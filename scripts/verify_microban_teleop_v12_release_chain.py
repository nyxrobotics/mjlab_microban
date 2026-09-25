#!/usr/bin/env python3
"""Verify the archived v12 stage chain with each stage's pinned evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "artifacts"
    / "teleop_v12_releases"
    / "microban_teleop_v12_full_chain_manifest.json"
)
EXPECTED_STAGES = (
    (3000, 2999, True),
    (3100, 3099, False),
    (7000, 6999, True),
    (7100, 7099, False),
    (10000, 9999, True),
    (10100, 10099, False),
    (15000, 14999, True),
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_ALLOWED_DATA_PREFIXES = (
    "artifacts/teleop_v12_",
    "logs/rsl_rl/mjlab_microban_teleop_v12/",
)
_RELOCATED_VALIDATOR = (
    PROJECT_ROOT / "scripts" / "validate_microban_teleop_v12_relocated_gate.py"
)
_RELOCATED_MIGRATION = (
    PROJECT_ROOT / "scripts" / "validate_microban_teleop_v12_relocated_migration.py"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(value: str) -> tuple[str, Path]:
    if not isinstance(value, str) or not value.startswith("repo://"):
        raise ValueError(f"Expected repo:// path, got {value!r}")
    relative = value.removeprefix("repo://")
    parsed = Path(relative)
    if (
        not relative
        or parsed.is_absolute()
        or parsed.as_posix() != relative
        or any(part in ("", ".", "..") for part in parsed.parts)
        or not relative.startswith(_ALLOWED_DATA_PREFIXES)
    ):
        raise ValueError(f"Invalid repository path: {value}")
    candidate = PROJECT_ROOT / parsed
    cursor = PROJECT_ROOT
    for part in parsed.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"Repository path traverses a symlink: {value}")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"Repository path escapes checkout: {value}") from exc
    return relative, candidate


def _git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(PROJECT_ROOT), *args], text=True
    ).strip()


def _require_clean_head_path(relative: str) -> None:
    index = (
        subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "ls-files", "--stage", "--", relative],
            check=True,
            capture_output=True,
            text=True,
        )
        .stdout.strip()
        .splitlines()
    )
    if len(index) != 1 or index[0].split(maxsplit=1)[0] not in {"100644", "100755"}:
        raise ValueError(f"Archived path is not one regular Git file: {relative}")
    tree = _git_output("ls-tree", "HEAD", "--", relative).splitlines()
    if len(tree) != 1 or tree[0].split(maxsplit=1)[0] not in {"100644", "100755"}:
        raise ValueError(f"Archived path is absent or irregular in HEAD: {relative}")
    for command in (
        ("diff", "--quiet", "--", relative),
        ("diff", "--cached", "--quiet", "HEAD", "--", relative),
    ):
        result = subprocess.run(["git", "-C", str(PROJECT_ROOT), *command], check=False)
        if result.returncode != 0:
            raise ValueError(f"Archived path differs from committed HEAD: {relative}")


def _require_archived(value: str, expected_sha256: str | None = None) -> Path:
    relative, path = _repo_path(value)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Archived file is missing, linked, or irregular: {value}")
    _require_clean_head_path(relative)
    if expected_sha256 is not None:
        if (
            not isinstance(expected_sha256, str)
            or _SHA256_RE.fullmatch(expected_sha256) is None
        ):
            raise ValueError(f"Invalid expected SHA-256 for {value}")
        actual = _sha256(path)
        if actual != expected_sha256:
            raise ValueError(
                f"SHA-256 mismatch for {value}: {actual} != {expected_sha256}"
            )
    return path


def _collect_repo_references(initial: list[Path]) -> tuple[set[str], set[str]]:
    pending = list(initial)
    visited: set[Path] = set()
    references: set[str] = set()
    hash_bound: set[str] = set()

    def bind(path_value: object, digest: object) -> None:
        if isinstance(path_value, str) and path_value.startswith("repo://"):
            if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
                raise ValueError(
                    f"Repository reference has no valid SHA-256: {path_value}"
                )
            _require_archived(path_value, digest)
            hash_bound.add(path_value)

    def walk(value: object) -> None:
        if isinstance(value, Mapping):
            path_value = value.get("path")
            if path_value is not None or "sha256" in value:
                bind(path_value, value.get("sha256"))
            reports = value.get("reports")
            report_hashes = value.get("report_sha256")
            if reports is not None or report_hashes is not None:
                if (
                    not isinstance(reports, dict)
                    or not isinstance(report_hashes, dict)
                    or set(reports) != set(report_hashes)
                ):
                    raise ValueError("Report paths and hashes are not one-to-one")
                for name, report in reports.items():
                    bind(report, report_hashes[name])
            for key, digest in value.items():
                if not isinstance(key, str) or not key.endswith("_sha256"):
                    continue
                stem = key.removesuffix("_sha256")
                for candidate in (stem, f"{stem}_path"):
                    if candidate in value:
                        bind(value[candidate], digest)
                        break
            for nested in value.values():
                walk(nested)
        elif isinstance(value, list):
            for nested in value:
                walk(nested)
        elif isinstance(value, str) and value.startswith("repo://"):
            references.add(value)

    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        if path.suffix != ".json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        before = set(references)
        walk(payload)
        for reference in references - before:
            referenced = _require_archived(reference)
            if referenced.suffix == ".json" and referenced not in visited:
                pending.append(referenced)
    unbound = references - hash_bound
    if unbound:
        raise ValueError(
            "Repository references are not hash-bound: " + ", ".join(sorted(unbound))
        )
    return references, hash_bound


def _check_manifest(manifest_path: Path) -> tuple[dict[str, object], set[str]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "microban_teleop_v12_release_chain_v1":
        raise ValueError("Unexpected release-chain schema")
    if (
        manifest.get("status") != "pass"
        or manifest.get("training_contract_version") != "12"
    ):
        raise ValueError("Release-chain manifest is not an accepted contract-v12 chain")
    recorded_roots = manifest.get("recorded_project_roots")
    if (
        not isinstance(recorded_roots, list)
        or not recorded_roots
        or not all(isinstance(value, str) for value in recorded_roots)
        or len(set(recorded_roots)) != len(recorded_roots)
    ):
        raise ValueError("Recorded project roots are missing or duplicated")
    for value in recorded_roots:
        root = Path(value) if isinstance(value, str) else Path()
        if (
            not isinstance(value, str)
            or not root.is_absolute()
            or root == Path(root.anchor)
            or any(part in (".", "..") for part in root.parts)
        ):
            raise ValueError(f"Invalid recorded project root: {value!r}")
    migration = manifest.get("bilateral_migration")
    if not isinstance(migration, dict) or migration.get("status") != "pass":
        raise ValueError("Bilateral migration archive is missing or not accepted")
    required_migration_fields = {
        "raw_checkpoint",
        "raw_checkpoint_sha256",
        "migrated_checkpoint",
        "migrated_checkpoint_sha256",
        "migration_receipt",
        "migration_receipt_sha256",
        "recovery_receipt",
        "recovery_receipt_sha256",
        "corner_parent_checkpoint",
        "corner_parent_checkpoint_sha256",
        "corner_parent_tracking_report",
        "corner_parent_tracking_report_sha256",
        "corner_superseded_tracking_report",
        "corner_superseded_tracking_report_sha256",
        "raw_run_context",
        "replay_run_context",
        "status",
    }
    if set(migration) != required_migration_fields:
        raise ValueError("Bilateral migration archive fields drifted")
    context_fields = {
        "agent_params",
        "agent_params_sha256",
        "environment_params",
        "environment_params_sha256",
        "source_diff",
        "source_diff_sha256",
    }
    for name in ("raw_run_context", "replay_run_context"):
        context = migration.get(name)
        if not isinstance(context, dict) or set(context) != context_fields:
            raise ValueError(f"Bilateral migration {name} fields drifted")
    stages = manifest.get("stages")
    if not isinstance(stages, list) or len(stages) != len(EXPECTED_STAGES):
        raise ValueError("Release-chain stage count is invalid")

    json_roots = [manifest_path]
    for stage, expected in zip(stages, EXPECTED_STAGES, strict=True):
        completed_updates, _iteration, canonical = expected
        if (
            stage.get("completed_updates"),
            stage.get("iteration"),
            stage.get("canonical_boundary"),
        ) != expected:
            raise ValueError(f"Unexpected stage sequence entry: {stage}")
        if stage.get("kind") != (
            "canonical_boundary" if canonical else "activation_canary"
        ):
            raise ValueError(
                f"Stage kind is inconsistent at update {completed_updates}"
            )
        validator_commit = stage.get("validator_commit")
        if (
            not isinstance(validator_commit, str)
            or _COMMIT_RE.fullmatch(validator_commit) is None
        ):
            raise ValueError(
                f"Validator commit is not pinned at update {completed_updates}"
            )
        resolved_validator = _git_output(
            "rev-parse", "--verify", f"{validator_commit}^{{commit}}"
        )
        if resolved_validator != validator_commit:
            raise ValueError(
                f"Validator ref is not an immutable full commit: {validator_commit}"
            )
        ancestor = subprocess.run(
            [
                "git",
                "-C",
                str(PROJECT_ROOT),
                "merge-base",
                "--is-ancestor",
                validator_commit,
                "HEAD",
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if ancestor.returncode != 0:
            raise ValueError(f"Validator commit is not an ancestor: {validator_commit}")

        gate = _require_archived(stage["gate"], stage["gate_sha256"])
        checkpoint = _require_archived(stage["checkpoint"], stage["checkpoint_sha256"])
        gate_payload = json.loads(gate.read_text(encoding="utf-8"))
        if gate_payload.get("status") != "pass":
            raise ValueError(f"Stage gate is not pass: {stage['gate']}")
        if (
            gate_payload.get("completed_updates"),
            gate_payload.get("iteration"),
            gate_payload.get("canonical_boundary"),
        ) != expected:
            raise ValueError(f"Stage gate metadata mismatch: {stage['gate']}")
        if gate_payload.get("checkpoint") != stage["checkpoint"]:
            raise ValueError(f"Stage gate checkpoint path mismatch: {stage['gate']}")
        if gate_payload.get("checkpoint_sha256") != _sha256(checkpoint):
            raise ValueError(f"Stage gate checkpoint hash mismatch: {stage['gate']}")
        json_roots.append(gate)

    deployment = manifest.get("deployment")
    if not isinstance(deployment, dict):
        raise TypeError("Deployment block is missing")
    receipt_path = _require_archived(
        deployment["receipt"], deployment["receipt_sha256"]
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    final_stage = stages[-1]
    packaged_policy = _require_archived(
        deployment["packaged_policy"], deployment["packaged_policy_sha256"]
    )
    runtime = receipt.get("microban_runtime_validator")
    smoke = (
        runtime.get("onnxruntime_compatibility_smoke")
        if isinstance(runtime, dict)
        else None
    )
    walk = runtime.get("walk_fallback") if isinstance(runtime, dict) else None
    if not all(
        (
            receipt.get("status") == "pass",
            receipt.get("completed_updates") == 15000,
            receipt.get("checkpoint_sha256") == final_stage["checkpoint_sha256"],
            receipt.get("stage_gate_sha256") == final_stage["gate_sha256"],
            receipt.get("output_sha256") == deployment.get("packaged_policy_sha256"),
            _sha256(packaged_policy) == receipt.get("output_sha256"),
            deployment.get("input_shape") == [1, 83],
            deployment.get("output_shape") == [1, 18],
            isinstance(runtime, dict) and runtime.get("status") == "pass",
            isinstance(runtime, dict) and runtime.get("input_width") == 83,
            isinstance(runtime, dict) and runtime.get("output_width") == 18,
            isinstance(smoke, dict) and smoke.get("status") == "pass",
            isinstance(smoke, dict)
            and smoke.get("providers") == [deployment.get("runtime_provider")],
            isinstance(smoke, dict)
            and smoke.get("sample_count") == deployment.get("runtime_samples"),
            isinstance(walk, dict) and walk.get("status") == "pass",
        )
    ):
        raise ValueError("Deployment receipt is not a complete runtime pass")
    json_roots.append(receipt_path)

    references, _ = _collect_repo_references(json_roots)
    for reference in references:
        _require_archived(reference)
    return manifest, references


def _smoke_packaged_policy(manifest: dict[str, object]) -> None:
    import numpy as np
    import onnxruntime as ort

    deployment = manifest["deployment"]
    policy = _require_archived(
        deployment["packaged_policy"], deployment["packaged_policy_sha256"]
    )
    session = ort.InferenceSession(
        str(policy), providers=[deployment["runtime_provider"]]
    )
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if (
        len(inputs) != 1
        or inputs[0].shape != [1, 83]
        or inputs[0].type != "tensor(float)"
        or len(outputs) != 1
        or outputs[0].shape != [1, 18]
        or outputs[0].type != "tensor(float)"
        or session.get_providers() != ["CPUExecutionProvider"]
    ):
        raise ValueError("Archived packaged ONNX I/O/provider contract drifted")
    corpus = (
        np.arange(16 * 83, dtype=np.float32).reshape(16, 83) % np.float32(29.0)
    ) / np.float32(29.0)
    for sample in corpus:
        result = session.run(
            [outputs[0].name], {inputs[0].name: sample.reshape(1, 83)}
        )[0]
        if (
            result.shape != (1, 18)
            or result.dtype != np.float32
            or not np.isfinite(result).all()
        ):
            raise ValueError("Archived packaged ONNX runtime smoke failed")
    print("[PASS] packaged ONNX CPU runtime samples=16 shape=[1,83]->[1,18]")


def _run_pinned_validators(
    manifest: dict[str, object], references: set[str], manifest_path: Path
) -> None:
    for support in (
        Path(__file__).resolve(),
        _RELOCATED_VALIDATOR,
        _RELOCATED_MIGRATION,
    ):
        relative = support.relative_to(PROJECT_ROOT).as_posix()
        if not support.is_file() or support.is_symlink():
            raise ValueError(f"Verifier support file is irregular: {relative}")
        _require_clean_head_path(relative)
    release_revision = _git_output("rev-parse", "HEAD")
    manifest_relative = manifest_path.relative_to(PROJECT_ROOT).as_posix()
    _require_archived(f"repo://{manifest_relative}")

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for stage in manifest["stages"]:
        grouped[stage["validator_commit"]].append(stage)

    checkout_paths = sorted(
        {_repo_path(reference)[0] for reference in references}
        | {
            _repo_path(stage[field])[0]
            for stage in manifest["stages"]
            for field in ("gate", "checkpoint")
        }
    )
    temporary_root = Path(tempfile.mkdtemp(prefix="microban-v12-chain-"))
    try:
        for validator_commit, stages in grouped.items():
            worktree = temporary_root / validator_commit[:12]
            added = False
            try:
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(PROJECT_ROOT),
                        "worktree",
                        "add",
                        "--detach",
                        str(worktree),
                        validator_commit,
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
                added = True
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(worktree),
                        "checkout",
                        release_revision,
                        "--",
                        *checkout_paths,
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
                environment = os.environ.copy()
                environment["PYTHONPATH"] = str(worktree / "src")
                environment["PYTHONNOUSERSITE"] = "1"
                environment["PYTHONSAFEPATH"] = "1"
                if validator_commit == manifest["stages"][-1]["validator_commit"]:
                    migration = manifest["bilateral_migration"]
                    migration_command = [
                        sys.executable,
                        "-sP",
                        str(_RELOCATED_MIGRATION),
                        "--checkout-root",
                        str(worktree),
                    ]
                    for recorded_root in manifest["recorded_project_roots"]:
                        migration_command.extend(("--recorded-root", recorded_root))
                    for option, field in (
                        ("--source", "raw_checkpoint"),
                        ("--migrated", "migrated_checkpoint"),
                        ("--migration-receipt", "migration_receipt"),
                        ("--recovery-receipt", "recovery_receipt"),
                        ("--corner-parent", "corner_parent_checkpoint"),
                        ("--corner-parent-tracking", "corner_parent_tracking_report"),
                        (
                            "--corner-superseded-tracking",
                            "corner_superseded_tracking_report",
                        ),
                    ):
                        migration_command.extend(
                            (option, str(worktree / _repo_path(migration[field])[0]))
                        )
                    for option, value in (
                        ("--stage-7100", manifest["stages"][3]["checkpoint"]),
                        ("--selected-10000", manifest["stages"][4]["checkpoint"]),
                        (
                            "--raw-agent-params",
                            migration["raw_run_context"]["agent_params"],
                        ),
                        (
                            "--replay-agent-params",
                            migration["replay_run_context"]["agent_params"],
                        ),
                    ):
                        migration_command.extend(
                            (option, str(worktree / _repo_path(value)[0]))
                        )
                    subprocess.run(
                        migration_command,
                        cwd=worktree,
                        env=environment,
                        check=True,
                        stdout=subprocess.DEVNULL,
                    )
                    print("[PASS] relocated bilateral migration and corner parent")
                for stage in stages:
                    gate = worktree / _repo_path(stage["gate"])[0]
                    checkpoint = worktree / _repo_path(stage["checkpoint"])[0]
                    command = [
                        sys.executable,
                        "-sP",
                        str(_RELOCATED_VALIDATOR),
                        "--checkout-root",
                        str(worktree),
                    ]
                    for recorded_root in manifest["recorded_project_roots"]:
                        command.extend(("--recorded-root", recorded_root))
                    command.extend((str(gate), str(checkpoint)))
                    subprocess.run(
                        command,
                        cwd=worktree,
                        env=environment,
                        check=True,
                        stdout=subprocess.DEVNULL,
                    )
                    print(
                        "[PASS] pinned validator "
                        f"{validator_commit[:12]} update={stage['completed_updates']}"
                    )
            finally:
                if added:
                    subprocess.run(
                        [
                            "git",
                            "-C",
                            str(PROJECT_ROOT),
                            "worktree",
                            "remove",
                            "--force",
                            str(worktree),
                        ],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)
        subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "worktree", "prune"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--paths-only",
        action="store_true",
        help="verify tracked paths, hashes, metadata and receipt without rerunning evaluators",
    )
    args = parser.parse_args()
    manifest_path = args.manifest.expanduser()
    if not manifest_path.is_absolute():
        manifest_path = Path.cwd() / manifest_path
    try:
        manifest_relative = manifest_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError as exc:
        raise ValueError("Manifest must be inside the repository") from exc
    manifest_path = _require_archived(f"repo://{manifest_relative}")
    manifest, references = _check_manifest(manifest_path)
    for stage in manifest["stages"]:
        print(
            f"[PASS] archived update={stage['completed_updates']} "
            f"checkpoint={stage['checkpoint_sha256']}"
        )
    if not args.paths_only:
        _smoke_packaged_policy(manifest)
        _run_pinned_validators(manifest, references, manifest_path)
    print("MICROBAN_TELEOP_V12_RELEASE_CHAIN=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
