#!/usr/bin/env python3
"""Verify the archived v12 stage chain with each stage's pinned evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


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
    if not relative or Path(relative).is_absolute():
        raise ValueError(f"Invalid repository path: {value}")
    resolved = (PROJECT_ROOT / relative).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"Repository path escapes checkout: {value}") from exc
    return relative, resolved


def _require_archived(value: str, expected_sha256: str | None = None) -> Path:
    relative, path = _repo_path(value)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Archived file is missing, linked, or irregular: {value}")
    tracked = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "ls-files", "--error-unmatch", "--", relative],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if tracked.returncode != 0:
        raise ValueError(f"Archived file is not tracked by Git: {value}")
    if expected_sha256 is not None:
        actual = _sha256(path)
        if actual != expected_sha256:
            raise ValueError(
                f"SHA-256 mismatch for {value}: {actual} != {expected_sha256}"
            )
    return path


def _collect_repo_references(initial: list[Path]) -> set[str]:
    pending = list(initial)
    visited: set[Path] = set()
    references: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
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
    return references


def _check_manifest(manifest_path: Path) -> tuple[dict[str, Any], set[str]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "microban_teleop_v12_release_chain_v1":
        raise ValueError("Unexpected release-chain schema")
    if manifest.get("status") != "pass" or manifest.get("training_contract_version") != "12":
        raise ValueError("Release-chain manifest is not an accepted contract-v12 chain")
    stages = manifest.get("stages")
    if not isinstance(stages, list) or len(stages) != len(EXPECTED_STAGES):
        raise ValueError("Release-chain stage count is invalid")

    json_roots = [manifest_path]
    for stage, expected in zip(stages, EXPECTED_STAGES, strict=True):
        completed_updates, iteration, canonical = expected
        if (
            stage.get("completed_updates"),
            stage.get("iteration"),
            stage.get("canonical_boundary"),
        ) != expected:
            raise ValueError(f"Unexpected stage sequence entry: {stage}")
        if stage.get("kind") != (
            "canonical_boundary" if canonical else "activation_canary"
        ):
            raise ValueError(f"Stage kind is inconsistent at update {completed_updates}")
        validator_commit = stage.get("validator_commit")
        if not isinstance(validator_commit, str) or len(validator_commit) != 40:
            raise ValueError(f"Validator commit is not pinned at update {completed_updates}")
        subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "cat-file", "-e", f"{validator_commit}^{{commit}}"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        gate = _require_archived(stage["gate"], stage["gate_sha256"])
        checkpoint = _require_archived(
            stage["checkpoint"], stage["checkpoint_sha256"]
        )
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
        raise ValueError("Deployment block is missing")
    receipt_path = _require_archived(
        deployment["receipt"], deployment["receipt_sha256"]
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    final_stage = stages[-1]
    runtime = receipt.get("microban_runtime_validator")
    smoke = runtime.get("onnxruntime_compatibility_smoke") if isinstance(runtime, dict) else None
    walk = runtime.get("walk_fallback") if isinstance(runtime, dict) else None
    if not all(
        (
            receipt.get("status") == "pass",
            receipt.get("completed_updates") == 15000,
            receipt.get("checkpoint_sha256") == final_stage["checkpoint_sha256"],
            receipt.get("stage_gate_sha256") == final_stage["gate_sha256"],
            receipt.get("output_sha256") == deployment.get("packaged_policy_sha256"),
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

    references = _collect_repo_references(json_roots)
    for reference in references:
        _require_archived(reference)
    return manifest, references


def _run_pinned_validators(manifest: dict[str, Any], references: set[str]) -> None:
    release_revision = subprocess.check_output(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    manifest_relative = DEFAULT_MANIFEST.relative_to(PROJECT_ROOT).as_posix()
    committed = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "cat-file", "-e", f"{release_revision}:{manifest_relative}"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if committed.returncode != 0:
        raise ValueError("Commit the release-chain archive before full verification")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
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
            try:
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
                for stage in stages:
                    gate = worktree / _repo_path(stage["gate"])[0]
                    checkpoint = worktree / _repo_path(stage["checkpoint"])[0]
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "mjlab_microban.scripts.teleop_v12_stage",
                            "validate",
                            str(gate),
                            str(checkpoint),
                        ],
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
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--paths-only",
        action="store_true",
        help="verify tracked paths, hashes, metadata and receipt without rerunning evaluators",
    )
    args = parser.parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    if manifest_path != DEFAULT_MANIFEST.resolve():
        try:
            manifest_path.relative_to(PROJECT_ROOT)
        except ValueError as exc:
            raise ValueError("Manifest must be inside the repository") from exc
    manifest, references = _check_manifest(manifest_path)
    for stage in manifest["stages"]:
        print(
            f"[PASS] archived update={stage['completed_updates']} "
            f"checkpoint={stage['checkpoint_sha256']}"
        )
    if not args.paths_only:
        _run_pinned_validators(manifest, references)
    print("MICROBAN_TELEOP_V12_RELEASE_CHAIN=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
