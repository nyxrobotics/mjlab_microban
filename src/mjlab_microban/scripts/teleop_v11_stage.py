# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Fail-closed planning and receipts for canonical contract-v11 stages."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from mjlab_microban.scripts import evaluate_teleop_checkpoint as evaluator
from mjlab_microban.scripts import teleop_v10_stage as receipt_primitives
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_TELEOP_ACTOR_INITIALIZATION,
    MICROBAN_TELEOP_RECIPE_REVISION,
    MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION,
    TELEOP_FINAL_CANONICAL_STAGE_TARGET,
    validate_teleop_checkpoint_contract,
)
from mjlab_microban.tasks.microban_safe_velocity_checkpoint import (
    inspect_safe_velocity_checkpoint,
)
from mjlab_microban.tasks.microban_teleop_bootstrap import (
    validate_safe_velocity_acceptance_receipt,
)
from mjlab_microban.tasks.microban_teleop_provenance import (
    MICROBAN_TELEOP_CANONICAL_STAGE_MODE,
    MICROBAN_TELEOP_TRAINING_PROVENANCE_KEY,
    MICROBAN_TELEOP_TRAINING_PROVENANCE_SHA256_KEY,
    sha256_file,
    validate_canonical_stage_critical_config,
    validate_training_provenance,
)

V11_STAGE_BOUNDARIES = (3_000, 7_000, 10_000, 15_000)
V11_CANARY_UPDATE_INTERVAL = 100
V11_EVALUATION_SEEDS = (42, 43, 44)
V11_CANARY_EVALUATION_SEEDS = (42,)
V11_GATE_SCHEMA_VERSION = 3
V11_CANARY_RECEIPT_SCHEMA_VERSION = 1
V11_CANARY_SCENARIOS = (
    "neutral",
    "low_forward",
    "low_yaw_left",
    "mid_yaw_left",
    "low_yaw_right",
    "mid_yaw_right",
)
_SIGNED_LOW_MID_SCENARIOS = (
    "neutral",
    "low_forward",
    "mid_forward",
    "low_backward",
    "mid_backward",
    "low_lateral_left",
    "mid_lateral_left",
    "low_lateral_right",
    "mid_lateral_right",
    "low_yaw_left",
    "mid_yaw_left",
    "low_yaw_right",
    "mid_yaw_right",
)
_LOCOMOTION_SCENARIOS = (
    *_SIGNED_LOW_MID_SCENARIOS,
    "max_forward",
    "max_backward",
    "max_lateral_left",
    "max_lateral_right",
    "max_moving_yaw_left",
    "max_moving_yaw_right",
    "max_stationary_yaw_left",
    "max_stationary_yaw_right",
    "mixed_twist_forward_left",
    "mixed_twist_backward_right",
)
_HAND_SCENARIOS = (*_LOCOMOTION_SCENARIOS, "max_hands_left", "max_hands_right")


@dataclass(frozen=True)
class StageInterval:
    """The canonical v11 interval containing a completed-update count."""

    start_boundary: int
    target_boundary: int
    interrupted: bool


def is_v11_canary_checkpoint(completed_iterations: int) -> bool:
    """Accept deliberate segment ends and RSL-RL zero-based periodic saves."""

    return completed_iterations % V11_CANARY_UPDATE_INTERVAL in (0, 1)


def resolve_stage_interval(completed_iterations: int) -> StageInterval:
    """Resolve an exact boundary or interrupted v11 checkpoint."""

    if isinstance(completed_iterations, bool) or not isinstance(
        completed_iterations, int
    ):
        raise TypeError("completed_iterations must be an integer")
    if completed_iterations < 0:
        raise ValueError("completed_iterations must be non-negative")
    if completed_iterations >= V11_STAGE_BOUNDARIES[-1]:
        if completed_iterations == V11_STAGE_BOUNDARIES[-1]:
            raise ValueError(
                "The final 15,000-update boundary is already reached; evaluate/export it"
            )
        raise ValueError("Checkpoint exceeds the final v11 stage boundary")
    previous = 0
    for index, boundary in enumerate(V11_STAGE_BOUNDARIES):
        if completed_iterations == boundary:
            return StageInterval(boundary, V11_STAGE_BOUNDARIES[index + 1], False)
        if completed_iterations < boundary:
            return StageInterval(previous, boundary, completed_iterations != previous)
        previous = boundary
    raise AssertionError("Unreachable v11 stage interval")


def scenarios_for_boundary(boundary: int) -> tuple[str, ...] | None:
    """Return ordered scenario coverage; ``None`` denotes the full suite."""

    if boundary == 3_000:
        return _SIGNED_LOW_MID_SCENARIOS
    if boundary == 7_000:
        return _LOCOMOTION_SCENARIOS
    if boundary == 10_000:
        return _HAND_SCENARIOS
    if boundary == 15_000:
        return None
    raise ValueError(f"Not a canonical v11 boundary: {boundary}")


def boundary_requires_moving_hmd(boundary: int) -> bool:
    scenarios_for_boundary(boundary)
    return boundary >= 10_000


def boundary_enforces_performance(boundary: int) -> bool:
    scenarios_for_boundary(boundary)
    return boundary == TELEOP_FINAL_CANONICAL_STAGE_TARGET


def validate_bootstrap(
    checkpoint_path: str | Path, receipt_path: str | Path
) -> dict[str, str]:
    """Authenticate the pinned fresh actor source before simulator startup."""

    source = inspect_safe_velocity_checkpoint(checkpoint_path)
    receipt = validate_safe_velocity_acceptance_receipt(receipt_path, source)
    return {
        "checkpoint": str(source.path),
        "checkpoint_sha256": source.sha256,
        "receipt": str(receipt.path),
        "receipt_sha256": receipt.sha256,
    }


def _checkpoint_infos(checkpoint_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("infos"), dict):
        raise TypeError("Checkpoint does not contain an infos dictionary")
    return payload, payload["infos"]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _current_checkpoint_identity(
    checkpoint_path: str | Path, *, completed_iterations: int
) -> tuple[Path, str, str]:
    checkpoint = Path(checkpoint_path).resolve(strict=True)
    contract = validate_teleop_checkpoint_contract(checkpoint, map_location="cpu")
    if contract.iteration != completed_iterations - 1:
        raise ValueError("Checkpoint iteration does not match completed updates")
    training = contract.training_provenance_identity
    if training is None or not training.canonical_stage:
        raise ValueError("Checkpoint is not a canonical v11 stage")
    if training.stage_target_boundary is None:
        raise ValueError("Checkpoint stage target is missing")
    _, infos = _checkpoint_infos(checkpoint)
    training_digest = infos.get(MICROBAN_TELEOP_TRAINING_PROVENANCE_SHA256_KEY)
    if not isinstance(training_digest, str):
        raise TypeError("Checkpoint training provenance SHA-256 is missing")
    return checkpoint, sha256_file(checkpoint), training_digest


def validate_checkpoint_for_evaluation(
    checkpoint_path: str | Path,
    *,
    completed_iterations: int,
    canary: bool,
) -> dict[str, Any]:
    """Reject wrong-stage checkpoints before an expensive GPU evaluation."""

    checkpoint, checkpoint_digest, training_digest = _current_checkpoint_identity(
        checkpoint_path, completed_iterations=completed_iterations
    )
    contract = validate_teleop_checkpoint_contract(checkpoint, map_location="cpu")
    training = contract.training_provenance_identity
    assert training is not None
    interval = resolve_stage_interval(completed_iterations)
    if canary:
        if not interval.interrupted:
            raise ValueError("Canary evaluation requires an interrupted stage")
        if not is_v11_canary_checkpoint(completed_iterations):
            raise ValueError("Canary checkpoint is not a canonical recovery save")
        if training.stage_start_boundary != interval.start_boundary:
            raise ValueError("Canary checkpoint stage start mismatch")
        if training.stage_target_boundary != interval.target_boundary:
            raise ValueError("Canary checkpoint stage target mismatch")
    else:
        scenarios_for_boundary(completed_iterations)
        if training.stage_target_boundary != completed_iterations:
            raise ValueError("Boundary checkpoint provenance target mismatch")
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_digest,
        "training_provenance_sha256": training_digest,
        "stage_start_boundary": interval.start_boundary,
        "stage_target_boundary": interval.target_boundary,
    }


def publish_stage_gate(
    gate_path: str | Path,
    checkpoint_path: str | Path,
    run_name: str,
    completed_iterations: int,
    report_paths: Sequence[str | Path],
    moving_hmd_report_paths: Sequence[str | Path],
) -> dict[str, Any]:
    """Validate reports and publish a v11 boundary gate atomically.

    The report schema is intentionally shared with v10. The checkpoint
    validator underneath it accepts only the current contract-v11 identity, so
    importing these byte-stable receipt primitives cannot admit a v10 model or
    the retired full-state migration path.
    """

    return receipt_primitives.publish_stage_gate(
        gate_path,
        checkpoint_path,
        run_name,
        completed_iterations,
        report_paths,
        moving_hmd_report_paths,
    )


def validate_stage_gate(
    gate_path: str | Path,
    *,
    expected_boundary: int,
    expected_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate a v11 boundary gate and all hash-bound artifacts."""

    return receipt_primitives.validate_stage_gate(
        gate_path,
        expected_boundary=expected_boundary,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )


def publish_canary_receipt(
    receipt_path: str | Path,
    checkpoint_path: str | Path,
    run_name: str,
    completed_iterations: int,
    report_path: str | Path,
) -> dict[str, Any]:
    """Publish non-deployment hard-safety evidence for an interrupted stage."""

    interval = resolve_stage_interval(completed_iterations)
    if not interval.interrupted or not is_v11_canary_checkpoint(completed_iterations):
        raise ValueError(
            "Canary receipts require a deliberate segment end or an RSL-RL "
            "periodic recovery checkpoint inside a stage"
        )
    checkpoint, checkpoint_digest, training_digest = _current_checkpoint_identity(
        checkpoint_path, completed_iterations=completed_iterations
    )
    report = Path(report_path).resolve(strict=True)
    digest_before = sha256_file(report)
    receipt_primitives._validate_report(
        report,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_digest,
        checkpoint_iteration=completed_iterations - 1,
        training_provenance_sha256=training_digest,
        seed=V11_CANARY_EVALUATION_SEEDS[0],
        scenarios=V11_CANARY_SCENARIOS,
        acceptance_profile=evaluator.CANARY_HARD_SAFETY_PROFILE,
        performance_enforced=False,
        moving_hmd=False,
        final_nominal=False,
    )
    digest_after = sha256_file(report)
    if digest_after != digest_before:
        raise RuntimeError(f"Evaluator report changed during validation: {report}")
    receipt = {
        "schema_version": V11_CANARY_RECEIPT_SCHEMA_VERSION,
        "receipt_kind": "canonical_v11_interrupted_canary",
        "status": "pass",
        "deployment_authority": False,
        "training_contract_version": MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION,
        "recipe_revision": MICROBAN_TELEOP_RECIPE_REVISION,
        "training_provenance_sha256": training_digest,
        "evaluator_revision": evaluator.TELEOP_EVALUATOR_REVISION,
        "acceptance_revision": evaluator.TELEOP_ACCEPTANCE_REVISION,
        "acceptance_profile": evaluator.CANARY_HARD_SAFETY_PROFILE,
        "evaluator_source_sha256": sha256_file(Path(evaluator.__file__)),
        "run_name": run_name,
        "stage_start_boundary": interval.start_boundary,
        "stage_target_boundary": interval.target_boundary,
        "completed_iterations": completed_iterations,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_digest,
        "evaluation_seeds": list(V11_CANARY_EVALUATION_SEEDS),
        "scenarios": list(V11_CANARY_SCENARIOS),
        "reports": [str(report)],
        "report_sha256": {str(report): digest_after},
        "moving_hmd_reports": [],
        "moving_hmd_report_sha256": {},
    }
    output = Path(receipt_path).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace canary receipt: {output}")
    _atomic_json(output, receipt)
    return receipt


def validate_canary_receipt(
    receipt_path: str | Path,
    *,
    checkpoint_path: str | Path,
    completed_iterations: int,
) -> dict[str, Any]:
    """Validate an interrupted checkpoint's non-deployment v11 canary."""

    receipt_file = Path(receipt_path).resolve(strict=True)
    receipt_digest_before = sha256_file(receipt_file)
    checkpoint = Path(checkpoint_path).resolve(strict=True)
    receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise TypeError("V11 canary receipt must contain a JSON object")
    interval = resolve_stage_interval(completed_iterations)
    if not interval.interrupted:
        raise ValueError("Canary receipt checkpoint must be inside a stage")
    if not is_v11_canary_checkpoint(completed_iterations):
        raise ValueError("Canary receipt checkpoint is not a canonical recovery save")
    _, checkpoint_digest, training_digest = _current_checkpoint_identity(
        checkpoint, completed_iterations=completed_iterations
    )
    expected = {
        "schema_version": V11_CANARY_RECEIPT_SCHEMA_VERSION,
        "receipt_kind": "canonical_v11_interrupted_canary",
        "status": "pass",
        "deployment_authority": False,
        "training_contract_version": MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION,
        "recipe_revision": MICROBAN_TELEOP_RECIPE_REVISION,
        "evaluator_revision": evaluator.TELEOP_EVALUATOR_REVISION,
        "acceptance_revision": evaluator.TELEOP_ACCEPTANCE_REVISION,
        "acceptance_profile": evaluator.CANARY_HARD_SAFETY_PROFILE,
        "evaluator_source_sha256": sha256_file(Path(evaluator.__file__)),
        "run_name": checkpoint.parent.name,
        "stage_start_boundary": interval.start_boundary,
        "stage_target_boundary": interval.target_boundary,
        "training_provenance_sha256": training_digest,
        "completed_iterations": completed_iterations,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_digest,
        "evaluation_seeds": list(V11_CANARY_EVALUATION_SEEDS),
        "scenarios": list(V11_CANARY_SCENARIOS),
        "moving_hmd_reports": [],
        "moving_hmd_report_sha256": {},
    }
    failures = [key for key, value in expected.items() if receipt.get(key) != value]
    reports = receipt.get("reports")
    hashes = receipt.get("report_sha256")
    if not isinstance(reports, list) or len(reports) != 1:
        failures.append("reports")
    elif not isinstance(hashes, dict) or set(hashes) != set(reports):
        failures.append("report SHA-256 coverage")
    else:
        report = Path(reports[0])
        if not report.is_file():
            failures.append("changed canary report")
        else:
            digest_before = sha256_file(report)
            if digest_before != hashes[reports[0]]:
                failures.append("changed canary report")
            else:
                try:
                    receipt_primitives._validate_report(
                        report,
                        checkpoint=checkpoint,
                        checkpoint_sha256=checkpoint_digest,
                        checkpoint_iteration=completed_iterations - 1,
                        training_provenance_sha256=training_digest,
                        seed=V11_CANARY_EVALUATION_SEEDS[0],
                        scenarios=V11_CANARY_SCENARIOS,
                        acceptance_profile=evaluator.CANARY_HARD_SAFETY_PROFILE,
                        performance_enforced=False,
                        moving_hmd=False,
                        final_nominal=False,
                    )
                except (OSError, TypeError, ValueError):
                    failures.append("invalid canary report")
                if sha256_file(report) != digest_before:
                    failures.append("canary report changed during validation")
    if sha256_file(receipt_file) != receipt_digest_before:
        failures.append("canary receipt changed during validation")
    if failures:
        raise ValueError("Invalid v11 canary receipt: " + "; ".join(failures))
    return receipt


def _unique_gate_by_sha256(root: Path, pattern: str, expected_sha256: str) -> Path:
    matches = [
        candidate.resolve()
        for candidate in sorted(root.glob(pattern))
        if candidate.is_file() and sha256_file(candidate) == expected_sha256
    ]
    if len(matches) != 1:
        raise ValueError(
            "Pinned parent receipt was not found uniquely by SHA-256 "
            f"({len(matches)} matches)"
        )
    return matches[0]


def validate_interrupted_stage_checkpoint(
    checkpoint_path: str | Path,
    *,
    completed_iterations: int,
    interval: StageInterval,
    gate_root: str | Path,
    canary_root: str | Path,
) -> dict[str, Any]:
    """Validate v11 lineage and a hard-safety receipt before another segment."""

    if not interval.interrupted:
        raise ValueError("Interrupted-stage validation requires an in-stage interval")
    checkpoint = Path(checkpoint_path).resolve(strict=True)
    contract = validate_teleop_checkpoint_contract(checkpoint, map_location="cpu")
    if contract.iteration != completed_iterations - 1:
        raise ValueError("Checkpoint iteration does not match completed updates")
    _, infos = _checkpoint_infos(checkpoint)
    manifest = validate_training_provenance(
        infos.get(MICROBAN_TELEOP_TRAINING_PROVENANCE_KEY),
        infos.get(MICROBAN_TELEOP_TRAINING_PROVENANCE_SHA256_KEY),
        require_canonical_stage=True,
        expected_contract_version=MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION,
        expected_recipe_revision=MICROBAN_TELEOP_RECIPE_REVISION,
        expected_actor_initialization=MICROBAN_TELEOP_ACTOR_INITIALIZATION,
    )
    validate_canonical_stage_critical_config(manifest)
    invocation = manifest.get("invocation")
    if not isinstance(invocation, dict):
        raise TypeError("Checkpoint canonical invocation is malformed")
    expected_invocation = {
        "mode": MICROBAN_TELEOP_CANONICAL_STAGE_MODE,
        "stage_start_boundary": interval.start_boundary,
        "stage_target_boundary": interval.target_boundary,
    }
    mismatches = [
        key
        for key, value in expected_invocation.items()
        if invocation.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "Interrupted checkpoint invocation mismatch: " + ", ".join(mismatches)
        )
    parent_checkpoint_sha256 = invocation.get("parent_checkpoint_sha256")
    parent_gate_sha256 = invocation.get("parent_gate_sha256")
    parent_gate: Path | None = None
    if interval.start_boundary == 0:
        if parent_checkpoint_sha256 is not None or parent_gate_sha256 is not None:
            raise ValueError("Initial v11 stage must not claim parent gate lineage")
    else:
        if not isinstance(parent_checkpoint_sha256, str) or not isinstance(
            parent_gate_sha256, str
        ):
            raise ValueError("Interrupted v11 stage is missing pinned parent SHAs")
        parent_gate = _unique_gate_by_sha256(
            Path(gate_root).resolve(),
            f"*_boundary_{interval.start_boundary}_gate.json",
            parent_gate_sha256,
        )
        validate_stage_gate(
            parent_gate,
            expected_boundary=interval.start_boundary,
            expected_checkpoint_sha256=parent_checkpoint_sha256,
        )

    canary_pattern = f"*_model_{contract.iteration}_canary_gate.json"
    canary_matches: list[Path] = []
    for candidate in sorted(Path(canary_root).resolve().glob(canary_pattern)):
        try:
            validate_canary_receipt(
                candidate,
                checkpoint_path=checkpoint,
                completed_iterations=completed_iterations,
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        canary_matches.append(candidate.resolve())
    if len(canary_matches) != 1:
        raise ValueError(
            "Interrupted checkpoint requires exactly one passing v11 canary "
            f"receipt ({len(canary_matches)} found)"
        )
    return {
        "stage_start_boundary": interval.start_boundary,
        "stage_target_boundary": interval.target_boundary,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "parent_gate_sha256": parent_gate_sha256,
        "parent_gate": None if parent_gate is None else str(parent_gate),
        "canary_receipt": str(canary_matches[0]),
        "training_provenance_sha256": infos[
            MICROBAN_TELEOP_TRAINING_PROVENANCE_SHA256_KEY
        ],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    bootstrap = subparsers.add_parser("validate-bootstrap")
    bootstrap.add_argument("checkpoint", type=Path)
    bootstrap.add_argument("receipt", type=Path)
    interval = subparsers.add_parser("resolve-interval")
    interval.add_argument("completed_iterations", type=int)
    interrupted = subparsers.add_parser("validate-interrupted")
    interrupted.add_argument("checkpoint", type=Path)
    interrupted.add_argument("--completed-iterations", type=int, required=True)
    interrupted.add_argument("--stage-start-boundary", type=int, required=True)
    interrupted.add_argument("--stage-target-boundary", type=int, required=True)
    interrupted.add_argument("--gate-root", type=Path, required=True)
    interrupted.add_argument("--canary-root", type=Path, required=True)
    boundary = subparsers.add_parser("validate-boundary")
    boundary.add_argument("gate", type=Path)
    boundary.add_argument("checkpoint", type=Path)
    boundary.add_argument("--boundary", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "validate-bootstrap":
        result = validate_bootstrap(args.checkpoint, args.receipt)
        print(f"[PASS] safe bootstrap checkpoint sha256={result['checkpoint_sha256']}")
        print(f"[PASS] safe bootstrap receipt sha256={result['receipt_sha256']}")
        return
    if args.command == "resolve-interval":
        interval = resolve_stage_interval(args.completed_iterations)
        print(interval.start_boundary)
        print(interval.target_boundary)
        print(int(interval.interrupted))
        return
    if args.command == "validate-boundary":
        checkpoint = args.checkpoint.resolve(strict=True)
        checkpoint_digest = sha256_file(checkpoint)
        validate_stage_gate(
            args.gate,
            expected_boundary=args.boundary,
            expected_checkpoint_sha256=checkpoint_digest,
        )
        print(checkpoint_digest)
        print(sha256_file(args.gate))
        return
    interval = StageInterval(
        start_boundary=args.stage_start_boundary,
        target_boundary=args.stage_target_boundary,
        interrupted=True,
    )
    expected = resolve_stage_interval(args.completed_iterations)
    if interval != expected:
        raise ValueError(
            f"Requested interval {interval} != resolved interval {expected}"
        )
    result = validate_interrupted_stage_checkpoint(
        args.checkpoint,
        completed_iterations=args.completed_iterations,
        interval=interval,
        gate_root=args.gate_root,
        canary_root=args.canary_root,
    )
    print(result["stage_start_boundary"])
    print(result["stage_target_boundary"])
    print(result["parent_checkpoint_sha256"] or "none")
    print(result["parent_gate_sha256"] or "none")


if __name__ == "__main__":
    main()
