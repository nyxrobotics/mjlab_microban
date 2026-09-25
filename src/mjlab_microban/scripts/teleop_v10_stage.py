# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Fail-closed planning and receipts for canonical contract-v10 stages."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from mjlab_microban.scripts import evaluate_teleop_checkpoint as evaluator
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_TELEOP_ACTOR_INITIALIZATION,
    MICROBAN_TELEOP_RECIPE_REVISION,
    MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION,
    TELEOP_FINAL_CANONICAL_STAGE_TARGET,
    validate_teleop_checkpoint_contract,
    validate_v9_migration_source_checkpoint,
)
from mjlab_microban.tasks.microban_teleop_provenance import (
    MICROBAN_TELEOP_CANONICAL_MIGRATION_STAGE_MODE,
    MICROBAN_TELEOP_CANONICAL_STAGE_MODE,
    MICROBAN_TELEOP_TRAINING_PROVENANCE_KEY,
    MICROBAN_TELEOP_TRAINING_PROVENANCE_SHA256_KEY,
    MICROBAN_TELEOP_V10_LEGACY_CHECKPOINT_ITERATION,
    MICROBAN_TELEOP_V10_LEGACY_CHECKPOINT_SHA256,
    MICROBAN_TELEOP_V10_LEGACY_COMMON_STEP_COUNTER,
    MICROBAN_TELEOP_V10_LEGACY_GATE_SHA256,
    MICROBAN_TELEOP_V10_LEGACY_OPTIMIZER_LEARNING_RATE,
    MICROBAN_TELEOP_V10_LEGACY_SOURCE_TREE_SHA256,
    MICROBAN_TELEOP_V10_LEGACY_TRAINING_PROVENANCE_SHA256,
    MICROBAN_TELEOP_V10_MIGRATION_START_BOUNDARY,
    sha256_file,
    validate_canonical_stage_critical_config,
    validate_training_provenance,
    validate_v10_migration_source_identity,
)

# Contract v10 started from the pinned 1,500-update v9 checkpoint and then
# advanced through these four boundaries.  Do not alias the current contract's
# boundary tuple: contract v11 added the fresh zero boundary for its own stage
# planner, which would make this historical planner return 1,500 -> 0.
V10_STAGE_BOUNDARIES = (3_000, 7_000, 10_000, 15_000)
V10_GATE_SCHEMA_VERSION = 3
V10_CANARY_RECEIPT_SCHEMA_VERSION = 1
V10_CANARY_UPDATE_INTERVAL = 100
V10_EVALUATION_SEEDS = (42, 43, 44)
V10_CANARY_EVALUATION_SEEDS = (42,)

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
V10_CANARY_SCENARIOS = (
    "neutral",
    "low_forward",
    "low_yaw_left",
    "mid_yaw_left",
    "low_yaw_right",
    "mid_yaw_right",
)


def is_v10_canary_checkpoint(completed_iterations: int) -> bool:
    """Accept deliberate segment ends and RSL-RL's zero-based periodic saves."""

    return completed_iterations % V10_CANARY_UPDATE_INTERVAL in (0, 1)


@dataclass(frozen=True)
class StageInterval:
    """The canonical v10 interval containing a completed-update count."""

    start_boundary: int
    target_boundary: int
    interrupted: bool
    migration: bool = False


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
    raise ValueError(f"Not a canonical v10 boundary: {boundary}")


def boundary_requires_moving_hmd(boundary: int) -> bool:
    scenarios_for_boundary(boundary)
    return boundary >= 10_000


def boundary_enforces_performance(boundary: int) -> bool:
    scenarios_for_boundary(boundary)
    return boundary == TELEOP_FINAL_CANONICAL_STAGE_TARGET


def resolve_stage_interval(completed_iterations: int) -> StageInterval:
    """Resolve an exact boundary or interrupted v10 checkpoint."""

    if isinstance(completed_iterations, bool) or not isinstance(
        completed_iterations, int
    ):
        raise TypeError("completed_iterations must be an integer")
    if completed_iterations < MICROBAN_TELEOP_V10_MIGRATION_START_BOUNDARY:
        raise ValueError("v10 can only migrate from the pinned 1,500-update source")
    if completed_iterations >= V10_STAGE_BOUNDARIES[-1]:
        if completed_iterations == V10_STAGE_BOUNDARIES[-1]:
            raise ValueError(
                "The final 15,000-update boundary is already reached; "
                "evaluate/export it"
            )
        raise ValueError("Checkpoint exceeds the final v10 stage boundary")

    previous = MICROBAN_TELEOP_V10_MIGRATION_START_BOUNDARY
    for index, boundary in enumerate(V10_STAGE_BOUNDARIES):
        if completed_iterations == previous and previous == (
            MICROBAN_TELEOP_V10_MIGRATION_START_BOUNDARY
        ):
            return StageInterval(previous, boundary, interrupted=False, migration=True)
        if completed_iterations == boundary:
            return StageInterval(
                boundary,
                V10_STAGE_BOUNDARIES[index + 1],
                interrupted=False,
                migration=False,
            )
        if completed_iterations < boundary:
            return StageInterval(
                previous,
                boundary,
                interrupted=True,
                migration=previous
                == MICROBAN_TELEOP_V10_MIGRATION_START_BOUNDARY,
            )
        previous = boundary
    raise AssertionError("Unreachable v10 stage interval")


def _checkpoint_infos(checkpoint_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("infos"), dict):
        raise TypeError("Checkpoint does not contain an infos dictionary")
    return payload, payload["infos"]


def validate_legacy_migration_source(
    checkpoint_path: str | Path, gate_path: str | Path
) -> dict[str, Any]:
    """Authenticate the one legacy checkpoint/gate pair allowed into v10."""

    checkpoint = Path(checkpoint_path).resolve(strict=True)
    gate = Path(gate_path).resolve(strict=True)
    if not checkpoint.is_file() or not gate.is_file():
        raise ValueError("Migration source checkpoint and gate must be files")
    if checkpoint.name != (
        f"model_{MICROBAN_TELEOP_V10_LEGACY_CHECKPOINT_ITERATION}.pt"
    ):
        raise ValueError("Migration source checkpoint filename is not canonical")
    checkpoint_digest = sha256_file(checkpoint)
    gate_digest = sha256_file(gate)
    if checkpoint_digest != MICROBAN_TELEOP_V10_LEGACY_CHECKPOINT_SHA256:
        raise ValueError("Migration source checkpoint SHA-256 mismatch")
    if gate_digest != MICROBAN_TELEOP_V10_LEGACY_GATE_SHA256:
        raise ValueError("Migration source gate SHA-256 mismatch")

    # The core validator checks actor/critic state, Adam moments, bounded-action
    # buffers, v9 provenance, common step, and the inherited safe-source chain.
    validate_v9_migration_source_checkpoint(checkpoint, map_location="cpu")

    payload, infos = _checkpoint_infos(checkpoint)
    failures: list[str] = []
    if payload.get("iter") != MICROBAN_TELEOP_V10_LEGACY_CHECKPOINT_ITERATION:
        failures.append("checkpoint iteration")
    env_state = infos.get("env_state")
    if not isinstance(env_state, dict) or env_state.get(
        "common_step_counter"
    ) != MICROBAN_TELEOP_V10_LEGACY_COMMON_STEP_COUNTER:
        failures.append("common step counter")
    if (
        infos.get(MICROBAN_TELEOP_TRAINING_PROVENANCE_SHA256_KEY)
        != MICROBAN_TELEOP_V10_LEGACY_TRAINING_PROVENANCE_SHA256
    ):
        failures.append("training provenance SHA-256")
    provenance = infos.get(MICROBAN_TELEOP_TRAINING_PROVENANCE_KEY)
    source = provenance.get("source") if isinstance(provenance, dict) else None
    if not isinstance(source, dict) or source.get(
        "tree_sha256"
    ) != MICROBAN_TELEOP_V10_LEGACY_SOURCE_TREE_SHA256:
        failures.append("training source tree SHA-256")
    optimizer = payload.get("optimizer_state_dict")
    groups = optimizer.get("param_groups") if isinstance(optimizer, dict) else None
    if not isinstance(groups, list) or len(groups) != 1:
        failures.append("optimizer parameter groups")
    else:
        learning_rate = groups[0].get("lr")
        if (
            isinstance(learning_rate, bool)
            or not isinstance(learning_rate, (int, float))
            or not math.isclose(
                float(learning_rate),
                MICROBAN_TELEOP_V10_LEGACY_OPTIMIZER_LEARNING_RATE,
                rel_tol=1.0e-12,
                abs_tol=0.0,
            )
        ):
            failures.append("optimizer learning rate")

    receipt = json.loads(gate.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != 3 or receipt.get("status") != "pass":
        failures.append("legacy gate schema/status")
    if receipt.get("completed_iterations") != (
        MICROBAN_TELEOP_V10_MIGRATION_START_BOUNDARY
    ):
        failures.append("legacy gate boundary")
    if receipt.get("checkpoint_sha256") != checkpoint_digest:
        failures.append("legacy gate checkpoint SHA-256")
    if receipt.get("training_provenance_sha256") != (
        MICROBAN_TELEOP_V10_LEGACY_TRAINING_PROVENANCE_SHA256
    ):
        failures.append("legacy gate training provenance")
    if receipt.get("evaluation_seeds") != list(V10_EVALUATION_SEEDS):
        failures.append("legacy gate seed coverage")
    if failures:
        raise ValueError("Invalid v10 migration source: " + "; ".join(failures))
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_iteration": MICROBAN_TELEOP_V10_LEGACY_CHECKPOINT_ITERATION,
        "gate": str(gate),
        "gate_sha256": gate_digest,
    }


def _validate_report(
    report_path: Path,
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
    checkpoint_iteration: int,
    training_provenance_sha256: str,
    seed: int,
    scenarios: tuple[str, ...] | None,
    acceptance_profile: str,
    performance_enforced: bool,
    moving_hmd: bool,
    final_nominal: bool,
) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    expected_scenarios = (
        [item.name for item in evaluator.default_scenarios()]
        if scenarios is None
        else list(scenarios)
    )
    checks = (
        (report.get("seed") == seed, "seed"),
        (report.get("checkpoint_iteration") == checkpoint_iteration, "iteration"),
        (
            Path(report.get("checkpoint", "")).resolve() == checkpoint,
            "checkpoint path",
        ),
        (report.get("checkpoint_sha256") == checkpoint_sha256, "checkpoint SHA-256"),
        (
            report.get("evaluator_revision") == evaluator.TELEOP_EVALUATOR_REVISION,
            "evaluator revision",
        ),
        (
            report.get("acceptance_revision") == evaluator.TELEOP_ACCEPTANCE_REVISION,
            "acceptance revision",
        ),
        (report.get("acceptance_profile") == acceptance_profile, "profile"),
        (report.get("steps_per_scenario") == 1000, "steps"),
        (report.get("settle_steps") == 50, "settle steps"),
    )
    failures.extend(label for passed, label in checks if not passed)
    contract = report.get("training_contract", {})
    if contract.get("version") != MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION:
        failures.append("training contract")
    if contract.get("training_provenance_sha256") != training_provenance_sha256:
        failures.append("training provenance")
    if contract.get("canonical_training_stage") is not True:
        failures.append("canonical training stage")
    if contract.get("deployment_compatible") is not True:
        failures.append("deployment compatibility")
    summary = report.get("summary", {})
    if summary.get("hard_safety_checks_passed") is not True:
        failures.append("hard safety")
    if summary.get("acceptance_checks_passed") is not True:
        failures.append("acceptance")
    if summary.get("hmd_motion_evidence_passed") is not True:
        failures.append("HMD motion evidence summary")
    if summary.get("performance_acceptance_checks_enforced") is not (
        performance_enforced
    ):
        failures.append("performance enforcement")
    scenario_reports = report.get("scenarios", [])
    actual_scenarios = [
        item.get("name") for item in scenario_reports if isinstance(item, dict)
    ]
    if actual_scenarios != expected_scenarios:
        failures.append("scenario coverage")
    if any(
        item.get("acceptance", {}).get("performance_checks_enforced")
        is not performance_enforced
        for item in scenario_reports
        if isinstance(item, dict)
    ):
        failures.append("scenario performance enforcement")
    expected_status = "pass" if final_nominal else "diagnostic"
    if report.get("status") != expected_status:
        failures.append(f"status {expected_status}")
    if summary.get("canonical_coverage") is not final_nominal:
        failures.append("canonical coverage")
    hmd = report.get("hmd_neck_motion", {})
    nominal_environment = report.get("nominal_environment", {})
    if hmd.get("enabled") is not moving_hmd:
        failures.append("moving-HMD flag")
    if nominal_environment.get("hmd_neck_motion") is not moving_hmd:
        failures.append("nominal environment HMD flag")
    if moving_hmd:
        params = hmd.get("params")
        if not isinstance(params, dict) or params.get("neutral_probability") != 0.0:
            failures.append("moving-HMD parameters")
        evidence = hmd.get("evidence")
        if not isinstance(evidence, dict) or evidence.get("passed") is not True:
            failures.append("moving-HMD evidence")
        else:
            membership = evidence.get("active_event_membership", {})
            if (
                membership.get("all_scenarios") is not True
                or membership.get("inactive_scenarios") != []
                or membership.get("malformed_scenarios") != []
            ):
                failures.append("moving-HMD membership")
            if evidence.get("minimum_required_target_peak_to_peak_rad") != (
                evaluator.HMD_TARGET_PEAK_TO_PEAK_MIN_RAD
            ):
                failures.append("moving-HMD target threshold")
            if evidence.get("minimum_required_actual_peak_to_peak_rad") != (
                evaluator.HMD_ACTUAL_PEAK_TO_PEAK_MIN_RAD
            ):
                failures.append("moving-HMD actual threshold")
            for key, minimum in (
                (
                    "minimum_observed_target_peak_to_peak_rad_by_axis",
                    evaluator.HMD_TARGET_PEAK_TO_PEAK_MIN_RAD,
                ),
                (
                    "minimum_observed_actual_peak_to_peak_rad_by_axis",
                    evaluator.HMD_ACTUAL_PEAK_TO_PEAK_MIN_RAD,
                ),
            ):
                values = evidence.get(key)
                if not isinstance(values, dict) or set(values) != {
                    "head",
                    "neck_roll",
                    "neck_pitch",
                } or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or value < minimum
                    for value in values.values()
                ):
                    failures.append(key)
    elif hmd.get("params") is not None:
        failures.append("nominal HMD parameters")
    if failures:
        raise ValueError(
            f"Invalid evaluator report {report_path.name}: " + "; ".join(failures)
        )
    return report


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
        raise ValueError("Checkpoint is not a canonical v10 stage")
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
        if not is_v10_canary_checkpoint(completed_iterations):
            raise ValueError("Canary checkpoint is not a canonical recovery save")
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
    """Validate three-seed reports and atomically publish a v10 boundary gate."""

    boundary = completed_iterations
    scenarios = scenarios_for_boundary(boundary)
    performance = boundary_enforces_performance(boundary)
    moving_required = boundary_requires_moving_hmd(boundary)
    checkpoint, checkpoint_digest, training_digest = _current_checkpoint_identity(
        checkpoint_path, completed_iterations=completed_iterations
    )
    if len(report_paths) != len(V10_EVALUATION_SEEDS):
        raise ValueError("A stage gate requires exactly three nominal reports")
    expected_moving_count = len(V10_EVALUATION_SEEDS) if moving_required else 0
    if len(moving_hmd_report_paths) != expected_moving_count:
        raise ValueError("Moving-HMD report count does not match this boundary")
    profile = (
        evaluator.DEPLOYMENT_PERFORMANCE_PROFILE
        if performance
        else evaluator.INTERMEDIATE_HARD_SAFETY_PROFILE
    )
    nominal_paths = [Path(value).resolve(strict=True) for value in report_paths]
    moving_paths = [
        Path(value).resolve(strict=True) for value in moving_hmd_report_paths
    ]
    report_digests: dict[str, str] = {}
    for seed, report_path in zip(V10_EVALUATION_SEEDS, nominal_paths, strict=True):
        digest_before = sha256_file(report_path)
        _validate_report(
            report_path,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_digest,
            checkpoint_iteration=completed_iterations - 1,
            training_provenance_sha256=training_digest,
            seed=seed,
            scenarios=scenarios,
            acceptance_profile=profile,
            performance_enforced=performance,
            moving_hmd=False,
            final_nominal=performance,
        )
        digest_after = sha256_file(report_path)
        if digest_after != digest_before:
            raise RuntimeError(
                f"Evaluator report changed during validation: {report_path}"
            )
        report_digests[str(report_path)] = digest_after
    for seed, report_path in zip(V10_EVALUATION_SEEDS, moving_paths, strict=True):
        digest_before = sha256_file(report_path)
        _validate_report(
            report_path,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_digest,
            checkpoint_iteration=completed_iterations - 1,
            training_provenance_sha256=training_digest,
            seed=seed,
            scenarios=scenarios,
            acceptance_profile=profile,
            performance_enforced=performance,
            moving_hmd=True,
            final_nominal=False,
        )
        digest_after = sha256_file(report_path)
        if digest_after != digest_before:
            raise RuntimeError(
                f"Evaluator report changed during validation: {report_path}"
            )
        report_digests[str(report_path)] = digest_after
    gate = {
        "schema_version": V10_GATE_SCHEMA_VERSION,
        "status": "pass",
        "training_contract_version": MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION,
        "recipe_revision": MICROBAN_TELEOP_RECIPE_REVISION,
        "training_provenance_sha256": training_digest,
        "evaluator_revision": evaluator.TELEOP_EVALUATOR_REVISION,
        "acceptance_revision": evaluator.TELEOP_ACCEPTANCE_REVISION,
        "acceptance_profile": profile,
        "evaluator_source_sha256": sha256_file(Path(evaluator.__file__)),
        "run_name": run_name,
        "completed_iterations": completed_iterations,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_digest,
        "evaluation_seeds": list(V10_EVALUATION_SEEDS),
        "scenarios": list(scenarios) if scenarios is not None else "canonical",
        "reports": [str(path) for path in nominal_paths],
        "report_sha256": {
            str(path): report_digests[str(path)] for path in nominal_paths
        },
        "moving_hmd_reports": [str(path) for path in moving_paths],
        "moving_hmd_report_sha256": {
            str(path): report_digests[str(path)] for path in moving_paths
        },
    }
    output = Path(gate_path).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace gate receipt: {output}")
    _atomic_json(output, gate)
    return gate


def publish_canary_receipt(
    receipt_path: str | Path,
    checkpoint_path: str | Path,
    run_name: str,
    completed_iterations: int,
    report_path: str | Path,
) -> dict[str, Any]:
    """Publish non-deployment hard-safety evidence for one interrupted checkpoint."""

    interval = resolve_stage_interval(completed_iterations)
    if not interval.interrupted or not is_v10_canary_checkpoint(
        completed_iterations
    ):
        raise ValueError(
            "Canary receipts require a deliberate segment end or an RSL-RL "
            "periodic recovery checkpoint inside a stage"
        )
    checkpoint, checkpoint_digest, training_digest = _current_checkpoint_identity(
        checkpoint_path, completed_iterations=completed_iterations
    )
    report = Path(report_path).resolve(strict=True)
    report_digest_before = sha256_file(report)
    _validate_report(
        report,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_digest,
        checkpoint_iteration=completed_iterations - 1,
        training_provenance_sha256=training_digest,
        seed=V10_CANARY_EVALUATION_SEEDS[0],
        scenarios=V10_CANARY_SCENARIOS,
        acceptance_profile=evaluator.CANARY_HARD_SAFETY_PROFILE,
        performance_enforced=False,
        moving_hmd=False,
        final_nominal=False,
    )
    report_digest_after = sha256_file(report)
    if report_digest_after != report_digest_before:
        raise RuntimeError(f"Evaluator report changed during validation: {report}")
    receipt = {
        "schema_version": V10_CANARY_RECEIPT_SCHEMA_VERSION,
        "receipt_kind": "canonical_v10_interrupted_canary",
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
        "evaluation_seeds": list(V10_CANARY_EVALUATION_SEEDS),
        "scenarios": list(V10_CANARY_SCENARIOS),
        "reports": [str(report)],
        "report_sha256": {str(report): report_digest_after},
        "moving_hmd_reports": [],
        "moving_hmd_report_sha256": {},
    }
    output = Path(receipt_path).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace canary receipt: {output}")
    _atomic_json(output, receipt)
    return receipt


def validate_stage_gate(
    gate_path: str | Path,
    *,
    expected_boundary: int,
    expected_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate an existing v10 boundary gate and every pinned artifact."""

    scenarios = scenarios_for_boundary(expected_boundary)
    performance = boundary_enforces_performance(expected_boundary)
    moving_required = boundary_requires_moving_hmd(expected_boundary)
    path = Path(gate_path).resolve(strict=True)
    gate_digest_before = sha256_file(path)
    gate = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(gate, dict):
        raise TypeError("V10 boundary gate must contain a JSON object")
    profile = (
        evaluator.DEPLOYMENT_PERFORMANCE_PROFILE
        if performance
        else evaluator.INTERMEDIATE_HARD_SAFETY_PROFILE
    )
    failures: list[str] = []
    expected_scalars = {
        "schema_version": V10_GATE_SCHEMA_VERSION,
        "status": "pass",
        "training_contract_version": MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION,
        "recipe_revision": MICROBAN_TELEOP_RECIPE_REVISION,
        "evaluator_revision": evaluator.TELEOP_EVALUATOR_REVISION,
        "acceptance_revision": evaluator.TELEOP_ACCEPTANCE_REVISION,
        "acceptance_profile": profile,
        "evaluator_source_sha256": sha256_file(Path(evaluator.__file__)),
        "completed_iterations": expected_boundary,
        "evaluation_seeds": list(V10_EVALUATION_SEEDS),
        "scenarios": list(scenarios) if scenarios is not None else "canonical",
    }
    failures.extend(
        key for key, value in expected_scalars.items() if gate.get(key) != value
    )
    checkpoint = Path(gate.get("checkpoint", "")).resolve()
    checkpoint_digest = gate.get("checkpoint_sha256")
    training_digest: str | None = None
    if not checkpoint.is_file() or sha256_file(checkpoint) != checkpoint_digest:
        failures.append("checkpoint")
    else:
        try:
            _, observed_checkpoint_digest, training_digest = (
                _current_checkpoint_identity(
                    checkpoint, completed_iterations=expected_boundary
                )
            )
        except (OSError, TypeError, ValueError):
            failures.append("checkpoint contract")
        else:
            if observed_checkpoint_digest != checkpoint_digest:
                failures.append("checkpoint SHA-256")
            if gate.get("training_provenance_sha256") != training_digest:
                failures.append("training provenance SHA-256")
            if gate.get("run_name") != checkpoint.parent.name:
                failures.append("run name")
    if (
        expected_checkpoint_sha256 is not None
        and checkpoint_digest != expected_checkpoint_sha256
    ):
        failures.append("expected checkpoint SHA-256")
    expected_counts = {
        "reports": len(V10_EVALUATION_SEEDS),
        "moving_hmd_reports": len(V10_EVALUATION_SEEDS) if moving_required else 0,
    }
    for paths_key, count in expected_counts.items():
        hashes_key = (
            "report_sha256"
            if paths_key == "reports"
            else "moving_hmd_report_sha256"
        )
        values = gate.get(paths_key)
        hashes = gate.get(hashes_key)
        if not isinstance(values, list) or len(values) != count:
            failures.append(paths_key)
            continue
        if not isinstance(hashes, dict) or set(hashes) != set(values):
            failures.append(hashes_key)
            continue
        for index, value in enumerate(values):
            report = Path(value)
            if not report.is_file():
                failures.append(f"changed report {value}")
                continue
            digest_before = sha256_file(report)
            if digest_before != hashes[value]:
                failures.append(f"changed report {value}")
                continue
            if training_digest is not None and isinstance(checkpoint_digest, str):
                try:
                    _validate_report(
                        report,
                        checkpoint=checkpoint,
                        checkpoint_sha256=checkpoint_digest,
                        checkpoint_iteration=expected_boundary - 1,
                        training_provenance_sha256=training_digest,
                        seed=V10_EVALUATION_SEEDS[index],
                        scenarios=scenarios,
                        acceptance_profile=profile,
                        performance_enforced=performance,
                        moving_hmd=paths_key == "moving_hmd_reports",
                        final_nominal=(
                            performance and paths_key == "reports"
                        ),
                    )
                except (OSError, TypeError, ValueError):
                    failures.append(f"invalid report {value}")
            if sha256_file(report) != digest_before:
                failures.append(f"report changed during validation {value}")
    if sha256_file(path) != gate_digest_before:
        failures.append("gate changed during validation")
    if failures:
        raise ValueError("Invalid v10 boundary gate: " + "; ".join(failures))
    return gate


def validate_canary_receipt(
    receipt_path: str | Path,
    *,
    checkpoint_path: str | Path,
    completed_iterations: int,
) -> dict[str, Any]:
    """Validate that an interrupted checkpoint passed its non-deployment canary."""

    receipt_file = Path(receipt_path).resolve(strict=True)
    receipt_digest_before = sha256_file(receipt_file)
    checkpoint = Path(checkpoint_path).resolve(strict=True)
    receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise TypeError("V10 canary receipt must contain a JSON object")
    interval = resolve_stage_interval(completed_iterations)
    if not interval.interrupted:
        raise ValueError("Canary receipt checkpoint must be inside a stage")
    if not is_v10_canary_checkpoint(completed_iterations):
        raise ValueError("Canary receipt checkpoint is not a canonical recovery save")
    _, checkpoint_digest, training_digest = _current_checkpoint_identity(
        checkpoint, completed_iterations=completed_iterations
    )
    expected = {
        "schema_version": V10_CANARY_RECEIPT_SCHEMA_VERSION,
        "receipt_kind": "canonical_v10_interrupted_canary",
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
        "evaluation_seeds": list(V10_CANARY_EVALUATION_SEEDS),
        "scenarios": list(V10_CANARY_SCENARIOS),
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
                    _validate_report(
                        report,
                        checkpoint=checkpoint,
                        checkpoint_sha256=checkpoint_digest,
                        checkpoint_iteration=completed_iterations - 1,
                        training_provenance_sha256=training_digest,
                        seed=V10_CANARY_EVALUATION_SEEDS[0],
                        scenarios=V10_CANARY_SCENARIOS,
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
        raise ValueError("Invalid v10 canary receipt: " + "; ".join(failures))
    return receipt


def _unique_gate_by_sha256(
    root: Path, pattern: str, expected_sha256: str
) -> Path:
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
    legacy_gate_root: str | Path,
    canary_root: str | Path,
) -> dict[str, Any]:
    """Validate lineage plus the hard-safety receipt before another segment."""

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
        require_current_source=True,
    )
    validate_canonical_stage_critical_config(manifest)
    invocation = manifest.get("invocation")
    if not isinstance(invocation, dict):
        raise TypeError("Checkpoint canonical invocation is malformed")
    expected_mode = (
        MICROBAN_TELEOP_CANONICAL_MIGRATION_STAGE_MODE
        if interval.migration
        else MICROBAN_TELEOP_CANONICAL_STAGE_MODE
    )
    expected_invocation = {
        "mode": expected_mode,
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
    if interval.migration:
        if parent_checkpoint_sha256 != MICROBAN_TELEOP_V10_LEGACY_CHECKPOINT_SHA256:
            raise ValueError("Migration parent checkpoint SHA-256 mismatch")
        if parent_gate_sha256 != MICROBAN_TELEOP_V10_LEGACY_GATE_SHA256:
            raise ValueError("Migration parent gate SHA-256 mismatch")
        legacy_gate = _unique_gate_by_sha256(
            Path(legacy_gate_root).resolve(),
            "*_boundary_1500_gate.json",
            parent_gate_sha256,
        )
        migration_source = validate_v10_migration_source_identity(
            manifest.get("migration_source")
        )
        legacy_checkpoint = Path(migration_source["source_checkpoint_path"])
        validate_legacy_migration_source(legacy_checkpoint, legacy_gate)
        parent_gate = legacy_gate
    else:
        if not isinstance(parent_checkpoint_sha256, str) or not isinstance(
            parent_gate_sha256, str
        ):
            raise ValueError("Interrupted stage is missing pinned parent SHAs")
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
            "Interrupted checkpoint requires exactly one passing canary receipt "
            f"({len(canary_matches)} found)"
        )
    return {
        "stage_start_boundary": interval.start_boundary,
        "stage_target_boundary": interval.target_boundary,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "parent_gate_sha256": parent_gate_sha256,
        "parent_gate": str(parent_gate),
        "canary_receipt": str(canary_matches[0]),
        "training_provenance_sha256": infos[
            MICROBAN_TELEOP_TRAINING_PROVENANCE_SHA256_KEY
        ],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    migration = subparsers.add_parser("validate-migration")
    migration.add_argument("checkpoint", type=Path)
    migration.add_argument("gate", type=Path)
    interrupted = subparsers.add_parser("validate-interrupted")
    interrupted.add_argument("checkpoint", type=Path)
    interrupted.add_argument("--completed-iterations", type=int, required=True)
    interrupted.add_argument("--stage-start-boundary", type=int, required=True)
    interrupted.add_argument("--stage-target-boundary", type=int, required=True)
    interrupted.add_argument("--gate-root", type=Path, required=True)
    interrupted.add_argument("--legacy-gate-root", type=Path, required=True)
    interrupted.add_argument("--canary-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "validate-migration":
        result = validate_legacy_migration_source(args.checkpoint, args.gate)
        print(result["checkpoint"])
        print(result["checkpoint_sha256"])
        print(result["gate"])
        print(result["gate_sha256"])
        return
    interval = StageInterval(
        start_boundary=args.stage_start_boundary,
        target_boundary=args.stage_target_boundary,
        interrupted=True,
        migration=(
            args.stage_start_boundary
            == MICROBAN_TELEOP_V10_MIGRATION_START_BOUNDARY
        ),
    )
    result = validate_interrupted_stage_checkpoint(
        args.checkpoint,
        completed_iterations=args.completed_iterations,
        interval=interval,
        gate_root=args.gate_root,
        legacy_gate_root=args.legacy_gate_root,
        canary_root=args.canary_root,
    )
    print(result["stage_start_boundary"])
    print(result["stage_target_boundary"])
    print(result["parent_checkpoint_sha256"])
    print(result["parent_gate_sha256"])


if __name__ == "__main__":
    main()
