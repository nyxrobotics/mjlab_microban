# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Pure receipt/check helpers for the GPU locomotion-prior dynamics gate.

The simulator entry point lives in ``scripts.evaluate_locomotion_prior_dynamics``.
Keeping receipt validation and pass/fail arithmetic here makes the safety contract
unit-testable without importing torch, Warp, or allocating a GPU context.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mjlab_microban.locomotion_prior_suitability import (
    DEFAULT_END_FRAME,
    DEFAULT_START_FRAME,
    LOCOMOTION_PRIOR_SUITABILITY_SCHEMA_VERSION,
    canonical_json_sha256,
    sha256_file,
    validate_suitability_receipt_digest,
)

DYNAMIC_GATE_SCHEMA_VERSION = 2
DYNAMIC_GATE_REVISION = "microban_locomotion_prior_dynamic_gate_v2"
DYNAMIC_GATE_NUM_ENVS = 256
DYNAMIC_GATE_SEED = 42
DYNAMIC_GATE_EXPECTED_POLICY_STEPS = 159


@dataclass(frozen=True)
class LocomotionPriorDynamicThresholds:
    """Required bounds for the full-clip actuator replay."""

    minimum_root_height_m: float = 0.10
    maximum_soft_limit_violation_rad: float = 1.0e-6
    maximum_target_projection_rad: float = 0.001
    minimum_forward_velocity_p05_m_s: float = 0.05
    maximum_xy_velocity_mae_p95_m_s: float = 0.075

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(
                    f"Dynamic threshold {name} must be finite/non-negative"
                )


DEFAULT_DYNAMIC_THRESHOLDS = LocomotionPriorDynamicThresholds()


def summarize_first_exit_steps(
    steps: Sequence[int], *, expected_envs: int = DYNAMIC_GATE_NUM_ENVS
) -> dict[str, Any]:
    """Summarize per-environment first-exit steps deterministically.

    ``-1`` means the environment never exited during the bounded replay.  The
    evaluator uses the same representation for first termination, first removal,
    and successful-completion steps, so the receipt makes staggered failures
    explicit instead of collapsing them into one global stop step.
    """

    if (
        isinstance(expected_envs, bool)
        or not isinstance(expected_envs, int)
        or expected_envs <= 0
    ):
        raise ValueError("expected_envs must be a positive integer")
    if len(steps) != expected_envs:
        raise ValueError(
            f"Expected {expected_envs} exit-step values, observed {len(steps)}"
        )
    normalized: list[int] = []
    for value in steps:
        if isinstance(value, bool) or not isinstance(value, int) or value < -1:
            raise ValueError("Exit steps must be integers greater than or equal to -1")
        normalized.append(value)
    observed = [value for value in normalized if value >= 0]
    counts = {str(step): observed.count(step) for step in sorted(set(observed))}
    return {
        "counts_by_step": counts,
        "earliest_zero_based": min(observed) if observed else None,
        "exited_envs": len(observed),
        "latest_zero_based": max(observed) if observed else None,
        "not_exited_envs": expected_envs - len(observed),
    }


def load_and_validate_static_receipt(
    path: Path,
    *,
    expected_prior_sha256: str,
    expected_robot_xml_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Load the prerequisite receipt and bind it to both dynamic inputs."""

    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Static suitability receipt does not exist: {path}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Static suitability receipt is not valid JSON: {path}"
        ) from exc
    if not isinstance(report, dict):
        raise TypeError("Static suitability receipt root must be an object")
    validate_suitability_receipt_digest(report)
    if report.get("schema_version") != LOCOMOTION_PRIOR_SUITABILITY_SCHEMA_VERSION:
        raise ValueError("Static suitability receipt schema version mismatch")
    if (
        report.get("status") != "pass"
        or report.get("summary", {}).get("passed") is not True
    ):
        raise ValueError("Static suitability gate did not pass")
    configuration = report.get("configuration", {})
    if (
        configuration.get("start_frame_inclusive") != DEFAULT_START_FRAME
        or configuration.get("end_frame_inclusive") != DEFAULT_END_FRAME
    ):
        raise ValueError("Static suitability receipt frame window mismatch")
    inputs = report.get("inputs", {})
    prior = inputs.get("locomotion_prior", {})
    robot = inputs.get("robot_xml", {})
    if prior.get("sha256") != expected_prior_sha256:
        raise ValueError("Static receipt locomotion-prior SHA-256 mismatch")
    if robot.get("sha256") != expected_robot_xml_sha256:
        raise ValueError("Static receipt robot XML SHA-256 mismatch")
    provenance = prior.get("provenance")
    if not isinstance(provenance, dict):
        raise TypeError("Static receipt is missing producer provenance")
    if provenance.get("retarget_model_sha256") != expected_robot_xml_sha256:
        raise ValueError("Static receipt retarget-model SHA-256 mismatch")
    return report, sha256_file(path)


def dynamic_gate_checks(
    measurements: Mapping[str, Any],
    *,
    thresholds: LocomotionPriorDynamicThresholds = DEFAULT_DYNAMIC_THRESHOLDS,
    num_envs: int = DYNAMIC_GATE_NUM_ENVS,
) -> dict[str, dict[str, Any]]:
    """Turn aggregate simulator measurements into explicit acceptance checks."""

    thresholds.validate()
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
        raise ValueError("num_envs must be a positive integer")

    def exact_zero(name: str) -> dict[str, Any]:
        value = int(measurements[name])
        return {"maximum": 0, "passed": value == 0, "value": value}

    checks: dict[str, dict[str, Any]] = {
        "executed_policy_steps": {
            "expected": DYNAMIC_GATE_EXPECTED_POLICY_STEPS,
            "passed": int(measurements["executed_policy_steps"])
            == DYNAMIC_GATE_EXPECTED_POLICY_STEPS,
            "value": int(measurements["executed_policy_steps"]),
        },
        "full_clip_completed_envs": {
            "expected": num_envs,
            "passed": int(measurements["full_clip_completed_envs"]) == num_envs,
            "value": int(measurements["full_clip_completed_envs"]),
        },
        "fall_envs": exact_zero("fall_envs"),
        "unexpected_termination_envs": exact_zero("unexpected_termination_envs"),
        "nonfinite_envs": exact_zero("nonfinite_envs"),
        "self_collision_contacts": exact_zero("self_collision_contacts"),
        "source_frame_sequence_complete": {
            "expected": True,
            "passed": measurements["source_frame_sequence_complete"] is True,
            "value": measurements["source_frame_sequence_complete"],
        },
        "minimum_root_height_m": {
            "minimum": thresholds.minimum_root_height_m,
            "passed": float(measurements["minimum_root_height_m"])
            >= thresholds.minimum_root_height_m,
            "value": float(measurements["minimum_root_height_m"]),
        },
        "maximum_soft_limit_violation_rad": {
            "maximum": thresholds.maximum_soft_limit_violation_rad,
            "passed": float(measurements["maximum_soft_limit_violation_rad"])
            <= thresholds.maximum_soft_limit_violation_rad,
            "value": float(measurements["maximum_soft_limit_violation_rad"]),
        },
        "maximum_target_projection_rad": {
            "maximum": thresholds.maximum_target_projection_rad,
            "passed": float(measurements["maximum_target_projection_rad"])
            <= thresholds.maximum_target_projection_rad,
            "value": float(measurements["maximum_target_projection_rad"]),
        },
        "left_foot_airborne_envs": {
            "expected": num_envs,
            "passed": int(measurements["left_foot_airborne_envs"]) == num_envs,
            "value": int(measurements["left_foot_airborne_envs"]),
        },
        "right_foot_airborne_envs": {
            "expected": num_envs,
            "passed": int(measurements["right_foot_airborne_envs"]) == num_envs,
            "value": int(measurements["right_foot_airborne_envs"]),
        },
        "simultaneous_flight_env_steps": exact_zero("simultaneous_flight_env_steps"),
        "forward_velocity_p05_m_s": {
            "minimum": thresholds.minimum_forward_velocity_p05_m_s,
            "passed": float(measurements["forward_velocity_p05_m_s"])
            >= thresholds.minimum_forward_velocity_p05_m_s,
            "value": float(measurements["forward_velocity_p05_m_s"]),
        },
        "xy_velocity_mae_p95_m_s": {
            "maximum": thresholds.maximum_xy_velocity_mae_p95_m_s,
            "passed": float(measurements["xy_velocity_mae_p95_m_s"])
            <= thresholds.maximum_xy_velocity_mae_p95_m_s,
            "value": float(measurements["xy_velocity_mae_p95_m_s"]),
        },
    }
    for name, check in checks.items():
        value = check["value"]
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Dynamic measurement {name} must be finite")
    return checks


def finalize_dynamic_report(payload: dict[str, Any]) -> dict[str, Any]:
    """Add status, failed-check list, and a canonical tamper-evident digest."""

    if "receipt_payload_sha256" in payload:
        raise ValueError("Dynamic payload must not already contain a receipt digest")
    checks = payload.get("checks")
    if not isinstance(checks, dict) or not checks:
        raise ValueError("Dynamic payload must contain non-empty checks")
    failed = sorted(
        name for name, check in checks.items() if check.get("passed") is not True
    )
    complete = {
        **payload,
        "schema_version": DYNAMIC_GATE_SCHEMA_VERSION,
        "audit_revision": DYNAMIC_GATE_REVISION,
        "status": "pass" if not failed else "fail",
        "summary": {"failed_checks": failed, "passed": not failed},
    }
    return {**complete, "receipt_payload_sha256": canonical_json_sha256(complete)}


def validate_dynamic_receipt_digest(report: dict[str, Any]) -> None:
    """Reject missing or stale dynamic receipt payload digests."""

    recorded = report.get("receipt_payload_sha256")
    if not isinstance(recorded, str):
        raise TypeError("Dynamic receipt payload SHA-256 is missing")
    payload = dict(report)
    del payload["receipt_payload_sha256"]
    if canonical_json_sha256(payload) != recorded:
        raise ValueError("Dynamic receipt payload SHA-256 mismatch")
