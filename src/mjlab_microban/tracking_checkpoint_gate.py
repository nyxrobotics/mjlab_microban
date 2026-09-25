# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Pure acceptance and receipt helpers for Microban tracking checkpoints.

The simulator entry point is
``mjlab_microban.scripts.evaluate_tracking_checkpoint``.  This module contains
only deterministic arithmetic and JSON receipt validation so the safety gate can
be exercised on a CPU without importing torch, Warp, or creating a simulator.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

TRACKING_CHECKPOINT_GATE_SCHEMA_VERSION = 1
TRACKING_CHECKPOINT_GATE_REVISION = "microban_tracking_checkpoint_gate_v1"
TRACKING_CHECKPOINT_GATE_NUM_ENVS = 256
TRACKING_CHECKPOINT_GATE_SEED = 42
TRACKING_CHECKPOINT_GATE_INITIAL_FRAME = 0
TRACKING_CHECKPOINT_GATE_FIRST_TARGET_FRAME = 1
TRACKING_CHECKPOINT_GATE_END_FRAME = 267
TRACKING_CHECKPOINT_GATE_FRAME_COUNT = (
    TRACKING_CHECKPOINT_GATE_END_FRAME - TRACKING_CHECKPOINT_GATE_INITIAL_FRAME + 1
)
TRACKING_CHECKPOINT_GATE_EXPECTED_STEPS = (
    TRACKING_CHECKPOINT_GATE_END_FRAME - TRACKING_CHECKPOINT_GATE_FIRST_TARGET_FRAME + 1
)
TRACKING_CHECKPOINT_GATE_MODES = ("nominal", "robust")

# Exact concatenation order produced by make_microban_tracking_env_cfg().  Widths
# are kept beside the receipt logic because the current tracking ONNX metadata
# records names but not term widths.
TRACKING_ACTOR_OBSERVATION_SCHEMA: tuple[tuple[str, int], ...] = (
    ("command", 36),
    ("motion_anchor_ori_b", 6),
    ("base_ang_vel", 3),
    ("joint_pos", 18),
    ("joint_vel", 18),
    ("actions", 18),
)
TRACKING_ACTOR_OBSERVATION_WIDTH = sum(
    width for _, width in TRACKING_ACTOR_OBSERVATION_SCHEMA
)


@dataclass(frozen=True)
class TrackingCheckpointThresholds:
    """Conservative thresholds inherited from existing Microban safety gates.

    The geometry/dynamics prior gate already requires a 10 cm minimum root
    height, at least 5 cm/s p05 forward velocity, at most 7.5 cm/s p95 XY
    velocity error, at most 1 mrad target projection and no flight.  The teleop
    checkpoint gate already limits target clipping to 0.1% and actual soft-limit
    violation to one microradian.  Reusing those values avoids silently making a
    new checkpoint evaluator easier than either prerequisite.
    """

    minimum_root_height_m: float = 0.10
    maximum_actual_soft_limit_violation_rad: float = 1.0e-6
    maximum_target_projection_rad: float = 0.001
    maximum_target_clip_fraction: float = 0.001
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
                    f"Tracking checkpoint threshold {name} must be finite and "
                    "non-negative"
                )


DEFAULT_TRACKING_CHECKPOINT_THRESHOLDS = TrackingCheckpointThresholds()

TRACKING_ACTOR_KIND_UNBOUNDED_SCALAR = "unbounded_gaussian_scalar"
TRACKING_ACTOR_KIND_UNBOUNDED_LOG = "unbounded_gaussian_log"
TRACKING_ACTOR_KIND_BOUNDED_LOG = "microban_bounded_gaussian_log"


def classify_tracking_actor_state_keys(keys: Iterable[str]) -> str:
    """Classify checkpoint distribution state without importing torch.

    The model_250 diagnostic predates the bounded-action actor now used for new
    training.  The evaluator must instantiate the architecture recorded by the
    checkpoint rather than silently reinterpret old weights with today's config.
    """

    normalized = set(keys)
    bounded = any(
        key.endswith(
            ("distribution.lower_bound", "distribution.operational_lower_bound")
        )
        for key in normalized
    )
    scalar_std = "std" in normalized or any(
        key.endswith("distribution.std_param") for key in normalized
    )
    log_std = "log_std" in normalized or any(
        key.endswith("distribution.log_std_param") for key in normalized
    )
    if bounded:
        if scalar_std or not log_std:
            raise ValueError(
                "Bounded tracking checkpoint must contain only log-std state"
            )
        return TRACKING_ACTOR_KIND_BOUNDED_LOG
    if scalar_std and log_std:
        raise ValueError("Tracking checkpoint contains ambiguous std state")
    if scalar_std:
        return TRACKING_ACTOR_KIND_UNBOUNDED_SCALAR
    if log_std:
        return TRACKING_ACTOR_KIND_UNBOUNDED_LOG
    raise ValueError("Tracking checkpoint actor distribution state is missing")


def canonical_json_sha256(value: object) -> str:
    """Hash a stable, whitespace-free JSON representation."""

    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_number(measurements: Mapping[str, Any], name: str) -> float:
    value = measurements.get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"Tracking measurement {name} must be finite")
    return float(value)


def _non_negative_integer(measurements: Mapping[str, Any], name: str) -> int:
    value = measurements.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Tracking measurement {name} must be a non-negative integer")
    return value


def tracking_checkpoint_checks(
    measurements: Mapping[str, Any],
    *,
    thresholds: TrackingCheckpointThresholds = (DEFAULT_TRACKING_CHECKPOINT_THRESHOLDS),
    num_envs: int = TRACKING_CHECKPOINT_GATE_NUM_ENVS,
    expected_steps: int = TRACKING_CHECKPOINT_GATE_EXPECTED_STEPS,
) -> dict[str, dict[str, Any]]:
    """Convert one nominal/robust rollout into explicit fail-closed checks."""

    thresholds.validate()
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
        raise ValueError("num_envs must be a positive integer")
    if (
        isinstance(expected_steps, bool)
        or not isinstance(expected_steps, int)
        or expected_steps <= 0
    ):
        raise ValueError("expected_steps must be a positive integer")

    def exact_zero(name: str) -> dict[str, Any]:
        value = _non_negative_integer(measurements, name)
        return {"maximum": 0, "passed": value == 0, "value": value}

    executed_steps = _non_negative_integer(measurements, "executed_policy_steps")
    completed = _non_negative_integer(measurements, "full_clip_completed_envs")
    left_airborne = _non_negative_integer(measurements, "left_foot_airborne_envs")
    right_airborne = _non_negative_integer(measurements, "right_foot_airborne_envs")
    source_complete = measurements.get("source_frame_sequence_complete")
    if type(source_complete) is not bool:
        raise ValueError(
            "Tracking measurement source_frame_sequence_complete must be boolean"
        )

    root_height = _finite_number(measurements, "minimum_root_height_m")
    soft_violation = _finite_number(
        measurements, "maximum_actual_soft_limit_violation_rad"
    )
    target_projection = _finite_number(measurements, "maximum_target_projection_rad")
    target_clip_fraction = _finite_number(measurements, "target_clip_fraction")
    forward_velocity = _finite_number(measurements, "forward_velocity_p05_m_s")
    xy_velocity_mae = _finite_number(measurements, "xy_velocity_mae_p95_m_s")
    forward_displacement = _finite_number(measurements, "forward_displacement_p05_m")

    # A strictly positive displacement is an independent direction check.  The
    # velocity threshold remains the quantitative capability check, so no new,
    # unvalidated displacement magnitude is invented here.
    checks: dict[str, dict[str, Any]] = {
        "executed_policy_steps": {
            "expected": expected_steps,
            "passed": executed_steps == expected_steps,
            "value": executed_steps,
        },
        "full_clip_completed_envs": {
            "expected": num_envs,
            "passed": completed == num_envs,
            "value": completed,
        },
        "fall_envs": exact_zero("fall_envs"),
        "unexpected_termination_envs": exact_zero("unexpected_termination_envs"),
        "nonfinite_envs": exact_zero("nonfinite_envs"),
        "self_collision_contacts": exact_zero("self_collision_contacts"),
        "source_frame_sequence_complete": {
            "expected": True,
            "passed": source_complete,
            "value": source_complete,
        },
        "minimum_root_height_m": {
            "minimum": thresholds.minimum_root_height_m,
            "passed": root_height >= thresholds.minimum_root_height_m,
            "value": root_height,
        },
        "maximum_actual_soft_limit_violation_rad": {
            "maximum": thresholds.maximum_actual_soft_limit_violation_rad,
            "passed": (
                soft_violation <= thresholds.maximum_actual_soft_limit_violation_rad
            ),
            "value": soft_violation,
        },
        "maximum_target_projection_rad": {
            "maximum": thresholds.maximum_target_projection_rad,
            "passed": (target_projection <= thresholds.maximum_target_projection_rad),
            "value": target_projection,
        },
        "target_clip_fraction": {
            "maximum": thresholds.maximum_target_clip_fraction,
            "passed": target_clip_fraction <= thresholds.maximum_target_clip_fraction,
            "value": target_clip_fraction,
        },
        "left_foot_airborne_envs": {
            "expected": num_envs,
            "passed": left_airborne == num_envs,
            "value": left_airborne,
        },
        "right_foot_airborne_envs": {
            "expected": num_envs,
            "passed": right_airborne == num_envs,
            "value": right_airborne,
        },
        "simultaneous_flight_env_steps": exact_zero("simultaneous_flight_env_steps"),
        "forward_velocity_p05_m_s": {
            "minimum": thresholds.minimum_forward_velocity_p05_m_s,
            "passed": forward_velocity >= thresholds.minimum_forward_velocity_p05_m_s,
            "value": forward_velocity,
        },
        "forward_displacement_direction": {
            "minimum_exclusive_m": 0.0,
            "passed": forward_displacement > 0.0,
            "value": forward_displacement,
        },
        "xy_velocity_mae_p95_m_s": {
            "maximum": thresholds.maximum_xy_velocity_mae_p95_m_s,
            "passed": xy_velocity_mae <= thresholds.maximum_xy_velocity_mae_p95_m_s,
            "value": xy_velocity_mae,
        },
    }
    return checks


def finalize_tracking_checkpoint_report(payload: dict[str, Any]) -> dict[str, Any]:
    """Add aggregate pass/fail state and a tamper-evident receipt digest."""

    if "receipt_payload_sha256" in payload:
        raise ValueError("Tracking payload must not already contain a receipt digest")
    passes = payload.get("passes")
    if not isinstance(passes, dict) or tuple(passes) != TRACKING_CHECKPOINT_GATE_MODES:
        raise ValueError(
            "Tracking payload must contain ordered nominal and robust passes"
        )

    failed: list[str] = []
    for mode in TRACKING_CHECKPOINT_GATE_MODES:
        mode_report = passes.get(mode)
        if not isinstance(mode_report, dict):
            raise TypeError(f"Tracking pass {mode} must be an object")
        checks = mode_report.get("checks")
        if not isinstance(checks, dict) or not checks:
            raise ValueError(f"Tracking pass {mode} must contain checks")
        failed.extend(
            f"{mode}.{name}"
            for name, check in checks.items()
            if not isinstance(check, dict) or check.get("passed") is not True
        )

    complete = {
        **payload,
        "schema_version": TRACKING_CHECKPOINT_GATE_SCHEMA_VERSION,
        "audit_revision": TRACKING_CHECKPOINT_GATE_REVISION,
        "status": "pass" if not failed else "fail",
        "summary": {"failed_checks": sorted(failed), "passed": not failed},
    }
    return {
        **complete,
        "receipt_payload_sha256": canonical_json_sha256(complete),
    }


def validate_tracking_checkpoint_receipt_digest(report: dict[str, Any]) -> None:
    """Reject a missing, malformed, or stale receipt digest."""

    recorded = report.get("receipt_payload_sha256")
    if (
        not isinstance(recorded, str)
        or len(recorded) != 64
        or any(character not in "0123456789abcdef" for character in recorded)
    ):
        raise ValueError("Tracking receipt SHA-256 is missing or malformed")
    payload = dict(report)
    del payload["receipt_payload_sha256"]
    expected = canonical_json_sha256(payload)
    if recorded != expected:
        raise ValueError("Tracking receipt payload SHA-256 mismatch")


def tracking_checkpoint_exit_code(report: Mapping[str, Any]) -> int:
    """Return zero only for an explicitly passing canonical receipt."""

    return 0 if report.get("status") == "pass" else 2


if TRACKING_ACTOR_OBSERVATION_WIDTH != 99:
    raise RuntimeError(
        "Microban tracking actor observation schema must total 99 values, got "
        f"{TRACKING_ACTOR_OBSERVATION_WIDTH}"
    )
