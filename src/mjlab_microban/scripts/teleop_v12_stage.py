"""Create and validate hash-bound contract-v12 canary/boundary gates."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

from mjlab_microban.legacy_velocity_diagnostics import publish_json_atomic
from mjlab_microban.scripts.evaluate_teleop_v12_checkpoint import (
    MINIMUM_SIGNED_RESPONSE,
)
from mjlab_microban.scripts.evaluate_teleop_v12_checkpoint import (
    _acceptance as _locomotion_acceptance,
)
from mjlab_microban.scripts.evaluate_teleop_v12_tracking import (
    DIRECTIONAL_RESPONSE_MINIMUM,
    EXPANDED_LOCOMOTION_PROFILE,
    FINAL_PROFILE,
    FOOT_P95_MAX_M,
    FOOT_RMS_MAX_M,
    HAND_P95_MAX_M,
    HAND_RMS_MAX_M,
    HMD_ACTUAL_PEAK_TO_PEAK_MIN_RAD,
    HMD_TARGET_PEAK_TO_PEAK_MIN_RAD,
    TARGET_COLUMN_ABLATION_ACTION_DELTA_MIN,
    TARGET_COLUMN_ABLATION_METHOD,
    _aggregate_action_envelopes,
    required_tracking_check_names,
    required_tracking_profile,
    required_tracking_scenario_names,
    target_column_ablation_observation_columns,
)
from mjlab_microban.scripts.evaluate_teleop_v12_tracking import (
    _acceptance as _tracking_acceptance,
)
from mjlab_microban.scripts.evaluate_teleop_v12_tracking import (
    _scenarios as _tracking_scenarios,
)
from mjlab_microban.scripts.teleop_v12_bootstrap_gate import (
    ONNX_PARITY_TOLERANCE,
    PRISTINE_PARITY_TOLERANCE,
)
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_HMD_JOINT_NAMES,
    MICROBAN_TELEOP_ACTION_JOINT_NAMES,
)
from mjlab_microban.tasks.microban_teleop_v12_actor import (
    TELEOP_V12_ADAPTER_GRADIENT_SCHEDULE_REVISION,
    TELEOP_V12_ADAPTER_SANITIZATION_REVISION,
    TELEOP_V12_ADAPTER_SANITIZATION_SCHEMA_VERSION,
    TELEOP_V12_EXTRA_OBSERVATION_COLUMNS,
    teleop_v12_active_adapter_columns,
    teleop_v12_target_normalizer_metadata,
)
from mjlab_microban.tasks.microban_teleop_v12_bootstrap import (
    portable_bootstrap_artifact_path,
    resolve_bootstrap_artifact_path,
    sha256_file,
)
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MICROBAN_TELEOP_V12_RECIPE_REVISION,
    MICROBAN_TELEOP_V12_STAGE_BOUNDARIES,
)
from mjlab_microban.tasks.microban_teleop_v12_lr_order import (
    validate_bilateral_site_order_checkpoint,
)
from mjlab_microban.tasks.microban_teleop_v12_preview import (
    reject_preview_checkpoint,
)
from mjlab_microban.teleop_v12_safety import (
    ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD,
)

_VELOCITY_AXES = ("vx_m_s", "vy_m_s", "yaw_rad_s")
_LOCOMOTION_COMMANDS = {
    "neutral": (0.0, 0.0, 0.0),
    "forward_0p1": (0.1, 0.0, 0.0),
    "forward_0p2": (0.2, 0.0, 0.0),
    "backward_0p1": (-0.1, 0.0, 0.0),
    "backward_0p2": (-0.2, 0.0, 0.0),
    "lateral_left_0p1": (0.0, 0.1, 0.0),
    "lateral_right_0p1": (0.0, -0.1, 0.0),
    "yaw_left_0p5": (0.0, 0.0, 0.5),
    "yaw_right_0p5": (0.0, 0.0, -0.5),
}


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON duplicates key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> object:
    raise ValueError(f"JSON contains non-finite value {value!r}")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_nonfinite_json,
    )
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _require_report_identity(
    report: dict[str, Any], expected: dict[str, int | str], label: str
) -> None:
    identity = report.get("checkpoint")
    if not isinstance(identity, dict) or any(
        identity.get(name) != value for name, value in expected.items()
    ):
        raise ValueError(f"{label} report is not bound to this checkpoint")


def _require_exact_true_checks(
    report: dict[str, Any], expected_names: set[str] | frozenset[str], label: str
) -> dict[str, bool]:
    checks = report.get("checks")
    if not isinstance(checks, dict) or set(checks) != set(expected_names):
        raise ValueError(f"{label} report check set is incomplete or drifted")
    if not all(value is True for value in checks.values()):
        raise ValueError(f"{label} report contains a failing check")
    return checks


def _require_canonical_settings(
    report: dict[str, Any], expected: dict[str, object], label: str
) -> None:
    settings = report.get("settings")
    if not isinstance(settings, dict) or any(
        settings.get(name) != value for name, value in expected.items()
    ):
        raise ValueError(f"{label} report settings are not canonical")
    if not isinstance(settings.get("device"), str) or not settings["device"]:
        raise ValueError(f"{label} report device is missing")


def _validate_locomotion_report(
    report: dict[str, Any], expected_identity: dict[str, int | str]
) -> None:
    if (
        report.get("schema_version") != 1
        or report.get("gate") != "microban_teleop_v12_neutral_locomotion_9x300"
        or report.get("status") != "pass"
    ):
        raise ValueError("Locomotion report schema/status drifted")
    _require_report_identity(report, expected_identity, "Locomotion")
    _require_canonical_settings(
        report,
        {
            "seed": 42,
            "steps": 300,
            "settle_steps": 50,
            "action_clip": None,
            "previous_action": "raw_actor_output",
            "policy_observation_width": 83,
        },
        "Locomotion",
    )
    if report.get("thresholds") != {
        "actual_soft_limit_violation_rad_max": (
            ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD
        ),
        "minimum_signed_response": MINIMUM_SIGNED_RESPONSE,
    }:
        raise ValueError("Locomotion report thresholds drifted")
    results = report.get("results")
    expected_names = ("neutral", *MINIMUM_SIGNED_RESPONSE)
    if (
        not isinstance(results, list)
        or tuple(
            item.get("name") if isinstance(item, dict) else None for item in results
        )
        != expected_names
    ):
        raise ValueError("Locomotion report scenario set/order drifted")
    for result in results:
        assert isinstance(result, dict)
        name = result["name"]
        command = _LOCOMOTION_COMMANDS[name]
        expected_command = dict(zip(_VELOCITY_AXES, command, strict=True))
        measured = result.get("measured_velocity_body")
        if (
            result.get("command") != expected_command
            or not isinstance(measured, dict)
            or set(measured) != set(_VELOCITY_AXES)
            or any(
                not isinstance(measured.get(axis), dict)
                or measured[axis].get("count") != 250
                or not _finite_number(measured[axis].get("mean"))
                for axis in _VELOCITY_AXES
            )
            or result.get("completed") is not True
            or result.get("executed_steps") != 300
            or result.get("fell") is not False
            or result.get("nonfinite") is not None
            or result.get("termination_names") != []
            or result.get("raw_action_recurrence_verified_steps") != 300
            or result.get("neutral_foot_hand_target_verified_steps") != 300
            or not _finite_number(result.get("maximum_actual_soft_limit_violation_rad"))
            or float(result["maximum_actual_soft_limit_violation_rad"]) < 0.0
            or float(result["maximum_actual_soft_limit_violation_rad"])
            > ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD
        ):
            raise ValueError("Locomotion report scenario evidence failed")
        nonzero = [index for index, value in enumerate(command) if value != 0.0]
        response = result.get("directional_response")
        if not nonzero:
            if response is not None:
                raise ValueError("Neutral locomotion response must be absent")
            continue
        if len(nonzero) != 1 or not isinstance(response, dict):
            raise ValueError("Locomotion directional response is malformed")
        index = nonzero[0]
        axis = _VELOCITY_AXES[index]
        mean = float(measured[axis]["mean"])
        signed = mean * (1.0 if command[index] > 0.0 else -1.0)
        if (
            response.get("axis") != axis
            or response.get("command") != command[index]
            or response.get("measured_mean") != mean
            or response.get("signed_response") != signed
            or response.get("sign_matches") is not (signed > 0.0)
            or signed < MINIMUM_SIGNED_RESPONSE[name]
        ):
            raise ValueError("Locomotion directional evidence is inconsistent")
    try:
        recomputed, status = _locomotion_acceptance(results)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Locomotion result evidence is malformed") from exc
    checks = _require_exact_true_checks(
        report,
        {
            "all_scenarios_completed",
            "no_falls",
            "finite",
            "actual_soft_limits",
            "raw_action_recurrence",
            "neutral_targets",
            "directional_signs",
            "minimum_directional_response",
        },
        "Locomotion",
    )
    if status != "pass" or recomputed != checks:
        raise ValueError("Locomotion checks do not match result evidence")
    summary = report.get("summary")
    if not isinstance(summary, dict) or any(
        summary.get(name) != value
        for name, value in {
            "scenario_count": 9,
            "completed_scenario_count": 9,
            "fall_scenario_count": 0,
            "nonfinite_scenario_count": 0,
            "directionally_correct_scenario_count": 8,
            "directional_scenario_count": 8,
        }.items()
    ):
        raise ValueError("Locomotion report summary drifted")


def _require_action_summary(value: object, label: str) -> dict[str, list[float]]:
    if not isinstance(value, dict) or set(value) != {
        "minimum",
        "maximum",
        "absolute_maximum",
    }:
        raise ValueError(f"{label} action envelope fields drifted")
    parsed: dict[str, list[float]] = {}
    for name in ("minimum", "maximum", "absolute_maximum"):
        vector = value[name]
        if (
            not isinstance(vector, list)
            or len(vector) != 18
            or not all(_finite_number(item) for item in vector)
        ):
            raise ValueError(f"{label} action envelope {name} is malformed")
        parsed[name] = [float(item) for item in vector]
    for minimum, maximum, absolute in zip(
        parsed["minimum"],
        parsed["maximum"],
        parsed["absolute_maximum"],
        strict=True,
    ):
        if minimum > maximum or absolute != max(abs(minimum), abs(maximum)):
            raise ValueError(f"{label} action envelope is inconsistent")
    return parsed


def _require_action_envelope(
    value: object, *, label: str, aggregate: bool, scenario_count: int = 0
) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("joint_names") != list(
        MICROBAN_TELEOP_ACTION_JOINT_NAMES
    ):
        raise ValueError(f"{label} action joint order drifted")
    required = {
        "joint_names",
        "v12",
        "legacy_source",
        "learned_minus_source",
    }
    if aggregate:
        required.update(("scenario_count", "step_count"))
        if value.get("scenario_count") != scenario_count or not isinstance(
            value.get("step_count"), int
        ):
            raise ValueError(f"{label} aggregate counts drifted")
    if set(value) != required:
        raise ValueError(f"{label} action envelope fields drifted")
    for policy in ("v12", "legacy_source", "learned_minus_source"):
        _require_action_summary(value.get(policy), f"{label}/{policy}")
    return value


def _validate_tracking_report(
    report: dict[str, Any], expected_identity: dict[str, int | str]
) -> None:
    completed = int(expected_identity["completed_updates"])
    profile = required_tracking_profile(completed)
    if (
        report.get("schema_version") != 1
        or report.get("gate") != "microban_teleop_v12_tracking"
        or report.get("profile") != profile
        or report.get("status") != "pass"
    ):
        raise ValueError("Tracking report schema/profile/status drifted")
    _require_report_identity(report, expected_identity, "Tracking")
    _require_canonical_settings(
        report,
        {
            "seed": 42,
            "steps": 300,
            "settle_steps": 50,
            "moving_hmd": "forced_non_neutral",
            "perturbation": profile in (EXPANDED_LOCOMOTION_PROFILE, FINAL_PROFILE),
            "action_clip": None,
            "previous_action": "raw_actor_output",
            "target_column_ablation": TARGET_COLUMN_ABLATION_METHOD,
            "reachable_hand_target_fk": teleop_v12_target_normalizer_metadata()[
                "hand_target_fk"
            ],
        },
        "Tracking",
    )
    expected_thresholds = {
        "actual_soft_limit_violation_rad_max": (
            ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD
        ),
        "hmd_target_peak_to_peak_rad_min": HMD_TARGET_PEAK_TO_PEAK_MIN_RAD,
        "hmd_actual_peak_to_peak_rad_min": HMD_ACTUAL_PEAK_TO_PEAK_MIN_RAD,
        "hand_rms_m_max": HAND_RMS_MAX_M,
        "hand_p95_m_max": HAND_P95_MAX_M,
        "foot_rms_m_max": FOOT_RMS_MAX_M,
        "foot_p95_m_max": FOOT_P95_MAX_M,
        "directional_response_minimum": DIRECTIONAL_RESPONSE_MINIMUM,
        "target_column_ablation_action_delta_min": (
            TARGET_COLUMN_ABLATION_ACTION_DELTA_MIN
        ),
        "raw_action_amplitude": "reported_finite_only_no_invented_threshold",
    }
    if report.get("thresholds") != expected_thresholds:
        raise ValueError("Tracking report thresholds drifted")
    results = report.get("results")
    names = required_tracking_scenario_names(profile)
    if (
        not isinstance(results, list)
        or tuple(
            item.get("name") if isinstance(item, dict) else None for item in results
        )
        != names
    ):
        raise ValueError("Tracking report scenario set/order drifted")
    scenarios = _tracking_scenarios(profile)
    for result, scenario in zip(results, scenarios, strict=True):
        assert isinstance(result, dict)
        expected_command = {
            "twist": list(scenario.twist),
            "foot_target": [list(item) for item in scenario.foot_target],
            "hand_target": [list(item) for item in scenario.hand_target],
            "hand_active": list(scenario.hand_active),
        }
        coverage = result.get("observation_coverage")
        expects_foot = any(
            abs(item) > 0.0 for target in scenario.foot_target for item in target
        )
        expects_hand = any(scenario.hand_active)
        if (
            result.get("command") != expected_command
            or result.get("completed") is not True
            or result.get("executed_steps") != 300
            or result.get("fell") is not False
            or result.get("nonfinite") is not None
            or result.get("termination_names") != []
            or result.get("raw_action_recurrence_verified_steps") != 300
            or not _finite_number(result.get("maximum_actual_soft_limit_violation_rad"))
            or float(result["maximum_actual_soft_limit_violation_rad"]) < 0.0
            or float(result["maximum_actual_soft_limit_violation_rad"])
            > ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD
            or result.get("hmd_motion_evidence_passed") is not True
            or result.get("twist_directional_response_passed") is not True
            or not isinstance(coverage, dict)
            or coverage.get("passed") is not True
            or coverage.get("foot_target_expected") is not expects_foot
            or coverage.get("hand_target_expected") is not expects_hand
            or not isinstance(coverage.get("hmd_nonzero_steps"), int)
            or coverage["hmd_nonzero_steps"] <= 0
            or (expects_foot and coverage.get("foot_nonzero_steps") != 300)
            or (expects_hand and coverage.get("hand_nonzero_steps") != 300)
        ):
            raise ValueError("Tracking report scenario evidence failed")
        target_error = result.get("target_error")
        if not isinstance(target_error, dict) or set(target_error) != {
            "foot",
            "active_hand",
        }:
            raise ValueError("Tracking target-error evidence is malformed")
        active_foot_count = sum(
            any(abs(value) > 0.0 for value in target) for target in scenario.foot_target
        )
        active_hand_count = sum(bool(value) for value in scenario.hand_active)
        for target, expected, expected_samples in (
            ("foot", expects_foot, 250 * active_foot_count),
            ("active_hand", expects_hand, 250 * active_hand_count),
        ):
            values = target_error[target]
            if not isinstance(values, dict) or values.get("units") != "m":
                raise ValueError("Tracking target-error stats are malformed")
            metrics = ("min", "max", "mean", "rms", "p95")
            if expected:
                if (
                    values.get("sample_count") != expected_samples
                    or not all(_finite_number(values.get(name)) for name in metrics)
                    or not (
                        0.0
                        <= float(values["min"])
                        <= float(values["mean"])
                        <= float(values["rms"])
                        <= float(values["max"])
                    )
                    or not (
                        float(values["min"])
                        <= float(values["p95"])
                        <= float(values["max"])
                    )
                ):
                    raise ValueError("Tracking active target-error stats failed")
            elif values.get("sample_count") != 0 or any(
                values.get(name) is not None for name in metrics
            ):
                raise ValueError("Tracking inactive target-error stats drifted")
        ablation = result.get("target_column_ablation")
        if not isinstance(ablation, dict) or set(ablation) != {"hand", "foot"}:
            raise ValueError("Tracking target-column ablation evidence is malformed")
        fields = {
            "target_expected",
            "ablated_observation_columns",
            "preserved_observation_columns",
            "maximum_absolute_action_delta",
            "minimum_required_action_delta",
            "passed",
        }
        for target, expected in (("hand", expects_hand), ("foot", expects_foot)):
            evidence = ablation[target]
            ablated_columns, preserved_columns = (
                target_column_ablation_observation_columns(target)
            )
            if (
                not isinstance(evidence, dict)
                or set(evidence) != fields
                or evidence.get("target_expected") is not expected
                or evidence.get("ablated_observation_columns") != list(ablated_columns)
                or evidence.get("preserved_observation_columns")
                != list(preserved_columns)
            ):
                raise ValueError("Tracking target-column ablation target drifted")
            maximum = evidence.get("maximum_absolute_action_delta")
            if expected:
                response = (
                    _finite_number(maximum)
                    and float(maximum) >= 0.0
                    and float(maximum) > TARGET_COLUMN_ABLATION_ACTION_DELTA_MIN
                )
                if (
                    not _finite_number(maximum)
                    or float(maximum) < 0.0
                    or evidence.get("minimum_required_action_delta")
                    != TARGET_COLUMN_ABLATION_ACTION_DELTA_MIN
                    or evidence.get("passed") is not response
                ):
                    raise ValueError(
                        "Tracking target-column ablation evidence is inconsistent"
                    )
            elif (
                maximum is not None
                or evidence.get("minimum_required_action_delta") is not None
                or evidence.get("passed") is not True
            ):
                raise ValueError("Tracking inactive target-column ablation drifted")
        hmd = result.get("hmd_motion")
        per_axis = hmd.get("per_axis") if isinstance(hmd, dict) else None
        if (
            not isinstance(hmd, dict)
            or hmd.get("joint_names") != list(MICROBAN_HMD_JOINT_NAMES)
            or hmd.get("sample_count") != 301
            or hmd.get("active_event_member") is not True
            or not isinstance(per_axis, dict)
            or set(per_axis) != set(MICROBAN_HMD_JOINT_NAMES)
        ):
            raise ValueError("Tracking HMD motion evidence is malformed")
        for axis in MICROBAN_HMD_JOINT_NAMES:
            values = per_axis[axis]
            if (
                not isinstance(values, dict)
                or not _finite_number(values.get("target_peak_to_peak_rad"))
                or not _finite_number(values.get("actual_peak_to_peak_rad"))
                or float(values["target_peak_to_peak_rad"])
                < HMD_TARGET_PEAK_TO_PEAK_MIN_RAD
                or float(values["actual_peak_to_peak_rad"])
                < HMD_ACTUAL_PEAK_TO_PEAK_MIN_RAD
            ):
                raise ValueError("Tracking HMD per-axis motion evidence failed")
        response = result.get("directional_response")
        measured = result.get("measured_velocity_body")
        expected_axes = {
            axis
            for axis, command in zip(
                ("vx_m_s", "vy_m_s", "yaw_rad_s"), scenario.twist, strict=True
            )
            if command != 0.0
        }
        if (
            not isinstance(response, dict)
            or set(response) != expected_axes
            or not isinstance(measured, dict)
            or set(measured) != set(_VELOCITY_AXES)
            or any(
                not isinstance(measured.get(axis), dict)
                or measured[axis].get("sample_count") != 250
                or not _finite_number(measured[axis].get("mean"))
                for axis in _VELOCITY_AXES
            )
        ):
            raise ValueError("Tracking directional response axes drifted")
        for axis, command in zip(_VELOCITY_AXES, scenario.twist, strict=True):
            if command == 0.0:
                continue
            item = response[axis]
            mean = float(measured[axis]["mean"])
            signed = mean * (1.0 if command > 0.0 else -1.0)
            if (
                not isinstance(item, dict)
                or item.get("command") != command
                or item.get("measured_mean") != mean
                or item.get("signed_response") != signed
                or item.get("minimum_signed_response")
                != DIRECTIONAL_RESPONSE_MINIMUM[axis]
                or item.get("passed")
                is not (signed >= DIRECTIONAL_RESPONSE_MINIMUM[axis])
                or signed < DIRECTIONAL_RESPONSE_MINIMUM[axis]
            ):
                raise ValueError("Tracking directional response failed")
        _require_action_envelope(
            result.get("raw_action_envelope"),
            label=f"Tracking/{scenario.name}",
            aggregate=False,
        )
    try:
        recomputed, status = _tracking_acceptance(results, profile)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Tracking result evidence is malformed") from exc
    checks = _require_exact_true_checks(
        report, required_tracking_check_names(profile), "Tracking"
    )
    if status != "pass" or recomputed != checks:
        raise ValueError("Tracking checks do not match result evidence")
    envelope = _require_action_envelope(
        report.get("raw_action_envelope"),
        label="Tracking/aggregate",
        aggregate=True,
        scenario_count=len(results),
    )
    if envelope["step_count"] != 300 * len(results):
        raise ValueError("Tracking aggregate step count drifted")
    if envelope != _aggregate_action_envelopes(results):
        raise ValueError("Tracking aggregate action envelope is inconsistent")


def _validate_onnx_report(
    report: dict[str, Any], expected_identity: dict[str, int | str]
) -> tuple[Path, str]:
    if (
        report.get("schema_version") != 1
        or report.get("gate") != "microban_teleop_v12_checkpoint_onnx"
        or report.get("status") != "pass"
    ):
        raise ValueError("ONNX report schema/status drifted")
    _require_report_identity(report, expected_identity, "ONNX")
    neutral = report.get("neutral_legacy_parity")
    if (
        not isinstance(neutral, dict)
        or neutral.get("samples") != 10_000
        or neutral.get("teleop_only_columns") != "exact_zero"
        or neutral.get("tolerance") != PRISTINE_PARITY_TOLERANCE
        or not _finite_number(neutral.get("maximum_absolute_error"))
        or float(neutral["maximum_absolute_error"]) < 0.0
        or float(neutral["maximum_absolute_error"]) > PRISTINE_PARITY_TOLERANCE
    ):
        raise ValueError("ONNX neutral legacy parity evidence failed")
    onnx = report.get("onnx")
    if (
        not isinstance(onnx, dict)
        or onnx.get("opset") != 18
        or onnx.get("input_shape") != [1, 83]
        or onnx.get("output_shape") != [1, 18]
        or onnx.get("reference_samples") != 64
        or onnx.get("input_coverage") != "deterministic_nonzero_all_83_columns"
        or onnx.get("teleop_only_columns_nonzero") is not True
        or onnx.get("onnxruntime_providers") != ["CPUExecutionProvider"]
        or not isinstance(onnx.get("onnxruntime_version"), str)
        or not onnx["onnxruntime_version"]
        or onnx.get("tolerance") != ONNX_PARITY_TOLERANCE
    ):
        raise ValueError("ONNX full-83 CPU evidence drifted")
    for name in (
        "reference_evaluator_maximum_absolute_error",
        "onnxruntime_cpu_maximum_absolute_error",
    ):
        if (
            not _finite_number(onnx.get(name))
            or float(onnx[name]) < 0.0
            or float(onnx[name]) > ONNX_PARITY_TOLERANCE
        ):
            raise ValueError(f"ONNX parity evidence failed: {name}")
    onnx_path = resolve_bootstrap_artifact_path(onnx.get("path", ""))
    onnx_sha = onnx.get("sha256")
    if (
        not isinstance(onnx_sha, str)
        or len(onnx_sha) != 64
        or not onnx_path.is_file()
        or sha256_file(onnx_path) != onnx_sha
    ):
        raise ValueError("ONNX artifact path/hash mismatch")
    return onnx_path, onnx_sha


def _checkpoint_identity(path: Path) -> tuple[str, int, int, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("infos"), dict):
        raise TypeError("Checkpoint payload is malformed")
    if payload["infos"].get("microban_teleop_training_contract_version") != "12":
        raise ValueError("Checkpoint is not contract-v12")
    infos = payload["infos"]
    validate_bilateral_site_order_checkpoint(infos)
    reject_preview_checkpoint(infos)
    if infos.get("microban_teleop_recipe_revision") != (
        MICROBAN_TELEOP_V12_RECIPE_REVISION
    ) or infos.get("adapter_gradient_schedule_revision") != (
        TELEOP_V12_ADAPTER_GRADIENT_SCHEDULE_REVISION
    ):
        raise ValueError("Checkpoint is not the safe staged-mask v12 recipe")
    iteration = payload.get("iter")
    if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < 0:
        raise ValueError("Stage checkpoint iteration is invalid")
    completed = iteration + 1
    env_state = payload["infos"].get("env_state")
    if (
        not isinstance(env_state, dict)
        or env_state.get("common_step_counter") != completed * 24
    ):
        raise ValueError("Checkpoint common step does not match iteration")
    expected_active = list(teleop_v12_active_adapter_columns(completed * 24))
    if infos.get("active_actor_columns_at_save") != expected_active:
        raise ValueError("Checkpoint active adapter columns drifted from its clock")
    sanitization = infos.get("adapter_sanitization")
    if sanitization is not None:
        if not isinstance(sanitization, dict):
            raise ValueError("Checkpoint sanitization lineage is malformed")
        expected_sanitization = {
            "schema_version": TELEOP_V12_ADAPTER_SANITIZATION_SCHEMA_VERSION,
            "revision": TELEOP_V12_ADAPTER_SANITIZATION_REVISION,
            "completed_updates": sanitization.get("parent_iteration", -2) + 1,
            "zeroed_actor_columns": list(TELEOP_V12_EXTRA_OBSERVATION_COLUMNS),
            **teleop_v12_target_normalizer_metadata(),
        }
        if (
            any(
                sanitization.get(name) != value
                for name, value in expected_sanitization.items()
            )
            or not isinstance(sanitization.get("parent_checkpoint_sha256"), str)
            or len(sanitization["parent_checkpoint_sha256"]) != 64
        ):
            raise ValueError("Checkpoint sanitization lineage is malformed")
    return sha256_file(path), iteration, completed, infos


def _checkpoint_kind(completed: int, sanitization: object) -> str:
    if completed in MICROBAN_TELEOP_V12_STAGE_BOUNDARIES:
        return "canonical_boundary"
    if completed in (3_100, 7_100, 10_100):
        return "activation_canary"
    if (
        isinstance(sanitization, dict)
        and sanitization.get("completed_updates") == completed
    ):
        return "sanitized_recovery"
    return "interrupted_recovery"


def next_training_target(completed: int) -> tuple[int, bool]:
    """Return the next enforced endpoint and whether it is an activation canary."""

    if isinstance(completed, bool) or completed < 0:
        raise ValueError("completed updates must be a non-negative integer")
    if completed >= MICROBAN_TELEOP_V12_STAGE_BOUNDARIES[-1]:
        raise ValueError("Final 15000-update boundary already reached")
    for boundary, canary_end in ((3_000, 3_100), (7_000, 7_100), (10_000, 10_100)):
        if boundary <= completed < canary_end:
            return canary_end, True
    for boundary in MICROBAN_TELEOP_V12_STAGE_BOUNDARIES:
        if completed < boundary:
            return boundary, False
    raise AssertionError("Unreachable contract-v12 stage route")


def create_gate(
    *,
    checkpoint: Path,
    locomotion_report: Path,
    tracking_report: Path,
    onnx_report: Path,
) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    locomotion_report = locomotion_report.resolve()
    tracking_report = tracking_report.resolve()
    onnx_report = onnx_report.resolve()
    checkpoint_sha, iteration, completed, infos = _checkpoint_identity(checkpoint)
    locomotion = _load_json(locomotion_report)
    tracking = _load_json(tracking_report)
    onnx = _load_json(onnx_report)
    expected_report_identity = {
        "sha256": checkpoint_sha,
        "iteration": iteration,
        "completed_updates": completed,
    }
    _validate_locomotion_report(locomotion, expected_report_identity)
    _validate_tracking_report(tracking, expected_report_identity)
    onnx_path, onnx_sha = _validate_onnx_report(onnx, expected_report_identity)
    canonical = completed in MICROBAN_TELEOP_V12_STAGE_BOUNDARIES
    sanitization = infos.get("adapter_sanitization")
    return {
        "schema_version": 2,
        "gate": "microban_teleop_v12_stage",
        "status": "pass",
        "checkpoint": portable_bootstrap_artifact_path(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "iteration": iteration,
        "completed_updates": completed,
        "canonical_boundary": canonical,
        "checkpoint_kind": _checkpoint_kind(completed, sanitization),
        "tracking_profile": required_tracking_profile(completed),
        "adapter_sanitization": sanitization,
        "reports": {
            "locomotion": portable_bootstrap_artifact_path(locomotion_report),
            "tracking": portable_bootstrap_artifact_path(tracking_report),
            "onnx": portable_bootstrap_artifact_path(onnx_report),
        },
        "report_sha256": {
            "locomotion": sha256_file(locomotion_report),
            "tracking": sha256_file(tracking_report),
            "onnx": sha256_file(onnx_report),
        },
        "onnx": {
            "path": portable_bootstrap_artifact_path(onnx_path),
            "sha256": onnx_sha,
        },
    }


def validate_gate(gate_path: Path, checkpoint: Path) -> dict[str, Any]:
    gate_path = gate_path.resolve()
    checkpoint = checkpoint.resolve()
    gate = _load_json(gate_path)
    checkpoint_sha, iteration, completed, infos = _checkpoint_identity(checkpoint)
    canonical = completed in MICROBAN_TELEOP_V12_STAGE_BOUNDARIES
    sanitization = infos.get("adapter_sanitization")
    exact = {
        "schema_version": 2,
        "gate": "microban_teleop_v12_stage",
        "status": "pass",
        "checkpoint": portable_bootstrap_artifact_path(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "iteration": iteration,
        "completed_updates": completed,
        "canonical_boundary": canonical,
        "checkpoint_kind": _checkpoint_kind(completed, sanitization),
        "tracking_profile": required_tracking_profile(completed),
        "adapter_sanitization": sanitization,
    }
    if any(gate.get(name) != value for name, value in exact.items()):
        raise ValueError("V12 stage gate identity mismatch")
    reports = gate.get("reports")
    report_hashes = gate.get("report_sha256")
    if not isinstance(reports, dict) or not isinstance(report_hashes, dict):
        raise TypeError("V12 stage gate report references are malformed")
    for name in ("locomotion", "tracking", "onnx"):
        report = resolve_bootstrap_artifact_path(reports.get(name, ""))
        if not report.is_file() or sha256_file(report) != report_hashes.get(name):
            raise ValueError(f"V12 stage gate {name} report changed")
    rebuilt = create_gate(
        checkpoint=checkpoint,
        locomotion_report=resolve_bootstrap_artifact_path(reports["locomotion"]),
        tracking_report=resolve_bootstrap_artifact_path(reports["tracking"]),
        onnx_report=resolve_bootstrap_artifact_path(reports["onnx"]),
    )
    if rebuilt != gate:
        raise ValueError("V12 stage gate content is not canonical")
    onnx_path = resolve_bootstrap_artifact_path(gate["onnx"]["path"])
    if not onnx_path.is_file() or sha256_file(onnx_path) != gate["onnx"]["sha256"]:
        raise ValueError("V12 stage ONNX artifact changed")
    return gate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("checkpoint", type=Path)
    create.add_argument("locomotion_report", type=Path)
    create.add_argument("tracking_report", type=Path)
    create.add_argument("onnx_report", type=Path)
    create.add_argument("output", type=Path)
    create.add_argument("--force", action="store_true")
    validate = subparsers.add_parser("validate")
    validate.add_argument("gate", type=Path)
    validate.add_argument("checkpoint", type=Path)
    route = subparsers.add_parser("route")
    route.add_argument("completed_updates", type=int)
    route.add_argument("--shell", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "route":
        target, mandatory = next_training_target(args.completed_updates)
        if args.shell:
            print(f"{target} {int(mandatory)}", flush=True)
        else:
            print(
                json.dumps(
                    {
                        "completed_updates": args.completed_updates,
                        "target_updates": target,
                        "mandatory_activation_canary": mandatory,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        return 0
    if args.command == "create":
        if args.output.exists() and not args.force:
            raise FileExistsError("Gate exists (pass --force)")
        gate = create_gate(
            checkpoint=args.checkpoint,
            locomotion_report=args.locomotion_report,
            tracking_report=args.tracking_report,
            onnx_report=args.onnx_report,
        )
        publish_json_atomic(args.output, gate)
    else:
        gate = validate_gate(args.gate, args.checkpoint)
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
