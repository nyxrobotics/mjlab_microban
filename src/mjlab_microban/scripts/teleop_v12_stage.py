"""Create and validate the hash-bound gate of the checkpoint a PICO run ends with.

A PICO run (one process, mjlab_microban/schedules.py) is judged once: the locomotion
(9x300, seed 42), PICO judgment (seeds 42 and 43) and ONNX reports of the
checkpoint it ends with (its last update, ``PICO_TOTAL_UPDATES``) are
validated and bound into one gate file, which the packager requires.

A PICO judgment whose behaviour checks failed can still be gated when the user
approves installing it anyway: ``create --waive-tracking-checks NAMES
--waiver-note TEXT`` names exactly the failing checks.  The gate then has the
status ``pass_with_waiver`` and records the approval and every failing
measurement of each waived check (``tracking_waiver``); the packager copies the
record into the ONNX metadata.  Only behaviour checks
(``policy_contract.PICO_WAIVABLE_TRACKING_CHECKS``) can be waived: the safety
and integrity checks of the PICO judgment, the 9x300 locomotion and the ONNX
gate cannot.  Without a waiver nothing changes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from mjlab_microban import policy_contract
from mjlab_microban.legacy_velocity_diagnostics import publish_json_atomic
from mjlab_microban.schedules import PICO_TOTAL_UPDATES
from mjlab_microban.scripts.evaluate_teleop_v12_checkpoint import (
    LOCOMOTION_MOVING_SCENARIOS,
    locomotion_twist_judgments,
)
from mjlab_microban.scripts.evaluate_teleop_v12_checkpoint import (
    _acceptance as _locomotion_acceptance,
)
from mjlab_microban.scripts.evaluate_teleop_v12_tracking import (
    CHECK_NAMES as TRACKING_CHECK_NAMES,
)
from mjlab_microban.scripts.evaluate_teleop_v12_tracking import (
    FOOT_AIR_MIN_SHARE,
    FOOT_LIFT_MIN_SHARE,
    PUSH_FALL_RATE_MARGIN,
    SEED_COUNT,
    STILL_TOUCHDOWNS_PER_S_MAX,
    SUPPORT_FOOT_MOVE_MAX_M,
    ZMP_CENTRE_MAX_M,
    arm_walk_min_speed,
    canonical_settings,
    crouch_drop_error_m,
    foot_error_limit_m,
    push_fall_rates_by_group,
    required_tracking_check_names,
    required_tracking_profile,
)
from mjlab_microban.scripts.evaluate_teleop_v12_tracking import (
    _acceptance as _tracking_acceptance,
)
from mjlab_microban.scripts.evaluate_teleop_v12_tracking import (
    thresholds as tracking_thresholds,
)
from mjlab_microban.scripts.teleop_v12_bootstrap_gate import (
    ONNX_PARITY_TOLERANCE,
    PRISTINE_PARITY_TOLERANCE,
)
from mjlab_microban.scripts.teleop_v12_scenarios import (
    ENVS_PER_SCENARIO,
    arm_scenarios,
    foot_scenarios,
    push_scenarios,
)
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_TELEOP_ACTION_JOINT_NAMES,
    MICROBAN_TELEOP_OBSERVATION_WIDTH,
)
from mjlab_microban.tasks.microban_teleop_v12_actor import (
    TELEOP_V12_ADAPTER_GRADIENT_SCHEDULE_REVISION,
    teleop_v12_active_adapter_columns,
)
from mjlab_microban.tasks.microban_teleop_v12_bootstrap import (
    portable_bootstrap_artifact_path,
    resolve_bootstrap_artifact_path,
    sha256_file,
)
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MICROBAN_TELEOP_V12_ACTION_CLIP,
    MICROBAN_TELEOP_V12_TRAINING_CONTRACT_VERSION,
)
from mjlab_microban.tasks.microban_teleop_v12_home_pose import (
    TELEOP_V12_HOME_POSE_INFO_KEY,
    validate_teleop_v12_home_pose,
)
from mjlab_microban.tasks.microban_teleop_v12_runner import (
    require_bilateral_site_order,
)
from mjlab_microban.teleop_v12_safety import (
    ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD,
)
from mjlab_microban.twist_pass_line import AXES as TWIST_AXES
from mjlab_microban.twist_pass_line import STANDING_DRIFT_MAX, twist_pass_line_record

if set(policy_contract.PICO_WAIVABLE_TRACKING_CHECKS) | set(
    policy_contract.PICO_UNWAIVABLE_TRACKING_CHECKS
) != set(TRACKING_CHECK_NAMES) or set(policy_contract.PICO_WAIVABLE_TRACKING_CHECKS) & set(
    policy_contract.PICO_UNWAIVABLE_TRACKING_CHECKS
):
    raise AssertionError("PICO waivable/unwaivable check split drifted from the judgment")

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
            "action_clip": list(MICROBAN_TELEOP_V12_ACTION_CLIP),
            "previous_action": "raw_actor_output",
            "policy_observation_width": MICROBAN_TELEOP_OBSERVATION_WIDTH,
        },
        "Locomotion",
    )
    if report.get("thresholds") != {
        "actual_soft_limit_violation_rad_max": (
            ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD
        ),
        "twist_pass_line": twist_pass_line_record(),
    }:
        raise ValueError("Locomotion report thresholds drifted")
    results = report.get("results")
    expected_names = ("neutral", *LOCOMOTION_MOVING_SCENARIOS)
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
            or result.get("neutral_target_verified_steps") != 300
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
        ):
            raise ValueError("Locomotion directional evidence is inconsistent")
    try:
        recomputed, status = _locomotion_acceptance(results)
        judgments = locomotion_twist_judgments(results)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Locomotion result evidence is malformed") from exc
    if report.get("twist_judgment") != judgments:
        raise ValueError("Locomotion twist judgment does not match result evidence")
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
            "twist_direction",
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


def _number_or_none(value: object) -> float | None:
    return float(value) if _finite_number(value) else None


def _record_failure(
    entries: list[dict[str, Any]],
    *,
    scenario: str,
    quantity: str,
    measured: object,
    limit: float | None,
    rule: str,
) -> None:
    """Append the measurement when it breaks its limit (or is not a number)."""

    value = _number_or_none(measured)
    if value is not None and limit is not None:
        if rule == "at_most" and value <= limit:
            return
        if rule == "at_least" and value >= limit:
            return
    entries.append(
        {"scenario": scenario, "quantity": quantity, "measured": value, "limit": limit, "rule": rule}
    )


def _standing_failures(entries: list[dict[str, Any]], item: dict[str, Any]) -> None:
    """A standing command's touchdowns and ZMP distance (J1's crouches, J3)."""

    name = str(item["name"])
    _record_failure(entries, scenario=name, quantity="touchdowns_per_s", measured=item["touchdowns_per_s"],
                    limit=STILL_TOUCHDOWNS_PER_S_MAX, rule="at_most")
    _record_failure(entries, scenario=name, quantity="zmp_distance_median_m",
                    measured=item["zmp_distance_median_m"], limit=ZMP_CENTRE_MAX_M, rule="at_most")


def tracking_check_failures(results: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Every failing measurement of each waivable PICO check.

    Read with the same limits as evaluate_teleop_v12_tracking._acceptance;
    create_gate cross-checks the two (a check fails exactly when it lists a
    measurement here).
    """

    failures: dict[str, list[dict[str, Any]]] = {
        name: [] for name in policy_contract.PICO_WAIVABLE_TRACKING_CHECKS
    }
    feet, push, arms = results["feet"], results["push"], results["arms"]
    for item in feet:
        name = str(item["name"])
        if all(item["lifted"]):
            dz = float(item["foot_goal"][0][2])
            _record_failure(failures["foot_lift"], scenario=name, quantity="trunk_drop_median_m",
                            measured=item["trunk_drop_median_m"], limit=FOOT_LIFT_MIN_SHARE * dz,
                            rule="at_least")
            _record_failure(failures["foot_error"], scenario=name, quantity="trunk_drop_error_m",
                            measured=crouch_drop_error_m(item), limit=foot_error_limit_m(dz), rule="at_most")
            _standing_failures(failures["standing_still"], item)
        else:
            for side, lifted, goal, median, air in zip(
                ("left", "right"), item["lifted"], item["foot_goal"], item["lift_median_m"],
                item["air_share_median"], strict=True,
            ):
                if lifted:
                    _record_failure(failures["foot_lift"], scenario=name,
                                    quantity=f"{side}_foot_lift_median_m", measured=median,
                                    limit=FOOT_LIFT_MIN_SHARE * float(goal[2]), rule="at_least")
                    _record_failure(failures["foot_lift"], scenario=name,
                                    quantity=f"{side}_foot_air_share_median", measured=air,
                                    limit=FOOT_AIR_MIN_SHARE, rule="at_least")
            _record_failure(failures["support_foot_still"], scenario=name,
                            quantity="support_move_median_m", measured=item["support_move_median_m"],
                            limit=SUPPORT_FOOT_MOVE_MAX_M, rule="at_most")
        _record_failure(failures["foot_error"], scenario=name, quantity="relative_error_rms_m",
                        measured=item["relative_error_rms_m"],
                        limit=foot_error_limit_m(float(item["relative_target_m"])), rule="at_most")
    for item in (*feet, *arms):
        _record_failure(failures["no_falls_without_push"], scenario=str(item["name"]),
                        quantity="falls", measured=item["falls"], limit=0.0, rule="at_most")
    for group, rates in push_fall_rates_by_group(push).items():
        _record_failure(failures["push_falls"], scenario=f"{group}_push_scenarios",
                        quantity="pico_fall_rate", measured=rates["pico"],
                        limit=rates["walker"] + PUSH_FALL_RATE_MARGIN, rule="at_most")
    home = {item["base"]: item for item in arms if item["arms"] == "home"}
    for item in arms:
        name = str(item["name"])
        if all(value == 0.0 for value in item["twist"]):
            _standing_failures(failures["standing_still"], item)
            for axis, value, limit in zip(TWIST_AXES, item["mean_twist"], STANDING_DRIFT_MAX, strict=True):
                number = _number_or_none(value)
                _record_failure(failures["standing_still"], scenario=name, quantity=f"abs_mean_{axis}",
                                measured=None if number is None else abs(number), limit=limit,
                                rule="at_most")
        elif item["arms"] != "home":
            reference = _number_or_none(home[item["base"]]["signed_speed"]) if item["base"] in home else None
            command = max(abs(float(value)) for value in item["twist"])
            _record_failure(failures["walking_with_arms"], scenario=name, quantity="signed_speed_m_s",
                            measured=item["signed_speed"],
                            limit=None if reference is None else arm_walk_min_speed(reference, command),
                            rule="at_least")
    return failures


def _require_waivable_failures(checks: dict[str, Any], waived: Sequence[str]) -> None:
    """The report passes, or ``waived`` names exactly its failing behaviour checks."""

    failing = sorted(name for name, value in checks.items() if value is not True)
    if not waived:
        if failing:
            raise ValueError("Tracking report contains a failing check")
        return
    names = list(waived)
    if len(set(names)) != len(names):
        raise ValueError("Waived tracking checks are listed twice")
    unwaivable = sorted(set(failing) & set(policy_contract.PICO_UNWAIVABLE_TRACKING_CHECKS))
    if unwaivable:
        raise ValueError("Safety checks failed and cannot be waived: " + ", ".join(unwaivable))
    if set(names) & set(policy_contract.PICO_UNWAIVABLE_TRACKING_CHECKS):
        raise ValueError("Safety checks cannot be waived")
    if set(names) != set(failing):
        raise ValueError(
            "A waiver must name exactly the failing tracking checks: failing="
            + ",".join(failing) + " waived=" + ",".join(sorted(names))
        )


def _tracking_waiver(
    tracking: dict[str, Any],
    waived: Sequence[str],
    approval: str | None,
) -> dict[str, Any] | None:
    """The waiver record of a validated failing judgment, or None without a waiver."""

    if not waived:
        if approval is not None:
            raise ValueError("A waiver note needs --waive-tracking-checks")
        return None
    checks, results = tracking["checks"], tracking["results"]
    _require_waivable_failures(checks, waived)
    if approval is None or not approval.strip():
        raise ValueError("A tracking waiver needs the user's approval (--waiver-note)")
    try:
        failures = tracking_check_failures(results)
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
        raise ValueError("Tracking result evidence is malformed") from exc
    for name in policy_contract.PICO_WAIVABLE_TRACKING_CHECKS:
        if bool(failures[name]) == checks[name]:
            raise ValueError(f"Waiver measurements of {name} disagree with its check")
    record = {
        "approval": approval.strip(),
        "checks": {name: failures[name] for name in sorted(waived)},
    }
    try:
        return policy_contract.check_tracking_waiver(record)
    except policy_contract.PolicyContractError as exc:
        raise ValueError(f"Tracking waiver cannot be recorded: {exc}") from exc


def _validate_tracking_report(
    report: dict[str, Any],
    expected_identity: dict[str, int | str],
    *,
    waived: Sequence[str] = (),
) -> str:
    """Validate one PICO judgment report; return its profile.

    The checks are recomputed from the recorded measurements of every
    scenario (evaluate_teleop_v12_tracking._acceptance).  The report must
    pass, unless ``waived`` names exactly its failing (behaviour) checks.
    """

    completed = int(expected_identity["completed_updates"])
    profile = required_tracking_profile(completed)
    if (
        report.get("schema_version") != 2
        or report.get("gate") != "microban_teleop_v12_tracking"
        or report.get("profile") != profile
        or report.get("status") != ("fail" if waived else "pass")
    ):
        raise ValueError("Tracking report schema/profile/status drifted")
    _require_report_identity(report, expected_identity, "Tracking")
    expected_settings = canonical_settings(device="", seed=42)
    expected_settings.pop("device")
    _require_canonical_settings(report, expected_settings, "Tracking")
    if report.get("thresholds") != tracking_thresholds():
        raise ValueError("Tracking report thresholds drifted")
    results = report.get("results")
    if not isinstance(results, dict) or set(results) != {"feet", "push", "arms", "safety"}:
        raise ValueError("Tracking report results are malformed")
    expected_names = {
        "feet": [s.name for s in foot_scenarios()],
        "push": [s.name for s in push_scenarios()],
        "arms": [s.name for s in arm_scenarios()],
    }
    rollouts = ENVS_PER_SCENARIO * SEED_COUNT
    for part, names in expected_names.items():
        records = results[part]
        if (
            not isinstance(records, list)
            or [item.get("name") if isinstance(item, dict) else None for item in records] != names
            or any(item.get("rollouts") != rollouts for item in records)
        ):
            raise ValueError(f"Tracking report {part} scenario set/order drifted")
    for item, scenario in zip(results["feet"], foot_scenarios(), strict=True):
        if item.get("foot_goal") != [list(goal) for goal in scenario.foot_goal] or item.get(
            "lifted"
        ) != list(scenario.lifted):
            raise ValueError("Tracking report foot targets drifted")
    for item, scenario in zip(results["push"], push_scenarios(), strict=True):
        if item.get("foot_goal") != [list(goal) for goal in scenario.foot_goal]:
            raise ValueError("Tracking report push foot targets drifted")
    try:
        recomputed, status = _tracking_acceptance(results)
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
        raise ValueError("Tracking result evidence is malformed") from exc
    checks = report.get("checks")
    if not isinstance(checks, dict) or set(checks) != set(required_tracking_check_names(profile)):
        raise ValueError("Tracking report check set is incomplete or drifted")
    if recomputed != checks or status != report.get("status"):
        raise ValueError("Tracking checks do not match result evidence")
    _require_waivable_failures(checks, waived)
    _require_action_envelope(
        report.get("raw_action_envelope"),
        label="Tracking/aggregate",
        aggregate=True,
        scenario_count=sum(len(names) for names in expected_names.values()),
    )
    return profile


def _validate_onnx_report(
    report: dict[str, Any],
    expected_identity: dict[str, int | str],
    *,
    parity_tolerance: float = ONNX_PARITY_TOLERANCE,
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
        or neutral.get("residual") != "excluded"
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
        or onnx.get("input_shape") != [1, MICROBAN_TELEOP_OBSERVATION_WIDTH]
        or onnx.get("output_shape") != [1, 18]
        or onnx.get("reference_samples") != 64
        or onnx.get("input_coverage") != f"deterministic_nonzero_all_{MICROBAN_TELEOP_OBSERVATION_WIDTH}_columns"
        or onnx.get("teleop_only_columns_nonzero") is not True
        or onnx.get("onnxruntime_providers") != ["CPUExecutionProvider"]
        or not isinstance(onnx.get("onnxruntime_version"), str)
        or not onnx["onnxruntime_version"]
        or onnx.get("tolerance") != parity_tolerance
    ):
        raise ValueError("ONNX full-width CPU evidence drifted")
    if "relative_tolerance" in onnx or "parity_rule" in onnx:
        # Per-sample atol + rtol*max|expected| bound (teleop_v12_onnx_gate).
        from mjlab_microban.scripts.teleop_v12_onnx_gate import (
            ONNX_PARITY_RELATIVE_TOLERANCE,
            ONNX_PARITY_RULE,
        )

        magnitude = onnx.get("maximum_absolute_expected_output")
        if (
            onnx.get("relative_tolerance") != ONNX_PARITY_RELATIVE_TOLERANCE
            or onnx.get("parity_rule") != ONNX_PARITY_RULE
            or not _finite_number(magnitude)
            or float(magnitude) < 0.0
        ):
            raise ValueError("ONNX relative parity evidence drifted")
        absolute_cap = parity_tolerance + ONNX_PARITY_RELATIVE_TOLERANCE * float(
            magnitude
        )
        for name in (
            "reference_evaluator_maximum_bound_ratio",
            "onnxruntime_cpu_maximum_bound_ratio",
        ):
            if (
                not _finite_number(onnx.get(name))
                or float(onnx[name]) < 0.0
                or float(onnx[name]) > 1.0
            ):
                raise ValueError(f"ONNX parity evidence failed: {name}")
    else:
        absolute_cap = parity_tolerance
    for name in (
        "reference_evaluator_maximum_absolute_error",
        "onnxruntime_cpu_maximum_absolute_error",
    ):
        if (
            not _finite_number(onnx.get(name))
            or float(onnx[name]) < 0.0
            or float(onnx[name]) > absolute_cap
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
    if payload["infos"].get("microban_teleop_training_contract_version") != (
        MICROBAN_TELEOP_V12_TRAINING_CONTRACT_VERSION
    ):
        raise ValueError("Checkpoint is not of the current PICO training contract")
    infos = payload["infos"]
    validate_teleop_v12_home_pose(infos)
    require_bilateral_site_order(infos)
    iteration = payload.get("iter")
    if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < 0:
        raise ValueError("Stage checkpoint iteration is invalid")
    checkpoint_sha = sha256_file(path)
    if infos.get("adapter_gradient_schedule_revision") != (
        TELEOP_V12_ADAPTER_GRADIENT_SCHEDULE_REVISION
    ):
        raise ValueError("Checkpoint is not the safe staged-mask v12 recipe")
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
    return checkpoint_sha, iteration, completed, infos


GATE_SCHEMA_VERSION = 3


def _require_final_clock(completed: int) -> None:
    if completed != PICO_TOTAL_UPDATES:
        raise ValueError(
            f"A PICO run ends at update {PICO_TOTAL_UPDATES}; "
            f"this checkpoint completed {completed}"
        )


def create_gate(
    *,
    checkpoint: Path,
    locomotion_report: Path,
    tracking_report: Path,
    onnx_report: Path,
    waive_tracking_checks: Sequence[str] = (),
    waiver_note: str | None = None,
) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    locomotion_report = locomotion_report.resolve()
    tracking_report = tracking_report.resolve()
    onnx_report = onnx_report.resolve()
    checkpoint_sha, iteration, completed, infos = _checkpoint_identity(checkpoint)
    _require_final_clock(completed)
    locomotion = _load_json(locomotion_report)
    tracking = _load_json(tracking_report)
    onnx = _load_json(onnx_report)
    expected_report_identity = {
        "sha256": checkpoint_sha,
        "iteration": iteration,
        "completed_updates": completed,
    }
    _validate_locomotion_report(locomotion, expected_report_identity)
    waived = tuple(waive_tracking_checks)
    if waived:
        tracking_profile = _validate_tracking_report(
            tracking, expected_report_identity, waived=waived
        )
    else:
        tracking_profile = _validate_tracking_report(tracking, expected_report_identity)
    waiver = _tracking_waiver(tracking, waived, waiver_note)
    onnx_path, onnx_sha = _validate_onnx_report(onnx, expected_report_identity)
    result = {
        "schema_version": GATE_SCHEMA_VERSION,
        "gate": "microban_teleop_v12_stage",
        "status": (
            policy_contract.GATE_STATUS_PASS
            if waiver is None
            else policy_contract.GATE_STATUS_WAIVED
        ),
        "checkpoint": portable_bootstrap_artifact_path(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "iteration": iteration,
        "completed_updates": completed,
        TELEOP_V12_HOME_POSE_INFO_KEY: deepcopy(infos[TELEOP_V12_HOME_POSE_INFO_KEY]),
        "tracking_profile": tracking_profile,
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
    if waiver is not None:
        result["tracking_waiver"] = waiver
    return result


def _gate_waiver_arguments(gate: dict[str, Any]) -> tuple[tuple[str, ...], str | None]:
    """The waiver a gate was created with (validated by rebuilding the gate)."""

    if "tracking_waiver" not in gate:
        return (), None
    waiver = gate["tracking_waiver"]
    try:
        policy_contract.check_tracking_waiver(waiver)
    except policy_contract.PolicyContractError as exc:
        raise ValueError(f"V12 stage gate tracking waiver is malformed: {exc}") from exc
    return tuple(waiver["checks"]), waiver["approval"]


def validate_gate(gate_path: Path, checkpoint: Path) -> dict[str, Any]:
    gate_path = gate_path.resolve()
    checkpoint = checkpoint.resolve()
    gate = _load_json(gate_path)
    checkpoint_sha, iteration, completed, infos = _checkpoint_identity(checkpoint)
    _require_final_clock(completed)
    waived, approval = _gate_waiver_arguments(gate)
    exact = {
        "schema_version": GATE_SCHEMA_VERSION,
        "gate": "microban_teleop_v12_stage",
        "status": (
            policy_contract.GATE_STATUS_WAIVED if waived else policy_contract.GATE_STATUS_PASS
        ),
        "checkpoint": portable_bootstrap_artifact_path(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "iteration": iteration,
        "completed_updates": completed,
        TELEOP_V12_HOME_POSE_INFO_KEY: deepcopy(infos[TELEOP_V12_HOME_POSE_INFO_KEY]),
        "tracking_profile": required_tracking_profile(completed),
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
        waive_tracking_checks=waived,
        waiver_note=approval,
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
    create.add_argument(
        "--waive-tracking-checks",
        default="",
        help="comma-separated failing PICO behaviour checks the user approved installing with",
    )
    create.add_argument("--waiver-note", help="the user's approval, recorded in the gate and the ONNX")
    validate = subparsers.add_parser("validate")
    validate.add_argument("gate", type=Path)
    validate.add_argument("checkpoint", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "create":
        if args.output.exists() and not args.force:
            raise FileExistsError("Gate exists (pass --force)")
        gate = create_gate(
            checkpoint=args.checkpoint,
            locomotion_report=args.locomotion_report,
            tracking_report=args.tracking_report,
            onnx_report=args.onnx_report,
            waive_tracking_checks=tuple(
                name.strip() for name in args.waive_tracking_checks.split(",") if name.strip()
            ),
            waiver_note=args.waiver_note,
        )
        publish_json_atomic(args.output, gate)
    else:
        gate = validate_gate(args.gate, args.checkpoint)
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
