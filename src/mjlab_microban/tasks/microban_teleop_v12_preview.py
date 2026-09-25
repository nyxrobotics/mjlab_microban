"""Fail-closed identity contract for simulation-only v12 preview checkpoints."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from mjlab_microban.tasks.microban_teleop_provenance import (
    collect_training_source_manifest,
)
from mjlab_microban.tasks.microban_teleop_v12_actor import (
    TELEOP_V12_EXTRA_OBSERVATION_COLUMNS,
    TELEOP_V12_FOOT_OBSERVATION_COLUMNS,
    TELEOP_V12_HAND_OBSERVATION_COLUMNS,
    TELEOP_V12_HMD_OBSERVATION_COLUMNS,
)
from mjlab_microban.teleop_v12_safety import (
    ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD,
)

MICROBAN_TELEOP_V12_PREVIEW_TASK_ID = "Mjlab-Teleop-V12-Preview-Microban"
TELEOP_V12_PREVIEW_INFO_KEY = "teleop_v12_preview"
TELEOP_V12_PREVIEW_PHASE1_ACCEPTANCE_INFO_KEY = "teleop_v12_preview_phase1_acceptance"
TELEOP_V12_PREVIEW_LEGACY_REVISION = (
    "sanitized601_clock_lift_to10001_full20_sim_only_v1"
)
TELEOP_V12_PREVIEW_REVISION = "sanitized601_staged_hmd_hand_then_full20_sim_only_v2"
TELEOP_V12_PREVIEW_PHASE_HMD_HAND = "hmd_hand"
TELEOP_V12_PREVIEW_PHASE_FULL_BODY = "full_body"

TELEOP_V12_PREVIEW_SOURCE_SHA256 = (
    "ab0dbe0db9cadd6bb937e5eebf3d2b8f6bb3fadcc5e025aa15d28fb87e85d3a2"
)
TELEOP_V12_PREVIEW_SOURCE_ITERATION = 600
TELEOP_V12_PREVIEW_SOURCE_COMPLETED_UPDATES = 601
TELEOP_V12_PREVIEW_PHASE1_LIFTED_ITERATION = 7_000
TELEOP_V12_PREVIEW_PHASE1_LIFTED_COMPLETED_UPDATES = 7_001
TELEOP_V12_PREVIEW_PHASE1_TRAINED_ITERATION = 7_100
TELEOP_V12_PREVIEW_PHASE1_TRAINED_COMPLETED_UPDATES = 7_101
TELEOP_V12_PREVIEW_PHASE2_LIFTED_ITERATION = 10_000
TELEOP_V12_PREVIEW_PHASE2_LIFTED_COMPLETED_UPDATES = 10_001
# At least one full-body update is required.  Actual live authority additionally
# requires a hash-bound strict/visual acceptance receipt, so 10/20-update
# deadline candidates remain evaluable without making the zero-foot seed live.
TELEOP_V12_PREVIEW_PHASE2_MINIMUM_LIVE_ITERATION = 10_001
TELEOP_V12_PREVIEW_PHASE1_STRICT_QUALITY = "strict_hmd_hand_acceptance_v1"
TELEOP_V12_PREVIEW_PHASE1_VISUAL_QUALITY = "visual_only_relaxed_hand_tracking_v1"
TELEOP_V12_PREVIEW_FULLBODY_STRICT_QUALITY = "strict_fullbody_acceptance_v1"
TELEOP_V12_PREVIEW_FULLBODY_VISUAL_QUALITY = "visual_only_relaxed_fullbody_tracking_v1"

# Old names remain importable only for explicit v1 receipt readers.
TELEOP_V12_PREVIEW_LIFTED_ITERATION = 10_000
TELEOP_V12_PREVIEW_LIFTED_COMPLETED_UPDATES = 10_001

TELEOP_V12_PREVIEW_HMD_HAND_COLUMNS = (
    *TELEOP_V12_HMD_OBSERVATION_COLUMNS,
    *TELEOP_V12_HAND_OBSERVATION_COLUMNS,
)
_LOWER_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LOCOMOTION_SCENARIOS = (
    "neutral",
    "forward_0p1",
    "forward_0p2",
    "backward_0p1",
    "backward_0p2",
    "lateral_left_0p1",
    "lateral_right_0p1",
    "yaw_left_0p5",
    "yaw_right_0p5",
)
_FINAL_TRACKING_SCENARIOS = (
    "low_forward",
    "max_hands_left",
    "max_hands_right",
    "max_keypoints_left",
    "max_keypoints_right",
    "bounded_both_feet",
    "mixed_forward_left",
    "mixed_backward_right",
)
_HMD_HAND_TRACKING_SCENARIOS = (
    "low_forward",
    "max_hands_left",
    "max_hands_right",
    "max_keypoints_left",
)
_LOCOMOTION_CHECKS = {
    "all_scenarios_completed",
    "no_falls",
    "finite",
    "actual_soft_limits",
    "raw_action_recurrence",
    "neutral_targets",
    "directional_signs",
    "minimum_directional_response",
}
_FINAL_TRACKING_CHECKS = {
    "all_scenarios_completed",
    "no_falls",
    "finite",
    "actual_soft_limits",
    "raw_action_recurrence",
    "forced_hmd_motion",
    "nonzero_observation_coverage",
    "target_column_ablation_response",
    "twist_directional_response",
    "hand_tracking_rms",
    "hand_tracking_p95",
    "foot_tracking_rms",
    "foot_tracking_p95",
}
_HMD_HAND_TRACKING_CHECKS = _FINAL_TRACKING_CHECKS - {
    "foot_tracking_rms",
    "foot_tracking_p95",
}


def legacy_preview_info() -> dict[str, Any]:
    """Return the exact v1 marker, for explicit read-only legacy audits only."""

    result = {
        "schema_version": 1,
        "revision": TELEOP_V12_PREVIEW_LEGACY_REVISION,
        "simulation_only": True,
        "source_checkpoint_sha256": TELEOP_V12_PREVIEW_SOURCE_SHA256,
        "source_iteration": TELEOP_V12_PREVIEW_SOURCE_ITERATION,
        "source_completed_updates": TELEOP_V12_PREVIEW_SOURCE_COMPLETED_UPDATES,
        "lifted_iteration": TELEOP_V12_PREVIEW_LIFTED_ITERATION,
        "lifted_completed_updates": TELEOP_V12_PREVIEW_LIFTED_COMPLETED_UPDATES,
        "active_actor_columns": list(TELEOP_V12_EXTRA_OBSERVATION_COLUMNS),
        "target_activation_asserted": True,
        "reward_activation_asserted": True,
    }
    return result


def canonical_preview_info() -> dict[str, Any]:
    """Deprecated alias for parsing already-created v1 preview artifacts."""

    return legacy_preview_info()


def staged_preview_info(
    *,
    phase: str,
    phase_source_checkpoint_sha256: str,
    phase1_acceptance_receipt_sha256: str | None = None,
    phase1_quality_class: str | None = None,
) -> dict[str, Any]:
    """Build the exact staged-v2 marker for one isolated preview phase."""

    if not _LOWER_SHA256.fullmatch(phase_source_checkpoint_sha256):
        raise ValueError("Preview phase source SHA-256 must be lowercase hex")
    if phase == TELEOP_V12_PREVIEW_PHASE_HMD_HAND:
        if phase_source_checkpoint_sha256 != TELEOP_V12_PREVIEW_SOURCE_SHA256:
            raise ValueError("HMD/hand preview must start from sanitized601")
        if phase1_acceptance_receipt_sha256 is not None:
            raise ValueError("HMD/hand seed cannot have a phase1 acceptance receipt")
        if phase1_quality_class is not None:
            raise ValueError("HMD/hand seed cannot have a phase1 quality class")
        source_iteration = TELEOP_V12_PREVIEW_SOURCE_ITERATION
        source_completed = TELEOP_V12_PREVIEW_SOURCE_COMPLETED_UPDATES
        lifted_iteration = TELEOP_V12_PREVIEW_PHASE1_LIFTED_ITERATION
        lifted_completed = TELEOP_V12_PREVIEW_PHASE1_LIFTED_COMPLETED_UPDATES
        active = TELEOP_V12_PREVIEW_HMD_HAND_COLUMNS
        inactive = TELEOP_V12_FOOT_OBSERVATION_COLUMNS
        foot_active = False
    elif phase == TELEOP_V12_PREVIEW_PHASE_FULL_BODY:
        if not isinstance(phase1_acceptance_receipt_sha256, str) or not (
            _LOWER_SHA256.fullmatch(phase1_acceptance_receipt_sha256)
        ):
            raise ValueError("Full-body preview requires a phase1 receipt SHA-256")
        if phase1_quality_class not in {
            TELEOP_V12_PREVIEW_PHASE1_STRICT_QUALITY,
            TELEOP_V12_PREVIEW_PHASE1_VISUAL_QUALITY,
        }:
            raise ValueError("Full-body preview phase1 quality class is invalid")
        source_iteration = TELEOP_V12_PREVIEW_PHASE1_TRAINED_ITERATION
        source_completed = TELEOP_V12_PREVIEW_PHASE1_TRAINED_COMPLETED_UPDATES
        lifted_iteration = TELEOP_V12_PREVIEW_PHASE2_LIFTED_ITERATION
        lifted_completed = TELEOP_V12_PREVIEW_PHASE2_LIFTED_COMPLETED_UPDATES
        active = TELEOP_V12_EXTRA_OBSERVATION_COLUMNS
        inactive = ()
        foot_active = True
    else:
        raise ValueError(f"Unknown staged preview phase: {phase!r}")
    result = {
        "schema_version": 2,
        "revision": TELEOP_V12_PREVIEW_REVISION,
        "phase": phase,
        "simulation_only": True,
        "canonical_deployment_accepted": False,
        "root_source_checkpoint_sha256": TELEOP_V12_PREVIEW_SOURCE_SHA256,
        "root_source_iteration": TELEOP_V12_PREVIEW_SOURCE_ITERATION,
        "root_source_completed_updates": TELEOP_V12_PREVIEW_SOURCE_COMPLETED_UPDATES,
        "phase_source_checkpoint_sha256": phase_source_checkpoint_sha256,
        "phase_source_iteration": source_iteration,
        "phase_source_completed_updates": source_completed,
        "phase1_acceptance_receipt_sha256": phase1_acceptance_receipt_sha256,
        "lifted_iteration": lifted_iteration,
        "lifted_completed_updates": lifted_completed,
        "active_actor_columns": list(active),
        "inactive_actor_columns": list(inactive),
        "hmd_target_active": True,
        "hand_target_active": True,
        "foot_target_active": foot_active,
        "hand_tracking_reward_active": True,
        "foot_tracking_reward_active": foot_active,
        "clock_lift_preserved_actor_critic_optimizer": True,
        "live_candidate_class": phase == TELEOP_V12_PREVIEW_PHASE_FULL_BODY,
    }
    if phase == TELEOP_V12_PREVIEW_PHASE_FULL_BODY:
        result["phase1_quality_class"] = phase1_quality_class
    return result


def validate_preview_marker(
    infos: object,
    *,
    iteration: int,
    allow_legacy_v1: bool = False,
    required_phase: str | None = None,
    require_live_candidate: bool = False,
) -> dict[str, Any]:
    """Authenticate a staged preview marker for its explicit consumer."""

    if not isinstance(infos, dict):
        raise TypeError("Preview checkpoint infos are malformed")
    if infos.get("preview_non_deployable") is not True:
        raise ValueError("Preview checkpoint lacks preview_non_deployable=true")
    marker = infos.get(TELEOP_V12_PREVIEW_INFO_KEY)
    if not isinstance(marker, dict):
        raise TypeError("Preview checkpoint lineage marker is missing")
    if marker.get("revision") == TELEOP_V12_PREVIEW_LEGACY_REVISION:
        if not allow_legacy_v1 or marker != legacy_preview_info():
            raise ValueError("Legacy v1 preview requires explicit read-only opt-in")
        if required_phase is not None or require_live_candidate:
            raise ValueError("Legacy v1 preview is never a staged/live candidate")
        if iteration < TELEOP_V12_PREVIEW_LIFTED_ITERATION:
            raise ValueError("Legacy preview checkpoint predates its clock lift")
        return marker

    phase = marker.get("phase")
    source_sha = marker.get("phase_source_checkpoint_sha256")
    receipt_sha = marker.get("phase1_acceptance_receipt_sha256")
    quality_class = marker.get("phase1_quality_class")
    expected = staged_preview_info(
        phase=phase,
        phase_source_checkpoint_sha256=source_sha,
        phase1_acceptance_receipt_sha256=receipt_sha,
        phase1_quality_class=quality_class,
    )
    if marker != expected:
        raise ValueError("Staged preview checkpoint lineage marker drifted")
    if required_phase is not None and phase != required_phase:
        raise ValueError(
            f"Preview phase {phase!r} is not required phase {required_phase!r}"
        )
    if iteration < expected["lifted_iteration"]:
        raise ValueError("Preview checkpoint iteration predates its phase clock lift")
    if phase == TELEOP_V12_PREVIEW_PHASE_HMD_HAND and iteration >= (
        TELEOP_V12_PREVIEW_PHASE2_LIFTED_ITERATION
    ):
        raise ValueError("HMD/hand preview ran past its isolated phase window")
    if require_live_candidate and (
        phase != TELEOP_V12_PREVIEW_PHASE_FULL_BODY
        or iteration < TELEOP_V12_PREVIEW_PHASE2_MINIMUM_LIVE_ITERATION
    ):
        raise ValueError("Only a trained staged-v2 full-body preview is live-eligible")
    return marker


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preview_evaluator_source_manifest() -> dict[str, Any]:
    """Hash the transitive local source/model contract behind preview evidence."""

    root = Path(__file__).resolve().parents[3]
    source = collect_training_source_manifest(root)
    return {
        "schema_version": 2,
        "algorithm": source["algorithm"],
        "files": source["files"],
        "aggregate_sha256": source["tree_sha256"],
    }


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and __import__("math").isfinite(float(value))
    )


def _validate_tracking_result_evidence(report: dict[str, Any], *, profile: str) -> None:
    """Recompute tracking checks from scenario evidence, including hard safety."""

    # Runtime imports avoid a module cycle: the evaluator imports this contract
    # in order to authenticate preview checkpoints before producing a report.
    from mjlab_microban.robot.microban_hand_fk import microban_hand_fk_metadata
    from mjlab_microban.scripts.evaluate_teleop_v12_tracking import (
        DIRECTIONAL_RESPONSE_MINIMUM,
        FOOT_P95_MAX_M,
        FOOT_RMS_MAX_M,
        HAND_P95_MAX_M,
        HAND_RMS_MAX_M,
        HMD_ACTUAL_PEAK_TO_PEAK_MIN_RAD,
        HMD_TARGET_PEAK_TO_PEAK_MIN_RAD,
        TARGET_COLUMN_ABLATION_ACTION_DELTA_MIN,
        TARGET_COLUMN_ABLATION_METHOD,
        _acceptance,
        _aggregate_action_envelopes,
        _scenarios,
        target_column_ablation_observation_columns,
    )
    from mjlab_microban.scripts.teleop_v12_stage import _require_action_envelope
    from mjlab_microban.tasks.microban_policy_export import MICROBAN_HMD_JOINT_NAMES

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
        raise ValueError("Preview tracking thresholds drifted")
    settings = report.get("settings")
    if (
        not isinstance(settings, dict)
        or settings.get("target_column_ablation") != TARGET_COLUMN_ABLATION_METHOD
        or settings.get("reachable_hand_target_fk") != microban_hand_fk_metadata()
    ):
        raise ValueError("Preview tracking input contract drifted")
    results = report["results"]
    scenarios = _scenarios(profile)
    for result, scenario in zip(results, scenarios, strict=True):
        expected_command = {
            "twist": list(scenario.twist),
            "foot_target": [list(item) for item in scenario.foot_target],
            "hand_target": [list(item) for item in scenario.hand_target],
            "hand_active": list(scenario.hand_active),
        }
        expects_foot = any(
            abs(item) > 0.0 for target in scenario.foot_target for item in target
        )
        expects_hand = any(scenario.hand_active)
        coverage = result.get("observation_coverage")
        actual_limit_violation = result.get("maximum_actual_soft_limit_violation_rad")
        if (
            result.get("command") != expected_command
            or result.get("completed") is not True
            or result.get("executed_steps") != 300
            or result.get("fell") is not False
            or result.get("nonfinite") is not None
            or result.get("termination_names") != []
            or result.get("raw_action_recurrence_verified_steps") != 300
            or not _finite_number(actual_limit_violation)
            or float(actual_limit_violation) < 0.0
            or not isinstance(coverage, dict)
            or coverage.get("foot_target_expected") is not expects_foot
            or coverage.get("hand_target_expected") is not expects_hand
            or coverage.get("hmd_nonzero_steps", 0) <= 0
            or (expects_foot and coverage.get("foot_nonzero_steps") != 300)
            or (expects_hand and coverage.get("hand_nonzero_steps") != 300)
        ):
            raise ValueError("Preview tracking scenario hard evidence is malformed")
        coverage_passed = (
            coverage["hmd_nonzero_steps"] > 0
            and (not expects_foot or coverage.get("foot_nonzero_steps") == 300)
            and (not expects_hand or coverage.get("hand_nonzero_steps") == 300)
        )
        if coverage.get("passed") is not coverage_passed:
            raise ValueError("Preview tracking coverage boolean is inconsistent")

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
            raise ValueError("Preview tracking HMD evidence is malformed")
        hmd_passed = True
        for axis in MICROBAN_HMD_JOINT_NAMES:
            values = per_axis[axis]
            if not isinstance(values, dict) or not all(
                _finite_number(values.get(key))
                for key in ("target_peak_to_peak_rad", "actual_peak_to_peak_rad")
            ):
                raise ValueError("Preview tracking HMD span is malformed")
            hmd_passed = hmd_passed and (
                float(values["target_peak_to_peak_rad"])
                >= HMD_TARGET_PEAK_TO_PEAK_MIN_RAD
                and float(values["actual_peak_to_peak_rad"])
                >= HMD_ACTUAL_PEAK_TO_PEAK_MIN_RAD
            )
        if result.get("hmd_motion_evidence_passed") is not hmd_passed:
            raise ValueError("Preview tracking HMD boolean is inconsistent")

        axes = ("vx_m_s", "vy_m_s", "yaw_rad_s")
        measured = result.get("measured_velocity_body")
        response = result.get("directional_response")
        expected_axes = {
            axis
            for axis, command in zip(axes, scenario.twist, strict=True)
            if command != 0.0
        }
        if (
            not isinstance(measured, dict)
            or set(measured) != set(axes)
            or not isinstance(response, dict)
            or set(response) != expected_axes
            or any(
                not isinstance(measured.get(axis), dict)
                or measured[axis].get("sample_count") != 250
                or not _finite_number(measured[axis].get("mean"))
                for axis in axes
            )
        ):
            raise ValueError("Preview tracking velocity evidence is malformed")
        directional_passed = True
        for axis, command in zip(axes, scenario.twist, strict=True):
            if command == 0.0:
                continue
            item = response[axis]
            mean = float(measured[axis]["mean"])
            signed = mean * (1.0 if command > 0.0 else -1.0)
            item_passed = signed >= DIRECTIONAL_RESPONSE_MINIMUM[axis]
            if (
                not isinstance(item, dict)
                or item.get("command") != command
                or item.get("measured_mean") != mean
                or item.get("signed_response") != signed
                or item.get("minimum_signed_response")
                != DIRECTIONAL_RESPONSE_MINIMUM[axis]
                or item.get("passed") is not item_passed
            ):
                raise ValueError(
                    "Preview tracking directional evidence is inconsistent"
                )
            directional_passed = directional_passed and item_passed
        if result.get("twist_directional_response_passed") is not directional_passed:
            raise ValueError("Preview tracking directional boolean is inconsistent")

        target_error = result.get("target_error")
        if not isinstance(target_error, dict) or set(target_error) != {
            "foot",
            "active_hand",
        }:
            raise ValueError("Preview tracking target-error evidence is malformed")
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
                raise ValueError("Preview tracking target-error stats are malformed")
            sample_count = values.get("sample_count")
            metrics = ("min", "max", "mean", "rms", "p95")
            if expected:
                minimum = values.get("min")
                maximum = values.get("max")
                mean = values.get("mean")
                rms = values.get("rms")
                p95 = values.get("p95")
                if (
                    sample_count != expected_samples
                    or not all(_finite_number(values.get(name)) for name in metrics)
                    or not (
                        0.0
                        <= float(minimum)
                        <= float(mean)
                        <= float(rms)
                        <= float(maximum)
                    )
                    or not (float(minimum) <= float(p95) <= float(maximum))
                ):
                    raise ValueError("Preview active target-error stats are malformed")
            elif sample_count != 0 or any(
                values.get(name) is not None for name in metrics
            ):
                raise ValueError("Preview inactive target-error stats are inconsistent")

        ablation = result.get("target_column_ablation")
        if not isinstance(ablation, dict) or set(ablation) != {"hand", "foot"}:
            raise ValueError("Preview target-column ablation evidence is malformed")
        for target, expected in (("hand", expects_hand), ("foot", expects_foot)):
            item = ablation[target]
            ablated_columns, preserved_columns = (
                target_column_ablation_observation_columns(target)
            )
            if (
                not isinstance(item, dict)
                or set(item)
                != {
                    "target_expected",
                    "ablated_observation_columns",
                    "preserved_observation_columns",
                    "maximum_absolute_action_delta",
                    "minimum_required_action_delta",
                    "passed",
                }
                or item.get("target_expected") is not expected
                or item.get("ablated_observation_columns") != list(ablated_columns)
                or item.get("preserved_observation_columns") != list(preserved_columns)
            ):
                raise ValueError("Preview target-column ablation target drifted")
            maximum = item.get("maximum_absolute_action_delta")
            if expected:
                passed = _finite_number(maximum) and float(maximum) > (
                    TARGET_COLUMN_ABLATION_ACTION_DELTA_MIN
                )
                if (
                    item.get("minimum_required_action_delta")
                    != TARGET_COLUMN_ABLATION_ACTION_DELTA_MIN
                    or item.get("passed") is not passed
                ):
                    raise ValueError("Preview target-column ablation is inconsistent")
            elif (
                maximum is not None
                or item.get("minimum_required_action_delta") is not None
                or item.get("passed") is not True
            ):
                raise ValueError("Preview inactive target-column ablation drifted")
        _require_action_envelope(
            result.get("raw_action_envelope"),
            label=f"Preview tracking/{scenario.name}",
            aggregate=False,
        )

    try:
        recomputed, status = _acceptance(results, profile)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Preview tracking result evidence is malformed") from exc
    if recomputed != report["checks"] or status != report["status"]:
        raise ValueError("Preview tracking checks do not match result evidence")
    envelope = _require_action_envelope(
        report.get("raw_action_envelope"),
        label="Preview tracking/aggregate",
        aggregate=True,
        scenario_count=len(results),
    )
    if envelope.get("step_count") != 300 * len(results):
        raise ValueError("Preview tracking aggregate step count drifted")
    if envelope != _aggregate_action_envelopes(results):
        raise ValueError("Preview tracking aggregate action envelope is inconsistent")


def _validate_child_report(
    report: object,
    *,
    gate: str,
    checks: set[str],
    scenarios: tuple[str, ...],
    checkpoint_sha256: str,
    iteration: int,
    tracking_profile: str | None = None,
) -> bool:
    if not isinstance(report, dict):
        raise TypeError(f"Preview receipt {gate} report is malformed")
    actual_checks = report.get("checks")
    results = report.get("results")
    checkpoint = report.get("checkpoint")
    settings = report.get("settings")
    if (
        not isinstance(actual_checks, dict)
        or set(actual_checks) != checks
        or any(type(value) is not bool for value in actual_checks.values())
    ):
        raise ValueError(f"Preview receipt {gate} check set is malformed")
    passed = all(actual_checks.values())
    expected_status = "pass" if passed else "fail"
    common_settings = {
        "device": "cuda:0",
        "seed": 42,
        "steps": 300,
        "settle_steps": 50,
        "action_clip": None,
        "previous_action": "raw_actor_output",
    }
    if (
        report.get("schema_version") != 1
        or report.get("gate") != gate
        or report.get("status") != expected_status
        or not isinstance(results, list)
        or len(results) != len(scenarios)
        or not all(isinstance(result, dict) for result in results)
        or tuple(result.get("name") for result in results) != scenarios
        or not isinstance(checkpoint, dict)
        or checkpoint.get("sha256") != checkpoint_sha256
        or checkpoint.get("iteration") != iteration
        or checkpoint.get("completed_updates") != iteration + 1
        or not isinstance(settings, dict)
        or any(
            settings.get(key, object()) != value
            for key, value in common_settings.items()
        )
    ):
        raise ValueError(f"Preview receipt {gate} report failed strict validation")
    if tracking_profile is None:
        if settings.get("policy_observation_width") != 83:
            raise ValueError("Preview locomotion report observation width drifted")
        from mjlab_microban.scripts.evaluate_teleop_v12_checkpoint import (
            _acceptance as locomotion_acceptance,
        )

        try:
            recomputed, recomputed_status = locomotion_acceptance(results)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Preview locomotion result evidence is malformed") from exc
        if recomputed != actual_checks or recomputed_status != expected_status:
            raise ValueError("Preview locomotion checks do not match result evidence")
        # A live/visual authority always requires a passing locomotion child. Reuse
        # the canonical validator for its full command/velocity/summary checks.
        if passed:
            from mjlab_microban.scripts.teleop_v12_stage import (
                _validate_locomotion_report,
            )

            _validate_locomotion_report(
                report,
                {
                    "sha256": checkpoint_sha256,
                    "iteration": iteration,
                    "completed_updates": iteration + 1,
                },
            )
    else:
        from mjlab_microban.scripts.evaluate_teleop_v12_tracking import FINAL_PROFILE

        if (
            report.get("profile") != tracking_profile
            or settings.get("moving_hmd") != "forced_non_neutral"
            or settings.get("perturbation") is not (tracking_profile == FINAL_PROFILE)
        ):
            raise ValueError("Preview tracking report settings/profile drifted")
        _validate_tracking_result_evidence(report, profile=tracking_profile)
    return passed


def validate_preview_evaluation_report(
    report: object,
    *,
    checkpoint_sha256: str,
    iteration: int,
    marker: dict[str, Any],
) -> dict[str, Any]:
    """Validate exact nested evidence in a strict phase1/fullbody evaluation."""

    if not isinstance(report, dict):
        raise TypeError("Preview evaluation report is malformed")
    from mjlab_microban.scripts.evaluate_teleop_v12_tracking import (
        FINAL_PROFILE,
        HMD_HAND_PROFILE,
    )

    phase = marker.get("phase")
    if phase == TELEOP_V12_PREVIEW_PHASE_HMD_HAND:
        profile = HMD_HAND_PROFILE
        tracking_checks = _HMD_HAND_TRACKING_CHECKS
        tracking_scenarios = _HMD_HAND_TRACKING_SCENARIOS
        tracking_check_name = "hmd_hand_performance_foot_exposure"
        quality_class = TELEOP_V12_PREVIEW_PHASE1_STRICT_QUALITY
        live_candidate_phase = False
    elif phase == TELEOP_V12_PREVIEW_PHASE_FULL_BODY:
        profile = FINAL_PROFILE
        tracking_checks = _FINAL_TRACKING_CHECKS
        tracking_scenarios = _FINAL_TRACKING_SCENARIOS
        tracking_check_name = "full_body_perturbation_8x300"
        quality_class = TELEOP_V12_PREVIEW_FULLBODY_STRICT_QUALITY
        live_candidate_phase = True
    else:
        raise ValueError("Strict preview evaluation requires a staged-v2 marker")
    locomotion_passed = _validate_child_report(
        report.get("locomotion"),
        gate="microban_teleop_v12_neutral_locomotion_9x300",
        checks=_LOCOMOTION_CHECKS,
        scenarios=_LOCOMOTION_SCENARIOS,
        checkpoint_sha256=checkpoint_sha256,
        iteration=iteration,
    )
    tracking_passed = _validate_child_report(
        report.get("tracking"),
        gate="microban_teleop_v12_tracking",
        checks=tracking_checks,
        scenarios=tracking_scenarios,
        checkpoint_sha256=checkpoint_sha256,
        iteration=iteration,
        tracking_profile=profile,
    )
    expected_checks = {
        "legacy_locomotion_9x300": locomotion_passed,
        tracking_check_name: tracking_passed,
    }
    settings = report.get("settings")
    checkpoint = report.get("checkpoint")
    passed = locomotion_passed and tracking_passed
    if (
        report.get("schema_version") != 2
        or report.get("gate") != "microban_teleop_v12_nondeployable_preview_acceptance"
        or report.get("status") != ("pass" if passed else "fail")
        or report.get("simulation_only") is not True
        or report.get("preview_non_deployable") is not True
        or report.get("legacy_preview_read_only") is not False
        or report.get("quality_class") != quality_class
        or report.get("live_simulation_candidate")
        is not (passed and live_candidate_phase)
        or report.get(TELEOP_V12_PREVIEW_INFO_KEY) != marker
        or report.get("checks") != expected_checks
        or not isinstance(checkpoint, dict)
        or checkpoint.get("sha256") != checkpoint_sha256
        or checkpoint.get("iteration") != iteration
        or checkpoint.get("completed_updates") != iteration + 1
        or not isinstance(settings, dict)
        or settings.get("seed") != 42
        or settings.get("steps_per_scenario") != 300
        or settings.get("settle_steps") != 50
        or settings.get("locomotion_scenario_count") != len(_LOCOMOTION_SCENARIOS)
        or settings.get("tracking_profile") != profile
        or settings.get("tracking_scenario_count") != len(tracking_scenarios)
        or settings.get("canonical_deployment_accepted") is not False
        or settings.get("device") != "cuda:0"
        or report.get("source_manifest") != preview_evaluator_source_manifest()
    ):
        raise ValueError("Preview evaluation parent receipt failed strict validation")
    return report


def validate_preview_acceptance_receipt(
    path: str | Path,
    *,
    expected_sha256: str,
    checkpoint_sha256: str,
    iteration: int,
    marker: dict[str, Any],
) -> dict[str, Any]:
    """Authenticate the full acceptance receipt required by the live consumer."""

    if not _LOWER_SHA256.fullmatch(expected_sha256):
        raise ValueError("Preview acceptance receipt SHA-256 must be lowercase hex")
    if not _LOWER_SHA256.fullmatch(checkpoint_sha256):
        raise ValueError("Preview checkpoint SHA-256 must be lowercase hex")
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise ValueError("Preview acceptance receipt cannot be a symlink")
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise ValueError("Preview acceptance receipt must be a regular file")
    if _sha256_file(resolved) != expected_sha256:
        raise ValueError("Preview acceptance receipt SHA-256 mismatch")
    report = json.loads(resolved.read_text())
    validate_preview_evaluation_report(
        report,
        checkpoint_sha256=checkpoint_sha256,
        iteration=iteration,
        marker=marker,
    )
    if (
        report.get("status") != "pass"
        or report.get("live_simulation_candidate") is not True
    ):
        raise ValueError("Preview strict acceptance receipt did not pass")
    if _sha256_file(resolved) != expected_sha256:
        raise ValueError("Preview acceptance receipt changed while validating")
    return report


def reject_preview_checkpoint(infos: object) -> None:
    """Reject either preview marker independently, including partial forgery."""

    if not isinstance(infos, dict):
        return
    if (
        infos.get("preview_non_deployable") is not None
        or (infos.get(TELEOP_V12_PREVIEW_INFO_KEY) is not None)
        or (infos.get(TELEOP_V12_PREVIEW_PHASE1_ACCEPTANCE_INFO_KEY) is not None)
    ):
        raise ValueError(
            "Simulation-only v12 preview checkpoint is forbidden in the "
            "canonical/deployment path"
        )
