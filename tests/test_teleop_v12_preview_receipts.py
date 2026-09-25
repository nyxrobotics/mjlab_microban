"""Fail-closed evidence tests for simulation-only v12 preview receipts."""

from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

from mjlab_microban.robot.microban_hand_fk import microban_hand_fk_metadata
from mjlab_microban.scripts.evaluate_teleop_v12_checkpoint import (
    MINIMUM_SIGNED_RESPONSE,
)
from mjlab_microban.scripts.evaluate_teleop_v12_checkpoint import (
    _acceptance as locomotion_acceptance,
)
from mjlab_microban.scripts.evaluate_teleop_v12_tracking import (
    DIRECTIONAL_RESPONSE_MINIMUM,
    FINAL_PROFILE,
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
    required_tracking_check_names,
    required_tracking_scenario_names,
    target_column_ablation_observation_columns,
)
from mjlab_microban.scripts.promote_teleop_v12_preview_visual import (
    FOOT_P95_MAX_M as VISUAL_FOOT_P95_MAX_M,
)
from mjlab_microban.scripts.promote_teleop_v12_preview_visual import (
    FOOT_RMS_MAX_M as VISUAL_FOOT_RMS_MAX_M,
)
from mjlab_microban.scripts.promote_teleop_v12_preview_visual import (
    FULLBODY_VISUAL_GATE,
    LEARNED_SOURCE_DELTA_MIN,
    _embedded_report_sha256,
    _learned_delta_evidence,
    _target_column_ablation_evidence,
    validate_visual_promotion_report,
    visual_promotion_source_manifest,
)
from mjlab_microban.scripts.promote_teleop_v12_preview_visual import (
    HAND_P95_MAX_M as VISUAL_HAND_P95_MAX_M,
)
from mjlab_microban.scripts.promote_teleop_v12_preview_visual import (
    HAND_RMS_MAX_M as VISUAL_HAND_RMS_MAX_M,
)
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_HMD_JOINT_NAMES,
    MICROBAN_TELEOP_ACTION_JOINT_NAMES,
)
from mjlab_microban.tasks.microban_teleop_v12_preview import (
    _FINAL_TRACKING_CHECKS,
    _FINAL_TRACKING_SCENARIOS,
    TELEOP_V12_PREVIEW_FULLBODY_STRICT_QUALITY,
    TELEOP_V12_PREVIEW_FULLBODY_VISUAL_QUALITY,
    TELEOP_V12_PREVIEW_INFO_KEY,
    TELEOP_V12_PREVIEW_PHASE1_VISUAL_QUALITY,
    TELEOP_V12_PREVIEW_PHASE_FULL_BODY,
    TELEOP_V12_PREVIEW_PHASE_HMD_HAND,
    TELEOP_V12_PREVIEW_SOURCE_SHA256,
    _validate_child_report,
    _validate_tracking_result_evidence,
    preview_evaluator_source_manifest,
    staged_preview_info,
)
from mjlab_microban.teleop_v12_safety import (
    ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD,
)


def _stats(
    *,
    active: bool,
    sample_count: int = 250,
    rms: float = 0.001,
    p95: float = 0.001,
) -> dict:
    if not active:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "rms": None,
            "p95": None,
            "sample_count": 0,
            "units": "m",
        }
    return {
        "min": 0.001,
        "max": max(rms, p95),
        "mean": rms,
        "rms": rms,
        "p95": p95,
        "sample_count": sample_count,
        "units": "m",
    }


def _action_summary() -> dict:
    zero = [0.0] * len(MICROBAN_TELEOP_ACTION_JOINT_NAMES)
    return {
        "minimum": list(zero),
        "maximum": list(zero),
        "absolute_maximum": list(zero),
    }


def _action_envelope(*, learned_delta: float = 0.0) -> dict:
    learned = _action_summary()
    learned["maximum"][0] = learned_delta
    learned["absolute_maximum"][0] = learned_delta
    return {
        "joint_names": list(MICROBAN_TELEOP_ACTION_JOINT_NAMES),
        "v12": _action_summary(),
        "legacy_source": _action_summary(),
        "learned_minus_source": learned,
    }


def _ablation(*, target: str, expected: bool) -> dict:
    ablated_columns, preserved_columns = target_column_ablation_observation_columns(
        target
    )
    return {
        "target_expected": expected,
        "ablated_observation_columns": list(ablated_columns),
        "preserved_observation_columns": list(preserved_columns),
        "maximum_absolute_action_delta": 0.01 if expected else None,
        "minimum_required_action_delta": (
            TARGET_COLUMN_ABLATION_ACTION_DELTA_MIN if expected else None
        ),
        "passed": True,
    }


def _tracking_report(*, visual_quality_failure: bool = False) -> dict:
    results = []
    axes = ("vx_m_s", "vy_m_s", "yaw_rad_s")
    for scenario in _scenarios(FINAL_PROFILE):
        expects_foot = any(
            abs(value) > 0.0 for target in scenario.foot_target for value in target
        )
        expects_hand = any(scenario.hand_active)
        measured = {}
        response = {}
        for axis, command in zip(axes, scenario.twist, strict=True):
            mean = command if command != 0.0 else 0.0
            measured[axis] = {"sample_count": 250, "mean": mean}
            if command != 0.0:
                response[axis] = {
                    "command": command,
                    "measured_mean": mean,
                    "signed_response": abs(mean),
                    "minimum_signed_response": DIRECTIONAL_RESPONSE_MINIMUM[axis],
                    "passed": abs(mean) >= DIRECTIONAL_RESPONSE_MINIMUM[axis],
                }
        hmd_axes = {
            name: {
                "target_peak_to_peak_rad": HMD_TARGET_PEAK_TO_PEAK_MIN_RAD,
                "actual_peak_to_peak_rad": HMD_ACTUAL_PEAK_TO_PEAK_MIN_RAD,
            }
            for name in MICROBAN_HMD_JOINT_NAMES
        }
        results.append(
            {
                "name": scenario.name,
                "command": {
                    "twist": list(scenario.twist),
                    "foot_target": [list(value) for value in scenario.foot_target],
                    "hand_target": [list(value) for value in scenario.hand_target],
                    "hand_active": list(scenario.hand_active),
                },
                "completed": True,
                "executed_steps": 300,
                "fell": False,
                "nonfinite": None,
                "termination_names": [],
                "raw_action_recurrence_verified_steps": 300,
                "maximum_actual_soft_limit_violation_rad": 0.0,
                "hmd_motion": {
                    "joint_names": list(MICROBAN_HMD_JOINT_NAMES),
                    "sample_count": 301,
                    "active_event_member": True,
                    "per_axis": hmd_axes,
                },
                "hmd_motion_evidence_passed": True,
                "observation_coverage": {
                    "hmd_nonzero_steps": 299,
                    "foot_nonzero_steps": 300 if expects_foot else 0,
                    "hand_nonzero_steps": 300 if expects_hand else 0,
                    "foot_target_expected": expects_foot,
                    "hand_target_expected": expects_hand,
                    "passed": True,
                },
                "measured_velocity_body": measured,
                "directional_response": response,
                "twist_directional_response_passed": True,
                "target_error": {
                    "foot": _stats(
                        active=expects_foot,
                        sample_count=(
                            250
                            * sum(
                                any(abs(value) > 0.0 for value in target)
                                for target in scenario.foot_target
                            )
                        ),
                        rms=0.05 if visual_quality_failure else 0.001,
                        p95=0.10 if visual_quality_failure else 0.001,
                    ),
                    "active_hand": _stats(
                        active=expects_hand,
                        sample_count=250 * sum(scenario.hand_active),
                        rms=0.10 if visual_quality_failure else 0.001,
                        p95=0.12 if visual_quality_failure else 0.001,
                    ),
                },
                "target_column_ablation": {
                    "hand": _ablation(target="hand", expected=expects_hand),
                    "foot": _ablation(target="foot", expected=expects_foot),
                },
                "raw_action_envelope": _action_envelope(
                    learned_delta=0.01 if (expects_hand or expects_foot) else 0.0
                ),
            }
        )
    checks, status = _acceptance(results, FINAL_PROFILE)
    return {
        "schema_version": 1,
        "gate": "microban_teleop_v12_tracking",
        "profile": FINAL_PROFILE,
        "status": status,
        "checkpoint": {
            "sha256": "a" * 64,
            "iteration": 10_100,
            "completed_updates": 10_101,
        },
        "settings": {
            "device": "cuda:0",
            "seed": 42,
            "steps": 300,
            "settle_steps": 50,
            "moving_hmd": "forced_non_neutral",
            "perturbation": True,
            "action_clip": None,
            "previous_action": "raw_actor_output",
            "target_column_ablation": TARGET_COLUMN_ABLATION_METHOD,
            "reachable_hand_target_fk": microban_hand_fk_metadata(),
        },
        "thresholds": {
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
        },
        "checks": checks,
        "raw_action_envelope": _aggregate_action_envelopes(results),
        "results": results,
    }


def _locomotion_report() -> dict:
    commands = {
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
    axes = ("vx_m_s", "vy_m_s", "yaw_rad_s")
    results = []
    for name, command in commands.items():
        measured = {
            axis: {"count": 250, "mean": value}
            for axis, value in zip(axes, command, strict=True)
        }
        nonzero = [index for index, value in enumerate(command) if value != 0.0]
        response = None
        if nonzero:
            index = nonzero[0]
            axis = axes[index]
            response = {
                "axis": axis,
                "command": command[index],
                "measured_mean": command[index],
                "signed_response": abs(command[index]),
                "sign_matches": True,
            }
        results.append(
            {
                "name": name,
                "command": dict(zip(axes, command, strict=True)),
                "completed": True,
                "executed_steps": 300,
                "fell": False,
                "nonfinite": None,
                "termination_names": [],
                "raw_action_recurrence_verified_steps": 300,
                "neutral_foot_hand_target_verified_steps": 300,
                "maximum_actual_soft_limit_violation_rad": 0.0,
                "measured_velocity_body": measured,
                "directional_response": response,
            }
        )
    checks, status = locomotion_acceptance(results)
    return {
        "schema_version": 1,
        "gate": "microban_teleop_v12_neutral_locomotion_9x300",
        "status": status,
        "checkpoint": {
            "sha256": "a" * 64,
            "iteration": 10_100,
            "completed_updates": 10_101,
        },
        "settings": {
            "device": "cuda:0",
            "seed": 42,
            "steps": 300,
            "settle_steps": 50,
            "action_clip": None,
            "previous_action": "raw_actor_output",
            "policy_observation_width": 83,
        },
        "thresholds": {
            "actual_soft_limit_violation_rad_max": (
                ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD
            ),
            "minimum_signed_response": MINIMUM_SIGNED_RESPONSE,
        },
        "checks": checks,
        "results": results,
        "summary": {
            "scenario_count": 9,
            "completed_scenario_count": 9,
            "fall_scenario_count": 0,
            "nonfinite_scenario_count": 0,
            "directionally_correct_scenario_count": 8,
            "directional_scenario_count": 8,
        },
    }


def _marker() -> dict:
    return staged_preview_info(
        phase=TELEOP_V12_PREVIEW_PHASE_FULL_BODY,
        phase_source_checkpoint_sha256="b" * 64,
        phase1_acceptance_receipt_sha256="c" * 64,
        phase1_quality_class=TELEOP_V12_PREVIEW_PHASE1_VISUAL_QUALITY,
    )


def _strict_visual_source() -> dict:
    tracking = _tracking_report(visual_quality_failure=True)
    marker = _marker()
    return {
        "schema_version": 2,
        "gate": "microban_teleop_v12_nondeployable_preview_acceptance",
        "status": "fail",
        "simulation_only": True,
        "preview_non_deployable": True,
        "live_simulation_candidate": False,
        "legacy_preview_read_only": False,
        "quality_class": TELEOP_V12_PREVIEW_FULLBODY_STRICT_QUALITY,
        "source_manifest": preview_evaluator_source_manifest(),
        "checkpoint": {
            "path": "/portable/model_10100.pt",
            "sha256": "a" * 64,
            "iteration": 10_100,
            "completed_updates": 10_101,
        },
        TELEOP_V12_PREVIEW_INFO_KEY: marker,
        "settings": {
            "device": "cuda:0",
            "seed": 42,
            "steps_per_scenario": 300,
            "settle_steps": 50,
            "locomotion_scenario_count": 9,
            "tracking_profile": FINAL_PROFILE,
            "tracking_scenario_count": 8,
            "canonical_deployment_accepted": False,
        },
        "checks": {
            "legacy_locomotion_9x300": True,
            "full_body_perturbation_8x300": False,
        },
        "locomotion": _locomotion_report(),
        "tracking": tracking,
    }


def _visual_report() -> dict:
    strict = _strict_visual_source()
    results = strict["tracking"]["results"]
    marker = strict[TELEOP_V12_PREVIEW_INFO_KEY]
    delta = {
        "hand_scenario_max_abs": _learned_delta_evidence(results, "active_hand"),
        "foot_scenario_max_abs": _learned_delta_evidence(results, "foot"),
    }
    ablation = {
        "hand_scenario_max_abs": _target_column_ablation_evidence(
            results, "active_hand", "hand"
        ),
        "foot_scenario_max_abs": _target_column_ablation_evidence(
            results, "foot", "foot"
        ),
    }
    false_checks = sorted(
        name for name, passed in strict["tracking"]["checks"].items() if not passed
    )
    return {
        "schema_version": 1,
        "gate": FULLBODY_VISUAL_GATE,
        "status": "pass",
        "simulation_only": True,
        "preview_non_deployable": True,
        "canonical_deployment_accepted": False,
        "quality_class": TELEOP_V12_PREVIEW_FULLBODY_VISUAL_QUALITY,
        "source_manifest": visual_promotion_source_manifest(),
        "checkpoint": {
            "path": "/portable/model_10100.pt",
            "sha256": "a" * 64,
            "iteration": 10_100,
            "completed_updates": 10_101,
        },
        TELEOP_V12_PREVIEW_INFO_KEY: marker,
        "source_evaluation_receipt": {
            "path": "/portable/strict.json",
            "sha256": _embedded_report_sha256(strict),
            "schema_version": 2,
            "gate": "microban_teleop_v12_nondeployable_preview_acceptance",
            "status": "fail",
        },
        "thresholds": {
            "hand_rms_m_max": VISUAL_HAND_RMS_MAX_M,
            "hand_p95_m_max": VISUAL_HAND_P95_MAX_M,
            "foot_rms_m_max": VISUAL_FOOT_RMS_MAX_M,
            "foot_p95_m_max": VISUAL_FOOT_P95_MAX_M,
            "learned_source_delta_min": LEARNED_SOURCE_DELTA_MIN,
        },
        "evidence": {
            "maximum_hand_rms_m": 0.10,
            "maximum_hand_p95_m": 0.12,
            "maximum_foot_rms_m": 0.05,
            "maximum_foot_p95_m": 0.10,
            "strict_failed_tracking_checks": false_checks,
            "learned_source_delta_coverage": delta,
            "target_column_ablation": ablation,
        },
        "checks": {
            "source_hash_bound": True,
            "checkpoint_hash_bound": True,
            "strict_hard_safety_passed": True,
            "strict_locomotion_passed": True,
            "only_tracking_quality_failed": True,
            "relaxed_hand_rms_passed": True,
            "relaxed_hand_p95_passed": True,
            "learned_source_delta_coverage_passed": True,
            "canonical_deployment_forbidden": True,
            "relaxed_foot_rms_passed": True,
            "relaxed_foot_p95_passed": True,
            "target_column_ablation_response_passed": True,
        },
        "strict_evaluation": strict,
    }


class PreviewReceiptEvidenceTest(unittest.TestCase):
    def test_fullbody_result_evidence_is_self_consistent(self) -> None:
        _validate_tracking_result_evidence(_tracking_report(), profile=FINAL_PROFILE)

    def test_raw_safety_and_motion_tampering_is_rejected(self) -> None:
        mutations = (
            lambda report: report["results"][0].update(fell=True),
            lambda report: report["results"][0].update(
                maximum_actual_soft_limit_violation_rad=999.0
            ),
            lambda report: report["results"][0].update(
                maximum_actual_soft_limit_violation_rad=-999.0
            ),
            lambda report: report["results"][1]["target_error"]["active_hand"].update(
                min=-1.0, max=-1.0, mean=-1.0, rms=-1.0, p95=-1.0
            ),
            lambda report: report["results"][0]["hmd_motion"]["per_axis"][
                MICROBAN_HMD_JOINT_NAMES[0]
            ].update(actual_peak_to_peak_rad=0.0),
            lambda report: report["results"][0]["directional_response"][
                "vx_m_s"
            ].update(signed_response=999.0),
            lambda report: report["settings"].update(
                target_column_ablation="zero_all_hand_columns"
            ),
            lambda report: next(
                result
                for result in report["results"]
                if result["target_column_ablation"]["hand"]["target_expected"]
            )["target_column_ablation"]["hand"].update(
                ablated_observation_columns=list(range(75, 83))
            ),
            lambda report: next(
                result
                for result in report["results"]
                if result["target_column_ablation"]["hand"]["target_expected"]
            )["target_column_ablation"]["hand"].update(
                preserved_observation_columns=[]
            ),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                report = _tracking_report()
                mutate(report)
                with self.assertRaises(ValueError):
                    _validate_tracking_result_evidence(report, profile=FINAL_PROFILE)

    def test_child_report_rejects_cpu_and_non_dict_scenario(self) -> None:
        report = _tracking_report()
        with self.assertRaisesRegex(ValueError, "strict validation"):
            cpu = copy.deepcopy(report)
            cpu["settings"]["device"] = "cpu"
            _validate_child_report(
                cpu,
                gate="microban_teleop_v12_tracking",
                checks=set(required_tracking_check_names(FINAL_PROFILE)),
                scenarios=required_tracking_scenario_names(FINAL_PROFILE),
                checkpoint_sha256="a" * 64,
                iteration=10_100,
                tracking_profile=FINAL_PROFILE,
            )
        with self.assertRaisesRegex(ValueError, "strict validation"):
            malformed = copy.deepcopy(report)
            malformed["results"][-1] = None
            _validate_child_report(
                malformed,
                gate="microban_teleop_v12_tracking",
                checks=set(required_tracking_check_names(FINAL_PROFILE)),
                scenarios=required_tracking_scenario_names(FINAL_PROFILE),
                checkpoint_sha256="a" * 64,
                iteration=10_100,
                tracking_profile=FINAL_PROFILE,
            )

    def test_self_contained_fullbody_visual_receipt_passes(self) -> None:
        report = _visual_report()
        validated = validate_visual_promotion_report(
            report,
            checkpoint_sha256="a" * 64,
            iteration=10_100,
            marker=report[TELEOP_V12_PREVIEW_INFO_KEY],
        )
        self.assertIs(validated, report)
        self.assertEqual(
            set(report["strict_evaluation"]["tracking"]["checks"]),
            _FINAL_TRACKING_CHECKS,
        )
        self.assertEqual(
            tuple(
                result["name"]
                for result in report["strict_evaluation"]["tracking"]["results"]
            ),
            _FINAL_TRACKING_SCENARIOS,
        )

    def test_phase1_visual_requires_hand_target_column_ablation(self) -> None:
        strict = {
            "status": "fail",
            "locomotion": {"checks": {"hard": True}},
            "tracking": {
                "checks": {"hand_tracking_rms": False},
                "results": _tracking_report(visual_quality_failure=True)["results"],
            },
        }
        active = next(
            result
            for result in strict["tracking"]["results"]
            if result["target_error"]["active_hand"]["sample_count"] > 0
        )
        active["target_column_ablation"]["hand"].update(
            maximum_absolute_action_delta=LEARNED_SOURCE_DELTA_MIN,
            passed=False,
        )
        marker = staged_preview_info(
            phase=TELEOP_V12_PREVIEW_PHASE_HMD_HAND,
            phase_source_checkpoint_sha256=TELEOP_V12_PREVIEW_SOURCE_SHA256,
        )
        with (
            patch(
                "mjlab_microban.scripts.promote_teleop_v12_preview_visual."
                "validate_preview_evaluation_report"
            ),
            self.assertRaisesRegex(ValueError, "ablation response"),
        ):
            validate_visual_promotion_report(
                {"strict_evaluation": strict},
                checkpoint_sha256="a" * 64,
                iteration=7_100,
                marker=marker,
            )

    def test_visual_receipt_rejects_hard_threshold_ablation_and_manifest_tamper(
        self,
    ) -> None:
        mutations = (
            lambda report: report["strict_evaluation"]["tracking"]["results"][0].update(
                fell=True
            ),
            lambda report: report["strict_evaluation"]["tracking"]["results"][0].update(
                maximum_actual_soft_limit_violation_rad=-999.0
            ),
            lambda report: report["strict_evaluation"]["tracking"]["results"][1][
                "target_error"
            ]["active_hand"].update(rms=0.20),
            lambda report: report["strict_evaluation"]["tracking"]["results"][1][
                "target_error"
            ]["active_hand"].update(
                min=-1.0,
                max=-1.0,
                mean=-1.0,
                rms=-1.0,
                p95=-1.0,
            ),
            lambda report: report["strict_evaluation"]["tracking"]["results"][1][
                "target_column_ablation"
            ]["hand"].update(
                maximum_absolute_action_delta=LEARNED_SOURCE_DELTA_MIN,
                passed=False,
            ),
            lambda report: report["source_manifest"].update(aggregate_sha256="e" * 64),
            lambda report: report["source_evaluation_receipt"].update(sha256="d" * 64),
            lambda report: report["strict_evaluation"]["tracking"]["settings"].update(
                device="cpu"
            ),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                report = _visual_report()
                mutate(report)
                with self.assertRaises(ValueError):
                    validate_visual_promotion_report(
                        report,
                        checkpoint_sha256="a" * 64,
                        iteration=10_100,
                        marker=report[TELEOP_V12_PREVIEW_INFO_KEY],
                    )


if __name__ == "__main__":
    unittest.main()
