"""CPU-only regression tests for v12 stage routing and evidence gates."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from copy import deepcopy
from unittest import mock
from pathlib import Path

import torch

from mjlab_microban.robot.microban_hand_fk import (
    microban_hand_fk_metadata,
    microban_reachable_hand_evaluation_offsets,
)
from mjlab_microban.scripts.evaluate_teleop_v12_checkpoint import (
    MINIMUM_SIGNED_RESPONSE,
)
from mjlab_microban.scripts.evaluate_teleop_v12_checkpoint import (
    _acceptance as _locomotion_acceptance,
)
from mjlab_microban.scripts.evaluate_teleop_v12_tracking import (
    DEADLINE_CANARY_FALLBACK_PROFILE,
    DEADLINE_FINAL_FALLBACK_PROFILE,
    DIRECTIONAL_RESPONSE_MINIMUM,
    EXPANDED_LOCOMOTION_PROFILE,
    FINAL_COMPLETION_ALLOWANCE_PROFILE,
    FINAL_DEPLOYED_ACCURACY_PROFILE,
    FINAL_PROFILE,
    FOOT_ACTIVATION_CANARY_DEPLOYED_ACCURACY_PROFILE,
    FOOT_ACTIVATION_CANARY_PROFILE,
    HMD_ACTUAL_PEAK_TO_PEAK_MIN_RAD,
    HMD_HAND_ACTIVATION_CANARY_PROFILE,
    HMD_HAND_DEPLOYED_ACCURACY_PROFILE,
    HMD_HAND_HAND_RMS_40MM_PROFILE,
    FOOT_CANARY_HAND_RMS_40MM_PROFILE,
    HMD_HAND_PROFILE,
    HMD_TARGET_PEAK_TO_PEAK_MIN_RAD,
    PRE_ACTIVATION_EXPOSURE_PROFILE,
    STRICT_PROFILE_BY_COMPLETION_ALLOWANCE_PROFILE,
    STRICT_PROFILE_BY_DEPLOYED_ACCURACY_PROFILE,
    TARGET_COLUMN_ABLATION_ACTION_DELTA_MIN,
    TARGET_COLUMN_ABLATION_METHOD,
    TRACKING_PROFILES,
    WHOLE_BODY_DEPLOYED_ACCURACY_PROFILE,
    WHOLE_BODY_PROFILE,
    _acceptance,
    _active_foot_tracking_error,
    _aggregate_action_envelopes,
    _scenarios,
    accepted_tracking_profiles,
    foot_tracking_p95_max_m,
    foot_tracking_rms_max_m,
    hand_tracking_p95_max_m,
    hand_tracking_rms_max_m,
    required_target_column_ablation_targets,
    required_tracking_check_names,
    required_tracking_profile,
    required_tracking_scenario_names,
    target_column_ablated_observation,
    target_column_ablation_observation_columns,
    tracking_profile_completion_allowance,
    tracking_profile_uses_perturbation,
)
from mjlab_microban.scripts.teleop_v12_bootstrap_gate import (
    ONNX_PARITY_TOLERANCE,
    PRISTINE_PARITY_TOLERANCE,
)
from mjlab_microban.scripts.teleop_v12_stage import (
    _checkpoint_kind,
    _validate_tracking_report,
    checkpoint_recipe_kind,
    create_gate,
    next_training_target,
    validate_gate,
)
from mjlab_microban.tasks.mdp import MICROBAN_BILATERAL_SITE_ORDER_REVISION
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_HMD_JOINT_NAMES,
    MICROBAN_TELEOP_ACTION_JOINT_NAMES,
)
from mjlab_microban.tasks.microban_teleop_v12_actor import (
    TELEOP_V12_ADAPTER_GRADIENT_SCHEDULE_REVISION,
    TELEOP_V12_ADAPTER_SANITIZATION_REVISION,
    TELEOP_V12_ADAPTER_SANITIZATION_SCHEMA_VERSION,
    TELEOP_V12_EXTRA_OBSERVATION_COLUMNS,
    TELEOP_V12_FOOT_OBSERVATION_COLUMNS,
    TELEOP_V12_TARGET_POSITION_NORMALIZER_STORED_STD,
    teleop_v12_active_adapter_columns,
    teleop_v12_target_normalizer_metadata,
)
from mjlab_microban.tasks.microban_teleop_v12_bootstrap import sha256_file
from mjlab_microban.tasks.microban_teleop_v12_corner_rescue import (
    MICROBAN_TELEOP_V12_CORNER_RESCUE_ACTIVE_COLUMNS,
    MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY,
    MICROBAN_TELEOP_V12_CORNER_RESCUE_RECIPE_REVISION,
    MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_OPTIMIZER_STEP,
    corner_rescue_marker,
)
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION,
    MICROBAN_TELEOP_V12_RECIPE_REVISION,
)
from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release_lineage import (
    HAND_POSE_RELEASE_RECIPE_SWITCH_PARENT_CHECKPOINT_SHA256,
    MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_EXPERIMENTAL_SWITCH_INFO_KEY,
    MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_SWITCH_INFO_KEY,
    hand_pose_release_recipe_switch_marker,
)
from mjlab_microban.tasks.microban_teleop_v12_home_pose import (
    TELEOP_V12_HOME_POSE_INFO_KEY,
    teleop_v12_home_pose_marker,
)
from mjlab_microban.tasks.microban_teleop_v12_lr_order import (
    BILATERAL_SITE_ORDER_INFO_KEY,
)
from mjlab_microban.teleop_v12_safety import (
    ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD,
)


def _ablation(
    *, target: str, expected: bool, maximum: float = 0.01
) -> dict[str, object]:
    ablated_columns, preserved_columns = target_column_ablation_observation_columns(
        target
    )
    return {
        "target_expected": expected,
        "ablated_observation_columns": list(ablated_columns),
        "preserved_observation_columns": list(preserved_columns),
        "maximum_absolute_action_delta": maximum if expected else None,
        "minimum_required_action_delta": (
            TARGET_COLUMN_ABLATION_ACTION_DELTA_MIN if expected else None
        ),
        "passed": (not expected or maximum > TARGET_COLUMN_ABLATION_ACTION_DELTA_MIN),
    }


def _result(**overrides):
    value = {
        "completed": True,
        "fell": False,
        "nonfinite": None,
        "maximum_actual_soft_limit_violation_rad": 0.0,
        "raw_action_recurrence_verified_steps": 300,
        "executed_steps": 300,
        "hmd_motion_evidence_passed": True,
        "observation_coverage": {"passed": True},
        "twist_directional_response_passed": True,
        "target_error": {
            "active_hand": {"sample_count": 1, "rms": 0.01, "p95": 0.02},
            "foot": {"sample_count": 1, "rms": 0.01, "p95": 0.02},
        },
        "command": {
            "foot_target": [[0.0, 0.0, 0.02], [0.0, 0.0, 0.0]],
            "hand_active": [True, True],
        },
        "target_column_ablation": {
            "hand": _ablation(target="hand", expected=True),
            "foot": _ablation(target="foot", expected=True),
        },
    }
    value.update(overrides)
    return value


def _locomotion_report(identity: dict[str, object]) -> dict[str, object]:
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
    axis_names = ("vx_m_s", "vy_m_s", "yaw_rad_s")
    results = []
    for name in ("neutral", *MINIMUM_SIGNED_RESPONSE):
        twist = commands[name]
        measured = {axis: 0.0 for axis in axis_names}
        response = None
        if name != "neutral":
            index = next(index for index, value in enumerate(twist) if value != 0.0)
            axis = axis_names[index]
            measured[axis] = (
                MINIMUM_SIGNED_RESPONSE[name]
                if twist[index] > 0.0
                else -MINIMUM_SIGNED_RESPONSE[name]
            )
            response = {
                "axis": axis,
                "command": twist[index],
                "measured_mean": measured[axis],
                "sign_matches": True,
                "signed_response": MINIMUM_SIGNED_RESPONSE[name],
            }
        results.append(
            {
                "name": name,
                "completed": True,
                "executed_steps": 300,
                "fell": False,
                "nonfinite": None,
                "termination_names": [],
                "maximum_actual_soft_limit_violation_rad": 0.0,
                "raw_action_recurrence_verified_steps": 300,
                "neutral_foot_hand_target_verified_steps": 300,
                "directional_response": response,
                "command": dict(zip(axis_names, twist, strict=True)),
                "measured_velocity_body": {
                    axis: {"count": 250, "mean": mean}
                    for axis, mean in measured.items()
                },
            }
        )
    checks, status = _locomotion_acceptance(results)
    return {
        "schema_version": 1,
        "gate": "microban_teleop_v12_neutral_locomotion_9x300",
        "status": status,
        "checkpoint": identity,
        "settings": {
            "device": "cpu",
            "seed": 42,
            "steps": 300,
            "settle_steps": 50,
            "action_clip": [-math.pi, math.pi],
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
            "minimum_signed_response": min(MINIMUM_SIGNED_RESPONSE.values()),
            "maximum_actual_soft_limit_violation_rad": 0.0,
        },
    }


def _zero_action_envelope() -> dict[str, object]:
    zero = [0.0] * 18
    summary = {
        "minimum": zero.copy(),
        "maximum": zero.copy(),
        "absolute_maximum": zero.copy(),
    }
    return {
        "joint_names": list(MICROBAN_TELEOP_ACTION_JOINT_NAMES),
        "v12": deepcopy(summary),
        "legacy_source": deepcopy(summary),
        "learned_minus_source": deepcopy(summary),
    }


def _tracking_report(
    identity: dict[str, object],
    *,
    profile: str | None = None,
    hand_rms: float = 0.01,
) -> dict[str, object]:
    if profile is None:
        profile = required_tracking_profile(int(identity["completed_updates"]))
    results = []
    for scenario in _scenarios(profile):
        expects_foot = any(
            abs(value) > 0.0 for target in scenario.foot_target for value in target
        )
        expects_hand = any(scenario.hand_active)
        directional = {}
        for axis, command in zip(
            ("vx_m_s", "vy_m_s", "yaw_rad_s"), scenario.twist, strict=True
        ):
            if command == 0.0:
                continue
            signed = DIRECTIONAL_RESPONSE_MINIMUM[axis]
            directional[axis] = {
                "command": command,
                "measured_mean": signed if command > 0 else -signed,
                "signed_response": signed,
                "minimum_signed_response": signed,
                "passed": True,
            }
        hmd_axes = {
            name: {
                "target_peak_to_peak_rad": HMD_TARGET_PEAK_TO_PEAK_MIN_RAD,
                "actual_peak_to_peak_rad": HMD_ACTUAL_PEAK_TO_PEAK_MIN_RAD,
            }
            for name in MICROBAN_HMD_JOINT_NAMES
        }

        def target_error(
            *, expected: bool, samples: int, hand: bool = False
        ) -> dict[str, object]:
            if not expected:
                return {
                    "sample_count": 0,
                    "min": None,
                    "max": None,
                    "mean": None,
                    "rms": None,
                    "p95": None,
                    "units": "m",
                }
            return {
                "sample_count": samples,
                "min": 0.005,
                "max": max(0.02, hand_rms) if hand else 0.02,
                "mean": 0.008,
                "rms": hand_rms if hand else 0.01,
                "p95": 0.015,
                "units": "m",
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
                "maximum_actual_soft_limit_violation_rad": 0.0,
                "raw_action_recurrence_verified_steps": 300,
                "hmd_motion_evidence_passed": True,
                "hmd_motion": {
                    "joint_names": list(MICROBAN_HMD_JOINT_NAMES),
                    "sample_count": 301,
                    "active_event_member": True,
                    "per_axis": hmd_axes,
                },
                "observation_coverage": {
                    "hmd_nonzero_steps": 299,
                    "foot_nonzero_steps": 300 if expects_foot else 0,
                    "hand_nonzero_steps": 300 if expects_hand else 0,
                    "foot_target_expected": expects_foot,
                    "hand_target_expected": expects_hand,
                    "passed": True,
                },
                "directional_response": directional,
                "twist_directional_response_passed": True,
                "measured_velocity_body": {
                    axis: {
                        "sample_count": 250,
                        "mean": (
                            directional[axis]["measured_mean"]
                            if axis in directional
                            else 0.0
                        ),
                    }
                    for axis in ("vx_m_s", "vy_m_s", "yaw_rad_s")
                },
                "target_error": {
                    "active_hand": target_error(
                        expected=expects_hand,
                        hand=True,
                        samples=250
                        * sum(bool(value) for value in scenario.hand_active),
                    ),
                    "foot": target_error(
                        expected=expects_foot,
                        samples=250
                        * sum(
                            any(abs(value) > 0.0 for value in target)
                            for target in scenario.foot_target
                        ),
                    ),
                },
                "target_column_ablation": {
                    "hand": _ablation(target="hand", expected=expects_hand),
                    "foot": _ablation(target="foot", expected=expects_foot),
                },
                "raw_action_envelope": _zero_action_envelope(),
            }
        )
    checks, status = _acceptance(results, profile)
    return {
        "schema_version": 1,
        "gate": "microban_teleop_v12_tracking",
        "profile": profile,
        "status": status,
        "checkpoint": identity,
        "settings": {
            "device": "cpu",
            "seed": 42,
            "steps": 300,
            "settle_steps": 50,
            "moving_hmd": "forced_non_neutral",
            "perturbation": tracking_profile_uses_perturbation(profile),
            "action_clip": [-math.pi, math.pi],
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
            "hand_rms_m_max": hand_tracking_rms_max_m(profile),
            "hand_p95_m_max": hand_tracking_p95_max_m(profile),
            "foot_rms_m_max": foot_tracking_rms_max_m(profile),
            "foot_p95_m_max": foot_tracking_p95_max_m(profile),
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


def _onnx_report(identity: dict[str, object], onnx_path: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "gate": "microban_teleop_v12_checkpoint_onnx",
        "status": "pass",
        "checkpoint": identity,
        "neutral_legacy_parity": {
            "samples": 10_000,
            "maximum_absolute_error": 0.0,
            "tolerance": PRISTINE_PARITY_TOLERANCE,
            "teleop_only_columns": "exact_zero",
        },
        "onnx": {
            "path": str(onnx_path),
            "sha256": sha256_file(onnx_path),
            "opset": 18,
            "input_shape": [1, 83],
            "output_shape": [1, 18],
            "reference_samples": 64,
            "input_coverage": "deterministic_nonzero_all_83_columns",
            "teleop_only_columns_nonzero": True,
            "reference_evaluator_maximum_absolute_error": 0.0,
            "onnxruntime_cpu_maximum_absolute_error": 0.0,
            "onnxruntime_version": "unit-test",
            "onnxruntime_providers": ["CPUExecutionProvider"],
            "tolerance": ONNX_PARITY_TOLERANCE,
        },
    }


class TeleopV12StageTest(unittest.TestCase):
    def test_tracking_scenarios_use_fk_reachable_hand_targets(self) -> None:
        poses = dict(microban_reachable_hand_evaluation_offsets())
        scenarios = {scenario.name: scenario for scenario in _scenarios(FINAL_PROFILE)}
        self.assertEqual(
            scenarios["max_hands_left"].hand_target,
            (poses["F"][0], poses["B"][1]),
        )
        self.assertEqual(
            scenarios["max_hands_right"].hand_target,
            (poses["B"][0], poses["F"][1]),
        )
        self.assertEqual(
            scenarios["mixed_forward_left"].hand_target,
            (poses["f"][0], poses["b"][1]),
        )
        self.assertEqual(
            scenarios["mixed_backward_right"].hand_target,
            (poses["b"][0], poses["f"][1]),
        )

    def test_route_covers_recovery_boundaries_and_activation_canaries(self) -> None:
        expected = {
            601: (3_000, False),
            2_999: (3_000, False),
            3_000: (3_100, True),
            3_001: (3_100, True),
            3_099: (3_100, True),
            3_100: (7_000, False),
            7_000: (7_100, True),
            7_099: (7_100, True),
            7_100: (10_000, False),
            10_000: (10_100, True),
            10_099: (10_100, True),
            10_100: (15_000, False),
            14_999: (15_000, False),
        }
        for completed, route in expected.items():
            with self.subTest(completed=completed):
                self.assertEqual(next_training_target(completed), route)
        with self.assertRaisesRegex(ValueError, "Final"):
            next_training_target(15_000)

    def test_tracking_profiles_follow_trained_then_next_exposure_semantics(
        self,
    ) -> None:
        self.assertEqual(
            required_tracking_profile(601), PRE_ACTIVATION_EXPOSURE_PROFILE
        )
        self.assertEqual(
            required_tracking_profile(3_000), PRE_ACTIVATION_EXPOSURE_PROFILE
        )
        self.assertEqual(required_tracking_profile(3_001), EXPANDED_LOCOMOTION_PROFILE)
        self.assertEqual(required_tracking_profile(7_000), EXPANDED_LOCOMOTION_PROFILE)
        self.assertEqual(
            required_tracking_profile(7_001), HMD_HAND_ACTIVATION_CANARY_PROFILE
        )
        self.assertEqual(
            required_tracking_profile(7_100), HMD_HAND_ACTIVATION_CANARY_PROFILE
        )
        self.assertEqual(
            required_tracking_profile(7_101), HMD_HAND_DEPLOYED_ACCURACY_PROFILE
        )
        self.assertEqual(
            required_tracking_profile(10_000), HMD_HAND_DEPLOYED_ACCURACY_PROFILE
        )
        self.assertEqual(
            required_tracking_profile(10_001),
            FOOT_ACTIVATION_CANARY_DEPLOYED_ACCURACY_PROFILE,
        )
        self.assertEqual(
            required_tracking_profile(10_100),
            FOOT_ACTIVATION_CANARY_DEPLOYED_ACCURACY_PROFILE,
        )
        self.assertEqual(
            required_tracking_profile(10_101), WHOLE_BODY_DEPLOYED_ACCURACY_PROFILE
        )
        self.assertEqual(
            required_tracking_profile(14_999), WHOLE_BODY_DEPLOYED_ACCURACY_PROFILE
        )
        self.assertEqual(
            required_tracking_profile(15_000), FINAL_DEPLOYED_ACCURACY_PROFILE
        )
        self.assertEqual(accepted_tracking_profiles(7_100), (
            HMD_HAND_ACTIVATION_CANARY_PROFILE,
        ))
        self.assertEqual(accepted_tracking_profiles(10_000), (
            HMD_HAND_DEPLOYED_ACCURACY_PROFILE,
            HMD_HAND_PROFILE,
        ))
        self.assertEqual(accepted_tracking_profiles(10_100), (
            FOOT_ACTIVATION_CANARY_DEPLOYED_ACCURACY_PROFILE,
            FOOT_ACTIVATION_CANARY_PROFILE,
        ))
        self.assertEqual(accepted_tracking_profiles(15_000), (
            FINAL_DEPLOYED_ACCURACY_PROFILE,
            FINAL_PROFILE,
        ))

    def test_deployed_accuracy_profiles_relax_only_accuracy_limits(self) -> None:
        # User instruction (2026-10-04): judge with the deployed model's
        # accuracy (hand 0.035 m, foot 0.05 m); every other check is unchanged.
        # (The 10000/10100 profiles do not check feet; their reported foot
        # limits are the deployed ones too.)
        expected_limits = {
            HMD_HAND_DEPLOYED_ACCURACY_PROFILE: (0.035, 0.05, 0.05, 0.08),
            FOOT_ACTIVATION_CANARY_DEPLOYED_ACCURACY_PROFILE: (
                0.035,
                0.05,
                0.05,
                0.08,
            ),
            WHOLE_BODY_DEPLOYED_ACCURACY_PROFILE: (0.035, 0.07, 0.05, 0.08),
            FINAL_DEPLOYED_ACCURACY_PROFILE: (0.035, 0.07, 0.05, 0.08),
        }
        self.assertEqual(
            set(expected_limits), set(STRICT_PROFILE_BY_DEPLOYED_ACCURACY_PROFILE)
        )
        for deployed, limits in expected_limits.items():
            strict = STRICT_PROFILE_BY_DEPLOYED_ACCURACY_PROFILE[deployed]
            with self.subTest(profile=deployed):
                self.assertEqual(deployed, f"{strict}_deployed_accuracy_v1")
                self.assertIn(deployed, TRACKING_PROFILES)
                self.assertIn(strict, TRACKING_PROFILES)
                self.assertEqual(
                    (
                        hand_tracking_rms_max_m(deployed),
                        hand_tracking_p95_max_m(deployed),
                        foot_tracking_rms_max_m(deployed),
                        foot_tracking_p95_max_m(deployed),
                    ),
                    limits,
                )
                self.assertEqual(
                    (
                        hand_tracking_rms_max_m(strict),
                        hand_tracking_p95_max_m(strict),
                        foot_tracking_rms_max_m(strict),
                        foot_tracking_p95_max_m(strict),
                    ),
                    (0.03, 0.05, 0.015, 0.025),
                )
                self.assertEqual(
                    required_tracking_scenario_names(deployed),
                    required_tracking_scenario_names(strict),
                )
                self.assertEqual(_scenarios(deployed), _scenarios(strict))
                self.assertEqual(
                    required_tracking_check_names(deployed),
                    required_tracking_check_names(strict),
                )
                self.assertEqual(
                    required_target_column_ablation_targets(deployed),
                    required_target_column_ablation_targets(strict),
                )
                self.assertEqual(
                    tracking_profile_uses_perturbation(deployed),
                    tracking_profile_uses_perturbation(strict),
                )
        self.assertTrue(tracking_profile_uses_perturbation(FINAL_PROFILE))
        self.assertFalse(tracking_profile_uses_perturbation(WHOLE_BODY_PROFILE))

    def test_stage_accepts_deployed_and_strict_canary_reports_only(self) -> None:
        identity = {
            "sha256": "a" * 64,
            "iteration": 10_099,
            "completed_updates": 10_100,
        }
        # model_10099 of the 2026-10-04 chain: hand RMS 0.0327 m.
        deployed = _tracking_report(identity, hand_rms=0.0327)
        self.assertEqual(
            deployed["profile"], FOOT_ACTIVATION_CANARY_DEPLOYED_ACCURACY_PROFILE
        )
        self.assertEqual(deployed["status"], "pass")
        self.assertEqual(deployed["thresholds"]["hand_rms_m_max"], 0.035)
        self.assertEqual(
            _validate_tracking_report(deployed, identity),
            FOOT_ACTIVATION_CANARY_DEPLOYED_ACCURACY_PROFILE,
        )
        strict_fail = _tracking_report(
            identity, profile=FOOT_ACTIVATION_CANARY_PROFILE, hand_rms=0.0327
        )
        self.assertEqual(strict_fail["status"], "fail")
        self.assertEqual(
            {name for name, value in strict_fail["checks"].items() if not value},
            {"hand_tracking_rms"},
        )
        with self.assertRaises(ValueError):
            _validate_tracking_report(strict_fail, identity)
        strict_pass = _tracking_report(identity, profile=FOOT_ACTIVATION_CANARY_PROFILE)
        self.assertEqual(
            _validate_tracking_report(strict_pass, identity),
            FOOT_ACTIVATION_CANARY_PROFILE,
        )
        over = _tracking_report(identity, hand_rms=0.0351)
        self.assertEqual(over["status"], "fail")
        with self.assertRaises(ValueError):
            _validate_tracking_report(over, identity)
        for profile in (DEADLINE_CANARY_FALLBACK_PROFILE, WHOLE_BODY_PROFILE):
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                _validate_tracking_report(
                    _tracking_report(identity, profile=profile), identity
                )
        relabelled = deepcopy(deployed)
        relabelled["profile"] = FOOT_ACTIVATION_CANARY_PROFILE
        with self.assertRaisesRegex(ValueError, "thresholds drifted"):
            _validate_tracking_report(relabelled, identity)

    def test_canary_gate_records_and_revalidates_either_accepted_profile(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model_10099.pt"
            infos = {
                "microban_teleop_training_contract_version": "12",
                "microban_teleop_recipe_revision": MICROBAN_TELEOP_V12_RECIPE_REVISION,
                BILATERAL_SITE_ORDER_INFO_KEY: MICROBAN_BILATERAL_SITE_ORDER_REVISION,
                TELEOP_V12_HOME_POSE_INFO_KEY: teleop_v12_home_pose_marker(),
                "adapter_gradient_schedule_revision": (
                    TELEOP_V12_ADAPTER_GRADIENT_SCHEDULE_REVISION
                ),
                "active_actor_columns_at_save": list(
                    teleop_v12_active_adapter_columns(10_100 * 24)
                ),
                "env_state": {"common_step_counter": 10_100 * 24},
            }
            torch.save({"iter": 10_099, "infos": infos}, checkpoint)
            identity = {
                "sha256": sha256_file(checkpoint),
                "iteration": 10_099,
                "completed_updates": 10_100,
            }
            locomotion_path = root / "locomotion.json"
            onnx_report_path = root / "onnx.json"
            onnx_path = root / "policy.onnx"
            onnx_path.write_bytes(b"unit-test-onnx")
            locomotion_path.write_text(json.dumps(_locomotion_report(identity)))
            onnx_report_path.write_text(
                json.dumps(_onnx_report(identity, onnx_path))
            )
            for profile, hand_rms in (
                (FOOT_ACTIVATION_CANARY_DEPLOYED_ACCURACY_PROFILE, 0.0327),
                (FOOT_ACTIVATION_CANARY_PROFILE, 0.01),
            ):
                with self.subTest(profile=profile):
                    tracking_path = root / f"tracking_{profile}.json"
                    tracking_path.write_text(
                        json.dumps(
                            _tracking_report(
                                identity, profile=profile, hand_rms=hand_rms
                            )
                        )
                    )
                    gate = create_gate(
                        checkpoint=checkpoint,
                        locomotion_report=locomotion_path,
                        tracking_report=tracking_path,
                        onnx_report=onnx_report_path,
                    )
                    self.assertEqual(gate["tracking_profile"], profile)
                    self.assertEqual(gate["checkpoint_kind"], "activation_canary")
                    gate_path = root / f"gate_{profile}.json"
                    gate_path.write_text(json.dumps(gate))
                    self.assertEqual(validate_gate(gate_path, checkpoint), gate)
                    other = next(
                        name
                        for name in accepted_tracking_profiles(10_100)
                        if name != profile
                    )
                    tampered = deepcopy(gate)
                    tampered["tracking_profile"] = other
                    gate_path.write_text(json.dumps(tampered))
                    with self.assertRaises(ValueError):
                        validate_gate(gate_path, checkpoint)

    def test_completion_allowance_is_the_pose_release_final_profile_only(
        self,
    ) -> None:
        pose_release = MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION
        allowance = FINAL_COMPLETION_ALLOWANCE_PROFILE
        self.assertEqual(
            allowance,
            "full_body_reachable_performance_perturbation_v2_completion_allowance_v1",
        )
        self.assertIn(allowance, TRACKING_PROFILES)
        self.assertEqual(
            required_tracking_profile(15_000, recipe_revision=pose_release),
            allowance,
        )
        for recipe in (None, MICROBAN_TELEOP_V12_RECIPE_REVISION, "other"):
            with self.subTest(recipe=recipe):
                self.assertEqual(
                    required_tracking_profile(15_000, recipe_revision=recipe),
                    FINAL_DEPLOYED_ACCURACY_PROFILE,
                )
        # Every other clock except the 10000 boundary and the 10100 canary
        # (hand-RMS allowances, tested below) keeps its profile for the
        # pose-release lineage.
        for completed in (3_000, 7_000, 7_100, 9_999, 10_099, 14_999):
            with self.subTest(completed=completed):
                self.assertEqual(
                    required_tracking_profile(completed, recipe_revision=pose_release),
                    required_tracking_profile(completed),
                )
                self.assertEqual(
                    accepted_tracking_profiles(
                        completed, recipe_revision=pose_release
                    ),
                    accepted_tracking_profiles(completed),
                )
        self.assertEqual(
            accepted_tracking_profiles(15_000, recipe_revision=pose_release),
            (allowance, FINAL_DEPLOYED_ACCURACY_PROFILE, FINAL_PROFILE),
        )
        self.assertEqual(
            (
                hand_tracking_rms_max_m(allowance),
                hand_tracking_p95_max_m(allowance),
                foot_tracking_rms_max_m(allowance),
                foot_tracking_p95_max_m(allowance),
            ),
            (0.045, 0.08, 0.055, 0.11),
        )
        # Only accuracy limits change: structure is the final profile's.
        self.assertEqual(
            STRICT_PROFILE_BY_COMPLETION_ALLOWANCE_PROFILE, {allowance: FINAL_PROFILE}
        )
        self.assertNotIn(allowance, STRICT_PROFILE_BY_DEPLOYED_ACCURACY_PROFILE)
        self.assertEqual(
            required_tracking_scenario_names(allowance),
            required_tracking_scenario_names(FINAL_PROFILE),
        )
        self.assertEqual(_scenarios(allowance), _scenarios(FINAL_PROFILE))
        self.assertEqual(
            required_tracking_check_names(allowance),
            required_tracking_check_names(FINAL_PROFILE),
        )
        self.assertEqual(
            required_target_column_ablation_targets(allowance),
            required_target_column_ablation_targets(FINAL_PROFILE),
        )
        self.assertTrue(tracking_profile_uses_perturbation(allowance))
        record = tracking_profile_completion_allowance(allowance)
        assert record is not None
        self.assertEqual(record["structure_profile"], FINAL_PROFILE)
        self.assertEqual(record["relaxes_profile"], FINAL_DEPLOYED_ACCURACY_PROFILE)
        self.assertEqual(record["recipe_revisions"], [pose_release])
        self.assertIn("near-fall", record["reason"])
        for profile in TRACKING_PROFILES:
            if profile not in (
                allowance,
                HMD_HAND_HAND_RMS_40MM_PROFILE,
                FOOT_CANARY_HAND_RMS_40MM_PROFILE,
            ):
                with self.subTest(profile=profile):
                    self.assertIsNone(tracking_profile_completion_allowance(profile))

    def test_hand_rms_40mm_is_the_pose_release_10000_profile_only(self) -> None:
        pose_release = MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION
        allowance = HMD_HAND_HAND_RMS_40MM_PROFILE
        self.assertEqual(
            allowance,
            "hmd_hand_reachable_performance_foot_exposure_v2_deployed_accuracy_v1_"
            "hand_rms_40mm_v1",
        )
        self.assertIn(allowance, TRACKING_PROFILES)
        self.assertEqual(
            required_tracking_profile(10_000, recipe_revision=pose_release),
            allowance,
        )
        self.assertEqual(
            accepted_tracking_profiles(10_000, recipe_revision=pose_release),
            (allowance, HMD_HAND_DEPLOYED_ACCURACY_PROFILE, HMD_HAND_PROFILE),
        )
        for recipe in (None, MICROBAN_TELEOP_V12_RECIPE_REVISION, "other"):
            with self.subTest(recipe=recipe):
                self.assertEqual(
                    required_tracking_profile(10_000, recipe_revision=recipe),
                    HMD_HAND_DEPLOYED_ACCURACY_PROFILE,
                )
        # Only the exact 10000 boundary (and its 10100 canary, below) move:
        # interrupted 7101..9999 clocks and the 15000 final keep their profiles.
        for completed in (7_101, 9_000, 9_999):
            with self.subTest(completed=completed):
                self.assertEqual(
                    required_tracking_profile(completed, recipe_revision=pose_release),
                    HMD_HAND_DEPLOYED_ACCURACY_PROFILE,
                )
        self.assertEqual(
            required_tracking_profile(10_100, recipe_revision=pose_release),
            FOOT_CANARY_HAND_RMS_40MM_PROFILE,
        )
        self.assertEqual(
            required_tracking_profile(15_000, recipe_revision=pose_release),
            FINAL_COMPLETION_ALLOWANCE_PROFILE,
        )
        # Identical to the HMD/hand deployed-accuracy profile except hand RMS.
        relaxed = HMD_HAND_DEPLOYED_ACCURACY_PROFILE
        self.assertEqual(hand_tracking_rms_max_m(allowance), 0.040)
        self.assertEqual(hand_tracking_rms_max_m(relaxed), 0.035)
        for limit in (
            hand_tracking_p95_max_m,
            foot_tracking_rms_max_m,
            foot_tracking_p95_max_m,
        ):
            with self.subTest(limit=limit.__name__):
                self.assertEqual(limit(allowance), limit(relaxed))
        self.assertEqual(
            (
                hand_tracking_p95_max_m(allowance),
                foot_tracking_rms_max_m(allowance),
                foot_tracking_p95_max_m(allowance),
            ),
            (0.05, 0.05, 0.08),
        )
        self.assertNotIn(allowance, STRICT_PROFILE_BY_DEPLOYED_ACCURACY_PROFILE)
        self.assertNotIn(allowance, STRICT_PROFILE_BY_COMPLETION_ALLOWANCE_PROFILE)
        self.assertEqual(
            required_tracking_scenario_names(allowance),
            required_tracking_scenario_names(relaxed),
        )
        self.assertEqual(_scenarios(allowance), _scenarios(relaxed))
        self.assertEqual(
            required_tracking_check_names(allowance),
            required_tracking_check_names(relaxed),
        )
        self.assertEqual(
            required_target_column_ablation_targets(allowance),
            required_target_column_ablation_targets(relaxed),
        )
        self.assertEqual(
            tracking_profile_uses_perturbation(allowance),
            tracking_profile_uses_perturbation(relaxed),
        )
        record = tracking_profile_completion_allowance(allowance)
        self.assertEqual(
            record,
            {
                "revision": "hand_rms_40mm_v1",
                "structure_profile": HMD_HAND_PROFILE,
                "relaxes_profile": relaxed,
                "boundary_completed_updates": 10_000,
                "recipe_revisions": [pose_release],
                "hand_rms_m_max": 0.040,
                "hand_p95_m_max": 0.05,
                "foot_rms_m_max": 0.05,
                "foot_p95_m_max": 0.08,
                "relaxed_profile_hand_rms_m_max": 0.035,
                "reason": record["reason"] if record else None,
            },
        )
        assert record is not None
        self.assertIn("0.040 m", record["reason"])
        # The 15000 completion allowance record is unchanged.
        final_record = tracking_profile_completion_allowance(
            FINAL_COMPLETION_ALLOWANCE_PROFILE
        )
        assert final_record is not None
        self.assertEqual(final_record["revision"], "completion_allowance_v1")
        self.assertNotIn("boundary_completed_updates", final_record)

    def test_hand_rms_40mm_is_the_pose_release_10100_canary_profile_only(
        self,
    ) -> None:
        pose_release = MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION
        allowance = FOOT_CANARY_HAND_RMS_40MM_PROFILE
        relaxed = FOOT_ACTIVATION_CANARY_DEPLOYED_ACCURACY_PROFILE
        self.assertEqual(
            allowance,
            "whole_body_foot_activation_canary_reachable_safety_v1_"
            "deployed_accuracy_v1_hand_rms_40mm_v1",
        )
        self.assertIn(allowance, TRACKING_PROFILES)
        self.assertEqual(
            required_tracking_profile(10_100, recipe_revision=pose_release),
            allowance,
        )
        self.assertEqual(
            accepted_tracking_profiles(10_100, recipe_revision=pose_release),
            (allowance, relaxed, FOOT_ACTIVATION_CANARY_PROFILE),
        )
        for recipe in (None, MICROBAN_TELEOP_V12_RECIPE_REVISION, "other"):
            with self.subTest(recipe=recipe):
                self.assertEqual(
                    required_tracking_profile(10_100, recipe_revision=recipe),
                    relaxed,
                )
        # Interrupted 10001..10099 clocks keep the deployed-accuracy canary.
        for completed in (10_001, 10_050, 10_099):
            with self.subTest(completed=completed):
                self.assertEqual(
                    required_tracking_profile(completed, recipe_revision=pose_release),
                    relaxed,
                )
        self.assertEqual(hand_tracking_rms_max_m(allowance), 0.040)
        self.assertEqual(hand_tracking_rms_max_m(relaxed), 0.035)
        for limit in (
            hand_tracking_p95_max_m,
            foot_tracking_rms_max_m,
            foot_tracking_p95_max_m,
        ):
            with self.subTest(limit=limit.__name__):
                self.assertEqual(limit(allowance), limit(relaxed))
        self.assertEqual(
            required_tracking_scenario_names(allowance),
            required_tracking_scenario_names(relaxed),
        )
        self.assertEqual(_scenarios(allowance), _scenarios(relaxed))
        self.assertEqual(
            required_tracking_check_names(allowance),
            required_tracking_check_names(relaxed),
        )
        self.assertEqual(
            required_target_column_ablation_targets(allowance),
            required_target_column_ablation_targets(relaxed),
        )
        self.assertEqual(
            tracking_profile_uses_perturbation(allowance),
            tracking_profile_uses_perturbation(relaxed),
        )
        record = tracking_profile_completion_allowance(allowance)
        assert record is not None
        self.assertEqual(record["revision"], "hand_rms_40mm_v1")
        self.assertEqual(record["structure_profile"], FOOT_ACTIVATION_CANARY_PROFILE)
        self.assertEqual(record["relaxes_profile"], relaxed)
        self.assertEqual(record["boundary_completed_updates"], 10_100)
        self.assertEqual(record["recipe_revisions"], [pose_release])
        self.assertEqual(record["hand_rms_m_max"], 0.040)
        self.assertEqual(record["relaxed_profile_hand_rms_m_max"], 0.035)
        self.assertIn("10100-canary", record["reason"])

    def test_pose_release_10000_gate_uses_and_records_the_hand_rms_allowance(
        self,
    ) -> None:
        base_infos = {
            "microban_teleop_training_contract_version": "12",
            "microban_teleop_recipe_revision": (
                MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION
            ),
            BILATERAL_SITE_ORDER_INFO_KEY: MICROBAN_BILATERAL_SITE_ORDER_REVISION,
            TELEOP_V12_HOME_POSE_INFO_KEY: teleop_v12_home_pose_marker(),
            "adapter_gradient_schedule_revision": (
                TELEOP_V12_ADAPTER_GRADIENT_SCHEDULE_REVISION
            ),
            "active_actor_columns_at_save": list(
                teleop_v12_active_adapter_columns(10_000 * 24)
            ),
            "env_state": {"common_step_counter": 10_000 * 24},
        }
        rescue_marker = corner_rescue_marker(
            parent_checkpoint_sha256="a" * 64,
            parent_strict_tracking_report_sha256="b" * 64,
            hand_pose_release=True,
            parent_strict_failed_checks=("hand_tracking_rms",),
        )
        allowance = HMD_HAND_HAND_RMS_40MM_PROFILE
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            onnx_path = root / "policy.onnx"
            onnx_path.write_bytes(b"unit-test-onnx")

            def evidence(name: str, infos: dict[str, object]) -> tuple:
                checkpoint = root / name / "model_9999.pt"
                checkpoint.parent.mkdir()
                torch.save({"iter": 9_999, "infos": infos}, checkpoint)
                identity = {
                    "sha256": sha256_file(checkpoint),
                    "iteration": 9_999,
                    "completed_updates": 10_000,
                }
                locomotion = root / name / "locomotion.json"
                locomotion.write_text(json.dumps(_locomotion_report(identity)))
                onnx_report = root / name / "onnx.json"
                onnx_report.write_text(
                    json.dumps(_onnx_report(identity, onnx_path))
                )
                return checkpoint, identity, locomotion, onnx_report

            for lineage, extra in (
                ("fresh", {}),
                (
                    "corner_rescue",
                    {MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY: rescue_marker},
                ),
            ):
                with self.subTest(lineage=lineage):
                    checkpoint, identity, locomotion, onnx_report = evidence(
                        lineage, {**base_infos, **extra}
                    )
                    # 0.0365 m hand RMS: over the deployed-accuracy 0.035 m
                    # limit, inside the 0.040 m allowance.
                    allowed = _tracking_report(
                        identity, profile=allowance, hand_rms=0.0365
                    )
                    self.assertEqual(allowed["status"], "pass")
                    self.assertEqual(allowed["thresholds"]["hand_rms_m_max"], 0.040)
                    self.assertEqual(
                        _tracking_report(
                            identity,
                            profile=HMD_HAND_DEPLOYED_ACCURACY_PROFILE,
                            hand_rms=0.0365,
                        )["status"],
                        "fail",
                    )
                    over = _tracking_report(identity, profile=allowance, hand_rms=0.0401)
                    self.assertEqual(over["status"], "fail")
                    tracking = root / lineage / "tracking.json"
                    tracking.write_text(json.dumps(allowed))
                    gate = create_gate(
                        checkpoint=checkpoint,
                        locomotion_report=locomotion,
                        tracking_report=tracking,
                        onnx_report=onnx_report,
                    )
                    self.assertEqual(gate["tracking_profile"], allowance)
                    self.assertEqual(gate["checkpoint_kind"], "canonical_boundary")
                    self.assertEqual(
                        gate["tracking_profile_completion_allowance"],
                        tracking_profile_completion_allowance(allowance),
                    )
                    self.assertEqual(
                        gate.get(MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY),
                        extra.get(MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY),
                    )
                    gate_path = root / lineage / "gate.json"
                    gate_path.write_text(json.dumps(gate))
                    self.assertEqual(validate_gate(gate_path, checkpoint), gate)
                    for field in (
                        "tracking_profile_completion_allowance",
                        "tracking_profile",
                    ):
                        tampered = deepcopy(gate)
                        if field == "tracking_profile":
                            tampered[field] = HMD_HAND_DEPLOYED_ACCURACY_PROFILE
                        else:
                            del tampered[field]
                        gate_path.write_text(json.dumps(tampered))
                        with self.assertRaises(ValueError):
                            validate_gate(gate_path, checkpoint)
                    # A stricter deployed-accuracy report is still accepted.
                    strict_tracking = root / lineage / "tracking_deployed.json"
                    strict_tracking.write_text(
                        json.dumps(
                            _tracking_report(
                                identity, profile=HMD_HAND_DEPLOYED_ACCURACY_PROFILE
                            )
                        )
                    )
                    strict_gate = create_gate(
                        checkpoint=checkpoint,
                        locomotion_report=locomotion,
                        tracking_report=strict_tracking,
                        onnx_report=onnx_report,
                    )
                    self.assertEqual(
                        strict_gate["tracking_profile"],
                        HMD_HAND_DEPLOYED_ACCURACY_PROFILE,
                    )
                    self.assertNotIn(
                        "tracking_profile_completion_allowance", strict_gate
                    )
                    over_tracking = root / lineage / "tracking_over.json"
                    over_tracking.write_text(json.dumps(over))
                    with self.assertRaises(ValueError):
                        create_gate(
                            checkpoint=checkpoint,
                            locomotion_report=locomotion,
                            tracking_report=over_tracking,
                            onnx_report=onnx_report,
                        )

            # The canonical v11 lineage cannot use the allowance at 10000.
            v11_checkpoint, v11_identity, v11_locomotion, v11_onnx = evidence(
                "canonical",
                {
                    **base_infos,
                    "microban_teleop_recipe_revision": (
                        MICROBAN_TELEOP_V12_RECIPE_REVISION
                    ),
                },
            )
            v11_tracking = root / "canonical" / "tracking.json"
            v11_tracking.write_text(
                json.dumps(
                    _tracking_report(v11_identity, profile=allowance, hand_rms=0.0365)
                )
            )
            with self.assertRaises(ValueError):
                create_gate(
                    checkpoint=v11_checkpoint,
                    locomotion_report=v11_locomotion,
                    tracking_report=v11_tracking,
                    onnx_report=v11_onnx,
                )

    def test_pose_release_final_gate_uses_and_records_the_allowance(self) -> None:
        base_infos = {
            "microban_teleop_training_contract_version": "12",
            "microban_teleop_recipe_revision": (
                MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION
            ),
            BILATERAL_SITE_ORDER_INFO_KEY: MICROBAN_BILATERAL_SITE_ORDER_REVISION,
            TELEOP_V12_HOME_POSE_INFO_KEY: teleop_v12_home_pose_marker(),
            "adapter_gradient_schedule_revision": (
                TELEOP_V12_ADAPTER_GRADIENT_SCHEDULE_REVISION
            ),
            "active_actor_columns_at_save": list(
                teleop_v12_active_adapter_columns(15_000 * 24)
            ),
            "env_state": {"common_step_counter": 15_000 * 24},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            onnx_path = root / "policy.onnx"
            onnx_path.write_bytes(b"unit-test-onnx")

            def evidence(name: str, infos: dict[str, object]) -> tuple:
                checkpoint = root / name / "model_14999.pt"
                checkpoint.parent.mkdir()
                torch.save({"iter": 14_999, "infos": infos}, checkpoint)
                identity = {
                    "sha256": sha256_file(checkpoint),
                    "iteration": 14_999,
                    "completed_updates": 15_000,
                }
                locomotion = root / name / "locomotion.json"
                locomotion.write_text(json.dumps(_locomotion_report(identity)))
                onnx_report = root / name / "onnx.json"
                onnx_report.write_text(
                    json.dumps(_onnx_report(identity, onnx_path))
                )
                return checkpoint, identity, locomotion, onnx_report

            checkpoint, identity, locomotion, onnx_report = evidence(
                "pose_release", base_infos
            )
            # 0.040 m hand RMS: over the deployed-accuracy 0.035 m limit, inside
            # the 0.045 m allowance.
            allowed = _tracking_report(
                identity, profile=FINAL_COMPLETION_ALLOWANCE_PROFILE, hand_rms=0.040
            )
            self.assertEqual(allowed["status"], "pass")
            self.assertEqual(
                _tracking_report(
                    identity, profile=FINAL_DEPLOYED_ACCURACY_PROFILE, hand_rms=0.040
                )["status"],
                "fail",
            )
            over = _tracking_report(
                identity, profile=FINAL_COMPLETION_ALLOWANCE_PROFILE, hand_rms=0.0451
            )
            self.assertEqual(over["status"], "fail")
            tracking = root / "pose_release" / "tracking.json"
            tracking.write_text(json.dumps(allowed))
            gate = create_gate(
                checkpoint=checkpoint,
                locomotion_report=locomotion,
                tracking_report=tracking,
                onnx_report=onnx_report,
            )
            self.assertEqual(gate["tracking_profile"], FINAL_COMPLETION_ALLOWANCE_PROFILE)
            self.assertEqual(gate["checkpoint_kind"], "canonical_boundary")
            self.assertEqual(
                gate["tracking_profile_completion_allowance"],
                tracking_profile_completion_allowance(
                    FINAL_COMPLETION_ALLOWANCE_PROFILE
                ),
            )
            gate_path = root / "pose_release" / "gate.json"
            gate_path.write_text(json.dumps(gate))
            self.assertEqual(validate_gate(gate_path, checkpoint), gate)
            for field in ("tracking_profile_completion_allowance", "tracking_profile"):
                with self.subTest(tampered=field):
                    tampered = deepcopy(gate)
                    if field == "tracking_profile":
                        tampered[field] = FINAL_DEPLOYED_ACCURACY_PROFILE
                    else:
                        del tampered[field]
                    gate_path.write_text(json.dumps(tampered))
                    with self.assertRaises(ValueError):
                        validate_gate(gate_path, checkpoint)
            # A stricter deployed-accuracy report is still accepted (no record).
            strict_tracking = root / "pose_release" / "tracking_deployed.json"
            strict_tracking.write_text(
                json.dumps(
                    _tracking_report(identity, profile=FINAL_DEPLOYED_ACCURACY_PROFILE)
                )
            )
            strict_gate = create_gate(
                checkpoint=checkpoint,
                locomotion_report=locomotion,
                tracking_report=strict_tracking,
                onnx_report=onnx_report,
            )
            self.assertEqual(
                strict_gate["tracking_profile"], FINAL_DEPLOYED_ACCURACY_PROFILE
            )
            self.assertNotIn("tracking_profile_completion_allowance", strict_gate)
            over_tracking = root / "pose_release" / "tracking_over.json"
            over_tracking.write_text(json.dumps(over))
            with self.assertRaises(ValueError):
                create_gate(
                    checkpoint=checkpoint,
                    locomotion_report=locomotion,
                    tracking_report=over_tracking,
                    onnx_report=onnx_report,
                )

            # The canonical v11 lineage cannot use the allowance at 15000.
            v11_checkpoint, v11_identity, v11_locomotion, v11_onnx = evidence(
                "canonical",
                {
                    **base_infos,
                    "microban_teleop_recipe_revision": (
                        MICROBAN_TELEOP_V12_RECIPE_REVISION
                    ),
                },
            )
            v11_tracking = root / "canonical" / "tracking.json"
            v11_tracking.write_text(
                json.dumps(
                    _tracking_report(
                        v11_identity,
                        profile=FINAL_COMPLETION_ALLOWANCE_PROFILE,
                        hand_rms=0.040,
                    )
                )
            )
            with self.assertRaises(ValueError):
                create_gate(
                    checkpoint=v11_checkpoint,
                    locomotion_report=v11_locomotion,
                    tracking_report=v11_tracking,
                    onnx_report=v11_onnx,
                )

    def test_pose_release_lineages_gate_under_the_unchanged_profiles(
        self,
    ) -> None:
        # The forward-lean HOME pins no switch parent; the switch mechanism is
        # exercised under a test-only pin.
        self.assertIsNone(HAND_POSE_RELEASE_RECIPE_SWITCH_PARENT_CHECKPOINT_SHA256)
        pin = mock.patch(
            "mjlab_microban.tasks.microban_teleop_v12_hand_pose_release_lineage."
            "HAND_POSE_RELEASE_RECIPE_SWITCH_PARENT_CHECKPOINT_SHA256",
            "d" * 64,
        )
        pin.start()
        self.addCleanup(pin.stop)
        marker = hand_pose_release_recipe_switch_marker(
            parent_checkpoint_path="repo://logs/run_7000_to7100/model_7099.pt",
            parent_checkpoint_sha256="d" * 64,
            parent_stage_gate_path="repo://artifacts/run_model_7099_gate.json",
            parent_stage_gate_sha256="b" * 64,
        )
        base_infos = {
            "microban_teleop_training_contract_version": "12",
            "microban_teleop_recipe_revision": (
                MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION
            ),
            BILATERAL_SITE_ORDER_INFO_KEY: MICROBAN_BILATERAL_SITE_ORDER_REVISION,
            TELEOP_V12_HOME_POSE_INFO_KEY: teleop_v12_home_pose_marker(),
            "adapter_gradient_schedule_revision": (
                TELEOP_V12_ADAPTER_GRADIENT_SCHEDULE_REVISION
            ),
            "active_actor_columns_at_save": list(
                teleop_v12_active_adapter_columns(10_000 * 24)
            ),
            "env_state": {"common_step_counter": 10_000 * 24},
        }
        parent_infos = {
            **base_infos,
            "microban_teleop_recipe_revision": MICROBAN_TELEOP_V12_RECIPE_REVISION,
        }
        verify = mock.patch(
            "mjlab_microban.tasks.microban_teleop_v12_hand_pose_release_lineage."
            "verify_hand_pose_release_switch_parent",
            return_value=parent_infos,
        )
        with tempfile.TemporaryDirectory() as directory, verify as verified:
            root = Path(directory)
            onnx_path = root / "policy.onnx"
            onnx_path.write_bytes(b"unit-test-onnx")
            for lineage, extra in (
                ("fresh", {}),
                ("switch", {MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_SWITCH_INFO_KEY: marker}),
            ):
                with self.subTest(lineage=lineage):
                    checkpoint = root / lineage / "model_9999.pt"
                    checkpoint.parent.mkdir()
                    torch.save(
                        {"iter": 9_999, "infos": {**base_infos, **extra}}, checkpoint
                    )
                    identity = {
                        "sha256": sha256_file(checkpoint),
                        "iteration": 9_999,
                        "completed_updates": 10_000,
                    }
                    reports = {
                        "locomotion_report": root / lineage / "locomotion.json",
                        "tracking_report": root / lineage / "tracking.json",
                        "onnx_report": root / lineage / "onnx.json",
                    }
                    reports["locomotion_report"].write_text(
                        json.dumps(_locomotion_report(identity))
                    )
                    reports["tracking_report"].write_text(
                        json.dumps(_tracking_report(identity))
                    )
                    reports["onnx_report"].write_text(
                        json.dumps(_onnx_report(identity, onnx_path))
                    )
                    gate = create_gate(checkpoint=checkpoint, **reports)
                    self.assertEqual(
                        gate["tracking_profile"], required_tracking_profile(10_000)
                    )
                    self.assertEqual(
                        gate.get(
                            MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_SWITCH_INFO_KEY
                        ),
                        extra.get(
                            MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_SWITCH_INFO_KEY
                        ),
                    )
                    gate_path = root / lineage / "gate.json"
                    gate_path.write_text(json.dumps(gate))
                    self.assertEqual(validate_gate(gate_path, checkpoint), gate)
                    if extra:
                        tampered = deepcopy(gate)
                        del tampered[
                            MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_SWITCH_INFO_KEY
                        ]
                        gate_path.write_text(json.dumps(tampered))
                        with self.assertRaises(ValueError):
                            validate_gate(gate_path, checkpoint)
            self.assertTrue(verified.called)

            # The experimental switch is evidence-only: no gate.
            experimental = root / "experimental" / "model_9999.pt"
            experimental.parent.mkdir()
            torch.save(
                {
                    "iter": 9_999,
                    "infos": {
                        **base_infos,
                        MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_EXPERIMENTAL_SWITCH_INFO_KEY: {
                            "schema_version": 1,
                            "release_eligible": False,
                            "parent_recipe_revision": MICROBAN_TELEOP_V12_RECIPE_REVISION,
                            "recipe_revision": (
                                MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION
                            ),
                            "parent_checkpoint_sha256": "0" * 64,
                            "parent_iteration": 7_099,
                        },
                    },
                },
                experimental,
            )
            with self.assertRaisesRegex(ValueError, "not release-eligible"):
                create_gate(
                    checkpoint=experimental,
                    locomotion_report=root / "fresh" / "locomotion.json",
                    tracking_report=root / "fresh" / "tracking.json",
                    onnx_report=root / "fresh" / "onnx.json",
                )
            # The switch marker cannot appear before its 7099 parent.
            early = root / "early" / "model_7099.pt"
            early.parent.mkdir()
            torch.save(
                {
                    "iter": 7_099,
                    "infos": {
                        **base_infos,
                        "active_actor_columns_at_save": list(
                            teleop_v12_active_adapter_columns(7_100 * 24)
                        ),
                        "env_state": {"common_step_counter": 7_100 * 24},
                        MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_SWITCH_INFO_KEY: marker,
                    },
                },
                early,
            )
            with self.assertRaisesRegex(ValueError, "descendant clock"):
                checkpoint_recipe_kind(early)

    def test_deadline_final_tracking_report_requires_perturbation(self) -> None:
        identity = {
            "sha256": "a" * 64,
            "iteration": 14_999,
            "completed_updates": 15_000,
        }
        report = _tracking_report(
            identity, profile=DEADLINE_FINAL_FALLBACK_PROFILE
        )
        self.assertTrue(report["settings"]["perturbation"])
        self.assertEqual(
            report["thresholds"]["hand_rms_m_max"],
            hand_tracking_rms_max_m(DEADLINE_FINAL_FALLBACK_PROFILE),
        )
        _validate_tracking_report(
            report,
            identity,
            profile_override=DEADLINE_FINAL_FALLBACK_PROFILE,
        )

        without_perturbation = deepcopy(report)
        without_perturbation["settings"]["perturbation"] = False
        with self.assertRaisesRegex(ValueError, "settings are not canonical"):
            _validate_tracking_report(
                without_perturbation,
                identity,
                profile_override=DEADLINE_FINAL_FALLBACK_PROFILE,
            )

    def test_activation_canaries_require_causality_before_final_quality(self) -> None:
        hand_canary_checks = required_tracking_check_names(
            HMD_HAND_ACTIVATION_CANARY_PROFILE
        )
        self.assertNotIn("hand_tracking_rms", hand_canary_checks)
        self.assertNotIn("hand_tracking_p95", hand_canary_checks)
        self.assertEqual(
            required_tracking_scenario_names(HMD_HAND_ACTIVATION_CANARY_PROFILE),
            ("low_forward", "max_hands_left", "max_hands_right"),
        )
        self.assertFalse(
            any(
                any(
                    abs(value) > 0.0
                    for target in scenario.foot_target
                    for value in target
                )
                for scenario in _scenarios(HMD_HAND_ACTIVATION_CANARY_PROFILE)
            )
        )

        foot_canary_checks = required_tracking_check_names(
            FOOT_ACTIVATION_CANARY_PROFILE
        )
        self.assertIn("hand_tracking_rms", foot_canary_checks)
        self.assertIn("hand_tracking_p95", foot_canary_checks)
        self.assertNotIn("foot_tracking_rms", foot_canary_checks)
        self.assertNotIn("foot_tracking_p95", foot_canary_checks)
        self.assertTrue(
            any(
                any(
                    abs(value) > 0.0
                    for target in scenario.foot_target
                    for value in target
                )
                for scenario in _scenarios(FOOT_ACTIVATION_CANARY_PROFILE)
            )
        )

    def test_tracking_acceptance_rejects_done_coverage_direction_and_active_error(
        self,
    ) -> None:
        checks, status = _acceptance([_result()], WHOLE_BODY_PROFILE)
        self.assertEqual(status, "pass")
        self.assertTrue(all(checks.values()))
        for mutation in (
            {"completed": False, "termination_names": ["out_of_terrain_bounds"]},
            {"observation_coverage": {"passed": False}},
            {"twist_directional_response_passed": False},
            {
                "maximum_actual_soft_limit_violation_rad": (
                    ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD + 1.0e-9
                )
            },
        ):
            with self.subTest(mutation=mutation):
                failed, failed_status = _acceptance(
                    [_result(**mutation)], WHOLE_BODY_PROFILE
                )
                self.assertEqual(failed_status, "fail")
                self.assertFalse(all(failed.values()))

        allowed, allowed_status = _acceptance(
            [
                _result(
                    maximum_actual_soft_limit_violation_rad=(
                        ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD
                    )
                )
            ],
            WHOLE_BODY_PROFILE,
        )
        self.assertEqual(allowed_status, "pass")
        self.assertTrue(allowed["actual_soft_limits"])

    def test_target_column_ablation_acceptance_follows_activation_schedule(
        self,
    ) -> None:
        self.assertEqual(
            target_column_ablation_observation_columns("hand"),
            (tuple(range(75, 81)), tuple(range(81, 83))),
        )
        self.assertEqual(
            target_column_ablation_observation_columns("foot"),
            (tuple(range(69, 75)), ()),
        )
        with self.assertRaisesRegex(ValueError, "Unknown"):
            target_column_ablation_observation_columns("head")
        observation = torch.arange(83, dtype=torch.float32).unsqueeze(0)
        hand_ablated = target_column_ablated_observation(observation, "hand")
        torch.testing.assert_close(
            hand_ablated[:, 75:81], torch.zeros((1, 6)), rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            hand_ablated[:, 81:83], observation[:, 81:83], rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            hand_ablated[:, :75], observation[:, :75], rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            observation,
            torch.arange(83, dtype=torch.float32).unsqueeze(0),
            rtol=0.0,
            atol=0.0,
        )
        with self.assertRaisesRegex(ValueError, r"\[batch, 83\]"):
            target_column_ablated_observation(torch.zeros((1, 82)), "hand")
        self.assertEqual(
            required_target_column_ablation_targets(PRE_ACTIVATION_EXPOSURE_PROFILE),
            frozenset(),
        )
        self.assertEqual(
            required_target_column_ablation_targets(EXPANDED_LOCOMOTION_PROFILE),
            frozenset(),
        )
        self.assertEqual(
            required_target_column_ablation_targets(HMD_HAND_ACTIVATION_CANARY_PROFILE),
            frozenset(("hand",)),
        )
        self.assertEqual(
            required_target_column_ablation_targets(HMD_HAND_PROFILE),
            frozenset(("hand",)),
        )
        self.assertEqual(
            required_target_column_ablation_targets(FOOT_ACTIVATION_CANARY_PROFILE),
            frozenset(("hand", "foot")),
        )
        self.assertEqual(
            required_target_column_ablation_targets(WHOLE_BODY_PROFILE),
            frozenset(("hand", "foot")),
        )

        unresponsive = _result(
            target_column_ablation={
                "hand": _ablation(
                    target="hand",
                    expected=True,
                    maximum=TARGET_COLUMN_ABLATION_ACTION_DELTA_MIN,
                ),
                "foot": _ablation(
                    target="foot",
                    expected=True,
                    maximum=TARGET_COLUMN_ABLATION_ACTION_DELTA_MIN,
                ),
            }
        )
        for profile in (
            PRE_ACTIVATION_EXPOSURE_PROFILE,
            EXPANDED_LOCOMOTION_PROFILE,
        ):
            with self.subTest(profile=profile):
                checks, status = _acceptance([unresponsive], profile)
                self.assertEqual(status, "pass")
                self.assertTrue(checks["target_column_ablation_response"])

        hand_only = deepcopy(unresponsive)
        hand_only["target_column_ablation"]["hand"] = _ablation(
            target="hand", expected=True
        )
        for profile in (HMD_HAND_ACTIVATION_CANARY_PROFILE, HMD_HAND_PROFILE):
            checks, status = _acceptance([hand_only], profile)
            self.assertEqual(status, "pass")
            self.assertTrue(checks["target_column_ablation_response"])
        for profile in (
            FOOT_ACTIVATION_CANARY_PROFILE,
            WHOLE_BODY_PROFILE,
            FINAL_PROFILE,
        ):
            with self.subTest(profile=profile):
                checks, status = _acceptance([hand_only], profile)
                self.assertEqual(status, "fail")
                self.assertFalse(checks["target_column_ablation_response"])

    def test_tracking_ablation_evidence_is_exact_and_fail_closed(self) -> None:
        identity = {
            "sha256": "a" * 64,
            "iteration": 10_000,
            "completed_updates": 10_001,
        }
        report = _tracking_report(identity)
        self.assertEqual(
            set(report["checks"]),
            set(required_tracking_check_names(FOOT_ACTIVATION_CANARY_PROFILE)),
        )
        self.assertIn("hand_tracking_rms", report["checks"])
        self.assertNotIn("foot_tracking_rms", report["checks"])
        _validate_tracking_report(report, identity)
        active_index = next(
            index
            for index, result in enumerate(report["results"])
            if result["target_column_ablation"]["hand"]["target_expected"]
        )

        def active_hand(candidate: dict[str, object]) -> dict[str, object]:
            return candidate["results"][active_index]["target_column_ablation"]["hand"]

        mutations = (
            lambda candidate: active_hand(candidate).update(target_expected=False),
            lambda candidate: active_hand(candidate).update(
                ablated_observation_columns=list(range(75, 83))
            ),
            lambda candidate: active_hand(candidate).update(
                preserved_observation_columns=[]
            ),
            lambda candidate: active_hand(candidate).update(
                maximum_absolute_action_delta=-1.0
            ),
            lambda candidate: active_hand(candidate).update(
                maximum_absolute_action_delta=float("nan")
            ),
            lambda candidate: active_hand(candidate).update(
                minimum_required_action_delta=0.0
            ),
            lambda candidate: active_hand(candidate).update(passed=False),
            lambda candidate: active_hand(candidate).update(untrusted=True),
            lambda candidate: active_hand(candidate).pop(
                "maximum_absolute_action_delta"
            ),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                corrupted = deepcopy(report)
                mutate(corrupted)
                with self.assertRaises(ValueError):
                    _validate_tracking_report(corrupted, identity)

        below_floor = deepcopy(report)
        active_hand(below_floor).update(
            maximum_absolute_action_delta=TARGET_COLUMN_ABLATION_ACTION_DELTA_MIN,
            passed=False,
        )
        with self.assertRaisesRegex(ValueError, "checks do not match"):
            _validate_tracking_report(below_floor, identity)

    def test_active_foot_metric_excludes_inactive_foot(self) -> None:
        default = torch.zeros(1, 2, 3)
        target = torch.tensor([[[0.0, 0.0, 0.02], [0.0, 0.0, 0.0]]])
        current = torch.tensor([[[0.0, 0.0, 0.03], [9.0, 9.0, 9.0]]])
        error = _active_foot_tracking_error(current, default, target)
        self.assertEqual(tuple(error.shape), (1,))
        torch.testing.assert_close(error, torch.tensor([0.01]))

    def test_schema2_gate_accepts_hash_bound_sanitized_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model_600.pt"
            infos = {
                "microban_teleop_training_contract_version": "12",
                "microban_teleop_recipe_revision": MICROBAN_TELEOP_V12_RECIPE_REVISION,
                BILATERAL_SITE_ORDER_INFO_KEY: MICROBAN_BILATERAL_SITE_ORDER_REVISION,
                TELEOP_V12_HOME_POSE_INFO_KEY: teleop_v12_home_pose_marker(),
                "adapter_gradient_schedule_revision": (
                    TELEOP_V12_ADAPTER_GRADIENT_SCHEDULE_REVISION
                ),
                "active_actor_columns_at_save": [],
                "env_state": {"common_step_counter": 601 * 24},
                "adapter_sanitization": {
                    "schema_version": (TELEOP_V12_ADAPTER_SANITIZATION_SCHEMA_VERSION),
                    "revision": TELEOP_V12_ADAPTER_SANITIZATION_REVISION,
                    "parent_checkpoint_sha256": "a" * 64,
                    "parent_iteration": 600,
                    "completed_updates": 601,
                    "zeroed_actor_columns": list(TELEOP_V12_EXTRA_OBSERVATION_COLUMNS),
                    **teleop_v12_target_normalizer_metadata(),
                },
            }
            torch.save({"iter": 600, "infos": infos}, checkpoint)
            checkpoint_sha = sha256_file(checkpoint)
            identity = {
                "sha256": checkpoint_sha,
                "iteration": 600,
                "completed_updates": 601,
            }
            locomotion_path = root / "locomotion.json"
            tracking_path = root / "tracking.json"
            onnx_report_path = root / "onnx.json"
            onnx_path = root / "policy.onnx"
            onnx_path.write_bytes(b"unit-test-onnx")
            locomotion = _locomotion_report(identity)
            tracking = _tracking_report(identity)
            onnx = _onnx_report(identity, onnx_path)
            locomotion_path.write_text(json.dumps(locomotion))
            tracking_path.write_text(json.dumps(tracking))
            onnx_report_path.write_text(json.dumps(onnx))
            gate = create_gate(
                checkpoint=checkpoint,
                locomotion_report=locomotion_path,
                tracking_report=tracking_path,
                onnx_report=onnx_report_path,
            )
            self.assertEqual(gate["schema_version"], 2)
            self.assertEqual(gate["checkpoint_kind"], "sanitized_recovery")
            gate_path = root / "gate.json"
            gate_path.write_text(json.dumps(gate))
            self.assertEqual(validate_gate(gate_path, checkpoint), gate)
            self.assertEqual(
                _checkpoint_kind(602, infos["adapter_sanitization"]),
                "interrupted_recovery",
            )

            corruptions = (
                (locomotion_path, locomotion, ("checks",), {}),
                (tracking_path, tracking, ("results",), []),
                (
                    tracking_path,
                    tracking,
                    ("results", 0, "hmd_motion", "per_axis"),
                    {},
                ),
                (
                    tracking_path,
                    tracking,
                    ("results", 0, "directional_response"),
                    {},
                ),
                (
                    onnx_report_path,
                    onnx,
                    ("onnx", "onnxruntime_cpu_maximum_absolute_error"),
                    ONNX_PARITY_TOLERANCE * 2.0,
                ),
                (
                    onnx_report_path,
                    onnx,
                    ("neutral_legacy_parity", "maximum_absolute_error"),
                    -1.0,
                ),
                (
                    onnx_report_path,
                    onnx,
                    ("onnx", "reference_evaluator_maximum_absolute_error"),
                    -1.0,
                ),
                (
                    tracking_path,
                    tracking,
                    (
                        "results",
                        next(
                            index
                            for index, result in enumerate(tracking["results"])
                            if result["target_error"]["active_hand"]["sample_count"] > 0
                        ),
                        "target_error",
                        "active_hand",
                        "rms",
                    ),
                    -1.0,
                ),
            )
            for path, original, keys, replacement in corruptions:
                with self.subTest(keys=keys):
                    corrupted = deepcopy(original)
                    target = corrupted
                    for key in keys[:-1]:
                        target = target[key]
                    target[keys[-1]] = replacement
                    path.write_text(json.dumps(corrupted))
                    with self.assertRaises(ValueError):
                        create_gate(
                            checkpoint=checkpoint,
                            locomotion_report=locomotion_path,
                            tracking_report=tracking_path,
                            onnx_report=onnx_report_path,
                        )
                    locomotion_path.write_text(json.dumps(locomotion))
                    tracking_path.write_text(json.dumps(tracking))
                    onnx_report_path.write_text(json.dumps(onnx))

    def test_schema2_gate_binds_exact_final_corner_rescue_lineage(self) -> None:
        _CORNER_MARKER = corner_rescue_marker(
            parent_checkpoint_sha256="a" * 64,
            parent_strict_tracking_report_sha256="b" * 64,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model_9999.pt"
            stored_std = torch.tensor(
                TELEOP_V12_TARGET_POSITION_NORMALIZER_STORED_STD
            )
            mean = torch.zeros(1, 83)
            var = torch.ones(1, 83)
            std = torch.ones(1, 83)
            var[:, 69:81] = stored_std.square()
            std[:, 69:81] = stored_std
            weight = torch.ones(512, 83)
            weight[:, TELEOP_V12_FOOT_OBSERVATION_COLUMNS] = 0.0
            first_moment = torch.ones(512, 83)
            second_moment = torch.ones(512, 83)
            first_moment[:, TELEOP_V12_FOOT_OBSERVATION_COLUMNS] = 0.0
            second_moment[:, TELEOP_V12_FOOT_OBSERVATION_COLUMNS] = 0.0
            infos = {
                "microban_teleop_training_contract_version": "12",
                "microban_teleop_recipe_revision": (
                    MICROBAN_TELEOP_V12_CORNER_RESCUE_RECIPE_REVISION
                ),
                BILATERAL_SITE_ORDER_INFO_KEY: MICROBAN_BILATERAL_SITE_ORDER_REVISION,
                "adapter_gradient_schedule_revision": (
                    TELEOP_V12_ADAPTER_GRADIENT_SCHEDULE_REVISION
                ),
                "active_actor_columns_at_save": list(
                    MICROBAN_TELEOP_V12_CORNER_RESCUE_ACTIVE_COLUMNS
                ),
                "env_state": {"common_step_counter": 10_000 * 24},
                MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY: _CORNER_MARKER,
            }
            torch.save(
                {
                    "iter": 9_999,
                    "infos": infos,
                    "actor_state_dict": {
                        "obs_normalizer._mean": mean,
                        "obs_normalizer._var": var,
                        "obs_normalizer._std": std,
                        "mlp.0.weight": weight,
                    },
                    "optimizer_state_dict": {
                        "state": {
                            1: {
                                "step": torch.tensor(
                                    float(
                                        MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_OPTIMIZER_STEP
                                    )
                                ),
                                "exp_avg": first_moment,
                                "exp_avg_sq": second_moment,
                            },
                            2: {
                                "step": torch.tensor(
                                    float(
                                        MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_OPTIMIZER_STEP
                                    )
                                ),
                                "exp_avg": torch.ones(18),
                                "exp_avg_sq": torch.ones(18),
                            },
                        }
                    },
                },
                checkpoint,
            )
            identity = {
                "sha256": sha256_file(checkpoint),
                "iteration": 9_999,
                "completed_updates": 10_000,
            }
            locomotion_path = root / "locomotion.json"
            tracking_path = root / "tracking.json"
            onnx_report_path = root / "onnx.json"
            onnx_path = root / "policy.onnx"
            onnx_path.write_bytes(b"unit-test-onnx")
            locomotion_path.write_text(json.dumps(_locomotion_report(identity)))
            tracking_path.write_text(json.dumps(_tracking_report(identity)))
            onnx_report_path.write_text(
                json.dumps(_onnx_report(identity, onnx_path))
            )
            gate = create_gate(
                checkpoint=checkpoint,
                locomotion_report=locomotion_path,
                tracking_report=tracking_path,
                onnx_report=onnx_report_path,
            )
            self.assertEqual(
                gate[MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY],
                _CORNER_MARKER,
            )
            gate_path = root / "gate.json"
            gate_path.write_text(json.dumps(gate))
            self.assertEqual(validate_gate(gate_path, checkpoint), gate)

            tampered = deepcopy(gate)
            tampered[MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY][
                "sampler_probabilities"
            ]["left_forward_right_backward"] = 0.39
            gate_path.write_text(json.dumps(tampered))
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                validate_gate(gate_path, checkpoint)


if __name__ == "__main__":
    unittest.main()
