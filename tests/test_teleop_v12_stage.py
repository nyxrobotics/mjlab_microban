"""CPU-only regression tests of the PICO judgment's evidence and its gate file."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import torch

from mjlab_microban.robot.microban_hand_fk import (
    microban_hand_fk_metadata,
    microban_reachable_hand_evaluation_offsets,
)
from mjlab_microban.scripts.evaluate_teleop_v12_checkpoint import (
    LOCOMOTION_MOVING_SCENARIOS,
    locomotion_twist_judgments,
)
from mjlab_microban.scripts.evaluate_teleop_v12_checkpoint import (
    _acceptance as _locomotion_acceptance,
)
from mjlab_microban.scripts.evaluate_teleop_v12_tracking import (
    FINAL_PROFILE,
    HMD_ACTUAL_PEAK_TO_PEAK_MIN_RAD,
    HMD_TARGET_PEAK_TO_PEAK_MIN_RAD,
    TARGET_COLUMN_ABLATION_ACTION_DELTA_MIN,
    TARGET_COLUMN_ABLATION_METHOD,
    TRACKING_PROFILES,
    _acceptance,
    _active_foot_tracking_error,
    _aggregate_action_envelopes,
    _scenarios,
    foot_tracking_p95_max_m,
    foot_tracking_rms_max_m,
    hand_tracking_p95_max_m,
    hand_tracking_rms_max_m,
    required_target_column_ablation_targets,
    required_tracking_check_names,
    required_tracking_profile,
    target_column_ablated_observation,
    target_column_ablation_observation_columns,
    tracking_profile_uses_perturbation,
)
from mjlab_microban.twist_pass_line import twist_judgment, twist_pass_line_record
from mjlab_microban.scripts.teleop_v12_bootstrap_gate import (
    ONNX_PARITY_TOLERANCE,
    PRISTINE_PARITY_TOLERANCE,
)
from mjlab_microban.scripts.teleop_v12_stage import (
    _validate_tracking_report,
    checkpoint_recipe_kind,
    create_gate,
    validate_gate,
)
from mjlab_microban.schedules import (
    PICO_MIN_FINAL_UPDATES,
    PICO_SCHEDULE,
    PICO_TOTAL_UPDATES,
)
from mjlab_microban.tasks.mdp import MICROBAN_BILATERAL_SITE_ORDER_REVISION
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_HMD_JOINT_NAMES,
    MICROBAN_TELEOP_ACTION_JOINT_NAMES,
)
from mjlab_microban.tasks.microban_teleop_v12_actor import (
    TELEOP_V12_ADAPTER_GRADIENT_SCHEDULE_REVISION,
    teleop_v12_active_adapter_columns,
)
from mjlab_microban.tasks.microban_teleop_v12_bootstrap import sha256_file
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION,
)
from mjlab_microban.tasks.microban_teleop_v12_home_pose import (
    TELEOP_V12_HOME_POSE_INFO_KEY,
    teleop_v12_home_pose_marker,
)
from mjlab_microban.tasks.microban_teleop_v12_runner import (
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
        "twist_beats_standing_passed": True,
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
    for name in ("neutral", *LOCOMOTION_MOVING_SCENARIOS):
        twist = commands[name]
        measured = {axis: 0.0 for axis in axis_names}
        response = None
        if name != "neutral":
            # Half the command along it: the reward 0.75 beats standing (0.5).
            index = next(index for index, value in enumerate(twist) if value != 0.0)
            axis = axis_names[index]
            measured[axis] = 0.5 * twist[index]
            response = {
                "axis": axis,
                "command": twist[index],
                "measured_mean": measured[axis],
                "sign_matches": True,
                "signed_response": abs(measured[axis]),
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
            "twist_pass_line": twist_pass_line_record(),
        },
        "checks": checks,
        "twist_judgment": locomotion_twist_judgments(results),
        "results": results,
        "summary": {
            "scenario_count": 9,
            "completed_scenario_count": 9,
            "fall_scenario_count": 0,
            "nonfinite_scenario_count": 0,
            "directionally_correct_scenario_count": 8,
            "directional_scenario_count": 8,
            "minimum_signed_response": 0.025,
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
            # Half the command along it beats standing still.
            directional[axis] = {
                "command": command,
                "measured_mean": 0.5 * command,
                "signed_response": abs(0.5 * command),
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
                "twist_judgment": (
                    None
                    if all(value == 0.0 for value in scenario.twist)
                    else twist_judgment(
                        scenario.twist, [0.5 * value for value in scenario.twist]
                    )
                ),
                "twist_beats_standing_passed": True,
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
            "twist_pass_line": twist_pass_line_record(),
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

    def test_one_profile_for_the_checkpoint_a_run_ends_with(self) -> None:
        for completed in (PICO_SCHEDULE["foot_tighten"] + 1, PICO_MIN_FINAL_UPDATES, PICO_TOTAL_UPDATES):
            with self.subTest(completed=completed):
                self.assertEqual(required_tracking_profile(completed), FINAL_PROFILE)
        for completed in (PICO_SCHEDULE["foot_tighten"], PICO_TOTAL_UPDATES + 1, 1):
            with self.subTest(completed=completed), self.assertRaises(ValueError):
                required_tracking_profile(completed)
        self.assertEqual(TRACKING_PROFILES, (FINAL_PROFILE,))
        # The robot validator accepts this published final name.
        self.assertEqual(
            FINAL_PROFILE,
            "full_body_reachable_performance_perturbation_v2_deployed_accuracy_v1",
        )

    def test_accuracy_limits_are_one_table(self) -> None:
        # User decision: hand RMS 0.040 m at every HOME; hand P95 0.07 m;
        # foot 0.05/0.08 m.
        self.assertEqual(hand_tracking_rms_max_m(FINAL_PROFILE), 0.040)
        self.assertEqual(foot_tracking_rms_max_m(FINAL_PROFILE), 0.05)
        self.assertEqual(foot_tracking_p95_max_m(FINAL_PROFILE), 0.08)
        self.assertEqual(hand_tracking_p95_max_m(FINAL_PROFILE), 0.07)
        self.assertTrue(tracking_profile_uses_perturbation(FINAL_PROFILE))
        with self.assertRaises(ValueError):
            hand_tracking_rms_max_m("full_body_reachable_performance_perturbation_v2")

    def test_the_gate_judges_the_report_under_the_final_profile(self) -> None:
        identity = {
            "sha256": "a" * 64,
            "iteration": PICO_TOTAL_UPDATES - 1,
            "completed_updates": PICO_TOTAL_UPDATES,
        }
        within = _tracking_report(identity, hand_rms=0.039)
        self.assertEqual(within["status"], "pass")
        self.assertEqual(_validate_tracking_report(within, identity), FINAL_PROFILE)
        over = _tracking_report(identity, hand_rms=0.041)
        self.assertEqual(over["status"], "fail")
        with self.assertRaises(ValueError):
            _validate_tracking_report(over, identity)
        relabelled = deepcopy(within)
        relabelled["profile"] = "whole_body_reachable_performance_v2_deployed_accuracy_v1"
        with self.assertRaisesRegex(ValueError, "schema/profile/status drifted"):
            _validate_tracking_report(relabelled, identity)
        held_out = deepcopy(within)
        held_out["settings"]["seed"] = 101
        with self.assertRaises(ValueError):
            _validate_tracking_report(held_out, identity)

    def test_final_gate_records_the_final_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / f"model_{PICO_TOTAL_UPDATES - 1}.pt"
            infos = {
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
                    teleop_v12_active_adapter_columns(PICO_TOTAL_UPDATES * 24)
                ),
                "env_state": {"common_step_counter": PICO_TOTAL_UPDATES * 24},
            }
            torch.save({"iter": PICO_TOTAL_UPDATES - 1, "infos": infos}, checkpoint)
            identity = {
                "sha256": sha256_file(checkpoint),
                "iteration": PICO_TOTAL_UPDATES - 1,
                "completed_updates": PICO_TOTAL_UPDATES,
            }
            paths = {
                "locomotion_report": root / "locomotion.json",
                "tracking_report": root / "tracking.json",
                "onnx_report": root / "onnx.json",
            }
            onnx_path = root / "policy.onnx"
            onnx_path.write_bytes(b"unit-test-onnx")
            paths["locomotion_report"].write_text(json.dumps(_locomotion_report(identity)))
            paths["tracking_report"].write_text(
                json.dumps(_tracking_report(identity, hand_rms=0.039))
            )
            paths["onnx_report"].write_text(json.dumps(_onnx_report(identity, onnx_path)))
            gate = create_gate(checkpoint=checkpoint, **paths)
            self.assertEqual(gate["tracking_profile"], FINAL_PROFILE)
            self.assertNotIn("tracking_profile_completion_allowance", gate)
            gate_path = root / "gate.json"
            gate_path.write_text(json.dumps(gate))
            self.assertEqual(validate_gate(gate_path, checkpoint), gate)
            tampered = deepcopy(gate)
            tampered["tracking_profile"] = "whole_body_reachable_performance_v2_deployed_accuracy_v1"
            gate_path.write_text(json.dumps(tampered))
            with self.assertRaises(ValueError):
                validate_gate(gate_path, checkpoint)

    def test_an_adopted_earlier_checkpoint_is_gated_but_not_before_the_minimum(self) -> None:
        for completed, accepted in ((PICO_MIN_FINAL_UPDATES, True), (PICO_MIN_FINAL_UPDATES - 1, False)):
            infos = {
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
                    teleop_v12_active_adapter_columns(completed * 24)
                ),
                "env_state": {"common_step_counter": completed * 24},
            }
            with self.subTest(completed=completed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                onnx_path = root / "policy.onnx"
                onnx_path.write_bytes(b"unit-test-onnx")
                checkpoint = root / f"model_{completed - 1}.pt"
                torch.save({"iter": completed - 1, "infos": infos}, checkpoint)
                identity = {
                    "sha256": sha256_file(checkpoint),
                    "iteration": completed - 1,
                    "completed_updates": completed,
                }
                reports = {
                    "locomotion_report": root / "locomotion.json",
                    "tracking_report": root / "tracking.json",
                    "onnx_report": root / "onnx.json",
                }
                reports["locomotion_report"].write_text(json.dumps(_locomotion_report(identity)))
                reports["tracking_report"].write_text(
                    json.dumps(_tracking_report(identity, profile=FINAL_PROFILE))
                )
                reports["onnx_report"].write_text(json.dumps(_onnx_report(identity, onnx_path)))
                if not accepted:
                    with self.assertRaisesRegex(ValueError, "A PICO run ends between"):
                        create_gate(checkpoint=checkpoint, **reports)
                    continue
                gate = create_gate(checkpoint=checkpoint, **reports)
                self.assertEqual(gate["schema_version"], 3)
                self.assertEqual(gate["tracking_profile"], FINAL_PROFILE)
                self.assertEqual(checkpoint_recipe_kind(checkpoint), "hand_pose_release")
                gate_path = root / "gate.json"
                gate_path.write_text(json.dumps(gate))
                self.assertEqual(validate_gate(gate_path, checkpoint), gate)

    def test_tracking_acceptance_rejects_done_coverage_direction_and_active_error(
        self,
    ) -> None:
        checks, status = _acceptance([_result()], FINAL_PROFILE)
        self.assertEqual(status, "pass")
        self.assertTrue(all(checks.values()))
        for mutation in (
            {"completed": False, "termination_names": ["out_of_terrain_bounds"]},
            {"observation_coverage": {"passed": False}},
            {"twist_beats_standing_passed": False},
            {
                "maximum_actual_soft_limit_violation_rad": (
                    ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD + 1.0e-9
                )
            },
        ):
            with self.subTest(mutation=mutation):
                failed, failed_status = _acceptance(
                    [_result(**mutation)], FINAL_PROFILE
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
            FINAL_PROFILE,
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
            required_target_column_ablation_targets(FINAL_PROFILE), frozenset(("hand", "foot"))
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
        checks, status = _acceptance([unresponsive], FINAL_PROFILE)
        self.assertEqual(status, "fail")
        self.assertFalse(checks["target_column_ablation_response"])
        hand_only = deepcopy(unresponsive)
        hand_only["target_column_ablation"]["hand"] = _ablation(
            target="hand", expected=True
        )
        checks, status = _acceptance([hand_only], FINAL_PROFILE)
        self.assertEqual(status, "fail")
        self.assertFalse(checks["target_column_ablation_response"])
        both = deepcopy(hand_only)
        both["target_column_ablation"]["foot"] = _ablation(target="foot", expected=True)
        checks, status = _acceptance([both], FINAL_PROFILE)
        self.assertEqual(status, "pass")

    def test_tracking_ablation_evidence_is_exact_and_fail_closed(self) -> None:
        identity = {
            "sha256": "a" * 64,
            "iteration": PICO_TOTAL_UPDATES - 1,
            "completed_updates": PICO_TOTAL_UPDATES,
        }
        report = _tracking_report(identity)
        self.assertEqual(
            set(report["checks"]),
            set(required_tracking_check_names(FINAL_PROFILE)),
        )
        self.assertIn("hand_tracking_rms", report["checks"])
        self.assertIn("foot_tracking_rms", report["checks"])
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

    def test_gate_evidence_is_hash_bound_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / f"model_{PICO_TOTAL_UPDATES - 1}.pt"
            infos = {
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
                    teleop_v12_active_adapter_columns(PICO_TOTAL_UPDATES * 24)
                ),
                "env_state": {"common_step_counter": PICO_TOTAL_UPDATES * 24},
            }
            torch.save({"iter": PICO_TOTAL_UPDATES - 1, "infos": infos}, checkpoint)
            checkpoint_sha = sha256_file(checkpoint)
            identity = {
                "sha256": checkpoint_sha,
                "iteration": PICO_TOTAL_UPDATES - 1,
                "completed_updates": PICO_TOTAL_UPDATES,
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
            self.assertEqual(gate["schema_version"], 3)
            gate_path = root / "gate.json"
            gate_path.write_text(json.dumps(gate))
            self.assertEqual(validate_gate(gate_path, checkpoint), gate)

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


if __name__ == "__main__":
    unittest.main()
