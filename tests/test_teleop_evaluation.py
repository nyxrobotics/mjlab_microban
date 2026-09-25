# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for the deterministic teleop checkpoint evaluator."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from tensordict import TensorDict

from mjlab_microban.scripts.evaluate_teleop_checkpoint import (
    CANONICAL_ACTIVE_FOOT_Z_LOWER_EDGE_M,
    DEPLOYMENT_PERFORMANCE_PROFILE,
    HMD_ACTUAL_PEAK_TO_PEAK_MIN_RAD,
    HMD_TARGET_PEAK_TO_PEAK_MIN_RAD,
    INTERMEDIATE_HARD_SAFETY_PROFILE,
    EvaluationScenario,
    HmdMotionStats,
    _configure_nominal_evaluation,
    _contact_aligned_foot_site_ids,
    _copy_forced_moving_hmd_neck_event,
    _hmd_neck_motion_report,
    _patch_initial_command_observation,
    _percentile,
    _publish_json_report,
    build_report,
    checkpoint_sha256,
    command_axis_sign_diagnostic,
    default_scenarios,
    evaluate_scenario_acceptance,
    evaluation_exit_code,
    resolve_checkpoint,
    select_scenarios,
    summarize_hmd_motion_evidence,
    validate_scenarios,
)
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_HMD_JOINT_NAMES,
    MICROBAN_TELEOP_OBSERVATION_SCHEMA,
    MICROBAN_TELEOP_OBSERVATION_WIDTH,
    MICROBAN_TELEOP_PREVIOUS_ACTION_SEMANTICS,
    TeleopCheckpointContract,
)
from mjlab_microban.tasks.microban_teleop_mdp import (
    MICROBAN_HMD_RETARGET_INTERVAL_S,
    MICROBAN_HMD_RUNTIME_LIMITS_RAD,
    MICROBAN_HMD_SLEW_RATES_RAD_S,
    MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M,
    HmdNeckTargetMotion,
)


class ScenarioContractTest(unittest.TestCase):
    def test_default_suite_covers_live_extrema_and_both_feet(self) -> None:
        scenarios = default_scenarios()
        validate_scenarios(scenarios)
        by_name = {scenario.name: scenario for scenario in scenarios}

        self.assertEqual(len(scenarios), 33)
        self.assertEqual(MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M, 0.0025)
        self.assertEqual(CANONICAL_ACTIVE_FOOT_Z_LOWER_EDGE_M, 0.0026)
        self.assertEqual(
            by_name["neutral"].foot_target,
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        )
        self.assertEqual(
            by_name["floor_band_edge_single"].foot_target,
            (
                (0.0, 0.0, CANONICAL_ACTIVE_FOOT_Z_LOWER_EDGE_M),
                (0.0, 0.0, 0.0),
            ),
        )
        self.assertEqual(
            by_name["floor_band_edge_both"].foot_target,
            (
                (0.0, 0.0, CANONICAL_ACTIVE_FOOT_Z_LOWER_EDGE_M),
                (0.0, 0.0, CANONICAL_ACTIVE_FOOT_Z_LOWER_EDGE_M),
            ),
        )
        self.assertGreater(
            CANONICAL_ACTIVE_FOOT_Z_LOWER_EDGE_M,
            MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M,
        )
        self.assertEqual(
            by_name["floor_band_edge_single"].twist,
            (0.0, 0.0, 0.0),
        )
        self.assertEqual(
            by_name["floor_band_edge_both"].twist,
            (0.0, 0.0, 0.0),
        )
        self.assertEqual(by_name["max_forward"].twist, (0.7, 0.0, 0.0))
        self.assertEqual(by_name["max_backward"].twist, (-0.5, 0.0, 0.0))
        self.assertEqual(by_name["mid_lateral_left"].twist, (0.0, 0.2, 0.0))
        self.assertEqual(by_name["mid_lateral_right"].twist, (0.0, -0.2, 0.0))
        self.assertEqual(by_name["max_moving_yaw_left"].twist, (0.0, 0.0, 1.5))
        self.assertEqual(by_name["max_moving_yaw_right"].twist, (0.0, 0.0, -1.5))
        self.assertEqual(by_name["max_stationary_yaw_left"].twist, (0.0, 0.0, 3.0))
        self.assertEqual(
            by_name["mixed_twist_forward_left"].twist,
            (0.7, 0.3, 1.5),
        )
        self.assertEqual(
            by_name["mixed_twist_forward_left"].foot_target,
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        )
        self.assertEqual(
            by_name["mixed_twist_forward_left"].hand_active,
            (False, False),
        )
        self.assertEqual(
            by_name["max_hands_left"].foot_target,
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        )
        self.assertEqual(by_name["max_hands_left"].hand_active, (True, True))
        both_feet = by_name["bounded_both_feet"].foot_target
        self.assertTrue(
            all(any(abs(value) > 0.0 for value in foot) for foot in both_feet)
        )
        self.assertEqual(
            both_feet,
            ((0.008, -0.008, 0.016), (-0.008, 0.008, 0.016)),
        )
        self.assertEqual(by_name["bounded_both_feet"].twist, (0.0, 0.0, 0.0))

    def test_rejects_command_outside_runtime_envelope(self) -> None:
        zero = (0.0, 0.0, 0.0)
        invalid = EvaluationScenario(
            "too_fast",
            (0.71, 0.0, 0.0),
            (zero, zero),
            (zero, zero),
            (False, False),
        )
        with self.assertRaisesRegex(ValueError, "vx exceeds"):
            validate_scenarios((invalid,))

    def test_simultaneous_both_feet_require_narrow_bounds_and_zero_twist(self) -> None:
        hands = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        too_wide = EvaluationScenario(
            "both_too_wide",
            (0.0, 0.0, 0.0),
            ((0.0080001, 0.0, 0.01), (-0.008, 0.0, 0.01)),
            hands,
            (False, False),
        )
        with self.assertRaisesRegex(ValueError, "simultaneous foot XY"):
            validate_scenarios((too_wide,))

        moving = EvaluationScenario(
            "both_moving",
            (0.01, 0.0, 0.0),
            ((0.005, 0.0, 0.01), (-0.005, 0.0, 0.01)),
            hands,
            (False, False),
        )
        with self.assertRaisesRegex(ValueError, "require zero twist"):
            validate_scenarios((moving,))

    def test_floor_band_requires_exact_zero_or_active_above_boundary(self) -> None:
        zero = (0.0, 0.0, 0.0)
        inside_floor_band = EvaluationScenario(
            "unprojected_floor_band",
            zero,
            ((1.0e-12, 0.0, MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M), zero),
            (zero, zero),
            (False, False),
        )
        with self.assertRaisesRegex(ValueError, "floor band.*exact XYZ zero"):
            validate_scenarios((inside_floor_band,))

        lower_edge_single = next(
            scenario
            for scenario in default_scenarios()
            if scenario.name == "floor_band_edge_single"
        )
        lower_edge_both = next(
            scenario
            for scenario in default_scenarios()
            if scenario.name == "floor_band_edge_both"
        )
        validate_scenarios((lower_edge_single, lower_edge_both))

    def test_subset_selection_preserves_requested_order(self) -> None:
        selected = select_scenarios(default_scenarios(), "max_backward,neutral")
        self.assertEqual([item.name for item in selected], ["max_backward", "neutral"])
        with self.assertRaisesRegex(ValueError, "Unknown scenarios"):
            select_scenarios(default_scenarios(), "not-a-scenario")


class MovingHmdConfigurationTest(unittest.TestCase):
    @staticmethod
    def _evaluation_cfg() -> SimpleNamespace:
        return SimpleNamespace(
            scene=SimpleNamespace(num_envs=64),
            auto_reset=True,
            episode_length_s=0.0,
            decimation=4,
            sim=SimpleNamespace(mujoco=SimpleNamespace(timestep=0.005)),
            events={
                "reset_base": SimpleNamespace(
                    mode="reset",
                    params={"pose_range": {"x": (-1.0, 1.0)}, "velocity_range": {}},
                ),
                "reset_robot_joints": SimpleNamespace(mode="reset", params={}),
                "push_robot": SimpleNamespace(mode="interval", params={}),
            },
            observations={"actor": SimpleNamespace(enable_corruption=True)},
            curriculum={"training_stage": object()},
        )

    @staticmethod
    def _training_cfg() -> SimpleNamespace:
        return SimpleNamespace(
            events={
                "hmd_neck_target_motion": SimpleNamespace(
                    mode="step",
                    func=HmdNeckTargetMotion,
                    params={
                        "asset_cfg": SimpleNamespace(
                            joint_names=MICROBAN_HMD_JOINT_NAMES
                        ),
                        "position_ranges_rad": MICROBAN_HMD_RUNTIME_LIMITS_RAD,
                        "slew_rates_rad_s": MICROBAN_HMD_SLEW_RATES_RAD_S,
                        "retarget_interval_s": MICROBAN_HMD_RETARGET_INTERVAL_S,
                        "neutral_probability": 1.0,
                    },
                )
            }
        )

    def test_forced_motion_copies_only_hmd_event_and_disables_other_randomness(
        self,
    ) -> None:
        training_cfg = self._training_cfg()
        copied = _copy_forced_moving_hmd_neck_event(training_cfg)
        self.assertEqual(copied.params["neutral_probability"], 0.0)
        self.assertEqual(
            training_cfg.events["hmd_neck_target_motion"].params["neutral_probability"],
            1.0,
        )

        cfg = self._evaluation_cfg()
        _configure_nominal_evaluation(
            cfg,
            steps=1000,
            moving_hmd_neck_event=copied,
        )
        self.assertEqual(
            set(cfg.events),
            {"reset_base", "reset_robot_joints", "hmd_neck_target_motion"},
        )
        self.assertFalse(cfg.observations["actor"].enable_corruption)
        self.assertEqual(cfg.curriculum, {})
        self.assertFalse(cfg.auto_reset)
        self.assertEqual(cfg.scene.num_envs, 1)
        self.assertAlmostEqual(cfg.episode_length_s, 20.0)

        report = _hmd_neck_motion_report(cfg)
        self.assertTrue(report["enabled"])
        self.assertEqual(report["params"]["neutral_probability"], 0.0)
        self.assertEqual(
            report["params"]["joint_names"], list(MICROBAN_HMD_JOINT_NAMES)
        )

    def test_nominal_configuration_has_no_hmd_event(self) -> None:
        cfg = self._evaluation_cfg()
        _configure_nominal_evaluation(cfg, steps=300)
        self.assertEqual(
            set(cfg.events),
            {"reset_base", "reset_robot_joints"},
        )
        self.assertEqual(
            _hmd_neck_motion_report(cfg),
            {"enabled": False, "params": None},
        )


class MovingHmdEvidenceTest(unittest.TestCase):
    @staticmethod
    def _scenario_motion(
        *, target_peak_to_peak: float, actual_peak_to_peak: float
    ) -> dict[str, object]:
        return {
            "active_event_member": True,
            "sample_count": 1001,
            "joint_names": list(MICROBAN_HMD_JOINT_NAMES),
            "per_axis": {
                name: {
                    "target_peak_to_peak_rad": target_peak_to_peak,
                    "actual_peak_to_peak_rad": actual_peak_to_peak,
                }
                for name in MICROBAN_HMD_JOINT_NAMES
            },
        }

    def test_streaming_stats_report_per_axis_target_and_actual_excursion(self) -> None:
        stats = HmdMotionStats.start(
            joint_names=MICROBAN_HMD_JOINT_NAMES,
            target=torch.tensor([0.0, -0.1, 0.2]),
            actual=torch.tensor([0.0, -0.05, 0.1]),
        )
        stats.add(
            target=torch.tensor([0.2, 0.1, -0.1]),
            actual=torch.tensor([0.1, 0.02, -0.05]),
        )
        report = stats.report(step_dt=0.02)
        self.assertTrue(report["active_event_member"])
        self.assertEqual(report["sample_count"], 2)
        self.assertAlmostEqual(
            report["per_axis"]["head"]["target_peak_to_peak_rad"], 0.2
        )
        self.assertAlmostEqual(
            report["per_axis"]["neck_pitch"]["actual_peak_to_peak_rad"],
            0.15,
            places=5,
        )
        self.assertAlmostEqual(
            report["per_axis"]["head"]["maximum_target_slew_rad_s"],
            10.0,
            places=5,
        )

    def test_motion_evidence_requires_active_membership_and_both_excursions(
        self,
    ) -> None:
        passing = [
            {
                "name": "neutral",
                "hmd_neck_motion": self._scenario_motion(
                    target_peak_to_peak=HMD_TARGET_PEAK_TO_PEAK_MIN_RAD,
                    actual_peak_to_peak=HMD_ACTUAL_PEAK_TO_PEAK_MIN_RAD,
                ),
            }
        ]
        evidence = summarize_hmd_motion_evidence(passing, required=True)
        self.assertTrue(evidence["passed"])
        self.assertTrue(evidence["active_event_membership"]["all_scenarios"])

        inactive = [
            {
                "name": "neutral",
                "hmd_neck_motion": {
                    "active_event_member": False,
                    "joint_names": [],
                    "per_axis": {},
                },
            }
        ]
        self.assertFalse(
            summarize_hmd_motion_evidence(inactive, required=True)["passed"]
        )

        insufficient = [
            {
                "name": "neutral",
                "hmd_neck_motion": self._scenario_motion(
                    target_peak_to_peak=HMD_TARGET_PEAK_TO_PEAK_MIN_RAD,
                    actual_peak_to_peak=HMD_ACTUAL_PEAK_TO_PEAK_MIN_RAD - 1.0e-6,
                ),
            }
        ]
        evidence = summarize_hmd_motion_evidence(insufficient, required=True)
        self.assertFalse(evidence["passed"])
        self.assertFalse(evidence["actual_excursion_passed"])


class CheckpointResolutionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def _checkpoint(path: Path, *, mtime: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"checkpoint")
        os.utime(path, (mtime, mtime))

    def test_uses_highest_numeric_stable_checkpoint(self) -> None:
        run = self.root / "2026-09-24_10-42-43"
        self._checkpoint(run / "model_9500.pt", mtime=900.0)
        self._checkpoint(run / "model_10000.pt", mtime=995.0)
        selected = resolve_checkpoint(
            None,
            self.root,
            minimum_age_s=10.0,
            now_s=1000.0,
        )
        self.assertEqual(selected, (run / "model_9500.pt").resolve())

    def test_rejects_recent_explicit_checkpoint(self) -> None:
        path = self.root / "model_500.pt"
        self._checkpoint(path, mtime=995.0)
        with self.assertRaisesRegex(RuntimeError, "may still be written"):
            resolve_checkpoint(
                path,
                minimum_age_s=10.0,
                now_s=1000.0,
            )

    def test_checkpoint_sha256_is_reproducible(self) -> None:
        path = self.root / "model_500.pt"
        path.write_bytes(b"checkpoint")
        self.assertEqual(
            checkpoint_sha256(path),
            "47320987f9a49d5b00119b960f247a956773f57543982b8bfcb6da5bb3afd9ef",
        )


class ReportPublicationTest(unittest.TestCase):
    def test_no_force_publishes_a_new_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "report.json"

            _publish_json_report(output, '{"new": true}', force=False)
            self.assertEqual(output.read_text(encoding="utf-8"), '{"new": true}\n')

    def test_legacy_contract_report_can_never_pass(self) -> None:
        scenarios = []
        for scenario in default_scenarios():
            scenarios.append(
                {
                    "name": scenario.name,
                    "fell_over": False,
                    "finite": True,
                    "completed_steps": 1000,
                    "requested_steps": 1000,
                    "termination_terms": ["time_out"],
                    "reached_time_limit": True,
                    "acceptance": {"passed": True},
                    "foot_slip": {"max": None},
                    "target_error": {},
                    "self_collision": {"step_fraction": None},
                    "joint_soft_limits": {"max_actual_violation_rad": 0.0},
                }
            )
        common = {
            "checkpoint": Path("model_14999.pt"),
            "checkpoint_digest": "0" * 64,
            "checkpoint_size_bytes": 1,
            "device": "cpu",
            "seed": 42,
            "steps": 1000,
            "settle_steps": 50,
            "saturation_margin_ratio": 0.01,
            "scenario_reports": scenarios,
            "control_hz": 50.0,
        }
        current_report = build_report(
            **common,
            checkpoint_contract=TeleopCheckpointContract(
                version="8",
                previous_action_semantics=MICROBAN_TELEOP_PREVIOUS_ACTION_SEMANTICS,
                iteration=14999,
                common_step_counter=360000,
            ),
        )
        self.assertEqual(current_report["status"], "pass")
        self.assertEqual(
            current_report["acceptance_profile"], DEPLOYMENT_PERFORMANCE_PROFILE
        )
        self.assertTrue(
            current_report["summary"]["performance_acceptance_checks_enforced"]
        )
        self.assertEqual(current_report["schema_version"], 8)
        self.assertFalse(current_report["hmd_neck_motion"]["enabled"])
        self.assertIsNone(current_report["hmd_neck_motion"]["params"])
        self.assertTrue(current_report["hmd_neck_motion"]["evidence"]["passed"])
        self.assertFalse(current_report["nominal_environment"]["hmd_neck_motion"])

        moving_scenarios = []
        for scenario_report in scenarios:
            moving_scenarios.append(
                {
                    **scenario_report,
                    "hmd_neck_motion": MovingHmdEvidenceTest._scenario_motion(
                        target_peak_to_peak=HMD_TARGET_PEAK_TO_PEAK_MIN_RAD,
                        actual_peak_to_peak=HMD_ACTUAL_PEAK_TO_PEAK_MIN_RAD,
                    ),
                }
            )
        moving_hmd_report = build_report(
            **{**common, "scenario_reports": moving_scenarios},
            checkpoint_contract=TeleopCheckpointContract(
                version="8",
                previous_action_semantics=MICROBAN_TELEOP_PREVIOUS_ACTION_SEMANTICS,
                iteration=14999,
                common_step_counter=360000,
            ),
            hmd_neck_motion={
                "enabled": True,
                "params": {"neutral_probability": 0.0},
            },
        )
        self.assertEqual(moving_hmd_report["status"], "diagnostic")
        self.assertEqual(evaluation_exit_code(moving_hmd_report), 3)
        self.assertFalse(moving_hmd_report["summary"]["canonical_coverage"])
        self.assertTrue(moving_hmd_report["summary"]["acceptance_checks_passed"])
        self.assertTrue(moving_hmd_report["hmd_neck_motion"]["enabled"])
        self.assertTrue(moving_hmd_report["hmd_neck_motion"]["evidence"]["passed"])
        self.assertTrue(moving_hmd_report["nominal_environment"]["hmd_neck_motion"])

        failing_scenarios = [dict(item) for item in scenarios]
        failing_scenarios[0] = {
            **failing_scenarios[0],
            "acceptance": {"passed": False},
        }
        failed_current_report = build_report(
            **{**common, "scenario_reports": failing_scenarios},
            checkpoint_contract=TeleopCheckpointContract(
                version="8",
                previous_action_semantics=MICROBAN_TELEOP_PREVIOUS_ACTION_SEMANTICS,
                iteration=14999,
                common_step_counter=360000,
            ),
        )
        self.assertTrue(failed_current_report["summary"]["canonical_coverage"])
        self.assertFalse(failed_current_report["summary"]["acceptance_checks_passed"])
        self.assertEqual(failed_current_report["status"], "fail")
        self.assertEqual(evaluation_exit_code(failed_current_report), 2)
        self.assertEqual(evaluation_exit_code(current_report), 0)

        intermediate_report = build_report(
            **common,
            checkpoint_contract=TeleopCheckpointContract(
                version="9",
                previous_action_semantics=MICROBAN_TELEOP_PREVIOUS_ACTION_SEMANTICS,
                iteration=1499,
                common_step_counter=36000,
            ),
            intermediate_hard_safety_only=True,
        )
        self.assertEqual(intermediate_report["status"], "diagnostic")
        self.assertEqual(
            intermediate_report["acceptance_profile"],
            INTERMEDIATE_HARD_SAFETY_PROFILE,
        )
        self.assertFalse(
            intermediate_report["summary"][
                "performance_acceptance_checks_enforced"
            ]
        )
        self.assertEqual(evaluation_exit_code(intermediate_report), 3)

        legacy_report = build_report(
            **common,
            checkpoint_contract=TeleopCheckpointContract(
                version="legacy_unversioned_v1",
                previous_action_semantics="raw_policy_output_before_target_clip",
                iteration=14999,
                common_step_counter=360000,
                diagnostic_legacy=True,
            ),
        )
        self.assertEqual(legacy_report["status"], "diagnostic")
        self.assertEqual(evaluation_exit_code(legacy_report), 3)
        self.assertTrue(legacy_report["training_contract"]["diagnostic_legacy"])
        self.assertFalse(legacy_report["training_contract"]["deployment_compatible"])
        self.assertFalse(legacy_report["summary"]["deployment_certified"])

        pristine_report = build_report(
            **{**common, "checkpoint": Path("model_pristine.pt")},
            checkpoint_contract=TeleopCheckpointContract(
                version="8",
                previous_action_semantics=MICROBAN_TELEOP_PREVIOUS_ACTION_SEMANTICS,
                iteration=-1,
                common_step_counter=0,
                pristine_pre_update=True,
            ),
        )
        self.assertEqual(pristine_report["status"], "diagnostic")
        self.assertTrue(pristine_report["training_contract"]["pristine_pre_update"])
        self.assertFalse(pristine_report["training_contract"]["deployment_compatible"])
        self.assertFalse(pristine_report["summary"]["deployment_certified"])
        self.assertEqual(evaluation_exit_code({"status": "unknown"}), 2)
        self.assertEqual(evaluation_exit_code({}), 2)

    def test_no_force_never_replaces_an_existing_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "report.json"
            output.write_text("original\n", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "refusing to replace"):
                _publish_json_report(output, '{"new": true}', force=False)
            self.assertEqual(output.read_text(encoding="utf-8"), "original\n")

    def test_force_atomically_replaces_an_existing_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "report.json"
            output.write_text("original\n", encoding="utf-8")

            _publish_json_report(output, '{"new": true}', force=True)
            self.assertEqual(output.read_text(encoding="utf-8"), '{"new": true}\n')


class ObservationPatchTest(unittest.TestCase):
    def test_only_command_slices_are_replaced(self) -> None:
        original_actor = torch.arange(
            MICROBAN_TELEOP_OBSERVATION_WIDTH, dtype=torch.float32
        ).unsqueeze(0)
        observations = TensorDict(
            {"actor": original_actor.clone()},
            batch_size=[1],
        )
        values = {
            "twist": torch.tensor([[0.1, 0.2, 0.3]]),
            "foot_target": torch.arange(6, dtype=torch.float32).unsqueeze(0) + 100.0,
            "hand_target": torch.arange(8, dtype=torch.float32).unsqueeze(0) + 200.0,
        }
        terms = {name: SimpleNamespace(command=value) for name, value in values.items()}
        manager = SimpleNamespace(get_term=lambda name: terms[name])
        env = SimpleNamespace(num_envs=1, command_manager=manager)

        patched = _patch_initial_command_observation(observations, env)
        self.assertTrue(torch.equal(observations["actor"], original_actor))

        offset = 0
        for name, width in MICROBAN_TELEOP_OBSERVATION_SCHEMA:
            actual = patched["actor"][:, offset : offset + width]
            source_name = "twist" if name == "command" else name
            if source_name in values:
                self.assertTrue(torch.equal(actual, values[source_name]))
            else:
                self.assertTrue(
                    torch.equal(actual, original_actor[:, offset : offset + width])
                )
            offset += width


class MetricHelperTest(unittest.TestCase):
    def test_contact_order_is_mapped_to_foot_site_order(self) -> None:
        foot_contact = SimpleNamespace(
            _slots=[
                SimpleNamespace(primary_name="foot", field_name="found"),
                SimpleNamespace(primary_name="foot", field_name="force"),
                SimpleNamespace(primary_name="foot_2", field_name="found"),
            ]
        )
        foot = SimpleNamespace(
            cfg=SimpleNamespace(foot_site_names=("left_foot", "right_foot")),
            _foot_asset_cfg=SimpleNamespace(site_ids=(10, 20)),
        )

        self.assertEqual(
            _contact_aligned_foot_site_ids(foot_contact, foot),
            [20, 10],
        )

    def test_percentile_interpolates_and_handles_empty_input(self) -> None:
        self.assertIsNone(_percentile([], 0.95))
        self.assertAlmostEqual(_percentile([0.0, 10.0], 0.95), 9.5)

    def test_command_axis_sign_diagnostic_uses_signed_measured_means(self) -> None:
        zero = (0.0, 0.0, 0.0)
        scenario = EvaluationScenario(
            "mixed_velocity",
            (0.7, -0.3, 3.0),
            (zero, zero),
            (zero, zero),
            (False, False),
        )
        measured = {
            "linear_x": {"mean": 0.1},
            "linear_y": {"mean": -0.01},
            "yaw": {"mean": 0.2},
        }

        diagnostic = command_axis_sign_diagnostic(scenario, measured)
        self.assertTrue(diagnostic["applied"])
        self.assertTrue(diagnostic["passed"])
        self.assertEqual(diagnostic["axes"]["linear_y"]["expected_sign"], -1)
        self.assertEqual(diagnostic["axes"]["linear_y"]["actual_sign"], -1)

        measured["linear_y"]["mean"] = 0.01
        diagnostic = command_axis_sign_diagnostic(scenario, measured)
        self.assertFalse(diagnostic["passed"])
        self.assertFalse(diagnostic["axes"]["linear_y"]["passed"])

        measured["linear_y"]["mean"] = None
        diagnostic = command_axis_sign_diagnostic(scenario, measured)
        self.assertFalse(diagnostic["passed"])
        self.assertIsNone(diagnostic["axes"]["linear_y"]["actual_sign"])

        neutral = EvaluationScenario(
            "neutral",
            zero,
            (zero, zero),
            (zero, zero),
            (False, False),
        )
        missing = {name: {"mean": None} for name in ("linear_x", "linear_y", "yaw")}
        diagnostic = command_axis_sign_diagnostic(neutral, missing)
        self.assertFalse(diagnostic["applied"])
        self.assertTrue(diagnostic["passed"])

    def test_command_axis_sign_is_an_acceptance_gate(self) -> None:
        zero = (0.0, 0.0, 0.0)
        scenario = EvaluationScenario(
            "wrong_way",
            (0.7, 0.0, 0.0),
            (zero, zero),
            (zero, zero),
            (False, False),
        )
        measured = {
            "linear_x": {"mean": -0.01},
            "linear_y": {"mean": 0.0},
            "yaw": {"mean": 0.0},
        }
        report = {
            "finite": True,
            "fell_over": False,
            "reached_time_limit": True,
            "self_collision": {"contact_count_total": 0},
            "action": {"target_clip_fraction": 0.0},
            "joint_soft_limits": {"max_actual_violation_rad": 0.0},
            "velocity_tracking_error": {
                "linear_xy": {"mean": 0.01},
                "yaw": {"mean": 0.01},
            },
            "target_error": {"active_hand": {}, "foot": {}},
            "command_axis_sign": command_axis_sign_diagnostic(scenario, measured),
        }

        acceptance = evaluate_scenario_acceptance(scenario, report)
        self.assertFalse(acceptance["passed"])
        self.assertEqual(acceptance["failed_checks"], ["command_axis_sign"])

        safety_only = evaluate_scenario_acceptance(
            scenario, report, enforce_performance=False
        )
        self.assertTrue(safety_only["passed"])
        self.assertFalse(safety_only["performance_checks_enforced"])
        self.assertNotIn("command_axis_sign", safety_only["checks"])


if __name__ == "__main__":
    unittest.main()
