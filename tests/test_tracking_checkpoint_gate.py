# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""CPU-only tests for the learned tracking-checkpoint gate."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import torch

from mjlab_microban.scripts.evaluate_tracking_checkpoint import (
    _agent_cfg_for_checkpoint,
    _checkpoint_actor_contract,
    _runtime_source_evidence,
)
from mjlab_microban.tracking_checkpoint_gate import (
    TRACKING_ACTOR_KIND_BOUNDED_LOG,
    TRACKING_ACTOR_KIND_UNBOUNDED_LOG,
    TRACKING_ACTOR_KIND_UNBOUNDED_SCALAR,
    TRACKING_ACTOR_OBSERVATION_SCHEMA,
    TRACKING_ACTOR_OBSERVATION_WIDTH,
    TRACKING_CHECKPOINT_GATE_EXPECTED_STEPS,
    TRACKING_CHECKPOINT_GATE_FIRST_TARGET_FRAME,
    TRACKING_CHECKPOINT_GATE_FRAME_COUNT,
    TRACKING_CHECKPOINT_GATE_INITIAL_FRAME,
    TRACKING_CHECKPOINT_GATE_NUM_ENVS,
    classify_tracking_actor_state_keys,
    finalize_tracking_checkpoint_report,
    tracking_checkpoint_checks,
    tracking_checkpoint_exit_code,
    validate_tracking_checkpoint_receipt_digest,
)


def _passing_measurements() -> dict[str, float | int | bool]:
    return {
        "executed_policy_steps": TRACKING_CHECKPOINT_GATE_EXPECTED_STEPS,
        "fall_envs": 0,
        "forward_displacement_p05_m": 1.0e-9,
        "forward_velocity_p05_m_s": 0.05,
        "full_clip_completed_envs": TRACKING_CHECKPOINT_GATE_NUM_ENVS,
        "left_foot_airborne_envs": TRACKING_CHECKPOINT_GATE_NUM_ENVS,
        "maximum_actual_soft_limit_violation_rad": 1.0e-6,
        "maximum_target_projection_rad": 0.001,
        "minimum_root_height_m": 0.10,
        "nonfinite_envs": 0,
        "right_foot_airborne_envs": TRACKING_CHECKPOINT_GATE_NUM_ENVS,
        "self_collision_contacts": 0,
        "simultaneous_flight_env_steps": 0,
        "source_frame_sequence_complete": True,
        "target_clip_fraction": 0.001,
        "unexpected_termination_envs": 0,
        "xy_velocity_mae_p95_m_s": 0.075,
    }


class TrackingCheckpointGateTest(unittest.TestCase):
    def test_runtime_provenance_hashes_every_local_gate_dependency(self) -> None:
        evidence = _runtime_source_evidence()
        paths = {item["path"] for item in evidence}
        self.assertIn("src/mjlab_microban/tracking_checkpoint_gate.py", paths)
        self.assertIn("src/mjlab_microban/tasks/microban_tracking_env_cfg.py", paths)
        self.assertIn("src/mjlab_microban/robot/xc330_params.json", paths)
        self.assertIn("uv.lock", paths)
        for item in evidence:
            self.assertEqual(len(item["sha256"]), 64)
            self.assertGreater(item["size_bytes"], 0)

    def test_checkpoint_actor_distribution_is_classified_from_state(self) -> None:
        self.assertEqual(
            classify_tracking_actor_state_keys(("distribution.std_param",)),
            TRACKING_ACTOR_KIND_UNBOUNDED_SCALAR,
        )
        self.assertEqual(
            classify_tracking_actor_state_keys(("distribution.log_std_param",)),
            TRACKING_ACTOR_KIND_UNBOUNDED_LOG,
        )
        self.assertEqual(
            classify_tracking_actor_state_keys(
                (
                    "distribution.log_std_param",
                    "distribution.lower_bound",
                    "distribution.operational_lower_bound",
                )
            ),
            TRACKING_ACTOR_KIND_BOUNDED_LOG,
        )
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            classify_tracking_actor_state_keys(
                ("distribution.std_param", "distribution.log_std_param")
            )
        with self.assertRaisesRegex(ValueError, "missing"):
            classify_tracking_actor_state_keys(("mlp.0.weight",))

    def test_checkpoint_actor_normalizer_is_reconstructed_from_state(self) -> None:
        cases = (
            (
                "unbounded_normalized",
                {
                    "distribution.std_param": torch.ones(18),
                    "obs_normalizer._mean": torch.zeros(99),
                },
                TRACKING_ACTOR_KIND_UNBOUNDED_SCALAR,
                True,
            ),
            (
                "bounded_normalized",
                {
                    "distribution.log_std_param": torch.zeros(18),
                    "distribution.lower_bound": -torch.ones(18),
                    "distribution.operational_lower_bound": -torch.ones(18),
                    "obs_normalizer.count": torch.tensor(1),
                },
                TRACKING_ACTOR_KIND_BOUNDED_LOG,
                True,
            ),
            (
                "bounded_raw",
                {
                    "distribution.log_std_param": torch.zeros(18),
                    "distribution.lower_bound": -torch.ones(18),
                    "distribution.operational_lower_bound": -torch.ones(18),
                },
                TRACKING_ACTOR_KIND_BOUNDED_LOG,
                False,
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            for name, state, expected_kind, expected_normalization in cases:
                with self.subTest(name=name):
                    checkpoint = Path(directory) / f"{name}.pt"
                    torch.save({"actor_state_dict": state}, checkpoint)
                    kind, normalization = _checkpoint_actor_contract(checkpoint)
                    self.assertEqual(kind, expected_kind)
                    self.assertEqual(normalization, expected_normalization)
                    cfg, evidence = _agent_cfg_for_checkpoint(
                        kind,
                        actor_obs_normalization=normalization,
                    )
                    self.assertEqual(cfg.actor.obs_normalization, normalization)
                    self.assertEqual(
                        evidence["checkpoint_actor_observation_normalization"],
                        expected_normalization,
                    )

    def test_actor_observation_contract_is_exactly_99_values(self) -> None:
        self.assertEqual(
            TRACKING_ACTOR_OBSERVATION_SCHEMA,
            (
                ("command", 36),
                ("motion_anchor_ori_b", 6),
                ("base_ang_vel", 3),
                ("joint_pos", 18),
                ("joint_vel", 18),
                ("actions", 18),
            ),
        )
        self.assertEqual(TRACKING_ACTOR_OBSERVATION_WIDTH, 99)

    def test_full_clip_is_initial_state_plus_all_remaining_targets(self) -> None:
        self.assertEqual(TRACKING_CHECKPOINT_GATE_INITIAL_FRAME, 0)
        self.assertEqual(TRACKING_CHECKPOINT_GATE_FIRST_TARGET_FRAME, 1)
        self.assertEqual(TRACKING_CHECKPOINT_GATE_FRAME_COUNT, 268)
        self.assertEqual(TRACKING_CHECKPOINT_GATE_EXPECTED_STEPS, 267)

    def test_exact_existing_acceptance_boundaries_pass(self) -> None:
        checks = tracking_checkpoint_checks(_passing_measurements())
        self.assertTrue(all(check["passed"] for check in checks.values()))

    def test_every_safety_or_capability_failure_is_fail_closed(self) -> None:
        mutations: dict[str, float | int | bool] = {
            "executed_policy_steps": TRACKING_CHECKPOINT_GATE_EXPECTED_STEPS - 1,
            "fall_envs": 1,
            "forward_displacement_p05_m": 0.0,
            "forward_velocity_p05_m_s": 0.049999,
            "full_clip_completed_envs": TRACKING_CHECKPOINT_GATE_NUM_ENVS - 1,
            "left_foot_airborne_envs": TRACKING_CHECKPOINT_GATE_NUM_ENVS - 1,
            "maximum_actual_soft_limit_violation_rad": 1.0001e-6,
            "maximum_target_projection_rad": 0.001001,
            "minimum_root_height_m": 0.09999,
            "nonfinite_envs": 1,
            "right_foot_airborne_envs": TRACKING_CHECKPOINT_GATE_NUM_ENVS - 1,
            "self_collision_contacts": 1,
            "simultaneous_flight_env_steps": 1,
            "source_frame_sequence_complete": False,
            "target_clip_fraction": 0.001001,
            "unexpected_termination_envs": 1,
            "xy_velocity_mae_p95_m_s": 0.075001,
        }
        check_for_measurement = {
            "forward_displacement_p05_m": "forward_displacement_direction",
            "maximum_actual_soft_limit_violation_rad": (
                "maximum_actual_soft_limit_violation_rad"
            ),
            **{
                name: name
                for name in mutations
                if name
                not in {
                    "forward_displacement_p05_m",
                    "maximum_actual_soft_limit_violation_rad",
                }
            },
        }
        for measurement, failing_value in mutations.items():
            with self.subTest(measurement=measurement):
                values = _passing_measurements()
                values[measurement] = failing_value
                checks = tracking_checkpoint_checks(values)
                self.assertFalse(checks[check_for_measurement[measurement]]["passed"])

    def test_both_passes_are_required_and_receipt_is_tamper_evident(self) -> None:
        passing = tracking_checkpoint_checks(_passing_measurements())
        report = finalize_tracking_checkpoint_report(
            {
                "passes": {
                    "nominal": {"checks": copy.deepcopy(passing)},
                    "robust": {"checks": copy.deepcopy(passing)},
                }
            }
        )
        self.assertEqual(report["status"], "pass")
        self.assertEqual(tracking_checkpoint_exit_code(report), 0)
        validate_tracking_checkpoint_receipt_digest(report)

        failing = copy.deepcopy(report)
        del failing["receipt_payload_sha256"]
        failing["passes"]["robust"]["checks"]["fall_envs"]["passed"] = False
        failing["passes"]["robust"]["checks"]["fall_envs"]["value"] = 1
        failed_report = finalize_tracking_checkpoint_report(failing)
        self.assertEqual(failed_report["status"], "fail")
        self.assertEqual(tracking_checkpoint_exit_code(failed_report), 2)
        self.assertIn("robust.fall_envs", failed_report["summary"]["failed_checks"])

        tampered = copy.deepcopy(report)
        tampered["passes"]["nominal"]["checks"]["fall_envs"]["value"] = 1
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            validate_tracking_checkpoint_receipt_digest(tampered)

    def test_missing_pass_or_invalid_measurement_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "nominal and robust"):
            finalize_tracking_checkpoint_report(
                {"passes": {"nominal": {"checks": {"x": {"passed": True}}}}}
            )

        invalid = _passing_measurements()
        invalid["minimum_root_height_m"] = float("nan")
        with self.assertRaisesRegex(ValueError, "must be finite"):
            tracking_checkpoint_checks(invalid)


if __name__ == "__main__":
    unittest.main()
