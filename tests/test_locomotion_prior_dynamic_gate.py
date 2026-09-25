# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""CPU-only contract tests for the GPU locomotion-prior dynamics gate."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from mjlab_microban.locomotion_prior_dynamic_gate import (
    DYNAMIC_GATE_NUM_ENVS,
    dynamic_gate_checks,
    finalize_dynamic_report,
    load_and_validate_static_receipt,
    summarize_first_exit_steps,
    validate_dynamic_receipt_digest,
)
from mjlab_microban.locomotion_prior_suitability import (
    LOCOMOTION_PRIOR_SUITABILITY_SCHEMA_VERSION,
    canonical_json_sha256,
    sha256_file,
)

PRIOR_SHA256 = "1" * 64
ROBOT_SHA256 = "2" * 64


def _passing_measurements() -> dict[str, float | int]:
    return {
        "executed_policy_steps": 159,
        "fall_envs": 0,
        "forward_velocity_p05_m_s": 0.05,
        "full_clip_completed_envs": DYNAMIC_GATE_NUM_ENVS,
        "left_foot_airborne_envs": DYNAMIC_GATE_NUM_ENVS,
        "maximum_soft_limit_violation_rad": 1.0e-6,
        "maximum_target_projection_rad": 0.001,
        "minimum_root_height_m": 0.10,
        "nonfinite_envs": 0,
        "right_foot_airborne_envs": DYNAMIC_GATE_NUM_ENVS,
        "self_collision_contacts": 0,
        "simultaneous_flight_env_steps": 0,
        "source_frame_sequence_complete": True,
        "unexpected_termination_envs": 0,
        "xy_velocity_mae_p95_m_s": 0.075,
    }


def _static_receipt(*, passed: bool = True) -> dict:
    payload = {
        "configuration": {
            "end_frame_inclusive": 267,
            "start_frame_inclusive": 109,
        },
        "inputs": {
            "locomotion_prior": {
                "provenance": {"retarget_model_sha256": ROBOT_SHA256},
                "sha256": PRIOR_SHA256,
            },
            "robot_xml": {"sha256": ROBOT_SHA256},
        },
        "schema_version": LOCOMOTION_PRIOR_SUITABILITY_SCHEMA_VERSION,
        "status": "pass" if passed else "fail",
        "summary": {"passed": passed},
    }
    return {**payload, "receipt_payload_sha256": canonical_json_sha256(payload)}


class DynamicCheckContractTest(unittest.TestCase):
    def test_exact_acceptance_boundaries_pass(self) -> None:
        checks = dynamic_gate_checks(_passing_measurements())

        self.assertTrue(all(check["passed"] for check in checks.values()))

    def test_every_required_failure_is_fail_closed(self) -> None:
        mutations = {
            "executed_policy_steps": 158,
            "fall_envs": 1,
            "forward_velocity_p05_m_s": 0.049999,
            "full_clip_completed_envs": DYNAMIC_GATE_NUM_ENVS - 1,
            "left_foot_airborne_envs": DYNAMIC_GATE_NUM_ENVS - 1,
            "maximum_soft_limit_violation_rad": 1.0001e-6,
            "maximum_target_projection_rad": 0.001001,
            "minimum_root_height_m": 0.09999,
            "nonfinite_envs": 1,
            "right_foot_airborne_envs": DYNAMIC_GATE_NUM_ENVS - 1,
            "self_collision_contacts": 1,
            "simultaneous_flight_env_steps": 1,
            "source_frame_sequence_complete": False,
            "unexpected_termination_envs": 1,
            "xy_velocity_mae_p95_m_s": 0.075001,
        }
        expected_check = {
            "executed_policy_steps": "executed_policy_steps",
            "fall_envs": "fall_envs",
            "forward_velocity_p05_m_s": "forward_velocity_p05_m_s",
            "full_clip_completed_envs": "full_clip_completed_envs",
            "left_foot_airborne_envs": "left_foot_airborne_envs",
            "maximum_soft_limit_violation_rad": ("maximum_soft_limit_violation_rad"),
            "maximum_target_projection_rad": "maximum_target_projection_rad",
            "minimum_root_height_m": "minimum_root_height_m",
            "nonfinite_envs": "nonfinite_envs",
            "right_foot_airborne_envs": "right_foot_airborne_envs",
            "self_collision_contacts": "self_collision_contacts",
            "simultaneous_flight_env_steps": "simultaneous_flight_env_steps",
            "source_frame_sequence_complete": "source_frame_sequence_complete",
            "unexpected_termination_envs": "unexpected_termination_envs",
            "xy_velocity_mae_p95_m_s": "xy_velocity_mae_p95_m_s",
        }
        for measurement, failing_value in mutations.items():
            with self.subTest(measurement=measurement):
                values = _passing_measurements()
                values[measurement] = failing_value
                checks = dynamic_gate_checks(values)
                self.assertFalse(checks[expected_check[measurement]]["passed"])

    def test_final_receipt_is_tamper_evident(self) -> None:
        report = finalize_dynamic_report(
            {"checks": dynamic_gate_checks(_passing_measurements())}
        )
        self.assertEqual(report["status"], "pass")
        validate_dynamic_receipt_digest(report)

        tampered = copy.deepcopy(report)
        tampered["checks"]["fall_envs"]["value"] = 1
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            validate_dynamic_receipt_digest(tampered)

    def test_staggered_first_exit_summary_is_explicit_and_deterministic(self) -> None:
        self.assertEqual(
            summarize_first_exit_steps([-1, 5, 3, 5], expected_envs=4),
            {
                "counts_by_step": {"3": 1, "5": 2},
                "earliest_zero_based": 3,
                "exited_envs": 3,
                "latest_zero_based": 5,
                "not_exited_envs": 1,
            },
        )

    def test_first_exit_summary_rejects_invalid_cardinality_and_steps(self) -> None:
        with self.assertRaisesRegex(ValueError, "Expected 2"):
            summarize_first_exit_steps([1], expected_envs=2)
        with self.assertRaisesRegex(ValueError, "greater than or equal to -1"):
            summarize_first_exit_steps([-2], expected_envs=1)


class StaticPrerequisiteTest(unittest.TestCase):
    def _write(self, directory: str, report: dict) -> Path:
        path = Path(directory) / "static.json"
        path.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def test_passing_receipt_is_bound_to_both_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, _static_receipt())
            report, receipt_sha256 = load_and_validate_static_receipt(
                path,
                expected_prior_sha256=PRIOR_SHA256,
                expected_robot_xml_sha256=ROBOT_SHA256,
            )
            self.assertEqual(report["status"], "pass")
            self.assertEqual(receipt_sha256, sha256_file(path))

    def test_failed_or_tampered_static_receipt_cannot_start_dynamics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            failed_path = self._write(directory, _static_receipt(passed=False))
            with self.assertRaisesRegex(ValueError, "did not pass"):
                load_and_validate_static_receipt(
                    failed_path,
                    expected_prior_sha256=PRIOR_SHA256,
                    expected_robot_xml_sha256=ROBOT_SHA256,
                )

            tampered = _static_receipt()
            tampered["status"] = "fail"
            tampered_path = self._write(directory, tampered)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_and_validate_static_receipt(
                    tampered_path,
                    expected_prior_sha256=PRIOR_SHA256,
                    expected_robot_xml_sha256=ROBOT_SHA256,
                )

    def test_input_digest_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, _static_receipt())
            with self.assertRaisesRegex(ValueError, "locomotion-prior SHA-256"):
                load_and_validate_static_receipt(
                    path,
                    expected_prior_sha256="3" * 64,
                    expected_robot_xml_sha256=ROBOT_SHA256,
                )


if __name__ == "__main__":
    unittest.main()
