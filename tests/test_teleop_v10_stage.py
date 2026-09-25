# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for fail-closed canonical-v10 staging and canary receipts."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mjlab_microban.scripts import evaluate_teleop_checkpoint, teleop_v10_stage
from mjlab_microban.scripts.teleop_v10_stage import (
    V10_CANARY_SCENARIOS,
    StageInterval,
    boundary_enforces_performance,
    boundary_requires_moving_hmd,
    is_v10_canary_checkpoint,
    publish_canary_receipt,
    publish_stage_gate,
    resolve_stage_interval,
    scenarios_for_boundary,
    validate_canary_receipt,
    validate_legacy_migration_source,
    validate_stage_gate,
)
from mjlab_microban.tasks.microban_teleop_provenance import sha256_file


class StagePlanTest(unittest.TestCase):
    def test_migration_and_boundaries_advance_exactly_one_stage(self) -> None:
        self.assertEqual(
            resolve_stage_interval(1500),
            StageInterval(1500, 3000, interrupted=False, migration=True),
        )
        self.assertEqual(
            resolve_stage_interval(3000),
            StageInterval(3000, 7000, interrupted=False, migration=False),
        )
        self.assertEqual(
            resolve_stage_interval(10000),
            StageInterval(10000, 15000, interrupted=False, migration=False),
        )

    def test_interrupted_checkpoint_keeps_stage_and_migration_identity(self) -> None:
        self.assertEqual(
            resolve_stage_interval(1600),
            StageInterval(1500, 3000, interrupted=True, migration=True),
        )
        self.assertEqual(
            resolve_stage_interval(6900),
            StageInterval(3000, 7000, interrupted=True, migration=False),
        )
        self.assertEqual(
            resolve_stage_interval(14900),
            StageInterval(10000, 15000, interrupted=True, migration=False),
        )

    def test_invalid_or_final_counts_are_not_resumable(self) -> None:
        for value in (0, 1499, 15000, 15001):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_stage_interval(value)
        with self.assertRaises(TypeError):
            resolve_stage_interval(True)

    def test_canary_alignment_includes_rsl_rl_periodic_recovery_saves(self) -> None:
        self.assertTrue(is_v10_canary_checkpoint(1600))
        self.assertTrue(is_v10_canary_checkpoint(1601))
        self.assertTrue(is_v10_canary_checkpoint(2901))
        self.assertFalse(is_v10_canary_checkpoint(1650))

    def test_boundary_coverage_grows_and_final_is_full_suite(self) -> None:
        stage_3000 = scenarios_for_boundary(3000)
        stage_7000 = scenarios_for_boundary(7000)
        stage_10000 = scenarios_for_boundary(10000)
        self.assertEqual(len(stage_3000 or ()), 13)
        self.assertEqual(len(stage_7000 or ()), 23)
        self.assertEqual(len(stage_10000 or ()), 25)
        self.assertTrue(set(stage_3000 or ()).issubset(stage_7000 or ()))
        self.assertTrue(set(stage_7000 or ()).issubset(stage_10000 or ()))
        self.assertIsNone(scenarios_for_boundary(15000))
        self.assertFalse(boundary_requires_moving_hmd(7000))
        self.assertTrue(boundary_requires_moving_hmd(10000))
        self.assertFalse(boundary_enforces_performance(10000))
        self.assertTrue(boundary_enforces_performance(15000))
        self.assertEqual(
            V10_CANARY_SCENARIOS,
            (
                "neutral",
                "low_forward",
                "low_yaw_left",
                "mid_yaw_left",
                "low_yaw_right",
                "mid_yaw_right",
            ),
        )


class ReceiptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.checkpoint = self.root / "model_1599.pt"
        self.checkpoint.write_bytes(b"v10 checkpoint")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _identity(self) -> tuple[Path, str, str]:
        return self.checkpoint.resolve(), sha256_file(self.checkpoint), "a" * 64

    def test_canary_receipt_is_explicitly_non_deployment_and_hash_bound(self) -> None:
        report = self.root / "canary.json"
        report.write_text("{}\n", encoding="utf-8")
        receipt_path = self.root / "run_model_1599_canary_gate.json"
        with (
            patch.object(
                teleop_v10_stage,
                "_current_checkpoint_identity",
                return_value=self._identity(),
            ),
            patch.object(teleop_v10_stage, "_validate_report"),
        ):
            receipt = publish_canary_receipt(
                receipt_path,
                self.checkpoint,
                self.root.name,
                1600,
                report,
            )
        self.assertFalse(receipt["deployment_authority"])
        self.assertEqual(
            receipt["acceptance_profile"],
            evaluate_teleop_checkpoint.CANARY_HARD_SAFETY_PROFILE,
        )
        self.assertEqual(receipt["evaluation_seeds"], [42])
        with (
            patch.object(
                teleop_v10_stage,
                "_current_checkpoint_identity",
                return_value=self._identity(),
            ),
            patch.object(teleop_v10_stage, "_validate_report") as validate_report,
        ):
            validate_canary_receipt(
                receipt_path,
                checkpoint_path=self.checkpoint,
                completed_iterations=1600,
            )
        validate_report.assert_called_once()

        changed = json.loads(receipt_path.read_text(encoding="utf-8"))
        changed["deployment_authority"] = True
        receipt_path.write_text(json.dumps(changed), encoding="utf-8")
        with (
            patch.object(
                teleop_v10_stage,
                "_current_checkpoint_identity",
                return_value=self._identity(),
            ),
            patch.object(teleop_v10_stage, "_validate_report"),
            self.assertRaisesRegex(ValueError, "deployment_authority"),
        ):
            validate_canary_receipt(
                receipt_path,
                checkpoint_path=self.checkpoint,
                completed_iterations=1600,
            )

    def test_boundary_10000_binds_three_nominal_and_three_moving_reports(self) -> None:
        self.checkpoint = self.root / "model_9999.pt"
        self.checkpoint.write_bytes(b"boundary checkpoint")
        nominal = []
        moving = []
        for seed in (42, 43, 44):
            nominal_path = self.root / f"seed_{seed}.json"
            moving_path = self.root / f"seed_{seed}_moving.json"
            nominal_path.write_text("{}\n", encoding="utf-8")
            moving_path.write_text("{}\n", encoding="utf-8")
            nominal.append(nominal_path)
            moving.append(moving_path)
        gate_path = self.root / "run_boundary_10000_gate.json"
        with (
            patch.object(
                teleop_v10_stage,
                "_current_checkpoint_identity",
                return_value=self._identity(),
            ),
            patch.object(teleop_v10_stage, "_validate_report"),
        ):
            gate = publish_stage_gate(
                gate_path,
                self.checkpoint,
                self.root.name,
                10000,
                nominal,
                moving,
            )
        self.assertNotIn("deployment_authority", gate)
        self.assertEqual(gate["evaluation_seeds"], [42, 43, 44])
        self.assertEqual(len(gate["moving_hmd_reports"]), 3)
        with (
            patch.object(
                teleop_v10_stage,
                "_current_checkpoint_identity",
                return_value=self._identity(),
            ),
            patch.object(teleop_v10_stage, "_validate_report") as validate_report,
        ):
            validate_stage_gate(
                gate_path,
                expected_boundary=10000,
                expected_checkpoint_sha256=sha256_file(self.checkpoint),
            )
        self.assertEqual(validate_report.call_count, 6)

    def test_migration_rejects_any_unpinned_checkpoint_before_loading(self) -> None:
        gate = self.root / "legacy_gate.json"
        gate.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "filename|SHA-256"):
            validate_legacy_migration_source(self.checkpoint, gate)


class WrapperContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[1]

    def _script(self, name: str) -> str:
        return (self.project_root / "scripts" / name).read_text(encoding="utf-8")

    def test_shell_scripts_parse_and_pin_training_invariants(self) -> None:
        scripts = (
            self.project_root / "scripts" / "train_microban_teleop_v10.sh",
            self.project_root / "scripts" / "evaluate_microban_teleop_v10_stage.sh",
        )
        subprocess.run(["bash", "-n", *(str(path) for path in scripts)], check=True)
        trainer = self._script("train_microban_teleop_v10.sh")
        for value in (
            "--agent.num-steps-per-env 24",
            "--agent.save-interval 100",
            "--agent.algorithm.learning-rate 0.00001",
            "--agent.algorithm.schedule fixed",
            "--env.scene.num-envs 2048",
            "--env.seed 42",
            "--agent.seed 42",
            "canonical_v10_migration_stage",
            "At most 100 updates remain; use a normal boundary run",
        ):
            with self.subTest(value=value):
                self.assertIn(value, trainer)

    def test_evaluator_has_distinct_canary_and_final_paths(self) -> None:
        script = self._script("evaluate_microban_teleop_v10_stage.sh")
        self.assertIn("--canary-hard-safety-only", script)
        self.assertIn("--intermediate-hard-safety-only", script)
        self.assertIn("moving_hmd_required=1", script)
        self.assertIn("performance_enforced=1", script)
        self.assertIn("publish_canary_receipt", script)
        self.assertIn("publish_stage_gate", script)

    def test_trainer_rejects_unsafe_output_run_suffix_before_file_access(self) -> None:
        result = subprocess.run(
            [
                "bash",
                str(self.project_root / "scripts" / "train_microban_teleop_v10.sh"),
                "migrate",
                "missing-checkpoint",
                "missing-gate",
                "--agent.run-name",
                "../escape",
            ],
            cwd=self.project_root,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("safe literal suffix", result.stderr)


if __name__ == "__main__":
    unittest.main()
