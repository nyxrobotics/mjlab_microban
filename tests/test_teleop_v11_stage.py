# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Focused tests for canonical-v11 stage planning and wrappers."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mjlab_microban.scripts import evaluate_teleop_checkpoint, teleop_v11_stage
from mjlab_microban.scripts.teleop_v11_stage import (
    V11_CANARY_SCENARIOS,
    StageInterval,
    boundary_enforces_performance,
    boundary_requires_moving_hmd,
    is_v11_canary_checkpoint,
    publish_canary_receipt,
    resolve_stage_interval,
    scenarios_for_boundary,
    validate_canary_receipt,
)
from mjlab_microban.tasks.microban_teleop_provenance import sha256_file


class StagePlanTest(unittest.TestCase):
    def test_fresh_interrupted_and_boundary_intervals(self) -> None:
        self.assertEqual(resolve_stage_interval(0), StageInterval(0, 3000, False))
        self.assertEqual(resolve_stage_interval(100), StageInterval(0, 3000, True))
        self.assertEqual(resolve_stage_interval(3000), StageInterval(3000, 7000, False))
        self.assertEqual(
            resolve_stage_interval(10000), StageInterval(10000, 15000, False)
        )
        self.assertEqual(
            resolve_stage_interval(14900), StageInterval(10000, 15000, True)
        )

    def test_invalid_and_final_counts_are_not_resumable(self) -> None:
        for value in (-1, 15000, 15001):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_stage_interval(value)
        with self.assertRaises(TypeError):
            resolve_stage_interval(True)

    def test_canary_alignment_handles_periodic_and_final_process_saves(self) -> None:
        self.assertTrue(is_v11_canary_checkpoint(100))
        self.assertTrue(is_v11_canary_checkpoint(101))
        self.assertTrue(is_v11_canary_checkpoint(2901))
        self.assertFalse(is_v11_canary_checkpoint(150))

    def test_boundary_coverage_expands_to_full_final_suite(self) -> None:
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
            V11_CANARY_SCENARIOS,
            (
                "neutral",
                "low_forward",
                "low_yaw_left",
                "mid_yaw_left",
                "low_yaw_right",
                "mid_yaw_right",
            ),
        )


class CanaryReceiptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.checkpoint = self.root / "model_99.pt"
        self.checkpoint.write_bytes(b"v11 checkpoint")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _identity(self) -> tuple[Path, str, str]:
        return self.checkpoint.resolve(), sha256_file(self.checkpoint), "a" * 64

    def test_receipt_is_non_deployment_v11_and_hash_bound(self) -> None:
        report = self.root / "canary.json"
        report.write_text("{}\n", encoding="utf-8")
        receipt_path = self.root / "run_model_99_canary_gate.json"
        with (
            patch.object(
                teleop_v11_stage,
                "_current_checkpoint_identity",
                return_value=self._identity(),
            ),
            patch.object(teleop_v11_stage.receipt_primitives, "_validate_report"),
        ):
            receipt = publish_canary_receipt(
                receipt_path,
                self.checkpoint,
                self.root.name,
                100,
                report,
            )
        self.assertEqual(receipt["receipt_kind"], "canonical_v11_interrupted_canary")
        self.assertFalse(receipt["deployment_authority"])
        self.assertEqual(
            receipt["acceptance_profile"],
            evaluate_teleop_checkpoint.CANARY_HARD_SAFETY_PROFILE,
        )
        with (
            patch.object(
                teleop_v11_stage,
                "_current_checkpoint_identity",
                return_value=self._identity(),
            ),
            patch.object(
                teleop_v11_stage.receipt_primitives, "_validate_report"
            ) as validate_report,
        ):
            validate_canary_receipt(
                receipt_path,
                checkpoint_path=self.checkpoint,
                completed_iterations=100,
            )
        validate_report.assert_called_once()

        changed = json.loads(receipt_path.read_text(encoding="utf-8"))
        changed["deployment_authority"] = True
        receipt_path.write_text(json.dumps(changed), encoding="utf-8")
        with (
            patch.object(
                teleop_v11_stage,
                "_current_checkpoint_identity",
                return_value=self._identity(),
            ),
            patch.object(teleop_v11_stage.receipt_primitives, "_validate_report"),
            self.assertRaisesRegex(ValueError, "deployment_authority"),
        ):
            validate_canary_receipt(
                receipt_path,
                checkpoint_path=self.checkpoint,
                completed_iterations=100,
            )


class WrapperContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[1]

    def _script(self, name: str) -> str:
        return (self.project_root / "scripts" / name).read_text(encoding="utf-8")

    def test_shell_scripts_parse_and_pin_training_invariants(self) -> None:
        scripts = (
            self.project_root / "scripts" / "train_microban_teleop_v11.sh",
            self.project_root / "scripts" / "evaluate_microban_teleop_v11_stage.sh",
        )
        subprocess.run(["bash", "-n", *(str(path) for path in scripts)], check=True)
        trainer = self._script("train_microban_teleop_v11.sh")
        for value in (
            "start [--canary]",
            "--agent.num-steps-per-env 24",
            "--agent.save-interval 100",
            "--agent.algorithm.learning-rate 0.0001",
            "--agent.algorithm.schedule fixed",
            "--agent.algorithm.entropy-coef 0.005",
            "--agent.algorithm.num-learning-epochs 5",
            "--agent.save-pristine-checkpoint True",
            "--env.scene.num-envs 2048",
            "--env.seed 42",
            "--agent.seed 42",
            "canonical_v11_stage",
            "416a8b16f7f7980822e4e1df81ffaf9515bc18a246e6fc257405a2c46ceece93",
            "e68701b11774dd30c8e45a2fd89614a2e4423a9486d01a0d936f0fa6fb760492",
        ):
            with self.subTest(value=value):
                self.assertIn(value, trainer)

    def test_evaluator_has_canary_intermediate_and_final_paths(self) -> None:
        script = self._script("evaluate_microban_teleop_v11_stage.sh")
        self.assertIn("--canary-hard-safety-only", script)
        self.assertIn("--intermediate-hard-safety-only", script)
        self.assertIn("moving_hmd_required=1", script)
        self.assertIn("performance_enforced=1", script)
        self.assertIn("publish_canary_receipt", script)
        self.assertIn("publish_stage_gate", script)

    def test_trainer_rejects_unsafe_output_suffix_before_bootstrap_access(self) -> None:
        result = subprocess.run(
            [
                "bash",
                str(self.project_root / "scripts" / "train_microban_teleop_v11.sh"),
                "start",
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
