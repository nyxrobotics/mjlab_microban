# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for fail-closed canonical-v9 stage continuation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from mjlab_microban.scripts import evaluate_teleop_checkpoint, teleop_v8_stage
from mjlab_microban.scripts.teleop_v8_stage import (
    StageInterval,
    resolve_stage_interval,
    validate_interrupted_stage_checkpoint,
)
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION,
)
from mjlab_microban.tasks.microban_teleop_provenance import (
    MICROBAN_TELEOP_TRAINING_PROVENANCE_KEY,
    MICROBAN_TELEOP_TRAINING_PROVENANCE_SHA256_KEY,
    sha256_file,
)


class StageIntervalTest(unittest.TestCase):
    def test_boundary_advances_one_stage(self) -> None:
        self.assertEqual(
            resolve_stage_interval(1500),
            StageInterval(1500, 3000, interrupted=False),
        )
        self.assertEqual(
            resolve_stage_interval(18000),
            StageInterval(18000, 20000, interrupted=False),
        )

    def test_interrupted_checkpoint_keeps_existing_stage_target(self) -> None:
        self.assertEqual(
            resolve_stage_interval(500),
            StageInterval(0, 1500, interrupted=True),
        )
        self.assertEqual(
            resolve_stage_interval(1501),
            StageInterval(1500, 3000, interrupted=True),
        )
        self.assertEqual(
            resolve_stage_interval(11999),
            StageInterval(8000, 12000, interrupted=True),
        )

    def test_final_or_invalid_count_is_not_resumable(self) -> None:
        for value in (0, 20000, 20001):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_stage_interval(value)


class StageWrapperReceiptContractTest(unittest.TestCase):
    @staticmethod
    def _script(name: str) -> str:
        return (Path(__file__).resolve().parents[1] / "scripts" / name).read_text(
            encoding="utf-8"
        )

    def test_force_invalidates_pass_receipt_before_any_evaluator_run(self) -> None:
        script = self._script("evaluate_microban_teleop_v8_stage.sh")
        invalidation = script.index('mv -- "${GATE_PATH}" "${INVALIDATED_GATE_PATH}"')
        evaluation_loop = script.index('for seed in "${EVALUATION_SEEDS[@]}"')
        self.assertLess(invalidation, evaluation_loop)
        self.assertIn('"schema_version": 3', script)

    def test_receipt_and_resume_bind_reports_evaluator_and_recipe(self) -> None:
        evaluator = self._script("evaluate_microban_teleop_v8_stage.sh")
        trainer = self._script("train_microban_teleop_v9.sh")
        required_fields = (
            "training_provenance_sha256",
            "recipe_revision",
            "evaluator_revision",
            "acceptance_revision",
            "acceptance_profile",
            "evaluator_source_sha256",
            "report_sha256",
            "moving_hmd_report_sha256",
        )
        for field in required_fields:
            with self.subTest(script="evaluator", field=field):
                self.assertIn(field, evaluator)
            with self.subTest(script="trainer", field=field):
                self.assertIn(field, trainer)


class InterruptedStageProvenanceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.gate_root = self.root / "gates"
        self.gate_root.mkdir()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _checkpoint(self, *, invocation: dict[str, object]) -> Path:
        checkpoint = self.root / "model_1999.pt"
        torch.save(
            {
                "infos": {
                    MICROBAN_TELEOP_TRAINING_PROVENANCE_KEY: {
                        "invocation": invocation
                    },
                    MICROBAN_TELEOP_TRAINING_PROVENANCE_SHA256_KEY: "a" * 64,
                }
            },
            checkpoint,
        )
        return checkpoint

    def _validate(
        self,
        checkpoint: Path,
        interval: StageInterval,
        *,
        completed_iterations: int = 2000,
    ) -> dict:
        manifest = torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )["infos"][MICROBAN_TELEOP_TRAINING_PROVENANCE_KEY]
        with (
            patch.object(
                teleop_v8_stage,
                "validate_teleop_checkpoint_contract",
                return_value=SimpleNamespace(iteration=completed_iterations - 1),
            ),
            patch.object(
                teleop_v8_stage,
                "validate_training_provenance",
                return_value=manifest,
            ),
            patch.object(
                teleop_v8_stage, "validate_canonical_stage_critical_config"
            ),
        ):
            return validate_interrupted_stage_checkpoint(
                checkpoint,
                completed_iterations=completed_iterations,
                interval=interval,
                gate_root=self.gate_root,
            )

    def test_initial_stage_requires_null_parents(self) -> None:
        checkpoint = self._checkpoint(
            invocation={
                "stage_start_boundary": 0,
                "stage_target_boundary": 1500,
                "parent_checkpoint_sha256": None,
                "parent_gate_sha256": None,
            }
        )
        result = self._validate(
            checkpoint,
            StageInterval(0, 1500, interrupted=True),
            completed_iterations=500,
        )
        self.assertIsNone(result["parent_gate"])

    def test_resumed_stage_requires_exact_pinned_parent_gate(self) -> None:
        parent_checkpoint = self.root / "parent_model_1499.pt"
        parent_checkpoint.write_bytes(b"pinned parent checkpoint")
        parent_checkpoint_sha256 = sha256_file(parent_checkpoint)
        gate = self.gate_root / "parent_boundary_1500_gate.json"
        gate.write_text(
            json.dumps(
                {
                    "status": "pass",
                    "training_contract_version": (
                        MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION
                    ),
                    "completed_iterations": 1500,
                    "evaluator_revision": (
                        evaluate_teleop_checkpoint.TELEOP_EVALUATOR_REVISION
                    ),
                    "acceptance_revision": (
                        evaluate_teleop_checkpoint.TELEOP_ACCEPTANCE_REVISION
                    ),
                    "acceptance_profile": (
                        evaluate_teleop_checkpoint.INTERMEDIATE_HARD_SAFETY_PROFILE
                    ),
                    "checkpoint": str(parent_checkpoint),
                    "checkpoint_sha256": parent_checkpoint_sha256,
                }
            ),
            encoding="utf-8",
        )
        parent_gate_sha256 = sha256_file(gate)
        checkpoint = self._checkpoint(
            invocation={
                "stage_start_boundary": 1500,
                "stage_target_boundary": 3000,
                "parent_checkpoint_sha256": parent_checkpoint_sha256,
                "parent_gate_sha256": parent_gate_sha256,
            }
        )
        result = self._validate(
            checkpoint, StageInterval(1500, 3000, interrupted=True)
        )
        self.assertEqual(result["parent_gate"], str(gate.resolve()))

        gate.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not found uniquely"):
            self._validate(
                checkpoint, StageInterval(1500, 3000, interrupted=True)
            )

    def test_rejects_changed_target(self) -> None:
        checkpoint = self._checkpoint(
            invocation={
                "stage_start_boundary": 0,
                "stage_target_boundary": 1500,
                "parent_checkpoint_sha256": None,
                "parent_gate_sha256": None,
            }
        )
        with self.assertRaisesRegex(ValueError, "stage-target"):
            self._validate(
                checkpoint, StageInterval(0, 3000, interrupted=True)
            )


if __name__ == "__main__":
    unittest.main()
