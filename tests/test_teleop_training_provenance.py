# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");

"""Focused tests for schema-2, full-state Microban teleop provenance."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import unittest
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_TELEOP_ACTOR_INITIALIZATION,
    MICROBAN_TELEOP_RECIPE_REVISION,
    MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION,
)
from mjlab_microban.tasks.microban_teleop_env_cfg import (
    MicrobanTeleopRlCfg,
    make_microban_teleop_env_cfg,
)
from mjlab_microban.tasks.microban_teleop_provenance import (
    MICROBAN_TELEOP_CANONICAL_MIGRATION_STAGE_MODE,
    MICROBAN_TELEOP_CANONICAL_STAGE_MODE,
    MICROBAN_TELEOP_TRAINING_PROVENANCE_SCHEMA_VERSION,
    MICROBAN_TELEOP_V11_ENTROPY_COEF,
    MICROBAN_TELEOP_V11_FIXED_LEARNING_RATE,
    MICROBAN_TELEOP_V11_NUM_LEARNING_EPOCHS,
    canonical_json_sha256,
    collect_training_provenance,
    collect_training_source_manifest,
    validate_canonical_stage_critical_config,
    validate_training_provenance,
)


@dataclass
class _FakeEnvCfg:
    seed: int
    reward_weight: float


class TrainingProvenanceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        source = self.root / "src/mjlab_microban/task.py"
        source.parent.mkdir(parents=True)
        source.write_text("RECIPE = 1\n", encoding="utf-8")
        self.source = source
        repository = Path(__file__).resolve().parents[1]
        self.legacy_checkpoint = (
            repository
            / "logs/rsl_rl/mjlab_microban_teleop"
            / "2026-09-25_10-58-50_teleop_v9_safe_v3_v92r_stage0_1500"
            / "model_1499.pt"
        ).resolve()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def _runner_cfg(**updates: object) -> dict[str, object]:
        result: dict[str, object] = {
            "seed": 42,
            "num_steps_per_env": 24,
            "save_interval": 100,
            "max_iterations": 100,
            "resume": False,
            "logger": "tensorboard",
            "upload_model": False,
            "checkpoint_consumer_mode": False,
            "safe_velocity_checkpoint": None,
            "safe_velocity_checkpoint_sha256": None,
            "safe_velocity_acceptance_receipt": None,
            "save_pristine_checkpoint": False,
            "algorithm": {
                "learning_rate": MICROBAN_TELEOP_V11_FIXED_LEARNING_RATE,
                "schedule": "fixed",
                "entropy_coef": MICROBAN_TELEOP_V11_ENTROPY_COEF,
                "num_learning_epochs": MICROBAN_TELEOP_V11_NUM_LEARNING_EPOCHS,
            },
        }
        result.update(updates)
        return result

    @staticmethod
    def _env(*, seed: int = 42, num_envs: int = 2048) -> SimpleNamespace:
        return SimpleNamespace(
            clip_actions=None,
            unwrapped=SimpleNamespace(
                cfg=_FakeEnvCfg(seed=seed, reward_weight=1.0), num_envs=num_envs
            ),
        )

    def _collect(
        self, *, runner_cfg: dict[str, object] | None = None
    ) -> tuple[dict, str]:
        return collect_training_provenance(
            self._env(),
            runner_cfg or self._runner_cfg(),
            training_contract_version=MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION,
            recipe_revision=MICROBAN_TELEOP_RECIPE_REVISION,
            actor_initialization=MICROBAN_TELEOP_ACTOR_INITIALIZATION,
            project_root=self.root,
        )

    def _migration_environment(self) -> dict[str, str]:
        return {
            "MICROBAN_TELEOP_PROVENANCE_MODE": (
                MICROBAN_TELEOP_CANONICAL_MIGRATION_STAGE_MODE
            ),
            "MICROBAN_TELEOP_STAGE_START_BOUNDARY": "1500",
            "MICROBAN_TELEOP_STAGE_TARGET_BOUNDARY": "3000",
            "MICROBAN_TELEOP_PARENT_CHECKPOINT_SHA256": "a" * 64,
            "MICROBAN_TELEOP_PARENT_GATE_SHA256": "b" * 64,
            "MICROBAN_TELEOP_RESUME_SOURCE_CHECKPOINT_PATH": str(
                self.legacy_checkpoint
            ),
            "MICROBAN_TELEOP_RESUME_SOURCE_CHECKPOINT_SHA256": "c" * 64,
            "MICROBAN_TELEOP_RESUME_SOURCE_CHECKPOINT_ITERATION": "1499",
        }

    def _migration_manifest(self) -> tuple[dict, str]:
        with patch.dict(os.environ, self._migration_environment(), clear=True):
            return self._collect()

    @staticmethod
    def _fresh_v11_environment() -> dict[str, str]:
        return {
            "MICROBAN_TELEOP_PROVENANCE_MODE": MICROBAN_TELEOP_CANONICAL_STAGE_MODE,
            "MICROBAN_TELEOP_STAGE_START_BOUNDARY": "0",
            "MICROBAN_TELEOP_STAGE_TARGET_BOUNDARY": "3000",
        }

    def _fresh_v11_manifest(self) -> tuple[dict, str]:
        runner = self._runner_cfg(
            resume=False,
            safe_velocity_checkpoint="/safe/model_7.pt",
            safe_velocity_checkpoint_sha256="e" * 64,
            safe_velocity_acceptance_receipt="/safe/receipt.json",
            save_pristine_checkpoint=True,
        )
        with patch.dict(os.environ, self._fresh_v11_environment(), clear=True):
            return self._collect(runner_cfg=runner)

    def test_digest_binds_resolved_config_and_exact_source_bytes(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            manifest, digest = self._collect()
            repeated, repeated_digest = self._collect()
        self.assertEqual(manifest, repeated)
        self.assertEqual(digest, repeated_digest)
        with patch.dict(os.environ, {}, clear=True):
            changed_cfg, changed_cfg_digest = self._collect(
                runner_cfg=self._runner_cfg(
                    algorithm={"learning_rate": 2.0e-5, "schedule": "fixed"}
                )
            )
        self.assertNotEqual(digest, changed_cfg_digest)
        self.assertNotEqual(
            manifest["resolved_config"]["runner"],
            changed_cfg["resolved_config"]["runner"],
        )
        self.source.write_text("RECIPE = 2\n", encoding="utf-8")
        with patch.dict(os.environ, {}, clear=True):
            changed_source, changed_source_digest = self._collect()
        self.assertNotEqual(digest, changed_source_digest)
        self.assertNotEqual(
            manifest["source"]["tree_sha256"],
            changed_source["source"]["tree_sha256"],
        )

    def test_source_manifest_binds_xc330_identification_json_bytes(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        copied_root = self.root / "copied-project"
        shutil.copytree(project_root / "src", copied_root / "src")
        relative_path = "src/mjlab_microban/robot/xc330_params.json"
        copied_json = copied_root / relative_path
        original = collect_training_source_manifest(copied_root)
        self.assertIn(relative_path, original["files"])
        self.assertEqual(
            original["files"][relative_path],
            hashlib.sha256(copied_json.read_bytes()).hexdigest(),
        )
        copied_json.write_bytes(copied_json.read_bytes() + b"\n")
        changed = collect_training_source_manifest(copied_root)
        self.assertNotEqual(original["tree_sha256"], changed["tree_sha256"])

    def test_real_resolved_v11_configs_are_deterministically_serializable(self) -> None:
        env_cfg = make_microban_teleop_env_cfg(play=False)
        env_cfg.scene.num_envs = 2048
        env_cfg.seed = 42
        runner_cfg = asdict(MicrobanTeleopRlCfg)
        runner_cfg.update(
            seed=42,
            logger="tensorboard",
            upload_model=False,
            resume=True,
            max_iterations=100,
            save_interval=100,
        )
        env = SimpleNamespace(
            clip_actions=None,
            unwrapped=SimpleNamespace(cfg=env_cfg, num_envs=2048),
        )
        with patch.dict(os.environ, {}, clear=True):
            manifest, digest = collect_training_provenance(
                env,
                runner_cfg,
                training_contract_version=MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION,
                recipe_revision=MICROBAN_TELEOP_RECIPE_REVISION,
                actor_initialization=MICROBAN_TELEOP_ACTOR_INITIALIZATION,
            )
        self.assertEqual(digest, canonical_json_sha256(manifest))
        self.assertEqual(
            manifest["schema_version"],
            MICROBAN_TELEOP_TRAINING_PROVENANCE_SCHEMA_VERSION,
        )

    def test_retired_v10_migration_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "canonical_v11_stage"):
            self._migration_manifest()

    def test_fresh_v11_stage_rejects_parent_and_resume_source(self) -> None:
        wrong_parent = self._fresh_v11_environment()
        wrong_parent["MICROBAN_TELEOP_PARENT_GATE_SHA256"] = "a" * 64
        with (
            patch.dict(os.environ, wrong_parent, clear=True),
            self.assertRaisesRegex(ValueError, "fresh stage must have null parents"),
        ):
            self._collect()

        resume_source = self.root / "model_0.pt"
        resume_source.write_bytes(b"not valid for a fresh launch")
        resume_sha = hashlib.sha256(resume_source.read_bytes()).hexdigest()
        wrong_source = self._fresh_v11_environment()
        wrong_source.update(
            MICROBAN_TELEOP_RESUME_SOURCE_CHECKPOINT_PATH=str(resume_source),
            MICROBAN_TELEOP_RESUME_SOURCE_CHECKPOINT_SHA256=resume_sha,
            MICROBAN_TELEOP_RESUME_SOURCE_CHECKPOINT_ITERATION="0",
        )
        with (
            patch.dict(os.environ, wrong_source, clear=True),
            self.assertRaisesRegex(ValueError, "fresh stage cannot claim a resume source"),
        ):
            self._collect()

    def test_normal_v11_stage_accepts_exact_100_update_canary(self) -> None:
        resume_source = self.root / "model_2999.pt"
        resume_source.write_bytes(b"contract-v11 boundary")
        resume_sha = hashlib.sha256(resume_source.read_bytes()).hexdigest()
        environment = {
            "MICROBAN_TELEOP_PROVENANCE_MODE": MICROBAN_TELEOP_CANONICAL_STAGE_MODE,
            "MICROBAN_TELEOP_STAGE_START_BOUNDARY": "3000",
            "MICROBAN_TELEOP_STAGE_TARGET_BOUNDARY": "7000",
            "MICROBAN_TELEOP_PARENT_CHECKPOINT_SHA256": resume_sha,
            "MICROBAN_TELEOP_PARENT_GATE_SHA256": "c" * 64,
            "MICROBAN_TELEOP_RESUME_SOURCE_CHECKPOINT_PATH": str(resume_source),
            "MICROBAN_TELEOP_RESUME_SOURCE_CHECKPOINT_SHA256": resume_sha,
            "MICROBAN_TELEOP_RESUME_SOURCE_CHECKPOINT_ITERATION": "2999",
        }
        runner = self._runner_cfg(resume=True, max_iterations=100)
        with patch.dict(os.environ, environment, clear=True):
            current, digest = self._collect(runner_cfg=runner)
        validate_training_provenance(
            current,
            digest,
            require_canonical_stage=True,
            expected_contract_version="11",
        )
        self.assertIsNone(current["migration_source"])
        validate_canonical_stage_critical_config(current)
        tampered = deepcopy(current)
        tampered["migration_source"] = {"retired": True}
        with self.assertRaisesRegex(ValueError, "retired v10 migration_source"):
            validate_canonical_stage_critical_config(tampered)

    def test_v11_stages_cannot_skip_boundaries(self) -> None:
        resume_source = self.root / "model_2999.pt"
        resume_source.write_bytes(b"boundary")
        resume_sha = hashlib.sha256(resume_source.read_bytes()).hexdigest()
        environment = {
            "MICROBAN_TELEOP_PROVENANCE_MODE": MICROBAN_TELEOP_CANONICAL_STAGE_MODE,
            "MICROBAN_TELEOP_STAGE_START_BOUNDARY": "3000",
            "MICROBAN_TELEOP_STAGE_TARGET_BOUNDARY": "10000",
            "MICROBAN_TELEOP_PARENT_CHECKPOINT_SHA256": resume_sha,
            "MICROBAN_TELEOP_PARENT_GATE_SHA256": "d" * 64,
            "MICROBAN_TELEOP_RESUME_SOURCE_CHECKPOINT_PATH": str(resume_source),
            "MICROBAN_TELEOP_RESUME_SOURCE_CHECKPOINT_SHA256": resume_sha,
            "MICROBAN_TELEOP_RESUME_SOURCE_CHECKPOINT_ITERATION": "2999",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            self.assertRaisesRegex(ValueError, "adjacent contract-v11 boundary"),
        ):
            self._collect()

    def test_fixed_lr_save_interval_and_stage_limit_fail_closed(self) -> None:
        manifest, _ = self._fresh_v11_manifest()
        for field, value, pattern in (
            ("save_interval", 500, "save_interval"),
            (
                "max_iterations_for_process",
                500,
                "complete initial stage or the canonical 100-update canary",
            ),
        ):
            with self.subTest(field=field):
                changed = deepcopy(manifest)
                changed["resolved_config"]["critical"][field] = value
                with self.assertRaisesRegex(ValueError, pattern):
                    validate_canonical_stage_critical_config(changed)
        changed = deepcopy(manifest)
        changed["resolved_config"]["runner"]["algorithm"]["learning_rate"] = 2e-5
        with self.assertRaisesRegex(ValueError, "learning_rate"):
            validate_canonical_stage_critical_config(changed)

    def test_tampered_manifest_or_digest_fails_closed(self) -> None:
        manifest, digest = self._fresh_v11_manifest()
        manifest["resolved_config"]["critical"]["num_envs"] = 64
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            validate_training_provenance(manifest, digest)
        with self.assertRaisesRegex(ValueError, "canonical v11 stage config"):
            validate_canonical_stage_critical_config(manifest)


if __name__ == "__main__":
    unittest.main()
