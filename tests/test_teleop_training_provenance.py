# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Focused tests for resolved-config/source-bound teleop provenance."""

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

from mjlab_microban.tasks.microban_teleop_env_cfg import (
    MicrobanTeleopRlCfg,
    make_microban_teleop_env_cfg,
)
from mjlab_microban.tasks.microban_teleop_provenance import (
    MICROBAN_TELEOP_CANONICAL_STAGE_MODE,
    canonical_json_sha256,
    collect_training_provenance,
    collect_training_source_manifest,
    inherit_initial_stage_safe_velocity_identity,
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

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def _runner_cfg(**updates: object) -> dict[str, object]:
        result: dict[str, object] = {
            "seed": 42,
            "num_steps_per_env": 24,
            "save_interval": 500,
            "max_iterations": 1500,
            "resume": False,
            "logger": "tensorboard",
            "upload_model": False,
            "safe_velocity_checkpoint": "/pinned/model_7.pt",
            "safe_velocity_checkpoint_sha256": "a" * 64,
            "safe_velocity_acceptance_receipt": "/pinned/acceptance.json",
            "save_pristine_checkpoint": False,
            "algorithm": {"learning_rate": 1.0e-4},
        }
        result.update(updates)
        return result

    @staticmethod
    def _env(*, seed: int = 42, num_envs: int = 2048) -> SimpleNamespace:
        return SimpleNamespace(
            clip_actions=None,
            unwrapped=SimpleNamespace(
                cfg=_FakeEnvCfg(seed=seed, reward_weight=1.0),
                num_envs=num_envs,
            ),
        )

    def _collect(
        self, *, runner_cfg: dict[str, object] | None = None
    ) -> tuple[dict, str]:
        return collect_training_provenance(
            self._env(),
            runner_cfg or self._runner_cfg(),
            training_contract_version="9",
            recipe_revision="recipe-test",
            actor_initialization="actor-test",
            project_root=self.root,
        )

    def test_digest_binds_resolved_config_and_exact_source_bytes(self) -> None:
        manifest, digest = self._collect()
        repeated, repeated_digest = self._collect()
        self.assertEqual(manifest, repeated)
        self.assertEqual(digest, repeated_digest)

        changed_cfg, changed_cfg_digest = self._collect(
            runner_cfg=self._runner_cfg(algorithm={"learning_rate": 2.0e-4})
        )
        self.assertNotEqual(digest, changed_cfg_digest)
        self.assertNotEqual(
            manifest["resolved_config"]["runner"],
            changed_cfg["resolved_config"]["runner"],
        )

        self.source.write_text("RECIPE = 2\n", encoding="utf-8")
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
        self.assertNotEqual(
            original["files"][relative_path], changed["files"][relative_path]
        )
        self.assertNotEqual(original["tree_sha256"], changed["tree_sha256"])

    def test_real_resolved_v9_configs_are_deterministically_serializable(self) -> None:
        env_cfg = make_microban_teleop_env_cfg(play=False)
        env_cfg.scene.num_envs = 2048
        env_cfg.seed = 42
        runner_cfg = asdict(MicrobanTeleopRlCfg)
        runner_cfg.update(
            seed=42,
            logger="tensorboard",
            upload_model=False,
            max_iterations=1500,
            save_interval=500,
        )
        env = SimpleNamespace(
            clip_actions=None,
            unwrapped=SimpleNamespace(cfg=env_cfg, num_envs=2048),
        )
        manifest, digest = collect_training_provenance(
            env,
            runner_cfg,
            training_contract_version="9",
            recipe_revision="recipe-test",
            actor_initialization="actor-test",
        )
        self.assertEqual(digest, canonical_json_sha256(manifest))
        self.assertGreater(len(manifest["resolved_config"]["environment"]), 1)

    def test_tampered_manifest_or_digest_fails_closed(self) -> None:
        manifest, digest = self._collect()
        validated = validate_training_provenance(
            manifest,
            digest,
            expected_contract_version="9",
            expected_recipe_revision="recipe-test",
            expected_actor_initialization="actor-test",
        )
        self.assertIs(validated, manifest)

        manifest["resolved_config"]["critical"]["num_envs"] = 64
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            validate_training_provenance(manifest, digest)

        forged_digest = canonical_json_sha256(manifest)
        with self.assertRaisesRegex(ValueError, "canonical v9 stage config"):
            validate_canonical_stage_critical_config(manifest)
        validate_training_provenance(manifest, forged_digest)

    def test_only_explicit_canonical_launch_can_make_canonical_manifest(self) -> None:
        generic, generic_digest = self._collect()
        self.assertFalse(generic["canonical_stage"])
        with self.assertRaisesRegex(ValueError, "canonical stage driver"):
            validate_training_provenance(
                generic, generic_digest, require_canonical_stage=True
            )

        canonical_environment = {
            "MICROBAN_TELEOP_PROVENANCE_MODE": (
                MICROBAN_TELEOP_CANONICAL_STAGE_MODE
            ),
            "MICROBAN_TELEOP_STAGE_START_BOUNDARY": "0",
            "MICROBAN_TELEOP_STAGE_TARGET_BOUNDARY": "1500",
            "MICROBAN_TELEOP_PARENT_CHECKPOINT_SHA256": "",
            "MICROBAN_TELEOP_PARENT_GATE_SHA256": "",
        }
        with patch.dict(os.environ, canonical_environment, clear=False):
            canonical, canonical_digest = self._collect()
        validate_training_provenance(
            canonical, canonical_digest, require_canonical_stage=True
        )
        validate_canonical_stage_critical_config(canonical)
        self.assertEqual(
            canonical["invocation"],
            {
                "mode": MICROBAN_TELEOP_CANONICAL_STAGE_MODE,
                "stage_start_boundary": 0,
                "stage_target_boundary": 1500,
                "parent_checkpoint_sha256": None,
                "parent_gate_sha256": None,
                "resume_source_checkpoint_path": None,
                "resume_source_checkpoint_sha256": None,
                "resume_source_checkpoint_iteration": None,
            },
        )

    def test_resumed_canonical_stage_requires_both_parent_digests(self) -> None:
        environment = {
            "MICROBAN_TELEOP_PROVENANCE_MODE": (
                MICROBAN_TELEOP_CANONICAL_STAGE_MODE
            ),
            "MICROBAN_TELEOP_STAGE_START_BOUNDARY": "1500",
            "MICROBAN_TELEOP_STAGE_TARGET_BOUNDARY": "3000",
            "MICROBAN_TELEOP_PARENT_CHECKPOINT_SHA256": "a" * 64,
            "MICROBAN_TELEOP_PARENT_GATE_SHA256": "",
        }
        with (
            patch.dict(os.environ, environment, clear=False),
            self.assertRaisesRegex(ValueError, "parent gate"),
        ):
            self._collect()

    def test_canonical_stage_cannot_skip_a_gate_boundary(self) -> None:
        environment = {
            "MICROBAN_TELEOP_PROVENANCE_MODE": (
                MICROBAN_TELEOP_CANONICAL_STAGE_MODE
            ),
            "MICROBAN_TELEOP_STAGE_START_BOUNDARY": "0",
            "MICROBAN_TELEOP_STAGE_TARGET_BOUNDARY": "3000",
            "MICROBAN_TELEOP_PARENT_CHECKPOINT_SHA256": "",
            "MICROBAN_TELEOP_PARENT_GATE_SHA256": "",
        }
        with (
            patch.dict(os.environ, environment, clear=False),
            self.assertRaisesRegex(ValueError, "adjacent v9 boundary pair"),
        ):
            self._collect()

    def test_interrupted_initial_stage_inherits_only_checkpoint_safe_identity(
        self,
    ) -> None:
        fresh_environment = {
            "MICROBAN_TELEOP_PROVENANCE_MODE": (
                MICROBAN_TELEOP_CANONICAL_STAGE_MODE
            ),
            "MICROBAN_TELEOP_STAGE_START_BOUNDARY": "0",
            "MICROBAN_TELEOP_STAGE_TARGET_BOUNDARY": "1500",
            "MICROBAN_TELEOP_PARENT_CHECKPOINT_SHA256": "",
            "MICROBAN_TELEOP_PARENT_GATE_SHA256": "",
        }
        with patch.dict(os.environ, fresh_environment, clear=False):
            loaded, _loaded_digest = self._collect()
        resume_source = self.root / "model_499.pt"
        resume_source.write_bytes(b"interrupted initial checkpoint")
        resume_source_sha256 = hashlib.sha256(resume_source.read_bytes()).hexdigest()
        resume_environment = {
            **fresh_environment,
            "MICROBAN_TELEOP_RESUME_SOURCE_CHECKPOINT_PATH": str(resume_source),
            "MICROBAN_TELEOP_RESUME_SOURCE_CHECKPOINT_SHA256": (
                resume_source_sha256
            ),
            "MICROBAN_TELEOP_RESUME_SOURCE_CHECKPOINT_ITERATION": "499",
        }
        with patch.dict(os.environ, resume_environment, clear=False):
            current, _current_digest = self._collect(
                runner_cfg=self._runner_cfg(
                    resume=True,
                    safe_velocity_checkpoint=None,
                    safe_velocity_checkpoint_sha256=None,
                    safe_velocity_acceptance_receipt=None,
                )
            )
        bootstrap_info = {
            "source_checkpoint_path": "/pinned/model_7.pt",
            "source_checkpoint_sha256": "a" * 64,
            "source_acceptance_receipt_path": "/pinned/acceptance.json",
        }

        inherited, digest = inherit_initial_stage_safe_velocity_identity(
            current, loaded, bootstrap_info
        )
        self.assertEqual(digest, canonical_json_sha256(inherited))
        self.assertEqual(
            inherited["resolved_config"]["critical"][
                "safe_velocity_checkpoint"
            ],
            "/pinned/model_7.pt",
        )
        self.assertIsNone(
            inherited["resolved_config"]["runner"][
                "safe_velocity_checkpoint"
            ]
        )
        self.assertTrue(inherited["resolved_config"]["critical"]["resume"])
        validate_canonical_stage_critical_config(inherited)

        changed = deepcopy(loaded)
        changed["resolved_config"]["critical"][
            "safe_velocity_acceptance_receipt"
        ] = "/pinned/different.json"
        with self.assertRaisesRegex(ValueError, "disagrees"):
            inherit_initial_stage_safe_velocity_identity(
                current, changed, bootstrap_info
            )

    def test_later_stage_still_rejects_resupplied_safe_source(self) -> None:
        resume_source = self.root / "model_1499.pt"
        resume_source.write_bytes(b"boundary checkpoint")
        resume_source_sha256 = hashlib.sha256(resume_source.read_bytes()).hexdigest()
        environment = {
            "MICROBAN_TELEOP_PROVENANCE_MODE": (
                MICROBAN_TELEOP_CANONICAL_STAGE_MODE
            ),
            "MICROBAN_TELEOP_STAGE_START_BOUNDARY": "1500",
            "MICROBAN_TELEOP_STAGE_TARGET_BOUNDARY": "3000",
            "MICROBAN_TELEOP_PARENT_CHECKPOINT_SHA256": resume_source_sha256,
            "MICROBAN_TELEOP_PARENT_GATE_SHA256": "c" * 64,
            "MICROBAN_TELEOP_RESUME_SOURCE_CHECKPOINT_PATH": str(resume_source),
            "MICROBAN_TELEOP_RESUME_SOURCE_CHECKPOINT_SHA256": (
                resume_source_sha256
            ),
            "MICROBAN_TELEOP_RESUME_SOURCE_CHECKPOINT_ITERATION": "1499",
        }
        with patch.dict(os.environ, environment, clear=False):
            valid, _digest = self._collect(
                runner_cfg=self._runner_cfg(
                    resume=True,
                    safe_velocity_checkpoint=None,
                    safe_velocity_checkpoint_sha256=None,
                    safe_velocity_acceptance_receipt=None,
                )
            )
            invalid, _invalid_digest = self._collect(
                runner_cfg=self._runner_cfg(resume=True)
            )
        validate_canonical_stage_critical_config(valid)
        with self.assertRaisesRegex(ValueError, "source bootstrap identity"):
            validate_canonical_stage_critical_config(invalid)


if __name__ == "__main__":
    unittest.main()
