"""CPU-only tests for the pinned model9200 L/R recovery route."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch

from mjlab_microban.scripts.migrate_teleop_v12_lr_order import migrate_checkpoint
from mjlab_microban.scripts.teleop_v12_lr_recovery import (
    EXPECTED_ACTIVE_ACTOR_COLUMNS,
    PINNED_SOURCE_COMMON_STEP_COUNTER,
    PINNED_SOURCE_ITERATION,
    RECOVERY_PROCESS_UPDATES,
    create_recovery_receipt,
    validate_recovery_receipt,
    validate_recovery_seed,
)
from mjlab_microban.tasks.microban_teleop_v12_actor import (
    TELEOP_V12_FOOT_OBSERVATION_COLUMNS,
)
from mjlab_microban.tasks.microban_teleop_v12_bootstrap import sha256_file
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MICROBAN_TELEOP_V12_TRAINING_CONTRACT_VERSION,
)

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/train_microban_teleop_v12_lr_recovery.sh"


def _normalizer(width: int) -> dict[str, torch.Tensor]:
    return {
        "obs_normalizer._mean": torch.zeros(1, width),
        "obs_normalizer._var": torch.ones(1, width),
        "obs_normalizer._std": torch.ones(1, width),
        "obs_normalizer.count": torch.tensor(10.0),
    }


def _source_payload(*, iteration: int = PINNED_SOURCE_ITERATION) -> dict:
    generator = torch.Generator().manual_seed(9200)
    actor_w0 = torch.randn(512, 83, generator=generator)
    actor_w0[:, TELEOP_V12_FOOT_OBSERVATION_COLUMNS] = 0.0
    critic_w0 = torch.randn(512, 137, generator=generator)
    actor_state = {
        **_normalizer(83),
        "mlp.0.weight": actor_w0,
        "mlp.0.bias": torch.randn(512, generator=generator),
    }
    critic_state = {
        **_normalizer(137),
        "mlp.0.weight": critic_w0,
        "mlp.0.bias": torch.randn(512, generator=generator),
    }
    actor_first = torch.randn(512, 83, generator=generator) * 0.01
    actor_second = torch.rand(512, 83, generator=generator) * 0.001
    actor_first[:, TELEOP_V12_FOOT_OBSERVATION_COLUMNS] = 0.0
    actor_second[:, TELEOP_V12_FOOT_OBSERVATION_COLUMNS] = 0.0
    return {
        "actor_state_dict": actor_state,
        "critic_state_dict": critic_state,
        "optimizer_state_dict": {
            "state": {
                1: {
                    "step": torch.tensor(9201.0),
                    "exp_avg": actor_first,
                    "exp_avg_sq": actor_second,
                },
                9: {
                    "step": torch.tensor(9201.0),
                    "exp_avg": torch.randn(512, 137, generator=generator) * 0.01,
                    "exp_avg_sq": torch.rand(512, 137, generator=generator) * 0.001,
                },
            },
            "param_groups": [{"params": [1, 9], "lr": 1.0e-4}],
        },
        "iter": iteration,
        "infos": {
            "microban_teleop_training_contract_version": (
                MICROBAN_TELEOP_V12_TRAINING_CONTRACT_VERSION
            ),
            "env_state": {
                "common_step_counter": (iteration + 1) * 24,
            },
            "active_actor_columns_at_save": list(EXPECTED_ACTIVE_ACTOR_COLUMNS),
        },
    }


class TeleopV12LrRecoveryTest(unittest.TestCase):
    def _prepare(
        self, directory: Path, *, iteration: int = PINNED_SOURCE_ITERATION
    ) -> tuple[Path, Path, Path, str]:
        source = directory / "raw_model_9200.pt"
        checkpoint = directory / "model_9200.pt"
        migration_receipt = directory / "migration.json"
        torch.save(_source_payload(iteration=iteration), source)
        source_sha256 = sha256_file(source)
        migrate_checkpoint(
            source=source,
            expected_sha256=source_sha256,
            output=checkpoint,
            receipt=migration_receipt,
            strategy="swap",
            force=False,
        )
        return source, checkpoint, migration_receipt, source_sha256

    def test_complete_route_receipt_is_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, checkpoint, migration, digest = self._prepare(root)
            recovery = root / "recovery.json"
            created = create_recovery_receipt(
                source_checkpoint=source,
                migrated_checkpoint=checkpoint,
                migration_receipt=migration,
                output=recovery,
                force=False,
                expected_source_sha256=digest,
            )
            validated = validate_recovery_receipt(
                recovery_receipt=recovery,
                source_checkpoint=source,
                migrated_checkpoint=checkpoint,
                migration_receipt=migration,
                expected_source_sha256=digest,
            )
            self.assertEqual(created, validated)
            self.assertEqual(created["status"], "pass")
            self.assertEqual(created["route"]["process_updates"], 799)
            self.assertEqual(
                created["source_checkpoint"]["common_step_counter"],
                PINNED_SOURCE_COMMON_STEP_COUNTER,
            )
            self.assertTrue(all(created["checks"].values()))

    def test_wrong_source_hash_is_rejected_before_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source, checkpoint, migration, _digest = self._prepare(Path(temporary))
            with self.assertRaisesRegex(ValueError, "Pinned raw model9200 SHA"):
                validate_recovery_seed(
                    source_checkpoint=source,
                    migrated_checkpoint=checkpoint,
                    migration_receipt=migration,
                    expected_source_sha256="0" * 64,
                )

    def test_noncanonical_clock_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source, checkpoint, migration, digest = self._prepare(
                Path(temporary), iteration=9_199
            )
            with self.assertRaisesRegex(ValueError, "Recovery clock"):
                validate_recovery_seed(
                    source_checkpoint=source,
                    migrated_checkpoint=checkpoint,
                    migration_receipt=migration,
                    expected_source_sha256=digest,
                )

    def test_tampered_migration_receipt_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, checkpoint, migration, digest = self._prepare(root)
            receipt = json.loads(migration.read_text(encoding="utf-8"))
            receipt["strategy"] = "zero_hand"
            migration.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "migration_receipt"):
                validate_recovery_seed(
                    source_checkpoint=source,
                    migrated_checkpoint=checkpoint,
                    migration_receipt=migration,
                    expected_source_sha256=digest,
                )

    def test_tampered_recovery_route_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, checkpoint, migration, digest = self._prepare(root)
            recovery = root / "recovery.json"
            create_recovery_receipt(
                source_checkpoint=source,
                migrated_checkpoint=checkpoint,
                migration_receipt=migration,
                output=recovery,
                force=False,
                expected_source_sha256=digest,
            )
            receipt = json.loads(recovery.read_text(encoding="utf-8"))
            receipt["route"]["process_updates"] = 798
            recovery.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "recovery_receipt"):
                validate_recovery_receipt(
                    recovery_receipt=recovery,
                    source_checkpoint=source,
                    migrated_checkpoint=checkpoint,
                    migration_receipt=migration,
                    expected_source_sha256=digest,
                )

    def test_launcher_is_syntax_checked_and_route_is_fixed(self) -> None:
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
        help_text = subprocess.run(
            ["bash", str(LAUNCHER), "--help"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        self.assertIn("exactly 799 updates", help_text)
        launcher = LAUNCHER.read_text(encoding="utf-8")
        for exact_argument in (
            "--env.scene.num-envs 2048",
            "--env.seed 42",
            "--agent.seed 42",
            "--agent.num-steps-per-env 24",
            "--agent.save-interval 100",
            "--agent.resume True",
            '--agent.load-checkpoint "^model_${SOURCE_ITERATION}[.]pt$"',
        ):
            self.assertIn(exact_argument, launcher)
        self.assertEqual(RECOVERY_PROCESS_UPDATES, 799)


if __name__ == "__main__":
    unittest.main()
