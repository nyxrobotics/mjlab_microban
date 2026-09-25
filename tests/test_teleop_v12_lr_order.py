"""CPU-only regression tests for the bilateral site-order recovery."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import torch

from mjlab_microban.scripts.evaluate_teleop_v12_checkpoint import _load_actor
from mjlab_microban.scripts.migrate_teleop_v12_lr_order import (
    migrate_checkpoint,
    migrate_checkpoint_payload,
)
from mjlab_microban.scripts.teleop_v12_stage import _checkpoint_identity
from mjlab_microban.tasks.mdp import (
    MICROBAN_BILATERAL_SITE_ORDER_REVISION,
    _resolve_ordered_site_cfg,
)
from mjlab_microban.tasks.microban_teleop_v12_actor import (
    TELEOP_V12_FOOT_OBSERVATION_COLUMNS,
    TELEOP_V12_HAND_OBSERVATION_COLUMNS,
)
from mjlab_microban.tasks.microban_teleop_v12_lr_order import (
    ACTOR_PERMUTATION,
    BILATERAL_SITE_ORDER_INFO_KEY,
    CRITIC_PERMUTATION,
    MIGRATION_INFO_KEY,
    validate_bilateral_site_order_checkpoint,
    validate_lr_order_checkpoint_lineage,
    validate_lr_order_migration_marker,
)
from mjlab_microban.tasks.microban_teleop_v12_runner import (
    TELEOP_V12_OBSERVATION_TERM_LAYOUTS,
    teleop_v12_observation_term_slices,
)


class _FakeEntity:
    def __init__(self, *, ignore_preserve_order: bool = False):
        self.site_names = ["right_hand", "right_foot", "left_hand", "left_foot"]
        self.num_sites = len(self.site_names)
        self.ignore_preserve_order = ignore_preserve_order
        self.preserve_order_calls: list[bool] = []

    def find_sites(self, names, *, preserve_order=False):
        self.preserve_order_calls.append(preserve_order)
        requested = list(names)
        if preserve_order and not self.ignore_preserve_order:
            resolved = requested
        else:
            resolved = [name for name in self.site_names if name in requested]
        return [self.site_names.index(name) for name in resolved], resolved


def _state(width: int, *, seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "obs_normalizer._mean": torch.randn(1, width, generator=generator),
        "obs_normalizer._var": torch.rand(1, width, generator=generator) + 0.1,
        "obs_normalizer._std": torch.rand(1, width, generator=generator) + 0.1,
        "obs_normalizer.count": torch.tensor(1234),
        "mlp.0.weight": torch.randn(512, width, generator=generator),
        "mlp.0.bias": torch.randn(512, generator=generator),
    }


def _checkpoint() -> dict:
    actor = _state(83, seed=1)
    critic = _state(137, seed=2)
    actor["mlp.0.weight"][:, TELEOP_V12_FOOT_OBSERVATION_COLUMNS] = 0.0
    actor_moments = {
        "step": torch.tensor(10.0),
        "exp_avg": torch.randn(512, 83),
        "exp_avg_sq": torch.rand(512, 83),
    }
    actor_moments["exp_avg"][:, TELEOP_V12_FOOT_OBSERVATION_COLUMNS] = 0.0
    actor_moments["exp_avg_sq"][:, TELEOP_V12_FOOT_OBSERVATION_COLUMNS] = 0.0
    critic_moments = {
        "step": torch.tensor(10.0),
        "exp_avg": torch.randn(512, 137),
        "exp_avg_sq": torch.rand(512, 137),
    }
    return {
        "actor_state_dict": actor,
        "critic_state_dict": critic,
        "optimizer_state_dict": {
            "state": {1: actor_moments, 9: critic_moments},
            "param_groups": [{"params": list(range(17)), "lr": 1.0e-4}],
        },
        "iter": 9200,
        "infos": {
            "env_state": {"common_step_counter": 9201 * 24},
            "microban_teleop_training_contract_version": "12",
            "active_actor_columns_at_save": [
                6,
                7,
                8,
                27,
                28,
                29,
                *TELEOP_V12_HAND_OBSERVATION_COLUMNS,
            ],
        },
    }


def _first_layer(
    state: dict[str, torch.Tensor], observation: torch.Tensor
) -> torch.Tensor:
    normalized = (observation - state["obs_normalizer._mean"]) / (
        state["obs_normalizer._std"] + 0.01
    )
    return torch.nn.functional.linear(
        normalized, state["mlp.0.weight"], state["mlp.0.bias"]
    )


class BilateralSiteOrderTest(unittest.TestCase):
    def test_requested_left_right_order_is_preserved(self) -> None:
        entity = _FakeEntity()
        cfg = _resolve_ordered_site_cfg(
            {"robot": entity},
            entity_name="robot",
            site_names=("left_hand", "right_hand"),
            label="test",
        )
        self.assertTrue(cfg.preserve_order)
        self.assertEqual(cfg.site_names, ["left_hand", "right_hand"])
        self.assertEqual(cfg.site_ids, [2, 0])
        self.assertEqual(entity.preserve_order_calls, [True])

    def test_resolver_order_drift_fails_during_construction(self) -> None:
        entity = _FakeEntity(ignore_preserve_order=True)
        with self.assertRaisesRegex(RuntimeError, "site order drifted"):
            _resolve_ordered_site_cfg(
                {"robot": entity},
                entity_name="robot",
                site_names=("left_foot", "right_foot"),
                label="test",
            )

    def test_resolved_observation_term_slices_are_fail_closed(self) -> None:
        manager = SimpleNamespace(
            active_terms={
                group: [name for name, _width in layout]
                for group, layout in TELEOP_V12_OBSERVATION_TERM_LAYOUTS.items()
            },
            group_obs_term_dim={
                group: [(width,) for _name, width in layout]
                for group, layout in TELEOP_V12_OBSERVATION_TERM_LAYOUTS.items()
            },
        )
        slices = teleop_v12_observation_term_slices(manager)
        self.assertEqual(slices["actor"]["hand_target"], slice(75, 83))
        self.assertEqual(slices["critic"]["hand_target"], slice(90, 98))
        manager.active_terms["critic"][-2:] = reversed(
            manager.active_terms["critic"][-2:]
        )
        with self.assertRaisesRegex(ValueError, "observation layout drifted"):
            teleop_v12_observation_term_slices(manager)

    def test_direct_actor_and_stage_consumers_reject_raw_prefix_checkpoint(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model_9200.pt"
            torch.save(_checkpoint(), checkpoint)
            with self.assertRaisesRegex(ValueError, "predates"):
                _load_actor(checkpoint, device="cpu")
            with self.assertRaisesRegex(ValueError, "predates"):
                _checkpoint_identity(checkpoint)


class TeleopV12LrMigrationTest(unittest.TestCase):
    def test_swap_is_functionally_the_exact_input_permutation(self) -> None:
        source = _checkpoint()
        migrated, evidence = migrate_checkpoint_payload(
            source,
            source_path=Path("/pinned/model_9200.pt"),
            source_sha256="a" * 64,
            strategy="swap",
        )
        generator = torch.Generator().manual_seed(42)
        actor_input = torch.randn(64, 83, generator=generator)
        critic_input = torch.randn(64, 137, generator=generator)
        torch.testing.assert_close(
            _first_layer(migrated["actor_state_dict"], actor_input),
            _first_layer(source["actor_state_dict"], actor_input[:, ACTOR_PERMUTATION]),
            rtol=1.0e-5,
            atol=1.0e-5,
        )
        torch.testing.assert_close(
            _first_layer(migrated["critic_state_dict"], critic_input),
            _first_layer(
                source["critic_state_dict"], critic_input[:, CRITIC_PERMUTATION]
            ),
            rtol=1.0e-5,
            atol=1.0e-5,
        )
        self.assertTrue(evidence["clock_preserved"])
        self.assertEqual(migrated["iter"], source["iter"])
        self.assertEqual(migrated["infos"]["env_state"], source["infos"]["env_state"])
        self.assertEqual(
            migrated["infos"][BILATERAL_SITE_ORDER_INFO_KEY],
            MICROBAN_BILATERAL_SITE_ORDER_REVISION,
        )

    def test_adam_and_normalizer_follow_same_semantic_permutation(self) -> None:
        source = _checkpoint()
        migrated, _evidence = migrate_checkpoint_payload(
            source,
            source_path=Path("/pinned/model_9200.pt"),
            source_sha256="b" * 64,
            strategy="swap",
        )
        for state_name, permutation in (
            ("actor_state_dict", ACTOR_PERMUTATION),
            ("critic_state_dict", CRITIC_PERMUTATION),
        ):
            for tensor_name in (
                "obs_normalizer._mean",
                "obs_normalizer._var",
                "obs_normalizer._std",
            ):
                self.assertTrue(
                    torch.equal(
                        migrated[state_name][tensor_name],
                        source[state_name][tensor_name][..., permutation],
                    )
                )
        for parameter_id, permutation in (
            (1, ACTOR_PERMUTATION),
            (9, CRITIC_PERMUTATION),
        ):
            for moment in ("exp_avg", "exp_avg_sq"):
                self.assertTrue(
                    torch.equal(
                        migrated["optimizer_state_dict"]["state"][parameter_id][moment],
                        source["optimizer_state_dict"]["state"][parameter_id][moment][
                            ..., permutation
                        ],
                    )
                )

    def test_foot_columns_and_every_out_of_scope_tensor_remain_exact(self) -> None:
        source = _checkpoint()
        migrated, evidence = migrate_checkpoint_payload(
            source,
            source_path=Path("/pinned/model_9200.pt"),
            source_sha256="c" * 64,
            strategy="swap",
        )
        self.assertTrue(
            torch.equal(
                migrated["actor_state_dict"]["mlp.0.weight"][
                    :, TELEOP_V12_FOOT_OBSERVATION_COLUMNS
                ],
                source["actor_state_dict"]["mlp.0.weight"][
                    :, TELEOP_V12_FOOT_OBSERVATION_COLUMNS
                ],
            )
        )
        integrity = evidence["tensor_integrity"]
        self.assertTrue(integrity["passed"])
        self.assertGreater(integrity["unchanged_tensor_count"], 0)
        for item in integrity["partially_transformed_tensors"].values():
            self.assertEqual(
                item["untouched_source_sha256"], item["untouched_output_sha256"]
            )

    def test_zero_hand_strategy_erases_actor_hand_w0_and_moments_only(self) -> None:
        source = _checkpoint()
        migrated, _evidence = migrate_checkpoint_payload(
            source,
            source_path=Path("/pinned/model_9200.pt"),
            source_sha256="d" * 64,
            strategy="zero_hand",
        )
        columns = TELEOP_V12_HAND_OBSERVATION_COLUMNS
        self.assertEqual(
            torch.count_nonzero(
                migrated["actor_state_dict"]["mlp.0.weight"][:, columns]
            ).item(),
            0,
        )
        for name in ("exp_avg", "exp_avg_sq"):
            self.assertEqual(
                torch.count_nonzero(
                    migrated["optimizer_state_dict"]["state"][1][name][:, columns]
                ).item(),
                0,
            )

    def test_migration_rejects_nonzero_locked_foot_columns(self) -> None:
        source = _checkpoint()
        source["actor_state_dict"]["mlp.0.weight"][0, 69] = 1.0
        with self.assertRaisesRegex(ValueError, "Inactive foot adapter"):
            migrate_checkpoint_payload(
                source,
                source_path=Path("/pinned/model_9200.pt"),
                source_sha256="e" * 64,
                strategy="swap",
            )

    def test_marker_is_exact_and_tampering_is_rejected(self) -> None:
        source = _checkpoint()
        migrated, _evidence = migrate_checkpoint_payload(
            source,
            source_path=Path("/pinned/model_9200.pt"),
            source_sha256="f" * 64,
            strategy="swap",
        )
        marker = validate_lr_order_checkpoint_lineage(migrated["infos"])
        self.assertIsNotNone(marker)
        tampered = deepcopy(marker)
        assert tampered is not None
        tampered["actor_permutation"][75] = 75
        with self.assertRaisesRegex(ValueError, "exact permutation"):
            validate_lr_order_migration_marker(tampered)

    def test_raw_prefixed_checkpoint_cannot_resume_without_migration(self) -> None:
        with self.assertRaisesRegex(ValueError, "predates"):
            validate_bilateral_site_order_checkpoint({})
        self.assertIsNone(
            validate_bilateral_site_order_checkpoint(
                {
                    BILATERAL_SITE_ORDER_INFO_KEY: (
                        MICROBAN_BILATERAL_SITE_ORDER_REVISION
                    )
                }
            )
        )

    def test_cli_artifacts_are_hash_bound_and_reloadable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "model_9200.pt"
            output = root / "model_9200_migrated.pt"
            receipt = root / "migration.json"
            torch.save(_checkpoint(), source)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            report = migrate_checkpoint(
                source=source,
                expected_sha256=digest,
                output=output,
                receipt=receipt,
                strategy="swap",
                force=False,
            )
            self.assertEqual(report["status"], "pass")
            self.assertEqual(
                report["output_checkpoint"]["sha256"],
                hashlib.sha256(output.read_bytes()).hexdigest(),
            )
            loaded = torch.load(output, map_location="cpu", weights_only=False)
            self.assertIn(MIGRATION_INFO_KEY, loaded["infos"])
            self.assertTrue(receipt.is_file())


if __name__ == "__main__":
    unittest.main()
