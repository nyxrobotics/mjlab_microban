"""CPU gates for the authenticated contract-v12 bootstrap chain."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from mjlab_microban.tasks.microban_teleop_v12_actor import (
    LEGACY_TO_TELEOP_OBSERVATION_INDEX,
    LEGACY_VELOCITY_CHECKPOINT_SHA256,
    TELEOP_V12_EXTRA_OBSERVATION_COLUMNS,
    LegacyAdapterTeleopActor,
)
from mjlab_microban.tasks.microban_teleop_v12_bootstrap import (
    PINNED_LEGACY_TELEOP_PROBE_SHA256,
    assert_actor_frozen_against_source,
    bootstrap_legacy_actor,
    inspect_legacy_velocity_checkpoint,
    serialize_bootstrap_provenance,
    validate_bootstrap_provenance,
)
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MicrobanTeleopV12RlCfg,
    make_microban_teleop_v12_env_cfg,
)

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoints/xc330_velocity/model_14999.pt"
RECEIPT = ROOT / "artifacts/legacy_teleop_probe/model_14999_teleop83_raw_9x300.json"


def _observations(width: int) -> TensorDict:
    return TensorDict({"actor": torch.zeros(1, width)}, batch_size=[1])


def _source_model() -> MLPModel:
    return MLPModel(
        obs=_observations(63),
        obs_groups={"actor": ["actor"]},
        obs_set="actor",
        output_dim=18,
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    )


def _target_model() -> LegacyAdapterTeleopActor:
    return LegacyAdapterTeleopActor(
        obs=_observations(83),
        obs_groups={"actor": ["actor"]},
        obs_set="actor",
        output_dim=18,
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    )


class TeleopV12BootstrapTest(unittest.TestCase):
    def test_authenticated_pristine_actor_parity(self) -> None:
        source_identity, source_state = inspect_legacy_velocity_checkpoint(
            CHECKPOINT, LEGACY_VELOCITY_CHECKPOINT_SHA256
        )
        source = _source_model()
        source.load_state_dict(source_state, strict=True)
        target = _target_model()
        provenance = bootstrap_legacy_actor(
            target,
            CHECKPOINT,
            LEGACY_VELOCITY_CHECKPOINT_SHA256,
            RECEIPT,
            PINNED_LEGACY_TELEOP_PROBE_SHA256,
        )
        self.assertEqual(provenance.source, source_identity)
        self.assertEqual(
            int(target.obs_normalizer.count.item()), source_identity.normalizer_count
        )
        self.assertEqual(target.trainable_actor_parameter_names(), ("mlp.0.weight",))

        generator = torch.Generator().manual_seed(20260925)
        teleop = torch.randn(10_000, 83, generator=generator)
        legacy = teleop[
            :, [target for _source, target in LEGACY_TO_TELEOP_OBSERVATION_INDEX]
        ]
        with torch.inference_mode():
            expected = source(TensorDict({"actor": legacy}, batch_size=[10_000]))
            actual = target(TensorDict({"actor": teleop}, batch_size=[10_000]))
        max_absolute = float(torch.max(torch.abs(actual - expected)).item())
        self.assertLessEqual(max_absolute, 2.0e-5)
        torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-5)

        validated = validate_bootstrap_provenance(
            serialize_bootstrap_provenance(provenance), verify_files=True
        )
        self.assertEqual(validated, provenance)
        assert_actor_frozen_against_source(target, provenance)

    def test_source_guard_allows_only_new_first_layer_columns(self) -> None:
        target = _target_model()
        provenance = bootstrap_legacy_actor(
            target,
            CHECKPOINT,
            LEGACY_VELOCITY_CHECKPOINT_SHA256,
            RECEIPT,
            PINNED_LEGACY_TELEOP_PROBE_SHA256,
        )
        first = target.mlp[0]
        assert isinstance(first, torch.nn.Linear)
        with torch.no_grad():
            first.weight[0, TELEOP_V12_EXTRA_OBSERVATION_COLUMNS[0]] += 0.5
        assert_actor_frozen_against_source(target, provenance)
        target.assert_frozen_legacy_state()
        with torch.no_grad():
            first.weight[0, 0] += 0.5
        with self.assertRaisesRegex(RuntimeError, "pinned legacy"):
            assert_actor_frozen_against_source(target, provenance)

    def test_v12_environment_and_runner_config_use_raw_legacy_semantics(self) -> None:
        cfg = make_microban_teleop_v12_env_cfg(play=True)
        self.assertIsNone(cfg.actions["joint_pos"].clip)
        self.assertNotIn("target_clip_excess", cfg.rewards)
        self.assertNotIn("target_near_limit", cfg.rewards)
        self.assertNotIn("raw_action_l2", cfg.rewards)
        self.assertIn("joint_soft_limit_guard", cfg.rewards)
        self.assertEqual(
            cfg.observations["actor"].terms["actions"].func.__name__, "last_action"
        )
        self.assertEqual(
            cfg.observations["critic"].terms["actions"].func.__name__, "last_action"
        )
        self.assertTrue(MicrobanTeleopV12RlCfg.actor.obs_normalization)
        self.assertEqual(
            MicrobanTeleopV12RlCfg.actor.distribution_cfg["std_type"], "scalar"
        )
        self.assertEqual(MicrobanTeleopV12RlCfg.clip_actions, None)


if __name__ == "__main__":
    unittest.main()
