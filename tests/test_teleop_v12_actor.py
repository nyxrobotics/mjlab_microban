"""CPU invariants for the legacy-preserving contract-v12 actor."""

from __future__ import annotations

import unittest

import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from mjlab_microban.tasks.microban_teleop_v12_actor import (
    LEGACY_TO_TELEOP_OBSERVATION_INDEX,
    TELEOP_V12_EXTRA_OBSERVATION_COLUMNS,
    TELEOP_V12_FOOT_OBSERVATION_COLUMNS,
    TELEOP_V12_HAND_OBSERVATION_COLUMNS,
    TELEOP_V12_HMD_OBSERVATION_COLUMNS,
    TELEOP_V12_SHARED_OBSERVATION_COLUMNS,
    FrozenEmpiricalNormalization,
    LegacyAdapterTeleopActor,
    teleop_v12_active_adapter_columns,
    transplant_legacy_actor_state_to_teleop83,
)


def _observation(width: int) -> TensorDict:
    return TensorDict({"actor": torch.zeros(1, width)}, batch_size=[1])


def _legacy_model() -> MLPModel:
    return MLPModel(
        obs=_observation(63),
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
        obs=_observation(83),
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


def _transplanted_pair() -> tuple[MLPModel, LegacyAdapterTeleopActor]:
    torch.manual_seed(7)
    source = _legacy_model()
    source_state = source.state_dict()
    source_state["obs_normalizer._mean"].normal_()
    source_state["obs_normalizer._var"].uniform_(0.1, 2.0)
    source_state["obs_normalizer._std"].copy_(
        torch.sqrt(source_state["obs_normalizer._var"])
    )
    source_state["obs_normalizer.count"].fill_(1_474_560_000)
    source_state["distribution.std_param"].uniform_(0.6, 1.0)
    source.load_state_dict(source_state, strict=True)
    target = _target_model()
    mapped = transplant_legacy_actor_state_to_teleop83(
        source_state,
        target.state_dict(),
        LEGACY_TO_TELEOP_OBSERVATION_INDEX,
    )
    target.load_state_dict(mapped, strict=True)
    target.bind_frozen_legacy_reference()
    return source, target


class TeleopV12ActorTest(unittest.TestCase):
    def test_semantic_mapping_has_expected_shared_and_extra_columns(self) -> None:
        self.assertEqual(
            LEGACY_TO_TELEOP_OBSERVATION_INDEX,
            (
                *((index, index) for index in range(6)),
                *((index, index + 3) for index in range(6, 24)),
                *((index, index + 6) for index in range(24, 63)),
            ),
        )
        self.assertEqual(len(TELEOP_V12_SHARED_OBSERVATION_COLUMNS), 63)
        self.assertEqual(
            TELEOP_V12_EXTRA_OBSERVATION_COLUMNS,
            (6, 7, 8, 27, 28, 29, *range(69, 83)),
        )

    def test_pristine_actor_matches_legacy_with_float32_tolerance(self) -> None:
        source, target = _transplanted_pair()
        generator = torch.Generator().manual_seed(20260925)
        teleop_obs = torch.randn(10_000, 83, generator=generator)
        source_obs = teleop_obs[:, list(TELEOP_V12_SHARED_OBSERVATION_COLUMNS)]
        with torch.inference_mode():
            expected = source(TensorDict({"actor": source_obs}, batch_size=[10_000]))
            actual = target(TensorDict({"actor": teleop_obs}, batch_size=[10_000]))
        torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-5)
        self.assertLessEqual(float(torch.max(torch.abs(actual - expected))), 2.0e-5)

    def test_frozen_normalizer_ignores_updates_while_training(self) -> None:
        normalizer = FrozenEmpiricalNormalization(83)
        before = {
            name: value.clone() for name, value in normalizer.state_dict().items()
        }
        normalizer.train()
        normalizer.update(torch.randn(128, 83))
        self.assertTrue(normalizer.training)
        for name, expected in before.items():
            self.assertTrue(torch.equal(normalizer.state_dict()[name], expected))

    def test_only_extra_first_layer_columns_change(self) -> None:
        _source, target = _transplanted_pair()
        target.bind_common_step_provider(lambda: 10_000 * 24)
        self.assertEqual(target.trainable_actor_parameter_names(), ("mlp.0.weight",))
        optimizer = torch.optim.Adam(target.parameters(), lr=1.0e-3)
        before = {name: value.clone() for name, value in target.state_dict().items()}
        obs = torch.randn(64, 83)
        loss = target(TensorDict({"actor": obs}, batch_size=[64])).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        target.assert_frozen_legacy_state()
        target.assert_optimizer_invariant(optimizer)

        after = target.state_dict()
        self.assertTrue(
            torch.equal(
                before["mlp.0.weight"][:, TELEOP_V12_SHARED_OBSERVATION_COLUMNS],
                after["mlp.0.weight"][:, TELEOP_V12_SHARED_OBSERVATION_COLUMNS],
            )
        )
        self.assertFalse(
            torch.equal(
                before["mlp.0.weight"][:, TELEOP_V12_EXTRA_OBSERVATION_COLUMNS],
                after["mlp.0.weight"][:, TELEOP_V12_EXTRA_OBSERVATION_COLUMNS],
            )
        )
        for name, expected in before.items():
            if name != "mlp.0.weight":
                self.assertTrue(torch.equal(after[name], expected), name)

    def test_resume_rejects_adam_momentum_in_shared_columns(self) -> None:
        _source, target = _transplanted_pair()
        target.bind_common_step_provider(lambda: 10_000 * 24)
        optimizer = torch.optim.Adam(target.parameters(), lr=1.0e-3)
        obs = torch.randn(16, 83)
        optimizer.zero_grad()
        target(TensorDict({"actor": obs}, batch_size=[16])).sum().backward()
        optimizer.step()
        target.assert_optimizer_invariant(optimizer)
        first = target.mlp[0]
        assert isinstance(first, torch.nn.Linear)
        optimizer.state[first.weight]["exp_avg"][0, 0] = 1.0
        with self.assertRaisesRegex(RuntimeError, "schedule-locked"):
            target.assert_optimizer_invariant(optimizer)

    def test_gradient_schedule_prevents_early_hmd_noise_contamination(self) -> None:
        self.assertEqual(teleop_v12_active_adapter_columns(0), ())
        self.assertEqual(teleop_v12_active_adapter_columns(7_000 * 24), ())
        self.assertEqual(
            teleop_v12_active_adapter_columns(7_000 * 24 + 24),
            (*TELEOP_V12_HMD_OBSERVATION_COLUMNS, *TELEOP_V12_HAND_OBSERVATION_COLUMNS),
        )
        self.assertEqual(
            teleop_v12_active_adapter_columns(10_000 * 24),
            (*TELEOP_V12_HMD_OBSERVATION_COLUMNS, *TELEOP_V12_HAND_OBSERVATION_COLUMNS),
        )
        self.assertEqual(
            teleop_v12_active_adapter_columns(10_000 * 24 + 24),
            TELEOP_V12_EXTRA_OBSERVATION_COLUMNS,
        )

        _source, target = _transplanted_pair()
        common_step = 0
        target.bind_common_step_provider(lambda: common_step)
        optimizer = torch.optim.Adam(target.parameters(), lr=1.0e-3)
        first = target.mlp[0]
        assert isinstance(first, torch.nn.Linear)
        initial = first.weight.detach().clone()

        def update() -> None:
            optimizer.zero_grad()
            obs = torch.randn(64, 83)
            target(
                TensorDict({"actor": obs}, batch_size=[64])
            ).square().mean().backward()
            optimizer.step()

        update()
        self.assertTrue(torch.equal(first.weight, initial))
        target.assert_optimizer_invariant(optimizer)

        common_step = 7_000 * 24 + 24
        update()
        active = (
            *TELEOP_V12_HMD_OBSERVATION_COLUMNS,
            *TELEOP_V12_HAND_OBSERVATION_COLUMNS,
        )
        self.assertFalse(torch.equal(first.weight[:, active], initial[:, active]))
        self.assertTrue(
            torch.equal(
                first.weight[:, TELEOP_V12_FOOT_OBSERVATION_COLUMNS],
                initial[:, TELEOP_V12_FOOT_OBSERVATION_COLUMNS],
            )
        )
        target.assert_optimizer_invariant(optimizer)


if __name__ == "__main__":
    unittest.main()
