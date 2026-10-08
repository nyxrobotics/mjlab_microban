"""CPU invariants for the legacy-preserving contract-v12 actor."""

from __future__ import annotations

import unittest

import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from mjlab_microban.schedules import PICO_SCHEDULE

from mjlab_microban.tasks.microban_teleop_v12_actor import (
    LEGACY_TO_TELEOP_OBSERVATION_INDEX,
    TELEOP_V12_EXTRA_OBSERVATION_COLUMNS,
    TELEOP_V12_FOOT_OBSERVATION_COLUMNS,
    TELEOP_V12_ARM_OBSERVATION_COLUMNS,
    TELEOP_V12_ARM_TARGET_NORMALIZER_DENOMINATORS,
    TELEOP_V12_HMD_OBSERVATION_COLUMNS,
    TELEOP_V12_SHARED_OBSERVATION_COLUMNS,
    FrozenEmpiricalNormalization,
    TELEOP_RESIDUAL_STATE_KEYS,
    TELEOP_TRAINABLE_ACTOR_PARAMETERS,
    LegacyAdapterTeleopActor,
    teleop_residual_trainable,
    teleop_v12_active_adapter_columns,
    transplant_legacy_actor_state_to_teleop,
    without_residual,
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
        obs=_observation(81),
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
    mapped = transplant_legacy_actor_state_to_teleop(
        source_state,
        target.state_dict(),
        LEGACY_TO_TELEOP_OBSERVATION_INDEX,
    )
    target.load_state_dict(mapped, strict=True)
    target.bind_frozen_legacy_reference()
    return source, target


class TeleopV12ActorTest(unittest.TestCase):
    def test_arm_target_columns_and_their_normalizer(self) -> None:
        self.assertEqual(TELEOP_V12_ARM_OBSERVATION_COLUMNS, tuple(range(75, 81)))
        # The largest distance from HOME to the box edge: pitch 100, roll 110,
        # elbow 90 deg, so HOME reads 0 and the box at most 1.
        expected = [1.7453292519943295, 1.9198621771937625, 1.5707963267948966] * 2
        for value, want in zip(TELEOP_V12_ARM_TARGET_NORMALIZER_DENOMINATORS, expected, strict=True):
            self.assertAlmostEqual(value, want, places=12)

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
            (6, 7, 8, 27, 28, 29, *range(69, 81)),
        )

    def test_pristine_actor_matches_legacy_with_float32_tolerance(self) -> None:
        source, target = _transplanted_pair()
        generator = torch.Generator().manual_seed(20260925)
        teleop_obs = torch.randn(10_000, 81, generator=generator)
        source_obs = teleop_obs[:, list(TELEOP_V12_SHARED_OBSERVATION_COLUMNS)]
        with torch.inference_mode():
            expected = source(TensorDict({"actor": source_obs}, batch_size=[10_000]))
            actual = target(TensorDict({"actor": teleop_obs}, batch_size=[10_000]))
        torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-5)
        self.assertLessEqual(float(torch.max(torch.abs(actual - expected))), 2.0e-5)

    def test_frozen_normalizer_ignores_updates_while_training(self) -> None:
        normalizer = FrozenEmpiricalNormalization(81)
        before = {
            name: value.clone() for name, value in normalizer.state_dict().items()
        }
        normalizer.train()
        normalizer.update(torch.randn(128, 81))
        self.assertTrue(normalizer.training)
        for name, expected in before.items():
            self.assertTrue(torch.equal(normalizer.state_dict()[name], expected))

    def test_only_extra_first_layer_columns_change(self) -> None:
        _source, target = _transplanted_pair()
        target.bind_common_step_provider(lambda: 10_000 * 24)
        trainable = [name for name, value in target.named_parameters() if value.requires_grad]
        self.assertEqual(sorted(trainable), sorted(TELEOP_TRAINABLE_ACTOR_PARAMETERS))
        optimizer = torch.optim.Adam(target.parameters(), lr=1.0e-3)
        before = {name: value.clone() for name, value in target.state_dict().items()}
        obs = torch.randn(64, 81)
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
            if name != "mlp.0.weight" and name not in TELEOP_RESIDUAL_STATE_KEYS:
                self.assertTrue(torch.equal(after[name], expected), name)

    def test_optimizer_invariant_rejects_adam_momentum_in_locked_columns(self) -> None:
        _source, target = _transplanted_pair()
        target.bind_common_step_provider(lambda: 10_000 * 24)
        optimizer = torch.optim.Adam(target.parameters(), lr=1.0e-3)
        obs = torch.randn(16, 81)
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
        ARM, FOOT = PICO_SCHEDULE["arm"], PICO_SCHEDULE["foot"]
        self.assertEqual(teleop_v12_active_adapter_columns(0), ())
        self.assertEqual(teleop_v12_active_adapter_columns(ARM * 24), ())
        self.assertEqual(
            teleop_v12_active_adapter_columns(ARM * 24 + 24),
            (*TELEOP_V12_HMD_OBSERVATION_COLUMNS, *TELEOP_V12_ARM_OBSERVATION_COLUMNS),
        )
        self.assertEqual(
            teleop_v12_active_adapter_columns(FOOT * 24),
            (*TELEOP_V12_HMD_OBSERVATION_COLUMNS, *TELEOP_V12_ARM_OBSERVATION_COLUMNS),
        )
        self.assertEqual(
            teleop_v12_active_adapter_columns(FOOT * 24 + 24),
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
            obs = torch.randn(64, 81)
            target(
                TensorDict({"actor": obs}, batch_size=[64])
            ).square().mean().backward()
            optimizer.step()

        update()
        self.assertTrue(torch.equal(first.weight, initial))
        target.assert_optimizer_invariant(optimizer)

        common_step = ARM * 24 + 24
        update()
        active = (
            *TELEOP_V12_HMD_OBSERVATION_COLUMNS,
            *TELEOP_V12_ARM_OBSERVATION_COLUMNS,
        )
        self.assertFalse(torch.equal(first.weight[:, active], initial[:, active]))
        self.assertTrue(
            torch.equal(
                first.weight[:, TELEOP_V12_FOOT_OBSERVATION_COLUMNS],
                initial[:, TELEOP_V12_FOOT_OBSERVATION_COLUMNS],
            )
        )
        target.assert_optimizer_invariant(optimizer)


def _nonzero_residual(target: LegacyAdapterTeleopActor) -> None:
    output = target.residual[-1]
    assert isinstance(output, torch.nn.Linear)
    with torch.no_grad():
        output.weight.normal_(0.0, 0.1, generator=torch.Generator().manual_seed(3))
        output.bias.fill_(0.05)


class TeleopResidualTest(unittest.TestCase):
    """The residual MLP added to the frozen walker's action mean."""

    def test_pristine_actor_is_the_walker_exactly(self) -> None:
        _source, target = _transplanted_pair()
        self.assertEqual(
            sorted(name for name in target.state_dict() if name.startswith("residual.")),
            sorted(TELEOP_RESIDUAL_STATE_KEYS),
        )
        self.assertEqual(tuple(target.residual[0].weight.shape), (64, 81))
        self.assertEqual(tuple(target.residual[2].weight.shape), (64, 64))
        self.assertEqual(tuple(target.residual[4].weight.shape), (18, 64))
        obs = TensorDict({"actor": torch.randn(256, 81)}, batch_size=[256])
        with torch.inference_mode():
            self.assertTrue(torch.equal(target(obs), without_residual(target)(obs)))
            self.assertTrue(torch.equal(target(obs), target.mlp(target.get_latent(obs))))

    def test_residual_adds_to_the_walker_mean(self) -> None:
        _source, target = _transplanted_pair()
        _nonzero_residual(target)
        obs = TensorDict({"actor": torch.randn(64, 81)}, batch_size=[64])
        with torch.inference_mode():
            latent = target.get_latent(obs)
            expected = target.mlp(latent) + target.residual(latent)
            torch.testing.assert_close(target(obs), expected, rtol=0.0, atol=0.0)
            self.assertFalse(torch.equal(target(obs), without_residual(target)(obs)))
        target.distribution.update(expected)
        torch.testing.assert_close(target.distribution.mean, expected, rtol=0.0, atol=0.0)

    def test_residual_init_leaves_the_global_random_stream(self) -> None:
        torch.manual_seed(11)
        _target_model()
        with_residual = torch.rand(4)
        torch.manual_seed(11)
        MLPModel(
            obs=_observation(81),
            obs_groups={"actor": ["actor"]},
            obs_set="actor",
            output_dim=18,
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        )
        self.assertTrue(torch.equal(with_residual, torch.rand(4)))

    def test_residual_trains_only_from_the_arm_stage(self) -> None:
        ARM = PICO_SCHEDULE["arm"]
        self.assertFalse(teleop_residual_trainable(0))
        self.assertFalse(teleop_residual_trainable(ARM * 24))
        self.assertTrue(teleop_residual_trainable(ARM * 24 + 24))

        _source, target = _transplanted_pair()
        common_step = 0
        target.bind_common_step_provider(lambda: common_step)
        optimizer = torch.optim.Adam(target.parameters(), lr=1.0e-3)
        initial = {name: value.clone() for name, value in target.residual.state_dict().items()}

        def update() -> None:
            optimizer.zero_grad()
            obs = torch.randn(64, 81)
            target(TensorDict({"actor": obs}, batch_size=[64]), stochastic_output=True)
            target.get_output_log_prob(torch.randn(64, 18)).mean().backward()
            optimizer.step()

        update()
        for name, value in target.residual.state_dict().items():
            self.assertTrue(torch.equal(value, initial[name]), name)
        target.assert_optimizer_invariant(optimizer)
        target.assert_schedule_locked_weights_zero()

        common_step = ARM * 24 + 24
        update()
        update()
        self.assertFalse(torch.equal(target.residual[4].weight, initial["4.weight"]))
        self.assertFalse(torch.equal(target.residual[0].weight, initial["0.weight"]))
        target.assert_frozen_legacy_state()
        target.assert_optimizer_invariant(optimizer)

    def test_locked_residual_rejects_a_nonzero_output_layer(self) -> None:
        _source, target = _transplanted_pair()
        target.bind_common_step_provider(lambda: 0)
        target.assert_schedule_locked_weights_zero()
        _nonzero_residual(target)
        with self.assertRaisesRegex(RuntimeError, "residual"):
            target.assert_schedule_locked_weights_zero()
        target.bind_common_step_provider(lambda: PICO_SCHEDULE["arm"] * 24 + 24)
        target.assert_schedule_locked_weights_zero()

    def test_onnx_is_one_graph_with_the_residual(self) -> None:
        import tempfile
        from pathlib import Path

        import numpy as np
        import onnxruntime as ort

        from mjlab_microban.scripts.teleop_v12_bootstrap_gate import _export_onnx_atomic

        _source, target = _transplanted_pair()
        _nonzero_residual(target)
        target.eval()
        obs = torch.randn(32, 81, generator=torch.Generator().manual_seed(5))
        with torch.inference_mode():
            expected = target(TensorDict({"actor": obs}, batch_size=[32])).numpy()
            walker = without_residual(target)(TensorDict({"actor": obs}, batch_size=[32])).numpy()
            jit = target.as_jit()(obs).numpy()
        self.assertGreater(float(np.max(np.abs(expected - walker))), 1.0e-2)
        np.testing.assert_allclose(jit, expected, rtol=0.0, atol=1.0e-6)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pico.onnx"
            _export_onnx_atomic(target, path)
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            actual = np.concatenate(
                [session.run(None, {"obs": row[None].numpy()})[0] for row in obs]
            )
        np.testing.assert_allclose(actual, expected, rtol=1.0e-5, atol=1.0e-5)


if __name__ == "__main__":
    unittest.main()
