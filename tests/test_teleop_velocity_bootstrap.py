# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Actor-only XC330 velocity-to-teleop bootstrap contract tests."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from mjlab.rl.runner import MjlabOnPolicyRunner

from mjlab_microban.tasks.microban_policy_export import MicrobanTeleopOnPolicyRunner
from mjlab_microban.tasks.microban_teleop_bootstrap import (
    TELEOP_ACTOR_OBSERVATION_WIDTH,
    TELEOP_SHOULDER_ROLL_ACTION_INDICES,
    TELEOP_SHOULDER_ROLL_INITIAL_LATENT_BIASES,
    VELOCITY_ACTOR_OBSERVATION_WIDTH,
    VELOCITY_TO_TELEOP_OBSERVATION_INDEX,
    bootstrap_teleop_actor_state,
    expand_velocity_observation_to_teleop,
    load_velocity_actor_bootstrap,
)


def _synthetic_states() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(20260924)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=generator)

    source = {
        "obs_normalizer._mean": randn(1, VELOCITY_ACTOR_OBSERVATION_WIDTH),
        "obs_normalizer._var": torch.rand(
            (1, VELOCITY_ACTOR_OBSERVATION_WIDTH), generator=generator
        )
        + 0.5,
        "obs_normalizer._std": torch.rand(
            (1, VELOCITY_ACTOR_OBSERVATION_WIDTH), generator=generator
        )
        + 0.5,
        "obs_normalizer.count": torch.tensor(1_474_560_000.0),
        "distribution.std_param": torch.full((18,), 9.0),
        "mlp.0.weight": randn(7, VELOCITY_ACTOR_OBSERVATION_WIDTH),
        "mlp.0.bias": randn(7),
        "mlp.2.weight": randn(5, 7),
        "mlp.2.bias": randn(5),
        "mlp.4.weight": randn(3, 5),
        "mlp.4.bias": randn(3),
        "mlp.6.weight": randn(18, 3),
        "mlp.6.bias": randn(18),
    }
    target = {
        "obs_normalizer._mean": randn(1, TELEOP_ACTOR_OBSERVATION_WIDTH),
        "obs_normalizer._var": torch.rand(
            (1, TELEOP_ACTOR_OBSERVATION_WIDTH), generator=generator
        )
        + 0.5,
        "obs_normalizer._std": torch.rand(
            (1, TELEOP_ACTOR_OBSERVATION_WIDTH), generator=generator
        )
        + 0.5,
        "obs_normalizer.count": torch.tensor(12.0),
        "distribution.log_std_param": torch.full((18,), -3.0),
        "mlp.0.weight": randn(7, TELEOP_ACTOR_OBSERVATION_WIDTH),
        "mlp.0.bias": randn(7),
        "mlp.2.weight": randn(5, 7),
        "mlp.2.bias": randn(5),
        "mlp.4.weight": randn(3, 5),
        "mlp.4.bias": randn(3),
        "mlp.6.weight": randn(18, 3),
        "mlp.6.bias": randn(18),
    }
    return source, target


def _actor_mean(
    state: dict[str, torch.Tensor], observation: torch.Tensor
) -> torch.Tensor:
    value = (observation - state["obs_normalizer._mean"]) / state["obs_normalizer._std"]
    for layer in (0, 2, 4):
        value = torch.nn.functional.elu(
            torch.nn.functional.linear(
                value, state[f"mlp.{layer}.weight"], state[f"mlp.{layer}.bias"]
            )
        )
    return torch.nn.functional.linear(value, state["mlp.6.weight"], state["mlp.6.bias"])


class VelocityBootstrapMappingTest(unittest.TestCase):
    def test_unguarded_neutral_output_parity_and_new_columns_are_zero(self) -> None:
        source, target = _synthetic_states()
        mapped = bootstrap_teleop_actor_state(target, source)

        neutral_velocity = torch.zeros((1, VELOCITY_ACTOR_OBSERVATION_WIDTH))
        neutral_velocity[:, 5] = -1.0
        neutral_teleop = expand_velocity_observation_to_teleop(neutral_velocity)
        source_mean = _actor_mean(source, neutral_velocity)
        mapped_mean = _actor_mean(mapped, neutral_teleop)
        unguarded = sorted(set(range(18)) - set(TELEOP_SHOULDER_ROLL_ACTION_INDICES))
        torch.testing.assert_close(
            source_mean[:, unguarded],
            mapped_mean[:, unguarded],
            rtol=1e-6,
            atol=1e-6,
        )
        torch.testing.assert_close(
            mapped_mean[:, list(TELEOP_SHOULDER_ROLL_ACTION_INDICES)],
            mapped_mean.new_tensor([TELEOP_SHOULDER_ROLL_INITIAL_LATENT_BIASES]),
        )

        mapped_columns = {
            target_index
            for _source_index, target_index in VELOCITY_TO_TELEOP_OBSERVATION_INDEX
        }
        new_columns = sorted(
            set(range(TELEOP_ACTOR_OBSERVATION_WIDTH)) - mapped_columns
        )
        torch.testing.assert_close(
            mapped["mlp.0.weight"][:, new_columns],
            torch.zeros_like(mapped["mlp.0.weight"][:, new_columns]),
        )
        torch.testing.assert_close(
            mapped["obs_normalizer._mean"][:, new_columns],
            torch.zeros_like(mapped["obs_normalizer._mean"][:, new_columns]),
        )
        for key in ("obs_normalizer._var", "obs_normalizer._std"):
            torch.testing.assert_close(
                mapped[key][:, new_columns],
                torch.ones_like(mapped[key][:, new_columns]),
            )

    def test_only_actor_mlp_and_normalizer_are_copied_with_guarded_shoulder_head(
        self,
    ) -> None:
        source, target = _synthetic_states()
        target_distribution = target["distribution.log_std_param"].clone()
        mapped = bootstrap_teleop_actor_state(target, source)

        torch.testing.assert_close(
            mapped["distribution.log_std_param"], target_distribution
        )
        self.assertNotIn("distribution.std_param", mapped)
        self.assertEqual(
            mapped["obs_normalizer.count"].item(),
            source["obs_normalizer.count"].item(),
        )
        for key in (
            "mlp.0.bias",
            "mlp.2.weight",
            "mlp.2.bias",
            "mlp.4.weight",
            "mlp.4.bias",
        ):
            torch.testing.assert_close(mapped[key], source[key])
        shoulder_indices = list(TELEOP_SHOULDER_ROLL_ACTION_INDICES)
        unguarded = sorted(set(range(18)) - set(shoulder_indices))
        torch.testing.assert_close(
            mapped["mlp.6.weight"][unguarded], source["mlp.6.weight"][unguarded]
        )
        torch.testing.assert_close(
            mapped["mlp.6.bias"][unguarded], source["mlp.6.bias"][unguarded]
        )
        torch.testing.assert_close(
            mapped["mlp.6.weight"][shoulder_indices],
            torch.zeros_like(mapped["mlp.6.weight"][shoulder_indices]),
        )
        torch.testing.assert_close(
            mapped["mlp.6.bias"][shoulder_indices],
            mapped["mlp.6.bias"].new_tensor(TELEOP_SHOULDER_ROLL_INITIAL_LATENT_BIASES),
        )

    def test_loader_requires_exact_sha_before_installing_actor(self) -> None:
        source, target = _synthetic_states()

        class TargetActor:
            def __init__(self) -> None:
                self.state = target
                self.loaded: dict[str, torch.Tensor] | None = None

            def state_dict(self) -> dict[str, torch.Tensor]:
                return self.state

            def load_state_dict(
                self, state: dict[str, torch.Tensor], strict: bool
            ) -> None:
                self.loaded = state
                if not strict:
                    raise AssertionError("bootstrap must load strictly")

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model_14999.pt"
            torch.save(
                {
                    "actor_state_dict": source,
                    "critic_state_dict": {"must_not_copy": torch.tensor(99.0)},
                    "optimizer_state_dict": {"must_not_copy": 99},
                    "iter": 14999,
                },
                checkpoint,
            )
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            actor = TargetActor()
            with self.assertRaisesRegex(ValueError, "mismatch"):
                load_velocity_actor_bootstrap(actor, checkpoint, "0" * 64)  # type: ignore[arg-type]
            self.assertIsNone(actor.loaded)

            provenance = load_velocity_actor_bootstrap(  # type: ignore[arg-type]
                actor, checkpoint, digest
            )
            self.assertEqual(provenance.checkpoint_sha256, digest)
            self.assertEqual(
                provenance.installed_normalizer_count,
                provenance.source_normalizer_count,
            )
            self.assertEqual(
                provenance.shoulder_roll_initial_latent_biases,
                TELEOP_SHOULDER_ROLL_INITIAL_LATENT_BIASES,
            )
            self.assertIsNotNone(actor.loaded)
            self.assertNotIn("must_not_copy", actor.loaded or {})


class VelocityBootstrapRunnerGuardTest(unittest.TestCase):
    def test_contract_v8_rejects_velocity_bootstrap_options(self) -> None:
        env = SimpleNamespace(clip_actions=None, num_actions=18)
        with (
            patch.object(MjlabOnPolicyRunner, "__init__") as base_init,
            self.assertRaisesRegex(ValueError, "requires a clean actor"),
        ):
            MicrobanTeleopOnPolicyRunner(
                env,
                {"bootstrap_velocity_checkpoint": "/tmp/model.pt"},
            )
        base_init.assert_not_called()

        with (
            patch.object(MjlabOnPolicyRunner, "__init__") as base_init,
            self.assertRaisesRegex(ValueError, "requires a clean actor"),
        ):
            MicrobanTeleopOnPolicyRunner(
                env,
                {"bootstrap_velocity_checkpoint_sha256": "0" * 64},
            )
        base_init.assert_not_called()

    def test_contract_v8_rejects_bootstrap_even_with_resume(self) -> None:
        env = SimpleNamespace(clip_actions=None, num_actions=18)
        with (
            patch.object(MjlabOnPolicyRunner, "__init__") as base_init,
            self.assertRaisesRegex(ValueError, "requires a clean actor"),
        ):
            MicrobanTeleopOnPolicyRunner(
                env,
                {
                    "resume": True,
                    "bootstrap_velocity_checkpoint": "/tmp/model.pt",
                    "bootstrap_velocity_checkpoint_sha256": "0" * 64,
                },
            )
        base_init.assert_not_called()

    def test_contract_v8_rejects_pristine_checkpoint_option(self) -> None:
        env = SimpleNamespace(clip_actions=None, num_actions=18)
        with (
            patch.object(MjlabOnPolicyRunner, "__init__") as base_init,
            self.assertRaisesRegex(ValueError, "requires a clean actor"),
        ):
            MicrobanTeleopOnPolicyRunner(
                env,
                {"save_pristine_checkpoint": True},
            )
        base_init.assert_not_called()


if __name__ == "__main__":
    unittest.main()
