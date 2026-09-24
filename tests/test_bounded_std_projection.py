"""Regression tests for trainable bounded-Gaussian exploration width."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from mjlab_microban.tasks.microban_policy_export import (
    validate_bounded_actor_checkpoint_buffers,
)
from mjlab_microban.tasks.microban_teleop_mdp import (
    AsymmetricBoundedGaussianDistribution,
    LatentActionPPO,
)


class BoundedStdProjectionTest(unittest.TestCase):
    @staticmethod
    def _distribution(
        *, std_type: str = "log"
    ) -> AsymmetricBoundedGaussianDistribution:
        return AsymmetricBoundedGaussianDistribution(
            output_dim=3,
            init_std=(0.30, 0.20, 0.10),
            lower_bound=(-1.0, -1.0, -1.0),
            upper_bound=(1.0, 1.0, 1.0),
            std_type=std_type,
        )

    def test_explicit_projection_recovers_from_above_max_without_schema_change(
        self,
    ) -> None:
        for std_type, parameter_name in (
            ("scalar", "std_param"),
            ("log", "log_std_param"),
        ):
            with self.subTest(std_type=std_type):
                distribution = self._distribution(std_type=std_type)
                state_keys = tuple(distribution.state_dict())
                parameter = getattr(distribution, parameter_name)
                upper = (
                    distribution.max_std
                    if std_type == "scalar"
                    else torch.log(distribution.max_std)
                )
                with torch.no_grad():
                    parameter.copy_(upper + 1.0)

                distribution.project_std_parameters_()

                torch.testing.assert_close(parameter, upper)
                self.assertEqual(tuple(distribution.state_dict()), state_keys)

    def test_max_boundary_can_learn_back_toward_the_interior(self) -> None:
        distribution = self._distribution()
        with torch.no_grad():
            distribution.log_std_param.copy_(torch.log(distribution.max_std))
        optimizer = torch.optim.SGD([distribution.log_std_param], lr=0.1)

        distribution.update(torch.zeros((2, 3)))
        optimizer.zero_grad()
        distribution.std.sum().backward()
        optimizer.step()

        self.assertTrue(
            torch.all(distribution.log_std_param < torch.log(distribution.max_std))
        )

    def test_latent_ppo_projects_std_after_every_optimizer_step(self) -> None:
        num_envs = 2
        observations = TensorDict(
            {"policy": torch.zeros(num_envs, 4)}, batch_size=[num_envs]
        )
        observation_groups = {"actor": ["policy"], "critic": ["policy"]}
        actor = MLPModel(
            obs=observations,
            obs_groups=observation_groups,
            obs_set="actor",
            output_dim=18,
            hidden_dims=(8,),
            distribution_cfg={
                "class_name": AsymmetricBoundedGaussianDistribution,
                "init_std": (0.30,) * 18,
                "lower_bound": (-1.0,) * 18,
                "upper_bound": (1.0,) * 18,
                "std_type": "log",
            },
        )
        critic = MLPModel(
            obs=observations,
            obs_groups=observation_groups,
            obs_set="critic",
            output_dim=1,
            hidden_dims=(8,),
        )
        storage = RolloutStorage("rl", num_envs, 2, observations, [18], "cpu")
        algorithm = LatentActionPPO(
            actor,
            critic,
            storage,
            device="cpu",
            rnd_cfg=None,
            symmetry_cfg=None,
        )
        distribution = actor.distribution
        assert isinstance(distribution, AsymmetricBoundedGaussianDistribution)
        with torch.no_grad():
            distribution.log_std_param.copy_(
                torch.log(distribution.max_std) + 1.0
            )
        distribution.log_std_param.grad = torch.zeros_like(
            distribution.log_std_param
        )

        algorithm.optimizer.step()

        self.assertTrue(
            torch.all(
                distribution.log_std_param <= torch.log(distribution.max_std)
            )
        )

    def test_checkpoint_validation_rejects_out_of_range_log_std(self) -> None:
        distribution = self._distribution()
        expected = {
            f"distribution.{key}": value.detach().clone()
            for key, value in distribution.state_dict().items()
        }
        candidate = {key: value.clone() for key, value in expected.items()}
        candidate["distribution.log_std_param"] = (
            torch.log(candidate["distribution.max_std"]) + 1.0
        )

        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model_0.pt"
            torch.save({"actor_state_dict": candidate}, checkpoint)
            with self.assertRaisesRegex(ValueError, "outside.*std contract"):
                validate_bounded_actor_checkpoint_buffers(checkpoint, expected)


if __name__ == "__main__":
    unittest.main()
