"""Contract-v9 tests proving rejected walk004 BC cannot reach the optimizer."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from mjlab_microban.tasks.microban_safe_velocity_mdp import (
    MicrobanSafeVelocityBoundedGaussianDistribution,
)
from mjlab_microban.tasks.microban_teleop_env_cfg import (
    microban_teleop_action_delta_bounds,
    microban_teleop_initial_action_std,
)
from mjlab_microban.tasks.microban_teleop_mdp import LatentActionPPO


class LocomotionPriorBcDisabledTest(unittest.TestCase):
    @staticmethod
    def _algorithm(
        *, locomotion_prior_bc_cfg: dict[str, object] | None = None
    ) -> LatentActionPPO:
        observations = TensorDict(
            {
                "actor": torch.zeros(8, 83),
                # The retired 39-wide critic payload remains topology-only.
                "critic": torch.zeros(8, 45),
            },
            batch_size=[8],
        )
        groups = {"actor": ["actor"], "critic": ["critic"]}
        lower, upper = microban_teleop_action_delta_bounds()
        actor = MLPModel(
            obs=observations,
            obs_groups=groups,
            obs_set="actor",
            output_dim=18,
            hidden_dims=(16,),
            obs_normalization=False,
            distribution_cfg={
                "class_name": MicrobanSafeVelocityBoundedGaussianDistribution,
                "init_std": microban_teleop_initial_action_std(),
                "lower_bound": lower,
                "upper_bound": upper,
                "std_type": "log",
            },
        )
        critic = MLPModel(
            obs=observations,
            obs_groups=groups,
            obs_set="critic",
            output_dim=1,
            hidden_dims=(16,),
        )
        storage = RolloutStorage("rl", 8, 1, observations, [18], "cpu")
        return LatentActionPPO(
            actor,
            critic,
            storage,
            device="cpu",
            learning_rate=1.0e-3,
            rnd_cfg=None,
            symmetry_cfg=None,
            locomotion_prior_bc_cfg=locomotion_prior_bc_cfg,
        )

    def test_none_is_the_only_accepted_bc_configuration(self) -> None:
        algorithm = self._algorithm()
        self.assertIsNone(algorithm._locomotion_prior_bc_cfg)
        with self.assertRaisesRegex(ValueError, "Contract v9 forbids walk004"):
            self._algorithm(
                locomotion_prior_bc_cfg={
                    "forward_coefficient": 0.0,
                    "neutral_leg_coefficient": 0.0,
                    "arm_home_coefficient": 0.0,
                }
            )

    def test_update_delegates_without_any_post_ppo_teacher_step(self) -> None:
        algorithm = self._algorithm()
        expected = {"surrogate": 1.25}
        with patch.object(PPO, "update", return_value=expected) as base_update:
            result = algorithm.update()
        base_update.assert_called_once_with()
        self.assertIs(result, expected)
        self.assertNotIn("locomotion_prior_bc", result)


if __name__ == "__main__":
    unittest.main()
