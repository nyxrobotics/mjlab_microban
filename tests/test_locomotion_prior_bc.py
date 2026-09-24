# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for v8i's deployment-invisible locomotion teacher."""

from __future__ import annotations

import unittest

import torch
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from mjlab_microban.tasks.microban_locomotion_prior import (
    MICROBAN_LOCOMOTION_PRIOR_LEG_JOINT_NAMES,
)
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_TELEOP_ACTION_JOINT_NAMES,
)
from mjlab_microban.tasks.microban_teleop_env_cfg import (
    microban_teleop_action_delta_bounds,
    microban_teleop_initial_action_std,
)
from mjlab_microban.tasks.microban_teleop_mdp import (
    AsymmetricBoundedGaussianDistribution,
    LatentActionPPO,
)


class LocomotionPriorBcTest(unittest.TestCase):
    def _algorithm(self, batch_size: int = 16) -> tuple[LatentActionPPO, TensorDict]:
        observations = TensorDict(
            {
                "actor": torch.zeros(batch_size, 83),
                # Six unrelated critic values prove that the named slice does
                # not rely on the prior remaining at a magic absolute offset.
                "critic": torch.zeros(batch_size, 6 + 39),
            },
            batch_size=[batch_size],
        )
        groups = {"actor": ["actor"], "critic": ["critic"]}
        lower, upper = microban_teleop_action_delta_bounds()
        actor = MLPModel(
            obs=observations,
            obs_groups=groups,
            obs_set="actor",
            output_dim=18,
            hidden_dims=(16,),
            distribution_cfg={
                "class_name": AsymmetricBoundedGaussianDistribution,
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
        storage = RolloutStorage("rl", batch_size, 1, observations, [18], "cpu")
        leg_ids = tuple(
            MICROBAN_TELEOP_ACTION_JOINT_NAMES.index(name)
            for name in MICROBAN_LOCOMOTION_PRIOR_LEG_JOINT_NAMES
        )
        algorithm = LatentActionPPO(
            actor,
            critic,
            storage,
            device="cpu",
            learning_rate=1.0e-2,
            rnd_cfg=None,
            symmetry_cfg=None,
            locomotion_prior_bc_cfg={
                "coefficient": 0.5,
                "error_scale_rad": 0.15,
                "chunks": 4,
                "critic_group": "critic",
                "actor_group": "actor",
                "prior_start": 6,
                "prior_stop": 45,
                "command_start": 66,
                "command_stop": 69,
                "action_scale": (1.0,) * 18,
                "action_offset": (0.0,) * 18,
                "leg_action_ids": leg_ids,
                "max_target_projection_rad": 1.0e-3,
                "neutral_anchor_relative_weight": 1.0,
            },
        )
        return algorithm, observations

    @staticmethod
    def _set_prior(
        observations: TensorDict, *, weight: float, target: float = 0.05
    ) -> None:
        prior = observations["critic"][:, 6:45]
        prior.zero_()
        if weight > 0.0:
            prior[:, 0] = weight
            prior[:, 2] = 1.0
            prior[:, 27:39] = target

    def test_active_payload_drives_only_actor_mean_toward_physical_target(self) -> None:
        algorithm, observations = self._algorithm()
        self._set_prior(observations, weight=1.0)
        with torch.no_grad():
            for parameter in algorithm.actor.mlp.parameters():
                parameter.zero_()
        prepared = algorithm._prepare_locomotion_prior_bc(observations)
        assert prepared is not None
        weight, target, stats = prepared
        leg_ids = algorithm._locomotion_prior_bc_cfg["leg_action_ids"]  # type: ignore[index]
        with torch.no_grad():
            before = algorithm.actor(observations)[:, leg_ids]
            before_mse = torch.square(before - target).mean()
        critic_before = {
            name: value.detach().clone()
            for name, value in algorithm.critic.state_dict().items()
        }
        std_before = algorithm.actor.distribution.log_std_param.detach().clone()

        result = algorithm._run_locomotion_prior_bc(observations, weight, target, stats)

        with torch.no_grad():
            after = algorithm.actor(observations)[:, leg_ids]
            after_mse = torch.square(after - target).mean()
        self.assertLess(float(after_mse.item()), float(before_mse.item()))
        self.assertGreater(result["locomotion_prior_bc"], 0.0)
        self.assertEqual(result["locomotion_prior_bc_active_fraction"], 1.0)
        torch.testing.assert_close(
            algorithm.actor.distribution.log_std_param, std_before
        )
        for name, value in algorithm.critic.state_dict().items():
            torch.testing.assert_close(value, critic_before[name])

    def test_zero_weight_skips_optimizer_and_rejects_nonzero_inactive_payload(
        self,
    ) -> None:
        algorithm, observations = self._algorithm()
        self._set_prior(observations, weight=0.0)
        prepared = algorithm._prepare_locomotion_prior_bc(observations)
        assert prepared is not None
        weight, target, stats = prepared
        before = {
            name: value.detach().clone()
            for name, value in algorithm.actor.state_dict().items()
        }
        result = algorithm._run_locomotion_prior_bc(observations, weight, target, stats)
        self.assertEqual(result["locomotion_prior_bc"], 0.0)
        for name, value in algorithm.actor.state_dict().items():
            torch.testing.assert_close(value, before[name])

        observations["critic"][:, 6 + 3] = 1.0
        with self.assertRaisesRegex(ValueError, "exact zero"):
            algorithm._prepare_locomotion_prior_bc(observations)

    def test_blend_scales_loss_without_active_row_renormalization(self) -> None:
        algorithm, observations = self._algorithm()
        with torch.no_grad():
            for parameter in algorithm.actor.mlp.parameters():
                parameter.zero_()
        self._set_prior(observations, weight=1.0)
        full = algorithm._prepare_locomotion_prior_bc(observations)
        assert full is not None
        full_weight, full_target, full_stats = full
        full_result = algorithm._run_locomotion_prior_bc(
            observations, full_weight, full_target, full_stats
        )

        with torch.no_grad():
            for parameter in algorithm.actor.mlp.parameters():
                parameter.zero_()
        self._set_prior(observations, weight=0.5)
        half = algorithm._prepare_locomotion_prior_bc(observations)
        assert half is not None
        half_weight, half_target, half_stats = half
        half_result = algorithm._run_locomotion_prior_bc(
            observations, half_weight, half_target, half_stats
        )
        self.assertAlmostEqual(
            half_result["locomotion_prior_bc"],
            0.5 * full_result["locomotion_prior_bc"],
            places=5,
        )

    def test_neutral_rows_are_anchored_but_other_commands_are_not(self) -> None:
        algorithm, observations = self._algorithm(batch_size=12)
        prior = observations["critic"][:, 6:45]
        prior.zero_()
        prior[:4, 0] = 0.75
        prior[:4, 2] = 1.0
        prior[:4, 27:39] = 0.05
        observations["actor"][:4, 66] = 0.08
        # Rows 4:8 remain exact-zero twist; rows 8:12 are lateral commands.
        observations["actor"][8:, 67] = 0.2

        prepared = algorithm._prepare_locomotion_prior_bc(observations)
        assert prepared is not None
        teacher_weight, target, stats = prepared
        torch.testing.assert_close(teacher_weight[:8], torch.full((8,), 0.75))
        torch.testing.assert_close(teacher_weight[8:], torch.zeros(4))
        torch.testing.assert_close(target[:4], torch.full((4, 12), 0.05))
        torch.testing.assert_close(target[4:], torch.zeros(8, 12))
        self.assertAlmostEqual(
            stats["locomotion_prior_bc_neutral_anchor_fraction"], 1.0 / 3.0
        )


if __name__ == "__main__":
    unittest.main()
