"""The target soft-limit barrier of walking and PICO (CPU)."""

from __future__ import annotations

import unittest

import torch

from mjlab_microban.tasks.microban_teleop_mdp import (
    normalized_target_soft_limit_excess_l1_sum,
    target_soft_limit_excess_l1,
)
from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release import (
    make_microban_teleop_v12_hand_pose_release_env_cfg,
)
from mjlab_microban.tasks.microban_velocity_env_cfg import (
    WALK_TARGET_SOFT_LIMIT_EXCESS_WEIGHT,
    make_microban_velocity_env_cfg,
)


class TargetSoftLimitExcessTest(unittest.TestCase):
    def test_zero_inside_and_linear_outside_without_saturation(self) -> None:
        lower = torch.tensor([[-2.0, -0.4]])
        upper = torch.tensor([[2.0, 0.4]])
        inside = target_soft_limit_excess_l1(torch.tensor([[1.99, -0.39]]), lower, upper)
        self.assertEqual(float(inside), 0.0)
        # One half-range is 2.0 and 0.4: 1 rad past the first is 0.5, 0.2 rad
        # below the second 0.5; far past the servo range it keeps growing.
        self.assertAlmostEqual(float(target_soft_limit_excess_l1(torch.tensor([[3.0, -0.6]]), lower, upper)), 1.0)
        far = target_soft_limit_excess_l1(torch.tensor([[20.0, 0.0]]), lower, upper)
        farther = target_soft_limit_excess_l1(torch.tensor([[21.0, 0.0]]), lower, upper)
        self.assertAlmostEqual(float(farther - far), 0.5)

    def test_walking_and_pico_register_it_at_the_twist_ratio_of_weights(self) -> None:
        for play in (False, True):
            walk = make_microban_velocity_env_cfg(play=play).rewards
            pico = make_microban_teleop_v12_hand_pose_release_env_cfg(play=play).rewards
            for rewards in (walk, pico):
                term = rewards["target_soft_limit_excess"]
                self.assertIs(term.func, normalized_target_soft_limit_excess_l1_sum)
                self.assertEqual(term.params, {"action_name": "joint_pos"})
            self.assertEqual(walk["target_soft_limit_excess"].weight, WALK_TARGET_SOFT_LIMIT_EXCESS_WEIGHT)
            self.assertLess(WALK_TARGET_SOFT_LIMIT_EXCESS_WEIGHT, 0.0)
            self.assertAlmostEqual(
                pico["target_soft_limit_excess"].weight / walk["target_soft_limit_excess"].weight,
                pico["twist_ratio_velocity"].weight / walk["twist_ratio_velocity"].weight,
            )


if __name__ == "__main__":
    unittest.main()
