"""The walking checks' pass line is the walking reward's own verdict."""

from __future__ import annotations

import math
import unittest

import torch

from mjlab_microban.tasks.microban_twist_ratio_mdp import twist_ratio_reward
from mjlab_microban.teleop_v12_safety import ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD
from mjlab_microban.twist_pass_line import (
    STANDING_DRIFT_VALUE_MIN,
    STANDING_STILL_VALUE,
    twist_judgment,
    twist_pass_line_record,
    twist_passes,
    twist_value,
)


class PassLineTest(unittest.TestCase):
    def test_joint_overshoot_allowance_is_the_users_quarter_radian(self) -> None:
        self.assertEqual(ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD, 0.25)

    def test_value_is_the_planar_walking_reward(self) -> None:
        torch.manual_seed(0)
        commands = torch.randn(64, 3, dtype=torch.float64) * torch.tensor([0.3, 0.15, 0.8], dtype=torch.float64)
        twists = torch.randn(64, 3, dtype=torch.float64) * 0.2
        expected = twist_ratio_reward(commands, twists, uncommanded=None, uncommanded_scale=None)
        for c, v, want in zip(commands.tolist(), twists.tolist(), expected.tolist(), strict=True):
            self.assertEqual(twist_value(c, v), want)

    def test_a_moving_command_passes_exactly_when_the_reward_beats_standing_still(self) -> None:
        torch.manual_seed(1)
        for _ in range(512):
            c = (torch.randn(3) * torch.tensor([0.3, 0.15, 0.8])).tolist()
            v = (torch.randn(3) * torch.tensor([0.2, 0.1, 0.6])).tolist()
            standing = twist_value(c, [0.0, 0.0, 0.0])
            self.assertAlmostEqual(standing, STANDING_STILL_VALUE, places=12)
            self.assertEqual(twist_passes(c, v), twist_value(c, v) > standing)

    def test_checked_commands(self) -> None:
        # 0.1 / 0.2 m/s forward and backward, 0.1 m/s lateral, 0.5 rad/s yaw:
        # any progress along the command passes, also at twice the command;
        # standing still, the wrong way, or mostly sideways fails.
        for command in ((0.1, 0, 0), (0.2, 0, 0), (-0.1, 0, 0), (-0.2, 0, 0), (0, 0.1, 0), (0, -0.1, 0),
                        (0, 0, 0.5), (0, 0, -0.5)):
            for k in (0.05, 0.5, 1.0, 2.0):
                self.assertTrue(twist_passes(command, [k * x for x in command]), (command, k))
            self.assertFalse(twist_passes(command, [0.0, 0.0, 0.0]), command)
            self.assertFalse(twist_passes(command, [-0.3 * x for x in command]), command)
        # The first release walker on 0.1 m/s forward: -0.0396 m/s.
        self.assertFalse(twist_passes((0.1, 0.0, 0.0), (-0.0396, 0.0, 0.0)))
        # Half of 0.1 m/s forward: drifting 0.1 m/s sideways still beats
        # standing (0.537), 0.15 m/s sideways does not (0.455).
        self.assertTrue(twist_passes((0.1, 0.0, 0.0), (0.05, 0.1, 0.0)))
        self.assertFalse(twist_passes((0.1, 0.0, 0.0), (0.05, 0.15, 0.0)))

    def test_standing_command_drift_line(self) -> None:
        self.assertAlmostEqual(STANDING_DRIFT_VALUE_MIN, 0.5 * math.exp(-1.0 / 7.0), places=12)
        self.assertTrue(twist_passes((0, 0, 0), (0.0, 0.0, 0.0)))
        self.assertTrue(twist_passes((0, 0, 0), (0.099, 0.0, 0.0)))
        self.assertFalse(twist_passes((0, 0, 0), (0.101, 0.0, 0.0)))
        self.assertTrue(twist_passes((0, 0, 0), (0.0, 0.042, 0.0)))
        self.assertFalse(twist_passes((0, 0, 0), (0.0, 0.044, 0.0)))
        self.assertTrue(twist_passes((0, 0, 0), (0.0, 0.0, 0.21)))
        self.assertFalse(twist_passes((0, 0, 0), (0.0, 0.0, 0.22)))

    def test_nonfinite_twist_fails(self) -> None:
        self.assertFalse(twist_passes((0.1, 0.0, 0.0), (float("nan"), 0.0, 0.0)))
        self.assertFalse(twist_judgment((0.1, 0.0, 0.0), (float("inf"), 0.0, 0.0))["passed"])

    def test_record_is_reproducible(self) -> None:
        self.assertEqual(twist_pass_line_record(), twist_pass_line_record())
        record = twist_judgment((0.1, 0.0, 0.0), (0.06, 0.01, 0.05))
        self.assertEqual(record, twist_judgment([0.1, 0.0, 0.0], [0.06, 0.01, 0.05]))
        self.assertEqual(record["line"], STANDING_STILL_VALUE)


if __name__ == "__main__":
    unittest.main()
