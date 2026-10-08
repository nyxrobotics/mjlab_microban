"""Unit tests of the PICO judgment's scenarios and command helpers."""

from __future__ import annotations

import math
import unittest

import torch
from tensordict import TensorDict

from mjlab_microban.scripts.teleop_v12_scenarios import (
    ENVS_PER_SCENARIO,
    PUSH_DIRECTIONS,
    PUSH_SPEED_M_S,
    FootTargetRamp,
    arm_scenarios,
    foot_scenarios,
    patch_observation,
    per_env,
    push_velocity,
    scenario_index,
)


class FootScenarioTest(unittest.TestCase):
    def test_the_left_and_right_targets_are_mirror_images(self) -> None:
        by_name = {s.name: s for s in foot_scenarios()}
        lefts = [name for name in by_name if name.startswith("left_") and "/" not in name]
        self.assertEqual(len(lefts), 6)
        for left in lefts:
            goal = by_name[left].foot_goal[0]
            right = by_name["right_" + left.removeprefix("left_")]
            self.assertEqual(right.foot_goal, ((0.0, 0.0, 0.0), (goal[0], -goal[1], goal[2])))
        # The corners of the sent box (80 % of the trained one).
        self.assertEqual(by_name["left_front_out"].foot_goal[0], (0.8 * 0.03, 0.8 * 0.03, 0.8 * 0.05))
        # Two corners with the arm on the lifted side reaching out.
        reach = by_name["left_front_in/arms_reach_left"]
        self.assertEqual((reach.foot_goal, reach.arms), (by_name["left_front_in"].foot_goal, "reach_left"))
        reach = by_name["right_back_in/arms_reach_right"]
        self.assertEqual((reach.foot_goal, reach.arms), (by_name["right_back_in"].foot_goal, "reach_right"))

    def test_both_feet_get_the_same_offset(self) -> None:
        both = [s for s in foot_scenarios() if all(s.lifted)]
        self.assertEqual([s.name for s in both], ["both_front_left", "both_back_right"])
        for scenario in both:
            left, right = scenario.foot_goal
            self.assertEqual(left, right)
            for value, limit in zip(left, (0.8 * 0.01, 0.8 * 0.01, 0.8 * 0.02)):
                self.assertAlmostEqual(abs(value), limit)

    def test_arm_scenarios(self) -> None:
        names = [s.name for s in arm_scenarios()]
        self.assertEqual(len(names), 3 * 9 + 2)
        self.assertEqual(names[-2:], ["neutral/arms_reach_left", "neutral/arms_reach_right"])
        self.assertIn("forward_0p1/arms_raised", names)

    def test_teleop_ramp_and_floor_band(self) -> None:
        ramp = FootTargetRamp(torch.tensor([[[0.024, 0.024, 0.04], [0.0, 0.0, 0.0]]]), step_dt=0.02)
        ramp.advance(True)
        # 0.12 m/s along a straight line; the first 2.4 mm is still in the 2.5 mm floor band.
        self.assertAlmostEqual(float(torch.linalg.vector_norm(ramp.internal[0, 0])), 0.0024, places=7)
        self.assertTrue(torch.equal(ramp.observed, torch.zeros(1, 2, 3)))
        for _ in range(30):
            ramp.advance(True)
        torch.testing.assert_close(ramp.observed, ramp.goal)
        ramp.advance(False)
        self.assertLess(float(ramp.observed[0, 0, 2]), 0.04)


class BatchHelperTest(unittest.TestCase):
    def test_each_scenario_gets_its_environments(self) -> None:
        scenarios = foot_scenarios()[:2]
        index = scenario_index(2, 3 * ENVS_PER_SCENARIO, "cpu")
        self.assertEqual(int((index == 1).sum()), ENVS_PER_SCENARIO)
        self.assertEqual(int((index == -1).sum()), ENVS_PER_SCENARIO)
        spec = per_env(scenarios, index)
        torch.testing.assert_close(spec["foot_goal"][ENVS_PER_SCENARIO, 1], torch.tensor([0.0, 0.0, 0.02]))
        self.assertTrue(torch.equal(spec["foot_goal"][-1], torch.zeros(2, 3)))
        with self.assertRaises(ValueError):
            scenario_index(4, 3 * ENVS_PER_SCENARIO, "cpu")

    def test_pushes_come_from_eight_directions(self) -> None:
        kick = push_velocity(torch.zeros(ENVS_PER_SCENARIO, dtype=torch.long))
        torch.testing.assert_close(torch.linalg.vector_norm(kick, dim=-1), torch.full((ENVS_PER_SCENARIO,), PUSH_SPEED_M_S))
        angles = {round(math.degrees(math.atan2(y, x))) % 360 for x, y in kick.tolist()}
        self.assertEqual(angles, {45 * k for k in range(PUSH_DIRECTIONS)})

    def test_only_the_given_columns_are_replaced(self) -> None:
        observations = TensorDict({"actor": torch.arange(12.0).reshape(2, 6)}, batch_size=[2])
        patched = patch_observation(observations, {slice(1, 3): torch.zeros(2, 2)})
        self.assertTrue(torch.equal(observations["actor"], torch.arange(12.0).reshape(2, 6)))
        self.assertTrue(torch.equal(patched["actor"][:, 1:3], torch.zeros(2, 2)))
        self.assertTrue(torch.equal(patched["actor"][:, 3:], observations["actor"][:, 3:]))


if __name__ == "__main__":
    unittest.main()
