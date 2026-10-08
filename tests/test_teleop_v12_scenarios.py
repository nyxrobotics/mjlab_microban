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
    foot_scenarios,
    mirrored_foot_pairs,
    patch_observation,
    per_env,
    push_velocity,
    scenario_index,
)


class FootScenarioTest(unittest.TestCase):
    def test_the_left_and_right_targets_are_mirror_images(self) -> None:
        by_name = {s.name: s for s in foot_scenarios()}
        self.assertEqual(len(mirrored_foot_pairs()), 6)
        for left, right in mirrored_foot_pairs():
            goal = by_name[left].foot_goal[0]
            self.assertEqual(by_name[right].foot_goal, ((0.0, 0.0, 0.0), (goal[0], -goal[1], goal[2])))
        # The corners of the sent box (80 % of the trained one) and both feet.
        self.assertEqual(by_name["left_front_out"].foot_goal[0], (0.8 * 0.03, 0.8 * 0.03, 0.8 * 0.05))
        both = by_name["both_front_out"].foot_goal
        self.assertEqual(both[1], (both[0][0], -both[0][1], both[0][2]))
        self.assertEqual(by_name["both_back_in"].lifted, (True, True))

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
