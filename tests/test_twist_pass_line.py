"""The walking direction's pass line: the fixed signed-response minimums."""

from __future__ import annotations

import unittest

from mjlab_microban.ik_tracking_v12_safety import ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD
from mjlab_microban.twist_pass_line import (
    axis_minimums,
    twist_judgment,
    twist_pass_line_record,
    twist_passes,
    worst_margin,
)

# The single-axis minimums: the 9x300 probe and the IK tracking locomotion judgment,
# the walking judgment's commands.
NINE_BY_300 = {
    (0.1, 0.0, 0.0): 0.04, (0.2, 0.0, 0.0): 0.08, (-0.1, 0.0, 0.0): 0.02, (-0.2, 0.0, 0.0): 0.04,
    (0.0, 0.1, 0.0): 0.02, (0.0, -0.1, 0.0): 0.02, (0.0, 0.0, 0.5): 0.2, (0.0, 0.0, -0.5): 0.2,
}
WALK_SINGLE = {(0.2, 0.0, 0.0): 0.08, (-0.2, 0.0, 0.0): 0.04, (0.0, 0.1, 0.0): 0.02, (0.0, 0.0, 0.5): 0.2}


class PassLineTest(unittest.TestCase):
    def test_joint_overshoot_limit_is_a_quarter_radian(self) -> None:
        self.assertEqual(ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD, 0.25)

    def test_single_and_multi_axis_minimums(self) -> None:
        for table in (NINE_BY_300, WALK_SINGLE):
            for command, minimum in table.items():
                (value,) = axis_minimums(command).values()
                self.assertAlmostEqual(value, minimum, places=12, msg=command)
        # Commands on several axes: one minimum per commanded axis.
        self.assertEqual(axis_minimums((0.35, 0.15, 0.75)), {0: 0.04, 1: 0.02, 2: 0.2})
        self.assertEqual(axis_minimums((-0.25, -0.15, 0.0)), {0: 0.04, 1: 0.02})

    def test_a_moving_command_passes_at_its_minimum_the_commanded_way(self) -> None:
        self.assertTrue(twist_passes((0.1, 0.0, 0.0), (0.04, 0.3, -1.0)))
        self.assertFalse(twist_passes((0.1, 0.0, 0.0), (0.0399, 0.0, 0.0)))
        self.assertTrue(twist_passes((-0.1, 0.0, 0.0), (-0.02, 0.0, 0.0)))
        self.assertFalse(twist_passes((-0.1, 0.0, 0.0), (0.02, 0.0, 0.0)))
        self.assertTrue(twist_passes((0.35, -0.15, 0.75), (0.05, -0.03, 0.2)))
        self.assertFalse(twist_passes((0.35, -0.15, 0.75), (0.05, 0.03, 0.2)))
        self.assertAlmostEqual(worst_margin((0.0, 0.0, -0.5), (0.0, 0.0, -0.3)), 0.1, places=12)

    def test_standing_command_drift_limits(self) -> None:
        self.assertTrue(twist_passes((0, 0, 0), (0.05, -0.05, 0.2)))
        self.assertFalse(twist_passes((0, 0, 0), (0.051, 0.0, 0.0)))
        self.assertFalse(twist_passes((0, 0, 0), (0.0, -0.051, 0.0)))
        self.assertFalse(twist_passes((0, 0, 0), (0.0, 0.0, 0.21)))

    def test_nonfinite_twist_fails(self) -> None:
        self.assertFalse(twist_passes((0.1, 0.0, 0.0), (float("nan"), 0.0, 0.0)))
        self.assertFalse(twist_judgment((0.1, 0.0, 0.0), (float("inf"), 0.0, 0.0))["passed"])
        self.assertFalse(twist_passes((0.0, 0.0, 0.0), (float("nan"), 0.0, 0.0)))

    def test_a_judged_floor_may_loosen_the_moving_minimums(self) -> None:
        # A floor's twist_minimum_scale (config/pipeline.yaml lines): every minimum times the scale, the standing
        # drift kept.  At 0.5 the 0.1 m/s side step needs 0.01 m/s: the walker 82e9585c's 0.015 on the 0.1 s floor
        # passes, a walker that does not step sideways does not.
        for table in (NINE_BY_300, WALK_SINGLE):
            for command, minimum in table.items():
                (value,) = axis_minimums(command, 0.5).values()
                self.assertAlmostEqual(value, 0.5 * minimum, places=12, msg=command)
        self.assertEqual(axis_minimums((0.35, 0.15, 0.75), 0.5), {0: 0.02, 1: 0.01, 2: 0.1})
        self.assertFalse(twist_passes((0.0, -0.1, 0.0), (0.0, -0.015, 0.0)))
        self.assertTrue(twist_passes((0.0, -0.1, 0.0), (0.0, -0.015, 0.0), 0.5))
        self.assertFalse(twist_passes((0.0, -0.1, 0.0), (0.0, -0.0099, 0.0), 0.5))
        self.assertAlmostEqual(worst_margin((0.0, -0.1, 0.0), (0.0, -0.015, 0.0), 0.5), 0.005, places=12)
        self.assertFalse(twist_passes((0, 0, 0), (0.051, 0.0, 0.0), 0.5))
        # Scale 1 is the rigid floor's line, recorded as it always was (the IK tracking stage gate recomputes stored
        # reports); another scale is recorded with the judgment.
        record = twist_judgment((0.1, 0.0, 0.0), (0.06, 0.01, 0.05))
        self.assertEqual(twist_judgment((0.1, 0.0, 0.0), (0.06, 0.01, 0.05), 1.0), record)
        self.assertEqual(set(record), {"command", "mean_twist", "signed_response", "minimum_signed_response", "passed"})
        loose = twist_judgment((0.1, 0.0, 0.0), (0.03, 0.0, 0.0), 0.5)
        self.assertEqual((loose["minimum_signed_response"], loose["minimum_scale"], loose["passed"]),
                         ({"vx_m_s": 0.02}, 0.5, True))
        for scale in (0.0, -0.5, 1.01, float("nan")):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                axis_minimums((0.1, 0.0, 0.0), scale)

    def test_record_is_reproducible(self) -> None:
        self.assertEqual(twist_pass_line_record(), twist_pass_line_record())
        record = twist_judgment((0.1, 0.0, 0.0), (0.06, 0.01, 0.05))
        self.assertEqual(record, twist_judgment([0.1, 0.0, 0.0], [0.06, 0.01, 0.05]))
        self.assertEqual(record["minimum_signed_response"], {"vx_m_s": 0.04})


if __name__ == "__main__":
    unittest.main()
