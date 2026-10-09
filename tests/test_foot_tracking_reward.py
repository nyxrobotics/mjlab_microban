"""The foot reward (mdp.foot_target_tracking_error_exp) on its row kinds.

One foot up (the published targets differ in z by the threshold or more): the
lifted foot from the support foot, in the trunk's heading frame levelled by
gravity.  Every other row: each foot from the trunk in the HOME-levelled
trunk frame, as before.
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_from_euler_xyz, quat_mul

from mjlab_microban.robot.microban_constants import HOME_ROOT_QUAT_WXYZ, HOME_TRUNK_PITCH_RAD
from mjlab_microban.tasks.mdp import FootTargetCommand, foot_target_tracking_error_exp

STD = 0.03
THRESHOLD = 0.010
TRUNK = (0.0, 0.0, 0.1704)
FEET = ((0.0068, 0.0464, 0.0035), (0.0068, -0.0464, 0.0035))  # left, right (world, HOME)


def _quat(roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0) -> torch.Tensor:
    """The HOME trunk turned by yaw about world z, then tilted by roll and
    pitch about the turned (heading) axes."""
    t = lambda v: torch.tensor([v], dtype=torch.float64)  # noqa: E731
    home = torch.tensor([HOME_ROOT_QUAT_WXYZ], dtype=torch.float64)
    return quat_mul(quat_from_euler_xyz(t(roll), t(pitch), t(yaw)), home)


class _Rows:
    """A fake FootTargetCommand: n rows of trunk pose, feet and offsets."""

    def __init__(self) -> None:
        self.trunk_pos: list[torch.Tensor] = []
        self.trunk_quat: list[torch.Tensor] = []
        self.feet: list[torch.Tensor] = []
        self.offsets: list[torch.Tensor] = []

    def add(self, offsets, feet=FEET, trunk=TRUNK, quat=None) -> None:
        self.trunk_pos.append(torch.tensor(trunk, dtype=torch.float64))
        self.trunk_quat.append((_quat() if quat is None else quat)[0])
        self.feet.append(torch.tensor(feet, dtype=torch.float64))
        self.offsets.append(torch.tensor(offsets, dtype=torch.float64))

    def reward(self, twist: torch.Tensor | None = None, **kw) -> torch.Tensor:
        n = len(self.feet)
        command = FootTargetCommand.__new__(FootTargetCommand)
        command._env = SimpleNamespace(num_envs=n)
        command.cfg = SimpleNamespace(trunk_pitch=HOME_TRUNK_PITCH_RAD)
        command._foot_asset_cfg = SimpleNamespace(site_ids=[0, 1])
        # The reset reference: every row at HOME.
        home = SimpleNamespace(root_link_pos_w=torch.tensor([TRUNK] * n, dtype=torch.float64),
                               root_link_quat_w=_quat().expand(n, 4),
                               site_pos_w=torch.tensor([FEET] * n, dtype=torch.float64))
        command.robot = SimpleNamespace(data=home)
        command._default_foot_pos_b = command.current_foot_pos_b()
        command.robot = SimpleNamespace(data=SimpleNamespace(
            root_link_pos_w=torch.stack(self.trunk_pos), root_link_quat_w=torch.stack(self.trunk_quat),
            site_pos_w=torch.stack(self.feet)))
        command.foot_target_offset_b = torch.stack(self.offsets)
        twist = torch.zeros(n, 3, dtype=torch.float64) if twist is None else twist
        manager = SimpleNamespace(get_term=lambda name: command, get_command=lambda name: twist)
        env = SimpleNamespace(num_envs=n, command_manager=manager)
        self.command = command
        return foot_target_tracking_error_exp(env, "foot_target", STD, THRESHOLD, **kw)


def _moved(feet, d_left=(0.0, 0.0, 0.0), d_right=(0.0, 0.0, 0.0)):
    return tuple(tuple(a + b for a, b in zip(foot, d)) for foot, d in zip(feet, (d_left, d_right)))


def _turned(points, trunk_quat_yaw: torch.Tensor):
    """Points turned about world z through the origin (the heading)."""
    p = torch.tensor(points, dtype=torch.float64).reshape(-1, 3)
    return quat_apply(trunk_quat_yaw.expand(len(p), 4), p).reshape(torch.tensor(points).shape).tolist()


LEFT_UP = ((0.012, 0.006, 0.03), (0.0, 0.0, 0.0))


class FootTrackingRewardTest(unittest.TestCase):
    def test_one_foot_up_at_its_place_from_the_support_foot_scores_1_wherever_the_trunk_is(self) -> None:
        rows = _Rows()
        lifted = _moved(FEET, d_left=LEFT_UP[0])
        # The trunk over the support (right) foot, by 36 and 52 mm, lower, and
        # back where it was.
        for trunk in ((0.0, -0.036, 0.168), (0.003, -0.052, 0.164), TRUNK):
            rows.add(LEFT_UP, feet=lifted, trunk=trunk)
        torch.testing.assert_close(rows.reward(), torch.ones(3, dtype=torch.float64), rtol=0.0, atol=1e-12)

    def test_the_support_foot_under_the_trunk_is_not_in_the_error(self) -> None:
        rows = _Rows()
        # Both feet 15 mm off the target (the left one 15 mm short), the
        # support foot anywhere under the trunk (the whole robot moved).
        for shift in ((0.0, 0.0, 0.0), (0.02, -0.05, 0.0), (-0.01, 0.03, -0.004)):
            feet = _moved(FEET, d_left=tuple(a + b for a, b in zip(LEFT_UP[0], shift)), d_right=shift)
            feet = _moved(feet, d_left=(0.0, 0.0, -0.015))
            rows.add(LEFT_UP, feet=feet)
        expected = math.exp(-(0.015**2) / STD**2)
        torch.testing.assert_close(rows.reward(), torch.full((3,), expected, dtype=torch.float64),
                                   rtol=0.0, atol=1e-12)

    def test_trunk_roll_and_pitch_do_not_change_the_error_but_its_heading_does_turn_the_target(self) -> None:
        yaw = 0.7
        heading = quat_from_euler_xyz(*(torch.tensor([v], dtype=torch.float64) for v in (0.0, 0.0, yaw)))
        feet = _turned(_moved(FEET, d_left=(0.012, 0.006, 0.02)), heading)  # 10 mm short in z
        rows = _Rows()
        for roll, pitch in ((0.0, 0.0), (0.15, 0.0), (0.0, -0.2), (-0.1, 0.25)):
            rows.add(LEFT_UP, feet=feet, trunk=TRUNK, quat=_quat(roll, pitch, yaw))
        values = rows.reward()
        torch.testing.assert_close(values, torch.full((4,), math.exp(-(0.01**2) / STD**2), dtype=torch.float64),
                                   rtol=0.0, atol=1e-12)
        # The same feet seen from an unturned trunk are off by the turn.
        rows = _Rows()
        rows.add(LEFT_UP, feet=feet, trunk=TRUNK, quat=_quat())
        self.assertLess(float(rows.reward()[0]), 0.1)

    def test_other_rows_keep_each_foot_from_the_trunk(self) -> None:
        rows = _Rows()
        cases = [
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),  # no target
            ((0.005, -0.004, 0.008), (0.005, -0.004, 0.008)),  # both feet by one offset
            ((0.01, 0.0, 0.0099), (0.0, 0.0, 0.0)),  # one foot below the 10 mm threshold
            ((0.0, 0.0, 0.03), (0.0, 0.0, 0.021)),  # 9 mm apart
        ]
        feet = _moved(FEET, d_left=(0.004, 0.0, 0.006), d_right=(0.0, -0.003, 0.0))
        trunk = (0.002, -0.02, 0.169)
        quat = _quat(0.05, -0.04, 0.3)
        for offsets in cases:
            rows.add(offsets, feet=feet, trunk=trunk, quat=quat)
        values = rows.reward()
        command = rows.command
        error = torch.square(command.current_foot_pos_b() - command._default_foot_pos_b
                             - command.foot_target_offset_b).sum(-1).mean(-1)
        torch.testing.assert_close(values, torch.exp(-error / STD**2), rtol=0.0, atol=1e-12)
        self.assertTrue(bool((values < 0.9).all()))  # the trunk move is in it
        # A row at the threshold switches to the relative error.
        rows = _Rows()
        rows.add(((0.0, 0.0, THRESHOLD), (0.0, 0.0, 0.0)), feet=_moved(FEET, d_left=(0.0, 0.0, THRESHOLD)),
                 trunk=(0.0, -0.04, 0.168))
        self.assertAlmostEqual(float(rows.reward()[0]), 1.0, places=12)

    def test_left_and_right_mirror(self) -> None:
        mirror = lambda v: (v[0], -v[1], v[2])  # noqa: E731
        left_target = ((0.012, 0.02, 0.04), (0.0, 0.0, 0.0))
        left_feet = _moved(FEET, d_left=(0.004, 0.012, 0.03), d_right=(0.002, 0.001, 0.0))
        left_trunk = (0.001, -0.04, 0.167)
        rows = _Rows()
        rows.add(left_target, feet=left_feet, trunk=left_trunk, quat=_quat(0.1, 0.05, 0.0))
        rows.add((mirror(left_target[1]), mirror(left_target[0])),
                 feet=(mirror(left_feet[1]), mirror(left_feet[0])), trunk=mirror(left_trunk),
                 quat=_quat(-0.1, 0.05, 0.0))
        values = rows.reward()
        self.assertAlmostEqual(float(values[0]), float(values[1]), places=12)
        dx, dy, dz = 0.004 - 0.002 - 0.012, 0.012 - 0.001 - 0.02, 0.03 - 0.04
        self.assertAlmostEqual(float(values[0]), math.exp(-(dx * dx + dy * dy + dz * dz) / STD**2), places=12)

    def test_a_moving_command_fades_both_kinds(self) -> None:
        rows = _Rows()
        rows.add(LEFT_UP, feet=_moved(FEET, d_left=LEFT_UP[0]), trunk=(0.0, -0.04, 0.168))
        rows.add(((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
        twist = torch.tensor([[0.075, 0.0, 0.0], [0.0, 0.0, 0.2]], dtype=torch.float64)
        values = rows.reward(twist, velocity_fade_range=(0.0, 0.15))
        torch.testing.assert_close(values, torch.tensor([0.5, 0.0], dtype=torch.float64), rtol=0.0, atol=1e-12)

    def test_single_foot_weight_scales_only_the_one_foot_up_rows(self) -> None:
        rows = _Rows()
        rows.add(LEFT_UP, feet=_moved(FEET, d_left=(0.012, 0.006, 0.02)), trunk=(0.0, -0.04, 0.168))
        rows.add(((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), feet=_moved(FEET, d_left=(0.004, 0.0, 0.0)))
        rows.add(((0.01, 0.0, 0.0099), (0.0, 0.0, 0.0)))  # below the threshold
        plain = rows.reward()
        torch.testing.assert_close(rows.reward(single_foot_weight=5.0), plain * torch.tensor(
            [5.0, 1.0, 1.0], dtype=torch.float64), rtol=0.0, atol=1e-12)

    def test_pico_pays_the_one_foot_up_rows_as_the_lift_at_every_stage(self) -> None:
        from mjlab_microban.tasks.microban_teleop_env_cfg import (
            MICROBAN_TELEOP_SINGLE_FOOT_TRACKING_WEIGHT,
            MICROBAN_TELEOP_UPPER_FOOT_LIFT_WEIGHT,
            TELEOP_STAGES,
        )
        from mjlab_microban.tasks.microban_teleop_v13_arm_overlay import (
            make_microban_teleop_v13_arm_overlay_env_cfg,
        )

        self.assertEqual(MICROBAN_TELEOP_SINGLE_FOOT_TRACKING_WEIGHT, MICROBAN_TELEOP_UPPER_FOOT_LIFT_WEIGHT)
        term = make_microban_teleop_v13_arm_overlay_env_cfg().rewards["foot_target_tracking"]
        weight, single = term.weight, term.params["single_foot_weight"]
        self.assertAlmostEqual(weight * single, MICROBAN_TELEOP_SINGLE_FOOT_TRACKING_WEIGHT, places=12)
        for stage in TELEOP_STAGES:
            for s in stage.settings:
                if s.manager == "reward" and s.term == "foot_target_tracking":
                    weight = s.value if s.path == "weight" else weight
                    single = s.value if s.path == "params.single_foot_weight" else single
            self.assertAlmostEqual(weight * single, MICROBAN_TELEOP_SINGLE_FOOT_TRACKING_WEIGHT, places=12,
                                   msg=stage.name)

    def test_pico_uses_the_unload_threshold(self) -> None:
        from mjlab_microban.tasks.microban_teleop_v13_arm_overlay import (
            make_microban_teleop_v13_arm_overlay_env_cfg,
        )

        cfg = make_microban_teleop_v13_arm_overlay_env_cfg()
        term = cfg.rewards["foot_target_tracking"]
        self.assertIs(term.func, foot_target_tracking_error_exp)
        self.assertEqual(term.params["lift_threshold"], cfg.rewards["upper_foot_unload"].params["lift_threshold"])


if __name__ == "__main__":
    unittest.main()
