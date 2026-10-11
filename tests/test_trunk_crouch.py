"""The crouch reward (mdp.trunk_crouch): on a standing two-foot row, the trunk's
drop over the feet since the reset (HOME) against the targets' mean height."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul

from mjlab_microban.robot.microban_constants import HOME_ROOT_QUAT_WXYZ, HOME_TRUNK_PITCH_RAD
from mjlab_microban.tasks.mdp import FootTargetCommand, trunk_crouch

THRESHOLD = 0.010
BAND = 0.0025
TRUNK = (0.0, 0.0, 0.1704)
FEET = ((0.0068, 0.0464, 0.0035), (0.0068, -0.0464, 0.0035))  # left, right (world, HOME)
HOME_HEIGHT = TRUNK[2] - FEET[0][2]


def _quat(roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0) -> torch.Tensor:
    t = lambda v: torch.tensor([v], dtype=torch.float64)  # noqa: E731
    home = torch.tensor([HOME_ROOT_QUAT_WXYZ], dtype=torch.float64)
    return quat_mul(quat_from_euler_xyz(t(roll), t(pitch), t(yaw)), home)[0]


def _both(dx: float, dy: float, dz: float):
    return ((dx, dy, dz), (dx, dy, dz))


class _Rows:
    """A FootTargetCommand reset at HOME (at reset_xy, turned by reset_yaw),
    then n rows of trunk pose, feet and published offsets."""

    def __init__(self) -> None:
        self.rows: list[tuple] = []

    def add(self, offsets, drop=0.0, feet=FEET, trunk=TRUNK, quat=None, reset_yaw=0.0) -> None:
        trunk = (trunk[0], trunk[1], trunk[2] - drop)
        self.rows.append((offsets, feet, trunk, _quat() if quat is None else quat, reset_yaw))

    def reward(self, twist: torch.Tensor | None = None) -> torch.Tensor:
        n = len(self.rows)
        f64 = lambda v: torch.tensor(v, dtype=torch.float64)  # noqa: E731
        command = FootTargetCommand.__new__(FootTargetCommand)
        command._env = SimpleNamespace(num_envs=n)
        command.cfg = SimpleNamespace(trunk_pitch=HOME_TRUNK_PITCH_RAD)
        command._foot_asset_cfg = SimpleNamespace(site_ids=[0, 1])
        # The reset reference, every row at HOME (anywhere on the floor).
        reset_quat = torch.stack([_quat(yaw=r[4]) for r in self.rows])
        command.robot = SimpleNamespace(data=SimpleNamespace(
            root_link_pos_w=f64([TRUNK] * n) + 0.3, root_link_quat_w=reset_quat,
            site_pos_w=f64([FEET] * n) + 0.3))
        command._default_foot_pos_b = command.current_foot_pos_b()
        command.robot = SimpleNamespace(data=SimpleNamespace(
            root_link_pos_w=f64([r[2] for r in self.rows]), root_link_quat_w=torch.stack([r[3] for r in self.rows]),
            site_pos_w=f64([r[1] for r in self.rows])))
        command.foot_target_offset_b = f64([r[0] for r in self.rows])
        twist = torch.zeros(n, 3, dtype=torch.float64) if twist is None else twist
        terms = {"twist": SimpleNamespace(command=twist), "foot_target": command}
        manager = SimpleNamespace(get_term=terms.__getitem__, get_command=lambda name: terms[name].command)
        env = SimpleNamespace(num_envs=n, command_manager=manager)
        self.command = command
        return trunk_crouch(env, "foot_target", lift_threshold=THRESHOLD, floor_band=BAND)


def _rounded(values: torch.Tensor) -> list[float]:
    return [round(v, 6) for v in values.tolist()]


class TrunkCrouchTest(unittest.TestCase):
    def test_row_kinds(self) -> None:
        rows = _Rows()
        rows.add(_both(0.008, 0.008, 0.016), drop=0.016)  # two feet, the trunk 16 mm down
        rows.add(_both(-0.008, -0.008, 0.016), drop=0.008)  # half way
        rows.add(((0.0, 0.0, 0.020), (0.0, 0.0, 0.011)), drop=0.0155)  # 9 mm apart: two feet, dz = mean
        rows.add(((0.01, 0.0, 0.04), (0.0, 0.0, 0.0)), drop=0.016)  # one foot up
        rows.add(((0.0, 0.0, 0.030), (0.0, 0.0, 0.016)), drop=0.016)  # 14 mm apart: one foot up
        rows.add(((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), drop=0.016)  # no target
        rows.add(_both(0.0, 0.0, BAND), drop=BAND)  # in the floor band
        rows.add(((0.0, 0.0, 0.008), (0.0, 0.0, BAND)), drop=0.005)  # one foot in the band
        self.assertEqual(_rounded(rows.reward()), [1.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # A walking command pays nothing; one below 1 cm/s still pays.
        twist = torch.zeros(8, 3, dtype=torch.float64)
        twist[0] = torch.tensor([0.08, 0.0, 0.0])
        twist[1] = torch.tensor([0.004, 0.0, 0.004])
        twist[2] = torch.tensor([0.0, 0.0, 0.4])
        self.assertEqual(_rounded(rows.reward(twist)), [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_the_drop_is_from_the_home_height_over_the_feet(self) -> None:
        rows = _Rows()
        target = _both(0.0, 0.0, 0.016)
        rows.add(target)  # at HOME: no drop
        rows.add(target, drop=-0.005)  # trunk raised
        # The whole robot 10 mm up (a hop) or down: the trunk over the feet
        # does not change.
        up = tuple((x, y, z + 0.01) for x, y, z in FEET)
        rows.add(target, drop=-0.01, feet=up)
        rows.add(target, drop=0.016 - 0.01, feet=up)  # 16 mm down over feet 10 mm up
        # Elsewhere on the floor, turned, the trunk tilted: only the heights count.
        rows.add(target, drop=0.012, trunk=(0.2, -0.1, TRUNK[2]),
                 feet=tuple((x + 0.2, y - 0.1, z) for x, y, z in FEET), quat=_quat(0.1, -0.1, 1.0), reset_yaw=-2.0)
        values = rows.reward()
        self.assertEqual(_rounded(values), [0.0, 0.0, 0.0, 1.0, 0.75])
        torch.testing.assert_close(rows.command.trunk_over_feet_at_reset(),
                                   torch.full((5,), HOME_HEIGHT, dtype=torch.float64), rtol=0.0, atol=1e-12)
        drop = rows.command.trunk_over_feet_at_reset() - rows.command.trunk_over_feet()
        torch.testing.assert_close(drop, torch.tensor([0.0, -0.005, 0.0, 0.016, 0.012], dtype=torch.float64),
                                   rtol=0.0, atol=1e-12)

    def test_a_trunk_lower_than_the_target_costs_as_much(self) -> None:
        rows = _Rows()
        for drop in (0.016, 0.020, 0.024, 0.032, 0.040):
            rows.add(_both(0.0, 0.0, 0.016), drop=drop)
        self.assertEqual(_rounded(rows.reward()), [1.0, 0.75, 0.5, 0.0, 0.0])

    def test_left_and_right_mirror(self) -> None:
        mirror = lambda v: (v[0], -v[1], v[2])  # noqa: E731
        target = ((0.006, 0.008, 0.018), (0.006, 0.008, 0.014))
        feet = ((0.0068, 0.0544, 0.0035), (0.0048, -0.0414, 0.0046))
        trunk = (0.003, 0.012, TRUNK[2] - 0.011)
        rows = _Rows()
        rows.add(target, feet=feet, trunk=trunk, quat=_quat(0.08, 0.05, 0.2), reset_yaw=0.4)
        rows.add((mirror(target[1]), mirror(target[0])), feet=(mirror(feet[1]), mirror(feet[0])),
                 trunk=mirror(trunk), quat=_quat(-0.08, 0.05, -0.2), reset_yaw=-0.4)
        values = rows.reward()
        self.assertAlmostEqual(float(values[0]), float(values[1]), places=12)
        drop = 0.011 + (0.0046 - 0.0035) / 2.0
        self.assertAlmostEqual(float(values[0]), 1.0 - abs(drop - 0.016) / 0.016, places=12)
        self.assertGreater(float(values[0]), 0.0)

    def test_pico_rewards_the_crouch_from_the_foot_stage(self) -> None:
        from mjlab_microban.tasks.microban_teleop_env_cfg import (
            MICROBAN_TELEOP_SINGLE_SUPPORT_LIFT_THRESHOLD_M,
            MICROBAN_TELEOP_TRUNK_CROUCH_WEIGHT,
            MICROBAN_TELEOP_UPPER_FOOT_LIFT_WEIGHT,
            TELEOP_STAGES,
        )
        from mjlab_microban.tasks.microban_teleop_mdp import MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M
        from mjlab_microban.tasks.microban_teleop_v13_arm_overlay import (
            make_microban_teleop_v13_arm_overlay_env_cfg,
        )

        settings = [(stage.name, s.path, s.value) for stage in TELEOP_STAGES for s in stage.settings
                    if s.manager == "reward" and s.term == "trunk_crouch"]
        self.assertEqual(settings, [(TELEOP_STAGES[1].name, "weight", MICROBAN_TELEOP_TRUNK_CROUCH_WEIGHT)])
        self.assertEqual(MICROBAN_TELEOP_TRUNK_CROUCH_WEIGHT, MICROBAN_TELEOP_UPPER_FOOT_LIFT_WEIGHT)
        rewards = make_microban_teleop_v13_arm_overlay_env_cfg().rewards
        term = rewards["trunk_crouch"]
        self.assertIs(term.func, trunk_crouch)
        self.assertEqual(term.weight, 0.0)
        lift = rewards["upper_foot_lift"].params
        self.assertEqual(term.params, {
            "foot_target_command_name": lift["foot_target_command_name"],
            "lift_threshold": MICROBAN_TELEOP_SINGLE_SUPPORT_LIFT_THRESHOLD_M,
            "floor_band": MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M,
            "command_name": lift["command_name"],
            "command_threshold": lift["command_threshold"],
        })
        self.assertEqual(BAND, MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M)
        self.assertEqual(THRESHOLD, MICROBAN_TELEOP_SINGLE_SUPPORT_LIFT_THRESHOLD_M)


if __name__ == "__main__":
    unittest.main()
