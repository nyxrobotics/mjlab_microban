"""PICO foot targets (microban_teleop_foot_command.py): single-foot targets with a
standing twist, the 0.12 m/s target speed and the floor band."""

from __future__ import annotations

import re
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from mjlab_microban.tasks import microban_teleop_foot_command as fc
from mjlab_microban.tasks.mdp import FootTargetCommand, feet_zmp, lifted_support_feet, support_zmp, upper_foot_lift
from mjlab_microban.tasks.microban_teleop_mdp import ResetFixedFootTargetCommand


def _command(stationary_probability: float, n: int = 3):
    command = object.__new__(fc.StationaryFootTargetCommand)
    twist = torch.tensor([[0.3, 0.1, 0.5], [0.2, 0.0, 0.0], [-0.1, 0.0, 1.0]])[:n]
    velocity = SimpleNamespace(command_counter=torch.zeros(n), vel_command_b=twist.clone(),
                               vel_command_w=twist.clone(), is_rotation_env=torch.zeros(n, dtype=torch.bool))
    command._env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: velocity), device="cpu")
    command.cfg = SimpleNamespace(velocity_command_name="twist",
                                  single_support_stationary_probability=stationary_probability)
    command._reference_pending = torch.zeros(n, dtype=torch.bool)
    command.is_single_support_env = torch.zeros(n, dtype=torch.bool)
    command.is_both_feet_env = torch.zeros(n, dtype=torch.bool)
    command._previous_both_feet_env = torch.zeros(n, dtype=torch.bool)
    command._velocity_cache_valid = torch.zeros(n, dtype=torch.bool)
    command._velocity_command_counter = None
    command._saved_vel_command_b = command._saved_vel_command_w = command._saved_is_rotation_env = None
    command.is_stationary_single_support_env = torch.zeros(n, dtype=torch.bool)
    command.foot_target_offset_b = torch.zeros(n, 2, 3)
    command.foot_target_goal_b = torch.zeros(n, 2, 3)
    command.foot_target_slewed_b = torch.zeros(n, 2, 3)
    command._max_step_m = fc.MICROBAN_TELEOP_FOOT_TARGET_SLEW_M_S * 0.02
    return command, velocity, twist


class _Scene(SimpleNamespace):
    def __getitem__(self, name):
        return self.entities[name]


def _zmp_env(published, single, lifted, twist=None, soles=None):
    """A standing env for support_zmp: soles (N, 2, 3) world (left, right),
    the feet sensor listing right before left."""
    n = published.shape[0]
    twist = torch.zeros(n, 3) if twist is None else twist
    if soles is None:
        soles = torch.tensor([[0.0, 0.047, 0.0], [0.0, -0.047, 0.0]]).repeat(n, 1, 1)
    foot = SimpleNamespace(command=published.reshape(n, 6), is_single_support_env=single, lifted_foot_idx=lifted)
    terms = {"twist": SimpleNamespace(command=twist), "foot_target": foot}
    manager = SimpleNamespace(get_term=terms.__getitem__, get_command=lambda name: terms[name].command)
    sensor = SimpleNamespace(data=SimpleNamespace(found=torch.ones(n, 2), force=torch.zeros(n, 2, 3),
                                                  torque=torch.zeros(n, 2, 3), pos=torch.zeros(n, 2, 3)))
    robot = SimpleNamespace(data=SimpleNamespace(geom_pos_w=soles))
    scene = _Scene(sensors={"feet": sensor}, entities={"robot": robot})
    return SimpleNamespace(num_envs=n, command_manager=manager, scene=scene), twist, sensor, soles


def _zmp_reward(env, sensor, load_left_right, at_left, at_right):
    """support_zmp with each foot's load (N up) pressing at one point (world x, y)."""
    n = env.num_envs
    load = torch.tensor(load_left_right, dtype=torch.float32)
    at = torch.stack([at_left, at_right])  # (2, 2)
    sensor.data.force = torch.zeros(n, 2, 3)
    sensor.data.force[..., 0] = 0.3  # a sideways push does not count
    sensor.data.force[..., 2] = -load  # the sensor reports the foot's push on the floor
    sensor.data.force = sensor.data.force.flip(1)
    sensor.data.pos = torch.cat([at.expand(n, 2, 2), torch.zeros(n, 2, 1)], -1).flip(1)
    sensor.data.torque = torch.zeros(n, 2, 3)
    sensor.data.found = (load > 0).float().flip(1)
    values = support_zmp(env, "feet", "foot_target", sensor_foot_ids=(1, 0), lift_threshold=0.01,
                         zmp_tolerance=0.047, sole_cfg=SimpleNamespace(name="robot", geom_ids=[0, 1]))
    return [round(v, 4) for v in values.tolist()]


class StationaryFootTargetTest(unittest.TestCase):
    def test_a_stationary_single_foot_target_holds_the_twist_at_zero(self) -> None:
        command, velocity, twist = _command(0.5)
        command.is_single_support_env[:2] = True
        command.is_stationary_single_support_env[0] = True
        command._update_command()
        self.assertTrue(torch.equal(velocity.vel_command_b[0], torch.zeros(3)))
        self.assertTrue(torch.equal(velocity.vel_command_b[1:], twist[1:]))  # moving single foot, no target
        self.assertFalse(bool(command.is_both_feet_env.any()))  # restored
        # The target ends: the twist comes back at once.
        command.is_stationary_single_support_env[0] = False
        command.is_single_support_env[0] = False
        command._update_command()
        self.assertTrue(torch.equal(velocity.vel_command_b, twist))

    def test_two_foot_targets_still_stand(self) -> None:
        command, velocity, twist = _command(0.0)
        command.is_both_feet_env[2] = True
        command._update_command()
        self.assertTrue(torch.equal(velocity.vel_command_b[2], torch.zeros(3)))
        self.assertTrue(torch.equal(velocity.vel_command_b[:2], twist[:2]))
        self.assertTrue(bool(command.is_both_feet_env[2]))

    def test_the_share_of_stationary_single_foot_targets(self) -> None:
        def parent(self, env_ids):
            self.is_single_support_env[env_ids] = True

        for probability, expected in ((0.0, 0), (1.0, 4000)):
            command, _velocity, _twist = _command(probability)
            n = 4000
            command.is_single_support_env = torch.zeros(n, dtype=torch.bool)
            command.is_stationary_single_support_env = torch.zeros(n, dtype=torch.bool)
            command.foot_target_offset_b = torch.zeros(n, 2, 3)
            command.foot_target_goal_b = torch.zeros(n, 2, 3)
            with mock.patch.object(ResetFixedFootTargetCommand, "_resample_command", parent):
                command._resample_command(torch.arange(n))
            self.assertEqual(int(command.is_stationary_single_support_env.sum()), expected)
        command, _velocity, _twist = _command(0.5)
        n = 4000
        command.is_single_support_env = torch.zeros(n, dtype=torch.bool)
        command.is_stationary_single_support_env = torch.zeros(n, dtype=torch.bool)
        command.foot_target_offset_b = torch.zeros(n, 2, 3)
        command.foot_target_goal_b = torch.zeros(n, 2, 3)
        torch.manual_seed(0)
        with mock.patch.object(ResetFixedFootTargetCommand, "_resample_command", parent):
            command._resample_command(torch.arange(n))
        self.assertAlmostEqual(float(command.is_stationary_single_support_env.float().mean()), 0.5, delta=0.03)

    def test_the_target_moves_at_the_teleop_speed_and_drops_the_floor_band(self) -> None:
        command, _velocity, _twist = _command(0.0, n=1)
        command.foot_target_goal_b[0, 0] = torch.tensor([0.0, 0.0, 0.04])
        heights = []
        for _ in range(25):
            command._update_command()
            heights.append(float(command.foot_target_offset_b[0, 0, 2]))
        # 0.12 m/s at 50 Hz is 2.4 mm per step; the first step (2.4 mm) is in
        # the 2.5 mm floor band and reads exactly zero, like the wire.
        self.assertEqual(heights[0], 0.0)
        self.assertTrue(torch.equal(command.foot_target_offset_b[0, 1], torch.zeros(3)))
        self.assertAlmostEqual(heights[1], 0.0048, places=6)
        self.assertAlmostEqual(heights[16], 0.04, places=6)  # 17 steps for 40 mm
        steps = [b - a for a, b in zip(heights[1:], heights[2:])]
        self.assertLessEqual(max(steps), 0.0024 + 1e-7)
        # Back down at the same speed when the target ends.
        command.foot_target_goal_b.zero_()
        command._update_command()
        self.assertAlmostEqual(float(command.foot_target_offset_b[0, 0, 2]), 0.04 - 0.0024, places=6)

    def test_two_published_feet_hold_the_twist_until_one_is_down(self) -> None:
        command, velocity, twist = _command(0.0, n=1)
        command.foot_target_slewed_b[0] = torch.tensor([[0.0, 0.0, 0.01], [0.0, 0.0, 0.006]])
        command._update_command()
        self.assertTrue(torch.equal(velocity.vel_command_b[0], torch.zeros(3)))
        self.assertFalse(bool(command.is_both_feet_env.any()))
        command._update_command()  # the right foot reaches the floor band
        self.assertTrue(torch.equal(command.foot_target_offset_b[0, 1], torch.zeros(3)))
        self.assertTrue(torch.equal(velocity.vel_command_b, twist))

    def test_a_two_foot_target_moves_both_feet_by_one_offset(self) -> None:
        n = 2000
        command = object.__new__(ResetFixedFootTargetCommand)
        command.cfg = SimpleNamespace(rel_both_feet_envs=1.0, both_feet_reach_xy_range=(-0.01, 0.01),
                                      both_feet_lift_height_range=(0.0025, 0.012))
        command._env = SimpleNamespace(device="cpu")
        command._default_foot_pos_b = torch.zeros(n, 2, 3)
        command.is_both_feet_env = torch.zeros(n, dtype=torch.bool)
        command.is_single_support_env = torch.zeros(n, dtype=torch.bool)
        command.foot_target_offset_b = torch.zeros(n, 2, 3)
        with mock.patch.object(FootTargetCommand, "_resample_command", lambda self, ids: None):
            command._resample_command(torch.arange(n))
        offsets = command.foot_target_offset_b
        self.assertTrue(bool(command.is_both_feet_env.all()))
        self.assertTrue(torch.equal(offsets[:, 0], offsets[:, 1]))
        self.assertGreater(float(offsets[:, 0, :2].std()), 0.004)  # still spread over +-10 mm
        self.assertTrue(bool(((offsets[..., 2] >= 0.0025) & (offsets[..., 2] <= 0.012)).all()))

    def test_lifted_support_feet_counts_the_feet_up_that_should_be_down(self) -> None:
        # Rows (feet in the foot target's order, left then right): no target,
        # single target on the left (right supports), single target on the
        # right drawn but still in the floor band, single target on the left
        # coming down, both feet by one offset, a single target on the right
        # handing over to a two-foot one, a single target on the right drawn
        # while the left one is still coming down.
        published = torch.zeros(7, 2, 3)
        published[3, 0, 2] = 0.02
        published[4] = torch.tensor([0.005, -0.004, 0.01])
        published[5] = torch.tensor([[0.0, 0.0, 0.01], [0.0, 0.0, 0.03]])
        published[6, 0, 2] = 0.03
        single = torch.tensor([False, True, True, False, False, False, True])
        lifted = torch.tensor([0, 0, 1, 0, 0, 1, 1])
        published[1, 0] = torch.tensor([0.01, 0.0, 0.04])
        foot = SimpleNamespace(command=published.view(7, 6), is_single_support_env=single,
                               lifted_foot_idx=lifted)
        twist = torch.zeros(7, 3)
        terms = {"twist": SimpleNamespace(command=twist), "foot_target": foot}
        manager = SimpleNamespace(get_term=terms.__getitem__, get_command=lambda name: terms[name].command)
        sensor = SimpleNamespace(data=SimpleNamespace(found=torch.ones(7, 2)))
        env = SimpleNamespace(num_envs=7, command_manager=manager, scene=SimpleNamespace(sensors={"feet": sensor}))

        def count(found_left_right):  # the sensor lists right before left
            sensor.data.found = torch.tensor(found_left_right, dtype=torch.float32).flip(-1)
            return lifted_support_feet(env, "feet", "foot_target", sensor_foot_ids=(1, 0)).tolist()

        self.assertEqual(count([[1, 1]] * 7), [0.0] * 7)
        # Only the support foot or the lower target's foot counts; a support
        # foot whose own target is still coming down does not.
        self.assertEqual(count([[1, 0]] * 7), [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        self.assertEqual(count([[0, 1]] * 7), [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
        both_up = [[0, 0]] * 7
        self.assertEqual(count(both_up), [2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.0])
        # A walking command costs nothing, a slow one below 1 cm/s still counts.
        twist[:3] = torch.tensor([[0.1, 0.0, 0.0], [0.0, 0.0, 0.4], [0.004, 0.0, 0.004]])
        self.assertEqual(count(both_up), [0.0, 0.0, 1.0, 1.0, 2.0, 1.0, 0.0])

    def test_feet_zmp_is_the_vertical_force_weighted_mean_of_the_contact_points(self) -> None:
        # Contact points of each foot (world x, y on the floor z = 0) with
        # forces of the floor on the foot (sideways parts included), reduced
        # as mujoco_warp's netforce does: the sum, the |F|-weighted centroid
        # and the torque about it; the sensor reports the foot's push (minus).
        torch.manual_seed(0)
        n, k = 5, 7
        points = torch.cat([torch.rand(n, 2, k, 2) * 0.06, torch.zeros(n, 2, k, 1)], dim=-1)
        points[:, 1, :, 1] += 0.09
        forces = torch.cat([torch.randn(n, 2, k, 2) * 0.3, torch.rand(n, 2, k, 1) * 2.0], dim=-1)
        forces[0, 1] = 0.0  # one foot in the air
        weight = forces.norm(dim=-1, keepdim=True)
        centroid = (points * weight).sum(2) / weight.sum(2).clamp(min=1e-12)
        net = forces.sum(2)
        torque = torch.cross(points - centroid[:, :, None], forces, dim=-1).sum(2)
        centroid[0, 1] = 0.0
        sensor = SimpleNamespace(data=SimpleNamespace(force=-net.flip(1), torque=-torque.flip(1), pos=centroid.flip(1)))
        zmp, load = feet_zmp(sensor, (1, 0))  # the sensor lists right before left
        fz = forces[..., 2]
        expected = (points[..., :2] * fz[..., None]).sum(dim=(1, 2)) / fz.sum(dim=(1, 2))[:, None]
        torch.testing.assert_close(zmp, expected)
        torch.testing.assert_close(load, fz.sum(2))

    def test_support_zmp_pays_the_centre_of_pressure_where_the_target_puts_the_weight(self) -> None:
        # Rows (feet left, right): no target; single target on the left at
        # 30 mm; single target on the right at 30 mm; single target on the
        # right at 5 mm (half the 10 mm threshold); single target on the
        # left at exactly 10 mm; a left target coming down (no single
        # target) at 2 mm; both feet by one offset; a single target on the
        # right drawn while the left one (40 mm) is still coming down.
        published = torch.zeros(8, 2, 3)
        published[1, 0] = torch.tensor([0.01, 0.0, 0.03])
        published[2, 1, 2] = 0.03
        published[3, 1, 2] = 0.005
        published[4, 0, 2] = 0.010
        published[5, 0, 2] = 0.002
        published[6] = torch.tensor([0.0, 0.0, 0.012])
        published[7] = torch.tensor([[0.0, 0.0, 0.04], [0.0, 0.0, 0.01]])
        single = torch.tensor([False, True, True, True, True, False, False, True])
        lifted = torch.tensor([0, 0, 1, 1, 0, 0, 0, 1])
        env, twist, sensor, stand = _zmp_env(published, single, lifted)
        left, right = torch.tensor([0.0, 0.047]), torch.tensor([0.0, -0.047])

        def reward(load_left_right, at_left=left, at_right=right):
            return _zmp_reward(env, sensor, load_left_right, at_left, at_right)

        # Target points (y, mm): the support sole's centre from the threshold
        # up; at 5 mm a quarter of the way to the higher (right) sole, +23.5;
        # at 2 mm 0.4 of the way to the higher (left) sole, -9.4.  Over the
        # midpoint (even weight on the sole centres): 0 from the threshold
        # up (46 mm off, the tolerance), 0.5 at 5 mm, 0.8 at 2 mm; no row
        # without a height difference pays.
        self.assertEqual(reward([[5.5, 5.5]] * 8), [0.0, 0.0, 0.0, 0.5, 0.0, 0.8, 0.0, 0.0])
        # All on the right sole's centre.
        self.assertEqual(reward([[0.0, 11.0]] * 8), [0.0, 1.0, 0.0, 0.0, 1.0, 0.2, 0.0, 1.0])
        # All on the left sole's centre.
        self.assertEqual(reward([[11.0, 0.0]] * 8), [0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0])
        # A quarter on the higher foot, each foot pressing at its centre.
        self.assertEqual(reward([[2.75, 8.25]] * 8), [0.0, 0.5, 0.0, 0.0, 0.5, 0.7, 0.0, 0.5])
        self.assertEqual(reward([[8.25, 2.75]] * 8), [0.0, 0.0, 0.5, 1.0, 0.0, 0.3, 0.0, 0.0])
        # On one foot, the centre of pressure 20 mm off the sole's centre (at
        # its side edge, either way, or forward): 1 - 20 / 47.
        edge = round(1.0 - 20.0 / 47.0, 4)
        for shift in ([0.0, 0.02], [0.0, -0.02], [0.02, 0.0]):
            off = torch.tensor(shift)
            self.assertEqual(reward([[0.0, 11.0]] * 8, at_right=right + off)[1], edge)
            self.assertEqual(reward([[11.0, 0.0]] * 8, at_left=left + off)[2], edge)
        # Both feet in the air (or all but a trace): 0.
        self.assertEqual(reward([[0.0, 0.0]] * 8), [0.0] * 8)
        self.assertEqual(reward([[0.0, 0.05]] * 8), [0.0] * 8)
        # A walking command pays nothing; a slow one below 1 cm/s still pays.
        twist[1] = torch.tensor([0.1, 0.0, 0.0])
        twist[2] = torch.tensor([0.0, 0.0, 0.4])
        twist[4] = torch.tensor([0.004, 0.0, 0.004])
        self.assertEqual(reward([[0.0, 11.0]] * 8), [0.0, 0.0, 0.0, 0.0, 1.0, 0.2, 0.0, 1.0])
        self.assertEqual(reward([[11.0, 0.0]] * 8), [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0])

    def test_support_zmp_reads_the_torque_and_is_mirror_symmetric(self) -> None:
        # Random rows, soles and loads; each foot's centroid is off its centre
        # of pressure and the torque about the centroid puts it back.
        torch.manual_seed(1)
        n = 64
        published = torch.zeros(n, 2, 3)
        published[..., 2] = torch.rand(n, 2) * 0.03 * (torch.rand(n, 2) < 0.6)
        lifted = torch.randint(0, 2, (n,))
        single = torch.rand(n) < 0.5
        soles = torch.cat([torch.randn(n, 2, 2) * 0.01, torch.zeros(n, 2, 1)], dim=-1)
        soles[:, 0, 1] += 0.047
        soles[:, 1, 1] -= 0.047
        cop = soles[..., :2] + torch.randn(n, 2, 2) * 0.015  # (left, right)
        load = torch.rand(n, 2) * 6.0 * (torch.rand(n, 2) < 0.85)
        twist = torch.zeros(n, 3)

        def run(published, lifted, soles, cop, load):
            env, _, sensor, _ = _zmp_env(published, single, lifted, twist=twist, soles=soles)
            centroid = cop + torch.randn(n, 2, 2) * 0.01
            # L p = c L + (tau_y, -tau_x)
            tau_y = (cop[..., 0] - centroid[..., 0]) * load
            tau_x = -(cop[..., 1] - centroid[..., 1]) * load
            sensor.data.force = torch.stack([torch.full_like(load, 0.2), -0.1 * load, -load], -1).flip(1)
            sensor.data.torque = torch.stack([tau_x, tau_y, torch.full_like(load, 0.003)], -1).flip(1)
            sensor.data.pos = torch.cat([centroid, torch.zeros(n, 2, 1)], -1).flip(1)
            return support_zmp(env, "feet", "foot_target", sensor_foot_ids=(1, 0), lift_threshold=0.01,
                               zmp_tolerance=0.047, sole_cfg=SimpleNamespace(name="robot", geom_ids=[0, 1]))

        values = run(published, lifted, soles, cop, load)
        # The same from the per-foot centres of pressure directly.
        upper = published[..., 2] > published.flip(1)[..., 2]
        dz = (published[:, 0, 2] - published[:, 1, 2]).abs()
        share = 0.5 * (1.0 - dz / 0.01).clamp(min=0.0)
        sup = torch.where(upper[:, :1], soles[:, 1, :2], soles[:, 0, :2])
        up = torch.where(upper[:, :1], soles[:, 0, :2], soles[:, 1, :2])
        target = sup + share[:, None] * (up - sup)
        zmp = (cop * load[..., None]).sum(1) / load.sum(1).clamp(min=1e-9)[:, None]
        expected = (1.0 - (zmp - target).norm(dim=-1) / 0.047).clamp(0.0, 1.0)
        expected = expected * ((dz > 0) & (load.sum(1) >= 0.1)).float()
        torch.testing.assert_close(values, expected)
        self.assertGreater(int((values > 0).sum()), n // 4)
        # Mirrored (y -> -y, left <-> right): the same reward.
        flip_y = torch.tensor([1.0, -1.0])
        mirrored = run(published.flip(1), 1 - lifted, soles.flip(1) * torch.tensor([1.0, -1.0, 1.0]),
                       cop.flip(1) * flip_y, load.flip(1))
        torch.testing.assert_close(mirrored, values)

    def test_upper_foot_lift_pays_the_higher_foot_off_the_floor_by_its_height(self) -> None:
        # Rows (feet left, right): no target; single target on the left at
        # 40 mm; single target on the right at 40 mm; single target on the
        # right at 5 mm (below the 10 mm threshold); single target on the
        # left at exactly 10 mm; both feet by one offset.
        published = torch.zeros(6, 2, 3)
        published[1, 0] = torch.tensor([0.01, 0.0, 0.04])
        published[2, 1, 2] = 0.04
        published[3, 1, 2] = 0.005
        published[4, 0, 2] = 0.010
        published[5] = torch.tensor([0.0, 0.0, 0.012])
        single = torch.tensor([False, True, True, True, True, False])
        lifted = torch.tensor([0, 0, 1, 1, 0, 0])
        rise = torch.zeros(6)  # left foot z minus right foot z
        foot = SimpleNamespace(command=published.view(6, 6), is_single_support_env=single, lifted_foot_idx=lifted,
                               left_from_right_level=lambda: torch.nn.functional.pad(rise[:, None], (2, 0)))
        twist = torch.zeros(6, 3)
        terms = {"twist": SimpleNamespace(command=twist), "foot_target": foot}
        manager = SimpleNamespace(get_term=terms.__getitem__, get_command=lambda name: terms[name].command)
        sensor = SimpleNamespace(data=SimpleNamespace(found=torch.ones(6, 2)))
        env = SimpleNamespace(num_envs=6, command_manager=manager, scene=SimpleNamespace(sensors={"feet": sensor}))

        def reward(left_minus_right_m, found_left_right):  # the sensor lists right before left
            rise[:] = left_minus_right_m
            sensor.data.found = torch.tensor(found_left_right, dtype=torch.float32).flip(-1)
            values = upper_foot_lift(env, "feet", "foot_target", sensor_foot_ids=(1, 0), lift_threshold=0.01)
            return [round(v, 4) for v in values.tolist()]

        left_up, right_up = [[0, 1]] * 6, [[1, 0]] * 6
        # The left foot 20 mm up and off the floor: half the 40 mm target,
        # all of the 10 mm one; not on the rows whose higher target is the
        # right foot, below the threshold, without a height difference.
        self.assertEqual(reward(0.02, left_up), [0.0, 0.5, 0.0, 0.0, 1.0, 0.0])
        self.assertEqual(reward(0.002, left_up), [0.0, 0.05, 0.0, 0.0, 0.2, 0.0])
        # The right foot 30 mm up and off the floor (left minus right -30 mm).
        self.assertEqual(reward(-0.03, right_up), [0.0, 0.0, 0.75, 0.0, 0.0, 0.0])
        # The same heights with the higher foot touching the floor (a heel
        # raised on the toes): nothing.
        self.assertEqual(reward(0.02, [[1, 1]] * 6), [0.0] * 6)
        self.assertEqual(reward(-0.03, [[1, 1]] * 6), [0.0] * 6)
        # Both feet in the air (a hop): nothing.
        self.assertEqual(reward(0.02, [[0, 0]] * 6), [0.0] * 6)
        # The wrong foot up, or the higher foot below the lower one: nothing.
        self.assertEqual(reward(0.02, right_up), [0.0] * 6)
        self.assertEqual(reward(-0.01, left_up), [0.0] * 6)
        # Higher than the target: capped at 1.
        self.assertEqual(reward(0.06, left_up), [0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
        # A walking command pays nothing; a slow one below 1 cm/s still pays.
        twist[1] = torch.tensor([0.1, 0.0, 0.0])
        twist[4] = torch.tensor([0.004, 0.0, 0.004])
        self.assertEqual(reward(0.02, left_up), [0.0, 0.0, 0.0, 0.0, 1.0, 0.0])

    def test_pico_rewards_the_higher_foot_lift_from_the_foot_stage(self) -> None:
        from mjlab_microban.tasks.microban_teleop_env_cfg import (
            MICROBAN_TELEOP_SINGLE_SUPPORT_LIFT_THRESHOLD_M,
            MICROBAN_TELEOP_SUPPORT_ZMP_WEIGHT,
            MICROBAN_TELEOP_UPPER_FOOT_LIFT_WEIGHT,
            TELEOP_STAGES,
        )
        from mjlab_microban.tasks.microban_teleop_v13_arm_overlay import (
            make_microban_teleop_v13_arm_overlay_env_cfg,
        )

        settings = [(stage.name, s.path, s.value) for stage in TELEOP_STAGES for s in stage.settings
                    if s.manager == "reward" and s.term == "upper_foot_lift"]
        self.assertEqual(settings, [(TELEOP_STAGES[1].name, "weight", MICROBAN_TELEOP_UPPER_FOOT_LIFT_WEIGHT)])
        self.assertEqual(MICROBAN_TELEOP_UPPER_FOOT_LIFT_WEIGHT, MICROBAN_TELEOP_SUPPORT_ZMP_WEIGHT)
        rewards = make_microban_teleop_v13_arm_overlay_env_cfg().rewards
        term = rewards["upper_foot_lift"]
        self.assertIs(term.func, upper_foot_lift)
        self.assertEqual(term.weight, 0.0)
        self.assertEqual(term.params, {**rewards["lifted_support_feet"].params,
                                       "lift_threshold": MICROBAN_TELEOP_SINGLE_SUPPORT_LIFT_THRESHOLD_M})

    def test_pico_rewards_the_support_zmp_from_the_foot_stage(self) -> None:
        from mjlab_microban.robot.home_pose import HOME
        from mjlab_microban.tasks.microban_teleop_env_cfg import (
            MICROBAN_TELEOP_SINGLE_SUPPORT_LIFT_THRESHOLD_M,
            MICROBAN_TELEOP_SUPPORT_ZMP_WEIGHT,
            TELEOP_STAGES,
        )
        from mjlab_microban.tasks.microban_teleop_mdp import MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M
        from mjlab_microban.tasks.microban_teleop_v13_arm_overlay import (
            make_microban_teleop_v13_arm_overlay_env_cfg,
        )
        from mjlab_microban.tasks.microban_velocity_env_cfg import make_microban_velocity_env_cfg

        settings = [(stage.name, s.path, s.value) for stage in TELEOP_STAGES for s in stage.settings
                    if s.manager == "reward" and s.term == "support_zmp"]
        self.assertEqual(settings, [(TELEOP_STAGES[1].name, "weight", MICROBAN_TELEOP_SUPPORT_ZMP_WEIGHT)])
        # Above the 6.2 /s the measured shift to one foot costs in the other terms.
        self.assertGreater(MICROBAN_TELEOP_SUPPORT_ZMP_WEIGHT, 6.2)
        cfg = make_microban_teleop_v13_arm_overlay_env_cfg()
        self.assertNotIn("upper_foot_unload", cfg.rewards)
        term = cfg.rewards["support_zmp"]
        self.assertIs(term.func, support_zmp)
        self.assertEqual(term.weight, 0.0)
        params = dict(term.params)
        sole = params.pop("sole_cfg")
        self.assertEqual((sole.name, tuple(sole.geom_names), sole.preserve_order),
                         ("robot", ("left_foot_collision_1", "right_foot_collision_1"), True))
        self.assertEqual(params.pop("zmp_tolerance"), HOME.feet_lateral_m / 2.0)
        threshold = params.pop("lift_threshold")
        self.assertEqual(params, cfg.rewards["lifted_support_feet"].params)
        self.assertEqual(threshold, MICROBAN_TELEOP_SINGLE_SUPPORT_LIFT_THRESHOLD_M)
        # Above the floor band, below the highest single-foot target.
        self.assertGreater(threshold, MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M)
        self.assertLess(threshold, cfg.commands["foot_target"].lift_height_range[1])
        # The PICO feet sensor adds the centroid and torque to the net force;
        # the walking task's is left as it was.
        sensors = {s.name: s for s in cfg.scene.sensors}
        walk = {s.name: s for s in make_microban_velocity_env_cfg().scene.sensors}["feet_ground_contact"]
        feet = sensors[params["sensor_name"]]
        self.assertEqual((feet.fields, feet.reduce), (("found", "force", "torque", "pos"), "netforce"))
        self.assertEqual(walk.fields, ("found", "force"))
        self.assertEqual(replace(feet, fields=walk.fields), walk)

    def test_the_sole_geoms_are_centred_on_the_sole(self) -> None:
        import mujoco

        xml = Path(fc.__file__).resolve().parents[1] / "robot" / "microban" / "robot.xml"
        model = mujoco.MjModel.from_xml_path(str(xml))
        for side in ("left", "right"):
            geoms = [model.geom(f"{side}_foot_collision_{i}") for i in range(1, 7)]
            for g in geoms[1:]:  # the six nested boxes share one centre
                self.assertTrue(np.allclose(g.pos, geoms[0].pos) and np.allclose(g.quat, geoms[0].quat))

    def test_pico_penalizes_lifted_support_feet_from_the_arm_stage_instead_of_no_stepping(self) -> None:
        from mjlab_microban.tasks.microban_teleop_env_cfg import (
            MICROBAN_TELEOP_LIFTED_SUPPORT_FEET_WEIGHT,
            TELEOP_STAGES,
        )
        from mjlab_microban.tasks.microban_teleop_v13_arm_overlay import (
            make_microban_teleop_v13_arm_overlay_env_cfg,
        )

        settings = [(stage.name, s.path, s.value) for stage in TELEOP_STAGES for s in stage.settings
                    if s.manager == "reward" and s.term in ("lifted_support_feet", "no_stepping")]
        self.assertEqual(settings, [(TELEOP_STAGES[0].name, "weight", MICROBAN_TELEOP_LIFTED_SUPPORT_FEET_WEIGHT)])
        self.assertEqual([s.term for s in TELEOP_STAGES[0].settings if s.path == "weight"], ["lifted_support_feet"])
        rewards = make_microban_teleop_v13_arm_overlay_env_cfg().rewards
        self.assertEqual(rewards["no_stepping"].weight, 0.0)
        term = rewards["lifted_support_feet"]
        self.assertIs(term.func, lifted_support_feet)
        self.assertEqual(term.params["sensor_name"], "feet_ground_contact")
        self.assertEqual(term.params["sensor_foot_ids"], (1, 0))

    def test_the_contact_sensor_lists_the_right_foot_first(self) -> None:
        import mujoco

        from mjlab_microban.tasks.microban_velocity_env_cfg import make_microban_velocity_env_cfg

        sensor = next(s for s in make_microban_velocity_env_cfg().scene.sensors if s.name == "feet_ground_contact")
        xml = Path(fc.__file__).resolve().parents[1] / "robot" / "microban" / "robot.xml"
        model = mujoco.MjModel.from_xml_path(str(xml))
        bodies = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
        sensed = [b for b in bodies if re.fullmatch(sensor.primary.pattern, b)]  # model order
        site_body = {s: bodies[model.site_bodyid[model.site(s).id]] for s in ("left_foot", "right_foot")}
        self.assertEqual([sensed.index(site_body[s]) for s in ("left_foot", "right_foot")], [1, 0])

    def test_the_foot_stage_trains_half_the_samples_on_one_foot_standing(self) -> None:
        from mjlab_microban.tasks.microban_teleop_env_cfg import TELEOP_STAGES

        def values(path):
            return {stage.name: s.value for stage in TELEOP_STAGES for s in stage.settings
                    if s.manager == "command" and s.term == "foot_target" and s.path == path}

        foot = TELEOP_STAGES[1].name
        self.assertEqual(values("rel_single_support_envs"), {foot: 0.6})
        self.assertEqual(values("single_support_stationary_probability"), {foot: 0.85})
        # The two-foot targets are drawn first; the rest are single-foot ones.
        both = values("rel_both_feet_envs")
        for share in both.values():
            self.assertAlmostEqual(0.6 * (1.0 - share) * 0.85, 0.5, delta=0.05)

    def test_pico_feet_distance_is_below_every_single_foot_target(self) -> None:
        import mujoco

        from mjlab_microban.tasks.microban_teleop_env_cfg import TELEOP_STAGES
        from mjlab_microban.tasks.microban_teleop_v13_arm_overlay import (
            make_microban_teleop_v13_arm_overlay_env_cfg,
        )
        from mjlab_microban.tasks.microban_velocity_env_cfg import make_microban_velocity_env_cfg

        cfg = make_microban_teleop_v13_arm_overlay_env_cfg()
        xml = Path(fc.__file__).resolve().parents[1] / "robot" / "microban" / "robot.xml"
        model = mujoco.MjModel.from_xml_path(str(xml))
        data = mujoco.MjData(model)
        for name, value in cfg.scene.entities["robot"].init_state.joint_pos.items():
            data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
        mujoco.mj_kinematics(model, data)
        home = float(abs(data.site("left_foot").xpos[1] - data.site("right_foot").xpos[1]))
        self.assertAlmostEqual(home, 0.0928, delta=0.0005)
        # The lifted foot reaches at most this far in; the two-foot targets
        # move both feet by one offset.
        reach_in = max(abs(v) for v in cfg.commands["foot_target"].reach_xy_range)
        self.assertNotIn("reach_xy_range", [s.path for stage in TELEOP_STAGES for s in stage.settings])
        min_dist = cfg.rewards["feet_distance"].params["min_dist"]
        self.assertEqual(min_dist, 0.06)
        self.assertLess(min_dist, home - reach_in)
        self.assertEqual(make_microban_velocity_env_cfg().rewards["feet_distance"].params["min_dist"], 0.08)


if __name__ == "__main__":
    unittest.main()
