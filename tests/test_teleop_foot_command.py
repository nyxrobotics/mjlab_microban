"""PICO foot targets (microban_teleop_foot_command.py): single-foot targets with a
standing twist, the 0.12 m/s target speed and the floor band."""

from __future__ import annotations

import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from mjlab_microban.tasks import microban_teleop_foot_command as fc
from mjlab_microban.tasks.mdp import FootTargetCommand, lifted_support_feet, upper_foot_lift, upper_foot_unload
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

    def test_upper_foot_unload_pays_the_higher_foot_off_its_share_of_the_weight(self) -> None:
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
        foot = SimpleNamespace(command=published.view(8, 6), is_single_support_env=single,
                               lifted_foot_idx=lifted)
        twist = torch.zeros(8, 3)
        terms = {"twist": SimpleNamespace(command=twist), "foot_target": foot}
        manager = SimpleNamespace(get_term=terms.__getitem__, get_command=lambda name: terms[name].command)
        sensor = SimpleNamespace(data=SimpleNamespace(found=torch.ones(8, 2), force=torch.zeros(8, 2, 3)))
        env = SimpleNamespace(num_envs=8, command_manager=manager, scene=SimpleNamespace(sensors={"feet": sensor}))

        def reward(load_left_right):  # newtons up from the floor; the sensor lists right before left
            load = torch.tensor(load_left_right, dtype=torch.float32).flip(-1)
            sensor.data.force = torch.zeros(8, 2, 3)
            sensor.data.force[..., 0] = 0.3  # a sideways push does not count
            sensor.data.force[..., 2] = -load  # the sensor reports the foot's push on the floor
            sensor.data.found = (load > 0).float()
            values = upper_foot_unload(env, "feet", "foot_target", sensor_foot_ids=(1, 0), lift_threshold=0.01)
            return [round(v, 4) for v in values.tolist()]

        # Target shares of the higher foot: 0, 0, 0, 0.25, 0, 0.4, (none), 0;
        # the higher foot is left, right, right, right, left, left, -, left.
        # Even weight: 1 - 2 x 0.5 = 0 from the threshold up, 0.5 at 5 mm,
        # 0.8 at 2 mm; no row without a height difference pays.
        self.assertEqual(reward([[5.5, 5.5]] * 8), [0.0, 0.0, 0.0, 0.5, 0.0, 0.8, 0.0, 0.0])
        # All on the right foot: the higher left foot carries nothing.
        self.assertEqual(reward([[0.0, 11.0]] * 8), [0.0, 1.0, 0.0, 0.0, 1.0, 0.2, 0.0, 1.0])
        # All on the left foot: the higher right foot carries nothing.
        self.assertEqual(reward([[11.0, 0.0]] * 8), [0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0])
        # A quarter on the higher foot: 1 - 2 x 0.25 above the threshold,
        # exactly the 5 mm target, 1 - 2 x 0.15 at 2 mm.
        self.assertEqual(reward([[2.75, 8.25]] * 8), [0.0, 0.5, 0.0, 0.0, 0.5, 0.7, 0.0, 0.5])
        self.assertEqual(reward([[8.25, 2.75]] * 8), [0.0, 0.0, 0.5, 1.0, 0.0, 0.3, 0.0, 0.0])
        # Both feet in the air (or all but a trace): 0.
        self.assertEqual(reward([[0.0, 0.0]] * 8), [0.0] * 8)
        self.assertEqual(reward([[0.0, 0.05]] * 8), [0.0] * 8)
        # The higher foot is never one lifted_support_feet asks down: all the
        # weight on the other foot pays in full from the threshold up and
        # lifted_support_feet counts no foot.
        higher_left = [[0.0, 11.0]] * 8
        for row in (2, 3):
            higher_left[row] = [11.0, 0.0]
        self.assertEqual(reward(higher_left), [0.0, 1.0, 1.0, 0.5, 1.0, 0.2, 0.0, 1.0])
        sensor.data.found[0] = 1.0
        sensor.data.found[6] = 1.0
        self.assertEqual(lifted_support_feet(env, "feet", "foot_target", sensor_foot_ids=(1, 0)).tolist(), [0.0] * 8)
        # A walking command pays nothing; a slow one below 1 cm/s still pays.
        twist[1] = torch.tensor([0.1, 0.0, 0.0])
        twist[2] = torch.tensor([0.0, 0.0, 0.4])
        twist[4] = torch.tensor([0.004, 0.0, 0.004])
        self.assertEqual(reward([[0.0, 11.0]] * 8), [0.0, 0.0, 0.0, 0.0, 1.0, 0.2, 0.0, 1.0])
        self.assertEqual(reward([[11.0, 0.0]] * 8), [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0])

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
            MICROBAN_TELEOP_UPPER_FOOT_LIFT_WEIGHT,
            MICROBAN_TELEOP_UPPER_FOOT_UNLOAD_WEIGHT,
            TELEOP_STAGES,
        )
        from mjlab_microban.tasks.microban_teleop_v13_arm_overlay import (
            make_microban_teleop_v13_arm_overlay_env_cfg,
        )

        settings = [(stage.name, s.path, s.value) for stage in TELEOP_STAGES for s in stage.settings
                    if s.manager == "reward" and s.term == "upper_foot_lift"]
        self.assertEqual(settings, [(TELEOP_STAGES[1].name, "weight", MICROBAN_TELEOP_UPPER_FOOT_LIFT_WEIGHT)])
        self.assertEqual(MICROBAN_TELEOP_UPPER_FOOT_LIFT_WEIGHT, MICROBAN_TELEOP_UPPER_FOOT_UNLOAD_WEIGHT)
        rewards = make_microban_teleop_v13_arm_overlay_env_cfg().rewards
        term = rewards["upper_foot_lift"]
        self.assertIs(term.func, upper_foot_lift)
        self.assertEqual(term.weight, 0.0)
        self.assertEqual(term.params, rewards["upper_foot_unload"].params)

    def test_pico_rewards_unloading_the_higher_foot_from_the_foot_stage(self) -> None:
        from mjlab_microban.tasks.microban_teleop_env_cfg import (
            MICROBAN_TELEOP_SINGLE_SUPPORT_LIFT_THRESHOLD_M,
            MICROBAN_TELEOP_UPPER_FOOT_UNLOAD_WEIGHT,
            TELEOP_STAGES,
        )
        from mjlab_microban.tasks.microban_teleop_mdp import MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M
        from mjlab_microban.tasks.microban_teleop_v13_arm_overlay import (
            make_microban_teleop_v13_arm_overlay_env_cfg,
        )

        settings = [(stage.name, s.path, s.value) for stage in TELEOP_STAGES for s in stage.settings
                    if s.manager == "reward" and s.term == "upper_foot_unload"]
        self.assertEqual(settings, [(TELEOP_STAGES[1].name, "weight", MICROBAN_TELEOP_UPPER_FOOT_UNLOAD_WEIGHT)])
        # Above the 6.2 /s the measured shift to one foot costs in the other terms.
        self.assertGreater(MICROBAN_TELEOP_UPPER_FOOT_UNLOAD_WEIGHT, 6.2)
        cfg = make_microban_teleop_v13_arm_overlay_env_cfg()
        self.assertNotIn("single_support", cfg.rewards)
        term = cfg.rewards["upper_foot_unload"]
        self.assertIs(term.func, upper_foot_unload)
        self.assertEqual(term.weight, 0.0)
        penalty = cfg.rewards["lifted_support_feet"].params
        self.assertEqual({k: v for k, v in term.params.items() if k != "lift_threshold"}, penalty)
        threshold = term.params["lift_threshold"]
        self.assertEqual(threshold, MICROBAN_TELEOP_SINGLE_SUPPORT_LIFT_THRESHOLD_M)
        # Above the floor band, below the highest single-foot target.
        self.assertGreater(threshold, MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M)
        self.assertLess(threshold, cfg.commands["foot_target"].lift_height_range[1])

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


if __name__ == "__main__":
    unittest.main()
