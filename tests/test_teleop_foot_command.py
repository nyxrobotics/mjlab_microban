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
from mjlab_microban.tasks.mdp import FootTargetCommand, lifted_support_feet
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

    def test_the_foot_stage_sets_half(self) -> None:
        from mjlab_microban.tasks.microban_teleop_env_cfg import TELEOP_STAGES

        values = [s.value for stage in TELEOP_STAGES for s in stage.settings
                  if s.path == "single_support_stationary_probability"]
        self.assertEqual(values, [0.5])


if __name__ == "__main__":
    unittest.main()
