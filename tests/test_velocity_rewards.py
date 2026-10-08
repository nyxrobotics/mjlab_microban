"""The velocity rewards of walking and PICO, and the PICO play environment.

Walking tracks velocity with mjlab's two exp tracking terms in the
HOME-levelled frame, PICO with its planar and yaw terms in that frame.  The
reward term tables of both tasks are pinned, PICO's command envelope peaks at
the robot's moving command limits, and the PICO play environment the 9x300
source probe and the stage gates build on is pinned.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from mjlab_microban.robot.microban_constants import HOME_TRUNK_PITCH_RAD
from mjlab_microban.tasks.microban_teleop_env_cfg import MICROBAN_TELEOP_FINAL_VELOCITY_ENVELOPE
from mjlab_microban.tasks.microban_teleop_velocity_rewards import (
    commanded_planar_velocity_progress,
    linear_velocity_tracking_error_l1,
    planar_velocity_tracking_exp,
    yaw_velocity_tracking_error_l1,
)
from mjlab_microban.tasks.microban_teleop_v13_arm_overlay import (
    make_microban_teleop_v13_arm_overlay_env_cfg,
)
from mjlab_microban.tasks.microban_velocity_env_cfg import make_microban_velocity_env_cfg
from mjlab_microban.tasks.microban_velocity_tracking import (
    track_angular_velocity_home_frame,
    track_linear_velocity_home_frame,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "teleop_play_env_snapshot.json"
# microban src/constants.py: VX_MAX, VY_MAX, VTHETA_MAX_MOVING (scale_velocity's
# limits while translating).
ROBOT_MOVING_COMMAND_LIMITS = (0.7, 0.3, 1.5)

WALK_REWARD_WEIGHTS = {
    "action_rate_l2": -0.1, "air_time": 3.0, "angular_momentum": -0.02, "body_ang_vel": -0.05,
    "dof_pos_limits": -1.0, "feet_distance": -1000.0, "foot_clearance": -2.0, "foot_slip": -1.0,
    "foot_swing_height": -0.25, "no_stepping": 0.0, "pose": 1.0, "self_collisions": -1.0,
    "track_angular_velocity": 2.0, "track_linear_velocity": 2.0, "upright": 1.0,
}
PICO_ONLY_REWARD_WEIGHTS = {
    "action_rate_l2": -0.02, "dof_pos_limits": -10.0, "feet_distance": -100.0,
    "foot_target_tracking": 1.0, "joint_soft_limit_guard": -5.0,
    "track_linear_velocity": 5.0, "commanded_planar_velocity_progress": 2.0,
    "linear_velocity_error_l1": -16.0, "yaw_velocity_error_l1": -1.0,
}


def play_env_snapshot(cfg) -> dict:
    """The PICO play-env values the 9x300 probe and the stage gates run under."""

    twist = cfg.commands["twist"]
    names = (
        "rel_standing_envs", "rel_heading_envs", "rel_rotation_envs", "rel_forward_envs",
        "rel_world_envs", "init_velocity_prob", "resampling_time_range",
        "rotation_env_ang_vel_range", "rotation_min_ang_vel", "signed_axis_ranges",
        "signed_axis_probabilities",
    )
    foot = cfg.commands["foot_target"]
    snapshot = {
        "twist": {name: getattr(twist, name) for name in names if hasattr(twist, name)},
        "twist_ranges": {
            name: getattr(twist.ranges, name) for name in ("lin_vel_x", "lin_vel_y", "ang_vel_z", "heading")
        },
        "push": cfg.events["push_robot"].params["velocity_range"],
        "foot": {
            name: getattr(foot, name)
            for name in (
                "rel_single_support_envs", "rel_both_feet_envs", "lift_height_range",
                "both_feet_lift_height_range", "reach_xy_range", "both_feet_reach_xy_range",
            )
        },
        "action": type(cfg.actions["joint_pos"]).__name__,
        "events": sorted(cfg.events),
        "curriculum": sorted(cfg.curriculum),
        "terminations": sorted(cfg.terminations),
    }
    return json.loads(json.dumps(snapshot, default=list))


class VelocityRewardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.walk = make_microban_velocity_env_cfg()
        cls.pico = make_microban_teleop_v13_arm_overlay_env_cfg()

    def test_pico_envelope_peaks_at_the_robot_moving_limits(self) -> None:
        envelope = MICROBAN_TELEOP_FINAL_VELOCITY_ENVELOPE
        peaks = tuple(
            max(abs(value) for value in envelope[axis]) for axis in ("lin_vel_x", "lin_vel_y", "ang_vel_z")
        )
        self.assertEqual(peaks, ROBOT_MOVING_COMMAND_LIMITS)

    def test_the_velocity_terms(self) -> None:
        # Walking: mjlab's exp tracking at HOME, std sqrt(0.1) / sqrt(0.5),
        # weight 2 each.
        linear, angular = self.walk.rewards["track_linear_velocity"], self.walk.rewards["track_angular_velocity"]
        self.assertIs(linear.func, track_linear_velocity_home_frame)
        self.assertIs(angular.func, track_angular_velocity_home_frame)
        self.assertAlmostEqual(linear.params["std"] ** 2, 0.1)
        self.assertAlmostEqual(angular.params["std"] ** 2, 0.5)
        for term in (linear, angular):
            self.assertEqual(term.params["trunk_pitch"], HOME_TRUNK_PITCH_RAD)
        # PICO: planar exp tracking, progress and L1 errors, final stds.
        terms = self.pico.rewards
        self.assertIs(terms["track_linear_velocity"].func, planar_velocity_tracking_exp)
        self.assertEqual(terms["track_linear_velocity"].params["std"], 0.5)
        self.assertIs(terms["track_angular_velocity"].func, track_angular_velocity_home_frame)
        self.assertEqual(terms["track_angular_velocity"].params["std"], 1.25)
        self.assertIs(terms["commanded_planar_velocity_progress"].func, commanded_planar_velocity_progress)
        self.assertIs(terms["linear_velocity_error_l1"].func, linear_velocity_tracking_error_l1)
        self.assertIs(terms["yaw_velocity_error_l1"].func, yaw_velocity_tracking_error_l1)
        for name in ("track_linear_velocity", "commanded_planar_velocity_progress", "linear_velocity_error_l1",
                     "yaw_velocity_error_l1"):
            self.assertEqual(terms[name].params["trunk_pitch"], HOME_TRUNK_PITCH_RAD, name)

    def test_reward_tables(self) -> None:
        self.assertEqual({k: v.weight for k, v in self.walk.rewards.items()}, WALK_REWARD_WEIGHTS)
        pico = {k: v.weight for k, v in self.pico.rewards.items()}
        for name, weight in PICO_ONLY_REWARD_WEIGHTS.items():
            self.assertEqual(pico.pop(name), weight, name)
        for name, weight in pico.items():
            self.assertEqual(WALK_REWARD_WEIGHTS[name], weight, name)

    def test_pico_play_env_is_unchanged(self) -> None:
        current = play_env_snapshot(make_microban_teleop_v13_arm_overlay_env_cfg(play=True))
        self.assertEqual(current, json.loads(FIXTURE.read_text()))

    def test_pico_training_samples_the_full_envelope_with_pushes(self) -> None:
        twist = self.pico.commands["twist"]
        envelope = MICROBAN_TELEOP_FINAL_VELOCITY_ENVELOPE
        self.assertEqual(twist.ranges.lin_vel_x, envelope["lin_vel_x"])
        self.assertEqual(twist.ranges.lin_vel_y, envelope["lin_vel_y"])
        self.assertEqual(twist.ranges.ang_vel_z, envelope["ang_vel_z"])
        self.assertEqual(twist.rotation_env_ang_vel_range, envelope["rotation_ang_vel_z"])
        self.assertEqual(
            self.pico.events["push_robot"].params["velocity_range"], {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}
        )


if __name__ == "__main__":
    unittest.main()
