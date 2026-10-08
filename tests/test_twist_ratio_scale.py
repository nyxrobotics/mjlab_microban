"""The twist-ratio velocity term as walking and PICO use it.

(a) PICO's command envelope peaks at the reward's axis scale on every axis;
(b) that scale is the robot's moving command limits; (c) walking and PICO pass
the same scale and HOME trunk pitch.  The reward term tables of both tasks are
pinned, and the PICO play environment the 9x300 source probe and the stage
gates build on is unchanged by the reward and schedule edits.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from mjlab_microban.robot.microban_constants import HOME_TRUNK_PITCH_RAD
from mjlab_microban.tasks.microban_teleop_env_cfg import (
    MICROBAN_TELEOP_FINAL_VELOCITY_ENVELOPE,
    PICO_TWIST_RATIO_WEIGHT,
)
from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release import (
    make_microban_teleop_v12_hand_pose_release_env_cfg,
)
from mjlab_microban.tasks.microban_twist_ratio_mdp import twist_ratio_velocity
from mjlab_microban.tasks.microban_velocity_env_cfg import (
    TWIST_AXIS_SCALE,
    WALK_TWIST_RATIO_WEIGHT,
    make_microban_velocity_env_cfg,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "teleop_play_env_snapshot.json"
# microban src/constants.py: VX_MAX, VY_MAX, VTHETA_MAX_MOVING (scale_velocity's
# limits while translating).
ROBOT_MOVING_COMMAND_LIMITS = (0.7, 0.3, 1.5)

WALK_REWARD_WEIGHTS = {
    "action_rate_l2": -0.1, "air_time": 3.0, "angular_momentum": -0.02, "body_ang_vel": -0.05,
    "dof_pos_limits": -1.0, "feet_distance": -1000.0, "foot_clearance": -2.0, "foot_slip": -1.0,
    "foot_swing_height": -0.25, "no_stepping": 0.0, "pose": 1.0, "self_collisions": -1.0,
    "twist_ratio_velocity": 8.0, "upright": 1.0,
}
PICO_ONLY_REWARD_WEIGHTS = {
    "action_rate_l2": -0.02, "dof_pos_limits": -10.0, "feet_distance": -100.0,
    "foot_target_tracking": 0.0, "hand_target_tracking": 0.0, "joint_soft_limit_guard": -5.0,
    "twist_ratio_velocity": 32.0,
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
        "hand": {"rel_active": cfg.commands["hand_target"].rel_active},
        "events": sorted(cfg.events),
        "curriculum": sorted(cfg.curriculum),
        "terminations": sorted(cfg.terminations),
    }
    return json.loads(json.dumps(snapshot, default=list))


class TwistRatioScaleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.walk = make_microban_velocity_env_cfg()
        cls.pico = make_microban_teleop_v12_hand_pose_release_env_cfg()

    def test_pico_envelope_peaks_at_the_axis_scale(self) -> None:
        envelope = MICROBAN_TELEOP_FINAL_VELOCITY_ENVELOPE
        peaks = tuple(
            max(abs(value) for value in envelope[axis]) for axis in ("lin_vel_x", "lin_vel_y", "ang_vel_z")
        )
        self.assertEqual(peaks, TWIST_AXIS_SCALE)

    def test_a_quarter_of_the_walking_commands_stand(self) -> None:
        # 2026-10-08: a tenth was too few for the walker to stop stepping in
        # place on the standing command (W6).
        self.assertEqual(self.walk.commands["twist"].rel_standing_envs, 0.25)
        self.assertEqual(make_microban_velocity_env_cfg(play=True).commands["twist"].rel_standing_envs, 0.0)

    def test_axis_scale_is_the_robot_moving_limits(self) -> None:
        self.assertEqual(TWIST_AXIS_SCALE, ROBOT_MOVING_COMMAND_LIMITS)

    def test_walk_and_pico_share_the_term(self) -> None:
        for cfg, weight in ((self.walk, WALK_TWIST_RATIO_WEIGHT), (self.pico, PICO_TWIST_RATIO_WEIGHT)):
            term = cfg.rewards["twist_ratio_velocity"]
            self.assertIs(term.func, twist_ratio_velocity)
            self.assertEqual(term.weight, weight)
            self.assertEqual(tuple(term.params["axis_scale"]), TWIST_AXIS_SCALE)
            self.assertEqual(term.params["trunk_pitch"], HOME_TRUNK_PITCH_RAD)
            self.assertEqual(term.params["command_name"], "twist")
            self.assertEqual(term.params["direction_penalty"], 1.0)
        self.assertEqual(PICO_TWIST_RATIO_WEIGHT, 4 * WALK_TWIST_RATIO_WEIGHT)

    def test_reward_tables(self) -> None:
        self.assertEqual({k: v.weight for k, v in self.walk.rewards.items()}, WALK_REWARD_WEIGHTS)
        pico = {k: v.weight for k, v in self.pico.rewards.items()}
        for name, weight in PICO_ONLY_REWARD_WEIGHTS.items():
            self.assertEqual(pico.pop(name), weight, name)
        for name in ("locomotion_prior_action_target", "locomotion_prior_joint_position"):
            pico.pop(name, None)
        for name, weight in pico.items():
            self.assertEqual(WALK_REWARD_WEIGHTS[name], weight, name)
        for removed in (
            "track_linear_velocity", "track_angular_velocity", "commanded_planar_velocity_progress",
            "linear_velocity_error_l1", "yaw_velocity_error_l1",
        ):
            self.assertNotIn(removed, self.walk.rewards)
            self.assertNotIn(removed, self.pico.rewards)

    def test_pico_play_env_is_unchanged(self) -> None:
        current = play_env_snapshot(make_microban_teleop_v12_hand_pose_release_env_cfg(play=True))
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
