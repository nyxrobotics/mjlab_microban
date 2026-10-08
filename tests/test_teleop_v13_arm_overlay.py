"""V13 arm-overlay recipe: the leg pose release, the arm overlay and recipe gating."""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import torch
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp as velocity_mdp

from mjlab_microban import policy_contract
from mjlab_microban.robot.home_pose import HOME
from mjlab_microban.robot.microban_hand_fk import microban_hand_positions_from_arm_joints
from mjlab_microban.scripts.evaluate_teleop_v12_checkpoint import recipe_report_settings
from mjlab_microban.scripts.export_teleop_v12_deployment import _deployment_recipe_revision
from mjlab_microban.scripts.teleop_v12_scenarios import (
    ARMS_FORWARD_70,
    ARMS_HALF_LEFT,
    ARMS_HALF_RIGHT,
    ARMS_REACH_LEFT,
    ARMS_REACH_RIGHT,
)
from mjlab_microban.tasks import microban_teleop_mdp as mdp
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MICROBAN_TELEOP_V12_RECIPE_REVISION,
    MICROBAN_TELEOP_V13_ARM_OVERLAY_RECIPE_REVISION,
)
from mjlab_microban.tasks.microban_teleop_v12_home_pose import (
    TELEOP_V12_HOME_POSE_INFO_KEY,
    teleop_v12_home_pose_marker,
    validate_teleop_v12_home_pose,
)
from mjlab_microban.tasks.microban_teleop_v13_arm_overlay import (
    MICROBAN_TELEOP_LEG_JOINT_NAMES,
    foot_target_leg_released_posture,
    make_microban_teleop_v13_arm_overlay_env_cfg,
)

JOINTS = ["left_knee", "right_knee"]
STD = {r".*knee.*": 0.15}


class _Asset:
    def __init__(self, joint_pos: torch.Tensor) -> None:
        self.data = SimpleNamespace(default_joint_pos=torch.zeros_like(joint_pos), joint_pos=joint_pos)

    def find_joints(self, names):
        del names
        return list(range(len(JOINTS))), list(JOINTS)


def _env(joint_pos: torch.Tensor, flags: torch.Tensor, published: torch.Tensor, twist: torch.Tensor):
    foot = SimpleNamespace(
        is_single_support_env=flags, is_both_feet_env=torch.zeros_like(flags), command=published
    )
    commands = SimpleNamespace(get_command=lambda name: twist, get_term=lambda name: foot)
    rewards = SimpleNamespace(
        get_term_cfg=lambda name: SimpleNamespace(params={"velocity_fade_range": (0.0, 0.15)})
    )
    return SimpleNamespace(
        scene={"robot": _Asset(joint_pos)},
        device="cpu",
        num_envs=joint_pos.shape[0],
        command_manager=commands,
        reward_manager=rewards,
    )


def _cfg() -> RewardTermCfg:
    asset_cfg = SceneEntityCfg("robot", joint_names=tuple(JOINTS))
    asset_cfg.joint_ids = slice(None)
    return RewardTermCfg(
        func=foot_target_leg_released_posture,
        weight=1.0,
        params={
            "asset_cfg": asset_cfg,
            "command_name": "twist",
            "foot_command_name": "foot_target",
            "foot_reward_name": "foot_target_tracking",
            "std_standing": STD,
            "std_walking": STD,
            "std_running": STD,
            "walking_threshold": 0.01,
        },
    )


class LegPoseReleaseTest(unittest.TestCase):
    def test_foot_target_rows_are_released_and_others_keep_the_leg_term(self) -> None:
        joint_pos = torch.full((4, 2), 0.6)  # a lifted foot bends the knee
        flags = torch.tensor([False, True, False, True])
        published = torch.zeros(4, 6)
        published[2, 2] = 0.01  # a target still on its way down
        twist = torch.zeros(4, 3)
        twist[3, 0] = 0.2  # walking: the foot reward is faded out
        env = _env(joint_pos, flags, published, twist)
        cfg = _cfg()
        params = dict(cfg.params)
        released = foot_target_leg_released_posture(cfg, env)(env, **params)
        params.pop("foot_command_name")
        params.pop("foot_reward_name")
        canonical = velocity_mdp.variable_posture(cfg, env)(env, **params)
        torch.testing.assert_close(released[[0, 3]], canonical[[0, 3]])
        torch.testing.assert_close(released[[1, 2]], torch.ones(2))
        self.assertLess(float(canonical[0]), 0.01)

    def test_the_task_poses_only_the_legs(self) -> None:
        cfg = make_microban_teleop_v13_arm_overlay_env_cfg()
        pose = cfg.rewards["pose"]
        self.assertIs(pose.func, foot_target_leg_released_posture)
        self.assertEqual(tuple(pose.params["asset_cfg"].joint_names), MICROBAN_TELEOP_LEG_JOINT_NAMES)
        self.assertEqual(len(MICROBAN_TELEOP_LEG_JOINT_NAMES), 12)
        for key in ("std_standing", "std_walking", "std_running"):
            self.assertFalse(any("shoulder" in k or "elbow" in k or "neck" in k or "head" in k
                                 for k in pose.params[key]))
        self.assertIsInstance(cfg.actions["joint_pos"], mdp.PicoArmOverlayJointPositionActionCfg)


class ArmTargetTest(unittest.TestCase):
    def test_the_box_is_the_robot_contract(self) -> None:
        self.assertEqual(policy_contract.PICO_ARM_SLEW_RATE_RAD_S, 4.0)
        degrees = [round(math.degrees(v), 9) for v in policy_contract.PICO_ARM_LOWER_RAD]
        self.assertEqual(degrees, [-100.0, 10.0, -110.0, -100.0, -120.0, -110.0])
        degrees = [round(math.degrees(v), 9) for v in policy_contract.PICO_ARM_UPPER_RAD]
        self.assertEqual(degrees, [100.0, 120.0, 0.0, 100.0, -10.0, 0.0])
        for name, lower, upper in zip(policy_contract.PICO_ARM_JOINT_NAMES, policy_contract.PICO_ARM_LOWER_RAD,
                                      policy_contract.PICO_ARM_UPPER_RAD, strict=True):
            self.assertTrue(lower <= HOME.joint_pos_rad[name] <= upper, name)

    def test_samples_stay_in_the_box_with_the_hands_clear_of_the_trunk(self) -> None:
        torch.manual_seed(3)
        draws = mdp.sample_pico_arm_targets(4000, device="cpu")
        lower = torch.tensor(policy_contract.PICO_ARM_LOWER_RAD)
        upper = torch.tensor(policy_contract.PICO_ARM_UPPER_RAD)
        self.assertTrue(bool(((draws >= lower) & (draws <= upper)).all()))
        hands = microban_hand_positions_from_arm_joints(draws.view(-1, 2, 3))
        self.assertTrue(bool((hands[:, 0, 1] >= 0.0).all() and (hands[:, 1, 1] <= 0.0).all()))
        # Every joint still covers most of its box.
        span = (draws.amax(0) - draws.amin(0)) / (upper - lower)
        self.assertGreater(float(span.min()), 0.9)

    def test_evaluation_poses_are_drawable(self) -> None:
        home = torch.tensor([HOME.joint_pos_rad[n] for n in policy_contract.PICO_ARM_JOINT_NAMES])
        lower = torch.tensor(policy_contract.PICO_ARM_LOWER_RAD)
        upper = torch.tensor(policy_contract.PICO_ARM_UPPER_RAD)
        for pose in (ARMS_FORWARD_70, ARMS_REACH_LEFT, ARMS_REACH_RIGHT, ARMS_HALF_LEFT, ARMS_HALF_RIGHT):
            absolute = home + torch.tensor(pose)
            self.assertTrue(bool(((absolute >= lower) & (absolute <= upper)).all()), pose)
            hands = microban_hand_positions_from_arm_joints(absolute.view(2, 3))
            self.assertGreater(float(hands[0, 1]), 0.0)
            self.assertLess(float(hands[1, 1]), 0.0)
        # A negative shoulder pitch raises the arm forward.
        forward = microban_hand_positions_from_arm_joints((home + torch.tensor(ARMS_FORWARD_70)).view(2, 3))
        self.assertGreater(float(forward[0, 0]), 0.08)


class ArmOverlayRecipeTest(unittest.TestCase):
    def _infos(self, recipe: str) -> dict:
        return {
            "microban_teleop_recipe_revision": recipe,
            TELEOP_V12_HOME_POSE_INFO_KEY: teleop_v12_home_pose_marker(),
        }

    def test_the_recipe_string(self) -> None:
        self.assertTrue(MICROBAN_TELEOP_V13_ARM_OVERLAY_RECIPE_REVISION.startswith(f"{HOME.tag}_"))
        self.assertIn("arm_overlay_leg_pose_release", MICROBAN_TELEOP_V13_ARM_OVERLAY_RECIPE_REVISION)
        infos = self._infos(MICROBAN_TELEOP_V13_ARM_OVERLAY_RECIPE_REVISION)
        validate_teleop_v12_home_pose(infos)
        self.assertEqual(_deployment_recipe_revision(infos), MICROBAN_TELEOP_V13_ARM_OVERLAY_RECIPE_REVISION)
        self.assertEqual(recipe_report_settings(infos),
                         {"recipe_revision": MICROBAN_TELEOP_V13_ARM_OVERLAY_RECIPE_REVISION})
        self.assertEqual(recipe_report_settings(self._infos(MICROBAN_TELEOP_V12_RECIPE_REVISION)), {})

    def test_another_recipe_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "Checkpoint recipe does not match"):
            validate_teleop_v12_home_pose(self._infos(MICROBAN_TELEOP_V12_RECIPE_REVISION))
        with self.assertRaisesRegex(ValueError, "Only an arm-overlay checkpoint"):
            _deployment_recipe_revision(self._infos(MICROBAN_TELEOP_V12_RECIPE_REVISION))


if __name__ == "__main__":
    unittest.main()
