"""Opt-in active-hand arm pose-release recipe: reward math and recipe gating."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp as velocity_mdp

from mjlab_microban.tasks.microban_teleop_v12_corner_rescue import (
    validate_corner_rescue_canonical_lineage,
)
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION,
    MICROBAN_TELEOP_V12_RECIPE_REVISION,
)
from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release import (
    active_hand_arm_released_posture,
)
from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release_runner import (
    validate_hand_pose_release_switch_marker,
)
from mjlab_microban.tasks.microban_teleop_v12_home_pose import (
    TELEOP_V12_HOME_POSE_INFO_KEY,
    teleop_v12_home_pose_marker,
    validate_teleop_v12_home_pose,
)

JOINTS = [
    "head",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_elbow",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_elbow",
    "left_knee",
]
STD = {r".*": 0.1}


class _Asset:
    def __init__(self, joint_pos: torch.Tensor) -> None:
        self.data = SimpleNamespace(
            default_joint_pos=torch.zeros_like(joint_pos), joint_pos=joint_pos
        )

    def find_joints(self, names):
        del names
        return list(range(len(JOINTS))), list(JOINTS)


def _env(joint_pos: torch.Tensor, active: torch.Tensor):
    asset = _Asset(joint_pos)
    hand = SimpleNamespace(is_active=active)
    twist = torch.zeros(joint_pos.shape[0], 3)
    commands = SimpleNamespace(
        get_command=lambda name: twist, get_term=lambda name: hand
    )
    return SimpleNamespace(
        scene={"robot": asset}, device="cpu", command_manager=commands
    )


def _cfg() -> RewardTermCfg:
    asset_cfg = SceneEntityCfg("robot", joint_names=(".*",))
    asset_cfg.joint_ids = slice(None)
    return RewardTermCfg(
        func=active_hand_arm_released_posture,
        weight=1.0,
        params={
            "asset_cfg": asset_cfg,
            "command_name": "twist",
            "hand_command_name": "hand_target",
            "std_standing": STD,
            "std_walking": STD,
            "std_running": STD,
            "walking_threshold": 0.01,
        },
    )


def _call(term, env, cfg):
    return term(env, **cfg.params)


class HandPoseReleaseRewardTest(unittest.TestCase):
    def test_inactive_rows_equal_v11_and_active_arms_are_dropped(self) -> None:
        joint_pos = torch.zeros(4, len(JOINTS))
        joint_pos[:, 1:4] = 0.3  # left arm far from HOME
        joint_pos[:, 4:7] = 0.2  # right arm
        joint_pos[:, 7] = 0.05
        active = torch.tensor(
            [[False, False], [True, False], [False, True], [True, True]]
        )
        env = _env(joint_pos, active)
        cfg = _cfg()
        term = active_hand_arm_released_posture(cfg, env)
        base = velocity_mdp.variable_posture(cfg, env)
        released = _call(term, env, cfg)
        params = dict(cfg.params)
        params.pop("hand_command_name")
        v11 = base(env, **params)
        self.assertTrue(torch.equal(released[0], v11[0]))
        err = joint_pos.square() / 0.01
        expected = torch.stack(
            [
                torch.exp(-err[0].mean()),
                torch.exp(-err[1, [0, 4, 5, 6, 7]].mean()),
                torch.exp(-err[2, [0, 1, 2, 3, 7]].mean()),
                torch.exp(-err[3, [0, 7]].mean()),
            ]
        )
        torch.testing.assert_close(released, expected)
        self.assertGreater(float(released[3]), float(v11[3]))

    def test_no_active_hand_is_bit_identical_to_v11(self) -> None:
        joint_pos = torch.randn(8, len(JOINTS)) * 0.2
        env = _env(joint_pos, torch.zeros(8, 2, dtype=torch.bool))
        cfg = _cfg()
        params = dict(cfg.params)
        params.pop("hand_command_name")
        v11 = velocity_mdp.variable_posture(cfg, env)(env, **params)
        released = _call(active_hand_arm_released_posture(cfg, env), env, cfg)
        self.assertTrue(torch.equal(released, v11))


class HandPoseReleaseRecipeGateTest(unittest.TestCase):
    def _infos(self, recipe: str) -> dict:
        return {
            "microban_teleop_recipe_revision": recipe,
            TELEOP_V12_HOME_POSE_INFO_KEY: teleop_v12_home_pose_marker(),
        }

    def test_canonical_validators_refuse_the_new_recipe_by_default(self) -> None:
        infos = self._infos(MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION)
        with self.assertRaises(ValueError):
            validate_teleop_v12_home_pose(infos)
        with self.assertRaises(ValueError):
            validate_corner_rescue_canonical_lineage(infos, iteration=9999)
        validate_teleop_v12_home_pose(infos, allow_hand_pose_release_recipe=True)
        self.assertIsNone(
            validate_corner_rescue_canonical_lineage(
                infos, iteration=9999, allow_hand_pose_release_recipe=True
            )
        )
        validate_teleop_v12_home_pose(
            self._infos(MICROBAN_TELEOP_V12_RECIPE_REVISION),
            allow_hand_pose_release_recipe=True,
        )

    def test_switch_marker_is_never_release_eligible(self) -> None:
        marker = {
            "schema_version": 1,
            "release_eligible": False,
            "parent_recipe_revision": MICROBAN_TELEOP_V12_RECIPE_REVISION,
            "recipe_revision": MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION,
            "parent_checkpoint_sha256": "0" * 64,
            "parent_iteration": 7099,
        }
        validate_hand_pose_release_switch_marker(marker)
        with self.assertRaises(ValueError):
            validate_hand_pose_release_switch_marker(
                {**marker, "release_eligible": True}
            )


if __name__ == "__main__":
    unittest.main()
