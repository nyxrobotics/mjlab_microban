"""Opt-in v12 recipe: release the arms of active hands from the HOME pose reward.

The inherited velocity ``pose`` term (weight 1.0, arm std 0.1 rad while
standing) pulls every joint toward HOME.  Once hand targets activate (update
7000) it fights the hand-tracking reward: a full reach to a corner of the
reachable box drops the pose term from about 0.31 to 0.02 per step, and the v11
chain settles on a steady ~40 % undershoot at the box corners.

This recipe keeps the pose term exactly as in v11 except that, per env, the
shoulder-pitch, shoulder-roll and elbow of each hand whose target is active are
left out of the mean.  An inactive hand's arm and all other joints keep the v11
term.  Before update 7000 no hand is active, so the term equals v11 bit for bit.

It is a separate recipe revision with its own task; the canonical task, stage
gates and exporter are unchanged and keep refusing it.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp as velocity_mdp

from mjlab_microban.robot.microban_hand_fk import (
    MICROBAN_ARM_JOINT_ORDER,
    MICROBAN_HAND_SIDE_ORDER,
)
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MicrobanTeleopV12RlCfg,
    MicrobanTeleopV12RunnerCfg,
    make_microban_teleop_v12_env_cfg,
)

MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_TASK_ID = (
    "Mjlab-Teleop-V12-HandPoseRelease-Microban"
)
# Checkpoint infos key written only when a run switched an existing v11
# checkpoint to this recipe (an experiment); a fresh chain never carries it.
MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_SWITCH_INFO_KEY = (
    "microban_teleop_v12_experimental_recipe_switch"
)
MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_ARM_JOINT_NAMES = tuple(
    tuple(f"{side}_{joint}" for joint in MICROBAN_ARM_JOINT_ORDER)
    for side in MICROBAN_HAND_SIDE_ORDER
)


class active_hand_arm_released_posture(velocity_mdp.variable_posture):
    """``variable_posture`` without the arm joints of hands with an active target.

    Per env the reward is ``exp(-mean_j(err_j^2 / std_j^2))`` over the joints
    that remain after removing, for every active hand, that side's
    shoulder-pitch/shoulder-roll/elbow.  Rows with no active hand use the
    unmodified parent formula, so they are bit-identical to v11.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        asset = env.scene[cfg.params["asset_cfg"].name]
        _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)
        joint_names = list(joint_names)
        release = torch.zeros(
            len(MICROBAN_HAND_SIDE_ORDER), len(joint_names), dtype=torch.float32
        )
        for side_index, side_names in enumerate(
            MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_ARM_JOINT_NAMES
        ):
            for name in side_names:
                if joint_names.count(name) != 1:
                    raise ValueError(f"Pose reward must cover arm joint {name} once")
                release[side_index, joint_names.index(name)] = 1.0
        # (2, J): 1 where that side's arm joint is released when it is active.
        self.arm_release_mask = release.to(env.device)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        std_standing,
        std_walking,
        std_running,
        asset_cfg: SceneEntityCfg,
        command_name: str,
        hand_command_name: str,
        walking_threshold: float = 0.5,
        running_threshold: float = 1.5,
    ) -> torch.Tensor:
        parent = super().__call__(
            env,
            std_standing,
            std_walking,
            std_running,
            asset_cfg,
            command_name,
            walking_threshold,
            running_threshold,
        )
        hand = env.command_manager.get_term(hand_command_name)
        active = hand.is_active.to(dtype=torch.float32)  # (N, 2)
        any_active = active.sum(dim=-1) > 0.0
        if not bool(any_active.any().item()):
            return parent

        command = env.command_manager.get_command(command_name)
        total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
        standing = (total_speed < walking_threshold).float().unsqueeze(1)
        walking = (
            ((total_speed >= walking_threshold) & (total_speed < running_threshold))
            .float()
            .unsqueeze(1)
        )
        running = (total_speed >= running_threshold).float().unsqueeze(1)
        std = (
            self.std_standing * standing
            + self.std_walking * walking
            + self.std_running * running
        )
        asset = env.scene[asset_cfg.name]
        error_squared = torch.square(
            asset.data.joint_pos[:, asset_cfg.joint_ids]
            - self.default_joint_pos[:, asset_cfg.joint_ids]
        )
        released = torch.clamp(active @ self.arm_release_mask, max=1.0)  # (N, J)
        kept = 1.0 - released
        masked_mean = torch.sum(kept * error_squared / std.square(), dim=1) / (
            kept.sum(dim=1)
        )
        return torch.where(any_active, torch.exp(-masked_mean), parent)


def apply_active_hand_arm_pose_release(cfg: ManagerBasedRlEnvCfg) -> None:
    """Swap the inherited pose term for the active-hand arm-release variant."""

    pose = cfg.rewards.get("pose")
    if pose is None or pose.func is not velocity_mdp.variable_posture:
        raise TypeError("Hand pose release requires the inherited variable_posture")
    if "hand_target" not in cfg.commands:
        raise KeyError("Hand pose release requires the hand_target command")
    pose.func = active_hand_arm_released_posture
    pose.params["hand_command_name"] = "hand_target"


def make_microban_teleop_v12_hand_pose_release_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Build the v12 task with only the pose term changed (see module doc)."""

    cfg = make_microban_teleop_v12_env_cfg(play=play)
    apply_active_hand_arm_pose_release(cfg)
    return cfg


@dataclass
class MicrobanTeleopV12HandPoseReleaseRunnerCfg(MicrobanTeleopV12RunnerCfg):
    """Adds the explicit experiment switch for resuming a v11 checkpoint."""

    # Resuming a v11 checkpoint into this recipe mixes two recipes in one
    # lineage.  It is refused unless this flag is set, and every save of such a
    # run records a not-for-release switch marker.  A fresh chain leaves it off.
    experimental_recipe_switch: bool = False


MicrobanTeleopV12HandPoseReleaseRlCfg = MicrobanTeleopV12HandPoseReleaseRunnerCfg(
    **{
        name: deepcopy(getattr(MicrobanTeleopV12RlCfg, name))
        for name in MicrobanTeleopV12RlCfg.__dataclass_fields__
    }
)
MicrobanTeleopV12HandPoseReleaseRlCfg.wandb_project = (
    "mjlab_microban_teleop_v12_hand_pose_release"
)
