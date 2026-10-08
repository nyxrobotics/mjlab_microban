"""V13 PICO recipe: arms driven from outside, legs released from the HOME pose reward.

The robot's direct-IK ``pico_arms`` move owns the six arm servos during PICO
teleoperation, so this recipe trains the same way: the arm targets come from
outside (microban_teleop_mdp.PicoArmOverlayJointPositionAction) and the policy
observes them (``arm_target``), with no hand-tracking reward.

The inherited velocity ``pose`` term pulls every joint toward HOME.  HOME's
knee is straight, so lifting a foot 40 mm costs the term almost all of its
value (1 -> 0.004 with the standing stds) and training never lifted a foot.
Here the term covers only the twelve leg joints the policy drives (the arms
and the neck are driven from outside), and on a row with a foot target whose
foot reward is not faded out it is 1.0: the legs are free to follow the foot
target.  (1.0, not 0: a smaller value would make a foot-target row less worth
surviving.)
"""

from __future__ import annotations

from copy import deepcopy

import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp as velocity_mdp

from mjlab_microban.tasks.mdp import foot_target_active
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MicrobanTeleopV12RlCfg,
    make_microban_teleop_v12_env_cfg,
)

MICROBAN_TELEOP_V13_ARM_OVERLAY_TASK_ID = "Mjlab-Teleop-V13-ArmOverlay-Microban"
# The joints the PICO policy drives: the legs (natural joint order).
MICROBAN_TELEOP_LEG_JOINT_NAMES = tuple(
    f"{side}_{joint}"
    for side in ("right", "left")
    for joint in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll")
)
_LEG_STD_PATTERNS = (
    r".*hip_roll.*",
    r".*hip_pitch.*",
    r".*hip_yaw.*",
    r".*knee.*",
    r".*ankle_pitch.*",
    r".*ankle_roll.*",
)


class foot_target_leg_released_posture(velocity_mdp.variable_posture):
    """``variable_posture`` over the leg joints, 1.0 on rows tracking a foot target.

    A row tracks a foot target when one is drawn or still published
    (mdp.foot_target_active) and the foot reward's velocity fade is above
    zero there.  Every other row gets the parent formula over the legs.
    """

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        std_standing,
        std_walking,
        std_running,
        asset_cfg: SceneEntityCfg,
        command_name: str,
        foot_command_name: str,
        foot_reward_name: str,
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
        lo, hi = env.reward_manager.get_term_cfg(foot_reward_name).params[
            "velocity_fade_range"
        ]
        command = env.command_manager.get_command(command_name)
        speed = torch.norm(command[:, :2], dim=-1) + torch.abs(command[:, 2])
        fade = 1.0 - torch.clamp((speed - lo) / (hi - lo), 0.0, 1.0)
        released = foot_target_active(env, foot_command_name) & (fade > 0.0)
        return torch.where(released, torch.ones_like(parent), parent)


def apply_leg_pose_release(cfg: ManagerBasedRlEnvCfg) -> None:
    """Swap the inherited pose term for the leg-only, foot-target-released one."""

    pose = cfg.rewards.get("pose")
    if pose is None or pose.func is not velocity_mdp.variable_posture:
        raise TypeError("Leg pose release requires the inherited variable_posture")
    if "foot_target" not in cfg.commands or "foot_target_tracking" not in cfg.rewards:
        raise KeyError("Leg pose release requires the foot target and its reward")
    pose.func = foot_target_leg_released_posture
    pose.params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=MICROBAN_TELEOP_LEG_JOINT_NAMES
    )
    for key in ("std_standing", "std_walking", "std_running"):
        pose.params[key] = {
            pattern: value
            for pattern, value in pose.params[key].items()
            if pattern in _LEG_STD_PATTERNS
        }
        if set(pose.params[key]) != set(_LEG_STD_PATTERNS):
            raise ValueError(f"Pose {key} lacks a leg joint std")
    pose.params["foot_command_name"] = "foot_target"
    pose.params["foot_reward_name"] = "foot_target_tracking"


def make_microban_teleop_v13_arm_overlay_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Build the PICO task with the leg pose release (see module doc)."""

    cfg = make_microban_teleop_v12_env_cfg(play=play)
    apply_leg_pose_release(cfg)
    return cfg


MicrobanTeleopV13ArmOverlayRlCfg = deepcopy(MicrobanTeleopV12RlCfg)
MicrobanTeleopV13ArmOverlayRlCfg.wandb_project = "mjlab_microban_teleop_v13_arm_overlay"
