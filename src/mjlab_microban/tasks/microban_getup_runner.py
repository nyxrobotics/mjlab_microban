"""Checkpoint provenance for the get-up policy's physical action contract."""

from __future__ import annotations

import math

import torch

from mjlab.envs.mdp.observations import builtin_sensor
from mjlab.rl.runner import MjlabOnPolicyRunner

from mjlab_microban.tasks.microban_getup_action import (
    GETUP_TARGET_SLEW_RAD_S,
    SlewLimitedGetupJointPositionAction,
    effective_getup_action_after_target_slew,
)
from mjlab_microban.tasks.microban_getup_actuator import (
    GETUP_BODY_KP_FW,
    GetupBamActuatorCfg,
)
from mjlab_microban.tasks.microban_getup_env_cfg import GETUP_EPISODE_LENGTH_S
from mjlab_microban.tasks.microban_teleop_mdp import normalized_target_clip_excess_l1_sum


GETUP_CONTRACT_VERSION = "v2"
GETUP_ANGULAR_VELOCITY_FRAME = "imu_sensor_xyz"


class MicrobanGetupOnPolicyRunner(MjlabOnPolicyRunner):
    """Stamp compatible checkpoints and reject old get-up resumes."""

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        unwrapped = env.unwrapped
        action = unwrapped.action_manager.get_term("joint_pos")
        if not isinstance(action, SlewLimitedGetupJointPositionAction):
            raise ValueError("Get-up v2 requires the slew-limited joint-position action")
        if env.clip_actions is not None or env.num_actions != 18:
            raise ValueError("Get-up v2 requires 18 actions without wrapper clipping")
        if not math.isclose(unwrapped.step_dt, 0.02, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError("Get-up v2 requires a 20 ms policy step")
        if unwrapped.cfg.episode_length_s != GETUP_EPISODE_LENGTH_S:
            raise ValueError("Get-up v2 requires a 20-second training episode")
        if action.cfg.clip != {r".*": (-1.57, 1.57)}:
            raise ValueError("Get-up v2 requires absolute target clipping at ±1.57 rad")
        if action.cfg.max_target_speed_rad_s != GETUP_TARGET_SLEW_RAD_S:
            raise ValueError("Get-up v2 requires target slew at 0.5 rad/s")
        if action.cfg.scale != 1.0 or action.cfg.offset != 0.0 or not action.cfg.use_default_offset:
            raise ValueError("Get-up v2 requires unit-scale default-relative actions")
        gyro_term = unwrapped.cfg.observations["actor"].terms["base_ang_vel"]
        if gyro_term.func is not builtin_sensor or gyro_term.params != {"sensor_name": "robot/imu_ang_vel"}:
            raise ValueError("Get-up v2 requires raw IMU-sensor-frame angular velocity")
        for group in ("actor", "critic"):
            term = unwrapped.cfg.observations[group].terms["actions"]
            if term.func is not effective_getup_action_after_target_slew:
                raise ValueError(f"Get-up v2 requires post-slew {group} action feedback")
        robot_cfg = unwrapped.cfg.scene.entities["robot"]
        actuator_cfgs = robot_cfg.articulation.actuators if robot_cfg.articulation else ()
        if len(actuator_cfgs) != 1 or not isinstance(actuator_cfgs[0], GetupBamActuatorCfg):
            raise ValueError("Get-up v2 requires its body/neck XC330 actuator model")
        if actuator_cfgs[0].kp_fw != GETUP_BODY_KP_FW or actuator_cfgs[0].max_current != 0.91:
            raise ValueError("Get-up v2 requires body P125 and XC330 0.91 A current limit")
        raw_clip_reward = unwrapped.cfg.rewards.get("raw_target_clip_excess")
        if (
            raw_clip_reward is None
            or raw_clip_reward.func is not normalized_target_clip_excess_l1_sum
            or raw_clip_reward.weight != -0.2
        ):
            raise ValueError("Get-up v2 requires the raw target clip-excess reward")
        super().__init__(env, train_cfg, log_dir, device)

    def save(self, path: str, infos=None) -> None:
        infos = {
            **(infos or {}),
            "microban_getup_contract": GETUP_CONTRACT_VERSION,
            "microban_getup_target_slew_rad_s": GETUP_TARGET_SLEW_RAD_S,
            "microban_getup_angular_velocity_frame": GETUP_ANGULAR_VELOCITY_FRAME,
        }
        super().save(path, infos)

    def load(self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None) -> dict:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        infos = checkpoint.get("infos")
        if not isinstance(infos, dict):
            raise ValueError("Checkpoint lacks get-up v2 training metadata; start a fresh run")
        if (
            infos.get("microban_getup_contract") != GETUP_CONTRACT_VERSION
            or infos.get("microban_getup_target_slew_rad_s") != GETUP_TARGET_SLEW_RAD_S
            or infos.get("microban_getup_angular_velocity_frame") != GETUP_ANGULAR_VELOCITY_FRAME
        ):
            raise ValueError("Checkpoint lacks the get-up v2 action and IMU-frame contract; start a fresh run")
        return super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)
