"""Checkpoint provenance for the get-up policy's physical action contract."""

from __future__ import annotations

import math

import torch

from mjlab.envs.mdp.observations import builtin_sensor
from mjlab.rl.runner import MjlabOnPolicyRunner

from mjlab_microban.robot.microban_constants import HOME_FRAME
from mjlab_microban.tasks.microban_getup_action import (
    GetupJointPositionAction,
    raw_getup_action,
)
from mjlab_microban.tasks.microban_getup_actuator import (
    GETUP_BODY_KP_FW,
    GetupBamActuatorCfg,
)
from mjlab_microban.tasks.microban_getup_env_cfg import (
    GETUP_ACTION_CLIP,
    GETUP_EPISODE_LENGTH_S,
)


# v4: flat +-1.57 rad absolute target clip on every body joint, the policy's
# RAW previous output as previous-action feedback, and no clip-excess
# penalty -- the contract every standing get-up policy was trained under
# (see microban_getup_env_cfg.py's module docstring). v3
# (per-joint soft-limit clip, post-clip feedback) never produced a stand.
# Old v3 checkpoints/ONNX are incompatible: their previous-action input
# meant the applied target, not the raw output.
GETUP_CONTRACT_VERSION = "v4"
GETUP_ANGULAR_VELOCITY_FRAME = "imu_sensor_xyz"


def getup_home_pose() -> dict[str, object]:
    """Return the complete training HOME in a checkpoint-safe form."""

    joints = {name: float(value) for name, value in sorted(HOME_FRAME.joint_pos.items())}
    if len(joints) != 21:
        raise ValueError("Get-up HOME must define all 21 Microban joints")
    return {
        "root_pos_m": [float(value) for value in HOME_FRAME.pos],
        "root_quat_wxyz": [float(value) for value in HOME_FRAME.rot],
        "joint_pos_rad": joints,
    }


def require_current_getup_home_pose(infos: dict) -> None:
    if infos.get("microban_getup_home_pose") != getup_home_pose():
        raise ValueError(
            "Checkpoint has a different or unknown get-up HOME pose; "
            "train from scratch with the current task"
        )


class MicrobanGetupOnPolicyRunner(MjlabOnPolicyRunner):
    """Stamp compatible checkpoints and reject old get-up resumes."""

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        unwrapped = env.unwrapped
        action = unwrapped.action_manager.get_term("joint_pos")
        if not isinstance(action, GetupJointPositionAction):
            raise ValueError("Get-up v4 requires the get-up joint-position action")
        if env.clip_actions is not None or env.num_actions != 18:
            raise ValueError("Get-up v4 requires 18 actions without wrapper clipping")
        if not math.isclose(unwrapped.step_dt, 0.02, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError("Get-up v4 requires a 20 ms policy step")
        if unwrapped.cfg.episode_length_s != GETUP_EPISODE_LENGTH_S:
            raise ValueError("Get-up v4 requires a 20-second training episode")
        if action.cfg.clip != dict(GETUP_ACTION_CLIP):
            raise ValueError("Get-up v4 requires the flat +-1.57 rad absolute target clip")
        if action.cfg.scale != 1.0 or action.cfg.offset != 0.0 or not action.cfg.use_default_offset:
            raise ValueError("Get-up v4 requires unit-scale default-relative actions")
        gyro_term = unwrapped.cfg.observations["actor"].terms["base_ang_vel"]
        if gyro_term.func is not builtin_sensor or gyro_term.params != {"sensor_name": "robot/imu_ang_vel"}:
            raise ValueError("Get-up v4 requires raw IMU-sensor-frame angular velocity")
        for group in ("actor", "critic"):
            term = unwrapped.cfg.observations[group].terms["actions"]
            if term.func is not raw_getup_action:
                raise ValueError(f"Get-up v4 requires raw previous-action {group} feedback")
            if term.delay_max_lag != 0:
                raise ValueError(f"Get-up v4 requires undelayed {group} action feedback")
        robot_cfg = unwrapped.cfg.scene.entities["robot"]
        actuator_cfgs = robot_cfg.articulation.actuators if robot_cfg.articulation else ()
        if len(actuator_cfgs) != 1 or not isinstance(actuator_cfgs[0], GetupBamActuatorCfg):
            raise ValueError("Get-up v4 requires its body/neck XC330 actuator model")
        if actuator_cfgs[0].kp_fw != GETUP_BODY_KP_FW or actuator_cfgs[0].max_current != 0.91:
            raise ValueError("Get-up v4 requires body P125 and XC330 0.91 A current limit")
        if "raw_target_clip_excess" in unwrapped.cfg.rewards:
            raise ValueError(
                "Get-up v4 must not penalize raw output beyond the clip: standing "
                "needs targets far past the joint angle for torque at P125"
            )
        super().__init__(env, train_cfg, log_dir, device)

    def save(self, path: str, infos=None) -> None:
        infos = {
            **(infos or {}),
            "microban_getup_contract": GETUP_CONTRACT_VERSION,
            "microban_getup_angular_velocity_frame": GETUP_ANGULAR_VELOCITY_FRAME,
            "microban_getup_home_pose": getup_home_pose(),
        }
        super().save(path, infos)

    def load(self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None) -> dict:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        infos = checkpoint.get("infos")
        if not isinstance(infos, dict):
            raise ValueError("Checkpoint lacks get-up v4 training metadata; start a fresh run")
        if (
            infos.get("microban_getup_contract") != GETUP_CONTRACT_VERSION
            or infos.get("microban_getup_angular_velocity_frame") != GETUP_ANGULAR_VELOCITY_FRAME
        ):
            raise ValueError("Checkpoint lacks the get-up v4 action and IMU-frame contract; start a fresh run")
        require_current_getup_home_pose(infos)
        return super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)
