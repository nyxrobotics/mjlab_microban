# Adapted from Teleopit f926386 train_mimic/tasks/tracking/mdp/rewards.py
# (Apache-2.0, https://github.com/BotRunner64/Teleopit).  Unchanged in form;
# the stds are set per robot in env_cfg.py.

"""Tracking rewards: exp(-error / std^2) of the reference and robot states."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_error_magnitude, yaw_quat

from mjlab_microban.wholebody.motion import MotionCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _command(env: ManagerBasedRlEnv, command_name: str) -> MotionCommand:
    return cast(MotionCommand, env.command_manager.get_term(command_name))


def body_indexes(command: MotionCommand, body_names: tuple[str, ...] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if body_names is None or name in body_names]


def motion_global_anchor_position_error_exp(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
    command = _command(env, command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
    command = _command(env, command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_global_anchor_heading_cos(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """(1 + cos e) / 2 of the heading error e: not Teleopit's.  The exp terms give nothing past about 1 rad, so a
    robot left behind in heading had no reason to turn back."""
    command = _command(env, command_name)
    error = quat_error_magnitude(yaw_quat(command.anchor_quat_w), yaw_quat(command.robot_anchor_quat_w))
    return 0.5 * (1.0 + torch.cos(error))


def motion_global_anchor_linear_velocity_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
    command = _command(env, command_name)
    error = torch.sum(torch.square(command.anchor_lin_vel_w - command.robot_anchor_lin_vel_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_angular_velocity_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
    command = _command(env, command_name)
    error = torch.sum(torch.square(command.anchor_ang_vel_w - command.robot_anchor_ang_vel_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float, body_names: tuple[str, ...] | None = None
) -> torch.Tensor:
    command = _command(env, command_name)
    idx = body_indexes(command, body_names)
    error = torch.sum(torch.square(command.body_pos_relative_w[:, idx] - command.robot_body_pos_w[:, idx]), dim=-1)
    return torch.exp(-error.mean(-1) / std**2)


def body_points(
    command: MotionCommand,
    body_names: tuple[str, ...],
    body_offsets: tuple[tuple[float, float, float], ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Points fixed on tracked bodies: (reference moved under the robot, robot), each (N, k, 3)."""
    index_by_name = {name: i for i, name in enumerate(command.cfg.body_names)}
    missing = [name for name in body_names if name not in index_by_name]
    if missing:
        raise ValueError(f"body_names must be tracked bodies: missing {missing}")
    if len(body_names) != len(body_offsets):
        raise ValueError("body_offsets needs one offset per body")
    idx = [index_by_name[name] for name in body_names]
    offsets = torch.tensor(body_offsets, dtype=command.body_pos_relative_w.dtype, device=command.device).expand(
        command.num_envs, -1, -1
    )
    ref_points = command.body_pos_relative_w[:, idx] + quat_apply(command.body_quat_relative_w[:, idx], offsets)
    robot_points = command.robot_body_pos_w[:, idx] + quat_apply(command.robot_body_quat_w[:, idx], offsets)
    return ref_points, robot_points


def motion_relative_body_point_position_error_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    body_names: tuple[str, ...],
    body_offsets: tuple[tuple[float, float, float], ...],
) -> torch.Tensor:
    """Points fixed on bodies (here the hands on the forearms), relative like the bodies."""
    ref_points, robot_points = body_points(_command(env, command_name), body_names, body_offsets)
    error = torch.sum(torch.square(ref_points - robot_points), dim=-1)
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float, body_names: tuple[str, ...] | None = None
) -> torch.Tensor:
    command = _command(env, command_name)
    idx = body_indexes(command, body_names)
    error = quat_error_magnitude(command.body_quat_relative_w[:, idx], command.robot_body_quat_w[:, idx]) ** 2
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float, body_names: tuple[str, ...] | None = None
) -> torch.Tensor:
    command = _command(env, command_name)
    idx = body_indexes(command, body_names)
    error = torch.sum(torch.square(command.body_lin_vel_w[:, idx] - command.robot_body_lin_vel_w[:, idx]), dim=-1)
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float, body_names: tuple[str, ...] | None = None
) -> torch.Tensor:
    command = _command(env, command_name)
    idx = body_indexes(command, body_names)
    error = torch.sum(torch.square(command.body_ang_vel_w[:, idx] - command.robot_body_ang_vel_w[:, idx]), dim=-1)
    return torch.exp(-error.mean(-1) / std**2)


def motion_joint_position_error_exp(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
    command = _command(env, command_name)
    error = torch.square(command.joint_pos - command.robot_joint_pos)
    return torch.exp(-error.mean(-1) / std**2)


def motion_joint_velocity_error_exp(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
    command = _command(env, command_name)
    error = torch.square(command.joint_vel - command.robot_joint_vel)
    return torch.exp(-error.mean(-1) / std**2)


def survival(env: ManagerBasedRlEnv) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def self_collision_cost(env: ManagerBasedRlEnv, sensor_name: str, force_threshold: float) -> torch.Tensor:
    """Number of sensor slots with a self contact above ``force_threshold`` in the step's substeps."""
    data = env.scene[sensor_name].data
    if data.force_history is None:
        raise ValueError(f"{sensor_name} needs force history (history_length > 0)")
    force = torch.norm(data.force_history, dim=-1)  # (B, N, H)
    return (force > force_threshold).any(dim=2).sum(dim=-1).float()
