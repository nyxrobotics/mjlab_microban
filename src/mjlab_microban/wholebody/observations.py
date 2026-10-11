# Adapted from Teleopit f926386 train_mimic/tasks/tracking/mdp/observations.py
# (Apache-2.0, https://github.com/BotRunner64/Teleopit).  Changes: the
# heading-frame variants and the future-window term are dropped.

"""Observation terms of the reference and of the tracked bodies."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply, quat_inv, subtract_frame_transforms

from mjlab_microban.wholebody.motion import MotionCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _command(env: ManagerBasedRlEnv, command_name: str) -> MotionCommand:
    return cast(MotionCommand, env.command_manager.get_term(command_name))


def ref_joint_pos(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Reference angles of the policy's joints (absolute, rad)."""
    return _command(env, command_name).joint_pos


def ref_joint_vel(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return _command(env, command_name).joint_vel


def ref_anchor_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = _command(env, command_name)
    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w, command.robot_anchor_quat_w, command.anchor_pos_w, command.anchor_quat_w
    )
    return pos.view(env.num_envs, -1)


def ref_anchor_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Reference trunk orientation in the robot trunk frame: first two rotation-matrix columns."""
    command = _command(env, command_name)
    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w, command.robot_anchor_quat_w, command.anchor_pos_w, command.anchor_quat_w
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_tracking_body_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = _command(env, command_name)
    n = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, n, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, n, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    return pos_b.view(env.num_envs, -1)


def robot_tracking_body_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = _command(env, command_name)
    n = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, n, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, n, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def ref_anchor_lin_vel_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Reference trunk linear velocity in the robot trunk frame. (N, 3)"""
    command = _command(env, command_name)
    return quat_apply(quat_inv(command.robot_anchor_quat_w), command.anchor_lin_vel_w)


def ref_anchor_ang_vel_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Reference trunk angular velocity in the robot trunk frame. (N, 3)"""
    command = _command(env, command_name)
    return quat_apply(quat_inv(command.robot_anchor_quat_w), command.anchor_ang_vel_w)


def ref_projected_gravity_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Gravity direction in the reference trunk frame (the reference's tilt). (N, 3)"""
    command = _command(env, command_name)
    asset = env.scene[command.cfg.entity_name]
    return quat_apply(quat_inv(command.anchor_quat_w), asset.data.gravity_vec_w)


def ref_anchor_height(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Reference trunk height above the floor. (N, 1)"""
    command = _command(env, command_name)
    return command.anchor_pos_w[:, 2:3] - env.scene.env_origins[:, 2:3]
