# Adapted from Teleopit f926386 train_mimic/tasks/tracking/mdp/terminations.py
# (Apache-2.0, https://github.com/BotRunner64/Teleopit).  Changes: only the
# terms the task uses (height-only anchor and end points, gravity z) and the
# trunk's lag across the floor (training only, see env_cfg.py); the end
# points may sit off their body's origin (Microban's hands are points on the
# forearms); the end of a window is a time-out.

"""Termination terms: the robot left the reference."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from mjlab_microban.wholebody.motion import MotionCommand
from mjlab_microban.wholebody.rewards import body_points

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def bad_anchor_pos_z_only(env: ManagerBasedRlEnv, command_name: str, threshold: float) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def bad_anchor_pos_xy(env: ManagerBasedRlEnv, command_name: str, threshold: float) -> torch.Tensor:
    """The robot trunk is more than ``threshold`` across the floor from the reference trunk."""
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    return torch.linalg.norm(command.anchor_pos_w[:, :2] - command.robot_anchor_pos_w[:, :2], dim=-1) > threshold


def bad_anchor_ori(env: ManagerBasedRlEnv, command_name: str, threshold: float) -> torch.Tensor:
    """The z components of gravity in the reference and robot trunk frames differ by more than ``threshold``."""
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    gravity = env.scene[command.cfg.entity_name].data.gravity_vec_w
    motion = quat_apply_inverse(command.anchor_quat_w, gravity)
    robot = quat_apply_inverse(command.robot_anchor_quat_w, gravity)
    return (motion[:, 2] - robot[:, 2]).abs() > threshold


def bad_motion_body_point_z_only(
    env: ManagerBasedRlEnv,
    command_name: str,
    threshold: float,
    body_names: tuple[str, ...],
    body_offsets: tuple[tuple[float, float, float], ...],
) -> torch.Tensor:
    """A point fixed on a tracked body is more than ``threshold`` off the reference's height."""
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    ref_points, robot_points = body_points(command, body_names, body_offsets)
    return torch.any(torch.abs(ref_points[..., 2] - robot_points[..., 2]) > threshold, dim=-1)


def clip_end(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """The reference window ran out (a time-out: the episode is cut, not failed)."""
    return cast(MotionCommand, env.command_manager.get_term(command_name)).clip_ended.clone()
