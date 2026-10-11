# Adapted from Teleopit f926386 train_mimic/tasks/tracking/mdp/terminations.py
# (Apache-2.0, https://github.com/BotRunner64/Teleopit).  Changes: only the
# three terms the task uses (height-only anchor and end bodies, gravity z).

"""Termination terms: the robot left the reference."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from mjlab_microban.wholebody.motion import MotionCommand
from mjlab_microban.wholebody.rewards import body_indexes

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def bad_anchor_pos_z_only(env: ManagerBasedRlEnv, command_name: str, threshold: float) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def bad_anchor_ori(env: ManagerBasedRlEnv, command_name: str, threshold: float) -> torch.Tensor:
    """The z components of gravity in the reference and robot trunk frames differ by more than ``threshold``."""
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    gravity = env.scene[command.cfg.entity_name].data.gravity_vec_w
    motion = quat_apply_inverse(command.anchor_quat_w, gravity)
    robot = quat_apply_inverse(command.robot_anchor_quat_w, gravity)
    return (motion[:, 2] - robot[:, 2]).abs() > threshold


def bad_motion_body_pos_z_only(
    env: ManagerBasedRlEnv, command_name: str, threshold: float, body_names: tuple[str, ...] | None = None
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    idx = body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, idx, -1] - command.robot_body_pos_w[:, idx, -1])
    return torch.any(error > threshold, dim=-1)
