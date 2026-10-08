"""PICO's velocity reward terms before the twist-ratio term (8595c2e).

The user (2026-10-08): "とりあえず歩行の報酬は今までうまく言ってた速度追従方法に
戻し、止まってたら足踏みしなくなってる状態で手足のトラッキングを頑張る".  PICO
tracks velocity with the terms the earlier forward-lean PICO was trained with:
planar_velocity_tracking_exp (weight 5, std 0.5 m/s), commanded planar
progress (2), the inherited track_angular_velocity (2, std 1.25 rad/s) and the
two L1 errors (-16 linear, -1 yaw), all read in the HOME-levelled trunk frame.
These are the functions 8595c2e removed from microban_teleop_mdp.py, restored
here as they were (microban_teleop_mdp.py is a get-up training input).
"""

from __future__ import annotations

import math

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from mjlab_microban.tasks.mdp import _home_levelled_quat

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def home_levelled_root_lin_vel_b(
    env: ManagerBasedRlEnv, trunk_pitch: float, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Root linear velocity in the HOME-levelled trunk frame."""

    asset: Entity = env.scene[asset_cfg.name]
    if trunk_pitch == 0.0:
        return asset.data.root_link_lin_vel_b
    frame = _home_levelled_quat(asset.data.root_link_quat_w, trunk_pitch)
    return quat_apply_inverse(frame, asset.data.root_link_lin_vel_w)


def home_levelled_root_ang_vel_b(
    env: ManagerBasedRlEnv, trunk_pitch: float, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Root angular velocity in the HOME-levelled trunk frame."""

    asset: Entity = env.scene[asset_cfg.name]
    if trunk_pitch == 0.0:
        return asset.data.root_link_ang_vel_b
    frame = _home_levelled_quat(asset.data.root_link_quat_w, trunk_pitch)
    return quat_apply_inverse(frame, asset.data.root_link_ang_vel_w)


def linear_velocity_tracking_error_l1(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    trunk_pitch: float = 0.0,
) -> torch.Tensor:
    """Return body-frame planar velocity error with a non-vanishing gradient."""

    command = env.command_manager.get_command(command_name)
    actual = home_levelled_root_lin_vel_b(env, trunk_pitch, asset_cfg)
    return torch.abs(command[:, :2] - actual[:, :2]).sum(dim=-1)


def yaw_velocity_tracking_error_l1(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    trunk_pitch: float = 0.0,
) -> torch.Tensor:
    """Return absolute body-frame yaw-rate error (HOME-levelled)."""

    command = env.command_manager.get_command(command_name)
    actual = home_levelled_root_ang_vel_b(env, trunk_pitch, asset_cfg)
    return torch.abs(command[:, 2] - actual[:, 2])


def planar_velocity_tracking_exp(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str = "twist",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    trunk_pitch: float = 0.0,
) -> torch.Tensor:
    """Track body-frame XY velocity without penalizing gait vertical motion.

    Mjlab's generic linear-velocity reward adds squared base vertical velocity
    inside the same exponential, which suppresses the vertical motion needed
    to unload and lift a foot.  Vertical stability remains covered by the
    upright, pose, body-angular-velocity, contact, and fall terms.
    """

    if not math.isfinite(std) or std <= 0.0:
        raise ValueError("planar velocity tracking std must be finite and positive")
    command = env.command_manager.get_command(command_name)
    actual = home_levelled_root_lin_vel_b(env, trunk_pitch, asset_cfg)
    error = torch.square(command[:, :2] - actual[:, :2]).sum(dim=-1)
    return torch.exp(-error / std**2)


def commanded_planar_velocity_progress(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    command_threshold: float = 0.01,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    trunk_pitch: float = 0.0,
) -> torch.Tensor:
    """Reward commanded body-frame XY progress, bounded to ``[0, 1]``."""

    if not math.isfinite(command_threshold) or command_threshold <= 0.0:
        raise ValueError("command_threshold must be finite and positive")
    command = env.command_manager.get_command(command_name)
    if command is None or command.ndim != 2 or command.shape != (env.num_envs, 3):
        raise ValueError("planar velocity progress requires an (num_envs, 3) command")
    if not bool(torch.isfinite(command).all()):
        raise ValueError("planar velocity progress command must be finite")
    actual = home_levelled_root_lin_vel_b(env, trunk_pitch, asset_cfg)
    if actual.ndim != 2 or actual.shape != (env.num_envs, 3):
        raise ValueError("planar velocity progress requires an (num_envs, 3) body velocity")
    if not bool(torch.isfinite(actual).all()):
        raise ValueError("planar velocity progress body velocity must be finite")
    command_xy = command[:, :2]
    actual_xy = actual[:, :2]
    command_norm_sq = torch.square(command_xy).sum(dim=-1)
    aligned_progress = (actual_xy * command_xy).sum(dim=-1)
    active_command = command_norm_sq > command_threshold**2
    aligned_fraction = aligned_progress / torch.clamp(command_norm_sq, min=command_threshold**2)
    reward = torch.clamp(aligned_fraction, min=0.0, max=1.0) * active_command
    env.extras["log"]["Metrics/commanded_planar_velocity_progress"] = reward.mean()
    return reward
