"""The walking task's velocity reward: mjlab's two exp tracking terms at HOME.

The user (2026-10-08): "もう歩行の速度の報酬はmicrobanの初期状態でいいや" --
walking tracks velocity with the reward it had before the twist-ratio term
(8595c2e), the reward the forward-lean walker lean_walk_cont2 model_29000
was trained with: mjlab's track_linear_velocity (std sqrt(0.1)) and
track_angular_velocity (std sqrt(0.5)), weight 2 each, read in the
HOME-levelled trunk frame.  These are the functions 8595c2e removed from
mdp.py, restored here as they were (mdp.py is an input of every step; this
module is a walking input only).
"""

from __future__ import annotations

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from mjlab_microban.tasks.mdp import _home_levelled_quat

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_linear_velocity_home_frame(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    trunk_pitch: float = 0.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """mjlab's track_linear_velocity, in the HOME-levelled trunk frame.

    mjlab reads root_link_lin_vel_b. With a HOME trunk leaning
    ``trunk_pitch`` forward that frame tips the walking velocity: at 10 deg
    the forward speed reads 1.5 % low and 17 % of it shows up as vertical
    velocity, which the z term penalizes (0.7 m/s: -14 % reward). Rotating
    the lean back out keeps the reward's meaning of the upright-HOME tasks.
    trunk_pitch = 0 is exactly mjlab's term.
    """

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    frame = _home_levelled_quat(asset.data.root_link_quat_w, trunk_pitch)
    actual = quat_apply_inverse(frame, asset.data.root_link_lin_vel_w)
    xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
    z_error = torch.square(actual[:, 2])
    return torch.exp(-(xy_error + z_error) / std**2)


def track_angular_velocity_home_frame(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    trunk_pitch: float = 0.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """mjlab's track_angular_velocity, in the HOME-levelled trunk frame.

    In the leaning trunk frame a pure yaw rate w reads w*cos(lean) about z
    and w*sin(lean) about x, which the xy term penalizes (10 deg, 1.5 rad/s:
    -13 % reward). trunk_pitch = 0 is exactly mjlab's term.
    """

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    frame = _home_levelled_quat(asset.data.root_link_quat_w, trunk_pitch)
    actual = quat_apply_inverse(frame, asset.data.root_link_ang_vel_w)
    z_error = torch.square(command[:, 2] - actual[:, 2])
    xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
    return torch.exp(-(z_error + xy_error) / std**2)
