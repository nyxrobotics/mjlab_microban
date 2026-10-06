"""Ratio-keeping twist velocity reward (one term for walking and PICO teleop).

User specification (2026-10-06):

* beyond the robot's speed, keep the ratio of the commanded twist components
  and walk at the largest speed it can manage ("速度の成分の比は合わせつつ、移動
  できる最大の速度で歩く"): execute ``s * c`` with one common scale
  ``s <= 1`` (``s = 1`` whenever the command is feasible);
* reward speed only through the component of the motion along the command
  direction ("司令方向成分だけ取り出して報酬"), so moving fast in another direction
  earns nothing;
* moving against the command is bad too ("反対向きもだめ").

The twist is (v_x, v_y, w_z) in the HOME-levelled trunk frame.  Each axis is
divided by the command envelope's maximum (default 0.7 m/s, 0.3 m/s,
1.5 rad/s, the robot's moving ``scale_velocity`` limits) so metres per second
and radians per second are comparable.  With ``c^ = c / scale``,
``v^ = v / scale``, ``n = |c^|``, ``u = c^ / n`` and ``p = v^ . u``:

``a = clamp(p, 0, n)``
    progress along the command, capped at the command;
``speed = a / max(n, min_command_norm)``
    the along-command speed fraction, in [0, 1];
``error = |v^ - a u| + max(0, -p)``
    everything that is not capped forward progress along the command
    (the perpendicular part, the overshoot beyond the command, any backward
    part) plus the backward part once more.  At the same speed k,
    perpendicular motion costs k and opposite motion 2k; the error grows
    monotonically with the angle to the command.

The reward is ``speed - direction_penalty * error``.  For a zero command
(standing) ``speed = 0`` and ``error = |v^|``.  Its maximum over the twists a
robot can reach lies on the commanded ray: ``s * c`` with the largest feasible
``s`` (exact tracking when the command is feasible).  ``min_command_norm``
keeps very small commands from carrying the full speed reward (the speed
fraction of a command smaller than it is at most ``n / min_command_norm``).

``twist_ratio_velocity_reward`` is meant to replace the separate velocity
tracking terms (exp kernels, L1 errors and the xy projection progress) of the
walking and PICO base reward configurations.  The module depends only on
torch and mjlab (no robot constants, no other task module): the caller passes
the HOME trunk pitch and, if its command envelope differs, the axis scale.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_mul

# Max |.| of the command envelope per axis (v_x m/s, v_y m/s, w_z rad/s).
TWIST_RATIO_AXIS_SCALE = (0.7, 0.3, 1.5)
TWIST_RATIO_MIN_COMMAND_NORM = 0.2
TWIST_RATIO_DIRECTION_PENALTY = 1.0

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class TwistRatio(NamedTuple):
    """Per-env decomposition in envelope-normalized units (see module doc)."""

    command_norm: torch.Tensor  # (N,) n = |c^|
    along: torch.Tensor  # (N,) p = v^ . u (0 for a zero command)
    speed: torch.Tensor  # (N,) a / max(n, min_command_norm), in [0, 1]
    error: torch.Tensor  # (N,) |v^ - a u| + max(0, -p)


def twist_ratio(
    command: torch.Tensor,
    twist: torch.Tensor,
    axis_scale: Sequence[float] = TWIST_RATIO_AXIS_SCALE,
    min_command_norm: float = TWIST_RATIO_MIN_COMMAND_NORM,
) -> TwistRatio:
    """Decompose (N, 3) twists against (N, 3) commands (v_x, v_y, w_z)."""

    if command.ndim != 2 or command.shape[-1] != 3 or twist.shape != command.shape:
        raise ValueError("twist_ratio needs (N, 3) command and twist tensors")
    if not min_command_norm > 0.0:
        raise ValueError("min_command_norm must be positive")
    scale = torch.as_tensor(
        tuple(float(value) for value in axis_scale),
        dtype=command.dtype,
        device=command.device,
    )
    if scale.shape != (3,) or not bool((scale > 0.0).all()):
        raise ValueError("axis_scale must be three positive values")
    c = command / scale
    v = twist.to(command.dtype) / scale
    norm = torch.linalg.vector_norm(c, dim=-1)
    moving = norm > 0.0
    unit = c / torch.where(moving, norm, torch.ones_like(norm)).unsqueeze(-1)
    along = (v * unit).sum(dim=-1)
    progress = torch.minimum(torch.clamp(along, min=0.0), norm)
    speed = progress / torch.clamp(norm, min=min_command_norm)
    error = torch.linalg.vector_norm(
        v - progress.unsqueeze(-1) * unit, dim=-1
    ) + torch.clamp(-along, min=0.0)
    return TwistRatio(norm, along, speed, error)


def twist_ratio_reward(
    command: torch.Tensor,
    twist: torch.Tensor,
    axis_scale: Sequence[float] = TWIST_RATIO_AXIS_SCALE,
    min_command_norm: float = TWIST_RATIO_MIN_COMMAND_NORM,
    direction_penalty: float = TWIST_RATIO_DIRECTION_PENALTY,
) -> torch.Tensor:
    """``speed - direction_penalty * error`` per env (pure tensor form)."""

    if not direction_penalty > 0.0:
        raise ValueError("direction_penalty must be positive")
    parts = twist_ratio(command, twist, axis_scale, min_command_norm)
    return parts.speed - direction_penalty * parts.error


def home_levelled_twist(
    env: ManagerBasedRlEnv,
    trunk_pitch: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """(N, 3) twist (v_x, v_y, w_z) in the HOME-levelled trunk frame.

    The root frame with the HOME trunk's forward lean ``trunk_pitch`` rotated
    back out: at HOME it is level and faces the robot's heading.
    ``trunk_pitch = 0`` is the root body frame.
    """

    data = env.scene[asset_cfg.name].data
    if trunk_pitch == 0.0:
        linear = data.root_link_lin_vel_b
        angular = data.root_link_ang_vel_b
    else:
        quat_w = data.root_link_quat_w
        half = -0.5 * trunk_pitch
        unpitch = torch.tensor(
            (math.cos(half), 0.0, math.sin(half), 0.0),
            device=quat_w.device,
            dtype=quat_w.dtype,
        ).expand_as(quat_w)
        frame = quat_mul(quat_w, unpitch)
        linear = quat_apply_inverse(frame, data.root_link_lin_vel_w)
        angular = quat_apply_inverse(frame, data.root_link_ang_vel_w)
    return torch.stack((linear[:, 0], linear[:, 1], angular[:, 2]), dim=-1)


def twist_ratio_velocity_reward(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    trunk_pitch: float = 0.0,
    axis_scale: Sequence[float] = TWIST_RATIO_AXIS_SCALE,
    min_command_norm: float = TWIST_RATIO_MIN_COMMAND_NORM,
    direction_penalty: float = TWIST_RATIO_DIRECTION_PENALTY,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward term: the ratio-keeping velocity reward of the module doc."""

    command = env.command_manager.get_command(command_name)[:, :3]
    twist = home_levelled_twist(env, trunk_pitch, asset_cfg)
    parts = twist_ratio(command, twist, axis_scale, min_command_norm)
    extras = getattr(env, "extras", None)
    if isinstance(extras, dict):
        log = extras.setdefault("log", {})
        log["Metrics/twist_ratio_speed"] = parts.speed.mean()
        log["Metrics/twist_ratio_error"] = parts.error.mean()
    return parts.speed - direction_penalty * parts.error
