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

The commanded twist is (v_x, v_y, w_z) in the HOME-levelled trunk frame.  Each
axis is divided by the command envelope's maximum (default 0.7 m/s, 0.3 m/s,
1.5 rad/s, the robot's moving ``scale_velocity`` limits) so metres per second
and radians per second are comparable.  With ``c^ = c / scale``,
``v^ = v / scale``, ``n = |c^|``, ``u = c^ / n`` and ``p = v^ . u``:

``a = clamp(p, 0, n)``
    progress along the command, capped at the command;
``speed = a / n`` when ``n >= min_command_norm``
    the along-command speed fraction, in [0, 1];
``speed = clamp(1 - |v^ - c^| / min_command_norm, 0, 1)`` for a smaller command
    (standing included): a small command is tracked for precision, on the
    scale of ``min_command_norm``.  Both branches agree on the commanded ray
    at ``n = min_command_norm``, and every command, standing included, has
    the same best value (speed 1 at exact tracking);
``error = sqrt(|v^ - a u|^2 + |w^|^2) + max(0, -p)``
    the norm of everything that is not capped forward progress along the
    command -- the perpendicular part, the overshoot beyond the command, any
    backward part, and the uncommanded motion ``w^`` (vertical velocity, roll
    and pitch rates, each divided by its scale, default 0.7 m/s, 1.5 rad/s,
    1.5 rad/s) -- plus the backward part once more.  At the same speed k,
    perpendicular motion costs k and opposite motion 2k; the error grows
    monotonically with the angle to the command.

The reward is ``(1 + speed) / 2 * exp(-direction_penalty * error)``, in
[0, 1]: 1 at exact tracking of any command (standing still on a standing
command included), 1/2 for standing still on a moving command, less for
anything off the command, and never negative.  For a feasible command its
maximum is exact tracking.  For an infeasible one it is near the commanded
ray but not exactly on it: when only some axes are limited (typically
forward/backward), over-producing the easier axes buys a little speed for a
little error.  Over the reach box measured for the forward-lean walker
(forward 0.2, backward 0.11, lateral 0.18 m/s, yaw 1.4 rad/s) and a grid of
173 infeasible commands (forward -0.5..0.7, lateral +-0.3, yaw +-1.5), 42
have their best twist more than 1 deg off the ray, at most 24.2 deg, where
backward is the limit and yaw or lateral is commanded too (c = (-0.3, 0,
0.75): best (-0.11, 0, 0.80), reward 0.705 against 0.683 on the ray); the
diagonal commands of the walk check W1 have it about 15 deg off.  The user's
own form ``1 + speed - error`` has the same property (70 commands, at most
28.5 deg); this form only weakens it, and W1 (angle to the command ray)
judges the walker that results.  For a zero
command ``error = sqrt(|v^|^2 + |w^|^2)``.

Why this form (walker trained from scratch at the forward-lean HOME,
2026-10-06, held-out probes on seeds 101-105):

* ``speed - error`` (no constant part): no positive reward for staying upright
  where the exp tracking kernels it replaces gave about half of the early
  positive reward; episodes stayed near 45 steps at update 200 (old reward
  190).
* ``1 + speed - error`` with ``speed = a / max(n, min_command_norm)``: a
  standing command could earn at most 1 against 2 for a moving one, and
  falling ends an episode without a penalty and draws a new command; at update
  4000 the walker fell within one second in every standing rollout.  The
  small-command branch of ``speed`` gives every command the same best value.
* ``1 + speed - error`` with that ``speed``: the reward goes negative after a
  push, so falling can pay; 36-43 of 90 pushed diagonal rollouts fell from
  update 5000 to 8000 (old reward 2 of 90 at 5000), and single-axis commands
  were overshot about twice.
* This bounded form: 4 of 90 at update 3000 with single-axis commands
  tracked.

The uncommanded motion keeps a robot that topples in the commanded direction
from collecting speed reward on the way down (the tracking kernels this term
replaces penalised vertical velocity and roll/pitch rates the same way);
``uncommanded_scale=None`` leaves it out.

The reward term ``twist_ratio_velocity`` evaluates it on time-filtered
command and motion (see its doc; on the instantaneous motion the per-step gait
sway made walking cost more than standing still on small commands: the
unfiltered walker stopped answering single-axis commands at update 4000).  It
replaces the separate velocity tracking terms (exp kernels, L1 errors and the
xy projection progress) of the walking and PICO base reward configurations.  The module depends only on
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
# Scale of the uncommanded motion (v_z m/s, w_x rad/s, w_y rad/s).
TWIST_RATIO_UNCOMMANDED_SCALE = (0.7, 1.5, 1.5)
# Below this command norm the speed is the precision branch (see the module
# doc).  0.1, not 0.2: at 0.2 the 0.1 m/s forward and backward commands
# (n = 0.14) were in the precision branch, where standing still (0.64) beats
# walking at twice the command (0.53); the 2026-10-07 release walker, whose
# gait sped up with the update-3000 command widening, stood still on them
# from update 4000 (9x300 forward_0p1 and backward_0p1 at 0.000).  In the
# along-command branch any progress beats standing (2x: 0.83 against 0.5).
TWIST_RATIO_MIN_COMMAND_NORM = 0.1
TWIST_RATIO_DIRECTION_PENALTY = 1.0
TWIST_RATIO_FILTER_TIME_CONSTANT_S = 0.5

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class TwistRatio(NamedTuple):
    """Per-env decomposition in envelope-normalized units (see module doc)."""

    command_norm: torch.Tensor  # (N,) n = |c^|
    along: torch.Tensor  # (N,) p = v^ . u (0 for a zero command)
    speed: torch.Tensor  # (N,) along-command speed fraction in [0, 1] (see doc)
    error: torch.Tensor  # (N,) sqrt(|v^ - a u|^2 + |w^|^2) + max(0, -p)


def _scale_tensor(values: Sequence[float], like: torch.Tensor, width: int) -> torch.Tensor:
    scale = torch.as_tensor(
        tuple(float(value) for value in values), dtype=like.dtype, device=like.device
    )
    if scale.shape != (width,) or not bool((scale > 0.0).all()):
        raise ValueError(f"a twist scale must be {width} positive values")
    return scale


def twist_ratio(
    command: torch.Tensor,
    twist: torch.Tensor,
    axis_scale: Sequence[float] = TWIST_RATIO_AXIS_SCALE,
    min_command_norm: float = TWIST_RATIO_MIN_COMMAND_NORM,
    uncommanded: torch.Tensor | None = None,
    uncommanded_scale: Sequence[float] | None = TWIST_RATIO_UNCOMMANDED_SCALE,
) -> TwistRatio:
    """Decompose (N, 3) twists against (N, 3) commands (v_x, v_y, w_z).

    ``uncommanded`` is the (N, K) motion no command asks for (by default
    v_z, w_x, w_y), divided by ``uncommanded_scale``; ``None`` leaves it out.
    """

    if command.ndim != 2 or command.shape[-1] != 3 or twist.shape != command.shape:
        raise ValueError("twist_ratio needs (N, 3) command and twist tensors")
    if not min_command_norm > 0.0:
        raise ValueError("min_command_norm must be positive")
    scale = _scale_tensor(axis_scale, command, 3)
    c = command / scale
    v = twist.to(command.dtype) / scale
    norm = torch.linalg.vector_norm(c, dim=-1)
    moving = norm > 0.0
    unit = c / torch.where(moving, norm, torch.ones_like(norm)).unsqueeze(-1)
    along = (v * unit).sum(dim=-1)
    progress = torch.minimum(torch.clamp(along, min=0.0), norm)
    large = norm >= min_command_norm
    speed = torch.where(
        large,
        progress / torch.where(large, norm, torch.ones_like(norm)),
        torch.clamp(
            1.0 - torch.linalg.vector_norm(v - c, dim=-1) / min_command_norm, min=0.0
        ),
    )
    off = v - progress.unsqueeze(-1) * unit
    if uncommanded is not None:
        if uncommanded_scale is None:
            raise ValueError("uncommanded motion needs its scale")
        if uncommanded.ndim != 2 or uncommanded.shape[0] != command.shape[0]:
            raise ValueError("uncommanded motion must be (N, K)")
        w = uncommanded.to(command.dtype) / _scale_tensor(
            uncommanded_scale, command, uncommanded.shape[-1]
        )
        off = torch.cat((off, w), dim=-1)
    error = torch.linalg.vector_norm(off, dim=-1) + torch.clamp(-along, min=0.0)
    return TwistRatio(norm, along, speed, error)


def twist_ratio_reward(
    command: torch.Tensor,
    twist: torch.Tensor,
    axis_scale: Sequence[float] = TWIST_RATIO_AXIS_SCALE,
    min_command_norm: float = TWIST_RATIO_MIN_COMMAND_NORM,
    direction_penalty: float = TWIST_RATIO_DIRECTION_PENALTY,
    uncommanded: torch.Tensor | None = None,
    uncommanded_scale: Sequence[float] | None = TWIST_RATIO_UNCOMMANDED_SCALE,
) -> torch.Tensor:
    """``(1 + speed) / 2 * exp(-direction_penalty * error)`` per env."""

    if not direction_penalty > 0.0:
        raise ValueError("direction_penalty must be positive")
    parts = twist_ratio(
        command, twist, axis_scale, min_command_norm, uncommanded, uncommanded_scale
    )
    return _bounded(parts, direction_penalty)


def _bounded(parts: TwistRatio, direction_penalty: float) -> torch.Tensor:
    return 0.5 * (1.0 + parts.speed) * torch.exp(-direction_penalty * parts.error)


def home_levelled_velocities(
    env: ManagerBasedRlEnv,
    trunk_pitch: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(N, 3) linear and angular root velocity in the HOME-levelled trunk frame.

    The root frame with the HOME trunk's forward lean ``trunk_pitch`` rotated
    back out: at HOME it is level and faces the robot's heading.
    ``trunk_pitch = 0`` is the root body frame.
    """

    data = env.scene[asset_cfg.name].data
    if trunk_pitch == 0.0:
        return data.root_link_lin_vel_b, data.root_link_ang_vel_b
    quat_w = data.root_link_quat_w
    half = -0.5 * trunk_pitch
    unpitch = torch.tensor(
        (math.cos(half), 0.0, math.sin(half), 0.0),
        device=quat_w.device,
        dtype=quat_w.dtype,
    ).expand_as(quat_w)
    frame = quat_mul(quat_w, unpitch)
    return (
        quat_apply_inverse(frame, data.root_link_lin_vel_w),
        quat_apply_inverse(frame, data.root_link_ang_vel_w),
    )


def home_levelled_twist(
    env: ManagerBasedRlEnv,
    trunk_pitch: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """(N, 3) twist (v_x, v_y, w_z) in the HOME-levelled trunk frame."""

    linear, angular = home_levelled_velocities(env, trunk_pitch, asset_cfg)
    return torch.stack((linear[:, 0], linear[:, 1], angular[:, 2]), dim=-1)


class twist_ratio_velocity:
    """Reward term: the bounded twist-ratio reward on time-filtered motion.

    The command and the measured motion are both low-pass filtered with the
    same first-order filter (time constant ``filter_time_constant``, 0.5 s by
    default; 0 uses the instantaneous values), and the reward of the module doc
    is evaluated on the filtered values.  The reward is about the direction and
    speed the robot walks at, not about the swaying within a stride: the
    per-step lateral sway, vertical bob and roll/pitch rates of a normal gait
    average out, while a sustained drift, a turn of the walking direction or a
    fall do not.  Filtering the command the same way keeps a robot that follows
    a new command at once on target while both settle.  Both filters start at
    the current values on the first step of an episode.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        num_envs, device = env.num_envs, env.device
        self.step_dt = float(env.step_dt)
        self.command = torch.zeros(num_envs, 3, device=device)
        self.twist = torch.zeros(num_envs, 3, device=device)
        self.uncommanded = torch.zeros(num_envs, 3, device=device)
        self.fresh = torch.ones(num_envs, dtype=torch.bool, device=device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self.fresh[slice(None) if env_ids is None else env_ids] = True

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str = "twist",
        trunk_pitch: float = 0.0,
        axis_scale: Sequence[float] = TWIST_RATIO_AXIS_SCALE,
        min_command_norm: float = TWIST_RATIO_MIN_COMMAND_NORM,
        direction_penalty: float = TWIST_RATIO_DIRECTION_PENALTY,
        uncommanded_scale: Sequence[float] | None = TWIST_RATIO_UNCOMMANDED_SCALE,
        filter_time_constant: float = TWIST_RATIO_FILTER_TIME_CONSTANT_S,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        if not filter_time_constant >= 0.0:
            raise ValueError("filter_time_constant must be non-negative")
        command = env.command_manager.get_command(command_name)[:, :3]
        linear, angular = home_levelled_velocities(env, trunk_pitch, asset_cfg)
        twist = torch.stack((linear[:, 0], linear[:, 1], angular[:, 2]), dim=-1)
        uncommanded = torch.stack((linear[:, 2], angular[:, 0], angular[:, 1]), dim=-1)
        gain = (
            1.0
            if filter_time_constant == 0.0
            else 1.0 - math.exp(-self.step_dt / filter_time_constant)
        )
        fresh = self.fresh.unsqueeze(-1)
        self.command = torch.where(fresh, command, self.command + gain * (command - self.command))
        self.twist = torch.where(fresh, twist, self.twist + gain * (twist - self.twist))
        self.uncommanded = torch.where(
            fresh, uncommanded, self.uncommanded + gain * (uncommanded - self.uncommanded)
        )
        self.fresh[:] = False
        parts = twist_ratio(
            self.command,
            self.twist,
            axis_scale,
            min_command_norm,
            None if uncommanded_scale is None else self.uncommanded,
            uncommanded_scale,
        )
        extras = getattr(env, "extras", None)
        if isinstance(extras, dict):
            log = extras.setdefault("log", {})
            log["Metrics/twist_ratio_speed"] = parts.speed.mean()
            log["Metrics/twist_ratio_error"] = parts.error.mean()
        return _bounded(parts, direction_penalty)
