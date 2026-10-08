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
``v^ = v / scale``, ``n = |c^|``, ``u = c^ / max(n, eps)`` and ``p = v^ . u``:

``a = clamp(p, 0, n)``
    progress along the command, capped at the command;
``speed = 1 - (n - a) / max(n, eps)``
    the share of the command achieved, in [0, 1], for every command: one
    formula, no separate small-command branch.  For ``n >= eps`` it is
    ``a / n``, the along-command speed fraction.  ``eps`` (0.01 in normalized
    units, ``TWIST_RATIO_EPS``) keeps the divisions finite: the direction is
    ``u = c^ / max(n, eps)`` too, so for ``n >= eps`` it is the unit command
    direction and below it shrinks continuously to 0.  A standing command
    (``n = 0``) has ``u = 0``, ``p = a = 0`` and speed 1: nothing is left
    undone, and all its motion is off-command motion, so standing still
    scores the full 1 there, as exact tracking does on a moving command;
``error = sqrt(|v^ - a u|^2 + |w^|^2) + max(0, -p)``
    the norm of everything that is not capped forward progress along the
    command -- the perpendicular part, the overshoot beyond the command, any
    backward part, and the uncommanded motion ``w^`` (vertical velocity, roll
    and pitch rates, each divided by its scale, default 0.7 m/s, 1.5 rad/s,
    1.5 rad/s) -- plus the backward part once more.  At the same speed k,
    perpendicular motion costs k and opposite motion 2k; the error grows
    monotonically with the angle to the command.

The reward is ``(1 + speed) / 2 * exp(-direction_penalty * (error / sigma)^2)``
with ``sigma = TWIST_RATIO_ERROR_SCALE`` (1.0, normalized units), in [0, 1]: 1 at exact tracking of every command (standing still on a standing
command included), 1/2 for standing still on a moving command
(``n >= eps``), less for anything off the command, and never negative.
The speed was ``a / max(n, eps)`` until 2026-10-07: it gave standing still
on a standing command speed 0 (1/2), hardly above drifting forward on it
(0.46), and the release walker trained with it drifted +0.05..0.08 m/s
forward and -0.07..-0.15 rad/s in yaw on the standing command, which
cancelled the slow backward commands; the standing command's best was half
of every other command's.  On every moving
command any progress along it beats standing still, also at twice the
command (the overshoot only costs error ``(k - 1) n``), and moving against
it is worse than standing; the reward is continuous in the command, also
where ``n`` crosses ``eps`` and at 0.  It replaces a small-command branch
(``speed = clamp(1 - |v^ - c^| / min_command_norm, 0, 1)`` below the norm
``min_command_norm`` 0.2) that scored the 0.1 m/s commands like standing
ones, where standing still beat walking at twice the command: the
2026-10-07 release walker stood still on them.  For a feasible command its
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
28.5 deg); this form only weakens it.  The walking checks therefore judge
the direction by this reward itself (mjlab_microban/twist_pass_line.py: a
moving command passes when its mean twist beats standing still), not by an
angle to the command ray.  For a zero
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
  4000 the walker fell within one second in every standing rollout.
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

The error enters squared (2026-10-08, user: the former reward made the
walker stand still on the standing command by update 4000 with the same 10 %
standing commands).  The linear form ``exp(-error)`` was sensitive to the
small sway of every stride, so it was evaluated on 0.5 s filtered motion;
that filter let a stepping-in-place gait cost nothing and the walkers trained
with it stepped in place on the standing command from the first updates
(walk_c1ab6dcb, walk_91b4dbec: 3.5 touchdowns per second).  The former
reward's exp kernels ``exp(-|v - c|^2 / sigma^2)`` (sigma 0.316 m/s linear,
0.707 rad/s yaw) were tolerant of small sway and evaluated on the
instantaneous motion, where stepping in place does cost.  The squared form
does the same on the instantaneous command and motion (no filter): small
errors cost little, large ones (a broken ratio, a fall) a lot.  Sigma: the
former kernels' sigmas are 0.45 (forward), 1.05 (lateral) and 0.47 (yaw) in
normalized units; this error is one norm that also holds the uncommanded
vertical velocity and roll/pitch rates, and on the instantaneous motion of
the forward-lean walker that stands still (lean_walk_cont2 model_29000, no
pushes) walking at 0.1 m/s lateral beats standing still only for sigma >=
0.8 (0.39 at 0.45, 0.57 at 0.8, 0.62 at 1.0), so sigma is 1.0, the lateral
kernel's: on that walker the small commands score 0.62-0.82, stepping in
place on the standing command (walk_91b4dbec model_4500) 0.81 against 1.0
standing still.  The reward term ``twist_ratio_velocity`` evaluates it on
the instantaneous command and motion.  It replaces the separate velocity tracking terms (exp kernels, L1 errors and the
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
# Floor of the command norm in the divisions (normalized units).  It keeps
# speed = 1 - (n - a) / max(n, eps) and u = c^ / max(n, eps) finite near a
# standing command (see the module doc).
TWIST_RATIO_EPS = 0.01
TWIST_RATIO_DIRECTION_PENALTY = 1.0
# sigma of the squared error, normalized units (module doc).
TWIST_RATIO_ERROR_SCALE = 1.0

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class TwistRatio(NamedTuple):
    """Per-env decomposition in envelope-normalized units (see module doc)."""

    command_norm: torch.Tensor  # (N,) n = |c^|
    along: torch.Tensor  # (N,) p = v^ . u with u = c^ / max(n, eps)
    speed: torch.Tensor  # (N,) 1 - (n - a) / max(n, eps), in [0, 1] (a / n for n >= eps)
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
    uncommanded: torch.Tensor | None = None,
    uncommanded_scale: Sequence[float] | None = TWIST_RATIO_UNCOMMANDED_SCALE,
) -> TwistRatio:
    """Decompose (N, 3) twists against (N, 3) commands (v_x, v_y, w_z).

    ``uncommanded`` is the (N, K) motion no command asks for (by default
    v_z, w_x, w_y), divided by ``uncommanded_scale``; ``None`` leaves it out.
    """

    if command.ndim != 2 or command.shape[-1] != 3 or twist.shape != command.shape:
        raise ValueError("twist_ratio needs (N, 3) command and twist tensors")
    scale = _scale_tensor(axis_scale, command, 3)
    c = command / scale
    v = twist.to(command.dtype) / scale
    norm = torch.linalg.vector_norm(c, dim=-1)
    floor = torch.clamp(norm, min=TWIST_RATIO_EPS)
    unit = c / floor.unsqueeze(-1)
    along = (v * unit).sum(dim=-1)
    progress = torch.minimum(torch.clamp(along, min=0.0), norm)
    speed = 1.0 - (norm - progress) / floor
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
    direction_penalty: float = TWIST_RATIO_DIRECTION_PENALTY,
    uncommanded: torch.Tensor | None = None,
    uncommanded_scale: Sequence[float] | None = TWIST_RATIO_UNCOMMANDED_SCALE,
) -> torch.Tensor:
    """``(1 + speed) / 2 * exp(-direction_penalty * (error / sigma)^2)`` per env."""

    if not direction_penalty > 0.0:
        raise ValueError("direction_penalty must be positive")
    parts = twist_ratio(
        command, twist, axis_scale, uncommanded, uncommanded_scale
    )
    return _bounded(parts, direction_penalty)


def _bounded(parts: TwistRatio, direction_penalty: float) -> torch.Tensor:
    scaled = parts.error / TWIST_RATIO_ERROR_SCALE
    return 0.5 * (1.0 + parts.speed) * torch.exp(-direction_penalty * scaled * scaled)


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
    """Reward term: the bounded twist-ratio reward on the instantaneous motion.

    The command and the measured HOME-levelled motion of the current step,
    no filter (module doc: the squared error tolerates the small sway of a
    stride; on a standing command any motion, stepping in place included,
    costs).  ``command``, ``twist`` and ``uncommanded`` keep the last values
    evaluated (diagnostics).
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        num_envs, device = env.num_envs, env.device
        self.command = torch.zeros(num_envs, 3, device=device)
        self.twist = torch.zeros(num_envs, 3, device=device)
        self.uncommanded = torch.zeros(num_envs, 3, device=device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        return None

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str = "twist",
        trunk_pitch: float = 0.0,
        axis_scale: Sequence[float] = TWIST_RATIO_AXIS_SCALE,
        direction_penalty: float = TWIST_RATIO_DIRECTION_PENALTY,
        uncommanded_scale: Sequence[float] | None = TWIST_RATIO_UNCOMMANDED_SCALE,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        if not direction_penalty > 0.0:
            raise ValueError("direction_penalty must be positive")
        command = env.command_manager.get_command(command_name)[:, :3]
        linear, angular = home_levelled_velocities(env, trunk_pitch, asset_cfg)
        self.command = command
        self.twist = torch.stack((linear[:, 0], linear[:, 1], angular[:, 2]), dim=-1)
        self.uncommanded = torch.stack((linear[:, 2], angular[:, 0], angular[:, 1]), dim=-1)
        parts = twist_ratio(
            self.command,
            self.twist,
            axis_scale,
            None if uncommanded_scale is None else self.uncommanded,
            uncommanded_scale,
        )
        extras = getattr(env, "extras", None)
        if isinstance(extras, dict):
            log = extras.setdefault("log", {})
            log["Metrics/twist_ratio_speed"] = parts.speed.mean()
            log["Metrics/twist_ratio_error"] = parts.error.mean()
        return _bounded(parts, direction_penalty)
