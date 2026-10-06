"""Ratio-keeping twist velocity reward (one term for walking and PICO teleop).

User specification (2026-10-06):

* beyond the robot's speed, keep the ratio of the commanded twist components
  and walk at the largest speed it can manage ("速度の成分の比は合わせつつ、移動
  できる最大の速度で歩く"): execute ``s * c`` with one common scale
  ``s <= 1`` (``s = 1`` whenever the command is feasible);
* reward speed only through the component of the motion along the command
  direction ("司令方向成分だけ取り出して報酬"), so moving fast in another direction
  earns nothing;
* moving against the command is bad too ("反対向きもだめ");
* standing still must never pay better than moving the commanded way: a
  deviating step forward is still a plus, an exact one a big plus ("多少ずれて
  ても＋で、一致してたらめっちゃ＋").

The commanded twist is (v_x, v_y, w_z) in the HOME-levelled trunk frame.  Each
axis is divided by the command envelope's maximum (default 0.7 m/s, 0.3 m/s,
1.5 rad/s, the robot's moving ``scale_velocity`` limits) so metres per second
and radians per second are comparable.  With ``c^ = c / scale``,
``v^ = v / scale``, ``n = |c^|``, ``u = c^ / n`` and ``p = v^ . u``:

``a = clamp(p, 0, n)``
    progress along the command, capped at the command;
``s = a / n`` when ``n >= min_command_norm``
    the along-command speed fraction, in [0, 1]; for a smaller command
    (standing included) ``s = clamp(1 - |v^ - c^| / min_command_norm, 0, 1)``:
    a small command is tracked for precision on the scale of
    ``min_command_norm`` (both branches agree on the commanded ray at
    ``n = min_command_norm``);
``e = sqrt(|v^ - a u|^2 + |w^|^2)``
    the deviation: everything that is not capped progress along the command
    (the perpendicular part, the overshoot beyond the command, any backward
    part) and the uncommanded motion ``w^`` (vertical velocity, roll and pitch
    rates divided by 0.7 m/s, 1.5 rad/s, 1.5 rad/s).

The reward (form B4) is::

    1 + s * (floor + (1 - floor) * exp(-e / deviation_scale))
      - backward_penalty * max(0, -p) / max(n, min_command_norm)

* standing still on a moving command: 1; moving the commanded way: above 1
  however large the deviation (at least ``1 + floor * s``); exact tracking:
  ``1 + s``, so 2 for a feasible command and for standing still on a standing
  command -- every command has the same best value;
* motion perpendicular to the command: 1 (it earns nothing); motion against
  the command: below 1;
* on an infeasible command the best reachable twist keeps the commanded ratio
  at the largest scale as long as the deviation gain
  ``s (1 - floor) / deviation_scale`` beats the extra progress an off-ray
  twist can buy (``deviation_scale`` small: the reward rises sharply near
  exact agreement).

Both the command and the motion are low-pass filtered with the same
first-order filter (0.5 s) before the reward is evaluated
(``twist_ratio_velocity``): the reward is about the direction and speed the
robot walks at, not the sway within a stride.

Forms tried before (walker from scratch at the forward-lean HOME, held-out
probes on seeds 101-105; details in the validation branch history):

* ``speed - error``: no positive reward for standing upright; episodes stayed
  near 45 steps at update 200 (old reward 190).
* ``1 + speed - error``: a standing command could earn at most half of a moving
  one and falling (no penalty) draws a new command, so the walker fell at once
  when told to stand (update 4000); with equal best values (A2) the reward
  could go negative after a push and 36-43 of 90 pushed rollouts fell.
* ``(1 + speed) / 2 * exp(-error)`` (B2/B3): never negative, but a large
  deviation made moving worth less than standing still: the walker stopped
  answering small single-axis commands (B2, update 4000) or turning right
  (B3, update 4000).  B4 keeps moving the commanded way above standing still.

``twist_ratio_velocity`` is meant to replace the separate velocity tracking
terms (exp kernels, L1 errors and the xy projection progress) of the walking
and PICO base reward configurations.  The module depends only on torch and
mjlab (no robot constants, no other task module): the caller passes the HOME
trunk pitch and, if its command envelope differs, the axis scale.
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
TWIST_RATIO_MIN_COMMAND_NORM = 0.2
TWIST_RATIO_PROGRESS_FLOOR = 0.4
TWIST_RATIO_DEVIATION_SCALE = 0.15
TWIST_RATIO_BACKWARD_PENALTY = 1.0
TWIST_RATIO_FILTER_TIME_CONSTANT_S = 0.5

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class TwistRatio(NamedTuple):
    """Per-env decomposition in envelope-normalized units (see module doc)."""

    command_norm: torch.Tensor  # (N,) n = |c^|
    along: torch.Tensor  # (N,) p = v^ . u (0 for a zero command)
    speed: torch.Tensor  # (N,) s in [0, 1]
    deviation: torch.Tensor  # (N,) e = sqrt(|v^ - a u|^2 + |w^|^2)
    backward: torch.Tensor  # (N,) max(0, -p) / max(n, min_command_norm)


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
    deviation = torch.linalg.vector_norm(off, dim=-1)
    backward = torch.clamp(-along, min=0.0) / torch.clamp(norm, min=min_command_norm)
    return TwistRatio(norm, along, speed, deviation, backward)


def twist_ratio_value(
    parts: TwistRatio,
    progress_floor: float = TWIST_RATIO_PROGRESS_FLOOR,
    deviation_scale: float = TWIST_RATIO_DEVIATION_SCALE,
    backward_penalty: float = TWIST_RATIO_BACKWARD_PENALTY,
) -> torch.Tensor:
    """Form B4 of the module doc from a decomposition."""

    if not 0.0 < progress_floor < 1.0:
        raise ValueError("progress_floor must be in (0, 1)")
    if not deviation_scale > 0.0 or not backward_penalty > 0.0:
        raise ValueError("deviation_scale and backward_penalty must be positive")
    gain = progress_floor + (1.0 - progress_floor) * torch.exp(
        -parts.deviation / deviation_scale
    )
    return 1.0 + parts.speed * gain - backward_penalty * parts.backward


def twist_ratio_reward(
    command: torch.Tensor,
    twist: torch.Tensor,
    axis_scale: Sequence[float] = TWIST_RATIO_AXIS_SCALE,
    min_command_norm: float = TWIST_RATIO_MIN_COMMAND_NORM,
    uncommanded: torch.Tensor | None = None,
    uncommanded_scale: Sequence[float] | None = TWIST_RATIO_UNCOMMANDED_SCALE,
    progress_floor: float = TWIST_RATIO_PROGRESS_FLOOR,
    deviation_scale: float = TWIST_RATIO_DEVIATION_SCALE,
    backward_penalty: float = TWIST_RATIO_BACKWARD_PENALTY,
) -> torch.Tensor:
    """The reward of the module doc per env (pure tensor form, no filter)."""

    parts = twist_ratio(
        command, twist, axis_scale, min_command_norm, uncommanded, uncommanded_scale
    )
    return twist_ratio_value(parts, progress_floor, deviation_scale, backward_penalty)


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
    """Reward term: the reward of the module doc on time-filtered motion.

    The command and the measured motion are both low-pass filtered with the
    same first-order filter (time constant ``filter_time_constant``, 0.5 s by
    default; 0 uses the instantaneous values), and the reward is evaluated on
    the filtered values.  The per-step lateral sway, vertical bob and
    roll/pitch rates of a normal gait average out, while a sustained drift, a
    wrong walking direction or a fall do not.  Filtering the command the same
    way keeps a robot that follows a new command at once on target while both
    settle.  Both filters start at the current values on the first step of an
    episode.
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
        uncommanded_scale: Sequence[float] | None = TWIST_RATIO_UNCOMMANDED_SCALE,
        progress_floor: float = TWIST_RATIO_PROGRESS_FLOOR,
        deviation_scale: float = TWIST_RATIO_DEVIATION_SCALE,
        backward_penalty: float = TWIST_RATIO_BACKWARD_PENALTY,
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
            log["Metrics/twist_ratio_deviation"] = parts.deviation.mean()
            log["Metrics/twist_ratio_backward"] = parts.backward.mean()
        return twist_ratio_value(parts, progress_floor, deviation_scale, backward_penalty)
