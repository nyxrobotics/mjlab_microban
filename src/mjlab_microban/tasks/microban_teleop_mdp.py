# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Teleoperation-only MDP terms for Microban.

The 18-DoF body policy observes, but never controls, the three camera-neck
joints.  During training those joints therefore need an external command that
resembles the independent HMD controller used on the physical robot.  The term
below generates random HMD waypoints and slews the position targets towards
them at the same bounded rate as the robot runtime.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microban.tasks.mdp import (
    FootTargetCommand,
    FootTargetCommandCfg,
    HandTargetCommand,
    HandTargetCommandCfg,
)
from mjlab_microban.tasks.microban_policy_export import MICROBAN_HMD_JOINT_NAMES

# Runtime command limits from microban/src/moves/hmd_head.py.  The event further
# intersects these with Entity.data.soft_joint_pos_limits, so the effective
# training bounds retain the articulation's 10% mechanical-limit margin.
MICROBAN_HMD_RUNTIME_LIMITS_RAD: dict[str, tuple[float, float]] = {
    "head": (-math.radians(85.0), math.radians(85.0)),
    "neck_roll": (-math.radians(23.0), math.radians(23.0)),
    "neck_pitch": (-math.radians(85.0), math.radians(23.0)),
}

# HmdHeadTrackingMove uses one 2.5 rad/s limiter for all three axes.  Matching
# it here makes target motion no faster than deployment even when a random
# waypoint is on the opposite side of the range.
MICROBAN_HMD_SLEW_RATES_RAD_S: dict[str, float] = {
    name: 2.5 for name in MICROBAN_HMD_JOINT_NAMES
}

MICROBAN_HMD_RETARGET_INTERVAL_S = (0.35, 1.50)

# The live PICO bridge treats a support-foot target at or below this height as
# measurement jitter and projects the complete XYZ vector to exact zero.  V2
# training samples active feet from this boundary upward so the first live value
# above the floor band remains inside learned support.
MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M = 0.0025

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _joint_position_action_tensors(
    env: ManagerBasedRlEnv,
    action_name: str,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Return raw, effective-raw, target, lower and upper action tensors."""

    action = env.action_manager.get_term(action_name)
    if not isinstance(action, JointPositionAction):
        raise TypeError(f"{action_name!r} must be a JointPositionAction")
    if action.cfg.clip is None or not hasattr(action, "_clip"):
        raise ValueError(f"{action_name!r} must define absolute target clips")

    raw = action.raw_action
    scale = torch.as_tensor(action.scale, dtype=raw.dtype, device=raw.device)
    offset = torch.as_tensor(action.offset, dtype=raw.dtype, device=raw.device)
    if not bool(torch.isfinite(scale).all().item()) or not bool(
        torch.all(scale > 0.0).item()
    ):
        raise ValueError("joint-position action scale must be finite and positive")
    if not bool(torch.isfinite(offset).all().item()):
        raise ValueError("joint-position action offset must be finite")

    clip = action._clip
    lower = clip[..., 0]
    upper = clip[..., 1]
    if not bool(torch.all(lower < upper).item()):
        raise ValueError("joint-position action clips must have lower < upper")

    target = raw * scale + offset
    clipped_target = torch.clamp(target, min=lower, max=upper)
    effective_raw = (clipped_target - offset) / scale
    return raw, effective_raw, target, lower, upper


def normalized_target_clip_excess_l1_sum(
    env: ManagerBasedRlEnv,
    action_name: str = "joint_pos",
) -> torch.Tensor:
    """Return a non-vanishing, range-normalized target saturation barrier.

    The absolute target excess is normalized by each soft range's half-width,
    then summed over joints.  Values inside the target range are exactly zero.
    L1 intentionally keeps a linear reward/return signal immediately outside
    the boundary; the v3 joint-mean smooth-L1 term divided sparse violations by
    18 and made that signal quadratic near the boundary, allowing deterministic
    target clipping to grow despite a nominally large reward weight.
    """

    _, _, target, lower, upper = _joint_position_action_tensors(env, action_name)
    clipped_target = torch.clamp(target, min=lower, max=upper)
    half_range = 0.5 * (upper - lower)
    normalized_excess = (target - clipped_target) / half_range
    return torch.abs(normalized_excess).sum(dim=-1)


def target_beyond_joint_limit_l1_sum(
    env: ManagerBasedRlEnv,
    joint_names: Sequence[str],
    action_name: str = "joint_pos",
) -> torch.Tensor:
    """Radians by which the unclipped targets of ``joint_names`` lie past the joints' stops.

    A target past a joint's mechanical range (MJCF limit) leaves the joint on
    its stop wherever the target is, so every position-based term is flat in
    that action and a policy mean that drifts there gets no gradient back.
    Linear in the excess, zero inside the range; ``joint_names`` are regexes
    over the action's joints.
    """

    action = env.action_manager.get_term(action_name)
    if not isinstance(action, JointPositionAction):
        raise TypeError(f"{action_name!r} must be a JointPositionAction")
    entity = action._entity
    ids = torch.as_tensor(action.target_ids, device=env.device).reshape(-1)
    names = [entity.joint_names[int(i)] for i in ids]
    selected = [k for k, name in enumerate(names) if any(re.fullmatch(p, name) for p in joint_names)]
    if not selected:
        raise ValueError(f"no action joint matches {list(joint_names)}")
    raw = action.raw_action
    scale = torch.as_tensor(action.scale, dtype=raw.dtype, device=raw.device)
    offset = torch.as_tensor(action.offset, dtype=raw.dtype, device=raw.device)
    target = torch.broadcast_to(raw * scale + offset, raw.shape)[:, selected]
    limits = entity.data.joint_pos_limits[:, ids][:, selected]
    excess = torch.clamp(limits[..., 0] - target, min=0.0) + torch.clamp(target - limits[..., 1], min=0.0)
    return excess.sum(dim=-1)


def normalized_joint_soft_limit_guard_l1_sum(
    env: ManagerBasedRlEnv,
    action_name: str = "joint_pos",
    margin_ratio: float = 0.05,
    lookahead_s: float = 0.12,
) -> torch.Tensor:
    """Penalize controlled joints near a soft limit now or after a short coast.

    Bounding position *targets* is not sufficient to bound the measured joint
    state: actuator delay, gravity and momentum can carry a joint past its soft
    limit while its target remains legal.  This guard evaluates both the current
    position and a constant-velocity projection over the maximum configured
    actuator delay (six 50 Hz policy samples).  The more dangerous value on each
    side is compared with an inward ``margin_ratio`` and normalized by half of
    that joint's soft range before summing.

    An inward margin is clamped at the configured default pose, so the guard
    never asks the measured joint to move away from neutral; it does penalize
    gravity sag or projected motion past that neutral boundary.
    Microban's shoulder-roll defaults are only one degree inside their soft
    limits, so the policy can learn the inward *target* needed to hold the
    measured shoulder at neutral without changing that neutral pose.  This is a
    training signal, not a deployment-time safety filter; deterministic
    evaluation must still reject every actual violation.
    """

    if not math.isfinite(margin_ratio) or not 0.0 < margin_ratio < 0.5:
        raise ValueError("margin_ratio must be finite and in (0, 0.5)")
    if not math.isfinite(lookahead_s) or lookahead_s < 0.0:
        raise ValueError("lookahead_s must be finite and non-negative")

    action = env.action_manager.get_term(action_name)
    if not isinstance(action, JointPositionAction):
        raise TypeError(f"{action_name!r} must be a JointPositionAction")
    entity = action._entity
    limits = entity.data.soft_joint_pos_limits[:, action.target_ids]
    joint_pos = entity.data.joint_pos[:, action.target_ids]
    joint_vel = entity.data.joint_vel[:, action.target_ids]
    if not bool(
        torch.isfinite(limits).all()
        and torch.isfinite(joint_pos).all()
        and torch.isfinite(joint_vel).all()
    ):
        raise ValueError("joint soft-limit guard inputs must be finite")

    lower = limits[..., 0]
    upper = limits[..., 1]
    if not bool(torch.all(lower < upper).item()):
        raise ValueError("joint soft-limit guard requires lower < upper")
    projected_pos = joint_pos + lookahead_s * joint_vel
    dangerous_lower = torch.minimum(joint_pos, projected_pos)
    dangerous_upper = torch.maximum(joint_pos, projected_pos)
    span = upper - lower
    default_target = torch.as_tensor(
        action.offset, dtype=joint_pos.dtype, device=joint_pos.device
    )
    default_target = torch.broadcast_to(default_target, joint_pos.shape)
    if not bool(
        torch.isfinite(default_target).all()
        and torch.all((default_target >= lower) & (default_target <= upper)).item()
    ):
        raise ValueError("joint soft-limit guard default target must be in bounds")
    preferred_lower = torch.minimum(default_target, lower + margin_ratio * span)
    preferred_upper = torch.maximum(default_target, upper - margin_ratio * span)
    lower_excess = torch.clamp(preferred_lower - dangerous_lower, min=0.0)
    upper_excess = torch.clamp(dangerous_upper - preferred_upper, min=0.0)
    return ((lower_excess + upper_excess) / (0.5 * span)).sum(dim=-1)


class ResetFixedFootTargetCommand(FootTargetCommand):
    """Foot offsets whose trunk-frame zero is fixed for one episode.

    ``CommandTerm`` also calls ``_resample_command`` when its timer expires.
    The shared Microban term snapshots its reference in that method, which makes
    the meaning of a zero offset drift every 3--8 seconds.  This task-specific
    term captures the reference only after an episode reset.  Capture is deferred
    to the first command-manager compute so MuJoCo forward kinematics already
    reflects the newly reset joint state.
    """

    def __init__(self, cfg: ResetFixedFootTargetCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        if not 0.0 <= cfg.rel_both_feet_envs <= 1.0:
            raise ValueError("rel_both_feet_envs must be in [0, 1]")
        if not (
            cfg.both_feet_lift_height_range[0] <= cfg.both_feet_lift_height_range[1]
        ):
            raise ValueError("both-feet lift range must have lower <= upper")
        if not (cfg.both_feet_reach_xy_range[0] <= cfg.both_feet_reach_xy_range[1]):
            raise ValueError("both-feet XY range must have lower <= upper")
        self._reference_pending = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.is_both_feet_env = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._previous_both_feet_env = torch.zeros_like(self.is_both_feet_env)
        self._velocity_cache_valid = torch.zeros_like(self.is_both_feet_env)
        self._velocity_command_counter: torch.Tensor | None = None
        self._saved_vel_command_b: torch.Tensor | None = None
        self._saved_vel_command_w: torch.Tensor | None = None
        self._saved_is_rotation_env: torch.Tensor | None = None

    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        if not isinstance(env_ids, torch.Tensor):
            raise TypeError("Foot target reset requires explicit environment IDs")
        self._reference_pending[env_ids] = True
        extras = super().reset(env_ids)

        # Command counters restart from the same value every episode, so equality
        # alone cannot distinguish a fresh twist from the previous episode's
        # cached command.  Clear both transition history and cache validity for
        # exactly the reset rows.  The first post-reset update will snapshot the
        # new episode's twist before applying any both-feet stationary mask.
        self._previous_both_feet_env[env_ids] = False
        self._velocity_cache_valid[env_ids] = False
        return extras

    def _capture_pending_reference(self) -> None:
        env_ids = self._reference_pending.nonzero(as_tuple=False).flatten()
        if len(env_ids) == 0:
            return
        self._default_foot_pos_b[env_ids] = self.current_foot_pos_b()[env_ids]
        self._reference_pending[env_ids] = False

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        # Reuse the shared sampler while preventing it from replacing our
        # episode-fixed reference with the current mid-episode pose.
        reference = self._default_foot_pos_b[env_ids].clone()
        super()._resample_command(env_ids)
        self._default_foot_pos_b[env_ids] = reference

        both_random = torch.rand(len(env_ids), device=self.device)
        both_mask = both_random < self.cfg.rel_both_feet_envs
        both_ids = env_ids[both_mask]
        self.is_both_feet_env[env_ids] = False
        self.is_both_feet_env[both_ids] = True
        self.is_single_support_env[both_ids] = False
        if len(both_ids) == 0:
            return

        xy_lower, xy_upper = self.cfg.both_feet_reach_xy_range
        z_lower, z_upper = self.cfg.both_feet_lift_height_range
        offsets = torch.empty((len(both_ids), 2, 3), device=self.device)
        offsets[..., :2].uniform_(xy_lower, xy_upper)
        offsets[..., 2].uniform_(z_lower, z_upper)
        self.foot_target_offset_b[both_ids] = offsets

    def _update_metrics(self) -> None:
        self._capture_pending_reference()
        super()._update_metrics()

    def _update_command(self) -> None:
        self._capture_pending_reference()
        velocity = self._env.command_manager.get_term(self.cfg.velocity_command_name)
        if self._velocity_command_counter is None:
            self._velocity_command_counter = velocity.command_counter.clone()
            self._saved_vel_command_b = velocity.vel_command_b.clone()
            if hasattr(velocity, "vel_command_w"):
                self._saved_vel_command_w = velocity.vel_command_w.clone()
            if hasattr(velocity, "is_rotation_env"):
                self._saved_is_rotation_env = velocity.is_rotation_env.clone()

        assert self._saved_vel_command_b is not None
        assert self._velocity_command_counter is not None
        resampled = (~self._velocity_cache_valid) | (
            velocity.command_counter != self._velocity_command_counter
        )
        entered = self.is_both_feet_env & ~self._previous_both_feet_env
        save_ids = resampled | entered
        self._saved_vel_command_b[save_ids] = velocity.vel_command_b[save_ids]
        if self._saved_vel_command_w is not None:
            self._saved_vel_command_w[save_ids] = velocity.vel_command_w[save_ids]
        if self._saved_is_rotation_env is not None:
            self._saved_is_rotation_env[save_ids] = velocity.is_rotation_env[save_ids]

        # A foot-target resample may leave the both-feet regime before the
        # independent twist timer fires. Restore the latest unmasked twist now;
        # otherwise the zero injected below can persist for several seconds.
        exited_ids = (
            (self._previous_both_feet_env & ~self.is_both_feet_env)
            .nonzero(as_tuple=False)
            .flatten()
        )
        velocity.vel_command_b[exited_ids] = self._saved_vel_command_b[exited_ids]
        if self._saved_vel_command_w is not None:
            velocity.vel_command_w[exited_ids] = self._saved_vel_command_w[exited_ids]
        if self._saved_is_rotation_env is not None:
            velocity.is_rotation_env[exited_ids] = self._saved_is_rotation_env[
                exited_ids
            ]

        both_ids = self.is_both_feet_env.nonzero(as_tuple=False).flatten()
        velocity.vel_command_b[both_ids] = 0.0
        if hasattr(velocity, "vel_command_w"):
            velocity.vel_command_w[both_ids] = 0.0
        if hasattr(velocity, "is_rotation_env"):
            velocity.is_rotation_env[both_ids] = False
        self._previous_both_feet_env.copy_(self.is_both_feet_env)
        self._velocity_command_counter.copy_(velocity.command_counter)
        self._velocity_cache_valid.fill_(True)


@dataclass(kw_only=True)
class ResetFixedFootTargetCommandCfg(FootTargetCommandCfg):
    """Episode-fixed feet, including conservative stationary two-foot targets."""

    lift_height_range: tuple[float, float] = (
        MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M,
        0.05,
    )
    rel_both_feet_envs: float = 0.0
    both_feet_lift_height_range: tuple[float, float] = (
        MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M,
        0.012,
    )
    both_feet_reach_xy_range: tuple[float, float] = (-0.01, 0.01)
    velocity_command_name: str = "twist"

    def build(self, env: ManagerBasedRlEnv) -> ResetFixedFootTargetCommand:
        return ResetFixedFootTargetCommand(self, env)


class ResetFixedHandTargetCommand(HandTargetCommand):
    """Hand offsets whose trunk-frame zero is fixed for one episode."""

    def __init__(self, cfg: ResetFixedHandTargetCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._reference_pending = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        if not isinstance(env_ids, torch.Tensor):
            raise TypeError("Hand target reset requires explicit environment IDs")
        self._reference_pending[env_ids] = True
        return super().reset(env_ids)

    def _capture_pending_reference(self) -> None:
        env_ids = self._reference_pending.nonzero(as_tuple=False).flatten()
        if len(env_ids) == 0:
            return
        self._default_hand_pos_b[env_ids] = self.current_hand_pos_b()[env_ids]
        self._reference_pending[env_ids] = False

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        reference = self._default_hand_pos_b[env_ids].clone()
        super()._resample_command(env_ids)
        self._default_hand_pos_b[env_ids] = reference

    def _update_metrics(self) -> None:
        self._capture_pending_reference()
        super()._update_metrics()

    def _update_command(self) -> None:
        self._capture_pending_reference()


@dataclass(kw_only=True)
class ResetFixedHandTargetCommandCfg(HandTargetCommandCfg):
    """Configuration for episode-reset-fixed hand targets."""

    def build(self, env: ManagerBasedRlEnv) -> ResetFixedHandTargetCommand:
        return ResetFixedHandTargetCommand(self, env)


def _ordered_values(
    values: Mapping[str, Any], names: Sequence[str], label: str
) -> list[Any]:
    missing = set(names) - values.keys()
    extra = values.keys() - set(names)
    if missing or extra:
        raise ValueError(
            f"{label} keys must be exactly {list(names)}; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return [values[name] for name in names]


class HmdNeckTargetMotion:
    """Stateful, CUDA-vectorized external target motion for the camera neck.

    This is a ``mode='step'`` event term.  Each environment independently
    samples a new waypoint after a seeded random interval, while its actuator
    target moves every 50 Hz policy step with a strict per-axis slew bound.
    The policy action manager does not include these joints, so this term is the
    sole owner of their simulation targets.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv) -> None:
        params = cfg.params
        asset_cfg = params.get("asset_cfg")
        if not isinstance(asset_cfg, SceneEntityCfg):
            raise TypeError("HmdNeckTargetMotion requires a SceneEntityCfg")

        names = tuple(asset_cfg.joint_names or ())
        if names != MICROBAN_HMD_JOINT_NAMES:
            raise ValueError(
                f"HMD joint order must be {MICROBAN_HMD_JOINT_NAMES}, got {names}"
            )
        if not isinstance(asset_cfg.joint_ids, list):
            raise TypeError("HMD joint selection must resolve to three explicit IDs")

        self.asset: Entity = env.scene[asset_cfg.name]
        self.joint_names = names
        self.joint_ids = torch.tensor(
            asset_cfg.joint_ids, dtype=torch.long, device=env.device
        )

        configured_limits = _ordered_values(
            params["position_ranges_rad"], names, "position_ranges_rad"
        )
        configured_lower = torch.tensor(
            [float(bounds[0]) for bounds in configured_limits],
            dtype=torch.float32,
            device=env.device,
        ).unsqueeze(0)
        configured_upper = torch.tensor(
            [float(bounds[1]) for bounds in configured_limits],
            dtype=torch.float32,
            device=env.device,
        ).unsqueeze(0)
        if not bool(torch.all(configured_lower < configured_upper).item()):
            raise ValueError("Every HMD position range must have lower < upper")

        # Limits may be per-environment after domain randomization.  Keep the
        # intersection as an (N, 3) tensor rather than assuming one global row.
        soft_limits = self.asset.data.soft_joint_pos_limits[:, self.joint_ids]
        self.position_lower = torch.maximum(configured_lower, soft_limits[..., 0])
        self.position_upper = torch.minimum(configured_upper, soft_limits[..., 1])
        if not bool(torch.all(self.position_lower < self.position_upper).item()):
            raise ValueError("Configured HMD ranges do not overlap the soft limits")

        slew_rates = _ordered_values(
            params["slew_rates_rad_s"], names, "slew_rates_rad_s"
        )
        self.slew_rates_rad_s = torch.tensor(
            [float(value) for value in slew_rates],
            dtype=torch.float32,
            device=env.device,
        ).unsqueeze(0)
        if not bool(torch.all(self.slew_rates_rad_s > 0.0).item()):
            raise ValueError("HMD slew rates must be positive")

        interval = params["retarget_interval_s"]
        if len(interval) != 2:
            raise ValueError("retarget_interval_s must contain (minimum, maximum)")
        self.retarget_interval_s = (float(interval[0]), float(interval[1]))
        if not (0.0 < self.retarget_interval_s[0] <= self.retarget_interval_s[1]):
            raise ValueError("HMD retarget interval must be finite and positive")
        if not all(math.isfinite(value) for value in self.retarget_interval_s):
            raise ValueError("HMD retarget interval must be finite")

        self.neutral_probability = float(params.get("neutral_probability", 0.2))
        if not 0.0 <= self.neutral_probability <= 1.0:
            raise ValueError("neutral_probability must be in [0, 1]")

        # The "neutral" HMD pose used for resets and neutral waypoints.  By
        # default it is HOME (default_joint_pos, read when used).
        # ``neutral_position_rad`` (absolute joint angles) moves it, e.g. to the
        # pose a level headset produces on a trunk that leans forward at HOME.
        neutral_position = params.get("neutral_position_rad")
        self._neutral_override: torch.Tensor | None = None
        if neutral_position is not None:
            values = [
                float(value)
                for value in _ordered_values(
                    neutral_position, names, "neutral_position_rad"
                )
            ]
            if not all(math.isfinite(value) for value in values):
                raise ValueError("HMD neutral position must be finite")
            override = (
                torch.tensor(values, dtype=torch.float32, device=env.device)
                .unsqueeze(0)
                .expand(env.num_envs, -1)
                .clone()
            )
            if not bool(
                torch.all(
                    (override >= self.position_lower)
                    & (override <= self.position_upper)
                ).item()
            ):
                raise ValueError("HMD neutral position lies outside the HMD limits")
            self._neutral_override = override

        all_envs = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
        self.current_target = self.neutral_target(all_envs).clone()
        self.goal_target = self.current_target.clone()
        # Zero causes a waypoint to be sampled on the first post-reset step.
        self.time_to_retarget_s = torch.zeros(
            env.num_envs, dtype=torch.float32, device=env.device
        )

    def neutral_target(self, indices: torch.Tensor) -> torch.Tensor:
        """Return the clamped neutral HMD targets of ``indices``. Shape (n, 3)."""

        if self._neutral_override is not None:
            return self._neutral_override[indices]
        default = self.asset.data.default_joint_pos[indices][:, self.joint_ids]
        return torch.clamp(
            default,
            min=self.position_lower[indices],
            max=self.position_upper[indices],
        )

    def _explicit_env_ids(self, env_ids: torch.Tensor | slice | None) -> torch.Tensor:
        if env_ids is None or isinstance(env_ids, slice):
            return torch.arange(
                self.current_target.shape[0],
                dtype=torch.long,
                device=self.current_target.device,
            )[env_ids if isinstance(env_ids, slice) else slice(None)]
        return env_ids.to(device=self.current_target.device, dtype=torch.long)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        indices = self._explicit_env_ids(env_ids)
        neutral = self.neutral_target(indices)
        self.current_target[indices] = neutral
        self.goal_target[indices] = neutral
        self.time_to_retarget_s[indices] = 0.0
        self.asset.set_joint_position_target(
            neutral,
            joint_ids=self.joint_ids,
            env_ids=indices.unsqueeze(-1),
        )

    def _sample_waypoints(self, indices: torch.Tensor) -> None:
        count = indices.numel()
        if count == 0:
            return
        lower = self.position_lower[indices]
        upper = self.position_upper[indices]
        random_unit = torch.rand(
            (count, len(self.joint_names)), device=self.current_target.device
        )
        sampled = lower + random_unit * (upper - lower)

        if self.neutral_probability > 0.0:
            neutral_mask = (
                torch.rand((count, 1), device=self.current_target.device)
                < self.neutral_probability
            )
            # Neutral waypoints: the override (validated inside the limits), or
            # HOME itself (unclamped, as before the override existed; the
            # goal is clamped below).
            neutral = (
                self._neutral_override[indices]
                if self._neutral_override is not None
                else self.asset.data.default_joint_pos[indices][:, self.joint_ids]
            )
            sampled = torch.where(neutral_mask, neutral, sampled)

        self.goal_target[indices] = torch.clamp(sampled, min=lower, max=upper)
        minimum, maximum = self.retarget_interval_s
        self.time_to_retarget_s[indices] = (
            torch.rand(count, device=self.current_target.device) * (maximum - minimum)
            + minimum
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        neutral_probability: float | None = None,
        **_unused: Any,
    ) -> None:
        # Step events are always dispatched for all environments.  Per-env
        # episode resets are handled by reset(), which only rewrites those rows.
        if env_ids is not None:
            raise ValueError("HmdNeckTargetMotion step event expects env_ids=None")
        # The params are passed on every call, so a curriculum stage that
        # writes params["neutral_probability"] takes effect at once.
        if neutral_probability is not None:
            if not 0.0 <= neutral_probability <= 1.0:
                raise ValueError("neutral_probability must be in [0, 1]")
            self.neutral_probability = float(neutral_probability)

        self.time_to_retarget_s -= env.step_dt
        due = (self.time_to_retarget_s <= 0.0).nonzero().flatten()
        self._sample_waypoints(due)

        max_step = self.slew_rates_rad_s * env.step_dt
        delta = self.goal_target - self.current_target
        delta = torch.maximum(torch.minimum(delta, max_step), -max_step)
        self.current_target = torch.clamp(
            self.current_target + delta,
            min=self.position_lower,
            max=self.position_upper,
        )
        self.asset.set_joint_position_target(
            self.current_target, joint_ids=self.joint_ids
        )
