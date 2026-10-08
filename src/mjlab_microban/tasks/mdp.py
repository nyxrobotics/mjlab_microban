# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from mjlab.entity import Entity
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
    quat_apply_inverse,
    quat_from_euler_xyz,
    quat_mul,
    sample_uniform,
    subtract_frame_transforms,
)
from mjlab.tasks.velocity.mdp.velocity_command import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)

from mjlab_microban.robot.home_pose import HOME
from mjlab_microban.robot.microban_hand_fk import (
    MICROBAN_ARM_HOME_JOINT_RAD,
    sample_microban_reachable_hand_targets,
)

# HOME forward trunk lean (rad); 0 = vertical trunk (config/home_pose.yaml).
HOME_TRUNK_PITCH_RAD = HOME.trunk_pitch_rad

############################ COMMANDS #############################


_SIGNED_AXIS_MODES = (
    "standing",
    "forward",
    "backward",
    "lateral_left",
    "lateral_right",
    "yaw_left",
    "yaw_right",
    "mixed",
)
_SIGNED_AXIS_RANGE_KEYS = (
    "forward",
    "backward",
    "lateral_left",
    "lateral_right",
    "yaw_left",
    "yaw_right",
)
MICROBAN_BILATERAL_SITE_ORDER_REVISION = "preserve_requested_left_right_sites_v1"


def _resolve_ordered_site_cfg(
    scene: object,
    *,
    entity_name: str,
    site_names: tuple[str, str],
    label: str,
) -> SceneEntityCfg:
    """Resolve one bilateral site pair without losing semantic L/R order.

    ``Entity.find_sites`` follows model order unless ``preserve_order`` is set.
    Microban's XML stores the right site before the left site, while every
    teleoperation command tensor is explicitly ``(left, right)``.  Silently
    accepting model order therefore crosses the two targets.  Keep the
    requested order and fail during environment construction if the resolver
    ever returns anything else.
    """

    expected = tuple(site_names)
    if len(expected) != 2 or len(set(expected)) != 2:
        raise ValueError(f"{label} site names must be one unique bilateral pair")
    asset_cfg = SceneEntityCfg(
        entity_name,
        site_names=expected,
        preserve_order=True,
    )
    asset_cfg.resolve(scene)  # type: ignore[arg-type]
    resolved = tuple(asset_cfg.site_names or ())
    if resolved != expected:
        raise RuntimeError(
            f"{label} site order drifted: expected {expected}, resolved {resolved}"
        )
    return asset_cfg


def _validate_signed_axis_sampler_cfg(
    cfg: UniformVelocityCommandCfg,
) -> tuple[tuple[float, ...], dict[str, tuple[float, float]]] | None:
    """Validate and materialize the opt-in signed-axis command sampler.

    The legacy sampler remains selected when both opt-in fields are ``None``.
    Validation runs at construction and before every resample because curriculum
    stages mutate command configuration in place.
    """

    raw_probabilities = getattr(cfg, "signed_axis_probabilities", None)
    raw_ranges = getattr(cfg, "signed_axis_ranges", None)
    if raw_probabilities is None and raw_ranges is None:
        return None
    if raw_probabilities is None or raw_ranges is None:
        raise ValueError(
            "signed-axis sampling requires both signed_axis_probabilities and "
            "signed_axis_ranges"
        )
    if not isinstance(raw_probabilities, dict):
        raise TypeError("signed_axis_probabilities must be a dictionary")
    if set(raw_probabilities) != set(_SIGNED_AXIS_MODES):
        missing = sorted(set(_SIGNED_AXIS_MODES) - set(raw_probabilities))
        unknown = sorted(set(raw_probabilities) - set(_SIGNED_AXIS_MODES))
        raise ValueError(
            "signed_axis_probabilities must contain exactly the supported modes; "
            f"missing={missing}, unknown={unknown}"
        )

    probabilities: list[float] = []
    for name in _SIGNED_AXIS_MODES:
        try:
            probability = float(raw_probabilities[name])
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"signed-axis probability {name!r} must be a finite number"
            ) from exc
        if not math.isfinite(probability) or probability < 0.0:
            raise ValueError(
                f"signed-axis probability {name!r} must be finite and non-negative"
            )
        probabilities.append(probability)
    probability_sum = sum(probabilities)
    if not math.isclose(probability_sum, 1.0, rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError(
            f"signed_axis_probabilities must sum to 1; got {probability_sum:.9g}"
        )

    if not isinstance(raw_ranges, dict):
        raise TypeError("signed_axis_ranges must be a dictionary")
    if set(raw_ranges) != set(_SIGNED_AXIS_RANGE_KEYS):
        missing = sorted(set(_SIGNED_AXIS_RANGE_KEYS) - set(raw_ranges))
        unknown = sorted(set(raw_ranges) - set(_SIGNED_AXIS_RANGE_KEYS))
        raise ValueError(
            "signed_axis_ranges must contain exactly the supported ranges; "
            f"missing={missing}, unknown={unknown}"
        )

    ranges: dict[str, tuple[float, float]] = {}
    for name in _SIGNED_AXIS_RANGE_KEYS:
        raw_range = raw_ranges[name]
        if not isinstance(raw_range, (tuple, list)) or len(raw_range) != 2:
            raise TypeError(f"signed-axis range {name!r} must contain two numbers")
        try:
            lower, upper = (float(value) for value in raw_range)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"signed-axis range {name!r} must contain finite numbers"
            ) from exc
        if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
            raise ValueError(
                f"signed-axis range {name!r} must be finite with lower <= upper"
            )
        positive = name in ("forward", "lateral_left", "yaw_left")
        if positive and lower <= 0.0:
            raise ValueError(f"signed-axis range {name!r} must be strictly positive")
        if not positive and upper >= 0.0:
            raise ValueError(f"signed-axis range {name!r} must be strictly negative")
        ranges[name] = (lower, upper)

    def validate_envelope(name: str, raw_envelope: object) -> tuple[float, float]:
        if not isinstance(raw_envelope, (tuple, list)) or len(raw_envelope) != 2:
            raise TypeError(f"signed-axis {name} envelope must contain two numbers")
        try:
            lower, upper = (float(value) for value in raw_envelope)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"signed-axis {name} envelope must contain finite numbers"
            ) from exc
        if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
            raise ValueError(
                f"signed-axis {name} envelope must be finite with lower <= upper"
            )
        return lower, upper

    linear_x_envelope = validate_envelope("linear-x", cfg.ranges.lin_vel_x)
    linear_y_envelope = validate_envelope("linear-y", cfg.ranges.lin_vel_y)
    moving_yaw_envelope = validate_envelope("moving-yaw", cfg.ranges.ang_vel_z)
    rotation_yaw_range = getattr(cfg, "rotation_env_ang_vel_range", None)
    rotation_yaw_envelope = (
        validate_envelope("rotation-yaw", rotation_yaw_range)
        if rotation_yaw_range is not None
        else moving_yaw_envelope
    )
    envelope_by_range = {
        "forward": linear_x_envelope,
        "backward": linear_x_envelope,
        "lateral_left": linear_y_envelope,
        "lateral_right": linear_y_envelope,
        "yaw_left": rotation_yaw_envelope,
        "yaw_right": rotation_yaw_envelope,
    }
    for name, bounds in ranges.items():
        envelope = envelope_by_range[name]
        if bounds[0] < envelope[0] or bounds[1] > envelope[1]:
            raise ValueError(
                f"signed-axis range {name!r}={bounds} escapes its envelope {envelope}"
            )

    # Mixed commands use the same explicit dead bands, intersected with the
    # ordinary moving-yaw envelope (pure yaw may use the wider rotation range).
    mixed_envelope_by_range = {
        "forward": linear_x_envelope,
        "backward": linear_x_envelope,
        "lateral_left": linear_y_envelope,
        "lateral_right": linear_y_envelope,
        "yaw_left": moving_yaw_envelope,
        "yaw_right": moving_yaw_envelope,
    }
    if probabilities[_SIGNED_AXIS_MODES.index("mixed")] > 0.0:
        for name, bounds in list(ranges.items()):
            envelope = mixed_envelope_by_range[name]
            intersection = (max(bounds[0], envelope[0]), min(bounds[1], envelope[1]))
            if intersection[0] > intersection[1]:
                raise ValueError(
                    f"signed-axis range {name!r}={bounds} has no intersection "
                    f"with the mixed-command envelope {envelope}"
                )
            ranges[f"mixed_{name}"] = intersection

    incompatible_fractions = {
        "rel_standing_envs": getattr(cfg, "rel_standing_envs", 0.0),
        "rel_forward_envs": getattr(cfg, "rel_forward_envs", 0.0),
        "rel_rotation_envs": getattr(cfg, "rel_rotation_envs", 0.0),
        "rel_heading_envs": getattr(cfg, "rel_heading_envs", 0.0),
        "rel_world_envs": getattr(cfg, "rel_world_envs", 0.0),
        "init_velocity_prob": getattr(cfg, "init_velocity_prob", 0.0),
    }
    nonzero = {
        name: value
        for name, value in incompatible_fractions.items()
        if not math.isclose(float(value), 0.0, rel_tol=0.0, abs_tol=0.0)
    }
    if nonzero:
        raise ValueError(
            "signed-axis sampling requires legacy/world/heading/initial-velocity "
            f"fractions to be zero; got {nonzero}"
        )

    return tuple(probability / probability_sum for probability in probabilities), ranges


class UniformVelocityCommandWithRotation(UniformVelocityCommand):
    """Extends UniformVelocityCommand with a `rel_rotation_envs` fraction.

    Rotation-only environments receive zero linear velocity and a non-zero angular 
    velocity in [`cfg.rotation_env_ang_vel_range[0]`, `cfg.rotation_env_ang_vel_range[1]`], 
    with an absolute value of at least `cfg.rotation_min_ang_vel`.
    """

    cfg: "UniformVelocityCommandWithRotationCfg"

    def __init__(self, cfg: "UniformVelocityCommandWithRotationCfg", env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.is_rotation_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        _validate_signed_axis_sampler_cfg(self.cfg)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        signed_axis_settings = _validate_signed_axis_sampler_cfg(self.cfg)
        if signed_axis_settings is not None:
            self._resample_signed_axis_command(env_ids, *signed_axis_settings)
            return

        super()._resample_command(env_ids)

        rel_rotation_envs = getattr(self.cfg, "rel_rotation_envs", 0.0)

        r = torch.empty(len(env_ids), device=self.device)
        self.is_rotation_env[env_ids] = r.uniform_(0.0, 1.0) <= rel_rotation_envs

        rot_ids = env_ids[self.is_rotation_env[env_ids]]
        if len(rot_ids) == 0:
            return

        self.vel_command_b[rot_ids, 0] = 0.0
        self.vel_command_b[rot_ids, 1] = 0.0

        # Sample angular velocity from the rotation-specific range if provided,
        # otherwise reuse what the parent sampled from cfg.ranges.ang_vel_z.
        if self.cfg.rotation_env_ang_vel_range is not None:
            ang = torch.empty(len(rot_ids), device=self.device).uniform_(
                *self.cfg.rotation_env_ang_vel_range
            )
        else:
            ang = self.vel_command_b[rot_ids, 2]

        # Ensure non-zero angular velocity.
        min_abs_ang = self.cfg.rotation_min_ang_vel
        too_small = ang.abs() < min_abs_ang
        if too_small.any():
            signs = torch.where(
                torch.rand(too_small.sum(), device=self.device) > 0.5,
                torch.ones(too_small.sum(), device=self.device),
                -torch.ones(too_small.sum(), device=self.device),
            )
            ang[too_small] = signs * min_abs_ang
        self.vel_command_b[rot_ids, 2] = ang

    def _resample_signed_axis_command(
        self,
        env_ids: torch.Tensor,
        probabilities: tuple[float, ...],
        ranges: dict[str, tuple[float, float]],
    ) -> None:
        """Sample one exclusive signed-axis mode for every requested environment."""

        if len(env_ids) == 0:
            return

        probability_tensor = torch.tensor(
            probabilities, dtype=torch.float32, device=self.device
        )
        cumulative = torch.cumsum(probability_tensor, dim=0)
        cumulative[-1] = 1.0
        draws = torch.rand(len(env_ids), device=self.device)
        mode_indices = torch.searchsorted(cumulative, draws, right=True)

        self.vel_command_b[env_ids] = 0.0
        self.is_standing_env[env_ids] = False
        self.is_forward_env[env_ids] = False
        self.is_heading_env[env_ids] = False
        self.is_world_env[env_ids] = False
        self.is_rotation_env[env_ids] = False

        mode_ids: dict[str, torch.Tensor] = {}
        for index, name in enumerate(_SIGNED_AXIS_MODES):
            mode_ids[name] = env_ids[mode_indices == index]

        standing_ids = mode_ids["standing"]
        self.is_standing_env[standing_ids] = True

        def sample_axis(ids: torch.Tensor, axis: int, range_name: str) -> None:
            if len(ids) > 0:
                self.vel_command_b[ids, axis] = torch.empty(
                    len(ids), device=self.device
                ).uniform_(*ranges[range_name])

        forward_ids = mode_ids["forward"]
        sample_axis(forward_ids, 0, "forward")
        self.is_forward_env[forward_ids] = True
        sample_axis(mode_ids["backward"], 0, "backward")
        sample_axis(mode_ids["lateral_left"], 1, "lateral_left")
        sample_axis(mode_ids["lateral_right"], 1, "lateral_right")

        yaw_left_ids = mode_ids["yaw_left"]
        yaw_right_ids = mode_ids["yaw_right"]
        sample_axis(yaw_left_ids, 2, "yaw_left")
        sample_axis(yaw_right_ids, 2, "yaw_right")
        self.is_rotation_env[yaw_left_ids] = True
        self.is_rotation_env[yaw_right_ids] = True

        mixed_ids = mode_ids["mixed"]
        if len(mixed_ids) > 0:
            mixed_axes = (
                (0, "forward", "backward"),
                (1, "lateral_left", "lateral_right"),
                (2, "yaw_left", "yaw_right"),
            )
            for axis, positive_name, negative_name in mixed_axes:
                positive = torch.rand(len(mixed_ids), device=self.device) < 0.5
                sample_axis(mixed_ids[positive], axis, f"mixed_{positive_name}")
                sample_axis(mixed_ids[~positive], axis, f"mixed_{negative_name}")

        # World-frame commands are disallowed in this opt-in mode.  Keep the
        # storage coherent for diagnostics and any future zero-world transition.
        self.vel_command_w[env_ids] = self.vel_command_b[env_ids]


@dataclass(kw_only=True)
class UniformVelocityCommandWithRotationCfg(UniformVelocityCommandCfg):
    """Configuration for UniformVelocityCommandWithRotation."""

    rel_rotation_envs: float = 0.0
    """Fraction of environments that receive pure-rotation commands
    (zero linear velocity, non-zero angular velocity)."""

    rotation_min_ang_vel: float = 0.3
    """Minimum absolute angular velocity assigned to rotation-only environments."""

    rotation_env_ang_vel_range: tuple[float, float] | None = None
    """Angular velocity range for rotation-only environments.
    If None, uses cfg.ranges.ang_vel_z (same range as normal environments)."""

    signed_axis_probabilities: dict[str, float] | None = None
    """Opt-in probabilities for exclusive signed-axis command modes.

    When set, this must contain exactly ``standing``, ``forward``, ``backward``,
    ``lateral_left``, ``lateral_right``, ``yaw_left``, ``yaw_right`` and
    ``mixed`` and sum to one.  All legacy mode fractions, world/heading fractions
    and ``init_velocity_prob`` must be zero.  ``None`` preserves legacy sampling.
    """

    signed_axis_ranges: dict[str, tuple[float, float]] | None = None
    """Strictly signed dead-band ranges for the six non-standing axis modes.

    Required keys are the six non-standing, non-mixed mode names.  Mixed commands
    independently choose either sign on every axis from these ranges; their yaw
    range is intersected with ``ranges.ang_vel_z``, while pure-yaw modes may use
    the wider ``rotation_env_ang_vel_range``.
    """

    def build(self, env: ManagerBasedRlEnv) -> UniformVelocityCommandWithRotation:
        return UniformVelocityCommandWithRotation(self, env)


class FootTargetCommand(CommandTerm):
    """Live per-env target offset (dx, dy, dz) for each foot, relative to that foot's
    own position measured right after reset (i.e. the robot's default/home stance),
    expressed in the trunk frame with the HOME lean rotated out
    (``cfg.trunk_pitch``; level at HOME, x forward, y left, z up).

    This drives the "whole-body tracking" leg behavior: trained alongside the existing
    velocity command so ONE policy learns both walking and holding a commanded foot
    pose (including single-leg stances), rather than switching to a kinematic-IK leg
    controller when idle — kinematic IK can't provide the continuous balance
    correction a single-leg stance needs.

    Most sampled episodes keep both targets at zero offset (normal stance). A fraction
    lift one foot (randomly chosen) to train single-support balance.
    """

    cfg: FootTargetCommandCfg

    def __init__(self, cfg: FootTargetCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.entity_name]

        self._foot_asset_cfg = _resolve_ordered_site_cfg(
            env.scene,
            entity_name=cfg.entity_name,
            site_names=cfg.foot_site_names,
            label="FootTargetCommand",
        )

        # Offset target (dx, dy, dz) per env, per foot (left, right), in the
        # HOME-levelled trunk frame (see current_foot_pos_b).
        self.foot_target_offset_b = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.is_single_support_env = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.lifted_foot_idx = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        # Per-episode reference ("zero offset") foot position, snapshotted at reset time
        # (see _resample_command) since it depends on wherever reset_robot_joints landed.
        self._default_foot_pos_b = torch.zeros(self.num_envs, 2, 3, device=self.device)

        self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.foot_target_offset_b.view(self.num_envs, -1)

    def current_foot_pos_b(self) -> torch.Tensor:
        """Live foot positions (left, right) in the HOME-levelled trunk frame
        (``_home_levelled_quat`` with ``cfg.trunk_pitch``; 0 = the trunk frame).
        Shape (N, 2, 3)."""
        trunk_pos_w = self.robot.data.root_link_pos_w
        trunk_quat_w = _home_levelled_quat(
            self.robot.data.root_link_quat_w, self.cfg.trunk_pitch
        )
        foot_pos_w = self.robot.data.site_pos_w[:, self._foot_asset_cfg.site_ids, :]
        num_feet = foot_pos_w.shape[1]
        pos_b, _ = subtract_frame_transforms(
            trunk_pos_w[:, None, :].repeat(1, num_feet, 1).reshape(-1, 3),
            trunk_quat_w[:, None, :].repeat(1, num_feet, 1).reshape(-1, 4),
            foot_pos_w.reshape(-1, 3),
        )
        return pos_b.view(self.num_envs, num_feet, 3)

    def _update_metrics(self) -> None:
        error = torch.sum(
            torch.square(
                self.current_foot_pos_b()
                - self._default_foot_pos_b
                - self.foot_target_offset_b
            ),
            dim=-1,
        )
        self.metrics["error_pos"] += error.mean(-1)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        # Snapshot each foot's current (post-reset, default-stance) position as this
        # episode's zero-offset reference.
        if not hasattr(self, "_default_foot_pos_b"):
            self._default_foot_pos_b = torch.zeros(
                self.num_envs, 2, 3, device=self.device
            )
        self._default_foot_pos_b[env_ids] = self.current_foot_pos_b()[env_ids]

        self.foot_target_offset_b[env_ids] = 0.0

        r = torch.empty(len(env_ids), device=self.device)
        self.is_single_support_env[env_ids] = (
            r.uniform_(0.0, 1.0) <= self.cfg.rel_single_support_envs
        )

        support_ids = env_ids[self.is_single_support_env[env_ids]]
        if len(support_ids) == 0:
            return

        r2 = torch.empty(len(support_ids), device=self.device)
        self.lifted_foot_idx[support_ids] = (r2.uniform_(0.0, 1.0) < 0.5).long()

        dx = torch.empty(len(support_ids), device=self.device).uniform_(
            *self.cfg.reach_xy_range
        )
        dy = torch.empty(len(support_ids), device=self.device).uniform_(
            *self.cfg.reach_xy_range
        )
        dz = torch.empty(len(support_ids), device=self.device).uniform_(
            *self.cfg.lift_height_range
        )

        offsets = torch.stack([dx, dy, dz], dim=-1)  # (n, 3)
        self.foot_target_offset_b[support_ids, self.lifted_foot_idx[support_ids], :] = (
            offsets
        )

    def _update_command(self) -> None:
        pass


@dataclass(kw_only=True)
class FootTargetCommandCfg(CommandTermCfg):
    """Configuration for FootTargetCommand."""

    entity_name: str = "robot"
    foot_site_names: tuple[str, str] = ("left_foot", "right_foot")
    rel_single_support_envs: float = 0.3
    """Fraction of environments that get a single-leg-stance target (one foot lifted)."""
    lift_height_range: tuple[float, float] = (0.01, 0.05)
    reach_xy_range: tuple[float, float] = (-0.03, 0.03)
    trunk_pitch: float = 0.0
    """HOME forward trunk lean (rad).  Offsets are expressed in the trunk frame
    with this lean rotated out, R_trunk * R_y(-trunk_pitch), which is level at
    HOME.  0 keeps the plain trunk frame."""

    def build(self, env: ManagerBasedRlEnv) -> FootTargetCommand:
        return FootTargetCommand(self, env)


class HandTargetCommand(CommandTerm):
    """Live per-env, per-hand target offset (dx, dy, dz), relative to that hand's own
    position measured right after reset, in the HOME-levelled trunk frame
    (``cfg.trunk_pitch``; see current_hand_pos_b).

    Activation is PER HAND (independent left/right), matching the real controller UX:
    each hand's tracking is meant to be enabled by that hand's own controller trigger,
    not tied to walking state (see foot_target_tracking_error_exp's velocity-based fade
    — hands have no such fade, they track whenever that hand is "active").

    A hand with no active target (``is_active`` False, e.g. trigger not held / that
    controller not connected) contributes nothing to the tracking reward at all (not
    "pulled to zero offset"), so that arm is free to move however helps gait/balance,
    rather than being locked toward a rest position it was never asked to hold.
    ``command`` exposes ``is_active`` (one flag per hand) alongside the offsets so the
    actor can tell "holding position zero" and "not tracking at all" apart.

    Active training targets are never sampled from a Cartesian cube.  A Microban
    shoulder-pitch/roll/elbow tuple is sampled uniformly inside the audited joint
    box and converted to an XYZ offset with the exact robot.xml kinematic chain;
    tuples whose offset leaves the receiver's per-axis hand box (+-0.8 * 0.08 m)
    are rejected and redrawn.
    """

    cfg: HandTargetCommandCfg

    def __init__(self, cfg: HandTargetCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.entity_name]

        self._hand_asset_cfg = _resolve_ordered_site_cfg(
            env.scene,
            entity_name=cfg.entity_name,
            site_names=cfg.hand_site_names,
            label="HandTargetCommand",
        )

        self.hand_target_offset_b = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self._default_hand_pos_b = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.is_active = torch.zeros(
            self.num_envs, 2, dtype=torch.bool, device=self.device
        )
        self.sampled_arm_joint_pos_rad = (
            torch.tensor(
                MICROBAN_ARM_HOME_JOINT_RAD,
                dtype=self.hand_target_offset_b.dtype,
                device=self.device,
            )
            .expand(self.num_envs, -1, -1)
            .clone()
        )

        self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return torch.cat(
            [self.hand_target_offset_b.view(self.num_envs, -1), self.is_active.float()],
            dim=-1,
        )

    def current_hand_pos_b(self) -> torch.Tensor:
        """Live hand positions (left, right) in the HOME-levelled trunk frame
        (``_home_levelled_quat`` with ``cfg.trunk_pitch``; 0 = the trunk frame).
        Shape (N, 2, 3)."""
        trunk_pos_w = self.robot.data.root_link_pos_w
        trunk_quat_w = _home_levelled_quat(
            self.robot.data.root_link_quat_w, self.cfg.trunk_pitch
        )
        hand_pos_w = self.robot.data.site_pos_w[:, self._hand_asset_cfg.site_ids, :]
        num_hands = hand_pos_w.shape[1]
        pos_b, _ = subtract_frame_transforms(
            trunk_pos_w[:, None, :].repeat(1, num_hands, 1).reshape(-1, 3),
            trunk_quat_w[:, None, :].repeat(1, num_hands, 1).reshape(-1, 4),
            hand_pos_w.reshape(-1, 3),
        )
        return pos_b.view(self.num_envs, num_hands, 3)

    def _update_metrics(self) -> None:
        error = torch.sum(
            torch.square(
                self.current_hand_pos_b()
                - self._default_hand_pos_b
                - self.hand_target_offset_b
            ),
            dim=-1,
        )
        active = self.is_active.float()
        self.metrics["error_pos"] += (error * active).sum(-1) / active.sum(-1).clamp(
            min=1.0
        )

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        self._default_hand_pos_b[env_ids] = self.current_hand_pos_b()[env_ids]

        r = torch.empty(len(env_ids), 2, device=self.device)
        self.is_active[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_active

        # FK offsets are trunk-frame; rotate them by R_y(trunk_pitch) into the
        # HOME-levelled frame the targets and current_hand_pos_b use.  Joint
        # samples whose target leaves the robot receiver's +-64 mm hand box
        # are redrawn, so every target is reachable and deliverable.
        sampled_joints, offsets = sample_microban_reachable_hand_targets(
            self.is_active[env_ids],
            dtype=self.hand_target_offset_b.dtype,
            trunk_pitch=self.cfg.trunk_pitch,
        )
        self.sampled_arm_joint_pos_rad[env_ids] = sampled_joints
        self.hand_target_offset_b[env_ids] = offsets

    def _update_command(self) -> None:
        pass


@dataclass(kw_only=True)
class HandTargetCommandCfg(CommandTermCfg):
    """Configuration for joint-box/FK reachable Microban hand targets."""

    entity_name: str = "robot"
    hand_site_names: tuple[str, str] = ("left_hand", "right_hand")
    rel_active: float = 0.7
    """Per-hand probability of being active at each resample (independent left/right,
    matching each controller's own trigger). Inactive hands contribute nothing to the
    tracking reward, so the policy learns that arm is free to move naturally."""
    trunk_pitch: float = 0.0
    """HOME forward trunk lean (rad).  Offsets are expressed in the trunk frame
    with this lean rotated out, R_trunk * R_y(-trunk_pitch), and the FK samples
    are rotated by R_y(trunk_pitch) into it.  0 keeps the plain trunk frame."""

    def build(self, env: ManagerBasedRlEnv) -> HandTargetCommand:
        return HandTargetCommand(self, env)


########################## OBSERVATIONS ############################


def foot_target_offset_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Flattened (left_dx, left_dy, left_dz, right_dx, right_dy, right_dz) target foot
    offset, in the command's HOME-levelled trunk frame. Lets the actor see what
    leg-tracking target (if any) is
    currently commanded, alongside the velocity command."""
    command: FootTargetCommand = env.command_manager.get_term(command_name)
    return command.command


def hand_target_offset_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Flattened (left_dx, left_dy, left_dz, right_dx, right_dy, right_dz) target hand
    offset, in the command's HOME-levelled trunk frame."""
    command: HandTargetCommand = env.command_manager.get_term(command_name)
    return command.command


############################ REWARDS ##############################

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class upright:
    """Reward for keeping the base at a target pitch orientation.

    Penalizes deviation from a given pitch angle (in radians) rather than
    always rewarding a perfectly vertical posture.

    Args:
        std: Standard deviation of the Gaussian kernel (controls reward sharpness).
        pitch: Target pitch angle in radians. 0.0 = perfectly upright.
               Positive values mean leaning forward.
        asset_cfg: Scene entity configuration for the robot body to track.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        pass

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        std: float,
        pitch: float = 0.0,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]

        if asset_cfg.body_ids:
            body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
        else:
            body_quat_w = asset.data.root_link_quat_w

        gravity_w = asset.data.gravity_vec_w
        projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)

        # Normalize to unit vector so the error is scale-independent.
        gravity_norm = projected_gravity_b.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        projected_gravity_b_unit = projected_gravity_b / gravity_norm

        # At pitch angle θ, the normalised gravity unit vector in body frame has
        # x = sin(θ), y = 0 (assuming flat ground, no roll, no terrain slope).
        target_gx = math.sin(pitch)
        xy_error = (
            torch.square(projected_gravity_b_unit[:, 0] - target_gx)
            + torch.square(projected_gravity_b_unit[:, 1])
        )
        return torch.exp(-xy_error / std**2)

    def reset(self, env_ids: torch.Tensor) -> None:
        del env_ids  # Unused.


def _home_levelled_quat(quat_w: torch.Tensor, trunk_pitch: float) -> torch.Tensor:
    """The trunk frame with the HOME forward lean taken out.

    At HOME (trunk pitched ``trunk_pitch`` forward) this frame is level and
    faces the robot's heading, so velocities expressed in it read like the
    body-frame velocities of an upright-trunk HOME.
    """

    if trunk_pitch == 0.0:
        return quat_w
    half = -0.5 * trunk_pitch
    unpitch = torch.tensor(
        (math.cos(half), 0.0, math.sin(half), 0.0), device=quat_w.device, dtype=quat_w.dtype
    ).expand_as(quat_w)
    return quat_mul(quat_w, unpitch)


def reset_root_state_uniform_world_yaw(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]] | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """mjlab's reset_root_state_uniform, with the yaw turned about world z.

    mjlab composes default_rot * R(roll, pitch, yaw), i.e. it turns the yaw
    about the DEFAULT body's z axis. With an upright HOME that is the world
    z axis; with a HOME trunk leaning forward it is the tilted trunk axis, so
    a reset yawed by 180 deg would lean the robot backward relative to its
    heading with both soles tipped. Here orientation = R_z(yaw) * default_rot
    * R(roll, pitch): the roll/pitch noise is applied in the HOME trunk frame
    and the whole HOME stance is then turned about world z, so every yaw
    keeps the soles flat and the lean forward. With an identity default it
    equals mjlab's term up to the order of yaw and roll/pitch noise.
    """

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    asset: Entity = env.scene[asset_cfg.name]
    keys = ("x", "y", "z", "roll", "pitch", "yaw")
    ranges = torch.tensor([pose_range.get(key, (0.0, 0.0)) for key in keys], device=env.device)
    samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device)
    root_states = asset.data.default_root_state[env_ids].clone()
    positions = root_states[:, 0:3] + samples[:, 0:3] + env.scene.env_origins[env_ids]
    zeros = torch.zeros_like(samples[:, 0])
    tilt = quat_from_euler_xyz(samples[:, 3], samples[:, 4], zeros)
    heading = quat_from_euler_xyz(zeros, zeros, samples[:, 5])
    orientations = quat_mul(heading, quat_mul(root_states[:, 3:7], tilt))
    velocity_range = velocity_range or {}
    ranges = torch.tensor([velocity_range.get(key, (0.0, 0.0)) for key in keys], device=env.device)
    velocities = root_states[:, 7:13] + sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device
    )
    asset.write_root_link_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)


def feet_distance_penalty(
    env: ManagerBasedRlEnv,
    min_dist: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize the feet getting too close to each other in the horizontal plane.

    Discourages the robot from stepping on its own feet by adding a smooth,
    anticipatory cost as soon as the horizontal distance between the two foot
    sites drops below ``min_dist``. Above the threshold the cost is zero.

    Returns ``clamp(min_dist - d, min=0)`` per env (use with a negative weight),
    where ``d`` is the horizontal (xy) distance between the two foot sites.

    Args:
        min_dist: Minimum desired horizontal distance between feet, in meters.
        asset_cfg: Scene entity configuration whose ``site_names`` select exactly
            the two foot sites.
    """
    asset: Entity = env.scene[asset_cfg.name]
    foot_pos_xy = asset.data.site_pos_w[:, asset_cfg.site_ids, :2]  # [B, 2, 2]
    dist = torch.norm(foot_pos_xy[:, 0] - foot_pos_xy[:, 1], dim=-1)  # [B]
    return torch.clamp(min_dist - dist, min=0.0)


def foot_target_tracking_error_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    velocity_command_name: str = "twist",
    velocity_fade_range: tuple[float, float] = (0.0, 0.15),
) -> torch.Tensor:
    """Reward for matching the commanded per-foot target offset (FootTargetCommand),
    faded out smoothly as the velocity command grows so legs prioritize walking over
    holding a static foot target once actually moving. This is a continuous weight
    (no hard cutoff/mode switch): at velocity_fade_range[0] and below, full weight;
    at velocity_fade_range[1] and above, zero; linear in between.
    """
    command: FootTargetCommand = env.command_manager.get_term(command_name)
    error = torch.sum(
        torch.square(
            command.current_foot_pos_b()
            - command._default_foot_pos_b
            - command.foot_target_offset_b
        ),
        dim=-1,
    ).mean(-1)
    tracking_reward = torch.exp(-error / std**2)

    velocity_command = env.command_manager.get_command(velocity_command_name)
    speed = torch.norm(velocity_command[:, :2], dim=-1) + torch.abs(
        velocity_command[:, 2]
    )
    lo, hi = velocity_fade_range
    fade = 1.0 - torch.clamp((speed - lo) / (hi - lo), 0.0, 1.0)

    return tracking_reward * fade


def hand_target_tracking_error_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
) -> torch.Tensor:
    """Reward for matching the commanded per-hand target offset (HandTargetCommand).

    No velocity-based fade: hand tracking is meant to be on whenever that hand is
    active (unlike foot_target_tracking_error_exp), since the arms don't need to give
    way to locomotion the way legs do. Averaged over active hands only — an inactive
    hand contributes nothing (positive or negative) so it's free to move naturally; an
    env with neither hand active gets zero from this term entirely.
    """
    command: HandTargetCommand = env.command_manager.get_term(command_name)
    error = torch.sum(
        torch.square(
            command.current_hand_pos_b()
            - command._default_hand_pos_b
            - command.hand_target_offset_b
        ),
        dim=-1,
    )
    per_hand_reward = torch.exp(-error / std**2)
    active = command.is_active.float()
    return (per_hand_reward * active).sum(-1) / active.sum(-1).clamp(min=1.0)


def no_stepping_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str = "twist",
    command_threshold: float = 0.01,
    foot_target_command_name: str | None = None,
) -> torch.Tensor:
    """Penalize feet in the air when the commanded speed is below threshold.

    Discourages marching in place when the robot should stand still.
    When ``foot_target_command_name`` is provided, rows with an explicitly
    active single- or two-foot target are exempt: lifting a commanded foot must
    not simultaneously incur the stationary no-stepping cost.

    Returns the count of airborne feet per environment (use with a negative weight).
    """
    command = env.command_manager.get_command(command_name)  # (N, 3)
    cmd_speed = torch.norm(command[:, :2], dim=-1) + torch.abs(command[:, 2])
    below_threshold = cmd_speed < command_threshold
    if foot_target_command_name is not None:
        foot_target = env.command_manager.get_term(foot_target_command_name)
        single_support = getattr(foot_target, "is_single_support_env", None)
        if not isinstance(single_support, torch.Tensor) or single_support.shape != (
            env.num_envs,
        ):
            raise ValueError(
                "foot target command must expose is_single_support_env with "
                "shape (num_envs,)"
            )
        active_foot_target = single_support.bool()
        both_feet = getattr(foot_target, "is_both_feet_env", None)
        if both_feet is not None:
            if not isinstance(both_feet, torch.Tensor) or both_feet.shape != (
                env.num_envs,
            ):
                raise ValueError(
                    "foot target command is_both_feet_env must have shape (num_envs,)"
                )
            active_foot_target = active_foot_target | both_feet.bool()
        below_threshold &= ~active_foot_target

    sensor = env.scene.sensors[sensor_name]
    found = sensor.data.found  # (N, num_feet) or (N, num_feet, num_slots)
    if found.dim() == 3:
        found = found.any(dim=-1)  # (N, num_feet)
    in_air = ~found.bool()

    return in_air.float().sum(dim=-1) * below_threshold.float()


########################## CURRICULUM #############################


class reward_based_staged_curriculum:
    """
    Curriculum based on stages ending while a reward component gets its mean 
    episode reward accross all environments above a threshold.

    Stage definitions example:
    stages = [
        {
            "name": "stage 1",
            "reward_term_name": "term_name",
            "threshold": 0.5,
            "apply": lambda env: env.reward_manager.get_term_cfg("term_name").weight = 1.0,
        },
        ...
    ]
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
        self.rewards = torch.zeros(env.num_envs, device=env.device)
        self.current_stage = 0
        self.stage_first_step = 0
        
    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor,
        stages: list[dict],
    ) -> dict[str, torch.Tensor]:
        self.rewards[env_ids] = (
            env.reward_manager._episode_sums[stages[self.current_stage]["reward_term_name"]][env_ids]
            / env.max_episode_length_s
        )
        mean_reward = self.rewards.mean().item()

        if (
            self.current_stage < len(stages)
            and mean_reward >= stages[self.current_stage]["threshold"]
            and env.common_step_counter >= self.stage_first_step + 100 * 24
        ):
            stage = stages[self.current_stage]
            print(
                f"Curriculum stage {self.current_stage + 1}: {stage['name']} at step {env.common_step_counter} (mean episode reward: {mean_reward:.4f})"
            )
            stage["apply"](env)
            self.current_stage += 1
            self.stage_first_step = env.common_step_counter
            self.rewards.zero_()  # Reset rewards to avoid immediately triggering the next stage

        return {"stage": self.current_stage}


        # step right after a reset is harmless, since is_standing gates the whole
        # term to 0 right at that moment anyway (a just-reset env starts fallen).


