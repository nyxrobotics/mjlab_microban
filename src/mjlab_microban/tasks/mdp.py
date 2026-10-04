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
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp.velocity_command import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)
from mjlab.utils.lab_api.math import (
    quat_apply,
    quat_apply_inverse,
    quat_from_euler_xyz,
    quat_mul,
    sample_uniform,
    subtract_frame_transforms,
    yaw_quat,
)
from mjlab.utils.lab_api.string import resolve_matching_names_values

from mjlab_microban.robot.microban_hand_fk import (
    MICROBAN_ARM_HOME_JOINT_RAD,
    sample_microban_reachable_hand_targets,
)

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

    cfg: UniformVelocityCommandWithRotationCfg

    def __init__(
        self, cfg: UniformVelocityCommandWithRotationCfg, env: ManagerBasedRlEnv
    ):
        super().__init__(cfg, env)
        self.is_rotation_env = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
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
    correction a single-leg stance needs (see microban_teleop/docs/retargeting_research.md).

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
    box and converted to an XYZ offset with the exact robot.xml kinematic chain.
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
        # HOME-levelled frame the targets and current_hand_pos_b use.
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
            body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(
                1
            )
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
        xy_error = torch.square(
            projected_gravity_b_unit[:, 0] - target_gx
        ) + torch.square(projected_gravity_b_unit[:, 1])
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


def home_levelled_root_lin_vel_b(
    env: ManagerBasedRlEnv, trunk_pitch: float, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Root linear velocity in the HOME-levelled trunk frame (see above)."""

    asset: Entity = env.scene[asset_cfg.name]
    if trunk_pitch == 0.0:
        return asset.data.root_link_lin_vel_b
    frame = _home_levelled_quat(asset.data.root_link_quat_w, trunk_pitch)
    return quat_apply_inverse(frame, asset.data.root_link_lin_vel_w)


def home_levelled_root_ang_vel_b(
    env: ManagerBasedRlEnv, trunk_pitch: float, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Root angular velocity in the HOME-levelled trunk frame (see above)."""

    asset: Entity = env.scene[asset_cfg.name]
    if trunk_pitch == 0.0:
        return asset.data.root_link_ang_vel_b
    frame = _home_levelled_quat(asset.data.root_link_quat_w, trunk_pitch)
    return quat_apply_inverse(frame, asset.data.root_link_ang_vel_w)


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


def extreme_joint_velocity(
    env: ManagerBasedRlEnv,
    max_joint_vel: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Terminate envs whose joint velocity has run away to an unphysical (but still
    finite) magnitude, before it can corrupt reward accumulation / the value function.

    ``nan_detection`` alone isn't enough: a state can spend several steps at an
    enormous-but-finite velocity (observed directly: body_ang_vel reward reaching
    ~1e12 in early hold_airborne training) before actually overflowing to NaN/Inf,
    and by then the episode return and value-loss for that env are already ruined —
    poisoning the batch mean that PPO's gradient step uses, which is how a single
    still-diverging env corrupts every other env's policy via one bad update. No real
    servo on this robot exceeds ~10 rad/s; 50 rad/s is a generous, clearly-diverging
    threshold rather than a tuned operating limit.
    """
    asset: Entity = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return joint_vel.abs().amax(dim=-1) > max_joint_vel


_TRUNK_TO_HEAD_OFFSET = 0.07324


def _head_height(env: ManagerBasedRlEnv, head_asset_cfg: SceneEntityCfg) -> torch.Tensor:
    # Despite the head_asset_cfg name, this no longer reads the actual "head" body
    # through it — it only uses .name to resolve the robot entity, then computes a
    # virtual point above the TRUNK's own center of mass/orientation (fixed offset,
    # matching the standing pose's own trunk-to-head distance). The real head body's
    # own world Z can read as "tall" even upside-down/inverted if the neck happens to
    # bend the right way; a point fixed relative to the trunk's own up-axis cannot.
    asset: Entity = env.scene[head_asset_cfg.name]
    com_pos_w = asset.data.root_com_pos_w
    com_quat_w = asset.data.root_com_quat_w
    local_up = torch.zeros_like(com_pos_w)
    local_up[:, 2] = _TRUNK_TO_HEAD_OFFSET
    virtual_head_pos_w = com_pos_w + quat_apply(com_quat_w, local_up)
    return virtual_head_pos_w[:, 2]


def _standing_gate(height: torch.Tensor, height_threshold: float) -> torch.Tensor:
    """Hard 0/1 standing gate for the post-standing rewards.

    A soft sigmoid version (softness 0.03 m) was measured paying these
    terms 37% of full weight while propped on one forearm at 0.205 m head
    height -- i.e. rewarding the robot for holding still in the exact
    local optimum it could not leave. Hard gates cannot leak that way.
    """
    return (height > height_threshold).float()


def standing_bonus(
    env: ManagerBasedRlEnv,
    height_threshold: float,
    target_height: float,
    head_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Dense reward active only once the robot is near-standing (head height above
    height_threshold), rewarding staying there — and, above that threshold,
    continuing to scale with how much of the way to target_height (the TRUE
    standing height) it's actually reached, rather than paying flat full credit
    the instant it merely crosses the threshold.

    HoST's "post-task" mechanism (arXiv:2502.08378): since this only pays out once
    standing is reached, reaching it earlier and holding it accumulates strictly more
    of it over a fixed-length episode. This term is a targeted "finish the job"
    incentive that a dense height reward alone doesn't provide: partial credit for
    progress isn't the same as a payoff specifically for completing it.

    The continued scaling above threshold (rather than a flat 1.0) was added after
    a real plateau: training stalled with head height sitting almost exactly AT
    height_threshold for thousands of iterations. Mechanism (confirmed, not just
    suspected): this reward and standing_pose both gated flat/binary on the same
    threshold, so once crossed there was no further gradient from either to keep
    pushing toward target_height, while standing_torque's effort cost keeps rising
    the higher (and more extended) the stance gets — net incentive was to stop
    right at the minimum height that still banks the bonuses, not the true target.

    Previously also multiplied by a separate trunk-upright reward (checking
    projected gravity against a target std) on top of the height gate. Dropped: once
    height_threshold is set relative to the TRUE standing head height (see
    HEAD_STANDING_HEIGHT in microban_getup_env_cfg.py — this used to be a lower,
    empirically-lowered value a seated posture could also satisfy, which is exactly
    why the separate upright check seemed necessary), reaching it is only physically
    possible while upright — a seated, kneeling, or inverted-but-elevated pose cannot
    reach the head that high.
    """
    height = _head_height(env, head_asset_cfg)
    is_standing = height > height_threshold
    scale = torch.clamp(height / target_height, min=0.0, max=1.0)
    return torch.where(is_standing, scale, torch.zeros_like(scale))


def standing_torque_penalty(
    env: ManagerBasedRlEnv,
    height_threshold: float,
    head_asset_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize total and peak joint torque, but only once (nearly) standing.

    A more principled alternative to penalizing deviation from a specific default
    pose: encourages settling into whichever stance costs the least continuous
    effort to hold, rather than any height/upright/on-feet-satisfying configuration
    regardless of torque cost (e.g. a wide-splayed stance with heavily tilted ankles
    needs much more holding torque than a naturally balanced one). Gated like
    standing_bonus — during the actual recovery motion large torques are necessary,
    so this only applies once actually up.

    Returns total |torque| + peak |torque| (single worst-loaded motor) — both matter:
    total for overall effort/heat, peak for any one servo's real current/torque limit.
    """
    asset: Entity = env.scene[asset_cfg.name]
    height = _head_height(env, head_asset_cfg)
    torque = torch.abs(asset.data.actuator_force[:, asset_cfg.actuator_ids])
    total = torch.sum(torque, dim=-1)
    peak = torch.amax(torque, dim=-1)
    is_standing = height > height_threshold
    return torch.where(is_standing, total + peak, torch.zeros_like(total))


def standing_stability_reward(
    env: ManagerBasedRlEnv,
    height_threshold: float,
    head_asset_cfg: SceneEntityCfg,
    lin_vel_std: float,
    ang_vel_std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward LOW measured trunk linear+angular velocity, but only once standing.

    HoST (arXiv:2502.08378) names this exact failure mode directly: a policy
    that reaches standing height via fast/ballistic motion, with nothing
    penalizing HOW it arrived, is still carrying real momentum right at the
    height threshold and falls back over almost immediately after -- their
    own ablation shows this specific failure (reach height, don't sustain it)
    is what a dedicated post-standing "arrive and stay calm" term fixes,
    distinct from a height-based sustain bonus like standing_bonus (which
    only cares THAT height is maintained, not how calmly it was reached).
    Nothing else in this reward set penalizes MEASURED (not commanded-target)
    trunk velocity: action_rate_l2/home_stillness both act on the commanded
    target, which can already be smooth while the actual body is still
    rocking or mid-fall.

    Measured directly (16-env rollout, iteration 10500 of a from-scratch
    run): mean max head height 124% of target_height (genuine overshoot,
    real upward momentum still to bleed off at the moment of crossing) but
    only 1/16 envs still standing a second later -- exactly HoST's diagnosed
    shape. lin_vel_std/ang_vel_std are a reasoned starting guess (this
    robot's scale, not yet a measured calibration the way most other
    thresholds in this file are) -- watch whether this saturates near 0 (too
    strict, no gradient) or near 1 (too loose, no signal) on the first run
    and retune from there.

    Hard-gated with _standing_gate (see that helper for why not soft).
    """
    asset: Entity = env.scene[asset_cfg.name]
    height = _head_height(env, head_asset_cfg)
    gate = _standing_gate(height, height_threshold)
    lin_speed = torch.linalg.norm(asset.data.root_com_lin_vel_w, dim=-1)
    ang_speed = torch.linalg.norm(asset.data.root_com_ang_vel_w, dim=-1)
    stability = torch.exp(-((lin_speed / lin_vel_std) ** 2)) * torch.exp(
        -((ang_speed / ang_vel_std) ** 2)
    )
    return gate * stability


def upright_balance_reward(
    env: ManagerBasedRlEnv,
    height_threshold: float,
    head_asset_cfg: SceneEntityCfg,
    tilt_std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    pitch: float = 0.0,
) -> torch.Tensor:
    """Reward LOW trunk tilt (from projected gravity), but only once standing.

    Added after a direct sanity-check experiment: start the robot EXACTLY at
    its true standing pose and command it to hold EXACTLY that fixed joint
    configuration (no learned policy, no correction) for several seconds.
    Result: holds for ~1s (tilt drifting slowly, ~0.1deg -> ~3deg, joint
    tracking error staying under a few degrees -- not an actuator-gain
    problem), then catastrophically topples (47deg tilt by 2s, 92deg -- fully
    fallen -- by 4s). Standing upright is an inherently unstable equilibrium
    here, like any real inverted-pendulum-style biped: it requires
    CONTINUOUS ACTIVE CORRECTION, not just reaching/holding a fixed target
    pose. Neither standing_bonus (height) nor standing_stability_reward
    (velocity) catches the early drift phase: in that same experiment, head
    height barely moved until tilt already exceeded ~40 degrees -- height is
    a real but LATE, insensitive proxy for balance, and velocity picks up
    only once already properly falling. Nothing else in this reward set
    rewards low tilt as a dense, continuously-available signal from the
    start of the standing phase.

    Reuses projected_gravity_b (the same quantity the actor already
    observes) rather than introducing a separate tilt computation: at
    perfectly upright this is (0, 0, -1), so its horizontal-plane norm is
    exactly 0 and grows smoothly with tilt angle (sin of the tilt angle, for
    small-to-moderate tilt) -- the same signal the policy could in principle
    already use to self-correct, just not yet rewarded for acting on.
    tilt_std is a reasoned starting guess (not yet a measured calibration):
    watch for saturation near 0 or 1 on the first run.

    Hard-gated with _standing_gate (see that helper for why not soft).

    ``pitch`` is the trunk's forward lean at HOME (rad, positive = forward):
    the reward peaks where the projected gravity equals its HOME value
    (sin(pitch), 0, -cos(pitch)) instead of (0, 0, -1), so a HOME with a
    leaning trunk is not pulled upright.
    """
    asset: Entity = env.scene[asset_cfg.name]
    height = _head_height(env, head_asset_cfg)
    gate = _standing_gate(height, height_threshold)
    gravity_xy = asset.data.projected_gravity_b[:, :2]
    target_xy = torch.tensor((math.sin(pitch), 0.0), device=gravity_xy.device, dtype=gravity_xy.dtype)
    tilt = torch.linalg.norm(gravity_xy - target_xy, dim=-1)
    balance = torch.exp(-((tilt / tilt_std) ** 2))
    return gate * balance


def feet_stance_reward(
    env: ManagerBasedRlEnv,
    height_threshold: float,
    head_asset_cfg: SceneEntityCfg,
    axis: str,
    target: float,
    scale: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward one axis of a HOME-like stance, once standing.

    Measures the two foot bodies' separation in the trunk's heading frame
    along ``axis`` ("lateral" or "fore_aft"): error = | |separation| - target |
    (metres), reward = exp(-error / scale). One term per axis, on purpose: a
    single summed-error term let a policy narrow its stance from 20 to 12 cm
    by staggering one foot forward (7.5 -> 13 cm), barely changing the sum.
    The L1-exponential kernel keeps a usable gradient from far off. Hard-gated
    with _standing_gate. asset_cfg names the two feet; order is irrelevant.
    """
    column = {"fore_aft": 0, "lateral": 1}[axis]
    asset: Entity = env.scene[asset_cfg.name]
    feet = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    separation = quat_apply_inverse(yaw_quat(asset.data.root_link_quat_w), feet[:, 0] - feet[:, 1])
    error = torch.abs(torch.abs(separation[:, column]) - target)
    gate = _standing_gate(_head_height(env, head_asset_cfg), height_threshold)
    return gate * torch.exp(-error / scale)


def standing_joint_vel_l2(
    env: ManagerBasedRlEnv,
    height_threshold: float,
    head_asset_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Sum of squared MEASURED joint velocities, counted only once standing.

    Targets the trembling directly. The HOME-stance policy holds its stance
    by bang-bang targets (~78 % on the clip) and trembles at ~0.8 rad/s;
    home_stillness and a commanded-target-rate penalty both read the
    commanded target, which flips every step, so they sat near zero or
    changed nothing. Measured velocity is what shakes the robot.
    """
    asset: Entity = env.scene[asset_cfg.name]
    vel_sq = torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=-1)
    return _standing_gate(_head_height(env, head_asset_cfg), height_threshold) * vel_sq


def standing_target_error_l1(
    env: ManagerBasedRlEnv,
    height_threshold: float,
    head_asset_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Sum of |commanded target - measured angle| (rad), counted once standing.

    Proportional to each servo's P-term effort, so it catches a joint pressed
    into its stop: there the measured angle cannot move (a pose reward on it
    has no gradient), but the commanded target -- and this term -- can. The
    calm get-up policy held its right shoulder_roll target 1.57 rad past the
    0-deg stop at 0.44 Nm while standing.
    """
    asset: Entity = env.scene[asset_cfg.name]
    error = torch.abs(
        asset.data.joint_pos_target[:, asset_cfg.joint_ids] - asset.data.joint_pos[:, asset_cfg.joint_ids]
    )
    return _standing_gate(_head_height(env, head_asset_cfg), height_threshold) * torch.sum(error, dim=-1)


def standing_pose_reward(
    env: ManagerBasedRlEnv,
    gate_center: float,
    gate_sharpness: float,
    std: float,
    head_asset_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward joint angles close to default pose, weighted by a steep sigmoid gate on
    head height instead of a hard on/off threshold — near-zero below gate_center,
    rapidly ramping to full strength just above it (gate_sharpness controls how
    quickly; small values = steeper). A linear height-based scale was avoided because
    it would keep this reward weak most of the way to the target instead of ever
    really taking over there; this gives a clearer regime change once actually near
    standing. Kept as a secondary nudge, not a dominant term — not falling over
    matters more than exactly matching the default pose, so its configured weight
    stays modest relative to height/standing_bonus/standing_torque.
    """
    asset: Entity = env.scene[asset_cfg.name]
    height = _head_height(env, head_asset_cfg)
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    error = torch.sum(torch.square(joint_pos - default_pos), dim=-1)
    pose_reward = torch.exp(-error / std**2)
    gate = torch.sigmoid((height - gate_center) / gate_sharpness)
    return gate * pose_reward


def foot_flat_reward(
    env: ManagerBasedRlEnv,
    std: float,
    height_threshold: float,
    head_asset_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward both feet flat/level (sole parallel to the ground) once standing.

    on_feet_reward only checks that the feet are touching, not what angle they're
    touching at — a foot resting on its edge or toe still counts. Same
    projected-gravity-error shape as standing_bonus's orientation check, applied per
    foot (asset_cfg should resolve to both foot bodies) and averaged. Gated on head
    height, not trunk (see head_height_reward) — during recovery a tilted foot is
    often unavoidable/necessary, so this only applies once actually up.
    """
    asset: Entity = env.scene[asset_cfg.name]
    height = _head_height(env, head_asset_cfg)
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]
    gravity_w = asset.data.gravity_vec_w.unsqueeze(1).expand(
        -1, len(asset_cfg.body_ids), -1
    )
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)
    gravity_norm = projected_gravity_b.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    projected_gravity_b_unit = projected_gravity_b / gravity_norm
    flat_error = torch.square(projected_gravity_b_unit[..., 0]) + torch.square(
        projected_gravity_b_unit[..., 1]
    )
    flat_reward = torch.exp(-flat_error / std**2).mean(dim=-1)
    is_standing = height > height_threshold
    return torch.where(is_standing, flat_reward, torch.zeros_like(flat_reward))


def on_feet_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    target_height: float,
    head_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward both feet bearing weight (in contact) while the head is reasonably high
    off the ground — distinguishes genuinely standing-on-feet from other stable-but-
    not-standing configurations (e.g. sitting/kneeling) that could otherwise also
    score reasonably on height+upright alone. Gated on head height, not trunk (see
    head_height_reward).
    """
    found = env.scene[sensor_name].data.found
    both_feet = (found > 0).all(dim=-1).float()
    height_frac = torch.clamp(
        _head_height(env, head_asset_cfg) / target_height, min=0.0, max=1.0
    )
    return both_feet * height_frac


def head_height_reward(
    env: ManagerBasedRlEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg,
    sensor_name: str | None = None,
    power: float = 1.0,
) -> torch.Tensor:
    """Dense reward proportional to head height, capped once standing head height is
    reached.

    Using a virtual point above the trunk's own center of mass (see _head_height)
    rather than raw trunk-Z height: trunk-Z alone is orientation-blind (rewards
    getting the trunk high regardless of which way the robot is facing, which
    gives a useful gradient from any starting pose, but can't distinguish "trunk
    high, right-side up" from "trunk high, upside down" — e.g. some
    inverted/handstand-like configuration). This doesn't have that failure mode:
    an inverted trunk puts the virtual point low even with the trunk itself high,
    so it only rewards a genuinely upright-and-high pose, while keeping the same
    useful gradient from a flat starting pose (virtual point low there too).
    Matches HumanUP's (arXiv:2502.12152) choice to reward head height directly
    rather than trunk height, without actually reading the (in this task,
    independently actuated) head/neck bodies themselves.

    ``power`` (default 1.0, i.e. the original linear-up-then-flat shape) blends in
    an extra bonus for how reward rises toward target_height AND, unlike the
    original, falls again past it:

        shape = clamp(1 - |height/target_height - 1|, 0, 1)   # peak at target_height
        reward = 0.5 * shape + 0.5 * shape ** power

    ``shape`` alone is a peak centered exactly at target_height, symmetric in the
    height FRACTION (not raw meters) on either side, reaching exactly 1.0 only at
    target_height. Blended 50/50 with a HALF-WEIGHTED linear (``shape``) term
    rather than using ``shape ** power`` alone: measured directly that the power
    term by itself crushes the reward for early, modest height gains too much to
    bootstrap the get-up motion at all (e.g. power=3 pays frac=0.2 only 0.008, vs
    0.2 for plain linear — confirmed as a real regression, not just theoretical:
    head_height reward stayed flat at ~0.03-0.09 through iteration 100+ with a
    pure-power shape, dramatically worse than every run using a linear or
    blended shape at the same point). The linear half guarantees a reasonable
    baseline gradient at every height; the power half is a top-up bonus for two
    problems, both found by watching live rollouts of training-in-progress
    checkpoints:
    1. (power > 1) A flat linear reward pays the same marginal amount whether
       closing the last 10% of the height gap or the first 10% — nothing
       specifically discouraged settling into a stable SEATED local optimum well
       short of standing. d/dx[x^p] = p*x^(p-1) grows with x for p > 1, so the
       last stretch pays disproportionately more.
    2. (falling off past target_height, not clamped flat at 1.0) A reward that
       merely saturates at "at least target_height" gives no reason to avoid
       overshooting it either (standing on tiptoes, a jump, or other
       overextension) — peaking exactly at the true standing height and paying
       less on EITHER side keeps target_height as the one specific point being
       optimized for, not a floor.
    """
    height = _head_height(env, asset_cfg)
    frac = height / target_height
    shape = torch.clamp(1.0 - torch.abs(frac - 1.0), min=0.0, max=1.0)
    reward = 0.5 * shape + 0.5 * shape**power
    if sensor_name is not None:
        airborne = _feet_airborne(env, sensor_name)
        reward = torch.where(airborne, torch.zeros_like(reward), reward)
    return reward


def _feet_airborne(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """True where neither foot has ground contact (bool, shape [B])."""
    found = env.scene[sensor_name].data.found
    return (found <= 0).all(dim=-1)


class hold_airborne:
    """Suspend the robot by applying a gravity-compensating external force to the
    trunk for a sampled duration, with a cooldown between holds — simulates a human
    picking the robot up and holding it. Mirrors mjlab's own
    ``mjlab.envs.mdp.events.apply_body_impulse`` state-machine (cooldown -> trigger ->
    sustain -> expire, one independent timer per env), but the force is biased upward
    (toward weight support) instead of zero-centered.

    Unlike a kinematic teleport of the root pose, this keeps the trunk dynamically
    simulated — internal joint torques can still swing/tilt the body against the
    supporting force, so a flailing policy visibly fights the hold instead of being
    rigidly pinned, and a compliant one hangs still. write_external_wrench_to_sim
    persists until overwritten (same assumption apply_body_impulse makes), so the
    force only needs to be (re)written at trigger and expire, not every step.

    Use with mode="step".
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self._asset: Entity = env.scene[cfg.params["asset_cfg"].name]
        self._body_ids = cfg.params["asset_cfg"].body_ids
        self._device = env.device
        self._step_dt = env.step_dt
        self._num_bodies = (
            len(self._body_ids)
            if isinstance(self._body_ids, list)
            else self._asset.num_bodies
        )
        self._cooldown_s = cfg.params["cooldown_s"]

        self._time_remaining = torch.zeros(env.num_envs, device=self._device)
        # Staggered, not zero: an all-zero init makes every env's first hold trigger
        # in lockstep on step 1 — reproduced as a real MuJoCo solver blowup (NaN
        # qpos/qvel within ~65 steps at 4096 envs) from applying a near-mg force to
        # every single env simultaneously, on top of the already-extreme randomized
        # fallen pose reset. Sampling from cooldown_s here (and in reset()) spreads
        # first triggers out like the ones after every subsequent expiry already are.
        lo, hi = self._cooldown_s
        self._interval_time_left = (
            torch.rand(env.num_envs, device=self._device) * (hi - lo) + lo
        )
        self._active = torch.zeros(env.num_envs, device=self._device, dtype=torch.bool)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        force_z_range: tuple[float, float],
        force_lateral_range: tuple[float, float],
        torque_range: tuple[float, float],
        duration_s: tuple[float, float],
        cooldown_s: tuple[float, float],
        asset_cfg: SceneEntityCfg,
    ) -> None:
        del env, env_ids, asset_cfg  # Unused; step events always operate on all envs.
        dt = self._step_dt

        self._time_remaining[self._active] -= dt

        expired = self._active & (self._time_remaining <= 0)
        if expired.any():
            expired_ids = expired.nonzero(as_tuple=False).squeeze(-1)
            zeros = torch.zeros(
                (len(expired_ids), self._num_bodies, 3), device=self._device
            )
            self._asset.write_external_wrench_to_sim(
                zeros, zeros, env_ids=expired_ids, body_ids=self._body_ids
            )
            self._active[expired_ids] = False
            self._time_remaining[expired_ids] = 0.0
            lo, hi = cooldown_s
            self._interval_time_left[expired_ids] = (
                torch.rand(len(expired_ids), device=self._device) * (hi - lo) + lo
            )

        self._interval_time_left -= dt

        eligible = (~self._active) & (self._interval_time_left <= 0)
        if eligible.any():
            trigger_ids = eligible.nonzero(as_tuple=False).squeeze(-1)
            n = len(trigger_ids)
            forces = sample_uniform(
                *force_lateral_range, (n, self._num_bodies, 3), self._device
            )
            forces[..., 2] = sample_uniform(
                *force_z_range, (n, self._num_bodies), self._device
            )
            torques = sample_uniform(
                *torque_range, (n, self._num_bodies, 3), self._device
            )
            self._asset.write_external_wrench_to_sim(
                forces, torques, env_ids=trigger_ids, body_ids=self._body_ids
            )

            lo, hi = duration_s
            self._time_remaining[trigger_ids] = (
                torch.rand(n, device=self._device) * (hi - lo) + lo
            )
            self._active[trigger_ids] = True

    def reset(self, env_ids: torch.Tensor) -> None:
        # An env can reset mid-hold (episode end while _active); the external wrench
        # written by __call__ otherwise persists onto the freshly-reset state.
        zeros = torch.zeros((len(env_ids), self._num_bodies, 3), device=self._device)
        self._asset.write_external_wrench_to_sim(
            zeros, zeros, env_ids=env_ids, body_ids=self._body_ids
        )

        self._time_remaining[env_ids] = 0.0
        lo, hi = self._cooldown_s
        self._interval_time_left[env_ids] = (
            torch.rand(len(env_ids), device=self._device) * (hi - lo) + lo
        )
        self._active[env_ids] = False


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


class step_based_staged_curriculum:
    """
    Curriculum based on step count stages. Each stage is applied once when
    env.common_step_counter reaches the stage's step threshold.

    Stage definitions example:
    stages = [
        {
            "name": "stage 1",
            "step": 10_000 * 24,
            "apply": lambda env: env.reward_manager.get_term_cfg("term_name").weight = 1.0,
        },
        ...
    ]
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
        self.current_stage = 0

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor,
        stages: list[dict],
    ) -> dict[str, torch.Tensor]:
        del env_ids
        if (
            self.current_stage < len(stages)
            and env.common_step_counter >= stages[self.current_stage]["step"]
        ):
            stage = stages[self.current_stage]
            print(
                f"Curriculum stage {self.current_stage + 1}: {stage['name']} at step {env.common_step_counter}"
            )
            stage["apply"](env)
            self.current_stage += 1

        return {"stage": self.current_stage}


class reward_based_staged_curriculum:
    """
    Curriculum based on stages ending while a reward component gets its mean
    episode reward accross all environments above a threshold.

    "reward_term_name" can be a single term name, or a list of term names — a
    stage with a list only advances once EVERY named term's mean episode reward
    has crossed its threshold, not just one of them. Added for a case where two
    terms (standing_pose/hip_pose) were raised to the same weight and a live
    rollout showed one of them (hip_pose) lagging behind the other — gating the
    next stage on standing_pose alone would have let it fire before hip_pose had
    actually caught up. "threshold" is then EITHER one number (applied to every
    named term) OR a list the same length as "reward_term_name" (one threshold
    per term, in the same order) — added because hip_pose's own achievable
    ceiling measured meaningfully lower than standing_pose's at the same weight
    (added joint dof, presumably harder to converge), so gating both on the same
    number was either too easy for one or unreachable for the other.

    Stage definitions example:
    stages = [
        {
            "name": "stage 1",
            "reward_term_name": "term_name",  # or ["term_a", "term_b"]
            "threshold": 0.5,  # or [0.5, 0.3] matching ["term_a", "term_b"]
            "apply": lambda env: env.reward_manager.get_term_cfg("term_name").weight = 1.0,
        },
        ...
    ]
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
        self.rewards: dict[str, torch.Tensor] = {}
        self.current_stage = 0
        self.stage_first_step = 0

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor,
        stages: list[dict],
    ) -> dict[str, torch.Tensor]:
        # Pre-existing bug, fixed here: this used to index stages[self.current_stage]
        # unconditionally before checking self.current_stage < len(stages), so once
        # every stage had actually been applied (self.current_stage == len(stages))
        # the very next call crashed with IndexError instead of just staying done —
        # only surfaces once a reward_based_staged_curriculum's LAST stage actually
        # triggers, which apparently hadn't happened before with this class in this
        # codebase. Guarding the whole body on the bounds check fixes it.
        if self.current_stage >= len(stages):
            return {"stage": self.current_stage}

        stage = stages[self.current_stage]
        term_names = stage["reward_term_name"]
        if isinstance(term_names, str):
            term_names = [term_names]
        thresholds = stage["threshold"]
        if not isinstance(thresholds, (list, tuple)):
            thresholds = [thresholds] * len(term_names)

        mean_rewards = {}
        for name in term_names:
            if name not in self.rewards:
                self.rewards[name] = torch.zeros(env.num_envs, device=env.device)
            self.rewards[name][env_ids] = (
                env.reward_manager._episode_sums[name][env_ids] / env.max_episode_length_s
            )
            mean_rewards[name] = self.rewards[name].mean().item()

        if (
            all(mean_rewards[name] >= t for name, t in zip(term_names, thresholds))
            and env.common_step_counter >= self.stage_first_step + 100 * 24
        ):
            rewards_str = ", ".join(f"{name}={r:.4f}" for name, r in mean_rewards.items())
            print(
                f"Curriculum stage {self.current_stage + 1}: {stage['name']} at step {env.common_step_counter} (mean episode reward: {rewards_str})"
            )
            stage["apply"](env)
            self.current_stage += 1
            self.stage_first_step = env.common_step_counter
            for name in term_names:
                self.rewards[name].zero_()  # Reset rewards to avoid immediately triggering the next stage

        return {"stage": self.current_stage}


class reward_based_curriculum:
    """
    Curriculum based on the mean episode reward of a specific term accross all environments.
    Once the mean reward across envs exceeds a threshold, a new curriculum stage is applied.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
        self.rewards = torch.zeros(env.num_envs, device=env.device)
        self.current_stage = 0
        self.stage_first_step = 0

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor,
        reward_term_name: str,
        stages: list[dict],
    ) -> dict[str, torch.Tensor]:
        self.rewards[env_ids] = (
            env.reward_manager._episode_sums[reward_term_name][env_ids]
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

        return {"stage": self.current_stage}


def set_command_velocity(
    env,
    lin_vel_x=None,
    lin_vel_y=None,
    ang_vel_z=None,
    rotation_env_ang_vel_z=None,
) -> None:
    """
    Helper function to set the command velocity parameters in the environment.
    """
    cmd = env.command_manager.get_term_cfg("twist")
    if lin_vel_x is not None:
        cmd.ranges.lin_vel_x = lin_vel_x
    if lin_vel_y is not None:
        cmd.ranges.lin_vel_y = lin_vel_y
    if ang_vel_z is not None:
        cmd.ranges.ang_vel_z = ang_vel_z
    if rotation_env_ang_vel_z is not None:
        cmd.rotation_env_ang_vel_range = rotation_env_ang_vel_z


def set_stepping_parameters(
    env,
    air_time_weight: float | None = None,
    no_stepping_penalty_weight: float | None = None,
    rel_standing_envs: float | None = None,
    rel_rotation_envs: float | None = None,
) -> None:
    """
    Helper function to set stepping/standing curriculum parameters.
    """
    if air_time_weight is not None:
        env.reward_manager.get_term_cfg("air_time").weight = air_time_weight
    if no_stepping_penalty_weight is not None:
        env.reward_manager.get_term_cfg(
            "no_stepping"
        ).weight = no_stepping_penalty_weight
    if rel_standing_envs is not None:
        env.command_manager.get_term_cfg("twist").rel_standing_envs = rel_standing_envs
    if rel_rotation_envs is not None:
        env.command_manager.get_term_cfg("twist").rel_rotation_envs = rel_rotation_envs


def hold_at_default_pose(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
) -> None:
    """Set the given joints' position TARGET (not raw ctrl) to default_joint_pos at
    reset.

    BAM actuators (bam.mjlab.BamActuator, used for all 21 joints here) compute force
    from ``data.joint_pos_target``, not from MuJoCo's raw ``ctrl`` — writing ctrl
    directly (e.g. via Entity.write_ctrl_to_sim) gets overwritten the next physics
    step regardless. EntityData resets joint_pos_target to 0.0 for every joint on
    every episode reset, and the action pipeline (JointPositionAction) only ever
    updates it for the actuators it owns — so any actuator OUTSIDE the RL action term
    (measured 2026-09-22) has its target silently left at 0 rather than
    default_joint_pos, driving joints like the elbows/shoulders (whose default is far
    from 0: -20/±10 deg) to visibly drift there every episode. That's an unintended
    disturbance injected into every training run that excluded them from actions —
    likely the real cause behind repeated failed attempts at a legs-only policy, more
    than any of the reward/curriculum/observation changes tried first.
    """
    asset: Entity = env.scene[asset_cfg.name]
    default_pos = asset.data.default_joint_pos[env_ids][:, asset_cfg.joint_ids]
    asset.set_joint_position_target(
        default_pos, joint_ids=asset_cfg.joint_ids, env_ids=env_ids.unsqueeze(-1)
    )


def randomize_upper_body_pose_reset(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
) -> None:
    """Teleport the given joints (arms/neck/head) to a random position within their
    own soft limits at reset, and set their position actuators to match (so there's no
    snap at episode start — see randomize_upper_body_pose_interval for the ongoing,
    mid-episode counterpart that keeps them actually moving).

    These joints aren't RL-controlled — they're driven mathematically (IK for the
    arms, the closed-form stabilization law for the neck) once deployed.

    asset_cfg must resolve both joint_names and actuator_names to the SAME set of
    joints (in corresponding order — each actuator drives its own like-named joint).
    """
    asset: Entity = env.scene[asset_cfg.name]

    pose = _sample_upper_body_pose(env, env_ids, asset_cfg)
    zero_vel = torch.zeros_like(pose)

    asset.write_joint_state_to_sim(
        pose, zero_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids
    )
    asset.set_joint_position_target(
        pose, joint_ids=asset_cfg.joint_ids, env_ids=env_ids.unsqueeze(-1)
    )


def randomize_upper_body_pose_interval(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
) -> None:
    """Retarget the arm/neck/head position actuators to a new random pose, WITHOUT
    teleporting the joint state — the position servo drives them there smoothly over
    the next few control steps, same as a real IK-driven arm or the neck stabilization
    law continuously retargeting while the operator moves.

    Fired on an interval (see the "interval_range_s" on this event's EventTermCfg) so
    the upper body keeps moving throughout each episode instead of holding one fixed
    pose — a static per-episode pose (randomize_upper_body_pose_reset alone) never
    exposes the legs to the ongoing, changing momentum a moving upper body imparts
    while actually walking.
    """
    asset: Entity = env.scene[asset_cfg.name]
    pose = _sample_upper_body_pose(env, env_ids, asset_cfg)
    asset.set_joint_position_target(
        pose, joint_ids=asset_cfg.joint_ids, env_ids=env_ids.unsqueeze(-1)
    )


def _sample_upper_body_pose(
    env: ManagerBasedRlEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    limits = asset.data.soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
    r = torch.rand(limits.shape[0], limits.shape[1], device=env.device)
    return limits[..., 0] + r * (limits[..., 1] - limits[..., 0])


def set_push_parameters(
    env,
    velocity_range: dict[str, tuple[float, float]] | None = None,
    interval_range: tuple[float, float] | None = None,
) -> None:
    """
    Helper function to set push event parameters.
    Returns a dict of the current (post-update) values for wandb logging.
    """
    push_event_cfg = env.event_manager.get_term_cfg("push_robot")
    if velocity_range is not None:
        push_event_cfg.params["velocity_range"] = velocity_range
    if interval_range is not None:
        push_event_cfg.params["interval_range"] = interval_range


def penalize_stepping_while_standing(
    env: ManagerBasedRlEnv,
    air_time_weight: float,
    no_stepping_penalty_weight: float,
) -> torch.Tensor:
    """
    Updating the air_time and no_stepping reward weights to penalize stepping while standing.
    """
    env.reward_manager.get_term_cfg("air_time").weight = air_time_weight
    env.reward_manager.get_term_cfg("no_stepping").weight = no_stepping_penalty_weight


def stepping_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    air_time_weight: float,
    no_stepping_penalty_weight: float,
    rel_standing_envs: float = 0.0,
    rel_rotation_envs: float = 0.0,
    step: int = 10000 * 24,
) -> dict[str, torch.Tensor]:
    """
    Updating the air_time and no_stepping reward weights to penalize stepping while standing
    after a certain number of iterations.
    """
    del env_ids  # Unused.

    if env.common_step_counter >= step:
        env.reward_manager.get_term_cfg("air_time").weight = air_time_weight
        env.reward_manager.get_term_cfg(
            "no_stepping"
        ).weight = no_stepping_penalty_weight
        env.command_manager.get_term_cfg("twist").rel_standing_envs = rel_standing_envs
        env.command_manager.get_term_cfg("twist").rel_rotation_envs = rel_rotation_envs

    return {
        "air_time_weight": torch.tensor(
            env.reward_manager.get_term_cfg("air_time").weight
        ),
        "no_stepping_penalty_weight": torch.tensor(
            env.reward_manager.get_term_cfg("no_stepping").weight
        ),
        "rel_standing_envs": torch.tensor(
            env.command_manager.get_term_cfg("twist").rel_standing_envs
        ),
        "rel_rotation_envs": torch.tensor(
            env.command_manager.get_term_cfg("twist").rel_rotation_envs
        ),
    }


class target_rate_l2:
    """Penalize the commanded joint TARGET (asset.data.joint_pos_target) changing
    between steps — a drop-in replacement for mjlab's built-in action_rate_l2,
    which penalizes env.action_manager.action/prev_action instead: those are
    explicitly the RAW, pre-scale/offset/clip network output (per
    ActionManager.process_action's own docstring), not what the robot actually
    ends up commanded to do. Once that raw value saturates past this task's own
    clip range (measured directly: RMS raw action reaching >100 over an episode,
    while the actual clipped target stays within +-1.57), action_rate_l2 can
    penalize a target that ISN'T physically changing at all just because the
    raw pre-clip number is still swinging — the opposite of what a smoothness
    penalty is supposed to measure. This reads the same joint_pos_target field
    home_stillness_reward does, which is bounded by the real clip and reflects
    what's actually asked of the servo.

    A previous joint_pos_target isn't separately exposed anywhere, so (like
    home_stillness_reward) this class caches its own (self._prev_target,
    updated every call) rather than being a plain function.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        asset: Entity = env.scene[cfg.params["asset_cfg"].name]
        self._joint_ids = cfg.params["asset_cfg"].joint_ids
        self._prev_target = asset.data.joint_pos_target[:, self._joint_ids].clone()

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]
        target = asset.data.joint_pos_target[:, asset_cfg.joint_ids]
        rate_sq = torch.sum(torch.square(target - self._prev_target), dim=-1)
        self._prev_target = target.clone()
        return rate_sq

    def reset(self, env_ids: torch.Tensor) -> None:
        del env_ids  # Unused — a stale cross-episode rate spike for one step
        # right after a reset is a minor, brief inaccuracy, not worth the extra
        # bookkeeping (matches home_stillness_reward's own reset() reasoning).




class home_stillness_reward:
    """Reward the commanded joint TARGET (asset.data.joint_pos_target, not the
    measured joint_pos/joint_vel) holding still near home, but only once actually
    standing (head height above height_threshold) — a triple AND (standing,
    target-near-home, target-not-changing) via multiplication/gating, not a
    penalty gated by any one of these alone.

    Went through several earlier versions of this idea, each found to backfire:
    1. A hard is_near_home threshold switching a joint-velocity PENALTY on/off:
       made trembling WORSE on a live rollout — a step function right at the
       boundary gives a policy sitting near it a reason to keep crossing back and
       forth (avoiding the penalty on one side costs nothing the reward
       otherwise cares about). Note this is specifically a penalty-gate pathology,
       not a general problem with binary gates: home_pose_reward's own gate is
       also binary (on height, not closeness-to-home) but gates a REWARD rather
       than a penalty, so a policy oscillating across it isn't rewarded for the
       oscillation itself the way it would be for dodging a penalty here.
    2. Smoothing that penalty into a continuous ramp fixed the boundary problem
       but kept a different one: as a pure penalty gated by closeness, being FAR
       from home paid exactly 0 regardless of velocity, while being CLOSE always
       carried at least the risk of a penalty unless velocity was exactly 0 — a
       net asymmetric incentive to just not fully converge onto home at all.
    3. A positive reward using env.action_manager.action (the policy's RAW,
       pre-scale/offset/clip network output): measured via a live debug rollout
       to be essentially unbounded (RMS growing past 100 over an episode, no
       ceiling) — this task's JointPositionActionCfg clips the fully-processed
       action to (-1.57, 1.57) rad, but that clip is applied to
       raw*scale+offset, a quantity the action MANAGER never exposes; the RAW
       action alone has no such bound and drifts arbitrarily, since nothing
       downstream of that clip pushes back on its scale. Any fixed "worst case"
       constant for it is meaningless — it measured to either saturate every
       ramp to exactly 0 or (once loosened far enough to stop doing that)
       plainly not correspond to anything physical.
    This version reads asset.data.joint_pos_target instead: the ACTUAL,
    post-scale/offset/clip position each joint's PD controller is tracking right
    now (what apply_actions() in mjlab's JointPositionAction writes via
    set_joint_position_target) — same physical radians and indexing as joint_pos/
    default_joint_pos, genuinely bounded by the action term's own clip, and
    exactly "the target value" in the sense meant throughout this reward's
    design: what the policy is currently asking for, independent of how well the
    real joint is tracking it.

    A previous joint_pos_target isn't separately exposed anywhere, so this class
    caches its own (self._prev_target, updated every call) rather than being a
    plain function like most other reward terms here.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        asset: Entity = env.scene[cfg.params["asset_cfg"].name]
        self._joint_ids = cfg.params["asset_cfg"].joint_ids
        self._prev_target = asset.data.joint_pos_target[:, self._joint_ids].clone()

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        height_threshold: float,
        target_pose_worst: float,
        target_rate_worst: float,
        head_asset_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]
        target_pos = asset.data.joint_pos_target[:, asset_cfg.joint_ids]
        default_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]

        target_offset = target_pos - default_pos
        target_pose_error = torch.sqrt(torch.mean(torch.square(target_offset), dim=-1))
        pose_closeness = torch.clamp(1.0 - target_pose_error / target_pose_worst, min=0.0, max=1.0)

        target_rate = torch.sqrt(
            torch.mean(torch.square((target_pos - self._prev_target) / env.step_dt), dim=-1)
        )
        target_stillness = torch.clamp(1.0 - target_rate / target_rate_worst, min=0.0, max=1.0)
        self._prev_target = target_pos.clone()

        height = _head_height(env, head_asset_cfg)
        is_standing = height > height_threshold

        reward = pose_closeness * target_stillness
        return torch.where(is_standing, reward, torch.zeros_like(reward))

    def reset(self, env_ids: torch.Tensor) -> None:
        del env_ids  # Unused — a stale cross-episode target_rate spike for one
        # step right after a reset is harmless, since is_standing gates the whole
        # term to 0 right at that moment anyway (a just-reset env starts fallen).




class home_pose_reward:
    """Reward joint angles close to the robot's own default/home pose, gated by a
    simple binary threshold on head height (0 below it, full reward above it — see
    __call__; this replaced an earlier smooth ramp-shaped gate once the extra
    tunable shape wasn't earning its complexity back over just reusing the one
    threshold every other "are we actually standing now" reward already gates on).

    The error-shape problem this term was designed to fix (found by surveying HoST
    arXiv:2502.08378, FRASA arXiv:2410.08655, and HumanUP arXiv:2502.12152 — see
    microban_getup_env_cfg.py for how this term is wired): summing squared error
    across ~18 joints then applying ONE exp(-sum/std^2) saturates near 0 whenever
    several joints are simultaneously off (the normal case for most of a recovery
    motion), collapsing the gradient into an undifferentiated "far from home"
    signal that can't tell "1 joint off" from "8 joints off". This uses per-joint
    standard deviations and the MEAN (or, for hip_pose, MAX — see the reduction
    param) of error^2/std^2 — identical mean shape to
    mjlab.tasks.velocity.mdp.rewards.variable_posture, the reward already driving
    this same robot's walking policy's own standing/walking pose term (see
    microban_velocity_env_cfg.py's std_standing) — averaging keeps the scale
    independent of joint count, and per-joint std lets tight/loose joints be tuned
    individually.

    Still gated at all (not literally dense from frame 1 like FRASA, which trains
    with an off-policy algorithm less prone to single-critic reward interference —
    see HoST's own single-critic-vs-multi-critic ablation): with on-policy PPO and a
    single critic here, giving large pose-matching gradient while the robot is still
    mid-flip (where large joint excursions from home are the correct behavior) would
    fight the get-up motion itself. The height gate keeps this a "which of the
    successful strategies do you converge to" signal, not a "do this instead of
    getting up" signal.

    ``std`` is re-resolved from ``cfg.params["std"]`` on every call (only the joint
    NAME list is cached from ``__init__``), not precomputed once into a fixed
    tensor. Nothing currently mutates it mid-training (only this term's own weight
    gets ramped by the env cfg's curriculum, via ``get_term_cfg("standing_pose")
    .weight``), but the re-resolution keeps that option open cheaply. It matters
    which std you start with regardless of any curriculum: the walking policy's own
    tight std_standing values (0.1-0.15 rad) saturate this term to ~0 for the
    still-imperfect joint configurations a mid-training get-up policy actually
    reaches, even once the gate is open — hence the much looser flat
    HOME_POSE_STD in microban_getup_env_cfg.py.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        asset: Entity = env.scene[cfg.params["asset_cfg"].name]
        _, self._joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        std: dict[str, float],
        height_threshold: float,
        head_asset_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
        reduction: str = "mean",
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]
        _, _, std_values = resolve_matching_names_values(
            data=std,
            list_of_strings=self._joint_names,
        )
        std_t = torch.tensor(std_values, device=env.device, dtype=torch.float32)
        height = _head_height(env, head_asset_cfg)
        joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
        default_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
        error_sq = torch.square(joint_pos - default_pos)
        scaled_error_sq = error_sq / std_t**2
        # "mean" (default, used by standing_pose): scale-independent of joint count,
        # matches variable_posture's own shape. "max" (used by hip_pose): gradient
        # flows ONLY through the single worst joint in this group each step, instead
        # of being diluted 1/N across N joints — requested because a mean over just
        # the 4 hip joints still let 3-already-converged joints outvote the one
        # (left_hip_pitch) still ~90+deg off; this makes the term care about
        # shrinking whichever hip joint is currently worst, not the group average.
        if reduction == "max":
            reduced = torch.amax(scaled_error_sq, dim=-1)
        elif reduction == "mean":
            reduced = torch.mean(scaled_error_sq, dim=-1)
        else:
            raise ValueError(f"Unknown reduction: {reduction!r}")
        pose_reward = torch.exp(-reduced)
        # Simple binary gate at the SAME height_threshold as standing_bonus (pass
        # HEAD_STANDING_THRESHOLD from the env cfg) — no ramp, no separate minimum
        # fraction: exactly 0 below it, full pose_reward above it. Simplified from
        # an earlier smooth-ramp-with-hard-floor version once it became clear the
        # extra tunable shape (gate_target_height/gate_power/gate_min_frac) wasn't
        # earning its complexity back over just reusing the one threshold every
        # other "are we actually standing now" reward already gates on.
        is_standing = height > height_threshold
        return torch.where(is_standing, pose_reward, torch.zeros_like(pose_reward))

    def reset(self, env_ids: torch.Tensor) -> None:
        del env_ids  # Unused.




def hands_released_reward(
    env: ManagerBasedRlEnv,
    height_threshold: float,
    sensor_name: str,
    head_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward NOT bracing on the hands/forearms once past a (deliberately early, not
    "nearly standing") height threshold.

    Added after watching a live rollout of a training-in-progress checkpoint: the
    policy reliably props itself up on its hands to roughly half height, then gets
    stuck there — it never lets go to complete the extension onto its feet alone.
    Every other reward term here rewards being tall/on-feet/home-postured, but none
    of them ever penalizes STAYING propped on the hands once there's clearly no need
    to be (the trunk is already elevated) — so "prop up and stop" is a stable
    stopping point under the existing reward stack, not just a slow-to-leave one.

    Gated at a LOWER height than standing_bonus/on_feet/foot_flat (which all key off
    HEAD_STANDING_THRESHOLD, i.e. near-complete) specifically because the failure
    mode happens well before that point — gating this the same way would never
    engage during the exact phase where the policy is stuck. height_threshold should
    be passed in around the "propped up on hands" height observed in practice (much
    lower than standing height), not the standing threshold itself.
    """
    found = env.scene[sensor_name].data.found
    hands_off_ground = (found <= 0).all(dim=-1).float()
    height = _head_height(env, head_asset_cfg)
    is_past_threshold = height > height_threshold
    return torch.where(is_past_threshold, hands_off_ground, torch.zeros_like(hands_off_ground))




def reset_near_home_fraction(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    rel_near_home_envs: float,
    joint_noise_range: tuple[float, float],
    orientation_noise_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """After the fallen-pose reset events (reset_base / reset_robot_joints) have
    already placed ``env_ids`` in an extreme random fallen configuration — this
    task's main training distribution — re-place a random fraction of THOSE SAME
    env_ids back into the home/standing pose plus small noise instead.

    Both FRASA (arXiv:2410.08655, Rhoban's own fall-recovery policy for a similarly
    small biped — its ``reset_final_p`` fraction) and HumanUP (arXiv:2502.12152 —
    ``_reset_stand_and_lie_states`` / ``standing_init_prob``) independently use this
    exact mechanism, found while surveying the fall-recovery RL literature for this
    task (see microban_getup_env_cfg.py). Without it, the pose-matching reward
    (home_pose_reward) only ever gets on-policy gradient from whatever pose the
    get-up policy happens to arrive at organically at the tail of a long, noisy
    recovery rollout — it never directly practices "stay AT home", only "pass
    through/near home once". Spawning some episodes already there gives dense,
    correctly-attributed gradient for stabilizing at exactly the target the pose
    reward is shaping toward.

    Relies on dict insertion order in ``cfg.events``: this event must be registered
    AFTER ``reset_base``/``reset_robot_joints`` so it overrides their sampled fallen
    pose for the selected subset rather than being overwritten by them.
    """
    mask = torch.rand(len(env_ids), device=env.device) < rel_near_home_envs
    near_home_ids = env_ids[mask]
    if len(near_home_ids) == 0:
        return

    lo, hi = orientation_noise_range
    # Roll/pitch noise about the HOME trunk frame, yaw about world z, so the
    # leaning HOME keeps its flat soles at every heading.
    reset_root_state_uniform_world_yaw(
        env,
        near_home_ids,
        pose_range={
            "x": (-0.02, 0.02),
            "y": (-0.02, 0.02),
            "z": (0.0, 0.005),
            "roll": (lo, hi),
            "pitch": (lo, hi),
            "yaw": (-3.14159, 3.14159),
        },
    )
    envs_mdp.reset_joints_by_offset(
        env,
        near_home_ids,
        position_range=joint_noise_range,
        velocity_range=(0.0, 0.0),
        asset_cfg=asset_cfg,
    )




