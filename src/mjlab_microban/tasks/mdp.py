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
from mjlab.utils.lab_api.math import (
    quat_apply,
    quat_apply_inverse,
    quat_from_euler_xyz,
    quat_mul,
    sample_uniform,
    subtract_frame_transforms,
    yaw_quat,
)
from mjlab.tasks.velocity.mdp.velocity_command import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)
from mjlab.utils.lab_api.string import resolve_matching_names_values

from mjlab_microban.robot.home_pose import HOME

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

    def left_from_right_level(self) -> torch.Tensor:
        """The left foot's position from the right one, in the trunk's heading
        frame levelled by gravity: the yaw of the HOME-levelled trunk frame
        (x forward, y left, z up), so the trunk's roll and pitch do not turn
        it.  Shape (N, 3).  At HOME it is the HOME-levelled trunk frame."""
        feet_w = self.robot.data.site_pos_w[:, self._foot_asset_cfg.site_ids, :]
        heading = yaw_quat(
            _home_levelled_quat(self.robot.data.root_link_quat_w, self.cfg.trunk_pitch)
        )
        return quat_apply_inverse(heading, feet_w[:, 0] - feet_w[:, 1])

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


########################## OBSERVATIONS ############################


def foot_target_offset_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Flattened (left_dx, left_dy, left_dz, right_dx, right_dy, right_dz) target foot
    offset, in the command's HOME-levelled trunk frame. Lets the actor see what
    leg-tracking target (if any) is
    currently commanded, alongside the velocity command."""
    command: FootTargetCommand = env.command_manager.get_term(command_name)
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
    # Despite the head_asset_cfg name, this does not read the actual "head" body
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

    The continued scaling above threshold (rather than a flat 1.0) prevents a
    plateau with the head height sitting AT height_threshold: with this reward
    and standing_pose both flat/binary on the same threshold there is no further
    gradient toward target_height once crossed, while standing_torque's effort
    cost keeps rising the higher (and more extended) the stance gets.

    No separate trunk-upright factor: height_threshold is relative to the TRUE
    standing head height (HEAD_STANDING_HEIGHT in microban_getup_env_cfg.py), so
    reaching it is only physically possible while upright -- a seated, kneeling,
    or inverted-but-elevated pose cannot reach the head that high.
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

    lin_vel_std / ang_vel_std are set to this robot's scale.

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

    Started EXACTLY at its standing pose and holding EXACTLY that fixed joint
    configuration (no policy, no correction), the robot holds for ~1s (tilt
    drifting ~0.1deg -> ~3deg, joint tracking error under a few degrees --
    not an actuator-gain problem), then topples (47deg tilt by 2s, fully
    fallen by 4s). Standing upright is an unstable equilibrium that requires
    CONTINUOUS ACTIVE CORRECTION, not just holding a fixed target pose.
    Neither standing_bonus (height) nor standing_stability_reward (velocity)
    catches the early drift: head height barely moves until tilt exceeds
    ~40 degrees, and velocity picks up only once already falling. This term
    rewards low tilt as a dense signal from the start of the standing phase.

    Reuses projected_gravity_b (the same quantity the actor already
    observes) rather than introducing a separate tilt computation: at
    perfectly upright this is (0, 0, -1), so its horizontal-plane norm is
    exactly 0 and grows smoothly with tilt angle (sin of the tilt angle, for
    small-to-moderate tilt) -- the same signal the policy could in principle
    already use to self-correct, just not yet rewarded for acting on.

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
    if pitch == 0.0:
        tilt = torch.linalg.norm(gravity_xy, dim=-1)
    else:
        target_xy = torch.tensor(
            (math.sin(pitch), 0.0), device=gravity_xy.device, dtype=gravity_xy.dtype
        )
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
    single summed-error term lets a policy narrow its stance by staggering
    one foot forward, barely changing the sum.
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

    Targets the trembling directly. Without it the HOME-stance policy holds
    its stance by bang-bang targets on the clip and trembles; home_stillness
    and a commanded-target-rate penalty both read the commanded target,
    which flips every step, so they do not see it. Measured velocity is what
    shakes the robot.
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
    rather than using ``shape ** power`` alone: the power term by itself crushes
    the reward for early, modest height gains too much to bootstrap the get-up
    motion at all (power=3 pays frac=0.2 only 0.008, vs 0.2 for plain linear;
    with a pure-power shape head_height stays flat at ~0.03-0.09 through
    iteration 100+). The linear half guarantees a reasonable baseline gradient
    at every height; the power half is a top-up bonus for two problems:
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
    lift_threshold: float,
    velocity_command_name: str = "twist",
    velocity_fade_range: tuple[float, float] = (0.0, 0.15),
    single_foot_weight: float = 1.0,
) -> torch.Tensor:
    """Reward for matching the commanded per-foot target offset (FootTargetCommand),
    exp(-error / std^2), faded out smoothly as the velocity command grows so legs
    prioritize walking over holding a static foot target once actually moving. This
    is a continuous weight (no hard cutoff/mode switch): at velocity_fade_range[0]
    and below, full weight; at velocity_fade_range[1] and above, zero; linear in
    between.

    error, with d the feet at reset (HOME) and o the published offsets, in the
    command's HOME-levelled trunk frame:
      one foot up (``_single_foot_rows``: the published targets differ in z by
      ``lift_threshold`` or more): the lifted foot from the support foot, in the
      trunk's heading frame levelled by gravity (``left_from_right_level``),
      |p_L - p_R - (d_L - d_R + o_L - o_R)|^2.  To stand on one foot the trunk
      moves over the support foot (36-52 mm at Microban's size), so the support
      foot's place under the trunk is not part of it (that it stays down is
      ``lifted_support_feet``'s), and the trunk's roll and pitch do not move
      the target;
      otherwise (no target, both feet by one offset, or below the threshold):
      each foot from the trunk, mean_feet |p_b - d - o|^2.

    The one-foot-up rows are paid ``single_foot_weight`` times the others.
    """
    command: FootTargetCommand = env.command_manager.get_term(command_name)
    offset = command.foot_target_offset_b  # (N, 2, 3)
    default = command._default_foot_pos_b
    error = torch.sum(
        torch.square(command.current_foot_pos_b() - default - offset), dim=-1
    ).mean(-1)
    relative_target = default[:, 0] - default[:, 1] + offset[:, 0] - offset[:, 1]
    relative_error = torch.sum(
        torch.square(command.left_from_right_level() - relative_target), dim=-1
    )
    single = _single_foot_rows(offset[..., 2], lift_threshold)
    error = torch.where(single, relative_error, error)
    tracking_reward = torch.exp(-error / std**2) * torch.where(single, single_foot_weight, 1.0)

    velocity_command = env.command_manager.get_command(velocity_command_name)
    speed = torch.norm(velocity_command[:, :2], dim=-1) + torch.abs(
        velocity_command[:, 2]
    )
    lo, hi = velocity_fade_range
    fade = 1.0 - torch.clamp((speed - lo) / (hi - lo), 0.0, 1.0)

    return tracking_reward * fade


def foot_target_active(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Rows (bool, shape [N]) with a foot target: drawn or still non-zero."""

    foot_target = env.command_manager.get_term(command_name)
    active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name in ("is_single_support_env", "is_both_feet_env"):
        flag = getattr(foot_target, name, None)
        if flag is None and name == "is_both_feet_env":
            continue
        if not isinstance(flag, torch.Tensor) or flag.shape != (env.num_envs,):
            raise ValueError(f"foot target command must expose {name} with shape (num_envs,)")
        active |= flag.bool()
    published = foot_target.command.reshape(env.num_envs, -1)
    return active | (published != 0.0).any(dim=-1)


def no_stepping_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str = "twist",
    command_threshold: float = 0.01,
) -> torch.Tensor:
    """Penalize feet in the air when the commanded speed is below threshold.

    Discourages marching in place when the robot should stand still.

    Returns the count of airborne feet per environment (use with a negative weight).
    """
    command = env.command_manager.get_command(command_name)  # (N, 3)
    cmd_speed = torch.norm(command[:, :2], dim=-1) + torch.abs(command[:, 2])
    below_threshold = cmd_speed < command_threshold

    sensor = env.scene.sensors[sensor_name]
    found = sensor.data.found  # (N, num_feet) or (N, num_feet, num_slots)
    if found.dim() == 3:
        found = found.any(dim=-1)  # (N, num_feet)
    in_air = ~found.bool()

    return in_air.float().sum(dim=-1) * below_threshold.float()


def _single_foot_rows(height: torch.Tensor, lift_threshold: float) -> torch.Tensor:
    """Rows (N,) whose published foot targets (heights (N, 2), left and right)
    differ in z by ``lift_threshold`` or more: the higher foot is up, the lower
    one supports."""
    return (height[:, 0] - height[:, 1]).abs() >= lift_threshold


def _higher_foot(height: torch.Tensor) -> torch.Tensor:
    """Per foot (N, 2), whether its published target (heights (N, 2), left and
    right) is higher than the other one's: none on a row with equal heights."""
    return height > height.flip(-1)


def _standing_and_feet_down(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    foot_target_command_name: str,
    command_name: str,
    command_threshold: float,
    sensor_foot_ids: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The standing rows and, per foot of the foot target (left, right), its
    published target height, whether it should be down and whether it is.

    The feet that should be down:
      no foot target: both;
      a single-foot target: the support foot, unless its published target is
      still the higher one (handing over from a target on that foot);
      a two-foot target (both feet by one offset, the trunk lowers): both;
      published feet apart without a single-foot target (a target coming
      back down, or handing over to a two-foot one): all but the foot whose
      published target is higher.
    ``sensor_foot_ids`` are the sensor's indices of the foot target's (left,
    right) feet.

    Returns (standing (N,), height (N, 2), should_be_down (N, 2), down (N, 2)).
    """
    command = env.command_manager.get_command(command_name)  # (N, 3)
    cmd_speed = torch.norm(command[:, :2], dim=-1) + torch.abs(command[:, 2])
    standing = cmd_speed < command_threshold

    foot_target = env.command_manager.get_term(foot_target_command_name)
    height = foot_target.command.reshape(env.num_envs, 2, 3)[..., 2]
    should_be_down = ~_higher_foot(height)  # (N, 2)
    support = torch.arange(2, device=height.device) != foot_target.lifted_foot_idx[:, None]
    single = foot_target.is_single_support_env.bool()[:, None]
    should_be_down = torch.where(single, support & should_be_down, should_be_down)

    down = feet_down(env.scene.sensors[sensor_name], sensor_foot_ids)
    return standing, height, should_be_down, down


def feet_down(sensor, sensor_foot_ids: tuple[int, int]) -> torch.Tensor:
    """Per foot (N, 2), in the foot target's (left, right) order, whether the
    contact ``sensor`` finds it touching; ``sensor_foot_ids`` are the sensor's
    indices of those feet."""
    found = sensor.data.found  # (N, num_feet) or (N, num_feet, num_slots)
    if found.dim() == 3:
        found = found.any(dim=-1)  # (N, num_feet)
    return found.bool()[:, list(sensor_foot_ids)]


def lifted_support_feet(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    foot_target_command_name: str,
    command_name: str = "twist",
    command_threshold: float = 0.01,
    sensor_foot_ids: tuple[int, int] = (0, 1),
) -> torch.Tensor:
    """Penalize feet in the air that should be down, when the commanded speed is below threshold.

    The foot-target task's form of ``no_stepping_penalty``; the feet that
    should be down are those of ``_standing_and_feet_down``.

    Returns the count of those feet in the air per environment, 0 when the
    commanded speed is at or above threshold (use with a negative weight).
    """
    standing, _, should_be_down, down = _standing_and_feet_down(
        env, sensor_name, foot_target_command_name, command_name, command_threshold, sensor_foot_ids
    )
    return (~down & should_be_down).float().sum(dim=-1) * standing.float()


def upper_foot_unload(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    foot_target_command_name: str,
    lift_threshold: float,
    command_name: str = "twist",
    command_threshold: float = 0.01,
    sensor_foot_ids: tuple[int, int] = (0, 1),
    min_force: float = 0.1,
) -> torch.Tensor:
    """Reward the higher foot carrying the share of the weight its target asks.

    On a standing command (as ``lifted_support_feet``) whose two published
    foot targets differ in height by dz > 0, the higher foot
    (``_higher_foot``, never one ``lifted_support_feet`` asks down) carries
    the share s = Fz_up / (Fz_up + Fz_down) of the floor's vertical push on
    the feet.  The target share s* = 0.5 max(0, 1 - dz / lift_threshold) is
    even at dz = 0 and 0 (the higher foot carries nothing) from the
    threshold up, so a partial shift of the weight pays below and above it.
    Returns clamp(1 - 2 |s - s*|, 0, 1); 0 on other rows and when the feet
    carry less than ``min_force`` newtons together (both in the air).
    ``sensor_foot_ids`` are the sensor's indices of the foot target's (left,
    right) feet; the sensor's net force (world frame) is the foot's push on
    the floor, so the floor's push up on a foot is -z.
    """
    standing, height, _, _ = _standing_and_feet_down(
        env, sensor_name, foot_target_command_name, command_name, command_threshold, sensor_foot_ids
    )
    upper = _higher_foot(height)
    dz = (height[:, 0] - height[:, 1]).abs()
    target = 0.5 * torch.clamp(1.0 - dz / lift_threshold, min=0.0)

    force = env.scene.sensors[sensor_name].data.force[:, list(sensor_foot_ids), 2]
    load = torch.clamp(-force, min=0.0)  # (N, 2)
    total = load.sum(dim=-1)
    share = (load * upper).sum(dim=-1) / torch.clamp(total, min=min_force)
    reward = torch.clamp(1.0 - 2.0 * (share - target).abs(), 0.0, 1.0)
    return reward * (standing & (dz > 0.0) & (total >= min_force)).float()


def upper_foot_lift(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    foot_target_command_name: str,
    lift_threshold: float,
    command_name: str = "twist",
    command_threshold: float = 0.01,
    sensor_foot_ids: tuple[int, int] = (0, 1),
) -> torch.Tensor:
    """Reward the higher foot's height over the lower one, in proportion to its target's.

    On a standing command (as ``lifted_support_feet``) whose two published
    foot targets differ in height by dz >= ``lift_threshold``: the higher
    foot (``_higher_foot``) above the lower one, measured as the foot reward
    measures it (``left_from_right_level``, z), over dz, clamped to [0, 1],
    while the higher foot touches nothing and the lower one is down.  Each
    millimetre of lift pays the same; a heel raised with the toes on the
    floor and a hop on both feet pay nothing; 0 on other rows.
    ``sensor_foot_ids`` are the sensor's indices of the foot target's (left,
    right) feet.
    """
    standing, height, _, down = _standing_and_feet_down(
        env, sensor_name, foot_target_command_name, command_name, command_threshold, sensor_foot_ids
    )
    upper = _higher_foot(height)
    dz = (height[:, 0] - height[:, 1]).abs()
    left_over_right = env.command_manager.get_term(foot_target_command_name).left_from_right_level()[:, 2]
    rise = torch.where(upper[:, 0], left_over_right, -left_over_right)
    lift = torch.clamp(rise / torch.clamp(dz, min=1e-6), 0.0, 1.0)
    stance = ~(down & upper).any(dim=-1) & (down & ~upper).any(dim=-1)
    return lift * (standing & _single_foot_rows(height, lift_threshold) & stance).float()


########################## CURRICULUM #############################


class reward_based_staged_curriculum:
    """
    Curriculum based on stages ending while a reward component gets its mean 
    episode reward accross all environments above a threshold.

    "reward_term_name" can be a single term name, or a list of term names -- a
    stage with a list only advances once EVERY named term's mean episode reward
    has crossed its threshold (e.g. standing_pose and hip_pose, whose rewards
    rise at different rates).  "threshold" is then EITHER one number (applied
    to every named term) OR a list the same length as "reward_term_name" (one
    threshold per term, in the same order), since the terms' achievable
    ceilings differ.

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
        # Stay done once every stage has been applied.
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

    def state(self) -> dict:
        return {"stage": self.current_stage, "stage_first_step": int(self.stage_first_step),
                "rewards": {name: r.detach().cpu().clone() for name, r in self.rewards.items()}}

    def resume(self, env: ManagerBasedRlEnv, state: dict | None, stages: list[dict]) -> None:
        """Replay the stages a checkpoint passed and take back its per-env episode rewards.

        The rewards come back only with the same number of environments; with
        another, they start at 0 and refill as episodes end.
        """

        if state is None:
            raise ValueError("Checkpoint does not record the reward-gated curriculum state")
        for stage in stages[: state["stage"]]:
            stage["apply"](env)
        self.current_stage = state["stage"]
        self.stage_first_step = state["stage_first_step"]
        saved = state.get("rewards", {})
        if all(r.shape == (env.num_envs,) for r in saved.values()):
            self.rewards = {name: r.to(env.device).clone() for name, r in saved.items()}


class home_stillness_reward:
    """Reward the commanded joint TARGET (asset.data.joint_pos_target, not the
    measured joint_pos/joint_vel) holding still near home, but only once actually
    standing (head height above height_threshold) — a triple AND (standing,
    target-near-home, target-not-changing) via multiplication/gating, not a
    penalty gated by any one of these alone.

    Why a gated reward on the target:
    1. A joint-velocity PENALTY switched by a near-home threshold makes
       trembling worse: a step right at the boundary gives a policy sitting
       near it a reason to keep crossing back and forth.
    2. A penalty ramped by closeness pays 0 far from home regardless of
       velocity, an incentive not to converge onto home at all.
    3. The policy's RAW output (env.action_manager.action) is unbounded (RMS
       past 100 over an episode), so no fixed "worst case" constant for it is
       physical.
    So it reads asset.data.joint_pos_target: the ACTUAL, post-scale/offset/clip
    position each joint's PD controller is tracking right now (what apply_actions() in mjlab's JointPositionAction writes via
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
    simple binary threshold on head height (0 below it, full reward above it -- the
    same threshold every other "are we actually standing now" reward gates on).

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
        # HEAD_STANDING_THRESHOLD from the env cfg) -- no ramp, no separate minimum
        # fraction: exactly 0 below it, full pose_reward above it.
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

    Without it the policy reliably props itself up on its hands to roughly half
    height, then gets stuck there — it never lets go to complete the extension onto its feet alone.
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
    # Roll/pitch noise about the HOME trunk frame, yaw about world z, so a
    # leaning HOME keeps its flat soles at every heading.  A vertical-trunk
    # HOME keeps mjlab's term (its yaw axis is world z already).
    reset_root = (
        envs_mdp.reset_root_state_uniform
        if HOME_TRUNK_PITCH_RAD == 0.0
        else reset_root_state_uniform_world_yaw
    )
    reset_root(
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


