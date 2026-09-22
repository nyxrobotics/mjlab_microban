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
from mjlab.utils.lab_api.math import quat_apply_inverse, subtract_frame_transforms
from mjlab.tasks.velocity.mdp.velocity_command import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)


############################ COMMANDS #############################

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

    def _resample_command(self, env_ids: torch.Tensor) -> None:
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

    def build(self, env: ManagerBasedRlEnv) -> UniformVelocityCommandWithRotation:
        return UniformVelocityCommandWithRotation(self, env)
    

class FootTargetCommand(CommandTerm):
    """Live per-env target offset (dx, dy, dz) for each foot, relative to that foot's
    own position measured right after reset (i.e. the robot's default/home stance),
    expressed in the trunk frame.

    This drives the "whole-body tracking" leg behavior: trained alongside the existing
    velocity command so ONE policy learns both walking and holding a commanded foot
    pose (including single-leg stances), rather than switching to a kinematic-IK leg
    controller when idle — kinematic IK can't provide the continuous balance
    correction a single-leg stance needs (see microban_teleop/docs/retargeting_research.md).

    Most sampled episodes keep both targets at zero offset (normal stance). A fraction
    lift one foot (randomly chosen) to train single-support balance.
    """

    cfg: "FootTargetCommandCfg"

    def __init__(self, cfg: "FootTargetCommandCfg", env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.entity_name]

        self._foot_asset_cfg = SceneEntityCfg(cfg.entity_name, site_names=cfg.foot_site_names)
        self._foot_asset_cfg.resolve(env.scene)

        # Offset target (dx, dy, dz) per env, per foot (left, right), in the trunk frame.
        self.foot_target_offset_b = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.is_single_support_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.lifted_foot_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Per-episode reference ("zero offset") foot position, snapshotted at reset time
        # (see _resample_command) since it depends on wherever reset_robot_joints landed.
        self._default_foot_pos_b = torch.zeros(self.num_envs, 2, 3, device=self.device)

        self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.foot_target_offset_b.view(self.num_envs, -1)

    def current_foot_pos_b(self) -> torch.Tensor:
        """Live foot positions (left, right) in the trunk frame. Shape (N, 2, 3)."""
        trunk_pos_w = self.robot.data.root_link_pos_w
        trunk_quat_w = self.robot.data.root_link_quat_w
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
            torch.square(self.current_foot_pos_b() - self._default_foot_pos_b - self.foot_target_offset_b),
            dim=-1,
        )
        self.metrics["error_pos"] += error.mean(-1)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        # Snapshot each foot's current (post-reset, default-stance) position as this
        # episode's zero-offset reference.
        if not hasattr(self, "_default_foot_pos_b"):
            self._default_foot_pos_b = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self._default_foot_pos_b[env_ids] = self.current_foot_pos_b()[env_ids]

        self.foot_target_offset_b[env_ids] = 0.0

        r = torch.empty(len(env_ids), device=self.device)
        self.is_single_support_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_single_support_envs

        support_ids = env_ids[self.is_single_support_env[env_ids]]
        if len(support_ids) == 0:
            return

        r2 = torch.empty(len(support_ids), device=self.device)
        self.lifted_foot_idx[support_ids] = (r2.uniform_(0.0, 1.0) < 0.5).long()

        dx = torch.empty(len(support_ids), device=self.device).uniform_(*self.cfg.reach_xy_range)
        dy = torch.empty(len(support_ids), device=self.device).uniform_(*self.cfg.reach_xy_range)
        dz = torch.empty(len(support_ids), device=self.device).uniform_(*self.cfg.lift_height_range)

        offsets = torch.stack([dx, dy, dz], dim=-1)  # (n, 3)
        self.foot_target_offset_b[support_ids, self.lifted_foot_idx[support_ids], :] = offsets

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

    def build(self, env: ManagerBasedRlEnv) -> FootTargetCommand:
        return FootTargetCommand(self, env)


class HandTargetCommand(CommandTerm):
    """Live per-env, per-hand target offset (dx, dy, dz), relative to that hand's own
    position measured right after reset, in the trunk frame.

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
    """

    cfg: "HandTargetCommandCfg"

    def __init__(self, cfg: "HandTargetCommandCfg", env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.entity_name]

        self._hand_asset_cfg = SceneEntityCfg(cfg.entity_name, site_names=cfg.hand_site_names)
        self._hand_asset_cfg.resolve(env.scene)

        self.hand_target_offset_b = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self._default_hand_pos_b = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.is_active = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)

        self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return torch.cat(
            [self.hand_target_offset_b.view(self.num_envs, -1), self.is_active.float()],
            dim=-1,
        )

    def current_hand_pos_b(self) -> torch.Tensor:
        """Live hand positions (left, right) in the trunk frame. Shape (N, 2, 3)."""
        trunk_pos_w = self.robot.data.root_link_pos_w
        trunk_quat_w = self.robot.data.root_link_quat_w
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
            torch.square(self.current_hand_pos_b() - self._default_hand_pos_b - self.hand_target_offset_b),
            dim=-1,
        )
        active = self.is_active.float()
        self.metrics["error_pos"] += (error * active).sum(-1) / active.sum(-1).clamp(min=1.0)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        self._default_hand_pos_b[env_ids] = self.current_hand_pos_b()[env_ids]

        r = torch.empty(len(env_ids), 2, device=self.device)
        self.is_active[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_active

        offsets = torch.zeros(len(env_ids), 2, 3, device=self.device)
        lo_xy, hi_xy = self.cfg.reach_xy_range
        lo_z, hi_z = self.cfg.reach_z_range
        r2 = torch.empty(len(env_ids), 2, 3, device=self.device)
        r2[..., 0].uniform_(lo_xy, hi_xy)
        r2[..., 1].uniform_(lo_xy, hi_xy)
        r2[..., 2].uniform_(lo_z, hi_z)
        active_mask = self.is_active[env_ids].unsqueeze(-1)
        offsets = torch.where(active_mask, r2, offsets)
        self.hand_target_offset_b[env_ids] = offsets

    def _update_command(self) -> None:
        pass


@dataclass(kw_only=True)
class HandTargetCommandCfg(CommandTermCfg):
    """Configuration for HandTargetCommand."""

    entity_name: str = "robot"
    hand_site_names: tuple[str, str] = ("left_hand", "right_hand")
    reach_xy_range: tuple[float, float] = (-0.08, 0.08)
    reach_z_range: tuple[float, float] = (-0.08, 0.08)
    rel_active: float = 0.7
    """Per-hand probability of being active at each resample (independent left/right,
    matching each controller's own trigger). Inactive hands contribute nothing to the
    tracking reward, so the policy learns that arm is free to move naturally."""

    def build(self, env: ManagerBasedRlEnv) -> HandTargetCommand:
        return HandTargetCommand(self, env)


########################## OBSERVATIONS ############################

def foot_target_offset_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Flattened (left_dx, left_dy, left_dz, right_dx, right_dy, right_dz) target foot
    offset, in the trunk frame. Lets the actor see what leg-tracking target (if any) is
    currently commanded, alongside the velocity command."""
    command: FootTargetCommand = env.command_manager.get_term(command_name)
    return command.command


def hand_target_offset_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Flattened (left_dx, left_dy, left_dz, right_dx, right_dy, right_dz) target hand
    offset, in the trunk frame."""
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


def getup_height_reward(
    env: ManagerBasedRlEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Dense reward proportional to trunk height, capped once standing height is
    reached. Rewards progress toward standing from any starting orientation, without
    assuming which direction is "up" relative to the trunk (unlike upright, which needs
    the trunk already close to vertical to give a useful gradient)."""
    asset: Entity = env.scene[asset_cfg.name]
    height = asset.data.root_link_pos_w[:, 2]
    return torch.clamp(height / target_height, min=0.0, max=1.0)


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
            command.current_foot_pos_b() - command._default_foot_pos_b - command.foot_target_offset_b
        ),
        dim=-1,
    ).mean(-1)
    tracking_reward = torch.exp(-error / std**2)

    velocity_command = env.command_manager.get_command(velocity_command_name)
    speed = torch.norm(velocity_command[:, :2], dim=-1) + torch.abs(velocity_command[:, 2])
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
            command.current_hand_pos_b() - command._default_hand_pos_b - command.hand_target_offset_b
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
) -> torch.Tensor:
    """
    Penalizes feet in the air when the commanded speed is below threshold.
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
        env.reward_manager.get_term_cfg("no_stepping").weight = no_stepping_penalty_weight
    if rel_standing_envs is not None:
        env.command_manager.get_term_cfg("twist").rel_standing_envs = rel_standing_envs
    if rel_rotation_envs is not None:
        env.command_manager.get_term_cfg("twist").rel_rotation_envs = rel_rotation_envs

def randomize_upper_body_pose(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
) -> None:
    """Pose the given joints (arms/neck/head) to a random position within their own
    soft limits at reset, and set their position actuators to hold there for the rest
    of the episode.

    These joints aren't RL-controlled — they're driven mathematically (IK for the
    arms, the closed-form stabilization law for the neck) once deployed. This event
    approximates that during training: instead of a fixed default pose, the legs learn
    to balance under a different, but per-episode-fixed, arm/neck configuration each
    time, so the walking policy is robust to whatever pose the operator's tracking
    happens to be commanding rather than only ever having seen one.

    asset_cfg must resolve both joint_names and actuator_names to the SAME set of
    joints (in corresponding order — each actuator drives its own like-named joint).
    """
    asset: Entity = env.scene[asset_cfg.name]

    joint_ids = asset_cfg.joint_ids
    limits = asset.data.soft_joint_pos_limits[env_ids][:, joint_ids]
    r = torch.rand(limits.shape[0], limits.shape[1], device=env.device)
    pose = limits[..., 0] + r * (limits[..., 1] - limits[..., 0])
    zero_vel = torch.zeros_like(pose)

    asset.write_joint_state_to_sim(pose, zero_vel, joint_ids=joint_ids, env_ids=env_ids)
    asset.write_ctrl_to_sim(pose, ctrl_ids=asset_cfg.actuator_ids, env_ids=env_ids)


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
        env.reward_manager.get_term_cfg("no_stepping").weight = no_stepping_penalty_weight
        env.command_manager.get_term_cfg("twist").rel_standing_envs = rel_standing_envs
        env.command_manager.get_term_cfg("twist").rel_rotation_envs = rel_rotation_envs

    return {
        "air_time_weight": torch.tensor(env.reward_manager.get_term_cfg("air_time").weight),
        "no_stepping_penalty_weight": torch.tensor(env.reward_manager.get_term_cfg("no_stepping").weight),
        "rel_standing_envs": torch.tensor(env.command_manager.get_term_cfg("twist").rel_standing_envs),
        "rel_rotation_envs": torch.tensor(env.command_manager.get_term_cfg("twist").rel_rotation_envs),
    }
