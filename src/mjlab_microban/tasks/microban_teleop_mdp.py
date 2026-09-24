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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.curriculum_manager import CurriculumTermCfg
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


class ResumeSafeStepBasedStagedCurriculum:
    """Apply every curriculum stage due at the environment's global step.

    MjLab persists ``common_step_counter`` in a checkpoint, but manager-term
    instances are rebuilt when a run resumes.  A single-stage ``if`` therefore
    leaves a resumed environment temporarily (or permanently, if no episode
    resets) at stage zero.  Advancing in a ``while`` loop reconstructs the exact
    stage-derived reward and command configuration in one manager compute call.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv) -> None:
        del env
        stages = cfg.params.get("stages")
        if not isinstance(stages, list):
            raise TypeError("Resume-safe curriculum requires a list of stages")

        previous_step = -1
        for index, stage in enumerate(stages):
            if not isinstance(stage, dict):
                raise TypeError(f"Curriculum stage {index} must be a dictionary")
            missing = {"name", "step", "apply"} - stage.keys()
            if missing:
                raise ValueError(
                    f"Curriculum stage {index} is missing {sorted(missing)}"
                )
            step = stage["step"]
            if not isinstance(step, int) or isinstance(step, bool) or step < 0:
                raise ValueError(
                    f"Curriculum stage {index} step must be a non-negative integer"
                )
            if step <= previous_step:
                raise ValueError("Curriculum stage steps must be strictly increasing")
            if not callable(stage["apply"]):
                raise TypeError(f"Curriculum stage {index} apply must be callable")
            previous_step = step

        self.current_stage = 0

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | slice,
        stages: list[dict[str, Any]],
    ) -> dict[str, int]:
        del env_ids
        while (
            self.current_stage < len(stages)
            and env.common_step_counter >= stages[self.current_stage]["step"]
        ):
            stage = stages[self.current_stage]
            print(
                f"Curriculum stage {self.current_stage + 1}: "
                f"{stage['name']} at step {env.common_step_counter}"
            )
            stage["apply"](env)
            self.current_stage += 1
        return {"stage": self.current_stage}


class ResetFixedFootTargetCommand(FootTargetCommand):
    """Foot offsets whose trunk-frame zero is fixed for one episode.

    ``CommandTerm`` also calls ``_resample_command`` when its timer expires.
    The shared Microban term snapshots its reference in that method, which makes
    the meaning of a zero offset drift every 3--8 seconds.  This task-specific
    term captures the reference only after an episode reset.  Capture is deferred
    to the first command-manager compute so MuJoCo forward kinematics already
    reflects the newly reset joint state.
    """

    def __init__(self, cfg: "ResetFixedFootTargetCommandCfg", env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._reference_pending = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        if not isinstance(env_ids, torch.Tensor):
            raise TypeError("Foot target reset requires explicit environment IDs")
        self._reference_pending[env_ids] = True
        return super().reset(env_ids)

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

    def _update_metrics(self) -> None:
        self._capture_pending_reference()
        super()._update_metrics()

    def _update_command(self) -> None:
        self._capture_pending_reference()


@dataclass(kw_only=True)
class ResetFixedFootTargetCommandCfg(FootTargetCommandCfg):
    """Configuration for episode-reset-fixed foot targets."""

    def build(self, env: ManagerBasedRlEnv) -> ResetFixedFootTargetCommand:
        return ResetFixedFootTargetCommand(self, env)


class ResetFixedHandTargetCommand(HandTargetCommand):
    """Hand offsets whose trunk-frame zero is fixed for one episode."""

    def __init__(self, cfg: "ResetFixedHandTargetCommandCfg", env: ManagerBasedRlEnv):
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
                "HMD joint order must be "
                f"{MICROBAN_HMD_JOINT_NAMES}, got {names}"
            )
        if not isinstance(asset_cfg.joint_ids, list):
            raise ValueError("HMD joint selection must resolve to three explicit IDs")

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
        self.position_lower = torch.maximum(
            configured_lower, soft_limits[..., 0]
        )
        self.position_upper = torch.minimum(
            configured_upper, soft_limits[..., 1]
        )
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
        if not (
            0.0 < self.retarget_interval_s[0] <= self.retarget_interval_s[1]
        ):
            raise ValueError("HMD retarget interval must be finite and positive")
        if not all(math.isfinite(value) for value in self.retarget_interval_s):
            raise ValueError("HMD retarget interval must be finite")

        self.neutral_probability = float(params.get("neutral_probability", 0.2))
        if not 0.0 <= self.neutral_probability <= 1.0:
            raise ValueError("neutral_probability must be in [0, 1]")

        default = self.asset.data.default_joint_pos[:, self.joint_ids]
        self.current_target = torch.clamp(
            default.clone(), min=self.position_lower, max=self.position_upper
        )
        self.goal_target = self.current_target.clone()
        # Zero causes a waypoint to be sampled on the first post-reset step.
        self.time_to_retarget_s = torch.zeros(
            env.num_envs, dtype=torch.float32, device=env.device
        )

    def _explicit_env_ids(
        self, env_ids: torch.Tensor | slice | None
    ) -> torch.Tensor:
        if env_ids is None or isinstance(env_ids, slice):
            return torch.arange(
                self.current_target.shape[0],
                dtype=torch.long,
                device=self.current_target.device,
            )[env_ids if isinstance(env_ids, slice) else slice(None)]
        return env_ids.to(device=self.current_target.device, dtype=torch.long)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        indices = self._explicit_env_ids(env_ids)
        default = self.asset.data.default_joint_pos[indices][:, self.joint_ids]
        default = torch.clamp(
            default,
            min=self.position_lower[indices],
            max=self.position_upper[indices],
        )
        self.current_target[indices] = default
        self.goal_target[indices] = default
        self.time_to_retarget_s[indices] = 0.0
        self.asset.set_joint_position_target(
            default,
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
            neutral = self.asset.data.default_joint_pos[indices][:, self.joint_ids]
            sampled = torch.where(neutral_mask, neutral, sampled)

        self.goal_target[indices] = torch.clamp(
            sampled, min=lower, max=upper
        )
        minimum, maximum = self.retarget_interval_s
        self.time_to_retarget_s[indices] = (
            torch.rand(count, device=self.current_target.device)
            * (maximum - minimum)
            + minimum
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        **_unused: Any,
    ) -> None:
        # Step events are always dispatched for all environments.  Per-env
        # episode resets are handled by reset(), which only rewrites those rows.
        if env_ids is not None:
            raise ValueError("HmdNeckTargetMotion step event expects env_ids=None")

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
