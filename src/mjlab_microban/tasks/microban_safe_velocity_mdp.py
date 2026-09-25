"""Safety-critical MDP helpers for Microban's bounded velocity policy.

The legacy velocity task intentionally remains unchanged.  This module owns the
separate bounded-action contract used by ``Mjlab-SafeVelocity-Microban``:

* PPO stores and scores the Gaussian latent while the simulator receives the
  monotonically bounded physical action;
* the deterministic actor starts at the configured home pose;
* position targets and measured joints are kept inside a default-preserving
  preferred margin; and
* the measured-state guard includes the 120 ms worst-case actuator-delay
  lookahead used by the real XC330 configuration.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from rsl_rl.algorithms import PPO
from torch import nn

from mjlab_microban.tasks.microban_teleop_mdp import (
    AsymmetricBoundedGaussianDistribution,
)

MICROBAN_SAFE_VELOCITY_ACTION_WIDTH = 18
MICROBAN_SAFE_VELOCITY_GUARD_MARGIN_RATIO = 0.05
MICROBAN_SAFE_VELOCITY_GUARD_LOOKAHEAD_S = 0.12
MICROBAN_SAFE_VELOCITY_RECIPE_INFO_KEY = "microban_safe_velocity_recipe_revision"
MICROBAN_SAFE_VELOCITY_RESUME_PARENT_INFO_KEY = "microban_safe_velocity_resume_parent"
MICROBAN_SAFE_VELOCITY_RECIPE_REVISION = (
    "scratch_bounded_inward_shoulder_sagittal_bodyprogress_v9"
)

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class MicrobanSafeVelocityBoundedGaussianDistribution(
    AsymmetricBoundedGaussianDistribution
):
    """Bounded Gaussian with a neutral body and inward shoulder-roll targets.

    Starting every output row at zero is important here: the shoulder-roll home
    positions have only about one degree of soft-limit headroom.  Generic random
    output rows can ask for a large outward target before PPO has observed a
    single transition.  The two shoulder-roll rows retain the parent's already
    validated inward latent biases, while every other row starts at exact zero.
    Small, per-joint stochastic exploration remains active.
    """

    def init_mlp_weights(self, mlp: nn.Module) -> None:
        super().init_mlp_weights(mlp)
        linear_layers = [
            module for module in mlp.modules() if isinstance(module, nn.Linear)
        ]
        if not linear_layers:
            raise ValueError("Safe velocity actor MLP has no linear output layer")
        output = linear_layers[-1]
        if output.out_features != MICROBAN_SAFE_VELOCITY_ACTION_WIDTH:
            raise ValueError("Safe velocity actor output must be exactly 18-wide")
        if output.bias is None:
            raise ValueError("Safe velocity actor output layer must have a bias")
        from mjlab_microban.tasks.microban_teleop_bootstrap import (
            TELEOP_SHOULDER_ROLL_ACTION_INDICES,
        )

        shoulder_ids = set(TELEOP_SHOULDER_ROLL_ACTION_INDICES)
        neutral_ids = [
            index
            for index in range(MICROBAN_SAFE_VELOCITY_ACTION_WIDTH)
            if index not in shoulder_ids
        ]
        with torch.no_grad():
            output.weight[neutral_ids] = 0.0
            output.bias[neutral_ids] = 0.0


class MicrobanSafeVelocityBoundedPPO(PPO):
    """Store PPO latents while sending only bounded actions to the environment."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        distribution = self.actor.distribution
        if not isinstance(
            distribution, MicrobanSafeVelocityBoundedGaussianDistribution
        ):
            raise TypeError(
                "MicrobanSafeVelocityBoundedPPO requires its bounded distribution"
            )
        if distribution.output_dim != MICROBAN_SAFE_VELOCITY_ACTION_WIDTH:
            raise ValueError("Safe velocity PPO requires exactly 18 actions")
        if self.symmetry is not None:
            raise ValueError("Safe velocity PPO does not support symmetry")
        if self.rnd is not None:
            raise ValueError("Safe velocity PPO does not support RND")
        distribution.project_std_parameters_()
        self._std_projection_hook_handle = self.optimizer.register_step_post_hook(
            self._project_std_after_optimizer_step
        )

    def _project_std_after_optimizer_step(
        self,
        _optimizer: torch.optim.Optimizer,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
    ) -> None:
        distribution = self.actor.distribution
        if not isinstance(
            distribution, MicrobanSafeVelocityBoundedGaussianDistribution
        ):
            raise TypeError("Safe velocity actor lost its bounded distribution")
        distribution.project_std_parameters_()

    def act(self, obs: Any) -> torch.Tensor:
        latent = super().act(obs)
        distribution = self.actor.distribution
        assert isinstance(distribution, MicrobanSafeVelocityBoundedGaussianDistribution)
        return distribution.to_environment_action(latent)


class MicrobanSafeVelocityOnPolicyRunner(VelocityOnPolicyRunner):
    """Persist and require the exact safe-velocity training recipe revision."""

    def save(self, path: str, infos: Mapping[str, Any] | None = None) -> None:
        checkpoint_infos = dict(infos or {})
        existing = checkpoint_infos.get(MICROBAN_SAFE_VELOCITY_RECIPE_INFO_KEY)
        if existing not in (None, MICROBAN_SAFE_VELOCITY_RECIPE_REVISION):
            raise ValueError(
                "Refusing to overwrite a different safe velocity recipe revision"
            )
        checkpoint_infos[MICROBAN_SAFE_VELOCITY_RECIPE_INFO_KEY] = (
            MICROBAN_SAFE_VELOCITY_RECIPE_REVISION
        )
        resume_parent = getattr(self, "_safe_velocity_resume_parent", None)
        if resume_parent is not None:
            checkpoint_infos[MICROBAN_SAFE_VELOCITY_RESUME_PARENT_INFO_KEY] = dict(
                resume_parent
            )
        super().save(path, checkpoint_infos)

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        source_path = Path(path).expanduser().resolve()
        payload = torch.load(source_path, map_location="cpu", weights_only=True)
        infos = payload.get("infos") if isinstance(payload, Mapping) else None
        recipe = (
            infos.get(MICROBAN_SAFE_VELOCITY_RECIPE_INFO_KEY)
            if isinstance(infos, Mapping)
            else None
        )
        if recipe != MICROBAN_SAFE_VELOCITY_RECIPE_REVISION:
            raise ValueError(
                "Safe velocity checkpoint recipe mismatch: "
                f"{recipe!r} != {MICROBAN_SAFE_VELOCITY_RECIPE_REVISION!r}"
            )
        env_state = infos.get("env_state") if isinstance(infos, Mapping) else None
        common_step_counter = (
            env_state.get("common_step_counter")
            if isinstance(env_state, Mapping)
            else None
        )
        source_iteration = payload.get("iter") if isinstance(payload, Mapping) else None
        if (
            not isinstance(source_iteration, int)
            or isinstance(source_iteration, bool)
            or source_iteration < 0
        ):
            raise ValueError("Safe velocity resume source has invalid iteration")
        if (
            not isinstance(common_step_counter, int)
            or isinstance(common_step_counter, bool)
            or common_step_counter < 0
        ):
            raise ValueError(
                "Safe velocity resume source has invalid common_step_counter"
            )
        expected_counter = (source_iteration + 1) * 24
        if common_step_counter != expected_counter:
            raise ValueError(
                "Safe velocity resume source iteration/step mismatch: "
                f"{common_step_counter} != {expected_counter}"
            )
        digest = hashlib.sha256()
        with source_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        loaded_infos = super().load(str(source_path), load_cfg, strict, map_location)
        # RSL-RL labels a checkpoint with the update index that just completed.
        # Continue at the next index so resumed runs cannot overwrite that
        # identity with different policy bytes.
        self.current_learning_iteration = source_iteration + 1
        self._safe_velocity_resume_parent = {
            "path": str(source_path),
            "sha256": digest.hexdigest(),
            "iteration": source_iteration,
        }
        # Simulation state is not checkpointed.  Reset the fresh vector only
        # after common_step_counter is restored so the curriculum applies every
        # overdue stage before commands and policy observations are sampled.
        self.env.unwrapped.reset()
        return loaded_infos


def _joint_position_action_tensors(
    env: ManagerBasedRlEnv,
    action_name: str,
) -> tuple[
    JointPositionAction,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    action = env.action_manager.get_term(action_name)
    if not isinstance(action, JointPositionAction):
        raise TypeError(f"{action_name!r} must be a JointPositionAction")
    if action.cfg.clip is None or not hasattr(action, "_clip"):
        raise ValueError(f"{action_name!r} must define absolute target clips")

    raw = action.raw_action
    scale = torch.as_tensor(action.scale, dtype=raw.dtype, device=raw.device)
    offset = torch.as_tensor(action.offset, dtype=raw.dtype, device=raw.device)
    scale = torch.broadcast_to(scale, raw.shape)
    offset = torch.broadcast_to(offset, raw.shape)
    clip = torch.broadcast_to(action._clip, (*raw.shape, 2))
    lower = clip[..., 0]
    upper = clip[..., 1]
    if not bool(
        torch.isfinite(raw).all()
        and torch.isfinite(scale).all()
        and torch.isfinite(offset).all()
        and torch.isfinite(clip).all()
    ):
        raise ValueError("safe velocity action tensors must be finite")
    if not bool(torch.all(scale > 0.0).item()):
        raise ValueError("safe velocity action scale must be positive")
    if not bool(torch.all(lower < upper).item()):
        raise ValueError("safe velocity action clips require lower < upper")
    target = offset + scale * raw
    return action, raw, target, lower, upper


def preferred_joint_position_bounds(
    default: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    margin_ratio: float = MICROBAN_SAFE_VELOCITY_GUARD_MARGIN_RATIO,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return an inward limit margin that never excludes the home position."""

    if not math.isfinite(margin_ratio) or not 0.0 < margin_ratio < 0.5:
        raise ValueError("margin_ratio must be finite and in (0, 0.5)")
    try:
        default, lower, upper = torch.broadcast_tensors(default, lower, upper)
    except RuntimeError as error:
        raise ValueError("default/lower/upper bounds are not broadcastable") from error
    if not bool(
        torch.isfinite(default).all()
        and torch.isfinite(lower).all()
        and torch.isfinite(upper).all()
    ):
        raise ValueError("preferred joint bounds must be finite")
    if not bool(torch.all(lower < upper).item()):
        raise ValueError("preferred joint bounds require lower < upper")
    if not bool(torch.all((default >= lower) & (default <= upper)).item()):
        raise ValueError("default joint positions must lie inside soft limits")
    span = upper - lower
    preferred_lower = torch.minimum(default, lower + margin_ratio * span)
    preferred_upper = torch.maximum(default, upper - margin_ratio * span)
    return preferred_lower, preferred_upper


def effective_action_after_absolute_clip(
    env: ManagerBasedRlEnv,
    action_name: str = "joint_pos",
) -> torch.Tensor:
    """Return the action actually represented by the absolute clipped target."""

    action, raw, target, lower, upper = _joint_position_action_tensors(env, action_name)
    offset = torch.broadcast_to(
        torch.as_tensor(action.offset, dtype=raw.dtype, device=raw.device), raw.shape
    )
    scale = torch.broadcast_to(
        torch.as_tensor(action.scale, dtype=raw.dtype, device=raw.device), raw.shape
    )
    return (torch.clamp(target, min=lower, max=upper) - offset) / scale


def preferred_target_margin_l1_sum(
    env: ManagerBasedRlEnv,
    action_name: str = "joint_pos",
    margin_ratio: float = MICROBAN_SAFE_VELOCITY_GUARD_MARGIN_RATIO,
) -> torch.Tensor:
    """Penalize absolute targets outside the default-preserving 5% margin."""

    action, raw, target, lower, upper = _joint_position_action_tensors(env, action_name)
    default = torch.broadcast_to(
        torch.as_tensor(action.offset, dtype=raw.dtype, device=raw.device), raw.shape
    )
    preferred_lower, preferred_upper = preferred_joint_position_bounds(
        default, lower, upper, margin_ratio
    )
    preferred = torch.clamp(target, min=preferred_lower, max=preferred_upper)
    return (torch.abs(target - preferred) / (0.5 * (upper - lower))).sum(dim=-1)


def measured_joint_margin_lookahead_l1_sum(
    env: ManagerBasedRlEnv,
    action_name: str = "joint_pos",
    margin_ratio: float = MICROBAN_SAFE_VELOCITY_GUARD_MARGIN_RATIO,
    lookahead_s: float = MICROBAN_SAFE_VELOCITY_GUARD_LOOKAHEAD_S,
) -> torch.Tensor:
    """Guard both ``q`` and ``q + lookahead_s * qdot`` against soft limits."""

    if not math.isfinite(lookahead_s) or lookahead_s < 0.0:
        raise ValueError("lookahead_s must be finite and non-negative")
    action, raw, _target, _clip_lower, _clip_upper = _joint_position_action_tensors(
        env, action_name
    )
    entity = action._entity
    joint_pos = entity.data.joint_pos[:, action.target_ids]
    joint_vel = entity.data.joint_vel[:, action.target_ids]
    limits = entity.data.soft_joint_pos_limits[:, action.target_ids]
    lower = limits[..., 0]
    upper = limits[..., 1]
    default = torch.broadcast_to(
        torch.as_tensor(action.offset, dtype=raw.dtype, device=raw.device), raw.shape
    )
    if not bool(
        torch.isfinite(joint_pos).all()
        and torch.isfinite(joint_vel).all()
        and torch.isfinite(limits).all()
    ):
        raise ValueError("measured joint guard inputs must be finite")
    preferred_lower, preferred_upper = preferred_joint_position_bounds(
        default, lower, upper, margin_ratio
    )
    projected = joint_pos + lookahead_s * joint_vel
    dangerous_lower = torch.minimum(joint_pos, projected)
    dangerous_upper = torch.maximum(joint_pos, projected)
    excess = torch.clamp(preferred_lower - dangerous_lower, min=0.0)
    excess += torch.clamp(dangerous_upper - preferred_upper, min=0.0)
    return (excess / (0.5 * (upper - lower))).sum(dim=-1)


def raw_action_l2(
    env: ManagerBasedRlEnv,
    action_name: str = "joint_pos",
) -> torch.Tensor:
    """Keep the bounded target delta close to the mechanically neutral action."""

    _action, raw, _target, _lower, _upper = _joint_position_action_tensors(
        env, action_name
    )
    return torch.square(raw).mean(dim=-1)


def planar_velocity_error_l1(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Non-saturating body-frame XY velocity tracking error."""

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    return torch.abs(command[:, :2] - asset.data.root_link_lin_vel_b[:, :2]).sum(dim=-1)


def planar_velocity_tracking_exp(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str = "twist",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Track body-frame XY velocity without penalizing gait vertical motion.

    Mjlab's generic linear-velocity reward adds squared base vertical velocity
    inside the same exponential.  At Microban's low command speeds and the
    sharpened 0.10 m/s scale, that suppresses the vertical motion needed to
    unload and lift a foot.  Vertical stability remains covered by the upright,
    pose, body-angular-velocity, contact, and fall terms.
    """

    if not math.isfinite(std) or std <= 0.0:
        raise ValueError("planar velocity tracking std must be finite and positive")
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error = torch.square(command[:, :2] - asset.data.root_link_lin_vel_b[:, :2]).sum(
        dim=-1
    )
    return torch.exp(-error / std**2)


def commanded_planar_velocity_progress(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    command_threshold: float = 0.01,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward commanded body-frame XY progress, bounded to ``[0, 1]``."""

    if not math.isfinite(command_threshold) or command_threshold <= 0.0:
        raise ValueError("command_threshold must be finite and positive")
    command = env.command_manager.get_command(command_name)
    if command is None or command.ndim != 2 or command.shape != (env.num_envs, 3):
        raise ValueError("planar velocity progress requires an (num_envs, 3) command")
    if not bool(torch.isfinite(command).all()):
        raise ValueError("planar velocity progress command must be finite")

    asset: Entity = env.scene[asset_cfg.name]
    actual = asset.data.root_link_lin_vel_b
    if actual.ndim != 2 or actual.shape != (env.num_envs, 3):
        raise ValueError(
            "planar velocity progress requires an (num_envs, 3) body velocity"
        )
    if not bool(torch.isfinite(actual).all()):
        raise ValueError("planar velocity progress body velocity must be finite")

    command_xy = command[:, :2]
    actual_xy = actual[:, :2]
    command_norm_sq = torch.square(command_xy).sum(dim=-1)
    aligned_progress = (actual_xy * command_xy).sum(dim=-1)
    if not bool(torch.isfinite(command_norm_sq).all()) or not bool(
        torch.isfinite(aligned_progress).all()
    ):
        raise ValueError("planar velocity progress arithmetic must remain finite")
    active_command = command_norm_sq > command_threshold**2
    aligned_fraction = aligned_progress / torch.clamp(
        command_norm_sq, min=command_threshold**2
    )
    reward = torch.clamp(aligned_fraction, min=0.0, max=1.0) * active_command
    env.extras["log"]["Metrics/commanded_planar_velocity_progress"] = reward.mean()
    return reward


def yaw_velocity_error_l1(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Non-saturating body-frame yaw-rate tracking error."""

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    return torch.abs(command[:, 2] - asset.data.root_link_ang_vel_b[:, 2])


def set_safe_velocity_command_ranges(
    env: ManagerBasedRlEnv,
    *,
    lin_vel_x: Sequence[float],
    lin_vel_y: Sequence[float] = (0.0, 0.0),
    ang_vel_z: Sequence[float] = (0.0, 0.0),
) -> None:
    """Apply one validated command-envelope curriculum stage."""

    def pair(name: str, values: Sequence[float]) -> tuple[float, float]:
        if len(values) != 2:
            raise ValueError(f"{name} must contain exactly two values")
        lower, upper = (float(values[0]), float(values[1]))
        if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
            raise ValueError(f"{name} must be a finite ordered pair")
        return lower, upper

    command = env.command_manager.get_term_cfg("twist")
    command.ranges.lin_vel_x = pair("lin_vel_x", lin_vel_x)
    command.ranges.lin_vel_y = pair("lin_vel_y", lin_vel_y)
    command.ranges.ang_vel_z = pair("ang_vel_z", ang_vel_z)


def _validate_safe_velocity_reward_overrides(
    overrides: Mapping[str, Mapping[str, Any]],
) -> None:
    """Validate the complete override mapping before any runtime mutation."""

    if not isinstance(overrides, Mapping):
        raise TypeError("Safe velocity reward overrides must be a mapping")
    allowed_fields = {
        "track_linear_velocity": {"weight", "std"},
        "linear_velocity_error_l1": {"weight"},
        "air_time": {"weight"},
    }
    for term_name, values in overrides.items():
        if term_name not in allowed_fields:
            raise ValueError(f"Unsupported safe velocity reward override: {term_name}")
        if not isinstance(values, Mapping) or not values:
            raise TypeError(
                f"Reward override {term_name!r} must be a non-empty mapping"
            )
        unexpected = set(values) - allowed_fields[term_name]
        if unexpected:
            raise ValueError(
                f"Unsupported {term_name!r} reward fields: {sorted(unexpected)}"
            )
        if "weight" in values:
            if isinstance(values["weight"], bool):
                raise TypeError(f"Reward weight for {term_name!r} must be numeric")
            weight = float(values["weight"])
            if not math.isfinite(weight):
                raise ValueError(f"Reward weight for {term_name!r} must be finite")
        if "std" in values:
            if isinstance(values["std"], bool):
                raise TypeError("track_linear_velocity std must be numeric")
            std = float(values["std"])
            if not math.isfinite(std) or std <= 0.0:
                raise ValueError(
                    "track_linear_velocity std must be finite and positive"
                )


def set_safe_velocity_reward_overrides(
    env: ManagerBasedRlEnv,
    overrides: Mapping[str, Mapping[str, Any]],
) -> None:
    """Apply atomically validated runtime reward changes for the curriculum."""

    _validate_safe_velocity_reward_overrides(overrides)
    for term_name, values in overrides.items():
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        if term_cfg is None:
            raise ValueError(f"Safe velocity reward term is unavailable: {term_name}")
        if "weight" in values:
            term_cfg.weight = float(values["weight"])
        if "std" in values:
            std = float(values["std"])
            term_cfg.params["std"] = std


class SafeVelocityStagedCurriculum:
    """Resume-safe, ordered step curriculum for the scratch velocity policy."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv) -> None:
        del env
        stages = cfg.params.get("stages")
        if not isinstance(stages, list):
            raise TypeError("safe velocity curriculum requires a stage list")
        previous_step = -1
        for index, stage in enumerate(stages):
            if not isinstance(stage, dict):
                raise TypeError(f"safe velocity stage {index} must be a dictionary")
            if set(stage) != {"name", "step", "ranges", "reward_overrides"}:
                raise ValueError(
                    "safe velocity stage "
                    f"{index} must contain name/step/ranges/reward_overrides"
                )
            step = stage["step"]
            if not isinstance(step, int) or isinstance(step, bool) or step < 0:
                raise ValueError(f"safe velocity stage {index} has invalid step")
            if step <= previous_step:
                raise ValueError("safe velocity stage steps must increase")
            if not isinstance(stage["ranges"], dict):
                raise TypeError(f"safe velocity stage {index} ranges must be a dict")
            if not isinstance(stage["reward_overrides"], dict):
                raise TypeError(
                    f"safe velocity stage {index} reward_overrides must be a dict"
                )
            _validate_safe_velocity_reward_overrides(stage["reward_overrides"])
            previous_step = step
        self.current_stage = 0

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor,
        stages: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        del env_ids
        while (
            self.current_stage < len(stages)
            and env.common_step_counter >= stages[self.current_stage]["step"]
        ):
            stage = stages[self.current_stage]
            set_safe_velocity_command_ranges(env, **stage["ranges"])
            set_safe_velocity_reward_overrides(env, stage["reward_overrides"])
            print(
                "Safe velocity curriculum stage "
                f"{self.current_stage + 1}: {stage['name']} at step "
                f"{env.common_step_counter}"
            )
            self.current_stage += 1
        return {
            "stage": torch.tensor(
                self.current_stage, dtype=torch.float32, device=env.device
            )
        }
