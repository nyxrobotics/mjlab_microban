"""Evaluate one bounded safe-velocity checkpoint on a fixed forward command."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.lab_api.math import quat_apply
from mjlab.utils.nan_guard import NanGuard

from mjlab_microban.tasks.microban_safe_velocity_checkpoint import (
    load_frozen_safe_velocity_actor,
)
from mjlab_microban.tasks.microban_safe_velocity_env_cfg import (
    MICROBAN_SAFE_VELOCITY_JOINT_NAMES,
    MICROBAN_SAFE_VELOCITY_OBSERVATION_SCHEMA,
    MICROBAN_SAFE_VELOCITY_OBSERVATION_WIDTH,
    make_microban_safe_velocity_env_cfg,
)
from mjlab_microban.tasks.microban_safe_velocity_mdp import (
    MICROBAN_SAFE_VELOCITY_GUARD_LOOKAHEAD_S,
    MICROBAN_SAFE_VELOCITY_GUARD_MARGIN_RATIO,
    preferred_joint_position_bounds,
)

DEFAULT_OUTPUT = Path("artifacts/microban_safe_velocity_checkpoint_gate.json")


class _TerminalStateRecorder(RecorderTerm):
    """Capture done-row physics immediately before the environment overwrites it."""

    def __init__(self, cfg: RecorderTermCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        self.cfg = cfg
        robot = env.scene["robot"]
        self.mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.joint_position = torch.empty_like(robot.data.joint_pos)
        self.joint_velocity = torch.empty_like(robot.data.joint_vel)
        self.root_position = torch.empty_like(robot.data.root_link_pos_w)
        self.root_pose = torch.empty_like(robot.data.root_link_pose_w)
        self.root_velocity = torch.empty_like(robot.data.root_link_vel_w)
        self.root_forward_velocity = torch.empty(
            env.num_envs, dtype=robot.data.root_link_lin_vel_b.dtype, device=env.device
        )
        self.reward = torch.empty(
            env.num_envs, dtype=robot.data.joint_pos.dtype, device=env.device
        )
        self.fell = torch.zeros_like(self.mask)
        self.physics_nonfinite = torch.zeros_like(self.mask)

    def begin_step(self) -> None:
        self.mask.zero_()

    def record_pre_reset(self, env_ids: torch.Tensor) -> None:
        robot = self._env.scene["robot"]
        self.mask[env_ids] = True
        self.joint_position[env_ids] = robot.data.joint_pos[env_ids]
        self.joint_velocity[env_ids] = robot.data.joint_vel[env_ids]
        self.root_position[env_ids] = robot.data.root_link_pos_w[env_ids]
        self.root_pose[env_ids] = robot.data.root_link_pose_w[env_ids]
        self.root_velocity[env_ids] = robot.data.root_link_vel_w[env_ids]
        self.root_forward_velocity[env_ids] = robot.data.root_link_lin_vel_b[env_ids, 0]
        self.reward[env_ids] = self._env.reward_buf[env_ids]
        self.fell[env_ids] = self._env.termination_manager.get_term("fell_over")[
            env_ids
        ]
        self.physics_nonfinite[env_ids] = NanGuard.detect_nans(self._env.sim.data)[
            env_ids
        ]


def _make_evaluation_env_cfg(
    *, num_envs: int, steps: int, command_vx_m_s: float, seed: int
) -> Any:
    cfg = make_microban_safe_velocity_env_cfg(play=True)
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    # Auto-reset is required to avoid public partial-reset calls advancing
    # delay buffers for still-active rows.  The recorder captures terminal
    # physics in record_pre_reset, before reset overwrites done rows.
    cfg.auto_reset = True
    cfg.recorders = {
        "safe_velocity_terminal_state": RecorderTermCfg(func=_TerminalStateRecorder)
    }
    cfg.episode_length_s = (steps + 2) * cfg.decimation * cfg.sim.mujoco.timestep
    command_cfg = cfg.commands["twist"]
    command_cfg.ranges.lin_vel_x = (command_vx_m_s, command_vx_m_s)
    command_cfg.ranges.lin_vel_y = (0.0, 0.0)
    command_cfg.ranges.ang_vel_z = (0.0, 0.0)
    command_cfg.resampling_time_range = (1.0e6, 1.0e6)
    return cfg


def _percentile(values: torch.Tensor, percentile: float, *, fallback: float) -> float:
    array = values.detach().to(dtype=torch.float64).cpu().numpy()
    array = array[np.isfinite(array)]
    if array.size == 0:
        return fallback
    return float(np.percentile(array, percentile, method="linear"))


def _publish_json(path: Path, report: Mapping[str, Any]) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            report, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
        )
        + "\n"
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=resolved.parent,
        prefix=f".{resolved.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(encoded)
        stream.flush()
    try:
        temporary.replace(resolved)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _finite_envs(named: Mapping[str, torch.Tensor], count: int) -> torch.Tensor:
    device = next(iter(named.values())).device
    result = torch.ones(count, dtype=torch.bool, device=device)
    for name, value in named.items():
        if value.shape[0] != count:
            raise ValueError(f"{name} has no environment-leading dimension")
        result &= torch.isfinite(value).reshape(count, -1).all(dim=-1)
    return result


def _require_fixed_forward_command(
    env: ManagerBasedRlEnv,
    *,
    count: int,
    command_vx_m_s: float,
    context: str,
) -> None:
    """Fail closed if command routing modifies the requested evaluation twist."""

    actual = env.command_manager.get_command("twist")
    expected_row = torch.tensor(
        (command_vx_m_s, 0.0, 0.0), dtype=actual.dtype, device=actual.device
    )
    expected = expected_row.expand(count, -1)
    if tuple(actual.shape) != (count, 3) or not bool(torch.equal(actual, expected)):
        maximum_error = (
            float(torch.abs(actual - expected).max().item())
            if tuple(actual.shape) == (count, 3)
            else math.inf
        )
        raise ValueError(
            f"Safe velocity {context} command drifted from exact fixed twist "
            f"({command_vx_m_s}, 0, 0); shape={tuple(actual.shape)}, "
            f"max_error={maximum_error}"
        )


def evaluate_safe_velocity_checkpoint(
    *,
    checkpoint: str | Path,
    expected_sha256: str | None,
    device: str,
    num_envs: int,
    steps: int,
    command_vx_m_s: float,
    seed: int,
) -> dict[str, Any]:
    """Run a deterministic first-episode gate and return JSON-safe evidence."""

    if num_envs < 1 or steps < 1:
        raise ValueError("num_envs and steps must be positive")
    if not math.isfinite(command_vx_m_s) or not 0.0 < command_vx_m_s <= 0.25:
        raise ValueError("command_vx_m_s must be finite and in (0, 0.25]")
    actor, identity = load_frozen_safe_velocity_actor(
        checkpoint, device=device, expected_sha256=expected_sha256
    )

    cfg = _make_evaluation_env_cfg(
        num_envs=num_envs,
        steps=steps,
        command_vx_m_s=command_vx_m_s,
        seed=seed,
    )
    env: ManagerBasedRlEnv | None = None
    wrapped: RslRlVecEnvWrapper | None = None
    try:
        env = ManagerBasedRlEnv(cfg=cfg, device=device)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
        recorder_terms = getattr(env.recorder_manager, "_terms", ())
        if len(recorder_terms) != 1 or not isinstance(
            recorder_terms[0], _TerminalStateRecorder
        ):
            raise TypeError("Safe velocity terminal recorder was not installed")
        terminal_recorder = recorder_terms[0]
        env.reset(seed=seed)
        _require_fixed_forward_command(
            env,
            count=num_envs,
            command_vx_m_s=command_vx_m_s,
            context="post-reset",
        )
        observations = wrapped.get_observations()
        if tuple(observations["actor"].shape) != (
            num_envs,
            MICROBAN_SAFE_VELOCITY_OBSERVATION_WIDTH,
        ):
            raise ValueError(
                f"Safe velocity actor observation shape drifted: "
                f"{tuple(observations['actor'].shape)}"
            )
        expected_terms = tuple(
            name for name, _width in MICROBAN_SAFE_VELOCITY_OBSERVATION_SCHEMA
        )
        if tuple(env.observation_manager.active_terms["actor"]) != expected_terms:
            raise ValueError("Safe velocity actor observation order drifted")

        action = env.action_manager.get_term("joint_pos")
        robot = env.scene["robot"]
        if tuple(action.target_names) != MICROBAN_SAFE_VELOCITY_JOINT_NAMES:
            raise ValueError("Safe velocity action joint order drifted")
        action_limits = robot.data.soft_joint_pos_limits[:, action.target_ids]
        expanded_clip = torch.broadcast_to(action._clip, action_limits.shape)
        clip_error = float(torch.abs(expanded_clip - action_limits).max().item())
        if not bool(
            torch.allclose(expanded_clip, action_limits, atol=1.0e-6, rtol=0.0)
        ):
            raise ValueError("Safe velocity absolute action clips drifted")
        scale = torch.broadcast_to(
            torch.as_tensor(
                action.scale, dtype=action.raw_action.dtype, device=env.device
            ),
            action.raw_action.shape,
        )
        offset = torch.broadcast_to(
            torch.as_tensor(
                action.offset, dtype=action.raw_action.dtype, device=env.device
            ),
            action.raw_action.shape,
        )
        default = robot.data.default_joint_pos[:, action.target_ids]
        if not bool(torch.equal(offset, default)):
            raise ValueError("Safe velocity action offset is not the robot default")
        preferred_lower, preferred_upper = preferred_joint_position_bounds(
            default,
            action_limits[..., 0],
            action_limits[..., 1],
            MICROBAN_SAFE_VELOCITY_GUARD_MARGIN_RATIO,
        )

        initial_root_pos = robot.data.root_link_pos_w.clone()
        initial_root_quat = robot.data.root_link_quat_w.clone()
        local_forward = torch.tensor(
            (1.0, 0.0, 0.0), dtype=torch.float32, device=env.device
        ).expand(num_envs, -1)
        forward_xy = quat_apply(initial_root_quat, local_forward)[:, :2]
        forward_xy /= torch.linalg.vector_norm(
            forward_xy, dim=-1, keepdim=True
        ).clamp_min(1.0e-9)

        active = torch.ones(num_envs, dtype=torch.bool, device=env.device)
        fell = torch.zeros_like(active)
        nonfinite = torch.zeros_like(active)
        forward_velocity_sum = torch.zeros(
            num_envs, dtype=torch.float64, device=env.device
        )
        forward_velocity_count = torch.zeros(
            num_envs, dtype=torch.long, device=env.device
        )
        forward_displacement = torch.zeros(
            num_envs, dtype=torch.float64, device=env.device
        )
        minimum_root_height = robot.data.root_link_pos_w[:, 2].clone()
        maximum_soft_violation = torch.zeros((), device=env.device)
        maximum_lookahead_soft_violation = torch.zeros((), device=env.device)
        maximum_preferred_lookahead_violation = torch.zeros((), device=env.device)
        maximum_target_clip = torch.zeros((), device=env.device)
        executed_steps = 0

        for _step in range(steps):
            if not bool(active.any().item()):
                break
            terminal_recorder.begin_step()
            active_before = active.clone()
            with torch.no_grad():
                bounded_action = actor(observations)
            if not isinstance(bounded_action, torch.Tensor):
                raise TypeError("Safe velocity actor did not return a tensor")
            action_finite = torch.isfinite(bounded_action).all(dim=-1)
            safe_action = bounded_action.clone()
            safe_action[~active_before | ~action_finite] = 0.0
            target = offset + scale * safe_action
            target_clip = torch.maximum(
                torch.clamp(action_limits[..., 0] - target, min=0.0),
                torch.clamp(target - action_limits[..., 1], min=0.0),
            )
            if bool(active_before.any().item()):
                maximum_target_clip = torch.maximum(
                    maximum_target_clip, target_clip[active_before].max()
                )

            observations, rewards, dones, _extras = wrapped.step(safe_action)
            executed_steps += 1
            _require_fixed_forward_command(
                env,
                count=num_envs,
                command_vx_m_s=command_vx_m_s,
                context=f"post-step-{executed_steps}",
            )
            terminal_mask = terminal_recorder.mask.clone()
            if not bool(torch.equal(terminal_mask, dones.bool())):
                raise RuntimeError(
                    "Safe velocity terminal recorder/done masks disagree"
                )
            measured_joint_position = robot.data.joint_pos.clone()
            measured_joint_velocity = robot.data.joint_vel.clone()
            measured_root_position = robot.data.root_link_pos_w.clone()
            measured_root_pose = robot.data.root_link_pose_w.clone()
            measured_root_velocity = robot.data.root_link_vel_w.clone()
            measured_forward_velocity = robot.data.root_link_lin_vel_b[:, 0].clone()
            measured_reward = rewards.clone()
            physics_nonfinite = NanGuard.detect_nans(env.sim.data)
            current_fell_term = env.termination_manager.get_term("fell_over").clone()
            if bool(terminal_mask.any().item()):
                measured_joint_position[terminal_mask] = (
                    terminal_recorder.joint_position[terminal_mask]
                )
                measured_joint_velocity[terminal_mask] = (
                    terminal_recorder.joint_velocity[terminal_mask]
                )
                measured_root_position[terminal_mask] = terminal_recorder.root_position[
                    terminal_mask
                ]
                measured_root_pose[terminal_mask] = terminal_recorder.root_pose[
                    terminal_mask
                ]
                measured_root_velocity[terminal_mask] = terminal_recorder.root_velocity[
                    terminal_mask
                ]
                measured_forward_velocity[terminal_mask] = (
                    terminal_recorder.root_forward_velocity[terminal_mask]
                )
                measured_reward[terminal_mask] = terminal_recorder.reward[terminal_mask]
                physics_nonfinite[terminal_mask] = terminal_recorder.physics_nonfinite[
                    terminal_mask
                ]
                current_fell_term[terminal_mask] = terminal_recorder.fell[terminal_mask]
            state_finite = _finite_envs(
                {
                    "actor_observation": observations["actor"],
                    "reward": measured_reward.unsqueeze(-1),
                    "joint_position": measured_joint_position,
                    "joint_velocity": measured_joint_velocity,
                    "root_pose": measured_root_pose,
                    "root_velocity": measured_root_velocity,
                },
                num_envs,
            )
            if bool(terminal_mask.any().item()):
                terminal_state_finite = _finite_envs(
                    {
                        "reward": measured_reward.unsqueeze(-1),
                        "joint_position": measured_joint_position,
                        "joint_velocity": measured_joint_velocity,
                        "root_pose": measured_root_pose,
                        "root_velocity": measured_root_velocity,
                    },
                    num_envs,
                )
                state_finite[terminal_mask] = terminal_state_finite[terminal_mask]
            step_nonfinite = active_before & (
                ~action_finite | ~state_finite | physics_nonfinite
            )
            nonfinite |= step_nonfinite
            valid = active_before & ~step_nonfinite

            minimum_root_height[valid] = torch.minimum(
                minimum_root_height[valid], measured_root_position[valid, 2]
            )
            current_fell = current_fell_term & valid
            current_fell |= valid & (measured_root_position[:, 2] < 0.10)
            fell |= current_fell

            actual_violation = torch.maximum(
                torch.clamp(
                    robot.data.soft_joint_pos_limits[..., 0] - measured_joint_position,
                    min=0.0,
                ),
                torch.clamp(
                    measured_joint_position - robot.data.soft_joint_pos_limits[..., 1],
                    min=0.0,
                ),
            )
            if bool(valid.any().item()):
                displacement_xy = (
                    measured_root_position[:, :2] - initial_root_pos[:, :2]
                )
                forward_displacement[valid] = (
                    (displacement_xy[valid] * forward_xy[valid])
                    .sum(dim=-1)
                    .to(torch.float64)
                )
                maximum_soft_violation = torch.maximum(
                    maximum_soft_violation, actual_violation[valid].max()
                )
                q = measured_joint_position[:, action.target_ids]
                qd = measured_joint_velocity[:, action.target_ids]
                projected = q + MICROBAN_SAFE_VELOCITY_GUARD_LOOKAHEAD_S * qd
                dangerous_lower = torch.minimum(q, projected)
                dangerous_upper = torch.maximum(q, projected)
                lookahead_soft_excess = torch.maximum(
                    torch.clamp(action_limits[..., 0] - dangerous_lower, min=0.0),
                    torch.clamp(dangerous_upper - action_limits[..., 1], min=0.0),
                )
                maximum_lookahead_soft_violation = torch.maximum(
                    maximum_lookahead_soft_violation,
                    lookahead_soft_excess[valid].max(),
                )
                lookahead_excess = torch.maximum(
                    torch.clamp(preferred_lower - dangerous_lower, min=0.0),
                    torch.clamp(dangerous_upper - preferred_upper, min=0.0),
                )
                maximum_preferred_lookahead_violation = torch.maximum(
                    maximum_preferred_lookahead_violation,
                    lookahead_excess[valid].max(),
                )
                forward_velocity_sum[valid] += measured_forward_velocity[valid].to(
                    torch.float64
                )
                forward_velocity_count[valid] += 1

            active &= ~dones.bool() & ~step_nonfinite & ~current_fell

        average_forward_velocity = torch.full(
            (num_envs,), float("nan"), dtype=torch.float64, device=env.device
        )
        measured = forward_velocity_count > 0
        average_forward_velocity[measured] = forward_velocity_sum[
            measured
        ] / forward_velocity_count[measured].to(torch.float64)
        completion_fraction = float(active.float().mean().item())
        fall_fraction = float(fell.float().mean().item())
        nonfinite_fraction = float(nonfinite.float().mean().item())
        forward_velocity_p05 = _percentile(
            average_forward_velocity, 5.0, fallback=-math.inf
        )
        forward_displacement_p05 = _percentile(
            forward_displacement, 5.0, fallback=-math.inf
        )
        maximum_preferred_lookahead_excess = float(
            maximum_preferred_lookahead_violation.item()
        )
        thresholds = {
            "completion_fraction_min": 0.95,
            "fall_fraction_max": 0.0,
            "nonfinite_fraction_max": 0.0,
            "forward_velocity_p05_m_s_min": 0.01,
            "forward_displacement_p05_m_min": 0.02,
            "actual_soft_limit_violation_rad_max": 1.0e-6,
            "actual_lookahead_soft_limit_violation_rad_max": 1.0e-6,
            "target_clip_rad_max": 1.0e-7,
        }
        measured_values = {
            "completion_fraction": completion_fraction,
            "fall_fraction": fall_fraction,
            "nonfinite_fraction": nonfinite_fraction,
            "forward_velocity_p05_m_s": forward_velocity_p05,
            "forward_displacement_p05_m": forward_displacement_p05,
            "maximum_actual_soft_limit_violation_rad": float(
                maximum_soft_violation.item()
            ),
            "maximum_actual_lookahead_soft_limit_violation_rad": float(
                maximum_lookahead_soft_violation.item()
            ),
            "maximum_preferred_margin_lookahead_excess_rad": (
                maximum_preferred_lookahead_excess
            ),
            "maximum_target_clip_rad": float(maximum_target_clip.item()),
        }
        checks = {
            "completion": completion_fraction >= thresholds["completion_fraction_min"],
            "no_falls": fall_fraction <= thresholds["fall_fraction_max"],
            "finite": nonfinite_fraction <= thresholds["nonfinite_fraction_max"],
            "forward_velocity": forward_velocity_p05
            >= thresholds["forward_velocity_p05_m_s_min"],
            "forward_displacement": forward_displacement_p05
            >= thresholds["forward_displacement_p05_m_min"],
            "actual_soft_limits": float(maximum_soft_violation.item())
            <= thresholds["actual_soft_limit_violation_rad_max"],
            "actual_lookahead_soft_limits": float(
                maximum_lookahead_soft_violation.item()
            )
            <= thresholds["actual_lookahead_soft_limit_violation_rad_max"],
            "absolute_target_clip": float(maximum_target_clip.item())
            <= thresholds["target_clip_rad_max"],
        }
        failed = sorted(name for name, passed in checks.items() if not passed)
        return {
            "schema_version": 2,
            "gate": "microban_safe_velocity_fixed_forward_v3",
            "checkpoint": {
                "path": str(identity.path),
                "sha256": identity.sha256,
                "iteration": identity.iteration,
                "checkpoint_schema_version": identity.schema_version,
                "recipe_revision": identity.recipe_revision,
                "actor_topology": list(identity.actor_topology),
                "actor_obs_normalization": identity.actor_obs_normalization,
                "observation_schema": [
                    list(item) for item in identity.observation_schema
                ],
                "action_joint_names": list(identity.action_joint_names),
            },
            "configuration": {
                "device": device,
                "num_envs": num_envs,
                "steps_requested": steps,
                "steps_executed": executed_steps,
                "seed": seed,
                "command_vx_m_s": command_vx_m_s,
                "actual_command_exactly_verified_each_step": True,
                "step_dt_s": env.step_dt,
                "absolute_clip_max_tensor_error_rad": clip_error,
                "guard_margin_ratio": MICROBAN_SAFE_VELOCITY_GUARD_MARGIN_RATIO,
                "guard_lookahead_s": MICROBAN_SAFE_VELOCITY_GUARD_LOOKAHEAD_S,
            },
            "metrics": {
                **measured_values,
                "forward_velocity_median_m_s": _percentile(
                    average_forward_velocity, 50.0, fallback=-math.inf
                ),
                "forward_displacement_median_m": _percentile(
                    forward_displacement, 50.0, fallback=-math.inf
                ),
                "minimum_root_height_m": float(minimum_root_height.min().item()),
            },
            "thresholds": thresholds,
            "checks": checks,
            "status": "pass" if not failed else "fail",
            "summary": {"passed": not failed, "failed_checks": failed},
        }
    finally:
        if wrapped is not None:
            wrapped.close()
        elif env is not None:
            env.close()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--command-vx", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = evaluate_safe_velocity_checkpoint(
        checkpoint=args.checkpoint,
        expected_sha256=args.expected_sha256,
        device=args.device,
        num_envs=args.num_envs,
        steps=args.steps,
        command_vx_m_s=args.command_vx,
        seed=args.seed,
    )
    _publish_json(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "failed_checks": report["summary"]["failed_checks"],
                "output": str(args.output.expanduser().resolve()),
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
