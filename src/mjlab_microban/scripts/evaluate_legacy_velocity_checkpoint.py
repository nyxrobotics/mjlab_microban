"""Diagnose the audited legacy walk actor in its original play environment.

This is deliberately not a deployment gate.  It answers whether the separately
trained velocity policy actually walks, turns and lifts both feet before any
teleoperation safety projection changes its outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.utils.lab_api.math import quat_apply
from mjlab.utils.nan_guard import NanGuard
from mjlab.utils.torch import configure_torch_backends

from mjlab_microban.legacy_velocity_diagnostics import (
    DEFAULT_LEGACY_VELOCITY_CHECKPOINT,
    DEFAULT_LEGACY_VELOCITY_SHA256,
    TwistScenario,
    build_report,
    checkpoint_sha256,
    default_scenarios,
    publish_json_atomic,
    resolve_output_path,
    select_scenarios,
    summarize_samples,
)
from mjlab_microban.tasks.microban_velocity_env_cfg import (
    MicrobanVelocityRlCfg,
    make_microban_velocity_env_cfg,
)

_FOOT_HEIGHT_ORDER = ("left", "right")
_CONTACT_BODY_TO_SIDE = {"foot": "right", "foot_2": "left"}
_LEGACY_ACTION_OBSERVATION_SLICE = slice(42, 60)


def _configure_environment(*, steps: int, seed: int) -> Any:
    cfg = make_microban_velocity_env_cfg(play=True)
    cfg.scene.num_envs = 1
    cfg.seed = seed
    cfg.auto_reset = False
    cfg.episode_length_s = (steps + 2) * cfg.decimation * cfg.sim.mujoco.timestep
    command = cfg.commands["twist"]
    command.debug_vis = False
    command.heading_command = False
    command.ranges.heading = None
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.rel_world_envs = 0.0
    command.rel_forward_envs = 0.0
    command.rel_rotation_envs = 0.0
    command.init_velocity_prob = 0.0
    command.ranges.lin_vel_x = (0.0, 0.0)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.resampling_time_range = (1.0e6, 1.0e6)
    return cfg


def _contact_indices(sensor: Any) -> dict[str, int]:
    primary_names = [
        slot.primary_name for slot in sensor._slots if slot.field_name == "found"
    ]
    if set(primary_names) != set(_CONTACT_BODY_TO_SIDE):
        raise ValueError(f"Unexpected foot contact layout: {primary_names}")
    return {
        _CONTACT_BODY_TO_SIDE[name]: index
        for index, name in enumerate(primary_names)
    }


def _termination_names(env: ManagerBasedRlEnv) -> list[str]:
    return [
        name
        for name in env.termination_manager.active_terms
        if bool(env.termination_manager.get_term(name)[0].item())
    ]


def _all_finite(named: dict[str, torch.Tensor]) -> tuple[bool, str | None]:
    for name, value in named.items():
        if not bool(torch.isfinite(value).all().item()):
            return False, name
    return True, None


def _parameter_tensor(value: torch.Tensor | float, reference: torch.Tensor) -> torch.Tensor:
    return torch.broadcast_to(
        torch.as_tensor(value, dtype=reference.dtype, device=reference.device),
        reference.shape,
    )


def _evaluate_scenario(
    *,
    env: ManagerBasedRlEnv,
    wrapped: RslRlVecEnvWrapper,
    policy: Any,
    scenario: TwistScenario,
    steps: int,
    settle_steps: int,
    seed: int,
    execute_soft_limit_projection: bool,
) -> dict[str, Any]:
    env.reset(seed=seed)
    command = env.command_manager.get_term("twist")
    expected_command = torch.tensor(
        scenario.twist, dtype=command.vel_command_b.dtype, device=env.device
    ).unsqueeze(0)
    command.vel_command_b[:] = expected_command
    observations = wrapped.get_observations()
    if not bool(torch.equal(command.vel_command_b, expected_command)):
        raise RuntimeError("Fixed legacy velocity command was not installed")

    robot = env.scene["robot"]
    action = env.action_manager.get_term("joint_pos")
    if action.cfg.clip is not None:
        raise ValueError("Original velocity play task unexpectedly clips action targets")
    if wrapped.clip_actions is not None:
        raise ValueError("Original velocity runner unexpectedly clips policy actions")
    soft_limits = robot.data.soft_joint_pos_limits[:, action.target_ids]
    scale = _parameter_tensor(action.scale, action.raw_action)
    offset = _parameter_tensor(action.offset, action.raw_action)
    if tuple(soft_limits.shape) != (*action.raw_action.shape, 2):
        raise ValueError("Legacy velocity action/soft-limit dimensions disagree")

    foot_contact = env.scene.sensors["feet_ground_contact"]
    foot_height = env.scene.sensors["foot_height_scan"]
    contact_indices = _contact_indices(foot_contact)
    height_frame_names = tuple(ref.name for ref in foot_height.cfg.frame)
    if height_frame_names != ("left_foot", "right_foot"):
        raise ValueError(f"Unexpected foot-height frame order: {height_frame_names}")
    if foot_contact.data.found is None:
        raise RuntimeError("Foot contact sensor does not expose found")
    previous_contact = {
        side: bool(foot_contact.data.found[0, index].item() > 0)
        for side, index in contact_indices.items()
    }

    initial_position = robot.data.root_link_pos_w[0].clone()
    initial_quaternion = robot.data.root_link_quat_w[0].clone()
    local_axes = torch.tensor(
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=initial_position.dtype,
        device=env.device,
    )
    initial_axes_world = quat_apply(
        initial_quaternion.unsqueeze(0).expand(2, -1), local_axes
    )[:, :2]
    fall_height_m = float(
        env.termination_manager.get_term_cfg("fell_over").params["minimum_height"]
    )

    velocity_samples = {name: [] for name in ("vx_m_s", "vy_m_s", "yaw_rad_s")}
    velocity_error_samples = {
        name: [] for name in ("vx_m_s", "vy_m_s", "yaw_rad_s")
    }
    linear_vector_error: list[float] = []
    root_heights: list[float] = []
    foot_heights = {side: [] for side in _FOOT_HEIGHT_ORDER}
    foot_airborne_steps = {side: 0 for side in _FOOT_HEIGHT_ORDER}
    foot_takeoffs = {side: 0 for side in _FOOT_HEIGHT_ORDER}
    foot_landings = {side: 0 for side in _FOOT_HEIGHT_ORDER}
    foot_ever_airborne = {side: False for side in _FOOT_HEIGHT_ORDER}
    foot_max_current_air = {side: 0.0 for side in _FOOT_HEIGHT_ORDER}
    foot_max_last_air = {side: 0.0 for side in _FOOT_HEIGHT_ORDER}
    both_airborne_steps = 0
    exactly_one_airborne_steps = 0
    integrated_yaw_rad = 0.0
    target_values = 0
    projected_target_values = 0
    projected_target_steps = 0
    maximum_target_projection = 0.0
    maximum_executed_target_violation = 0.0
    executed_action_modified_values = 0
    maximum_target_projection_by_joint = {
        name: 0.0 for name in action.target_names
    }
    actual_joint_values = 0
    actual_violation_values = 0
    maximum_actual_violation = 0.0
    minimum_root_height = float(robot.data.root_link_pos_w[0, 2].item())
    termination_names: list[str] = []
    nonfinite: dict[str, Any] | None = None
    fell = False
    fall_step: int | None = None
    executed_steps = 0
    previous_action_consistency_verified_steps = 0

    for step in range(steps):
        finite, tensor_name = _all_finite({"actor_observation": observations["actor"]})
        if not finite:
            nonfinite = {"phase": "before_policy", "step": step, "tensor": tensor_name}
            break
        with torch.inference_mode():
            actions = policy(observations)
        finite, tensor_name = _all_finite({"policy_action": actions})
        if not finite:
            nonfinite = {"phase": "policy_output", "step": step, "tensor": tensor_name}
            break
        if tuple(actions.shape) != tuple(action.raw_action.shape):
            raise ValueError(
                f"Legacy policy action shape {tuple(actions.shape)} != "
                f"environment shape {tuple(action.raw_action.shape)}"
            )

        target = actions * scale + offset
        projected_target = torch.clamp(
            target, min=soft_limits[..., 0], max=soft_limits[..., 1]
        )
        projection = torch.abs(projected_target - target)
        projected_mask = projection > 1.0e-7
        target_values += projection.numel()
        projected_target_values += int(projected_mask.sum().item())
        projected_target_steps += int(bool(projected_mask.any().item()))
        maximum_target_projection = max(
            maximum_target_projection, float(projection.max().item())
        )
        for joint_index, joint_name in enumerate(action.target_names):
            maximum_target_projection_by_joint[joint_name] = max(
                maximum_target_projection_by_joint[joint_name],
                float(projection[0, joint_index].item()),
            )

        if execute_soft_limit_projection:
            if bool((scale == 0.0).any().item()):
                raise ValueError("Cannot invert a zero legacy action scale")
            executed_actions = (projected_target - offset) / scale
        else:
            executed_actions = actions
        executed_action_modified_values += int(
            (torch.abs(executed_actions - actions) > 1.0e-7).sum().item()
        )
        executed_target = executed_actions * scale + offset
        executed_target_violation = torch.maximum(
            torch.clamp(soft_limits[..., 0] - executed_target, min=0.0),
            torch.clamp(executed_target - soft_limits[..., 1], min=0.0),
        )
        maximum_executed_target_violation = max(
            maximum_executed_target_violation,
            float(executed_target_violation.max().item()),
        )

        # Feeding the transformed raw action to the environment also makes the
        # next `last_action` observation equal the action that was really
        # executed.  This keeps the actor's closed-loop recurrence consistent.
        observations, rewards, dones, _extras = wrapped.step(executed_actions)
        executed_steps += 1
        if not bool(torch.equal(action.raw_action, executed_actions)):
            raise RuntimeError("Action term did not retain the executed raw action")
        actor_observation = observations["actor"]
        if tuple(actor_observation.shape) != (1, 63):
            raise ValueError(
                "Legacy actor observation shape drifted from (1, 63): "
                f"{tuple(actor_observation.shape)}"
            )
        if not bool(
            torch.equal(
                actor_observation[:, _LEGACY_ACTION_OBSERVATION_SLICE],
                executed_actions,
            )
        ):
            raise RuntimeError(
                "Legacy previous-action observation does not match the raw action "
                "actually executed"
            )
        previous_action_consistency_verified_steps += 1
        if not bool(torch.equal(command.vel_command_b, expected_command)):
            raise RuntimeError(
                f"Command drifted during {scenario.name} at step {executed_steps}"
            )
        physics_nonfinite = bool(NanGuard.detect_nans(env.sim.data)[0].item())
        finite, tensor_name = _all_finite(
            {
                "actor_observation": observations["actor"],
                "joint_position": robot.data.joint_pos,
                "joint_velocity": robot.data.joint_vel,
                "reward": rewards,
                "root_pose": robot.data.root_link_pose_w,
                "root_velocity": robot.data.root_link_vel_w,
            }
        )
        if physics_nonfinite or not finite:
            nonfinite = {
                "phase": "after_step",
                "step": step,
                "tensor": "mujoco_warp_physics_state" if physics_nonfinite else tensor_name,
            }
            break

        root_height_m = float(robot.data.root_link_pos_w[0, 2].item())
        root_heights.append(root_height_m)
        minimum_root_height = min(minimum_root_height, root_height_m)
        current_fell = bool(
            env.termination_manager.get_term("fell_over")[0].item()
        ) or root_height_m < fall_height_m
        if current_fell and not fell:
            fell = True
            fall_step = executed_steps

        actual_position = robot.data.joint_pos
        all_limits = robot.data.soft_joint_pos_limits
        actual_violation = torch.maximum(
            torch.clamp(all_limits[..., 0] - actual_position, min=0.0),
            torch.clamp(actual_position - all_limits[..., 1], min=0.0),
        )
        actual_joint_values += actual_violation.numel()
        actual_violation_values += int((actual_violation > 1.0e-7).sum().item())
        maximum_actual_violation = max(
            maximum_actual_violation, float(actual_violation.max().item())
        )

        measured = (
            float(robot.data.root_link_lin_vel_b[0, 0].item()),
            float(robot.data.root_link_lin_vel_b[0, 1].item()),
            float(robot.data.root_link_ang_vel_b[0, 2].item()),
        )
        integrated_yaw_rad += measured[2] * env.step_dt
        if step >= settle_steps:
            for name, target_value, actual_value in zip(
                velocity_samples, scenario.twist, measured, strict=True
            ):
                velocity_samples[name].append(actual_value)
                velocity_error_samples[name].append(abs(target_value - actual_value))
            linear_vector_error.append(
                math.hypot(
                    scenario.twist[0] - measured[0],
                    scenario.twist[1] - measured[1],
                )
            )

        found = foot_contact.data.found
        current_air_time = foot_contact.data.current_air_time
        last_air_time = foot_contact.data.last_air_time
        heights = foot_height.data.heights
        if (
            found is None
            or current_air_time is None
            or last_air_time is None
            or tuple(heights.shape) != (1, 2)
        ):
            raise RuntimeError("Foot contact/height evidence is unavailable")
        airborne_by_side: dict[str, bool] = {}
        for height_index, side in enumerate(_FOOT_HEIGHT_ORDER):
            contact_index = contact_indices[side]
            in_contact = bool(found[0, contact_index].item() > 0)
            airborne = not in_contact
            airborne_by_side[side] = airborne
            foot_ever_airborne[side] |= airborne
            foot_airborne_steps[side] += int(airborne)
            foot_takeoffs[side] += int(previous_contact[side] and airborne)
            foot_landings[side] += int(not previous_contact[side] and in_contact)
            previous_contact[side] = in_contact
            foot_heights[side].append(float(heights[0, height_index].item()))
            foot_max_current_air[side] = max(
                foot_max_current_air[side],
                float(current_air_time[0, contact_index].item()),
            )
            foot_max_last_air[side] = max(
                foot_max_last_air[side],
                float(last_air_time[0, contact_index].item()),
            )
        airborne_count = sum(airborne_by_side.values())
        both_airborne_steps += int(airborne_count == 2)
        exactly_one_airborne_steps += int(airborne_count == 1)

        if bool(dones[0].item()):
            termination_names = _termination_names(env)
            break

    displacement_world = robot.data.root_link_pos_w[0, :2] - initial_position[:2]
    displacement_forward = float(
        torch.dot(displacement_world, initial_axes_world[0]).item()
    )
    displacement_lateral = float(
        torch.dot(displacement_world, initial_axes_world[1]).item()
    )
    measured_velocity = {
        name: summarize_samples(values) for name, values in velocity_samples.items()
    }
    velocity_error = {
        name: summarize_samples(values)
        for name, values in velocity_error_samples.items()
    }
    completed = (
        executed_steps == steps
        and not termination_names
        and nonfinite is None
        and not fell
    )
    return {
        "command": {
            "vx_m_s": scenario.twist[0],
            "vy_m_s": scenario.twist[1],
            "yaw_rad_s": scenario.twist[2],
        },
        "completed": completed,
        "displacement_initial_body_frame": {
            "forward_m": displacement_forward,
            "integrated_yaw_rad": integrated_yaw_rad,
            "lateral_m": displacement_lateral,
        },
        "executed_steps": executed_steps,
        "fall_step": fall_step,
        "fell": fell,
        "foot_evidence": {
            side: {
                "airborne_step_count": foot_airborne_steps[side],
                "airborne_step_fraction": (
                    foot_airborne_steps[side] / executed_steps if executed_steps else 0.0
                ),
                "ever_airborne": foot_ever_airborne[side],
                "height_m": summarize_samples(foot_heights[side]),
                "landing_count": foot_landings[side],
                "maximum_current_air_time_s": foot_max_current_air[side],
                "maximum_completed_air_time_s": foot_max_last_air[side],
                "takeoff_count": foot_takeoffs[side],
            }
            for side in _FOOT_HEIGHT_ORDER
        }
        | {
            "both_feet_airborne_step_count": both_airborne_steps,
            "exactly_one_foot_airborne_step_count": exactly_one_airborne_steps,
        },
        "linear_velocity_vector_error_m_s": summarize_samples(linear_vector_error),
        "measured_velocity_body": measured_velocity,
        "minimum_root_height_m": minimum_root_height,
        "name": scenario.name,
        "nonfinite": nonfinite,
        "nonfinite_detected": nonfinite is not None,
        "requested_steps": steps,
        "previous_action_recurrence": {
            "consistent": previous_action_consistency_verified_steps
            == executed_steps,
            "verified_steps": previous_action_consistency_verified_steps,
        },
        "root_height_m": summarize_samples(root_heights),
        "seed": seed,
        "target_soft_limits": {
            "actual_joint_value_count": actual_joint_values,
            "actual_violation_fraction": (
                actual_violation_values / actual_joint_values
                if actual_joint_values
                else 0.0
            ),
            "actual_violation_value_count": actual_violation_values,
            "executed_action_modified_value_count": executed_action_modified_values,
            "execution_applied_projection": execute_soft_limit_projection,
            "maximum_executed_target_violation_rad": (
                maximum_executed_target_violation
            ),
            "maximum_actual_violation_rad": maximum_actual_violation,
            "maximum_hypothetical_target_projection_by_joint_rad": (
                maximum_target_projection_by_joint
            ),
            "maximum_hypothetical_target_projection_rad": maximum_target_projection,
            "projected_target_step_count": projected_target_steps,
            "projected_target_value_count": projected_target_values,
            "projected_target_value_fraction": (
                projected_target_values / target_values if target_values else 0.0
            ),
            "target_value_count": target_values,
        },
        "termination_names": termination_names,
        "velocity_absolute_error": velocity_error,
    }


def evaluate_checkpoint(
    *,
    checkpoint: Path,
    expected_sha256: str,
    device: str,
    seed: int,
    steps: int,
    settle_steps: int,
    scenarios: tuple[TwistScenario, ...],
    execute_soft_limit_projection: bool,
) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Legacy velocity checkpoint not found: {checkpoint}")
    digest_before = checkpoint_sha256(checkpoint)
    if digest_before != expected_sha256:
        raise ValueError(
            f"Legacy velocity checkpoint SHA-256 mismatch: {digest_before}"
        )
    if steps < 1 or not 0 <= settle_steps < steps:
        raise ValueError("Require steps > settle_steps >= 0")

    configure_torch_backends(allow_tf32=False, deterministic=True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    cfg = _configure_environment(steps=steps, seed=seed)
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    agent_cfg = MicrobanVelocityRlCfg
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    try:
        runner = VelocityOnPolicyRunner(wrapped, asdict(agent_cfg), device=device)
        runner.load(
            str(checkpoint),
            load_cfg={"actor": True},
            strict=True,
            map_location=device,
        )
        policy = runner.get_inference_policy(device=device)
        results = []
        for scenario in scenarios:
            print(f"[INFO] evaluating legacy walk scenario {scenario.name}", flush=True)
            results.append(
                _evaluate_scenario(
                    env=env,
                    wrapped=wrapped,
                    policy=policy,
                    scenario=scenario,
                    steps=steps,
                    settle_steps=settle_steps,
                    seed=seed,
                    execute_soft_limit_projection=execute_soft_limit_projection,
                )
            )
        digest_after = checkpoint_sha256(checkpoint)
        if digest_after != digest_before:
            raise RuntimeError("Legacy velocity checkpoint changed during evaluation")
        return build_report(
            checkpoint=checkpoint,
            checkpoint_sha256=digest_after,
            device=device,
            seed=seed,
            steps=steps,
            settle_steps=settle_steps,
            step_dt_s=env.step_dt,
            results=results,
            execute_soft_limit_projection=execute_soft_limit_projection,
        )
    finally:
        wrapped.close()
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_LEGACY_VELOCITY_CHECKPOINT)
    parser.add_argument("--expected-sha256", default=DEFAULT_LEGACY_VELOCITY_SHA256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--settle-steps", type=int, default=50)
    parser.add_argument("--scenarios")
    parser.add_argument(
        "--execute-soft-limit-projection",
        action="store_true",
        help=(
            "Execute the actor after converting its absolute targets to the "
            "current joint soft limits; previous-action observations then contain "
            "the transformed action actually executed"
        ),
    )
    parser.add_argument("--list-scenarios", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scenarios = select_scenarios(default_scenarios(), args.scenarios)
    if args.list_scenarios:
        print("\n".join(scenario.name for scenario in scenarios))
        return 0
    output = resolve_output_path(
        args.output,
        execute_soft_limit_projection=args.execute_soft_limit_projection,
    )
    if output.expanduser().exists() and not args.force:
        raise FileExistsError(f"Output already exists (pass --force): {output}")
    report = evaluate_checkpoint(
        checkpoint=args.checkpoint,
        expected_sha256=args.expected_sha256,
        device=args.device,
        seed=args.seed,
        steps=args.steps,
        settle_steps=args.settle_steps,
        scenarios=scenarios,
        execute_soft_limit_projection=args.execute_soft_limit_projection,
    )
    output = publish_json_atomic(output, report)
    print(
        json.dumps(
            {
                "checkpoint_sha256": report["checkpoint"]["sha256"],
                "output": str(output),
                "summary": report["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
