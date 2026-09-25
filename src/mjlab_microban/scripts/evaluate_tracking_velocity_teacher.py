# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Evaluate a frozen velocity actor as a safe tracking-BC proposal teacher.

This is a simulation-only prerequisite gate.  The legacy actor never reaches
the environment directly: each output passes through the tracking student's
deterministic action closure and the action term's absolute soft limits.  The
measured state is checked at ``q`` and ``q + 0.12*qdot`` before a label is
admitted.  More than 1 mrad of target projection marks that label as no longer
faithful to the legacy actor, while the projected controller is judged on its
own complete dynamic rollout.

The projected action is executed in simulation, which lets the receipt
distinguish "dynamically incapable" from "stable but substantially changed by
the safety projection".  A non-finite or measured-state-unsafe row receives raw
action zero and is never admitted as a label.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommand
from mjlab.utils.lab_api.math import quat_apply
from mjlab.utils.nan_guard import NanGuard
from mjlab.utils.torch import configure_torch_backends

from mjlab_microban.locomotion_prior_suitability import (
    DEFAULT_LOCOMOTION_PRIOR_PATH,
    DEFAULT_LOCOMOTION_PRIOR_SHA256,
    sha256_file,
)
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_TELEOP_ACTION_JOINT_NAMES,
)
from mjlab_microban.tasks.microban_tracking_env_cfg import (
    MICROBAN_TRACKING_ACTION_JOINT_NAMES,
    make_microban_tracking_env_cfg,
    microban_tracking_action_delta_bounds,
    microban_tracking_initial_action_std,
)
from mjlab_microban.tasks.microban_tracking_mdp import (
    MicrobanTrackingBoundedGaussianDistribution,
)
from mjlab_microban.tasks.microban_tracking_teacher import (
    MICROBAN_TRACKING_TEACHER_CHECKPOINT_ITERATION,
    MICROBAN_TRACKING_TEACHER_CHECKPOINT_SHA256,
    MICROBAN_TRACKING_TEACHER_MAX_TARGET_PROJECTION_RAD,
    assemble_velocity_teacher_observation,
    bounded_student_deterministic_action_closure,
    load_frozen_velocity_teacher,
    project_velocity_teacher_labels,
    reference_velocity_command_b,
)
from mjlab_microban.tracking_checkpoint_gate import (
    TRACKING_CHECKPOINT_GATE_END_FRAME,
    TRACKING_CHECKPOINT_GATE_EXPECTED_STEPS,
    TRACKING_CHECKPOINT_GATE_FIRST_TARGET_FRAME,
    TRACKING_CHECKPOINT_GATE_FRAME_COUNT,
    TRACKING_CHECKPOINT_GATE_MODES,
    TRACKING_CHECKPOINT_GATE_NUM_ENVS,
    TRACKING_CHECKPOINT_GATE_SEED,
    tracking_checkpoint_checks,
)

DEFAULT_OUTPUT = Path("artifacts/microban_tracking_velocity_teacher_gate.json")
_ROOT_HEIGHT_FALL_THRESHOLD_M = 0.10
_EXPECTED_STARTUP_EVENTS = {"base_com", "encoder_bias", "foot_friction"}


def _percentile(values: torch.Tensor, percentile: float, *, fallback: float) -> float:
    array = values.detach().to(dtype=torch.float64).cpu().numpy()
    array = array[np.isfinite(array)]
    if array.size == 0:
        return fallback
    return float(np.percentile(array, percentile, method="linear"))


def _publish_json(path: Path, report: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        report, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(encoded)
        stream.flush()
    try:
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _feet_ground_sensor_cfg() -> ContactSensorCfg:
    return ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree", pattern=r"^(foot|foot_2)$", entity="robot"
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )


def _configure(mode: str, motion: Path, num_envs: int) -> Any:
    if mode not in TRACKING_CHECKPOINT_GATE_MODES:
        raise ValueError(f"Unsupported teacher gate mode: {mode}")
    cfg = make_microban_tracking_env_cfg(play=True, motion_file=motion)
    cfg.scene.num_envs = num_envs
    cfg.seed = TRACKING_CHECKPOINT_GATE_SEED
    cfg.auto_reset = False
    cfg.curriculum = {}
    cfg.episode_length_s = (TRACKING_CHECKPOINT_GATE_EXPECTED_STEPS + 2) * (
        cfg.decimation * cfg.sim.mujoco.timestep
    )
    cfg.observations["actor"].enable_corruption = mode == "robust"
    cfg.commands["motion"].debug_vis = False
    startup_events = {
        name: term for name, term in cfg.events.items() if term.mode == "startup"
    }
    if set(startup_events) != _EXPECTED_STARTUP_EVENTS:
        raise ValueError(f"Tracking startup event set drifted: {sorted(startup_events)}")
    cfg.events = {} if mode == "nominal" else startup_events
    cfg.scene.sensors = (*cfg.scene.sensors, _feet_ground_sensor_cfg())
    return cfg


def _install_non_wrapping_advance(command: MotionCommand) -> None:
    def advance_without_wrap(self: MotionCommand) -> None:
        self.time_steps.add_(1)
        self.time_steps.clamp_(max=TRACKING_CHECKPOINT_GATE_END_FRAME)
        self.update_relative_body_poses()

    command._update_command = MethodType(  # type: ignore[method-assign]
        advance_without_wrap, command
    )


def _broadcast_action_parameter(
    value: torch.Tensor | float, reference: torch.Tensor
) -> torch.Tensor:
    return torch.broadcast_to(
        torch.as_tensor(value, dtype=reference.dtype, device=reference.device),
        reference.shape,
    )


def _finite_envs(named: Mapping[str, torch.Tensor], num_envs: int) -> torch.Tensor:
    device = next(iter(named.values())).device
    result = torch.ones(num_envs, dtype=torch.bool, device=device)
    for name, value in named.items():
        if value.shape[0] != num_envs:
            raise ValueError(f"{name} does not have an environment-leading dimension")
        result &= torch.isfinite(value).reshape(num_envs, -1).all(dim=-1)
    return result


def _contact_indices(sensor: Any) -> dict[str, int]:
    primaries = [
        slot.primary_name for slot in sensor._slots if slot.field_name == "found"
    ]
    if len(primaries) != 2 or set(primaries) != {"foot", "foot_2"}:
        raise ValueError(f"Unexpected foot-contact layout: {primaries}")
    return {
        ("right" if name == "foot" else "left"): index
        for index, name in enumerate(primaries)
    }


def _evaluate_pass(
    *,
    mode: str,
    teacher_checkpoint: Path,
    teacher_sha256: str,
    teacher_iteration: int,
    motion: Path,
    num_envs: int,
    teacher_command_scale: float,
    device: str,
) -> dict[str, Any]:
    env: ManagerBasedRlEnv | None = None
    wrapped: RslRlVecEnvWrapper | None = None
    try:
        cfg = _configure(mode, motion, num_envs)
        env = ManagerBasedRlEnv(cfg=cfg, device=device)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
        teacher = load_frozen_velocity_teacher(
            teacher_checkpoint,
            checkpoint_sha256=teacher_sha256,
            checkpoint_iteration=teacher_iteration,
            device=device,
        )
        env.reset(seed=TRACKING_CHECKPOINT_GATE_SEED)
        observations = wrapped.get_observations()
        robot = env.scene["robot"]
        action = env.action_manager.get_term("joint_pos")
        command = env.command_manager.get_term("motion")
        if not isinstance(command, MotionCommand):
            raise TypeError("Tracking motion command did not build correctly")
        if tuple(action.target_names) != MICROBAN_TRACKING_ACTION_JOINT_NAMES:
            raise ValueError("Tracking action order differs from teacher action order")
        if tuple(action.target_names) != MICROBAN_TELEOP_ACTION_JOINT_NAMES:
            raise ValueError("Legacy velocity and tracking action orders differ")
        if not bool((command.time_steps == 1).all().item()):
            raise ValueError("Tracking reset must expose source frame 1 first")
        if command.motion.time_step_total != TRACKING_CHECKPOINT_GATE_FRAME_COUNT:
            raise ValueError("Teacher gate requires exactly 268 source frames")
        if not math.isclose(env.step_dt, 0.02, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("Teacher gate requires exact 50 Hz policy steps")
        _install_non_wrapping_advance(command)

        distribution = MicrobanTrackingBoundedGaussianDistribution(
            18,
            microban_tracking_initial_action_std(),
            *microban_tracking_action_delta_bounds(),
            std_type="log",
        ).to(device)
        closure_lower, closure_upper = bounded_student_deterministic_action_closure(
            distribution
        )
        offset = _broadcast_action_parameter(action.offset, action.raw_action)
        scale = _broadcast_action_parameter(action.scale, action.raw_action)
        soft = torch.broadcast_to(action._clip, (*action.raw_action.shape, 2))
        soft_lower = soft[..., 0]
        soft_upper = soft[..., 1]
        default = robot.data.default_joint_pos[:, action.target_ids]

        initial_root_pos = robot.data.root_link_pos_w.clone()
        initial_reference_quat = command.anchor_quat_w.clone()
        local_forward = torch.tensor(
            (1.0, 0.0, 0.0), dtype=torch.float32, device=env.device
        ).expand(num_envs, -1)
        reference_forward_xy = quat_apply(
            initial_reference_quat, local_forward
        )[:, :2]
        reference_forward_xy /= torch.linalg.vector_norm(
            reference_forward_xy, dim=-1, keepdim=True
        ).clamp_min(1.0e-9)

        active = torch.ones(num_envs, dtype=torch.bool, device=env.device)
        fall = torch.zeros_like(active)
        nonfinite = torch.zeros_like(active)
        unexpected_termination = torch.zeros_like(active)
        full_clip = torch.zeros_like(active)
        hard_guard_rejected = torch.zeros_like(active)
        projection_rejected = torch.zeros_like(active)
        ever_airborne = {
            side: torch.zeros_like(active) for side in ("left", "right")
        }
        minimum_height = robot.data.root_link_pos_w[:, 2].clone()
        forward_displacement = torch.zeros(
            num_envs, dtype=torch.float64, device=env.device
        )
        velocity_sum = torch.zeros_like(forward_displacement)
        velocity_error_sum = torch.zeros_like(forward_displacement)
        sample_count = torch.zeros(num_envs, dtype=torch.long, device=env.device)
        max_actual_violation = torch.zeros((), device=env.device)
        max_legacy_projection = torch.zeros((), device=env.device)
        max_environment_projection = torch.zeros((), device=env.device)
        label_candidate_count = 0
        accepted_label_count = 0
        faithful_label_count = 0
        preferred_label_count = 0
        projected_value_count = 0
        total_target_value_count = 0
        simultaneous_flight_env_steps = 0
        self_collision_contacts = 0
        phase_sequence_valid = True
        executed_steps = 0
        step_trace: list[dict[str, Any]] = []

        foot_contact = env.scene.sensors["feet_ground_contact"]
        foot_indices = _contact_indices(foot_contact)
        self_collision = env.scene.sensors["self_collision"]

        for policy_step in range(TRACKING_CHECKPOINT_GATE_EXPECTED_STEPS):
            active_before = active.clone()
            if not bool(active_before.any().item()):
                break
            expected_frame = TRACKING_CHECKPOINT_GATE_FIRST_TARGET_FRAME + policy_step
            phase_ok = bool(
                (command.time_steps[active_before] == expected_frame).all().item()
            )
            phase_sequence_valid &= phase_ok

            teacher_command = reference_velocity_command_b(
                reference_quat_w=command.anchor_quat_w,
                reference_lin_vel_w=command.anchor_lin_vel_w,
                reference_ang_vel_w=command.anchor_ang_vel_w,
            ) * teacher_command_scale
            joint_pos = robot.data.joint_pos[:, action.target_ids]
            joint_vel = robot.data.joint_vel[:, action.target_ids]
            teacher_observation = assemble_velocity_teacher_observation(
                base_ang_vel=robot.data.root_link_ang_vel_b,
                projected_gravity=robot.data.projected_gravity_b,
                joint_pos=joint_pos - default,
                joint_vel=(
                    joint_vel
                    - robot.data.default_joint_vel[:, action.target_ids]
                ),
                previous_action=action.raw_action,
                command=teacher_command,
            )
            proposal = teacher(teacher_observation)
            projected = project_velocity_teacher_labels(
                proposal,
                teacher_scale=scale,
                teacher_offset=offset,
                student_scale=scale,
                student_offset=offset,
                soft_lower=soft_lower,
                soft_upper=soft_upper,
                student_action_lower=closure_lower,
                student_action_upper=closure_upper,
                joint_pos=joint_pos,
                joint_vel=joint_vel,
                default_joint_pos=default,
            )
            active_candidates = int(active_before.sum().item())
            label_candidate_count += active_candidates
            accepted_label_count += int(
                (projected.accepted & active_before).sum().item()
            )
            faithful_label_count += int(
                (projected.faithful_to_legacy & active_before).sum().item()
            )
            preferred_label_count += int(
                (
                    (projected.bc_weight > 0.0)
                    & active_before.unsqueeze(-1)
                ).sum().item()
            )
            current_projection_rejected = (
                ~projected.faithful_to_legacy & active_before
            )
            current_hard_rejected = (
                ~projected.measured_soft_limit_safe & active_before
            )
            projection_rejected |= current_projection_rejected
            hard_guard_rejected |= current_hard_rejected
            if active_candidates:
                max_legacy_projection = torch.maximum(
                    max_legacy_projection,
                    projected.maximum_target_projection_rad[active_before].max(),
                )
                final_unclipped_target = offset + scale * projected.label_action
                final_clipped_target = torch.clamp(
                    final_unclipped_target, min=soft_lower, max=soft_upper
                )
                environment_projection = torch.abs(
                    final_clipped_target - final_unclipped_target
                )
                max_environment_projection = torch.maximum(
                    max_environment_projection,
                    environment_projection[active_before].max(),
                )
                environment_projected_values = (
                    environment_projection[active_before] > 1.0e-7
                )
                projected_value_count += int(
                    environment_projected_values.sum().item()
                )
                total_target_value_count += int(
                    environment_projected_values.numel()
                )

            safe_action = projected.label_action.clone()
            execute_safe = projected.accepted & active_before
            safe_action[~execute_safe] = 0.0
            observations, rewards, dones, _ = wrapped.step(safe_action)
            executed_steps += 1

            finite_state = _finite_envs(
                {
                    "actor_observation": observations["actor"],
                    "joint_position": robot.data.joint_pos,
                    "joint_velocity": robot.data.joint_vel,
                    "reward": rewards.unsqueeze(-1),
                    "root_pose": robot.data.root_link_pose_w,
                    "root_velocity": robot.data.root_link_vel_w,
                },
                num_envs,
            )
            step_nonfinite = (
                NanGuard.detect_nans(env.sim.data) | ~finite_state
            ) & active_before
            nonfinite |= step_nonfinite
            valid = active_before & ~step_nonfinite

            root_height = robot.data.root_link_pos_w[:, 2]
            minimum_height[valid] = torch.minimum(
                minimum_height[valid], root_height[valid]
            )
            low_root = valid & (root_height < _ROOT_HEIGHT_FALL_THRESHOLD_M)
            fall |= low_root
            actual_violation = torch.maximum(
                torch.clamp(
                    robot.data.soft_joint_pos_limits[..., 0]
                    - robot.data.joint_pos,
                    min=0.0,
                ),
                torch.clamp(
                    robot.data.joint_pos
                    - robot.data.soft_joint_pos_limits[..., 1],
                    min=0.0,
                ),
            )
            if bool(valid.any().item()):
                max_actual_violation = torch.maximum(
                    max_actual_violation, actual_violation[valid].max()
                )
                velocity_w = robot.data.root_link_lin_vel_w[:, :2]
                reference_velocity_w = command.anchor_lin_vel_w[:, :2]
                forward_velocity = torch.sum(
                    velocity_w * reference_forward_xy, dim=-1
                )
                velocity_error = torch.linalg.vector_norm(
                    velocity_w - reference_velocity_w, dim=-1
                )
                velocity_sum[valid] += forward_velocity[valid].to(torch.float64)
                velocity_error_sum[valid] += velocity_error[valid].to(torch.float64)
                sample_count[valid] += 1
                displacement_xy = (
                    robot.data.root_link_pos_w[:, :2] - initial_root_pos[:, :2]
                )
                forward_displacement[valid] = torch.sum(
                    displacement_xy[valid] * reference_forward_xy[valid], dim=-1
                ).to(torch.float64)

            collision_found = self_collision.data.found
            if collision_found is None:
                raise RuntimeError("self_collision sensor does not expose found")
            self_collision_contacts += int(collision_found[valid].sum().item())
            contacts = foot_contact.data.found
            if contacts is None:
                raise RuntimeError("feet_ground_contact does not expose found")
            contact_by_side = {
                side: contacts[:, index] > 0
                for side, index in foot_indices.items()
            }
            for side in ("left", "right"):
                ever_airborne[side] |= ~contact_by_side[side] & valid
            simultaneous_flight_env_steps += int(
                (
                    ~contact_by_side["left"]
                    & ~contact_by_side["right"]
                    & valid
                ).sum().item()
            )

            done = dones.bool() & active_before
            unexpected_termination |= done
            removed = done | step_nonfinite | low_root | current_hard_rejected
            active &= ~removed
            step_trace.append(
                {
                    "accepted_labels": int(
                        (projected.accepted & active_before).sum().item()
                    ),
                    "active_envs_after_step": int(active.sum().item()),
                    "hard_guard_rejections": int(current_hard_rejected.sum().item()),
                    "maximum_legacy_projection_rad": float(
                        projected.maximum_target_projection_rad[
                            active_before
                        ].max().item()
                    ),
                    "policy_step_zero_based": policy_step,
                    "projection_rejections": int(
                        current_projection_rejected.sum().item()
                    ),
                    "source_frame": expected_frame,
                }
            )
            if policy_step % 32 == 0 or policy_step + 1 == (
                TRACKING_CHECKPOINT_GATE_EXPECTED_STEPS
            ):
                print(
                    f"[INFO] teacher {mode}: frame={expected_frame} "
                    f"active={int(active.sum().item())}/{num_envs} "
                    f"accepted={accepted_label_count}/{label_candidate_count}",
                    flush=True,
                )
            if policy_step + 1 == TRACKING_CHECKPOINT_GATE_EXPECTED_STEPS:
                full_clip |= active
                break
            if not bool(active.any().item()):
                break
            inactive_ids = (~active).nonzero(as_tuple=False).squeeze(-1)
            if inactive_ids.numel() > 0:
                env.reset(env_ids=inactive_ids)
                command.time_steps[active] = min(
                    expected_frame + 1, TRACKING_CHECKPOINT_GATE_END_FRAME
                )
                command.update_relative_body_poses()
                observations = wrapped.get_observations()

        sampled = sample_count > 0
        velocity_mean = torch.full_like(velocity_sum, float("nan"))
        velocity_error_mean = torch.full_like(velocity_error_sum, float("nan"))
        velocity_mean[sampled] = velocity_sum[sampled] / sample_count[sampled]
        velocity_error_mean[sampled] = (
            velocity_error_sum[sampled] / sample_count[sampled]
        )
        measurements = {
            "executed_policy_steps": executed_steps,
            "fall_envs": int(fall.sum().item()),
            "forward_displacement_p05_m": _percentile(
                forward_displacement, 5.0, fallback=0.0
            ),
            "forward_velocity_p05_m_s": _percentile(
                velocity_mean[sampled], 5.0, fallback=0.0
            ),
            "full_clip_completed_envs": int(full_clip.sum().item()),
            "left_foot_airborne_envs": int(ever_airborne["left"].sum().item()),
            "maximum_actual_soft_limit_violation_rad": float(
                max_actual_violation.item()
            ),
            "maximum_target_projection_rad": float(
                max_environment_projection.item()
            ),
            "minimum_root_height_m": float(minimum_height.min().item()),
            "nonfinite_envs": int(nonfinite.sum().item()),
            "right_foot_airborne_envs": int(ever_airborne["right"].sum().item()),
            "self_collision_contacts": self_collision_contacts,
            "simultaneous_flight_env_steps": simultaneous_flight_env_steps,
            "source_frame_sequence_complete": (
                phase_sequence_valid
                and executed_steps == TRACKING_CHECKPOINT_GATE_EXPECTED_STEPS
                and bool(full_clip.all().item())
            ),
            "target_clip_fraction": (
                projected_value_count / total_target_value_count
                if total_target_value_count
                else 1.0
            ),
            "unexpected_termination_envs": int(
                unexpected_termination.sum().item()
            ),
            "xy_velocity_mae_p95_m_s": _percentile(
                velocity_error_mean[sampled], 95.0, fallback=1.0e9
            ),
        }
        checks = tracking_checkpoint_checks(
            measurements, num_envs=num_envs
        )
        checks["teacher_label_acceptance"] = {
            "expected": label_candidate_count,
            "passed": accepted_label_count == label_candidate_count,
            "value": accepted_label_count,
        }
        checks["teacher_hard_lookahead_rejections"] = {
            "maximum": 0,
            "passed": not bool(hard_guard_rejected.any().item()),
            "value": int(hard_guard_rejected.sum().item()),
        }
        failed = sorted(name for name, value in checks.items() if not value["passed"])
        return {
            "checks": checks,
            "diagnostics": {
                "accepted_label_fraction": (
                    accepted_label_count / label_candidate_count
                    if label_candidate_count
                    else 0.0
                ),
                "faithful_legacy_label_fraction": (
                    faithful_label_count / label_candidate_count
                    if label_candidate_count
                    else 0.0
                ),
                "legacy_proposal_maximum_projection_rad": float(
                    max_legacy_projection.item()
                ),
                "bc_preferred_margin_fraction": (
                    preferred_label_count / (18 * label_candidate_count)
                    if label_candidate_count
                    else 0.0
                ),
                "hard_lookahead_rejected_envs": int(
                    hard_guard_rejected.sum().item()
                ),
                "projection_rejected_envs": int(
                    projection_rejected.sum().item()
                ),
                "step_trace": step_trace,
            },
            "failed_checks": failed,
            "measurements": measurements,
            "passed": not failed,
        }
    finally:
        if wrapped is not None:
            wrapped.close()
        elif env is not None:
            env.close()
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()


def evaluate_velocity_teacher(
    *,
    teacher_checkpoint: Path,
    teacher_sha256: str,
    teacher_iteration: int,
    motion: Path,
    expected_motion_sha256: str,
    num_envs: int,
    teacher_command_scale: float,
    device: str,
) -> dict[str, Any]:
    teacher_checkpoint = teacher_checkpoint.expanduser().resolve()
    motion = motion.expanduser().resolve()
    if not teacher_checkpoint.is_file():
        raise FileNotFoundError(f"Teacher checkpoint does not exist: {teacher_checkpoint}")
    if not motion.is_file():
        raise FileNotFoundError(f"Tracking motion does not exist: {motion}")
    actual_motion_sha256 = sha256_file(motion)
    if actual_motion_sha256 != expected_motion_sha256:
        raise ValueError(
            f"Tracking motion SHA-256 mismatch: {actual_motion_sha256}"
        )
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
        raise ValueError("num_envs must be a positive integer")
    if not math.isfinite(teacher_command_scale):
        raise ValueError("teacher_command_scale must be finite")
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Velocity-teacher dynamic gate requires CUDA")
    configure_torch_backends(allow_tf32=False, deterministic=True)
    torch.use_deterministic_algorithms(True, warn_only=True)

    passes = {
        mode: _evaluate_pass(
            mode=mode,
            teacher_checkpoint=teacher_checkpoint,
            teacher_sha256=teacher_sha256,
            teacher_iteration=teacher_iteration,
            motion=motion,
            num_envs=num_envs,
            teacher_command_scale=teacher_command_scale,
            device=device,
        )
        for mode in TRACKING_CHECKPOINT_GATE_MODES
    }
    failed = [
        f"{mode}.{name}"
        for mode, result in passes.items()
        for name in result["failed_checks"]
    ]
    return {
        "configuration": {
            "expected_policy_steps": TRACKING_CHECKPOINT_GATE_EXPECTED_STEPS,
            "lookahead_s": 0.12,
            "max_target_projection_rad": (
                MICROBAN_TRACKING_TEACHER_MAX_TARGET_PROJECTION_RAD
            ),
            "modes": list(TRACKING_CHECKPOINT_GATE_MODES),
            "num_envs_per_pass": num_envs,
            "seed": TRACKING_CHECKPOINT_GATE_SEED,
            "teacher_command_scale": teacher_command_scale,
        },
        "inputs": {
            "motion": {
                "path": str(motion),
                "sha256": actual_motion_sha256,
            },
            "teacher_checkpoint": {
                "iteration": teacher_iteration,
                "path": str(teacher_checkpoint),
                "sha256": sha256_file(teacher_checkpoint),
            },
        },
        "passes": passes,
        "status": "pass" if not failed else "fail",
        "summary": {"failed_checks": sorted(failed), "passed": not failed},
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-checkpoint", required=True, type=Path)
    parser.add_argument(
        "--teacher-sha256", default=MICROBAN_TRACKING_TEACHER_CHECKPOINT_SHA256
    )
    parser.add_argument(
        "--teacher-iteration",
        type=int,
        default=MICROBAN_TRACKING_TEACHER_CHECKPOINT_ITERATION,
    )
    parser.add_argument("--motion", type=Path, default=DEFAULT_LOCOMOTION_PRIOR_PATH)
    parser.add_argument(
        "--expected-motion-sha256", default=DEFAULT_LOCOMOTION_PRIOR_SHA256
    )
    parser.add_argument("--num-envs", type=int, default=TRACKING_CHECKPOINT_GATE_NUM_ENVS)
    parser.add_argument("--teacher-command-scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = evaluate_velocity_teacher(
        teacher_checkpoint=args.teacher_checkpoint,
        teacher_sha256=args.teacher_sha256,
        teacher_iteration=args.teacher_iteration,
        motion=args.motion,
        expected_motion_sha256=args.expected_motion_sha256,
        num_envs=args.num_envs,
        teacher_command_scale=args.teacher_command_scale,
        device=args.device,
    )
    _publish_json(args.output, report)
    print(
        json.dumps(
            {
                "failed_checks": report["summary"]["failed_checks"],
                "output": str(args.output.expanduser().resolve()),
                "status": report["status"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
