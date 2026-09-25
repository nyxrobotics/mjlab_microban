# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""GPU dynamics preflight for the short Microban locomotion prior.

This is an actuator replay, not a learned-policy evaluation.  It retains the
training environment's startup domain randomization and BAM actuator, removes
the interval push, teleports all environments to source frame 109, and commands
the next source frame at every 50 Hz policy step through frame 267.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.nan_guard import NanGuard
from mjlab.utils.torch import configure_torch_backends

from mjlab_microban.locomotion_prior_dynamic_gate import (
    DEFAULT_DYNAMIC_THRESHOLDS,
    DYNAMIC_GATE_EXPECTED_POLICY_STEPS,
    DYNAMIC_GATE_NUM_ENVS,
    DYNAMIC_GATE_SEED,
    dynamic_gate_checks,
    finalize_dynamic_report,
    load_and_validate_static_receipt,
    summarize_first_exit_steps,
)
from mjlab_microban.locomotion_prior_suitability import (
    DEFAULT_LOCOMOTION_PRIOR_PATH,
    DEFAULT_LOCOMOTION_PRIOR_SHA256,
    DEFAULT_ROBOT_XML_PATH,
    DEFAULT_ROBOT_XML_SHA256,
    PROJECT_ROOT,
    publish_suitability_receipt,
    sha256_file,
)
from mjlab_microban.tasks.mdp import UniformVelocityCommandWithRotation
from mjlab_microban.tasks.microban_locomotion_prior import (
    MICROBAN_LOCOMOTION_PRIOR_END_FRAME,
    MICROBAN_LOCOMOTION_PRIOR_LEG_JOINT_NAMES,
    MICROBAN_LOCOMOTION_PRIOR_NOMINAL_FORWARD_VELOCITY_M_S,
    MICROBAN_LOCOMOTION_PRIOR_START_FRAME,
    LocomotionPriorCommand,
)
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_TELEOP_ACTION_JOINT_NAMES,
)
from mjlab_microban.tasks.microban_teleop_env_cfg import (
    make_microban_teleop_env_cfg,
)

DEFAULT_STATIC_RECEIPT = Path("artifacts/microban_locomotion_prior_suitability.json")
DEFAULT_DYNAMIC_RECEIPT = Path("artifacts/microban_locomotion_prior_dynamic_gate.json")
_EXPECTED_STARTUP_EVENTS = {
    "foot_friction",
    "encoder_bias",
    "base_com",
    "dof_armature_randomization",
    "dof_friction_randomization",
}
_EXPECTED_RESET_EVENTS = {"reset_base", "reset_robot_joints"}
_EXPECTED_FINISH_TERM = "locomotion_prior_clip_finished"


def _portable_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.name


def _percentile(values: torch.Tensor, percentile: float) -> float:
    array = values.detach().to(dtype=torch.float64).cpu().numpy()
    return float(np.percentile(array, percentile, method="linear"))


def _distribution(values: torch.Tensor, *, units: str) -> dict[str, Any]:
    value = values.detach().to(dtype=torch.float64)
    if value.numel() == 0 or not bool(torch.isfinite(value).all().item()):
        raise ValueError("Distribution values must be non-empty and finite")
    return {
        "count": int(value.numel()),
        "max": float(value.max().item()),
        "mean": float(value.mean().item()),
        "min": float(value.min().item()),
        "p05": _percentile(value, 5.0),
        "p50": _percentile(value, 50.0),
        "p95": _percentile(value, 95.0),
        "units": units,
    }


def _configure_gate_env(
    *, prior_path: Path, expected_prior_sha256: str
) -> tuple[Any, dict[str, Any]]:
    cfg = make_microban_teleop_env_cfg(play=False)
    cfg.scene.num_envs = DYNAMIC_GATE_NUM_ENVS
    cfg.seed = DYNAMIC_GATE_SEED
    cfg.auto_reset = False
    cfg.episode_length_s = 10.0

    # Preserve startup domain randomization and deterministic reset events only.
    # In particular, interval pushes and moving-HMD step events are out of scope.
    retained = {
        name: term
        for name, term in cfg.events.items()
        if term.mode in {"startup", "reset"}
    }
    startup_names = {name for name, term in retained.items() if term.mode == "startup"}
    reset_names = {name for name, term in retained.items() if term.mode == "reset"}
    if startup_names != _EXPECTED_STARTUP_EVENTS:
        raise ValueError(
            f"Startup domain-randomization set drifted: {sorted(startup_names)}"
        )
    if reset_names != _EXPECTED_RESET_EVENTS:
        raise ValueError(f"Reset event set drifted: {sorted(reset_names)}")
    cfg.events = retained
    cfg.events["reset_base"].params["pose_range"] = {
        axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    cfg.events["reset_base"].params["velocity_range"] = {}
    cfg.events["reset_robot_joints"].params["position_range"] = (0.0, 0.0)
    cfg.events["reset_robot_joints"].params["velocity_range"] = (0.0, 0.0)
    cfg.curriculum = {}

    twist = cfg.commands["twist"]
    probabilities = {name: 0.0 for name in twist.signed_axis_probabilities}
    probabilities["forward"] = 1.0
    twist.signed_axis_probabilities = probabilities
    ranges = dict(twist.signed_axis_ranges)
    nominal = MICROBAN_LOCOMOTION_PRIOR_NOMINAL_FORWARD_VELOCITY_M_S
    ranges["forward"] = (nominal, nominal)
    twist.signed_axis_ranges = ranges
    twist.rel_standing_envs = 0.0
    twist.rel_forward_envs = 0.0
    twist.rel_rotation_envs = 0.0
    twist.rel_heading_envs = 0.0
    twist.rel_world_envs = 0.0
    twist.init_velocity_prob = 0.0
    twist.resampling_time_range = (1.0e9, 1.0e9)
    twist.debug_vis = False
    cfg.commands["foot_target"].debug_vis = False
    cfg.commands["hand_target"].debug_vis = False

    prior = cfg.commands["locomotion_prior"]
    prior.motion_file = str(prior_path)
    prior.expected_sha256 = expected_prior_sha256
    prior.enabled = True
    prior.debug_vis = False

    evidence = {
        "auto_reset": cfg.auto_reset,
        "episode_length_s": cfg.episode_length_s,
        "push_event_present": "push_robot" in cfg.events,
        "reset_events": sorted(reset_names),
        "startup_domain_randomization_events": sorted(startup_names),
    }
    return cfg, evidence


def _broadcast_action_parameter(
    value: torch.Tensor | float,
    *,
    reference: torch.Tensor,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return torch.broadcast_to(value, reference.shape)
    return torch.full_like(reference, float(value))


def _contact_layout(foot_contact: Any, robot: Any) -> dict[str, tuple[int, int]]:
    primaries = [
        slot.primary_name for slot in foot_contact._slots if slot.field_name == "found"
    ]
    if len(primaries) != 2 or len(set(primaries)) != 2:
        raise ValueError(f"Unexpected foot-contact primaries: {primaries}")
    site_for_primary = {"foot": "right_foot", "foot_2": "left_foot"}
    if set(primaries) != set(site_for_primary):
        raise ValueError(f"Unexpected Microban foot bodies: {primaries}")
    result: dict[str, tuple[int, int]] = {}
    for contact_index, primary in enumerate(primaries):
        side = "right" if primary == "foot" else "left"
        site_name = site_for_primary[primary]
        site_ids, resolved = robot.find_sites((site_name,), preserve_order=True)
        if tuple(resolved) != (site_name,):
            raise ValueError(f"Could not resolve foot site {site_name!r}")
        result[side] = (contact_index, int(site_ids[0]))
    return result


def _runtime_bam_evidence(robot: Any) -> dict[str, Any]:
    bam = [
        actuator
        for actuator in robot.actuators
        if actuator.__class__.__module__ == "bam.mjlab"
        and actuator.__class__.__name__ == "BamActuator"
    ]
    if len(bam) != 1:
        raise ValueError(f"Expected exactly one BAM actuator group, got {len(bam)}")
    actuator = bam[0]
    if tuple(actuator.target_names) != tuple(robot.joint_names):
        raise ValueError("BAM actuator must own all 21 robot joints in model order")
    if actuator.vin_tensor is None or actuator.vin_drop_gain is None:
        raise ValueError("BAM voltage and voltage-drop randomization must be active")
    if actuator._delay_buffer is None or not actuator.has_delay:
        raise ValueError("BAM command delay must be active")
    vin = actuator.vin_tensor.detach()
    drop = actuator.vin_drop_gain.detach()
    if float(vin.max() - vin.min()) <= 0.0 or float(drop.max() - drop.min()) <= 0.0:
        raise ValueError("BAM startup randomization did not vary across environments")
    return {
        "class": f"{actuator.__class__.__module__}.{actuator.__class__.__name__}",
        "delay_lag_physics_steps": [
            int(actuator.cfg.delay_min_lag),
            int(actuator.cfg.delay_max_lag),
        ],
        "joint_count": len(actuator.target_names),
        "vin_drop_gain_observed": [float(drop.min()), float(drop.max())],
        "vin_observed_v": [float(vin.min()), float(vin.max())],
    }


def _finite_env_mask(
    named_tensors: dict[str, torch.Tensor], num_envs: int
) -> torch.Tensor:
    result = torch.ones(
        num_envs, dtype=torch.bool, device=next(iter(named_tensors.values())).device
    )
    for name, tensor in named_tensors.items():
        if tensor.shape[0] != num_envs:
            raise ValueError(f"{name} does not have an environment-leading dimension")
        result &= torch.isfinite(tensor).reshape(num_envs, -1).all(dim=1)
    return result


def _roll_pitch_from_wxyz(
    quaternion: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return intrinsic XYZ roll/pitch from normalized WXYZ quaternions."""

    if quaternion.ndim != 2 or quaternion.shape[1] != 4:
        raise ValueError("Root quaternion tensor must have shape (num_envs, 4)")
    w, x, y, z = quaternion.unbind(dim=-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = torch.clamp(2.0 * (w * y - z * x), min=-1.0, max=1.0)
    return roll, torch.asin(sin_pitch)


def evaluate_dynamic_gate(
    *,
    prior_path: Path,
    expected_prior_sha256: str,
    robot_xml_path: Path,
    expected_robot_xml_sha256: str,
    static_receipt_path: Path,
    device: str,
) -> dict[str, Any]:
    """Run the 256-environment full-clip actuator replay and return a receipt."""

    prior_path = prior_path.expanduser().resolve()
    robot_xml_path = robot_xml_path.expanduser().resolve()
    prior_sha256 = sha256_file(prior_path)
    robot_sha256 = sha256_file(robot_xml_path)
    if prior_sha256 != expected_prior_sha256:
        raise ValueError("Locomotion-prior SHA-256 mismatch")
    if robot_sha256 != expected_robot_xml_sha256:
        raise ValueError("Robot XML SHA-256 mismatch")
    static_report, static_receipt_sha256 = load_and_validate_static_receipt(
        static_receipt_path,
        expected_prior_sha256=prior_sha256,
        expected_robot_xml_sha256=robot_sha256,
    )

    if not device.startswith("cuda"):
        raise ValueError("Dynamic gate requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    configure_torch_backends(allow_tf32=False, deterministic=True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    cfg, configuration_evidence = _configure_gate_env(
        prior_path=prior_path,
        expected_prior_sha256=prior_sha256,
    )

    env: ManagerBasedRlEnv | None = None
    try:
        env = ManagerBasedRlEnv(cfg=cfg, device=device)
        env.reset(seed=DYNAMIC_GATE_SEED)
        if env.num_envs != DYNAMIC_GATE_NUM_ENVS:
            raise ValueError("Dynamic gate environment count drifted")
        if not math.isclose(env.step_dt, 0.02, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("Dynamic gate requires an exact 50 Hz policy step")

        robot = env.scene["robot"]
        action = env.action_manager.get_term("joint_pos")
        prior = env.command_manager.get_term("locomotion_prior")
        twist = env.command_manager.get_term("twist")
        if not isinstance(prior, LocomotionPriorCommand):
            raise TypeError("locomotion_prior command did not build correctly")
        if not isinstance(twist, UniformVelocityCommandWithRotation):
            raise TypeError("twist command did not build correctly")
        if tuple(action.target_names) != MICROBAN_TELEOP_ACTION_JOINT_NAMES:
            raise ValueError("Dynamic gate action joint order mismatch")
        if tuple(
            prior.robot.joint_names[index] for index in prior.leg_joint_ids.tolist()
        ) != (MICROBAN_LOCOMOTION_PRIOR_LEG_JOINT_NAMES):
            raise ValueError("Dynamic gate prior leg joint order mismatch")

        nominal = MICROBAN_LOCOMOTION_PRIOR_NOMINAL_FORWARD_VELOCITY_M_S
        expected_twist = torch.tensor(
            (nominal, 0.0, 0.0), dtype=torch.float32, device=env.device
        ).expand(env.num_envs, -1)
        if not bool(
            torch.allclose(twist.vel_command_b, expected_twist, atol=1.0e-7, rtol=0.0)
        ):
            raise ValueError("Reset did not produce the exact nominal-forward command")
        if not bool(prior.eligible.all().item()):
            raise ValueError(
                "Not every environment is eligible for the locomotion prior"
            )
        if not bool(prior.teleported.all().item()) or bool(
            prior.launching.any().item()
        ):
            raise ValueError("Frame-109 reset must teleport every environment")
        if not bool(
            (prior.phase == MICROBAN_LOCOMOTION_PRIOR_START_FRAME).all().item()
        ):
            raise ValueError("Locomotion prior did not reset to frame 109")
        if not bool(torch.allclose(prior.phase_rate, torch.ones_like(prior.phase))):
            raise ValueError("Nominal-forward phase rate must be exactly one")

        bam_evidence = _runtime_bam_evidence(robot)
        foot_contact = env.scene.sensors["feet_ground_contact"]
        self_collision = env.scene.sensors["self_collision"]
        foot_layout = _contact_layout(foot_contact, robot)
        if _EXPECTED_FINISH_TERM not in env.termination_manager.active_terms:
            raise ValueError(
                f"Required termination term is missing: {_EXPECTED_FINISH_TERM}"
            )

        all_soft_limits = robot.data.soft_joint_pos_limits
        target_clip = action._clip
        if not bool(action.cfg.use_default_offset):
            raise ValueError("Dynamic gate requires absolute joint-position actions")
        action_soft_limits = all_soft_limits[:, action.target_ids]
        try:
            expanded_target_clip = torch.broadcast_to(
                target_clip, action_soft_limits.shape
            )
        except RuntimeError as exc:
            raise ValueError(
                "Action clip cannot broadcast to the entity soft-limit tensor"
            ) from exc
        action_clip_soft_limit_max_error = float(
            torch.abs(expanded_target_clip - action_soft_limits).max().item()
        )
        if not bool(
            torch.allclose(
                expanded_target_clip,
                action_soft_limits,
                atol=1.0e-6,
                rtol=0.0,
            )
        ):
            raise ValueError(
                "Action clip is not the absolute-radian entity soft-limit tensor; "
                f"max_error={action_clip_soft_limit_max_error:.9g} rad"
            )
        offset = _broadcast_action_parameter(action.offset, reference=action.raw_action)
        scale = _broadcast_action_parameter(action.scale, reference=action.raw_action)
        if bool((scale == 0.0).any().item()):
            raise ValueError("Action scale contains zero")

        root_min_by_env = robot.data.root_link_pos_w[:, 2].clone()
        max_soft_violation_by_joint = torch.zeros(
            len(robot.joint_names), dtype=torch.float32, device=env.device
        )
        max_projection_by_joint = torch.zeros(
            len(action.target_names), dtype=torch.float32, device=env.device
        )
        ever_airborne = {
            side: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            for side in ("left", "right")
        }
        ever_contact = {
            side: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            for side in ("left", "right")
        }
        foot_contact_env_steps = {"left": 0, "right": 0}
        foot_airborne_env_steps = {"left": 0, "right": 0}
        foot_slip_samples: dict[str, list[float]] = {"left": [], "right": []}
        simultaneous_flight_env_steps = 0
        simultaneous_flight_envs = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        self_collision_contacts = 0
        self_collision_env_steps = 0
        fall_envs = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        nonfinite_envs = torch.zeros_like(fall_envs)
        first_termination_seen = {
            name: torch.zeros_like(fall_envs)
            for name in env.termination_manager.active_terms
        }
        unattributed_termination_envs = torch.zeros_like(fall_envs)
        active_envs = torch.ones_like(fall_envs)
        full_clip_completed = torch.zeros_like(fall_envs)
        first_termination_step = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )
        first_exit_step = torch.full_like(first_termination_step, -1)
        completion_step = torch.full_like(first_termination_step, -1)
        vx_sum = torch.zeros(env.num_envs, dtype=torch.float64, device=env.device)
        xy_error_sum = torch.zeros_like(vx_sum)
        velocity_sample_count = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
        maximum_tracking_error_by_joint = torch.zeros(
            len(action.target_names), dtype=torch.float32, device=env.device
        )
        maximum_abs_root_roll_rad = 0.0
        maximum_abs_root_pitch_rad = 0.0
        step_trace: list[dict[str, Any]] = []
        steps_executed = 0
        target_phase_first: float | None = None
        target_phase_last: float | None = None

        initial_joint_pos = robot.data.joint_pos
        initial_violation = torch.maximum(
            torch.clamp(all_soft_limits[..., 0] - initial_joint_pos, min=0.0),
            torch.clamp(initial_joint_pos - all_soft_limits[..., 1], min=0.0),
        )
        max_soft_violation_by_joint = torch.maximum(
            max_soft_violation_by_joint, initial_violation.max(dim=0).values
        )

        maximum_steps = (
            MICROBAN_LOCOMOTION_PRIOR_END_FRAME
            - MICROBAN_LOCOMOTION_PRIOR_START_FRAME
            + 4
        )
        for step in range(maximum_steps):
            if not bool(active_envs.any().item()):
                break
            active_before = active_envs.clone()
            active_count = int(active_before.sum().item())
            if not bool(
                torch.allclose(
                    twist.vel_command_b[active_before],
                    expected_twist[active_before],
                    atol=1.0e-7,
                    rtol=0.0,
                )
            ):
                raise ValueError(f"Nominal-forward command drifted at step {step}")

            next_phase = torch.clamp(
                prior.phase + prior.phase_rate,
                max=float(MICROBAN_LOCOMOTION_PRIOR_END_FRAME),
            )
            active_next_phase = next_phase[active_before]
            phase_min = float(active_next_phase.min().item())
            phase_max = float(active_next_phase.max().item())
            if not math.isclose(phase_min, phase_max, rel_tol=0.0, abs_tol=0.0):
                raise ValueError(f"Active prior phases desynchronized at step {step}")
            if target_phase_first is None:
                target_phase_first = phase_min
            target_phase_last = phase_max
            expected_target_phase = float(
                min(
                    MICROBAN_LOCOMOTION_PRIOR_START_FRAME + step + 1,
                    MICROBAN_LOCOMOTION_PRIOR_END_FRAME,
                )
            )
            if not math.isclose(
                phase_min, expected_target_phase, rel_tol=0.0, abs_tol=0.0
            ):
                raise ValueError(
                    f"Unexpected next-frame phase at step {step}: {phase_min}"
                )
            desired = offset.clone()
            desired[:, prior.leg_action_ids] = prior._interpolate(
                prior.arrays.joint_pos, next_phase
            )
            clipped = torch.clamp(
                desired,
                min=expanded_target_clip[..., 0],
                max=expanded_target_clip[..., 1],
            )
            projection = torch.abs(clipped - desired)
            max_projection_by_joint = torch.maximum(
                max_projection_by_joint,
                projection[active_before].max(dim=0).values,
            )
            raw_action = (desired - offset) / scale
            raw_action[~active_before] = 0.0

            observations, rewards, terminated, truncated, _ = env.step(raw_action)
            steps_executed += 1

            physics_nonfinite = NanGuard.detect_nans(env.sim.data)
            finite = _finite_env_mask(
                {
                    "actor_observation": observations["actor"],
                    "joint_position": robot.data.joint_pos,
                    "joint_velocity": robot.data.joint_vel,
                    "processed_action": action._processed_actions,
                    "reward": rewards.unsqueeze(-1),
                    "root_pose": robot.data.root_link_pose_w,
                    "root_velocity": robot.data.root_link_vel_w,
                },
                env.num_envs,
            )
            step_nonfinite = (physics_nonfinite | ~finite) & active_before
            nonfinite_envs |= step_nonfinite
            valid_active = active_before & ~step_nonfinite
            valid_count = int(valid_active.sum().item())

            root_height = robot.data.root_link_pos_w[:, 2]
            root_min_by_env[valid_active] = torch.minimum(
                root_min_by_env[valid_active], root_height[valid_active]
            )
            fall_envs |= valid_active & (
                root_height < DEFAULT_DYNAMIC_THRESHOLDS.minimum_root_height_m
            )
            root_roll, root_pitch = _roll_pitch_from_wxyz(robot.data.root_link_quat_w)
            if valid_count:
                maximum_abs_root_roll_rad = max(
                    maximum_abs_root_roll_rad,
                    float(root_roll[valid_active].abs().max().item()),
                )
                maximum_abs_root_pitch_rad = max(
                    maximum_abs_root_pitch_rad,
                    float(root_pitch[valid_active].abs().max().item()),
                )

            joint_pos = robot.data.joint_pos
            soft_violation = torch.maximum(
                torch.clamp(all_soft_limits[..., 0] - joint_pos, min=0.0),
                torch.clamp(joint_pos - all_soft_limits[..., 1], min=0.0),
            )
            if valid_count:
                max_soft_violation_by_joint = torch.maximum(
                    max_soft_violation_by_joint,
                    soft_violation[valid_active].max(dim=0).values,
                )

            tracking_error = torch.abs(joint_pos[:, action.target_ids] - clipped)
            if valid_count:
                step_tracking_max = tracking_error[valid_active].max(dim=0).values
                maximum_tracking_error_by_joint = torch.maximum(
                    maximum_tracking_error_by_joint, step_tracking_max
                )
            else:
                step_tracking_max = torch.zeros_like(maximum_tracking_error_by_joint)

            collision_found = self_collision.data.found
            if collision_found is None:
                raise RuntimeError("self_collision sensor does not expose found")
            collision_mask = collision_found > 0
            self_collision_contacts += int(collision_found[valid_active].sum().item())
            self_collision_env_steps += int(
                (collision_mask.reshape(env.num_envs, -1).any(dim=1) & valid_active)
                .sum()
                .item()
            )

            contact_found = foot_contact.data.found
            if contact_found is None or contact_found.shape[1] != 2:
                raise RuntimeError("feet_ground_contact must expose two found columns")
            contacts: dict[str, torch.Tensor] = {}
            for side, (contact_index, site_id) in foot_layout.items():
                contact = contact_found[:, contact_index] > 0
                contacts[side] = contact
                active_contact = contact & valid_active
                active_airborne = ~contact & valid_active
                ever_contact[side] |= active_contact
                ever_airborne[side] |= active_airborne
                foot_contact_env_steps[side] += int(active_contact.sum().item())
                foot_airborne_env_steps[side] += int(active_airborne.sum().item())
                slip = robot.data.site_lin_vel_w[:, site_id, :2].norm(dim=-1)
                foot_slip_samples[side].extend(
                    slip[active_contact].detach().cpu().tolist()
                )
            simultaneous = ~contacts["left"] & ~contacts["right"] & valid_active
            simultaneous_flight_envs |= simultaneous
            simultaneous_flight_env_steps += int(simultaneous.sum().item())

            actual_xy = robot.data.root_link_lin_vel_b[:, :2]
            vx_sum[valid_active] += actual_xy[valid_active, 0].to(dtype=torch.float64)
            xy_error = torch.linalg.vector_norm(
                expected_twist[:, :2] - actual_xy, dim=-1
            ).to(dtype=torch.float64)
            xy_error_sum[valid_active] += xy_error[valid_active]
            velocity_sample_count[valid_active] += 1

            current_terms: dict[str, torch.Tensor] = {}
            any_current_term = torch.zeros_like(active_envs)
            for name in env.termination_manager.active_terms:
                current = env.termination_manager.get_term(name) & active_before
                current_terms[name] = current
                any_current_term |= current
            done = terminated | truncated
            done_active = done & active_before
            unattributed = done_active & ~any_current_term
            unattributed_termination_envs |= unattributed
            if bool(done_active.any().item()):
                first_termination_step[done_active] = step
                for name, current in current_terms.items():
                    first_termination_seen[name] |= current

            expected_now = current_terms[_EXPECTED_FINISH_TERM]
            unexpected_now = unattributed.clone()
            for name, current in current_terms.items():
                if name != _EXPECTED_FINISH_TERM:
                    unexpected_now |= current
            completed_now = (
                done_active
                & expected_now
                & ~unexpected_now
                & ~step_nonfinite
                & prior.finished
                & (prior.phase == MICROBAN_LOCOMOTION_PRIOR_END_FRAME)
            )
            full_clip_completed |= completed_now
            completion_step[completed_now] = step

            removed_now = done_active | step_nonfinite
            first_exit_step[removed_now] = step
            active_envs &= ~removed_now

            trace: dict[str, Any] = {
                "active_envs_after_step": int(active_envs.sum().item()),
                "active_envs_before_step": active_count,
                "nonfinite_envs_this_step": int(step_nonfinite.sum().item()),
                "policy_step_zero_based": step,
                "source_target_frame": phase_min,
                "terminations_this_step": {
                    name: int(mask.sum().item())
                    for name, mask in current_terms.items()
                    if bool(mask.any().item())
                },
            }
            if valid_count:
                valid_root_height = root_height[valid_active]
                valid_roll_abs = root_roll[valid_active].abs()
                valid_pitch_abs = root_pitch[valid_active].abs()
                valid_tracking = tracking_error[valid_active]
                trace.update(
                    {
                        "left_contact_envs": int(
                            (contacts["left"] & valid_active).sum().item()
                        ),
                        "right_contact_envs": int(
                            (contacts["right"] & valid_active).sum().item()
                        ),
                        "simultaneous_flight_envs": int(simultaneous.sum().item()),
                        "root_height_min_m": float(valid_root_height.min().item()),
                        "root_height_p05_m": _percentile(valid_root_height, 5.0),
                        "root_roll_abs_max_deg": math.degrees(
                            float(valid_roll_abs.max().item())
                        ),
                        "root_roll_abs_p95_deg": math.degrees(
                            _percentile(valid_roll_abs, 95.0)
                        ),
                        "root_pitch_abs_max_deg": math.degrees(
                            float(valid_pitch_abs.max().item())
                        ),
                        "root_pitch_abs_p95_deg": math.degrees(
                            _percentile(valid_pitch_abs, 95.0)
                        ),
                        "tracking_error_max_rad": float(valid_tracking.max().item()),
                        "tracking_error_p95_rad": _percentile(
                            valid_tracking.reshape(-1), 95.0
                        ),
                        "tracking_error_max_rad_by_joint": dict(
                            zip(
                                action.target_names,
                                step_tracking_max.detach().cpu().tolist(),
                                strict=True,
                            )
                        ),
                    }
                )
            step_trace.append(trace)

            if not bool(active_envs.any().item()):
                break

            # ``auto_reset=False`` preserves every true terminal observation.
            # Once captured above, reset all exited rows so MjLab permits another
            # vector step, then permanently exclude those rows from every metric.
            inactive_ids = (~active_envs).nonzero(as_tuple=False).squeeze(-1)
            if inactive_ids.numel() > 0:
                env.reset(env_ids=inactive_ids)

        if steps_executed <= 0:
            raise RuntimeError("Dynamic gate executed no simulation steps")

        unexpected = unattributed_termination_envs.clone()
        for name, mask in first_termination_seen.items():
            if name != _EXPECTED_FINISH_TERM:
                unexpected |= mask
        if "fell_over" in first_termination_seen:
            fall_envs |= first_termination_seen["fell_over"]

        safe_velocity_sample_count = velocity_sample_count.clamp_min(1).to(
            dtype=torch.float64
        )
        mean_vx = vx_sum / safe_velocity_sample_count
        mean_xy_error = xy_error_sum / safe_velocity_sample_count
        vx_p05 = _percentile(mean_vx, 5.0)
        xy_mae_p95 = _percentile(mean_xy_error, 95.0)
        expected_completion_step = DYNAMIC_GATE_EXPECTED_POLICY_STEPS - 1
        source_frame_sequence_complete = (
            target_phase_first == float(MICROBAN_LOCOMOTION_PRIOR_START_FRAME + 1)
            and target_phase_last == float(MICROBAN_LOCOMOTION_PRIOR_END_FRAME)
            and bool(full_clip_completed.all().item())
            and bool((completion_step == expected_completion_step).all().item())
        )
        measurements = {
            "executed_policy_steps": steps_executed,
            "fall_envs": int(fall_envs.sum().item()),
            "forward_velocity_p05_m_s": vx_p05,
            "full_clip_completed_envs": int(full_clip_completed.sum().item()),
            "left_foot_airborne_envs": int(ever_airborne["left"].sum().item()),
            "maximum_soft_limit_violation_rad": float(
                max_soft_violation_by_joint.max().item()
            ),
            "maximum_target_projection_rad": float(
                max_projection_by_joint.max().item()
            ),
            "minimum_root_height_m": float(root_min_by_env.min().item()),
            "nonfinite_envs": int(nonfinite_envs.sum().item()),
            "right_foot_airborne_envs": int(ever_airborne["right"].sum().item()),
            "self_collision_contacts": self_collision_contacts,
            "simultaneous_flight_env_steps": simultaneous_flight_env_steps,
            "source_frame_sequence_complete": source_frame_sequence_complete,
            "unexpected_termination_envs": int(unexpected.sum().item()),
            "xy_velocity_mae_p95_m_s": xy_mae_p95,
        }
        checks = dynamic_gate_checks(measurements)

        first_termination_steps = [
            int(value) for value in first_termination_step.detach().cpu().tolist()
        ]
        first_exit_steps = [
            int(value) for value in first_exit_step.detach().cpu().tolist()
        ]
        completion_steps = [
            int(value) for value in completion_step.detach().cpu().tolist()
        ]

        per_foot = {}
        for side in ("left", "right"):
            samples = torch.tensor(foot_slip_samples[side], dtype=torch.float64)
            per_foot[side] = {
                "airborne_env_steps": foot_airborne_env_steps[side],
                "contact_env_steps": foot_contact_env_steps[side],
                "envs_ever_airborne": int(ever_airborne[side].sum().item()),
                "envs_ever_in_contact": int(ever_contact[side].sum().item()),
                "slip_while_in_contact": (
                    _distribution(samples, units="m_s")
                    if samples.numel()
                    else {"count": 0, "units": "m_s"}
                ),
            }

        payload = {
            "checks": checks,
            "configuration": {
                **configuration_evidence,
                "end_frame_inclusive": MICROBAN_LOCOMOTION_PRIOR_END_FRAME,
                "num_envs": DYNAMIC_GATE_NUM_ENVS,
                "policy_step_hz": 1.0 / env.step_dt,
                "seed": DYNAMIC_GATE_SEED,
                "start_frame_inclusive": MICROBAN_LOCOMOTION_PRIOR_START_FRAME,
                "teacher_target": "next_source_frame_absolute_joint_position",
                "thresholds": asdict(DEFAULT_DYNAMIC_THRESHOLDS),
            },
            "diagnostics": {
                "bam": bam_evidence,
                "action_target_clip": {
                    "matches_entity_soft_limits": True,
                    "maximum_soft_limit_tensor_difference_rad": (
                        action_clip_soft_limit_max_error
                    ),
                    "semantics": "absolute_joint_position_rad",
                    "units": "rad",
                },
                "feet": per_foot,
                "foot_contact_column_order": [
                    {
                        "column": column,
                        "primary_body": ("foot" if side == "right" else "foot_2"),
                        "side": side,
                        "site_id": site_id,
                        "site_name": f"{side}_foot",
                    }
                    for side, (column, site_id) in sorted(
                        foot_layout.items(), key=lambda item: item[1][0]
                    )
                ],
                "forward_velocity_per_env_mean": _distribution(mean_vx, units="m_s"),
                "lifecycle": {
                    "completion_step": summarize_first_exit_steps(
                        completion_steps, expected_envs=env.num_envs
                    ),
                    "first_exit_step": summarize_first_exit_steps(
                        first_exit_steps, expected_envs=env.num_envs
                    ),
                    "first_termination_step": summarize_first_exit_steps(
                        first_termination_steps, expected_envs=env.num_envs
                    ),
                    "unattributed_termination_envs": int(
                        unattributed_termination_envs.sum().item()
                    ),
                },
                "joint_soft_limit_max_violation_rad_by_joint": dict(
                    zip(
                        robot.joint_names,
                        max_soft_violation_by_joint.detach().cpu().tolist(),
                        strict=True,
                    )
                ),
                "joint_tracking_error_max_rad_by_joint": dict(
                    zip(
                        action.target_names,
                        maximum_tracking_error_by_joint.detach().cpu().tolist(),
                        strict=True,
                    )
                ),
                "root_orientation_absolute_max_deg": {
                    "pitch": math.degrees(maximum_abs_root_pitch_rad),
                    "roll": math.degrees(maximum_abs_root_roll_rad),
                },
                "root_height_min_per_env": _distribution(root_min_by_env, units="m"),
                "self_collision_env_steps": self_collision_env_steps,
                "simultaneous_flight_envs": int(simultaneous_flight_envs.sum().item()),
                "step_trace": step_trace,
                "target_projection_max_rad_by_joint": dict(
                    zip(
                        action.target_names,
                        max_projection_by_joint.detach().cpu().tolist(),
                        strict=True,
                    )
                ),
                "termination_counts_by_term": {
                    name: int(mask.sum().item())
                    for name, mask in first_termination_seen.items()
                },
                "teacher_source_phase": {
                    "first_target_frame": target_phase_first,
                    "initial_reset_frame": float(MICROBAN_LOCOMOTION_PRIOR_START_FRAME),
                    "last_target_frame": target_phase_last,
                    "required_policy_steps": DYNAMIC_GATE_EXPECTED_POLICY_STEPS,
                },
                "velocity_sample_count_per_env": _distribution(
                    velocity_sample_count, units="policy_steps"
                ),
                "xy_velocity_error_per_env_mean": _distribution(
                    mean_xy_error, units="m_s"
                ),
            },
            "inputs": {
                "locomotion_prior": {
                    "path": _portable_path(prior_path),
                    "sha256": prior_sha256,
                    "size_bytes": prior_path.stat().st_size,
                },
                "robot_xml": {
                    "path": _portable_path(robot_xml_path),
                    "sha256": robot_sha256,
                    "size_bytes": robot_xml_path.stat().st_size,
                },
                "static_suitability_receipt": {
                    "path": _portable_path(static_receipt_path),
                    "receipt_payload_sha256": static_report["receipt_payload_sha256"],
                    "sha256": static_receipt_sha256,
                },
            },
            "measurements": measurements,
            "method": {
                "actions": (
                    "18 raw actions reconstruct absolute next-frame targets; "
                    "six arms hold their configured default offsets"
                ),
                "completion": (
                    "capture each environment's first terminal observation with "
                    "auto_reset=false; reset and permanently mask exited rows so "
                    "the remaining environments continue; only "
                    "locomotion_prior_clip_finished is expected"
                ),
                "velocity_reduction": (
                    "active-lifetime time mean per environment, then p05 forward / "
                    "p95 XY error"
                ),
            },
            "runtime": {
                "cuda_device": torch.cuda.get_device_name(torch.device(device)),
                "cuda_version": torch.version.cuda,
                "torch_version": torch.__version__,
            },
            "steps_executed": steps_executed,
        }
        return finalize_dynamic_report(payload)
    finally:
        if env is not None:
            env.close()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, default=DEFAULT_LOCOMOTION_PRIOR_PATH)
    parser.add_argument(
        "--expected-prior-sha256", default=DEFAULT_LOCOMOTION_PRIOR_SHA256
    )
    parser.add_argument("--robot-xml", type=Path, default=DEFAULT_ROBOT_XML_PATH)
    parser.add_argument("--expected-robot-xml-sha256", default=DEFAULT_ROBOT_XML_SHA256)
    parser.add_argument("--static-receipt", type=Path, default=DEFAULT_STATIC_RECEIPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_DYNAMIC_RECEIPT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        report = evaluate_dynamic_gate(
            prior_path=args.prior,
            expected_prior_sha256=args.expected_prior_sha256,
            robot_xml_path=args.robot_xml,
            expected_robot_xml_sha256=args.expected_robot_xml_sha256,
            static_receipt_path=args.static_receipt,
            device=args.device,
        )
        publish_suitability_receipt(args.output, report, force=args.force)
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"dynamic locomotion-prior gate error: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "failed_checks": report["summary"]["failed_checks"],
                "output": str(args.output),
                "status": report["status"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
