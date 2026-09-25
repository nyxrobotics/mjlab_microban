# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Headless full-clip gate for a learned Microban tracking checkpoint.

This program is simulation-only.  It opens no viewer, network listener, PICO
connection, or robot transport.  A checkpoint is evaluated deterministically in
two 256-environment passes: source frame 0 initializes simulator state, then the
policy tracks target frames 1 through 267 (267 control transitions):

``nominal``
    No MjLab startup event randomization.  The configured BAM voltage, voltage
    drop and measured command-delay envelope remain active, making this stricter
    than an ideal-actuator replay.

``robust``
    The same actuator envelope plus the tracking task's startup friction, base-COM
    and encoder-bias randomization and actor-observation corruption.  Interval
    pushes are excluded from this bounded capability gate.

Passing this receipt does not authorize physical deployment.  The exported
tracking artifact is still a fixed-motion, offline-only 99-observation policy and
is not compatible with the current live PICO 83-observation runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommand
from mjlab.utils.lab_api.math import quat_apply, quat_error_magnitude
from mjlab.utils.nan_guard import NanGuard
from mjlab.utils.torch import configure_torch_backends

from mjlab_microban.locomotion_prior_suitability import (
    DEFAULT_LOCOMOTION_PRIOR_PATH,
    DEFAULT_LOCOMOTION_PRIOR_SHA256,
    DEFAULT_ROBOT_XML_PATH,
    DEFAULT_ROBOT_XML_SHA256,
    PROJECT_ROOT,
    sha256_file,
)
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_TELEOP_ACTION_JOINT_NAMES,
)
from mjlab_microban.tasks.microban_tracking_env_cfg import (
    MICROBAN_BODY_NAMES,
    MICROBAN_END_EFFECTOR_BODY_NAMES,
    MICROBAN_JOINT_NAMES,
    MICROBAN_TRACKED_BODY_NAMES,
    MicrobanTrackingRlCfg,
    make_microban_tracking_env_cfg,
)
from mjlab_microban.tasks.microban_tracking_policy_export import (
    MicrobanTrackingOnPolicyRunner,
)
from mjlab_microban.tracking_checkpoint_gate import (
    DEFAULT_TRACKING_CHECKPOINT_THRESHOLDS,
    TRACKING_ACTOR_KIND_BOUNDED_LOG,
    TRACKING_ACTOR_KIND_UNBOUNDED_LOG,
    TRACKING_ACTOR_KIND_UNBOUNDED_SCALAR,
    TRACKING_ACTOR_OBSERVATION_SCHEMA,
    TRACKING_ACTOR_OBSERVATION_WIDTH,
    TRACKING_CHECKPOINT_GATE_END_FRAME,
    TRACKING_CHECKPOINT_GATE_EXPECTED_STEPS,
    TRACKING_CHECKPOINT_GATE_FIRST_TARGET_FRAME,
    TRACKING_CHECKPOINT_GATE_FRAME_COUNT,
    TRACKING_CHECKPOINT_GATE_INITIAL_FRAME,
    TRACKING_CHECKPOINT_GATE_MODES,
    TRACKING_CHECKPOINT_GATE_NUM_ENVS,
    TRACKING_CHECKPOINT_GATE_SEED,
    classify_tracking_actor_state_keys,
    finalize_tracking_checkpoint_report,
    tracking_checkpoint_checks,
    tracking_checkpoint_exit_code,
)

DEFAULT_OUTPUT = Path("artifacts/microban_tracking_checkpoint_walk004_gate.json")
_TRACKING_GATE_RUNTIME_SOURCE_PATHS = (
    PROJECT_ROOT / "src/mjlab_microban/tracking_checkpoint_gate.py",
    PROJECT_ROOT / "src/mjlab_microban/tasks/microban_tracking_env_cfg.py",
    PROJECT_ROOT / "src/mjlab_microban/tasks/microban_tracking_mdp.py",
    PROJECT_ROOT / "src/mjlab_microban/tasks/microban_tracking_policy_export.py",
    PROJECT_ROOT / "src/mjlab_microban/robot/microban_constants.py",
    PROJECT_ROOT / "src/mjlab_microban/robot/xc330_params.json",
    PROJECT_ROOT / "uv.lock",
)
_EXPECTED_ROBUST_STARTUP_EVENTS = {
    "base_com",
    "encoder_bias",
    "foot_friction",
}
_ROOT_HEIGHT_FALL_THRESHOLD_M = (
    DEFAULT_TRACKING_CHECKPOINT_THRESHOLDS.minimum_root_height_m
)


def _portable_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _runtime_source_evidence() -> list[dict[str, Any]]:
    """Hash every local source/config input that can change gate dynamics."""

    evidence: list[dict[str, Any]] = []
    for path in _TRACKING_GATE_RUNTIME_SOURCE_PATHS:
        if not path.is_file():
            raise FileNotFoundError(f"Tracking gate runtime source is missing: {path}")
        evidence.append(
            {
                "path": _portable_path(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return evidence


def _percentile(values: torch.Tensor, percentile: float) -> float:
    array = values.detach().to(dtype=torch.float64).cpu().numpy()
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Percentile input must be non-empty and finite")
    return float(np.percentile(array, percentile, method="linear"))


def _distribution(values: torch.Tensor, *, units: str) -> dict[str, Any]:
    finite = values.detach().to(dtype=torch.float64)
    finite = finite[torch.isfinite(finite)]
    if finite.numel() == 0:
        return {
            "count": 0,
            "max": None,
            "mean": None,
            "min": None,
            "p05": None,
            "p50": None,
            "p95": None,
            "units": units,
        }
    return {
        "count": int(finite.numel()),
        "max": float(finite.max().item()),
        "mean": float(finite.mean().item()),
        "min": float(finite.min().item()),
        "p05": _percentile(finite, 5.0),
        "p50": _percentile(finite, 50.0),
        "p95": _percentile(finite, 95.0),
        "units": units,
    }


def _publish_json_report(path: Path, report: dict[str, Any]) -> None:
    """Publish one complete receipt atomically."""

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            report,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary_path = Path(stream.name)
        stream.write(encoded)
        stream.flush()
    try:
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _validate_motion_archive(path: Path) -> dict[str, Any]:
    """Validate the fixed full-clip shape/order needed by the evaluator."""

    with np.load(path, allow_pickle=False) as archive:
        required_shapes = {
            "joint_pos": (TRACKING_CHECKPOINT_GATE_FRAME_COUNT, 21),
            "joint_vel": (TRACKING_CHECKPOINT_GATE_FRAME_COUNT, 21),
            "body_pos_w": (TRACKING_CHECKPOINT_GATE_FRAME_COUNT, 22, 3),
            "body_quat_w": (TRACKING_CHECKPOINT_GATE_FRAME_COUNT, 22, 4),
            "body_lin_vel_w": (TRACKING_CHECKPOINT_GATE_FRAME_COUNT, 22, 3),
            "body_ang_vel_w": (TRACKING_CHECKPOINT_GATE_FRAME_COUNT, 22, 3),
        }
        missing = sorted(set(required_shapes) - set(archive.files))
        if missing:
            raise ValueError(f"Tracking motion archive is missing arrays: {missing}")
        for name, expected_shape in required_shapes.items():
            values = np.asarray(archive[name])
            if values.shape != expected_shape:
                raise ValueError(
                    f"Tracking motion {name} shape {values.shape} != {expected_shape}"
                )
            if not np.isfinite(values).all():
                raise ValueError(f"Tracking motion {name} contains non-finite values")

        joint_names = tuple(str(value) for value in archive["joint_names"].tolist())
        body_names = tuple(str(value) for value in archive["body_names"].tolist())
        if joint_names != MICROBAN_JOINT_NAMES:
            raise ValueError("Tracking motion joint order does not match Microban")
        if body_names != MICROBAN_BODY_NAMES:
            raise ValueError("Tracking motion body order does not match Microban")
        fps_values = np.asarray(archive["fps"], dtype=np.float64).reshape(-1)
        if fps_values.size != 1 or not math.isclose(
            float(fps_values[0]), 50.0, rel_tol=0.0, abs_tol=1.0e-9
        ):
            raise ValueError("Tracking checkpoint gate requires an exact 50 Hz motion")

        root_index = body_names.index("trunk")
        root_pos = np.asarray(archive["body_pos_w"], dtype=np.float64)[:, root_index]
        root_lin_vel = np.asarray(archive["body_lin_vel_w"], dtype=np.float64)[
            :, root_index
        ]
        return {
            "body_count": len(body_names),
            "fps": float(fps_values[0]),
            "frame_count": int(required_shapes["joint_pos"][0]),
            "joint_count": len(joint_names),
            "reference_root_displacement_xyz_m": (root_pos[-1] - root_pos[0]).tolist(),
            "reference_root_mean_velocity_xyz_m_s": root_lin_vel.mean(axis=0).tolist(),
        }


def _checkpoint_actor_contract(path: Path) -> tuple[str, bool]:
    """Identify distribution and observation-normalizer state in a checkpoint.

    Tracking experiments made before the raw-actor correction contain an
    ``obs_normalizer`` in the actor state.  Reconstructing those weights with
    the current raw-actor config makes strict loading fail before the evaluator
    can produce the rejection receipt it exists to preserve.  The checkpoint
    keys are authoritative here: this only reconstructs the recorded model
    topology and never changes the acceptance thresholds.
    """

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("Tracking checkpoint root must be a mapping")
    actor_state = payload.get("actor_state_dict")
    if isinstance(actor_state, Mapping):
        keys = tuple(str(key) for key in actor_state)
        return (
            classify_tracking_actor_state_keys(keys),
            any(key.startswith("obs_normalizer.") for key in keys),
        )
    legacy_state = payload.get("model_state_dict")
    if isinstance(legacy_state, Mapping):
        keys = tuple(str(key) for key in legacy_state)
        return (
            classify_tracking_actor_state_keys(keys),
            any(key.startswith("obs_normalizer.") for key in keys),
        )
    raise ValueError("Tracking checkpoint contains no actor state")


def _agent_cfg_for_checkpoint(
    actor_kind: str,
    *,
    actor_obs_normalization: bool,
) -> tuple[Any, dict[str, str | bool]]:
    """Reconstruct old unbounded actors without reinterpreting their weights."""

    agent_cfg = deepcopy(MicrobanTrackingRlCfg)
    if actor_kind in {
        TRACKING_ACTOR_KIND_UNBOUNDED_SCALAR,
        TRACKING_ACTOR_KIND_UNBOUNDED_LOG,
    }:
        std_type = (
            "scalar" if actor_kind == TRACKING_ACTOR_KIND_UNBOUNDED_SCALAR else "log"
        )
        agent_cfg.actor.distribution_cfg = {
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": std_type,
        }
        agent_cfg.algorithm.class_name = "PPO"
    elif actor_kind != TRACKING_ACTOR_KIND_BOUNDED_LOG:
        raise ValueError(f"Unsupported tracking checkpoint actor kind: {actor_kind}")
    agent_cfg.actor.obs_normalization = actor_obs_normalization
    return agent_cfg, {
        "checkpoint_actor_distribution": actor_kind,
        "checkpoint_actor_observation_normalization": actor_obs_normalization,
        "checkpoint_algorithm_class": agent_cfg.algorithm.class_name,
    }


def _feet_ground_sensor_cfg() -> ContactSensorCfg:
    return ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(foot|foot_2)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )


def _configure_evaluation(
    *, mode: str, motion_path: Path
) -> tuple[Any, dict[str, Any]]:
    if mode not in TRACKING_CHECKPOINT_GATE_MODES:
        raise ValueError(f"Unsupported tracking gate mode: {mode}")
    cfg = make_microban_tracking_env_cfg(play=True, motion_file=motion_path)
    cfg.scene.num_envs = TRACKING_CHECKPOINT_GATE_NUM_ENVS
    cfg.seed = TRACKING_CHECKPOINT_GATE_SEED
    cfg.auto_reset = False
    cfg.episode_length_s = (TRACKING_CHECKPOINT_GATE_EXPECTED_STEPS + 2) * (
        cfg.decimation * cfg.sim.mujoco.timestep
    )
    cfg.curriculum = {}
    # The nominal pass isolates deterministic policy capability.  The robust
    # pass additionally exercises the same actor-observation corruption used in
    # training; it is reproducible because the environment seed is fixed.
    cfg.observations["actor"].enable_corruption = mode == "robust"
    cfg.commands["motion"].debug_vis = False

    startup_events = {
        name: term for name, term in cfg.events.items() if term.mode == "startup"
    }
    if set(startup_events) != _EXPECTED_ROBUST_STARTUP_EVENTS:
        raise ValueError(
            f"Tracking startup event set drifted: {sorted(startup_events)}"
        )
    cfg.events = {} if mode == "nominal" else startup_events

    sensor_names = tuple(sensor.name for sensor in cfg.scene.sensors)
    if sensor_names != ("self_collision",):
        raise ValueError(f"Unexpected tracking sensor set: {sensor_names}")
    cfg.scene.sensors = (*cfg.scene.sensors, _feet_ground_sensor_cfg())

    return cfg, {
        "actor_observation_corruption": (cfg.observations["actor"].enable_corruption),
        "auto_reset": cfg.auto_reset,
        "mode": mode,
        "startup_domain_randomization_events": sorted(cfg.events),
    }


def _broadcast_action_parameter(
    value: torch.Tensor | float,
    *,
    reference: torch.Tensor,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return torch.broadcast_to(value, reference.shape)
    return torch.full_like(reference, float(value))


def _finite_env_mask(
    named_tensors: dict[str, torch.Tensor], num_envs: int
) -> torch.Tensor:
    result = torch.ones(
        num_envs,
        dtype=torch.bool,
        device=next(iter(named_tensors.values())).device,
    )
    for name, tensor in named_tensors.items():
        if tensor.shape[0] != num_envs:
            raise ValueError(f"{name} does not have an environment-leading dimension")
        result &= torch.isfinite(tensor).reshape(num_envs, -1).all(dim=1)
    return result


def _contact_layout(foot_contact: Any, robot: Any) -> dict[str, tuple[int, int]]:
    primaries = [
        slot.primary_name for slot in foot_contact._slots if slot.field_name == "found"
    ]
    if len(primaries) != 2 or set(primaries) != {"foot", "foot_2"}:
        raise ValueError(f"Unexpected tracking foot contact bodies: {primaries}")
    result: dict[str, tuple[int, int]] = {}
    for contact_index, primary in enumerate(primaries):
        side = "right" if primary == "foot" else "left"
        site_name = f"{side}_foot"
        site_ids, resolved = robot.find_sites((site_name,), preserve_order=True)
        if tuple(resolved) != (site_name,):
            raise ValueError(f"Could not resolve tracking foot site {site_name!r}")
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
        raise ValueError(f"Expected one BAM actuator group, got {len(bam)}")
    actuator = bam[0]
    if actuator.vin_tensor is None or actuator.vin_drop_gain is None:
        raise ValueError("Tracking BAM voltage/drop envelope is not active")
    if actuator._delay_buffer is None or not actuator.has_delay:
        raise ValueError("Tracking BAM command delay is not active")
    vin = actuator.vin_tensor.detach()
    drop = actuator.vin_drop_gain.detach()
    if float(vin.max() - vin.min()) <= 0.0 or float(drop.max() - drop.min()) <= 0.0:
        raise ValueError(
            "Tracking BAM voltage/drop envelope did not vary across environments"
        )
    return {
        "class": f"{actuator.__class__.__module__}.{actuator.__class__.__name__}",
        "delay_lag_physics_steps": [
            int(actuator.cfg.delay_min_lag),
            int(actuator.cfg.delay_max_lag),
        ],
        "joint_count": len(actuator.target_names),
        "vin_drop_gain_observed": [
            float(drop.min().item()),
            float(drop.max().item()),
        ],
        "vin_observed_v": [
            float(vin.min().item()),
            float(vin.max().item()),
        ],
    }


def _safe_per_env_mean(total: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
    result = torch.full_like(total, float("nan"), dtype=torch.float64)
    valid = count > 0
    result[valid] = total[valid] / count[valid].to(dtype=torch.float64)
    return result


def _termination_masks(env: ManagerBasedRlEnv) -> dict[str, torch.Tensor]:
    return {
        name: env.termination_manager.get_term(name).clone()
        for name in env.termination_manager.active_terms
    }


def _install_non_wrapping_motion_advance(command: MotionCommand) -> None:
    """Clamp at the final frame so post-step measurements are not teleported.

    MjLab's training command resamples (and writes a newly sampled reference
    state) as soon as it advances past the last frame.  That is appropriate for
    continuing training episodes but would replace the actual frame-267 state
    before this evaluator can inspect it.
    """

    def advance_without_wrap(self: MotionCommand) -> None:
        self.time_steps.add_(1)
        self.time_steps.clamp_(max=TRACKING_CHECKPOINT_GATE_END_FRAME)
        self.update_relative_body_poses()

    command._update_command = MethodType(  # type: ignore[method-assign]
        advance_without_wrap, command
    )


def _evaluate_pass(
    *,
    mode: str,
    checkpoint: Path,
    checkpoint_actor_kind: str,
    checkpoint_actor_obs_normalization: bool,
    motion_path: Path,
    device: str,
) -> dict[str, Any]:
    cfg, configuration_evidence = _configure_evaluation(
        mode=mode, motion_path=motion_path
    )
    env: ManagerBasedRlEnv | None = None
    wrapped: RslRlVecEnvWrapper | None = None
    try:
        env = ManagerBasedRlEnv(cfg=cfg, device=device)
        agent_cfg, checkpoint_evidence = _agent_cfg_for_checkpoint(
            checkpoint_actor_kind,
            actor_obs_normalization=checkpoint_actor_obs_normalization,
        )
        agent_cfg.seed = TRACKING_CHECKPOINT_GATE_SEED
        if agent_cfg.clip_actions is not None:
            raise ValueError("Tracking checkpoint gate requires no wrapper action clip")
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = MicrobanTrackingOnPolicyRunner(
            wrapped,
            asdict(agent_cfg),
            log_dir=None,
            device=device,
        )
        runner.load(str(checkpoint), map_location=device)
        policy = runner.get_inference_policy(device=device)

        # RslRlVecEnvWrapper resets once during construction.  Reset again with
        # the receipt seed immediately before taking the measured initial state.
        env.reset(seed=TRACKING_CHECKPOINT_GATE_SEED)
        observations = wrapped.get_observations()
        if tuple(observations["actor"].shape) != (
            env.num_envs,
            TRACKING_ACTOR_OBSERVATION_WIDTH,
        ):
            raise ValueError(
                "Tracking actor observation shape drifted: "
                f"{tuple(observations['actor'].shape)}"
            )
        actor_term_names = tuple(env.observation_manager.active_terms["actor"])
        expected_term_names = tuple(
            name for name, _ in TRACKING_ACTOR_OBSERVATION_SCHEMA
        )
        if actor_term_names != expected_term_names:
            raise ValueError(
                f"Tracking actor term order {actor_term_names} != {expected_term_names}"
            )
        if not math.isclose(env.step_dt, 0.02, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("Tracking checkpoint gate requires exact 50 Hz steps")

        robot = env.scene["robot"]
        action = env.action_manager.get_term("joint_pos")
        command = env.command_manager.get_term("motion")
        if not isinstance(command, MotionCommand):
            raise TypeError("Tracking motion command did not build correctly")
        if tuple(action.target_names) != MICROBAN_TELEOP_ACTION_JOINT_NAMES:
            raise ValueError("Tracking action joint order mismatch")
        if tuple(command.cfg.body_names) != MICROBAN_TRACKED_BODY_NAMES:
            raise ValueError("Tracking body order mismatch")
        if int(command.motion.time_step_total) != TRACKING_CHECKPOINT_GATE_FRAME_COUNT:
            raise ValueError("Tracking motion does not contain exactly 268 frames")
        if not bool(
            (command.time_steps == TRACKING_CHECKPOINT_GATE_FIRST_TARGET_FRAME).all()
        ):
            raise ValueError(
                "Tracking reset did not initialize frame 0 and expose frame 1 as "
                "the first policy target"
            )
        _install_non_wrapping_motion_advance(command)

        all_soft_limits = robot.data.soft_joint_pos_limits
        action_soft_limits = all_soft_limits[:, action.target_ids]
        if not bool(action.cfg.use_default_offset):
            raise ValueError(
                "Tracking policy actions must be default-relative joint deltas"
            )
        expanded_clip = torch.broadcast_to(action._clip, action_soft_limits.shape)
        clip_tensor_max_error = float(
            torch.abs(expanded_clip - action_soft_limits).max().item()
        )
        if not bool(
            torch.allclose(
                expanded_clip,
                action_soft_limits,
                atol=1.0e-6,
                rtol=0.0,
            )
        ):
            raise ValueError("Tracking action clip differs from entity soft limits")
        offset = _broadcast_action_parameter(action.offset, reference=action.raw_action)
        scale = _broadcast_action_parameter(action.scale, reference=action.raw_action)
        expected_offset = robot.data.default_joint_pos[:, action.target_ids]
        offset_max_error = float(torch.abs(offset - expected_offset).max().item())
        if not bool(torch.allclose(offset, expected_offset, atol=1.0e-7, rtol=0.0)):
            raise ValueError(
                "Tracking action offset differs from the default joint pose"
            )
        if not bool(torch.allclose(scale, torch.ones_like(scale))):
            raise ValueError("Tracking checkpoint gate requires unit action scale")

        foot_contact = env.scene.sensors["feet_ground_contact"]
        self_collision = env.scene.sensors["self_collision"]
        foot_layout = _contact_layout(foot_contact, robot)
        bam_evidence = _runtime_bam_evidence(robot)
        tracked_anchor_index = command.cfg.body_names.index("trunk")
        ee_indices = tuple(
            command.cfg.body_names.index(name)
            for name in MICROBAN_END_EFFECTOR_BODY_NAMES
        )

        initial_root_pos = robot.data.root_link_pos_w.clone()
        initial_reference_quat = command.anchor_quat_w.clone()
        local_forward = torch.tensor(
            (1.0, 0.0, 0.0), dtype=torch.float32, device=env.device
        ).expand(env.num_envs, -1)
        reference_forward_w = quat_apply(initial_reference_quat, local_forward)
        reference_forward_xy = reference_forward_w[:, :2]
        reference_forward_xy /= torch.linalg.vector_norm(
            reference_forward_xy, dim=-1, keepdim=True
        ).clamp_min(1.0e-9)

        active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        nonfinite_envs = torch.zeros_like(active)
        fall_envs = torch.zeros_like(active)
        full_clip_completed = torch.zeros_like(active)
        ever_contact = {side: torch.zeros_like(active) for side in ("left", "right")}
        ever_airborne = {side: torch.zeros_like(active) for side in ("left", "right")}
        simultaneous_flight_envs = torch.zeros_like(active)
        first_exit_step = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )
        forward_displacement = torch.zeros(
            env.num_envs, dtype=torch.float64, device=env.device
        )
        minimum_root_height = robot.data.root_link_pos_w[:, 2].clone()
        maximum_actual_soft_violation_by_joint = torch.zeros(
            len(robot.joint_names), dtype=torch.float32, device=env.device
        )
        maximum_target_projection_by_joint = torch.zeros(
            len(action.target_names), dtype=torch.float32, device=env.device
        )
        maximum_raw_action_abs_by_joint = torch.zeros_like(
            maximum_target_projection_by_joint
        )
        maximum_target_tracking_error_by_joint = torch.zeros_like(
            maximum_target_projection_by_joint
        )
        maximum_reference_joint_error_by_joint = torch.zeros(
            len(robot.joint_names), dtype=torch.float32, device=env.device
        )
        termination_first_seen = {
            name: torch.zeros_like(active)
            for name in env.termination_manager.active_terms
        }
        unexpected_termination_envs = torch.zeros_like(active)
        unattributed_termination_envs = torch.zeros_like(active)

        per_env_sample_count = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
        forward_velocity_sum = torch.zeros(
            env.num_envs, dtype=torch.float64, device=env.device
        )
        xy_velocity_error_sum = torch.zeros_like(forward_velocity_sum)
        tracked_sums = {
            name: torch.zeros_like(forward_velocity_sum)
            for name in (
                "anchor_position_error_m",
                "anchor_orientation_error_rad",
                "ee_position_error_m",
                "joint_position_l2_error_rad",
                "joint_velocity_l2_error_rad_s",
                "mpkpe_m",
                "root_relative_mpkpe_m",
            )
        }
        foot_contact_env_steps = {"left": 0, "right": 0}
        foot_airborne_env_steps = {"left": 0, "right": 0}
        foot_slip_samples: dict[str, list[float]] = {"left": [], "right": []}
        simultaneous_flight_env_steps = 0
        self_collision_contacts = 0
        self_collision_env_steps = 0
        target_clip_values = 0
        target_value_count = 0
        phase_sequence_valid = True
        executed_steps = 0
        step_trace: list[dict[str, Any]] = []

        initial_violation = torch.maximum(
            torch.clamp(all_soft_limits[..., 0] - robot.data.joint_pos, min=0.0),
            torch.clamp(robot.data.joint_pos - all_soft_limits[..., 1], min=0.0),
        )
        maximum_actual_soft_violation_by_joint = torch.maximum(
            maximum_actual_soft_violation_by_joint,
            initial_violation.max(dim=0).values,
        )

        for policy_step in range(TRACKING_CHECKPOINT_GATE_EXPECTED_STEPS):
            if not bool(active.any().item()):
                break
            active_before = active.clone()
            active_count = int(active_before.sum().item())
            expected_frame = TRACKING_CHECKPOINT_GATE_FIRST_TARGET_FRAME + policy_step
            active_phases = command.time_steps[active_before]
            phase_ok = bool((active_phases == expected_frame).all().item())
            phase_sequence_valid &= phase_ok

            reference_joint_pos = command.joint_pos.clone()
            reference_joint_vel = command.joint_vel.clone()
            reference_anchor_pos = command.anchor_pos_w.clone()
            reference_anchor_quat = command.anchor_quat_w.clone()
            reference_anchor_lin_vel = command.anchor_lin_vel_w.clone()
            reference_body_pos = command.body_pos_relative_w.clone()

            with torch.no_grad():
                raw_actions = policy(observations)
            action_finite = torch.isfinite(raw_actions).all(dim=1)
            pre_step_nonfinite = active_before & ~action_finite
            safe_actions = raw_actions.clone()
            safe_actions[~active_before | ~action_finite] = 0.0

            unclipped_target = safe_actions * scale + offset
            clipped_target = torch.clamp(
                unclipped_target,
                min=expanded_clip[..., 0],
                max=expanded_clip[..., 1],
            )
            projection = torch.abs(clipped_target - unclipped_target)
            finite_active_before = active_before & action_finite
            finite_active_count = int(finite_active_before.sum().item())
            if finite_active_count:
                maximum_target_projection_by_joint = torch.maximum(
                    maximum_target_projection_by_joint,
                    projection[finite_active_before].max(dim=0).values,
                )
                maximum_raw_action_abs_by_joint = torch.maximum(
                    maximum_raw_action_abs_by_joint,
                    safe_actions[finite_active_before].abs().max(dim=0).values,
                )
                clipped_values = projection[finite_active_before] > 1.0e-7
                target_clip_values += int(clipped_values.sum().item())
                target_value_count += int(clipped_values.numel())

            observations, rewards, dones, _ = wrapped.step(safe_actions)
            executed_steps += 1

            physics_nonfinite = NanGuard.detect_nans(env.sim.data)
            finite_state = _finite_env_mask(
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
            step_nonfinite = (
                pre_step_nonfinite | physics_nonfinite | ~finite_state
            ) & active_before
            nonfinite_envs |= step_nonfinite
            valid_active = active_before & ~step_nonfinite
            valid_count = int(valid_active.sum().item())

            root_height = robot.data.root_link_pos_w[:, 2]
            minimum_root_height[valid_active] = torch.minimum(
                minimum_root_height[valid_active], root_height[valid_active]
            )
            low_root = valid_active & (root_height < _ROOT_HEIGHT_FALL_THRESHOLD_M)
            fall_envs |= low_root

            actual_violation = torch.maximum(
                torch.clamp(all_soft_limits[..., 0] - robot.data.joint_pos, min=0.0),
                torch.clamp(robot.data.joint_pos - all_soft_limits[..., 1], min=0.0),
            )
            target_tracking_error = torch.abs(
                robot.data.joint_pos[:, action.target_ids]
                - robot.data.joint_pos_target[:, action.target_ids]
            )
            reference_joint_error = torch.abs(
                robot.data.joint_pos - reference_joint_pos
            )
            if valid_count:
                maximum_actual_soft_violation_by_joint = torch.maximum(
                    maximum_actual_soft_violation_by_joint,
                    actual_violation[valid_active].max(dim=0).values,
                )
                maximum_target_tracking_error_by_joint = torch.maximum(
                    maximum_target_tracking_error_by_joint,
                    target_tracking_error[valid_active].max(dim=0).values,
                )
                maximum_reference_joint_error_by_joint = torch.maximum(
                    maximum_reference_joint_error_by_joint,
                    reference_joint_error[valid_active].max(dim=0).values,
                )

            collision_found = self_collision.data.found
            if collision_found is None:
                raise RuntimeError("self_collision sensor does not expose found")
            collision_mask = collision_found.reshape(env.num_envs, -1) > 0
            self_collision_contacts += int(collision_found[valid_active].sum().item())
            self_collision_env_steps += int(
                (collision_mask.any(dim=1) & valid_active).sum().item()
            )

            contact_found = foot_contact.data.found
            if contact_found is None or contact_found.shape[1] != 2:
                raise RuntimeError("feet_ground_contact must expose two columns")
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
            simultaneous_flight = ~contacts["left"] & ~contacts["right"] & valid_active
            simultaneous_flight_envs |= simultaneous_flight
            simultaneous_flight_env_steps += int(simultaneous_flight.sum().item())

            if valid_count:
                actual_root_xy_velocity = robot.data.root_link_lin_vel_w[:, :2]
                reference_root_xy_velocity = reference_anchor_lin_vel[:, :2]
                forward_velocity = torch.sum(
                    actual_root_xy_velocity * reference_forward_xy, dim=-1
                )
                xy_velocity_error = torch.linalg.vector_norm(
                    actual_root_xy_velocity - reference_root_xy_velocity, dim=-1
                )
                forward_velocity_sum[valid_active] += forward_velocity[valid_active].to(
                    dtype=torch.float64
                )
                xy_velocity_error_sum[valid_active] += xy_velocity_error[
                    valid_active
                ].to(dtype=torch.float64)
                per_env_sample_count[valid_active] += 1

                displacement_xy = (
                    robot.data.root_link_pos_w[:, :2] - initial_root_pos[:, :2]
                )
                forward_displacement[valid_active] = torch.sum(
                    displacement_xy[valid_active] * reference_forward_xy[valid_active],
                    dim=-1,
                ).to(dtype=torch.float64)

                anchor_position_error = torch.linalg.vector_norm(
                    robot.data.root_link_pos_w - reference_anchor_pos, dim=-1
                )
                anchor_orientation_error = quat_error_magnitude(
                    reference_anchor_quat, robot.data.root_link_quat_w
                )
                per_body_error = torch.linalg.vector_norm(
                    reference_body_pos - command.robot_body_pos_w, dim=-1
                )
                mpkpe = per_body_error.mean(dim=-1)
                reference_root_relative = (
                    reference_body_pos
                    - reference_body_pos[
                        :, tracked_anchor_index : tracked_anchor_index + 1
                    ]
                )
                actual_root_relative = (
                    command.robot_body_pos_w
                    - command.robot_body_pos_w[
                        :, tracked_anchor_index : tracked_anchor_index + 1
                    ]
                )
                root_relative_mpkpe = torch.linalg.vector_norm(
                    reference_root_relative - actual_root_relative, dim=-1
                ).mean(dim=-1)
                ee_position_error = per_body_error[:, ee_indices].mean(dim=-1)
                joint_position_l2_error = torch.linalg.vector_norm(
                    robot.data.joint_pos - reference_joint_pos, dim=-1
                )
                joint_velocity_l2_error = torch.linalg.vector_norm(
                    robot.data.joint_vel - reference_joint_vel, dim=-1
                )
                tracked_values = {
                    "anchor_position_error_m": anchor_position_error,
                    "anchor_orientation_error_rad": anchor_orientation_error,
                    "ee_position_error_m": ee_position_error,
                    "joint_position_l2_error_rad": joint_position_l2_error,
                    "joint_velocity_l2_error_rad_s": joint_velocity_l2_error,
                    "mpkpe_m": mpkpe,
                    "root_relative_mpkpe_m": root_relative_mpkpe,
                }
                for name, values in tracked_values.items():
                    tracked_sums[name][valid_active] += values[valid_active].to(
                        dtype=torch.float64
                    )

            current_terms = _termination_masks(env)
            done = dones.bool() & active_before
            attributed = torch.zeros_like(active)
            for name, mask in current_terms.items():
                current = mask & active_before
                termination_first_seen[name] |= current
                attributed |= current
            # This bounded replay has no expected terminal condition.  Every
            # done is therefore unexpected, while unattributed done signals are
            # retained separately to expose manager/accounting drift.
            unexpected_termination_envs |= done
            unattributed_termination_envs |= done & ~attributed
            if "fell_over" in current_terms:
                fall_envs |= current_terms["fell_over"] & active_before

            removed = done | step_nonfinite | low_root
            newly_removed = active_before & removed
            first_exit_step[newly_removed] = policy_step
            active &= ~removed

            step_trace.append(
                {
                    "active_envs_after_step": int(active.sum().item()),
                    "active_envs_before_step": active_count,
                    "nonfinite_envs_this_step": int(step_nonfinite.sum().item()),
                    "policy_step_zero_based": policy_step,
                    "source_frame": expected_frame,
                    "source_phase_valid": phase_ok,
                    "terminations_this_step": {
                        name: int((mask & active_before).sum().item())
                        for name, mask in current_terms.items()
                        if bool((mask & active_before).any().item())
                    },
                }
            )

            if policy_step % 32 == 0 or policy_step + 1 == (
                TRACKING_CHECKPOINT_GATE_EXPECTED_STEPS
            ):
                print(
                    f"[INFO] {mode}: frame={expected_frame} "
                    f"active={int(active.sum().item())}/{env.num_envs}",
                    flush=True,
                )

            if policy_step + 1 == TRACKING_CHECKPOINT_GATE_EXPECTED_STEPS:
                full_clip_completed |= active
                break
            if not bool(active.any().item()):
                break

            # auto_reset=False preserves terminal state for the measurements
            # above.  Clear MjLab's manual-reset latch before the next vector
            # step, while permanently excluding exited rows from all metrics.
            inactive_ids = (~active).nonzero(as_tuple=False).squeeze(-1)
            if inactive_ids.numel() > 0:
                env.reset(env_ids=inactive_ids)
                # Partial reset calls CommandManager.compute() for the whole
                # vector and advances still-active rows. Restore their exact
                # next target before policy observations are recomputed.
                next_frame = min(expected_frame + 1, TRACKING_CHECKPOINT_GATE_END_FRAME)
                command.time_steps[active] = next_frame
                command.update_relative_body_poses()
                observations = wrapped.get_observations()

        forward_velocity_mean = _safe_per_env_mean(
            forward_velocity_sum, per_env_sample_count
        )
        xy_velocity_error_mean = _safe_per_env_mean(
            xy_velocity_error_sum, per_env_sample_count
        )
        sampled = per_env_sample_count > 0
        forward_velocity_p05 = (
            _percentile(forward_velocity_mean[sampled], 5.0)
            if bool(sampled.any().item())
            else 0.0
        )
        xy_velocity_mae_p95 = (
            _percentile(xy_velocity_error_mean[sampled], 95.0)
            if bool(sampled.any().item())
            else 1.0e9
        )
        displacement_p05 = _percentile(forward_displacement, 5.0)
        target_clip_fraction = (
            target_clip_values / target_value_count if target_value_count else 1.0
        )
        source_complete = (
            phase_sequence_valid
            and executed_steps == TRACKING_CHECKPOINT_GATE_EXPECTED_STEPS
            and bool(full_clip_completed.all().item())
        )
        measurements = {
            "executed_policy_steps": executed_steps,
            "fall_envs": int(fall_envs.sum().item()),
            "forward_displacement_p05_m": displacement_p05,
            "forward_velocity_p05_m_s": forward_velocity_p05,
            "full_clip_completed_envs": int(full_clip_completed.sum().item()),
            "left_foot_airborne_envs": int(ever_airborne["left"].sum().item()),
            "maximum_actual_soft_limit_violation_rad": float(
                maximum_actual_soft_violation_by_joint.max().item()
            ),
            "maximum_target_projection_rad": float(
                maximum_target_projection_by_joint.max().item()
            ),
            "minimum_root_height_m": float(minimum_root_height.min().item()),
            "nonfinite_envs": int(nonfinite_envs.sum().item()),
            "right_foot_airborne_envs": int(ever_airborne["right"].sum().item()),
            "self_collision_contacts": self_collision_contacts,
            "simultaneous_flight_env_steps": simultaneous_flight_env_steps,
            "source_frame_sequence_complete": source_complete,
            "target_clip_fraction": target_clip_fraction,
            "unexpected_termination_envs": int(
                unexpected_termination_envs.sum().item()
            ),
            "xy_velocity_mae_p95_m_s": xy_velocity_mae_p95,
        }
        checks = tracking_checkpoint_checks(measurements)

        tracked_error_report = {
            name: _distribution(
                _safe_per_env_mean(total, per_env_sample_count),
                units=(
                    "rad_s"
                    if name == "joint_velocity_l2_error_rad_s"
                    else "rad"
                    if "orientation" in name or "joint_position" in name
                    else "m"
                ),
            )
            for name, total in tracked_sums.items()
        }
        per_foot = {}
        for side in ("left", "right"):
            per_foot[side] = {
                "airborne_env_steps": foot_airborne_env_steps[side],
                "contact_env_steps": foot_contact_env_steps[side],
                "envs_ever_airborne": int(ever_airborne[side].sum().item()),
                "envs_ever_in_contact": int(ever_contact[side].sum().item()),
                "slip_while_in_contact": _distribution(
                    torch.tensor(foot_slip_samples[side], dtype=torch.float64),
                    units="m_s",
                ),
            }

        return {
            "checks": checks,
            "configuration": {
                **configuration_evidence,
                **checkpoint_evidence,
                "bam": bam_evidence,
                "episode_length_s": cfg.episode_length_s,
                "policy_step_hz": 1.0 / env.step_dt,
            },
            "diagnostics": {
                "action_target_clip": {
                    "absolute_clip_matches_entity_soft_limits": True,
                    "clip_tensor_max_error_rad": clip_tensor_max_error,
                    "clipped_value_count": target_clip_values,
                    "default_offset_max_error_rad": offset_max_error,
                    "raw_action_semantics": (
                        "default_relative_joint_position_delta_rad"
                    ),
                    "scale": 1.0,
                    "target_value_count": target_value_count,
                },
                "feet": per_foot,
                "first_exit_step": _distribution(
                    first_exit_step[first_exit_step >= 0], units="policy_step"
                ),
                "forward_displacement_per_env": _distribution(
                    forward_displacement, units="m"
                ),
                "forward_velocity_per_env_mean": _distribution(
                    forward_velocity_mean, units="m_s"
                ),
                "joint_actual_soft_limit_max_violation_rad_by_joint": dict(
                    zip(
                        robot.joint_names,
                        maximum_actual_soft_violation_by_joint.detach().cpu().tolist(),
                        strict=True,
                    )
                ),
                "joint_reference_error_max_rad_by_joint": dict(
                    zip(
                        robot.joint_names,
                        maximum_reference_joint_error_by_joint.detach().cpu().tolist(),
                        strict=True,
                    )
                ),
                "policy_raw_action_abs_max_by_joint": dict(
                    zip(
                        action.target_names,
                        maximum_raw_action_abs_by_joint.detach().cpu().tolist(),
                        strict=True,
                    )
                ),
                "policy_target_projection_max_rad_by_joint": dict(
                    zip(
                        action.target_names,
                        maximum_target_projection_by_joint.detach().cpu().tolist(),
                        strict=True,
                    )
                ),
                "self_collision_env_steps": self_collision_env_steps,
                "simultaneous_flight_envs": int(simultaneous_flight_envs.sum().item()),
                "step_trace": step_trace,
                "target_tracking_error_max_rad_by_joint": dict(
                    zip(
                        action.target_names,
                        maximum_target_tracking_error_by_joint.detach().cpu().tolist(),
                        strict=True,
                    )
                ),
                "termination_env_counts_by_term": {
                    name: int(mask.sum().item())
                    for name, mask in termination_first_seen.items()
                },
                "unattributed_termination_envs": int(
                    unattributed_termination_envs.sum().item()
                ),
                "tracking_errors_per_env_mean": tracked_error_report,
                "valid_sample_count_per_env": _distribution(
                    per_env_sample_count, units="policy_steps"
                ),
                "xy_velocity_error_per_env_mean": _distribution(
                    xy_velocity_error_mean, units="m_s"
                ),
            },
            "measurements": measurements,
        }
    finally:
        if wrapped is not None:
            wrapped.close()
        elif env is not None:
            env.close()
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()


def evaluate_tracking_checkpoint(
    *,
    checkpoint: Path,
    motion_path: Path,
    expected_motion_sha256: str,
    robot_xml_path: Path,
    expected_robot_xml_sha256: str,
    device: str,
) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve()
    motion_path = motion_path.expanduser().resolve()
    robot_xml_path = robot_xml_path.expanduser().resolve()
    for name, path in (
        ("checkpoint", checkpoint),
        ("motion", motion_path),
        ("robot XML", robot_xml_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Tracking {name} does not exist: {path}")

    motion_sha256 = sha256_file(motion_path)
    robot_xml_sha256 = sha256_file(robot_xml_path)
    if motion_sha256 != expected_motion_sha256:
        raise ValueError("Tracking motion SHA-256 mismatch")
    if robot_xml_sha256 != expected_robot_xml_sha256:
        raise ValueError("Tracking robot XML SHA-256 mismatch")
    motion_evidence = _validate_motion_archive(motion_path)
    (
        checkpoint_actor_kind,
        checkpoint_actor_obs_normalization,
    ) = _checkpoint_actor_contract(checkpoint)

    if not device.startswith("cuda"):
        raise ValueError("The canonical 256-environment tracking gate requires CUDA")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    configure_torch_backends(allow_tf32=False, deterministic=True)
    torch.use_deterministic_algorithms(True, warn_only=True)

    passes = {
        mode: _evaluate_pass(
            mode=mode,
            checkpoint=checkpoint,
            checkpoint_actor_kind=checkpoint_actor_kind,
            checkpoint_actor_obs_normalization=(checkpoint_actor_obs_normalization),
            motion_path=motion_path,
            device=device,
        )
        for mode in TRACKING_CHECKPOINT_GATE_MODES
    }
    evaluator_path = Path(__file__).resolve()
    payload = {
        "configuration": {
            "actor_observation_schema": [
                {"name": name, "width": width}
                for name, width in TRACKING_ACTOR_OBSERVATION_SCHEMA
            ],
            "actor_observation_width": TRACKING_ACTOR_OBSERVATION_WIDTH,
            "end_frame_inclusive": TRACKING_CHECKPOINT_GATE_END_FRAME,
            "expected_policy_steps": TRACKING_CHECKPOINT_GATE_EXPECTED_STEPS,
            "first_policy_target_frame": (TRACKING_CHECKPOINT_GATE_FIRST_TARGET_FRAME),
            "initial_state_frame": TRACKING_CHECKPOINT_GATE_INITIAL_FRAME,
            "modes": list(TRACKING_CHECKPOINT_GATE_MODES),
            "num_envs_per_pass": TRACKING_CHECKPOINT_GATE_NUM_ENVS,
            "seed": TRACKING_CHECKPOINT_GATE_SEED,
            "source_frame_count": TRACKING_CHECKPOINT_GATE_FRAME_COUNT,
            "thresholds": asdict(DEFAULT_TRACKING_CHECKPOINT_THRESHOLDS),
        },
        "deployment_scope": (
            "simulation_fixed_motion_tracking_diagnostic_not_live_pico_acceptance"
        ),
        "inputs": {
            "checkpoint": {
                "path": _portable_path(checkpoint),
                "sha256": sha256_file(checkpoint),
                "size_bytes": checkpoint.stat().st_size,
            },
            "evaluator_source": {
                "path": _portable_path(evaluator_path),
                "sha256": sha256_file(evaluator_path),
            },
            "motion": {
                **motion_evidence,
                "path": _portable_path(motion_path),
                "sha256": motion_sha256,
                "size_bytes": motion_path.stat().st_size,
            },
            "runtime_sources": _runtime_source_evidence(),
            "robot_xml": {
                "path": _portable_path(robot_xml_path),
                "sha256": robot_xml_sha256,
                "size_bytes": robot_xml_path.stat().st_size,
            },
        },
        "passes": passes,
    }
    return finalize_tracking_checkpoint_report(payload)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--motion", type=Path, default=DEFAULT_LOCOMOTION_PRIOR_PATH)
    parser.add_argument(
        "--expected-motion-sha256",
        default=DEFAULT_LOCOMOTION_PRIOR_SHA256,
    )
    parser.add_argument("--robot-xml", type=Path, default=DEFAULT_ROBOT_XML_PATH)
    parser.add_argument(
        "--expected-robot-xml-sha256",
        default=DEFAULT_ROBOT_XML_SHA256,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = evaluate_tracking_checkpoint(
        checkpoint=args.checkpoint,
        motion_path=args.motion,
        expected_motion_sha256=args.expected_motion_sha256,
        robot_xml_path=args.robot_xml,
        expected_robot_xml_sha256=args.expected_robot_xml_sha256,
        device=args.device,
    )
    _publish_json_report(args.output, report)
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
    return tracking_checkpoint_exit_code(report)


if __name__ == "__main__":
    sys.exit(main())
