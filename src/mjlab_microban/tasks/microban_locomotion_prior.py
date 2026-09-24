# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Short-lived privileged locomotion prior for Microban teleoperation.

The prior is deliberately unavailable to the actor.  During the first 1,000
PPO updates it gives the asymmetric critic, two training rewards and the
training-only actor teacher a retargeted, forward-walking reference.  Early
episodes start directly at frame 109.  That teleport probability then fades
while non-teleported episodes receive a smooth 20-step launch from their real
reset pose into frame 109.  The clip advances to frame 267 at a command-scaled
rate and terminates instead of looping.  A teleport writes only the twelve leg
joints plus the floating root, preserving all six arm and three HMD-owned joint
states.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.envs.mdp.actions import JointPositionAction
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_TELEOP_ACTION_JOINT_NAMES,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


MICROBAN_LOCOMOTION_PRIOR_FILENAME = "microban_twist2_walk002_locomotion_prior.npz"
MICROBAN_LOCOMOTION_PRIOR_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "motions"
    / MICROBAN_LOCOMOTION_PRIOR_FILENAME
)
MICROBAN_LOCOMOTION_PRIOR_SHA256 = (
    "e789594b7711eb7e001edbde12064c9068e9649dfa6945626478171b48ad9fa0"
)
MICROBAN_LOCOMOTION_PRIOR_FPS = 50.0
MICROBAN_LOCOMOTION_PRIOR_START_FRAME = 109
MICROBAN_LOCOMOTION_PRIOR_END_FRAME = 267
MICROBAN_LOCOMOTION_PRIOR_LEAD_FRAMES = 5
MICROBAN_LOCOMOTION_PRIOR_NOMINAL_FORWARD_VELOCITY_M_S = 0.0780311897
MICROBAN_LOCOMOTION_PRIOR_FORWARD_VELOCITY_RANGE_M_S = (0.06, 0.11)
MICROBAN_LOCOMOTION_PRIOR_FULL_BLEND_END_STEP = 500 * 24
MICROBAN_LOCOMOTION_PRIOR_FADE_END_STEP = 1000 * 24
MICROBAN_LOCOMOTION_PRIOR_FULL_TELEPORT_END_STEP = 100 * 24
MICROBAN_LOCOMOTION_PRIOR_TELEPORT_FADE_END_STEP = 500 * 24
MICROBAN_LOCOMOTION_PRIOR_LAUNCH_STEPS = 20
MICROBAN_LOCOMOTION_PRIOR_COMMAND_WIDTH = 39

MICROBAN_LOCOMOTION_PRIOR_JOINT_NAMES: tuple[str, ...] = (
    "head",
    "neck_roll",
    "neck_pitch",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_elbow",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_elbow",
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
)
MICROBAN_LOCOMOTION_PRIOR_LEG_JOINT_NAMES: tuple[str, ...] = (
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
)


def locomotion_prior_blend(global_step: int) -> float:
    """Return the audited 1.0 -> 0.0 imitation blend schedule."""

    if isinstance(global_step, bool) or not isinstance(global_step, int):
        raise TypeError("global_step must be an integer")
    if global_step < 0:
        raise ValueError("global_step must be non-negative")
    if global_step <= MICROBAN_LOCOMOTION_PRIOR_FULL_BLEND_END_STEP:
        return 1.0
    if global_step >= MICROBAN_LOCOMOTION_PRIOR_FADE_END_STEP:
        return 0.0
    fade_steps = (
        MICROBAN_LOCOMOTION_PRIOR_FADE_END_STEP
        - MICROBAN_LOCOMOTION_PRIOR_FULL_BLEND_END_STEP
    )
    elapsed = global_step - MICROBAN_LOCOMOTION_PRIOR_FULL_BLEND_END_STEP
    return 1.0 - elapsed / fade_steps


def locomotion_prior_teleport_probability(global_step: int) -> float:
    """Fade reset teleport before imitation fades, exposing real launch states."""

    if isinstance(global_step, bool) or not isinstance(global_step, int):
        raise TypeError("global_step must be an integer")
    if global_step < 0:
        raise ValueError("global_step must be non-negative")
    if global_step <= MICROBAN_LOCOMOTION_PRIOR_FULL_TELEPORT_END_STEP:
        return 1.0
    if global_step >= MICROBAN_LOCOMOTION_PRIOR_TELEPORT_FADE_END_STEP:
        return 0.0
    fade_steps = (
        MICROBAN_LOCOMOTION_PRIOR_TELEPORT_FADE_END_STEP
        - MICROBAN_LOCOMOTION_PRIOR_FULL_TELEPORT_END_STEP
    )
    elapsed = global_step - MICROBAN_LOCOMOTION_PRIOR_FULL_TELEPORT_END_STEP
    return 1.0 - elapsed / fade_steps


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class _LocomotionPriorArrays:
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    root_pos: torch.Tensor
    root_quat: torch.Tensor
    root_lin_vel: torch.Tensor
    root_ang_vel: torch.Tensor


def _scalar_string(value: np.ndarray, name: str) -> str:
    if value.shape != () or value.dtype.kind not in ("U", "S"):
        raise ValueError(f"Locomotion prior {name} must be one string scalar")
    return str(value.item())


def _load_locomotion_prior(
    path: Path,
    *,
    expected_sha256: str,
    device: str,
) -> _LocomotionPriorArrays:
    """Load and fail-closed validate the exact audited retargeting artifact."""

    if not path.is_file():
        raise FileNotFoundError(f"Locomotion prior artifact is missing: {path}")
    if _sha256_file(path) != expected_sha256:
        raise ValueError("Locomotion prior artifact SHA-256 mismatch")

    required = {
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "joint_names",
        "reference_only_joint_names",
        "policy_action_joint_names",
        "body_names",
        "source_session_id",
        "source_schema",
        "retarget_profile",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(
                f"Locomotion prior artifact is missing fields: {sorted(missing)}"
            )
        fps = np.asarray(archive["fps"])
        joint_pos = np.asarray(archive["joint_pos"])
        joint_vel = np.asarray(archive["joint_vel"])
        body_pos = np.asarray(archive["body_pos_w"])
        body_quat = np.asarray(archive["body_quat_w"])
        body_lin_vel = np.asarray(archive["body_lin_vel_w"])
        body_ang_vel = np.asarray(archive["body_ang_vel_w"])
        joint_names = tuple(str(name) for name in archive["joint_names"].tolist())
        reference_only = tuple(
            str(name) for name in archive["reference_only_joint_names"].tolist()
        )
        policy_action_names = tuple(
            str(name) for name in archive["policy_action_joint_names"].tolist()
        )
        body_names = tuple(str(name) for name in archive["body_names"].tolist())
        source_session_id = _scalar_string(
            np.asarray(archive["source_session_id"]), "source_session_id"
        )
        source_schema = _scalar_string(
            np.asarray(archive["source_schema"]), "source_schema"
        )
        retarget_profile = _scalar_string(
            np.asarray(archive["retarget_profile"]), "retarget_profile"
        )

    if fps.shape != (1,) or float(fps[0]) != MICROBAN_LOCOMOTION_PRIOR_FPS:
        raise ValueError("Locomotion prior must be sampled at exactly 50 Hz")
    if joint_pos.shape != (268, 21) or joint_vel.shape != (268, 21):
        raise ValueError("Locomotion prior joint arrays must have shape (268, 21)")
    expected_body_shape = (268, 22)
    if body_pos.shape != (*expected_body_shape, 3):
        raise ValueError("Locomotion prior body_pos_w shape mismatch")
    if body_quat.shape != (*expected_body_shape, 4):
        raise ValueError("Locomotion prior body_quat_w shape mismatch")
    if body_lin_vel.shape != (*expected_body_shape, 3):
        raise ValueError("Locomotion prior body_lin_vel_w shape mismatch")
    if body_ang_vel.shape != (*expected_body_shape, 3):
        raise ValueError("Locomotion prior body_ang_vel_w shape mismatch")
    if joint_names != MICROBAN_LOCOMOTION_PRIOR_JOINT_NAMES:
        raise ValueError("Locomotion prior joint order mismatch")
    if reference_only != ("head", "neck_roll", "neck_pitch"):
        raise ValueError("Locomotion prior reference-only joint contract mismatch")
    if policy_action_names != MICROBAN_TELEOP_ACTION_JOINT_NAMES:
        raise ValueError("Locomotion prior policy action order mismatch")
    if not body_names or body_names[0] != "trunk":
        raise ValueError("Locomotion prior root body must be trunk at index zero")
    if source_session_id != "twist2-b06178f19a22-0807_yanjie_walk_002":
        raise ValueError("Locomotion prior source session mismatch")
    if source_schema != "microban.twist2.capture@1":
        raise ValueError("Locomotion prior source schema mismatch")
    if retarget_profile != "locomotion-prior":
        raise ValueError("Locomotion prior retarget profile mismatch")

    used = slice(
        MICROBAN_LOCOMOTION_PRIOR_START_FRAME,
        MICROBAN_LOCOMOTION_PRIOR_END_FRAME + 1,
    )
    used_arrays = (
        joint_pos[used],
        joint_vel[used],
        body_pos[used],
        body_quat[used],
        body_lin_vel[used],
        body_ang_vel[used],
    )
    if any(array.dtype != np.float32 for array in used_arrays):
        raise ValueError("Locomotion prior numeric arrays must use float32")
    if any(not np.isfinite(array).all() for array in used_arrays):
        raise ValueError("Locomotion prior contains non-finite values")

    source_leg_ids = [
        joint_names.index(name) for name in MICROBAN_LOCOMOTION_PRIOR_LEG_JOINT_NAMES
    ]
    tensor = lambda value: torch.as_tensor(
        value.copy(), dtype=torch.float32, device=device
    )
    return _LocomotionPriorArrays(
        joint_pos=tensor(joint_pos[:, source_leg_ids]),
        joint_vel=tensor(joint_vel[:, source_leg_ids]),
        root_pos=tensor(body_pos[:, 0]),
        root_quat=tensor(body_quat[:, 0]),
        root_lin_vel=tensor(body_lin_vel[:, 0]),
        root_ang_vel=tensor(body_ang_vel[:, 0]),
    )


class LocomotionPriorCommand(CommandTerm):
    """Per-environment, non-looping motion reference latched only on reset."""

    cfg: LocomotionPriorCommandCfg
    _env: ManagerBasedRlEnv

    def __init__(self, cfg: LocomotionPriorCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.entity_name]
        resolved_ids, resolved_names = self.robot.find_joints(
            MICROBAN_LOCOMOTION_PRIOR_LEG_JOINT_NAMES,
            preserve_order=True,
        )
        if tuple(resolved_names) != MICROBAN_LOCOMOTION_PRIOR_LEG_JOINT_NAMES:
            raise ValueError("Microban locomotion-prior leg joint order mismatch")
        self.leg_joint_ids = torch.tensor(
            resolved_ids, dtype=torch.long, device=self.device
        )
        self.leg_action_ids = torch.tensor(
            [
                MICROBAN_TELEOP_ACTION_JOINT_NAMES.index(name)
                for name in MICROBAN_LOCOMOTION_PRIOR_LEG_JOINT_NAMES
            ],
            dtype=torch.long,
            device=self.device,
        )
        self.arrays = _load_locomotion_prior(
            Path(cfg.motion_file).expanduser().resolve(),
            expected_sha256=cfg.expected_sha256,
            device=self.device,
        )
        self.phase = torch.full(
            (self.num_envs,),
            float(MICROBAN_LOCOMOTION_PRIOR_START_FRAME),
            dtype=torch.float32,
            device=self.device,
        )
        self.phase_rate = torch.zeros_like(self.phase)
        self.eligible = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.finished = torch.zeros_like(self.eligible)
        self.teleported = torch.zeros_like(self.eligible)
        self.launching = torch.zeros_like(self.eligible)
        self.launch_step = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.launch_start_joint_pos = torch.zeros(
            (self.num_envs, len(MICROBAN_LOCOMOTION_PRIOR_LEG_JOINT_NAMES)),
            dtype=torch.float32,
            device=self.device,
        )
        self.launch_dt = float(env.step_dt)
        if not math.isclose(
            self.launch_dt,
            1.0 / MICROBAN_LOCOMOTION_PRIOR_FPS,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError("Locomotion prior requires an exact 50 Hz policy step")
        # MjLab auto-reset happens inside ``env.step`` immediately before the
        # command manager's positive-dt compute.  Mark reset rows so that call
        # does not advance a reference whose reset pose has not been simulated.
        # An explicit ``env.reset`` instead consumes the marker in its dt=0 call.
        self._skip_next_advance = torch.zeros_like(self.eligible)

    @property
    def blend(self) -> float:
        if not self.cfg.enabled:
            return 0.0
        return locomotion_prior_blend(int(self._env.common_step_counter))

    @property
    def active(self) -> torch.Tensor:
        return self.eligible & (self.blend > 0.0)

    @property
    def imitation_weight(self) -> torch.Tensor:
        return self.active.to(dtype=torch.float32) * self.blend

    def _interpolate(self, values: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        bounded = torch.clamp(
            phase,
            min=float(MICROBAN_LOCOMOTION_PRIOR_START_FRAME),
            max=float(MICROBAN_LOCOMOTION_PRIOR_END_FRAME),
        )
        lower = torch.floor(bounded).to(dtype=torch.long)
        upper = torch.clamp(lower + 1, max=MICROBAN_LOCOMOTION_PRIOR_END_FRAME)
        alpha = (bounded - lower.to(dtype=bounded.dtype)).unsqueeze(-1)
        return values[lower] + alpha * (values[upper] - values[lower])

    @property
    def reference_joint_pos(self) -> torch.Tensor:
        reference = self._interpolate(self.arrays.joint_pos, self.phase)
        launch = self._launch_joint_pos(lookahead_source_frames=0.0)
        return torch.where(self.launching.unsqueeze(-1), launch, reference)

    @property
    def reference_joint_vel(self) -> torch.Tensor:
        velocity = self._interpolate(self.arrays.joint_vel, self.phase)
        velocity = velocity * self.phase_rate.unsqueeze(-1)
        progress = self.launch_step.to(dtype=torch.float32) / float(
            MICROBAN_LOCOMOTION_PRIOR_LAUNCH_STEPS
        )
        progress = torch.clamp(progress, min=0.0, max=1.0)
        smoothstep_derivative = 6.0 * progress * (1.0 - progress)
        launch_velocity = (
            self.arrays.joint_pos[MICROBAN_LOCOMOTION_PRIOR_START_FRAME]
            - self.launch_start_joint_pos
        ) * (
            smoothstep_derivative
            / (MICROBAN_LOCOMOTION_PRIOR_LAUNCH_STEPS * self.launch_dt)
        ).unsqueeze(-1)
        return torch.where(self.launching.unsqueeze(-1), launch_velocity, velocity)

    @property
    def lead_joint_pos(self) -> torch.Tensor:
        reference = self._interpolate(
            self.arrays.joint_pos,
            self.phase + float(MICROBAN_LOCOMOTION_PRIOR_LEAD_FRAMES),
        )
        launch = self._launch_joint_pos(
            lookahead_source_frames=float(MICROBAN_LOCOMOTION_PRIOR_LEAD_FRAMES)
        )
        return torch.where(self.launching.unsqueeze(-1), launch, reference)

    def _launch_joint_pos(self, *, lookahead_source_frames: float) -> torch.Tensor:
        """Evaluate launch and source-frame lookahead without a join jump."""

        safe_phase_rate = torch.where(
            self.phase_rate > 0.0,
            self.phase_rate,
            torch.ones_like(self.phase_rate),
        )
        future_step = self.launch_step.to(dtype=torch.float32) + (
            lookahead_source_frames / safe_phase_rate
        )
        progress = torch.clamp(
            future_step / float(MICROBAN_LOCOMOTION_PRIOR_LAUNCH_STEPS),
            min=0.0,
            max=1.0,
        )
        smoothstep = torch.square(progress) * (3.0 - 2.0 * progress)
        frame_109 = self.arrays.joint_pos[MICROBAN_LOCOMOTION_PRIOR_START_FRAME]
        transition = self.launch_start_joint_pos + smoothstep.unsqueeze(-1) * (
            frame_109 - self.launch_start_joint_pos
        )
        overflow_steps = torch.clamp(
            future_step - float(MICROBAN_LOCOMOTION_PRIOR_LAUNCH_STEPS), min=0.0
        )
        clip_phase = float(MICROBAN_LOCOMOTION_PRIOR_START_FRAME) + (
            overflow_steps * self.phase_rate
        )
        clip = self._interpolate(self.arrays.joint_pos, clip_phase)
        return torch.where(
            (future_step <= MICROBAN_LOCOMOTION_PRIOR_LAUNCH_STEPS).unsqueeze(-1),
            transition,
            clip,
        )

    @property
    def command(self) -> torch.Tensor:
        result = torch.zeros(
            (self.num_envs, MICROBAN_LOCOMOTION_PRIOR_COMMAND_WIDTH),
            dtype=torch.float32,
            device=self.device,
        )
        active = self.active
        progress = (self.phase - MICROBAN_LOCOMOTION_PRIOR_START_FRAME) / (
            MICROBAN_LOCOMOTION_PRIOR_END_FRAME - MICROBAN_LOCOMOTION_PRIOR_START_FRAME
        )
        angle = 2.0 * math.pi * progress
        payload = torch.cat(
            (
                self.imitation_weight.unsqueeze(-1),
                torch.sin(angle).unsqueeze(-1),
                torch.cos(angle).unsqueeze(-1),
                self.reference_joint_pos,
                self.reference_joint_vel,
                self.lead_joint_pos,
            ),
            dim=-1,
        )
        if payload.shape[1] != MICROBAN_LOCOMOTION_PRIOR_COMMAND_WIDTH:
            raise RuntimeError("Locomotion prior command width drifted")
        return torch.where(active.unsqueeze(-1), payload, result)

    def _forward_eligible(self, env_ids: torch.Tensor) -> torch.Tensor:
        twist = self._env.command_manager.get_command(self.cfg.velocity_command_name)
        if twist.ndim != 2 or twist.shape[0] != self.num_envs or twist.shape[1] < 3:
            raise ValueError("Locomotion prior requires a three-axis velocity command")
        selected = twist[env_ids, :3]
        if not bool(torch.isfinite(selected).all().item()):
            raise ValueError("Locomotion prior received non-finite velocity commands")
        lower, upper = self.cfg.forward_velocity_range_m_s
        tolerance = 1.0e-6
        return (
            (selected[:, 0] >= lower - tolerance)
            & (selected[:, 0] <= upper + tolerance)
            & (torch.abs(selected[:, 1]) <= tolerance)
            & (torch.abs(selected[:, 2]) <= tolerance)
        )

    def _write_reference_reset(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        phase = torch.full(
            (env_ids.numel(),),
            float(MICROBAN_LOCOMOTION_PRIOR_START_FRAME),
            dtype=torch.float32,
            device=self.device,
        )
        rate = self.phase_rate[env_ids]
        joint_pos = self._interpolate(self.arrays.joint_pos, phase)
        joint_vel = self._interpolate(self.arrays.joint_vel, phase)
        joint_vel = joint_vel * rate.unsqueeze(-1)
        self.robot.write_joint_state_to_sim(
            joint_pos,
            joint_vel,
            joint_ids=self.leg_joint_ids,
            env_ids=env_ids,
        )
        # Prevent the first decimation substep from pulling the teleported legs
        # back toward stale pre-reset targets.  No arm or HMD target is touched.
        self.robot.set_joint_position_target(
            joint_pos,
            joint_ids=self.leg_joint_ids,
            env_ids=env_ids.unsqueeze(-1),
        )

        frame = MICROBAN_LOCOMOTION_PRIOR_START_FRAME
        root_pos = self._env.scene.env_origins[env_ids].clone()
        root_pos[:, 2] += self.arrays.root_pos[frame, 2]
        root_quat = self.arrays.root_quat[frame].expand(env_ids.numel(), -1)
        root_lin_vel = self.arrays.root_lin_vel[frame].expand(
            env_ids.numel(), -1
        ) * rate.unsqueeze(-1)
        root_ang_vel = self.arrays.root_ang_vel[frame].expand(
            env_ids.numel(), -1
        ) * rate.unsqueeze(-1)
        root_state = torch.cat(
            (root_pos, root_quat, root_lin_vel, root_ang_vel), dim=-1
        )
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)

    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        if not isinstance(env_ids, torch.Tensor):
            raise TypeError("LocomotionPriorCommand.reset requires tensor env_ids")
        self.command_counter[env_ids] = 0
        self.time_left[env_ids] = float("inf")
        self.phase[env_ids] = float(MICROBAN_LOCOMOTION_PRIOR_START_FRAME)
        self.phase_rate[env_ids] = 0.0
        self.eligible[env_ids] = False
        self.finished[env_ids] = False
        self.teleported[env_ids] = False
        self.launching[env_ids] = False
        self.launch_step[env_ids] = 0
        self.launch_start_joint_pos[env_ids] = 0.0
        self._skip_next_advance[env_ids] = True

        if not self.cfg.enabled or self.blend <= 0.0:
            return {}
        eligible_local = self._forward_eligible(env_ids)
        eligible_ids = env_ids[eligible_local]
        self.eligible[eligible_ids] = True
        twist = self._env.command_manager.get_command(self.cfg.velocity_command_name)
        self.phase_rate[eligible_ids] = (
            twist[eligible_ids, 0]
            / MICROBAN_LOCOMOTION_PRIOR_NOMINAL_FORWARD_VELOCITY_M_S
        )
        probability = locomotion_prior_teleport_probability(
            int(self._env.common_step_counter)
        )
        if probability >= 1.0:
            teleported_ids = eligible_ids
            launch_ids = eligible_ids[:0]
        elif probability <= 0.0:
            teleported_ids = eligible_ids[:0]
            launch_ids = eligible_ids
        else:
            teleport_local = (
                torch.rand(eligible_ids.numel(), device=self.device) < probability
            )
            teleported_ids = eligible_ids[teleport_local]
            launch_ids = eligible_ids[~teleport_local]
        self.teleported[teleported_ids] = True
        self.launching[launch_ids] = True
        if launch_ids.numel() > 0:
            self.launch_start_joint_pos[launch_ids] = self.robot.data.joint_pos[
                launch_ids.unsqueeze(-1), self.leg_joint_ids
            ]
        self._write_reference_reset(teleported_ids)
        return {}

    def compute(self, dt: float) -> None:
        """Advance only for real policy steps; the post-reset dt=0 call is inert."""

        self._update_metrics()
        if dt <= 0.0 or not self.cfg.enabled:
            self._skip_next_advance.zero_()
            return
        active = (
            self.eligible
            & ~self.finished
            & ~self._skip_next_advance
            & (self.blend > 0.0)
        )
        self._skip_next_advance.zero_()
        launch_active = active & self.launching
        walking_active = active & ~self.launching
        self.launch_step[launch_active] += 1
        launch_finished = launch_active & (
            self.launch_step >= MICROBAN_LOCOMOTION_PRIOR_LAUNCH_STEPS
        )
        self.launching[launch_finished] = False
        next_phase = self.phase + torch.where(
            walking_active, self.phase_rate, torch.zeros_like(self.phase_rate)
        )
        newly_finished = walking_active & (
            next_phase >= MICROBAN_LOCOMOTION_PRIOR_END_FRAME
        )
        self.finished |= newly_finished
        self.phase[:] = torch.clamp(
            next_phase, max=float(MICROBAN_LOCOMOTION_PRIOR_END_FRAME)
        )

    def disable(self) -> None:
        self.cfg.enabled = False
        self.eligible.zero_()
        self.finished.zero_()
        self.teleported.zero_()
        self.launching.zero_()
        self.launch_step.zero_()
        self.launch_start_joint_pos.zero_()
        self.phase_rate.zero_()
        self._skip_next_advance.zero_()

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        del env_ids

    def _update_command(self) -> None:
        # ``compute`` implements dt-aware, non-looping advancement.
        pass


@dataclass(kw_only=True)
class LocomotionPriorCommandCfg(CommandTermCfg):
    motion_file: str
    expected_sha256: str
    entity_name: str = "robot"
    velocity_command_name: str = "twist"
    forward_velocity_range_m_s: tuple[float, float] = (
        MICROBAN_LOCOMOTION_PRIOR_FORWARD_VELOCITY_RANGE_M_S
    )
    enabled: bool = True

    def build(self, env: ManagerBasedRlEnv) -> LocomotionPriorCommand:
        lower, upper = self.forward_velocity_range_m_s
        if (
            not math.isfinite(lower)
            or not math.isfinite(upper)
            or not 0.0 < lower < upper
        ):
            raise ValueError("Locomotion prior forward velocity range is invalid")
        if len(self.expected_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.expected_sha256
        ):
            raise ValueError("Locomotion prior expected SHA-256 is malformed")
        return LocomotionPriorCommand(self, env)


def _locomotion_prior_command(
    env: ManagerBasedRlEnv, command_name: str
) -> LocomotionPriorCommand:
    command = env.command_manager.get_term(command_name)
    if not isinstance(command, LocomotionPriorCommand):
        raise TypeError(f"{command_name!r} is not a LocomotionPriorCommand")
    return command


def locomotion_prior_action_target_error_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    action_name: str,
    std: float,
) -> torch.Tensor:
    """Reward the clipped absolute leg target at the +5-frame reference."""

    if not math.isfinite(std) or std <= 0.0:
        raise ValueError("Locomotion-prior reward std must be finite and positive")
    command = _locomotion_prior_command(env, command_name)
    action = env.action_manager.get_term(action_name)
    if not isinstance(action, JointPositionAction):
        raise TypeError(f"{action_name!r} must be a JointPositionAction")
    if tuple(action.target_names) != MICROBAN_TELEOP_ACTION_JOINT_NAMES:
        raise ValueError("Locomotion-prior action joint order mismatch")
    if action.cfg.clip is None or not hasattr(action, "_clip"):
        raise ValueError("Locomotion-prior action target requires absolute clips")
    raw = action.raw_action
    scale = torch.as_tensor(action.scale, dtype=raw.dtype, device=raw.device)
    offset = torch.as_tensor(action.offset, dtype=raw.dtype, device=raw.device)
    target = raw * scale + offset
    target = torch.clamp(target, min=action._clip[..., 0], max=action._clip[..., 1])
    error = torch.square(
        target[:, command.leg_action_ids] - command.lead_joint_pos
    ).mean(dim=-1)
    return torch.exp(-error / std**2) * command.imitation_weight


def locomotion_prior_joint_position_error_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
) -> torch.Tensor:
    """Reward measured leg positions against the current reference frame."""

    if not math.isfinite(std) or std <= 0.0:
        raise ValueError("Locomotion-prior reward std must be finite and positive")
    command = _locomotion_prior_command(env, command_name)
    measured = command.robot.data.joint_pos[:, command.leg_joint_ids]
    error = torch.square(measured - command.reference_joint_pos).mean(dim=-1)
    return torch.exp(-error / std**2) * command.imitation_weight


def locomotion_prior_clip_finished(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    """Terminate eligible episodes at frame 267 instead of looping the clip."""

    command = _locomotion_prior_command(env, command_name)
    return command.finished & command.active


def set_locomotion_prior_enabled(
    env: ManagerBasedRlEnv,
    *,
    command_name: str,
    enabled: bool,
) -> None:
    """Update both the live command and its auditable resolved config."""

    command = _locomotion_prior_command(env, command_name)
    cfg = env.command_manager.get_term_cfg(command_name)
    if not isinstance(cfg, LocomotionPriorCommandCfg):
        raise TypeError("Locomotion prior command config type mismatch")
    cfg.enabled = bool(enabled)
    command.cfg.enabled = bool(enabled)
    if not enabled:
        command.disable()
