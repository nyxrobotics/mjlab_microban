# Adapted from Teleopit f926386 train_mimic/tasks/tracking/mdp/commands.py
# (Apache-2.0, https://github.com/BotRunner64/Teleopit).  Changes: no future
# window, heading terms or foot contact estimate; the dataset carries clip
# names and joint names; the rewards and the actor see the policy's joints
# (``joint_names``) while resets write all joints; ``reset_count`` lets the
# neck target snap after a reset; an env placed on the reference sees it one
# control step ahead whether env.step or env.reset placed it; the end of a
# window holds its last frame and ends the episode as a time-out (Teleopit
# placed the robot on a new window inside the command update); the reference
# can be held on its frame, standing still (``hold``, the evaluation's start).

"""Reference motion: a clip library on the device and the motion command.

Timing.  The reference the policy observes is the target of the coming
control step: an env placed on reference time ``t`` observes ``t + step_dt``,
and each step's rewards and terminations compare the robot after the step
with that reference.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast

import h5py
import mujoco
import numpy as np
import torch
from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

from mjlab_microban.wholebody.data.hdf5 import (
    FRAME_KEYS,
    find_motion_shards,
    motion_category,
    read_strings,
    validate_motion_shard,
)

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


def _batched_quat_slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Batched quaternion slerp.  q0, q1: (..., 4) wxyz, t: (...)."""
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs()
    t = t.unsqueeze(-1)
    omega = torch.acos(torch.clamp(dot, -1.0, 1.0))
    sin_omega = torch.sin(omega)
    safe = sin_omega.abs() > 1e-8
    w0 = torch.where(safe, torch.sin((1.0 - t) * omega) / sin_omega, 1.0 - t)
    w1 = torch.where(safe, torch.sin(t * omega) / sin_omega, t)
    result = w0 * q0 + w1 * q1
    return result / result.norm(dim=-1, keepdim=True)


class MotionLib:
    """All windows of a dataset directory (or shard), on ``device``.

    Joint arrays are reordered to ``joint_names`` and body arrays restricted to
    ``body_names`` (both in the caller's order).  Window ``i`` comes from the
    source clip ``window_clip_names[i]``.
    """

    def __init__(
        self,
        motion_file: str,
        body_names: tuple[str, ...],
        joint_names: tuple[str, ...],
        device: str = "cpu",
    ) -> None:
        self._device = device
        arrays: dict[str, list[torch.Tensor]] = {key: [] for key in FRAME_KEYS}
        offsets, lengths, fps, names = [], [], [], []
        base = 0
        for shard in find_motion_shards(motion_file):
            validate_motion_shard(shard)
            with h5py.File(shard, "r") as h5:
                shard_bodies = read_strings(h5, "body_names")
                shard_joints = read_strings(h5, "joint_names")
                missing = [n for n in (*body_names, *joint_names) if n not in shard_bodies + shard_joints]
                if missing:
                    raise ValueError(f"{shard} lacks {missing}; rebuild the dataset")
                body_idx = [shard_bodies.index(name) for name in body_names]
                joint_idx = [shard_joints.index(name) for name in joint_names]
                for key in FRAME_KEYS:
                    data = np.asarray(h5[key], dtype=np.float32)
                    data = data[:, joint_idx] if key.startswith("joint") else data[:, body_idx]
                    arrays[key].append(torch.from_numpy(np.ascontiguousarray(data)))
                clip_names = read_strings(h5, "clip_names")
                offsets.extend((base + np.asarray(h5["clip_starts"])).tolist())
                lengths.extend(np.asarray(h5["clip_lengths"]).tolist())
                fps.extend(np.asarray(h5["clip_fps"]).tolist())
                names.extend(clip_names[i] for i in np.asarray(h5["source_clip_ids"]))
                base += int(h5["joint_pos"].shape[0])
        if not lengths:
            raise ValueError(f"motion dataset is empty: {motion_file}")
        if min(lengths) < 2:
            raise ValueError("every window needs two frames to interpolate")
        dev = torch.device(device)
        self.tensors = {key: torch.cat(values, dim=0).to(dev) for key, values in arrays.items()}
        self.window_clip_names = tuple(names)
        self.window_categories = tuple(motion_category(name) for name in names)
        self.clip_frame_offsets = torch.tensor(offsets, dtype=torch.long, device=dev)
        self.clip_lengths = torch.tensor(lengths, dtype=torch.long, device=dev)
        self.clip_fps = torch.tensor(fps, dtype=torch.float32, device=dev)
        self.num_clips = len(lengths)
        self.clip_duration_s = (self.clip_lengths.float() - 1.0) / self.clip_fps
        # Sample centre frames in [0, length - 1): interpolation needs frame + 1.
        self.clip_sample_start_s = torch.zeros(self.num_clips, device=dev)
        self.clip_sample_end_s = self.clip_duration_s.clone()
        self.sample_weights = self.clip_duration_s / torch.sum(self.clip_duration_s)

    def sample_motion_ids(self, n: int) -> torch.Tensor:
        """``n`` window ids, weighted by duration."""
        return torch.multinomial(self.sample_weights, int(n), replacement=True)

    def sample_times(self, motion_ids: torch.Tensor) -> torch.Tensor:
        """Uniform times over each window's valid range."""
        start = self.clip_sample_start_s[motion_ids]
        end = self.clip_sample_end_s[motion_ids]
        return start + torch.rand_like(start) * (end - start)

    def get_frames(self, motion_ids: torch.Tensor, motion_times: torch.Tensor) -> dict[str, torch.Tensor]:
        """Interpolated reference state at ``(window, time)``."""
        fps = self.clip_fps[motion_ids]
        frame_f = torch.clamp(motion_times, min=0.0) * fps
        frame_i0 = frame_f.long()
        last = self.clip_lengths[motion_ids] - 1
        frame_i0 = torch.minimum(frame_i0, last)
        frame_i1 = torch.minimum(frame_i0 + 1, last)
        alpha = frame_f - frame_i0.float()
        idx0 = self.clip_frame_offsets[motion_ids] + frame_i0
        idx1 = self.clip_frame_offsets[motion_ids] + frame_i1
        out = {}
        for key in ("joint_pos", "joint_vel"):
            v0, v1 = self.tensors[key][idx0], self.tensors[key][idx1]
            out[key] = v0 + alpha[:, None] * (v1 - v0)
        for key in ("body_pos_w", "body_lin_vel_w", "body_ang_vel_w"):
            v0, v1 = self.tensors[key][idx0], self.tensors[key][idx1]
            out[key] = v0 + alpha[:, None, None] * (v1 - v0)
        q0, q1 = self.tensors["body_quat_w"][idx0], self.tensors["body_quat_w"][idx1]
        out["body_quat_w"] = _batched_quat_slerp(q0, q1, alpha[:, None].expand(q0.shape[:2]))
        return out


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg
    _env: ManagerBasedRlEnv

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        if cfg.rewind_min_steps < 0 or cfg.rewind_max_steps < cfg.rewind_min_steps:
            raise ValueError("rewind steps must satisfy 0 <= min <= max")
        if not 0.0 <= cfg.rewind_prob <= 1.0:
            raise ValueError("rewind_prob must be in [0, 1]")
        self.robot: Entity = env.scene[cfg.entity_name]
        if cfg.body_names[0] != self.robot.body_names[0]:
            raise ValueError("body_names[0] must be the root body (resets place it)")
        self.robot_anchor_body_index = self.robot.body_names.index(cfg.anchor_body_name)
        self.motion_anchor_body_index = cfg.body_names.index(cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )
        all_joints = tuple(self.robot.joint_names)
        self.tracked_joint_ids = torch.tensor(
            [all_joints.index(name) for name in cfg.joint_names], dtype=torch.long, device=self.device
        )
        self.motion = MotionLib(cfg.motion_file, cfg.body_names, all_joints, device=self.device)

        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_times = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        # Incremented whenever an env's robot is placed on the reference.
        self.reset_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Placed since the last update: advance by one control step.
        self._placed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # The window ran out: the reference holds its last frame and the
        # clip_end termination (a time-out) ends the episode.
        self.clip_ended = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Held (``hold``): the reference stays on its frame and its velocities read 0.
        self.held = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._step_dt = env.step_dt
        self._elapsed_s = env.step_dt
        self._cached_frames: dict[str, torch.Tensor] = {}
        self._refresh_frame_cache()

        nb = len(cfg.body_names)
        self.body_pos_relative_w = torch.zeros(self.num_envs, nb, 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, nb, 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0
        for name in ("anchor_pos", "anchor_rot", "anchor_lin_vel", "anchor_ang_vel", "body_pos", "body_rot",
                     "body_lin_vel", "body_ang_vel", "joint_pos", "joint_vel"):
            self.metrics[f"error_{name}"] = torch.zeros(self.num_envs, device=self.device)
        self._ghost_model: mujoco.MjModel | None = None

    def _refresh_frame_cache(self) -> None:
        frames = self.motion.get_frames(self.motion_ids, self.motion_times)
        for key in ("joint_vel", "body_lin_vel_w", "body_ang_vel_w"):
            held = self.held.view(-1, *([1] * (frames[key].ndim - 1)))
            frames[key] = torch.where(held, torch.zeros_like(frames[key]), frames[key])
        self._cached_frames = frames

    def hold(self, held: torch.Tensor) -> None:
        """Hold the reference of the envs in the mask ``held`` (N,) and let the others play.

        A held reference keeps its time and shows its frame standing still:
        the joint, trunk and body velocities read 0 (the evaluation's first
        second, where the operator stands at a frame while the robot comes
        from HOME; the frame's own velocities are those of the motion that
        passes through it).  An env placed while held still moves one control
        step past its placed time.  Takes effect from the next update, so a
        step's observed target and the target it is judged on stay the same.
        """
        self.held.copy_(held.to(device=self.device, dtype=torch.bool))

    # Reference (interpolated).

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def all_joint_pos(self) -> torch.Tensor:
        """Reference angles of every robot joint, in robot order."""
        return self._cached_frames["joint_pos"]

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._cached_frames["joint_pos"][:, self.tracked_joint_ids]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._cached_frames["joint_vel"][:, self.tracked_joint_ids]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._cached_frames["body_pos_w"] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._cached_frames["body_quat_w"]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._cached_frames["body_lin_vel_w"]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._cached_frames["body_ang_vel_w"]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self._cached_frames["body_pos_w"][:, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self._cached_frames["body_quat_w"][:, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self._cached_frames["body_lin_vel_w"][:, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self._cached_frames["body_ang_vel_w"][:, self.motion_anchor_body_index]

    # Robot.

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos[:, self.tracked_joint_ids]

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel[:, self.tracked_joint_ids]

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_link_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_link_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

    # Metrics.

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)
        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(-1)
        self.metrics["error_body_rot"] = quat_error_magnitude(
            self.body_quat_relative_w, self.robot_body_quat_w
        ).mean(-1)
        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(-1)
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(-1)
        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    # Sampling.

    def _uniform_sampling(self, env_ids: torch.Tensor) -> None:
        self.motion_ids[env_ids] = self.motion.sample_motion_ids(len(env_ids))
        self.motion_times[env_ids] = self.motion.sample_times(self.motion_ids[env_ids])

    def _rewind_sampling(self, env_ids: torch.Tensor) -> None:
        """A failed episode restarts 25-75 steps before its fall with ``rewind_prob``."""
        previous_ids = self.motion_ids[env_ids].clone()
        previous_times = self.motion_times[env_ids].clone()
        self._uniform_sampling(env_ids)
        if env_ids.numel() == 0 or self.cfg.rewind_prob <= 0.0:
            return
        failed = self._env.termination_manager.terminated[env_ids]
        rewind = failed & (torch.rand(env_ids.numel(), device=self.device) < self.cfg.rewind_prob)
        if not torch.any(rewind):
            return
        steps = torch.randint(
            self.cfg.rewind_min_steps, self.cfg.rewind_max_steps + 1, (int(rewind.sum()),), device=self.device
        )
        ids = previous_ids[rewind]
        times = torch.maximum(
            previous_times[rewind] - steps.to(self.motion_times.dtype) * float(self._step_dt),
            self.motion.clip_sample_start_s[ids],
        )
        self.motion_ids[env_ids[rewind]] = ids
        self.motion_times[env_ids[rewind]] = times

    def _resample_command(self, env_ids: torch.Tensor):
        if self.cfg.sampling_mode == "start":
            self.motion_ids[env_ids] = self.motion.sample_motion_ids(len(env_ids))
            self.motion_times[env_ids] = self.motion.clip_sample_start_s[self.motion_ids[env_ids]]
        elif self.cfg.sampling_mode == "uniform":
            self._uniform_sampling(env_ids)
        elif self.cfg.sampling_mode == "rewind":
            self._rewind_sampling(env_ids)
        else:
            raise ValueError(f"unsupported sampling_mode {self.cfg.sampling_mode!r}")
        self._reset_envs_to_current_reference(env_ids)

    def reset_to_motion(self, env_ids: torch.Tensor, motion_ids: torch.Tensor, motion_times: torch.Tensor) -> None:
        """Place ``env_ids`` exactly on ``(window, time)`` (deterministic evaluation)."""
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        motion_ids = motion_ids.to(device=self.device, dtype=torch.long)
        motion_times = motion_times.to(device=self.device, dtype=self.motion_times.dtype)
        if not (env_ids.ndim == motion_ids.ndim == motion_times.ndim == 1) or not (
            len(env_ids) == len(motion_ids) == len(motion_times)
        ):
            raise ValueError("env_ids, motion_ids and motion_times must be matching 1-D tensors")
        if len(env_ids) == 0:
            return
        if torch.any(motion_ids < 0) or torch.any(motion_ids >= self.motion.num_clips):
            raise ValueError("motion_ids out of range")
        start = self.motion.clip_sample_start_s[motion_ids]
        end = self.motion.clip_sample_end_s[motion_ids]
        if torch.any((motion_times < start) | (motion_times >= end)):
            raise ValueError("motion_times must lie in each window's sample range")
        self.motion_ids[env_ids] = motion_ids
        self.motion_times[env_ids] = motion_times
        self.time_left[env_ids] = float(self.cfg.resampling_time_range[1])
        self._reset_envs_to_current_reference(env_ids)

    def _reset_envs_to_current_reference(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self._refresh_frame_cache()
        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        keys = ["x", "y", "z", "roll", "pitch", "yaw"]
        ranges = torch.tensor([self.cfg.pose_range.get(k, (0.0, 0.0)) for k in keys], device=self.device)
        sample = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += sample[:, 0:3]
        root_ori[env_ids] = quat_mul(quat_from_euler_xyz(sample[:, 3], sample[:, 4], sample[:, 5]), root_ori[env_ids])
        ranges = torch.tensor([self.cfg.velocity_range.get(k, (0.0, 0.0)) for k in keys], device=self.device)
        sample = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += sample[:, :3]
        root_ang_vel[env_ids] += sample[:, 3:]

        joint_pos = self.all_joint_pos.clone()
        joint_vel = self._cached_frames["joint_vel"].clone()
        joint_pos += sample_uniform(
            self.cfg.joint_position_range[0], self.cfg.joint_position_range[1], joint_pos.shape, device=self.device
        )
        limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(joint_pos[env_ids], limits[:, :, 0], limits[:, :, 1])
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        root_state = torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], -1)
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
        self.robot.clear_state(env_ids=env_ids)
        self.reset_count[env_ids] += 1
        self._placed[env_ids] = True
        self.clip_ended[env_ids] = False

    def compute(self, dt: float) -> None:
        # dt is the elapsed time: step_dt in env.step, 0 in env.reset.
        self._elapsed_s = dt
        super().compute(dt)

    def _update_command(self):
        # An env placed since the last update (in env.step or env.reset) moves
        # one control step past its placed state; every other env by the
        # elapsed time, or not at all while held.
        advance = torch.full_like(self.motion_times, self._elapsed_s)
        advance[self.held] = 0.0
        advance[self._placed] = self._step_dt
        self._placed.zero_()
        self.motion_times += advance
        end_times = self.motion.clip_sample_end_s[self.motion_ids]
        ended = self.motion_times >= end_times
        self.motion_times = torch.where(
            ended,
            torch.nextafter(end_times, self.motion.clip_sample_start_s[self.motion_ids]),
            self.motion_times,
        )
        self.clip_ended |= ended
        self._refresh_frame_cache()

        nb = len(self.cfg.body_names)
        anchor_pos = self.anchor_pos_w[:, None, :].repeat(1, nb, 1)
        anchor_quat = self.anchor_quat_w[:, None, :].repeat(1, nb, 1)
        robot_anchor_pos = self.robot_anchor_pos_w[:, None, :].repeat(1, nb, 1)
        robot_anchor_quat = self.robot_anchor_quat_w[:, None, :].repeat(1, nb, 1)
        # The reference, moved under the robot's anchor: the robot's xy and the
        # reference's height, turned by the yaw between the two anchors.
        delta_pos = robot_anchor_pos
        delta_pos[..., 2] = anchor_pos[..., 2]
        delta_ori = yaw_quat(quat_mul(robot_anchor_quat, quat_inv(anchor_quat)))
        self.body_quat_relative_w = quat_mul(delta_ori, self.body_quat_w)
        self.body_pos_relative_w = delta_pos + quat_apply(delta_ori, self.body_pos_w - anchor_pos)

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        """The reference as a ghost robot."""
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return
        if self._ghost_model is None:
            self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
            self._ghost_model.geom_rgba[:] = np.array(self.cfg.ghost_color, dtype=np.float32)
        indexing = self.robot.indexing
        free_adr = indexing.free_joint_q_adr.cpu().numpy()
        joint_adr = indexing.joint_q_adr.cpu().numpy()
        for batch in env_indices:
            qpos = np.zeros(self._env.sim.mj_model.nq)
            qpos[free_adr[0:3]] = self.body_pos_w[batch, 0].cpu().numpy()
            qpos[free_adr[3:7]] = self.body_quat_w[batch, 0].cpu().numpy()
            qpos[joint_adr] = self.all_joint_pos[batch].cpu().numpy()
            visualizer.add_ghost_mesh(qpos, model=self._ghost_model, label=f"ghost_{batch}")


@dataclass(kw_only=True)
class MotionCommandCfg(CommandTermCfg):
    motion_file: str
    anchor_body_name: str
    body_names: tuple[str, ...]
    joint_names: tuple[str, ...]
    """The policy's joints: the actor's reference and the joint rewards."""
    entity_name: str
    pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    joint_position_range: tuple[float, float] = (-0.52, 0.52)
    sampling_mode: Literal["uniform", "start", "rewind"] = "rewind"
    rewind_prob: float = 0.8
    rewind_min_steps: int = 25
    rewind_max_steps: int = 75
    ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)

    def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
        return MotionCommand(self, env)


class ReferenceNeckTarget:
    """``mode='step'`` event: the neck targets follow the reference's neck angles.

    head, neck_roll and neck_pitch are not the policy's.  In training and on
    the robot (contract microban-wholebody-1) their targets are the
    reference's angles of these joints, clamped to the HMD controller's
    limits within the joint range and slewed at its 2.5 rad/s.  An env just
    placed on the reference (``MotionCommand.reset_count``) starts at the
    clamped reference.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
        params = cfg.params
        asset_cfg = params["asset_cfg"]
        names = tuple(asset_cfg.joint_names)
        if not isinstance(asset_cfg.joint_ids, list) or len(asset_cfg.joint_ids) != len(names):
            raise TypeError("the neck joints must resolve to explicit IDs")
        self.asset = env.scene[asset_cfg.name]
        self.joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.long, device=env.device)
        self.command_name = params["command_name"]
        ranges = params["position_ranges_rad"]
        lower = torch.tensor([float(ranges[n][0]) for n in names], device=env.device)
        upper = torch.tensor([float(ranges[n][1]) for n in names], device=env.device)
        soft = self.asset.data.soft_joint_pos_limits[:, self.joint_ids]
        self.lower = torch.maximum(lower, soft[..., 0])
        self.upper = torch.minimum(upper, soft[..., 1])
        self.slew = torch.tensor([float(params["slew_rates_rad_s"][n]) for n in names], device=env.device)
        self.target = torch.clamp(self.asset.data.default_joint_pos[:, self.joint_ids], self.lower, self.upper)
        self.seen = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Nothing: the next step snaps every env the command placed."""

    def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, **_unused) -> None:
        command = cast(MotionCommand, env.command_manager.get_term(self.command_name))
        goal = torch.clamp(command.all_joint_pos[:, self.joint_ids], self.lower, self.upper)
        step = self.slew * env.step_dt
        moved = torch.clamp(self.target + torch.clamp(goal - self.target, -step, step), self.lower, self.upper)
        placed = (command.reset_count != self.seen)[:, None]
        self.target = torch.where(placed, goal, moved)
        self.seen.copy_(command.reset_count)
        self.asset.set_joint_position_target(self.target, joint_ids=self.joint_ids)
