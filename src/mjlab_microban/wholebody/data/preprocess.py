# Adapted from Teleopit f926386 train_mimic/data/preprocess.py (Apache-2.0,
# https://github.com/BotRunner64/Teleopit).  Changes: frames that break a
# rule are cut out of the clip (Teleopit drops the whole clip), and the rules
# measure the retargeted pose on the training model (floor and self
# penetration, joint range, one-frame joint jumps, lifted above the
# termination height); the airborne rule is Teleopit's, at Microban's size
# and on the lowest collision point.

"""Cut the frames of a retargeted clip that the robot cannot be asked to follow.

A frame is marked when, on the training model (the task's collision geoms and
a floor plane at z = 0):

* a collision geom is more than ``floor_penetration_m`` below the floor;
* two robot geoms that collide in training overlap by more than
  ``self_penetration_m``;
* a joint is more than ``joint_range_excess_rad`` outside its MJCF range;
* a joint jumps between two frames: a step faster than ``joint_jump_rad_s``
  and more than ``joint_jump_ratio`` times the joint's steps before and after
  it (an IK branch switch of the retargeting; a fast human motion speeds up
  and slows down over several frames).  Both frames are marked;
* the lowest collision point is more than ``lifted_height_m`` above the floor
  (the termination height): a robot on the floor in that pose ends the
  episode, and GMR keeps human timing at 0.21 x the length, so the
  reference's flight has no gravity-consistent jump behind it;
* the lowest collision point stays above ``off_ground_height_m`` for longer
  than ``max_off_ground_s`` (Teleopit's 0.12 m x 0.21 Microban/G1 scale; the
  whole run is marked).

``pad_s`` around every marked frame is dropped too, and the remaining runs of
at least ``min_frames`` frames become separate clips.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from mjlab_microban.wholebody.data.fk import normalize_quaternion, set_qpos
from mjlab_microban.wholebody.env_cfg import HEIGHT_TERMINATION_M

REASONS = ("floor_penetration", "self_penetration", "joint_range", "joint_jump", "lifted", "off_ground")


@dataclass(frozen=True)
class ExclusionSpec:
    floor_penetration_m: float = 0.01
    self_penetration_m: float = 0.005
    joint_range_excess_rad: float = 0.1
    joint_jump_rad_s: float = 15.0
    joint_jump_ratio: float = 2.0
    lifted_height_m: float = HEIGHT_TERMINATION_M
    off_ground_height_m: float = 0.025
    max_off_ground_s: float = 0.8
    pad_s: float = 0.2
    min_frames: int = 22

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class ClipKinematics:
    body_pos_w: np.ndarray  # (T, B, 3)
    body_quat_w: np.ndarray  # (T, B, 4) wxyz
    lowest_point_z: np.ndarray  # (T,) lowest collision point
    self_penetration_m: np.ndarray  # (T,) deepest robot-robot overlap
    joint_excess_rad: np.ndarray  # (T,) largest excursion beyond a joint range
    joint_step_rad: np.ndarray  # (T - 1, J) |q[t + 1] - q[t]|


class ClipChecker:
    """FK and the frame rules on one compiled training model (robot + floor)."""

    def __init__(self, model: mujoco.MjModel, robot_prefix: str = "") -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.prefix = robot_prefix
        nbody = model.nbody
        self.body_ids = np.arange(1, nbody)
        self.body_names = tuple(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(b))[len(robot_prefix):]
            for b in self.body_ids
        )
        hinges = [j for j in range(model.njnt) if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE)]
        self.joint_names = tuple(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)[len(robot_prefix):] for j in hinges
        )
        if [int(model.jnt_qposadr[j]) for j in hinges] != list(range(7, 7 + len(hinges))):
            raise ValueError("expected one free joint followed by the hinges in qpos")
        limited = np.asarray([bool(model.jnt_limited[j]) for j in hinges])
        self.joint_lower = np.where(limited, model.jnt_range[hinges, 0], -np.inf)
        self.joint_upper = np.where(limited, model.jnt_range[hinges, 1], np.inf)
        robot_geoms = [
            g
            for g in range(model.ngeom)
            if model.geom_bodyid[g] != 0 and (model.geom_contype[g] or model.geom_conaffinity[g])
        ]
        box, cylinder = int(mujoco.mjtGeom.mjGEOM_BOX), int(mujoco.mjtGeom.mjGEOM_CYLINDER)
        if any(int(model.geom_type[g]) not in (box, cylinder) for g in robot_geoms):
            raise ValueError("lowest point is defined for box and cylinder collision geoms")
        self.geom_ids = np.asarray(robot_geoms)
        self.geom_is_box = model.geom_type[self.geom_ids] == box
        self.geom_size = model.geom_size[self.geom_ids].copy()
        self.robot_geom = np.zeros(model.ngeom, dtype=bool)
        self.robot_geom[self.geom_ids] = True

    def _lowest_point(self) -> float:
        z = self.data.geom_xpos[self.geom_ids, 2]
        rz = self.data.geom_xmat[self.geom_ids].reshape(-1, 3, 3)[:, 2, :]
        box = z - np.sum(np.abs(rz) * self.geom_size, axis=-1)
        axis = np.abs(rz[:, 2])
        cylinder = z - axis * self.geom_size[:, 1] - self.geom_size[:, 0] * np.sqrt(
            np.clip(1.0 - axis**2, 0.0, 1.0)
        )
        return float(np.min(np.where(self.geom_is_box, box, cylinder)))

    def _self_penetration(self) -> float:
        deepest = 0.0
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if self.robot_geom[contact.geom1] and self.robot_geom[contact.geom2]:
                deepest = max(deepest, -float(contact.dist))
        return deepest

    def run(self, root_pos: np.ndarray, root_quat_wxyz: np.ndarray, joint_pos: np.ndarray) -> ClipKinematics:
        root_pos = np.asarray(root_pos, dtype=np.float64)
        root_quat = np.asarray(normalize_quaternion(root_quat_wxyz), dtype=np.float64)
        joint_pos = np.asarray(joint_pos, dtype=np.float64)
        frames = root_pos.shape[0]
        if root_quat.shape != (frames, 4) or joint_pos.shape != (frames, len(self.joint_names)):
            raise ValueError("root_pos, root_quat and joint_pos must share the frame count")
        body_pos = np.empty((frames, len(self.body_ids), 3), dtype=np.float32)
        body_quat = np.empty((frames, len(self.body_ids), 4), dtype=np.float32)
        lowest = np.empty(frames)
        self_pen = np.empty(frames)
        for frame in range(frames):
            set_qpos(self.model, self.data, root_pos[frame], root_quat[frame], joint_pos[frame])
            mujoco.mj_collision(self.model, self.data)
            body_pos[frame] = self.data.xpos[self.body_ids]
            body_quat[frame] = self.data.xquat[self.body_ids]
            lowest[frame] = self._lowest_point()
            self_pen[frame] = self._self_penetration()
        excess = np.maximum(self.joint_lower - joint_pos, joint_pos - self.joint_upper)
        return ClipKinematics(
            body_pos_w=body_pos,
            body_quat_w=normalize_quaternion(body_quat),
            lowest_point_z=lowest,
            self_penetration_m=self_pen,
            joint_excess_rad=np.maximum(np.max(excess, axis=1), 0.0),
            joint_step_rad=np.abs(np.diff(joint_pos, axis=0)),
        )


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """``[start, end)`` runs of True."""
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    edges = np.diff(padded)
    return list(zip(np.where(edges == 1)[0].tolist(), np.where(edges == -1)[0].tolist()))


def joint_jumps(step: np.ndarray, fps: int, spec: ExclusionSpec) -> np.ndarray:
    """(T - 1,) the steps where some joint jumps (see the module docstring)."""
    if step.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    padded = np.pad(step, ((1, 1), (0, 0)))
    neighbours = np.maximum(padded[:-2], padded[2:])
    return np.any((step * fps > spec.joint_jump_rad_s) & (step > spec.joint_jump_ratio * neighbours), axis=1)


def frame_flags(kin: ClipKinematics, fps: int, spec: ExclusionSpec) -> dict[str, np.ndarray]:
    """Per-reason marks (T,) of one clip."""
    jump = joint_jumps(kin.joint_step_rad, fps, spec)
    jumped = np.zeros(kin.lowest_point_z.shape, dtype=bool)
    jumped[:-1] |= jump
    jumped[1:] |= jump
    airborne = kin.lowest_point_z > spec.off_ground_height_m
    off_ground = np.zeros_like(airborne)
    longest = int(round(spec.max_off_ground_s * fps))
    for start, end in _runs(airborne):
        if end - start > longest:
            off_ground[start:end] = True
    return {
        "floor_penetration": kin.lowest_point_z < -spec.floor_penetration_m,
        "self_penetration": kin.self_penetration_m > spec.self_penetration_m,
        "joint_range": kin.joint_excess_rad > spec.joint_range_excess_rad,
        "joint_jump": jumped,
        "lifted": kin.lowest_point_z > spec.lifted_height_m,
        "off_ground": off_ground,
    }


def kept_segments(flags: dict[str, np.ndarray], fps: int, spec: ExclusionSpec) -> list[tuple[int, int]]:
    """``[start, end)`` runs left after padding the marks, at least ``min_frames`` long."""
    marked = np.zeros_like(next(iter(flags.values())), dtype=bool)
    for mask in flags.values():
        marked |= mask
    pad = int(round(spec.pad_s * fps))
    index = np.flatnonzero(marked)
    edges = np.zeros(marked.size + 1, dtype=np.int64)
    np.add.at(edges, np.maximum(index - pad, 0), 1)
    np.add.at(edges, np.minimum(index + pad + 1, marked.size), -1)
    dropped = np.cumsum(edges)[:-1] > 0
    return [(s, e) for s, e in _runs(~dropped) if e - s >= spec.min_frames]
