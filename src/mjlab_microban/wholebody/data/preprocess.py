# Adapted from Teleopit f926386 train_mimic/data/preprocess.py (Apache-2.0,
# https://github.com/BotRunner64/Teleopit).  Changes: frames that break a
# rule are cut out of the clip (Teleopit drops the whole clip), and the rules
# measure the retargeted pose on the training model (floor and self
# penetration, joint range, one-frame joint jumps, lifted above the
# termination height); the airborne rule is Teleopit's, at Microban's size
# and on the lowest collision point.  Added: the unsupported rule.

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
  whole run is marked);
* the trunk is upright (within ``upright_tilt_deg`` of vertical), the
  robot's centre of mass moves slower than ``quasi_static_speed_m_s`` across
  the floor and, seen from above, is more than ``support_margin_m`` outside
  the convex hull of the collision geom corners within ``off_ground_height_m``
  of the floor, for longer than ``max_unsupported_s`` (the whole run is
  marked).  The reference leans its weight on something the simulation does
  not have: a chair or a stool (a seated human puts the robot's feet on the
  floor and its trunk in the air), a rail, a wall, a partner.

  Why these numbers.  The zero-moment point, which stays inside the support,
  is the centre of mass minus (h / g) times its acceleration; over 1 s below
  0.1 m/s their means differ by at most (h / g) (2 x 0.1 m/s) / 1 s = 3.6 mm
  (h = 0.18 m), so such a reference cannot keep its centre of mass outside
  the support (an inverted pendulum of that height leaves in sqrt(h / g) =
  0.13 s).  A moving centre of mass (a run, a jump, a fall, a step) is not
  judged.  The 1 cm margin is above the retargeting's own error and below
  sitting: in the motions of LAFAN1 that lean on nothing (walk, run, sprint,
  dance, fight, jumps, ...) 0.28 % of the slow 1 s windows have the centre of
  mass more than 1 cm outside (0.65 % beyond 0.5 cm, 0.03 % beyond 1.5 cm),
  while the rule marks 17.2 % of CMU's sit at 0.5 cm, 16.3 % at 1 cm, 13.0 %
  at 1.5 cm and 9.0 % at 2 cm (CMU 114_05, seated 1.8 to 1.9 cm behind the
  feet, stayed at 2 cm).  The support is wide (corners up to the off-ground
  height count) because the retargeting leaves a standing foot hovering:
  standing still, the higher foot is 1 to 2.5 cm above the floor in 7 % of
  those frames.  Lying and crawling (trunk past ``upright_tilt_deg``) are not
  judged: their trunk is near the floor and settles onto it.

``pad_s`` around every marked frame is dropped too, and the remaining runs of
at least ``min_frames`` frames become separate clips.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from mjlab_microban.wholebody.data.fk import normalize_quaternion, set_qpos
from mjlab_microban.wholebody.env_cfg import HEIGHT_TERMINATION_M

REASONS = ("floor_penetration", "self_penetration", "joint_range", "joint_jump", "lifted", "off_ground", "unsupported")


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
    support_margin_m: float = 0.01
    quasi_static_speed_m_s: float = 0.1
    upright_tilt_deg: float = 45.0
    max_unsupported_s: float = 1.0
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
    # (T,) horizontal distance of the centre of mass outside the support (the
    # hull of the collision geom corners near the floor); inf without contact.
    com_outside_support_m: np.ndarray
    trunk_tilt_rad: np.ndarray  # (T,) the root body's z axis from vertical
    com_xy_w: np.ndarray  # (T, 2) the centre of mass seen from above


def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Counter-clockwise hull of (N, 2) points (Andrew's monotone chain; 1 or 2 points for degenerate sets)."""
    unique = sorted(set(map(tuple, np.round(np.asarray(points, dtype=np.float64), 9))))
    if len(unique) <= 2:
        return np.asarray(unique, dtype=np.float64).reshape(-1, 2)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def distance_outside(point: np.ndarray, hull: np.ndarray) -> float:
    """Distance of a 2D point outside a convex hull (0 inside, inf for an empty hull)."""
    if len(hull) == 0:
        return float("inf")
    if len(hull) == 1:
        return float(np.linalg.norm(point - hull[0]))
    a, b = hull, np.roll(hull, -1, axis=0)
    edge = b - a
    rel = point - a
    if len(hull) > 2 and np.all(edge[:, 0] * rel[:, 1] - edge[:, 1] * rel[:, 0] >= 0.0):
        return 0.0
    t = np.clip(np.einsum("ij,ij->i", rel, edge) / np.maximum(np.einsum("ij,ij->i", edge, edge), 1e-18), 0.0, 1.0)
    return float(np.min(np.linalg.norm(rel - t[:, None] * edge, axis=1)))


class ClipChecker:
    """FK and the frame rules on one compiled training model (robot + floor).

    The support is the hull of the collision geom corners (a cylinder's end
    circles as 8 points each) within ``contact_height_m`` of the floor."""

    def __init__(
        self, model: mujoco.MjModel, robot_prefix: str = "", contact_height_m: float = ExclusionSpec.off_ground_height_m
    ) -> None:
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
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
        angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
        corner_geom, corner_local = [], []
        for geom, size in zip(self.geom_ids, self.geom_size):
            if int(model.geom_type[geom]) == box:
                local = signs * size
            else:
                ring = np.stack([size[0] * np.cos(angles), size[0] * np.sin(angles)], axis=1)
                local = np.concatenate([np.column_stack([ring, np.full(8, z)]) for z in (-size[1], size[1])])
            corner_geom.extend([geom] * len(local))
            corner_local.append(local)
        self.corner_geom = np.asarray(corner_geom)
        self.corner_local = np.concatenate(corner_local)
        self.contact_height_m = contact_height_m
        free = [j for j in range(model.njnt) if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)]
        self.root_body = int(model.jnt_bodyid[free[0]])

    def _support(self) -> tuple[float, float, np.ndarray]:
        """(centre of mass outside the support, trunk tilt, centre of mass (x, y)) of the current pose;
        mj_kinematics has run."""
        mujoco.mj_comPos(self.model, self.data)
        xmat = self.data.geom_xmat[self.corner_geom].reshape(-1, 3, 3)
        corners = self.data.geom_xpos[self.corner_geom] + np.einsum("nij,nj->ni", xmat, self.corner_local)
        hull = convex_hull_2d(corners[corners[:, 2] < self.contact_height_m, :2])
        com_xy = self.data.subtree_com[self.root_body, :2].copy()
        outside = distance_outside(com_xy, hull)
        tilt = float(np.arccos(np.clip(self.data.xmat[self.root_body, 8], -1.0, 1.0)))
        return outside, tilt, com_xy

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
        outside = np.empty(frames)
        tilt = np.empty(frames)
        com_xy = np.empty((frames, 2))
        for frame in range(frames):
            set_qpos(self.model, self.data, root_pos[frame], root_quat[frame], joint_pos[frame])
            mujoco.mj_collision(self.model, self.data)
            body_pos[frame] = self.data.xpos[self.body_ids]
            body_quat[frame] = self.data.xquat[self.body_ids]
            lowest[frame] = self._lowest_point()
            self_pen[frame] = self._self_penetration()
            outside[frame], tilt[frame], com_xy[frame] = self._support()
        excess = np.maximum(self.joint_lower - joint_pos, joint_pos - self.joint_upper)
        return ClipKinematics(
            body_pos_w=body_pos,
            body_quat_w=normalize_quaternion(body_quat),
            lowest_point_z=lowest,
            self_penetration_m=self_pen,
            joint_excess_rad=np.maximum(np.max(excess, axis=1), 0.0),
            joint_step_rad=np.abs(np.diff(joint_pos, axis=0)),
            com_outside_support_m=outside,
            trunk_tilt_rad=tilt,
            com_xy_w=com_xy,
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
    com_speed = (
        np.linalg.norm(np.gradient(kin.com_xy_w, axis=0), axis=1) * fps
        if len(kin.com_xy_w) > 1
        else np.zeros(len(kin.com_xy_w))
    )
    leaning = (
        np.isfinite(kin.com_outside_support_m)
        & (kin.com_outside_support_m > spec.support_margin_m)
        & (kin.trunk_tilt_rad < np.radians(spec.upright_tilt_deg))
        & (com_speed < spec.quasi_static_speed_m_s)
    )
    unsupported = np.zeros_like(leaning)
    longest = int(round(spec.max_unsupported_s * fps))
    for start, end in _runs(leaning):
        if end - start > longest:
            unsupported[start:end] = True
    return {
        "floor_penetration": kin.lowest_point_z < -spec.floor_penetration_m,
        "self_penetration": kin.self_penetration_m > spec.self_penetration_m,
        "joint_range": kin.joint_excess_rad > spec.joint_range_excess_rad,
        "joint_jump": jumped,
        "lifted": kin.lowest_point_z > spec.lifted_height_m,
        "off_ground": off_ground,
        "unsupported": unsupported,
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
