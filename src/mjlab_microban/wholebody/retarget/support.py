"""The support changes of the retargeting (``core.retarget_frames(..., support_spec=SupportSpec())``; off by default).

Microban's feet are far apart for its size (HOME's sole centres 93 mm apart, 0.75 leg lengths; a human's
0.2-0.25), and its dynamics are quick (sqrt(h / g) = 0.11 s at its centre of mass's height, a human's 0.3 s), so
the human's motion, scaled with the human's timing, puts Microban's weight outside the foot it stands on.  The
support changes change the targets, as little as needed and in a fixed order, so that Microban's zero-moment point
(ZMP) stays in its support:

1. the trunk moved horizontally (forward and sideways in its heading; its height but where a standing leg would
   not reach its foot, by at most ``max_drop_m``), the arms with it (their pose relative to the trunk is kept);
2. the trunk tilted (roll and pitch, about the hips' midpoint), the arms with it;
3. the feet rolled about their forward axis (the ankle roll);
4. the feet put elsewhere on the floor (each time a foot is on the floor, one offset; the support moves with it).

Each is allowed only around the frames where the ZMP stayed outside with the ones before it (stages): the clip is
solved with 1, then again around the frames still outside with 1 and 2, and so on; what 1-4 do not bring in is left.
Each change is capped (``max_lateral_m``, ``max_forward_m``, ``max_tilt_deg``, ``max_ankle_roll_deg``,
``max_foot_m``; the data command's ``--support-max-*``).  A foot off the floor moves with the trunk (its leg keeps
its pose relative to the trunk) and back to where it lands before it lands, as part of 1.

The feet on the floor are held where they stand (foot_lock): over each run on the floor a foot's target keeps its
sole's place and its heading (within HOLD_TURN_DEG of the human's), rolling over its toes (heels) as the human's
foot pitches; 4 moves a run's place.  The IK (core) then weighs a held foot LOCK_GAIN times and the trunk's
orientation TRUNK_TURN_GAIN times their usual weights, and solves the unchanged clip again from itself to
convergence before the changes are found (its first pass is not converged, and the changes, solved from it, would
carry the scatter).

* The ZMP is Microban's, from its whole-body dynamics: the qpos sequence through mj_inverse on robot.xml (the
  masses and inertias the training simulates; the arms' and the trunk's angular momentum included), without
  contacts.  The free joint's force and torque are then the wrench the floor has to give, and the ZMP is the point
  of the floor (z = 0) about which its horizontal moment is zero.  Frames whose vertical force is below MIN_LOAD of
  the weight (falling, flight) have none.  The human's motion's velocities and accelerations are those of a cubic
  fitted over ``smooth_s`` (0.3 s; with central differences the capture's frame-to-frame noise moves the ZMP by
  7-68 mm per frame), and the changes' (the tangent from the unchanged pose) those of a fit over ``change_s``
  (5 frames; Dyn): a 0.3 s cubic sees a 2 Hz acceleration at 0.78 of its size and a 3 Hz one at 0.56, so solved
  against it the changes would lean on what it does not see; 5 frames see them at 0.94 and 0.86, and pass the
  IK's frame-to-frame scatter (0.7 mm of trunk height, up to 20 mrad in a joint) at 0.22 of central differences,
  which moved the ZMP by 4 mm (median) against a 5-frame fit.
* The parts on the floor (robot_contact): a part is on the floor when the human's is (as low as when standing for
  a foot, as low as when lying on the floor for the others (core.CONTACT_RADIUS), and slow across the floor) and
  Microban's is within ROBOT_CONTACT_M of the floor (the training side's support height), or when Microban's
  touches the floor (ROBOT_TOUCH_M) and is slow.  Gaps of up to GAP_FRAMES and contacts of one frame are filled
  and dropped.  Lying and crawling (the trunk tilted past LYING_TILT_DEG with a part other than the feet on the
  floor) have no part on the floor.  The changed clip's parts are found again from its own poses by the same rule.
* The support S(t) is the convex hull of the floor contact areas of the parts on the floor, shrunk inward by
  ``margin_m`` (10 mm by default; the data command's ``--support-margin-mm``).  A part's contact area is the part
  of its training collision shapes within ``band_m`` (10 mm) of its lowest point, seen from above (a sole rolled
  past asin(band / width) = 14 degrees stands on its edge).  Where the hull is too narrow to shrink by
  ``margin_m``, S(t) is its deepest points (its middle line or point).
* The changes are 0 where nothing is on the floor (flight, lying, crawling) and where a part other than the feet
  is (kneeling, a hand down: the IK's knee and hand targets are too weak to keep them on the floor under a moving
  trunk); they go to 0 over ``fade_s`` before and come back over ``fade_s`` after.  A stage's new change fades in
  and out over ``fade_s`` around the frames it is allowed at.
* Before a part leaves the floor (lift-off), the ZMP moves into what remains: over the last ``lift_s`` of the
  contact phase (at most its length) the allowed area is S(t) within a distance of the remaining support that
  falls linearly to 0 at the last frame on the floor, and that frame and the first one off the floor weigh
  ``w_lift`` times the others.  A part that leaves the floor in the frame another lands (no frame with both) is
  counted on the floor in both frames.
* The per-frame changes (1-3) are cubic B-splines with knots ``knot_s`` apart (0.1 s: the quick dynamics need it;
  0.2 s left half of a walk's single-support frames outside with every limit lifted), the feet's offsets are
  held while the foot is on the floor and blend (smoothstep) while it is off.  A stage solves by least squares
  over the frames it may change: the ZMP outside the allowed area (aimed ``aim_m`` inside; squared up to
  ``huber_m``, linear beyond, so that a ZMP far out that nothing brings in, a push-off before a jump, does not
  swing the changes) weighs ``w_outside`` (enough that 1 mm outside costs more than any change at its cap), each
  change its size (``w_shift`` < ``w_tilt`` < ``w_ankle`` < ``w_foot`` per mm of ZMP it moves, so that within a
  stage the earlier ones are used first), its acceleration and the shift's forward speed (the trunk's progress),
  and, as near-hard limits, a standing leg stretched beyond its straight length (less ``reach_slack_m``; the trunk
  may go down by ``max_drop_m`` for it), a standing leg's hip roll within ``joint_margin_deg`` of the end of its
  range (at the end the IK lifts the foot instead; one already there unchanged is left to the IK) and the caps.
  The response of the ZMP and of the hip rolls to the per-frame changes is found numerically: the frames are
  solved again with one change moved by ``probe_m`` / ``probe_rad`` (the IK and the floor alignment, each frame
  started from the unchanged solution moved with the trunk), and the ZMP at t is recomputed with each frame of its
  changes' fit moved by 1 % of that.  A foot's offset moves the support with it, its leg's mass (LEG_SHARE
  of the weight) and its hip roll as the opposite shift does.  The frames are then solved with the new changes,
  the ZMP computed again and the least squares solved again from there, ``iterations`` times (``stage_iterations``
  after the first stage); the changes with the least ZMP outside are kept.  The ZMP of the clip's first and last
  half window of the fit is fitted from one side and not constrained, and the changes are held there (as at the
  first and last constrained frame): free, they moved there to make the acceleration the constrained frames' fits see.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass

import mujoco
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
from scipy.signal import savgol_coeffs
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation

from mjlab_microban.wholebody.data.preprocess import convex_hull_2d, distance_outside
from mjlab_microban.wholebody.retarget.model import training_collision_geoms

GRAVITY = 9.81
MIN_LOAD = 0.25  # the ZMP needs the floor to push up with at least this share of the weight
ROBOT_CONTACT_M = 0.025  # a part on the floor is at most this high (preprocess's off_ground_height_m)
# Microban's part touching the floor (this low and slower than this across it) is on the floor whatever the
# human's: lying on its side it lies on the trunk and an arm where the human's trunk is a little higher.
ROBOT_TOUCH_M = 0.005
ROBOT_TOUCH_SPEED = 0.1
# A part's gaps on the floor of up to this many frames are filled (a foot is not up and down again in 67 ms; the
# robot's height condition breaks the human's contact for a frame or two where the human's is whole).
GAP_FRAMES = 2
# Lying and crawling (the trunk more than this from upright with a part other than the feet on the floor) have no
# support, as by preprocess's unsupported rule: Microban's arms and trunk are 6-45 mm above the floor where the
# human's lie on it, so its zero-moment point is 70-140 mm outside what touches the floor.
LYING_TILT_DEG = 45.0
# A human part is on the floor when its joints are less than CONTACT_HEIGHT leg lengths above where they are on
# the floor and the slowest of them moves less than CONTACT_SPEED leg lengths per second across it (CMU 07_01: a
# standing foot 0-0.02 high and below 0.3 per second, a swinging one 0.05-0.14 high and 2.6-5.4 per second).
CONTACT_HEIGHT = 0.05
CONTACT_SPEED = 0.8
HARD = 1e3  # the weight of the near-hard limits, per w_outside
INSIDE_M = 1e-4  # a ZMP at most this far outside S(t) is in it
STAGES = ("shift", "tilt", "ankle", "foot")
# The per-frame changes (Changes.x's columns) and the stage each comes in at.
COLUMNS = ("lateral", "forward", "roll", "pitch", "left_ankle", "right_ankle")
COLUMN_STAGE = (1, 1, 2, 2, 3, 3)


@dataclass(frozen=True)
class SupportSpec:
    """The support changes' settings (module docstring)."""

    margin_m: float = 0.010  # S(t): the contact hull shrunk inward by this
    band_m: float = 0.010  # a part's contact area: its shapes within this of its lowest point
    smooth_s: float = 0.3  # the human motion's velocities and accelerations: a cubic fitted over this
    change_s: float = 0.17  # the changes' velocities and accelerations: a fit over this (5 frames at 30 fps)
    lift_s: float = 0.3  # the ZMP moves into the remaining support over this before a part leaves the floor
    knot_s: float = 0.1  # the per-frame changes are cubic B-splines with knots this far apart
    stance_s: float = 0.2  # a foot that left the floor stops lowering the trunk for its leg over this
    reach_slack_m: float = 0.002  # a standing leg reaches its foot this short of its straight length
    max_drop_m: float = 0.01  # the trunk goes down at most this for a standing leg to reach its foot
    joint_margin_deg: float = 1.0  # a standing leg's hip roll is kept this far from the end of its range
    # The caps of the changes, in the order they are allowed (STAGES).
    max_lateral_m: float = 0.04
    max_forward_m: float = 0.03
    max_tilt_deg: float = 10.0
    max_ankle_roll_deg: float = 5.0
    max_foot_m: float = 0.02
    # Least squares weights, per frame and squared: the ZMP outside (per m; w_lift times at a lift-off), the
    # changes' sizes (shift and foot per m, tilt and ankle per rad), their accelerations (per m/s^2; the tilt's and
    # the ankle's per rad/s^2 times lever_m), the shift's forward speed (per m/s) and a change where it must be 0.
    w_outside: float = 1e4
    w_lift: float = 10.0
    w_shift: float = 1e-2
    w_tilt: float = 1e-4
    w_ankle: float = 1e-3
    w_foot: float = 2e-1
    w_accel: float = 1e-4
    lever_m: float = 0.05
    w_speed_forward: float = 0.3  # 1e-2 let the trunk stop and go at every step (forward speed std x1.9)
    w_zero: float = 1e7
    fade_s: float = 0.2  # a change goes to 0 over this before a frame where it must be 0
    huber_m: float = 0.03  # the ZMP's distance outside counts squared up to this and linearly beyond
    aim_m: float = 0.002
    probe_m: float = 0.01
    probe_rad: float = 0.03
    iterations: int = 2  # (3 and 2: the same share in S on CMU 07_01, 16_18 and LAFAN1 walk3, 30 % more IK)
    stage_iterations: int = 1

    def __post_init__(self):
        for name in ("margin_m", "max_lateral_m", "max_forward_m", "max_tilt_deg", "max_ankle_roll_deg", "max_foot_m"):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and math.isfinite(value) and value >= 0.0):
                what = "the support margin" if name == "margin_m" else f"the support's {name}"
                raise ValueError(f"{what} must be a finite number >= 0, not {value!r}")

    def to_dict(self) -> dict:
        return asdict(self)

    def caps(self) -> np.ndarray:
        """The per-frame changes' caps (COLUMNS)."""
        tilt, ankle = math.radians(self.max_tilt_deg), math.radians(self.max_ankle_roll_deg)
        return np.array([self.max_lateral_m, self.max_forward_m, tilt, tilt, ankle, ankle])


# ---------------------------------------------------------------------------------------------------------
# The zero-moment point from the whole-body dynamics.


def _inverse_model(model: mujoco.MjModel) -> mujoco.MjModel:
    m = copy.copy(model)
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
    return m


def windows(T: int, fps: float, smooth_s: float):
    """Per frame the frames its derivatives are fitted over (T, W) and its place among them (T,): ``smooth_s``
    centred (at least 3 frames), moved inside the clip at its ends."""
    W = min(T, max(3, round(smooth_s * fps) // 2 * 2 + 1))
    start = np.clip(np.arange(T) - W // 2, 0, T - W)
    return start[:, None] + np.arange(W)[None], np.arange(T) - start


def _coefficients(W: int, fps: float):
    """The Savitzky-Golay (cubic, or quadratic over 3 frames) first and second derivative weights (W, W) by the
    frame's place in its window."""
    order = min(3, W - 1)
    c1 = np.stack([savgol_coeffs(W, order, deriv=1, delta=1.0 / fps, pos=p, use="dot") for p in range(W)])
    c2 = np.stack([savgol_coeffs(W, order, deriv=2, delta=1.0 / fps, pos=p, use="dot") for p in range(W)])
    return c1, c2


def tangent(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    """mj_differentiatePos(qa, qb) for one free joint and hinges, vectorised (..., nq) -> (..., nq - 1): the
    root's displacement (world) and turn (its own frame), then the joints."""
    shape = np.broadcast_shapes(qa.shape, qb.shape)
    qa, qb = np.broadcast_to(qa, shape), np.broadcast_to(qb, shape)
    out = np.empty(shape[:-1] + (shape[-1] - 1,))
    out[..., :3] = qb[..., :3] - qa[..., :3]
    ra = Rotation.from_quat(qa[..., 3:7].reshape(-1, 4), scalar_first=True)
    rb = Rotation.from_quat(qb[..., 3:7].reshape(-1, 4), scalar_first=True)
    out[..., 3:6] = (ra.inv() * rb).as_rotvec().reshape(shape[:-1] + (3,))
    out[..., 6:] = qb[..., 7:] - qa[..., 7:]
    return out


def _inverse(m: mujoco.MjModel, d: mujoco.MjData, qpos, qvel, qacc):
    """The ZMP (T, 2) and the floor's vertical force over the weight (T,) of the states (qpos, qvel, qacc)."""
    root = int(m.jnt_bodyid[0])
    weight = float(m.body_subtreemass[root]) * GRAVITY
    T = len(qpos)
    zmp_, load = np.full((T, 2), np.nan), np.zeros(T)
    for t in range(T):
        d.qpos[:] = qpos[t]
        d.qvel[:] = qvel[t]
        d.qacc[:] = qacc[t]
        mujoco.mj_inverse(m, d)
        f = d.qfrc_inverse[:6]
        force = f[:3]
        moment = d.xmat[root].reshape(3, 3) @ f[3:6]  # about the free joint's origin, world frame
        o = d.xpos[root]
        load[t] = force[2] / weight
        if force[2] > 1e-9:
            zmp_[t] = (o[0] - (moment[1] + o[2] * force[0]) / force[2], o[1] + (moment[0] - o[2] * force[1]) / force[2])
    return zmp_, load


def zmp_at(model: mujoco.MjModel, centre: np.ndarray, near: np.ndarray, pos: np.ndarray, fps: float):
    """The ZMP (T, 2) and the floor's vertical force over the weight (T,) at the poses ``centre`` (T, nq), with the
    velocities and accelerations of a Savitzky-Golay fit (cubic) over the poses ``near`` (T, W, nq), of which
    ``centre`` is the ``pos``-th."""
    m = _inverse_model(model)
    c1, c2 = _coefficients(near.shape[1], fps)
    u = tangent(centre[:, None], near)
    return _inverse(m, mujoco.MjData(m), centre, np.einsum("tw,twv->tv", c1[pos], u), np.einsum("tw,twv->tv", c2[pos], u))


def zmp(model: mujoco.MjModel, qs: np.ndarray, fps: float, smooth_s: float):
    """The ZMP (T, 2) and the load share (T,) of a qpos sequence (zmp_at, fitted over ``smooth_s``)."""
    if len(qs) < 3:
        return np.full((len(qs), 2), np.nan), np.zeros(len(qs))
    idx, pos = windows(len(qs), fps, smooth_s)
    return zmp_at(model, qs, qs[idx], pos, fps)


class Dyn:
    """The ZMP of changed poses of a clip (``base``, the unchanged qpos): the human motion's velocities and
    accelerations fitted over ``smooth_s`` (as zmp), the changes' (the tangent from the unchanged pose) over
    ``change_s``.  Of the unchanged clip it is zmp's."""

    def __init__(self, model: mujoco.MjModel, base: np.ndarray, fps: float, smooth_s: float, change_s: float):
        self.m = _inverse_model(model)
        self.d = mujoco.MjData(self.m)
        self.base, self.fps = base, fps
        T = len(base)
        idx, pos = windows(T, fps, smooth_s)
        c1, c2 = _coefficients(idx.shape[1], fps)
        u = tangent(base[:, None], base[idx])
        self.v, self.a = np.einsum("tw,twv->tv", c1[pos], u), np.einsum("tw,twv->tv", c2[pos], u)
        self.idx, self.pos = windows(T, fps, change_s)
        self.c1, self.c2 = _coefficients(self.idx.shape[1], fps)

    def derivatives(self, q: np.ndarray):
        u = tangent(self.base, q)[self.idx]
        return (self.v + np.einsum("tw,twv->tv", self.c1[self.pos], u),
                self.a + np.einsum("tw,twv->tv", self.c2[self.pos], u))

    def zmp(self, q: np.ndarray):
        """The ZMP (T, 2) and the load share (T,) of the changed poses ``q``."""
        if len(q) < 3:
            return np.full((len(q), 2), np.nan), np.zeros(len(q))
        return _inverse(self.m, self.d, q, *self.derivatives(q))

    def response(self, q: np.ndarray, dq: np.ndarray, step: float):
        """J (T, W, 2): the ZMP at t against the change ``dq`` (T, nv, a tangent per unit) of each frame of its
        changes' fit (self.idx), by a change of ``step`` units."""
        T = len(q)
        v, a = self.derivatives(q)
        z0, _ = _inverse(self.m, self.d, q, v, a)
        J = np.zeros((T, self.idx.shape[1], 2))
        rows = np.arange(T)
        for w in range(self.idx.shape[1]):
            other = dq[self.idx[:, w]] * step
            qm = q.copy()
            here = self.idx[:, w] == rows
            for t in np.flatnonzero(here & np.any(other != 0.0, axis=1)):
                mujoco.mj_integratePos(self.m, qm[t], other[t], 1.0)
            vm = v + self.c1[self.pos, w][:, None] * other
            am = a + self.c2[self.pos, w][:, None] * other
            on = np.flatnonzero(np.any(other != 0.0, axis=1))
            if len(on):
                z, _ = _inverse(self.m, self.d, qm[on], vm[on], am[on])
                J[on, w] = np.nan_to_num((z - z0[on]) / step)
        return J


# ---------------------------------------------------------------------------------------------------------
# Contact areas and the parts on the floor.

_BOX_SIGNS = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
_BOX_EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8) if np.sum(_BOX_SIGNS[i] != _BOX_SIGNS[j]) == 1]
_RING = 12


def _shape(g: dict):
    """Local vertices and edges of a training collision geom (a cylinder as a 12-sided prism)."""
    size = np.asarray(g["size"])
    if g["type"] == "box":
        return _BOX_SIGNS * size, _BOX_EDGES
    a = np.linspace(0.0, 2.0 * np.pi, _RING, endpoint=False)
    ring = np.stack([size[0] * np.cos(a), size[0] * np.sin(a)], 1)
    v = np.concatenate([np.c_[ring, np.full(_RING, -size[1])], np.c_[ring, np.full(_RING, size[1])]])
    edges = [(i, (i + 1) % _RING) for i in range(_RING)] + [(_RING + i, _RING + (i + 1) % _RING) for i in range(_RING)]
    edges += [(i, _RING + i) for i in range(_RING)]
    return v, edges


def _side(model: mujoco.MjModel, b: int):
    while b > 0:
        for j in range(model.body_jntnum[b]):
            n = model.joint(model.body_jntadr[b] + j).name
            if n.startswith(("left_", "right_")):
                return n.split("_")[0]
        b = model.body_parentid[b]
    return None


def _part(model: mujoco.MjModel, body: str) -> str:
    """The part a collision body belongs to: <side>_foot, <side>_leg (knee and shin), <side>_forearm (hand and
    elbow), <side>_upper_arm (shoulder) or trunk (pelvis, hips, head)."""
    side = _side(model, model.body(body).id)
    if body.startswith("foot"):
        return f"{side}_foot"
    if body.startswith(("tibia", "femur", "ankle_block", "hip_block")):
        return f"{side}_leg"
    if body.startswith("radius"):
        return f"{side}_forearm"
    if body.startswith(("humerus", "shoulder")):
        return f"{side}_upper_arm"
    return "trunk"


PARTS = ("left_foot", "right_foot", "left_leg", "right_leg", "left_forearm", "right_forearm",
         "left_upper_arm", "right_upper_arm", "trunk")
FEET = [PARTS.index("left_foot"), PARTS.index("right_foot")]
# The human roles that tell when a part is on the floor (skeleton.FORMATS roles).
PART_ROLES = {
    "left_foot": ("L.ankle", "L.toe"), "right_foot": ("R.ankle", "R.toe"),
    "left_leg": ("L.knee",), "right_leg": ("R.knee",),
    "left_forearm": ("L.elbow", "L.wrist"), "right_forearm": ("R.elbow", "R.wrist"),
    "left_upper_arm": ("L.shoulder",), "right_upper_arm": ("R.shoulder",),
    "trunk": ("pelvis", "spine", "neck"),
}


class Shapes:
    """The training collision shapes by part, posed from an IK model's bodies."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        geoms = training_collision_geoms()
        self.part_of = []
        verts, edges, body, offset = [], [], [], 0
        for g in geoms:
            v, e = _shape(g)
            rot = np.empty(9)
            mujoco.mju_quat2Mat(rot, np.asarray(g["quat"], float))
            verts.append(np.asarray(g["pos"]) + v @ rot.reshape(3, 3).T)
            edges += [(offset + i, offset + j) for i, j in e]
            body += [model.body(g["body"]).id] * len(v)
            self.part_of += [PARTS.index(_part(model, g["body"]))] * len(v)
            offset += len(v)
        self.local = np.concatenate(verts)
        self.body = np.asarray(body)
        self.edges = np.asarray(edges)
        self.part_of = np.asarray(self.part_of)

    def vertices(self, data: mujoco.MjData) -> np.ndarray:
        """(V, 3) world vertices; mj_kinematics has run."""
        xm = data.xmat[self.body].reshape(-1, 3, 3)
        return data.xpos[self.body] + np.einsum("vij,vj->vi", xm, self.local)

    def lowest(self, verts: np.ndarray) -> np.ndarray:
        """(P,) each part's lowest point."""
        out = np.full(len(PARTS), np.inf)
        np.minimum.at(out, self.part_of, verts[:, 2])
        return out

    def area(self, verts: np.ndarray, parts, band: float):
        """(N, 2) points whose hull is the contact area of ``parts`` and the part of each (N,): per part, its
        shapes cut at ``band`` above its lowest point (the vertices below and the edges' crossings)."""
        low = self.lowest(verts)
        cut = low[self.part_of] + band
        keep = np.isin(self.part_of, list(parts))
        below = keep & (verts[:, 2] <= cut)
        a, b = self.edges[:, 0], self.edges[:, 1]
        za, zb, c = verts[a, 2], verts[b, 2], cut[a]
        cross = keep[a] & ((za - c) * (zb - c) < 0.0)
        s = ((c - za) / np.where(cross, zb - za, 1.0))[cross]
        pts = np.concatenate([verts[below, :2], verts[a[cross], :2] + s[:, None] * (verts[b[cross], :2] - verts[a[cross], :2])])
        return pts, np.concatenate([self.part_of[below], self.part_of[a[cross]]])


@dataclass
class Poses:
    """Per frame of a qpos sequence: the collision shapes' vertices, the trunk's tilt (degrees) and heading (rad),
    and the hip and ankle centres (T, 2 (left, right), 3)."""

    verts: list
    tilt_deg: np.ndarray
    yaw: np.ndarray
    hip: np.ndarray
    ankle: np.ndarray


def poses(model: mujoco.MjModel, qs: np.ndarray, shapes: Shapes) -> Poses:
    d = mujoco.MjData(model)
    root, trunk = int(model.jnt_bodyid[0]), model.body("trunk").id
    sites = [[model.site(f"{s}_{p}_c").id for s in ("left", "right")] for p in ("hip", "ankle")]
    T = len(qs)
    verts, tilt, yaw = [], np.zeros(T), np.zeros(T)
    hip, ankle = np.zeros((T, 2, 3)), np.zeros((T, 2, 3))
    for t in range(T):
        d.qpos[:] = qs[t]
        mujoco.mj_kinematics(model, d)
        verts.append(shapes.vertices(d))
        tilt[t] = np.degrees(np.arccos(np.clip(d.xmat[root, 8], -1.0, 1.0)))
        # The heading from both horizontal axes (the trunk's turn about the vertical): defined at any tilt but
        # upside down, where the forward axis alone flips about once it points near the vertical (a deep bow).
        x = d.xmat[trunk]
        yaw[t] = np.arctan2(x[3] - x[1], x[0] + x[4])
        hip[t], ankle[t] = d.site_xpos[sites[0]], d.site_xpos[sites[1]]
    return Poses(verts, tilt, np.unwrap(yaw), hip, ankle)


def fill_runs(c: np.ndarray, value: bool, n: int) -> np.ndarray:
    """``c`` (T,) with its runs of ``not value`` of at most ``n`` frames between two of ``value`` set to ``value``."""
    c = c.copy()
    edges = np.flatnonzero(np.diff(np.r_[value, c == value, value].astype(int)))
    for a, b in zip(edges[::2], edges[1::2]):
        if 0 < a and b < len(c) and b - a <= n:
            c[a:b] = value
    return c


def human_contact(frames, ref, fmt, fps, floor, role_name, contact_radius) -> np.ndarray:
    """(T, P) the human's parts on the floor: the lowest of the part's joints less than CONTACT_HEIGHT leg lengths
    above its height on the floor (a foot's: as in the standing reference; the others': core.CONTACT_RADIUS) and
    the slowest moving less than CONTACT_SPEED leg lengths per second across the floor.  Gaps and contacts of one
    frame are filled and dropped."""
    leg = ref[fmt["L"]["hip"]][0][2] - ref[fmt["L"]["ankle"]][0][2]
    floor_ref = min(ref[fmt[s]["toe"]][0][2] for s in "LR")
    out = np.zeros((len(frames), len(PARTS)), bool)
    for p, part in enumerate(PARTS):
        heights, speeds = [], []
        for role in PART_ROLES[part]:
            name = role_name(fmt, role)
            base = role.split(".")[-1]
            z0 = ref[name][0][2] if base in ("ankle", "toe") else floor_ref + contact_radius[base] * leg
            pos = np.array([f[name][0] for f in frames])
            heights.append(pos[:, 2] - z0 - floor)
            v = np.r_[0.0, np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1)] * fps
            if len(v) > 1:
                v[0] = v[1]
            speeds.append(v)
        c = (np.min(heights, 0) < CONTACT_HEIGHT * leg) & (np.min(speeds, 0) < CONTACT_SPEED * leg)
        out[:, p] = fill_runs(fill_runs(c, True, 1), False, 1)
    return out


def robot_contact(P: Poses, human: np.ndarray, shapes: Shapes, fps: float) -> np.ndarray:
    """(T, P) Microban's parts on the floor in the poses ``P`` (module docstring)."""
    T = len(P.verts)
    low = np.array([shapes.lowest(v) for v in P.verts])
    centre = np.array([[v[shapes.part_of == p, :2].mean(0) for p in range(len(PARTS))] for v in P.verts])
    speed = np.linalg.norm(np.gradient(centre, axis=0), axis=2) * fps if T > 1 else np.zeros((T, len(PARTS)))
    contact = (human & (low < ROBOT_CONTACT_M)) | ((low < ROBOT_TOUCH_M) & (speed < ROBOT_TOUCH_SPEED))
    for p in range(len(PARTS)):
        contact[:, p] = fill_runs(fill_runs(contact[:, p], True, GAP_FRAMES), False, 1)
    lying = (P.tilt_deg > LYING_TILT_DEG) & np.delete(contact, FEET, axis=1).any(1)
    contact[lying] = False
    return contact


# ---------------------------------------------------------------------------------------------------------
# Regions as half-planes n . z <= h.

_DIRS = np.stack([np.cos(a := np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)), np.sin(a)], 1)


def halfplanes(points: np.ndarray):
    """(A (K, 2) unit outward normals, h (K,)) of the convex hull of ``points`` (a point or a segment too)."""
    hull = convex_hull_2d(points)
    if len(hull) >= 3:
        e = np.roll(hull, -1, axis=0) - hull
        n = np.stack([e[:, 1], -e[:, 0]], 1) / np.linalg.norm(e, axis=1, keepdims=True)
        return n, np.einsum("ij,ij->i", n, hull)
    if len(hull) == 2:
        u = (hull[1] - hull[0]) / max(np.linalg.norm(hull[1] - hull[0]), 1e-12)
        n = np.array([[-u[1], u[0]], [u[1], -u[0]], u, -u])
    else:
        n = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    return n, np.max(hull @ n.T, axis=0)


def clip_polygon(A: np.ndarray, h: np.ndarray, size: float = 10.0) -> np.ndarray:
    """Vertices (CCW) of {z : A z <= h} (a box of +-size to start); empty (0, 2) if none."""
    poly = np.array([[-size, -size], [size, -size], [size, size], [-size, size]])
    for n, c in zip(A, h):
        if len(poly) == 0:
            break
        s = poly @ n - c
        out = []
        for i in range(len(poly)):
            j = (i + 1) % len(poly)
            if s[i] <= 0.0:
                out.append(poly[i])
            if (s[i] <= 0.0) != (s[j] <= 0.0):
                out.append(poly[i] + (s[i] / (s[i] - s[j])) * (poly[j] - poly[i]))
        poly = np.asarray(out).reshape(-1, 2)
    return poly


def inradius(A: np.ndarray, h: np.ndarray) -> float:
    """The largest r with {z : A z + r <= h} not empty (the hull's inradius)."""
    res = linprog([0.0, 0.0, -1.0], A_ub=np.c_[A, np.ones(len(A))], b_ub=h, bounds=[(None, None)] * 2 + [(0.0, None)],
                  method="highs")
    return float(res.x[2]) if res.status == 0 else 0.0


def shrink(A: np.ndarray, h: np.ndarray, margin: float):
    """The hull shrunk by ``margin``, or by its inradius where that is less (its deepest points)."""
    if len(clip_polygon(A, h - margin)) >= 3:
        return A, h - margin
    return A, h - min(margin, inradius(A, h))


def region_vertices(A: np.ndarray, h: np.ndarray) -> np.ndarray:
    """The vertices of a region (a hair wider when it is a line or a point, so that clipping keeps it)."""
    poly = clip_polygon(A, h)
    return poly if len(poly) >= 3 else clip_polygon(A, h + 1e-6)


def outside(z: np.ndarray, verts: np.ndarray) -> float:
    return distance_outside(np.asarray(z, float), verts) if len(verts) else float("inf")


def foot_weights(A: np.ndarray, points: np.ndarray, part: np.ndarray) -> np.ndarray:
    """(K, 2): how far each half-plane of the hull of ``points`` moves along its normal with the left and the
    right foot's offset: the share of each foot among the points that hold it (the farthest along its normal)."""
    s = A @ points.T  # (K, N)
    hold = s >= s.max(1, keepdims=True) - 1e-7
    feet = np.stack([part == p for p in FEET], 1).astype(float)  # (N, 2)
    return (hold @ feet) / np.maximum(hold.sum(1, keepdims=True), 1)


@dataclass
class Supports:
    """Per frame: the parts on the floor (T, P), the allowed area (half-planes, or None) and how its half-planes
    move with the feet's offsets (K, 2), S(t)'s vertices and the weight of the ZMP outside the allowed area (T,)."""

    contact: np.ndarray
    A: list
    h: list
    feet: list
    S: list
    weight: np.ndarray


def _region(P: Poses, t: int, parts, shapes: Shapes, spec: SupportSpec):
    """S of the parts ``parts`` at frame t: (A, h, the half-planes' foot weights)."""
    pts, part = shapes.area(P.verts[t], parts, spec.band_m)
    A, h = shrink(*halfplanes(pts), spec.margin_m)
    return A, h, foot_weights(A, pts, part), pts, part


def supports(P: Poses, contact: np.ndarray, shapes: Shapes, spec: SupportSpec, fps: float, plan: bool = False) -> Supports:
    """S(t) of every frame from the poses ``P`` and the parts on the floor ``contact``.  With ``plan``, the allowed
    area and the weights the least squares uses (the lift-offs and the parts that change in one frame; module
    docstring); without, the allowed area is S(t)."""
    T = len(P.verts)
    contact = contact.copy()
    weight = np.ones(T)
    if plan:
        # A part that leaves the floor as another lands: both on the floor in both frames.
        for t1 in range(1, T):
            before, after = contact[t1 - 1], contact[t1]
            if (before & ~after).any() and after.any() and not (before & after).any():
                contact[t1 - 1] |= after
                contact[t1] |= before
    A, h, feet, S = [None] * T, [None] * T, [None] * T, [np.zeros((0, 2))] * T
    for t in range(T):
        parts = np.flatnonzero(contact[t])
        if len(parts):
            A[t], h[t], feet[t], _, _ = _region(P, t, parts, shapes, spec)
            S[t] = region_vertices(A[t], h[t])
    if not plan:
        return Supports(contact, A, h, feet, S, weight)
    n_lift = max(1, round(spec.lift_s * fps))
    for t1 in range(1, T):
        before, after = contact[t1 - 1], contact[t1]
        keep = before & after
        if not (before & ~after).any() or not keep.any():
            continue
        last = t1 - 1  # the last frame with the part on the floor
        t0 = last
        while t0 > 0 and last - t0 + 1 < n_lift and (contact[t0 - 1] == before).all():
            t0 -= 1
        far = 0.0
        for t in range(t0, last + 1):
            An, hn, fn, pts, part = _region(P, t, np.flatnonzero(keep), shapes, spec)
            rem = region_vertices(An, hn)
            if t == t0:
                far = max(outside(p, rem) for p in S[t]) if len(S[t]) else 0.0
            r = far * (last - t) / max(last - t0, 1)
            A[t] = np.r_[A[t], An, _DIRS]
            h[t] = np.r_[h[t], hn + r, np.max(rem @ _DIRS.T, axis=0) + r]
            feet[t] = np.r_[feet[t], fn, foot_weights(_DIRS, pts, part)]
        weight[[last, t1]] = spec.w_lift
    return Supports(contact, A, h, feet, S, weight)


def lift_offs(contact: np.ndarray) -> np.ndarray:
    """The frames (first off the floor) where a part leaves the floor and another stays."""
    before, after = contact[:-1], contact[1:]
    return np.flatnonzero((before & ~after).any(1) & (before & after).any(1)) + 1


def classes(contact: np.ndarray) -> dict:
    """Frame masks by the parts on the floor: one foot, both feet, a part other than the feet, none."""
    feet = contact[:, FEET].sum(1)
    other = np.delete(contact, FEET, axis=1).any(1)
    return {"single": (feet == 1) & ~other, "double": (feet == 2) & ~other, "other": other, "none": ~contact.any(1)}


def stats(z, load, sup: Supports, edge: int = 0):
    """Distances (T,) of the ZMP outside S(t) (nan without contact, without load and in the first and last ``edge``
    frames, where the derivatives are fitted from one side)."""
    out = np.full(len(z), np.nan)
    for t in range(edge, len(z) - edge):
        if sup.A[t] is not None and load[t] >= MIN_LOAD and np.isfinite(z[t]).all():
            out[t] = outside(z[t], sup.S[t])
    return out


def evaluate(model, qs, human, fps: float, spec: SupportSpec, shapes: Shapes | None = None, dyn: Dyn | None = None):
    """(ZMP, load, the parts on the floor, the distances outside S(t); nan in the clip's first and last half window
    of the fit) of a qpos sequence; also its Poses.  The ZMP is zmp's over ``smooth_s``, or ``dyn``'s."""
    shapes = shapes or Shapes(model)
    P = poses(model, qs, shapes)
    contact = robot_contact(P, human, shapes, fps)
    z, load = zmp(model, qs, fps, spec.smooth_s) if dyn is None else dyn.zmp(qs)
    edge = windows(len(qs), fps, spec.smooth_s)[0].shape[1] // 2
    return (z, load, contact, stats(z, load, supports(P, contact, shapes, spec, fps), edge)), P


def zmp_summary(dist, contact) -> dict:
    """Per class of frames (classes) and at the lift-offs: the frames with a ZMP, the share inside S(t) and the
    distance outside (p95, max; mm)."""
    def one(d):
        d = d[np.isfinite(d)]
        if not len(d):
            return {"frames_with_zmp": 0}
        return {"frames_with_zmp": len(d), "inside_share": round(float((d <= INSIDE_M).mean()), 4),
                "outside_mm_p95_max": [round(1000 * float(np.percentile(d, 95)), 1), round(1000 * float(d.max()), 1)]}
    out = {k: one(dist[m]) for k, m in classes(contact).items() if k != "none"}
    out["lift_off"] = one(dist[lift_offs(contact)])
    return out


def stage_reached(dists: list) -> np.ndarray:
    """(T,) per frame the stage (0: unchanged, 1-4: STAGES) from which its ZMP stayed in S(t) to the end, 5 where
    it is outside at the end, -1 without a ZMP to judge at the end.  ``dists``: the distances outside after each
    stage, the unchanged clip's first."""
    d = np.array(dists)
    inside = d <= INSIDE_M
    out = np.full(d.shape[1], 5, np.int8)
    stay = np.ones(d.shape[1], bool)
    for s in range(len(d) - 1, -1, -1):
        stay &= inside[s]
        out[stay] = s
    out[~np.isfinite(d[-1])] = -1
    return out


def stage_summary(stage: np.ndarray, contact: np.ndarray) -> dict:
    """Per class of frames (classes, at the end): the share of the frames with a ZMP whose ZMP came into S(t) at
    each stage (unchanged, STAGES) or stayed outside."""
    out = {}
    for k, m in classes(contact).items():
        s = stage[m & (stage >= 0)]
        if k != "none" and len(s):
            out[k] = {"frames": int(len(s)), **{name: round(float((s == i).mean()), 4)
                                                for i, name in enumerate(("unchanged",) + STAGES + ("outside",))}}
    return out


# ---------------------------------------------------------------------------------------------------------
# The changes.


def heading_axes(yaw: np.ndarray):
    """(T, 2) each: the heading's left and forward unit vectors (world)."""
    return np.stack([-np.sin(yaw), np.cos(yaw)], 1), np.stack([np.cos(yaw), np.sin(yaw)], 1)


@dataclass
class Adjust:
    """One frame's support changes for the IK (core.MicrobanRetargeting._adjusted), world: the trunk's shift (2,)
    and tilt (a rotation vector, 3), each foot's rotation (2, 3; left, right) and offset on the floor (2, 2)."""

    shift: np.ndarray
    tilt: np.ndarray
    ankle: np.ndarray
    foot: np.ndarray


@dataclass
class Changes:
    """A clip's support changes: per frame the COLUMNS (T, 6: the shift lateral and forward in the trunk's heading
    (m), the tilt's roll about the heading's forward axis and pitch about its left axis (rad), each foot's roll
    about the heading's forward axis (rad)), the feet's offsets on the floor (T, 2, 2; world, m) and the heading
    (T,)."""

    x: np.ndarray
    foot: np.ndarray
    yaw: np.ndarray

    @staticmethod
    def none(T: int) -> "Changes":
        return Changes(np.zeros((T, len(COLUMNS))), np.zeros((T, 2, 2)), np.zeros(T))

    def world(self):
        """The shift (T, 2), the tilt (T, 3) and the feet's rotations (T, 2, 3), world."""
        left, fwd = heading_axes(self.yaw)
        f3, l3 = np.c_[fwd, np.zeros(len(fwd))], np.c_[left, np.zeros(len(left))]
        x = self.x
        return (x[:, :1] * left + x[:, 1:2] * fwd, x[:, 2:3] * f3 + x[:, 3:4] * l3,
                np.stack([x[:, 4:5] * f3, x[:, 5:6] * f3], 1))

    def at(self, t: int) -> Adjust:
        if not hasattr(self, "_world"):
            self._world = self.world()
        shift, tilt, ankle = self._world
        return Adjust(shift[t], tilt[t], ankle[t], self.foot[t])

    def arrays(self) -> dict:
        """The npz's arrays: the shift (T, 2; world, m), the tilt (T, 2; roll, pitch, rad), the feet's roll (T, 2;
        rad) and their offsets on the floor (T, 2, 2; world, m)."""
        return {"support_shift": self.world()[0].astype(np.float32), "support_tilt": self.x[:, 2:4].astype(np.float32),
                "support_ankle_roll": self.x[:, 4:6].astype(np.float32), "support_foot": self.foot.astype(np.float32)}

    def summary(self) -> dict:
        """Per change: its size (p50, p95, max) and the share of the frames where it is over 0.5 mm or 0.5 deg."""
        sizes = {"shift_mm": 1000 * np.linalg.norm(self.x[:, :2], axis=1),
                 "tilt_deg": np.degrees(np.linalg.norm(self.x[:, 2:4], axis=1)),
                 "ankle_roll_deg": np.degrees(np.abs(self.x[:, 4:6]).max(1)),
                 "foot_mm": 1000 * np.linalg.norm(self.foot, axis=2).max(1)}
        return {k: {"p50_p95_max": [round(float(np.percentile(v, p)), 2) for p in (50, 95, 100)] if len(v) else None,
                    "share_used": round(float((v > 0.5).mean()), 4) if len(v) else 0.0} for k, v in sizes.items()}


def bspline(T: int, step: int) -> sp.csr_matrix:
    """(T, K) a uniform cubic B-spline's basis at the frames, knots ``step`` frames apart."""
    step = max(1, step)
    K = max(-(-(T - 1) // step), 1) + 3
    u = np.arange(T) / step
    i = np.minimum(np.floor(u).astype(int), K - 4)
    s = u - i
    w = np.stack([(1 - s) ** 3, 3 * s**3 - 6 * s**2 + 4, -3 * s**3 + 3 * s**2 + 3 * s + 1, s**3], 1) / 6.0
    rows = np.repeat(np.arange(T), 4)
    return sp.csr_matrix((w.reshape(-1), (rows, (i[:, None] + np.arange(4)).reshape(-1))), shape=(T, K))


def _diff(T: int, order: int) -> sp.csr_matrix:
    D = sp.eye(T, format="csr")
    for _ in range(order):
        D = (D[1:] - D[:-1]).tocsr()
    return D


def _runs(mask: np.ndarray):
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return list(zip(edges[::2], edges[1::2]))


def sole_points(model: mujoco.MjModel, shapes: "Shapes") -> list:
    """Per foot (left, right): its training collision shapes' vertices (V, 3) in its ankle centre's frame (the
    site ``<side>_ankle_c``, the frame of the IK's foot targets)."""
    out = []
    for side, part in zip(("left", "right"), FEET):
        site = model.site(f"{side}_ankle_c").id
        on = shapes.part_of == part
        assert (shapes.body[on] == model.site_bodyid[site]).all()
        rot = np.empty(9)
        mujoco.mju_quat2Mat(rot, model.site_quat[site])
        out.append((shapes.local[on] - model.site_pos[site]) @ rot.reshape(3, 3))
    return out


ROLL_PIVOT_DEG = 2.0  # a held foot pitched this much rolls on its toe's (heel's) edge, less on a point between
# A held foot's heading is held within about this of the human's (tanh): turning while it stands, the human's foot
# pivots 17 degrees over a run (median; 49 at 90 %, LAFAN1 walk3), and a heading held there twists the leg.
HOLD_TURN_DEG = 10.0


def foot_lock(contact: np.ndarray, pos: np.ndarray, quat: np.ndarray, soles: list):
    """The feet held where they stand, rolling as the human's: each run on the floor (``contact``) of a foot holds
    its sole where it lies flat (its ankle at the median over the run's flattest frames, its heading at the run's
    median within about HOLD_TURN_DEG of the human's) and lets it roll as the human's foot pitches: heel up, about
    its toe's edge, toe up, about its heel's (the edge's place where the flat sole has it; the human's ankle moves
    5-35 mm over a run while the foot pitches 40-80 degrees, and a held ankle stretches the leg instead).  Between runs the hold blends (smoothstep) from the
    one before to the one after, so that the target does not jump where the foot lands or leaves.  ``pos`` (T, 2,
    3) and ``quat`` (T, 2, 4; wxyz) are the feet's (left, right) targets, ``soles`` the sole points in their frames
    (sole_points).  Returns per frame and foot the horizontal offset of the target (T, 2, 2), its turn about the
    vertical (T, 2) and 1 where the foot stands (T, 2) (core.MicrobanRetargeting.lock)."""
    T = len(contact)
    offset, turn, stand = np.zeros((T, 2, 2)), np.zeros((T, 2)), np.zeros((T, 2))
    for i, p in enumerate(FEET):
        runs = _runs(contact[:, p])
        if not runs:
            continue
        bottom = soles[i][soles[i][:, 2] <= soles[i][:, 2].min() + 0.001]
        toe, heel = bottom[:, 0].max(), bottom[:, 0].min()
        middle = bottom.mean(0)
        rot = Rotation.from_quat(quat[:, i], scalar_first=True)
        m = rot.as_matrix()
        yaw = np.unwrap(np.arctan2(m[:, 1, 0], m[:, 0, 0]))
        pitch = np.degrees(-np.arcsin(np.clip(m[:, 2, 0], -1.0, 1.0)))  # heel up > 0
        hold = np.zeros((T, 3))
        cap = math.radians(HOLD_TURN_DEG)
        for a, b in runs:
            d_yaw = cap * np.tanh((np.median(yaw[a:b]) - yaw[a:b]) / cap)
            flat = np.abs(pitch[a:b]) <= np.percentile(np.abs(pitch[a:b]), 30)
            ankle0 = np.median(pos[a:b, i, :2][flat], 0)
            w = np.clip(pitch[a:b] / ROLL_PIVOT_DEG, -1.0, 1.0)
            pivot = np.tile(middle, (b - a, 1))
            pivot[:, 0] += np.where(w > 0, w * (toe - middle[0]), -w * (heel - middle[0]))
            about_z = lambda angle: Rotation.from_rotvec(np.c_[np.zeros((len(angle), 2)), angle])  # noqa: E731
            held = about_z(d_yaw) * rot[a:b]  # the target turned to the held heading
            flat_rot = about_z(yaw[a:b] + d_yaw)
            ankle = ankle0 + (flat_rot.apply(pivot) - held.apply(pivot))[:, :2]
            hold[a:b, :2] = ankle - pos[a:b, i, :2]
            hold[a:b, 2] = d_yaw
            stand[a:b, i] = 1.0
        hold[:runs[0][0]] = hold[runs[0][0]]
        hold[runs[-1][1]:] = hold[runs[-1][1] - 1]
        for (_, b0), (a1, _) in zip(runs, runs[1:]):
            u = np.arange(1, a1 - b0 + 1) / (a1 - b0 + 1)
            s = (u * u * (3.0 - 2.0 * u))[:, None]
            hold[b0:a1] = (1.0 - s) * hold[b0 - 1] + s * hold[a1]
        offset[:, i], turn[:, i] = hold[:, :2], hold[:, 2]
    return offset, turn, stand


def foot_phases(contact: np.ndarray, zero: np.ndarray):
    """Per foot (left, right): its runs on the floor [(start, end)] and the weights (T, runs) that give each frame's
    offset: a run's offset while the foot is on the floor, blended (smoothstep) between runs while it is off, 0
    where the changes must be 0 (and blended to and from it)."""
    T = len(contact)
    out = []
    for p in FEET:
        runs = _runs(contact[:, p] & ~zero)
        anchors = sorted([(a, b, j) for j, (a, b) in enumerate(runs)] + [(a, b, None) for a, b in _runs(zero)])
        W = np.zeros((T, len(runs)))
        for a, b, j in anchors:
            if j is not None:
                W[a:b, j] = 1.0
        for (_, b0, j0), (a1, _, j1) in zip(anchors, anchors[1:]):
            if a1 > b0:
                u = np.arange(1, a1 - b0 + 1) / (a1 - b0 + 1)
                s = u * u * (3.0 - 2.0 * u)
                if j0 is not None:
                    W[b0:a1, j0] = 1.0 - s
                if j1 is not None:
                    W[b0:a1, j1] = s
        if anchors and anchors[0][2] is not None:
            W[:anchors[0][0], anchors[0][2]] = 1.0
        if anchors and anchors[-1][2] is not None:
            W[anchors[-1][1]:, anchors[-1][2]] = 1.0
        out.append((runs, sp.csr_matrix(W)))
    return out


def hip_rolls(model: mujoco.MjModel):
    """The hip roll joints (left, right): qpos addresses (2,) and ranges (2, 2)."""
    ids = [model.joint(f"{s}_hip_roll").id for s in ("left", "right")]
    return model.jnt_qposadr[ids], model.jnt_range[ids]


def leg_length(model: mujoco.MjModel) -> float:
    """The hip centre to ankle centre distance with the knee straight."""
    d = mujoco.MjData(model)
    mujoco.mj_kinematics(model, d)
    return float(np.linalg.norm(d.site_xpos[model.site("left_hip_c").id] - d.site_xpos[model.site("left_ankle_c").id]))


def leg_share(model: mujoco.MjModel) -> float:
    """The share of the weight that moves with a foot's offset: the shank and the foot, and half the thigh."""
    side = "left"
    shank = model.body_subtreemass[model.body(f"tibia__configuration_{side}").id]
    thigh = model.body_mass[model.body(f"femur__configuration_{side}").id]
    return float((shank + 0.5 * thigh) / model.body_subtreemass[int(model.jnt_bodyid[0])])


def stance(contact: np.ndarray, fps: float, stance_s: float) -> np.ndarray:
    """(T, 2) per foot (left, right): 1 on the floor, falling linearly to 0 over ``stance_s`` from the floor."""
    n = max(1, round(stance_s * fps))
    return np.stack([np.clip(1.0 - _distance(contact[:, p]) / n, 0.0, 1.0) for p in FEET], 1)


def _distance(mask: np.ndarray) -> np.ndarray:
    """(T,) frames to the nearest True (inf without one)."""
    T = len(mask)
    d = np.full(T, np.inf)
    last = -np.inf
    for t in range(T):
        last = t if mask[t] else last
        d[t] = t - last
    last = np.inf
    for t in range(T - 1, -1, -1):
        last = t if mask[t] else last
        d[t] = min(d[t], last - t)
    return d


# ---------------------------------------------------------------------------------------------------------
# The least squares.


class Layout:
    """The least squares' variables: the per-frame changes' B-spline coefficients (K x COLUMNS, interleaved) and
    the feet's runs' offsets (world x, y; the left foot's runs, then the right's), and the maps from them to the
    frames: XM (T C, N) to the per-frame changes, FM[foot] (2 T, N) to the foot's offsets."""

    def __init__(self, T: int, fps: float, spec: SupportSpec, phases):
        C = len(COLUMNS)
        self.T, self.C = T, C
        self.B = bspline(T, round(spec.knot_s * fps))
        self.K = self.B.shape[1]
        self.nx = self.K * C
        runs = [W.shape[1] for _, W in phases]
        self.foot_at = [self.nx, self.nx + 2 * runs[0]]
        self.N = self.nx + 2 * sum(runs)
        self.XM = sp.kron(self.B, sp.eye(C), format="csr")
        self.XM.resize((T * C, self.N))
        self.FM = []
        for (_, W), at in zip(phases, self.foot_at):
            W = W.tocoo()
            rows = np.r_[2 * W.row, 2 * W.row + 1]
            cols = np.r_[at + 2 * W.col, at + 2 * W.col + 1]
            self.FM.append(sp.csr_matrix((np.r_[W.data, W.data], (rows, cols)), shape=(2 * T, self.N)))

    def values(self, u: np.ndarray):
        """(T, C) the per-frame changes and (T, 2, 2) the feet's offsets of the variables ``u``."""
        return (self.XM @ u).reshape(self.T, self.C), np.stack([(fm @ u).reshape(self.T, 2) for fm in self.FM], 1)


def reach_rows(P: Poses, contact: np.ndarray, lay: Layout, yaw: np.ndarray, length: float, spec: SupportSpec):
    """The changes that keep each standing leg's hip within its reach of its ankle, as rows (n, N) and bounds:
    |hip - ankle| at most the straight length less ``reach_slack_m`` (or as far as the unchanged leg, if farther),
    with the hip ``max_drop_m`` lower (16 tangent half-planes of that circle, the circle widening as the tilt
    lowers the hip; a secant over 6 mm)."""
    T, C = lay.T, lay.C
    left, fwd = heading_axes(yaw)
    reach = max(length - spec.reach_slack_m, 0.0)
    far = (math.hypot(spec.max_lateral_m, spec.max_forward_m) + 0.06 * math.sin(math.radians(spec.max_tilt_deg))
           + spec.max_foot_m + 0.005)
    pivot = P.hip.mean(1)
    rx, cx, vx, rf, cf, vf, b = [], [], [], [[], []], [[], []], [[], []], []
    n = 0
    for t in range(T):
        f3, l3 = np.r_[fwd[t], 0.0], np.r_[left[t], 0.0]
        for side in (0, 1):
            if not contact[t, FEET[side]]:
                continue
            v = P.ankle[t, side] - P.hip[t, side]
            r_max = max(reach, float(np.linalg.norm(v)))

            def radius(height):
                return math.sqrt(max(r_max**2 - max(height - spec.max_drop_m, 0.0) ** 2, 0.0))

            r0 = radius(-v[2])
            slope = (radius(-v[2] - 0.006) - r0) / 0.006
            bound = r0 + _DIRS @ v[:2]
            use = np.flatnonzero(bound < far)
            arm = P.hip[t, side] - pivot[t]
            g = np.stack([np.cross(f3, arm), np.cross(l3, arm)], 1)  # (3, 2): the hip against roll and pitch
            for i in use:
                d = _DIRS[i]
                coef = [d @ left[t], d @ fwd[t], d @ g[:2, 0] + slope * g[2, 0], d @ g[:2, 1] + slope * g[2, 1]]
                rx += [n] * 4
                cx += [t * C + c for c in range(4)]
                vx += coef
                rf[side] += [n, n]
                cf[side] += [2 * t, 2 * t + 1]
                vf[side] += [-d[0], -d[1]]
                b.append(bound[i])
                n += 1
    R = sp.csr_matrix((vx, (rx, cx)), shape=(n, T * C)) @ lay.XM
    for side in (0, 1):
        R = R + sp.csr_matrix((vf[side], (rf[side], cf[side])), shape=(n, 2 * T)) @ lay.FM[side]
    return R.tocsr(), np.asarray(b, float)


def solve_rows(u_k: np.ndarray, free: np.ndarray, Q: sp.csr_matrix, R: sp.csr_matrix, b: np.ndarray, w: np.ndarray,
               huber: np.ndarray) -> np.ndarray:
    """The variables minimising u Q u + sum w hinge(R u - b) (hinge: squared up to ``huber``, linear beyond) over
    the ``free`` ones, the others held at ``u_k``; Newton steps on the active set (the linear part as a reweighted
    square), from ``u_k``."""
    fixed = u_k * ~free
    Rf = R[:, free]
    bf = b - R @ fixed
    keep = np.diff(Rf.indptr) > 0
    Rf, bf, w, huber = Rf[keep], bf[keep], w[keep], huber[keep]
    Qf = (Q[free][:, free] + 1e-6 * sp.eye(int(free.sum()))).tocsc()  # (a small ridge: the knots at the ends act on few frames)
    g = Q[free] @ fixed

    def cost(v):
        r = np.maximum(Rf @ v - bf, 0.0)
        far = r > huber
        return float(v @ (Qf @ v) + 2.0 * g @ v + w @ np.where(far, 2.0 * huber * r - huber**2, r * r))

    v = u_k[free].copy()
    f = cost(v)
    for _ in range(50):
        r = Rf @ v - bf
        act = r > 0.0
        Ra = Rf[act]
        we = w[act] * np.minimum(1.0, huber[act] / r[act])
        v_new = spsolve((Qf + Ra.T @ sp.diags(we) @ Ra).tocsc(), Ra.T @ (we * bf[act]) - g)
        if not np.isfinite(v_new).all():
            break
        step, f_new = 1.0, cost(v_new)
        while f_new > f + 1e-15 and step > 1e-3:
            step *= 0.5
            f_new = cost(v + step * (v_new - v))
        v_next = v + step * (v_new - v)
        done = np.abs(v_next - v).max() < 1e-7
        v, f = v_next, f_new
        if done:
            break
    out = u_k.copy()
    out[free] = v
    return out


@dataclass
class Linear:
    """The responses to the per-frame changes, by column: J (T, W, 2) of the ZMP at t to the change at the frames
    of its changes' fit (Dyn.idx), Jr (T, 2) of the hip rolls (left, right)."""

    J: dict
    Jr: dict


def respond(dyn: Dyn, solve_x, q_k: np.ndarray, X_k: np.ndarray, F_k: np.ndarray, columns, frames: np.ndarray,
            roll: np.ndarray, spec: SupportSpec, lin: Linear) -> None:
    """Find the responses (Linear) of ``columns`` at ``frames`` around the solution ``q_k``: the frames solved again
    with the change moved by ``probe_m`` or ``probe_rad`` (``solve_x(X, F, subset, base)``); the responses at
    the other frames are kept."""
    if not len(frames):
        return
    T = len(q_k)
    on = np.zeros(T, bool)
    on[frames] = True
    near = on[dyn.idx]  # (T, W): the responses to a probed frame
    for c in columns:
        # Towards 0 (where the clip is solved): beyond a change at its cap a leg may be out of reach or its hip
        # roll at its end, and the IK's answer there is not the response.
        size = (spec.probe_m if c < 2 else spec.probe_rad) * np.where(X_k[:, c] > 0.0, -1.0, 1.0)
        X = X_k.copy()
        X[frames, c] += size[frames]
        qp = solve_x(X, F_k, frames, q_k)
        dq = tangent(q_k, qp) / size[:, None]
        # An arm that takes the other branch in the probe says nothing about the change: its joints are kept.
        dq[:, 6:][np.abs(qp[:, 7:] - q_k[:, 7:]) > 0.5] = 0.0
        J, Jr = lin.J.setdefault(c, np.zeros((T, dyn.idx.shape[1], 2))), lin.Jr.setdefault(c, np.zeros((T, 2)))
        J[near] = dyn.response(q_k, dq, 0.01 * abs(size[0]))[near]
        Jr[on] = ((qp[:, roll] - q_k[:, roll]) / size[:, None])[on]


def _rows(lay: Layout, lin: Linear, idx: np.ndarray, z_k, roll_k, u_k, sup: Supports, frames, roll_on, roll_range,
          share: float, yaw: np.ndarray, spec: SupportSpec):
    """The linearised rows (R, b, w, huber) around (u_k, z_k, roll_k): the ZMP outside the allowed area at
    ``frames`` and the standing legs' hip rolls (``roll_on``; a hip roll already past its bound is kept from
    going farther).  ``idx``: the frames of each frame's changes' fit (Dyn.idx)."""
    T, C = lay.T, lay.C
    blocks, bounds, weights, hubers = [], [], [], []
    # The ZMP (2 T) against the variables: the per-frame changes through J, the feet's offsets through their legs.
    rows, cols, vals = [], [], []
    for c, J in lin.J.items():
        t, w, i = np.nonzero(J)
        rows.append(2 * t + i)
        cols.append(idx[t, w] * C + c)
        vals.append(J[t, w, i])
    JX = sp.csr_matrix((np.concatenate(vals) if vals else [], (np.concatenate(rows) if rows else [], np.concatenate(cols) if cols else [])),
                       shape=(2 * T, T * C))
    ZJ = (JX @ lay.XM + share * (lay.FM[0] + lay.FM[1])).tocsr()
    if len(frames):
        Ar, Ac, Av, h, fw, wt = [], [], [], [], [], []
        n = 0
        for t in frames:
            A = sup.A[t]
            k = len(A)
            Ar.append(np.repeat(np.arange(n, n + k), 2))
            Ac.append(np.tile([2 * t, 2 * t + 1], k))
            Av.append(A.reshape(-1))
            h.append(sup.h[t])
            fw.append(sup.feet[t])
            wt.append(np.full(k, spec.w_outside * sup.weight[t]))
            n += k
        Abig = sp.csr_matrix((np.concatenate(Av), (np.concatenate(Ar), np.concatenate(Ac))), shape=(n, 2 * T))
        fw = np.concatenate(fw)
        G = Abig @ ZJ - sp.diags(fw[:, 0]) @ Abig @ lay.FM[0] - sp.diags(fw[:, 1]) @ Abig @ lay.FM[1]
        blocks.append(G)
        bounds.append(np.concatenate(h) - spec.aim_m - Abig @ z_k.reshape(-1) + G @ u_k)
        weights.append(np.concatenate(wt))
        hubers.append(np.full(n, spec.huber_m))
    # The standing legs' hip rolls.
    left, fwd = heading_axes(yaw)
    lo, hi = roll_range
    for side in (0, 1):
        tt = np.flatnonzero(roll_on[:, side])
        if not len(tt) or not lin.Jr:
            continue
        rr, cc, vv = [], [], []
        for c, Jr in lin.Jr.items():
            rr.append(np.arange(len(tt)))
            cc.append(tt * C + c)
            vv.append(Jr[tt, side])
        GX = sp.csr_matrix((np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))), shape=(len(tt), T * C)) @ lay.XM
        # A foot's offset turns its hip roll as the opposite shift does.
        js = (lin.Jr.get(0, np.zeros((T, 2)))[tt, side][:, None] * left[tt] + lin.Jr.get(1, np.zeros((T, 2)))[tt, side][:, None] * fwd[tt])
        GF = sp.csr_matrix((-js.reshape(-1), (np.repeat(np.arange(len(tt)), 2), np.c_[2 * tt, 2 * tt + 1].reshape(-1))),
                           shape=(len(tt), 2 * T)) @ lay.FM[side]
        G = (GX + GF).tocsr()
        lin_k = roll_k[tt, side] - G @ u_k
        blocks += [G, -G]
        bounds += [np.maximum(hi[side], roll_k[tt, side]) - lin_k, lin_k - np.minimum(lo[side], roll_k[tt, side])]
        weights += [np.full(len(tt), HARD * spec.w_outside)] * 2
        hubers += [np.full(len(tt), 1e6)] * 2
    return blocks, bounds, weights, hubers


def plan(model, solve, qs, human, fps: float, spec: SupportSpec):
    """The support changes of a clip.  ``solve(adjust, stance, subset, base)`` maps changes (Changes) and the
    feet's weights on the floor (T, 2; stance) to the clip's qpos on the floor (of the frames ``subset``, the others
    as in the solution ``base``); ``qs`` is the unchanged clip's, ``human`` the human's contacts (T, P).  Returns the
    qpos, the changes (Changes), a summary (the ZMP in S(t) before, after each stage and after, the stage each
    frame came in at, the cost of each iteration), (ZMP, load, the parts on the floor, the distances outside S(t))
    before and after (after: the parts found again from the changed poses), and the stage each frame came into
    S(t) at (stage_reached)."""
    shapes = Shapes(model)
    T = len(qs)
    dyn = Dyn(model, qs, fps, spec.smooth_s, spec.change_s) if T >= 3 else None
    before, P0 = evaluate(model, qs, human, fps, spec, shapes)
    z, load, contact, dist0 = before
    summary = {"margin_m": spec.margin_m, "before": zmp_summary(dist0, contact), "cost_before": None, "iterations": []}
    zero = ~contact.any(1) | np.delete(contact, FEET, axis=1).any(1)
    n_fade = max(1, round(spec.fade_s * fps))
    fade = np.clip(1.0 - _distance(zero) / n_fade, 0.0, 1.0) ** 2
    edge = np.zeros(T, bool)
    half = windows(T, fps, spec.smooth_s)[0].shape[1] // 2
    edge[:half], edge[T - half:] = True, True
    changes = Changes(np.zeros((T, len(COLUMNS))), np.zeros((T, 2, 2)), P0.yaw)
    if (zero | edge).all() or T < 3:
        summary["after"] = summary["before"]
        stage = stage_reached([dist0])
        return qs, changes, summary, before, before, stage
    feet = stance(contact, fps, spec.stance_s)
    yaw = P0.yaw
    phases = foot_phases(contact, zero)
    lay = Layout(T, fps, spec, phases)
    C = lay.C
    caps = spec.caps()
    w_size = np.array([spec.w_shift, spec.w_shift, spec.w_tilt, spec.w_tilt, spec.w_ankle, spec.w_ankle])
    w_acc = spec.w_accel * np.array([1.0, 1.0] + [spec.lever_m**2] * 4)
    share = leg_share(model)
    roll, rng = hip_rolls(model)
    m = math.radians(spec.joint_margin_deg)
    lo, hi = rng[:, 0] + m, rng[:, 1] - m
    roll_on = contact[:, FEET] & (qs[:, roll] > lo) & (qs[:, roll] < hi) & (fade < 1.0)[:, None]
    R_reach, b_reach = reach_rows(P0, contact, lay, yaw, leg_length(model), spec)
    D2 = sp.kron(_diff(T, 2), sp.eye(C), format="csr") @ lay.XM
    D1f = (_diff(T, 1) @ lay.XM[1::C]).tocsr()
    FF = (lay.FM[0].T @ lay.FM[0] + lay.FM[1].T @ lay.FM[1]).tocsr()
    acc_w = np.tile(w_acc, T - 2)
    # Held where the ZMP is not constrained (the clip's ends): the steps there weigh as a change where it must be 0
    # (free, the changes moved there to make the acceleration the constrained frames' differences see).
    D1 = sp.kron(_diff(T, 1), sp.eye(C), format="csr") @ lay.XM
    hold = np.repeat((edge[:-1] | edge[1:]).astype(float), C)
    Q_fixed = (D2.T @ sp.diags(acc_w * fps**4) @ D2 + spec.w_speed_forward * fps**2 * (D1f.T @ D1f) + spec.w_foot * FF
               + spec.w_zero * (D1.T @ sp.diags(hold) @ D1)).tocsr()

    def solve_x(X, F, subset=None, base=None):
        ch = Changes(X, F, yaw)
        return solve(ch, feet, subset, base)

    def assess(q, z_, load_):
        sup = supports(poses(model, q, shapes), contact, shapes, spec, fps, plan=True)
        viol = np.full(T, np.nan)
        for t in range(T):
            if sup.A[t] is not None and load_[t] >= MIN_LOAD and np.isfinite(z_[t]).all() and not (zero[t] or edge[t]):
                viol[t] = float(np.max(sup.A[t] @ z_[t] - sup.h[t]))
        on = np.isfinite(viol)
        cost = float(sum(sup.weight[t] * np.sum(np.maximum(sup.A[t] @ z_[t] - sup.h[t], 0.0) ** 2) for t in np.flatnonzero(on)))
        return sup, viol, cost

    lin = Linear({}, {})
    u = np.zeros(lay.N)
    X, F = lay.values(u)
    q_k, (z_k, load_k) = qs, (z, load)
    sup, viol, cost = assess(q_k, z_k, load_k)
    summary["cost_before"] = round(cost, 8)
    best = (cost, q_k, u.copy(), z_k, load_k)
    allow = np.ones((T, C))  # where each per-frame change may be used (1) or must be 0 (0), besides ``zero``
    dists, stages = [dist0], {}
    # The rounds: the first stage over the whole clip, then each stage (the first again) around the frames still
    # outside, with the responses found again there (the changes act from where they now are).
    for s, whole in ((1, True), (1, False), (2, False), (3, False), (4, False)):
        name = STAGES[s - 1]
        new = [c for c in range(C) if COLUMN_STAGE[c] == s and caps[c] > 0.0] if whole or s > 1 else []
        gate = np.isfinite(viol) & (viol > INSIDE_M)
        if whole:
            active = probing = np.ones(T, bool)
        else:
            to_gate = _distance(gate)
            active = to_gate <= n_fade + 2 * round(spec.knot_s * fps)
            probing = to_gate < n_fade  # where a new change may be used (allow > 0)
            allow[:, new] = np.clip(1.0 - to_gate / n_fade, 0.0, 1.0)[:, None]
        foot_free = np.zeros(lay.N - lay.nx, bool)
        if name == "foot" and spec.max_foot_m > 0.0:
            for side, (runs, _) in enumerate(phases):
                for k, (a, b) in enumerate(runs):
                    if gate[a:b].any():
                        at = lay.foot_at[side] - lay.nx + 2 * k
                        foot_free[at:at + 2] = True
        used = [c for c in range(C) if COLUMN_STAGE[c] <= s and caps[c] > 0.0]
        # The responses found (again) at the round's start: the first stage's around where it now is, a later
        # stage's of its own changes.
        probed = used if s == 1 else new
        if not gate.any() or not (used or foot_free.any()) or (s > 1 and not new and not foot_free.any()):
            if s == 1 and not whole:  # the first stage's result is the whole round's
                (_, _, cs, ds), _ = evaluate(model, q_k, human, fps, spec, shapes, dyn)
                dists.append(ds)
                stages[name] = zmp_summary(ds, cs)
            elif not whole:
                dists.append(dists[-1])
                stages[name] = {"skipped": True}
            continue
        probe = np.flatnonzero(probing & ~zero)
        coef_on = np.asarray(lay.B[np.flatnonzero(active)].sum(0)).ravel() > 0.0
        free = np.zeros(lay.N, bool)
        for c in used:
            free[np.flatnonzero(coef_on) * C + c] = True
        free[lay.nx:] = foot_free
        # The frames the free variables move (a coefficient acts two knots past the active frames, an offset over
        # its run and the blends next to it): their ZMP, limits and caps are in the least squares.
        reach_ = (np.asarray(lay.B[:, coef_on].sum(1)).ravel() > 0.0) if used else np.zeros(T, bool)
        for fm in lay.FM:
            reach_ |= np.asarray(abs(fm[:, lay.nx:][:, foot_free]).sum(1)).ravel().reshape(T, 2).any(1)
        active = active | reach_
        pen = np.maximum(fade[:, None], (1.0 - allow) ** 2)
        Q = (lay.XM.T @ sp.diags((w_size[None] + spec.w_zero * pen).reshape(-1)) @ lay.XM + Q_fixed).tocsr()
        # The caps (near-hard), on the free variables' frames.
        cap_rows, cap_b = [], []
        tt = np.flatnonzero(active)
        for c in used:
            Xc = lay.XM[tt * C + c]
            cap_rows += [Xc, -Xc]
            cap_b += [np.full(len(tt), caps[c])] * 2
        n_foot = int(foot_free.sum()) // 2
        if n_foot:
            at = lay.nx + np.flatnonzero(foot_free)[::2]
            Rf = sp.csr_matrix((np.r_[np.repeat(_DIRS[::2, 0], n_foot), np.repeat(_DIRS[::2, 1], n_foot)],
                                (np.r_[np.arange(8 * n_foot), np.arange(8 * n_foot)], np.r_[np.tile(at, 8), np.tile(at + 1, 8)])),
                               shape=(8 * n_foot, lay.N))
            cap_rows.append(Rf)
            cap_b.append(np.full(8 * n_foot, spec.max_foot_m))
        near = _distance(active) <= dyn.idx.shape[1] // 2
        respond(dyn, solve_x, q_k, X, F, probed, probe, roll, spec, lin)
        costs = []
        for _ in range(spec.iterations if whole else spec.stage_iterations):
            frames = np.flatnonzero(np.isfinite(viol) & near)
            blocks, bounds, weights, hubers = _rows(lay, lin, dyn.idx, np.nan_to_num(z_k), q_k[:, roll], u, sup, frames,
                                                    roll_on, (lo, hi), share, yaw, spec)
            n_lin = sum(len(x) for x in bounds)
            blocks += [R_reach] + cap_rows
            bounds += [b_reach] + cap_b
            R = sp.vstack(blocks).tocsr()
            b = np.concatenate(bounds)
            b[n_lin:] = np.maximum(b[n_lin:], R[n_lin:] @ u)  # a limit already passed is kept from getting worse
            w = np.r_[np.concatenate(weights) if weights else [], np.full(len(b) - n_lin, HARD * spec.w_outside)]
            h = np.r_[np.concatenate(hubers) if hubers else [], np.full(len(b) - n_lin, 1e6)]
            u_new = solve_rows(u, free, Q, R, b, w, h)
            X_new, F_new = lay.values(u_new)
            # A frame whose changes moved less than 0.01 mm (or 0.01 mrad) keeps its solution.
            moved = (np.abs(X_new - X).max(1) > 1e-5) | (np.abs(F_new - F).reshape(T, -1).max(1) > 1e-5)
            if moved.any():
                q_k = solve_x(X_new, F_new, np.flatnonzero(moved), q_k)
            u, X, F = u_new, X_new, F_new
            z_k, load_k = dyn.zmp(q_k)
            sup, viol, cost = assess(q_k, z_k, load_k)
            costs.append(round(cost, 8))
            if cost < best[0]:
                best = (cost, q_k, u.copy(), z_k, load_k)
        summary["iterations"].append(costs)
        _, q_k, u, z_k, load_k = best
        X, F = lay.values(u)
        sup, viol, _ = assess(q_k, z_k, load_k)
        if whole:
            continue
        (_, _, cs, ds), _ = evaluate(model, q_k, human, fps, spec, shapes, dyn)
        dists.append(ds)
        stages[name] = zmp_summary(ds, cs)
    _, q_b, u_b, z_b, load_b = best
    X_b, F_b = lay.values(u_b)
    (z_a, load_a, contact_a, dist_a), _ = evaluate(model, q_b, human, fps, spec, shapes, dyn)
    after = (z_a, load_a, contact_a, dist_a)
    summary["stages"] = stages
    summary["after"] = zmp_summary(dist_a, contact_a)
    stage = stage_reached(dists)
    summary["stage_reached"] = stage_summary(stage, contact_a)
    return q_b, Changes(X_b, F_b, yaw), summary, before, after, stage
