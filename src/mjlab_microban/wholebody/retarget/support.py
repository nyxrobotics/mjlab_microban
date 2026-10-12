"""The support shift of the retargeting (``core.retarget_frames(..., support_spec=SupportSpec())``; off by default).

Microban's feet are far apart for its size (HOME's sole centres 93 mm apart, 0.75 leg lengths; a human's
0.2-0.25), so the human's motion, scaled, puts Microban's weight outside the foot it stands on.  The support
shift moves the trunk horizontally (forward and sideways; its orientation is kept, and its height but where a
standing leg would not reach its foot, by at most ``max_drop_m``), the arms with it (their pose relative to the
trunk is kept) and the feet where they were (core.MicrobanRetargeting._shifted), so that Microban's zero-moment
point stays in its support:

* The zero-moment point (ZMP) is Microban's, from its whole-body dynamics: the retargeted qpos sequence through
  mj_inverse on robot.xml (the masses and inertias the training simulates; the arms' and the trunk's angular
  momentum included), without contacts.  The free joint's force and torque are then the wrench the floor has to
  give, and the ZMP is the point of the floor (z = 0) about which its horizontal moment is zero.  Frames whose
  vertical force is below MIN_LOAD of the weight (falling, flight) have none.  The velocities and accelerations
  are those of a cubic fitted over ``smooth_s`` (0.3 s): with central differences the capture's frame-to-frame
  noise moves the ZMP by 7-68 mm per frame (median; the PICO walk the most).
* The parts on the floor (robot_contact): a part is on the floor when the human's is (as low as when standing for
  a foot, as low as when lying on the floor for the others (core.CONTACT_RADIUS), and slow across the floor) and
  Microban's is within ROBOT_CONTACT_M of the floor (the training side's support height), or when Microban's
  touches the floor (ROBOT_TOUCH_M) and is slow.  Gaps of up to GAP_FRAMES and contacts of one frame are filled
  and dropped.  Lying and crawling (the trunk tilted past LYING_TILT_DEG with a part other than the feet on the
  floor) have no part on the floor.  The shifted clip's parts are found again from its own poses by the same rule.
* The support S(t) is the convex hull of the floor contact areas of the parts on the floor, shrunk inward by
  ``margin_m`` (10 mm by default; the data command's ``--support-margin-mm``).  A part's contact area is the part
  of its training collision shapes within ``band_m`` (10 mm) of its lowest point, seen from above.  Where the hull
  is too narrow to shrink by ``margin_m``, S(t) is its deepest points (its middle line or point).
* The shift d(t) is 0 where nothing is on the floor (flight, lying, crawling) and where a part other than the feet
  is (kneeling, a hand down: the IK's knee and hand targets are too weak to keep them on the floor under a moving
  trunk); it goes to 0 over ``fade_s`` before and comes back over ``fade_s`` after.
* Before a part leaves the floor (lift-off), the ZMP moves into what remains: over the last ``lift_s`` of the
  contact phase (at most its length) the allowed area is S(t) within a distance of the remaining support that
  falls linearly to 0 at the last frame on the floor, and that frame and the first one off the floor weigh
  ``w_lift`` times the others.  A part that leaves the floor in the frame another lands (no frame with both) is
  counted on the floor in both frames.
* The shift (in the heading frame, lateral and forward) is a cubic B-spline with knots ``knot_s`` apart, solved
  over the whole clip by least squares.  The B-spline and the acceleration term keep it smooth enough that the
  0.3 s fit, the 0.5 s fit and central differences see the same ZMP from it (with knots 0.1 s apart the ZMP the
  optimisation saw from the shift's 3-5 Hz part was not there with central differences).  The terms: the ZMP
  outside the allowed area (aimed ``aim_m`` inside; squared up to ``huber_m``, linear beyond, so that a ZMP far
  out that no shift brings in, a push-off before a jump, does not swing the shift), the shift's size (so that it
  stays near 0 where the ZMP is already in), its acceleration and its forward speed (the trunk's progress), the
  shift where it must be 0, and, as near-hard limits, a standing leg stretched beyond its straight length (less
  ``reach_slack_m``; the trunk may go down by ``max_drop_m`` for it), a standing leg's hip roll within
  ``joint_margin_deg`` of the end of its range (at the end the IK lifts the foot instead; one already there
  unshifted is left to the IK), and the shift beyond ``max_lateral_m`` / ``max_forward_m``.  The response of the
  ZMP and of the hip rolls to the shift is found numerically: the clip is solved again with the shift moved by
  ``probe_m`` along each axis (the IK and the floor alignment, each frame started from the unshifted solution
  moved by the shift), and the ZMP at t is recomputed with each frame of its fit moved by 1 % of that change.  The
  clip is then solved with the new shift, its ZMP computed again and the least squares solved again from there,
  ``iterations`` times; the shift with the least cost is kept.  The ZMP of the clip's first and last half window
  of the fit is fitted from one side and not constrained, and the shift is held there (as at the first and last
  constrained frame): free, it moved there to make the acceleration the constrained frames' fits see.
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


@dataclass(frozen=True)
class SupportSpec:
    """The support shift's settings (module docstring)."""

    margin_m: float = 0.010  # S(t): the contact hull shrunk inward by this
    axes: str = "both"  # "both" (lateral and forward, in the heading frame) or "lateral"
    band_m: float = 0.010  # a part's contact area: its shapes within this of its lowest point
    smooth_s: float = 0.3  # the ZMP's velocities and accelerations: a cubic fitted over this
    lift_s: float = 0.3  # the ZMP moves into the remaining support over this before a part leaves the floor
    knot_s: float = 0.2  # the shift is a cubic B-spline with knots this far apart
    stance_s: float = 0.2  # a foot that left the floor stops lowering the trunk for its leg over this
    reach_slack_m: float = 0.002  # a standing leg reaches its foot this short of its straight length
    max_drop_m: float = 0.01  # the trunk goes down at most this for a standing leg to reach its foot
    joint_margin_deg: float = 1.0  # a standing leg's hip roll is kept this far from the end of its range
    max_lateral_m: float = 0.04
    max_forward_m: float = 0.03
    # Least squares weights, per frame and squared: the ZMP outside (per m; w_lift times at a lift-off), the shift
    # (per m), its acceleration (per m/s^2), its forward speed (per m/s) and the shift where it must be 0 (per m).
    w_outside: float = 1.0
    w_lift: float = 3.0
    w_shift: float = 3e-2
    w_accel: float = 1e-4
    w_speed_forward: float = 1e-2
    w_zero: float = 1e3
    fade_s: float = 0.2  # the shift goes to 0 over this before a frame where it must be 0
    huber_m: float = 0.03  # the ZMP's distance outside counts squared up to this and linearly beyond
    aim_m: float = 0.002
    probe_m: float = 0.01
    iterations: int = 2

    def __post_init__(self):
        if not (math.isfinite(self.margin_m) and self.margin_m >= 0.0):
            raise ValueError(f"the support margin must be a finite length >= 0, not {self.margin_m!r}")
        if self.axes not in ("both", "lateral"):
            raise ValueError(f"unknown support axes {self.axes!r}")

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------------------------------------
# The zero-moment point from the whole-body dynamics.


def _inverse_model(model: mujoco.MjModel) -> mujoco.MjModel:
    m = copy.copy(model)
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
    return m


def windows(T: int, fps: float, smooth_s: float):
    """Per frame the frames its derivatives are fitted over (T, W) and its place among them (T,): ``smooth_s``
    centred, moved inside the clip at its ends."""
    W = min(T, max(3, round(smooth_s * fps) // 2 * 2 + 1))
    start = np.clip(np.arange(T) - W // 2, 0, T - W)
    return start[:, None] + np.arange(W)[None], np.arange(T) - start


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


def zmp_at(model: mujoco.MjModel, centre: np.ndarray, near: np.ndarray, pos: np.ndarray, fps: float):
    """The ZMP (T, 2) and the floor's vertical force over the weight (T,) at the poses ``centre`` (T, nq), with the
    velocities and accelerations of a Savitzky-Golay fit (cubic) over the poses ``near`` (T, W, nq), of which
    ``centre`` is the ``pos``-th."""
    m = _inverse_model(model)
    d = mujoco.MjData(m)
    root = int(m.jnt_bodyid[0])
    weight = float(m.body_subtreemass[root]) * GRAVITY
    T, W = near.shape[:2]
    order = min(3, W - 1)
    c1 = np.stack([savgol_coeffs(W, order, deriv=1, delta=1.0 / fps, pos=p, use="dot") for p in range(W)])
    c2 = np.stack([savgol_coeffs(W, order, deriv=2, delta=1.0 / fps, pos=p, use="dot") for p in range(W)])
    u = tangent(centre[:, None], near)
    qvel = np.einsum("tw,twv->tv", c1[pos], u)
    qacc = np.einsum("tw,twv->tv", c2[pos], u)
    zmp_, load = np.full((T, 2), np.nan), np.zeros(T)
    for t in range(T):
        d.qpos[:] = centre[t]
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


def zmp(model: mujoco.MjModel, qs: np.ndarray, fps: float, smooth_s: float):
    """The ZMP (T, 2) and the load share (T,) of a qpos sequence (zmp_at, fitted over ``smooth_s``)."""
    if len(qs) < 3:
        return np.full((len(qs), 2), np.nan), np.zeros(len(qs))
    idx, pos = windows(len(qs), fps, smooth_s)
    return zmp_at(model, qs, qs[idx], pos, fps)


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

    def area(self, verts: np.ndarray, parts, band: float) -> np.ndarray:
        """(N, 2) points whose hull is the contact area of ``parts``: per part, its shapes cut at ``band`` above
        its lowest point (the vertices below and the edges' crossings)."""
        low = self.lowest(verts)
        cut = low[self.part_of] + band
        keep = np.isin(self.part_of, list(parts))
        pts = [verts[keep & (verts[:, 2] <= cut), :2]]
        a, b = self.edges[:, 0], self.edges[:, 1]
        za, zb, c = verts[a, 2], verts[b, 2], cut[a]
        cross = keep[a] & ((za - c) * (zb - c) < 0.0)
        s = ((c - za) / np.where(cross, zb - za, 1.0))[cross]
        pts.append(verts[a[cross], :2] + s[:, None] * (verts[b[cross], :2] - verts[a[cross], :2]))
        return np.concatenate(pts)


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


@dataclass
class Supports:
    """Per frame: the parts on the floor (T, P), the allowed area (half-planes, or None), S(t)'s vertices and the
    weight of the ZMP outside the allowed area (T,)."""

    contact: np.ndarray
    A: list
    h: list
    S: list
    weight: np.ndarray


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
    A, h, S = [None] * T, [None] * T, [np.zeros((0, 2))] * T
    for t in range(T):
        parts = np.flatnonzero(contact[t])
        if len(parts):
            A[t], h[t] = shrink(*halfplanes(shapes.area(P.verts[t], parts, spec.band_m)), spec.margin_m)
            S[t] = region_vertices(A[t], h[t])
    if not plan:
        return Supports(contact, A, h, S, weight)
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
            An, hn = shrink(*halfplanes(shapes.area(P.verts[t], np.flatnonzero(keep), spec.band_m)), spec.margin_m)
            rem = region_vertices(An, hn)
            if t == t0:
                far = max(outside(p, rem) for p in S[t]) if len(S[t]) else 0.0
            r = far * (last - t) / max(last - t0, 1)
            A[t] = np.r_[A[t], An, _DIRS]
            h[t] = np.r_[h[t], hn + r, np.max(rem @ _DIRS.T, axis=0) + r]
        weight[[last, t1]] = spec.w_lift
    return Supports(contact, A, h, S, weight)


def lift_offs(contact: np.ndarray) -> np.ndarray:
    """The frames (first off the floor) where a part leaves the floor and another stays."""
    before, after = contact[:-1], contact[1:]
    return np.flatnonzero((before & ~after).any(1) & (before & after).any(1)) + 1


def classes(contact: np.ndarray) -> dict:
    """Frame masks by the parts on the floor: one foot, both feet, a part other than the feet, none."""
    feet = contact[:, FEET].sum(1)
    other = np.delete(contact, FEET, axis=1).any(1)
    return {"single": (feet == 1) & ~other, "double": (feet == 2) & ~other, "other": other, "none": ~contact.any(1)}


def stats(z, load, sup: Supports):
    """Distances (T,) of the ZMP outside S(t) (nan without contact or without load)."""
    out = np.full(len(z), np.nan)
    for t in range(len(z)):
        if sup.A[t] is not None and load[t] >= MIN_LOAD and np.isfinite(z[t]).all():
            out[t] = outside(z[t], sup.S[t])
    return out


def evaluate(model, qs, human, fps: float, spec: SupportSpec, shapes: Shapes | None = None):
    """(ZMP, load, the parts on the floor, the distances outside S(t)) of a qpos sequence; also its Poses."""
    shapes = shapes or Shapes(model)
    P = poses(model, qs, shapes)
    contact = robot_contact(P, human, shapes, fps)
    z, load = zmp(model, qs, fps, spec.smooth_s)
    return (z, load, contact, stats(z, load, supports(P, contact, shapes, spec, fps))), P


def zmp_summary(dist, contact) -> dict:
    """Per class of frames (classes) and at the lift-offs: the frames with a ZMP, the share inside S(t) and the
    distance outside (p95, max; mm)."""
    def one(d):
        d = d[np.isfinite(d)]
        if not len(d):
            return {"frames_with_zmp": 0}
        return {"frames_with_zmp": len(d), "inside_share": round(float((d <= 1e-4).mean()), 4),
                "outside_mm_p95_max": [round(1000 * float(np.percentile(d, 95)), 1), round(1000 * float(d.max()), 1)]}
    out = {k: one(dist[m]) for k, m in classes(contact).items() if k != "none"}
    out["lift_off"] = one(dist[lift_offs(contact)])
    return out


# ---------------------------------------------------------------------------------------------------------
# The least squares.


def heading_basis(yaw: np.ndarray, axes: str) -> np.ndarray:
    """(T, 2, k) the shift's axes in the world: lateral (the trunk heading's left) and, with "both", forward."""
    left = np.stack([-np.sin(yaw), np.cos(yaw)], 1)
    if axes == "lateral":
        return left[:, :, None]
    return np.stack([left, np.stack([np.cos(yaw), np.sin(yaw)], 1)], 2)


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


def hip_rolls(model: mujoco.MjModel):
    """The hip roll joints (left, right): qpos addresses (2,) and ranges (2, 2)."""
    ids = [model.joint(f"{s}_hip_roll").id for s in ("left", "right")]
    return model.jnt_qposadr[ids], model.jnt_range[ids]


def leg_length(model: mujoco.MjModel) -> float:
    """The hip centre to ankle centre distance with the knee straight."""
    d = mujoco.MjData(model)
    mujoco.mj_kinematics(model, d)
    return float(np.linalg.norm(d.site_xpos[model.site("left_hip_c").id] - d.site_xpos[model.site("left_ankle_c").id]))


@dataclass
class Problem:
    """What the least squares needs besides the linearisation point: the ZMP's response to the shift (J (T, W, 2,
    k) at the frames idx (T, W)) and the standing legs' hip roll's (Jr (T, 2, k); the legs and frames ``roll_on``
    (T, 2), its range (lo, hi) less the margin), how much the shift must be 0 (T,; 1 where it must, fading out
    around), the frames where it is held (T,; the clip's first and last half window) and the standing legs' reach
    (rows (n, T k), bounds (n,))."""

    J: np.ndarray
    idx: np.ndarray
    Jr: np.ndarray
    roll_on: np.ndarray
    roll_range: tuple
    zero: np.ndarray
    hold: np.ndarray
    reach_rows: sp.csr_matrix
    reach_b: np.ndarray


def reach_limits(P: Poses, contact: np.ndarray, basis: np.ndarray, length: float, spec: SupportSpec):
    """The shift that keeps each standing leg's hip within its reach of its ankle, as rows (n, T k) and bounds:
    |hip + s - ankle| at most the straight length less ``reach_slack_m`` (or as far as the unshifted leg, if
    farther), with the hip ``max_drop_m`` lower (16 tangent half-planes of that circle)."""
    T, _, k = basis.shape
    reach = max(length - spec.reach_slack_m, 0.0)
    far = math.hypot(spec.max_lateral_m, spec.max_forward_m)
    rows, cols, vals, b = [], [], [], []
    n = 0
    for t in range(T):
        for side in (0, 1):
            if not contact[t, FEET[side]]:
                continue
            v = P.ankle[t, side] - P.hip[t, side]
            c = v[:2]
            r_max = max(reach, float(np.linalg.norm(v)))
            r = math.sqrt(max(r_max**2 - max(-v[2] - spec.max_drop_m, 0.0) ** 2, 0.0))
            bound = r + _DIRS @ c
            use = np.flatnonzero(bound < far)
            if not len(use):
                continue
            g = _DIRS[use] @ basis[t]  # (m, k)
            for i, row in enumerate(g):
                rows += [n + i] * k
                cols += list(t * k + np.arange(k))
                vals += list(row)
            b.append(bound[use])
            n += len(use)
    R = sp.csr_matrix((vals, (rows, cols)), shape=(n, T * k))
    return R, (np.concatenate(b) if b else np.zeros(0))


def solve_shift(x_k, z_k, roll_k, sup: Supports, constrained, prob: Problem, spec: SupportSpec, fps: float,
                caps: np.ndarray, coef=None):
    """The shift (T, k) and its B-spline coefficients minimising the cost linearised around (x_k, z_k, roll_k):
    z(x) = z_k + sum_w J[:, w] (x - x_k) at the frames idx[:, w], the hip rolls (T, 2) roll_k + Jr (x - x_k).
    Hinges, squared (the ZMP's: squared up to ``huber_m`` and linear beyond, so that a ZMP far out that the shift
    cannot bring in, a push-off before a jump, does not swing the shift), solved by Newton steps on the active
    set (the linear part as a reweighted square)."""
    T, k = x_k.shape
    n = T * k
    J, idx = prob.J, prob.idx
    W = idx.shape[1]
    t, w, i, j = np.meshgrid(np.arange(T), np.arange(W), np.arange(2), np.arange(k), indexing="ij")
    Jb = sp.csr_matrix((J.reshape(-1), ((2 * t + i).reshape(-1), (k * idx[t, w] + j).reshape(-1))), shape=(2 * T, n))
    off = np.nan_to_num(z_k.reshape(-1) - Jb @ x_k.reshape(-1))
    blocks, bounds, weights = [], [], []
    # The ZMP outside the allowed area.
    for t in constrained:
        A, h = sup.A[t], sup.h[t]
        blocks.append(sp.csr_matrix(A) @ Jb[2 * t:2 * t + 2])
        bounds.append(h - spec.aim_m - A @ off[2 * t:2 * t + 2])
        weights.append(np.full(len(h), spec.w_outside * sup.weight[t]))
    n_zmp = sum(len(x) for x in bounds)
    # The standing legs' reach and hip roll.
    blocks.append(prob.reach_rows)
    bounds.append(prob.reach_b)
    weights.append(np.full(len(prob.reach_b), HARD * spec.w_outside))
    tt, ss = np.nonzero(prob.roll_on)
    if len(tt):
        g = prob.Jr[tt, ss]  # (r, k)
        G = sp.csr_matrix((g.reshape(-1), (np.repeat(np.arange(len(tt)), k), (tt[:, None] * k + np.arange(k)).reshape(-1))),
                          shape=(len(tt), n))
        lin = roll_k[tt, ss] - G @ x_k.reshape(-1)
        lo, hi = prob.roll_range
        blocks += [G, -G]
        bounds += [hi[ss] - lin, lin - lo[ss]]
        weights += [np.full(len(tt), HARD * spec.w_outside)] * 2
    # The caps.
    cap = np.tile(caps, T)
    blocks += [sp.eye(n, format="csr"), -sp.eye(n, format="csr")]
    bounds += [cap, cap]
    weights += [np.full(n, HARD * spec.w_outside)] * 2
    R = sp.vstack(blocks).tocsr()
    b = np.concatenate(bounds)
    wr = np.concatenate(weights)
    # The quadratic terms, on the shift; the shift is the B-spline M c.
    B = bspline(T, round(spec.knot_s * fps))
    M = sp.kron(B, sp.eye(k), format="csr")
    D2, D1 = sp.kron(_diff(T, 2), sp.eye(k)), _diff(T, 1)
    Q = spec.w_shift * sp.eye(n) + spec.w_accel * fps**4 * (D2.T @ D2)
    if k == 2:
        F = sp.kron(D1, sp.csr_matrix(([1.0], ([0], [1])), shape=(1, 2)))  # the forward axis's speed
        Q = Q + spec.w_speed_forward * fps**2 * (F.T @ F)
    Q = Q + sp.diags(spec.w_zero * np.repeat(prob.zero, k))
    # Held where the ZMP is not constrained (the clip's ends): its steps there weigh as the shift where it must be 0.
    pairs = sp.diags((prob.hold[:-1] | prob.hold[1:]).astype(float))
    Q = Q + spec.w_zero * sp.kron(D1.T @ pairs @ D1, sp.eye(k))
    # A small ridge on the coefficients: the knots at the clip's ends act on few frames.
    Qc = (M.T @ Q @ M + 1e-6 * sp.eye(M.shape[1])).tocsc()
    Rc = (R @ M).tocsr()
    huber = np.r_[np.full(n_zmp, spec.huber_m), np.full(len(b) - n_zmp, 1e6)]  # (the limits: squared)

    def cost(c):
        r = np.maximum(Rc @ c - b, 0.0)
        far = r > huber
        return float(c @ (Qc @ c) + wr @ np.where(far, 2.0 * huber * r - huber**2, r * r))

    c = np.zeros(M.shape[1]) if coef is None else coef.copy()
    f = cost(c)
    for _ in range(50):
        r = Rc @ c - b
        act = r > 0.0
        Ra = Rc[act]
        we = wr[act] * np.minimum(1.0, huber[act] / r[act])  # the linear part as a reweighted square
        H = Qc + Ra.T @ sp.diags(we) @ Ra
        c_new = spsolve(H.tocsc(), Ra.T @ (we * b[act]))
        if not np.isfinite(c_new).all():
            break
        step, f_new = 1.0, cost(c_new)
        while f_new > f + 1e-15 and step > 1e-3:
            step *= 0.5
            f_new = cost(c + step * (c_new - c))
        c_next = c + step * (c_new - c)
        done = np.abs(M @ (c_next - c)).max() < 1e-6
        c, f = c_next, f_new
        if done:
            break
    return (M @ c).reshape(T, k), c


def sensitivity(model, solve, qs, basis, fps: float, spec: SupportSpec, step: float = 0.01):
    """J (T, W, 2, k): the ZMP at t against the shift of the frames its derivatives are fitted over (windows),
    those frames (T, W), and Jr (T, 2, k): the hip rolls (left, right) against the shift.  The clip is solved again with the shift moved by ``probe_m`` along each axis (``solve``
    maps a world shift (T, 2) to qpos on the floor), and the ZMP at t is recomputed with one of those frames moved
    by ``step`` of that change (a single moved frame has a large acceleration, and the ZMP is not linear in it over
    a whole probe)."""
    T, _, k = basis.shape
    idx, pos = windows(T, fps, spec.smooth_s)
    W = idx.shape[1]
    near = qs[idx]
    z0, _ = zmp_at(model, qs, near, pos, fps)
    J = np.zeros((T, W, 2, k))
    Jr = np.zeros((T, 2, k))
    roll = hip_rolls(model)[0]
    v = np.zeros(model.nv)
    rows = np.arange(T)
    for j in range(k):
        qp = solve(basis[:, :, j] * spec.probe_m)
        # An arm that takes the other branch in the probe says nothing about the shift: its joints are kept.
        jump = np.abs(qp[:, 7:] - qs[:, 7:]) > 0.5
        qp[:, 7:][jump] = qs[:, 7:][jump]
        Jr[:, :, j] = (qp[:, roll] - qs[:, roll]) / spec.probe_m
        qm = qs.copy()
        for t in range(T):
            mujoco.mj_differentiatePos(model, v, 1.0, qs[t], qp[t])
            mujoco.mj_integratePos(model, qm[t], v, step)
        for w in range(W):
            moved = near.copy()
            moved[:, w] = qm[idx[:, w]]
            centre = moved[rows, pos]
            z, _ = zmp_at(model, centre, moved, pos, fps)
            J[:, w, :, j] = np.nan_to_num((z - z0) / (step * spec.probe_m))
    return J, idx, Jr


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


def plan(model, solve, qs, human, fps: float, spec: SupportSpec):
    """The support shift of a clip.  ``solve(shift, stance)`` maps a world shift (T, 2) and the feet's weights on
    the floor (T, 2; stance) to the clip's qpos on the floor; ``qs`` is the unshifted clip's, and
    ``human`` is the human's contacts (T, P).  Returns
    the qpos, the shift (T, 2), a summary (the ZMP in S(t) before and after, the cost of each iteration), and (ZMP,
    load, the parts on the floor, the distances outside S(t)) before and after (after: the parts found again from
    the shifted poses)."""
    shapes = Shapes(model)
    T = len(qs)
    before, P0 = evaluate(model, qs, human, fps, spec, shapes)
    z, load, contact, _ = before
    summary = {"margin_m": spec.margin_m, "before": zmp_summary(before[3], contact), "cost_before": None,
               "iterations": []}
    zero = ~contact.any(1) | np.delete(contact, FEET, axis=1).any(1)
    # The shift fades in and out over ``fade_s`` around the frames where it is 0.
    fade = np.clip(1.0 - _distance(zero) / max(1, round(spec.fade_s * fps)), 0.0, 1.0) ** 2
    # The ZMP of the first and last half window is fitted from one side: not constrained.
    edge = np.zeros(T, bool)
    half = windows(T, fps, spec.smooth_s)[0].shape[1] // 2
    edge[:half], edge[T - half:] = True, True
    if (zero | edge).all() or T < 3:
        summary["after"] = summary["before"]
        return qs, np.zeros((T, 2)), summary, before, before
    feet = stance(contact, fps, spec.stance_s)
    basis = heading_basis(P0.yaw, spec.axes)
    k = basis.shape[2]
    caps = np.array([spec.max_lateral_m, spec.max_forward_m][:k])

    def world(x):
        return np.einsum("tik,tk->ti", basis, x)

    J, idx, Jr = sensitivity(model, lambda s: solve(s, feet), qs, basis, fps, spec)
    # A standing leg's hip roll is kept off the end of its range (at the end the IK can only lift the foot), unless
    # the unshifted clip has it there.
    m = math.radians(spec.joint_margin_deg)
    roll, rng = hip_rolls(model)
    lo, hi = rng[:, 0] + m, rng[:, 1] - m
    roll_on = contact[:, FEET] & (qs[:, roll] > lo) & (qs[:, roll] < hi) & (fade < 1.0)[:, None]
    prob = Problem(J, idx, Jr, roll_on, (lo, hi), fade, edge, *reach_limits(P0, contact, basis, leg_length(model), spec))

    def assess(q, z, load):
        sup = supports(poses(model, q, shapes), contact, shapes, spec, fps, plan=True)
        frames = [t for t in range(T) if sup.A[t] is not None and load[t] >= MIN_LOAD and np.isfinite(z[t]).all()
                  and not (zero[t] or edge[t])]
        cost = float(sum(sup.weight[t] * np.sum(np.maximum(sup.A[t] @ z[t] - sup.h[t], 0.0) ** 2) for t in frames))
        return sup, frames, cost

    sup, constrained, cost = assess(qs, z, load)
    summary["cost_before"] = round(cost, 8)
    best = (cost, qs, np.zeros((T, k)), z, load)
    x, coef, q_k = np.zeros((T, k)), None, qs
    for _ in range(spec.iterations):
        x, coef = solve_shift(x, np.nan_to_num(z), q_k[:, roll], sup, constrained, prob, spec, fps, caps, coef)
        q_k = solve(world(x), feet)
        z, load = zmp(model, q_k, fps, spec.smooth_s)
        sup, constrained, cost = assess(q_k, z, load)
        summary["iterations"].append(round(cost, 8))
        if cost < best[0]:
            best = (cost, q_k, x.copy(), z, load)
    _, qs_b, x_b, z_b, load_b = best
    P = poses(model, qs_b, shapes)
    contact_b = robot_contact(P, human, shapes, fps)
    after = (z_b, load_b, contact_b, stats(z_b, load_b, supports(P, contact_b, shapes, spec, fps)))
    summary["after"] = zmp_summary(after[3], contact_b)
    return qs_b, world(x_b), summary, before, after
