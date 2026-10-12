"""The support shift of the retargeting (``core.retarget_frames(..., support_spec=SupportSpec())``; off by default).

Microban's feet are far apart for its size (HOME's sole centres 93 mm apart, 0.75 leg lengths; a human's
0.2-0.25), so the human's motion, scaled, puts Microban's weight outside the foot it stands on.  The support
shift moves the trunk horizontally, with the hands and feet where they were (core.MicrobanRetargeting._shifted),
so that Microban's zero-moment point stays in its support:

* The zero-moment point (ZMP) is Microban's, from its whole-body dynamics: the retargeted qpos sequence through
  mj_inverse on robot.xml (the masses and inertias the training simulates; the arms' and the trunk's angular
  momentum included), without contacts.  The free joint's force and torque are then the wrench the floor has to
  give, and the ZMP is the point of the floor (z = 0) about which its horizontal moment is zero.  Frames whose
  vertical force is below MIN_LOAD of the weight (falling, flight) have none.  The velocities and accelerations
  are those of a cubic fitted over ``smooth_s`` (0.3 s): with central differences the capture's frame-to-frame
  noise moves the ZMP by 7-68 mm per frame (median; the PICO walk the most), with 0.3 s by 0.3-7.6 mm, within
  9 mm (p95) of a 0.6 s fit.
* The support S(t) is the convex hull of the floor contact areas of every part on the floor (feet, legs,
  forearms, upper arms, trunk), shrunk inward by ``margin_m`` (10 mm by default; the data command's
  ``--support-margin-mm``).  A part is on the floor when the human's is (as low as when standing for a foot, as
  low as when lying on the floor for the others (core.CONTACT_RADIUS), and slow across the floor) and Microban's
  is within ROBOT_CONTACT_M of the floor, or when Microban's touches the floor (ROBOT_TOUCH_M) and is slow:
  lying on the floor (LAFAN1 ground1_subject5) the human's trunk often counts as up while Microban's touches
  the floor, and an arm alone would be the support.  Lying and crawling (LYING_TILT_DEG) have none.  The
  shifted clip keeps the unshifted clip's parts.  A
  part's contact area is the part of its training collision shapes within ``band_m`` (10 mm) of its lowest
  point, seen from above: a sole tilted less than 9 degrees counts whole (the retargeted standing foot is tilted
  6-10 degrees at the median, 7.4 standing on one leg in CMU 49_19), a foot rolling on its heel or toes counts
  its edge.  Where the hull is too narrow to shrink by ``margin_m``, S(t) is its deepest points (the hull
  shrunk by its inradius: its middle line or point), so that S changes continuously as a foot rolls.
* Before a part leaves the floor, the ZMP moves into what remains: over the last ``lift_s`` of the contact
  phase (at most its length) the allowed area is S(t) within a distance of the remaining support that falls
  linearly to 0 at lift-off.  After a landing the ZMP may be anywhere in the larger support.  With no part on
  the floor (a flight) the ZMP is not constrained, and the shift only joins the frames around.
* The shift d(t) (in the heading frame, lateral and forward) is solved over the whole clip by least squares:
  the ZMP outside the allowed area (half-planes, aimed ``aim_m`` inside), the shift's size and its acceleration,
  and the shift beyond ``max_lateral_m`` / ``max_forward_m``.  The ZMP's response to the shift is found
  numerically: the clip is solved again with the shift moved by ``probe_m`` along each axis (the IK and the
  floor alignment, each frame started from the unshifted solution), and the ZMP at t is recomputed with each
  frame of its fit moved by 1 % of that change (one moved frame has a large acceleration and the ZMP is not
  linear in it over the whole probe).  The clip is then solved with the new shift, its ZMP computed again and
  the least squares solved again from there, ``iterations`` times; the shift with the least cost is kept.
* Forward too (``axes="both"``), by the numbers on CMU 07_01, 16_18, 49_19, 132_24, LAFAN1 walk3_subject5 and
  PICO picoteleopit_rec06: shifting sideways only leaves the ZMP 10-54 mm (p95) outside on one foot and inside
  in 0-40 % of those frames, because the scaled walk's ZMP runs ahead of the short sole; with forward the p95
  is 0-14 mm and 50-100 % are inside (0-4 % before).  The trunk then slows over the sole and speeds up between
  steps (the forward speed's standard deviation 0.02 -> 0.07-0.09 m/s on the CMU walks), but the shift stays
  within 5 cm, so the trunk never falls behind or gets ahead of the unshifted one by more.
"""

from __future__ import annotations

import copy
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
# Lying and crawling (the trunk more than this from upright with a part other than the feet on the floor) are
# left alone, as by preprocess's unsupported rule: Microban's arms and trunk are 6-45 mm above the floor where
# the human's lie on it, so its zero-moment point is 70-140 mm outside what touches the floor, and the shift up
# to its limits only took LAFAN1 ground1_subject5's frames in S(t) from 12 to 16 %.
LYING_TILT_DEG = 45.0
# A human part is on the floor when its joints are less than CONTACT_HEIGHT leg lengths above where they are on
# the floor and the slowest of them moves less than CONTACT_SPEED leg lengths per second across it (CMU 07_01: a
# standing foot 0-0.02 high and below 0.3 per second, a swinging one 0.05-0.14 high and 2.6-5.4 per second).
CONTACT_HEIGHT = 0.05
CONTACT_SPEED = 0.8


@dataclass(frozen=True)
class SupportSpec:
    """The support shift's settings (module docstring)."""

    margin_m: float = 0.010  # S(t): the contact hull shrunk inward by this
    axes: str = "both"  # "both" (lateral and forward, in the heading frame) or "lateral"
    band_m: float = 0.010  # a part's contact area: its shapes within this of its lowest point
    smooth_s: float = 0.3  # the ZMP's velocities and accelerations: a cubic fitted over this
    lift_s: float = 0.3  # the ZMP moves into the remaining support over this before a part leaves the floor
    # The largest shift: sideways, both feet kept, 40 mm puts the far leg's hip roll at its 25 degree limit.
    max_lateral_m: float = 0.04
    max_forward_m: float = 0.03
    # Least squares weights, per frame: the ZMP outside (per m), the shift (per m), its acceleration (per m/s^2;
    # lateral, forward).  The ZMP is aimed ``aim_m`` inside S (the linearisation's error).
    w_outside: float = 1.0
    w_shift: float = 1e-3
    w_accel_lateral: float = 3e-5
    w_accel_forward: float = 3e-5
    aim_m: float = 0.002
    probe_m: float = 0.01
    iterations: int = 2

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
# Contact areas.

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
        for value in (True, False):
            mid = (c[:-2] == value) & (c[1:-1] != value) & (c[2:] == value)
            c[1:-1][mid] = value
        out[:, p] = c
    return out


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
    """Per frame: the parts on the floor (T, P), the allowed area (half-planes, or None) and S(t)'s vertices."""

    contact: np.ndarray
    A: list
    h: list
    S: list


def supports(model, qs, human, shapes: Shapes, spec: SupportSpec, fps: float, contact=None) -> Supports:
    """S(t) and the allowed area of every frame (module docstring) from the poses ``qs`` and the human's contacts
    (T, P); ``contact``: the parts on the floor as given (the shifted clip keeps the unshifted clip's)."""
    d = mujoco.MjData(model)
    T = len(qs)
    verts, tilt_cos = [], []
    root = int(model.jnt_bodyid[0])
    for t in range(T):
        d.qpos[:] = qs[t]
        mujoco.mj_kinematics(model, d)
        verts.append(shapes.vertices(d))
        tilt_cos.append(d.xmat[root, 8])
    if contact is None:
        lying = np.degrees(np.arccos(np.clip(tilt_cos, -1.0, 1.0))) > LYING_TILT_DEG
        low = np.array([shapes.lowest(v) for v in verts])
        centre = np.array([[v[shapes.part_of == p, :2].mean(0) for p in range(len(PARTS))] for v in verts])
        speed = np.linalg.norm(np.gradient(centre, axis=0), axis=2) * fps if T > 1 else np.zeros((T, len(PARTS)))
        contact = (human & (low < ROBOT_CONTACT_M)) | ((low < ROBOT_TOUCH_M) & (speed < ROBOT_TOUCH_SPEED))
        # Lying and crawling: no support (the frame is left alone).
        contact[lying & contact[:, [PARTS.index(p) for p in PARTS if not p.endswith("_foot")]].any(1)] = False
    contact = contact.copy()
    A, h, S = [None] * T, [None] * T, [np.zeros((0, 2))] * T
    for t in range(T):
        parts = np.flatnonzero(contact[t])
        if len(parts):
            A[t], h[t] = shrink(*halfplanes(shapes.area(verts[t], parts, spec.band_m)), spec.margin_m)
            S[t] = region_vertices(A[t], h[t])
    # Before a part leaves the floor: within a distance of the remaining support falling to 0 at lift-off.
    n_lift = max(1, round(spec.lift_s * fps))
    for t1 in range(1, T):
        before, after = contact[t1 - 1], contact[t1]
        keep = before & after
        if not (before & ~after).any() or not keep.any():
            continue
        t0 = t1 - 1
        while t0 > 0 and t1 - t0 < n_lift and (contact[t0 - 1] == before).all():
            t0 -= 1
        for t in range(t0, t1):
            An, hn = shrink(*halfplanes(shapes.area(verts[t], np.flatnonzero(keep), spec.band_m)), spec.margin_m)
            if t == t0:
                far = max(outside(p, region_vertices(An, hn)) for p in S[t]) if len(S[t]) else 0.0
            r = far * (t1 - t) / (t1 - t0 + 1)
            hd = np.r_[hn, np.max(region_vertices(An, hn) @ _DIRS.T, axis=0)] + r
            A[t] = np.r_[A[t], An, _DIRS]
            h[t] = np.r_[h[t], hd]
    return Supports(contact, A, h, S)


def heading_basis(model, qs, axes: str) -> np.ndarray:
    """(T, 2, k) the shift's axes in the world: lateral (the trunk heading's left) and, with "both", forward."""
    d = mujoco.MjData(model)
    trunk = model.body("trunk").id
    yaw = np.zeros(len(qs))
    for t, q in enumerate(qs):
        d.qpos[:] = q
        mujoco.mj_kinematics(model, d)
        yaw[t] = np.arctan2(d.xmat[trunk, 3], d.xmat[trunk, 0])
    yaw = np.unwrap(yaw)
    left = np.stack([-np.sin(yaw), np.cos(yaw)], 1)
    if axes == "lateral":
        return left[:, :, None]
    if axes == "both":
        return np.stack([left, np.stack([np.cos(yaw), np.sin(yaw)], 1)], 2)
    raise ValueError(f"unknown support axes {axes!r}")


# ---------------------------------------------------------------------------------------------------------
# The least squares.


def _diff(T: int, k: int, order: int) -> sp.csr_matrix:
    D = sp.eye(T, format="csr")
    for _ in range(order):
        D = (D[1:] - D[:-1]).tocsr()
    return sp.kron(D, sp.eye(k), format="csr")


def solve_shift(x_k, z_k, J, idx, sup: Supports, constrained, spec: SupportSpec, fps: float, caps: np.ndarray):
    """The shift (T, k) minimising the linearised cost around (x_k, z_k): z(x) = z_k + sum_w J[:, w] (x - x_k) at
    the frames idx[:, w] (J (T, W, 2, k)).  Squared hinges, solved by Newton steps on the active set."""
    T, k = x_k.shape
    n = T * k
    W = idx.shape[1]
    t, w, i, j = np.meshgrid(np.arange(T), np.arange(W), np.arange(2), np.arange(k), indexing="ij")
    Jb = sp.csr_matrix((J.reshape(-1), ((2 * t + i).reshape(-1), (k * idx[t, w] + j).reshape(-1))), shape=(2 * T, n))
    c = z_k.reshape(-1) - Jb @ x_k.reshape(-1)
    c = np.nan_to_num(c)
    # The constraint rows (all half-planes of the constrained frames).
    crow, cb = [], []
    for t in constrained:
        A, h = sup.A[t], sup.h[t]
        crow.append(sp.csr_matrix(A) @ Jb[2 * t:2 * t + 2])
        cb.append(h - spec.aim_m - A @ c[2 * t:2 * t + 2])
    C = sp.vstack(crow).tocsr() if crow else sp.csr_matrix((0, n))
    cb = np.concatenate(cb) if cb else np.zeros(0)
    D2 = _diff(T, k, 2)
    acc = sp.diags(np.tile([spec.w_accel_lateral, spec.w_accel_forward][:k], T - 2))
    base = (spec.w_shift * sp.eye(n) + fps**4 * (D2.T @ acc @ D2)).tocsc()
    cap = np.tile(caps, T)

    def cost(x):
        r = np.maximum(C @ x - cb, 0.0)
        e = np.maximum(np.abs(x) - cap, 0.0)
        return spec.w_outside * r @ r + x @ (base @ x) + 1e3 * spec.w_outside * e @ e

    x = x_k.reshape(-1).copy()
    f = cost(x)
    for _ in range(50):
        act = (C @ x - cb) > 0.0
        over = np.abs(x) > cap
        Ca = C[act]
        H = base + spec.w_outside * (Ca.T @ Ca) + sp.diags(1e3 * spec.w_outside * over.astype(float))
        g = spec.w_outside * (Ca.T @ cb[act]) + 1e3 * spec.w_outside * over * np.sign(x) * cap
        x_new = spsolve(H.tocsc(), g)
        step, f_new = 1.0, cost(x_new)
        while f_new > f + 1e-15 and step > 1e-3:
            step *= 0.5
            f_new = cost(x + step * (x_new - x))
        x_next = x + step * (x_new - x)
        done = np.abs(x_next - x).max() < 1e-6
        x, f = x_next, f_new
        if done:
            break
    return x.reshape(T, k)


def sensitivity(model, solve, qs, x, basis, fps: float, spec: SupportSpec, step: float = 0.01):
    """J (T, W, 2, k): the ZMP at t against the shift of the frames its derivatives are fitted over (windows).
    The clip is solved again with the shift moved by ``probe_m`` along each axis (``solve`` maps a world shift
    (T, 2) to qpos on the floor), and the ZMP at t is recomputed with one of those frames moved by ``step`` of
    that change (a single moved frame has a large acceleration, and the ZMP is not linear in it over a whole
    probe)."""
    T, k = x.shape
    idx, pos = windows(T, fps, spec.smooth_s)
    W = idx.shape[1]
    near = qs[idx]
    z0, _ = zmp_at(model, qs, near, pos, fps)
    J = np.zeros((T, W, 2, k))
    v = np.zeros(model.nv)
    rows = np.arange(T)
    for j in range(k):
        e = np.zeros(k)
        e[j] = spec.probe_m
        qp = solve(np.einsum("tik,tk->ti", basis, x + e))
        # An arm that takes the other branch in the probe says nothing about the shift: its joints are kept.
        jump = np.abs(qp[:, 7:] - qs[:, 7:]) > 0.5
        qp[:, 7:][jump] = qs[:, 7:][jump]
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
    return J


def stats(z, load, sup: Supports, frames=None):
    """Distances (T,) of the ZMP outside S(t) (nan without contact or without load)."""
    T = len(z)
    out = np.full(T, np.nan)
    for t in range(T) if frames is None else frames:
        if sup.A[t] is not None and load[t] >= MIN_LOAD and np.isfinite(z[t]).all():
            out[t] = outside(z[t], sup.S[t])
    return out


def plan(model, solve, human, fps: float, spec: SupportSpec):
    """The support shift of a clip.  ``solve`` maps a world shift (T, 2) (None: none) to the clip's qpos on the
    floor and ``human`` is the human's contacts (T, P).  Returns the qpos, the shift (T, 2), a summary (the ZMP in
    S(t) before and after, the cost of each iteration), and (ZMP, load, Supports, distances outside) before and
    after."""
    shapes = Shapes(model)
    qs = solve(None)
    T = len(qs)
    basis = heading_basis(model, qs, spec.axes)
    k = basis.shape[2]
    caps = np.array([spec.max_lateral_m, spec.max_forward_m][:k])
    z, load = zmp(model, qs, fps, spec.smooth_s)
    sup = supports(model, qs, human, shapes, spec, fps)
    contact = sup.contact
    constrained = [t for t in range(T) if sup.A[t] is not None and load[t] >= MIN_LOAD]
    dist0 = stats(z, load, sup)
    summary = {"before": zmp_summary(dist0), "cost_before": None, "iterations": []}
    if not constrained:
        summary["after"] = summary["before"]
        return qs, np.zeros((T, 2)), summary, (z, load, sup, dist0), (z, load, sup, dist0)
    x = np.zeros((T, k))
    J = sensitivity(model, solve, qs, x, basis, fps, spec)
    idx, _ = windows(T, fps, spec.smooth_s)

    def objective(z, load, sup):
        """The squared distance outside the allowed area, summed over the frames with a ZMP."""
        return float(sum(np.sum(np.maximum(sup.A[t] @ z[t] - sup.h[t], 0.0) ** 2) for t in range(T)
                         if sup.A[t] is not None and load[t] >= MIN_LOAD and np.isfinite(z[t]).all()))

    best = (objective(z, load, sup), qs, x, (z, load, sup, dist0))
    summary["cost_before"] = round(best[0], 8)
    first = best[3]
    for _ in range(spec.iterations):
        x = solve_shift(x, np.nan_to_num(z), J, idx, sup, constrained, spec, fps, caps)
        qs_i = solve(np.einsum("tik,tk->ti", basis, x))
        z, load = zmp(model, qs_i, fps, spec.smooth_s)
        sup = supports(model, qs_i, human, shapes, spec, fps, contact)
        cost = objective(z, load, sup)
        constrained = [t for t in range(T) if sup.A[t] is not None and load[t] >= MIN_LOAD]
        summary["iterations"].append(round(cost, 8))
        if cost < best[0]:
            best = (cost, qs_i, x.copy(), (z, load, sup, stats(z, load, sup)))
    _, qs_b, x_b, after = best
    summary["after"] = zmp_summary(after[3])
    return qs_b, np.einsum("tik,tk->ti", basis, x_b), summary, first, after


def zmp_summary(dist):
    """The frames with a ZMP, the share inside S(t) and the distance outside (p95, max; mm) of stats' distances."""
    ok = np.isfinite(dist)
    return {"frames_with_zmp": int(ok.sum()),
            "inside_share": round(float((dist[ok] <= 1e-4).mean()), 4) if ok.any() else None,
            "outside_mm_p95_max": [round(1000 * float(np.percentile(dist[ok], 95)), 1),
                                   round(1000 * float(dist[ok].max()), 1)] if ok.any() else None}
