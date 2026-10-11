"""Human skeletons: the BVH reader and the standing reference made from the skeleton alone.

* The BVH reader is our own (GMR's lafan_vendor reader is Ubisoft's,
  CC BY-NC-ND 4.0, and is not used).  Numbers may be separated by spaces or tabs.
* The world is turned Z-up (BVH's Y-up rotated +90 degrees about x).
  Lengths stay in the BVH's unit; the scale comes from the standing reference.
* The standing reference: from the BVH rest pose (every joint rotation 0),
  the spine (pelvis -> neck base) is turned straight up, and the legs
  (hip -> ankle) and arms (shoulder -> wrist) straight down, each by its
  smallest rotation; the face looks along +x, the pelvis is at x = y = 0 and
  the lowest joint at z = 0.  A skeleton whose rest is no standing pose
  (LAFAN1: the bones lie on one line, the feet along the shins, narrow
  shoulders) starts instead from a quiet standing frame of each subject
  (``ref_frames``); its knee and elbow bend and foot angles stay, so the
  subject's natural stance is laid onto HOME.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

YUP_TO_ZUP = R.from_euler("x", 90, degrees=True)  # (x, y, z) -> (x, -z, y)

_LIMBS = ("hip", "knee", "ankle", "toe", "shoulder", "elbow", "wrist")

# The names of each human format (GMR's convention: a name is the node that
# holds the rotation of the bone from that joint on).  ``spine`` is the root
# of the upper body (spine, neck, head, arms), ``neck`` the neck base (top of
# the chest, where the shoulders hang).
FORMATS = {
    # CMU mocap in the cgspeed BVH conversion.
    "bvh_cmu": {
        "skip_first": 1,  # frame 1 is a near-rest pose (the second frame is 70+ degrees away)
        "pelvis": "Hips",
        "spine": "LowerBack",
        "neck": "Neck",
        "L": {"hip": "LeftUpLeg", "knee": "LeftLeg", "ankle": "LeftFoot", "toe": "LeftToeBase",
              "shoulder": "LeftArm", "elbow": "LeftForeArm", "wrist": "LeftHand"},
        "R": {"hip": "RightUpLeg", "knee": "RightLeg", "ankle": "RightFoot", "toe": "RightToeBase",
              "shoulder": "RightArm", "elbow": "RightForeArm", "wrist": "RightHand"},
    },
    # LAFAN1 (Ubisoft): every subject on one skeleton whose rest is no standing
    # pose.  The reference frames are quiet standing frames whose knee bend is
    # near the subject's median and whose motion is small.
    "bvh_lafan1": {
        "skip_first": 0,
        "pelvis": "Hips",
        "spine": "Spine",
        "neck": "Neck",
        "L": {"hip": "LeftUpLeg", "knee": "LeftLeg", "ankle": "LeftFoot", "toe": "LeftToe",
              "shoulder": "LeftArm", "elbow": "LeftForeArm", "wrist": "LeftHand"},
        "R": {"hip": "RightUpLeg", "knee": "RightLeg", "ankle": "RightFoot", "toe": "RightToe",
              "shoulder": "RightArm", "elbow": "RightForeArm", "wrist": "RightHand"},
        "ref_frames": {"subject1": ("fallAndGetUp3_subject1.bvh", 1024), "subject2": ("sprint1_subject2.bvh", 3213),
                       "subject3": ("aiming2_subject3.bvh", 9079), "subject4": ("sprint1_subject4.bvh", 2220),
                       "subject5": ("pushAndStumble1_subject5.bvh", 309)},
    },
    # G1 robot recordings (the PICO sets) as pseudo-bones from G1's forward
    # kinematics (pico.py); the neck base is the midpoint of the shoulders.
    "g1": {
        "skip_first": 0, "pelvis": "pelvis", "spine": "torso", "neck": "neck",
        **{s: {p: f"{s}_{p}" for p in _LIMBS} for s in "LR"},
    },
    # SMPL-X (AMASS), joints 0-21 (smplx.py).  The rest pose stands straight.
    "smplx": {
        "skip_first": 0,
        "pelvis": "pelvis",
        "spine": "spine1",
        "neck": "neck",
        "L": {"hip": "left_hip", "knee": "left_knee", "ankle": "left_ankle", "toe": "left_foot",
              "shoulder": "left_shoulder", "elbow": "left_elbow", "wrist": "left_wrist"},
        "R": {"hip": "right_hip", "knee": "right_knee", "ankle": "right_ankle", "toe": "right_foot",
              "shoulder": "right_shoulder", "elbow": "right_elbow", "wrist": "right_wrist"},
    },
}


@dataclass
class Bvh:
    names: list[str]
    parents: np.ndarray  # -1 for the root
    offsets: np.ndarray  # (J, 3)
    channels: list[list[str]]
    motion: np.ndarray  # (T, C)
    frame_time: float


def read_bvh(path: str) -> Bvh:
    tokens = Path(path).read_text().split()
    i = 0
    names, parents, offsets, channels = [], [], [], []
    stack = []
    in_end = False
    while tokens[i] != "MOTION":
        t = tokens[i]
        if t in ("ROOT", "JOINT"):
            names.append(tokens[i + 1])
            parents.append(stack[-1] if stack else -1)
            offsets.append([0.0, 0.0, 0.0])
            channels.append([])
            i += 2
        elif t == "End":
            in_end = True
            i += 2
        elif t == "{":
            stack.append(-2 if in_end else len(names) - 1)
            i += 1
        elif t == "}":
            if stack.pop() == -2:
                in_end = False
            i += 1
        elif t == "OFFSET":
            if not in_end:
                offsets[-1] = [float(x) for x in tokens[i + 1:i + 4]]
            i += 4
        elif t == "CHANNELS":
            n = int(tokens[i + 1])
            channels[-1] = tokens[i + 2:i + 2 + n]
            i += 2 + n
        else:
            i += 1
    n_frames = int(tokens[i + 2])
    frame_time = float(tokens[i + 5])
    n_ch = sum(len(c) for c in channels)
    vals = np.asarray(tokens[i + 6:i + 6 + n_frames * n_ch], dtype=float).reshape(n_frames, n_ch)
    return Bvh(names, np.asarray(parents), np.asarray(offsets, float), channels, vals, frame_time)


def local_transforms(b: Bvh, rest: bool = False):
    """Each joint's local rotation (an R of T rotations) and translation (T, J, 3); one rest frame with ``rest``."""
    T = 1 if rest else b.motion.shape[0]
    lrot, lpos = [], np.repeat(b.offsets[None], T, axis=0).copy()
    col = 0
    for j, chans in enumerate(b.channels):
        v = b.motion[:, col:col + len(chans)] if not rest else np.zeros((1, len(chans)))
        rot_axes = "".join(c[0] for c in chans if c.endswith("rotation"))
        rot_vals = np.stack([v[:, k] for k, c in enumerate(chans) if c.endswith("rotation")], axis=1) if rot_axes else None
        for k, c in enumerate(chans):
            if c.endswith("position") and not rest:
                lpos[:, j, "XYZ".index(c[0])] = v[:, k]
        if rest and b.parents[j] < 0:
            lpos[:, j] = 0.0
        lrot.append(R.from_euler(rot_axes.upper(), rot_vals, degrees=True) if rot_axes else R.identity(T))
        col += len(chans)
    return lrot, lpos


def fk(b: Bvh, lrot, lpos):
    """Global rotations (an R of T per joint) and positions (T, J, 3), in the BVH's Y-up world."""
    J = len(b.names)
    grot, gpos = [None] * J, np.zeros_like(lpos)
    for j in range(J):
        p = b.parents[j]
        if p < 0:
            grot[j], gpos[:, j] = lrot[j], lpos[:, j]
        else:
            grot[j] = grot[p] * lrot[j]
            gpos[:, j] = gpos[:, p] + grot[p].apply(lpos[:, j])
    return grot, gpos


def subtree(b: Bvh, root: int) -> list[int]:
    out = [root]
    for j in range(len(b.names)):
        if b.parents[j] in out:
            out.append(j)
    return out


def _facing_yaw(pos, idx, fmt):
    """The facing direction (yaw in the Z-up world) from the line between the hips."""
    left = pos[idx[fmt["L"]["hip"]]] - pos[idx[fmt["R"]["hip"]]]
    fwd = np.cross(left, [0.0, 0.0, 1.0])
    return np.arctan2(fwd[1], fwd[0])


def standing_reference(b: Bvh, fmt_name: str, base: tuple[Bvh, int] | None = None):
    """The standing reference {name: (position, quaternion wxyz)} (Z-up, facing +x, floor z = 0).
    From the rest pose, or with ``base`` = (a Bvh of the same skeleton, frame) from that frame."""
    fmt = FORMATS[fmt_name]
    idx = {n: i for i, n in enumerate(b.names)}
    if base is None:
        lrot, lpos = local_transforms(b, rest=True)
    else:
        src, frame = base
        if src.names != b.names or np.abs(src.offsets[1:] - b.offsets[1:]).max() > 1e-3:
            raise ValueError("the base pose must come from the same skeleton")
        lrot, lpos = local_transforms(src)
        lrot = [r[[frame]] for r in lrot]
        lpos = lpos[frame:frame + 1].copy()
        lpos[:, 0] = 0.0
    grot, gpos = fk(b, lrot, lpos)
    grot = [g[0] for g in grot]
    gpos = gpos[0].copy()
    down = YUP_TO_ZUP.inv().apply([0.0, 0.0, -1.0])  # straight down in the BVH world
    v = gpos[idx[fmt["neck"]]] - gpos[idx[fmt["pelvis"]]]
    swing = R.align_vectors([-down], [v / np.linalg.norm(v)])[0]
    pivot = gpos[idx[fmt["pelvis"]]].copy()
    for j in subtree(b, idx[fmt["spine"]]):
        gpos[j] = pivot + swing.apply(gpos[j] - pivot)
        grot[j] = swing * grot[j]
    for side in ("L", "R"):
        for a, c in (("hip", "ankle"), ("shoulder", "wrist")):
            ja, jc = idx[fmt[side][a]], idx[fmt[side][c]]
            v = gpos[jc] - gpos[ja]
            swing = R.align_vectors([down], [v / np.linalg.norm(v)])[0]
            pivot = gpos[ja].copy()
            for j in subtree(b, ja):
                gpos[j] = pivot + swing.apply(gpos[j] - pivot)
                grot[j] = swing * grot[j]
    pos = YUP_TO_ZUP.apply(gpos)
    rots = [YUP_TO_ZUP * g for g in grot]
    yaw = _facing_yaw(pos, idx, fmt)
    rz = R.from_euler("z", -yaw)
    pos = rz.apply(pos)
    pos -= [pos[idx[fmt["pelvis"]], 0], pos[idx[fmt["pelvis"]], 1], pos[:, 2].min()]
    return {n: (pos[i], (rz * rots[i]).as_quat(scalar_first=True)) for i, n in enumerate(b.names)}


# A steeper floor is terrain, not a tilted capture floor (in CMU's 2548 clips,
# 89 tilt more than 3 degrees; most of those described are slopes, stairs,
# platforms or flips).
MAX_LEVEL_DEG = 3.0


def level_rotation(pos, legs, leg_len):
    """The rotation that levels the floor: a plane fitted to the lower envelope of each frame's lowest leg joint.
    The slope is fitted only along the directions in which the motion spreads over at least two leg lengths
    (the xy principal axes; along a narrow one the difference between the feet dominates)."""
    low = pos[np.arange(len(pos)), np.asarray(legs)[pos[:, legs, 2].argmin(1)]]
    xy = low[:, :2] - low[:, :2].mean(0)
    _, _, vt = np.linalg.svd(xy, full_matrices=False)
    axes = [u for u in vt if np.ptp(xy @ u) >= 2 * leg_len]
    if not axes:
        return R.identity(), 0.0
    A = np.c_[np.stack([xy @ u for u in axes], 1), np.ones(len(low))]
    keep = np.ones(len(low), bool)
    for _ in range(4):
        coef = np.linalg.lstsq(A[keep], low[keep, 2], rcond=None)[0]
        res = low[:, 2] - A @ coef
        keep = res <= np.percentile(res, 50)
    grad = sum(c * u for c, u in zip(coef[:-1], axes))
    n = np.array([-grad[0], -grad[1], 1.0])
    rot = R.align_vectors([[0.0, 0.0, 1.0]], [n / np.linalg.norm(n)])[0]
    return rot, float(np.degrees(rot.magnitude()))


def motion_frames(b: Bvh, fmt_name: str, fps_out: float):
    """Each frame {name: (position, quaternion wxyz)} (Z-up).  The first frame faces +x with the pelvis at
    x = y = 0; a floor tilt of at most MAX_LEVEL_DEG is levelled, and the clip's lowest leg joint is put at
    z = 0 (the world is re-framed; the pose is unchanged)."""
    fmt = FORMATS[fmt_name]
    idx = {n: i for i, n in enumerate(b.names)}
    lrot, lpos = local_transforms(b)
    grot, gpos = fk(b, lrot, lpos)
    step = max(1, int(round(1.0 / b.frame_time / fps_out)))
    sel = np.arange(fmt["skip_first"], b.motion.shape[0], step)
    pos = YUP_TO_ZUP.apply(gpos[sel].reshape(-1, 3)).reshape(len(sel), -1, 3)
    yaw = _facing_yaw(pos[0], idx, fmt)
    rz = R.from_euler("z", -yaw)
    pos = rz.apply(pos.reshape(-1, 3)).reshape(pos.shape)
    legs = sorted(set(subtree(b, idx[fmt["L"]["hip"]]) + subtree(b, idx[fmt["R"]["hip"]])))
    leg_len = np.linalg.norm(b.offsets[subtree(b, idx[fmt["L"]["hip"]])[1:3]], axis=1).sum()
    rl, tilt_deg = level_rotation(pos, legs, leg_len)
    if tilt_deg > MAX_LEVEL_DEG:  # levelling a slope or stairs would tilt the body
        rl, tilt_deg = R.identity(), 0.0
    pos = rl.apply(pos.reshape(-1, 3)).reshape(pos.shape)
    pos -= [pos[0, idx[fmt["pelvis"]], 0], pos[0, idx[fmt["pelvis"]], 1], pos[:, legs, 2].min()]
    quats = np.stack([(rl * rz * YUP_TO_ZUP * grot[j][sel]).as_quat(scalar_first=True) for j in range(len(b.names))], axis=1)
    frames = [{n: (pos[t, j], quats[t, j]) for j, n in enumerate(b.names)} for t in range(len(sel))]
    return frames, 1.0 / (b.frame_time * step), tilt_deg
