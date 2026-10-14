"""Public PICO recordings (joint angles retargeted to G1) as human frames for the evaluation set.

G1's forward kinematics turns each recording into pseudo-bones (the "g1"
format of skeleton.FORMATS), which go through the same retargeting as human
motion.  The recordings (``<raw>/``, placed by hand; docs/wholebody_data.md):

* ``teleopit/pico_record_shard_000.h5``: Teleopit's HDF5 (9 clips).  The
  source clips (``source_clip_*``) are used; ``clip_*`` are overlapping
  windows.  The joints are G1's 29 in g1_mocap_29dof.xml's qpos order.
* ``twist2/*.pkl``: TWIST2's GMR PKLs (root_rot xyzw), at 29.7-29.9 fps:
  resampled to 30 fps (linear, root orientation by slerp) because the
  training side reads integer rates only.
* ``any4hdmi/manifest.json`` and ``any4hdmi/*.npz``: qpos (root 7 + 29
  joints in the manifest's order) at 30 fps.

G1 bodies per role (position body / orientation body): pelvis / pelvis, neck
base = midpoint of the shoulder_roll_links / torso_link, hip hip_pitch_link /
hip_yaw_link, knee knee_link, ankle ankle_roll_link, toe toe_link /
ankle_roll_link, shoulder shoulder_roll_link / shoulder_yaw_link, elbow
elbow_link, wrist wrist_roll_link / wrist_yaw_link.  A root teleport (the
pelvis moving more than 0.15 m in one frame, 4.5 m/s) cuts a recording, and
so do the frames whose waist yaw turns the torso from the pelvis past the
human trunk's rotation (config/retarget_limits.yaml, human trunk rotation:
64.6 degrees; TWIST2's G1 holds its waist at the 90 degree end through some
walks, which turned Microban's legs 105+ degrees out); the parts of at least
one second become separate clips (``_p<n>``).  The standing
reference of each source is a quiet standing frame (chosen as for LAFAN1),
made like the human one (spine up, legs and arms down); G1's zero pose has
the elbows at 90 degrees and is not used.
"""

from __future__ import annotations

import functools
import json
import pickle
from pathlib import Path

import h5py
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from mjlab_microban.wholebody.retarget.model import retarget_limits

FPS = 30.0
TELEPORT_M = 0.15
G1_LEG_M = 0.654

# Pseudo-bone -> (position body, orientation body); "neck" is made from the shoulders.
BONES = {
    "pelvis": ("pelvis", "pelvis"), "torso": ("torso_link", "torso_link"),
    **{f"{s}_{p}": (f"{side}_{a}", f"{side}_{b}") for s, side in (("L", "left"), ("R", "right")) for p, a, b in (
        ("hip", "hip_pitch_link", "hip_yaw_link"), ("knee", "knee_link", "knee_link"), ("ankle", "ankle_roll_link", "ankle_roll_link"),
        ("toe", "toe_link", "ankle_roll_link"), ("shoulder", "shoulder_roll_link", "shoulder_yaw_link"),
        ("elbow", "elbow_link", "elbow_link"), ("wrist", "wrist_roll_link", "wrist_yaw_link"))},
}
NAMES = list(BONES) + ["neck"]
PARENT = {"pelvis": None, "torso": "pelvis", "neck": "torso",
          **{f"{s}_{p}": f"{s}_{q}" if q else "pelvis" for s in "LR" for p, q in (("hip", None), ("knee", "hip"), ("ankle", "knee"), ("toe", "ankle"))},
          **{f"{s}_{p}": (f"{s}_{q}" if q else "neck") for s in "LR" for p, q in (("shoulder", None), ("elbow", "shoulder"), ("wrist", "elbow"))}}
PARENTS = np.array([NAMES.index(PARENT[n]) if PARENT[n] else -1 for n in NAMES])
UPPER = ["torso", "neck"] + [f"{s}_{p}" for s in "LR" for p in ("shoulder", "elbow", "wrist")]


@functools.cache
def g1_model(xml: str) -> mujoco.MjModel:
    """G1 without its meshes (kinematics only; the meshes are not needed)."""
    spec = mujoco.MjSpec.from_file(xml)
    for geom in list(spec.geoms):
        if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
            spec.delete(geom)
    for mesh in list(spec.meshes):
        spec.delete(mesh)
    return spec.compile()


def g1_joint_names(model: mujoco.MjModel) -> list[str]:
    return [model.joint(j).name for j in range(1, model.njnt)]


def trunk_turn_kept(jp, g1_joints) -> np.ndarray:
    """(T,) the waist yaw (the torso's turn from the pelvis) within the human trunk's rotation range."""
    rom = retarget_limits()["human"]["joints"]["trunk"]["rotation"]["rom"]["adopted"]
    right = -np.degrees(np.asarray(jp)[:, g1_joints.index("waist_yaw_joint")])  # G1's yaw turns left; the range's + is right
    return (right >= rom["lower"]) & (right <= rom["upper"])


def pieces(root_pos, fps, kept=None):
    """[start, end) parts between root teleports (more than TELEPORT_M in one frame) and the frames not ``kept``,
    at least one second long."""
    n = len(root_pos)
    step = np.linalg.norm(np.diff(np.asarray(root_pos), axis=0), axis=1)
    kept = np.ones(n, dtype=bool) if kept is None else np.asarray(kept, dtype=bool)
    cuts = np.zeros(n + 1, dtype=bool)
    cuts[1:-1] = step > TELEPORT_M
    parts, start = [], None
    for t in range(n + 1):
        if t == n or not kept[t] or cuts[t]:
            if start is not None and t - start >= fps:
                parts.append((start, t))
            start = t if t < n and kept[t] else None
        elif start is None:
            start = t
    return parts


def resample(p, fps, name):
    """TWIST2: a non-integer rate to 30 fps (linear; the root orientation by slerp)."""
    t = np.arange(len(p["root_pos"])) / fps
    tn = np.arange(0.0, t[-1] + 1e-9, 1.0 / FPS)
    rot = Slerp(t, R.from_quat(p["root_rot"]))(tn)  # xyzw
    lin = lambda x: np.stack([np.interp(tn, t, x[:, k]) for k in range(x.shape[1])], 1)
    return ("twist2", name, FPS, lin(p["root_pos"]), rot.as_quat(scalar_first=True), lin(p["dof_pos"]))


def _split(src, name, fps, rp, rq, jp, g1_joints):
    return [(src, name + (f"_p{k}" if (a, b) != (0, len(rp)) else ""), fps, rp[a:b], rq[a:b], jp[a:b])
            for k, (a, b) in enumerate(pieces(rp, fps, trunk_turn_kept(jp, g1_joints)))]


def load_sources(raw: Path, g1_joints: list[str]):
    """[(source, name, fps, root position (T,3), root orientation wxyz (T,4), joints (T,29))]."""
    out = []
    shard = raw / "teleopit" / "pico_record_shard_000.h5"
    if shard.exists():
        with h5py.File(shard) as f:
            for i, (s, n) in enumerate(zip(f["source_clip_starts"][()], f["source_clip_lengths"][()])):
                if int(f["source_clip_fps"][i]) != 30:
                    raise ValueError(f"{shard}: source clip {i} is not 30 fps")
                out += _split("teleopit", f"picoteleopit_rec{i:02d}", 30.0, f["root_pos"][s:s + n].astype(float),
                              f["root_quat_w"][s:s + n].astype(float), f["joint_pos"][s:s + n].astype(float), g1_joints)
    for fn in sorted((raw / "twist2").glob("*.pkl")):
        with open(fn, "rb") as stream:
            p0 = pickle.load(stream)
        fps = float(p0["fps"])
        for k, (a, b) in enumerate(pieces(p0["root_pos"], fps, trunk_turn_kept(p0["dof_pos"], g1_joints))):
            p = {key: p0[key][a:b] for key in ("root_pos", "root_rot", "dof_pos")}
            out.append(resample(p, fps, f"picotwist_{fn.stem}" + (f"_p{k}" if (a, b) != (0, len(p0["root_pos"])) else "")))
    manifest = raw / "any4hdmi" / "manifest.json"
    if manifest.exists():
        man = json.loads(manifest.read_text())
        if man["qpos_names"][7:] != g1_joints:
            raise ValueError("any4hdmi's joint order is not G1's")
        fps = round(1.0 / man["timestep"], 6)
        for c in man["source"]["clips"]:
            q = np.load(raw / "any4hdmi" / f"{c['name']}.npz")["qpos"].astype(float)
            out += _split("any4hdmi", f"picoany_{c['name'].replace('-', '_')}", fps, q[:, :3], q[:, 3:7], q[:, 7:], g1_joints)
    return out


def g1_frames(model: mujoco.MjModel, root_pos, root_quat, jp):
    """Pseudo-bone frames ({name: (position, quaternion wxyz)}) from G1's forward kinematics."""
    d = mujoco.MjData(model)
    pid = {k: model.body(a).id for k, (a, _) in BONES.items()}
    oid = {k: model.body(b).id for k, (_, b) in BONES.items()}
    frames = []
    for t in range(len(jp)):
        d.qpos[:3] = root_pos[t]
        d.qpos[3:7] = root_quat[t] / np.linalg.norm(root_quat[t])
        d.qpos[7:] = jp[t]
        mujoco.mj_kinematics(model, d)
        f = {k: (d.xpos[pid[k]].copy(), d.xquat[oid[k]].copy()) for k in BONES}
        f["neck"] = (0.5 * (f["L_shoulder"][0] + f["R_shoulder"][0]), f["torso"][1].copy())
        frames.append(f)
    return frames


def standing_reference(frame):
    """The standing reference from one frame (as skeleton.standing_reference: spine up, legs and arms down,
    facing +x, pelvis at x = y = 0, lowest point z = 0)."""
    pos = {k: v[0].copy() for k, v in frame.items()}
    rot = {k: R.from_quat(v[1], scalar_first=True) for k, v in frame.items()}

    def swing(names, pivot, a, b, target):
        v = pos[b] - pos[a]
        r = R.align_vectors([target], [v / np.linalg.norm(v)])[0]
        p0 = pos[pivot].copy()
        for n in names:
            pos[n] = p0 + r.apply(pos[n] - p0)
            rot[n] = r * rot[n]
    swing(UPPER, "pelvis", "pelvis", "neck", [0, 0, 1.0])
    for s in "LR":
        swing([f"{s}_{p}" for p in ("hip", "knee", "ankle", "toe")], f"{s}_hip", f"{s}_hip", f"{s}_ankle", [0, 0, -1.0])
        swing([f"{s}_{p}" for p in ("shoulder", "elbow", "wrist")], f"{s}_shoulder", f"{s}_shoulder", f"{s}_wrist", [0, 0, -1.0])
    left = pos["L_hip"] - pos["R_hip"]
    fwd = np.cross(left, [0, 0, 1.0])
    rz = R.from_euler("z", -np.arctan2(fwd[1], fwd[0]))
    zmin = min(v[2] for v in pos.values())
    c = pos["pelvis"].copy()
    return {k: (rz.apply(pos[k] - [c[0], c[1], 0]) - [0, 0, zmin], (rz * rot[k]).as_quat(scalar_first=True)) for k in pos}


def normalize(frames):
    """The first frame facing +x, the pelvis at x = y = 0, the clip's lowest leg point at z = 0
    (skeleton.motion_frames' framing; the floor tilt is not levelled)."""
    f0 = frames[0]
    left = f0["L_hip"][0] - f0["R_hip"][0]
    fwd = np.cross(left, [0, 0, 1.0])
    rz = R.from_euler("z", -np.arctan2(fwd[1], fwd[0]))
    legs = [f"{s}_{p}" for s in "LR" for p in ("hip", "knee", "ankle", "toe")]
    zmin = min(f[n][0][2] for f in frames for n in legs)
    c = f0["pelvis"][0]
    off = np.array([c[0], c[1], 0.0])
    return [{k: (rz.apply(v[0] - off) - [0, 0, zmin], (rz * R.from_quat(v[1], scalar_first=True)).as_quat(scalar_first=True))
             for k, v in f.items()} for f in frames]


def _angle(u, v):
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    return np.degrees(np.arccos(np.clip((u * v).sum(-1), -1, 1)))


def stance_candidates(frames, fps):
    """Quiet standing frames (LAFAN1's conditions, lengths as a share of G1's leg): [(motion, frame, knee)]."""
    P = lambda n: np.array([f[n][0] for f in frames])
    kL = _angle(P("L_knee") - P("L_hip"), P("L_ankle") - P("L_knee"))
    kR = _angle(P("R_knee") - P("R_hip"), P("R_ankle") - P("R_knee"))
    spine = _angle(P("neck") - P("pelvis"), np.array([0, 0, 1.0]))
    arm = np.maximum(_angle(P("L_elbow") - P("L_shoulder"), np.array([0, 0, -1.0])),
                     _angle(P("R_elbow") - P("R_shoulder"), np.array([0, 0, -1.0])))
    allp = np.concatenate([P(n) for n in NAMES], 1)
    v = np.r_[0, np.linalg.norm(np.diff(allp, axis=0), axis=1)] * fps / G1_LEG_M
    feet = np.abs(P("L_ankle")[:, 2] - P("R_ankle")[:, 2]) / G1_LEG_M
    ok = (v < 0.15) & (feet < 0.01) & (spine < 10) & (np.abs(kL - kR) < 5) & (arm < 35)
    return [(float(v[t]), int(t), float((kL[t] + kR[t]) / 2)) for t in np.where(ok)[0]]


def references(sources, model):
    """Each source's standing reference: the candidate whose knee is nearest the source's median, preferring still frames."""
    cand = {}
    for src, name, fps, rp, rq, jp in sources:
        fr = normalize(g1_frames(model, rp, rq, jp))
        for v, t, kn in stance_candidates(fr, fps):
            cand.setdefault(src, []).append((v, name, t, kn, fr[t]))
    refs, chosen = {}, {}
    for src, c in cand.items():
        med = np.median([x[3] for x in c])
        v, name, t, kn, f = min(c, key=lambda x: abs(x[3] - med) + 20 * x[0])
        refs[src] = standing_reference(f)
        chosen[src] = {"clip": name, "frame": t, "knee_deg": round(kn, 1), "candidates": len(c), "median_knee_deg": round(float(med), 1)}
    return refs, chosen
