"""AMASS SMPL-X motions (``*_stageii.npz``) as a ``skeleton.Bvh``.

As a Bvh, an AMASS motion goes through the same standing reference, framing
(facing, floor, tilt), IK and drawing as a BVH file.  Only the body joints
0-21 are used; the joint positions follow from the rest joints of the
subject's shape (``betas``) and the joint rotations, so the body model is read
once per file.  The motion is resampled to exactly ``fps_out`` (the training
side reads integer rates only; AMASS rates differ per source).

Checked on a synthetic body model and motion only (tests); the keys of the
real SMPL-X v1.1 model and of AMASS files are as published, not yet seen.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from mjlab_microban.wholebody.retarget.skeleton import YUP_TO_ZUP, Bvh

BODY = ["pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle", "right_ankle",
        "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head", "left_shoulder",
        "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"]  # SMPL-X joints 0-21


def _dense(x):
    x = x.item() if x.dtype == object else x
    return np.asarray(x.todense()) if hasattr(x, "todense") else np.asarray(x)


def rest_joints(model_dir: str | Path, gender: str, betas: np.ndarray):
    """The rest joints of a body shape (SMPL-X frame: Y up, facing +Z) and their parents."""
    m = np.load(Path(model_dir) / f"SMPLX_{gender.upper()}.npz", allow_pickle=True)
    v = m["v_template"] + m["shapedirs"][:, :, :len(betas)] @ betas
    J = _dense(m["J_regressor"]) @ v  # pose correctives move the surface only, not the joints
    parents = m["kintree_table"][0].astype(int)
    parents[0] = -1
    return J[:len(BODY)], parents[:len(BODY)]


def load_amass(path: str | Path, model_dir: str | Path, fps_out: float = 30.0) -> Bvh:
    d = np.load(path, allow_pickle=True)
    g = np.asarray(d["gender"]).item()
    g = g.decode() if isinstance(g, bytes) else str(g)
    J, parents = rest_joints(model_dir, g, np.asarray(d["betas"], float))
    src = float(np.asarray(d["mocap_frame_rate"]))
    n = len(d["trans"])
    rv = np.concatenate([d["root_orient"], d["pose_body"]], axis=1).reshape(n, -1, 3)  # (T, 22, 3)
    t_src = np.arange(n) / src
    t = np.arange(0.0, t_src[-1] + 1e-9, 1.0 / fps_out)
    rots = [Slerp(t_src, R.from_rotvec(rv[:, j]))(t) for j in range(len(BODY))]
    trans = np.stack([np.interp(t, t_src, d["trans"][:, k]) for k in range(3)], axis=1)
    to_y = YUP_TO_ZUP.inv()  # AMASS's Z-up world -> the BVH's Y-up (motion_frames turns it back)
    cols = [to_y.apply(J[0] + trans), (to_y * rots[0]).as_euler("ZYX", degrees=True)]
    cols += [rots[j].as_euler("ZYX", degrees=True) for j in range(1, len(BODY))]
    offsets = J - J[np.maximum(parents, 0)]
    offsets[0] = 0.0
    rot3 = ["Zrotation", "Yrotation", "Xrotation"]
    channels = [["Xposition", "Yposition", "Zposition"] + rot3] + [rot3] * (len(BODY) - 1)
    return Bvh(list(BODY), parents, offsets, channels, np.concatenate(cols, axis=1), 1.0 / fps_out)
