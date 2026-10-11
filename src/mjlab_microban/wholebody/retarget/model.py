"""The robot model the retargeting solves on, and the training collision shapes.

The IK model is robot.xml with a site at every joint centre GMR matches (the
point where two joint axes come closest), oriented with the world axes at
HOME (x forward, y left, z up).  It is built from robot.xml at run time and
written once per process to a temporary file, because GMR loads its model
from a path.  The collision shapes are the whole-body task's
(``training_collision_model``), so the floor and the self-collision pairs of
the retargeting are the ones the training data are cut with.
"""

from __future__ import annotations

import atexit
import functools
import shutil
import tempfile
from pathlib import Path

import mujoco
import numpy as np

from mjlab_microban.robot.home_pose import MICROBAN_XML, load_home_pose


def home_qpos(model: mujoco.MjModel) -> np.ndarray:
    """qpos of HOME (config/home_pose.yaml) on ``model``."""
    home = load_home_pose()
    q = model.qpos0.copy()
    q[:3] = home.root_pos
    q[3:7] = home.root_quat_wxyz
    for name, value in home.joint_pos_rad.items():
        q[model.joint(name).qposadr[0]] = value
    return q


def _closest_points(p1, d1, p2, d2):
    """Midpoint of the closest points of two lines."""
    w = p1 - p2
    a, b, c = d1 @ d1, d1 @ d2, d2 @ d2
    d, e = d1 @ w, d2 @ w
    den = a * c - b * b
    s = (b * e - c * d) / den
    t = (a * e - b * d) / den
    return 0.5 * ((p1 + s * d1) + (p2 + t * d2))


def joint_centres(model: mujoco.MjModel) -> dict[str, tuple[str, np.ndarray]]:
    """Site name -> (body, world position at HOME)."""
    d = mujoco.MjData(model)
    d.qpos[:] = home_qpos(model)
    mujoco.mj_forward(model, d)

    def axis(joint):
        jid = model.joint(joint).id
        return d.xanchor[jid].copy(), d.xaxis[jid].copy()

    centres = {}
    for side in ("left", "right"):
        hip = _closest_points(*axis(f"{side}_hip_pitch"), *axis(f"{side}_hip_yaw"))
        ankle = _closest_points(*axis(f"{side}_ankle_pitch"), *axis(f"{side}_ankle_roll"))
        knee = _closest_points(*axis(f"{side}_knee"), hip, ankle - hip)
        shoulder = _closest_points(*axis(f"{side}_shoulder_pitch"), *axis(f"{side}_shoulder_roll"))
        hand = d.site_xpos[model.site(f"{side}_hand").id].copy()
        elbow = _closest_points(*axis(f"{side}_elbow"), shoulder, hand - shoulder)
        centres[f"{side}_hip_c"] = (f"femur__configuration_{side}", hip)
        centres[f"{side}_knee_c"] = (f"tibia__configuration_{side}", knee)
        centres[f"{side}_ankle_c"] = ("foot_2" if side == "left" else "foot", ankle)
        centres[f"{side}_shoulder_c"] = (f"humerus__configuration_{side}", shoulder)
        centres[f"{side}_elbow_c"] = ("radius_2" if side == "left" else "radius", elbow)
    centres["neck_c"] = ("camera_cover", _closest_points(*axis("neck_roll"), *axis("neck_pitch")))
    centres["trunk_c"] = ("trunk", d.xpos[model.body("trunk").id].copy())
    return centres


def gmr_spec() -> mujoco.MjSpec:
    """robot.xml with the joint-centre sites (group 3)."""
    model = mujoco.MjModel.from_xml_path(str(MICROBAN_XML))
    d = mujoco.MjData(model)
    d.qpos[:] = home_qpos(model)
    mujoco.mj_kinematics(model, d)
    spec = mujoco.MjSpec.from_file(str(MICROBAN_XML))
    spec.meshdir = str(Path(MICROBAN_XML).parent / "assets")
    for name, (body, world) in joint_centres(model).items():
        bid = model.body(body).id
        local = d.xmat[bid].reshape(3, 3).T @ (world - d.xpos[bid])
        world_aligned = np.empty(4)
        mujoco.mju_negQuat(world_aligned, d.xquat[bid])
        spec.body(body).add_site(name=name, pos=local, quat=world_aligned, group=3)
    return spec


@functools.cache
def gmr_model_xml() -> Path:
    """The IK model, written once per process (GMR loads its model from a file)."""
    folder = Path(tempfile.mkdtemp(prefix="microban_gmr_"))
    atexit.register(shutil.rmtree, folder, True)
    path = folder / "microban_gmr.xml"
    path.write_text(gmr_spec().to_xml())
    return path


@functools.cache
def training_collision_geoms() -> tuple[dict, ...]:
    """The whole-body task's collision geoms: name, body, type (box or cylinder), size, pos, quat (body frame)."""
    from mjlab_microban.wholebody.env_cfg import training_collision_model

    model = training_collision_model()
    kinds = {int(mujoco.mjtGeom.mjGEOM_BOX): "box", int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder"}
    out = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == 0 or not (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        out.append({
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g),
            "body": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])),
            "type": kinds[int(model.geom_type[g])],
            "size": model.geom_size[g].tolist(),
            "pos": model.geom_pos[g].tolist(),
            "quat": model.geom_quat[g].tolist(),
        })
    return tuple(out)
