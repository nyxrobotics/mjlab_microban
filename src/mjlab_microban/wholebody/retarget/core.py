"""Retarget human motion to Microban with GMR's IK.

The body difference is taken up by the human's standing reference (made
from the skeleton alone, ``skeleton.standing_reference``) against Microban's
HOME: the standing reference retargets exactly to HOME (joint error 0, root
error 0).

* Orientations: GMR's orientation offset (the fifth entry of an
  ik_match_table row) is R_off = R_human_ref^-1 * R_HOME per part.
* Trunk: Microban's trunk is one rigid body from pelvis to head, so its
  orientation comes from points, not from the human pelvis.  The tilt
  (pitch, roll) is the direction pelvis -> neck base; the yaw about that line
  is the line between the shoulders (the arms have no twist joint at the
  shoulder, so the trunk follows the shoulders and the hip yaw takes the
  pelvis twist).  The rotation from the standing reference (spine straight
  up) is applied to HOME's trunk (10 degrees forward lean).
* Ankles: as in GMR, the distance from the pelvis is scaled by the root scale
  (the leg length ratio) and the difference left at the reference (31-36 mm
  sideways) is added.  That difference turns with the human foot, but moves
  per frame no more than the human ankle (at robot size), so a planted foot's
  target stays put.
* The other positions (hip, knee, shoulder, elbow, hand): instead of GMR's
  "distance from the pelvis x scale", HOME's Microban segments (e.g. hip
  centre -> knee centre) are turned by the change of the human bone and
  chained.  Limb bones turn by their smallest rotation without twist, taken
  in the trunk frame (independent of the heading; a segment not parallel to
  the human bone does not move with the bone's twist).  Segments on the trunk
  (hips, shoulders) turn with the trunk.  The part of a segment not parallel
  to the human bone (HOME's elbow bend, say) fades once the bone is more than
  90 degrees from its reference and is 0 when opposite (arm straight up): the
  smallest rotation has no defined axis there and that part would swing with
  the axis and jump.
* Joints a human cannot bend that way: inside the IK only, the elbow bends
  the human way (<= 0) and the knee stops at straight (>= 0) (with the trunk
  on the human spine line, a deep forward bend puts the hip pitch at its -90
  degree limit and the knee would bend backwards to make up), and the hip yaw
  turns the leg outward by at most 105 degrees.  The robot's hip yaw runs
  from 60 degrees inward to 240 degrees outward, the latter 120 degrees
  inward the other way round.  With the hip roll at its 25 degrees, the yaw
  is the only joint that points a raised thigh sideways, so a thigh swung out
  to the side and on behind (CMU's wild-leg walk 132_56, kicks, animal
  imitations) was followed outward through 90 degrees, past where the
  inward end would have stopped it, up to the 240 degree end (past 120
  degrees in 0.3 % of CMU's evaluation frames).  105 degrees is the farthest LAFAN1's motions turn a
  leg outward (104), so they are unchanged.
* What the servos can do: between two frames (h = 1 / fps) a joint moves at
  most the XC330's speed x h and changes its per-frame motion by at most its
  acceleration x h^2 (``ServoLimit``; the values and their sources are in
  config/retarget_limits.yaml).  A motion faster than the servos is followed
  late; without it the IK swung a joint by up to 91 degrees in one frame
  (the hip yaw of CMU 132_56) and branch switches of the solution reached the
  data as joint jumps.
* Self-collision: mink's CollisionAvoidanceLimit in the IK (no closer than
  2 mm, active from 30 mm; it cannot push apart a pair already in contact, so
  the small overlaps the box approximation lets in remain and are cut on the
  training side).  The pairs: each leg's shin, ankle block and foot against
  its hip block and femur (a deep knee bend puts the heel into the hip
  block), the two legs' lower parts and the two femurs, legs against the
  trunk, forearms against the trunk; the shapes are the training collision
  shapes.  An IK step that cannot be solved with the limit is solved without it.
* Arms: two shoulder axes and one elbow, so turning the shoulder by 180
  degrees and bending the elbow the other way gives nearly the same elbow and
  hand positions.  The elbow bends the human way only (above); with the arm
  near sideways (along the shoulder pitch axis) the position does not fix the
  shoulder pitch and the IK drifts to the other branch, so the upper arm's
  orientation (the human upper-arm twist) is matched with a weight that grows
  as the arm comes sideways.  The first frame after a reset is also solved
  from the other branch and the better one is kept; later frames keep the
  branch they are on (a switch would be a jump the servos cannot make; the
  servo limits let an arm come round only at their speed).
* Root: the human pelvis motion scaled by the root scale (GMR's
  human_scale_table).
* Floor: each frame's lowest point of Microban (the training collision
  shapes) is put at the human clearance x the root scale.  The clearance is
  the smaller of the feet's (ankle and toe, against their height in the
  standing reference) and of the other parts that can touch the floor
  (pelvis, spine root, neck base, shoulders, elbows, wrists, knees, against
  their height when on the floor): lying, crawling and hands on the floor.
* Neck: the human head's turn from the standing reference, relative to the
  trunk frame (above), carried onto Microban's trunk as it stands at HOME and
  split into head (yaw), neck_roll and neck_pitch (the joints turn the head
  by Rz Rx Ry).  The angles are a posture target of the three joints in the
  IK, so the servo limits and the joint ranges hold for them as for the
  others.  The G1 recordings have no head: their head is the trunk frame, so
  the neck stays at HOME.

Outputs of ``retarget_frames`` (in ``out_dir``): ``<name>.npz`` (qpos, joint
names, the joints in HOME's order, the targets, the human skeleton for
drawing, per-frame marks), ``<name>.pkl`` (GMR's PKL with ``joint_names``,
read by wholebody/data/build.py), ``<name>_ik.json`` (the IK configuration)
and ``<name>_report.json`` (the numbers).  CPU only.
"""

from __future__ import annotations

import json
import pickle
import re
import time
from pathlib import Path

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from mjlab_microban.policy_contract import HEAD_JOINTS
from mjlab_microban.robot.home_pose import HOME_JOINT_NAMES
from mjlab_microban.wholebody.retarget import smplx
from mjlab_microban.wholebody.retarget.gmr import GeneralMotionRetargeting
from mjlab_microban.wholebody.retarget.gmr import params as gparams
from mjlab_microban.wholebody.retarget.model import (
    gmr_model_xml,
    home_qpos,
    retarget_limits,
    training_collision_geoms,
)
from mjlab_microban.wholebody.retarget.skeleton import (
    FORMATS,
    motion_frames,
    read_bvh,
    standing_reference,
)

ROBOT = "microban"

# Microban's targets (a body or a site) and the human roles.  A position is
# "the parent role's position + the segment", turned by the driver role's
# bone.  Without a driver the position is GMR's (distance from the pelvis x
# root scale + offset).  skeleton.FORMATS turns roles into bone names.
TARGETS = [
    # robot frame,       type,   role,         parent,       driver
    ("trunk",            "body", "pelvis",     None,         None),
    ("left_hip_c",       "site", "L.hip",      "pelvis",     "pelvis"),
    ("left_knee_c",      "site", "L.knee",     "L.hip",      "L.hip"),
    ("left_ankle_c",     "site", "L.ankle",    None,         None),
    ("left_shoulder_c",  "site", "L.shoulder", "pelvis",     "pelvis"),
    ("left_elbow_c",     "site", "L.elbow",    "L.shoulder", "L.shoulder"),
    ("left_hand",        "site", "L.wrist",    "L.elbow",    "L.elbow"),
    ("right_hip_c",      "site", "R.hip",      "pelvis",     "pelvis"),
    ("right_knee_c",     "site", "R.knee",     "R.hip",      "R.hip"),
    ("right_ankle_c",    "site", "R.ankle",    None,         None),
    ("right_shoulder_c", "site", "R.shoulder", "pelvis",     "pelvis"),
    ("right_elbow_c",    "site", "R.elbow",    "R.shoulder", "R.shoulder"),
    ("right_hand",       "site", "R.wrist",    "R.elbow",    "R.elbow"),
]

# Weights ((table 1 position, orientation), (table 2 position, orientation)),
# after GMR's bvh_lafan1_to_g1.json.
# - Position weights are multiplied by the G1/Microban leg length ratio (5.3):
#   position errors are in m and orientation errors in rad.
# - The trunk orientation is G1's pelvis + torso_link (table 1 10 + 100,
#   table 2 5 + 10); Microban's trunk is one body.
# - The arm positions are the elbow and the hand.  The shoulder (upper arm)
#   has an orientation only; its weight is table 2's value x (the sideways
#   part of the upper arm's target direction)^2, set every frame (about 0
#   with the arm down or forward, so it does not pull against the positions).
#   No wrist, so the hand's twist is not retargeted.
WEIGHTS = {
    "pelvis":   ((0, 110), (100, 15)),
    "hip":      ((0, 10), (10, 5)),
    "knee":     ((0, 10), (10, 5)),
    "ankle":    ((50, 10), (50, 10)),
    "shoulder": ((0, 0), (0, 4)),
    "elbow":    ((0, 0), (10, 0)),
    "wrist":    ((0, 0), (10, 0)),
}
# Joint ranges inside the IK only (the ways a human joint does not bend); the
# robot's ranges (elbow +-125, knee -45..135, hip yaw -60..240 outward degrees) are unchanged.
ELBOW_MAX_DEG = 0.0  # the elbow bends the human way only (HOME is -20)
KNEE_MIN_DEG = 0.0  # the knee does not bend past straight (HOME's 0)
# The hip yaw turns a leg outward (left +, right -) by at most this (config/retarget_limits.yaml: LAFAN1's largest, 104
# degrees), not round to the robot's 240 degree end (see the module docstring).  On CMU's frames that went past it
# (0.15 % of the leg frames, 49 clips) the knee and ankle are then 26.5 mm from their targets on average (28.9 mm
# before); LAFAN1 does not change.
HIP_YAW_OUT_MAX_DEG = float(retarget_limits()["robot"]["hip_yaw_outward_max_deg"]["value"])
# The neck's posture target in the IK (its three joints only; they move nothing the other targets see).
NECK_WEIGHT = 10.0
G1_LEG_M = 0.654  # GMR's g1_mocap_29dof.xml, hip_pitch_link to ankle_roll_link height
SOLE_GEOMS = {s: [f"{s}_foot_collision_{i}" for i in range(1, 7)] for s in ("left", "right")}
# Joint heights of the parts other than the feet when they lie on the floor,
# as a share of the reference leg length (hip -> ankle), above the toe-base
# floor: the lowest 0.5 % of each part's height in LAFAN1's lying and falling
# clips (ground, fallAndGetUp, pushAndFall; 13 clips).
CONTACT_RADIUS = {"pelvis": 0.09, "spine": 0.06, "neck": 0.12, "shoulder": 0.05, "elbow": -0.02, "wrist": -0.02, "knee": 0.01}


def _register_robot() -> str:
    gparams.ROBOT_XML_DICT[ROBOT] = gmr_model_xml()
    return str(gparams.ROBOT_XML_DICT[ROBOT])


class LowestPoint:
    """The lowest point of the training collision shapes (boxes and cylinders), as preprocess.ClipChecker."""

    def __init__(self, model: mujoco.MjModel):
        geoms = training_collision_geoms()
        self.body = np.array([model.body(g["body"]).id for g in geoms])
        self.pos = np.array([g["pos"] for g in geoms])
        self.rot = R.from_quat([g["quat"] for g in geoms], scalar_first=True).as_matrix()
        self.size = np.array([g["size"] for g in geoms])
        self.box = np.array([g["type"] == "box" for g in geoms])

    def __call__(self, data: mujoco.MjData) -> float:
        xm = data.xmat[self.body].reshape(-1, 3, 3)
        z = data.xpos[self.body, 2] + np.einsum("gj,gj->g", xm[:, 2, :], self.pos)
        rz = np.einsum("gj,gjk->gk", xm[:, 2, :], self.rot)
        box = z - np.sum(np.abs(rz) * self.size, axis=-1)
        axis = np.abs(rz[:, 2])
        cyl = z - axis * self.size[:, 1] - self.size[:, 0] * np.sqrt(np.clip(1.0 - axis**2, 0.0, 1.0))
        return float(np.min(np.where(self.box, box, cyl)))


def role_name(fmt: dict, role: str) -> str:
    if "." in role:
        side, part = role.split(".")
        return fmt[side][part]
    return fmt[role]


def frame_pose(model, data, name, ftype):
    if ftype == "body":
        i = model.body(name).id
        return data.xpos[i].copy(), data.xquat[i].copy()
    i = model.site(name).id
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, data.site_xmat[i])
    return data.site_xpos[i].copy(), q


def trunk_frame(data, pelvis, neck, left, right):
    """The trunk frame (columns x forward, y left, z up): z is pelvis -> neck base, y the left-right line made normal to z."""
    z = data[neck][0] - data[pelvis][0]
    z = z / np.linalg.norm(z)
    y = data[left][0] - data[right][0]
    y = y - y.dot(z) * z
    y = y / np.linalg.norm(y)
    return np.stack([np.cross(y, z), y, z], axis=1)


def calibrate(ref: dict, fmt_name: str, model: mujoco.MjModel) -> dict:
    """GMR's IK configuration (and the segment chain) that puts the standing reference ``ref`` on HOME."""
    fmt = FORMATS[fmt_name]
    pelvis = fmt["pelvis"]
    d = mujoco.MjData(model)
    d.qpos[:] = home_qpos(model)
    mujoco.mj_kinematics(model, d)
    home = {role: frame_pose(model, d, f, t) for f, t, role, *_ in TARGETS}
    leg_robot = home["L.hip"][0][2] - home["L.ankle"][0][2]
    leg_human = ref[fmt["L"]["hip"]][0][2] - ref[fmt["L"]["ankle"]][0][2]
    root_scale = leg_robot / leg_human
    pos_mult = G1_LEG_M / leg_robot
    t1, t2, chain, uniform = {}, {}, [], []
    for frame, ftype, role, parent, driver in TARGETS:
        hb = role_name(fmt, role)
        q_home = R.from_quat(home[role][1], scalar_first=True)
        q_off = (R.from_quat(ref[hb][1], scalar_first=True).inv() * q_home).as_quat(scalar_first=True).tolist()
        p_off = [0.0, 0.0, 0.0]
        if parent is None and role != "pelvis":
            # GMR's: the distance from the pelvis scaled by the root scale, plus the difference left at the reference (world frame)
            p_scaled = home["pelvis"][0] + root_scale * (ref[hb][0] - ref[pelvis][0])
            uniform.append({"human": hb, "offset_ref": (home[role][0] - p_scaled).tolist(), "ref_quat": ref[hb][1].tolist()})
        (p1, r1), (p2, r2) = WEIGHTS[role.split(".")[-1]]
        t1[frame] = [hb, p1 * pos_mult, r1, p_off, q_off, ftype]
        t2[frame] = [hb, p2 * pos_mult, r2, p_off, q_off, ftype]
        if parent is not None:
            db = role_name(fmt, driver)
            axis = None
            if driver == parent and driver != "pelvis":  # a limb bone: its direction (this joint -> next) in the bone frame
                a = ref[hb][0] - ref[db][0]
                axis = R.from_quat(ref[db][1], scalar_first=True).inv().apply(a / np.linalg.norm(a)).tolist()
            chain.append({"human": hb, "parent": role_name(fmt, parent), "driver": db,
                          "vec_home": (home[role][0] - home[parent][0]).tolist(),
                          "driver_ref_quat": ref[db][1].tolist(), "driver_axis_local": axis})
    return {
        "robot_root_name": "trunk",
        "human_root_name": pelvis,
        "ground_height": 0.0,
        "human_height_assumption": 1.0,
        "use_ik_match_table1": True,
        "use_ik_match_table2": True,
        "human_scale_table": {pelvis: root_scale},
        "ik_match_table1": t1,
        "ik_match_table2": t2,
        # Microban's additions (read by MicrobanRetargeting.scale_human_data).
        "chain": chain,
        "uniform": uniform,
        "root_ref": {"human_pos": ref[pelvis][0].tolist(), "robot_pos": home["pelvis"][0].tolist()},
        "trunk": {"pelvis_ref_quat": ref[pelvis][1].tolist(), "neck": fmt["neck"],
                  "lateral": [fmt["L"]["shoulder"], fmt["R"]["shoulder"]],
                  "frame_ref": trunk_frame(ref, pelvis, fmt["neck"], fmt["L"]["shoulder"], fmt["R"]["shoulder"]).tolist()},
        # The neck: the human head at the reference and Microban's trunk at HOME in the human trunk frame's axes.
        "neck": {"head": fmt["head"], "head_ref_quat": ref[fmt["head"]][1].tolist(),
                 "trunk_in_frame": (trunk_frame(ref, pelvis, fmt["neck"], fmt["L"]["shoulder"], fmt["R"]["shoulder"]).T
                                    @ R.from_quat(home["pelvis"][1], scalar_first=True).as_matrix()).tolist()},
        "info": {"leg_robot_m": leg_robot, "leg_human": leg_human, "pos_weight_mult": pos_mult},
    }


_SOLVE_IK = mink.solve_ik
COLLISION_FALLBACKS = [0]


def _solve_ik_with_fallback(configuration, tasks, dt, solver, damping, limits=None, **kw):
    """Solve an IK step that has no solution with the collision limit without it (counted in COLLISION_FALLBACKS)."""
    try:
        return _SOLVE_IK(configuration, tasks, dt, solver, damping, limits=limits, **kw)
    except mink.exceptions.NoSolutionFound:
        if not limits or not any(isinstance(lim, mink.CollisionAvoidanceLimit) for lim in limits):
            raise
        COLLISION_FALLBACKS[0] += 1
        return _SOLVE_IK(configuration, tasks, dt, solver, damping,
                         limits=[lim for lim in limits if not isinstance(lim, mink.CollisionAvoidanceLimit)], **kw)


mink.solve_ik = _solve_ik_with_fallback  # GMR calls mink.solve_ik too


def self_collision_pairs(model: mujoco.MjModel):
    """The collision geom pairs the IK keeps apart (mink.CollisionAvoidanceLimit's geom_pairs).  Parts by the
    body name's prefix, the side by the joints (the configuration_left/right of a part's name is not its side:
    ankle_block__configuration_left is on the right leg)."""
    def side(b):
        while b > 0:
            for j in range(model.body_jntnum[b]):
                n = model.joint(model.body_jntadr[b] + j).name
                if n.startswith(("left_", "right_")):
                    return n.split("_")[0]
            b = model.body_parentid[b]
        return None
    groups = {}
    for g in training_collision_geoms():
        if "floor_collision" in g["name"]:
            continue
        b = model.body(g["body"]).id
        gid = next(i for i in range(model.ngeom) if model.geom_bodyid[i] == b and np.allclose(model.geom_size[i], g["size"], atol=1e-6)
                   and np.allclose(model.geom_pos[i], g["pos"], atol=1e-6))
        part = next(k for k in ("hip_block", "femur", "tibia", "ankle_block", "foot", "radius", "trunk") if g["body"].startswith(k))
        part = "lower" if part in ("tibia", "ankle_block", "foot") else "upper" if part == "hip_block" else part
        groups.setdefault((side(b), part), []).append(gid)
    pairs = []
    for s in ("left", "right"):
        pairs.append((groups[(s, "lower")], groups[(s, "upper")] + groups[(s, "femur")]))
        pairs.append((groups[(s, "femur")] + groups[(s, "lower")], groups[(None, "trunk")]))
        pairs.append((groups[(s, "radius")], groups[(None, "trunk")]))
    pairs.append((groups[("left", "lower")], groups[("right", "lower")]))
    pairs.append((groups[("left", "femur")], groups[("right", "femur")]))
    return pairs


class JointTask(mink.Task):
    """Some hinge joints towards target angles (error q - target over those joints only, so the IK's convergence
    test sees no other joint)."""

    def __init__(self, model: mujoco.MjModel, joints, cost: float):
        super().__init__(cost=np.full(len(joints), cost))
        self.qpos = np.array([model.joint(n).qposadr[0] for n in joints])
        self.jacobian = np.eye(model.nv)[[model.joint(n).dofadr[0] for n in joints]]
        self.target = np.zeros(len(joints))

    def compute_error(self, configuration):
        return configuration.q[self.qpos] - self.target

    def compute_jacobian(self, configuration):
        return self.jacobian


class ServoLimit(mink.Limit):
    """The hinge joints within their ranges and within what the servos can do between two frames (h = 1 / fps):
    from the previous frame at most the speed x h, and from the previous frame's motion at most the acceleration x
    h^2 more or less (config/retarget_limits.yaml), and no faster towards an end of the range than lets the joint
    stop there at that acceleration (so the three can always be met; should rounding part them, the range, the
    speed and the stop hold).  After a reset only the range holds (the first frame places the robot), on the
    frame after it the speed and the stop too.  It takes the place of mink's ConfigurationLimit: a hinge
    integrates exactly, so the box needs no gain.

    Per frame: ``begin`` sets the box, ``end`` records the frame's angles and returns whether the speed and whether
    the acceleration bound (at the bound, 1e-6 rad)."""

    def __init__(self, model: mujoco.MjModel, fps: float):
        servo = retarget_limits()["robot"]["servo"]
        hinges = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        self.qpos = model.jnt_qposadr[hinges]
        rows = np.eye(model.nv)[model.jnt_dofadr[hinges]]
        self.G = np.vstack([rows, -rows])
        self.lo, self.hi = model.jnt_range[hinges, 0].copy(), model.jnt_range[hinges, 1].copy()
        self.step = np.full(len(hinges), float(servo["speed_rad_s"]["value"]) / fps)
        accel = servo["acceleration_rad_s2"]["value"]
        self.turn = np.array([float(accel[model.joint(j).name]) for j in hinges]) / fps**2
        self.reset()

    def reset(self) -> None:
        self.previous: list[np.ndarray] = []  # the last two frames' angles, the latest last
        self.begin()

    def begin(self) -> None:
        lower, upper = self.lo, self.hi
        if self.previous:
            centre = self.previous[-1]
            # Moving s per frame (a per frame squared), a joint stops within s^2 / (2 a) after this frame's s; keeping
            # s + s^2 / (2 a) within the room to the end leaves the next frame's s - a its room too.
            stop = lambda room: self.turn * (np.sqrt(1.0 + 2.0 * np.maximum(room, 0.0) / self.turn) - 1.0)
            lower = np.maximum(lower, centre - np.minimum(self.step, stop(centre - self.lo)))
            upper = np.minimum(upper, centre + np.minimum(self.step, stop(self.hi - centre)))
            if len(self.previous) == 2:
                motion = self.previous[-1] - self.previous[-2]
                low, high = np.maximum(lower, centre + motion - self.turn), np.minimum(upper, centre + motion + self.turn)
                met = low <= high
                lower, upper = np.where(met, low, lower), np.where(met, high, upper)
        self.lower, self.upper = lower, upper

    def end(self, q: np.ndarray) -> tuple[bool, bool]:
        angles = q[self.qpos]
        speed = accel = False
        if self.previous:
            move = angles - self.previous[-1]
            speed = bool(np.any(np.abs(move) >= self.step - 1e-6))
            if len(self.previous) == 2:
                accel = bool(np.any(np.abs(move - (self.previous[-1] - self.previous[-2])) >= self.turn - 1e-6))
        self.previous = [*self.previous[-1:], angles.copy()]
        return speed, accel

    def compute_qp_inequalities(self, configuration, dt):
        del dt
        angles = configuration.q[self.qpos]
        return mink.Constraint(G=self.G, h=np.concatenate([self.upper - angles, angles - self.lower]))


class MicrobanRetargeting(GeneralMotionRetargeting):
    """GMR with its position scaling (scale_human_data) replaced and the arm branch handling added.
    The IK, the orientation and position offsets and tables 1 and 2 are GMR's."""

    def __init__(self, cfg: dict, cfg_path: Path, fmt_name: str, fps: float, **kw):
        _register_robot()
        gparams.IK_CONFIG_DICT.setdefault(fmt_name, {})[ROBOT] = cfg_path
        super().__init__(src_human=fmt_name, tgt_robot=ROBOT, actual_human_height=None, verbose=False, **kw)
        self.root_scale = cfg["human_scale_table"][cfg["human_root_name"]]
        self.root_ref_h = np.asarray(cfg["root_ref"]["human_pos"])
        self.root_ref_r = np.asarray(cfg["root_ref"]["robot_pos"])
        tr = cfg["trunk"]
        self.pelvis_ref = R.from_quat(tr["pelvis_ref_quat"], scalar_first=True)
        self.neck, self.lateral = tr["neck"], tr["lateral"]
        self.frame_ref = np.asarray(tr["frame_ref"])
        self.uniform = [(u["human"], np.asarray(u["offset_ref"]), R.from_quat(u["ref_quat"], scalar_first=True).inv())
                        for u in cfg["uniform"]]
        self.ankle_offset = {}  # human ankle -> (current offset (world), the human ankle's previous position)
        self.chain = []
        for c in cfg["chain"]:
            q_ref = R.from_quat(c["driver_ref_quat"], scalar_first=True)
            axis = None if c["driver_axis_local"] is None else np.asarray(c["driver_axis_local"])
            a_ref = None if axis is None else q_ref.apply(axis)
            self.chain.append((c["human"], c["parent"], c["driver"], np.asarray(c["vec_home"]), axis, a_ref))
        # GMR keeps no offsets for a part whose table 1 weights are 0 and then fails in offset_human_data; take them from table 1.
        for e in cfg["ik_match_table1"].values():
            self.pos_offsets1.setdefault(e[0], np.asarray(e[3]) - self.ground)
            self.rot_offsets1.setdefault(e[0], R.from_quat(e[4], scalar_first=True))
        for side in ("left", "right"):
            self.model.jnt_range[self.model.joint(f"{side}_elbow").id, 1] = np.radians(ELBOW_MAX_DEG)
            self.model.jnt_range[self.model.joint(f"{side}_knee").id, 0] = np.radians(KNEE_MIN_DEG)
            # Outward is + for the left hip yaw and - for the right.
            self.model.jnt_range[self.model.joint(f"{side}_hip_yaw").id, int(side == "left")] = (
                np.radians(HIP_YAW_OUT_MAX_DEG) * (1.0 if side == "left" else -1.0))
        neck = cfg["neck"]
        self.head = neck["head"]
        self.head_ref = R.from_matrix(self.frame_ref).inv() * R.from_quat(neck["head_ref_quat"], scalar_first=True)
        self.trunk_in_frame = R.from_matrix(np.asarray(neck["trunk_in_frame"]))
        self.neck_task = JointTask(self.model, HEAD_JOINTS, NECK_WEIGHT)
        self.tasks1.append(self.neck_task)
        self.tasks2.append(self.neck_task)
        self.servo = ServoLimit(self.model, fps)
        self.limits_plain = [self.servo]
        self.limits_coll = self.limits_plain + [mink.CollisionAvoidanceLimit(self.model, self_collision_pairs(self.model),
                                                                            minimum_distance_from_collisions=0.002,
                                                                            collision_detection_distance=0.03)]
        self.ik_limits = self.limits_coll
        t1, t2 = cfg["ik_match_table1"], cfg["ik_match_table2"]
        # (human shoulder, human elbow, table 2's upper-arm orientation task, its weight,
        #  qpos addresses (pitch, roll, elbow), roll sign)
        self.arms = []
        for side, sign in (("left", 1.0), ("right", -1.0)):
            task = next(t for t in self.tasks2 if t.frame_name == f"{side}_shoulder_c")
            adr = [self.model.joint(f"{side}_{j}").qposadr[0] for j in ("shoulder_pitch", "shoulder_roll", "elbow")]
            self.arms.append((t1[f"{side}_shoulder_c"][0], t1[f"{side}_elbow_c"][0], task, t2[f"{side}_shoulder_c"][2], adr, sign))

    def reset_configuration(self):
        super().reset_configuration()
        self.ankle_offset = {}
        self.servo.reset()

    def retarget(self, human_data, offset_to_ground=False):
        """The frame's qpos; ``servo_bound`` is then whether the servos' speed and whether their acceleration
        bounded it."""
        first = self._warmup_needed
        self.servo.begin()
        q = super().retarget(human_data, offset_to_ground)
        if first:
            q = self._pick_arm_branch(range(len(self.arms)))
        self.servo_bound = self.servo.end(q)
        return q

    def _cost2(self):
        """Table 2's weighted squared error (what the IK minimises)."""
        return float(sum(np.sum((t.cost * t.compute_error(self.configuration)) ** 2) for t in self.tasks2))

    def _pick_arm_branch(self, arms):
        """Per arm, solve table 2 again from the other branch (shoulder pitch + 180 degrees, roll to the other
        side, elbow reversed) and keep the lower IK cost.  Used on the first frame after a reset (the servo limits
        hold from the next frame on); the next frames start from the kept solution.  Solved as GMR's warm-up."""
        dt = self._warmup_dt
        for _, _, _, _, (a_p, a_r, a_e), sign in (self.arms[i] for i in arms):
            cur = self.configuration.q.copy()
            c_cur = self._cost2()
            tw = cur.copy()
            tw[a_p] = (cur[a_p] + 2 * np.pi) % (2 * np.pi) - np.pi
            tw[a_r] = sign * np.pi - cur[a_r]
            tw[a_e] = min(-cur[a_e], np.radians(ELBOW_MAX_DEG))
            self.configuration.update(q=tw)
            err = self.error2()
            for _ in range(self._warmup_max_iter):
                vel = mink.solve_ik(self.configuration, self.tasks2, dt, self.solver, self.damping, limits=self.ik_limits)
                self.configuration.integrate_inplace(vel, dt)
                nxt = self.error2()
                if err - nxt <= 0.001:
                    break
                err = nxt
            if self._cost2() >= c_cur:
                self.configuration.update(q=cur)
        return self.configuration.data.qpos.copy()

    def scale_human_data(self, human_data, human_root_name, human_scale_table):
        root_pos = human_data[human_root_name][0]
        F = trunk_frame(human_data, human_root_name, self.neck, *self.lateral)
        T = R.from_matrix(F @ self.frame_ref.T)  # the trunk's rotation from the reference
        root_t = self.root_ref_r + self.root_scale * (root_pos - self.root_ref_h)
        out = {human_root_name: (root_t, (T * self.pelvis_ref).as_quat(scalar_first=True))}
        # The neck: the head's turn from the reference in the trunk frame's axes, on Microban's trunk.
        turn = R.from_matrix(F).inv() * R.from_quat(human_data[self.head][1], scalar_first=True) * self.head_ref.inv()
        self.neck_task.target = (self.trunk_in_frame.inv() * turn * self.trunk_in_frame).as_euler("ZXY")
        for hb, off_ref, ref_inv in self.uniform:
            # The difference left at the reference turns with the human foot, moving per frame no more than the human ankle (at robot size).
            w = (R.from_quat(human_data[hb][1], scalar_first=True) * ref_inv).apply(off_ref)
            prev = self.ankle_offset.get(hb)
            if prev is not None:
                step = self.root_scale * np.linalg.norm(human_data[hb][0] - prev[1])
                dw = w - prev[0]
                n = np.linalg.norm(dw)
                w = prev[0] + (dw if n <= step else dw * (step / n))
            self.ankle_offset[hb] = (w, np.array(human_data[hb][0]))
            out[hb] = (root_t + self.root_scale * (human_data[hb][0] - root_pos) + w, human_data[hb][1])
        for hb, parent, driver, vec, axis, a_ref in self.chain:
            if axis is None:  # a segment on the trunk (hips, shoulders): turns with the trunk
                seg = T.apply(vec)
            else:  # a limb bone: the smallest rotation without twist, in the trunk frame (independent of the heading)
                d = T.inv().apply(R.from_quat(human_data[driver][1], scalar_first=True).apply(axis))
                par = float(vec @ a_ref)
                w = min(1.0, max(0.0, 1.0 + float(d @ a_ref)))  # the sideways part fades past 90 degrees and is 0 when opposite
                side = R.align_vectors([d], [a_ref])[0].apply(vec - par * a_ref) if w > 0.0 else 0.0
                seg = T.apply(par * d + w * side)
            out[hb] = (out[parent][0] + seg, human_data[hb][1])
        # The upper-arm orientation weight: proportional to the squared sideways part of the upper arm's target
        # direction in the trunk frame (sideways, the shoulder pitch becomes the upper-arm twist and positions do not fix it).
        for sh, el, task, weight, *_ in self.arms:
            d = T.inv().apply(out[el][0] - out[sh][0])
            task.set_orientation_cost(weight * (d[1] / np.linalg.norm(d)) ** 2)
        return out


def human_clearance(frames, ref, fmt, fps, window_s=1.0):
    """The human body's clearance above the floor (in the source unit).  The ankle and toe base against their
    height in the standing reference, the other parts (CONTACT_RADIUS) against their height on the floor; h is
    the smallest (small when only a heel, a hand or the back is down).  The floor is the minimum of h within
    +-window_s, and the clearance is h minus it (absorbing floor unevenness, capture drift and the error of the
    part heights while keeping jumps shorter than a second)."""
    leg = ref[fmt["L"]["hip"]][0][2] - ref[fmt["L"]["ankle"]][0][2]
    floor_ref = min(ref[fmt[s]["toe"]][0][2] for s in "LR")
    parts = [(fmt[s][p], ref[fmt[s][p]][0][2]) for s in "LR" for p in ("ankle", "toe")]
    parts += [(role_name(fmt, r if r in ("pelvis", "spine", "neck") else f"{s}.{r}"), floor_ref + c * leg)
              for r, c in CONTACT_RADIUS.items() for s in (("L",) if r in ("pelvis", "spine", "neck") else ("L", "R"))]
    h = np.array([min(f[n][0][2] - z for n, z in parts) for f in frames])
    k = max(1, int(round(window_s * fps)))
    floor = np.array([h[max(0, t - k):t + k + 1].min() for t in range(len(h))])
    return h - floor, floor


def to_input(frame):
    return {k: [v[0].copy(), v[1].copy()] for k, v in frame.items()}


def sole_min_z(model, data, side=None):
    zs = []
    for s, geoms in SOLE_GEOMS.items():
        if side not in (None, s):
            continue
        for g in geoms:
            gid = model.geom(g).id
            c, m, h = data.geom_xpos[gid], data.geom_xmat[gid].reshape(3, 3), model.geom_size[gid]
            for sx in (-1, 1):
                for sy in (-1, 1):
                    for sz in (-1, 1):
                        zs.append((c + m @ (h * [sx, sy, sz]))[2])
    return min(zs)


def contact_slide(model, qs, frames, ref, fmt, scale, fps, floor):
    """While a human foot is down (its clearance below 0.3 units; floor as in human_clearance), the median and
    90 % point of the ankle's horizontal speed difference |v_Microban - v_human x scale| (mm/s)."""
    d = mujoco.MjData(model)
    ank = {"left": [], "right": []}
    for q in qs:
        d.qpos[:] = q
        mujoco.mj_kinematics(model, d)
        for side in ank:
            ank[side].append(d.site_xpos[model.site(f"{side}_ankle_c").id][:2].copy())
    out = {}
    for S, side in (("L", "left"), ("R", "right")):
        a = np.array([f[fmt[S]["ankle"]][0] for f in frames])
        t = np.array([f[fmt[S]["toe"]][0] for f in frames])
        h = np.minimum(a[:, 2] - ref[fmt[S]["ankle"]][0][2], t[:, 2] - ref[fmt[S]["toe"]][0][2]) - floor
        cont = (h[1:] < 0.3) & (h[:-1] < 0.3)
        dv = np.linalg.norm(np.diff(np.array(ank[side]), axis=0) - np.diff(a[:, :2], axis=0) * scale, axis=1) * fps * 1000
        out[side] = [round(float(np.median(dv[cont])), 1), round(float(np.percentile(dv[cont], 90)), 1)] if cont.any() else None
    return out


def load_motion(path: str | Path, fmt_name: str, fps_out: float, smplx_model_dir: str | Path | None = None):
    """A BVH file (or an AMASS SMPL-X npz with fmt_name "smplx") as (frames, fps, standing reference,
    removed floor tilt in degrees, the skeleton for drawing (names, parents))."""
    fmt = FORMATS[fmt_name]
    if fmt_name == "smplx":
        if smplx_model_dir is None:
            raise ValueError("SMPL-X motions need the body model folder")
        b = smplx.load_amass(path, smplx_model_dir, fps_out)
    else:
        b = read_bvh(str(path))
    base = None
    if "ref_frames" in fmt:  # a skeleton whose rest is no standing pose: from each subject's standing frame
        name, frame = fmt["ref_frames"][re.search(r"subject\d+", Path(path).stem).group(0)]
        base = (read_bvh(str(Path(path).parent / name)), frame)
    ref = standing_reference(b, fmt_name, base)
    if "head_neutral" in fmt:  # the reference's head turned to the subject's looking ahead (the neck's 0)
        subject = re.search(r"subject\d+", Path(path).stem).group(0)
        frame = trunk_frame(ref, fmt["pelvis"], fmt["neck"], fmt["L"]["shoulder"], fmt["R"]["shoulder"])
        head = R.from_matrix(frame) * R.from_quat(fmt["head_neutral"][subject], scalar_first=True)
        ref[fmt["head"]] = (ref[fmt["head"]][0], head.as_quat(scalar_first=True))
    frames, fps, tilt_deg = motion_frames(b, fmt_name, fps_out)
    return frames, fps, ref, tilt_deg, (list(b.names), b.parents)


def run(path: str | Path, fmt_name: str, fps_out: float, out_dir: Path, stem: str | None = None,
        smplx_model_dir: str | Path | None = None) -> dict:
    frames, fps, ref, tilt_deg, skel = load_motion(path, fmt_name, fps_out, smplx_model_dir)
    return retarget_frames(stem or Path(path).stem, fmt_name, frames, fps, ref, Path(out_dir), tilt_deg, skel)


def arm_flips(rob, tgt, trunk_quat, roll_deg):
    """Frames whose arm is on the other branch: the target elbow bends 30+ degrees while the robot's elbow bends
    90+ degrees away from the target's bend, or the shoulder roll is past 90 degrees while the target upper arm
    points down in the trunk frame (z < -0.3).  rob, tgt: (T, 3, 3) shoulder, elbow and hand positions."""
    def ang(a, b):
        return np.degrees(np.arccos(np.clip((a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1), -1, 1)))
    ut, ft = tgt[:, 1] - tgt[:, 0], tgt[:, 2] - tgt[:, 1]
    ur, fr = rob[:, 1] - rob[:, 0], rob[:, 2] - rob[:, 1]
    bt = ft - (ft * ut).sum(1, keepdims=True) * ut / (ut * ut).sum(1, keepdims=True)
    br = fr - (fr * ur).sum(1, keepdims=True) * ur / (ur * ur).sum(1, keepdims=True)
    wrong = (ang(ut, ft) > 30) & (ang(bt, br) > 90) & (ang(ut, ur) < 20)
    ul = R.from_quat(trunk_quat, scalar_first=True).inv().apply(ut)
    ul /= np.linalg.norm(ul, axis=1, keepdims=True)
    return wrong | ((np.abs(roll_deg) > 90) & (ul[:, 2] < -0.3))


def longest_run(mask) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def retarget_frames(stem: str, fmt_name: str, frames: list, fps: float, ref: dict, out_dir: Path,
                    tilt_deg: float = 0.0, skel=None) -> dict:
    """Retarget human frames ({name: (position, quaternion wxyz)}, Z-up, floor z = 0) and write the npz, PKL,
    IK configuration and report to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = FORMATS[fmt_name]
    model = mujoco.MjModel.from_xml_path(_register_robot())
    cfg = calibrate(ref, fmt_name, model)
    scale = cfg["human_scale_table"][fmt["pelvis"]]
    cfg_path = out_dir / f"{stem}_ik.json"
    cfg_path.write_text(json.dumps(cfg, indent=1))

    gmr = MicrobanRetargeting(cfg, cfg_path, fmt_name, fps)
    # Check: the standing reference retargets to HOME.
    gmr.reset_configuration()
    for _ in range(20):
        q_ref = gmr.retarget(to_input(ref))
    q_home = home_qpos(model)
    ref_joint_err = float(np.degrees(np.abs(q_ref[7:] - q_home[7:]).max()))
    ref_root_err = float(np.linalg.norm(q_ref[:3] - q_home[:3]))

    gmr.reset_configuration()
    fallbacks0 = COLLISION_FALLBACKS[0]
    qs, targets, target_quats, bound = [], [], [], []
    t0 = time.perf_counter()
    for f in frames:
        qs.append(gmr.retarget(to_input(f)).copy())
        bound.append(gmr.servo_bound)
        targets.append({k: np.asarray(v[0]).copy() for k, v in gmr.scaled_human_data.items()})
        target_quats.append({k: np.asarray(v[1]).copy() for k, v in gmr.scaled_human_data.items()})
    ik_fps = len(qs) / (time.perf_counter() - t0)
    qs = np.asarray(qs)
    jn = [model.joint(j).name for j in range(1, model.njnt)]
    # The frames the servos' speed, their acceleration and the hip yaw's outward limit bounded.
    speed_bound, accel_bound = np.asarray(bound, dtype=bool).reshape(-1, 2).T
    outward = np.stack([qs[:, 7 + jn.index("left_hip_yaw")], -qs[:, 7 + jn.index("right_hip_yaw")]], 1)
    hip_yaw_bound = np.any(outward >= np.radians(HIP_YAW_OUT_MAX_DEG) - 1e-6, axis=1)
    # The IK holds the joints in range only to its solver's tolerance (mink's limits let them past an end by up to
    # 5e-6 rad in LAFAN1, CMU and PICO): the angles are clipped to the robot's ranges (the MJCF's, not the IK's).
    lo, hi = model.jnt_range[1:, 0], model.jnt_range[1:, 1]
    qs[:, 7:] = np.clip(qs[:, 7:], lo, hi)

    # Floor: each frame's lowest point (training collision shapes) at the human clearance x root scale
    # (the joints are unchanged; only the root height moves).
    d = mujoco.MjData(model)
    lowest = LowestPoint(model)
    lows = []
    for q in qs:
        d.qpos[:] = q
        mujoco.mj_kinematics(model, d)
        lows.append(lowest(d))
    clear, floor = human_clearance(frames, ref, fmt, fps)
    dz = np.asarray(lows) - scale * clear
    qs[:, 2] -= dz
    for tg, z in zip(targets, dz):
        for k in tg:
            tg[k] = tg[k] - [0.0, 0.0, z]

    # The numbers.
    perr = {f: [] for f, *_ in TARGETS}
    rerr = {f: [] for f, *_ in TARGETS}
    sole = {"left": [], "right": []}
    arm_sites = {s: (f"{s}_shoulder_c", f"{s}_elbow_c", f"{s}_hand") for s in ("left", "right")}
    rob_arm = {s: [] for s in arm_sites}
    trunk_q, low_after = [], []
    for q, tg, tq in zip(qs, targets, target_quats):
        d.qpos[:] = q
        mujoco.mj_kinematics(model, d)
        for f, t, role, *_ in TARGETS:
            p_, q_ = frame_pose(model, d, f, t)
            hb = role_name(fmt, role)
            perr[f].append(np.linalg.norm(p_ - tg[hb]))
            rerr[f].append(np.degrees((R.from_quat(q_, scalar_first=True).inv() * R.from_quat(tq[hb], scalar_first=True)).magnitude()))
        for side in sole:
            sole[side].append(sole_min_z(model, d, side))
        for s, names in arm_sites.items():
            rob_arm[s].append([d.site_xpos[model.site(n).id].copy() for n in names])
        trunk_q.append(d.xquat[model.body("trunk").id].copy())
        low_after.append(lowest(d))
    lows = np.minimum(sole["left"], sole["right"])
    low_after = np.asarray(low_after)
    w = {f: WEIGHTS[role.split(".")[-1]] for f, _, role, *_ in TARGETS}
    near = (np.minimum(qs[:, 7:] - lo, hi - qs[:, 7:]) < np.radians(1.0)).sum(0)
    deg = np.degrees(qs[:, 7:])
    # Arm on the other branch (per frame, either arm).
    tf = [f for f, *_ in TARGETS]
    tg_arr = np.array([[tg[role_name(fmt, r)] for _, _, r, *_ in TARGETS] for tg in targets])
    flip = np.zeros(len(qs), bool)
    for s, names in arm_sites.items():
        flip |= arm_flips(np.array(rob_arm[s]), tg_arr[:, [tf.index(n) for n in names]], np.array(trunk_q),
                          deg[:, jn.index(f"{s}_shoulder_roll")])
    # Joint speed and jumps (as preprocess.joint_jumps: faster than 15 rad/s and more than twice the steps around it).
    step = np.abs(np.diff(qs[:, 7:], axis=0))
    speed = step.max(1) * fps if len(step) else np.zeros(1)
    padded = np.pad(step, ((1, 1), (0, 0)))
    jump = np.any((step * fps > 15.0) & (step > 2.0 * np.maximum(padded[:-2], padded[2:])), axis=1)
    # Target jumps: a limb segment's target (upper arm, forearm, thigh) turns 5+ degrees more than the human bone
    # in one frame (the shin is left out: the ankle target is GMR's scaled position, not a chained segment).
    def turn(v):
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        return np.degrees(np.arccos(np.clip((v[1:] * v[:-1]).sum(1), -1, 1)))
    tjump = np.zeros(max(len(qs) - 1, 0), bool)
    for s in ("L", "R"):
        side = "left" if s == "L" else "right"
        for a, b, fa, fb in (("shoulder", "elbow", "shoulder_c", "elbow_c"), ("elbow", "wrist", "elbow_c", "hand"),
                             ("hip", "knee", "hip_c", "knee_c")):
            ha, hb_ = fmt[s][a], fmt[s][b]
            hv = np.array([fr[hb_][0] - fr[ha][0] for fr in frames])
            tv = tg_arr[:, tf.index(f"{side}_{fb}")] - tg_arr[:, tf.index(f"{side}_{fa}")]
            tjump |= (turn(tv) - turn(hv)) > 5.0
    hand_err = np.maximum(perr["left_hand"], perr["right_hand"])
    report = {
        "clip": stem, "format": fmt_name, "frames": len(qs), "fps": round(fps, 3), "ik_fps_cpu": round(ik_fps, 1),
        "ik_steps_without_collision_limit": COLLISION_FALLBACKS[0] - fallbacks0,
        "floor_tilt_removed_deg": round(tilt_deg, 2),
        "root_scale_m_per_unit": round(scale, 6),
        "pos_weight_mult": round(cfg["info"]["pos_weight_mult"], 3),
        "ref_to_home_max_joint_err_deg": round(ref_joint_err, 4),
        "ref_to_home_root_err_mm": round(1000 * ref_root_err, 3),
        "pos_err_mm_mean_p95": {f.replace("_c", ""): [round(1000 * float(np.mean(v)), 1), round(1000 * float(np.percentile(v, 95)), 1)]
                                for f, v in perr.items() if w[f][1][0] > 0},
        "rot_err_deg_mean_p95": {f.replace("_c", ""): [round(float(np.mean(v)), 1), round(float(np.percentile(v, 95)), 1)]
                                 for f, v in rerr.items() if w[f][1][1] > 0},
        "lowest_sole_mm_p50_p95": [round(1000 * float(np.percentile(lows, p)), 1) for p in (50, 95)],
        "lowest_point_mm_min_p50_p99_max": [round(1000 * float(v), 1) for v in
                                            (low_after.min(), np.percentile(low_after, 50), np.percentile(low_after, 99), low_after.max())],
        "contact_ankle_slide_vs_human_mm_s_med_p90": contact_slide(model, qs, frames, ref, fmt, scale, fps, floor),
        "joint_speed_rad_s_p99_max": [round(float(np.percentile(speed, 99)), 2), round(float(speed.max()), 2)],
        "joint_jump_frames": int(jump.sum()),
        "servo_speed_bound_frames": int(speed_bound.sum()),
        "servo_accel_bound_frames": int(accel_bound.sum()),
        "hip_yaw_bound_frames": int(hip_yaw_bound.sum()),
        "target_jump_frames": int(tjump.sum()),
        "arm_flip_frames_longest": [int(flip.sum()), longest_run(flip)],
        "hand_err_mm_p99_max": [round(1000 * float(np.percentile(hand_err, 99)), 1), round(1000 * float(np.max(hand_err)), 1)],
        "frames_within_1deg_of_limit_any": int((np.minimum(qs[:, 7:] - lo, hi - qs[:, 7:]) < np.radians(1.0)).any(1).sum()),
        "joint_deg_min_max": {n: [round(float(deg[:, i].min()), 1), round(float(deg[:, i].max()), 1)] for i, n in enumerate(jn)},
        "frames_within_1deg_of_limit": {jn[i]: int(c) for i, c in enumerate(near) if c > 0},
        "root_z_m_min_max": [round(float(qs[:, 2].min()), 3), round(float(qs[:, 2].max()), 3)],
        "root_xy_end_m": np.round(qs[-1, :2], 3).tolist(),
    }
    home_order = list(HOME_JOINT_NAMES)
    extra = {}
    if skel is not None:
        names, parents = skel
        extra = {"human_pos": (np.array([[f[n][0] for n in names] for f in frames]) * scale).astype(np.float32),
                 "human_parents": parents, "human_names": np.array(names)}
    np.savez(out_dir / f"{stem}.npz",
             qpos=qs, fps=fps, joint_names=np.array(jn),
             joint_pos_home_order=qs[:, [7 + jn.index(n) for n in home_order]], home_joint_names=np.array(home_order),
             targets=tg_arr.astype(np.float32), target_frames=np.array(tf),
             flip=flip, joint_jump=jump, target_jump=tjump, speed_bound=speed_bound, accel_bound=accel_bound,
             hip_yaw_bound=hip_yaw_bound, hand_err=np.asarray(hand_err, np.float32),
             lowest_point=low_after.astype(np.float32), **extra)
    with open(out_dir / f"{stem}.pkl", "wb") as fh:  # the shape of GMR's scripts/bvh_to_robot.py output
        pickle.dump({"fps": fps, "root_pos": qs[:, :3], "root_rot": qs[:, [4, 5, 6, 3]],  # xyzw
                     "dof_pos": qs[:, 7:], "joint_names": jn, "local_body_pos": None, "link_body_list": None}, fh)
    (out_dir / f"{stem}_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    return report
