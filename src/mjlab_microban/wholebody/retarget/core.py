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
  degree limit and the knee would bend backwards to make up).
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
  as the arm comes sideways.  The first frame is also solved from the other
  branch and the better one is kept.
* Root: the human pelvis motion scaled by the root scale (GMR's
  human_scale_table).
* Floor: each frame's lowest point of Microban (the training collision
  shapes) is put at the human clearance x the root scale.  The clearance is
  the smaller of the feet's (ankle and toe, against their height in the
  standing reference) and of the other parts that can touch the floor
  (pelvis, spine root, neck base, shoulders, elbows, wrists, knees, against
  their height when on the floor): lying, crawling and hands on the floor.
* The neck is not retargeted (the HMD drives it): no neck target, so the neck
  stays at 0 (HOME).
* Support (``support_spec``, off by default; support.py): the targets
  changed so that Microban's zero-moment point stays in its support: the
  trunk moved horizontally and tilted about the hips' midpoint (the arms
  keep their pose relative to the trunk), the feet rolled about their
  forward axis and placed elsewhere on the floor; a knee moves by the share
  of the leg below it, and the thigh and shank turn with the leg's line.

Outputs of ``retarget_frames`` (in ``out_dir``): ``<name>.npz`` (qpos, joint
names, the joints in HOME's order, the targets, the human skeleton for
drawing, per-frame marks, the zero-moment point and the parts on the floor
before and after the support changes (after: found again from the changed
poses), the changes and the stage each frame came into the support at),
``<name>.pkl`` (GMR's PKL with ``joint_names``,
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

from mjlab_microban.robot.home_pose import HOME_JOINT_NAMES
from mjlab_microban.wholebody.retarget import smplx, support
from mjlab_microban.wholebody.retarget.gmr import GeneralMotionRetargeting
from mjlab_microban.wholebody.retarget.gmr import params as gparams
from mjlab_microban.wholebody.retarget.model import (
    gmr_model_xml,
    home_qpos,
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
# robot's ranges (elbow +-125, knee -45..135 degrees) are unchanged.
ELBOW_MAX_DEG = 0.0  # the elbow bends the human way only (HOME is -20)
KNEE_MIN_DEG = 0.0  # the knee does not bend past straight (HOME's 0)
G1_LEG_M = 0.654  # GMR's g1_mocap_29dof.xml, hip_pitch_link to ankle_roll_link height
SOLE_GEOMS = {s: [f"{s}_foot_collision_{i}" for i in range(1, 7)] for s in ("left", "right")}
# Joint heights of the parts other than the feet when they lie on the floor,
# as a share of the reference leg length (hip -> ankle), above the toe-base
# floor: the lowest 0.5 % of each part's height in LAFAN1's lying and falling
# clips (ground, fallAndGetUp, pushAndFall; 13 clips).
CONTACT_RADIUS = {"pelvis": 0.09, "spine": 0.06, "neck": 0.12, "shoulder": 0.05, "elbow": -0.02, "wrist": -0.02, "knee": 0.01}
# With the support changes, the IK solves on to this error decrease (GMR's 0.001) and at most this many steps.
SUPPORT_IK_MIN_GAIN = 1e-5
SUPPORT_IK_MAX_ITER = 50
HOLD_FADE_S = 0.2  # after a foot leaves the floor, its hold fades out over this
HOLD_LEG_STEPS = 12  # solve_leg's least squares steps at most
HOLD_LEG_TOL_M = 1e-5  # solve_leg stops within this (the turn: per HOLD_LEG_TURN_M)
HOLD_LEG_MAX_STEP = 0.1  # solve_leg moves a joint at most this (rad) a step
# A held foot its leg alone cannot bring within this of its target is out of reach there (hip pitch or roll at its
# end, the leg straight): the IK's leg is kept (the foot slides there; LAFAN1 walk3, its hips at their pitch ends in
# 55 % of the frames unchanged, otherwise took 7 % of its frames to joint jumps and 5 % to overlapping legs).
HOLD_FAIL_M = 0.002
# solve_leg's damping (per m^2 of the foot's error): near a straight knee the hip yaw and the ankle trade almost
# freely, and undamped the leg drifts along them from its seed (60 degrees of hip yaw at 1e-6).
HOLD_LEG_DAMPING = 1e-4
HOLD_ROUNDS = 3  # the feet held again on the feet found on the floor at most this many times (settle)
HOLD_LEG_TURN_M = 0.03  # hold_legs weighs a foot's turn per rad as its place per this many m (the sole's half length)


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


class MicrobanRetargeting(GeneralMotionRetargeting):
    """GMR with its position scaling (scale_human_data) replaced and the arm branch handling added.
    The IK, the orientation and position offsets and tables 1 and 2 are GMR's."""

    def __init__(self, cfg: dict, cfg_path: Path, fmt_name: str, **kw):
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
        self.limits_plain = [mink.ConfigurationLimit(self.model)]
        self.limits_coll = self.limits_plain + [mink.CollisionAvoidanceLimit(self.model, self_collision_pairs(self.model),
                                                                            minimum_distance_from_collisions=0.002,
                                                                            collision_detection_distance=0.03)]
        self.ik_limits = self.limits_coll
        t1, t2 = cfg["ik_match_table1"], cfg["ik_match_table2"]
        # (human shoulder, human elbow, table 2's upper-arm orientation task, its weight,
        #  qpos addresses (pitch, roll, elbow), roll sign, human wrist)
        self.arms = []
        for side, sign in (("left", 1.0), ("right", -1.0)):
            task = next(t for t in self.tasks2 if t.frame_name == f"{side}_shoulder_c")
            adr = [self.model.joint(f"{side}_{j}").qposadr[0] for j in ("shoulder_pitch", "shoulder_roll", "elbow")]
            self.arms.append((t1[f"{side}_shoulder_c"][0], t1[f"{side}_elbow_c"][0], task, t2[f"{side}_shoulder_c"][2], adr, sign,
                              t1[f"{side}_hand"][0]))
        self.trunk_id = self.model.body("trunk").id
        # The support changes of the current frame (support.py; None unless asked for; Adjust) and how much the
        # left and right foot stand (support.stance; 1 on the floor): a standing leg is at most ``leg_reach`` long,
        # the trunk going down by at most ``max_drop`` for it to reach its foot.
        self.adjust = None
        self.stance = np.zeros(2)
        self.leg_reach, self.max_drop = np.inf, 0.0
        # The feet's offsets (scale_human_data's) and the targets by frame: a clip solved again for some frames gets
        # the ones of its first, whole pass (``frame``: the frame being solved, None outside solve_frames).
        self.frame, self.ankle_offsets, self.target_cache = None, {}, {}
        # The feet held on the floor (support.hold_feet; None: not): the runs on the floor ``hold_on`` (T, 2; left,
        # right), the soles' points (support.sole_points) and, in a solve, the held targets of every frame ``held``
        # (positions (T, 2, 3), quaternions (T, 2, 4)).
        self.hold_on, self.soles, self.held = None, None, None
        self._feet_base = None  # held_feet's: the feet's and hips' kept targets of every frame
        # The geom pairs the IK keeps apart, one by one (move_frames checks a moved frame on them).
        pairs = np.array([(a, b) for ga, gb in self_collision_pairs(self.model) for a in ga for b in gb])
        self._coll_pairs = pairs, self.model.geom_rbound[pairs].sum(1)
        # Per leg (left, right) for hold_legs: the ankle centre's site, its six joints' qpos and dof addresses and
        # ranges (the IK's: the knee does not bend past straight).
        self._leg_data = mujoco.MjData(self.model)
        self._legs_ik = []
        for side in ("left", "right"):
            ids = [self.model.joint(f"{side}_{j}").id for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll")]
            self._legs_ik.append((self.model.site(f"{side}_ankle_c").id, self.model.jnt_qposadr[ids], self.model.jnt_dofadr[ids],
                                  self.model.jnt_range[ids, 0], self.model.jnt_range[ids, 1], self.model.site(f"{side}_hip_c").id))
        fmt = FORMATS[fmt_name]
        # (hip, knee, ankle, side) and (shoulder, elbow, wrist)
        self.legs = [tuple(role_name(fmt, f"{s}.{p}") for p in ("hip", "knee", "ankle")) + (i,) for i, s in enumerate("LR")]
        self.arm_parts = [tuple(role_name(fmt, f"{s}.{p}") for p in ("shoulder", "elbow", "wrist")) for s in "LR"]

    def reset_configuration(self):
        super().reset_configuration()
        self.ankle_offset = {}

    def update_targets(self, human_data, offset_to_ground=False):
        """GMR's, with the frame's targets kept (``frame``): the clip's frames are solved many times with the
        support changes, which (and the feet's hold) are put on them each time (_adjusted; they turn the targets
        from the left, so on GMR's orientation offsets as on the human's)."""
        kept = self.target_cache.get(self.frame) if self.frame is not None else None
        if kept is None:
            data = self.to_numpy(human_data)
            data = self.scale_human_data(data, self.human_root_name, self.human_scale_table)
            data = self.offset_human_data(data, self.pos_offsets1, self.rot_offsets1)
            data = self.apply_ground_offset(data)
            if offset_to_ground:
                data = self.offset_human_data_to_ground(data)
            kept = (data, self._arm_costs)
            if self.frame is not None:
                self.target_cache[self.frame] = kept
        data, arm_costs = kept
        for (_, _, task, *_), cost in zip(self.arms, arm_costs):
            task.set_orientation_cost(cost)
        if self.adjust is not None or (self.held is not None and self.frame is not None):
            data = self._adjusted(dict(data), self.human_root_name)
        self.scaled_human_data = data
        for table, use in ((self.human_body_to_task1, self.use_ik_match_table1), (self.human_body_to_task2, self.use_ik_match_table2)):
            if use:
                for body_name, task in table.items():
                    pos, rot = data[body_name]
                    task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))

    def retarget(self, human_data, offset_to_ground=False):
        first = self._warmup_needed
        q = super().retarget(human_data, offset_to_ground)
        arms = list(range(len(self.arms))) if first else [i for i in range(len(self.arms)) if self._arm_suspect(i)]
        if arms:
            q = self._pick_arm_branch(arms)
        return q

    def _arm_suspect(self, i) -> bool:
        """The arm may be on the other branch (arm_flips' condition on the target and the robot's joints): the
        target elbow bends 30+ degrees while the robot's is straight (within 2 degrees of the IK's 0 limit; the
        other branch cannot bend it the human way), or the shoulder roll is past 90 degrees (the other branch's
        side) while the target upper arm points down in the trunk frame."""
        sh, el, _, _, (_, a_r, a_e), _, wr = self.arms[i]
        tg = self.scaled_human_data
        u, f = tg[el][0] - tg[sh][0], tg[wr][0] - tg[el][0]
        q = self.configuration.q
        if np.degrees(np.arccos(np.clip(u @ f / np.linalg.norm(u) / np.linalg.norm(f), -1, 1))) > 30 and q[a_e] > np.radians(-2.0):
            return True
        if abs(q[a_r]) > np.pi / 2:
            trunk = R.from_matrix(self.configuration.data.xmat[self.trunk_id].reshape(3, 3))
            return trunk.inv().apply(u / np.linalg.norm(u))[2] < -0.3
        return False

    def _cost2(self):
        """Table 2's weighted squared error (what the IK minimises)."""
        return float(sum(np.sum((t.cost * t.compute_error(self.configuration)) ** 2) for t in self.tasks2))

    def _pick_arm_branch(self, arms):
        """Per arm, solve table 2 again from the other branch (shoulder pitch + 180 degrees, roll to the other
        side, elbow reversed) and keep the lower IK cost.  Used on the first frame after a reset and on frames
        whose arm may be on the other branch (_arm_suspect); the next frames start from the kept solution.
        Solved as GMR's warm-up after a reset."""
        dt = self._warmup_dt
        for _, _, _, _, (a_p, a_r, a_e), sign, _ in (self.arms[i] for i in arms):
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
        for hb, off_ref, ref_inv in self.uniform:
            # The difference left at the reference turns with the human foot, moving per frame no more than the human ankle (at robot size).
            w = self.ankle_offsets.get((self.frame, hb)) if self.frame is not None else None
            if w is None:
                w = (R.from_quat(human_data[hb][1], scalar_first=True) * ref_inv).apply(off_ref)
                prev = self.ankle_offset.get(hb)
                if prev is not None:
                    step = self.root_scale * np.linalg.norm(human_data[hb][0] - prev[1])
                    dw = w - prev[0]
                    n = np.linalg.norm(dw)
                    w = prev[0] + (dw if n <= step else dw * (step / n))
                self.ankle_offset[hb] = (w, np.array(human_data[hb][0]))
                if self.frame is not None:
                    self.ankle_offsets[(self.frame, hb)] = w
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
        # direction in the trunk frame (sideways, the shoulder pitch becomes the upper-arm twist and positions do not
        # fix it); set by update_targets (the support changes keep the arms' pose relative to the trunk).
        self._arm_costs = []
        for sh, el, _, weight, *_ in self.arms:
            d = T.inv().apply(out[el][0] - out[sh][0])
            self._arm_costs.append(weight * (d[1] / np.linalg.norm(d)) ** 2)
        return out

    def penetrates(self) -> bool:
        """Whether two of the parts the IK keeps apart overlap in ``_leg_data`` (mj_kinematics has run)."""
        m, d = self.model, self._leg_data
        pairs, reach = self._coll_pairs
        near = np.linalg.norm(d.geom_xpos[pairs[:, 0]] - d.geom_xpos[pairs[:, 1]], axis=1) < reach  # bounding spheres meet
        fromto = np.empty(6)
        return any(mujoco.mj_geomDistance(m, d, a, b, 0.002, fromto) < 0.0 for a, b in pairs[near])

    def hold_legs(self, q):
        """``q`` with the legs of the feet on the floor (``hold_on``) solved again, alone (their six joints, the
        trunk where the IK put it), to put each foot on its held target (``held``): the IK weighs the held feet
        against the other targets and leaves them a few mm off, which the next frame turns into a slide."""
        on = self.hold_on[self.frame]
        if not on.any():
            return q
        self._leg_data.qpos[:] = q
        for i in np.flatnonzero(on):
            adr = self._legs_ik[i][1]
            before = self._leg_data.qpos[adr].copy()
            if self.solve_leg(i, self.held[0][self.frame, i], self.held[1][self.frame, i], True) > HOLD_FAIL_M:
                self._leg_data.qpos[adr] = before  # out of the leg's reach: the IK's leg, nearest the human's
        return self._leg_data.qpos.copy()

    def solve_leg(self, i, goal_p, goal_quat, standing):
        """Solve leg ``i`` (left 0, right 1) alone in ``_leg_data`` (its qpos the pose; the trunk stays) to put its
        foot (the ankle centre's site) at ``goal_p``, ``goal_quat``: damped least squares within the joints' ranges
        (HOLD_LEG_STEPS at most).  A ``standing`` foot beyond the leg's reach (``leg_reach``) is raised to it, off the
        floor (Microban's legs are shorter for its steps than the human's: a trailing foot held on its toes at the
        push-off leaves the floor a little sooner); beyond it horizontally, as near as it gets.  Returns how far
        the foot is left from its target (m)."""
        m, d = self.model, self._leg_data
        site, adr, dof, lo, hi, hip = self._legs_ik[i]
        goal_p = np.array(goal_p, float)
        mujoco.mj_kinematics(m, d)
        if standing:
            v = d.site_xpos[hip] - goal_p
            horizontal = float(np.hypot(v[0], v[1]))
            if horizontal < self.leg_reach and v @ v > self.leg_reach**2:
                goal_p[2] = d.site_xpos[hip][2] - np.sqrt(self.leg_reach**2 - horizontal**2)
        jac, now, inv, diff, turn = np.zeros((6, m.nv)), np.empty(4), np.empty(4), np.empty(4), np.empty(3)
        err = np.empty(6)
        for k in range(HOLD_LEG_STEPS):
            if k:
                mujoco.mj_kinematics(m, d)
            mujoco.mju_mat2Quat(now, d.site_xmat[site])
            mujoco.mju_negQuat(inv, now)
            mujoco.mju_mulQuat(diff, goal_quat, inv)  # the turn still to go, world frame
            mujoco.mju_quat2Vel(turn, diff, 1.0)
            err[:3], err[3:] = goal_p - d.site_xpos[site], turn * HOLD_LEG_TURN_M
            if np.abs(err).max() < HOLD_LEG_TOL_M:
                break
            if k == HOLD_LEG_STEPS - 1:
                break
            mujoco.mj_comPos(m, d)
            mujoco.mj_jacSite(m, d, jac[:3], jac[3:], site)
            j = jac[:, dof]
            j[3:] *= HOLD_LEG_TURN_M
            step = j.T @ np.linalg.solve(j @ j.T + HOLD_LEG_DAMPING * np.eye(6), err)
            step *= min(1.0, HOLD_LEG_MAX_STEP / max(np.abs(step).max(), 1e-12))  # (near a straight knee: no leap to another branch)
            d.qpos[adr] = np.clip(d.qpos[adr] + step, lo, hi)
        return float(np.abs(err[:3]).max())

    def _place(self, out):
        """The map of a point by the trunk's change (turn by ``tilt`` about the hips' midpoint, move by ``shift``)
        of the current frame, by a share of it."""
        a = self.adjust
        turn = R.from_rotvec(a.tilt) if a is not None else R.identity()
        pivot = 0.5 * (out[self.legs[0][0]][0] + out[self.legs[1][0]][0])
        move = np.array([a.shift[0], a.shift[1], 0.0]) if a is not None else np.zeros(3)

        def place(x, share=1.0):
            return x + share * (pivot + turn.apply(x - pivot) + move - x)
        return place, turn

    def desired_feet(self, out):
        """The feet's targets of the current frame before the hold: the human's turned by the foot's ``ankle``
        rotation and moved by its ``foot`` offset, and, off the floor, with the trunk (by 1 - its stance weight,
        smoothed: the leg keeps its pose relative to the trunk, and the foot comes back to where it lands)."""
        a = self.adjust
        if a is None:
            return {i: out[ankle] for _, _, ankle, i in self.legs}
        place, _ = self._place(out)
        feet = {}
        for _, _, ankle, i in self.legs:
            off = 1.0 - self.stance[i]
            follow = off * off * (3.0 - 2.0 * off)
            turn = R.from_rotvec(a.ankle[i]) * R.from_rotvec(follow * np.asarray(a.tilt))
            feet[i] = (place(out[ankle][0], follow) + [a.foot[i][0], a.foot[i][1], 0.0],
                       (turn * R.from_quat(out[ankle][1], scalar_first=True)).as_quat(scalar_first=True))
        return feet

    def _adjusted(self, out, root):
        """The targets with the support changes (``adjust``, support.Adjust's frame; None: none) and the feet held
        (``held``; None: desired_feet): the trunk and the arms turn by ``tilt`` about the hips' midpoint and move
        by ``shift`` (the arms keep their pose relative to the trunk), a knee moves by the share of its leg below
        it, and the thigh and shank turn with the leg's line (hip -> ankle).  Where a standing leg would have to be
        longer than ``leg_reach`` (or than it was) to reach its foot, the trunk and the arms go down by up to
        ``max_drop`` (support.reach_rows keeps the changes within that)."""
        if self.held is not None and self.frame is not None:
            feet = {i: (self.held[0][self.frame, i], self.held[1][self.frame, i]) for i in range(2)}
        else:
            feet = self.desired_feet(out)
        place, turn = self._place(out)

        def rotate(rot, q):
            return (rot * R.from_quat(q, scalar_first=True)).as_quat(scalar_first=True)

        drop = 0.0
        if self.max_drop > 0.0:
            for hip, _, ankle, i in self.legs:
                p = place(out[hip][0]) - feet[i][0]
                # (Held, a foot is where the leg must reach; otherwise a leg the human's motion already stretched past
                # its length is left to the IK.)
                reach = self.leg_reach if self.held is not None else max(self.leg_reach, float(np.linalg.norm(out[hip][0] - out[ankle][0])))
                need = p[2] - np.sqrt(max(reach**2 - p[0] ** 2 - p[1] ** 2, 0.0))
                drop = min(max(drop, self.stance[i] * need), self.max_drop)
        down = np.array([0.0, 0.0, drop])
        out[root] = (place(out[root][0]) - down, rotate(turn, out[root][1]))
        for arm in self.arm_parts:
            for name in arm:
                out[name] = (place(out[name][0]) - down, rotate(turn, out[name][1]))
        for hip, knee, ankle, i in self.legs:
            b, m, e = out[hip][0], out[knee][0], out[ankle][0]
            b2, e2 = place(b) - down, feet[i][0]
            share = np.linalg.norm(e - m) / max(np.linalg.norm(m - b) + np.linalg.norm(e - m), 1e-9)
            line = R.align_vectors([b2 - e2], [b - e])[0]
            out[hip] = (b2, rotate(line, out[hip][1]))
            out[knee] = (m + share * (b2 - b) + (1.0 - share) * (e2 - e), rotate(line, out[knee][1]))
            out[ankle] = feet[i]
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
    frames, fps, tilt_deg = motion_frames(b, fmt_name, fps_out)
    return frames, fps, ref, tilt_deg, (list(b.names), b.parents)


def run(path: str | Path, fmt_name: str, fps_out: float, out_dir: Path, stem: str | None = None,
        smplx_model_dir: str | Path | None = None, support_spec: support.SupportSpec | None = None) -> dict:
    frames, fps, ref, tilt_deg, skel = load_motion(path, fmt_name, fps_out, smplx_model_dir)
    return retarget_frames(stem or Path(path).stem, fmt_name, frames, fps, ref, Path(out_dir), tilt_deg, skel, support_spec)


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


def solve_frames(gmr: MicrobanRetargeting, frames: list, adjust=None, seeds=None, stance=None, subset=None,
                 base_held=None):
    """IK of every frame (or of the frames ``subset``) from a reset: qpos (n, nq), the targets' positions and
    orientations per frame, the feet's held targets (``gmr.held``; None without the hold) and the frames solved,
    with the support changes ``adjust`` (support.Changes) and the feet's weights on the floor ``stance`` (T, 2)
    (MicrobanRetargeting.adjust, .stance).  With ``seeds`` (T, nq), each frame's IK starts from its seed (with the
    trunk changed as ``adjust`` changes it) instead of the frame before (a changed clip solved from the unchanged
    solution stays on its branches: a near-straight leg's hip yaw and ankle roll, the arms).  With the feet held
    (``gmr.hold_on``), the held targets are found over the whole clip first (support.hold_feet: a run's frames hold
    on the frames before them), and the frames whose held feet differ from ``base_held`` are solved too."""
    gmr.reset_configuration()
    T = len(frames)
    gmr.held = None
    if seeds is not None:
        gmr._warmup_needed = False
        data = mujoco.MjData(gmr.model)
        hips = [gmr.model.site(f"{s}_hip_c").id for s in ("left", "right")]
        if gmr.hold_on is not None:
            held = held_feet(gmr, adjust, stance)
            if subset is not None and base_held is not None:
                subset = np.union1d(subset, np.flatnonzero(held_moved(held, base_held))).astype(int)
            gmr.held = held
    qs, targets, target_quats = [], [], []
    for t in (range(T) if subset is None else subset):
        gmr.frame = t
        gmr.adjust = None if adjust is None else adjust.at(t)
        gmr.stance = np.zeros(2) if stance is None else np.asarray(stance[t], float)
        if seeds is not None:
            seed = seeds[t].copy()
            if gmr.adjust is not None:  # the trunk where the changes put it, the arms with it
                data.qpos[:] = seed
                mujoco.mj_kinematics(gmr.model, data)
                pivot = data.site_xpos[hips].mean(0)
                turn = R.from_rotvec(gmr.adjust.tilt)
                seed[:3] = pivot + turn.apply(seed[:3] - pivot) + [*gmr.adjust.shift, 0.0]
                seed[3:7] = (turn * R.from_quat(seed[3:7], scalar_first=True)).as_quat(scalar_first=True)
            gmr.configuration.update(q=seed)
        q = gmr.retarget(to_input(frames[t])).copy()
        if gmr.held is not None:
            q = gmr.hold_legs(q)
        qs.append(q)
        targets.append({k: np.asarray(v[0]).copy() for k, v in gmr.scaled_human_data.items()})
        target_quats.append({k: np.asarray(v[1]).copy() for k, v in gmr.scaled_human_data.items()})
    held = gmr.held
    gmr.frame, gmr.adjust, gmr.stance, gmr.held = None, None, np.zeros(2), None
    return np.asarray(qs), targets, target_quats, held, (np.arange(T) if subset is None else np.asarray(subset, int))


def held_feet(gmr: MicrobanRetargeting, adjust, stance):
    """The feet's held targets of every frame (support.hold_feet) with the changes ``adjust`` and the stance
    weights ``stance``: the desired feet (MicrobanRetargeting.desired_feet, for all frames at once on the frames'
    kept targets) held on the runs ``gmr.hold_on``."""
    T = len(gmr.hold_on)
    if gmr._feet_base is None:
        ankle_p, ankle_q, hips = np.zeros((T, 2, 3)), np.zeros((T, 2, 4)), np.zeros((T, 2, 3))
        for t in range(T):
            data = gmr.target_cache[t][0]
            for hip, _, ankle, i in gmr.legs:
                ankle_p[t, i], ankle_q[t, i] = data[ankle]
                hips[t, i] = data[hip][0]
        gmr._feet_base = ankle_p, ankle_q, hips
    ankle_p, ankle_q, hips = gmr._feet_base
    shift, tilt, ankle_turn = adjust.world()
    off = 1.0 - np.asarray(stance, float)
    follow = off * off * (3.0 - 2.0 * off)  # (T, 2)
    pivot = hips.mean(1)  # (T, 3)
    move = np.c_[shift, np.zeros(T)]
    pos, quat = np.zeros((T, 2, 3)), np.zeros((T, 2, 4))
    turn = Rotation_from_rotvec(tilt)
    for i in range(2):
        rel = ankle_p[:, i] - pivot
        moved = pivot + turn.apply(rel) + move - ankle_p[:, i]
        pos[:, i] = ankle_p[:, i] + follow[:, i, None] * moved + np.c_[adjust.foot[:, i], np.zeros(T)]
        rot = (Rotation_from_rotvec(ankle_turn[:, i]) * Rotation_from_rotvec(follow[:, i, None] * tilt)
               * R.from_quat(ankle_q[:, i], scalar_first=True))
        quat[:, i] = rot.as_quat(scalar_first=True)
    return support.hold_feet(pos, quat, gmr.hold_on, gmr.soles, max(1, round(HOLD_FADE_S * gmr.fps)))


def Rotation_from_rotvec(v) -> R:  # noqa: N802
    return R.from_rotvec(np.asarray(v, float).reshape(-1, 3))


def targets_of(gmr: MicrobanRetargeting, frames, adjust, stance, held, names):
    """The IK targets' positions (T, n, 3) and orientations (T, n, 4) (``names``' order) of every frame with the
    changes ``adjust``, the stance weights ``stance`` and the held feet ``held``, IK space."""
    T = len(frames)
    tg, tq = np.zeros((T, len(names), 3)), np.zeros((T, len(names), 4))
    gmr.held = held
    for t in range(T):
        gmr.frame, gmr.adjust, gmr.stance = t, adjust.at(t), np.asarray(stance[t], float)
        gmr.update_targets(to_input(frames[t]))
        for k, name in enumerate(names):
            tg[t, k], tq[t, k] = gmr.scaled_human_data[name]
    gmr.frame, gmr.adjust, gmr.stance, gmr.held = None, None, np.zeros(2), None
    return tg, tq


def held_moved(held, base_held) -> np.ndarray:
    """(T,) the frames whose held feet differ."""
    T = len(held[0])
    return (np.abs(held[0] - base_held[0]).reshape(T, -1).max(1) > 1e-7) | (np.abs(held[1] - base_held[1]).reshape(T, -1).max(1) > 1e-7)


def move_frames(gmr: MicrobanRetargeting, seeds, adjust, stance, held, subset) -> np.ndarray:
    """The frames ``subset`` with the support changes, without the IK: each from its seed (the unchanged solution),
    the trunk turned and moved as ``adjust`` says (the arms and the head with it: their pose relative to the trunk
    is kept, as the IK keeps it) and lowered for a standing leg's reach (by up to ``max_drop``), and each leg
    solved alone to its foot's target in ``held`` (MicrobanRetargeting.solve_leg: a leg has six joints for its
    foot's six coordinates, so the IK's legs differ only by its other targets' pull).  qpos (n, nq), IK space, and
    the frames (n,) to solve by the IK instead: where two parts the IK keeps apart overlap (the legs solved alone
    do not avoid each other) or a foot is beyond its leg's reach (HOLD_FAIL_M)."""
    m, d = gmr.model, gmr._leg_data
    hips = [m.site(f"{s}_hip_c").id for s in ("left", "right")]
    out = np.empty((len(subset), m.nq))
    overlap = np.zeros(len(subset), bool)
    for k, t in enumerate(subset):
        q = seeds[t].copy()
        a = adjust.at(t)
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        pivot = d.site_xpos[hips].mean(0)
        turn = R.from_rotvec(a.tilt)
        q[:3] = pivot + turn.apply(q[:3] - pivot) + [a.shift[0], a.shift[1], 0.0]
        q[3:7] = (turn * R.from_quat(q[3:7], scalar_first=True)).as_quat(scalar_first=True)
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        drop = 0.0
        for i in range(2):
            v = d.site_xpos[hips[i]] - held[0][t, i]
            need = v[2] - np.sqrt(max(gmr.leg_reach**2 - v[0] ** 2 - v[1] ** 2, 0.0))
            drop = min(max(drop, stance[t, i] * need), gmr.max_drop)
        d.qpos[:] = q
        d.qpos[2] -= drop
        missed = 0.0
        for i in range(2):
            # A leg off the floor and not coming back to it yet moves with the trunk as it is (its pose relative to
            # the trunk kept); the others are solved to their foot's target (a held foot out of reach: by the IK).
            standing = bool(gmr.hold_on[t, i])
            if standing or stance[t, i] > 0.0 or drop > 0.0:
                miss = gmr.solve_leg(i, held[0][t, i], held[1][t, i], standing)
                missed = max(missed, miss if standing else 0.0)
        out[k] = d.qpos
        mujoco.mj_kinematics(m, d)
        overlap[k] = missed > HOLD_FAIL_M or gmr.penetrates()
    return out, overlap


def put_on_floor(model, lowest: LowestPoint, qs, targets, height) -> None:
    """Floor: each frame's lowest point (training collision shapes) at ``height`` (the human clearance x root
    scale); the joints are unchanged, only the root height (and the targets' positions (T, n, 3)) move.  In place;
    returns the heights moved down (T,)."""
    d = mujoco.MjData(model)
    lows = []
    for q in qs:
        d.qpos[:] = q
        mujoco.mj_kinematics(model, d)
        lows.append(lowest(d))
    dz = np.asarray(lows) - height
    qs[:, 2] -= dz
    if targets is not None:
        targets[:, :, 2] -= dz[:, None]
    return dz


def retarget_frames(stem: str, fmt_name: str, frames: list, fps: float, ref: dict, out_dir: Path,
                    tilt_deg: float = 0.0, skel=None, support_spec: support.SupportSpec | None = None) -> dict:
    """Retarget human frames ({name: (position, quaternion wxyz)}, Z-up, floor z = 0) and write the npz, PKL,
    IK configuration and report to ``out_dir``.  ``support_spec``: with the support changes (support.py)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = FORMATS[fmt_name]
    model = mujoco.MjModel.from_xml_path(_register_robot())
    cfg = calibrate(ref, fmt_name, model)
    scale = cfg["human_scale_table"][fmt["pelvis"]]
    cfg_path = out_dir / f"{stem}_ik.json"
    cfg_path.write_text(json.dumps(cfg, indent=1))

    gmr = MicrobanRetargeting(cfg, cfg_path, fmt_name)
    if support_spec is not None:
        gmr.leg_reach = support.leg_length(model) - support_spec.reach_slack_m
        gmr.max_drop = support_spec.max_drop_m
    # Check: the standing reference retargets to HOME.
    gmr.reset_configuration()
    for _ in range(20):
        q_ref = gmr.retarget(to_input(ref))
    q_home = home_qpos(model)
    ref_joint_err = float(np.degrees(np.abs(q_ref[7:] - q_home[7:]).max()))
    ref_root_err = float(np.linalg.norm(q_ref[:3] - q_home[:3]))

    fallbacks0 = COLLISION_FALLBACKS[0]
    clear, floor = human_clearance(frames, ref, fmt, fps)
    lowest = LowestPoint(model)
    solved, seeds, target_names, frames_solved, ik_frames = {}, [], [], [0], [0]
    kinematic = False

    def solve(adjust=None, stance=None, subset=None, base=None, reseed=False):
        """The clip's qpos with the support changes ``adjust`` (support.Changes; None: none), each frame on the
        floor: of every frame, or of the frames ``subset`` with the others as in the solution ``base``.  The first
        solve is the unchanged clip; the later ones start each frame from it (``reseed``: and this one is it).
        Once ``kinematic`` is set, the frames are moved from the unchanged solution (move_frames) instead of solved
        by the IK, and their targets are made for the final solution only."""
        if kinematic:
            T = len(frames)
            held = held_feet(gmr, adjust, stance)
            sub = np.arange(T) if subset is None else np.asarray(subset, int)
            if base is not None:
                sub = np.union1d(sub, np.flatnonzero(held_moved(held, solved[id(base)][3]))).astype(int)
            qn, overlap = move_frames(gmr, seeds[0], adjust, stance, held, sub)
            frames_solved[0] += len(qn)
            if overlap.any():  # solved by the IK, which keeps the parts apart
                qn[overlap] = solve_frames(gmr, frames, adjust, seeds[0], stance, sub[overlap])[0]
                ik_frames[0] += int(overlap.sum())
            if reseed:
                seeds[0][sub] = qn
            qs = base.copy() if base is not None else np.empty((T, model.nq))
            dz = solved[id(base)][4].copy() if base is not None else np.zeros(T)
            dz[sub] = put_on_floor(model, lowest, qn, None, scale * clear[sub])
            qs[sub] = qn
            solved[id(qs)] = (qs, None, None, held, dz)
            return qs
        base_held = solved[id(base)][3] if base is not None else None
        qn, tg, tq, held, solved_frames = solve_frames(gmr, frames, adjust, seeds[0] if seeds else None, stance, subset,
                                                       base_held)
        subset = None if subset is None else solved_frames
        if reseed:
            if subset is None:
                seeds[0] = qn.copy()
            else:
                seeds[0][subset] = qn
        if not target_names:
            target_names.extend(tg[0])
        n = len(target_names)
        tg = np.array([[x[k] for k in target_names] for x in tg]).reshape(len(qn), n, 3)
        tq = np.array([[x[k] for k in target_names] for x in tq]).reshape(len(qn), n, 4)
        frames_solved[0] += len(qn)
        if subset is None:
            if not seeds:
                seeds.append(qn.copy())
            dz = put_on_floor(model, lowest, qn, tg, scale * clear)
            qs = qn
        else:
            dz_sub = put_on_floor(model, lowest, qn, tg, scale * clear[subset])
            _, tg_base, tq_base, _, dz = solved[id(base)]
            qs, dz = base.copy(), dz.copy()
            qs[subset], dz[subset] = qn, dz_sub
            tg_base, tq_base = tg_base.copy(), tq_base.copy()
            tg_base[subset], tq_base[subset] = tg, tq
            tg, tq = tg_base, tq_base
        solved[id(qs)] = (qs, tg, tq, held, dz)
        return qs

    human = support.human_contact(frames, ref, fmt, fps, floor, role_name, CONTACT_RADIUS)
    t0 = time.perf_counter()
    qs = solve()
    base_slide = None
    if support_spec is not None:
        # The feet held where they stand (support.hold_feet) over the runs on the floor of this first pass.
        shapes = support.Shapes(model)
        contact = support.robot_contact(support.poses(model, qs, shapes), human, shapes, fps)
        on = support.feet_on_floor(contact, support.feet_soles(model, qs, shapes))
        gmr.hold_on, gmr.soles, gmr.fps = on, support.sole_points(model, shapes), fps
        # Each frame solved again from its first solution, to convergence: the first pass, each frame from the
        # one before and stopped by GMR's error decrease, is up to 70 mrad from it (a near-straight knee), and the
        # changed clip, solved from it, would carry that scatter into its accelerations (support.Dyn).
        gmr.min_gain, gmr.max_iter = SUPPORT_IK_MIN_GAIN, SUPPORT_IK_MAX_ITER
        stance = support.stance(contact, fps, support_spec.stance_s)
        none = support.Changes.none(len(qs))
        qs = solve(none, stance, reseed=True)
        kinematic = True  # (the changes from here on move the frames from this solution: move_frames)

        def settle(qs, changes, reseed=False):
            """The feet held again on the feet on the floor found from the poses ``qs`` where they differ from the
            ones held, and those frames solved again, until they agree (HOLD_ROUNDS at most): holding a foot (or
            raising one beyond reach) moves it on or off the floor."""
            n = 0
            for _ in range(HOLD_ROUNDS):
                on = support.feet_on_floor(support.robot_contact(support.poses(model, qs, shapes), human, shapes, fps),
                                           support.feet_soles(model, qs, shapes))
                again = np.flatnonzero((on != gmr.hold_on).any(1))
                if not len(again):
                    break
                gmr.hold_on = on
                qs = solve(changes, stance, again, qs, reseed=reseed)
                n += len(again)
            return qs, n

        # The feet held before the changes: the unchanged clip settled on its own feet on the floor, and their slide.
        qs, held_again = settle(qs, none, reseed=True)
        base_contact = support.robot_contact(support.poses(model, qs, shapes), human, shapes, fps)
        soles = support.feet_soles(model, qs, shapes)
        speed = support.slide_speed(soles, support.feet_on_floor(base_contact, soles), fps)
        speed = speed[np.isfinite(speed)]
        base_slide = [round(float(np.percentile(speed, p_)), 1) for p_ in (50, 90, 100)] if len(speed) else None
    if support_spec is None:
        changes = support.Changes.none(len(qs))
        spec = support.SupportSpec()
        before, _ = support.evaluate(model, qs, human, fps, spec)
        summary = {"margin_m": spec.margin_m, "before": support.zmp_summary(before[3], before[2])}
        after, stage = before, np.full(len(qs), -1, np.int8)
    else:
        qs, changes, summary, before, after, stage, rejudge = support.plan(model, solve, qs, human, fps, support_spec)
        # The feet on the floor found again from the changed poses: where they changed, the feet are held again on
        # them (those frames solved again) and the ZMP is judged again.
        qs, again = settle(qs, changes)
        if again:
            after, stage = rejudge(qs)
        summary["held_again_frames"] = {"before_changes": held_again, "after_changes": again}
    z, _, contact, dist = after
    _, tg_all, tq_all, held_all, dz_all = solved[id(qs)]
    if tg_all is None:  # moved without the IK: the targets of the final solution
        tg_all, tq_all = targets_of(gmr, frames, changes, support.stance(before[2], fps, support_spec.stance_s), held_all,
                                    target_names)
        tg_all[:, :, 2] -= dz_all[:, None]
    targets = [dict(zip(target_names, x)) for x in tg_all]
    target_quats = [dict(zip(target_names, x)) for x in tq_all]
    solved.clear()
    ik_fps = frames_solved[0] / (time.perf_counter() - t0)
    d = mujoco.MjData(model)

    # The numbers.
    jn = [model.joint(j).name for j in range(1, model.njnt)]
    lo, hi = model.jnt_range[1:, 0], model.jnt_range[1:, 1]
    perr = {f: [] for f, *_ in TARGETS}
    rerr = {f: [] for f, *_ in TARGETS}
    sole = {"left": [], "right": []}
    arm_sites = {s: (f"{s}_shoulder_c", f"{s}_elbow_c", f"{s}_hand") for s in ("left", "right")}
    rob_arm = {s: [] for s in arm_sites}
    trunk_q, low_after, soles_w = [], [], []
    shapes_ = support.Shapes(model)
    foot_on = [shapes_.part_of == p_ for p_ in support.FEET]
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
        v_ = shapes_.vertices(d)
        soles_w.append([v_[on_] for on_ in foot_on])
    lows = np.minimum(sole["left"], sole["right"])
    # A standing foot's slide (support.slide_speed, the points support.hold_feet holds), on the parts on the floor
    # found from the final poses, and the frames where one slides (support.sliding).
    slide = support.slide_speed(np.asarray(soles_w), support.feet_on_floor(contact, np.asarray(soles_w)), fps)
    standing = slide[np.isfinite(slide)]
    slid = support.sliding(slide)
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
        "standing_foot_speed_mm_s_p50_p90_max": [round(float(np.percentile(standing, p_)), 1) for p_ in (50, 90, 100)] if len(standing) else None,
        "standing_foot_held_before_changes_mm_s_p50_p90_max": base_slide,
        "frames_sliding_share": round(float(slid.mean()), 4),
        "joint_speed_rad_s_p99_max": [round(float(np.percentile(speed, 99)), 2), round(float(speed.max()), 2)],
        "joint_jump_frames": int(jump.sum()),
        "target_jump_frames": int(tjump.sum()),
        "arm_flip_frames_longest": [int(flip.sum()), longest_run(flip)],
        "hand_err_mm_p99_max": [round(1000 * float(np.percentile(hand_err, 99)), 1), round(1000 * float(np.max(hand_err)), 1)],
        "frames_within_1deg_of_limit_any": int((np.minimum(qs[:, 7:] - lo, hi - qs[:, 7:]) < np.radians(1.0)).any(1).sum()),
        "joint_deg_min_max": {n: [round(float(deg[:, i].min()), 1), round(float(deg[:, i].max()), 1)] for i, n in enumerate(jn)},
        "frames_within_1deg_of_limit": {jn[i]: int(c) for i, c in enumerate(near) if c > 0},
        "root_z_m_min_max": [round(float(qs[:, 2].min()), 3), round(float(qs[:, 2].max()), 3)],
        "root_xy_end_m": np.round(qs[-1, :2], 3).tolist(),
        "ik_frames_solved_per_frame": round(frames_solved[0] / max(len(qs), 1), 2),
        "moved_frames_solved_by_the_ik_again": ik_frames[0],
        "support_shift": None if support_spec is None else support_spec.to_dict(),
        "support_changes": changes.summary(),
        "zmp_in_support": summary,
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
             flip=flip, joint_jump=jump, target_jump=tjump, hand_err=np.asarray(hand_err, np.float32),
             lowest_point=low_after.astype(np.float32), **changes.arrays(), zmp_stage=stage, sliding=slid,
             zmp=z.astype(np.float32), zmp_outside=dist.astype(np.float32), contact=contact,
             zmp_before=before[0].astype(np.float32), zmp_outside_before=before[3].astype(np.float32),
             contact_before=before[2], human_contact=human, contact_parts=np.array(support.PARTS),
             **extra)
    with open(out_dir / f"{stem}.pkl", "wb") as fh:  # the shape of GMR's scripts/bvh_to_robot.py output
        pickle.dump({"fps": fps, "root_pos": qs[:, :3], "root_rot": qs[:, [4, 5, 6, 3]],  # xyzw
                     "dof_pos": qs[:, 7:], "joint_names": jn, "local_body_pos": None, "link_body_list": None}, fh)
    (out_dir / f"{stem}_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    return report
