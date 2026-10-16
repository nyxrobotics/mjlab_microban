"""PICO body tracking retargeted to Microban frame by frame, from the frames before each one (docs/wholebody_live.md).

The data pipeline (wholebody/retarget/core.py ``retarget_frames``, wholebody/data/build.py) sees a whole clip.  Here
each output frame uses only what came before it; the parts that differ:

* World frame: the calibration frame faces +x with the pelvis at x = y = 0 (the data: a clip's first frame).
* Standing reference: one still standing frame picked from a calibration window, made by pico.standing_reference
  (the data: per source, a quiet standing frame of its clips).
* Input rate: the samples (25-50 Hz) are interpolated in time (positions linearly, orientations by slerp) to the
  output rate (50 Hz), ``delay`` behind the newest input so a sample after the rendered time has normally arrived.
* IK: core.MicrobanRetargeting, unchanged (GMR's IK, the servo, hip-yaw, elbow and knee limits, self-collision, the
  arm branch picked on the first frame), at the output rate; the joints are clipped to the MJCF ranges.
* Floor: the lowest collision point at the human clearance x the root scale, the clearance's floor being the
  minimum of the last second (the data: +-1 s); it follows a lower minimum at once and a higher one (an old low
  leaving the window) at FLOOR_RISE_M_S at most, so the trunk does not drop by the step in one frame.
* Poses: forward kinematics on the training collision model (body_pos_w, body_quat_w as build.py).
* Frames sent: the average of the last SMOOTH_TICKS frames (``Smoother``, (SMOOTH_TICKS - 1) / 2 ticks late; the data:
  its 30 fps frames, interpolated linearly at 50 Hz in training).
* Velocities: backward differences of the frames sent (the data: central differences).
* Exclusion: preprocess.py's rules and finalize's arm-flip cut, judged on the frames so far.  A marked frame and the
  pad_s after the last mark are not sent (the data cuts them): like a missing or late body, they interrupt the
  reference (``Carry``): the last motion fades out, the pose stands still, after HOLD_S (at once for a pose whose
  weight is outside its support) it goes to the calibrated standing pose, the lowest point on the floor; when the
  body is back the reference goes back to it smoothly (``Return``).  No frame changes more than the training
  reference does in one tick (``JOINT_STEP_LIMITS``, ``Guard``).

PICO's 24 joints become the pseudo-bones of pico.py (skeleton.FORMATS["g1"]), the joint whose global rotation is
the bone's rotation: thigh = hip, shin = knee, foot = ankle, upper arm = shoulder, forearm = elbow, hand = wrist; the
torso is spine_1 (position) with spine_3's (chest) rotation, the neck base the neck joint, the head the head joint
(the neck's target), the toe the foot joint with the ankle's rotation.
"""

from __future__ import annotations

import json
import tempfile
import threading
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from mjlab_microban.live import stream
from mjlab_microban.wholebody.data.fk import quat_conjugate, quat_multiply, set_qpos
from mjlab_microban.wholebody.data.preprocess import ClipChecker, ExclusionSpec
from mjlab_microban.wholebody.retarget import core, pico
from mjlab_microban.wholebody.retarget.model import collision_model, home_qpos
from mjlab_microban.wholebody.retarget.pipeline import MIN_FLIP_RUN
from mjlab_microban.wholebody.retarget.skeleton import FORMATS

FMT = "g1"
OUTPUT_HZ = 50.0
STALE_S = 0.25
CALIBRATION_S = 1.0
# An interruption (the body missing or late, or HELD; docs/wholebody_live.md "途切れたとき"): the last motion carries
# on and fades out in COAST_S (it goes on for COAST_S / 2 x its speed), the pose stands still until HOLD_S after the
# interruption began, then goes to the calibrated standing pose in STAND_S (longer if a joint or the trunk would move
# faster than the step limits' 99th percentiles below).  When the body is back, the difference to it fades out in as
# long as the interruption lasted, at most RETURN_S (longer by the same rule).
COAST_S = 0.2
HOLD_S = 1.0
STAND_S = 1.0
RETURN_S = 1.0
CATCH_UP_S = 0.5  # after a gap the return waits (at most this) for the IK to stop catching up the operator
# The floor's rise (robot m/s): a step of the trunk height by the floor's 1 s window becomes a descent at this speed
# (its start and end change the trunk's speed by a third of the training reference's 99.9th percentile per tick).
FLOOR_RISE_M_S = 0.05
# The frames sent are the reference (SMOOTH_TICKS - 1) / 2 ticks late, each the average of the SMOOTH_TICKS frames
# around it (``Smoother``): the trunk's speed and turn rate from 30 Hz input at 50 Hz otherwise change per tick by up
# to 3.4 and 2.2 x the training reference's 99.9th percentile (PICO recordings); averaged over 4 ticks (30 ms late), by
# at most 1.24 and 0.86 x (6 frames of 55,717 past it; over 5 ticks, 40 ms late, 0.99 and 0.69 x: the shorter delay
# for the operator was chosen).
SMOOTH_TICKS = 4
# The training reference per control tick (0.02 s), as the training env feeds it (motion.py: 30 fps clips, positions
# and velocities linear, orientations slerp, at any phase), on the deployed run's motion_file
# (scripts/wholebody_reference_steps.py):
# per joint (speed p99 rad/s, speed change per tick p99 rad/s, step p99.9 rad, speed change per tick p99.9 rad/s).
# The planned motions of an interruption stay within the 99th percentiles; every frame sent through one stays within
# the 99.9th (``Guard``).
JOINT_STEP_LIMITS = {
    "head": (4.09, 0.98, 0.182, 2.26),
    "neck_roll": (2.00, 0.62, 0.090, 1.39),
    "neck_pitch": (2.57, 0.77, 0.112, 1.75),
    "right_shoulder_pitch": (6.34, 1.83, 0.213, 2.22),
    "right_shoulder_roll": (5.53, 1.65, 0.213, 2.22),
    "right_elbow": (6.44, 1.94, 0.213, 2.28),
    "right_hip_yaw": (4.60, 1.75, 0.178, 2.25),
    "right_hip_roll": (3.29, 1.23, 0.124, 1.92),
    "right_hip_pitch": (6.50, 1.85, 0.213, 1.91),
    "right_knee": (8.12, 2.18, 0.213, 2.18),
    "right_ankle_pitch": (6.10, 2.28, 0.195, 2.28),
    "right_ankle_roll": (3.59, 1.66, 0.146, 2.29),
    "left_shoulder_pitch": (6.06, 1.73, 0.213, 2.22),
    "left_shoulder_roll": (5.09, 1.52, 0.205, 2.22),
    "left_elbow": (6.12, 1.83, 0.209, 2.28),
    "left_hip_yaw": (4.54, 1.69, 0.175, 2.25),
    "left_hip_roll": (3.39, 1.22, 0.125, 1.92),
    "left_hip_pitch": (6.36, 1.83, 0.213, 1.91),
    "left_knee": (7.98, 2.18, 0.213, 2.18),
    "left_ankle_pitch": (5.77, 2.28, 0.189, 2.28),
    "left_ankle_roll": (3.67, 1.64, 0.144, 2.29),
}
# The trunk: position (m/s, m/s per tick, m, m/s per tick, and its height's step p99.9 in m) and turn (rad/s, rad/s per
# tick, rad, rad/s per tick), the same percentiles of the speed, its change per tick, the step and the change per tick.
TRUNK_STEP_LIMITS = (0.447, 0.0832, 0.0146, 0.162, 0.00791)
TURN_STEP_LIMITS = (4.85, 1.30, 0.197, 2.30)
# The Guard keeps a frame this far inside the 99.9th percentiles (their last digit, float rounding).
GUARD_SHARE = 0.99
# PICO's body joints (XRoboToolkit, SMPL order).
PICO_JOINTS = (
    "pelvis", "left_hip", "right_hip", "spine_1", "left_knee", "right_knee", "spine_2", "left_ankle", "right_ankle",
    "spine_3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand",
)
# pseudo-bone -> (position joint, orientation joint)
PICO_BONES = {
    "pelvis": ("pelvis", "pelvis"), "torso": ("spine_1", "spine_3"), "neck": ("neck", "spine_3"), "head": ("head", "head"),
    **{f"{s}_{p}": (f"{side}_{a}", f"{side}_{b}") for s, side in (("L", "left"), ("R", "right")) for p, a, b in (
        ("hip", "hip", "hip"), ("knee", "knee", "knee"), ("ankle", "ankle", "ankle"), ("toe", "foot", "ankle"),
        ("shoulder", "shoulder", "shoulder"), ("elbow", "elbow", "elbow"), ("wrist", "wrist", "wrist"))},
}
BONES = tuple(pico.NAMES)
_POS = np.array([PICO_JOINTS.index(PICO_BONES[b][0]) for b in BONES])
_ROT = np.array([PICO_JOINTS.index(PICO_BONES[b][1]) for b in BONES])
# Calibration: a still standing frame (pico.stance_candidates' conditions; lengths as a share of the operator's leg).
STAND_SPEED = 0.15  # all points, leg lengths per second
STAND_FEET = 0.02  # ankle height difference, leg lengths
STAND_SPINE_DEG = 10.0
STAND_KNEES_DEG = 5.0
STAND_ARMS_DEG = 35.0


def pico_to_bones(pos: np.ndarray, quat_wxyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PICO's 24 joints (Microban axes; (24, 3), (24, 4) wxyz) as the pseudo-bones' positions and orientations."""
    return np.asarray(pos, float)[_POS], np.asarray(quat_wxyz, float)[_ROT]


def as_frame(p: np.ndarray, q: np.ndarray) -> dict:
    return {n: (p[i], q[i]) for i, n in enumerate(BONES)}


def slerp(q0: np.ndarray, q1: np.ndarray, a: float) -> np.ndarray:
    """Row-wise slerp of wxyz quaternions."""
    d = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(d < 0.0, -q1, q1)
    theta = np.arccos(np.clip(np.abs(d), 0.0, 1.0))
    s = np.sin(theta)
    small = s < 1e-6
    w0 = np.where(small, 1.0 - a, np.sin((1.0 - a) * theta) / np.where(small, 1.0, s))
    w1 = np.where(small, a, np.sin(a * theta) / np.where(small, 1.0, s))
    q = w0 * q0 + w1 * q1
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


class Interpolator:
    """Samples on the source clock; the pose at a time is interpolated between the samples around it, and past the
    newest sample the newest one (late)."""

    def __init__(self, keep: int = 64):
        self.t: deque[float] = deque(maxlen=keep)
        self.p: deque[np.ndarray] = deque(maxlen=keep)
        self.q: deque[np.ndarray] = deque(maxlen=keep)

    def push(self, t: float, p: np.ndarray, q: np.ndarray) -> bool:
        if self.t and t <= self.t[-1]:
            return False
        self.t.append(t)
        self.p.append(p)
        self.q.append(q)
        return True

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray, bool]:
        if t >= self.t[-1]:
            return self.p[-1], self.q[-1], t > self.t[-1]
        if t <= self.t[0]:
            return self.p[0], self.q[0], False
        k = int(np.searchsorted(np.asarray(self.t), t))
        t0, t1 = self.t[k - 1], self.t[k]
        a = (t - t0) / (t1 - t0)
        return (1.0 - a) * self.p[k - 1] + a * self.p[k], slerp(self.q[k - 1], self.q[k], a), False


class SourceClock:
    """The source clock seen locally: offset = the smallest (arrival - source time) of the last ``window_s`` (the
    fastest delivery).  The rendered source time is local - offset - delay; the automatic delay is the median input
    interval plus the 90th percentile of the arrivals' lateness over the fastest."""

    def __init__(self, delay_s: float | None = None, window_s: float = 5.0):
        self.fixed = delay_s
        self.window = window_s
        self.rows: deque[tuple[float, float]] = deque()  # (arrival, source)

    def push(self, source: float, arrival: float) -> None:
        self.rows.append((arrival, source))
        while self.rows and self.rows[0][0] < arrival - self.window:
            self.rows.popleft()

    @property
    def offset(self) -> float:
        return min(a - s for a, s in self.rows)

    @property
    def delay(self) -> float:
        if self.fixed is not None:
            return self.fixed
        src = np.array([s for _, s in self.rows])
        late = np.array([a - s for a, s in self.rows]) - self.offset
        gap = float(np.median(np.diff(src))) if len(src) > 1 else 0.04
        return float(np.clip(gap + np.percentile(late, 90), 0.0, 0.2))

    def render_time(self, now: float) -> float:
        return now - self.offset - self.delay


def standing_frame(frames: list[dict], fps: float) -> tuple[dict | None, str]:
    """The stillest frame that stands (pico.stance_candidates' checks), or None and why not."""
    if len(frames) < 3:
        return None, "too few frames"
    P = lambda n: np.array([f[n][0] for f in frames])
    leg = float(np.median(np.linalg.norm(P("L_hip") - P("L_ankle"), axis=1)))
    kL = pico._angle(P("L_knee") - P("L_hip"), P("L_ankle") - P("L_knee"))
    kR = pico._angle(P("R_knee") - P("R_hip"), P("R_ankle") - P("R_knee"))
    spine = pico._angle(P("neck") - P("pelvis"), np.array([0, 0, 1.0]))
    arm = np.maximum(pico._angle(P("L_elbow") - P("L_shoulder"), np.array([0, 0, -1.0])),
                     pico._angle(P("R_elbow") - P("R_shoulder"), np.array([0, 0, -1.0])))
    allp = np.concatenate([P(n) for n in BONES], 1)
    v = np.r_[np.inf, np.linalg.norm(np.diff(allp, axis=0), axis=1) * fps / leg]
    feet = np.abs(P("L_ankle")[:, 2] - P("R_ankle")[:, 2]) / leg
    checks = {"still": v < STAND_SPEED, "feet level": feet < STAND_FEET, "spine up": spine < STAND_SPINE_DEG,
              "knees alike": np.abs(kL - kR) < STAND_KNEES_DEG, "arms down": arm < STAND_ARMS_DEG}
    ok = np.logical_and.reduce(list(checks.values()))
    if not ok.any():
        failed = [k for k, c in checks.items() if not c[1:].any()] or [min(checks, key=lambda k: checks[k].sum())]
        return None, "not standing: " + ", ".join(failed)
    t = int(np.flatnonzero(ok)[np.argmin(v[ok])])
    return frames[t], f"frame {t} of {len(frames)}, knee {(kL[t] + kR[t]) / 2:.1f} deg"


def world_of(frame: dict) -> tuple[R, np.ndarray]:
    """The calibration's world frame (as pico.standing_reference / normalize): turn about z so the frame faces +x and
    put its pelvis at x = y = 0."""
    left = frame["L_hip"][0] - frame["R_hip"][0]
    fwd = np.cross(left, [0, 0, 1.0])
    c = frame["pelvis"][0]
    return R.from_euler("z", -np.arctan2(fwd[1], fwd[0])), np.array([c[0], c[1], 0.0])


class PastRules:
    """preprocess.py's frame rules and finalize's arm-flip cut on the frames so far (a run rule marks a frame once
    its run so far is too long; a joint jump compares a step with the step before it)."""

    def __init__(self, fps: float, spec: ExclusionSpec | None = None):
        self.fps, self.spec = fps, spec or ExclusionSpec()
        self.prev_q = self.prev_step = self.prev_com = None
        self.air = self.lean = self.flip = 0

    def __call__(self, kin, q: np.ndarray, flip: bool) -> list[str]:
        s, fps, out = self.spec, self.fps, []
        low = float(kin.lowest_point_z[0])
        if low < -s.floor_penetration_m:
            out.append("floor_penetration")
        if kin.self_penetration_m[0] > s.self_penetration_m:
            out.append("self_penetration")
        if kin.joint_excess_rad[0] > s.joint_range_excess_rad:
            out.append("joint_range")
        if self.prev_q is not None:
            step = np.abs(q - self.prev_q)
            if self.prev_step is not None and np.any((step * fps > s.joint_jump_rad_s) & (step > s.joint_jump_ratio * self.prev_step)):
                out.append("joint_jump")
            self.prev_step = step
        self.prev_q = q.copy()
        if low > s.lifted_height_m:
            out.append("lifted")
        self.air = self.air + 1 if low > s.off_ground_height_m else 0
        if self.air > round(s.max_off_ground_s * fps):
            out.append("off_ground")
        com = kin.com_xy_w[0]
        speed = 0.0 if self.prev_com is None else float(np.linalg.norm(com - self.prev_com)) * fps
        self.prev_com = com.copy()
        outside = float(kin.com_outside_support_m[0])
        leaning = (np.isfinite(outside) and outside > s.support_margin_m and kin.trunk_tilt_rad[0] < np.radians(s.upright_tilt_deg)
                   and speed < s.quasi_static_speed_m_s)
        self.lean = self.lean + 1 if leaning else 0
        if self.lean > round(s.max_unsupported_s * fps):
            out.append("unsupported")
        self.flip = self.flip + 1 if flip else 0
        if self.flip >= MIN_FLIP_RUN:
            out.append("arm_flip")
        return out


class Retargeter:
    """One calibration: the standing reference ``ref`` (pico.standing_reference) on HOME; ``step`` turns the next
    human frame (the calibration's world frame) into the training frame's arrays."""

    def __init__(self, ref: dict, fps: float = OUTPUT_HZ, workdir: Path | None = None):
        self.fps = fps
        self.fmt = FORMATS[FMT]
        self.model = mujoco.MjModel.from_xml_path(core._register_robot())
        self.data = mujoco.MjData(self.model)
        cfg = core.calibrate(ref, FMT, self.model)
        workdir = Path(workdir or tempfile.mkdtemp(prefix="microban_live_"))
        workdir.mkdir(parents=True, exist_ok=True)
        cfg_path = workdir / "live_ik.json"
        cfg_path.write_text(json.dumps(cfg))
        self.gmr = core.MicrobanRetargeting(cfg, cfg_path, FMT, fps)
        self.gmr.reset_configuration()
        self.scale = cfg["human_scale_table"][self.fmt["pelvis"]]
        self.lo, self.hi = self.model.jnt_range[1:, 0], self.model.jnt_range[1:, 1]
        self.lowest = core.LowestPoint(self.model)
        # human_clearance's parts: (name, height against which the clearance is taken)
        leg = ref[self.fmt["L"]["hip"]][0][2] - ref[self.fmt["L"]["ankle"]][0][2]
        floor_ref = min(ref[self.fmt[s]["toe"]][0][2] for s in "LR")
        self.parts = [(self.fmt[s][p], ref[self.fmt[s][p]][0][2]) for s in "LR" for p in ("ankle", "toe")]
        self.parts += [(core.role_name(self.fmt, r if r in ("pelvis", "spine", "neck") else f"{s}.{r}"), floor_ref + c * leg)
                       for r, c in core.CONTACT_RADIUS.items() for s in (("L",) if r in ("pelvis", "spine", "neck") else ("L", "R"))]
        self.clearance = deque(maxlen=round(1.0 * fps) + 1)
        self.floor: float | None = None  # the clearance's floor in use (robot metres)
        self.checker = ClipChecker(collision_model())
        names = [self.model.joint(j).name for j in range(1, self.model.njnt)]
        if tuple(names) != self.checker.joint_names or self.checker.joint_names != stream.JOINT_NAMES:
            raise ValueError("the IK model, the training model and the stream name the joints differently")
        if self.checker.body_names != stream.BODY_NAMES:
            raise ValueError("the training model's bodies are not the stream's")
        self.arm = [(s, [self.model.site(n).id for n in (f"{s}_shoulder_c", f"{s}_elbow_c", f"{s}_hand")],
                     [core.role_name(self.fmt, f"{s[0].upper()}.{p}") for p in ("shoulder", "elbow", "wrist")],
                     7 + names.index(f"{s}_shoulder_roll")) for s in ("left", "right")]
        self.trunk = self.model.body("trunk").id
        self.rules = PastRules(fps)
        # The limb bones (child, joint, the bone's direction in the joint's frame at the reference): the IK reads a
        # bone's direction from its joint's orientation, so the orientation must turn with the bone.  bone_err_deg is
        # the largest angle between that direction and the one from the positions (a few degrees when it does).
        self.bones = [(hb, driver, axis) for hb, _, driver, _, axis, _ in self.gmr.chain if axis is not None]
        self.bone_err_deg = 0.0

    def step(self, frame: dict) -> tuple[np.ndarray, object, list[str]]:
        """(qpos, the training model's kinematics of this frame, the rules it breaks)."""
        q = self.gmr.retarget(core.to_input(frame)).copy()
        self.bone_err_deg = max(
            float(np.degrees(np.arccos(np.clip(R.from_quat(frame[j][1], scalar_first=True).apply(a) @ _unit(frame[c][0] - frame[j][0]), -1, 1))))
            for c, j, a in self.bones)
        q[7:] = np.clip(q[7:], self.lo, self.hi)
        d = self.data
        d.qpos[:] = q
        mujoco.mj_kinematics(self.model, d)
        flip = False
        for _, sites, roles, roll in self.arm:
            rob = d.site_xpos[sites][None]
            tgt = np.array([self.gmr.scaled_human_data[r][0] for r in roles])[None]
            flip |= bool(core.arm_flips(rob, tgt, d.xquat[self.trunk][None], np.degrees([q[roll]]))[0])
        h = min(frame[n][0][2] - z for n, z in self.parts)
        self.clearance.append(h)
        floor = self.scale * min(self.clearance)
        if self.floor is not None:
            floor = min(floor, self.floor + FLOOR_RISE_M_S / self.fps)
        self.floor = floor
        q[2] -= self.lowest(d) - (self.scale * h - floor)
        kin = self.checker.run(q[None, :3], q[None, 3:7], q[None, 7:])
        return q, kin, self.rules(kin, q[7:], flip)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def home_arrays(checker: ClipChecker | None = None) -> dict[str, np.ndarray]:
    """HOME standing at the origin, still."""
    checker = checker or ClipChecker(collision_model())
    q = home_qpos(checker.model)
    kin = checker.run(q[None, :3], q[None, 3:7], q[None, 7:])
    return {"joint_pos": q[7:].astype(np.float32), "joint_vel": np.zeros(stream.NJ, np.float32),
            "body_pos_w": kin.body_pos_w[0], "body_quat_w": kin.body_quat_w[0],
            "body_lin_vel_w": np.zeros((stream.NB, 3), np.float32), "body_ang_vel_w": np.zeros((stream.NB, 3), np.float32)}


def backward_velocities(cur: dict, prev: dict | None, fps: float) -> dict:
    if prev is None:
        return {"joint_vel": np.zeros(stream.NJ), "body_lin_vel_w": np.zeros((stream.NB, 3)),
                "body_ang_vel_w": np.zeros((stream.NB, 3))}
    q1, q0 = cur["body_quat_w"].astype(np.float64), prev["body_quat_w"].astype(np.float64)
    q0 = np.where(np.sum(q0 * q1, -1, keepdims=True) < 0.0, -q0, q0)
    w = 2.0 * quat_multiply((q1 - q0) * fps, quat_conjugate(q1))[..., 1:4]
    return {"joint_vel": (cur["joint_pos"] - prev["joint_pos"]) * fps,
            "body_lin_vel_w": (cur["body_pos_w"] - prev["body_pos_w"]) * fps, "body_ang_vel_w": w}


# JOINT_STEP_LIMITS in the stream's joint order: speed p99, speed change per tick p99, step p99.9, speed change p99.9.
_JOINT = np.array([JOINT_STEP_LIMITS[n] for n in stream.JOINT_NAMES]).T
_SLOPE = 1.875  # smootherstep's largest slope
_BEND = 10.0 / np.sqrt(3.0)  # and its largest second derivative


def smootherstep(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * u * (u * (6.0 * u - 15.0) + 10.0)


def _faded(u: np.ndarray) -> np.ndarray:
    """The distance a motion of speed 1 covers by ``u`` while its speed fades as 1 - smoothstep over a duration of 1
    (all of it: 1 / 2)."""
    u = np.clip(u, 0.0, 1.0)
    return u - u**3 + 0.5 * u**4


def sweep_s(gaps: list[tuple[np.ndarray, np.ndarray, np.ndarray]], floor_s: float, fps: float = OUTPUT_HZ) -> float:
    """The shortest smootherstep (s, at least ``floor_s``) that moves every gap with its speed and its speed change per
    tick within their limits (``(gap, speed, speed change per tick)`` arrays)."""
    t = floor_s
    for gap, speed, change in gaps:
        gap = np.abs(np.asarray(gap, float))
        t = max(t, float(np.max(_SLOPE * gap / speed)), float(np.max(np.sqrt(_BEND * gap / (change * fps)))))
    return t


def heading(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def blend(a: np.ndarray, b: np.ndarray, w: float) -> np.ndarray:
    """qpos ``a`` to ``b`` by ``w``: positions and joints linear, the trunk's orientation slerp."""
    q = a + w * (b - a)
    q[3:7] = slerp(a[3:7], b[3:7], w)
    return q


def _turn(a: np.ndarray, b: np.ndarray) -> float:
    return float(2.0 * np.arccos(min(1.0, abs(float(np.dot(a, b))))))


class Carry:
    """The reference through an interruption, from the last frame sent (qpos ``q0``; per-second velocities ``v0``:
    joints, trunk linear, trunk angular in the world).  Its motion carries on, the speed fading as 1 - smoothstep over
    COAST_S (longer for a speed whose fading would brake harder than the training reference's 99th percentile change
    of speed; a joint sooner if it would pass its range ``lo``-``hi``), the pose stands still, and from HOLD_S (or
    from ``stand_now``) it goes to the calibrated standing pose (``home``'s joints, trunk height and attitude) at the
    trunk's x, y and heading.  The trunk's height is not carried on: it puts the lowest point (``lowest(qpos)``, the
    training collision model) on the floor (z = 0), from where the last frame had it by a smootherstep within the
    99th percentiles, so the feet neither sink into the floor nor stay off it while the joints move."""

    def __init__(self, q0: np.ndarray, v0: tuple, home: np.ndarray, lo: np.ndarray, hi: np.ndarray, lowest,
                 fps: float = OUTPUT_HZ):
        jv, self.lv, self.av = (np.asarray(v, float) for v in v0)
        self.q0, self.jv, self.lowest = np.asarray(q0, float).copy(), jv, lowest
        fade = lambda speed, change: np.maximum(COAST_S, 1.5 * np.abs(speed) / (change * fps))  # 1 - smoothstep's 1.5
        room = np.maximum(np.where(jv > 0.0, hi - self.q0[7:], self.q0[7:] - lo), 0.0)
        self.joint_s = fade(jv, _JOINT[1])
        self.joint_s = np.where(np.abs(jv) * self.joint_s / 2.0 > room, 2.0 * room / np.maximum(np.abs(jv), 1e-9),
                                self.joint_s)
        self.trunk_s = float(fade(np.linalg.norm(self.lv), TRUNK_STEP_LIMITS[1]))
        self.turn_s = float(fade(np.linalg.norm(self.av), TURN_STEP_LIMITS[1]))
        self.hold_s = HOLD_S
        self.r0 = R.from_quat(self.q0[3:7], scalar_first=True)
        self.low0 = lowest(self.q0)
        self.land_s = sweep_s([(self.low0, TRUNK_STEP_LIMITS[0], TRUNK_STEP_LIMITS[1])], COAST_S, fps)
        self.end = self._coast(np.inf)
        self.stand = self.end.copy()
        self.stand[2] = home[2]
        self.stand[3:7] = (R.from_euler("z", heading(self.end[3:7])) * R.from_quat(home[3:7], scalar_first=True)
                           ).as_quat(scalar_first=True)
        self.stand[7:] = home[7:]
        self.low_stand = lowest(self.stand)
        self.stand_s = sweep_s([(self.stand[7:] - self.end[7:], _JOINT[0], _JOINT[1]),
                                (self.stand[2] - self.end[2] + lowest(self.end),  # the trunk's height, feet down
                                 TRUNK_STEP_LIMITS[0], TRUNK_STEP_LIMITS[1]),
                                (_turn(self.end[3:7], self.stand[3:7]), TURN_STEP_LIMITS[0], TURN_STEP_LIMITS[1])],
                               STAND_S, fps)

    def stand_now(self, t: float) -> None:
        """Go to the standing pose from ``t`` s instead of holding the pose (the motion fades out on the way)."""
        self.hold_s = min(self.hold_s, t)

    def _coast(self, t: float) -> np.ndarray:
        q = self.q0.copy()
        q[7:] += self.jv * self.joint_s * _faded(t / np.maximum(self.joint_s, 1e-9))
        q[:3] += self.lv * self.trunk_s * float(_faded(t / self.trunk_s))
        q[3:7] = (R.from_rotvec(self.av * self.turn_s * float(_faded(t / self.turn_s))) * self.r0
                  ).as_quat(scalar_first=True)
        return q

    def at(self, t: float) -> np.ndarray:
        """The reference ``t`` s after the last frame sent."""
        q, low = self._coast(t), self.low0 * (1.0 - smootherstep(t / self.land_s))
        if t > self.hold_s:
            w = smootherstep((t - self.hold_s) / self.stand_s)
            q, low = blend(q, self.stand, w), low + w * self.low_stand
        q[2] += low - self.lowest(q)
        return q


def _quintic(x0: np.ndarray, v0: np.ndarray, T: float, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A difference ``x0`` changing at ``v0`` brought to rest at zero in ``T`` by the quintic that overshoots least
    (Bollo's inertialization: x0 (1 - u)^4 (1 + 4u) + v0 T u (1 - u)^4, u = t / T): its value, rate and acceleration
    at the times ``t`` (rows) for each column."""
    u = np.clip(np.asarray(t, float) / T, 0.0, 1.0)[:, None]
    w = 1.0 - u
    x = x0 * w**4 * (1.0 + 4.0 * u) + v0 * T * u * w**4
    v = -20.0 * x0 * u * w**3 / T + v0 * w**3 * (1.0 - 5.0 * u)
    acc = w**2 * (-20.0 * x0 * (1.0 - 4.0 * u) + v0 * T * (20.0 * u - 8.0)) / T**2
    return x, v, acc


# Per channel (21 joints, the trunk's x, y, z, its turn about x, y, z): the speed change per tick (the training
# reference's 99th percentile) a return plans within.
_CHANGE = np.concatenate([_JOINT[1], np.full(3, TRUNK_STEP_LIMITS[1]), np.full(3, TURN_STEP_LIMITS[1])])


class Return:
    """Back to the live frames: each live frame plus the difference the reference had to it (where the interruption
    took it, ``x0``, and how fast it moved against the operator, ``v0``), fading out in ``seconds`` as Bollo's quintic
    (no step in the position or the speed; at rest at the end).  ``seconds`` is the shortest from ``floor_s`` whose
    braking (largest at the start: (8 |v0| T + 20 |x0|) / T^2) stays within the training reference's 99th percentile
    change of speed per tick in every channel (``_CHANGE``).  The height's channel (x0[23], v0[23]) is the lowest
    point's (the training collision model) against the live frame's, not the trunk's: the trunk's height puts the
    lowest point there, so the feet keep to the floor while the joints come back."""

    def __init__(self, x0: np.ndarray, v0: np.ndarray, floor_s: float, fps: float = OUTPUT_HZ):
        self.x0, self.v0 = x0, v0
        a = _CHANGE * fps
        need = (8.0 * np.abs(v0) + np.sqrt(64.0 * v0**2 + 80.0 * a * np.abs(x0))) / (2.0 * a)
        T = max(floor_s, 2.0 / fps, float(np.max(need)))
        u = np.linspace(0.0, 1.0, 41)
        while np.any(np.abs(_quintic(x0, v0, T, u * T)[2]) / fps > _CHANGE * (1.0 + 1e-6)):
            T += 1.0 / fps
        self.seconds = T

    def at(self, t: float, live: np.ndarray, live_low: float, lo: np.ndarray, hi: np.ndarray,
           lowest) -> tuple[np.ndarray, bool]:
        """The reference ``t`` s into the return (``live``: this tick's live qpos, its lowest point ``live_low``; the
        joints kept in ``lo``-``hi``) and whether the difference is gone."""
        x = _quintic(self.x0, self.v0, self.seconds, np.array([t]))[0][0]
        q = live.copy()
        q[7:] = np.clip(q[7:] + x[:21], lo, hi)
        q[:2] += x[21:23]
        q[3:7] = (R.from_rotvec(x[24:27]) * R.from_quat(live[3:7], scalar_first=True)).as_quat(scalar_first=True)
        # Not below both the floor and the live frame's lowest point: the live frame may come down to the floor
        # faster than the difference fades (the IK catching up the operator after a gap).
        q[2] += max(live_low + x[23], min(live_low, 0.0)) - lowest(q)
        return q, t >= self.seconds


class Guard:
    """No frame sent changes a joint, the trunk's position, its height or its orientation by more than the training
    reference's 99.9th percentile step in one tick (``GUARD_SHARE`` of it), and none sent through an interruption or a
    return changes that tick's change (the speed) by more than its 99.9th percentile change of speed either, its joints
    also within the servos' box (core.ServoLimit: the range, the speed, the acceleration, braking before a range end).
    The motion planned there stays within the 99th percentiles; this acts when the operator moves faster still, and
    on the live frames only past the 99.9th percentile steps.  A frame it changes keeps its lowest point (``lowest``,
    the training collision model) where it was, within the trunk's limits, so the feet stay on the floor."""

    def __init__(self, model: mujoco.MjModel, lo: np.ndarray, hi: np.ndarray, lowest, fps: float = OUTPUT_HZ):
        self.lo, self.hi, self.lowest, self.dt = lo, hi, lowest, 1.0 / fps
        self.servo = core.ServoLimit(model, fps)

    @staticmethod
    def deltas(q: np.ndarray, prev: np.ndarray) -> np.ndarray:
        """Per tick changes: 21 joints, the trunk's position (3), its turn (a world rotation vector, 3)."""
        turn = R.from_quat(q[3:7], scalar_first=True) * R.from_quat(prev[3:7], scalar_first=True).inv()
        return np.concatenate([q[7:] - prev[7:], q[:3] - prev[:3], turn.as_rotvec()])

    def __call__(self, q: np.ndarray, prev: np.ndarray | None, d_prev: np.ndarray | None,
                 planned: bool) -> tuple[np.ndarray, bool]:
        """``q`` limited from the last frame sent (``prev``, its changes ``d_prev``); ``planned``: a frame of an
        interruption or a return (all the limits), else a live one (the steps only).  The step is first shortened
        as a whole (the frame between ``prev`` and ``q``), which keeps the pose's own relations, such as the feet on
        the floor; only if no share of it meets every limit is each channel limited on its own.  Then the trunk's
        height puts the lowest point where ``q`` has it (``_floored``).  Records the frame for the servo box; whether
        it changed it."""
        if prev is None:
            self.servo.end(q)
            return q, False
        d = self.deltas(q, prev)
        lower, upper = self._joint_bounds(prev, d_prev, planned)
        share = self._share(d, d_prev, lower, upper, planned)
        out = d.copy()
        if share is not None:
            out *= share
        else:
            out[:21] = np.clip(out[:21], lower, upper)
            out[24:27] = self._limit(out[24:27], d_prev[24:27], planned, TURN_STEP_LIMITS)
            out[21:24] = self._limit(out[21:24], d_prev[21:24], planned, TRUNK_STEP_LIMITS)
        clamped = not np.allclose(out, d, rtol=0.0, atol=1e-9)
        if clamped:
            q = self._stepped(prev, self._floored(prev, out, d_prev, planned, lower, upper, self.lowest(q)))
        self.servo.end(q)
        return q, clamped

    def _floored(self, prev: np.ndarray, out: np.ndarray, d_prev: np.ndarray, planned: bool, lower: np.ndarray,
                 upper: np.ndarray, low: float) -> np.ndarray:
        """The limited step ``out`` with the trunk's height putting the lowest point at ``low``, within the trunk's
        limits.  When the joints' steps need a height step the trunk's limits do not allow (the legs straightening
        faster than the trunk may speed up), they are shortened by the largest share that keeps the lowest point
        there and the joints within their limits ``lower``-``upper`` (the change of speed, the servos' box): the body
        slows down rather than the feet going into the floor.  The trunk's change of speed is given up for the floor
        only if no such share keeps both."""

        def placed(k: float) -> np.ndarray:
            o = out.copy()
            o[:21] *= k
            o[23] += low - self.lowest(self._stepped(prev, o))
            return o

        # The least share of the joints' step within their limits (a joint that must go on moving).
        x = out[:21]
        need = np.where(x > 1e-12, lower / np.where(x > 1e-12, x, 1.0), np.where(x < -1e-12, upper / np.where(
            x < -1e-12, x, 1.0), 0.0))
        least = float(np.clip(np.max(need, initial=0.0), 0.0, 1.0))
        for speed in (planned, False):
            fits = lambda o: np.allclose(self._limit(o[21:24], d_prev[21:24], speed, TRUNK_STEP_LIMITS), o[21:24],
                                         rtol=0.0, atol=1e-9)
            o = placed(1.0)
            if fits(o):
                return o
            if not fits(placed(least)):
                continue
            k, over = least, 1.0
            for _ in range(12):
                mid = 0.5 * (k + over)
                k, over = (mid, over) if fits(placed(mid)) else (k, mid)
            return placed(k)
        o = placed(least)
        o[21:24] = self._limit(o[21:24], d_prev[21:24], planned, TRUNK_STEP_LIMITS)
        return o

    @staticmethod
    def _stepped(prev: np.ndarray, out: np.ndarray) -> np.ndarray:
        """``prev`` moved by the per tick changes ``out``."""
        q = prev.copy()
        q[7:] = prev[7:] + out[:21]
        q[:3] = prev[:3] + out[21:24]
        q[3:7] = (R.from_rotvec(out[24:27]) * R.from_quat(prev[3:7], scalar_first=True)).as_quat(scalar_first=True)
        return q

    def _limit(self, x: np.ndarray, x_prev: np.ndarray, planned: bool, limits: tuple) -> np.ndarray:
        """The trunk's step ``x`` (position or turn) within its limits: the change of speed through an interruption,
        the step, the height's step."""
        _, _, step, speed_change, *height = limits
        x = x.copy()
        if planned:
            dd = x - x_prev
            n = float(np.linalg.norm(dd))
            if n > GUARD_SHARE * speed_change * self.dt:
                x = x_prev + dd * (GUARD_SHARE * speed_change * self.dt / n)
        n = float(np.linalg.norm(x))
        if n > GUARD_SHARE * step:
            x *= GUARD_SHARE * step / n
        if height:
            x[2] = np.clip(x[2], -GUARD_SHARE * height[0], GUARD_SHARE * height[0])
        return x

    def _joint_bounds(self, prev: np.ndarray, d_prev: np.ndarray, planned: bool) -> tuple[np.ndarray, np.ndarray]:
        """Each joint's allowed step: the 99.9th percentile step, through an interruption the change of speed too,
        within the servos' box (which comes first: the range, braking before its ends)."""
        step = GUARD_SHARE * _JOINT[2]
        lower, upper = -step, step
        if planned:
            change = GUARD_SHARE * _JOINT[3] * self.dt
            lower, upper = np.maximum(lower, d_prev[:21] - change), np.minimum(upper, d_prev[:21] + change)
        self.servo.begin()
        box_lo, box_hi = self.servo.lower - prev[7:], self.servo.upper - prev[7:]
        lower, upper = np.maximum(lower, box_lo), np.minimum(upper, box_hi)
        empty = lower > upper
        return np.where(empty, box_lo, lower), np.where(empty, box_hi, upper)

    def _share(self, d: np.ndarray, d_prev: np.ndarray, lower: np.ndarray, upper: np.ndarray,
               planned: bool) -> float | None:
        """The largest share (0..1) of the step ``d`` that meets every limit, or None."""
        lo, hi = 0.0, 1.0
        bounds = [(d[:21], lower, upper)]
        height = GUARD_SHARE * TRUNK_STEP_LIMITS[4]
        bounds.append((d[23:24], np.array([-height]), np.array([height])))
        for x, a, b in bounds:
            moving = np.abs(x) > 1e-12
            if np.any(~moving & ((a > 0.0) | (b < 0.0))):
                return None
            r1, r2 = a[moving] / x[moving], b[moving] / x[moving]
            lo = max(lo, float(np.max(np.minimum(r1, r2), initial=0.0)))
            hi = min(hi, float(np.min(np.maximum(r1, r2), initial=1.0)))
        for sl, (_, _, step, speed_change, *_) in ((slice(21, 24), TRUNK_STEP_LIMITS), (slice(24, 27), TURN_STEP_LIMITS)):
            n = float(np.linalg.norm(d[sl]))
            if n > 1e-12:
                hi = min(hi, GUARD_SHARE * step / n)
            if planned:
                # |share d - d_prev| <= c: a quadratic in the share
                c = GUARD_SHARE * speed_change * self.dt
                a2, b1, c0 = n * n, -2.0 * float(d[sl] @ d_prev[sl]), float(d_prev[sl] @ d_prev[sl]) - c * c
                if a2 <= 1e-24:
                    if c0 > 0.0:
                        return None
                    continue
                disc = b1 * b1 - 4.0 * a2 * c0
                if disc < 0.0:
                    return None
                root = np.sqrt(disc)
                lo, hi = max(lo, (-b1 - root) / (2.0 * a2)), min(hi, (-b1 + root) / (2.0 * a2))
        return hi if lo <= hi else None


class Smoother:
    """The frames sent: each the average of the last ``ticks`` frames of the reference (qpos: positions and joints
    linear, the trunk's orientation by its rotation vectors from the middle one), so a frame is (ticks - 1) / 2 ticks
    late.  An average of consecutive frames keeps the joints in their ranges and changes no channel per tick, nor its
    speed, more than the frames averaged did (the Guard's limits hold); it takes out the corners the 30 Hz input's
    linear interpolation and the tracking's jitter put into the trunk's speed at 50 Hz.  Right after a reset it
    averages the frames it has."""

    def __init__(self, ticks: int = SMOOTH_TICKS):
        self.frames: deque[np.ndarray] = deque(maxlen=ticks)

    def reset(self) -> None:
        self.frames.clear()

    def __call__(self, q: np.ndarray) -> np.ndarray:
        self.frames.append(np.asarray(q, float).copy())
        qs = np.array(self.frames)
        out = qs.mean(axis=0)
        mid = R.from_quat(qs[len(qs) // 2, 3:7], scalar_first=True)
        turn = (R.from_quat(qs[:, 3:7], scalar_first=True) * mid.inv()).as_rotvec().mean(axis=0)
        out[3:7] = (R.from_rotvec(turn) * mid).as_quat(scalar_first=True)
        return out


class Pipeline:
    """Samples in (source clock), calibrate and start requests, one stream frame out per ``tick`` (local clock)."""

    def __init__(self, fps: float = OUTPUT_HZ, delay_s: float | None = None, workdir: Path | None = None, log=print,
                 background: bool = False):
        """``background``: build a calibration (about 0.15 s and the IK's first solve) on a thread, so the output
        ticks go on meanwhile (the live command); otherwise at once."""
        self.fps, self.workdir, self.log, self.background = fps, workdir, log, background
        self.ready: tuple | None = None  # a calibration built on the thread, installed by the next tick
        self.interp = Interpolator()
        self.clock = SourceClock(delay_s)
        self.retargeter: Retargeter | None = None
        self.world: tuple[R, np.ndarray] | None = None
        checker = ClipChecker(collision_model())
        self.home = home_arrays(checker)
        self.home_q = home_qpos(checker.model)
        self.active = False
        self.restart = False
        self.seq = 0
        self.last_arrival = None
        self.capture: list | None = None  # calibration window: [end source time, samples]
        self.hold_until = -1
        self.ticks = 0
        self.rendered = None  # source time of the last rendered frame
        self._reset_output()

    def _reset_output(self) -> None:
        self.sent = None  # the last frame sent: (qpos, its per-tick deltas, its arrays)
        self.shift = np.zeros(2)  # added to the live frames' x, y (each return starts where the interruption was)
        self.carry: Carry | None = None
        self.carried = 0  # ticks into the interruption
        self.back: Return | None = None
        self.returned = 0  # ticks into the return
        self.calm = 0  # ticks in a row the return was over and unguarded
        self.live_prev: tuple | None = None  # the last live qpos, whether it followed new input, its lowest point
        self.marks: list[str] = []  # the rules this tick's live frame broke
        self.carry_end: float | None = None  # when (s into the carry) the input was back
        self.free_ticks = 0  # ticks in a row the IK followed new input unbounded by its servos
        self.transition: str | None = None  # "carry" or "return" (diagnostics)
        self.smoother = Smoother()
        self.out_prev: dict | None = None  # the last frame sent (its arrays, for the velocities)
        self.guarded = 0  # frames the Guard changed

    # Input.
    def push(self, source_s: float, arrival_s: float, p: np.ndarray, q: np.ndarray) -> None:
        """One sample of the pseudo-bones (positions (16, 3), wxyz (16, 4)) at source time ``source_s``."""
        if not self.interp.push(source_s, p, q):
            return
        self.clock.push(source_s, arrival_s)
        self.last_arrival = arrival_s
        if self.capture is not None:
            self.capture[1].append(as_frame(p, q))
            if source_s >= self.capture[0]:
                self._finish_calibration()

    def request_calibration(self) -> None:
        end = (self.interp.t[-1] if self.interp.t else 0.0) + CALIBRATION_S
        self.capture = [end, []]
        self.log(f"calibration: stand still for {CALIBRATION_S:.0f} s")

    def calibrate_with(self, ref: dict, world: tuple[R, np.ndarray]) -> None:
        self._install(Retargeter(ref, self.fps, self.workdir), world)

    def _install(self, retargeter: Retargeter, world: tuple[R, np.ndarray]) -> None:
        self.retargeter, self.world = retargeter, world
        self.guard = Guard(retargeter.model, retargeter.lo, retargeter.hi, self._lowest, self.fps)
        self._reset_output()
        self.hold_until = -1
        if self.active:
            # A new calibration moves the world frame: tracking starts again only from left Y.
            self.active = False
            self.log("tracking stopped for the calibration; left Y starts it again")

    def _finish_calibration(self) -> None:
        frames, self.capture = self.capture[1], None
        fps = len(frames) / CALIBRATION_S
        frame, why = standing_frame(frames, fps)
        if frame is None:
            self.log(f"calibration failed ({why}); kept {'the previous one' if self.retargeter else 'none'}")
            return
        ref, world = pico.standing_reference(frame), world_of(frame)

        def build():
            r = Retargeter(ref, self.fps, self.workdir)
            r.step(self._in_world(frame, world))  # the IK's first solve (the arm branch), off the output ticks
            self.ready = (r, world)
            self.log(f"calibrated ({why}, root scale {r.scale:.4f}); left Y starts tracking")

        if self.background:
            threading.Thread(target=build, daemon=True).start()
        else:
            build()
            self._install(*self.ready)
            self.ready = None

    @staticmethod
    def _in_world(frame: dict, world: tuple[R, np.ndarray]) -> dict:
        rz, c = world
        return {n: (rz.apply(v[0] - c), (rz * R.from_quat(v[1], scalar_first=True)).as_quat(scalar_first=True))
                for n, v in frame.items()}

    def set_active(self, active: bool) -> None:
        if active and self.retargeter is None:
            self.log("start refused: not calibrated yet (left X, then stand still for 1 s)")
            return
        if active and not self.active:
            self.restart = True
        self.active = active
        self.log("tracking " + ("started" if active else "stopped"))

    # Output.
    def tick(self, now_s: float, sent_time_s: float | None = None) -> stream.Frame:
        if self.ready is not None:
            self._install(*self.ready)
            self.ready = None
        flags = 0
        arrays = None
        if self.retargeter is not None:
            flags |= stream.CALIBRATED
            live = None
            if self.last_arrival is None or now_s - self.last_arrival > STALE_S:
                flags |= stream.STALE
            else:
                t = self.clock.render_time(now_s)
                p, q, late = self.interp.sample(t)
                self.rendered = t
                qpos, kin, marks = self.retargeter.step(self._in_world(as_frame(p, q), self.world))
                live = (qpos, kin)
                self.marks = marks
                if marks:
                    self.hold_until = self.ticks + round(ExclusionSpec.pad_s * self.fps)
                if late:
                    flags |= stream.LATE
                if marks or self.ticks <= self.hold_until:
                    flags |= stream.HELD
            self._send(live, bool(flags & (stream.STALE | stream.LATE | stream.HELD)),
                       not flags & (stream.STALE | stream.LATE))
            arrays = self._smoothed(self.sent[0])
        if arrays is None:
            arrays = self.home
        if self.active:
            flags |= stream.ACTIVE
            if self.restart and flags & stream.CALIBRATED:
                flags |= stream.RESTART
                self.restart = False
        out = stream.Frame(seq=self.seq, sent_time_s=now_s if sent_time_s is None else sent_time_s, flags=flags,
                           **{k: np.asarray(arrays[k], np.float32) for k, _ in stream.FIELDS})
        self.seq = (self.seq + 1) & 0xFFFFFFFF
        self.ticks += 1
        return out

    def _send(self, live: tuple | None, interrupted: bool, fresh: bool) -> dict:
        """This tick's frame: the live one (``(qpos, kinematics)`` of the IK this tick, None when the body is stale),
        or, while ``interrupted`` (the body missing, late or held), the Carry, and back from it the Return, through
        the Guard, with velocities from the frames sent.  ``fresh``: the IK followed new input this tick (not stale,
        not late)."""
        dt = 1.0 / self.fps
        q_live = None
        if live is not None:
            q_live = live[0].copy()
            q_live[:2] += self.shift
        free = fresh and not any(self.retargeter.gmr.servo_bound)
        self.free_ticks = self.free_ticks + 1 if free else 0
        if self.carry is not None and not interrupted and self.carry_end is None:
            self.carry_end = self.carried * dt
        if interrupted or (self.carry is not None and not self._caught_up(fresh)):
            if self.carry is None:
                if self.sent is None:
                    q0, v0 = self.home_q, (np.zeros(stream.NJ), np.zeros(3), np.zeros(3))
                else:
                    a = self.sent[2]
                    q0, v0 = self.sent[0], (a["joint_vel"], a["body_lin_vel_w"][0], a["body_ang_vel_w"][0])
                self.carry = Carry(q0, v0, self.home_q, self.retargeter.lo, self.retargeter.hi, self._lowest,
                                   self.fps)
                self.carried, self.back, self.carry_end = 0, None, None
            self.carried += 1
            if "unsupported" in self.marks:
                # A pose whose weight is outside its support is not held: it goes to the standing pose now.
                self.carry.stand_now(self.carried * dt)
            planned, self.transition = self.carry.at(self.carried * dt), "carry"
        else:
            if self.carry is not None:
                # Back: the live frames start where the reference is (x, y) and the difference fades out.  Their
                # speed is the operator's when the IK has caught up (free_ticks >= 2); if it has not after
                # CATCH_UP_S, the joints' speed is not known and taken as zero.
                prev, a = self.live_prev[0].copy(), self.sent[2]
                shift = self.sent[0][:2] - prev[:2]
                self.shift += shift
                q_live[:2] += shift
                prev[:2] += shift
                live_v = Guard.deltas(q_live, prev) / dt
                if self.free_ticks < 2:
                    live_v[:21] = 0.0
                out_v = np.concatenate([a["joint_vel"], a["body_lin_vel_w"][0], a["body_ang_vel_w"][0]])
                x0, v0 = -Guard.deltas(prev, self.sent[0]), out_v - live_v
                x0[23], v0[23] = self._lowest(self.sent[0]) - self.live_prev[2], 0.0  # the lowest point's
                self.back = Return(x0, v0, min(self.carried * dt, RETURN_S), self.fps)
                self.returned, self.calm, self.carry = 0, 0, None
            planned, self.transition = None, None
            if self.back is not None:
                self.returned += 1
                planned, done = self.back.at(self.returned * dt, q_live, float(live[1].lowest_point_z[0]),
                                             self.retargeter.lo, self.retargeter.hi, self._lowest)
                self.transition = "return"
        if live is not None:
            self.live_prev = (q_live.copy(), fresh, float(live[1].lowest_point_z[0]))
        prev = (None, None) if self.sent is None else self.sent[:2]
        q, clamped = self.guard(q_live if planned is None else planned, *prev, planned is not None)
        self.guarded += clamped
        if self.back is not None:
            # Over once the difference is gone and two frames in a row were the live ones unchanged (their speed too).
            self.calm = self.calm + 1 if done and not clamped else 0
            if self.calm >= 2:
                self.back = None
        if planned is None and not clamped:
            kin = live[1]
            pos = kin.body_pos_w[0].astype(np.float64)
            pos[:, :2] += self.shift
            arrays = {"joint_pos": q[7:], "body_pos_w": pos, "body_quat_w": kin.body_quat_w[0].astype(np.float64)}
        else:
            c = self.retargeter.checker
            set_qpos(c.model, c.data, q[:3], q[3:7], q[7:])
            arrays = {"joint_pos": q[7:], "body_pos_w": c.data.xpos[c.body_ids].copy(),
                      "body_quat_w": c.data.xquat[c.body_ids].copy()}
        arrays.update(backward_velocities(arrays, None if self.sent is None else self.sent[2], self.fps))
        deltas = np.zeros(27) if self.sent is None else Guard.deltas(q, self.sent[0])
        self.sent = (q, deltas, arrays)
        return arrays

    def _lowest(self, q: np.ndarray) -> float:
        """The lowest collision point of qpos ``q`` on the training model (as the floor rules measure it)."""
        c = self.retargeter.checker
        set_qpos(c.model, c.data, q[:3], q[3:7], q[7:])
        return c._lowest_point()

    def _smoothed(self, q: np.ndarray) -> dict:
        """The frame sent for the reference ``q`` of this tick (Smoother), its velocities from the frames sent."""
        q = self.smoother(q)
        c = self.retargeter.checker
        set_qpos(c.model, c.data, q[:3], q[3:7], q[7:])
        arrays = {"joint_pos": q[7:], "body_pos_w": c.data.xpos[c.body_ids].copy(),
                  "body_quat_w": c.data.xquat[c.body_ids].copy()}
        arrays.update(backward_velocities(arrays, self.out_prev, self.fps))
        self.out_prev = arrays
        return arrays

    def _caught_up(self, fresh: bool) -> bool:
        """Whether the live frames follow the operator again after an interruption: the IK followed new input with
        neither its servos' speed nor their acceleration bounding it (it is not catching up a gap) on this tick and
        the one before, so that the last step is the operator's speed; or CATCH_UP_S have passed since the input was
        back (two fresh frames)."""
        waited = self.carried / self.fps - self.carry_end
        before = self.live_prev is not None and self.live_prev[1]
        return self.free_ticks >= 2 or (fresh and before and waited > CATCH_UP_S)
