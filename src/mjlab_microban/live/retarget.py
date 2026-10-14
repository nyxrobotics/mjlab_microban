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
  minimum of the last second (the data: +-1 s).
* Poses: forward kinematics on the training collision model (body_pos_w, body_quat_w as build.py).
* Velocities: backward differences (the data: central differences).
* Exclusion: preprocess.py's rules and finalize's arm-flip cut, judged on the frames so far.  A marked frame and the
  pad_s after the last mark are held at the last unmarked frame (velocities 0) instead of being cut.

PICO's 24 joints become the pseudo-bones of pico.py (skeleton.FORMATS["g1"]), the joint whose global rotation is
the bone's rotation: thigh = hip, shin = knee, foot = ankle, upper arm = shoulder, forearm = elbow, hand = wrist; the
torso is spine_1 (position) with spine_3's (chest) rotation, the neck base the neck joint, the toe the foot joint
with the ankle's rotation.
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
from mjlab_microban.wholebody.data.fk import quat_conjugate, quat_multiply
from mjlab_microban.wholebody.data.preprocess import ClipChecker, ExclusionSpec
from mjlab_microban.wholebody.retarget import core, pico
from mjlab_microban.wholebody.retarget.model import collision_model, home_qpos
from mjlab_microban.wholebody.retarget.pipeline import MIN_FLIP_RUN
from mjlab_microban.wholebody.retarget.skeleton import FORMATS

FMT = "g1"
OUTPUT_HZ = 50.0
STALE_S = 0.25
CALIBRATION_S = 1.0
# PICO's body joints (XRoboToolkit, SMPL order).
PICO_JOINTS = (
    "pelvis", "left_hip", "right_hip", "spine_1", "left_knee", "right_knee", "spine_2", "left_ankle", "right_ankle",
    "spine_3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand",
)
# pseudo-bone -> (position joint, orientation joint)
PICO_BONES = {
    "pelvis": ("pelvis", "pelvis"), "torso": ("spine_1", "spine_3"), "neck": ("neck", "spine_3"),
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
        q[2] -= self.lowest(d) - self.scale * (h - min(self.clearance))
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
        self.home = home_arrays()
        self.active = False
        self.restart = False
        self.seq = 0
        self.last_arrival = None
        self.capture: list | None = None  # calibration window: [end source time, samples]
        self.prev_live = None  # the last frame's arrays before holding (velocities)
        self.last_sent = None  # the last unheld frame (what a hold repeats)
        self.hold_until = -1
        self.ticks = 0
        self.rendered = None  # source time of the last rendered frame

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
        self.prev_live = self.last_sent = None
        self.hold_until = -1
        self.restart = self.active

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
            self.log(f"calibrated ({why}, root scale {r.scale:.4f})")

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
        stale = self.last_arrival is None or now_s - self.last_arrival > STALE_S
        if self.retargeter is not None:
            flags |= stream.CALIBRATED
            if stale:
                flags |= stream.STALE
                arrays = self._still(self.last_sent)
            else:
                t = self.clock.render_time(now_s)
                p, q, late = self.interp.sample(t)
                self.rendered = t
                if late:
                    flags |= stream.LATE
                qpos, kin, marks = self.retargeter.step(self._in_world(as_frame(p, q), self.world))
                live = {"joint_pos": qpos[7:], "body_pos_w": kin.body_pos_w[0].astype(np.float64),
                        "body_quat_w": kin.body_quat_w[0].astype(np.float64)}
                vel = backward_velocities(live, self.prev_live, self.fps)
                self.prev_live = live
                if marks:
                    self.hold_until = self.ticks + round(ExclusionSpec.pad_s * self.fps)
                if marks or self.ticks <= self.hold_until:
                    flags |= stream.HELD
                    arrays = self._still(self.last_sent or {**live, **vel})
                else:
                    arrays = {**live, **vel}
                    self.last_sent = arrays
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

    def _still(self, arrays: dict | None) -> dict:
        if arrays is None:
            return self.home
        return {**arrays, "joint_vel": np.zeros(stream.NJ), "body_lin_vel_w": np.zeros((stream.NB, 3)),
                "body_ang_vel_w": np.zeros((stream.NB, 3))}
