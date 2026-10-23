# Adapted from Teleopit f926386 train_mimic/benchmarking.py and
# train_mimic/scripts/benchmark.py (Apache-2.0,
# https://github.com/BotRunner64/Teleopit).  Changes: the evaluation set run
# in pieces that start where the reference stands quietly, results per motion
# kind, error lines equal to the reward stds, a progress line on every piece
# whose reference travels, no reset during a run, the robot started at rest
# at HOME, a fall line and a line on the leg targets' step-to-step change.

"""Evaluate a checkpoint on the evaluation set, per motion kind.

Pass (decided 2026-10-15).  Whole-body tracking is complete when the robot
does not fall however the operator moves; the hands' and feet's height off
the reference (the tracking error) is accepted.  So the full run
(``verdicts["final"]["pass"]``) passes on the fall line on the
evaluation-only set (the PICO recordings) and the action-rate line on both
sets; success, the errors and progress are reported, not judged.

Start (as in teleoperation).  On the robot, teleoperation starts with the
robot at rest at HOME and the operator standing, and from then on the robot
sees the operator's motion without a break: the reference never jumps from
HOME into a pose in motion.  So every piece starts at a frame where the
reference stands quietly (``standing_frames``: both soles on the floor, the
trunk near upright, at standing height and slow).  The robot is placed at rest at HOME (joint
velocities 0, the trunk at HOME's height and tilt) under the reference
trunk's horizontal position, facing the reference's heading.  For the first
``SETTLE_S`` = 1 s the reference holds the piece's first frame (the operator
stands while the robot comes from HOME into the operator's pose), standing
still: its joint, trunk and body velocities read 0 (the frame's own are those
of the motion through it, up to 21.5 rad/s at a joint and 7.8 rad/s at the
trunk on the training kinds' set), and nothing is judged: no termination
fails the piece and no error is taken.  Then the
reference plays on from that frame without a break, and every one of its
steps is judged.

Pieces.  Each evaluation section (one window of the evaluation shard) is cut
at its standing frames: the first piece starts at the section's first
standing frame, and a new one at the first standing frame at least
``clip_seconds`` (10 s) after the previous start that leaves at least
``clip_seconds`` to the section's end.  A piece runs to the next one's start
(or the section's end), so a motion that keeps the operator from standing
(lying on the floor and getting up, a fight, a run) stays whole in one piece
however long it is (the PICO set's pieces run 1.1-203 s, 12.0 s on average;
the training kinds' set's 1.0-36 s).  The frames before a section's first
standing frame are not evaluated (the robot cannot meet them from HOME
without a jump of the reference) and a section without a standing frame is
left out: the CMU sections are the last 20 % of their takes and the LAFAN1
ones are cut at the frames the data excludes, so many start in motion.  The
PICO set keeps 2739 of its 3543 s, the training kinds' set 4729 of 8604 s,
and every kind keeps judged seconds (the fewest: cmurun, 6 s).  The earlier
start (pieces every 10 s from any frame, the reference playing from the
first step) judged 3030 and 5807 s, but 49-59 % of its pieces were past a
termination line on their first step, only because the reference jumped
from HOME to a pose in motion.

Conditions.  Observation noise, pushes and the physics randomization of
training are off.  The robot's own conditions (battery voltage and its drop,
actuator and sensor delays) are drawn from their ranges with the seed: those
of training, but for the goal delay (the robot's 3-6 physics steps redrawn at
every step; training draws 0-6 once per episode, env_cfg.py).  No env is
reset during a run, so the draws do not depend on the
policy: two checkpoints meet the same conditions piece by piece.  The
physics on the GPU is not bit-reproducible (its sums run in a varying
order; torch's deterministic settings do not change it), so two runs of one
checkpoint and seed differ piece by piece (the full run's model_10000 on the
PICO set's earlier pieces, twice: 5 of 489 pieces changed success, picotwist
91 and 88 %, 1.32 and 1.34 falls per minute).  A value within a few points
of a line can fall on either side of it in another run.  A piece that ended
(a termination) keeps running to its end for the fall and action-rate
measures.

Success and errors (reported).  A piece succeeds when no termination fires
in its judged steps (the end of the reference is a time-out, not a failure).
A kind's success rate is the share of its judged seconds in successful
pieces; the kinds below ``FINAL_SUCCESS_RATE`` are listed.  For every
successful piece the RMS over its judged steps of each tracking error (the
quantity inside the reward's exp, square-rooted) is taken against the
reference of the step (the observed target); per kind the RMS over all
judged steps of its successful pieces is compared with that reward's std
(RMS <= std: the reward stays above exp(-1) on average), the kinds above it
listed.

Progress (reported; a small-run check).  The trunk velocity cannot tell a
robot standing still from a walking one at Microban's speeds, so the trunk's
horizontal displacement over the judged steps of each successful piece is
projected on the reference's: progress = sum(d_robot . d_ref) /
sum(|d_ref|^2) over a kind's travelling pieces, 0 standing still, 1 covering
the reference's ground, and it is held to ``MIN_PROGRESS``.  A piece
travels when its reference trunk moves at least ``PROGRESS_TRAVEL_M`` =
BODY_POS_STD_M / (1 - MIN_PROGRESS) across the floor over the judged steps
(from the motion file, whatever the policy does): a robot that ends such a
piece within the trunk position std of the reference reaches MIN_PROGRESS,
one standing still does not.  It is taken on every kind with a travelling
piece, whatever its name (LAFAN1's walk, CMU's cmuwalk, the PICO recordings'
walking).

Action-rate line (violence).  Per kind, the mean over steps of the sum over
the 12 leg joints of the squared change of the servo target in one step (the
form of the action_rate_l2 reward, rad^2) is at most ``MAX_LEG_ACTION_RATE``
= 3.36: the IK tracking policy that runs on the robot without violent motion
(mjlab_microban_teleop_v12 run 2026-10-11_23-19-20_home_481503d292_pico_v13_v13g2,
model_14999) walking (its eight walking commands, arms at HOME, 128 rollouts
of 10 s; standing, left out, measures 0.0003, and with it the 144 rollouts
2.98) measures 3.355 over the same steps: from 1 s into a rollout until the
robot falls.  Every piece counts, failed ones included, from 1 s into the
piece (the judged steps) until it falls.

Fall line.  A piece falls when the robot trunk tilts past ``FALL_TILT_RAD`` =
45 deg from vertical (HOME stands at 10 deg) while the reference trunk is within
``FALL_REF_TILT_RAD`` = 30 deg of vertical, or when, past 45 deg, it stays
for ``FALL_PAST_REF_S`` = 0.5 s more than ``FALL_PAST_REF_TILT_RAD`` = 30 deg
past the reference trunk's tilt (tilts taken up to 90 deg) or
``FALL_BELOW_REF_M`` = 6 cm below the reference trunk (the fall then dates
from the first of those steps), at any step of the piece, the first second
included.  A robot that follows a reference leaning or lying down is no
fall; one down on the floor under a reference leaning on its feet is, without
waiting for the reference to come back upright (the upright rule alone
counted 25 such falls 0.06-12.7 s late and missed those whose piece ended
first; the numbers are at FALL_PAST_REF_TILT_RAD).  The pieces'
lengths vary and the rollouts of the line's source ran 6 s, so falls are
counted per minute standing: a kind's or a set's falls over the time its
pieces ran before their fall (a piece's whole run, the first second
included, when it does not fall).  On an evaluation-only set (the operator's
motions: the PICO recordings) at most ``MAX_FALLS_PER_MINUTE`` = 0.425: the
IK tracking policy above in its push check (144 rollouts of 6 s, standing and
walking in eight directions, pushed by 0.4 m/s every second) fell 6 times by
the same tilt rule over 846.58 s standing, 6 of 144 rollouts (its reference
is upright, so the rule under a leaning reference adds none there).  On the
training kinds' set the rate is reported, not judged: it holds motions the
operator does not ask the robot for (sprints, jumps, falls).

Kinds.  Every kind of the training set and walk, run and sprint must be in
the evaluation set; an evaluation-only set (the PICO recordings, whose kinds
are not training kinds) is judged on its own kinds, by the action-rate line
and the fall line.

The share of steps whose reference joint speed exceeds the XC330's no-load
speed is reported per kind; it is not a line.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import mujoco
import numpy as np

from mjlab_microban.policy_contract import ACTION_JOINT_NAMES
from mjlab_microban.robot.home_pose import FOOT_BODIES, HOME, MICROBAN_XML, SOLE_GEOMS
from mjlab_microban.wholebody.data.hdf5 import find_motion_shards, motion_category, read_strings
from mjlab_microban.wholebody.env_cfg import (
    ACTOR_GROUP,
    ANCHOR_BODY_NAME,
    BODY_POS_STD_M,
    JOINT_POS_STD_RAD,
    LIN_VEL_STD_M_S,
    ORI_STD_RAD,
)

# The error lines: (name, std) of the reward each error belongs to.
ERROR_LINES = {
    "body_pos_m": BODY_POS_STD_M,
    "trunk_ori_rad": ORI_STD_RAD,
    "joint_pos_rad": JOINT_POS_STD_RAD,
    "trunk_lin_vel_m_s": LIN_VEL_STD_M_S,
}
FINAL_SUCCESS_RATE = 0.9
LOCOMOTION_KINDS = ("walk", "run", "sprint")
MIN_PROGRESS = 0.5
PROGRESS_TRAVEL_M = BODY_POS_STD_M / (1.0 - MIN_PROGRESS)
SERVO_VIN_V = 11.1
# The start recorded in a report's metadata: at rest at HOME, the reference from a standing frame.
START = "standing"
SETTLE_S = 1.0
LEG_JOINT_NAMES = tuple(name for name in ACTION_JOINT_NAMES if "shoulder" not in name and "elbow" not in name)
MAX_LEG_ACTION_RATE = 3.36
ACTION_RATE_FROM_S = 1.0  # the IK tracking policy was measured from 1 s into its rollouts
FALL_TILT_RAD = math.radians(45.0)
FALL_REF_TILT_RAD = math.radians(30.0)
# Under a reference past FALL_REF_TILT_RAD, the robot is down past it: its trunk past FALL_TILT_RAD and past the
# reference trunk's tilt by FALL_PAST_REF_TILT_RAD (tilts taken up to 90 deg: lying is lying, so a robot lying under a
# reference lying or on all fours is no fall) or FALL_BELOW_REF_M below the reference trunk, for FALL_PAST_REF_S.
# Measured on the PICO set (model_29999 of runs 2026-10-12_07-24-30_wholebody_main and
# 2026-10-14_00-39-43_wholebody_v4) and the training kinds' set (the first), the reference held still in the first
# second: in the 181 runs past 45 deg the robot followed a leaning reference and came back up from, it was at most 12
# deg past the reference's tilt (99 %: 9) and at most 3.0 cm below it; in the 25 falls that the upright rule counted
# only when the reference came back within 30 deg (0.06-12.7 s after the robot passed 45 deg), every one went past 30
# deg or 6 cm (up to 60 deg and 15 cm).  30 deg and 6 cm are 2.5 and 2 times the following's largest, which never
# reaches them even for a step; 0.5 s is longer than a fall takes from 45 deg to the floor (a body other than the feet
# touches it 0.08 s later at the median, 0.28 s at 90 %), so a robot passing through is not counted, and a fall in a
# piece's last second is.
FALL_PAST_REF_TILT_RAD = math.radians(30.0)
FALL_BELOW_REF_M = 0.06
FALL_PAST_REF_S = 0.5
# The IK tracking policy's push check: 6 falls over 846.58 s standing (144 rollouts of 6 s).  Its reference (stand,
# walk) is upright, where a robot past 45 deg is a fall at once, so the rule under a leaning reference adds none there.
MAX_FALLS_PER_MINUTE = 60.0 * 6 / 846.58
# A standing frame (a piece's start).  A sole is on the floor when its lowest corner is within 10 mm of it:
# the PICO set's slow upright frames have the higher sole within 7 mm in 75 % and 13 mm in 90 % (tracking noise).
STAND_SOLE_M = 0.010
# Near upright: within the fall line's upright reference (30 deg; HOME stands at 10 deg).
STAND_TILT_RAD = FALL_REF_TILT_RAD
# Still: a third of the walking pace (the walk kinds' median trunk speed 0.12-0.16 m/s).
STAND_TRUNK_SPEED_M_S = 0.05
# Standing, not squatting: the trunk at most 2 cm below HOME's height (0.170 m); the PICO set's standing frames
# gather at 0.15-0.18 m, its squats and kneels (soles down, trunk upright) at 0.10-0.13 m.
STAND_TRUNK_DROP_M = 0.02


def servo_no_load_speed_rad_s(vin: float = SERVO_VIN_V) -> float:
    """XC330 no-load speed of the BAM identification: vin / kt."""
    params = json.loads((Path(__file__).resolve().parents[1] / "robot" / "xc330_params.json").read_text())
    return vin / float(params["kt"])


def settle_steps(step_dt: float) -> int:
    """Steps not judged at a piece's start: the way from HOME into the reference's standing pose."""
    return round(SETTLE_S / step_dt)


@dataclass(frozen=True)
class Piece:
    piece_id: int
    window_id: int
    clip_name: str
    kind: str
    start_s: float  # the standing frame the piece starts at (s into the window)
    steps: int  # run: settle_steps, then one judged step per reference step
    seconds: float
    settle_steps: int  # not judged (the way from HOME; the reference holds its first frame)
    judged_seconds: float
    overspeed_share: float
    ref_travel_m: float  # the reference trunk's horizontal displacement over the judged steps


@dataclass(frozen=True)
class PieceResult:
    piece_id: int
    window_id: int
    clip_name: str
    kind: str
    start_s: float
    seconds: float
    judged_seconds: float
    steps: int  # judged steps run until the piece ended or failed
    success: bool
    failure_reason: str | None
    overspeed_share: float
    ref_travel_m: float
    # RMS tracking errors of a successful piece (None after a termination).
    body_pos_m: float | None
    trunk_ori_rad: float | None
    joint_pos_rad: float | None
    trunk_lin_vel_m_s: float | None
    # Horizontal trunk displacements of a successful piece: d_robot . d_ref and |d_ref|^2.
    progress_dot_m2: float | None
    ref_displacement_sq_m2: float | None
    # The fall line: the time into the piece of the first fall, None if it did not fall.
    fall_s: float | None
    # The action-rate line: its mean per step over action_rate_steps steps (None without one).
    leg_action_rate: float | None
    action_rate_steps: int


def dataset_kinds(motion_file: str | Path) -> set[str]:
    """The motion kinds of a dataset's clips."""
    kinds = set()
    for shard in find_motion_shards(motion_file):
        with h5py.File(shard, "r") as h5:
            kinds.update(motion_category(name) for name in read_strings(h5, "clip_names"))
    return kinds


def _trunk_xy(trunk_xy: np.ndarray, start: int, length: int, fps: float, time_s: float) -> np.ndarray:
    """The reference trunk's (x, y) at ``time_s`` into a window (linear between frames)."""
    position = min(time_s * fps, length - 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, length - 1)
    weight = position - lower
    return (1.0 - weight) * trunk_xy[start + lower] + weight * trunk_xy[start + upper]


@lru_cache(maxsize=1)
def sole_corners() -> dict[str, np.ndarray]:
    """{foot body: (48, 3)} the corners of its six sole boxes in the foot body's frame."""
    model = mujoco.MjModel.from_xml_path(str(MICROBAN_XML))
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))
    corners = {}
    for side, body in FOOT_BODIES.items():
        boxes = []
        for name in SOLE_GEOMS[side]:
            geom = model.geom(name)
            if model.geom_bodyid[geom.id] != model.body(body).id:
                raise ValueError(f"{name} is not on {body}")
            rotation = np.zeros(9)
            mujoco.mju_quat2Mat(rotation, geom.quat)
            boxes.append(geom.pos + (signs * geom.size) @ rotation.reshape(3, 3).T)
        corners[body] = np.concatenate(boxes)
    return corners


def _rotate(quat: np.ndarray, points: np.ndarray) -> np.ndarray:
    """(T, P, 3): ``points`` (P, 3) turned by each of the (T, 4) (w, x, y, z) quaternions."""
    w, v = quat[:, None, :1], np.broadcast_to(quat[:, None, 1:], (len(quat), len(points), 3))
    t = 2.0 * np.cross(v, points[None])
    return points[None] + w * t + np.cross(v, t)


def standing_frames(h5: h5py.File, anchor_body: str = ANCHOR_BODY_NAME) -> np.ndarray:
    """(frames,) the reference stands quietly: both soles on the floor, the trunk near upright, at standing
    height and slow."""
    bodies = read_strings(h5, "body_names")
    pos = np.asarray(h5["body_pos_w"], dtype=np.float64)
    quat = np.asarray(h5["body_quat_w"], dtype=np.float64)
    standing = np.ones(len(pos), dtype=bool)
    for body, corners in sole_corners().items():
        index = bodies.index(body)
        lowest = pos[:, index, None, 2] + _rotate(quat[:, index], corners)[..., 2]
        standing &= lowest.min(axis=1) <= STAND_SOLE_M
    trunk = bodies.index(anchor_body)
    x, y = quat[:, trunk, 1], quat[:, trunk, 2]
    standing &= np.arccos(np.clip(1.0 - 2.0 * (x**2 + y**2), -1.0, 1.0)) <= STAND_TILT_RAD
    standing &= pos[:, trunk, 2] >= HOME.root_pos[2] - STAND_TRUNK_DROP_M
    standing &= np.linalg.norm(np.asarray(h5["body_lin_vel_w"][:, trunk]), axis=-1) <= STAND_TRUNK_SPEED_M_S
    return standing


def piece_starts(standing: np.ndarray, clip_seconds: float, fps: float) -> list[int]:
    """A section's piece starts (frames): its first standing frame, then the first standing frame at least
    ``clip_seconds`` after the previous start that leaves at least ``clip_seconds`` to the section's end."""
    last = len(standing) - 1
    starts: list[int] = []
    for frame in np.flatnonzero(standing).tolist():
        if not starts or (frame - starts[-1] >= clip_seconds * fps and last - frame >= clip_seconds * fps):
            starts.append(frame)
    return starts


def load_pieces(
    motion_file: str | Path,
    joint_names: Sequence[str],
    clip_seconds: float,
    step_dt: float,
    no_load_speed: float,
    anchor_body: str = ANCHOR_BODY_NAME,
) -> list[Piece]:
    """Every window (in MotionLib's order) cut at its standing frames (``piece_starts``); a piece runs to
    the next one's start or the window's end, after ``settle_steps`` holding its first frame."""
    settle = settle_steps(step_dt)
    pieces: list[Piece] = []
    window_id = 0
    for shard in find_motion_shards(motion_file):
        with h5py.File(shard, "r") as h5:
            names = read_strings(h5, "clip_names")
            shard_joints = read_strings(h5, "joint_names")
            columns = [shard_joints.index(name) for name in joint_names]
            fast = np.any(np.abs(np.asarray(h5["joint_vel"])[:, columns]) > no_load_speed, axis=1)
            standing = standing_frames(h5, anchor_body)
            anchor = read_strings(h5, "body_names").index(anchor_body)
            trunk_xy = np.asarray(h5["body_pos_w"][:, anchor, :2], dtype=np.float64)
            for first_frame, length, fps, source in zip(
                np.asarray(h5["clip_starts"]), np.asarray(h5["clip_lengths"]), np.asarray(h5["clip_fps"]),
                np.asarray(h5["source_clip_ids"]), strict=True,
            ):
                first_frame, length, fps = int(first_frame), int(length), float(fps)
                starts = piece_starts(standing[first_frame : first_frame + length], clip_seconds, fps)
                for start, end in zip(starts, [*starts[1:], length - 1]):
                    judged = int(math.floor((end - start) / fps / step_dt + 1e-6))
                    if judged < 1:
                        continue
                    begin = start / fps
                    last = min(int(math.ceil((begin + judged * step_dt) * fps)), length - 1)
                    # The judged steps' targets: the reference from begin + step_dt to begin + judged * step_dt.
                    travel = _trunk_xy(trunk_xy, first_frame, length, fps, begin + judged * step_dt) - _trunk_xy(
                        trunk_xy, first_frame, length, fps, begin + step_dt
                    )
                    pieces.append(Piece(
                        piece_id=len(pieces), window_id=window_id, clip_name=names[source],
                        kind=motion_category(names[source]), start_s=begin, steps=settle + judged,
                        seconds=(settle + judged) * step_dt, settle_steps=settle, judged_seconds=judged * step_dt,
                        overspeed_share=float(np.mean(fast[first_frame + start : first_frame + last + 1])),
                        ref_travel_m=float(np.linalg.norm(travel)),
                    ))
                window_id += 1
    return pieces


def _weighted(rows: Iterable[PieceResult], value: str, weight: str) -> float | None:
    rows = [r for r in rows if getattr(r, value) is not None]
    total = sum(getattr(r, weight) for r in rows)
    return sum(getattr(r, value) * getattr(r, weight) for r in rows) / total if total > 0 else None


def summarize(results: Sequence[PieceResult]) -> dict[str, Any]:
    by_kind: dict[str, list[PieceResult]] = defaultdict(list)
    for result in results:
        by_kind[result.kind].append(result)
    kinds = {}
    for kind, rows in sorted(by_kind.items()):
        succeeded = [r for r in rows if r.success]
        judged = sum(r.judged_seconds for r in rows)
        falls = sum(r.fall_s is not None for r in rows)
        standing = sum(r.seconds if r.fall_s is None else r.fall_s for r in rows)
        entry: dict[str, Any] = {
            "pieces": len(rows),
            "seconds": sum(r.seconds for r in rows),
            "judged_seconds": judged,
            "success_rate": sum(r.judged_seconds for r in succeeded) / judged,
            "falls": falls,
            "standing_seconds": standing,
            "falls_per_minute": 60.0 * falls / standing,
            "leg_action_rate": _weighted(rows, "leg_action_rate", "action_rate_steps"),
            "non_finite_pieces": sum(r.failure_reason == "non_finite" for r in rows),
            "overspeed_share": _weighted(rows, "overspeed_share", "seconds"),
        }
        reasons: dict[str, int] = defaultdict(int)
        for r in rows:
            if not r.success:
                reasons[r.failure_reason or "unknown"] += 1
        entry["failure_reasons"] = dict(sorted(reasons.items()))
        steps = sum(r.steps for r in succeeded)
        for name in ERROR_LINES:
            entry[name] = (
                math.sqrt(sum(getattr(r, name) ** 2 * r.steps for r in succeeded) / steps) if steps else None
            )
        # Progress on the travelling pieces; the line holds on a kind that has any (travel_seconds > 0).
        entry["travel_seconds"] = sum(r.judged_seconds for r in rows if r.ref_travel_m >= PROGRESS_TRAVEL_M)
        travelled = [r for r in succeeded if r.ref_travel_m >= PROGRESS_TRAVEL_M]
        ref_sq = sum(r.ref_displacement_sq_m2 for r in travelled)
        entry["progress"] = sum(r.progress_dot_m2 for r in travelled) / ref_sq if ref_sq > 0 else None
        kinds[kind] = entry
    return kinds


def verdicts(
    kinds: dict[str, Any],
    expected_kinds: Iterable[str] = (),
    require_locomotion: bool = True,
    hold_fall_line: bool = False,
) -> dict[str, Any]:
    """The full run's pass and the small run's checks.

    The full run passes (decided 2026-10-15: not falling however the operator
    moves; the tracking errors are accepted) when every kind's leg action rate
    is at most MAX_LEG_ACTION_RATE and, with ``hold_fall_line`` (an
    evaluation-only set), the falls over all pieces are at most
    MAX_FALLS_PER_MINUTE.  The kinds below 90 % success or above the error
    lines and the progress are reported, not judged.  The small run's checks:
    no kind at 0 %, progress on every kind with a travelling piece.  A kind
    expected but not evaluated or a non-finite piece fails both; walk, run and
    sprint are expected unless ``require_locomotion`` is off (an
    evaluation-only set)."""
    required = set(expected_kinds) | (set(LOCOMOTION_KINDS) if require_locomotion else set())
    missing = sorted(required - set(kinds))
    errors_ok = {k: all(v[n] is not None and v[n] <= std for n, std in ERROR_LINES.items()) for k, v in kinds.items()}
    progress = {k: v["progress"] for k, v in kinds.items() if v["travel_seconds"] > 0}
    progress_ok = not missing and all(v is not None and v >= MIN_PROGRESS for v in progress.values())
    finite = not any(v["non_finite_pieces"] for v in kinds.values())
    standing = sum(v["standing_seconds"] for v in kinds.values())
    falls_per_minute = 60.0 * sum(v["falls"] for v in kinds.values()) / standing if standing > 0 else None
    fall_ok = not hold_fall_line or (falls_per_minute is not None and falls_per_minute <= MAX_FALLS_PER_MINUTE)
    final = {
        "kinds_without_pieces": missing,
        "kinds_above_action_rate_line": sorted(
            k for k, v in kinds.items() if v["leg_action_rate"] is None or v["leg_action_rate"] > MAX_LEG_ACTION_RATE
        ),
        "falls_per_minute": falls_per_minute,
        "fall_line_held": hold_fall_line,
        "fall_ok": fall_ok,
        "finite": finite,
        # Reported, not judged.
        "kinds_below_success_rate": sorted(k for k, v in kinds.items() if v["success_rate"] < FINAL_SUCCESS_RATE),
        "kinds_above_error_lines": sorted(k for k, ok in errors_ok.items() if not ok),
        "progress": progress,
        "progress_ok": progress_ok,
    }
    final["pass"] = bool(kinds and not missing and not final["kinds_above_action_rate_line"] and fall_ok and finite)
    small = {
        "kinds_without_pieces": missing,
        "kinds_never_succeeding": sorted(k for k, v in kinds.items() if v["success_rate"] == 0.0),
        "progress": progress,
        "progress_ok": progress_ok,
        "finite": finite,
    }
    small["pass"] = bool(kinds and not missing and not small["kinds_never_succeeding"] and progress_ok and finite)
    return {"final": final, "small_run": small}


KIND_COLUMNS = (
    "kind", "pieces", "seconds", "judged_seconds", "success_rate", "falls_per_minute", "leg_action_rate",
    "overspeed_share", "travel_seconds", "progress", *ERROR_LINES,
)


def write_outputs(
    out_dir: Path,
    stem: str,
    metadata: dict[str, Any],
    results: Sequence[PieceResult],
    expected_kinds: Iterable[str],
    require_locomotion: bool = True,
    hold_fall_line: bool = False,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    kinds = summarize(results)
    report = {
        "metadata": metadata,
        "error_lines": ERROR_LINES,
        "final_success_rate": FINAL_SUCCESS_RATE,
        "min_progress": MIN_PROGRESS,
        "progress_travel_m": PROGRESS_TRAVEL_M,
        "settle_s": SETTLE_S,
        "stand_sole_m": STAND_SOLE_M,
        "stand_tilt_rad": STAND_TILT_RAD,
        "stand_trunk_speed_m_s": STAND_TRUNK_SPEED_M_S,
        "stand_trunk_drop_m": STAND_TRUNK_DROP_M,
        "max_leg_action_rate": MAX_LEG_ACTION_RATE,
        "leg_joint_names": LEG_JOINT_NAMES,
        "fall_tilt_rad": FALL_TILT_RAD,
        "fall_ref_tilt_rad": FALL_REF_TILT_RAD,
        "fall_past_ref_tilt_rad": FALL_PAST_REF_TILT_RAD,
        "fall_below_ref_m": FALL_BELOW_REF_M,
        "fall_past_ref_s": FALL_PAST_REF_S,
        "max_falls_per_minute": MAX_FALLS_PER_MINUTE,
        "verdicts": verdicts(kinds, expected_kinds, require_locomotion, hold_fall_line),
        "kinds": kinds,
        "pieces": [asdict(r) for r in results],
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(report, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    with (out_dir / f"{stem}-kinds.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(KIND_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for kind, entry in kinds.items():
            writer.writerow({"kind": kind, **entry})
    with (out_dir / f"{stem}-pieces.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(results[0])))
        writer.writeheader()
        writer.writerows(asdict(r) for r in results)
    return report


def format_summary(report: dict[str, Any]) -> str:
    def number(value: float | None, pattern: str) -> str:
        return "-" if value is None else format(value, pattern)

    lines = [f"{'kind':<18}{'n':>5}{'min':>6}{'success':>9}{'fall/min':>9}{'rate':>6}{'fast':>6}{'progress':>9}"
             + "".join(f"{n:>19}" for n in ERROR_LINES)]
    for kind, v in report["kinds"].items():
        errors = "".join(f"{number(v[n], '.3f'):>13}/{std:<5}" for n, std in ERROR_LINES.items())
        lines.append(f"{kind:<18}{v['pieces']:>5}{v['judged_seconds'] / 60:>6.1f}{100 * v['success_rate']:>8.0f}%"
                     f"{v['falls_per_minute']:>9.2f}{number(v['leg_action_rate'], '.2f'):>6}"
                     f"{100 * (v['overspeed_share'] or 0.0):>5.0f}%{number(v['progress'], '.2f'):>9}{errors}")
    final, small = report["verdicts"]["final"], report["verdicts"]["small_run"]
    progress = {k: None if v is None else round(v, 2) for k, v in small["progress"].items()}
    held = "" if final["fall_line_held"] else " (not a line here)"
    lines.append(f"final pass: {final['pass']}  falls {number(final['falls_per_minute'], '.2f')}/min "
                 f"(<= {MAX_FALLS_PER_MINUTE:.3f}){held}  "
                 f"above action-rate line ({MAX_LEG_ACTION_RATE}): {final['kinds_above_action_rate_line']}  "
                 f"missing kinds: {final['kinds_without_pieces']}")
    lines.append(f"reported, not judged: below 90%: {final['kinds_below_success_rate']}  "
                 f"above error lines: {final['kinds_above_error_lines']}  progress ok: {final['progress_ok']}")
    lines.append(f"small-run checks: {small['pass']}  never succeeding: {small['kinds_never_succeeding']}  "
                 f"progress (>= {MIN_PROGRESS} where the reference travels >= {PROGRESS_TRAVEL_M:.2f} m): {progress}")
    return "\n".join(lines)


def tilt_rad(quat: Any) -> Any:
    """Angle between a body's z axis and the world's, from (w, x, y, z) quaternions (torch)."""
    return (1.0 - 2.0 * (quat[..., 1] ** 2 + quat[..., 2] ** 2)).clamp(-1.0, 1.0).arccos()


def fell(robot_quat: Any, ref_quat: Any) -> Any:
    """The robot trunk past FALL_TILT_RAD while the reference trunk is within FALL_REF_TILT_RAD of upright."""
    return (tilt_rad(robot_quat) > FALL_TILT_RAD) & (tilt_rad(ref_quat) < FALL_REF_TILT_RAD)


def down_past_reference(robot_quat: Any, ref_quat: Any, robot_z: Any, ref_z: Any) -> Any:
    """The robot trunk past FALL_TILT_RAD and past the reference trunk's tilt by FALL_PAST_REF_TILT_RAD or below its
    height by FALL_BELOW_REF_M (a fall when it lasts FALL_PAST_REF_S)."""
    robot_tilt = tilt_rad(robot_quat)
    lying = math.pi / 2.0
    return (robot_tilt > FALL_TILT_RAD) & (
        (robot_tilt.clamp(max=lying) - tilt_rad(ref_quat).clamp(max=lying) > FALL_PAST_REF_TILT_RAD)
        | (ref_z - robot_z > FALL_BELOW_REF_M)
    )


def _place_at_home(env: Any, command: Any) -> None:
    """Put the robot at rest at HOME under the reference trunk, facing the reference's heading."""
    import torch
    from mjlab.utils.lab_api.math import quat_mul, yaw_quat

    robot = command.robot
    root = robot.data.default_root_state.clone()
    root[:, 0:2] = command.body_pos_w[:, 0, 0:2]
    root[:, 2] += env.scene.env_origins[:, 2]
    root[:, 3:7] = quat_mul(yaw_quat(command.body_quat_w[:, 0]), root[:, 3:7])
    root[:, 7:] = 0.0
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), torch.zeros_like(robot.data.default_joint_vel))
    robot.write_root_state_to_sim(root)
    robot.clear_state()


def rollout(
    env: Any,
    policy: Any,
    pieces: Sequence[Piece],
    *,
    record_every: int = 0,
) -> tuple[list[PieceResult], list[Any]]:
    """Run ``pieces`` (one env each, cut by ``load_pieces``) from HOME for their steps.

    ``env`` is the unwrapped ManagerBasedRlEnv (``auto_reset=False``).  With
    ``record_every`` > 0 the actor history of env 0 is recorded every that
    many steps until the longest piece ends.
    """
    import torch
    from mjlab.utils.lab_api.math import quat_error_magnitude
    from tensordict import TensorDict

    n = len(pieces)
    device = env.device
    settle = settle_steps(env.step_dt)
    if any(p.settle_steps != settle for p in pieces):
        raise ValueError(f"the pieces were not cut for a step of {env.step_dt} s")
    rate_from = round(ACTION_RATE_FROM_S / env.step_dt)
    past_steps = round(FALL_PAST_REF_S / env.step_dt)
    env.reset()
    command = env.command_manager.get_term("motion")
    ids = torch.arange(n, device=device)
    start_s = torch.tensor([p.start_s for p in pieces], device=device, dtype=command.motion_times.dtype)
    # The operator stands at the piece's first frame while the robot comes from HOME: held, standing still.
    command.hold(torch.ones(n, dtype=torch.bool, device=device))
    command.reset_to_motion(ids, torch.tensor([p.window_id for p in pieces], device=device), start_s)
    _place_at_home(env, command)
    # env.reset's sequence for the placed envs: the reference one step ahead,
    # the neck target, then fresh observation histories.
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.command_manager.compute(dt=0.0)
    env.event_manager.apply(mode="step", dt=env.step_dt)
    env.sim.sense()
    env.observation_manager.reset(ids)
    obs = TensorDict(env.observation_manager.compute(update_history=True), batch_size=[n])
    # Set at the settle's end, where the judged steps start.
    robot_start_xy = command.robot_anchor_pos_w[:, :2].clone()
    ref_start_xy = robot_start_xy.clone()

    leg_ids = command.tracked_joint_ids[[list(command.cfg.joint_names).index(name) for name in LEG_JOINT_NAMES]]
    budgets = torch.tensor([p.steps for p in pieces], device=device)
    sums = {name: torch.zeros(n, device=device) for name in ERROR_LINES}
    running = torch.ones(n, dtype=torch.bool, device=device)
    success = torch.zeros(n, dtype=torch.bool, device=device)
    steps = torch.zeros(n, dtype=torch.long, device=device)
    fall_step = torch.full((n,), -1, dtype=torch.long, device=device)
    rate_sum = torch.zeros(n, device=device)
    rate_steps = torch.zeros(n, dtype=torch.long, device=device)
    # The current run of down_past_reference: its length and its steps' share of the action-rate sums.
    past_run = torch.zeros(n, dtype=torch.long, device=device)
    past_rate_sum = torch.zeros(n, device=device)
    past_rate_steps = torch.zeros(n, dtype=torch.long, device=device)
    robot_end_xy = torch.zeros(n, 2, device=device)
    ref_end_xy = torch.zeros(n, 2, device=device)
    previous_target = command.robot.data.joint_pos_target[:, leg_ids].clone()
    reasons: list[str | None] = [None] * n
    recorded = []
    for step in range(int(budgets.max())):
        if record_every and step % record_every == record_every - 1:
            recorded.append(obs[ACTOR_GROUP][0].detach().cpu().clone())
        if step == settle:
            # The reference plays on from its frame: this step's target is still the held frame, the next one moves.
            command.hold(torch.zeros(n, dtype=torch.bool, device=device))
        # The step's target, which its rewards and terminations compare the robot with.
        ref = {
            "body_pos": command.body_pos_relative_w.clone(),
            "trunk_quat": command.anchor_quat_w.clone(),
            "joint_pos": command.joint_pos.clone(),
            "trunk_lin_vel": command.anchor_lin_vel_w.clone(),
            "trunk_xy": command.anchor_pos_w[:, :2].clone(),
            "trunk_z": command.anchor_pos_w[:, 2].clone(),
        }
        with torch.inference_mode():
            actions = policy(obs)
        raw, _, terminated, _, _ = env.step(actions)
        # No reset during a run: a reset would add a row to every env's
        # history and draw new conditions; an ended piece just keeps going.
        env._manual_reset_pending.zero_()
        obs = TensorDict(raw, batch_size=[n])
        in_piece = step < budgets
        fell_now = in_piece & (fall_step < 0) & fell(command.robot_anchor_quat_w, ref["trunk_quat"])
        fall_step[fell_now] = step
        past = down_past_reference(
            command.robot_anchor_quat_w, ref["trunk_quat"], command.robot_anchor_pos_w[:, 2], ref["trunk_z"]
        )
        past_run = torch.where(past, past_run + 1, torch.zeros_like(past_run))
        # The leg targets' change from the previous step, while standing from 1 s on.
        target = command.robot.data.joint_pos_target[:, leg_ids].clone()
        counted = in_piece & (fall_step < 0) & (step > rate_from)
        change = torch.where(counted, torch.square(target - previous_target).sum(-1), torch.zeros_like(rate_sum))
        rate_sum += change
        rate_steps += counted.long()
        past_rate_sum = torch.where(past, past_rate_sum + change, torch.zeros_like(past_rate_sum))
        past_rate_steps = torch.where(past, past_rate_steps + counted.long(), torch.zeros_like(past_rate_steps))
        previous_target = target
        # Down past the reference for FALL_PAST_REF_S: fallen since the run's first step, which no longer stands.
        fell_past = in_piece & (fall_step < 0) & (past_run >= past_steps)
        fall_step[fell_past] = step - past_steps + 1
        rate_sum -= torch.where(fell_past, past_rate_sum, torch.zeros_like(rate_sum))
        rate_steps -= torch.where(fell_past, past_rate_steps, torch.zeros_like(rate_steps))
        judged = running & (step >= settle)
        errors = {
            "body_pos_m": torch.square(ref["body_pos"] - command.robot_body_pos_w).sum(-1).mean(-1),
            "trunk_ori_rad": quat_error_magnitude(ref["trunk_quat"], command.robot_anchor_quat_w) ** 2,
            "joint_pos_rad": torch.square(ref["joint_pos"] - command.robot_joint_pos).mean(-1),
            "trunk_lin_vel_m_s": torch.square(ref["trunk_lin_vel"] - command.robot_anchor_lin_vel_w).sum(-1),
        }
        for name, value in errors.items():
            sums[name] += torch.where(judged, value, torch.zeros_like(value))
        steps += judged.long()
        failed = terminated & judged
        for index in failed.nonzero().flatten().tolist():
            reasons[index] = next(
                (name for name in env.termination_manager.active_terms
                 if not env.termination_manager.get_term_cfg(name).time_out
                 and bool(env.termination_manager.get_term(name)[index])),
                "terminated",
            )
        running &= ~failed
        finished = running & (step + 1 == budgets)
        robot_end_xy[finished] = command.robot_anchor_pos_w[finished, :2]
        ref_end_xy[finished] = ref["trunk_xy"][finished]
        success |= finished
        running &= ~finished
        if step + 1 == settle:
            # The judged steps start: displacements from here.
            robot_start_xy = command.robot_anchor_pos_w[:, :2].clone()
            ref_start_xy = ref["trunk_xy"].clone()

    robot_disp = (robot_end_xy - robot_start_xy).cpu()
    ref_disp = (ref_end_xy - ref_start_xy).cpu()
    results = []
    for index, piece in enumerate(pieces):
        ok = bool(success[index])
        rms = {name: math.sqrt(float(sums[name][index]) / max(int(steps[index]), 1)) for name in ERROR_LINES}
        if ok and not all(math.isfinite(value) for value in rms.values()):
            ok, reasons[index] = False, "non_finite"
        rate_n = int(rate_steps[index])
        results.append(
            PieceResult(
                piece_id=piece.piece_id, window_id=piece.window_id, clip_name=piece.clip_name, kind=piece.kind,
                start_s=piece.start_s, seconds=piece.seconds, judged_seconds=piece.judged_seconds,
                steps=int(steps[index]), success=ok, failure_reason=None if ok else (reasons[index] or "terminated"),
                overspeed_share=piece.overspeed_share, ref_travel_m=piece.ref_travel_m,
                **{name: (value if ok else None) for name, value in rms.items()},
                progress_dot_m2=float(robot_disp[index] @ ref_disp[index]) if ok else None,
                ref_displacement_sq_m2=float(ref_disp[index] @ ref_disp[index]) if ok else None,
                fall_s=None if int(fall_step[index]) < 0 else (int(fall_step[index]) + 1) * env.step_dt,
                leg_action_rate=float(rate_sum[index]) / rate_n if rate_n else None,
                action_rate_steps=rate_n,
            )
        )
    return results, recorded
