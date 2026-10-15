"""The whole-body evaluation: pieces start where the reference stands, and the verdicts judge every kind
by the fall and action-rate lines (success, errors and progress reported)."""

from __future__ import annotations

import csv
import math

import h5py
import numpy as np
import pytest
import torch

from mjlab_microban.robot.home_pose import HOME
from mjlab_microban.wholebody import benchmark
from mjlab_microban.wholebody.data.hdf5 import FRAME_KEYS, write_motion_shard

JOINTS = ("a", "b")
BODIES = ("trunk", "foot", "foot_2")


def _standing_feet() -> tuple[np.ndarray, dict[str, float]]:
    """A foot quaternion with the sole facing down, and each foot body's height with its sole on the floor."""
    quat = np.array([math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0])  # the body's -y down
    rotation = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    return quat, {body: -float((c @ rotation.T)[:, 2].min()) for body, c in benchmark.sole_corners().items()}


def _section(name: str, frames: int, fps: int = 30, moving: slice | None = None) -> dict:
    """Standing quietly (soles on the floor, trunk upright and still), the trunk moving in ``moving``."""
    clip = {key: np.zeros((frames, 2) if key.startswith("joint") else (frames, len(BODIES), 4 if "quat" in key else 3))
            for key in FRAME_KEYS}
    clip["body_quat_w"][..., 0] = 1.0
    clip["body_pos_w"][:, 0, 2] = HOME.root_pos[2]
    quat, heights = _standing_feet()
    for index, body in enumerate(BODIES[1:], start=1):
        clip["body_quat_w"][:, index] = quat
        clip["body_pos_w"][:, index, 2] = heights[body]
    if moving is not None:
        clip["body_lin_vel_w"][moving, 0, 0] = 0.2
    return {"name": name, "fps": fps, **clip}


def _write(path, sections) -> None:
    write_motion_shard(sections, path, body_names=BODIES, joint_names=JOINTS, max_window_frames=None)


def test_a_standing_frame_has_both_soles_down_and_the_trunk_upright_high_and_still(tmp_path):
    # Frame 0 stands; 1 lifts a foot 12 mm; 2 tilts the trunk 35 deg; 3 moves it at 0.06 m/s; 5 squats
    # (the trunk 2.5 cm below HOME's height); 4 is at the lines (a foot 10 mm up, 30 deg, 0.05 m/s, 2 cm down).
    section = _section("dance1_subject5", 6)
    section["body_pos_w"][1, 1, 2] += 0.012
    for frame, degrees in ((2, 35.0), (4, 29.9)):
        section["body_quat_w"][frame, 0] = (math.cos(math.radians(degrees) / 2), 0.0, math.sin(math.radians(degrees) / 2), 0.0)
    section["body_lin_vel_w"][3, 0] = (0.0, 0.06, 0.0)
    section["body_lin_vel_w"][4, 0] = (0.0, 0.0, 0.0499)
    section["body_pos_w"][4, 2, 2] += 0.0099
    section["body_pos_w"][4, 0, 2] -= 0.0199
    section["body_pos_w"][5, 0, 2] -= 0.025
    _write(tmp_path / "motion.h5", [section])
    with h5py.File(tmp_path / "motion.h5", "r") as h5:
        assert benchmark.standing_frames(h5).tolist() == [True, False, False, False, True, False]


def test_pieces_start_where_the_reference_stands_and_run_on_without_a_break(tmp_path):
    # 50 s at 30 fps, moving for its first 3 s and from 13.3 to 30 s: pieces from 3 s (the first standing
    # frame), 13 s (10 s later), 30 s (the next standing frame) and 40 s (leaving 10 s).  A section never
    # standing has none; one standing from 2 s to its end at 2.97 s has one; one standing only on its last
    # frame has nothing to judge.
    path = tmp_path / "motion.h5"
    _write(path, [
        _section("walk1_subject5__f0-1501", 1501, moving=slice(0, 90)),
        _section("run1_subject5__f0-90", 90, moving=slice(None)),
        _section("run2_subject5__f0-90", 90, moving=slice(0, 60)),
        _section("run3_subject5__f0-31", 31, moving=slice(0, 30)),
    ])
    # The 13.3-30 s motion (frames 400-899).
    with h5py.File(path, "r+") as h5:
        h5["body_lin_vel_w"][400:900, 0, 0] = 0.2
    pieces = benchmark.load_pieces(path, JOINTS, 10.0, 0.02, no_load_speed=10.0)
    assert benchmark.settle_steps(0.02) == 50
    assert [(p.window_id, p.kind, round(p.start_s, 6), p.steps - p.settle_steps) for p in pieces] == [
        (0, "walk", 3.0, 500), (0, "walk", 13.0, 850), (0, "walk", 30.0, 500), (0, "walk", 40.0, 500),
        (2, "run", 2.0, 48),
    ]
    # Every reference step is judged, after the settle (the reference held on the first frame).
    assert all(p.settle_steps == 50 and p.seconds == pytest.approx(p.steps * 0.02) for p in pieces)
    assert [round(p.judged_seconds, 6) for p in pieces] == [10.0, 17.0, 10.0, 10.0, 0.96]


def test_pieces_measure_the_reference_trunks_travel(tmp_path):
    # The trunk walks along x at 0.03 m/s for 10 s, then stands: over the judged steps (the reference
    # from 0.02 s into the piece) 0.2994 m, then 0.
    section = _section("dance1_subject5__f0-601", 601)
    t = np.arange(601) / 30.0
    section["body_pos_w"][:, 0, 0] = 0.03 * np.minimum(t, 10.0)
    path = tmp_path / "motion.h5"
    _write(path, [section])
    pieces = benchmark.load_pieces(path, JOINTS, 10.0, 0.02, no_load_speed=10.0)
    assert [round(p.ref_travel_m, 6) for p in pieces] == [0.2994, 0.0]


def _kind(
    success_rate: float,
    progress: float | None = 1.0,
    travel_seconds: float | None = None,
    falls: int = 0,
    standing_seconds: float = 600.0,
    rate: float | None = 0.5,
) -> dict:
    """A kind's summary; it travels (the progress line holds) when it has a progress."""
    travel = (60.0 if progress is not None else 0.0) if travel_seconds is None else travel_seconds
    return {"success_rate": success_rate, "non_finite_pieces": 0, "progress": progress, "travel_seconds": travel,
            "pieces": 10, "falls": falls, "standing_seconds": standing_seconds,
            "falls_per_minute": 60.0 * falls / standing_seconds, "leg_action_rate": rate,
            **{name: 0.5 * std for name, std in benchmark.ERROR_LINES.items()}}


def test_verdicts_fail_missing_kinds_and_a_small_run_standing_still():
    kinds = {"walk": _kind(1.0), "run": _kind(1.0), "sprint": _kind(0.95), "jumps": _kind(0.92, None)}
    assert benchmark.verdicts(kinds, {"walk", "jumps"})["final"]["pass"]
    # A training kind with no evaluation piece fails both.
    missing = benchmark.verdicts(kinds, {"walk", "jumps", "fallAndGetUp"})
    assert missing["final"]["kinds_without_pieces"] == ["fallAndGetUp"]
    assert not missing["final"]["pass"] and not missing["small_run"]["pass"]
    # So does a missing locomotion kind.
    missing = benchmark.verdicts({k: v for k, v in kinds.items() if k != "sprint"})
    assert not missing["final"]["pass"] and not missing["small_run"]["pass"]
    # A walk that stands still fails the small run; the full run reports it.
    still = benchmark.verdicts({**kinds, "walk": _kind(1.0, progress=0.05)})
    assert not still["final"]["progress_ok"] and still["final"]["pass"] and not still["small_run"]["pass"]
    # A non-finite piece fails both.
    broken = benchmark.verdicts({**kinds, "jumps": {**_kind(0.92, None), "non_finite_pieces": 1}}, {"walk", "jumps"})
    assert not broken["final"]["pass"] and not broken["small_run"]["pass"]


def test_progress_holds_on_every_kind_whose_reference_travels():
    kinds = {"walk": _kind(1.0), "run": _kind(1.0), "sprint": _kind(0.95)}
    # Whatever its name (CMU's walking, a fight that moves around), a travelling kind is held to progress ...
    for name in ("cmuwalk", "fight"):
        still = benchmark.verdicts({**kinds, name: _kind(1.0, progress=0.1)})
        assert still["final"]["progress"][name] == 0.1
        assert not still["final"]["progress_ok"] and not still["small_run"]["pass"]
    # ... and so is one whose travelling pieces all failed (no progress measured).
    assert not benchmark.verdicts({**kinds, "fight": _kind(0.5, progress=None, travel_seconds=20.0)})["small_run"]["pass"]
    # A kind that never travels has no progress line.
    judged = benchmark.verdicts({**kinds, "cmujump": _kind(1.0, progress=None)})
    assert judged["small_run"]["pass"] and "cmujump" not in judged["final"]["progress"]


def _result(
    kind: str,
    travel: float,
    dot: float,
    success: bool = True,
    seconds: float = 10.0,
    settle_s: float = 0.0,
    fall_s: float | None = None,
    rate: float | None = 0.5,
    rate_steps: int = 450,
):
    """A piece result whose reference trunk travelled ``travel`` m and the robot ``dot`` / ``travel`` of it."""
    errors = {name: 0.5 * std for name, std in benchmark.ERROR_LINES.items()}
    return benchmark.PieceResult(
        piece_id=0, window_id=0, clip_name=f"{kind}1_subject5", kind=kind, start_s=0.0, seconds=seconds,
        judged_seconds=seconds - settle_s, steps=round((seconds - settle_s) / 0.02), success=success,
        failure_reason=None if success else "anchor_pos", overspeed_share=0.0, ref_travel_m=travel,
        **{name: (value if success else None) for name, value in errors.items()},
        progress_dot_m2=dot if success else None, ref_displacement_sq_m2=travel**2 if success else None,
        fall_s=fall_s, leg_action_rate=rate, action_rate_steps=rate_steps if rate is not None else 0,
    )


def test_progress_is_taken_on_the_travelling_pieces():
    line = benchmark.PROGRESS_TRAVEL_M
    assert line == benchmark.BODY_POS_STD_M / (1.0 - benchmark.MIN_PROGRESS)
    kinds = benchmark.summarize([
        _result("walk", 0.4, 0.4 * 0.4),  # covers its ground
        _result("walk", 0.6 * line, -0.05),  # too short to judge: left out
        _result("walk", 0.5, 0.0, success=False),  # failed: counts as travelling time, not in progress
        _result("cmujump", 0.5 * line, 0.0),
    ])
    assert kinds["walk"]["progress"] == 1.0
    assert kinds["walk"]["travel_seconds"] == 20.0
    assert kinds["cmujump"]["progress"] is None and kinds["cmujump"]["travel_seconds"] == 0.0
    # A robot that ends a travelling piece within the trunk position std of the reference reaches the line.
    ref = np.array([line, 0.0])
    worst = ref + np.array([-benchmark.BODY_POS_STD_M, 0.0])
    assert float(worst @ ref) / float(ref @ ref) == pytest.approx(benchmark.MIN_PROGRESS)


def test_an_evaluation_only_set_is_judged_on_its_own_kinds():
    # The PICO set: no training kinds, no walk/run/sprint; the same lines.
    pico = {"picoteleopit": _kind(0.95), "picotwist": _kind(0.91), "picoany": _kind(0.9, None)}
    assert not benchmark.verdicts(pico)["small_run"]["pass"]  # the default still asks for locomotion
    judged = benchmark.verdicts(pico, require_locomotion=False)
    assert judged["final"]["pass"] and judged["small_run"]["pass"]
    assert judged["final"]["kinds_without_pieces"] == []
    assert sorted(judged["final"]["progress"]) == ["picoteleopit", "picotwist"]
    failing = benchmark.verdicts({**pico, "picoany": _kind(0.0, None)}, require_locomotion=False)
    assert failing["final"]["kinds_below_success_rate"] == ["picoany"]
    assert failing["small_run"]["kinds_never_succeeding"] == ["picoany"]
    assert failing["final"]["pass"] and not failing["small_run"]["pass"]
    # Its recordings walk without walk in their kind names: a robot standing still is caught.
    still = benchmark.verdicts({**pico, "picoteleopit": _kind(1.0, progress=0.05)}, require_locomotion=False)
    assert not still["final"]["progress_ok"] and not still["small_run"]["pass"]


def test_a_kind_is_judged_on_its_judged_seconds():
    # From HOME: success over the seconds past the settle, falls per minute
    # standing (a piece stands until its fall), the action rate over its
    # steps, travelling time on the judged seconds.
    kinds = benchmark.summarize([
        _result("walk", 0.4, 0.16, seconds=10.0, settle_s=1.0, rate=1.0, rate_steps=400),
        _result("walk", 0.4, 0.16, success=False, seconds=4.0, settle_s=1.0, fall_s=3.0, rate=3.0, rate_steps=100),
        _result("walk", 0.0, 0.0, success=False, seconds=2.0, settle_s=1.0, fall_s=0.5, rate=None),
    ])
    walk = kinds["walk"]
    assert walk["judged_seconds"] == pytest.approx(13.0) and walk["seconds"] == pytest.approx(16.0)
    assert walk["success_rate"] == pytest.approx(9.0 / 13.0)
    assert walk["falls"] == 2 and walk["standing_seconds"] == pytest.approx(10.0 + 3.0 + 0.5)
    assert walk["falls_per_minute"] == pytest.approx(60.0 * 2 / 13.5)
    assert walk["leg_action_rate"] == pytest.approx((1.0 * 400 + 3.0 * 100) / 500)
    assert walk["travel_seconds"] == pytest.approx(12.0)


def test_the_action_rate_line_holds_on_every_kind():
    # The PICO policy on the robot walks at 3.355 (rad^2 per step over the 12 leg joints).
    assert benchmark.MAX_LEG_ACTION_RATE == 3.36
    assert len(benchmark.LEG_JOINT_NAMES) == 12
    assert not any("shoulder" in name or "elbow" in name for name in benchmark.LEG_JOINT_NAMES)
    kinds = {"walk": _kind(1.0), "run": _kind(1.0), "sprint": _kind(0.95)}
    assert benchmark.verdicts(kinds)["final"]["pass"]
    for rate in (3.5, None):  # rougher than the PICO policy, or not measured
        rough = benchmark.verdicts({**kinds, "sprint": _kind(0.95, rate=rate)})
        assert rough["final"]["kinds_above_action_rate_line"] == ["sprint"]
        assert not rough["final"]["pass"]
        assert rough["small_run"]["pass"]  # not a small-run check
    assert benchmark.verdicts({**kinds, "sprint": _kind(0.95, rate=3.36)})["final"]["pass"]


def test_the_fall_line_holds_on_the_operators_motions():
    # The PICO policy fell 6 times in its 144 pushed rollouts of 6 s, over 846.58 s standing.
    assert benchmark.MAX_FALLS_PER_MINUTE == pytest.approx(0.425, abs=5e-4)
    pico = {"picoteleopit": _kind(0.95, falls=1, standing_seconds=400.0),
            "picotwist": _kind(0.95, falls=0, standing_seconds=300.0),
            "picoany": _kind(0.9, None, falls=2, standing_seconds=300.0)}
    judged = benchmark.verdicts(pico, require_locomotion=False, hold_fall_line=True)["final"]
    assert judged["falls_per_minute"] == pytest.approx(0.18) and judged["fall_ok"] and judged["pass"]
    # Over the whole set: 8 falls in 1000 s fail it, even with every kind's success above 90 %.
    falling = {**pico, "picotwist": _kind(0.95, falls=5, standing_seconds=300.0)}
    judged = benchmark.verdicts(falling, require_locomotion=False, hold_fall_line=True)["final"]
    assert judged["falls_per_minute"] == pytest.approx(0.48) and not judged["fall_ok"] and not judged["pass"]
    # The training kinds' set reports the rate without the line.
    reported = benchmark.verdicts(falling, require_locomotion=False)["final"]
    assert reported["falls_per_minute"] == pytest.approx(0.48) and reported["fall_ok"] and reported["pass"]


def test_the_full_run_passes_on_the_fall_and_action_rate_lines_alone():
    # Decided 2026-10-15: not falling however the operator moves; the tracking
    # errors are accepted.  Half the seconds failing on the end points' height,
    # every error above its line and no progress still pass ...
    off = {name: 2.0 * std for name, std in benchmark.ERROR_LINES.items()}
    loose = {kind: {**_kind(0.49, progress=0.1, falls=1, standing_seconds=600.0), **off}
             for kind in ("picoteleopit", "picotwist")}
    judged = benchmark.verdicts(loose, require_locomotion=False, hold_fall_line=True)["final"]
    assert judged["kinds_below_success_rate"] == judged["kinds_above_error_lines"] == ["picoteleopit", "picotwist"]
    assert not judged["progress_ok"] and judged["falls_per_minute"] == pytest.approx(0.1)
    assert judged["pass"]
    # ... and so does the training kinds' set, falling or not; a fall rate or an action rate past its line fails.
    assert benchmark.verdicts({**loose, "picotwist": {**loose["picotwist"], "falls": 9}}, require_locomotion=False)[
        "final"]["pass"]
    falling = {**loose, "picotwist": {**loose["picotwist"], "falls": 8}}  # 9 in 20 min: 0.45
    assert not benchmark.verdicts(falling, require_locomotion=False, hold_fall_line=True)["final"]["pass"]
    rough = {**loose, "picotwist": {**loose["picotwist"], "leg_action_rate": 3.37}}
    assert not benchmark.verdicts(rough, require_locomotion=False)["final"]["pass"]


def test_a_fall_is_a_tipped_trunk_under_an_upright_reference():
    def tilted(degrees: float) -> torch.Tensor:
        half = math.radians(degrees) / 2.0
        return torch.tensor([[math.cos(half), math.sin(half), 0.0, 0.0]])

    assert float(benchmark.tilt_rad(tilted(50.0))[0]) == pytest.approx(math.radians(50.0), abs=1e-6)
    assert bool(benchmark.fell(tilted(50.0), tilted(0.0))[0])
    assert not bool(benchmark.fell(tilted(40.0), tilted(0.0))[0])
    # A reference lying down (the robot follows it to the floor) is no fall.
    assert not bool(benchmark.fell(tilted(90.0), tilted(40.0))[0])


def test_a_robot_down_past_a_leaning_reference_is_down():
    def tilted(degrees: float) -> torch.Tensor:
        half = math.radians(degrees) / 2.0
        return torch.tensor([[math.cos(half), math.sin(half), 0.0, 0.0]])

    def down(robot_deg: float, ref_deg: float, robot_z: float = 0.15, ref_z: float = 0.15) -> bool:
        return bool(benchmark.down_past_reference(tilted(robot_deg), tilted(ref_deg), torch.tensor([robot_z]),
                                                  torch.tensor([ref_z]))[0])

    # Lying under a reference bent 40 deg over on its feet: 50 deg past it.
    assert down(90.0, 40.0)
    # Following a reference leaning 70 deg, or lying under one on all fours (tilts taken up to 90 deg).
    assert not down(75.0, 70.0) and not down(110.0, 75.0, 0.09, 0.11)
    # 7 cm below the reference's trunk, though only 10 deg past its tilt; not past 45 deg, no fall.
    assert down(50.0, 40.0, 0.08, 0.15) and not down(40.0, 40.0, 0.05, 0.15)


def test_the_kinds_table_carries_the_new_lines(tmp_path):
    results = [_result("picotwist", 0.4, 0.16, settle_s=1.0), _result("picotwist", 0.0, 0.0, settle_s=1.0, fall_s=4.0)]
    report = benchmark.write_outputs(tmp_path, "model", {}, results, (), require_locomotion=False, hold_fall_line=True)
    with (tmp_path / "model-kinds.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert list(rows[0]) == list(benchmark.KIND_COLUMNS)
    # One fall over 10 s and 4 s standing.
    assert float(rows[0]["falls_per_minute"]) == pytest.approx(60.0 / 14.0)
    assert float(rows[0]["leg_action_rate"]) == 0.5
    assert float(rows[0]["judged_seconds"]) == 18.0
    assert report["verdicts"]["final"]["falls_per_minute"] == pytest.approx(60.0 / 14.0)
    assert not report["verdicts"]["final"]["pass"]


def test_a_rollout_starts_at_rest_at_home_holds_the_first_frame_and_judges_after_the_settle(tmp_path):
    # On the simulator (CPU), two references at 50 fps whose trunk, tilted
    # 20 deg and heading 60 deg, stands at HOME's height on the floor (the end
    # points' termination is left out: the arms stay at 0), the right elbow
    # at 0.1 rad per second of the reference (its time) and the trunk turning
    # at 0.5 rad/s about the vertical.  The first moves for
    # its first 0.2 s: its piece starts there, 55 judged steps; the second's,
    # 85, its trunk bent 50 deg over from its 0.1 s on.  The policy alternates every action by +-0.05 rad, lifts
    # the first robot 10 cm in the settle (steps 10-14) and puts it back,
    # knocks the second over at step 60 and the first at step 110 (past its piece).
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.lab_api.math import quat_mul, yaw_quat

    from mjlab_microban.policy_contract import ACTION_JOINT_NAMES, joint_ranges
    from mjlab_microban.scripts.wholebody_benchmark import benchmark_env_cfg
    from mjlab_microban.wholebody.env_cfg import TRACKED_BODY_NAMES

    def axis_quat(angle: float, axis: int) -> torch.Tensor:
        quat = torch.zeros(4)
        quat[0], quat[1 + axis] = math.cos(angle / 2.0), math.sin(angle / 2.0)
        return quat

    joints, bodies = tuple(joint_ranges()), len(TRACKED_BODY_NAMES)
    trunk_quat = quat_mul(axis_quat(math.radians(60.0), 2), axis_quat(math.radians(20.0), 0))
    lean_quat = quat_mul(axis_quat(math.radians(60.0), 2), axis_quat(math.radians(50.0), 0))
    foot_quat, foot_heights = _standing_feet()

    def clip(name: str, frames: int, moving_frames: int, lean_from: int | None = None) -> dict:
        section = {"name": name, "fps": 50, "joint_pos": np.zeros((frames, len(joints))),
                   "joint_vel": np.zeros((frames, len(joints))), "body_pos_w": np.zeros((frames, bodies, 3)),
                   "body_quat_w": np.tile([1.0, 0.0, 0.0, 0.0], (frames, bodies, 1)),
                   "body_lin_vel_w": np.zeros((frames, bodies, 3)), "body_ang_vel_w": np.zeros((frames, bodies, 3))}
        section["body_pos_w"][:, 0] = (0.3, -0.2, HOME.root_pos[2])
        section["body_quat_w"][:, 0] = trunk_quat.numpy()
        if lean_from is not None:
            section["body_quat_w"][lean_from:, 0] = lean_quat.numpy()
        section["body_lin_vel_w"][:moving_frames, 0, 0] = 0.2
        section["joint_pos"][:, joints.index("right_elbow")] = 0.1 * np.arange(frames) / 50.0
        section["joint_vel"][:, joints.index("right_elbow")] = 0.1
        section["body_ang_vel_w"][:, 0, 2] = 0.5
        for body, height in foot_heights.items():
            section["body_pos_w"][:, TRACKED_BODY_NAMES.index(body), 2] = height
            section["body_quat_w"][:, TRACKED_BODY_NAMES.index(body)] = foot_quat
        return section

    path = tmp_path / "motion.h5"
    write_motion_shard([clip("walk1_subject5", 66, 10), clip("run1_subject5", 86, 0, lean_from=5)], path,
                       body_names=TRACKED_BODY_NAMES, joint_names=joints, max_window_frames=None)
    cfg = benchmark_env_cfg(str(path), 10.0, 2, 0)
    del cfg.terminations["ee_body_pos"]
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        pieces = benchmark.load_pieces(path, ACTION_JOINT_NAMES, 10.0, env.step_dt, no_load_speed=10.0)
        assert [(p.start_s, p.steps) for p in pieces] == [(0.2, 50 + 55), (0.0, 50 + 85)]
        command = env.command_manager.get_term("motion")
        robot = command.robot
        start, times, speeds = {}, [], []
        elbow = robot.joint_names.index("right_elbow")

        def policy(obs):
            step = start.setdefault("calls", 0)
            start["calls"] += 1
            times.append(command.all_joint_pos[:, elbow] / 0.1)  # the reference's time the policy sees
            speeds.append(torch.stack([command.joint_vel[:, ACTION_JOINT_NAMES.index("right_elbow")],
                                       command.anchor_ang_vel_w[:, 2], command.body_ang_vel_w[:, 0, 2]], -1))
            if step == 0:  # placed, before the first step
                start.update(trunk=robot.data.root_link_pose_w.clone(), vel=robot.data.root_link_vel_w.clone(),
                             joint_pos=robot.data.joint_pos.clone(), joint_vel=robot.data.joint_vel.clone(),
                             ref_xy=command.anchor_pos_w[:, :2].clone())
            if step in (10, 15):  # lifted, then back where it was placed
                state = torch.cat([start["trunk"][:1], torch.zeros(1, 6)], dim=-1)
                state[0, 2] += 0.1 if step == 10 else 0.0
                robot.write_root_state_to_sim(state, env_ids=torch.tensor([0]))
                robot.write_joint_state_to_sim(start["joint_pos"][:1], start["joint_vel"][:1], env_ids=torch.tensor([0]))
            if step in (60, 110):
                index = 1 if step == 60 else 0
                lying = torch.zeros(1, 13)
                lying[0, :3] = robot.data.root_link_pose_w[index, :3] * torch.tensor([1.0, 1.0, 0.0])
                lying[0, 2] = 0.08
                lying[0, 3:7] = axis_quat(math.pi / 2.0, 0)
                robot.write_root_state_to_sim(lying, env_ids=torch.tensor([index]))
            return torch.full((2, env.action_manager.total_action_dim), 0.05 * (-1.0) ** step)

        results, _ = benchmark.rollout(env, policy, pieces)
        # At rest at HOME, under the reference trunk and facing its heading (not its tilt).
        home = robot.data.default_root_state
        np.testing.assert_allclose(start["trunk"][:, :2], start["ref_xy"], atol=1e-5)
        np.testing.assert_allclose(start["trunk"][:, 2], home[:, 2] + env.scene.env_origins[:, 2], atol=1e-5)
        expected_quat = quat_mul(yaw_quat(trunk_quat.expand(2, 4)), home[:, 3:7])
        assert torch.allclose(start["trunk"][:, 3:7], expected_quat, atol=1e-5)
        assert torch.allclose(start["joint_pos"], robot.data.default_joint_pos, atol=1e-6)
        assert not start["joint_vel"].any() and not start["vel"].any()
        # The reference holds the piece's first frame (one step ahead) through
        # the settle, standing still (its velocities 0), then plays on from it.
        dt, first = env.step_dt, torch.tensor([0.2, 0.0])
        for call in (0, 25, 50):
            assert torch.allclose(times[call], first + dt, atol=1e-5)
            assert not speeds[call].any()
        for call in (51, 90):
            assert torch.allclose(times[call], first + (call - 49) * dt, atol=1e-5)
            assert torch.allclose(speeds[call], torch.tensor([0.1, 0.5, 0.5]).expand(2, 3), atol=1e-6)
        # The termination of the settle fails no piece; the fall does (on
        # step 60, the 11th judged step, under the bent-over reference: counted
        # 0.5 s later from step 60 on), and the piece runs to its end.
        assert [(r.steps, r.success, r.failure_reason) for r in results] == [(55, True, None), (11, False, "anchor_pos")]
        # The fall and the leg targets' change (12 leg joints x 0.1 rad per
        # step) are taken from 1 s into each piece until its end or its fall.
        assert results[0].fall_s is None and results[1].fall_s == pytest.approx(61 * dt)
        assert [r.action_rate_steps for r in results] == [105 - 51, 60 - 51]
        assert [r.leg_action_rate for r in results] == pytest.approx([12 * 0.1**2] * 2, rel=1e-4)
    finally:
        env.close()
