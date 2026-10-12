"""The whole-body evaluation: every evaluation frame is run once, and the verdicts judge every kind."""

from __future__ import annotations

import csv
import math

import numpy as np
import pytest
import torch

from mjlab_microban.wholebody import benchmark
from mjlab_microban.wholebody.data.hdf5 import FRAME_KEYS, write_motion_shard

JOINTS = ("a", "b")


def _section(name: str, frames: int, fps: int = 30) -> dict:
    clip = {key: np.zeros((frames, 2) if key.startswith("joint") else (frames, 1, 4 if "quat" in key else 3))
            for key in FRAME_KEYS}
    clip["body_quat_w"][..., 0] = 1.0
    return {"name": name, "fps": fps, **clip}


def test_pieces_cover_each_section_once(tmp_path):
    # 25 s and 2.97 s sections at 30 fps: pieces of 10, 10 and 5 s, and 2.96 s.
    path = tmp_path / "motion.h5"
    write_motion_shard([_section("walk1_subject5__f0-751", 751), _section("run1_subject5__f0-90", 90)], path,
                       body_names=("trunk",), joint_names=JOINTS, max_window_frames=None)
    pieces = benchmark.load_pieces(path, JOINTS, 10.0, 0.02, no_load_speed=10.0, start="reference")
    assert [(p.window_id, p.kind, p.start_s, p.steps) for p in pieces] == [
        (0, "walk", 0.0, 500), (0, "walk", 10.0, 500), (0, "walk", 20.0, 250), (1, "run", 0.0, 148),
    ]
    assert all(p.settle_steps == 0 and p.judged_seconds == p.seconds for p in pieces)


def test_pieces_measure_the_reference_trunks_travel(tmp_path):
    # The trunk walks along x at 0.03 m/s for 10 s, then stands: 0.3 m, then 0.
    section = _section("dance1_subject5__f0-601", 601)
    t = np.arange(601) / 30.0
    section["body_pos_w"][:, 0, 0] = 0.03 * np.minimum(t, 10.0)
    path = tmp_path / "motion.h5"
    write_motion_shard([section], path, body_names=("trunk",), joint_names=JOINTS, max_window_frames=None)
    pieces = benchmark.load_pieces(path, JOINTS, 10.0, 0.02, no_load_speed=10.0, start="reference")
    assert [round(p.ref_travel_m, 6) for p in pieces] == [0.3, 0.0]


def test_a_home_start_judges_each_piece_after_its_first_second(tmp_path):
    # From HOME the first second of a piece is the way to the reference pose:
    # 25 s at 0.03 m/s for 10 s, then standing; a 1.0 s section (nothing past
    # the settle, left out) and a 1.07 s one (0.06 s judged).
    section = _section("dance1_subject5__f0-751", 751)
    t = np.arange(751) / 30.0
    section["body_pos_w"][:, 0, 0] = 0.03 * np.minimum(t, 10.0)
    path = tmp_path / "motion.h5"
    write_motion_shard([section, _section("run1_subject5__f0-31", 31), _section("run2_subject5__f0-33", 33)], path,
                       body_names=("trunk",), joint_names=JOINTS, max_window_frames=None)
    pieces = benchmark.load_pieces(path, JOINTS, 10.0, 0.02, no_load_speed=10.0)
    assert benchmark.settle_steps("home", 0.02) == 50 and benchmark.settle_steps("reference", 0.02) == 0
    assert [(p.window_id, p.start_s, p.steps, p.settle_steps) for p in pieces] == [
        (0, 0.0, 500, 50), (0, 10.0, 500, 50), (0, 20.0, 250, 50), (2, 0.0, 53, 50),
    ]
    assert [round(p.judged_seconds, 6) for p in pieces] == [9.0, 9.0, 4.0, 0.06]
    # The reference's travel over the judged steps only: 0.3 m less the first second's 0.03 m.
    assert [round(p.ref_travel_m, 6) for p in pieces[:2]] == [0.27, 0.0]
    with pytest.raises(ValueError):
        benchmark.settle_steps("crouched", 0.02)


def _kind(
    success_rate: float,
    progress: float | None = 1.0,
    travel_seconds: float | None = None,
    falls: int = 0,
    pieces: int = 10,
    rate: float | None = 0.5,
) -> dict:
    """A kind's summary; it travels (the progress line holds) when it has a progress."""
    travel = (60.0 if progress is not None else 0.0) if travel_seconds is None else travel_seconds
    return {"success_rate": success_rate, "non_finite_pieces": 0, "progress": progress, "travel_seconds": travel,
            "pieces": pieces, "falls": falls, "fall_share": falls / pieces, "leg_action_rate": rate,
            **{name: 0.5 * std for name, std in benchmark.ERROR_LINES.items()}}


def test_verdicts_fail_missing_kinds_and_a_robot_standing_still():
    kinds = {"walk": _kind(1.0), "run": _kind(1.0), "sprint": _kind(0.95), "jumps": _kind(0.92, None)}
    assert benchmark.verdicts(kinds, {"walk", "jumps"})["final"]["pass"]
    # A training kind with no evaluation piece fails both lines.
    missing = benchmark.verdicts(kinds, {"walk", "jumps", "fallAndGetUp"})
    assert missing["final"]["kinds_without_pieces"] == ["fallAndGetUp"]
    assert not missing["final"]["pass"] and not missing["small_run"]["pass"]
    # So does a missing locomotion kind, and a walk that stands still.
    assert not benchmark.verdicts({k: v for k, v in kinds.items() if k != "sprint"})["small_run"]["pass"]
    still = benchmark.verdicts({**kinds, "walk": _kind(1.0, progress=0.05)})
    assert not still["final"]["pass"] and not still["small_run"]["pass"]


def test_progress_holds_on_every_kind_whose_reference_travels():
    kinds = {"walk": _kind(1.0), "run": _kind(1.0), "sprint": _kind(0.95)}
    # Whatever its name (CMU's walking, a fight that moves around), a travelling kind is held to progress ...
    for name in ("cmuwalk", "fight"):
        still = benchmark.verdicts({**kinds, name: _kind(1.0, progress=0.1)})
        assert still["final"]["progress"][name] == 0.1
        assert not still["final"]["pass"] and not still["small_run"]["pass"]
    # ... and so is one whose travelling pieces all failed (no progress measured).
    assert not benchmark.verdicts({**kinds, "fight": _kind(0.5, progress=None, travel_seconds=20.0)})["small_run"]["pass"]
    # A kind that never travels has no progress line.
    judged = benchmark.verdicts({**kinds, "cmujump": _kind(1.0, progress=None)})
    assert judged["final"]["pass"] and "cmujump" not in judged["final"]["progress"]


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
    assert not failing["final"]["pass"] and not failing["small_run"]["pass"]
    # Its recordings walk without walk in their kind names: a robot standing still fails.
    still = benchmark.verdicts({**pico, "picoteleopit": _kind(1.0, progress=0.05)}, require_locomotion=False)
    assert not still["final"]["pass"] and not still["small_run"]["pass"]


def test_a_kind_is_judged_on_its_judged_seconds():
    # From HOME: success over the seconds past the settle, falls over the pieces,
    # the action rate over its steps, travelling time on the judged seconds.
    kinds = benchmark.summarize([
        _result("walk", 0.4, 0.16, seconds=10.0, settle_s=1.0, rate=1.0, rate_steps=400),
        _result("walk", 0.4, 0.16, success=False, seconds=4.0, settle_s=1.0, fall_s=3.0, rate=3.0, rate_steps=100),
        _result("walk", 0.0, 0.0, success=False, seconds=2.0, settle_s=1.0, fall_s=0.5, rate=None),
    ])
    walk = kinds["walk"]
    assert walk["judged_seconds"] == pytest.approx(13.0) and walk["seconds"] == pytest.approx(16.0)
    assert walk["success_rate"] == pytest.approx(9.0 / 13.0)
    assert walk["falls"] == 2 and walk["fall_share"] == pytest.approx(2 / 3)
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
    # The PICO policy fell in 6 of its 144 pushed rollouts.
    assert benchmark.MAX_FALL_SHARE == 6 / 144
    pico = {"picoteleopit": _kind(0.95, falls=1, pieces=40), "picotwist": _kind(0.95, falls=0, pieces=30),
            "picoany": _kind(0.9, None, falls=1, pieces=30)}
    judged = benchmark.verdicts(pico, require_locomotion=False, hold_fall_line=True)["final"]
    assert judged["fall_share"] == pytest.approx(2 / 100) and judged["fall_ok"] and judged["pass"]
    # Over the whole set: 5 of 100 pieces fail it, even with every kind's success above 90 %.
    falling = {**pico, "picotwist": _kind(0.95, falls=3, pieces=30)}
    judged = benchmark.verdicts(falling, require_locomotion=False, hold_fall_line=True)["final"]
    assert judged["fall_share"] == pytest.approx(5 / 100) and not judged["fall_ok"] and not judged["pass"]
    # The training kinds' set reports the share without the line.
    reported = benchmark.verdicts(falling, require_locomotion=False)["final"]
    assert reported["fall_share"] == pytest.approx(5 / 100) and reported["fall_ok"] and reported["pass"]


def test_a_fall_is_a_tipped_trunk_under_an_upright_reference():
    def tilted(degrees: float) -> torch.Tensor:
        half = math.radians(degrees) / 2.0
        return torch.tensor([[math.cos(half), math.sin(half), 0.0, 0.0]])

    assert float(benchmark.tilt_rad(tilted(50.0))[0]) == pytest.approx(math.radians(50.0), abs=1e-6)
    assert bool(benchmark.fell(tilted(50.0), tilted(0.0))[0])
    assert not bool(benchmark.fell(tilted(40.0), tilted(0.0))[0])
    # A reference lying down (the robot follows it to the floor) is no fall.
    assert not bool(benchmark.fell(tilted(90.0), tilted(40.0))[0])


def test_the_kinds_table_carries_the_new_lines(tmp_path):
    results = [_result("picotwist", 0.4, 0.16, settle_s=1.0), _result("picotwist", 0.0, 0.0, settle_s=1.0, fall_s=4.0)]
    report = benchmark.write_outputs(tmp_path, "model", {}, results, (), require_locomotion=False, hold_fall_line=True)
    with (tmp_path / "model-kinds.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert list(rows[0]) == list(benchmark.KIND_COLUMNS)
    assert float(rows[0]["fall_share"]) == 0.5 and float(rows[0]["leg_action_rate"]) == 0.5
    assert float(rows[0]["judged_seconds"]) == 18.0
    assert report["verdicts"]["final"]["fall_share"] == 0.5 and not report["verdicts"]["final"]["pass"]
