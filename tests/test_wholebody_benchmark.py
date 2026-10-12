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
    standing_seconds: float = 600.0,
    rate: float | None = 0.5,
) -> dict:
    """A kind's summary; it travels (the progress line holds) when it has a progress."""
    travel = (60.0 if progress is not None else 0.0) if travel_seconds is None else travel_seconds
    return {"success_rate": success_rate, "non_finite_pieces": 0, "progress": progress, "travel_seconds": travel,
            "pieces": 10, "falls": falls, "standing_seconds": standing_seconds,
            "falls_per_minute": 60.0 * falls / standing_seconds, "leg_action_rate": rate,
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
    # One fall over 10 s and 4 s standing.
    assert float(rows[0]["falls_per_minute"]) == pytest.approx(60.0 / 14.0)
    assert float(rows[0]["leg_action_rate"]) == 0.5
    assert float(rows[0]["judged_seconds"]) == 18.0
    assert report["verdicts"]["final"]["falls_per_minute"] == pytest.approx(60.0 / 14.0)
    assert not report["verdicts"]["final"]["pass"]


def test_a_rollout_starts_at_rest_at_home_and_judges_after_the_settle(tmp_path):
    # On the simulator (CPU), two pieces of 65 and 85 steps whose reference
    # trunk, tilted 20 deg and heading 60 deg, lies on the floor for 0.88 s
    # (the trunk height termination fires) and then stands at HOME's height
    # (the end points' termination is left out: the other bodies stay at 0).
    # The policy alternates every action by +-0.05 rad and knocks the second
    # robot over at step 60, the first at step 70 (past its piece).
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.lab_api.math import quat_mul, yaw_quat

    from mjlab_microban.policy_contract import ACTION_JOINT_NAMES, joint_ranges
    from mjlab_microban.robot.home_pose import HOME
    from mjlab_microban.scripts.wholebody_benchmark import benchmark_env_cfg
    from mjlab_microban.wholebody.env_cfg import TRACKED_BODY_NAMES

    def axis_quat(angle: float, axis: int) -> torch.Tensor:
        quat = torch.zeros(4)
        quat[0], quat[1 + axis] = math.cos(angle / 2.0), math.sin(angle / 2.0)
        return quat

    joints, bodies = tuple(joint_ranges()), len(TRACKED_BODY_NAMES)
    trunk_quat = quat_mul(axis_quat(math.radians(60.0), 2), axis_quat(math.radians(20.0), 0))

    def clip(name: str, frames: int) -> dict:
        section = {"name": name, "fps": 50, "joint_pos": np.zeros((frames, len(joints))),
                   "joint_vel": np.zeros((frames, len(joints))), "body_pos_w": np.zeros((frames, bodies, 3)),
                   "body_quat_w": np.tile([1.0, 0.0, 0.0, 0.0], (frames, bodies, 1)),
                   "body_lin_vel_w": np.zeros((frames, bodies, 3)), "body_ang_vel_w": np.zeros((frames, bodies, 3))}
        section["body_pos_w"][:, 0] = (0.3, -0.2, HOME.root_pos[2])
        section["body_pos_w"][:44, 0, 2] = 0.0
        section["body_quat_w"][:, 0] = trunk_quat.numpy()
        return section

    path = tmp_path / "motion.h5"
    write_motion_shard([clip("walk1_subject5", 66), clip("run1_subject5", 86)], path, body_names=TRACKED_BODY_NAMES,
                       joint_names=joints, max_window_frames=None)
    cfg = benchmark_env_cfg(str(path), 10.0, 2, 0)
    del cfg.terminations["ee_body_pos"]
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        pieces = benchmark.load_pieces(path, ACTION_JOINT_NAMES, 10.0, env.step_dt, no_load_speed=10.0)
        assert [p.steps for p in pieces] == [65, 85]
        command = env.command_manager.get_term("motion")
        robot = command.robot
        start = {}

        def policy(obs):
            step = start.setdefault("calls", 0)
            start["calls"] += 1
            if step == 0:  # placed, before the first step
                start.update(trunk=robot.data.root_link_pose_w.clone(), vel=robot.data.root_link_vel_w.clone(),
                             joint_pos=robot.data.joint_pos.clone(), joint_vel=robot.data.joint_vel.clone(),
                             ref_xy=command.anchor_pos_w[:, :2].clone())
            if step in (60, 70):
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
        # The termination of the first second fails no piece; the fall does
        # (on step 60, the 11th judged step), and the piece runs to its end.
        assert [(r.steps, r.success, r.failure_reason) for r in results] == [(15, True, None), (11, False, "anchor_pos")]
        # The fall and the leg targets' change (12 leg joints x 0.1 rad per
        # step) are taken from 1 s into each piece until its end or its fall.
        assert results[0].fall_s is None and results[1].fall_s == pytest.approx(61 * env.step_dt)
        assert [r.action_rate_steps for r in results] == [65 - 51, 60 - 51]
        assert [r.leg_action_rate for r in results] == pytest.approx([12 * 0.1**2] * 2, rel=1e-4)
    finally:
        env.close()
