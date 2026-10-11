"""The whole-body evaluation: every evaluation frame is run once, and the verdicts judge every kind."""

from __future__ import annotations

import numpy as np

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
    pieces = benchmark.load_pieces(path, JOINTS, 10.0, 0.02, no_load_speed=10.0)
    assert [(p.window_id, p.kind, p.start_s, p.steps) for p in pieces] == [
        (0, "walk", 0.0, 500), (0, "walk", 10.0, 500), (0, "walk", 20.0, 250), (1, "run", 0.0, 148),
    ]


def _kind(success_rate: float, progress: float | None = 1.0) -> dict:
    return {"success_rate": success_rate, "non_finite_pieces": 0, "progress": progress,
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


def test_an_evaluation_only_set_is_judged_on_its_own_kinds():
    # The PICO set: no training kinds, no walk/run/sprint; the same lines.
    pico = {"picoteleopit": _kind(0.95, None), "picotwist": _kind(0.91, None), "picoany": _kind(0.9, None)}
    assert not benchmark.verdicts(pico)["small_run"]["pass"]  # the default still asks for locomotion
    judged = benchmark.verdicts(pico, require_locomotion=False)
    assert judged["final"]["pass"] and judged["small_run"]["pass"]
    assert judged["final"]["kinds_without_pieces"] == []
    failing = benchmark.verdicts({**pico, "picoany": _kind(0.0, None)}, require_locomotion=False)
    assert failing["final"]["kinds_below_success_rate"] == ["picoany"]
    assert failing["small_run"]["kinds_never_succeeding"] == ["picoany"]
    assert not failing["final"]["pass"] and not failing["small_run"]["pass"]
    # A locomotion kind it does have is still held to the progress line.
    still = benchmark.verdicts({**pico, "walk": _kind(1.0, progress=0.05)}, require_locomotion=False)
    assert not still["final"]["pass"]
