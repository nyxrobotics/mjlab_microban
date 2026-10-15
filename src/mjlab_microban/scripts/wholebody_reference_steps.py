"""The training reference's change per control tick: the live retargeting's step limits (live/retarget.py).

The reference the policy observes, as the training env feeds it (wholebody/motion.py MotionLib.get_frames): each
30 fps source clip sampled at the control step (0.02 s), positions and velocities linear, orientations slerp.  Over all
source clips of the motion shards, per joint: the speed (p99), its change per tick (p99 and p99.9) and the step per
tick (p99.9); the trunk: the same of its position (speed and step as vector norms) and of its turn (angular speed, the
rotation angle per tick).  Printed as live/retarget.py's JOINT_STEP_LIMITS, TRUNK_STEP_LIMITS and TURN_STEP_LIMITS.

    MICROBAN_WITHOUT_TRAINING=1 uv run python src/mjlab_microban/scripts/wholebody_reference_steps.py <motion_file>

``<motion_file>``: the run's params/env.yaml ``motion_file`` (a directory of shards or one shard).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from mjlab_microban.live.retarget import slerp
from mjlab_microban.wholebody.data.hdf5 import find_motion_shards, read_strings

STEP_S = 0.02


def _turn(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    return 2.0 * np.arccos(np.clip(np.abs(np.sum(q0 * q1, axis=-1)), 0.0, 1.0))


def sample(path: Path) -> tuple[list[str], dict[str, list[np.ndarray]]]:
    """Per-tick series of one shard, pooled over its source clips."""
    out: dict[str, list[np.ndarray]] = {k: [] for k in ("jp", "jv", "tp", "tq", "lv", "av")}
    with h5py.File(path, "r") as f:
        names = read_strings(f, "joint_names")
        arrays = {"jp": np.asarray(f["joint_pos"], np.float64), "jv": np.asarray(f["joint_vel"], np.float64),
                  "tp": np.asarray(f["body_pos_w"][:, 0, :], np.float64),
                  "tq": np.asarray(f["body_quat_w"][:, 0, :], np.float64),
                  "lv": np.asarray(f["body_lin_vel_w"][:, 0, :], np.float64),
                  "av": np.asarray(f["body_ang_vel_w"][:, 0, :], np.float64)}
        clips = zip(f["source_clip_starts"][()], f["source_clip_lengths"][()], f["source_clip_fps"][()])
        for start, length, fps in clips:
            if length < 3:
                continue
            frame = np.arange(0.0, (length - 1) / fps, STEP_S) * fps
            i0 = np.minimum(frame.astype(int), length - 1)
            i1 = np.minimum(i0 + 1, length - 1)
            a = (frame - i0)[:, None]
            for key, value in arrays.items():
                clip = value[start:start + length]
                out[key].append(slerp(clip[i0], clip[i1], a) if key == "tq" else clip[i0] + a * (clip[i1] - clip[i0]))
    return names, out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("motion_file", type=Path)
    ap.add_argument("--out", type=Path, help="also write the percentiles as JSON")
    args = ap.parse_args()
    steps: dict[str, list[np.ndarray]] = {k: [] for k in ("jv", "jv_step", "jp_step", "lv", "lv_step", "tp_step",
                                                         "av", "av_step", "turn")}
    names = None
    for shard in find_motion_shards(str(args.motion_file)):
        names, series = sample(Path(shard))
        for jp, jv, tp, tq, lv, av in zip(*(series[k] for k in ("jp", "jv", "tp", "tq", "lv", "av"))):
            steps["jv"].append(np.abs(jv))
            steps["jv_step"].append(np.abs(np.diff(jv, axis=0)))
            steps["jp_step"].append(np.abs(np.diff(jp, axis=0)))
            steps["lv"].append(np.linalg.norm(lv, axis=1))
            steps["lv_step"].append(np.linalg.norm(np.diff(lv, axis=0), axis=1))
            steps["tp_step"].append(np.linalg.norm(np.diff(tp, axis=0), axis=1))
            steps["av"].append(np.linalg.norm(av, axis=1))
            steps["av_step"].append(np.linalg.norm(np.diff(av, axis=0), axis=1))
            steps["turn"].append(_turn(tq[1:], tq[:-1]))
    pooled = {k: np.concatenate(v) for k, v in steps.items()}
    p = lambda key, q: np.percentile(pooled[key], q, axis=0)
    joints = {n: (p("jv", 99)[j], p("jv_step", 99)[j], p("jp_step", 99.9)[j], p("jv_step", 99.9)[j])
              for j, n in enumerate(names)}
    trunk = tuple(float(p(k, q)) for k, q in (("lv", 99), ("lv_step", 99), ("tp_step", 99.9), ("lv_step", 99.9)))
    turn = tuple(float(p(k, q)) for k, q in (("av", 99), ("av_step", 99), ("turn", 99.9), ("av_step", 99.9)))
    print(f"# {len(pooled['jv'])} ticks ({len(pooled['jv']) * STEP_S / 3600:.1f} h)")
    print("JOINT_STEP_LIMITS = {")
    for n, (a, b, c, d) in joints.items():
        print(f'    "{n}": ({a:.2f}, {b:.2f}, {c:.3f}, {d:.2f}),')
    print("}")
    print(f"TRUNK_STEP_LIMITS = ({trunk[0]:.3f}, {trunk[1]:.4f}, {trunk[2]:.4f}, {trunk[3]:.3f})")
    print(f"TURN_STEP_LIMITS = ({turn[0]:.2f}, {turn[1]:.2f}, {turn[2]:.3f}, {turn[3]:.2f})")
    if args.out:
        args.out.write_text(json.dumps({"joints": {n: list(map(float, v)) for n, v in joints.items()},
                                        "trunk": trunk, "turn": turn}, indent=1))


if __name__ == "__main__":
    main()
