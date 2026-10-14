"""The live retargeting (mjlab_microban/live) against the data pipeline's on the PICO evaluation recordings.

Each source clip of data/wholebody/raw/pico (G1 pseudo-bones, pico.py) is sampled at the input rate (``--rate``,
PICO's body tracking reaches the PC at 25-50 Hz), fed through the live pipeline at its arrival time (no network
jitter) and calibrated with the source's standing reference as the data pipeline picks it, so the difference is what
comes from making each frame from the frames before it (interpolation, IK at 50 Hz, floor, velocities, exclusion).
The data pipeline's frames are the evaluation shard (``<data>/pico_eval/eval/motion.h5``).  At the time of each of
its frames the live frames are interpolated by their rendered time; frames the live pipeline held are left out of the
differences and counted.  The lag: the live frames are also compared at the data's time + s, and the s with the
smallest mean joint (or joint velocity) difference is the lag of the content behind the rendered time; the rendered
time is itself ``delay`` behind the newest input.

    MICROBAN_WITHOUT_TRAINING=1 uv run python src/mjlab_microban/scripts/wholebody_live_compare.py \\
        --raw data/wholebody/raw --data data/wholebody --rate 25 --out live_compare_25hz.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("MICROBAN_WITHOUT_TRAINING", "1")

import h5py
import numpy as np

from mjlab_microban.live import stream
from mjlab_microban.live.retarget import BONES, Pipeline, slerp, world_of
from mjlab_microban.wholebody.retarget import pico

POLICY = slice(3, 21)  # the 18 policy joints (the neck is not retargeted)
SHIFTS_S = np.round(np.arange(-0.06, 0.1501, 0.005), 3)
KEYS = ("joint_pos", "joint_vel", "body_pos_w", "body_lin_vel_w", "body_ang_vel_w")


def sections_by_clip(data: Path) -> dict[str, list[tuple[int, int, int]]]:
    """gmr clip name -> [(first row in the shard, frames, first frame in the clip)]."""
    fin = json.loads((data / "pico_eval" / "finalize.json").read_text())
    pieces = {f"{row['clip']}_p{k}": (row["clip"], s) for row in fin["clips"] for k, (s, _) in enumerate(row["pieces"])}
    out: dict[str, list] = {}
    with h5py.File(data / "pico_eval" / "eval" / "motion.h5") as f:
        names = [n.decode() for n in f["clip_names"][()]]
        for name, start, length in zip(names, f["source_clip_starts"][()], f["source_clip_lengths"][()]):
            stem, frames = name.split("__f")
            a = int(frames.split("-")[0])
            clip, offset = pieces.get(stem, (stem, 0))
            out.setdefault(clip, []).append((int(start), int(length), offset + a))
    return out


def resample(P, Q, fps, rate):
    """Sensor samples at ``rate`` from the clip's frames (positions linear, orientations slerp)."""
    ts = np.arange(0.0, (len(P) - 1) / fps + 1e-9, 1.0 / rate)
    x = ts * fps
    i = np.minimum(np.floor(x).astype(int), len(P) - 2)
    a = (x - i)[:, None, None]
    return ts, (1 - a) * P[i] + a * P[i + 1], slerp(Q[i], Q[i + 1], a)


def interp_at(tau, X, t):
    """Rows of X (rendered at tau, increasing) at times t (inside tau's range)."""
    k = np.clip(np.searchsorted(tau, t), 1, len(tau) - 1)
    w = ((t - tau[k - 1]) / (tau[k] - tau[k - 1])).reshape((-1,) + (1,) * (X.ndim - 1))
    return (1 - w) * X[k - 1] + w * X[k], k


def run_clip(job):
    src, name, fps, rp, rq, jp, ref, g1_xml, rate, h5, sections = job
    g1 = pico.g1_model(g1_xml)
    frames = pico.g1_frames(g1, rp, rq, jp)
    P = np.array([[f[b][0] for b in BONES] for f in frames])
    Q = np.array([[f[b][1] for b in BONES] for f in frames])
    ts, sp, sq = resample(P, Q, fps, rate)
    pipe = Pipeline(log=lambda _: None)
    pipe.calibrate_with(ref, world_of(frames[0]))
    pipe.set_active(True)
    tau, flags, live, ms, k = [], [], {key: [] for key in KEYS}, [], 0
    t = ts[0]
    while t <= ts[-1] + 1e-9:
        while k < len(ts) and ts[k] <= t + 1e-9:
            pipe.push(ts[k], ts[k], sp[k], sq[k])
            k += 1
        t0 = time.perf_counter()
        fr = pipe.tick(t)
        ms.append(1000 * (time.perf_counter() - t0))
        tau.append(pipe.rendered)
        flags.append(fr.flags)
        for key in KEYS:
            live[key].append(getattr(fr, key).astype(np.float64))
        t += 1.0 / pipe.fps
    tau, flags = np.array(tau), np.array(flags)
    live = {key: np.array(v) for key, v in live.items()}
    held = (flags & stream.HELD) > 0
    # The data pipeline's frames of this clip.
    times, kept = [], np.zeros(len(frames), bool)
    with h5py.File(h5) as f:
        batch = {key: [] for key in KEYS}
        for start, length, first in sections:
            for key in KEYS:
                batch[key].append(f[key][start:start + length].astype(np.float64))
            times.append((first + np.arange(length)) / fps)
            kept[first:first + length] = True
    batch = {key: np.concatenate(v) for key, v in batch.items()}
    times = np.concatenate(times)
    inside = (times >= tau[0]) & (times <= tau[-1])
    times = times[inside]
    batch = {key: v[inside] for key, v in batch.items()}
    at, kk = interp_at(tau, live["joint_pos"], times)
    ok = ~(held[kk] | held[kk - 1])
    diff = {
        "joint_deg": np.degrees(np.abs(at - batch["joint_pos"]))[ok][:, POLICY].ravel(),
        "joint_vel_rad_s": np.abs(interp_at(tau, live["joint_vel"], times)[0] - batch["joint_vel"])[ok][:, POLICY].ravel(),
        "body_pos_mm": 1000 * np.linalg.norm(interp_at(tau, live["body_pos_w"], times)[0] - batch["body_pos_w"], axis=-1)[ok].ravel(),
        "trunk_pos_mm": 1000 * np.linalg.norm(interp_at(tau, live["body_pos_w"], times)[0][:, 0] - batch["body_pos_w"][:, 0], axis=-1)[ok],
        "body_lin_vel_m_s": np.linalg.norm(interp_at(tau, live["body_lin_vel_w"], times)[0] - batch["body_lin_vel_w"], axis=-1)[ok].ravel(),
        "body_ang_vel_rad_s": np.linalg.norm(interp_at(tau, live["body_ang_vel_w"], times)[0] - batch["body_ang_vel_w"], axis=-1)[ok].ravel(),
    }
    lag = {"joint": [], "joint_vel": []}
    for s in SHIFTS_S:
        tt = times + s
        m = ok & (tt >= tau[0]) & (tt <= tau[-1])
        if not m.any():
            lag["joint"].append((0.0, 0))
            lag["joint_vel"].append((0.0, 0))
            continue
        a = interp_at(tau, live["joint_pos"], tt[m])[0][:, POLICY]
        v = interp_at(tau, live["joint_vel"], tt[m])[0][:, POLICY]
        lag["joint"].append((float(np.degrees(np.abs(a - batch["joint_pos"][m][:, POLICY])).sum()), int(m.sum()) * 18))
        lag["joint_vel"].append((float(np.abs(v - batch["joint_vel"][m][:, POLICY]).sum()), int(m.sum()) * 18))
    # Exclusion agreement: live ticks against the data pipeline's kept frames.
    idx = np.clip(np.round(tau * fps).astype(int), 0, len(frames) - 1)
    agree = {"kept_held": int((kept[idx] & held).sum()), "kept_passed": int((kept[idx] & ~held).sum()),
             "cut_held": int((~kept[idx] & held).sum()), "cut_passed": int((~kept[idx] & ~held).sum())}
    return {"clip": name, "source": src, "diff": {k: v.astype(np.float32) for k, v in diff.items()}, "lag": lag,
            "agree": agree, "compared_frames": int(ok.sum()), "held_compared": int((~ok).sum()), "ms": np.array(ms, np.float32),
            "delay_s": pipe.clock.delay, "late_ticks": int(((flags & stream.LATE) > 0).sum()), "ticks": len(flags)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=Path("data/wholebody/raw"))
    ap.add_argument("--data", type=Path, default=Path("data/wholebody"))
    ap.add_argument("--rate", type=float, default=25.0, help="input rate of the body tracking (Hz)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", nargs="*", help="clip names")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    g1_xml = str(args.raw / "g1" / "g1_mocap_29dof.xml")
    g1 = pico.g1_model(g1_xml)
    sources = pico.load_sources(args.raw / "pico", pico.g1_joint_names(g1))
    refs, _ = pico.references(sources, g1)
    sections = sections_by_clip(args.data)
    h5 = str(args.data / "pico_eval" / "eval" / "motion.h5")
    jobs = [(s, n, f, rp, rq, jp, refs[s], g1_xml, args.rate, h5, sections[n]) for s, n, f, rp, rq, jp in sources
            if n in sections and (not args.only or n in args.only)]
    jobs.sort(key=lambda j: -len(j[3]))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run_clip, jobs))
    stat = lambda v: {"mean": round(float(np.mean(v)), 4), "p99": round(float(np.percentile(v, 99)), 4)}
    diff = {k: stat(np.concatenate([r["diff"][k] for r in results])) for k in results[0]["diff"]}
    lag = {}
    for k in ("joint", "joint_vel"):
        tot = np.array([[r["lag"][k][i][0] for i in range(len(SHIFTS_S))] for r in results]).sum(0)
        cnt = np.array([[r["lag"][k][i][1] for i in range(len(SHIFTS_S))] for r in results]).sum(0)
        mean = tot / np.maximum(cnt, 1)
        lag[k] = {"best_shift_ms": round(1000 * float(SHIFTS_S[int(np.argmin(mean))]), 1),
                  "mean_at_best": round(float(mean.min()), 4), "mean_at_0": round(float(mean[list(SHIFTS_S).index(0.0)]), 4)}
    agree = {k: sum(r["agree"][k] for r in results) for k in results[0]["agree"]}
    ms = np.concatenate([r["ms"] for r in results])
    report = {
        "input_rate_hz": args.rate, "clips": len(results), "compared_frames": sum(r["compared_frames"] for r in results),
        "held_frames_left_out": sum(r["held_compared"] for r in results),
        "difference_live_minus_data": diff,
        "units": {"joint_deg": "18 policy joints, deg", "joint_vel_rad_s": "rad/s", "body_pos_mm": "22 bodies, mm",
                  "trunk_pos_mm": "mm", "body_lin_vel_m_s": "22 bodies, m/s", "body_ang_vel_rad_s": "22 bodies, rad/s"},
        "render_delay_ms_median": round(1000 * float(np.median([r["delay_s"] for r in results])), 1),
        "content_lag_behind_render": lag,
        "late_tick_share": round(sum(r["late_ticks"] for r in results) / sum(r["ticks"] for r in results), 4),
        "exclusion_ticks": agree,
        "compute_ms_per_tick": {"mean": round(float(ms.mean()), 2), "p99": round(float(np.percentile(ms, 99)), 2),
                                "max": round(float(ms.max()), 2)},
        "per_clip": [{"clip": r["clip"], "frames": r["compared_frames"], "joint_deg_mean": round(float(r["diff"]["joint_deg"].mean()), 3)
                      if len(r["diff"]["joint_deg"]) else None} for r in results],
    }
    args.out.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "per_clip"}, indent=1))


if __name__ == "__main__":
    main()
