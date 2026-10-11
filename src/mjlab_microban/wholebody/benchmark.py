# Adapted from Teleopit f926386 train_mimic/benchmarking.py and
# train_mimic/scripts/benchmark.py (Apache-2.0,
# https://github.com/BotRunner64/Teleopit).  Changes: results per motion kind,
# error lines equal to the reward stds, windows whose reference outruns the
# servo counted apart.

"""Evaluate a checkpoint on the evaluation windows (deterministic, no randomization).

Each window of at least ``clip_seconds`` is started at its first frame and
run for ``clip_seconds`` (500 policy steps); it succeeds when no termination
fires.  For every successful window, the RMS over time of each tracking
error (the quantity inside the reward's exp, square-rooted) is compared with
that reward's std: RMS <= std means the reward stays above exp(-1) on
average.  A window whose reference joint speed exceeds the XC330's no-load
speed somewhere in the run cannot be followed by the robot: it is reported
apart and left out of the pass lines.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np

from mjlab_microban.wholebody.data.hdf5 import find_motion_shards, motion_category, read_strings
from mjlab_microban.wholebody.env_cfg import (
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
# Small-run gate: the trunk velocity line on these kinds (a standing-still
# solution fails it).
LOCOMOTION_KINDS = ("walk", "run", "sprint")
SERVO_VIN_V = 11.1


def servo_no_load_speed_rad_s(vin: float = SERVO_VIN_V) -> float:
    """XC330 no-load speed of the BAM identification: vin / kt."""
    params = json.loads((Path(__file__).resolve().parents[1] / "robot" / "xc330_params.json").read_text())
    return vin / float(params["kt"])


@dataclass(frozen=True)
class WindowSpec:
    window_id: int
    clip_name: str
    kind: str
    fps: float
    frames: int
    overspeed: bool


@dataclass(frozen=True)
class WindowResult:
    window_id: int
    clip_name: str
    kind: str
    overspeed: bool
    success: bool
    steps: int
    failure_reason: str | None
    # RMS tracking errors of a successful window (None after a fall).
    body_pos_m: float | None
    trunk_ori_rad: float | None
    joint_pos_rad: float | None
    trunk_lin_vel_m_s: float | None


def load_windows(
    motion_file: str | Path, joint_names: Sequence[str], clip_seconds: float, no_load_speed: float
) -> list[WindowSpec]:
    """All windows in MotionLib's order, with the overspeed flag over the evaluated span."""
    specs = []
    for shard in find_motion_shards(motion_file):
        with h5py.File(shard, "r") as h5:
            names = read_strings(h5, "clip_names")
            shard_joints = read_strings(h5, "joint_names")
            columns = [shard_joints.index(name) for name in joint_names]
            joint_vel = np.asarray(h5["joint_vel"])[:, columns]
            for start, length, fps, source in zip(
                np.asarray(h5["clip_starts"]), np.asarray(h5["clip_lengths"]), np.asarray(h5["clip_fps"]),
                np.asarray(h5["source_clip_ids"]), strict=True,
            ):
                span = min(int(length), int(math.ceil(clip_seconds * fps)) + 1)
                speed = float(np.max(np.abs(joint_vel[start : start + span])))
                specs.append(WindowSpec(len(specs), names[source], motion_category(names[source]), float(fps),
                                        int(length), speed > no_load_speed))
    return specs


def eligible(specs: Sequence[WindowSpec], clip_seconds: float) -> list[WindowSpec]:
    return [s for s in specs if (s.frames - 1) / s.fps + 1e-9 >= clip_seconds]


def summarize(results: Sequence[WindowResult]) -> dict[str, Any]:
    by_kind: dict[str, list[WindowResult]] = defaultdict(list)
    for result in results:
        by_kind[result.kind].append(result)
    kinds = {}
    for kind, rows in sorted(by_kind.items()):
        counted = [r for r in rows if not r.overspeed]
        fast = [r for r in rows if r.overspeed]
        succeeded = [r for r in counted if r.success]
        entry: dict[str, Any] = {
            "windows": len(counted),
            "success_rate": len(succeeded) / len(counted) if counted else None,
            "overspeed_windows": len(fast),
            "overspeed_success_rate": sum(r.success for r in fast) / len(fast) if fast else None,
            "non_finite_windows": sum(r.failure_reason == "non_finite" for r in rows),
        }
        reasons: dict[str, int] = defaultdict(int)
        for r in counted:
            if not r.success:
                reasons[r.failure_reason or "unknown"] += 1
        entry["failure_reasons"] = dict(sorted(reasons.items()))
        for name in ERROR_LINES:
            values = [getattr(r, name) for r in succeeded]
            entry[name] = float(np.sqrt(np.mean(np.square(values)))) if values else None
        kinds[kind] = entry
    return kinds


def verdicts(kinds: dict[str, Any]) -> dict[str, Any]:
    """The full run's pass (every kind at least 90 % and under the error lines) and
    the small run's checks (no kind at 0 %, the locomotion velocity line, finite errors)."""
    judged = {k: v for k, v in kinds.items() if v["windows"]}
    errors_ok = {
        k: all(v[n] is not None and v[n] <= std for n, std in ERROR_LINES.items()) for k, v in judged.items()
    }
    final = {
        "pass": bool(judged) and all(
            v["success_rate"] >= FINAL_SUCCESS_RATE and errors_ok[k] for k, v in judged.items()
        ),
        "kinds_below_success_rate": sorted(k for k, v in judged.items() if v["success_rate"] < FINAL_SUCCESS_RATE),
        "kinds_above_error_lines": sorted(k for k, ok in errors_ok.items() if not ok),
    }
    locomotion = {
        k: judged[k]["trunk_lin_vel_m_s"] for k in LOCOMOTION_KINDS if k in judged
    }
    small = {
        "kinds_never_succeeding": sorted(k for k, v in judged.items() if v["success_rate"] == 0.0),
        "locomotion_trunk_lin_vel_m_s": locomotion,
        "locomotion_velocity_ok": all(v is not None and v <= LIN_VEL_STD_M_S for v in locomotion.values()),
        "finite": not any(v["non_finite_windows"] for v in kinds.values()),
    }
    small["pass"] = (not small["kinds_never_succeeding"]) and small["locomotion_velocity_ok"] and small["finite"]
    return {"final": final, "small_run": small}


def write_outputs(out_dir: Path, stem: str, metadata: dict[str, Any], results: Sequence[WindowResult]) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    kinds = summarize(results)
    report = {
        "metadata": metadata,
        "error_lines": ERROR_LINES,
        "final_success_rate": FINAL_SUCCESS_RATE,
        "verdicts": verdicts(kinds),
        "kinds": kinds,
        "windows": [asdict(r) for r in results],
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(report, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    with (out_dir / f"{stem}-kinds.csv").open("w", newline="") as stream:
        fields = ["kind", "windows", "success_rate", "overspeed_windows", "overspeed_success_rate", *ERROR_LINES]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for kind, entry in kinds.items():
            writer.writerow({"kind": kind, **entry})
    with (out_dir / f"{stem}-windows.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(results[0])))
        writer.writeheader()
        writer.writerows(asdict(r) for r in results)
    return report


def format_summary(report: dict[str, Any]) -> str:
    lines = [f"{'kind':<18}{'n':>4}{'success':>9}{'fast n/ok':>11}" + "".join(f"{n:>19}" for n in ERROR_LINES)]
    for kind, v in report["kinds"].items():
        rate = "-" if v["success_rate"] is None else f"{100 * v['success_rate']:.0f}%"
        fast = f"{v['overspeed_windows']}/" + (
            "-" if v["overspeed_success_rate"] is None else f"{100 * v['overspeed_success_rate']:.0f}%"
        )
        errors = "".join(
            f"{'-' if v[n] is None else f'{v[n]:.3f}':>13}/{std:<5}" for n, std in ERROR_LINES.items()
        )
        lines.append(f"{kind:<18}{v['windows']:>4}{rate:>9}{fast:>11}{errors}")
    final, small = report["verdicts"]["final"], report["verdicts"]["small_run"]
    lines.append(f"final pass: {final['pass']}  below 90%: {final['kinds_below_success_rate']}  "
                 f"above error lines: {final['kinds_above_error_lines']}")
    lines.append(f"small-run checks: {small['pass']}  never succeeding: {small['kinds_never_succeeding']}  "
                 f"locomotion trunk velocity RMS: {small['locomotion_trunk_lin_vel_m_s']}")
    return "\n".join(lines)


def rollout(
    env: Any,
    policy: Any,
    windows: Sequence[WindowSpec],
    control_steps: int,
    *,
    record_every: int = 0,
) -> tuple[list[WindowResult], list[tuple[Any, Any]]]:
    """Run ``windows`` (one env each) from their first frame for ``control_steps``.

    ``env`` is the unwrapped ManagerBasedRlEnv (``auto_reset=False``).  With
    ``record_every`` > 0 the (actor, actor_history) observations of env 0 are
    recorded every that many steps for the whole run (a window that ended is
    replaced by a reset one).
    """
    import torch
    from mjlab.utils.lab_api.math import quat_error_magnitude
    from tensordict import TensorDict

    n = len(windows)
    device = env.device
    env.reset()
    command = env.command_manager.get_term("motion")
    ids = torch.arange(n, device=device)
    command.reset_to_motion(ids, torch.tensor([w.window_id for w in windows], device=device),
                            torch.zeros(n, device=device))
    env.scene.write_data_to_sim()
    env.sim.forward()
    # The reference moved under the robot (what the terminations compare) and the neck target.
    env.command_manager.compute(dt=0.0)
    env.event_manager.apply(mode="step", dt=env.step_dt)
    env.sim.sense()
    env.observation_manager.reset(ids)
    obs = TensorDict(env.observation_manager.compute(update_history=True), batch_size=[n])

    sums = {name: torch.zeros(n, device=device) for name in ERROR_LINES}
    active = torch.ones(n, dtype=torch.bool, device=device)
    steps = torch.zeros(n, dtype=torch.long, device=device)
    reasons: list[str | None] = [None] * n
    recorded = []
    for step in range(control_steps):
        if record_every and step % record_every == record_every - 1:
            recorded.append((obs["actor"][0].detach().cpu().clone(), obs["actor_history"][0].detach().cpu().clone()))
        with torch.inference_mode():
            actions = policy(obs)
        raw, _, terminated, truncated, _ = env.step(actions)
        obs = TensorDict(raw, batch_size=[n])
        errors = {
            "body_pos_m": torch.square(command.body_pos_relative_w - command.robot_body_pos_w).sum(-1).mean(-1),
            "trunk_ori_rad": quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2,
            "joint_pos_rad": torch.square(command.joint_pos - command.robot_joint_pos).mean(-1),
            "trunk_lin_vel_m_s": torch.square(command.anchor_lin_vel_w - command.robot_anchor_lin_vel_w).sum(-1),
        }
        for name, value in errors.items():
            sums[name] += torch.where(active, value, torch.zeros_like(value))
        steps += active.long()
        failed = terminated & active
        for index in failed.nonzero().flatten().tolist():
            reasons[index] = next(
                (name for name in env.termination_manager.active_terms
                 if not env.termination_manager.get_term_cfg(name).time_out
                 and bool(env.termination_manager.get_term(name)[index])),
                "terminated",
            )
        active &= ~terminated
        done = (terminated | truncated).nonzero().flatten()
        if done.numel() and step + 1 < control_steps:
            raw, _ = env.reset(env_ids=done)
            obs = TensorDict(raw, batch_size=[n])
        if not record_every and not bool(active.any()):
            break

    results = []
    for index, window in enumerate(windows):
        success = bool(active[index]) and int(steps[index]) == control_steps
        rms = {name: float(torch.sqrt(sums[name][index] / max(int(steps[index]), 1))) for name in ERROR_LINES}
        if success and not all(math.isfinite(value) for value in rms.values()):
            success, reasons[index] = False, "non_finite"
        results.append(
            WindowResult(
                window_id=window.window_id, clip_name=window.clip_name, kind=window.kind,
                overspeed=window.overspeed, success=success, steps=int(steps[index]),
                failure_reason=None if success else (reasons[index] or "terminated"),
                **{name: (value if success else None) for name, value in rms.items()},
            )
        )
    return results, recorded
