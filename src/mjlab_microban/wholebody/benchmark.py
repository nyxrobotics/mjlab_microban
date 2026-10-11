# Adapted from Teleopit f926386 train_mimic/benchmarking.py and
# train_mimic/scripts/benchmark.py (Apache-2.0,
# https://github.com/BotRunner64/Teleopit).  Changes: the evaluation set run
# in consecutive pieces, results per motion kind, error lines equal to the
# reward stds, a progress line for walking and running, no reset during a run.

"""Evaluate a checkpoint on the evaluation set, per motion kind.

Pieces.  Each evaluation section (one window of the evaluation shard) is run
in consecutive, non-overlapping pieces of at most ``clip_seconds``, so every
evaluated frame is run once.  A piece starts with the robot on the
reference's first frame and succeeds when no termination fires in it (the
end of the reference is a time-out, not a failure).  A kind's success rate
is the share of its evaluated seconds in successful pieces.

Conditions.  Observation noise, pushes and the physics randomization of
training are off.  The robot's own conditions (battery voltage and its drop,
actuator and sensor delays) are drawn from their training ranges with the
seed.  No env is reset during a run, so the draws do not depend on the
policy: two checkpoints meet the same conditions piece by piece.

Lines.  For every successful piece the RMS over time of each tracking error
(the quantity inside the reward's exp, square-rooted) is taken against the
reference of the step (the observed target); per kind the RMS over all steps
of its successful pieces is compared with that reward's std (RMS <= std: the
reward stays above exp(-1) on average).  The trunk velocity line cannot tell
a robot standing still from a walking one at Microban's speeds, so on walk,
run and sprint the trunk's horizontal displacement over each successful
piece is projected on the reference's: progress = sum(d_robot . d_ref) /
sum(|d_ref|^2), 0 standing still, 1 covering the reference's ground, and it
must reach ``MIN_PROGRESS``.  Every kind of the training set and walk, run
and sprint must be in the evaluation set.

The share of steps whose reference joint speed exceeds the XC330's no-load
speed is reported per kind; it is not a line.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np

from mjlab_microban.wholebody.data.hdf5 import find_motion_shards, motion_category, read_strings
from mjlab_microban.wholebody.env_cfg import (
    ACTOR_GROUP,
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
SERVO_VIN_V = 11.1


def servo_no_load_speed_rad_s(vin: float = SERVO_VIN_V) -> float:
    """XC330 no-load speed of the BAM identification: vin / kt."""
    params = json.loads((Path(__file__).resolve().parents[1] / "robot" / "xc330_params.json").read_text())
    return vin / float(params["kt"])


@dataclass(frozen=True)
class Piece:
    piece_id: int
    window_id: int
    clip_name: str
    kind: str
    start_s: float
    steps: int
    seconds: float
    overspeed_share: float


@dataclass(frozen=True)
class PieceResult:
    piece_id: int
    window_id: int
    clip_name: str
    kind: str
    start_s: float
    seconds: float
    steps: int
    success: bool
    failure_reason: str | None
    overspeed_share: float
    # RMS tracking errors of a successful piece (None after a fall).
    body_pos_m: float | None
    trunk_ori_rad: float | None
    joint_pos_rad: float | None
    trunk_lin_vel_m_s: float | None
    # Horizontal trunk displacements of a successful piece: d_robot . d_ref and |d_ref|^2.
    progress_dot_m2: float | None
    ref_displacement_sq_m2: float | None


def dataset_kinds(motion_file: str | Path) -> set[str]:
    """The motion kinds of a dataset's clips."""
    kinds = set()
    for shard in find_motion_shards(motion_file):
        with h5py.File(shard, "r") as h5:
            kinds.update(motion_category(name) for name in read_strings(h5, "clip_names"))
    return kinds


def load_pieces(
    motion_file: str | Path,
    joint_names: Sequence[str],
    clip_seconds: float,
    step_dt: float,
    no_load_speed: float,
) -> list[Piece]:
    """Every window (in MotionLib's order) cut into consecutive pieces of at most ``clip_seconds``."""
    pieces: list[Piece] = []
    window_id = 0
    for shard in find_motion_shards(motion_file):
        with h5py.File(shard, "r") as h5:
            names = read_strings(h5, "clip_names")
            shard_joints = read_strings(h5, "joint_names")
            columns = [shard_joints.index(name) for name in joint_names]
            fast = np.any(np.abs(np.asarray(h5["joint_vel"])[:, columns]) > no_load_speed, axis=1)
            for start, length, fps, source in zip(
                np.asarray(h5["clip_starts"]), np.asarray(h5["clip_lengths"]), np.asarray(h5["clip_fps"]),
                np.asarray(h5["source_clip_ids"]), strict=True,
            ):
                duration = (int(length) - 1) / float(fps)
                index = 0
                while True:
                    begin = index * clip_seconds
                    steps = int(math.floor(min(clip_seconds, duration - begin) / step_dt + 1e-6))
                    if steps < 1:
                        break
                    seconds = steps * step_dt
                    first = int(start) + int(math.floor(begin * fps))
                    last = int(start) + min(int(math.ceil((begin + seconds) * fps)), int(length) - 1)
                    pieces.append(Piece(
                        piece_id=len(pieces), window_id=window_id, clip_name=names[source],
                        kind=motion_category(names[source]), start_s=begin, steps=steps, seconds=seconds,
                        overspeed_share=float(np.mean(fast[first : last + 1])),
                    ))
                    index += 1
                window_id += 1
    return pieces


def _weighted(rows: Iterable[PieceResult], value: str, weight: str) -> float | None:
    rows = list(rows)
    total = sum(getattr(r, weight) for r in rows)
    return sum(getattr(r, value) * getattr(r, weight) for r in rows) / total if total > 0 else None


def summarize(results: Sequence[PieceResult]) -> dict[str, Any]:
    by_kind: dict[str, list[PieceResult]] = defaultdict(list)
    for result in results:
        by_kind[result.kind].append(result)
    kinds = {}
    for kind, rows in sorted(by_kind.items()):
        succeeded = [r for r in rows if r.success]
        seconds = sum(r.seconds for r in rows)
        entry: dict[str, Any] = {
            "pieces": len(rows),
            "seconds": seconds,
            "success_rate": sum(r.seconds for r in succeeded) / seconds,
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
        ref_sq = sum(r.ref_displacement_sq_m2 for r in succeeded)
        entry["progress"] = sum(r.progress_dot_m2 for r in succeeded) / ref_sq if ref_sq > 0 else None
        kinds[kind] = entry
    return kinds


def verdicts(kinds: dict[str, Any], expected_kinds: Iterable[str] = ()) -> dict[str, Any]:
    """The full run's pass (every kind at least 90 % and under the error lines,
    locomotion progress) and the small run's checks (no kind at 0 %,
    locomotion progress, finite errors).  A kind expected but not evaluated
    fails both."""
    missing = sorted((set(expected_kinds) | set(LOCOMOTION_KINDS)) - set(kinds))
    errors_ok = {k: all(v[n] is not None and v[n] <= std for n, std in ERROR_LINES.items()) for k, v in kinds.items()}
    progress = {k: kinds[k]["progress"] for k in LOCOMOTION_KINDS if k in kinds}
    progress_ok = not missing and all(v is not None and v >= MIN_PROGRESS for v in progress.values())
    finite = not any(v["non_finite_pieces"] for v in kinds.values())
    final = {
        "kinds_without_pieces": missing,
        "kinds_below_success_rate": sorted(k for k, v in kinds.items() if v["success_rate"] < FINAL_SUCCESS_RATE),
        "kinds_above_error_lines": sorted(k for k, ok in errors_ok.items() if not ok),
        "locomotion_progress": progress,
        "locomotion_progress_ok": progress_ok,
    }
    final["pass"] = bool(
        kinds and not missing and not final["kinds_below_success_rate"] and not final["kinds_above_error_lines"]
        and progress_ok and finite
    )
    small = {
        "kinds_without_pieces": missing,
        "kinds_never_succeeding": sorted(k for k, v in kinds.items() if v["success_rate"] == 0.0),
        "locomotion_progress": progress,
        "locomotion_progress_ok": progress_ok,
        "finite": finite,
    }
    small["pass"] = bool(kinds and not missing and not small["kinds_never_succeeding"] and progress_ok and finite)
    return {"final": final, "small_run": small}


def write_outputs(
    out_dir: Path, stem: str, metadata: dict[str, Any], results: Sequence[PieceResult], expected_kinds: Iterable[str]
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    kinds = summarize(results)
    report = {
        "metadata": metadata,
        "error_lines": ERROR_LINES,
        "final_success_rate": FINAL_SUCCESS_RATE,
        "min_progress": MIN_PROGRESS,
        "verdicts": verdicts(kinds, expected_kinds),
        "kinds": kinds,
        "pieces": [asdict(r) for r in results],
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(report, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    with (out_dir / f"{stem}-kinds.csv").open("w", newline="") as stream:
        fields = ["kind", "pieces", "seconds", "success_rate", "overspeed_share", "progress", *ERROR_LINES]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
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

    lines = [f"{'kind':<18}{'n':>5}{'min':>6}{'success':>9}{'fast':>6}{'progress':>9}"
             + "".join(f"{n:>19}" for n in ERROR_LINES)]
    for kind, v in report["kinds"].items():
        errors = "".join(f"{number(v[n], '.3f'):>13}/{std:<5}" for n, std in ERROR_LINES.items())
        lines.append(f"{kind:<18}{v['pieces']:>5}{v['seconds'] / 60:>6.1f}{100 * v['success_rate']:>8.0f}%"
                     f"{100 * (v['overspeed_share'] or 0.0):>5.0f}%{number(v['progress'], '.2f'):>9}{errors}")
    final, small = report["verdicts"]["final"], report["verdicts"]["small_run"]
    lines.append(f"final pass: {final['pass']}  missing kinds: {final['kinds_without_pieces']}  "
                 f"below 90%: {final['kinds_below_success_rate']}  above error lines: {final['kinds_above_error_lines']}")
    lines.append(f"small-run checks: {small['pass']}  never succeeding: {small['kinds_never_succeeding']}  "
                 f"locomotion progress (>= {MIN_PROGRESS}): {small['locomotion_progress']}")
    return "\n".join(lines)


def rollout(
    env: Any,
    policy: Any,
    pieces: Sequence[Piece],
    *,
    record_every: int = 0,
) -> tuple[list[PieceResult], list[Any]]:
    """Run ``pieces`` (one env each) from their start for their steps.

    ``env`` is the unwrapped ManagerBasedRlEnv (``auto_reset=False``).  With
    ``record_every`` > 0 the actor history of env 0 is recorded every that
    many steps until the longest piece ends.
    """
    import torch
    from mjlab.utils.lab_api.math import quat_error_magnitude
    from tensordict import TensorDict

    n = len(pieces)
    device = env.device
    env.reset()
    command = env.command_manager.get_term("motion")
    ids = torch.arange(n, device=device)
    command.reset_to_motion(ids, torch.tensor([p.window_id for p in pieces], device=device),
                            torch.tensor([p.start_s for p in pieces], device=device))
    # env.reset's sequence for the placed envs: the reference one step ahead,
    # the neck target, then fresh observation histories.
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.command_manager.compute(dt=0.0)
    env.event_manager.apply(mode="step", dt=env.step_dt)
    env.sim.sense()
    env.observation_manager.reset(ids)
    obs = TensorDict(env.observation_manager.compute(update_history=True), batch_size=[n])
    # Placed exactly on the reference (no pose offsets in play).
    start_xy = command.robot_anchor_pos_w[:, :2].clone()

    budgets = torch.tensor([p.steps for p in pieces], device=device)
    sums = {name: torch.zeros(n, device=device) for name in ERROR_LINES}
    running = torch.ones(n, dtype=torch.bool, device=device)
    success = torch.zeros(n, dtype=torch.bool, device=device)
    steps = torch.zeros(n, dtype=torch.long, device=device)
    robot_end_xy = torch.zeros(n, 2, device=device)
    ref_end_xy = torch.zeros(n, 2, device=device)
    reasons: list[str | None] = [None] * n
    recorded = []
    for step in range(int(budgets.max())):
        if record_every and step % record_every == record_every - 1:
            recorded.append(obs[ACTOR_GROUP][0].detach().cpu().clone())
        # The step's target, which its rewards and terminations compare the robot with.
        ref = {
            "body_pos": command.body_pos_relative_w.clone(),
            "trunk_quat": command.anchor_quat_w.clone(),
            "joint_pos": command.joint_pos.clone(),
            "trunk_lin_vel": command.anchor_lin_vel_w.clone(),
            "trunk_xy": command.anchor_pos_w[:, :2].clone(),
        }
        with torch.inference_mode():
            actions = policy(obs)
        raw, _, terminated, _, _ = env.step(actions)
        # No reset during a run: a reset would add a row to every env's
        # history and draw new conditions; an ended piece just keeps going.
        env._manual_reset_pending.zero_()
        obs = TensorDict(raw, batch_size=[n])
        errors = {
            "body_pos_m": torch.square(ref["body_pos"] - command.robot_body_pos_w).sum(-1).mean(-1),
            "trunk_ori_rad": quat_error_magnitude(ref["trunk_quat"], command.robot_anchor_quat_w) ** 2,
            "joint_pos_rad": torch.square(ref["joint_pos"] - command.robot_joint_pos).mean(-1),
            "trunk_lin_vel_m_s": torch.square(ref["trunk_lin_vel"] - command.robot_anchor_lin_vel_w).sum(-1),
        }
        for name, value in errors.items():
            sums[name] += torch.where(running, value, torch.zeros_like(value))
        steps += running.long()
        failed = terminated & running
        for index in failed.nonzero().flatten().tolist():
            reasons[index] = next(
                (name for name in env.termination_manager.active_terms
                 if not env.termination_manager.get_term_cfg(name).time_out
                 and bool(env.termination_manager.get_term(name)[index])),
                "terminated",
            )
        running &= ~failed
        finished = running & (steps == budgets)
        robot_end_xy[finished] = command.robot_anchor_pos_w[finished, :2]
        ref_end_xy[finished] = ref["trunk_xy"][finished]
        success |= finished
        running &= ~finished
        if not record_every and not bool(running.any()):
            break

    robot_disp = (robot_end_xy - start_xy).cpu()
    ref_disp = (ref_end_xy - start_xy).cpu()
    results = []
    for index, piece in enumerate(pieces):
        ok = bool(success[index])
        rms = {name: math.sqrt(float(sums[name][index]) / max(int(steps[index]), 1)) for name in ERROR_LINES}
        if ok and not all(math.isfinite(value) for value in rms.values()):
            ok, reasons[index] = False, "non_finite"
        results.append(
            PieceResult(
                piece_id=piece.piece_id, window_id=piece.window_id, clip_name=piece.clip_name, kind=piece.kind,
                start_s=piece.start_s, seconds=piece.seconds, steps=int(steps[index]), success=ok,
                failure_reason=None if ok else (reasons[index] or "terminated"),
                overspeed_share=piece.overspeed_share,
                **{name: (value if ok else None) for name, value in rms.items()},
                progress_dot_m2=float(robot_disp[index] @ ref_disp[index]) if ok else None,
                ref_displacement_sq_m2=float(ref_disp[index] @ ref_disp[index]) if ok else None,
            )
        )
    return results, recorded
