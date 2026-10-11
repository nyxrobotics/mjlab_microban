# Adapted from Teleopit f926386 train_mimic/scripts/convert_pkl_to_npz.py
# (convert_pkl_to_arrays; Apache-2.0, https://github.com/BotRunner64/Teleopit).
# Changes: the Microban model with the trunk as root, PKL input only, the
# frame rules of preprocess.py cut clips into named sections, and the
# LAFAN1 evaluation split.

"""Build the training and evaluation shards from GMR's retargeted PKL files.

Each PKL (GMR's standard output) holds ``fps``, ``root_pos`` (T, 3),
``root_rot`` (T, 4, xyzw) and ``dof_pos`` (T, 21: the hinge joints in
robot.xml qpos order, or in the order of an optional ``joint_names``).

Split: the ``subject5`` files are the evaluation set (LAFAN1's standard
split); every file of a kind without one gives its last 20 % of frames.  Both
sets go through the same cuts.  The training shard is cut into overlapping
windows (Teleopit's); each evaluation section is one window, so the
evaluation can run it in consecutive pieces.  An evaluation-only set (the
PICO recordings) puts every file in the evaluation set.
"""

from __future__ import annotations

import json
import pickle
import re
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from mjlab_microban.wholebody.data.fk import (
    compute_body_velocities,
    finite_diff_velocity,
    normalize_quaternion,
    quat_xyzw_to_wxyz,
)
from mjlab_microban.wholebody.data.hdf5 import motion_category, write_motion_shard
from mjlab_microban.wholebody.data.preprocess import (
    REASONS,
    ClipChecker,
    ExclusionSpec,
    frame_flags,
    kept_segments,
)

EVAL_SUBJECT = "subject5"
SINGLE_FILE_EVAL_FRACTION = 0.2
# A kind losing more than this share of its frames needs the retargeting
# checked before training.
REVIEW_EXCLUDED_FRACTION = 0.3


def load_retarget_pkl(path: Path, joint_names: Sequence[str]) -> dict[str, Any]:
    """Root pose (wxyz) and the joints in ``joint_names`` order of one GMR PKL."""
    with open(path, "rb") as stream:
        data = pickle.load(stream)
    missing = sorted({"fps", "root_pos", "root_rot", "dof_pos"} - set(data))
    if missing:
        raise ValueError(f"{path}: missing PKL keys {missing}")
    fps_value = float(data["fps"])
    fps = int(round(fps_value))
    if fps <= 0 or abs(fps_value - fps) > 0.01:
        raise ValueError(f"{path}: fps {fps_value} is not an integer rate")
    root_pos = np.asarray(data["root_pos"], dtype=np.float32)
    root_rot = np.asarray(data["root_rot"], dtype=np.float32)
    dof_pos = np.asarray(data["dof_pos"], dtype=np.float32)
    frames = root_pos.shape[0]
    if root_pos.shape != (frames, 3) or root_rot.shape != (frames, 4) or dof_pos.shape != (frames, len(joint_names)):
        raise ValueError(f"{path}: expected root_pos (T,3), root_rot (T,4), dof_pos (T,{len(joint_names)})")
    names = data.get("joint_names")
    if names is not None:
        names = [str(name) for name in names]
        if sorted(names) != sorted(joint_names):
            raise ValueError(f"{path}: joint_names are not the robot's hinge joints")
        dof_pos = dof_pos[:, [names.index(name) for name in joint_names]]
    if not (np.isfinite(root_pos).all() and np.isfinite(root_rot).all() and np.isfinite(dof_pos).all()):
        raise ValueError(f"{path}: non-finite values")
    return {
        "fps": fps,
        "root_pos": root_pos,
        "root_quat_w": normalize_quaternion(quat_xyzw_to_wxyz(root_rot)),
        "joint_pos": dof_pos,
    }


def split_files(names: Sequence[str]) -> dict[str, str]:
    """``train``, ``eval`` or ``tail`` (last 20 % to eval) for each file stem."""
    by_kind: dict[str, list[str]] = defaultdict(list)
    for name in sorted(names):
        by_kind[motion_category(name)].append(name)
    split = {}
    for files in by_kind.values():
        evaluation = [name for name in files if re.search(rf"{EVAL_SUBJECT}(?!\d)", name)]
        for name in files:
            split[name] = ("eval" if name in evaluation else "train") if evaluation else "tail"
    return split


def _section(name: str, clip: dict[str, Any], kin: Any, start: int, end: int, offset: int) -> dict[str, Any]:
    fps = int(clip["fps"])
    dt = 1.0 / fps
    joint_pos = clip["joint_pos"][start:end]
    body_pos = kin.body_pos_w[start:end]
    body_quat = kin.body_quat_w[start:end]
    lin_vel, ang_vel = compute_body_velocities(body_pos, body_quat, dt)
    return {
        "name": f"{name}__f{offset + start}-{offset + end}",
        "fps": fps,
        "joint_pos": joint_pos,
        "joint_vel": finite_diff_velocity(joint_pos, dt),
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "body_lin_vel_w": lin_vel,
        "body_ang_vel_w": ang_vel,
    }


_WORKER: ClipChecker | None = None


def _checker() -> ClipChecker:
    global _WORKER
    if _WORKER is None:
        from mjlab_microban.wholebody.env_cfg import training_collision_model

        _WORKER = ClipChecker(training_collision_model())
    return _WORKER


def process_file(path: Path, part: str, spec: ExclusionSpec) -> dict[str, Any]:
    """Cut one PKL; ``part`` is train, eval or tail.  Returns the sections and the counts."""
    checker = _checker()
    clip = load_retarget_pkl(path, checker.joint_names)
    frames = clip["joint_pos"].shape[0]
    pieces = [(part, 0, frames)]
    if part == "tail":
        cut = frames - int(round(SINGLE_FILE_EVAL_FRACTION * frames))
        pieces = [("train", 0, cut), ("eval", cut, frames)]
    result: dict[str, Any] = {
        "name": path.stem,
        "fps": clip["fps"],
        "body_names": checker.body_names,
        "joint_names": checker.joint_names,
        "parts": [],
    }
    for target, first, last in pieces:
        piece = {key: (value[first:last] if isinstance(value, np.ndarray) else value) for key, value in clip.items()}
        kin = checker.run(piece["root_pos"], piece["root_quat_w"], piece["joint_pos"])
        flags = frame_flags(kin, clip["fps"], spec)
        segments = kept_segments(flags, clip["fps"], spec)
        result["parts"].append(
            {
                "split": target,
                "frames": last - first,
                "kept_frames": int(sum(e - s for s, e in segments)),
                "flagged_frames": {reason: int(flags[reason].sum()) for reason in REASONS},
                "sections": [_section(path.stem, piece, kin, s, e, first) for s, e in segments],
            }
        )
    return result


def build_dataset(
    retarget_dir: Path,
    out_dir: Path,
    *,
    spec: ExclusionSpec = ExclusionSpec(),
    workers: int = 1,
    eval_only: bool = False,
) -> dict[str, Any]:
    """Write ``out_dir/{train,eval}/motion.h5`` (``eval_only``: the evaluation shard only) and ``out_dir/report.json``."""
    files = sorted(Path(retarget_dir).glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"no PKL files in {retarget_dir}")
    split = {path.stem: "eval" for path in files} if eval_only else split_files([path.stem for path in files])
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(process_file, files, [split[p.stem] for p in files], [spec] * len(files)))
    else:
        results = [process_file(path, split[path.stem], spec) for path in files]

    sections: dict[str, list[dict[str, Any]]] = {"train": [], "eval": []}
    kinds: dict[str, dict[str, Any]] = {}
    for result in results:
        kind = motion_category(result["name"])
        row = kinds.setdefault(
            kind,
            {"files": {"train": [], "eval": []}, "frames": 0, "seconds": 0.0, "kept_frames": 0,
             "flagged_frames": dict.fromkeys(REASONS, 0)},
        )
        for part in result["parts"]:
            sections[part["split"]].extend(part["sections"])
            row["files"][part["split"]].append(result["name"])
            row["frames"] += part["frames"]
            row["seconds"] += part["frames"] / result["fps"]
            row["kept_frames"] += part["kept_frames"]
            for reason, count in part["flagged_frames"].items():
                row["flagged_frames"][reason] += count
    for row in kinds.values():
        frames = max(row["frames"], 1)
        row["excluded_fraction"] = 1.0 - row["kept_frames"] / frames
        row["flagged_fraction"] = {reason: count / frames for reason, count in row["flagged_frames"].items()}
        row["needs_review"] = row["excluded_fraction"] > REVIEW_EXCLUDED_FRACTION

    shards = {}
    for part in ("eval",) if eval_only else ("train", "eval"):
        if not sections[part]:
            raise ValueError(f"no {part} sections survived the cuts")
        shards[part] = write_motion_shard(
            sections[part],
            Path(out_dir) / part / "motion.h5",
            body_names=results[0]["body_names"],
            joint_names=results[0]["joint_names"],
            **({} if part == "train" else {"max_window_frames": None}),
        )
    report = {
        "retarget_dir": str(Path(retarget_dir).resolve()),
        "exclusion": spec.to_dict(),
        "shards": shards,
        "kinds": dict(sorted(kinds.items())),
        "review_excluded_fraction": REVIEW_EXCLUDED_FRACTION,
    }
    report_path = Path(out_dir) / "report.json"
    report_path.write_text(json.dumps(report, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    return report


def format_report(report: dict[str, Any]) -> str:
    lines = [f"{'kind':<18}{'files tr/ev':>12}{'min':>7}{'excluded':>10}  flagged % ({'/'.join(REASONS)})"]
    for kind, row in report["kinds"].items():
        f = row["flagged_fraction"]
        minutes = row["seconds"] / 60.0
        lines.append(
            f"{kind:<18}{len(row['files']['train']):>5}/{len(row['files']['eval']):<6}{minutes:>7.1f}"
            f"{100 * row['excluded_fraction']:>9.1f}%  "
            + "/".join(f"{100 * f[r]:.1f}" for r in REASONS)
            + ("  REVIEW" if row["needs_review"] else "")
        )
    for part, shard in report["shards"].items():
        lines.append(f"{part}: {shard['source_clips']} sections, {shard['windows']} windows, "
                     f"{shard['duration_s'] / 60:.1f} min -> {shard['path']}")
    return "\n".join(lines)
