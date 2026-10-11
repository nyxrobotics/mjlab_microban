# Adapted from Teleopit f926386 train_mimic/data/dataset_lib.py (Apache-2.0,
# https://github.com/BotRunner64/Teleopit), its HDF5 part.  Changes: one
# format (the precomputed training shard, written in one pass), the clips'
# names (``clip_names``, one per source clip) and joint names.

"""The whole-body motion dataset: precomputed HDF5 shards cut into windows.

Frame arrays are flat over all source clips; ``clip_starts``/``clip_lengths``
are the training windows (at most ``max_window_frames``, overlapping by
``overlap_frames``) and ``source_clip_ids`` names each window's source clip,
whose name is ``clip_names[source_clip_ids[i]]``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

FORMAT = "microban_wholebody_motion"
VERSION = 1
FRAME_KEYS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
WINDOW_KEYS = ("clip_starts", "clip_lengths", "clip_fps", "source_clip_ids", "source_start_frames")
SOURCE_KEYS = ("source_clip_starts", "source_clip_lengths", "source_clip_fps", "clip_names")
DEFAULT_MAX_WINDOW_FRAMES = 512
DEFAULT_WINDOW_OVERLAP_FRAMES = 64


def motion_category(clip_name: str) -> str:
    """The motion kind of a clip: the letters of its source file name before the first digit."""
    base = clip_name.split("__", 1)[0]
    match = re.match(r"[A-Za-z]+", base)
    if match is None:
        raise ValueError(f"clip name {clip_name!r} does not start with a motion kind")
    return match.group(0)


def window_ranges(clip_length: int, max_window_frames: int, overlap_frames: int) -> list[tuple[int, int]]:
    """``(start, length)`` windows of one clip, relative to the clip."""
    if clip_length <= 0:
        raise ValueError(f"clip_length must be > 0, got {clip_length}")
    if not 0 <= overlap_frames < max_window_frames:
        raise ValueError("overlap_frames must be in [0, max_window_frames)")
    if clip_length <= max_window_frames:
        return [(0, clip_length)]
    stride = max_window_frames - overlap_frames
    starts = list(range(0, clip_length - max_window_frames + 1, stride))
    if starts[-1] != clip_length - max_window_frames:
        starts.append(clip_length - max_window_frames)
    return [(start, max_window_frames) for start in starts]


def write_motion_shard(
    clips: Sequence[Mapping[str, Any]],
    path: Path,
    *,
    body_names: Sequence[str],
    joint_names: Sequence[str],
    max_window_frames: int = DEFAULT_MAX_WINDOW_FRAMES,
    overlap_frames: int = DEFAULT_WINDOW_OVERLAP_FRAMES,
) -> dict[str, Any]:
    """Write ``clips`` (each: name, fps and the FRAME_KEYS arrays) as one shard."""
    if not clips:
        raise ValueError(f"no clips to write to {path}")
    names = [str(clip["name"]) for clip in clips]
    if len(set(names)) != len(names):
        raise ValueError("clip names must be unique")
    frames = {key: [] for key in FRAME_KEYS}
    source_starts, source_lengths, source_fps = [], [], []
    windows: dict[str, list[int]] = {key: [] for key in WINDOW_KEYS}
    offset = 0
    for source_id, clip in enumerate(clips):
        length = int(np.asarray(clip["joint_pos"]).shape[0])
        fps = int(clip["fps"])
        for key in FRAME_KEYS:
            array = np.asarray(clip[key], dtype=np.float32)
            if array.shape[0] != length or not np.isfinite(array).all():
                raise ValueError(f"{names[source_id]}: {key} is not {length} finite frames")
            frames[key].append(array)
        for start, window_length in window_ranges(length, max_window_frames, overlap_frames):
            windows["clip_starts"].append(offset + start)
            windows["clip_lengths"].append(window_length)
            windows["clip_fps"].append(fps)
            windows["source_clip_ids"].append(source_id)
            windows["source_start_frames"].append(start)
        source_starts.append(offset)
        source_lengths.append(length)
        source_fps.append(fps)
        offset += length

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    strings = h5py.string_dtype(encoding="utf-8")
    with h5py.File(temporary, "w") as h5:
        h5.attrs["format"] = FORMAT
        h5.attrs["version"] = VERSION
        h5.attrs["max_window_frames"] = int(max_window_frames)
        h5.attrs["overlap_frames"] = int(overlap_frames)
        h5.create_dataset("body_names", data=np.asarray(body_names, dtype=object), dtype=strings)
        h5.create_dataset("joint_names", data=np.asarray(joint_names, dtype=object), dtype=strings)
        for key in FRAME_KEYS:
            h5.create_dataset(key, data=np.concatenate(frames[key], axis=0), chunks=True)
        for key, values in windows.items():
            h5.create_dataset(key, data=np.asarray(values, dtype=np.int64))
        h5.create_dataset("source_clip_starts", data=np.asarray(source_starts, dtype=np.int64))
        h5.create_dataset("source_clip_lengths", data=np.asarray(source_lengths, dtype=np.int64))
        h5.create_dataset("source_clip_fps", data=np.asarray(source_fps, dtype=np.int64))
        h5.create_dataset("clip_names", data=np.asarray(names, dtype=object), dtype=strings)
    temporary.replace(path)
    validate_motion_shard(path)
    return {
        "path": str(path),
        "source_clips": len(clips),
        "windows": len(windows["clip_starts"]),
        "frames": offset,
        "duration_s": float(sum(n / f for n, f in zip(source_lengths, source_fps))),
    }


def read_strings(h5: h5py.File, key: str) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in h5[key][()]]


def validate_motion_shard(path: str | Path) -> None:
    with h5py.File(path, "r") as h5:
        if h5.attrs.get("format") != FORMAT or int(h5.attrs.get("version", 0)) != VERSION:
            raise ValueError(f"{path} is not a {FORMAT} v{VERSION} shard; rebuild it")
        missing = [k for k in (*FRAME_KEYS, *WINDOW_KEYS, *SOURCE_KEYS, "body_names", "joint_names") if k not in h5]
        if missing:
            raise ValueError(f"{path} is missing {missing}")
        frames = h5["joint_pos"].shape[0]
        bodies = len(h5["body_names"])
        joints = len(h5["joint_names"])
        expected = {
            "joint_pos": (frames, joints),
            "joint_vel": (frames, joints),
            "body_pos_w": (frames, bodies, 3),
            "body_quat_w": (frames, bodies, 4),
            "body_lin_vel_w": (frames, bodies, 3),
            "body_ang_vel_w": (frames, bodies, 3),
        }
        for key, shape in expected.items():
            if h5[key].shape != shape:
                raise ValueError(f"{path}: {key} is {h5[key].shape}, expected {shape}")
        windows = h5["clip_starts"].shape[0]
        sources = h5["source_clip_starts"].shape[0]
        if any(h5[key].shape != (windows,) for key in WINDOW_KEYS):
            raise ValueError(f"{path}: window arrays differ in length")
        if any(h5[key].shape != (sources,) for key in SOURCE_KEYS):
            raise ValueError(f"{path}: source clip arrays differ in length")
        ids = np.asarray(h5["source_clip_ids"])
        if windows and (ids.min() < 0 or ids.max() >= sources):
            raise ValueError(f"{path}: source_clip_ids outside clip_names")
        starts = np.asarray(h5["clip_starts"])
        lengths = np.asarray(h5["clip_lengths"])
        if np.any(starts < 0) or np.any(starts + lengths > frames):
            raise ValueError(f"{path}: windows outside the frame arrays")


def find_motion_shards(root: str | Path) -> list[Path]:
    """The shards of a dataset directory (recursively) or the shard itself."""
    path = Path(root).expanduser().resolve()
    if path.is_file():
        candidates = [path]
    elif path.is_dir():
        candidates = sorted(path.rglob("*.h5"))
    else:
        raise FileNotFoundError(f"motion dataset not found: {root}")
    shards = []
    for candidate in candidates:
        try:
            with h5py.File(candidate, "r") as h5:
                if h5.attrs.get("format") == FORMAT:
                    shards.append(candidate)
        except OSError:
            continue
    if not shards:
        raise FileNotFoundError(f"no {FORMAT} shards under {root}")
    return shards
