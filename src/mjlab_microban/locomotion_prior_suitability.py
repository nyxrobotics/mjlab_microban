# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""CPU-only geometric suitability audit for a Microban locomotion prior.

The audit deliberately performs kinematics only: it loads the tracked MuJoCo
model, writes each recorded floating-root pose and joint configuration, calls
``mj_forward``, and measures the world-space corners of the twelve numbered
foot collision boxes.  It neither creates an MjLab environment nor steps
physics, so it cannot reserve a GPU or send commands to a robot.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCOMOTION_PRIOR_PATH = (
    PROJECT_ROOT / "data" / "motions" / "microban_twist2_walk004_locomotion_prior.npz"
)
DEFAULT_ROBOT_XML_PATH = (
    PROJECT_ROOT / "src" / "mjlab_microban" / "robot" / "microban" / "robot.xml"
)

# These bind the defaults to the exact tracked inputs for which this audit was
# introduced.  A regenerated candidate can still be audited, but its expected
# digest must be supplied explicitly instead of silently changing provenance.
DEFAULT_LOCOMOTION_PRIOR_SHA256 = (
    "100656a04438e5b7c09d80e63f79f3e758650a20d87326f3e3970616e6fb69e2"
)
DEFAULT_ROBOT_XML_SHA256 = (
    "27a8a5731bc389ade2fa0cab6d3530515175988667cb0ea386cec577cd68ad8f"
)

LOCOMOTION_PRIOR_SUITABILITY_SCHEMA_VERSION = 3
LOCOMOTION_PRIOR_SUITABILITY_REVISION = (
    "microban_locomotion_prior_static_suitability_v3"
)
DEFAULT_START_FRAME = 109
DEFAULT_END_FRAME = 267
ROOT_FREEJOINT_NAME = "trunk_freejoint"
ROOT_BODY_NAME = "trunk"

FOOT_COLLISION_GEOM_NAMES: dict[str, tuple[str, ...]] = {
    side: tuple(f"{side}_foot_collision_{index}" for index in range(1, 7))
    for side in ("left", "right")
}

_BOX_CORNER_SIGNS = np.asarray(
    tuple(itertools.product((-1.0, 1.0), repeat=3)), dtype=np.float64
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
LOCOMOTION_PRIOR_RETARGET_RECIPE = "microban-bilateral-sagittal-sole-v2"
LOCOMOTION_PRIOR_SWING_BASELINE_ESTIMATOR = "midpoint-p05-p95-v1"
LOCOMOTION_PRIOR_SWING_DEADBAND_M = 0.001
LOCOMOTION_PRIOR_SWING_PHASE_FILTER = (0.0625, 0.25, 0.375, 0.25, 0.0625)
LOCOMOTION_PRIOR_SAGITTAL_CORRECTION_JOINT_SUFFIXES = (
    "hip_pitch",
    "knee",
    "ankle_pitch",
)
LOCOMOTION_PRIOR_SAGITTAL_CORRECTION_RAD = (-0.025, -0.025, 0.0125)
LOCOMOTION_PRIOR_SWING_TARGET_BOOST_M = 0.0


@dataclass(frozen=True)
class LocomotionPriorSuitabilityThresholds:
    """Static gates that a prior must pass before dynamics are considered."""

    grounding_error_m_max: float = 0.0005
    relative_swing_clearance_max_m_min: float = 0.015
    relative_swing_clearance_p95_m_min: float = 0.010

    def validate(self) -> None:
        values = asdict(self)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in values.values()
        ):
            raise ValueError("Suitability thresholds must be finite and non-negative")


DEFAULT_THRESHOLDS = LocomotionPriorSuitabilityThresholds()


@dataclass(frozen=True)
class _PriorPoseArrays:
    fps: float
    joint_names: tuple[str, ...]
    joint_pos: np.ndarray
    root_pos: np.ndarray
    root_quat: np.ndarray
    body_names: tuple[str, ...]
    source_session_id: str
    source_schema: str
    retarget_profile: str
    trim_start_s: float
    trim_end_s: float
    source_capture_sha256: str
    retarget_model_sha256: str
    retarget_root_translation_scale: float
    retarget_recipe: str
    source_swing_baseline_estimator: str
    source_swing_baseline_m: float
    source_swing_deadband_m: float
    source_swing_scale_m: tuple[float, float]
    source_right_swing_frame_count: int
    source_left_swing_frame_count: int
    source_swing_transition_count: int
    source_foot_height_difference_sha256: str
    swing_phase_filter: tuple[float, ...]
    sagittal_correction_joint_suffixes: tuple[str, ...]
    sagittal_correction_rad: tuple[float, ...]
    swing_target_boost_m: float
    archive_members: tuple[str, ...]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of one input without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    """Hash the stable, whitespace-free JSON representation of ``value``."""

    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def serialize_suitability_receipt(report: dict[str, Any]) -> str:
    """Serialize a receipt deterministically, including its final newline."""

    return (
        json.dumps(
            report,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def publish_suitability_receipt(
    output: Path,
    report: dict[str, Any],
    *,
    force: bool,
) -> None:
    """Atomically publish a complete receipt without accidental replacement."""

    output = output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = serialize_suitability_receipt(report)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())

        if force:
            os.replace(temporary, output)
            temporary = None
            return

        try:
            os.link(temporary, output)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Receipt already exists; refusing to replace: {output}"
            ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _portable_input_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        # The digest is the input identity.  Keeping host-specific absolute
        # prefixes out of a receipt makes copied-worktree runs byte-identical.
        return resolved.name


def _validate_expected_sha256(label: str, actual: str, expected: str | None) -> None:
    if expected is None:
        return
    if _SHA256_RE.fullmatch(expected) is None:
        raise ValueError(f"Expected {label} SHA-256 must be 64 lowercase hex digits")
    if actual != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected}, observed {actual}"
        )


def _scalar_text(value: np.ndarray, name: str) -> str:
    if value.shape != () or value.dtype.kind not in ("U", "S"):
        raise ValueError(f"Locomotion prior {name} must be one string scalar")
    return str(value.item())


def _scalar_float(value: np.ndarray, name: str) -> float:
    if value.shape not in ((), (1,)) or value.dtype.kind not in ("f", "i", "u"):
        raise ValueError(f"Locomotion prior {name} must be one numeric scalar")
    result = float(value.reshape(-1)[0])
    if not math.isfinite(result):
        raise ValueError(f"Locomotion prior {name} must be finite")
    return result


def _scalar_sha256(value: np.ndarray, name: str) -> str:
    result = _scalar_text(value, name)
    if _SHA256_RE.fullmatch(result) is None:
        raise ValueError(f"Locomotion prior {name} must be 64 lowercase hex digits")
    return result


def _load_prior_pose_arrays(
    path: Path,
    *,
    start_frame: int,
    end_frame: int,
) -> _PriorPoseArrays:
    required = {
        "fps",
        "joint_names",
        "joint_pos",
        "body_names",
        "body_pos_w",
        "body_quat_w",
        "source_session_id",
        "source_schema",
        "retarget_profile",
        "trim_start_s",
        "trim_end_s",
        "source_capture_sha256",
        "retarget_model_sha256",
        "retarget_root_translation_scale",
        "locomotion_prior_retarget_recipe",
        "locomotion_prior_source_foot_height_difference_m",
        "locomotion_prior_source_swing_baseline_estimator",
        "locomotion_prior_source_swing_baseline_m",
        "locomotion_prior_source_swing_deadband_m",
        "locomotion_prior_source_swing_side",
        "locomotion_prior_source_swing_scale_m",
        "locomotion_prior_swing_phase_filter",
        "locomotion_prior_sagittal_correction_joint_suffixes",
        "locomotion_prior_sagittal_correction_rad",
        "locomotion_prior_swing_target_boost_m",
    }
    with np.load(path, allow_pickle=False) as archive:
        archive_members = tuple(sorted(archive.files))
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"Locomotion prior is missing fields: {sorted(missing)}")
        fps = _scalar_float(np.asarray(archive["fps"]), "fps")
        joint_names_array = np.asarray(archive["joint_names"])
        joint_pos = np.asarray(archive["joint_pos"]).copy()
        body_names_array = np.asarray(archive["body_names"])
        body_pos = np.asarray(archive["body_pos_w"])
        body_quat = np.asarray(archive["body_quat_w"])
        source_session_id = _scalar_text(
            np.asarray(archive["source_session_id"]), "source_session_id"
        )
        source_schema = _scalar_text(
            np.asarray(archive["source_schema"]), "source_schema"
        )
        retarget_profile = _scalar_text(
            np.asarray(archive["retarget_profile"]), "retarget_profile"
        )
        trim_start_s = _scalar_float(
            np.asarray(archive["trim_start_s"]), "trim_start_s"
        )
        trim_end_s = _scalar_float(np.asarray(archive["trim_end_s"]), "trim_end_s")
        source_capture_sha256 = _scalar_sha256(
            np.asarray(archive["source_capture_sha256"]), "source_capture_sha256"
        )
        retarget_model_sha256 = _scalar_sha256(
            np.asarray(archive["retarget_model_sha256"]), "retarget_model_sha256"
        )
        retarget_root_translation_scale = _scalar_float(
            np.asarray(archive["retarget_root_translation_scale"]),
            "retarget_root_translation_scale",
        )
        retarget_recipe = _scalar_text(
            np.asarray(archive["locomotion_prior_retarget_recipe"]),
            "locomotion_prior_retarget_recipe",
        )
        source_swing_baseline_estimator = _scalar_text(
            np.asarray(archive["locomotion_prior_source_swing_baseline_estimator"]),
            "locomotion_prior_source_swing_baseline_estimator",
        )
        source_swing_baseline_m = _scalar_float(
            np.asarray(archive["locomotion_prior_source_swing_baseline_m"]),
            "locomotion_prior_source_swing_baseline_m",
        )
        source_swing_deadband_m = _scalar_float(
            np.asarray(archive["locomotion_prior_source_swing_deadband_m"]),
            "locomotion_prior_source_swing_deadband_m",
        )
        source_height_difference = np.asarray(
            archive["locomotion_prior_source_foot_height_difference_m"]
        ).copy()
        source_swing_side = np.asarray(
            archive["locomotion_prior_source_swing_side"]
        ).copy()
        source_swing_scale = np.asarray(
            archive["locomotion_prior_source_swing_scale_m"]
        ).copy()
        swing_phase_filter = np.asarray(
            archive["locomotion_prior_swing_phase_filter"]
        ).copy()
        correction_suffixes_array = np.asarray(
            archive["locomotion_prior_sagittal_correction_joint_suffixes"]
        )
        correction_rad = np.asarray(
            archive["locomotion_prior_sagittal_correction_rad"]
        ).copy()
        swing_target_boost_m = _scalar_float(
            np.asarray(archive["locomotion_prior_swing_target_boost_m"]),
            "locomotion_prior_swing_target_boost_m",
        )

        if fps != 50.0:
            raise ValueError("Locomotion prior must be sampled at exactly 50 Hz")
        if not source_session_id or not source_schema:
            raise ValueError("Locomotion prior source provenance must not be empty")
        if retarget_profile != "locomotion-prior":
            raise ValueError("Locomotion prior retarget_profile mismatch")
        if retarget_recipe != LOCOMOTION_PRIOR_RETARGET_RECIPE:
            raise ValueError("Locomotion prior retarget recipe mismatch")
        if source_swing_baseline_estimator != LOCOMOTION_PRIOR_SWING_BASELINE_ESTIMATOR:
            raise ValueError("Locomotion prior swing baseline estimator mismatch")
        if not math.isclose(
            source_swing_deadband_m,
            LOCOMOTION_PRIOR_SWING_DEADBAND_M,
            rel_tol=0.0,
            abs_tol=1.0e-7,
        ):
            raise ValueError("Locomotion prior swing deadband mismatch")
        if not math.isclose(
            swing_target_boost_m,
            LOCOMOTION_PRIOR_SWING_TARGET_BOOST_M,
            rel_tol=0.0,
            abs_tol=1.0e-7,
        ):
            raise ValueError("Locomotion prior swing target boost mismatch")
        if retarget_root_translation_scale <= 0.0:
            raise ValueError("Locomotion prior root translation scale must be positive")
        if trim_end_s < trim_start_s:
            raise ValueError("Locomotion prior trim interval is reversed")

        if joint_names_array.ndim != 1 or joint_names_array.dtype.kind not in (
            "U",
            "S",
        ):
            raise ValueError("Locomotion prior joint_names must be a string vector")
        if body_names_array.ndim != 1 or body_names_array.dtype.kind not in (
            "U",
            "S",
        ):
            raise ValueError("Locomotion prior body_names must be a string vector")
        joint_names = tuple(str(name) for name in joint_names_array.tolist())
        body_names = tuple(str(name) for name in body_names_array.tolist())
        if len(joint_names) != len(set(joint_names)):
            raise ValueError("Locomotion prior joint_names contains duplicates")
        if not body_names or body_names[0] != ROOT_BODY_NAME:
            raise ValueError(
                f"Locomotion prior body_names[0] must be {ROOT_BODY_NAME!r}"
            )
        if joint_pos.ndim != 2 or joint_pos.shape[1] != len(joint_names):
            raise ValueError("Locomotion prior joint_pos width must match joint_names")
        expected_body_prefix = (joint_pos.shape[0], len(body_names))
        if body_pos.shape != (*expected_body_prefix, 3):
            raise ValueError("Locomotion prior body_pos_w shape mismatch")
        if body_quat.shape != (*expected_body_prefix, 4):
            raise ValueError("Locomotion prior body_quat_w shape mismatch")
        if end_frame >= joint_pos.shape[0]:
            raise ValueError(
                f"Requested end frame {end_frame} is outside {joint_pos.shape[0]} frames"
            )

        frame_count = joint_pos.shape[0]
        if source_height_difference.shape != (frame_count,) or (
            source_height_difference.dtype.kind != "f"
            or not np.isfinite(source_height_difference).all()
        ):
            raise ValueError(
                "Locomotion prior source foot-height difference must be one finite "
                "float per frame"
            )
        if (
            source_swing_side.shape != (frame_count,)
            or source_swing_side.dtype.kind != "i"
        ):
            raise ValueError(
                "Locomotion prior source swing side must be one integer per frame"
            )
        if (
            source_swing_scale.shape != (2,)
            or source_swing_scale.dtype.kind != "f"
            or (not np.isfinite(source_swing_scale).all())
        ):
            raise ValueError("Locomotion prior source swing scale must be two floats")
        if (
            swing_phase_filter.shape != (5,)
            or swing_phase_filter.dtype.kind != "f"
            or (
                not np.allclose(
                    swing_phase_filter,
                    LOCOMOTION_PRIOR_SWING_PHASE_FILTER,
                    rtol=0.0,
                    atol=1.0e-8,
                )
            )
        ):
            raise ValueError("Locomotion prior swing phase filter mismatch")
        if correction_suffixes_array.shape != (3,) or (
            correction_suffixes_array.dtype.kind not in ("U", "S")
        ):
            raise ValueError("Locomotion prior correction suffixes must be strings")
        correction_suffixes = tuple(str(value) for value in correction_suffixes_array)
        if correction_suffixes != LOCOMOTION_PRIOR_SAGITTAL_CORRECTION_JOINT_SUFFIXES:
            raise ValueError("Locomotion prior correction joint suffixes mismatch")
        if (
            correction_rad.shape != (3,)
            or correction_rad.dtype.kind != "f"
            or (
                not np.allclose(
                    correction_rad,
                    LOCOMOTION_PRIOR_SAGITTAL_CORRECTION_RAD,
                    rtol=0.0,
                    atol=1.0e-8,
                )
            )
        ):
            raise ValueError("Locomotion prior sagittal correction mismatch")

        height64 = source_height_difference.astype(np.float64)
        low, high = np.percentile(height64, (5.0, 95.0), method="linear")
        expected_baseline = float(0.5 * (low + high))
        if not math.isclose(
            source_swing_baseline_m,
            expected_baseline,
            rel_tol=0.0,
            abs_tol=2.0e-7,
        ):
            raise ValueError("Locomotion prior source swing baseline is inconsistent")
        centered = height64 - expected_baseline
        expected_side = np.zeros(frame_count, dtype=np.int8)
        expected_side[centered > LOCOMOTION_PRIOR_SWING_DEADBAND_M] = 1
        expected_side[centered < -LOCOMOTION_PRIOR_SWING_DEADBAND_M] = -1
        if not np.array_equal(source_swing_side, expected_side):
            raise ValueError("Locomotion prior source swing side is inconsistent")
        expected_scales = np.asarray(
            [
                np.percentile(
                    signal[signal > LOCOMOTION_PRIOR_SWING_DEADBAND_M],
                    95.0,
                    method="linear",
                )
                for signal in (centered, -centered)
            ]
        )
        if not np.allclose(source_swing_scale, expected_scales, rtol=0.0, atol=2.0e-7):
            raise ValueError("Locomotion prior source swing scale is inconsistent")
        non_neutral_side = expected_side[expected_side != 0]
        transitions = int(np.sum(non_neutral_side[1:] != non_neutral_side[:-1]))

        used = slice(start_frame, end_frame + 1)
        used_arrays = (joint_pos[used], body_pos[used, 0], body_quat[used, 0])
        if any(array.dtype.kind != "f" for array in used_arrays):
            raise ValueError("Locomotion prior pose arrays must be floating point")
        if any(not np.isfinite(array).all() for array in used_arrays):
            raise ValueError("Locomotion prior pose arrays contain non-finite values")
        quaternion_norms = np.linalg.norm(body_quat[used, 0].astype(np.float64), axis=1)
        if not np.allclose(quaternion_norms, 1.0, rtol=0.0, atol=1.0e-6):
            raise ValueError("Locomotion prior root quaternions are not unit length")

        return _PriorPoseArrays(
            fps=fps,
            joint_names=joint_names,
            joint_pos=joint_pos,
            root_pos=body_pos[:, 0].copy(),
            root_quat=body_quat[:, 0].copy(),
            body_names=body_names,
            source_session_id=source_session_id,
            source_schema=source_schema,
            retarget_profile=retarget_profile,
            trim_start_s=trim_start_s,
            trim_end_s=trim_end_s,
            source_capture_sha256=source_capture_sha256,
            retarget_model_sha256=retarget_model_sha256,
            retarget_root_translation_scale=retarget_root_translation_scale,
            retarget_recipe=retarget_recipe,
            source_swing_baseline_estimator=source_swing_baseline_estimator,
            source_swing_baseline_m=source_swing_baseline_m,
            source_swing_deadband_m=source_swing_deadband_m,
            source_swing_scale_m=tuple(float(value) for value in source_swing_scale),
            source_right_swing_frame_count=int(np.sum(expected_side == 1)),
            source_left_swing_frame_count=int(np.sum(expected_side == -1)),
            source_swing_transition_count=transitions,
            source_foot_height_difference_sha256=hashlib.sha256(
                np.ascontiguousarray(source_height_difference).tobytes()
            ).hexdigest(),
            swing_phase_filter=tuple(float(value) for value in swing_phase_filter),
            sagittal_correction_joint_suffixes=correction_suffixes,
            sagittal_correction_rad=tuple(float(value) for value in correction_rad),
            swing_target_boost_m=swing_target_boost_m,
            archive_members=archive_members,
        )


def _validate_frame_range(start_frame: int, end_frame: int) -> None:
    if (
        isinstance(start_frame, bool)
        or not isinstance(start_frame, int)
        or isinstance(end_frame, bool)
        or not isinstance(end_frame, int)
    ):
        raise TypeError("Frame bounds must be integers")
    if start_frame < 0 or end_frame < start_frame:
        raise ValueError("Frame range must be non-negative and inclusive")


def _model_joint_qpos_addresses(
    model: mujoco.MjModel, joint_names: tuple[str, ...]
) -> tuple[int, tuple[int, ...]]:
    one_coordinate_types = {
        int(mujoco.mjtJoint.mjJNT_HINGE),
        int(mujoco.mjtJoint.mjJNT_SLIDE),
    }
    root_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, ROOT_FREEJOINT_NAME
    )
    if root_joint_id < 0:
        raise ValueError(f"Robot XML is missing freejoint {ROOT_FREEJOINT_NAME!r}")
    if model.jnt_type[root_joint_id] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(f"{ROOT_FREEJOINT_NAME!r} must be a freejoint")
    root_body_id = int(model.jnt_bodyid[root_joint_id])
    root_body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, root_body_id)
    if root_body_name != ROOT_BODY_NAME:
        raise ValueError(
            f"{ROOT_FREEJOINT_NAME!r} must belong to body {ROOT_BODY_NAME!r}"
        )
    root_qpos_address = int(model.jnt_qposadr[root_joint_id])

    addresses: list[int] = []
    resolved_ids: set[int] = set()
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Robot XML is missing prior joint {name!r}")
        if int(model.jnt_type[joint_id]) not in one_coordinate_types:
            raise ValueError(f"Prior joint {name!r} must have one qpos coordinate")
        resolved_ids.add(joint_id)
        addresses.append(int(model.jnt_qposadr[joint_id]))

    expected_ids = {
        joint_id
        for joint_id in range(model.njnt)
        if joint_id != root_joint_id
        and int(model.jnt_type[joint_id]) in one_coordinate_types
    }
    if resolved_ids != expected_ids:
        missing_names = sorted(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            for joint_id in expected_ids.difference(resolved_ids)
        )
        raise ValueError(
            "Locomotion prior must provide every one-coordinate robot joint; "
            f"missing={missing_names}"
        )
    expected_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in range(model.njnt)
        if joint_id in expected_ids
    )
    if joint_names != expected_names:
        raise ValueError("Locomotion prior joint order does not match the robot XML")
    return root_qpos_address, tuple(addresses)


def _foot_geom_ids(model: mujoco.MjModel) -> dict[str, tuple[int, ...]]:
    result: dict[str, tuple[int, ...]] = {}
    all_ids: set[int] = set()
    for side, names in FOOT_COLLISION_GEOM_NAMES.items():
        ids: list[int] = []
        for name in names:
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom_id < 0:
                raise ValueError(f"Robot XML is missing foot collision geom {name!r}")
            if geom_id in all_ids:
                raise ValueError(f"Foot collision geom {name!r} is not unique")
            if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_BOX:
                raise ValueError(f"Foot collision geom {name!r} must be a box")
            if not np.all(np.isfinite(model.geom_size[geom_id])) or np.any(
                model.geom_size[geom_id] <= 0.0
            ):
                raise ValueError(f"Foot collision geom {name!r} has invalid half-sizes")
            ids.append(geom_id)
            all_ids.add(geom_id)
        result[side] = tuple(ids)
    if len(all_ids) != 12:
        raise ValueError(
            "Exactly twelve unique numbered foot collision boxes are required"
        )
    return result


def _sole_min_world_z(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_ids: tuple[int, ...],
) -> float:
    minimum = math.inf
    for geom_id in geom_ids:
        local_corners = _BOX_CORNER_SIGNS * model.geom_size[geom_id]
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        world_corners = data.geom_xpos[geom_id] + local_corners @ rotation.T
        minimum = min(minimum, float(np.min(world_corners[:, 2])))
    if not math.isfinite(minimum):
        raise ValueError("Foot box forward kinematics produced a non-finite height")
    return minimum


def _json_float(value: float) -> float:
    """Remove insignificant platform noise while retaining sub-micrometre detail."""

    if not math.isfinite(value):
        raise ValueError("Receipt metrics must be finite")
    return float(format(value, ".12g"))


def _percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile, method="linear"))


def audit_locomotion_prior_suitability(
    prior_path: Path = DEFAULT_LOCOMOTION_PRIOR_PATH,
    robot_xml_path: Path = DEFAULT_ROBOT_XML_PATH,
    *,
    expected_prior_sha256: str | None = DEFAULT_LOCOMOTION_PRIOR_SHA256,
    expected_robot_xml_sha256: str | None = DEFAULT_ROBOT_XML_SHA256,
    start_frame: int = DEFAULT_START_FRAME,
    end_frame: int = DEFAULT_END_FRAME,
    thresholds: LocomotionPriorSuitabilityThresholds = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """Build a deterministic static-suitability receipt for one motion clip."""

    _validate_frame_range(start_frame, end_frame)
    thresholds.validate()
    prior_path = prior_path.expanduser().resolve()
    robot_xml_path = robot_xml_path.expanduser().resolve()
    if not prior_path.is_file():
        raise FileNotFoundError(f"Locomotion prior does not exist: {prior_path}")
    if not robot_xml_path.is_file():
        raise FileNotFoundError(f"Robot XML does not exist: {robot_xml_path}")

    prior_sha256 = sha256_file(prior_path)
    robot_xml_sha256 = sha256_file(robot_xml_path)
    _validate_expected_sha256("locomotion prior", prior_sha256, expected_prior_sha256)
    _validate_expected_sha256("robot XML", robot_xml_sha256, expected_robot_xml_sha256)

    arrays = _load_prior_pose_arrays(
        prior_path, start_frame=start_frame, end_frame=end_frame
    )
    if arrays.retarget_model_sha256 != robot_xml_sha256:
        raise ValueError(
            "Locomotion prior retarget_model_sha256 does not match robot XML"
        )
    model = mujoco.MjModel.from_xml_path(str(robot_xml_path))
    expected_body_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        for body_id in range(1, model.nbody)
    )
    if arrays.body_names != expected_body_names:
        raise ValueError("Locomotion prior body order does not match the robot XML")
    data = mujoco.MjData(model)
    root_qpos_address, joint_qpos_addresses = _model_joint_qpos_addresses(
        model, arrays.joint_names
    )
    geom_ids = _foot_geom_ids(model)

    frame_reports: list[dict[str, Any]] = []
    grounding_errors: list[float] = []
    grounding_translations: list[float] = []
    clearances: list[float] = []
    grounded_clearance_by_foot: dict[str, list[float]] = {
        "left": [],
        "right": [],
    }
    for frame in range(start_frame, end_frame + 1):
        data.qpos[:] = model.qpos0
        data.qvel[:] = 0.0
        data.qpos[root_qpos_address : root_qpos_address + 3] = arrays.root_pos[frame]
        data.qpos[root_qpos_address + 3 : root_qpos_address + 7] = arrays.root_quat[
            frame
        ]
        for source_index, qpos_address in enumerate(joint_qpos_addresses):
            data.qpos[qpos_address] = arrays.joint_pos[frame, source_index]
        mujoco.mj_forward(model, data)

        left_height = _sole_min_world_z(model, data, geom_ids["left"])
        right_height = _sole_min_world_z(model, data, geom_ids["right"])
        lower_height = min(left_height, right_height)
        grounding_error = abs(lower_height)
        grounding_translation = -lower_height
        clearance = abs(left_height - right_height)
        support_foot = "left" if left_height <= right_height else "right"
        swing_foot = "right" if support_foot == "left" else "left"

        grounding_errors.append(grounding_error)
        grounding_translations.append(grounding_translation)
        clearances.append(clearance)
        grounded_clearance_by_foot["left"].append(left_height - lower_height)
        grounded_clearance_by_foot["right"].append(right_height - lower_height)
        frame_reports.append(
            {
                "frame": frame,
                "grounded_left_sole_height_m": _json_float(left_height - lower_height),
                "grounded_right_sole_height_m": _json_float(
                    right_height - lower_height
                ),
                "grounding_error_m": _json_float(grounding_error),
                "grounding_passed": (
                    grounding_error <= thresholds.grounding_error_m_max
                ),
                "grounding_translation_m": _json_float(grounding_translation),
                "left_sole_min_world_z_m": _json_float(left_height),
                "lower_sole_min_world_z_m": _json_float(lower_height),
                "relative_swing_clearance_m": _json_float(clearance),
                "right_sole_min_world_z_m": _json_float(right_height),
                "support_foot": support_foot,
                "swing_foot": swing_foot,
            }
        )

    grounding_array = np.asarray(grounding_errors, dtype=np.float64)
    translation_array = np.asarray(grounding_translations, dtype=np.float64)
    clearance_array = np.asarray(clearances, dtype=np.float64)
    grounding_max = float(np.max(grounding_array))
    clearance_max = float(np.max(clearance_array))
    clearance_p95 = _percentile(clearance_array, 95.0)
    grounding_max_frame = start_frame + int(np.argmax(grounding_array))
    clearance_max_frame = start_frame + int(np.argmax(clearance_array))
    per_foot_clearance = {}
    for side, values in grounded_clearance_by_foot.items():
        side_array = np.asarray(values, dtype=np.float64)
        positive = side_array[side_array > 0.0]
        per_foot_clearance[side] = {
            "frames_higher_than_other_foot": int(positive.size),
            "max_m": _json_float(float(np.max(side_array))),
            "p95_all_frames_m": _json_float(_percentile(side_array, 95.0)),
            "p95_while_higher_m": (
                _json_float(_percentile(positive, 95.0)) if positive.size else None
            ),
        }
    checks = {
        "grounding_error_m_max": {
            "maximum": thresholds.grounding_error_m_max,
            "passed": grounding_max <= thresholds.grounding_error_m_max,
            "value": _json_float(grounding_max),
        },
        "relative_swing_clearance_max_m": {
            "minimum": thresholds.relative_swing_clearance_max_m_min,
            "passed": (clearance_max >= thresholds.relative_swing_clearance_max_m_min),
            "value": _json_float(clearance_max),
        },
        "relative_swing_clearance_p95_m": {
            "minimum": thresholds.relative_swing_clearance_p95_m_min,
            "passed": (clearance_p95 >= thresholds.relative_swing_clearance_p95_m_min),
            "value": _json_float(clearance_p95),
        },
    }
    passed = all(check["passed"] for check in checks.values())

    payload: dict[str, Any] = {
        "aggregate": {
            "checks": checks,
            "frame_count": len(frame_reports),
            "grounding": {
                "error_max_frame": grounding_max_frame,
                "error_max_m": _json_float(grounding_max),
                "error_mean_m": _json_float(float(np.mean(grounding_array))),
                "error_p95_m": _json_float(_percentile(grounding_array, 95.0)),
                "translation_max_m": _json_float(float(np.max(translation_array))),
                "translation_min_m": _json_float(float(np.min(translation_array))),
            },
            "relative_swing_clearance": {
                "max_frame": clearance_max_frame,
                "max_m": _json_float(clearance_max),
                "mean_m": _json_float(float(np.mean(clearance_array))),
                "per_foot_when_higher": per_foot_clearance,
                "p95_m": _json_float(clearance_p95),
            },
        },
        "audit_revision": LOCOMOTION_PRIOR_SUITABILITY_REVISION,
        "configuration": {
            "end_frame_inclusive": end_frame,
            "foot_collision_geoms": {
                side: list(names) for side, names in FOOT_COLLISION_GEOM_NAMES.items()
            },
            "start_frame_inclusive": start_frame,
            "thresholds": asdict(thresholds),
        },
        "frames": frame_reports,
        "inputs": {
            "locomotion_prior": {
                "archive_members": list(arrays.archive_members),
                "expected_sha256": expected_prior_sha256,
                "path": _portable_input_path(prior_path),
                "sha256": prior_sha256,
                "size_bytes": prior_path.stat().st_size,
                "source": {
                    "body_names": list(arrays.body_names),
                    "fps": arrays.fps,
                    "joint_names": list(arrays.joint_names),
                    "retarget_profile": arrays.retarget_profile,
                    "schema": arrays.source_schema,
                    "session_id": arrays.source_session_id,
                    "total_frames": int(arrays.joint_pos.shape[0]),
                    "trim_end_s": arrays.trim_end_s,
                    "trim_start_s": arrays.trim_start_s,
                },
                "provenance": {
                    "retarget_model_sha256": arrays.retarget_model_sha256,
                    "retarget_recipe": arrays.retarget_recipe,
                    "retarget_root_translation_scale": (
                        arrays.retarget_root_translation_scale
                    ),
                    "sagittal_correction": {
                        "joint_suffixes": list(
                            arrays.sagittal_correction_joint_suffixes
                        ),
                        "radians": list(arrays.sagittal_correction_rad),
                    },
                    "source_capture_sha256": arrays.source_capture_sha256,
                    "source_swing": {
                        "baseline_estimator": (arrays.source_swing_baseline_estimator),
                        "baseline_m": arrays.source_swing_baseline_m,
                        "deadband_m": arrays.source_swing_deadband_m,
                        "foot_height_difference_sha256": (
                            arrays.source_foot_height_difference_sha256
                        ),
                        "left_frame_count": arrays.source_left_swing_frame_count,
                        "right_frame_count": arrays.source_right_swing_frame_count,
                        "scale_m": list(arrays.source_swing_scale_m),
                        "transition_count": arrays.source_swing_transition_count,
                    },
                    "swing_phase_filter": list(arrays.swing_phase_filter),
                    "swing_target_boost_m": arrays.swing_target_boost_m,
                },
            },
            "robot_xml": {
                "expected_sha256": expected_robot_xml_sha256,
                "path": _portable_input_path(robot_xml_path),
                "sha256": robot_xml_sha256,
                "size_bytes": robot_xml_path.stat().st_size,
            },
        },
        "method": {
            "box_corner_count_per_geom": 8,
            "execution": "cpu_kinematics_only_no_physics_step",
            "forward_kinematics": "mujoco.mj_forward",
            "grounding_error": ("absolute world-z of the lower of the two sole minima"),
            "grounding_normalization": (
                "translate both soles by minus the lower sole minimum"
            ),
            "percentile": "numpy_linear",
            "relative_swing_clearance": (
                "absolute difference between left and right sole minimum world-z"
            ),
            "sole_height": (
                "minimum world-z over all transformed corners of all six "
                "numbered collision boxes for that foot"
            ),
        },
        "schema_version": LOCOMOTION_PRIOR_SUITABILITY_SCHEMA_VERSION,
        "status": "pass" if passed else "fail",
        "summary": {
            "failed_checks": sorted(
                name for name, check in checks.items() if not check["passed"]
            ),
            "passed": passed,
            "static_only": True,
        },
        "tool_versions": {
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
        },
    }
    payload_sha256 = canonical_json_sha256(payload)
    return {**payload, "receipt_payload_sha256": payload_sha256}


def validate_suitability_receipt_digest(report: dict[str, Any]) -> None:
    """Fail if a receipt's canonical payload hash is missing or stale."""

    recorded = report.get("receipt_payload_sha256")
    if not isinstance(recorded, str) or _SHA256_RE.fullmatch(recorded) is None:
        raise ValueError("Suitability receipt payload SHA-256 is missing or invalid")
    payload = dict(report)
    del payload["receipt_payload_sha256"]
    observed = canonical_json_sha256(payload)
    if observed != recorded:
        raise ValueError("Suitability receipt payload SHA-256 mismatch")
