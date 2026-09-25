"""Fail-closed live PICO reference bridge for the 99-value tracking actor.

The trained tracking actor does not consume the 24-joint PICO skeleton
directly.  Its first 42 values are a Microban reference: absolute position and
velocity for the 18 body-policy joints followed by the desired trunk
orientation relative to the measured trunk.  This module converts one already
validated XRoboToolkit/Unity ``TrackingFrame`` into exactly that contract.

The bridge is deliberately simulation-only.  It has no network or motor
output.  HMD-owned ``head``, ``neck_roll`` and ``neck_pitch`` joints are not
part of the IK solve or the actor action.  A released deadman maps to the robot
home reference; live IK becomes available only after a fresh, released-trigger
calibration frame has been observed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch

from mjlab_microban.robot.microban_constants import HOME_FRAME, MICROBAN_XML
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_TELEOP_ACTION_JOINT_NAMES,
)
from mjlab_microban.tasks.microban_tracking_env_cfg import (
    MICROBAN_BODY_JOINT_SOFT_LIMITS,
    MICROBAN_BODY_NAMES,
    MICROBAN_JOINT_NAMES,
)

LIVE_TRACKING_ACTOR_SCHEMA: tuple[tuple[str, int], ...] = (
    ("command", 36),
    ("motion_anchor_ori_b", 6),
    ("base_ang_vel", 3),
    ("joint_pos", 18),
    ("joint_vel", 18),
    ("actions", 18),
)
LIVE_TRACKING_ACTOR_WIDTH = sum(width for _, width in LIVE_TRACKING_ACTOR_SCHEMA)
LIVE_TRACKING_REFERENCE_WIDTH = 42

PICO_BODY_JOINT_NAMES: tuple[str, ...] = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine_1",
    "left_knee",
    "right_knee",
    "spine_2",
    "left_ankle",
    "right_ankle",
    "spine_3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
)

_PICO_INDEX = {name: index for index, name in enumerate(PICO_BODY_JOINT_NAMES)}
_POLICY_LIMITS = np.asarray(
    [
        MICROBAN_BODY_JOINT_SOFT_LIMITS[name]
        for name in MICROBAN_TELEOP_ACTION_JOINT_NAMES
    ],
    dtype=np.float64,
)


class LiveTrackingBridgeError(ValueError):
    """A live reference cannot safely be produced from the current frame."""


@dataclass(frozen=True, slots=True)
class PicoBodySample:
    """Validated named body sample in Microban's forward/left/up frame."""

    timestamp_ns: int
    positions_m: np.ndarray
    pelvis_orientation_xyzw: np.ndarray


@dataclass(frozen=True, slots=True)
class LiveTrackingReference:
    """Physical reference consumed by the first 42 tracking observations."""

    joint_pos_rad: tuple[float, ...]
    joint_vel_rad_s: tuple[float, ...]
    desired_trunk_quat_wxyz: tuple[float, float, float, float]
    source_timestamp_ns: int
    ik_error_m: float


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _finite_vector(value: Any, width: int, name: str) -> np.ndarray:
    if isinstance(value, (str, bytes, bytearray)):
        raise LiveTrackingBridgeError(f"{name} must contain {width} numbers")
    try:
        result = np.asarray(tuple(value), dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise LiveTrackingBridgeError(
            f"{name} must contain {width} finite numbers"
        ) from exc
    if result.shape != (width,) or not np.all(np.isfinite(result)):
        raise LiveTrackingBridgeError(f"{name} must contain {width} finite numbers")
    return result


def _normalized_xyzw(value: Any, name: str) -> np.ndarray:
    result = _finite_vector(value, 4, name)
    norm = float(np.linalg.norm(result))
    if norm < 1.0e-6:
        raise LiveTrackingBridgeError(f"{name} has zero length")
    return result / norm


def quaternion_xyzw_to_matrix(value: Any, name: str = "quaternion") -> np.ndarray:
    """Return a proper rotation matrix for a finite XYZW quaternion."""

    x, y, z, w = _normalized_xyzw(value, name)
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def quaternion_wxyz_to_matrix(value: Any, name: str = "quaternion") -> np.ndarray:
    """Return a proper rotation matrix for a finite WXYZ quaternion."""

    wxyz = _finite_vector(value, 4, name)
    return quaternion_xyzw_to_matrix((wxyz[1], wxyz[2], wxyz[3], wxyz[0]), name)


def matrix_to_quaternion_wxyz(matrix: Any) -> tuple[float, float, float, float]:
    """Convert a finite proper 3x3 rotation to normalized WXYZ."""

    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (3, 3) or not np.all(np.isfinite(value)):
        raise LiveTrackingBridgeError("rotation matrix must be finite 3x3")
    if not np.allclose(value.T @ value, np.eye(3), atol=2.0e-5, rtol=0.0):
        raise LiveTrackingBridgeError("rotation matrix is not orthonormal")
    if not math.isclose(float(np.linalg.det(value)), 1.0, abs_tol=2.0e-5):
        raise LiveTrackingBridgeError("rotation matrix is not proper")

    m00, m01, m02 = value[0]
    m10, m11, m12 = value[1]
    m20, m21, m22 = value[2]
    trace = float(m00 + m11 + m22)
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        result = np.asarray(
            (
                0.25 * scale,
                (m21 - m12) / scale,
                (m02 - m20) / scale,
                (m10 - m01) / scale,
            )
        )
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(max(0.0, 1.0 + m00 - m11 - m22)) * 2.0
        result = np.asarray(
            (
                (m21 - m12) / scale,
                0.25 * scale,
                (m01 + m10) / scale,
                (m02 + m20) / scale,
            )
        )
    elif m11 > m22:
        scale = math.sqrt(max(0.0, 1.0 + m11 - m00 - m22)) * 2.0
        result = np.asarray(
            (
                (m02 - m20) / scale,
                (m01 + m10) / scale,
                0.25 * scale,
                (m12 + m21) / scale,
            )
        )
    else:
        scale = math.sqrt(max(0.0, 1.0 + m22 - m00 - m11)) * 2.0
        result = np.asarray(
            (
                (m10 - m01) / scale,
                (m02 + m20) / scale,
                (m12 + m21) / scale,
                0.25 * scale,
            )
        )
    norm = float(np.linalg.norm(result))
    if norm < 1.0e-9 or not np.all(np.isfinite(result)):
        raise LiveTrackingBridgeError("rotation produced an invalid quaternion")
    result /= norm
    if result[0] < 0.0:
        result *= -1.0
    return tuple(float(item) for item in result)


def extract_pico_body_sample(frame: Any) -> PicoBodySample:
    """Validate freshness, exact joint order and the live body pose payload."""

    health = _field(frame, "body_health")
    if health is not None:
        if _field(health, "valid", False) is not True:
            raise LiveTrackingBridgeError("PICO body frame is invalid")
        if _field(health, "fresh", False) is not True:
            raise LiveTrackingBridgeError("PICO body frame is stale")
    elif _field(frame, "body_fresh", False) is not True:
        raise LiveTrackingBridgeError("PICO body freshness is unavailable")

    jumps = _field(frame, "body_jumps", ())
    if jumps:
        raise LiveTrackingBridgeError("PICO body frame contains a pose jump")
    body = _field(frame, "body")
    if body is None:
        raise LiveTrackingBridgeError("PICO body tracking is unavailable")
    timestamp = _field(body, "timestamp_ns")
    if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp <= 0:
        raise LiveTrackingBridgeError("PICO body timestamp must be positive")
    joints_value = _field(body, "joints")
    if isinstance(joints_value, (str, bytes, bytearray)):
        raise LiveTrackingBridgeError("PICO body joints are malformed")
    try:
        joints = tuple(joints_value)
    except TypeError as exc:
        raise LiveTrackingBridgeError("PICO body joints are malformed") from exc
    if len(joints) != len(PICO_BODY_JOINT_NAMES):
        raise LiveTrackingBridgeError("PICO body must contain exactly 24 joints")

    positions = np.empty((len(joints), 3), dtype=np.float64)
    pelvis_orientation: np.ndarray | None = None
    for index, (joint, expected_name) in enumerate(
        zip(joints, PICO_BODY_JOINT_NAMES, strict=True)
    ):
        if (
            _field(joint, "index", index) != index
            or _field(joint, "name") != expected_name
        ):
            raise LiveTrackingBridgeError(
                f"PICO body joint {index} must be {expected_name!r}"
            )
        pose = _field(joint, "pose")
        if pose is None:
            raise LiveTrackingBridgeError(f"{expected_name}.pose is missing")
        positions[index] = _finite_vector(
            _field(pose, "position"), 3, f"{expected_name}.position"
        )
        orientation = _normalized_xyzw(
            _field(pose, "orientation"), f"{expected_name}.orientation"
        )
        if expected_name == "pelvis":
            pelvis_orientation = orientation
    assert pelvis_orientation is not None
    return PicoBodySample(timestamp, positions, pelvis_orientation)


def _model_names(
    model: mujoco.MjModel, object_type: mujoco.mjtObj, count: int
) -> tuple[str, ...]:
    return tuple(
        mujoco.mj_id2name(model, object_type, index) or "" for index in range(count)
    )


def _safe_direction(value: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    return fallback.copy() if norm < 1.0e-5 else value / norm


class OnlineMicrobanRetargeter:
    """Warm-started DLS position IK matching the reviewed offline retargeter."""

    def __init__(
        self,
        model_path: str | Path = MICROBAN_XML,
        *,
        damping: float = 2.0e-4,
        regularization: float = 1.0e-3,
        max_iterations: int = 30,
    ) -> None:
        if not math.isfinite(damping) or damping <= 0.0:
            raise ValueError("damping must be finite and positive")
        if not math.isfinite(regularization) or regularization <= 0.0:
            raise ValueError("regularization must be finite and positive")
        if (
            isinstance(max_iterations, bool)
            or not isinstance(max_iterations, int)
            or max_iterations <= 0
        ):
            raise ValueError("max_iterations must be a positive integer")
        self.model = mujoco.MjModel.from_xml_path(str(Path(model_path).resolve()))
        self.data = mujoco.MjData(self.model)
        joints = _model_names(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.model.njnt)[
            1:
        ]
        bodies = _model_names(self.model, mujoco.mjtObj.mjOBJ_BODY, self.model.nbody)[
            1:
        ]
        if joints != MICROBAN_JOINT_NAMES or bodies != MICROBAN_BODY_NAMES:
            raise LiveTrackingBridgeError(
                "Microban model order does not match tracking policy"
            )
        self.damping = damping
        self.regularization = regularization
        self.max_iterations = max_iterations
        self.last_weighted_rms_error_m = math.inf
        self.home_qpos = self.model.qpos0.copy()
        self.home_qpos[:3] = np.asarray(HOME_FRAME.pos, dtype=np.float64)
        for name, value in (HOME_FRAME.joint_pos or {}).items():
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise LiveTrackingBridgeError(
                    f"Microban home pose references unknown joint {name!r}"
                )
            self.home_qpos[self.model.jnt_qposadr[joint_id]] = float(value)
        self.qpos = self.home_qpos.copy()
        self.active_q = np.asarray(
            [
                self.model.jnt_qposadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
                for name in MICROBAN_TELEOP_ACTION_JOINT_NAMES
            ],
            dtype=np.int32,
        )
        self.active_v = np.asarray(
            [
                self.model.jnt_dofadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
                for name in MICROBAN_TELEOP_ACTION_JOINT_NAMES
            ],
            dtype=np.int32,
        )
        self.neck_q = np.asarray(
            [
                self.model.jnt_qposadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
                for name in MICROBAN_JOINT_NAMES[:3]
            ],
            dtype=np.int32,
        )
        self.body_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in (
                "shoulder",
                "radius",
                "hip",
                "tibia__configuration_right",
                "shoulder_2",
                "radius_2",
                "hip_2",
                "tibia__configuration_left",
            )
        }
        self.site_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in ("right_hand", "right_foot", "left_hand", "left_foot")
        }
        if min((*self.body_ids.values(), *self.site_ids.values())) < 0:
            raise LiveTrackingBridgeError("Microban model lacks a live IK body/site")
        self._forward()
        self.chain_lengths = self._measure_chain_lengths()

    @property
    def home_action_joint_pos(self) -> np.ndarray:
        return self.home_qpos[self.active_q].copy()

    def reset(self) -> None:
        self.qpos[:] = self.home_qpos
        self.last_weighted_rms_error_m = math.inf
        self._forward()

    def _forward(self) -> None:
        self.data.qpos[:] = self.qpos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _measure_chain_lengths(self) -> dict[str, tuple[float, float]]:
        result: dict[str, tuple[float, float]] = {}
        for side, shoulder, elbow, hand, hip, knee, foot in (
            (
                "right",
                "shoulder",
                "radius",
                "right_hand",
                "hip",
                "tibia__configuration_right",
                "right_foot",
            ),
            (
                "left",
                "shoulder_2",
                "radius_2",
                "left_hand",
                "hip_2",
                "tibia__configuration_left",
                "left_foot",
            ),
        ):
            result[f"{side}_arm"] = (
                float(
                    np.linalg.norm(
                        self.data.xpos[self.body_ids[elbow]]
                        - self.data.xpos[self.body_ids[shoulder]]
                    )
                ),
                float(
                    np.linalg.norm(
                        self.data.site_xpos[self.site_ids[hand]]
                        - self.data.xpos[self.body_ids[elbow]]
                    )
                ),
            )
            result[f"{side}_leg"] = (
                float(
                    np.linalg.norm(
                        self.data.xpos[self.body_ids[knee]]
                        - self.data.xpos[self.body_ids[hip]]
                    )
                ),
                float(
                    np.linalg.norm(
                        self.data.site_xpos[self.site_ids[foot]]
                        - self.data.xpos[self.body_ids[knee]]
                    )
                ),
            )
        if min(component for pair in result.values() for component in pair) < 1.0e-4:
            raise LiveTrackingBridgeError(
                "Microban model contains a degenerate IK chain"
            )
        return result

    def _targets(
        self, human: np.ndarray, alignment: np.ndarray
    ) -> list[tuple[str, int, np.ndarray, float]]:
        result: list[tuple[str, int, np.ndarray, float]] = []
        for (
            side,
            shoulder_body,
            elbow_body,
            hand_site,
            hip_body,
            knee_body,
            foot_site,
        ) in (
            (
                "right",
                "shoulder",
                "radius",
                "right_hand",
                "hip",
                "tibia__configuration_right",
                "right_foot",
            ),
            (
                "left",
                "shoulder_2",
                "radius_2",
                "left_hand",
                "hip_2",
                "tibia__configuration_left",
                "left_foot",
            ),
        ):
            shoulder = human[_PICO_INDEX[f"{side}_shoulder"]]
            elbow = human[_PICO_INDEX[f"{side}_elbow"]]
            hand = human[_PICO_INDEX[f"{side}_hand"]]
            hip = human[_PICO_INDEX[f"{side}_hip"]]
            knee = human[_PICO_INDEX[f"{side}_knee"]]
            foot = human[_PICO_INDEX[f"{side}_foot"]]
            upper_arm, lower_arm = self.chain_lengths[f"{side}_arm"]
            thigh, shank = self.chain_lengths[f"{side}_leg"]
            lateral = 1.0 if side == "left" else -1.0
            upper_arm_direction = _safe_direction(
                alignment @ (elbow - shoulder), np.asarray((0.0, lateral, -0.2))
            )
            lower_arm_direction = _safe_direction(
                alignment @ (hand - elbow), upper_arm_direction
            )
            thigh_direction = _safe_direction(
                alignment @ (knee - hip), np.asarray((0.0, 0.0, -1.0))
            )
            shank_direction = _safe_direction(
                alignment @ (foot - knee), thigh_direction
            )
            elbow_target = (
                self.data.xpos[self.body_ids[shoulder_body]]
                + upper_arm * upper_arm_direction
            )
            hand_target = elbow_target + lower_arm * lower_arm_direction
            knee_target = (
                self.data.xpos[self.body_ids[hip_body]] + thigh * thigh_direction
            )
            foot_target = knee_target + shank * shank_direction
            result.extend(
                (
                    ("body", self.body_ids[elbow_body], elbow_target, 0.65),
                    ("site", self.site_ids[hand_site], hand_target, 1.0),
                    ("body", self.body_ids[knee_body], knee_target, 0.8),
                    ("site", self.site_ids[foot_site], foot_target, 1.25),
                )
            )
        return result

    def _position_jacobian(
        self, kind: str, index: int
    ) -> tuple[np.ndarray, np.ndarray]:
        jacobian_pos = np.zeros((3, self.model.nv), dtype=np.float64)
        jacobian_rot = np.zeros((3, self.model.nv), dtype=np.float64)
        if kind == "body":
            mujoco.mj_jacBody(self.model, self.data, jacobian_pos, jacobian_rot, index)
            position = self.data.xpos[index]
        else:
            mujoco.mj_jacSite(self.model, self.data, jacobian_pos, jacobian_rot, index)
            position = self.data.site_xpos[index]
        return position.copy(), jacobian_pos[:, self.active_v]

    def solve(
        self,
        positions_m: np.ndarray,
        *,
        alignment: np.ndarray,
        desired_trunk_matrix: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Return joint positions and the final maximum endpoint error.

        The weighted scalar RMS remains available as
        :attr:`last_weighted_rms_error_m` for diagnostics.  Safety decisions use
        the maximum three-dimensional endpoint error so one bad hand or foot
        cannot be hidden by averaging it with the other seven endpoints.
        """

        human = np.asarray(positions_m, dtype=np.float64)
        if human.shape != (24, 3) or not np.all(np.isfinite(human)):
            raise LiveTrackingBridgeError("live IK positions must be finite [24,3]")
        alignment = np.asarray(alignment, dtype=np.float64)
        desired_trunk_matrix = np.asarray(desired_trunk_matrix, dtype=np.float64)
        if alignment.shape != (3, 3):
            raise LiveTrackingBridgeError("live IK alignment must be 3x3")
        self.qpos[:3] = self.home_qpos[:3]
        self.qpos[3:7] = matrix_to_quaternion_wxyz(desired_trunk_matrix)
        self.qpos[self.neck_q] = self.home_qpos[self.neck_q]
        self._forward()
        targets = self._targets(human, alignment)
        residual_norm = math.inf
        for _ in range(self.max_iterations):
            residual_parts: list[np.ndarray] = []
            jacobian_parts: list[np.ndarray] = []
            for kind, index, target, weight in targets:
                current, jacobian = self._position_jacobian(kind, index)
                scale = math.sqrt(weight)
                residual_parts.append(scale * (target - current))
                jacobian_parts.append(scale * jacobian)
            residual = np.concatenate(residual_parts)
            jacobian = np.vstack(jacobian_parts)
            residual_norm = float(np.sqrt(np.mean(np.square(residual))))
            if residual_norm < 2.0e-4:
                break
            regularization_scale = math.sqrt(self.regularization)
            augmented_jacobian = np.vstack(
                (jacobian, regularization_scale * np.eye(len(self.active_v)))
            )
            augmented_residual = np.concatenate(
                (
                    residual,
                    regularization_scale
                    * (self.home_qpos[self.active_q] - self.qpos[self.active_q]),
                )
            )
            normal = augmented_jacobian.T @ augmented_jacobian
            normal.flat[:: normal.shape[0] + 1] += self.damping
            try:
                step = np.linalg.solve(
                    normal, augmented_jacobian.T @ augmented_residual
                )
            except np.linalg.LinAlgError as exc:
                raise LiveTrackingBridgeError("live IK linear solve failed") from exc
            velocity = np.zeros(self.model.nv, dtype=np.float64)
            velocity[self.active_v] = np.clip(step, -0.18, 0.18)
            mujoco.mj_integratePos(self.model, self.qpos, velocity, 1.0)
            self.qpos[self.active_q] = np.clip(
                self.qpos[self.active_q], _POLICY_LIMITS[:, 0], _POLICY_LIMITS[:, 1]
            )
            self.qpos[self.neck_q] = self.home_qpos[self.neck_q]
            self._forward()
        # Re-evaluate the qpos that is actually returned.  In particular, when
        # the final iteration takes a step, ``residual_norm`` above describes
        # the pre-step state and must not be used as the acceptance result.
        final_weighted_residuals: list[np.ndarray] = []
        final_endpoint_errors: list[float] = []
        for kind, index, target, weight in targets:
            current, _jacobian = self._position_jacobian(kind, index)
            delta = target - current
            final_weighted_residuals.append(math.sqrt(weight) * delta)
            final_endpoint_errors.append(float(np.linalg.norm(delta)))
        final_weighted = np.concatenate(final_weighted_residuals)
        self.last_weighted_rms_error_m = float(
            np.sqrt(np.mean(np.square(final_weighted)))
        )
        maximum_endpoint_error_m = max(final_endpoint_errors)
        result = self.qpos[self.active_q].copy()
        if (
            not np.all(np.isfinite(result))
            or not math.isfinite(self.last_weighted_rms_error_m)
            or not math.isfinite(maximum_endpoint_error_m)
        ):
            raise LiveTrackingBridgeError("live IK produced a non-finite result")
        return result, maximum_endpoint_error_m


class LivePicoTrackingReferenceBuilder:
    """Stateful calibration, derivative and safety gate for live IK references."""

    def __init__(
        self,
        retargeter: OnlineMicrobanRetargeter | None = None,
        *,
        max_body_gap_s: float = 0.1,
        max_ik_error_m: float = 0.02,
        max_joint_speed_rad_s: float = 5.0,
    ) -> None:
        for name, value in (
            ("max_body_gap_s", max_body_gap_s),
            ("max_ik_error_m", max_ik_error_m),
            ("max_joint_speed_rad_s", max_joint_speed_rad_s),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        self.retargeter = retargeter or OnlineMicrobanRetargeter()
        self.max_body_gap_ns = int(max_body_gap_s * 1.0e9)
        self.max_ik_error_m = max_ik_error_m
        self.max_joint_speed_rad_s = max_joint_speed_rad_s
        self.home = self.retargeter.home_action_joint_pos
        self.reset()

    @property
    def calibrated(self) -> bool:
        return self._source_calibration_matrix is not None

    def reset(self) -> None:
        self.retargeter.reset()
        self._source_calibration_matrix: np.ndarray | None = None
        self._robot_calibration_matrix: np.ndarray | None = None
        self._alignment: np.ndarray | None = None
        self._calibration_solution: np.ndarray | None = None
        self._last_sample: PicoBodySample | None = None
        self._last_reference: LiveTrackingReference | None = None

    def _neutral_reference(
        self, timestamp_ns: int, robot_matrix: np.ndarray, ik_error_m: float = 0.0
    ) -> LiveTrackingReference:
        return LiveTrackingReference(
            tuple(float(value) for value in self.home),
            (0.0,) * len(self.home),
            matrix_to_quaternion_wxyz(robot_matrix),
            timestamp_ns,
            float(ik_error_m),
        )

    def update(
        self,
        frame: Any,
        *,
        calibration_ready: bool,
        robot_calibration_ready: bool,
        enabled: bool,
        robot_trunk_quat_wxyz: Sequence[float],
    ) -> LiveTrackingReference | None:
        """Consume one frame; return neutral/live reference or fail closed.

        ``calibration_ready`` comes from the existing native mapper's stable
        released-trigger calibration.  ``enabled`` is its hold-to-walk result.
        Calibration is never created while the trigger is held.
        """

        if (
            type(calibration_ready) is not bool
            or type(robot_calibration_ready) is not bool
            or type(enabled) is not bool
        ):
            raise TypeError(
                "calibration_ready, robot_calibration_ready and enabled must be booleans"
            )
        sample = extract_pico_body_sample(frame)
        robot_matrix = quaternion_wxyz_to_matrix(
            robot_trunk_quat_wxyz, "robot_trunk_quat_wxyz"
        )
        previous_sample = self._last_sample
        if previous_sample is not None:
            delta_ns = sample.timestamp_ns - previous_sample.timestamp_ns
            if delta_ns < 0:
                self.reset()
                raise LiveTrackingBridgeError("PICO body timestamp regressed")
            if delta_ns == 0:
                if not (
                    np.array_equal(sample.positions_m, previous_sample.positions_m)
                    and np.array_equal(
                        sample.pelvis_orientation_xyzw,
                        previous_sample.pelvis_orientation_xyzw,
                    )
                ):
                    self.reset()
                    raise LiveTrackingBridgeError(
                        "PICO body changed without a new timestamp"
                    )
                return self._last_reference
            if delta_ns > self.max_body_gap_ns:
                self.reset()
                raise LiveTrackingBridgeError("PICO body timestamp has a gap")
        self._last_sample = sample

        if not calibration_ready:
            # A mapper disarm invalidates the entire calibration epoch.  Reset
            # the IK warm start too so a later released-trigger calibration can
            # never inherit a pose from an earlier owner/session.
            self.retargeter.reset()
            self._source_calibration_matrix = None
            self._robot_calibration_matrix = None
            self._alignment = None
            self._calibration_solution = None
            self._last_reference = None
            if enabled:
                raise LiveTrackingBridgeError(
                    "tracking was enabled before body calibration"
                )
            return None

        source_matrix = quaternion_xyzw_to_matrix(
            sample.pelvis_orientation_xyzw, "pelvis.orientation"
        )
        if not enabled and not robot_calibration_ready:
            # A released trigger is not sufficient by itself: calibration must
            # describe the explicit, upright and stationary robot HOME state.
            # Invalidate an older calibration as soon as the released robot is
            # no longer near HOME so it cannot be armed later with a stale base.
            self.retargeter.reset()
            self._source_calibration_matrix = None
            self._robot_calibration_matrix = None
            self._alignment = None
            self._calibration_solution = None
            self._last_reference = None
            return None
        if not self.calibrated:
            if enabled:
                raise LiveTrackingBridgeError(
                    "released-trigger body calibration is required"
                )
            self._source_calibration_matrix = source_matrix
            self._robot_calibration_matrix = robot_matrix
            self._alignment = robot_matrix @ source_matrix.T
            solution, error = self.retargeter.solve(
                sample.positions_m,
                alignment=self._alignment,
                desired_trunk_matrix=robot_matrix,
            )
            if error > self.max_ik_error_m:
                self.reset()
                raise LiveTrackingBridgeError(
                    f"calibration IK error {error:.6f} m exceeds "
                    f"{self.max_ik_error_m:.6f} m"
                )
            self._calibration_solution = solution
            self._last_reference = self._neutral_reference(
                sample.timestamp_ns, robot_matrix, error
            )
            return self._last_reference

        assert self._source_calibration_matrix is not None
        assert self._robot_calibration_matrix is not None
        assert self._alignment is not None
        assert self._calibration_solution is not None
        relative_source = self._source_calibration_matrix.T @ source_matrix
        desired_matrix = self._robot_calibration_matrix @ relative_source
        if not enabled:
            self._last_reference = self._neutral_reference(
                sample.timestamp_ns, self._robot_calibration_matrix
            )
            return self._last_reference

        solution, error = self.retargeter.solve(
            sample.positions_m,
            alignment=self._alignment,
            desired_trunk_matrix=desired_matrix,
        )
        if error > self.max_ik_error_m:
            self.reset()
            raise LiveTrackingBridgeError(
                f"live IK error {error:.6f} m exceeds {self.max_ik_error_m:.6f} m"
            )
        joint_pos = self.home + (solution - self._calibration_solution)
        if np.any(joint_pos < _POLICY_LIMITS[:, 0]) or np.any(
            joint_pos > _POLICY_LIMITS[:, 1]
        ):
            self.reset()
            raise LiveTrackingBridgeError(
                "calibration-relative live reference exceeds a policy soft limit"
            )
        previous_reference = self._last_reference
        if previous_reference is None:
            joint_vel = np.zeros_like(joint_pos)
        else:
            dt = (sample.timestamp_ns - previous_reference.source_timestamp_ns) * 1.0e-9
            if dt <= 0.0 or dt > self.max_body_gap_ns * 1.0e-9:
                self.reset()
                raise LiveTrackingBridgeError(
                    "invalid live reference derivative interval"
                )
            previous_joint_pos = np.asarray(
                previous_reference.joint_pos_rad, dtype=np.float64
            )
            joint_vel = (joint_pos - previous_joint_pos) / dt
        maximum_speed = float(np.max(np.abs(joint_vel)))
        if (
            not math.isfinite(maximum_speed)
            or maximum_speed > self.max_joint_speed_rad_s
        ):
            self.reset()
            raise LiveTrackingBridgeError(
                f"live reference joint speed {maximum_speed:.6f} rad/s exceeds "
                f"{self.max_joint_speed_rad_s:.6f} rad/s"
            )
        reference = LiveTrackingReference(
            tuple(float(value) for value in joint_pos),
            tuple(float(value) for value in joint_vel),
            matrix_to_quaternion_wxyz(desired_matrix),
            sample.timestamp_ns,
            float(error),
        )
        self._last_reference = reference
        return reference


def tracking_orientation_observation(
    current_trunk_quat_wxyz: Sequence[float],
    desired_trunk_quat_wxyz: Sequence[float],
) -> tuple[float, ...]:
    """Match MjLab ``motion_anchor_ori_b``'s six-value matrix encoding."""

    current = quaternion_wxyz_to_matrix(
        current_trunk_quat_wxyz, "current_trunk_quat_wxyz"
    )
    desired = quaternion_wxyz_to_matrix(
        desired_trunk_quat_wxyz, "desired_trunk_quat_wxyz"
    )
    relative = current.T @ desired
    return tuple(float(value) for value in relative[:, :2].reshape(-1))


def patch_tracking_actor_observation(
    observations: Any,
    reference: LiveTrackingReference,
    *,
    current_trunk_quat_wxyz: Sequence[float],
) -> Any:
    """Return a cloned TensorDict with only the live 42-value prefix replaced."""

    actor = observations["actor"]
    if actor.ndim != 2 or actor.shape[1] != LIVE_TRACKING_ACTOR_WIDTH:
        raise LiveTrackingBridgeError(
            f"tracking actor observation must be [N,{LIVE_TRACKING_ACTOR_WIDTH}]"
        )
    if actor.shape[0] != 1:
        raise LiveTrackingBridgeError(
            "live PICO bridge requires exactly one environment"
        )
    joint_pos = torch.as_tensor(
        reference.joint_pos_rad, dtype=actor.dtype, device=actor.device
    )
    joint_vel = torch.as_tensor(
        reference.joint_vel_rad_s, dtype=actor.dtype, device=actor.device
    )
    orientation = torch.as_tensor(
        tracking_orientation_observation(
            current_trunk_quat_wxyz, reference.desired_trunk_quat_wxyz
        ),
        dtype=actor.dtype,
        device=actor.device,
    )
    prefix = torch.cat((joint_pos, joint_vel, orientation), dim=0)
    if prefix.shape != (LIVE_TRACKING_REFERENCE_WIDTH,) or not bool(
        torch.isfinite(prefix).all().item()
    ):
        raise LiveTrackingBridgeError("live tracking observation prefix is malformed")
    patched = observations.clone()
    patched_actor = actor.clone()
    patched_actor[:, :LIVE_TRACKING_REFERENCE_WIDTH] = prefix.unsqueeze(0)
    patched["actor"] = patched_actor
    return patched


def clip_tracking_action_to_soft_limits(
    raw_action: torch.Tensor,
    *,
    scale: torch.Tensor,
    offset: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Project an 18-value raw action through the resolved absolute target clip."""

    if raw_action.ndim != 2 or raw_action.shape[1] != 18:
        raise LiveTrackingBridgeError("tracking action must have shape [N,18]")
    tensors = {
        "scale": torch.as_tensor(
            scale, dtype=raw_action.dtype, device=raw_action.device
        ),
        "offset": torch.as_tensor(
            offset, dtype=raw_action.dtype, device=raw_action.device
        ),
        "lower": torch.as_tensor(
            lower, dtype=raw_action.dtype, device=raw_action.device
        ),
        "upper": torch.as_tensor(
            upper, dtype=raw_action.dtype, device=raw_action.device
        ),
    }
    normalized: dict[str, torch.Tensor] = {}
    for name, tensor in tensors.items():
        if tensor.ndim == 0:
            tensor = tensor.expand_as(raw_action)
        elif tensor.shape == (18,):
            tensor = tensor.unsqueeze(0).expand_as(raw_action)
        elif tensor.shape != raw_action.shape:
            raise LiveTrackingBridgeError(
                f"tracking action {name} shape is incompatible"
            )
        if not bool(torch.isfinite(tensor).all().item()):
            raise LiveTrackingBridgeError(f"tracking action {name} is non-finite")
        normalized[name] = tensor
    if not bool(torch.isfinite(raw_action).all().item()):
        raise LiveTrackingBridgeError("tracking actor output is non-finite")
    if not bool((normalized["scale"] > 0.0).all().item()):
        raise LiveTrackingBridgeError("tracking action scale must be positive")
    if not bool((normalized["lower"] < normalized["upper"]).all().item()):
        raise LiveTrackingBridgeError("tracking action limits are invalid")
    target = raw_action * normalized["scale"] + normalized["offset"]
    target = torch.clamp(target, min=normalized["lower"], max=normalized["upper"])
    result = (target - normalized["offset"]) / normalized["scale"]
    # The inverse affine transform can round one ULP past an asymmetric limit
    # when the action manager reconstructs its absolute target.  Move the raw
    # value inward until the *reconstructed* target is inside the measured
    # articulation bounds, rather than relying on an approximate comparison.
    negative_infinity = torch.full_like(result, -torch.inf)
    positive_infinity = torch.full_like(result, torch.inf)
    for _ in range(4):
        reconstructed = result * normalized["scale"] + normalized["offset"]
        below = reconstructed < normalized["lower"]
        above = reconstructed > normalized["upper"]
        if not bool((below | above).any().item()):
            break
        result = torch.where(
            below,
            torch.nextafter(result, positive_infinity),
            torch.where(above, torch.nextafter(result, negative_infinity), result),
        )
    reconstructed = result * normalized["scale"] + normalized["offset"]
    if bool(
        ((reconstructed < normalized["lower"]) | (reconstructed > normalized["upper"]))
        .any()
        .item()
    ):
        raise LiveTrackingBridgeError("soft-limit action projection did not converge")
    if not bool(torch.isfinite(result).all().item()):
        raise LiveTrackingBridgeError("soft-limit action projection is non-finite")
    return result
