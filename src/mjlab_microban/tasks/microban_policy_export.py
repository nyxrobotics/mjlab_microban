# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Shared names and observation contract of Microban's 18-DoF teleop policy.

The PICO policy deliberately leaves ``head``, ``neck_roll`` and ``neck_pitch``
to the independent HMD controller: it acts on the 18 body joints and observes
all 21.  This module holds those orders, the 83-wide actor observation schema
and its environment check, and the deterministic all-column parity corpus that
the exporters use.
"""

from __future__ import annotations

import math

import numpy as np
from mjlab.envs import ManagerBasedRlEnv

from mjlab_microban.robot.microban_constants import HOME_PROJECTED_GRAVITY
from mjlab_microban.robot.microban_hand_fk import (
    MICROBAN_HAND_TARGET_FRAME,
    MICROBAN_HAND_TARGET_NORMALIZER_ABS_BOUND_M,
)

# Entity.find_joints_by_actuator_names resolves in the model's natural joint
# order.  Keep this explicit contract next to the exporter and verify it at env
# construction/export time so an XML reorder cannot silently change deployment.
MICROBAN_TELEOP_ACTION_JOINT_NAMES: tuple[str, ...] = (
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_elbow",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_elbow",
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
)
MICROBAN_HMD_JOINT_NAMES: tuple[str, ...] = (
    "head",
    "neck_roll",
    "neck_pitch",
)
# This tuple is the wire format consumed by the deployment runtime.  The order
# is just as important as the widths: concatenating the same terms in a
# different order produces a valid-shaped policy input with incorrect meaning.
MICROBAN_TELEOP_OBSERVATION_SCHEMA: tuple[tuple[str, int], ...] = (
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("joint_pos", 21),
    ("joint_vel", 21),
    ("actions", 18),
    ("command", 3),
    ("foot_target", 6),
    ("hand_target", 8),
)
MICROBAN_TELEOP_OBSERVATION_WIDTH = sum(
    width for _, width in MICROBAN_TELEOP_OBSERVATION_SCHEMA
)
MICROBAN_TELEOP_ACTION_WIDTH = len(MICROBAN_TELEOP_ACTION_JOINT_NAMES)
MICROBAN_TELEOP_NUM_STEPS_PER_ENV = 24
# The final curriculum stage expands simultaneous-foot lift support from the
# conservative play/early-training ceiling (12 mm) to 20 mm. Deployment
# metadata for an acceptance-backed final checkpoint must describe that final
# support even though export constructs a play environment with curriculum
# disabled. The curriculum imports this wire-contract constant too, avoiding a
# duplicated final-stage limit.
MICROBAN_TELEOP_FINAL_BOTH_FEET_LIFT_UPPER_M = 0.02
TELEOP_ONNX_PARITY_SEED = 20260924
TELEOP_ONNX_PARITY_SAMPLE_COUNT = 16


def _representative_observation_bounds() -> tuple[np.ndarray, np.ndarray]:
    """Return conservative finite bounds in the exact 83-value schema order."""

    lower = np.asarray(
        [-4.0] * 3
        + [-1.0] * 3
        + [-math.pi] * 21
        + [-12.0] * 21
        + [-1.0] * 18
        + [-1.0, -1.0, -2.0]
        + [-0.03, -0.03, 0.0] * 2
        + [-value for value in MICROBAN_HAND_TARGET_NORMALIZER_ABS_BOUND_M] * 2
        + [0.0, 0.0],
        dtype=np.float32,
    )
    upper = np.asarray(
        [4.0] * 3
        + [1.0] * 3
        + [math.pi] * 21
        + [12.0] * 21
        + [1.0] * 18
        + [1.0, 1.0, 2.0]
        + [0.03, 0.03, 0.05] * 2
        + list(MICROBAN_HAND_TARGET_NORMALIZER_ABS_BOUND_M) * 2
        + [1.0, 1.0],
        dtype=np.float32,
    )
    if lower.shape != (MICROBAN_TELEOP_OBSERVATION_WIDTH,) or upper.shape != (
        MICROBAN_TELEOP_OBSERVATION_WIDTH,
    ):
        raise RuntimeError("Parity observation bounds do not match the 83-value schema")
    return lower, upper


def deterministic_teleop_parity_inputs(
    *,
    seed: int = TELEOP_ONNX_PARITY_SEED,
    sample_count: int = TELEOP_ONNX_PARITY_SAMPLE_COUNT,
) -> np.ndarray:
    """Build repeatable neutral, boundary, and seeded finite observations."""

    if sample_count < 4:
        raise ValueError("Parity corpus requires at least four samples")
    lower, upper = _representative_observation_bounds()
    midpoint = (lower + upper) * np.float32(0.5)
    neutral = np.zeros(MICROBAN_TELEOP_OBSERVATION_WIDTH, dtype=np.float32)
    # Standing at HOME the trunk observes the HOME projected gravity
    # ((0, 0, -1) for the vertical-trunk HOME).
    neutral[3:6] = HOME_PROJECTED_GRAVITY
    rows = [neutral, lower, upper, midpoint]
    rng = np.random.default_rng(seed)
    if sample_count > len(rows):
        random_rows = rng.uniform(
            lower,
            upper,
            size=(sample_count - len(rows), MICROBAN_TELEOP_OBSERVATION_WIDTH),
        ).astype(np.float32)
        # The final two hand target values are activation flags, not positions.
        random_rows[:, -2:] = rng.integers(0, 2, size=(random_rows.shape[0], 2)).astype(
            np.float32
        )
        rows.extend(random_rows)
    observations = np.stack(rows, axis=0).reshape(
        sample_count, 1, MICROBAN_TELEOP_OBSERVATION_WIDTH
    )
    if not np.isfinite(observations).all():
        raise RuntimeError("Parity corpus contains non-finite observations")
    return observations


def validate_microban_teleop_observation_contract(
    env: ManagerBasedRlEnv,
) -> None:
    """Reject an environment whose actor vector differs from the wire schema."""

    manager = env.observation_manager
    expected_names = tuple(name for name, _ in MICROBAN_TELEOP_OBSERVATION_SCHEMA)
    actor_names = tuple(manager.active_terms["actor"])
    if actor_names != expected_names:
        raise ValueError(
            "Unsafe Microban actor observation order: "
            f"resolved {actor_names}, expected {expected_names}"
        )

    if not manager.group_obs_concatenate["actor"]:
        raise ValueError("Microban actor observations must be concatenated")

    actor_term_dims = manager.group_obs_term_dim["actor"]
    actor_widths = tuple(math.prod(dims) for dims in actor_term_dims)
    expected_widths = tuple(width for _, width in MICROBAN_TELEOP_OBSERVATION_SCHEMA)
    if actor_widths != expected_widths:
        raise ValueError(
            "Unsafe Microban actor observation widths: "
            f"resolved {actor_widths}, expected {expected_widths}"
        )

    actor_group_dim = manager.group_obs_dim["actor"]
    if actor_group_dim != (MICROBAN_TELEOP_OBSERVATION_WIDTH,):
        raise ValueError(
            "Unsafe Microban actor observation shape: "
            f"resolved {actor_group_dim}, expected "
            f"({MICROBAN_TELEOP_OBSERVATION_WIDTH},)"
        )

# PICO foot/hand target columns are offsets in the trunk frame with HOME's
# forward lean rotated out, R_trunk * R_y(-HOME_TRUNK_PITCH_RAD): level at HOME,
# x forward, y left, z up (the twist uses the same frame).  With a vertical
# trunk at HOME that is the trunk frame ("robot_trunk_xyz_forward_left_up").
MICROBAN_TELEOP_TARGET_FRAME = MICROBAN_HAND_TARGET_FRAME
