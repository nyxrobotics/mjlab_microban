# Adapted from Teleopit f926386 train_mimic/data/motion_fk.py (Apache-2.0,
# https://github.com/BotRunner64/Teleopit).  Changes: the Microban model and
# its trunk root, no NPZ consistency check.

"""Forward kinematics and finite-difference velocities of retargeted clips."""

from __future__ import annotations

import mujoco
import numpy as np


def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """Convert quaternion from xyzw to wxyz convention."""
    return np.asarray(q)[..., [3, 0, 1, 2]]


def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    """Convert quaternion from wxyz to xyzw convention."""
    return np.asarray(q)[..., [1, 2, 3, 0]]


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion array in wxyz convention."""
    arr = np.asarray(q, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    return arr / norms


def finite_diff_velocity(x: np.ndarray, dt: float) -> np.ndarray:
    """Finite-difference velocity with central interior and one-sided edges."""
    arr = np.asarray(x, dtype=np.float32)
    vel = np.zeros_like(arr, dtype=np.float32)
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0, got {dt}")
    if arr.shape[0] < 2:
        return vel
    inv_dt = 1.0 / float(dt)
    vel[1:-1] = (arr[2:] - arr[:-2]) * (0.5 * inv_dt)
    vel[0] = (arr[1] - arr[0]) * inv_dt
    vel[-1] = (arr[-1] - arr[-2]) * inv_dt
    return vel


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two wxyz quaternions: q1 * q2."""
    q1 = np.asarray(q1, dtype=np.float32)
    q2 = np.asarray(q2, dtype=np.float32)
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate of a wxyz quaternion."""
    conj = np.asarray(q, dtype=np.float32).copy()
    conj[..., 1:] = -conj[..., 1:]
    return conj


def quat_to_angular_velocity(q: np.ndarray, dt: float) -> np.ndarray:
    """World angular velocity of a wxyz quaternion sequence (T, ..., 4)."""
    quat = normalize_quaternion(q)
    if quat.shape[0] < 2:
        return np.zeros(quat.shape[:-1] + (3,), dtype=np.float32)

    flat = quat.reshape(quat.shape[0], -1, 4)
    dots = np.sum(flat[1:] * flat[:-1], axis=-1)
    signs = np.where(dots < 0.0, -1.0, 1.0).astype(np.float32)
    signs = np.concatenate([np.ones_like(signs[:1]), signs], axis=0)
    flat = flat * np.cumprod(signs, axis=0)[..., None]
    quat = flat.reshape(quat.shape)

    q_dot = finite_diff_velocity(quat, dt)
    product = quat_multiply(q_dot, quat_conjugate(quat))
    return (2.0 * product[..., 1:4]).astype(np.float32)


def compute_body_velocities(
    body_pos_w: np.ndarray, body_quat_w: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Linear and angular world velocities of FK body poses."""
    body_lin_vel_w = finite_diff_velocity(np.asarray(body_pos_w, dtype=np.float32), dt)
    body_ang_vel_w = quat_to_angular_velocity(normalize_quaternion(body_quat_w), dt)
    return body_lin_vel_w, body_ang_vel_w


def set_qpos(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    root_pos: np.ndarray,
    root_quat_wxyz: np.ndarray,
    joint_pos: np.ndarray,
) -> None:
    """Write one frame (root on the free joint, hinges in qpos order)."""
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[:3] = root_pos
    data.qpos[3:7] = root_quat_wxyz
    data.qpos[7 : 7 + joint_pos.shape[-1]] = joint_pos
    mujoco.mj_kinematics(model, data)
