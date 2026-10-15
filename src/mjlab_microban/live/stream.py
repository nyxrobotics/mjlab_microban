"""The whole-body reference stream from the PC retargeting to the Pi (docs/wholebody_reference_stream.md).

One UDP datagram per frame: a 24-byte header (``<4sHHHIdH``: magic, version,
joint count, body count, sequence number, send time, flags) and one frame of
the training HDF5 (``wholebody/data/hdf5.py`` FRAME_KEYS, in that order) as
little-endian float32.  The joint and body orders are the training HDF5's.
Only NumPy and the standard library: the Pi side can copy this file.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

MAGIC = b"MBWB"
# 2: the reference is carried through an interruption (STALE, HELD, LATE) and back to the operator without a jump
# (retarget.py Carry / Return / Guard).  Version 1 sent the held pose, then the live one in one frame; the Pi follows
# HELD for up to 1 s only from version 2 and refuses version 1's frames.
VERSION = 2
HEADER = struct.Struct("<4sHHHIdH")
DEFAULT_PORT = 5556

# The training HDF5's joint_names (contract.OBSERVATION_JOINT_NAMES) and body_names (robot.xml's bodies without the world).
JOINT_NAMES = (
    "head", "neck_roll", "neck_pitch",
    "right_shoulder_pitch", "right_shoulder_roll", "right_elbow",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "left_shoulder_pitch", "left_shoulder_roll", "left_elbow",
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle_pitch", "left_ankle_roll",
)
BODY_NAMES = (
    "trunk", "head", "roll_pitch_link", "camera_cover",
    "shoulder", "humerus__configuration_right", "radius",
    "hip", "hip_block__configuration_right", "femur__configuration_right", "tibia__configuration_right",
    "ankle_block__configuration_left", "foot",
    "shoulder_2", "humerus__configuration_left", "radius_2",
    "hip_2", "hip_block__configuration_left", "femur__configuration_left", "tibia__configuration_left",
    "ankle_block__configuration_right", "foot_2",
)
NJ, NB = len(JOINT_NAMES), len(BODY_NAMES)
# (key, shape) in FRAME_KEYS order.
FIELDS = (
    ("joint_pos", (NJ,)),
    ("joint_vel", (NJ,)),
    ("body_pos_w", (NB, 3)),
    ("body_quat_w", (NB, 4)),
    ("body_lin_vel_w", (NB, 3)),
    ("body_ang_vel_w", (NB, 3)),
)
FLOATS = sum(int(np.prod(shape)) for _, shape in FIELDS)
SIZE = HEADER.size + 4 * FLOATS
BODY_DTYPE = np.dtype("<f4")

# Flags (the doc's table).
CALIBRATED = 1 << 0
ACTIVE = 1 << 1
RESTART = 1 << 2
STALE = 1 << 3
HELD = 1 << 4
LATE = 1 << 5


class StreamError(ValueError):
    """A datagram that is not a frame of this format."""


@dataclass
class Frame:
    seq: int
    sent_time_s: float
    flags: int
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    body_lin_vel_w: np.ndarray
    body_ang_vel_w: np.ndarray


def encode_frame(frame: Frame) -> bytes:
    parts = []
    for key, shape in FIELDS:
        value = np.asarray(getattr(frame, key), dtype=BODY_DTYPE)
        if value.shape != shape:
            raise StreamError(f"{key} is {value.shape}, expected {shape}")
        parts.append(value.ravel())
    header = HEADER.pack(MAGIC, VERSION, NJ, NB, frame.seq & 0xFFFFFFFF, float(frame.sent_time_s), frame.flags & 0xFFFF)
    return header + np.concatenate(parts).astype(BODY_DTYPE).tobytes()


def decode_frame(data: bytes) -> Frame:
    if len(data) != SIZE:
        raise StreamError(f"{len(data)} bytes, expected {SIZE}")
    magic, version, nj, nb, seq, sent, flags = HEADER.unpack_from(data)
    if magic != MAGIC or version != VERSION:
        raise StreamError(f"magic {magic!r} version {version}, expected {MAGIC!r} {VERSION}")
    if (nj, nb) != (NJ, NB):
        raise StreamError(f"{nj} joints and {nb} bodies, expected {NJ} and {NB}")
    values = np.frombuffer(data, dtype=BODY_DTYPE, offset=HEADER.size)
    out, start = {}, 0
    for key, shape in FIELDS:
        n = int(np.prod(shape))
        out[key] = values[start:start + n].reshape(shape).copy()
        start += n
    return Frame(seq=seq, sent_time_s=sent, flags=flags, **out)


def seq_newer(seq: int, last: int) -> bool:
    """Whether ``seq`` comes after ``last`` (serial number arithmetic modulo 2^32)."""
    return 0 < (seq - last) % (1 << 32) < (1 << 31)
