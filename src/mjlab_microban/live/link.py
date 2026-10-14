"""PICO's body joints from microban_teleop's bridge to the live retargeting (PC-local UDP, docs/wholebody_live.md).

One datagram per bridge tick (50 Hz), little-endian: a 28-byte header (``<4sHIIIQH``: magic ``b"MBBD"``, version 1,
sequence number, calibrate presses, start presses, body timestamp in ns (the headset's clock), flags (bit 0: the body
is valid and fresh)) and 24 x (px, py, pz, qx, qy, qz, qw) float32 in Microban axes (x forward, y left, z up;
xyzw), in PICO's joint order (retarget.PICO_JOINTS).  The press counts are the cumulative left X (calibrate) and
left Y (start / stop) presses.  microban_teleop's ``wholebody_link.py`` writes it.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

MAGIC = b"MBBD"
VERSION = 1
HEADER = struct.Struct("<4sHIIIQH")
JOINTS = 24
SIZE = HEADER.size + JOINTS * 7 * 4
DEFAULT_PORT = 5557
BODY_VALID = 1


@dataclass
class Packet:
    seq: int
    calibrate_presses: int
    start_presses: int
    body_timestamp_ns: int
    flags: int
    pose: np.ndarray  # (24, 7): position, quaternion xyzw

    @property
    def body_valid(self) -> bool:
        return bool(self.flags & BODY_VALID) and self.body_timestamp_ns > 0

    def wxyz(self) -> tuple[np.ndarray, np.ndarray]:
        return self.pose[:, :3].astype(np.float64), self.pose[:, [6, 3, 4, 5]].astype(np.float64)


def encode(p: Packet) -> bytes:
    pose = np.asarray(p.pose, dtype="<f4")
    if pose.shape != (JOINTS, 7):
        raise ValueError(f"pose is {pose.shape}, expected ({JOINTS}, 7)")
    return HEADER.pack(MAGIC, VERSION, p.seq & 0xFFFFFFFF, p.calibrate_presses & 0xFFFFFFFF, p.start_presses & 0xFFFFFFFF,
                       p.body_timestamp_ns, p.flags) + pose.tobytes()


def decode(data: bytes) -> Packet:
    if len(data) != SIZE:
        raise ValueError(f"{len(data)} bytes, expected {SIZE}")
    magic, version, seq, cal, start, ts, flags = HEADER.unpack_from(data)
    if magic != MAGIC or version != VERSION:
        raise ValueError(f"magic {magic!r} version {version}")
    pose = np.frombuffer(data, dtype="<f4", offset=HEADER.size).reshape(JOINTS, 7).copy()
    return Packet(seq, cal, start, ts, flags, pose)
