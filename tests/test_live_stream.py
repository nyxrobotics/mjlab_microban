"""The PC -> Pi reference stream (docs/wholebody_reference_stream.md) and the bridge -> PC body link."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from mjlab_microban.live import link, stream
from mjlab_microban.wholebody import contract
from mjlab_microban.wholebody.data.preprocess import ClipChecker
from mjlab_microban.wholebody.env_cfg import training_collision_model


def _frame(seed=0, **kw):
    rng = np.random.default_rng(seed)
    arrays = {key: rng.normal(size=shape).astype(np.float32) for key, shape in stream.FIELDS}
    return stream.Frame(seq=kw.get("seq", 7), sent_time_s=kw.get("sent", 1760000000.25), flags=kw.get("flags", 0b11), **arrays)


def test_a_frame_round_trips_in_the_documented_layout():
    frame = _frame()
    data = stream.encode_frame(frame)
    assert len(data) == stream.SIZE == 24 + 4 * 328 == 1336
    # The header by hand, as the Pi side reads it.
    assert data[:4] == b"MBWB"
    version, nj, nb, seq = struct.unpack_from("<HHHI", data, 4)
    (sent,) = struct.unpack_from("<d", data, 14)
    (flags,) = struct.unpack_from("<H", data, 22)
    assert (version, nj, nb, seq, sent, flags) == (2, 21, 22, 7, 1760000000.25, 0b11)
    body = np.frombuffer(data, "<f4", offset=24)
    assert np.array_equal(body[:21], frame.joint_pos)
    assert np.array_equal(body[42:42 + 66].reshape(22, 3), frame.body_pos_w)
    assert np.array_equal(body[108:108 + 88].reshape(22, 4), frame.body_quat_w)
    back = stream.decode_frame(data)
    for key, _ in stream.FIELDS:
        assert np.array_equal(getattr(back, key), getattr(frame, key))
    assert (back.seq, back.sent_time_s, back.flags) == (7, 1760000000.25, 0b11)


@pytest.mark.parametrize("mutate", [
    lambda d: d[:-4],
    lambda d: b"XXXX" + d[4:],
    lambda d: d[:4] + struct.pack("<H", 1) + d[6:],  # the PC before the interruption reference (it jumps)
    lambda d: d[:6] + struct.pack("<H", 20) + d[8:],
])
def test_other_datagrams_are_refused(mutate):
    with pytest.raises(stream.StreamError):
        stream.decode_frame(mutate(stream.encode_frame(_frame())))


def test_sequence_numbers_wrap():
    assert stream.seq_newer(1, 0) and not stream.seq_newer(0, 1)
    assert stream.seq_newer(3, 0xFFFFFFFE) and not stream.seq_newer(0xFFFFFFFE, 3)


def test_the_names_are_the_training_hdf5s():
    checker = ClipChecker(training_collision_model())  # build.py's names of the HDF5
    assert stream.JOINT_NAMES == checker.joint_names == contract.OBSERVATION_JOINT_NAMES
    assert stream.BODY_NAMES == checker.body_names


def test_the_body_link_round_trips():
    pose = np.arange(24 * 7, dtype=np.float32).reshape(24, 7)
    p = link.Packet(seq=5, calibrate_presses=2, start_presses=3, body_timestamp_ns=123456789012, flags=link.BODY_VALID, pose=pose)
    data = link.encode(p)
    assert len(data) == link.SIZE == 28 + 24 * 7 * 4
    back = link.decode(data)
    assert (back.seq, back.calibrate_presses, back.start_presses, back.body_timestamp_ns, back.body_valid) == (5, 2, 3, 123456789012, True)
    assert np.array_equal(back.pose, pose)
    _, q = back.wxyz()
    assert np.array_equal(q[0], pose[0, [6, 3, 4, 5]])
