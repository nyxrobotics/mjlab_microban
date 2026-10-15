"""The live command's inputs (mjlab_microban/live/app.py): the bridge's press counts across a bridge restart and a
late packet, and a start pressed before the calibration (CPU, no downloaded data)."""

from __future__ import annotations

import queue
import socket
import threading

import numpy as np

from mjlab_microban.live import stream
from mjlab_microban.live.app import Presses, _receive
from mjlab_microban.live.retarget import BONES, Pipeline, as_frame, world_of
from mjlab_microban.wholebody.retarget import pico

BRIDGE, RESTARTED = ("127.0.0.1", 40001), ("127.0.0.1", 40002)


class FakePipe:
    def __init__(self):
        self.active, self.calls = False, []

    def request_calibration(self):
        self.calls.append("calibrate")

    def set_active(self, active):
        self.active = active
        self.calls.append(active)


def test_a_restarted_bridge_sets_a_new_base_instead_of_pressing():
    # The live command started while the bridge's Y count was odd, tracking is off, and the bridge starts again
    # (counts and sequence from 0, a new socket): the counts going 1 -> 0 used to toggle tracking on by themselves.
    pipe, log = FakePipe(), []
    presses = Presses(log=log.append)
    presses(pipe, 2, 1, seq=900, sender=BRIDGE)
    presses(pipe, 2, 1, seq=901, sender=BRIDGE)
    presses(pipe, 0, 0, seq=0, sender=RESTARTED)
    presses(pipe, 0, 0, seq=1, sender=RESTARTED)
    assert pipe.calls == [] and "started again" in log[-1]
    presses(pipe, 1, 1, seq=2, sender=RESTARTED)  # X and Y on the new bridge
    assert pipe.calls == ["calibrate", True]


def test_a_bridge_restarted_on_the_same_socket_or_counts_going_back_set_a_new_base():
    pipe = FakePipe()
    presses = Presses(log=lambda _: None)
    presses(pipe, 3, 5, seq=5000, sender=BRIDGE)
    presses(pipe, 0, 0, seq=0, sender=BRIDGE)  # far behind: a new bridge process
    presses(pipe, 0, 1, seq=1, sender=BRIDGE)
    assert pipe.calls == [True]
    presses(pipe, 0, 0, seq=2, sender=BRIDGE)  # the counts go back
    presses(pipe, 0, 0, seq=3, sender=BRIDGE)
    assert pipe.calls == [True]
    replay = Presses(log=lambda _: None)  # a recording's counts (no sender, no sequence)
    replay(pipe, 4, 7)
    replay(pipe, 0, 0)
    replay(pipe, 0, 1)
    assert pipe.calls == [True, False]


def test_a_late_packet_neither_presses_nor_moves_the_base():
    pipe = FakePipe()
    presses = Presses(log=lambda _: None)
    presses(pipe, 1, 1, seq=10, sender=BRIDGE)
    presses(pipe, 1, 2, seq=11, sender=BRIDGE)
    presses(pipe, 1, 1, seq=10, sender=BRIDGE)  # a late copy of the packet before
    presses(pipe, 1, 2, seq=12, sender=BRIDGE)
    assert pipe.calls == [True]


def test_the_receiver_hands_over_the_sender():
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tx.bind(("127.0.0.1", 0))
    inbox: queue.Queue = queue.Queue()
    threading.Thread(target=_receive, args=(rx, inbox), daemon=True).start()
    try:
        tx.sendto(b"MBBD", rx.getsockname())
        _, data, sender = inbox.get(timeout=1.0)
        assert data == b"MBBD" and sender == tx.getsockname()
    finally:
        tx.close()


def _standing_frame() -> dict:
    """A still standing operator facing +x (metres; legs and arms straight, as the standing reference makes them)."""
    side = {"hip": (0.0, 0.09, 0.92), "knee": (0.0, 0.09, 0.50), "ankle": (0.0, 0.09, 0.08), "toe": (0.12, 0.09, 0.02),
            "shoulder": (0.0, 0.18, 1.42), "elbow": (-0.02, 0.18, 1.15), "wrist": (0.0, 0.18, 0.88)}
    pos = {"pelvis": (0.0, 0.0, 0.95), "torso": (0.0, 0.0, 1.05), "neck": (0.0, 0.0, 1.45), "head": (0.0, 0.0, 1.6),
           **{f"{s}_{p}": (x, y * sign, z) for s, sign in (("L", 1.0), ("R", -1.0)) for p, (x, y, z) in side.items()}}
    return as_frame(np.array([pos[b] for b in BONES]), np.tile([1.0, 0.0, 0.0, 0.0], (len(BONES), 1)))


def test_a_start_pressed_before_the_calibration_is_refused_and_said():
    log = []
    pipe = Pipeline(log=log.append)
    presses = Presses(log=log.append)
    presses(pipe, 0, 0, seq=0, sender=BRIDGE)
    presses(pipe, 0, 1, seq=1, sender=BRIDGE)  # left Y before left X
    assert not pipe.active and "not calibrated" in log[-1]
    assert not pipe.tick(0.0).flags & stream.ACTIVE
    frame = _standing_frame()
    pipe.calibrate_with(pico.standing_reference(frame), world_of(frame))
    assert not pipe.active  # the refused press does not start tracking when the calibration comes
    presses(pipe, 0, 2, seq=2, sender=BRIDGE)
    assert pipe.active and log[-1] == "tracking started"
