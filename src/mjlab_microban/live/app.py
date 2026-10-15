"""The live retargeting's commands (``python -m mjlab_microban.live``; docs/wholebody_live.md).

* ``run``: PICO body joints from microban_teleop's bridge (UDP 5557) -> reference frames to the Pi (UDP 5556), 50 Hz.
* ``record``: the bridge's body joints to a file (no robot needed).
* ``replay``: a recording through the same pipeline, in real time to the Pi, or as fast as it goes to a file.
* ``from-clip``: a PICO evaluation recording (G1, data/wholebody/raw/pico) as a recording, with its standing reference.
"""

from __future__ import annotations

import argparse
import queue
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np

from mjlab_microban.live import link, stream
from mjlab_microban.live.retarget import (
    BONES,
    OUTPUT_HZ,
    Pipeline,
    as_frame,
    pico_to_bones,
    world_of,
)


class Presses:
    """Calibrate and start requests from the bridge's cumulative press counts.  The counts are one bridge process's own
    (from 0 when it starts): the first packet, and one from another sender (a restarted bridge's new socket) or whose
    sequence number or counts go back (a bridge restarted on the same socket), set the base and press nothing; a
    packet at most REORDER behind the newest one is a late copy and changes nothing."""

    REORDER = 50  # packets (1 s of the bridge's 50 Hz)

    def __init__(self, log=print):
        self.last = None  # (sender, seq, calibrate, start)
        self.log = log

    def __call__(self, pipe: Pipeline, calibrate: int, start: int, seq: int | None = None, sender=None) -> None:
        last = self.last
        restarted = last is not None and sender != last[0]
        if last is not None and not restarted and seq is not None and last[1] is not None:
            behind = (last[1] - seq) % (1 << 32)
            if behind <= self.REORDER:
                return
            restarted = behind < (1 << 31)
        self.last = (sender, seq, calibrate, start)
        if last is None:
            return
        if restarted or calibrate < last[2] or start < last[3]:
            self.log(f"the bridge started again ({sender}, packet {seq}): its press counts ({calibrate}, {start}) "
                     "are the new base, nothing pressed")
            return
        if calibrate != last[2]:
            pipe.request_calibration()
        if (start - last[3]) % 2:
            pipe.set_active(not pipe.active)


class Recording:
    """Body samples as they arrived: body time (s, the headset's clock), arrival (s, local), pose (24, 7) xyzw,
    press counts; optionally a standing reference (the pseudo-bones) and its world frame."""

    def __init__(self):
        self.rows: list[tuple[float, float, np.ndarray, int, int]] = []

    def add(self, source_s, arrival_s, pose, calibrate, start):
        self.rows.append((source_s, arrival_s, np.asarray(pose, np.float32), calibrate, start))

    def save(self, path: Path, **extra) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, source_s=np.array([r[0] for r in self.rows]), arrival_s=np.array([r[1] for r in self.rows]),
                            pose=np.array([r[2] for r in self.rows]), presses=np.array([r[3:] for r in self.rows], np.int64),
                            joints=np.array(link_joint_names()), **extra)


def link_joint_names():
    from mjlab_microban.live.retarget import PICO_JOINTS
    return PICO_JOINTS


def _stdin_commands(q: queue.Queue) -> None:
    for line in sys.stdin:
        q.put(line.strip().lower())


def _status(pipe: Pipeline, frame: stream.Frame, now: float, compute_ms: list[float]) -> str:
    names = [n for n, b in (("CAL", stream.CALIBRATED), ("ACTIVE", stream.ACTIVE), ("STALE", stream.STALE),
                            ("HELD", stream.HELD), ("LATE", stream.LATE)) if frame.flags & b]
    transition = getattr(pipe, "transition", None)  # the reference through an interruption: carry / return
    names += [transition.upper()] if transition else []
    delay = f"{1000 * pipe.clock.delay:.0f}" if pipe.clock.rows else "-"
    ms = np.array(compute_ms) if compute_ms else np.zeros(1)
    bone = f", bone dir err {pipe.retargeter.bone_err_deg:.0f} deg" if pipe.retargeter is not None else ""
    return (f"seq {frame.seq} [{' '.join(names) or 'idle'}] delay {delay} ms, compute {ms.mean():.1f}/{np.percentile(ms, 99):.1f} "
            f"ms (mean/p99){bone}, trunk {np.round(frame.body_pos_w[0], 3).tolist()}")


def _receive(rx: socket.socket, q: queue.Queue) -> None:
    """Datagrams with their arrival time (stamped here, so a busy output tick does not make them look late) and
    sender."""
    while True:
        data, sender = rx.recvfrom(4096)
        q.put((time.monotonic(), data, sender))


def run(args) -> None:
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind((args.listen_host, args.listen_port))
    inbox: queue.Queue = queue.Queue()
    threading.Thread(target=_receive, args=(rx, inbox), daemon=True).start()
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest = (socket.gethostbyname(args.robot), args.robot_port)
    pipe = Pipeline(delay_s=None if args.delay_ms is None else args.delay_ms / 1000.0, background=True)
    presses, rec, last_ts = Presses(), Recording() if args.record else None, 0
    commands: queue.Queue = queue.Queue()
    threading.Thread(target=_stdin_commands, args=(commands,), daemon=True).start()
    print(f"live retargeting: body joints on UDP {args.listen_host}:{args.listen_port} -> frames to {dest[0]}:{dest[1]} "
          f"at {OUTPUT_HZ:.0f} Hz.  Left X: calibrate (stand still 1 s), left Y: start / stop; or type c / s / q + Enter.",
          flush=True)
    period, next_t, last_status, compute = 1.0 / OUTPUT_HZ, time.monotonic(), 0.0, []
    try:
        while True:
            while not commands.empty():
                c = commands.get()
                if c == "c":
                    pipe.request_calibration()
                elif c == "s":
                    pipe.set_active(not pipe.active)
                elif c == "q":
                    return
            while not inbox.empty():
                arrival, data, sender = inbox.get()
                try:
                    pkt = link.decode(data)
                except ValueError as exc:
                    print(f"dropped a body packet: {exc}", flush=True)
                    continue
                presses(pipe, pkt.calibrate_presses, pkt.start_presses, pkt.seq, sender)
                if pkt.body_valid:
                    p, q = pkt.wxyz()
                    pipe.push(pkt.body_timestamp_ns * 1e-9, arrival, *pico_to_bones(p, q))
                    if rec is not None and pkt.body_timestamp_ns > last_ts:
                        rec.add(pkt.body_timestamp_ns * 1e-9, arrival, pkt.pose, pkt.calibrate_presses, pkt.start_presses)
                    last_ts = max(last_ts, pkt.body_timestamp_ns)
            now = time.monotonic()
            t0 = time.perf_counter()
            frame = pipe.tick(now, sent_time_s=time.time())
            compute.append(1000 * (time.perf_counter() - t0))
            try:
                tx.sendto(stream.encode_frame(frame), dest)
            except OSError as exc:
                print(f"send failed: {exc}", flush=True)
            if now - last_status >= 2.0:
                print(_status(pipe, frame, now, compute[-100:]), flush=True)
                last_status, compute = now, compute[-100:]
            next_t = max(next_t + period, time.monotonic() - period)
            time.sleep(max(0.0, next_t - time.monotonic()))
    except KeyboardInterrupt:
        pass
    finally:
        if rec is not None and rec.rows:
            rec.save(args.record)
            print(f"recorded {len(rec.rows)} body samples to {args.record}", flush=True)


def record(args) -> None:
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind((args.listen_host, args.listen_port))
    rx.settimeout(0.5)
    rec, last_ts, start, last_print = Recording(), 0, time.monotonic(), 0.0
    print(f"recording PICO body joints from UDP {args.listen_host}:{args.listen_port} for {args.seconds:.0f} s "
          f"to {args.out} (Ctrl-C ends early).  Stand still for the first seconds (calibration).", flush=True)
    try:
        while time.monotonic() - start < args.seconds:
            try:
                data = rx.recv(4096)
            except TimeoutError:
                continue
            pkt = link.decode(data)
            if pkt.body_valid and pkt.body_timestamp_ns > last_ts:
                last_ts = pkt.body_timestamp_ns
                rec.add(pkt.body_timestamp_ns * 1e-9, time.monotonic(), pkt.pose, pkt.calibrate_presses, pkt.start_presses)
            if time.monotonic() - last_print >= 5.0:
                last_print = time.monotonic()
                n = len(rec.rows)
                rate = (n - 1) / (rec.rows[-1][0] - rec.rows[0][0]) if n > 1 else 0.0
                print(f"{last_print - start:.0f} s: {n} samples ({rate:.1f} Hz)", flush=True)
    except KeyboardInterrupt:
        pass
    rec.save(args.out)
    print(f"saved {len(rec.rows)} samples to {args.out}", flush=True)


def load_recording(path: Path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def drive(pipe: Pipeline, rec: dict, *, realtime: bool, send=None, start_s: float = 0.0, log_every_s: float = 2.0):
    """Feed a recording through ``pipe`` on its arrival times; one frame per output tick.  Calibrates with the
    recording's reference at once and starts tracking when it has one, else follows the recorded presses."""
    arrival = rec["arrival_s"] - rec["arrival_s"][0]
    presses = Presses()
    if "ref_pos" in rec:
        from scipy.spatial.transform import Rotation as R
        ref = as_frame(rec["ref_pos"], rec["ref_quat"])
        pipe.calibrate_with(ref, (R.from_quat(rec["world_quat"], scalar_first=True), rec["world_c"]))
        pipe.set_active(True)
    frames, render, k, n = [], [], 0, len(arrival)
    t, period = start_s, 1.0 / pipe.fps
    wall0, last_log = time.monotonic(), 0.0
    while k < n or (pipe.last_arrival is not None and t - pipe.last_arrival < 0.1):
        while k < n and arrival[k] <= t:
            if "ref_pos" not in rec:
                presses(pipe, int(rec["presses"][k, 0]), int(rec["presses"][k, 1]))
            pose = rec["pose"][k]
            pipe.push(float(rec["source_s"][k]), float(arrival[k]), *pico_to_bones(pose[:, :3], pose[:, [6, 3, 4, 5]]))
            k += 1
        if pipe.last_arrival is None:
            t += period
            continue
        frame = pipe.tick(t, sent_time_s=time.time() if realtime else t)
        frames.append(frame)
        render.append(pipe.rendered if pipe.rendered is not None else np.nan)
        if send is not None:
            send(stream.encode_frame(frame))
        if realtime:
            time.sleep(max(0.0, wall0 + (t - start_s) + period - time.monotonic()))
            if time.monotonic() - last_log >= log_every_s:
                last_log = time.monotonic()
                print(_status(pipe, frame, t, []), flush=True)
        t += period
    return frames, np.array(render)


def replay(args) -> None:
    rec = load_recording(args.recording)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    send = None
    if args.robot:
        dest = (socket.gethostbyname(args.robot), args.robot_port)
        send = lambda data: tx.sendto(data, dest)
        print(f"replaying {args.recording} ({len(rec['source_s'])} samples) to {dest[0]}:{dest[1]}", flush=True)
    while True:
        pipe = Pipeline(delay_s=None if args.delay_ms is None else args.delay_ms / 1000.0)
        frames, render = drive(pipe, rec, realtime=send is not None, send=send)
        if args.out:
            save_frames(args.out, frames, render)
            print(f"wrote {len(frames)} frames to {args.out}", flush=True)
        if not args.loop or send is None:
            return


def save_frames(path: Path, frames: list[stream.Frame], render: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, seq=np.array([f.seq for f in frames]), flags=np.array([f.flags for f in frames]),
                        sent_time_s=np.array([f.sent_time_s for f in frames]), render_s=render,
                        joint_names=np.array(stream.JOINT_NAMES), body_names=np.array(stream.BODY_NAMES),
                        **{k: np.array([getattr(f, k) for f in frames]) for k, _ in stream.FIELDS})


def clip_recording(raw: Path, name: str) -> tuple[Recording, dict]:
    """A PICO evaluation clip (G1 pseudo-bones, pico.py) as a 30 Hz recording in PICO's 24 joints, with its source's
    standing reference (as the data pipeline picks it) and the clip's first frame as the world frame."""
    from mjlab_microban.live.retarget import PICO_BONES, PICO_JOINTS
    from mjlab_microban.wholebody.retarget import pico

    g1 = pico.g1_model(str(raw / "g1" / "g1_mocap_29dof.xml"))
    sources = pico.load_sources(raw / "pico", pico.g1_joint_names(g1))
    refs, _ = pico.references(sources, g1)
    src, _, fps, rp, rq, jp = next(s for s in sources if s[1] == name)
    frames = pico.g1_frames(g1, rp, rq, jp)
    rec = Recording()
    for t, f in enumerate(frames):
        pose = np.zeros((24, 7), np.float32)
        pose[:, 6] = 1.0
        for b, (pj, qj) in PICO_BONES.items():
            pose[PICO_JOINTS.index(pj), :3] = f[b][0]
            pose[PICO_JOINTS.index(qj), 3:] = f[b][1][[1, 2, 3, 0]]
        rec.add(t / fps, t / fps, pose, 0, 0)
    rz, c = world_of(frames[0])
    ref = refs[src]
    extra = {"ref_pos": np.array([ref[b][0] for b in BONES]), "ref_quat": np.array([ref[b][1] for b in BONES]),
             "world_quat": rz.as_quat(scalar_first=True), "world_c": c}
    return rec, extra


def from_clip(args) -> None:
    rec, extra = clip_recording(args.raw, args.clip)
    rec.save(args.out, **extra)
    print(f"wrote {len(rec.rows)} samples of {args.clip} to {args.out}", flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m mjlab_microban.live", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def listen(p):
        p.add_argument("--listen-host", default="127.0.0.1")
        p.add_argument("--listen-port", type=int, default=link.DEFAULT_PORT)

    def robot(p, required):
        p.add_argument("--robot", required=required, help="the Pi's host name or address")
        p.add_argument("--robot-port", type=int, default=stream.DEFAULT_PORT)
        p.add_argument("--delay-ms", type=float, default=None, help="interpolation delay (default: input interval + jitter)")

    p = sub.add_parser("run", help="retarget the PICO body live and send frames to the Pi")
    listen(p)
    robot(p, True)
    p.add_argument("--record", type=Path, help="also record the body samples to this file")
    p.set_defaults(fn=run)
    p = sub.add_parser("record", help="record the PICO body joints (no robot)")
    listen(p)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=300.0)
    p.set_defaults(fn=record)
    p = sub.add_parser("replay", help="a recording through the pipeline, to the Pi (real time) or to a file")
    p.add_argument("recording", type=Path)
    robot(p, False)
    p.add_argument("--out", type=Path, help="write the frames (npz)")
    p.add_argument("--loop", action="store_true")
    p.set_defaults(fn=replay)
    p = sub.add_parser("from-clip", help="a PICO evaluation clip (G1) as a recording")
    p.add_argument("clip", help="clip name, e.g. picoteleopit_rec04")
    p.add_argument("--raw", type=Path, default=Path("data/wholebody/raw"))
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(fn=from_clip)
    args = ap.parse_args(argv)
    args.fn(args)
