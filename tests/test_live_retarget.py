"""The live retargeting (mjlab_microban/live) on a synthetic standing operator (CPU, no downloaded data).

It runs without the training stack, its collision model and frame rules are the training ones, a still standing
operator calibrates onto HOME and is sent as HOME, one who is not standing is refused, and the flags follow the
input (start, stale input, a lifted pose held).
"""

from __future__ import annotations

import os
import subprocess
import sys

import mujoco
import numpy as np

from mjlab_microban.live import stream
from mjlab_microban.live.app import Presses
from mjlab_microban.live.retarget import (
    BONES,
    PICO_JOINTS,
    Interpolator,
    Pipeline,
    pico_to_bones,
)
from mjlab_microban.robot.home_pose import HOME
from mjlab_microban.wholebody import env_cfg
from mjlab_microban.wholebody.data import preprocess
from mjlab_microban.wholebody.retarget.model import collision_model

RATE = 25.0

# A standing operator facing +x (left = +y), metres: spine, legs (hip -> ankle) and arms (shoulder -> wrist) straight,
# as the standing reference makes them, so the operator's pose is the reference's and retargets to HOME.
STAND = {
    "pelvis": (0.0, 0.0, 0.95), "torso": (0.0, 0.0, 1.05), "neck": (0.0, 0.0, 1.45), "head": (0.0, 0.0, 1.6),
    **{f"{s}_{p}": (x, y * sign, z) for s, sign in (("L", 1.0), ("R", -1.0)) for p, (x, y, z) in {
        "hip": (0.0, 0.09, 0.92), "knee": (0.0, 0.09, 0.50), "ankle": (0.0, 0.09, 0.08), "toe": (0.12, 0.09, 0.02),
        "shoulder": (0.0, 0.18, 1.42), "elbow": (-0.02, 0.18, 1.15), "wrist": (0.0, 0.18, 0.88)}.items()},
}


def _pose(lift=0.0, arms_up=False):
    p = np.array([STAND[b] for b in BONES]) + [0.0, 0.0, lift]
    if arms_up:
        for s in "LR":
            sh = p[BONES.index(f"{s}_shoulder")]
            p[BONES.index(f"{s}_elbow")] = sh + [0, 0, 0.28]
            p[BONES.index(f"{s}_wrist")] = sh + [0, 0, 0.54]
    q = np.tile([1.0, 0.0, 0.0, 0.0], (len(BONES), 1))
    return p, q


def _feed(pipe, t0, seconds, **pose):
    """Samples at RATE and output ticks at 50 Hz over ``seconds``; returns the frames."""
    frames, t, k = [], t0, 0
    while t < t0 + seconds - 1e-9:
        while t0 + k / RATE <= t + 1e-9:
            pipe.push(t0 + k / RATE, t0 + k / RATE, *_pose(**pose))
            k += 1
        frames.append(pipe.tick(t))
        t += 1.0 / pipe.fps
    return frames, t


def test_the_live_command_does_not_load_the_training_stack():
    code = ("import sys; import mjlab_microban.live.app, mjlab_microban.live.retarget; "
            "print(sorted(m for m in ('mjlab', 'torch', 'warp') if m in sys.modules))")
    env = {**os.environ, "MICROBAN_WITHOUT_TRAINING": "1"}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_the_collision_model_and_rules_are_the_training_ones():
    a, b = env_cfg.training_collision_model(), collision_model()
    assert (a.ngeom, a.nbody) == (b.ngeom, b.nbody)
    for key in ("geom_type", "geom_size", "geom_pos", "geom_quat", "geom_bodyid", "geom_contype", "geom_conaffinity",
                "geom_condim", "geom_margin", "geom_gap", "body_pos", "body_quat", "jnt_range"):
        assert np.array_equal(getattr(a, key), getattr(b, key)), key
    names = lambda m: [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(m.ngeom)]
    assert names(a) == names(b)
    assert preprocess.HEIGHT_TERMINATION_M == env_cfg.HEIGHT_TERMINATION_M


def test_pico_joints_become_the_pseudo_bones():
    pos = np.arange(24 * 3, dtype=float).reshape(24, 3)
    quat = np.tile([1.0, 0, 0, 0], (24, 1)) * np.arange(1, 25)[:, None]
    p, q = pico_to_bones(pos, quat)
    assert np.array_equal(p[BONES.index("L_knee")], pos[PICO_JOINTS.index("left_knee")])
    assert np.array_equal(p[BONES.index("R_toe")], pos[PICO_JOINTS.index("right_foot")])
    assert np.array_equal(q[BONES.index("R_toe")], quat[PICO_JOINTS.index("right_ankle")])
    assert np.array_equal(p[BONES.index("neck")], pos[PICO_JOINTS.index("neck")])


def test_interpolation_between_samples():
    it = Interpolator()
    q0 = np.array([[1.0, 0, 0, 0]])
    q1 = np.array([[np.cos(0.5), 0, 0, np.sin(0.5)]])  # 1 rad about z
    it.push(0.0, np.zeros((1, 3)), q0)
    it.push(0.04, np.ones((1, 3)), q1)
    p, q, late = it.sample(0.02)
    assert np.allclose(p, 0.5) and not late
    assert np.allclose(q, [[np.cos(0.25), 0, 0, np.sin(0.25)]])
    assert it.sample(0.05)[2]


def test_a_still_standing_operator_is_sent_as_home():
    pipe = Pipeline(log=lambda _: None)
    frames, t = _feed(pipe, 100.0, 0.5)
    assert all(f.flags == 0 for f in frames)  # before the calibration: HOME, not calibrated
    pipe.request_calibration()
    frames, t = _feed(pipe, t, 1.2)
    assert pipe.retargeter is not None
    pipe.set_active(True)
    frames, t = _feed(pipe, t, 1.0)
    flags = np.array([f.flags for f in frames])
    assert flags[0] == stream.CALIBRATED | stream.ACTIVE | stream.RESTART
    assert np.all(flags[1:] == stream.CALIBRATED | stream.ACTIVE)
    home = np.array([HOME.joint_pos_rad[n] for n in stream.JOINT_NAMES])
    last = frames[-1]
    assert np.degrees(np.abs(last.joint_pos - home)).max() < 0.5
    assert abs(last.body_pos_w[0, 2] - HOME.root_pos[2]) < 0.002
    assert np.abs(last.joint_vel).max() < 0.05 and np.abs(last.body_lin_vel_w).max() < 0.01


def test_a_calibration_that_does_not_stand_is_refused():
    log = []
    pipe = Pipeline(log=log.append)
    _, t = _feed(pipe, 0.0, 0.2, arms_up=True)
    pipe.request_calibration()
    _feed(pipe, t, 1.2, arms_up=True)
    assert pipe.retargeter is None and "arms down" in log[-1]


def test_stale_input_and_a_lifted_pose_are_held():
    pipe = Pipeline(log=lambda _: None)
    _, t = _feed(pipe, 0.0, 0.2)
    pipe.request_calibration()
    _, t = _feed(pipe, t, 1.2)
    frames, t = _feed(pipe, t, 0.5)
    standing = frames[-1]
    # Jumping 0.5 m (the robot's lowest point ~0.13 m up, past the 0.05 m termination height): held while up and 0.2 s after.
    frames, t = _feed(pipe, t, 0.4, lift=0.5)
    held = np.array([(f.flags & stream.HELD) > 0 for f in frames])
    assert held[5:].all()
    assert np.allclose(frames[-1].body_pos_w, standing.body_pos_w) and not frames[-1].joint_vel.any()
    frames, t = _feed(pipe, t, 0.6)
    held = np.array([(f.flags & stream.HELD) > 0 for f in frames])
    assert held[:5].all() and not held[-5:].any()
    # No input for 0.3 s: stale.
    assert pipe.tick(t + 0.3).flags & stream.STALE


def test_presses_toggle_start_and_request_calibration():
    calls = []

    class Fake:
        active = False

        def request_calibration(self):
            calls.append("c")

        def set_active(self, a):
            self.active = a
            calls.append(a)

    fake, presses = Fake(), Presses()
    presses(fake, 4, 7)  # the first packet sets the base
    presses(fake, 5, 7)
    presses(fake, 5, 8)
    presses(fake, 5, 10)  # two presses between packets: no change
    presses(fake, 5, 11)
    assert calls == ["c", True, False]
