"""The servo model of every policy: the XC330-T288-T identification that the robot's simulator also loads (the
microban repository's tools/actuator_id/xc330_params.json, the same bytes), in bam.mjlab's actuator with the static
friction of bam's own simulator."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from bam.mjlab import BamActuator
from mjlab.actuator.actuator import ActuatorCmd

from mjlab_microban.policy_contract import SELF_TEST_MAX_JOINT_SPEED_RAD_S
from mjlab_microban.robot.microban_constants import actuators
from mjlab_microban.robot.xc330_actuator import XC330BamActuator

PARAMS = Path(__file__).resolve().parents[1] / "src" / "mjlab_microban" / "robot" / "xc330_params.json"
PARAMS_SHA256 = "d4206f02f31ac817804ae0785fcb94849388ce3f42e8987195b7e179683468ce"
INERTIA = 0.3 * 0.1**2 + 0.00163  # the bench's 0.3 kg at 0.1 m plus the armature [kg m^2]


def _torque(cls, gravity, target=0.0, pos=0.0, vel=0.0, drop_gain=None, prev_motor=0.0):
    """One actuator's torque on one joint, one environment per gravity torque on the joint (-qfrc_bias)."""
    gravity = torch.as_tensor(gravity, dtype=torch.float32).reshape(-1, 1)
    n = gravity.shape[0]

    def column(value):
        return torch.as_tensor(value, dtype=torch.float32).reshape(-1, 1).expand(n, 1).clone()

    act = cls(actuators, None, [0], ["joint"])
    act._data = SimpleNamespace(qfrc_bias=-gravity)
    act._mjwarp_model = SimpleNamespace(dof_invweight0=torch.tensor([1.0 / INERTIA]))
    act._dof_ids, act._dt, act._device = torch.tensor([0]), 0.005, "cpu"
    act.vin_tensor = torch.full((n, 1), 11.1)
    act.vin_drop_gain = None if drop_gain is None else column(drop_gain)
    act._prev_motor_torque = column(prev_motor)
    act.kp_scale, act.kd_scale = torch.ones(n, 1), torch.ones(n, 1)
    zeros = torch.zeros(n, 1)
    return act.compute(ActuatorCmd(position_target=column(target), velocity_target=zeros, effort_target=zeros,
                                   pos=column(pos), vel=column(vel)))


def test_every_policy_trains_on_the_shared_identification():
    assert hashlib.sha256(PARAMS.read_bytes()).hexdigest() == PARAMS_SHA256
    assert Path(actuators.json_path).resolve() == PARAMS.resolve()
    params = json.loads(PARAMS.read_text())
    model = actuators.build(None, [0], ["joint"])._bam_model
    assert {name: p.value for name, p in model.get_parameters().items()} == {
        name: params[name] for name in model.get_parameters()
    }
    assert model.actuator.get_extra_inertia() == params["armature"] and model.actuator.kp == 125
    # The self-test's joint-speed bound is this servo's no-load speed at a full pack.
    assert SELF_TEST_MAX_JOINT_SPEED_RAD_S == math.ceil(10 * max(actuators.vin_range) / params["kt"]) / 10


def test_static_friction_holds_against_gravity():
    # Without gravity the actuator is bam.mjlab's.
    g = torch.Generator().manual_seed(0)
    states = [torch.rand(256, generator=g) * 2 - 1 for _ in range(4)]
    target, pos, vel, prev = states[0] * 2, states[1] * 2, states[2] * 8, states[3]
    for drop_gain in (None, 0.2):
        ours = _torque(XC330BamActuator, torch.zeros(256), target, pos, vel, drop_gain, prev)
        theirs = _torque(BamActuator, torch.zeros(256), target, pos, vel, drop_gain, prev)
        assert torch.allclose(ours, theirs, atol=1e-6)
    # A joint at rest, no torque from the motor: the friction opposes gravity up to the static budget
    # friction_base + friction_stribeck + |gravity| (load_friction_external + load_friction_external_stribeck).
    p = json.loads(PARAMS.read_text())
    gravity = torch.tensor([-0.2, -0.05, -0.005, 0.0, 0.005, 0.05, 0.2])
    budget = (p["friction_base"] + p["friction_stribeck"]
              + gravity.abs() * (p["load_friction_external"] + p["load_friction_external_stribeck"]))
    friction = _torque(XC330BamActuator, gravity)[:, 0]
    assert torch.allclose(friction, -torch.clamp(gravity, -budget, budget), atol=1e-7)
