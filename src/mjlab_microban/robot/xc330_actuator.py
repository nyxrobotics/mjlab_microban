# Copyright 2026 nyxrobotics

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""bam actuator definition for the XC330-T288-T, which this bam branch doesn't
ship (only xl330, xl320, mx106, mx64, erob80:50/100) - mirrors XL330Actuator's
structure (bam/dynamixel/actuator.py). Unlike the PyPI better-actuator-models
package, this branch's VoltageControlledActuator.__init__ has no max_current
parameter - current clipping is applied at the BamActuatorCfg/mjlab layer via
its own max_current field instead (see microban_constants.py).

kt, R, the armature and the friction come from xc330_params.json (the
identification of a physical XC330-T288-T on the pendulum bench of
tools/actuator_id/ in the main microban repo, which carries the same file),
loaded via XC330ActuatorCfg(json_path=...); XC330Actuator only needs to exist
so load_model_from_dict can look up "xc330" in bam's actuator registry and
instantiate *something* to attach the loaded parameters to. Its initial
guesses are the values of that JSON so a future re-fit against this class
starts near them.

XC330ActuatorCfg builds bam.mjlab's actuator with one correction: the torque
that would stop the joint, which caps the static friction, takes gravity with
the sign of bam's own simulator (bam.simulate, which the identification fits).
bam.mjlab subtracts it, so its friction pushes a joint held against gravity
along gravity instead of holding it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from bam.actuator import VoltageControlledActuator
from bam.mjlab import BamActuator, BamActuatorCfg
from bam.parameter import Parameter
from bam.testbench import Testbench
from mjlab.actuator.actuator import ActuatorCmd

XC330_ENCODER_COUNTS_PER_REV = 4096  # ROBOTIS e-manual: 4096 pulse/rev, same as XL330
XC330_KP_DIVISOR = 256  # ASSUMED same as XL330 (same X-series control table/firmware gen) - unverified
XC330_PWM_LIMIT = 885  # ROBOTIS e-manual control table default, same value as XL330


class XC330Actuator(VoltageControlledActuator):
    """Represents a Dynamixel XC330-T288-T actuator."""

    def __init__(self, testbench_class: Testbench):
        super().__init__(
            testbench_class,
            vin=11.1,  # 3S nominal - overridden by BamActuatorCfg.vin/vin_range at the mjlab layer
            kp=400,
            error_gain=(XC330_ENCODER_COUNTS_PER_REV / (2 * np.pi))
            / (XC330_KP_DIVISOR * XC330_PWM_LIMIT),
            max_pwm=1.0,
        )

    def initialize(self):
        # Centred on xc330_params.json; R's bound is the identification's.
        self.model.kt = Parameter(1.490, 0.5, 2.5)
        self.model.R = Parameter(13.318, 1.5, 40.0)
        self.model.armature = Parameter(0.00163, 0.0001, 0.05)

    def get_extra_inertia(self) -> float:
        return self.model.armature.value


class XC330BamActuator(BamActuator):
    """bam.mjlab's BamActuator (rhoban/bam@doc 0007411) with bam.simulate's stopping torque."""

    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        bam = self._bam_model
        act = bam.actuator
        kt, R = bam.kt.value, bam.R.value

        # Supply voltage, dropped by the previous step's motor torques.
        vin = self.vin_tensor
        if self.vin_drop_gain is not None:
            vin = vin - self.vin_drop_gain * self._prev_motor_torque.abs().sum(dim=-1, keepdim=True)
            if self.cfg.vin_min is not None:
                vin = torch.clamp(vin, min=self.cfg.vin_min)

        # Firmware P control -> duty cycle -> voltage; DC motor with back-EMF; firmware current limit.
        duty = (cmd.position_target - cmd.pos) * act.kp * self.kp_scale * act.error_gain
        duty = torch.clamp(duty, -act.max_pwm, act.max_pwm)
        vel = cmd.vel
        motor_torque = kt * vin * duty / R - kt**2 * vel * self.kd_scale / R
        if self.cfg.max_current is not None:
            limit = self.cfg.max_current * kt
            motor_torque = torch.clamp(motor_torque, -limit, limit)

        # Gravity and Coriolis on the joint.
        qfrc_bias = self._data.qfrc_bias
        if not isinstance(qfrc_bias, torch.Tensor):
            qfrc_bias = torch.as_tensor(qfrc_bias, device=self._device)
        external_torque = -qfrc_bias[:, self._dof_ids]

        # BAM friction budget (bam.mjlab's).
        abs_vel = torch.abs(vel)
        if bam.stribeck:
            stribeck_coeff = torch.exp(-torch.pow(abs_vel / bam.dtheta_stribeck.value, bam.alpha.value))
        else:
            stribeck_coeff = torch.zeros_like(vel)
        budget = self._compute_friction_budget(motor_torque, external_torque, stribeck_coeff)
        budget = budget + bam.friction_viscous.value * abs_vel

        # Static friction (BAM Algorithm 1): at most the torque that stops the joint in one step,
        # (I / dt) vel + motor + external, as in bam.simulate.
        dof_invweight = self._mjwarp_model.dof_invweight0
        if not isinstance(dof_invweight, torch.Tensor):
            dof_invweight = torch.as_tensor(dof_invweight, device=self._device)
        if dof_invweight.ndim == 1:
            inertia = 1.0 / dof_invweight[self._dof_ids].unsqueeze(0)
        else:
            inertia = 1.0 / dof_invweight[:, self._dof_ids]
        tau_stop = (inertia / self._dt) * vel + motor_torque + external_torque
        friction_torque = -torch.sign(tau_stop) * torch.minimum(torch.abs(tau_stop), budget)

        self._prev_motor_torque = motor_torque.detach()
        return motor_torque + friction_torque


@dataclass(kw_only=True)
class XC330ActuatorCfg(BamActuatorCfg):
    """BamActuatorCfg whose actuator is XC330BamActuator."""

    def build(self, entity, target_ids: list[int], target_names: list[str]) -> XC330BamActuator:
        return XC330BamActuator(self, entity, target_ids, target_names)
