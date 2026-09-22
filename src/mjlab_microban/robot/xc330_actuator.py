# Copyright 2026 Marc Duclusaud

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

kt/R come from xc330_params.json (a real identification against a physical
XC330-T288-T - see tools/actuator_id/ in the main microban repo), loaded via
BamActuatorCfg(json_path=...); this class only needs to exist so
load_model_from_dict can look up "xc330" in bam's actuator registry and
instantiate *something* to attach the loaded parameters to. armature's initial
guess is deliberately kept close to the fitted value from that JSON so a
future re-fit against this class starts near the known-good point.
"""

from bam.actuator import VoltageControlledActuator
from bam.parameter import Parameter
from bam.testbench import Testbench

XC330_ENCODER_COUNTS_PER_REV = 4096  # ROBOTIS e-manual: 4096 pulse/rev, same as XL330
XC330_KP_DIVISOR = 256  # ASSUMED same as XL330 (same X-series control table/firmware gen) - unverified
XC330_PWM_LIMIT = 885  # ROBOTIS e-manual control table default, same value as XL330


class XC330Actuator(VoltageControlledActuator):
    """Represents a Dynamixel XC330-T288-T actuator."""

    def __init__(self, testbench_class: Testbench):
        import numpy as np

        super().__init__(
            testbench_class,
            vin=11.1,  # 3S nominal - overridden by BamActuatorCfg.vin/vin_range at the mjlab layer
            kp=400,
            error_gain=(XC330_ENCODER_COUNTS_PER_REV / (2 * np.pi))
            / (XC330_KP_DIVISOR * XC330_PWM_LIMIT),
            max_pwm=1.0,
        )

    def initialize(self):
        self.model.kt = Parameter(1.043, 0.5, 2.5)  # centered on tools/actuator_id/xc330_params.json
        self.model.R = Parameter(10.007, 1.5, 15.0)  # centered on tools/actuator_id/xc330_params.json
        self.model.armature = Parameter(0.0083, 0.0001, 0.05)

    def get_extra_inertia(self) -> float:
        return self.model.armature.value
