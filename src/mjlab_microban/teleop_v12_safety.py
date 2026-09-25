"""Shared numeric safety tolerances for the contract-v12 teleop pipeline.

The measured joint state may transiently overshoot a soft limit by at most five
degrees. This allowance never widens commanded joint targets: target-limit
excess remains a numerical-tolerance-only check.
"""

from __future__ import annotations

import math

ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_DEG = 5.0
ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD = math.radians(
    ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_DEG
)
COMMANDED_TARGET_SOFT_LIMIT_EXCESS_MAX_RAD = 1.0e-7
