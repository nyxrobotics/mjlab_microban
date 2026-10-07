"""Shared numeric safety tolerances of the walking checks and the PICO judgment.

The measured joint state may transiently overshoot a soft limit by at most
0.25 rad (user decision 2026-10-07: "「関節の限界を超えていないかを見るテスト」を
ちょっと緩和してとりあえず突破せよ"; it was 5 deg).  The walking reward does not
charge the measured joints beyond the soft limits, so the checks hold them to
a bound, not to the limits themselves.  This allowance never widens commanded
joint targets: target-limit excess remains a numerical-tolerance-only check.
One value for the walker's checks while training (pipeline.steps.probe_verdict),
the PICO source gate and every PICO evaluation.
"""

from __future__ import annotations

ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD = 0.25
COMMANDED_TARGET_SOFT_LIMIT_EXCESS_MAX_RAD = 1.0e-7
