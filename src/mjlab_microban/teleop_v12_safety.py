"""Shared numeric safety tolerances of the walking and PICO judgments.

The measured joint state may transiently overshoot a soft limit by at most
0.25 rad.  The walking reward does not charge the measured joints beyond the
soft limits, so the judgments hold them to a bound, not to the limits
themselves.  One value for the walker's judgment
(pipeline.steps.probe_verdict), the PICO source gate and every PICO
evaluation.
"""

from __future__ import annotations

ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD = 0.25
