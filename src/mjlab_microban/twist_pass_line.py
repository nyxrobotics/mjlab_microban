"""Pass line of every check of the walking direction: fixed signed-response minimums.

Walking and PICO track velocity with their initial reward terms again (user
decision 2026-10-08: "とりあえず歩行の報酬は今までうまく言ってた速度追従方法に戻し"),
so the checks of the walking direction go back to the fixed minimums they
had with those rewards (before c0d8d49):

* a single-axis command passes when the mean twist moves along that axis the
  commanded way by at least a fixed share of the command: 40 % forward, 20 %
  backward, 20 % lateral, 40 % yaw.  These reproduce the former tables: the
  9x300 probe (0.1 / 0.2 m/s forward 0.04 / 0.08, backward 0.02 / 0.04,
  lateral 0.1 m/s 0.02, yaw 0.5 rad/s 0.2) and the walk check's single-axis
  minimums (0.2 m/s forward 0.08, backward 0.04, lateral 0.02, yaw 0.2);
* a command on several axes (the PICO tracking judgment's mixed scenarios)
  passes when every commanded axis moves the commanded way by at least the
  former fixed minimums: 0.04 m/s, 0.02 m/s and 0.2 rad/s;
* a standing command passes when the drift stays within 0.05 m/s forward,
  0.05 m/s lateral and 0.2 rad/s yaw (the former walk check's standing
  limits).

The twist is the mean over the check's measurement window.  The names of the
checks that use it (``twist_beats_standing`` in the PICO reports) are kept.
The measured joints may overshoot the soft limits by
``teleop_v12_safety.ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD`` (0.25 rad,
user decision 2026-10-07) in every walking check and the PICO judgment.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

TWIST_PASS_LINE_REVISION = "signed_response_minimums_v1"
AXES = ("vx_m_s", "vy_m_s", "yaw_rad_s")
# Single-axis commands: the share of the command the mean twist must reach.
SINGLE_AXIS_MINIMUM_SHARE = {"forward": 0.4, "backward": 0.2, "lateral": 0.2, "yaw": 0.4}
# Commands on several axes: every commanded axis at least this (m/s, m/s, rad/s).
MIXED_AXIS_MINIMUM = (0.04, 0.02, 0.20)
# Standing command: the largest drift per axis (m/s, m/s, rad/s).
STANDING_DRIFT_MAX = (0.05, 0.05, 0.20)


def is_standing(command: Sequence[float]) -> bool:
    return all(float(x) == 0.0 for x in command)


def axis_minimums(command: Sequence[float]) -> dict[int, float]:
    """The signed response each commanded axis must reach, by axis index."""

    values = [float(x) for x in command]
    if len(values) != 3:
        raise ValueError("a command is (v_x, v_y, w_z)")
    axes = [i for i, value in enumerate(values) if value != 0.0]
    if len(axes) == 1:
        i = axes[0]
        kind = ("forward" if values[0] > 0 else "backward") if i == 0 else ("lateral" if i == 1 else "yaw")
        return {i: round(SINGLE_AXIS_MINIMUM_SHARE[kind] * abs(values[i]), 9)}
    return {i: MIXED_AXIS_MINIMUM[i] for i in axes}


def twist_judgment(command: Sequence[float], twist: Sequence[float]) -> dict[str, Any]:
    """One check's record: the signed response per commanded axis, its minimum and the verdict."""

    c = [float(x) for x in command]
    v = [float(x) for x in twist]
    if len(c) != 3 or len(v) != 3:
        raise ValueError("a twist and its command are (v_x, v_y, w_z)")
    finite = all(math.isfinite(x) for x in v)
    recorded = [x if math.isfinite(x) else None for x in v]  # JSON has no NaN
    if is_standing(c):
        passed = finite and all(abs(v[i]) <= STANDING_DRIFT_MAX[i] for i in range(3))
        return {"command": c, "mean_twist": recorded, "standing_drift_max": list(STANDING_DRIFT_MAX),
                "passed": passed}
    minimums = axis_minimums(c)
    signed = {AXES[i]: (v[i] * (1.0 if c[i] > 0 else -1.0) if finite else None) for i in minimums}
    minimum = {AXES[i]: value for i, value in minimums.items()}
    passed = finite and all(signed[axis] >= minimum[axis] for axis in minimum)
    return {"command": c, "mean_twist": recorded, "signed_response": signed,
            "minimum_signed_response": minimum, "passed": passed}


def twist_passes(command: Sequence[float], twist: Sequence[float]) -> bool:
    return bool(twist_judgment(command, twist)["passed"])


def worst_margin(command: Sequence[float], twist: Sequence[float]) -> float:
    """The smallest signed response minus its minimum (a moving command; nan if not finite)."""

    judgment = twist_judgment(command, twist)
    if "signed_response" not in judgment:
        raise ValueError("worst_margin is for moving commands")
    if not all(math.isfinite(float(x)) for x in twist):
        return float("nan")
    margins = [judgment["signed_response"][a] - judgment["minimum_signed_response"][a]
               for a in judgment["minimum_signed_response"]]
    return min(margins)


def twist_pass_line_record() -> dict[str, Any]:
    """The pass line as reports record it (a report with another line is stale)."""

    return {
        "revision": TWIST_PASS_LINE_REVISION,
        "single_axis_minimum_share": dict(SINGLE_AXIS_MINIMUM_SHARE),
        "mixed_axis_minimum": list(MIXED_AXIS_MINIMUM),
        "standing_drift_max": list(STANDING_DRIFT_MAX),
    }


__all__ = [
    "AXES",
    "MIXED_AXIS_MINIMUM",
    "SINGLE_AXIS_MINIMUM_SHARE",
    "STANDING_DRIFT_MAX",
    "TWIST_PASS_LINE_REVISION",
    "axis_minimums",
    "is_standing",
    "twist_judgment",
    "twist_pass_line_record",
    "twist_passes",
    "worst_margin",
]
