"""Pass line of every check of the walking direction: the walking reward's own verdict.

The walking and PICO base rewards score the twist with one term, the
twist-ratio reward (tasks/microban_twist_ratio_mdp.py): ``(1 + speed) / 2 *
exp(-error)``, which is 1/2 for standing still on every moving command and 1
for standing still on the standing command.  A check judges the motion the
way that reward does, so it never fails a motion the reward prefers and
never passes one the reward rejects:

* a moving command passes when the reward of the measured mean twist is
  higher than standing still on that command (1/2 for every checked
  command): the robot moved along the command more than it moved off it.  This one line holds for every command, feasible or
  not (the robot reaches about 0.2 m/s forward, 0.11 backward); a per-axis
  minimum or an angle limit would fail motions the reward prefers (its best
  answer to an infeasible diagonal command can be 24 deg off the command ray
  and give up an axis), and a fixed speed fraction is not in the reward.
* a standing command passes when the measured drift costs less than walking
  at the smallest command the checks ask for (0.1 m/s forward): its reward
  is at least that of moving at that command on a standing command
  (``exp(-1/7)``), i.e. the normalized drift ``|v / (0.7, 0.3, 1.5)|``
  is below 1/7 (0.1 m/s forward alone, 0.043 m/s lateral, 0.21 rad/s yaw).
  The reward prefers standing still; below this line the robot is not
  following any command the checks give.

The twist is the mean over the check's measurement window (it averages the
gait sway, as the reward's 0.5 s filter does); the motion no command asks for
(vertical, roll and pitch rates) is left out, since a fall is checked
separately.  The checks that read the body frame (the 9x300 probes and the
PICO evaluations) see the twist of the HOME-levelled frame scaled by
cos(HOME trunk pitch) (0.985 at the forward-lean 10 deg); every value here is
computed in float64 on the CPU, so a recomputation reproduces it exactly.

The measured joints may overshoot the soft limits by
``teleop_v12_safety.ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD`` (0.25 rad,
user decision 2026-10-07) in every walking check and the PICO judgment.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch

from mjlab_microban.tasks.microban_twist_ratio_mdp import (
    TWIST_RATIO_AXIS_SCALE,
    twist_ratio_reward,
)

TWIST_PASS_LINE_REVISION = "twist_ratio_reward_beats_standing_still_v2"
# The reward of standing still on every checked moving command (n >= eps).
STANDING_STILL_VALUE = 0.5
# The smallest command any walking check asks for (v_x m/s, v_y m/s, w_z rad/s).
SMALLEST_CHECKED_COMMAND = (0.1, 0.0, 0.0)


def twist_value(command: Sequence[float], twist: Sequence[float]) -> float:
    """The twist-ratio reward of a mean twist (v_x, v_y, w_z) for a command."""

    c = torch.tensor([[float(x) for x in command]], dtype=torch.float64)
    v = torch.tensor([[float(x) for x in twist]], dtype=torch.float64)
    if c.shape != (1, 3) or v.shape != (1, 3):
        raise ValueError("a twist and its command are (v_x, v_y, w_z)")
    if not bool(torch.isfinite(v).all()):
        return float("nan")
    return float(twist_ratio_reward(c, v, uncommanded=None, uncommanded_scale=None)[0])


# Standing command: the drift must earn at least what moving at the smallest
# checked command earns on it (normalized drift below |that command|).
STANDING_DRIFT_VALUE_MIN = twist_value((0.0, 0.0, 0.0), SMALLEST_CHECKED_COMMAND)


def is_standing(command: Sequence[float]) -> bool:
    return all(float(x) == 0.0 for x in command)


def twist_line(command: Sequence[float]) -> float:
    """The value a mean twist must exceed (moving: standing still on it) or reach (standing)."""

    return STANDING_DRIFT_VALUE_MIN if is_standing(command) else twist_value(command, (0.0, 0.0, 0.0))


def twist_passes(command: Sequence[float], twist: Sequence[float]) -> bool:
    value = twist_value(command, twist)
    if not math.isfinite(value):
        return False
    if is_standing(command):
        return value >= STANDING_DRIFT_VALUE_MIN
    return value > twist_line(command)


def twist_judgment(command: Sequence[float], twist: Sequence[float]) -> dict[str, Any]:
    """One check's record: the value, the line it is held to and the verdict."""

    return {
        "command": [float(x) for x in command],
        "mean_twist": [float(x) for x in twist],
        "value": twist_value(command, twist),
        "line": twist_line(command),
        "passed": twist_passes(command, twist),
    }


def twist_pass_line_record() -> dict[str, Any]:
    """The pass line as reports record it (a report with another line is stale)."""

    return {
        "revision": TWIST_PASS_LINE_REVISION,
        "axis_scale": list(TWIST_RATIO_AXIS_SCALE),
        "moving_command_value_above": STANDING_STILL_VALUE,
        "standing_command_value_min": STANDING_DRIFT_VALUE_MIN,
        "smallest_checked_command": list(SMALLEST_CHECKED_COMMAND),
    }


__all__ = [
    "SMALLEST_CHECKED_COMMAND",
    "STANDING_DRIFT_VALUE_MIN",
    "STANDING_STILL_VALUE",
    "TWIST_PASS_LINE_REVISION",
    "is_standing",
    "twist_judgment",
    "twist_line",
    "twist_pass_line_record",
    "twist_passes",
    "twist_value",
]
