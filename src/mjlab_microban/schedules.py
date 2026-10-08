"""Training lengths and switch updates of the policies, in one place.

Every number is a PPO update (iteration) of one training run.  The task
configs read them from here; under ``MICROBAN_SCHEDULE_SCALE`` (dry runs)
every value is scaled (``scaled``).  This module imports nothing, so any
tool can read it without loading mjlab.
"""

from __future__ import annotations

import os

SCHEDULE_SCALE_ENV = "MICROBAN_SCHEDULE_SCALE"


def schedule_scale() -> float:
    """The dry-run schedule scale (``MICROBAN_SCHEDULE_SCALE``, default 1)."""

    raw = os.environ.get(SCHEDULE_SCALE_ENV, "").strip()
    if not raw:
        return 1.0
    value = float(raw)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{SCHEDULE_SCALE_ENV} must be in (0, 1], got {raw!r}")
    return value


def scaled(iteration: int) -> int:
    """``iteration`` under the dry-run schedule scale (at least 1 if positive)."""

    scale = schedule_scale()
    if scale == 1.0 or iteration == 0:
        return iteration
    return max(1, round(iteration * scale))


# Walking: one stage (wider forward/yaw commands, no-stepping penalty).
WALK_WIDEN_UPDATE = scaled(3000)
WALK_TOTAL_UPDATES = scaled(30000)
