"""Training lengths and switch updates of the three policies, in one place.

Every number is a PPO update (iteration) of one training run.  The task
configs, the PICO adapter, the evaluators, the exporters and the pipeline
(scripts/retrain_all_for_home.py) read them from here; under
``MICROBAN_SCHEDULE_SCALE`` (dry runs) every value is scaled
(``scaled``).  This module imports nothing, so the pipeline and the HOME
contracts can read it without loading mjlab.

Every run trains its ``*_TOTAL_UPDATES`` from scratch and ends with
``model_<total - 1>``, the checkpoint the pipeline judges.
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

# Get-up: IMU latency, calm refinement, low effort, then pushes.  The pushes
# start after the low-effort stage has released the arms from their stops
# (5000 updates; a push-free effort stage releases them 3000-4500 updates in)
# and last 3000 updates.
GETUP_SCHEDULE = {
    "imu_delay": scaled(2500),
    "refine": scaled(4000),
    "effort": scaled(10000),
    "push": scaled(15000),
}
GETUP_TOTAL_UPDATES = scaled(18000)

# PICO: the critic of the frozen walker's adapter warms up for
# PICO_WARMUP_UPDATES (no actor column trains before hand targets open), then
# hands (tightened after 1500), feet (3000 after the hands, tightened 2000
# later) and 3000 more updates with everything active and tightened.
PICO_STEPS_PER_UPDATE = 24
PICO_WARMUP_UPDATES = 1000
PICO_CRITIC_WARMUP = scaled(PICO_WARMUP_UPDATES)
PICO_SCHEDULE = {
    "hand": PICO_CRITIC_WARMUP,
    "hand_tighten": scaled(PICO_WARMUP_UPDATES + 1500),
    "foot": scaled(PICO_WARMUP_UPDATES + 3000),
    "foot_tighten": scaled(PICO_WARMUP_UPDATES + 5000),
}
PICO_TOTAL_UPDATES = scaled(PICO_WARMUP_UPDATES + 8000)
# Recorded in every PICO checkpoint and package: the adapter's gradient
# schedule (which actor columns train after which update).
PICO_ADAPTER_SCHEDULE_REVISION = (
    f"freeze_extra_to{PICO_SCHEDULE['hand']}_then_hmd_hand_to{PICO_SCHEDULE['foot']}"
    f"_then_all_v2"
)


def pico_schedule_record() -> dict[str, int]:
    """The PICO curriculum as the package records it (pico_curriculum_json)."""

    return {
        "critic_warmup": PICO_CRITIC_WARMUP,
        "hand_start": PICO_SCHEDULE["hand"],
        "hand_tighten": PICO_SCHEDULE["hand_tighten"],
        "foot_start": PICO_SCHEDULE["foot"],
        "foot_tighten": PICO_SCHEDULE["foot_tighten"],
        "total": PICO_TOTAL_UPDATES,
    }
