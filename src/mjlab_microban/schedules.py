"""Training lengths and switch updates of the three policies, in one place.

Every number is a PPO update (iteration) of one training run.  The task
configs, the PICO adapter, the evaluators, the exporters and the pipeline
(scripts/retrain_all_for_home.py) read them from here; under
``MICROBAN_SCHEDULE_SCALE`` (dry runs) every value is scaled
(``scaled``).  This module imports nothing, so the pipeline and the HOME
contracts can read it without loading mjlab.

Early stop (pipeline monitor): a run may stop once two checks 1000 updates
apart pass, at the earliest at its ``*_MIN_FINAL_UPDATES``: get-up and PICO
once their last stage has lasted half its planned length; walking (no planned
end, ``WALK_MAX_UPDATES`` is a cap) once both checks saw at least 1000 updates
of its only stage.  A walking run that never passes two checks in a row ends
at ``WALK_MAX_UPDATES`` with its best checked checkpoint (the pipeline's rule
in config/pipeline.yaml ``walk.check``).
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
WALK_MAX_UPDATES = scaled(8000)
WALK_MIN_FINAL_UPDATES = WALK_WIDEN_UPDATE + scaled(2000)

# Get-up: IMU latency, calm refinement, low effort with pushes.
GETUP_SCHEDULE = {
    "imu_delay": scaled(2500),
    "refine": scaled(4000),
    "effort_push": scaled(10000),
}
GETUP_TOTAL_UPDATES = scaled(16500)
GETUP_MIN_FINAL_UPDATES = GETUP_SCHEDULE["effort_push"] + (
    GETUP_TOTAL_UPDATES - GETUP_SCHEDULE["effort_push"]
) // 2

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
PICO_MIN_FINAL_UPDATES = PICO_SCHEDULE["foot_tighten"] + (
    PICO_TOTAL_UPDATES - PICO_SCHEDULE["foot_tighten"]
) // 2
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
