"""Training lengths and switch updates of the three policies, in one place.

Every number is a PPO update (iteration) of one training run.  The task
configs, the IK tracking adapter, the evaluators, the exporters and the
pipeline (scripts/retrain_all_for_home.py) read them from here; under
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
    # The floor that gives sideways (tasks/floor_plate.py) under two thirds of
    # the environments, once the get-up stands on the rigid floor (600-1000
    # updates): on the plate from the first update, runs found standing at 600
    # or not by 6600, the same seed either way.
    "plate": scaled(3000),
    "refine": scaled(4000),
    "effort": scaled(10000),
    "push": scaled(15000),
}
GETUP_TOTAL_UPDATES = scaled(18000)

# IK tracking: the critic of the frozen walker's adapter warms up for
# IK_TRACKING_WARMUP_UPDATES (no actor column trains before the arms move),
# then the externally driven arms and the moving HMD (with their columns), feet
# 3000 updates later (tightened 2000 later) and 9000 more updates with
# everything active and tightened.
IK_TRACKING_STEPS_PER_UPDATE = 24
IK_TRACKING_WARMUP_UPDATES = 1000
IK_TRACKING_CRITIC_WARMUP = scaled(IK_TRACKING_WARMUP_UPDATES)
IK_TRACKING_SCHEDULE = {
    "arm": IK_TRACKING_CRITIC_WARMUP,
    "foot": scaled(IK_TRACKING_WARMUP_UPDATES + 3000),
    "foot_tighten": scaled(IK_TRACKING_WARMUP_UPDATES + 5000),
}
IK_TRACKING_TOTAL_UPDATES = scaled(IK_TRACKING_WARMUP_UPDATES + 14000)
# Recorded in every IK tracking checkpoint and package: the adapter's gradient
# schedule (which actor columns train after which update; the residual MLP
# trains with the HMD and arm columns).
IK_TRACKING_ADAPTER_SCHEDULE_REVISION = (
    f"freeze_extra_residual_to{IK_TRACKING_SCHEDULE['arm']}"
    f"_then_hmd_arm_residual_to{IK_TRACKING_SCHEDULE['foot']}_then_all_v4"
)


def ik_tracking_schedule_record() -> dict[str, int]:
    """The IK tracking curriculum as the package records it (pico_curriculum_json)."""

    return {
        "critic_warmup": IK_TRACKING_CRITIC_WARMUP,
        "arm_start": IK_TRACKING_SCHEDULE["arm"],
        "foot_start": IK_TRACKING_SCHEDULE["foot"],
        "foot_tighten": IK_TRACKING_SCHEDULE["foot_tighten"],
        "total": IK_TRACKING_TOTAL_UPDATES,
    }
