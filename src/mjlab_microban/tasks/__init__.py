# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

from mjlab.tasks.registry import register_mjlab_task

from mjlab_microban.tasks.microban_getup_env_cfg import (
    MicrobanGetupRlCfg,
    make_microban_getup_env_cfg,
)
from mjlab_microban.tasks.microban_getup_runner import MicrobanGetupOnPolicyRunner
from mjlab_microban.tasks.microban_ik_tracking_v13_arm_overlay import (
    MICROBAN_IK_TRACKING_V13_ARM_OVERLAY_TASK_ID,
    MicrobanIkTrackingV13ArmOverlayRlCfg,
    make_microban_ik_tracking_v13_arm_overlay_env_cfg,
)
from mjlab_microban.tasks.microban_ik_tracking_v13_arm_overlay_runner import (
    MicrobanIkTrackingV13ArmOverlayOnPolicyRunner,
)
from mjlab_microban.tasks.microban_velocity_env_cfg import (
    make_microban_velocity_env_cfg,
    MicrobanVelocityRlCfg,
)
from mjlab_microban.tasks.microban_velocity_runner import MicrobanVelocityOnPolicyRunner
from mjlab_microban.wholebody.env_cfg import TASK_ID as WHOLEBODY_TASK_ID
from mjlab_microban.wholebody.env_cfg import WholebodyRlCfg, make_wholebody_env_cfg
from mjlab_microban.wholebody.runner import WholebodyOnPolicyRunner

register_mjlab_task(
    task_id="Mjlab-Velocity-Microban",
    env_cfg=make_microban_velocity_env_cfg(),
    play_env_cfg=make_microban_velocity_env_cfg(play=True),
    rl_cfg=MicrobanVelocityRlCfg,
    # Binds the curriculum's update clock and stamps checkpoints with the
    # training HOME (another HOME is refused on load).
    runner_cls=MicrobanVelocityOnPolicyRunner,
)

# Get-up: one training run with step-scheduled switches (IMU latency at 2500,
# calm refinement at 4000, low effort at 10000, pushes at 15000; see
# mjlab_microban/schedules.py and docs/getup_training_export.md).
register_mjlab_task(
    task_id="Mjlab-Getup-Microban",
    env_cfg=make_microban_getup_env_cfg(),
    play_env_cfg=make_microban_getup_env_cfg(play=True),
    rl_cfg=MicrobanGetupRlCfg,
    runner_cls=MicrobanGetupOnPolicyRunner,
)
# IK tracking v13: arms driven from outside, legs released for foot targets, one run.
register_mjlab_task(
    task_id=MICROBAN_IK_TRACKING_V13_ARM_OVERLAY_TASK_ID,
    env_cfg=make_microban_ik_tracking_v13_arm_overlay_env_cfg(),
    play_env_cfg=make_microban_ik_tracking_v13_arm_overlay_env_cfg(play=True),
    rl_cfg=MicrobanIkTrackingV13ArmOverlayRlCfg,
    runner_cls=MicrobanIkTrackingV13ArmOverlayOnPolicyRunner,
)
# Whole-body motion tracking (Teleopit's task on Microban; src/mjlab_microban/wholebody).
register_mjlab_task(
    task_id=WHOLEBODY_TASK_ID,
    env_cfg=make_wholebody_env_cfg(),
    play_env_cfg=make_wholebody_env_cfg(play=True),
    rl_cfg=WholebodyRlCfg,
    runner_cls=WholebodyOnPolicyRunner,
)
