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
from mjlab_microban.tasks.microban_teleop_v13_arm_overlay import (
    MICROBAN_TELEOP_V13_ARM_OVERLAY_TASK_ID,
    MicrobanTeleopV13ArmOverlayRlCfg,
    make_microban_teleop_v13_arm_overlay_env_cfg,
)
from mjlab_microban.tasks.microban_teleop_v13_arm_overlay_runner import (
    MicrobanTeleopV13ArmOverlayOnPolicyRunner,
)
from mjlab_microban.tasks.microban_velocity_env_cfg import (
    make_microban_velocity_env_cfg,
    MicrobanVelocityRlCfg,
)
from mjlab_microban.tasks.microban_velocity_runner import MicrobanVelocityOnPolicyRunner

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
# PICO v13: arms driven from outside, legs released for foot targets, one run.
register_mjlab_task(
    task_id=MICROBAN_TELEOP_V13_ARM_OVERLAY_TASK_ID,
    env_cfg=make_microban_teleop_v13_arm_overlay_env_cfg(),
    play_env_cfg=make_microban_teleop_v13_arm_overlay_env_cfg(play=True),
    rl_cfg=MicrobanTeleopV13ArmOverlayRlCfg,
    runner_cls=MicrobanTeleopV13ArmOverlayOnPolicyRunner,
)
