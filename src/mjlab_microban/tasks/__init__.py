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
from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release import (
    MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_TASK_ID,
    MicrobanTeleopV12HandPoseReleaseRlCfg,
    make_microban_teleop_v12_hand_pose_release_env_cfg,
)
from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release_runner import (
    MicrobanTeleopV12HandPoseReleaseOnPolicyRunner,
)
from mjlab_microban.tasks.microban_velocity_env_cfg import (
    MicrobanVelocityRlCfg,
    make_microban_velocity_env_cfg,
)
from mjlab_microban.tasks.microban_velocity_runner import MicrobanVelocityOnPolicyRunner

register_mjlab_task(
    task_id="Mjlab-Velocity-Microban",
    env_cfg=make_microban_velocity_env_cfg(),
    play_env_cfg=make_microban_velocity_env_cfg(play=True),
    rl_cfg=MicrobanVelocityRlCfg,
    # Stamps checkpoints with the training HOME, refuses another HOME on load.
    runner_cls=MicrobanVelocityOnPolicyRunner,
)

# Get-up: one training run with step-scheduled switches (IMU latency at 2500,
# calm refinement at 4000, low effort and pushes at 10000; see
# microban_getup_env_cfg.GETUP_SCHEDULE and docs/getup_training_export.md).
register_mjlab_task(
    task_id="Mjlab-Getup-Microban",
    env_cfg=make_microban_getup_env_cfg(),
    play_env_cfg=make_microban_getup_env_cfg(play=True),
    rl_cfg=MicrobanGetupRlCfg,
    runner_cls=MicrobanGetupOnPolicyRunner,
)
# PICO v12: the active-hand arm pose-release recipe (every chain trains it;
# scripts/retrain_all_for_home.py trains it in one run).
register_mjlab_task(
    task_id=MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_TASK_ID,
    env_cfg=make_microban_teleop_v12_hand_pose_release_env_cfg(),
    play_env_cfg=make_microban_teleop_v12_hand_pose_release_env_cfg(play=True),
    rl_cfg=MicrobanTeleopV12HandPoseReleaseRlCfg,
    runner_cls=MicrobanTeleopV12HandPoseReleaseOnPolicyRunner,
)
