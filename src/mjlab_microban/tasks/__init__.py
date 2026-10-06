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

# Get-up (v5 action contract at the centered HOME, v6 at the forward-lean
# HOME, "v6_<tag>" at any other; see microban_getup_runner.py). The robot
# policy is trained in two stages:
#   1. Mjlab-Getup-Microban from scratch: HOME-stance reward set ("posture");
#      stands from fallen starts with a HOME stance by ~2000 iterations.
#   2. Mjlab-Getup-Microban-ImuDelay, resumed from stage 1: the same rewards
#      under the walking task's 0-3 tick simulated IMU latency (~500 iters).
register_mjlab_task(
    task_id="Mjlab-Getup-Microban",
    env_cfg=make_microban_getup_env_cfg(reward_set="posture"),
    play_env_cfg=make_microban_getup_env_cfg(play=True, reward_set="posture"),
    rl_cfg=MicrobanGetupRlCfg,
    runner_cls=MicrobanGetupOnPolicyRunner,
)
register_mjlab_task(
    task_id="Mjlab-Getup-Microban-ImuDelay",
    env_cfg=make_microban_getup_env_cfg(reward_set="posture", imu_delay_max_lag=3),
    play_env_cfg=make_microban_getup_env_cfg(play=True, reward_set="posture", imu_delay_max_lag=3),
    rl_cfg=MicrobanGetupRlCfg,
    runner_cls=MicrobanGetupOnPolicyRunner,
)
# Stages 3-5: calm, low-effort, push-tolerant fine-tunes, each resumed from
# the previous stage (see microban_getup_env_cfg._add_calm_rewards).
for _reward_set, _task_id in (
    ("calm_roll", "Mjlab-Getup-Microban-CalmRoll-ImuDelay"),
    ("calm_effort_strong", "Mjlab-Getup-Microban-CalmEffortStrong-ImuDelay"),
    ("calm_push", "Mjlab-Getup-Microban-CalmPush-ImuDelay"),
):
    register_mjlab_task(
        task_id=_task_id,
        env_cfg=make_microban_getup_env_cfg(reward_set=_reward_set, imu_delay_max_lag=3),
        play_env_cfg=make_microban_getup_env_cfg(play=True, reward_set=_reward_set, imu_delay_max_lag=3),
        rl_cfg=MicrobanGetupRlCfg,
        runner_cls=MicrobanGetupOnPolicyRunner,
    )
# The first v4 set that stood, with a wide braced stance ("redesign"): the
# scene scripts/home_pipeline/getup_eval.py evaluates every get-up stage in.
register_mjlab_task(
    task_id="Mjlab-Getup-Microban-Redesign",
    env_cfg=make_microban_getup_env_cfg(reward_set="redesign"),
    play_env_cfg=make_microban_getup_env_cfg(play=True, reward_set="redesign"),
    rl_cfg=MicrobanGetupRlCfg,
    runner_cls=MicrobanGetupOnPolicyRunner,
)
# PICO v12: the active-hand arm pose-release recipe (every chain trains it;
# scripts/train_microban_teleop_v12.sh).
register_mjlab_task(
    task_id=MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_TASK_ID,
    env_cfg=make_microban_teleop_v12_hand_pose_release_env_cfg(),
    play_env_cfg=make_microban_teleop_v12_hand_pose_release_env_cfg(play=True),
    rl_cfg=MicrobanTeleopV12HandPoseReleaseRlCfg,
    runner_cls=MicrobanTeleopV12HandPoseReleaseOnPolicyRunner,
)
