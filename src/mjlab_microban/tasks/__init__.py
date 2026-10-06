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
from mjlab_microban.tasks.microban_teleop_v12_corner_rescue import (
    MICROBAN_TELEOP_V12_CORNER_RESCUE_TASK_ID,
    MicrobanTeleopV12CornerRescueRlCfg,
    make_microban_teleop_v12_corner_rescue_env_cfg,
)
from mjlab_microban.tasks.microban_teleop_v12_corner_rescue_runner import (
    MicrobanTeleopV12CornerRescueOnPolicyRunner,
)
from mjlab_microban.tasks.microban_teleop_v12_final_rescue import (
    MICROBAN_TELEOP_V12_FINAL_RESCUE_TASK_ID,
    MicrobanTeleopV12FinalRescueRlCfg,
    make_microban_teleop_v12_final_rescue_env_cfg,
)
from mjlab_microban.tasks.microban_teleop_v12_final_rescue_runner import (
    MicrobanTeleopV12FinalRescueOnPolicyRunner,
)
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

register_mjlab_task(
    task_id=MICROBAN_TELEOP_V12_CORNER_RESCUE_TASK_ID,
    env_cfg=make_microban_teleop_v12_corner_rescue_env_cfg(),
    play_env_cfg=make_microban_teleop_v12_corner_rescue_env_cfg(play=True),
    rl_cfg=MicrobanTeleopV12CornerRescueRlCfg,
    runner_cls=MicrobanTeleopV12CornerRescueOnPolicyRunner,
)

# Pose-release variant of the 9901->10000 corner rescue (fresh pose-release
# chain's model_9900; registered sampler mixes; saves keep the pose-release
# recipe).
from mjlab_microban.tasks.microban_teleop_v12_corner_rescue import (  # noqa: E402
    MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_CORNER_RESCUE_TASK_ID,
    MicrobanTeleopV12HandPoseReleaseCornerRescueRlCfg,
    make_microban_teleop_v12_hand_pose_release_corner_rescue_env_cfg,
)
from mjlab_microban.tasks.microban_teleop_v12_corner_rescue_runner import (  # noqa: E402
    MicrobanTeleopV12HandPoseReleaseCornerRescueOnPolicyRunner,
)

register_mjlab_task(
    task_id=MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_CORNER_RESCUE_TASK_ID,
    env_cfg=make_microban_teleop_v12_hand_pose_release_corner_rescue_env_cfg(),
    play_env_cfg=make_microban_teleop_v12_hand_pose_release_corner_rescue_env_cfg(
        play=True
    ),
    rl_cfg=MicrobanTeleopV12HandPoseReleaseCornerRescueRlCfg,
    runner_cls=MicrobanTeleopV12HandPoseReleaseCornerRescueOnPolicyRunner,
)

# 14900->14999 final-scenario rescue; the sampler mix is read from
# MICROBAN_V12_FINAL_RESCUE_MIX when the package is imported (launcher-set).
register_mjlab_task(
    task_id=MICROBAN_TELEOP_V12_FINAL_RESCUE_TASK_ID,
    env_cfg=make_microban_teleop_v12_final_rescue_env_cfg(),
    play_env_cfg=make_microban_teleop_v12_final_rescue_env_cfg(play=True),
    rl_cfg=MicrobanTeleopV12FinalRescueRlCfg,
    runner_cls=MicrobanTeleopV12FinalRescueOnPolicyRunner,
)

# Pose-release variant of the 14900->14999 final-scenario rescue (pose-release
# model_14900 of a run whose 14999 gate failed; saves keep the pose-release
# recipe).  The mix is read from MICROBAN_V12_PR_FINAL_RESCUE_MIX at import.
from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release_final_rescue import (  # noqa: E402
    MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_TASK_ID,
    MicrobanTeleopV12HandPoseReleaseFinalRescueRlCfg,
    make_microban_teleop_v12_hand_pose_release_final_rescue_env_cfg,
)
from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release_final_rescue_runner import (  # noqa: E402
    MicrobanTeleopV12HandPoseReleaseFinalRescueOnPolicyRunner,
)

register_mjlab_task(
    task_id=MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_TASK_ID,
    env_cfg=make_microban_teleop_v12_hand_pose_release_final_rescue_env_cfg(),
    play_env_cfg=make_microban_teleop_v12_hand_pose_release_final_rescue_env_cfg(
        play=True
    ),
    rl_cfg=MicrobanTeleopV12HandPoseReleaseFinalRescueRlCfg,
    runner_cls=MicrobanTeleopV12HandPoseReleaseFinalRescueOnPolicyRunner,
)
