# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from mjlab_microban.tasks.microban_getup_env_cfg import (
    MicrobanGetupRlCfg,
    make_microban_getup_env_cfg,
)
from mjlab_microban.tasks.microban_getup_runner import MicrobanGetupOnPolicyRunner
from mjlab_microban.tasks.microban_getup_symmetry import with_getup_symmetry
from mjlab_microban.tasks.microban_policy_export import MicrobanTeleopOnPolicyRunner
from mjlab_microban.tasks.microban_safe_velocity_env_cfg import (
    MICROBAN_SAFE_VELOCITY_TASK_ID,
    MicrobanSafeVelocityRlCfg,
    make_microban_safe_velocity_env_cfg,
)
from mjlab_microban.tasks.microban_safe_velocity_mdp import (
    MicrobanSafeVelocityOnPolicyRunner,
)
from mjlab_microban.tasks.microban_teleop_env_cfg import (
    MicrobanTeleopRlCfg,
    make_microban_teleop_env_cfg,
)
from mjlab_microban.tasks.microban_teleop_v12_corner_rescue import (
    MICROBAN_TELEOP_V12_CORNER_RESCUE_TASK_ID,
    MicrobanTeleopV12CornerRescueRlCfg,
    make_microban_teleop_v12_corner_rescue_env_cfg,
)
from mjlab_microban.tasks.microban_teleop_v12_corner_rescue_runner import (
    MicrobanTeleopV12CornerRescueOnPolicyRunner,
)
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MICROBAN_TELEOP_V12_TASK_ID,
    MicrobanTeleopV12PreviewRlCfg,
    MicrobanTeleopV12RlCfg,
    make_microban_teleop_v12_env_cfg,
    make_microban_teleop_v12_preview_env_cfg,
)
from mjlab_microban.tasks.microban_teleop_v12_preview import (
    MICROBAN_TELEOP_V12_PREVIEW_TASK_ID,
)
from mjlab_microban.tasks.microban_teleop_v12_runner import (
    MicrobanTeleopV12OnPolicyRunner,
    MicrobanTeleopV12PreviewOnPolicyRunner,
)
from mjlab_microban.tasks.microban_teleop_upright_fullbody_env_cfg import (
    MICROBAN_TELEOP_UPRIGHT_FULLBODY_TASK_ID,
    MicrobanTeleopUprightFullbodyRlCfg,
    make_microban_teleop_upright_fullbody_env_cfg,
)
from mjlab_microban.tasks.microban_teleop_upright_fullbody_runner import (
    MicrobanTeleopUprightFullbodyOnPolicyRunner,
)
from mjlab_microban.tasks.microban_tracking_env_cfg import (
    MicrobanTrackingRlCfg,
    make_microban_tracking_env_cfg,
)
from mjlab_microban.tasks.microban_tracking_policy_export import (
    MicrobanTrackingOnPolicyRunner,
)
from mjlab_microban.tasks.microban_velocity_env_cfg import (
    MicrobanVelocityRlCfg,
    make_microban_velocity_env_cfg,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Microban",
    env_cfg=make_microban_velocity_env_cfg(),
    play_env_cfg=make_microban_velocity_env_cfg(play=True),
    rl_cfg=MicrobanVelocityRlCfg,
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id=MICROBAN_SAFE_VELOCITY_TASK_ID,
    env_cfg=make_microban_safe_velocity_env_cfg(),
    play_env_cfg=make_microban_safe_velocity_env_cfg(play=True),
    rl_cfg=MicrobanSafeVelocityRlCfg,
    runner_cls=MicrobanSafeVelocityOnPolicyRunner,
)

# Get-up (v4 action contract, see microban_getup_env_cfg.py). The robot
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
# Experimental stage 3: stop the stage-2 policy trembling (_add_calm_rewards).
for _reward_set, _task_id in (
    ("calm", "Mjlab-Getup-Microban-Calm-ImuDelay"),
    ("calm_strong", "Mjlab-Getup-Microban-CalmStrong-ImuDelay"),
):
    register_mjlab_task(
        task_id=_task_id,
        env_cfg=make_microban_getup_env_cfg(reward_set=_reward_set, imu_delay_max_lag=3),
        play_env_cfg=make_microban_getup_env_cfg(play=True, reward_set=_reward_set, imu_delay_max_lag=3),
        rl_cfg=MicrobanGetupRlCfg,
        runner_cls=MicrobanGetupOnPolicyRunner,
    )
# Reference variants of the same contract: the 09-25 recipe ("v42"), and the
# first v4 set that stood, with a wide braced stance ("redesign").
for _reward_set, _task_id in (
    ("v42", "Mjlab-Getup-Microban-V42"),
    ("redesign", "Mjlab-Getup-Microban-Redesign"),
):
    register_mjlab_task(
        task_id=_task_id,
        env_cfg=make_microban_getup_env_cfg(reward_set=_reward_set),
        play_env_cfg=make_microban_getup_env_cfg(play=True, reward_set=_reward_set),
        rl_cfg=MicrobanGetupRlCfg,
        runner_cls=MicrobanGetupOnPolicyRunner,
    )
# Stage 1 with left/right mirror data augmentation in PPO (HiFAR/HumanUP).
register_mjlab_task(
    task_id="Mjlab-Getup-Microban-Sym",
    env_cfg=make_microban_getup_env_cfg(reward_set="posture"),
    play_env_cfg=make_microban_getup_env_cfg(play=True, reward_set="posture"),
    rl_cfg=with_getup_symmetry(MicrobanGetupRlCfg),
    runner_cls=MicrobanGetupOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tracking-Microban",
    env_cfg=make_microban_tracking_env_cfg(),
    play_env_cfg=make_microban_tracking_env_cfg(play=True),
    rl_cfg=MicrobanTrackingRlCfg,
    runner_cls=MicrobanTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Teleop-Microban",
    env_cfg=make_microban_teleop_env_cfg(),
    play_env_cfg=make_microban_teleop_env_cfg(play=True),
    rl_cfg=MicrobanTeleopRlCfg,
    runner_cls=MicrobanTeleopOnPolicyRunner,
)

register_mjlab_task(
    task_id=MICROBAN_TELEOP_V12_TASK_ID,
    env_cfg=make_microban_teleop_v12_env_cfg(),
    play_env_cfg=make_microban_teleop_v12_env_cfg(play=True),
    rl_cfg=MicrobanTeleopV12RlCfg,
    runner_cls=MicrobanTeleopV12OnPolicyRunner,
)

register_mjlab_task(
    task_id=MICROBAN_TELEOP_UPRIGHT_FULLBODY_TASK_ID,
    env_cfg=make_microban_teleop_upright_fullbody_env_cfg(),
    play_env_cfg=make_microban_teleop_upright_fullbody_env_cfg(play=True),
    rl_cfg=MicrobanTeleopUprightFullbodyRlCfg,
    runner_cls=MicrobanTeleopUprightFullbodyOnPolicyRunner,
)

register_mjlab_task(
    task_id=MICROBAN_TELEOP_V12_PREVIEW_TASK_ID,
    env_cfg=make_microban_teleop_v12_preview_env_cfg(),
    play_env_cfg=make_microban_teleop_v12_preview_env_cfg(play=True),
    rl_cfg=MicrobanTeleopV12PreviewRlCfg,
    runner_cls=MicrobanTeleopV12PreviewOnPolicyRunner,
)

register_mjlab_task(
    task_id=MICROBAN_TELEOP_V12_CORNER_RESCUE_TASK_ID,
    env_cfg=make_microban_teleop_v12_corner_rescue_env_cfg(),
    play_env_cfg=make_microban_teleop_v12_corner_rescue_env_cfg(play=True),
    rl_cfg=MicrobanTeleopV12CornerRescueRlCfg,
    runner_cls=MicrobanTeleopV12CornerRescueOnPolicyRunner,
)
