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
from mjlab_microban.tasks.microban_teleop_v12_final_rescue import (
    MICROBAN_TELEOP_V12_FINAL_RESCUE_TASK_ID,
    MicrobanTeleopV12FinalRescueRlCfg,
    make_microban_teleop_v12_final_rescue_env_cfg,
)
from mjlab_microban.tasks.microban_teleop_v12_final_rescue_runner import (
    MicrobanTeleopV12FinalRescueOnPolicyRunner,
)
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MICROBAN_TELEOP_V12_TASK_ID,
    MicrobanTeleopV12PreviewRlCfg,
    MicrobanTeleopV12RlCfg,
    make_microban_teleop_v12_env_cfg,
    make_microban_teleop_v12_preview_env_cfg,
)
from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release import (
    MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_TASK_ID,
    MicrobanTeleopV12HandPoseReleaseRlCfg,
    make_microban_teleop_v12_hand_pose_release_env_cfg,
)
from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release_runner import (
    MicrobanTeleopV12HandPoseReleaseOnPolicyRunner,
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
from mjlab_microban.tasks.microban_velocity_runner import MicrobanVelocityOnPolicyRunner

# Walking checkpoints are stamped with the training HOME and refused on load
# under any other HOME (microban_velocity_runner.py).
register_mjlab_task(
    task_id="Mjlab-Velocity-Microban",
    env_cfg=make_microban_velocity_env_cfg(),
    play_env_cfg=make_microban_velocity_env_cfg(play=True),
    rl_cfg=MicrobanVelocityRlCfg,
    runner_cls=MicrobanVelocityOnPolicyRunner,
)

# VALIDATION EXPERIMENT ONLY: walking with the twist-ratio velocity reward.
from mjlab_microban.tasks.microban_velocity_env_cfg import (  # noqa: E402
    make_microban_velocity_twist_ratio_env_cfg,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-TwistRatio-Microban",
    env_cfg=make_microban_velocity_twist_ratio_env_cfg(),
    play_env_cfg=make_microban_velocity_twist_ratio_env_cfg(play=True),
    rl_cfg=MicrobanVelocityRlCfg,
    runner_cls=MicrobanVelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id=MICROBAN_SAFE_VELOCITY_TASK_ID,
    env_cfg=make_microban_safe_velocity_env_cfg(),
    play_env_cfg=make_microban_safe_velocity_env_cfg(play=True),
    rl_cfg=MicrobanSafeVelocityRlCfg,
    runner_cls=MicrobanSafeVelocityOnPolicyRunner,
)

# Get-up (v6 action contract, see microban_getup_runner.py). The robot
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

# Stage 1 with the near-home reset written out (10 %, +-5 deg); identical to
# Mjlab-Getup-Microban since that became the default again (kept: the
# 2026-10-03 servo-range stage-1 run was trained under this id).
register_mjlab_task(
    task_id="Mjlab-Getup-Microban-NearHome5deg",
    env_cfg=make_microban_getup_env_cfg(reward_set="posture", near_home_reset=(0.1, 0.09)),
    play_env_cfg=make_microban_getup_env_cfg(play=True, reward_set="posture", near_home_reset=(0.1, 0.09)),
    rl_cfg=MicrobanGetupRlCfg,
    runner_cls=MicrobanGetupOnPolicyRunner,
)

# Stage 1 with 957ab42's wide "tipping" near-home reset (20 %, +-34 deg).
register_mjlab_task(
    task_id="Mjlab-Getup-Microban-Tipping",
    env_cfg=make_microban_getup_env_cfg(reward_set="posture", near_home_reset=(0.2, 0.6)),
    play_env_cfg=make_microban_getup_env_cfg(play=True, reward_set="posture", near_home_reset=(0.2, 0.6)),
    rl_cfg=MicrobanGetupRlCfg,
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

# Successor recipe (active-hand arms leave the HOME pose reward).  Gates and the
# exporter accept its release-eligible lineages; see the module docstring.
register_mjlab_task(
    task_id=MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_TASK_ID,
    env_cfg=make_microban_teleop_v12_hand_pose_release_env_cfg(),
    play_env_cfg=make_microban_teleop_v12_hand_pose_release_env_cfg(play=True),
    rl_cfg=MicrobanTeleopV12HandPoseReleaseRlCfg,
    runner_cls=MicrobanTeleopV12HandPoseReleaseOnPolicyRunner,
)

# Lateral-fidelity variant of the pose-release recipe (restart at a gated
# fresh-chain model_7099).  The weight label is read from
# MICROBAN_V12_LATERAL_FIDELITY_WEIGHT when the package is imported.
from mjlab_microban.tasks.microban_teleop_v12_lateral_fidelity import (  # noqa: E402
    MICROBAN_TELEOP_V12_LATERAL_FIDELITY_TASK_ID,
    make_microban_teleop_v12_lateral_fidelity_env_cfg,
)

register_mjlab_task(
    task_id=MICROBAN_TELEOP_V12_LATERAL_FIDELITY_TASK_ID,
    env_cfg=make_microban_teleop_v12_lateral_fidelity_env_cfg(),
    play_env_cfg=make_microban_teleop_v12_lateral_fidelity_env_cfg(play=True),
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
# chain's model_9900; 5/60/35 sampler; saves keep the pose-release recipe).
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
