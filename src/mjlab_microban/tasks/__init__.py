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

register_mjlab_task(
    task_id="Mjlab-Getup-Microban",
    env_cfg=make_microban_getup_env_cfg(),
    play_env_cfg=make_microban_getup_env_cfg(play=True),
    rl_cfg=MicrobanGetupRlCfg,
    runner_cls=VelocityOnPolicyRunner,
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
