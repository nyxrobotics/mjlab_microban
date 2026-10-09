# Copyright 2026 nyxrobotics

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""PICO hybrid locomotion/keypoint policy for Microban.

This is intentionally a separate task from the deployed velocity policy.  Its
actor consumes only signals available on the physical robot plus PICO commands;
privileged simulation state may still be used by the critic during training.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import MISSING, fields

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microban.robot.microban_constants import (
    HOME_TRUNK_PITCH_RAD,
)
from mjlab_microban.tasks.curriculum import Setting, Stage, StagedCurriculum
from mjlab_microban.schedules import PICO_SCHEDULE
from mjlab_microban.tasks.mdp import (
    UniformVelocityCommandWithRotationCfg,
    foot_target_offset_b,
    foot_target_tracking_error_exp,
    lifted_support_feet,
    single_support,
)
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_HMD_JOINT_NAMES,
    MICROBAN_TELEOP_ACTION_JOINT_NAMES,
    MICROBAN_TELEOP_FINAL_BOTH_FEET_LIFT_UPPER_M,
)
from mjlab_microban.tasks.microban_teleop_mdp import (
    MICROBAN_HMD_RETARGET_INTERVAL_S,
    MICROBAN_HMD_RUNTIME_LIMITS_RAD,
    MICROBAN_HMD_SLEW_RATES_RAD_S,
    MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M,
    PICO_ARM_HOME_PROBABILITY,
    PICO_ARM_RETARGET_INTERVAL_S,
    HmdNeckTargetMotion,
    PicoArmOverlayJointPositionActionCfg,
    PicoArmTargetMotion,
    arm_target_rel,
    normalized_joint_soft_limit_guard_l1_sum,
)
from mjlab_microban.tasks.microban_teleop_foot_command import StationaryFootTargetCommandCfg
from mjlab_microban.tasks.microban_teleop_velocity_rewards import (
    commanded_planar_velocity_progress,
    linear_velocity_tracking_error_l1,
    planar_velocity_tracking_exp,
    yaw_velocity_tracking_error_l1,
)
from mjlab_microban.tasks.microban_velocity_env_cfg import make_microban_velocity_env_cfg


# The velocity tracking stds of the earlier forward-lean PICO's final stages.
MICROBAN_TELEOP_LINEAR_TRACKING_STD_M_S = 0.5
MICROBAN_TELEOP_ANGULAR_TRACKING_STD_RAD_S = 1.25
MICROBAN_TELEOP_NEUTRAL_FOOT_TRACKING_WEIGHT = 1.0
MICROBAN_TELEOP_FOOT_TRACKING_FINAL_STD_M = 0.03
PICO_SINGLE_SUPPORT_STATIONARY_PROBABILITY = 0.5
# The lifted-support-feet penalty (mdp.lifted_support_feet) on a standing
# command, per foot that should be down and is in the air (mjlab multiplies by
# dt: weight * value is a rate per second).  Measured on standing rows: r =
# 6.32 /s of the other terms, n = 0.25 feet up (update 1000) to 0.44-0.63
# (stepping v13 run, update 3000).  Stepping costs w n: |w| = r / (2 x 0.44)
# = 7.2 -> 7 makes it 28-70 % of r.  A standing row stays above zero while n <
# r / |w| = 0.9, so a fall does not pay; a stop/walk switch steps the 2 s value
# by (6.32 - 7 x 0.25 - 2.45) x 2 = 4 (a bonus of 7 per foot down: 36).
MICROBAN_TELEOP_LIFTED_SUPPORT_FEET_WEIGHT = -7.0
# The single-stance reward (mdp.single_support): 1 on a standing row whose
# published foot targets differ in z by the threshold or more, while the lower
# foot is down and the higher one in the air.  Threshold 10 mm: 4x the 2.5 mm
# floor band, reached by 84 % of the single-foot targets (z ~ U(2.5, 50) mm).
# On the same rows the foot reward measures the lifted foot from the support
# foot (mdp.foot_target_tracking_error_exp), so how high it pays is the foot
# reward's: 1 mm in the air is all this term asks.
# Weight (from the foot stage), per second on such a row, foot stage /
# tightened: standing on the lower foot with the other at its target pays
# w + G - L more than both feet down.
#   G, the foot reward's gain, static poses (the support foot where it stood,
#   the lifted one at its target from it), mean over the training targets:
#   0.91 (w 2, std 0.05) / 2.31 (w 3, std 0.03), of which a 1 mm lift gets
#   0.02-0.04.  These are upper bounds: they leave out what the pose itself
#   costs.  With the trunk level the lifted hip roll sits at its soft limit
#   (0.393 rad), inside joint_soft_limit_guard's 5 % margin (from 0.353):
#   -0.5 /s straight up, -2.0 to -2.5 at the inner front corner.  Rolling the
#   trunk over the support foot keeps the guard at 0, for upright 0.025 (1-10
#   mm), 0.095 (20-30), 0.20 (40-50), 0.46 (outer corners).  Net of that, the
#   target pays over a 1 mm lift: 10 mm +0.06 / +0.26, 20 mm +0.20 / +0.92,
#   40 mm +0.74 / +2.27, front outer corner +0.88 / +2.42.
#   L, what one foot down costs in the other terms (model_8999, stochastic,
#   paid rows against both feet down): 1.26 / 1.02 without pushes, 1.18 /
#   1.03 a second or more after a push; 2.01 / 1.87 with the second after a
#   push (20-23 % of the time) counted at its whole difference, 4.8-5.2 (a push
#   lifts a foot and adds the velocity errors at once: over all rows 4.5-4.7).
# w = 3: w + G - L = 3 + 0.91 - 2.01 = 1.9 / 3 + 2.31 - 1.87 = 3.4 with
# pushes and the upper-bound G, w = 1.5 L; at the least net G (10 mm, 0.09 /
# 0.29 with the 1 mm share) still 3 + 0.09 - 2.01 = 1.1 / 3 + 0.29 - 1.87 = 1.4.
MICROBAN_TELEOP_SINGLE_SUPPORT_LIFT_THRESHOLD_M = 0.010
MICROBAN_TELEOP_SINGLE_SUPPORT_WEIGHT = 3.0
MICROBAN_TELEOP_INITIAL_HMD_NEUTRAL_PROBABILITY = 1.0
MICROBAN_TELEOP_MOVING_HMD_NEUTRAL_PROBABILITY = 0.2
MICROBAN_TELEOP_JOINT_LIMIT_GUARD_MARGIN_RATIO = 0.05
MICROBAN_TELEOP_JOINT_LIMIT_GUARD_LOOKAHEAD_S = 0.12
# The PICO command envelope (the robot's moving scale_velocity limits, see
# tests/test_velocity_rewards.py).  Training samples it from the first update.
MICROBAN_TELEOP_FINAL_VELOCITY_ENVELOPE = {
    "lin_vel_x": (-0.5, 0.7),
    "lin_vel_y": (-0.3, 0.3),
    "ang_vel_z": (-1.5, 1.5),
    "rotation_ang_vel_z": (-3.0, 3.0),
}
MICROBAN_TELEOP_MIXED_AXIS_PROBABILITIES = {
    "standing": 0.10,
    "forward": 0.10,
    "backward": 0.10,
    "lateral_left": 0.10,
    "lateral_right": 0.10,
    "yaw_left": 0.10,
    "yaw_right": 0.10,
    "mixed": 0.30,
}
MICROBAN_TELEOP_FINAL_SIGNED_AXIS_RANGES = {
    "forward": (0.10, 0.70),
    "backward": (-0.50, -0.10),
    "lateral_left": (0.08, 0.30),
    "lateral_right": (-0.30, -0.08),
    "yaw_left": (0.40, 3.00),
    "yaw_right": (-3.00, -0.40),
}


def _materialize_rotation_command_cfg(
    command: object,
) -> UniformVelocityCommandWithRotationCfg:
    """Replace the velocity task's dynamically extended command config.

    The base task currently attaches a ``build`` lambda and three rotation-only
    attributes to a plain ``UniformVelocityCommandCfg`` instance.  Those
    instance-only extensions are lost when Tyro reconstructs the registered task
    config for ``uv run train``.  Copy every formal dataclass init field into the
    dedicated subclass so the command implementation and its rotation fields
    survive CLI serialization without changing the deployed velocity task.
    """

    values: dict[str, object] = {}
    missing: list[str] = []
    for field in fields(UniformVelocityCommandWithRotationCfg):
        if not field.init:
            continue
        if hasattr(command, field.name):
            values[field.name] = deepcopy(getattr(command, field.name))
        elif field.default is not MISSING:
            values[field.name] = deepcopy(field.default)
        elif field.default_factory is not MISSING:
            values[field.name] = field.default_factory()
        else:
            missing.append(field.name)
    if missing:
        raise TypeError(
            "Velocity command is missing fields required by "
            f"UniformVelocityCommandWithRotationCfg: {tuple(missing)}"
        )
    return UniformVelocityCommandWithRotationCfg(**values)


# The arms (and the moving HMD and the lifted-support-feet penalty) after the critic
# warm-up, foot targets later, tightened later (mjlab_microban/schedules.py).
# The adapter columns of the frozen walker open at the same updates
# (microban_teleop_v12_actor); before the arms move no actor column trains.
TELEOP_STAGES = (
    Stage(
        "enable moving-HMD, moving arms and the lifted-support-feet penalty",
        PICO_SCHEDULE["arm"],
        (
            Setting("reward", "lifted_support_feet", "weight", MICROBAN_TELEOP_LIFTED_SUPPORT_FEET_WEIGHT),
            Setting(
                "event",
                "hmd_neck_target_motion",
                "params.neutral_probability",
                MICROBAN_TELEOP_MOVING_HMD_NEUTRAL_PROBABILITY,
            ),
            Setting(
                "event", "pico_arm_target_motion", "params.home_probability", PICO_ARM_HOME_PROBABILITY
            ),
        ),
    ),
    Stage(
        "enable broad stationary foot tracking",
        PICO_SCHEDULE["foot"],
        (
            Setting("reward", "single_support", "weight", MICROBAN_TELEOP_SINGLE_SUPPORT_WEIGHT),
            Setting("reward", "foot_target_tracking", "weight", 2.0),
            Setting("reward", "foot_target_tracking", "params.std", 0.05),
            Setting("reward", "foot_target_tracking", "params.velocity_fade_range", (0.0, 0.15)),
            Setting("command", "foot_target", "rel_single_support_envs", 0.3),
            # Half of the single-foot targets come with a standing twist (the
            # foot reward fades out with the commanded speed): about 3.5 % ->
            # 17 % of the samples train a single-foot target at full weight
            # (microban_teleop_foot_command.py).
            Setting("command", "foot_target", "single_support_stationary_probability",
                    PICO_SINGLE_SUPPORT_STATIONARY_PROBABILITY),
            Setting("command", "foot_target", "rel_both_feet_envs", 0.05),
        ),
    ),
    Stage(
        "tighten foot tracking",
        PICO_SCHEDULE["foot_tighten"],
        (
            Setting("reward", "foot_target_tracking", "weight", 3.0),
            Setting(
                "reward", "foot_target_tracking", "params.std", MICROBAN_TELEOP_FOOT_TRACKING_FINAL_STD_M
            ),
            Setting("command", "foot_target", "rel_both_feet_envs", 0.1),
            Setting(
                "command",
                "foot_target",
                "both_feet_lift_height_range",
                (MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M, MICROBAN_TELEOP_FINAL_BOTH_FEET_LIFT_UPPER_M),
            ),
        ),
    ),
)


def make_microban_teleop_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build the independent PICO hybrid policy training environment."""

    # Reuse the validated locomotion dynamics/rewards without mutating the
    # deployed velocity configuration object.
    cfg = make_microban_velocity_env_cfg(play=play)
    # Rejected fall rollouts reached 248 contacts / 513 constraints.  Reserve
    # enough capacity to simulate the failure faithfully instead of silently
    # dropping contacts or constraints at the moment the policy is least stable.
    cfg.sim.nconmax = 512
    cfg.sim.njmax = 2048

    # Materialize the velocity task's runtime-only rotation extensions as a
    # proper dataclass before this config crosses the registry/Tyro CLI boundary.
    # In particular, do not carry over the dynamically assigned ``build`` lambda.
    cfg.commands["twist"] = _materialize_rotation_command_cfg(cfg.commands["twist"])
    if not play:
        # Training samples the full PICO command envelope with mixed-axis
        # replay from the first update, and the walking task's +-0.5 m/s
        # pushes stay on.  (Until the arms move the frozen walker
        # receives no policy gradient, so only the critic learns there.)
        twist = cfg.commands["twist"]
        envelope = MICROBAN_TELEOP_FINAL_VELOCITY_ENVELOPE
        twist.ranges.lin_vel_x = envelope["lin_vel_x"]
        twist.ranges.lin_vel_y = envelope["lin_vel_y"]
        twist.ranges.ang_vel_z = envelope["ang_vel_z"]
        twist.rotation_env_ang_vel_range = envelope["rotation_ang_vel_z"]
        twist.signed_axis_probabilities = deepcopy(MICROBAN_TELEOP_MIXED_AXIS_PROBABILITIES)
        twist.signed_axis_ranges = deepcopy(MICROBAN_TELEOP_FINAL_SIGNED_AXIS_RANGES)
        twist.rel_standing_envs = 0.0
        twist.rel_forward_envs = 0.0
        twist.rel_rotation_envs = 0.0
        twist.rel_heading_envs = 0.0
        twist.rel_world_envs = 0.0
        twist.init_velocity_prob = 0.0
        twist.resampling_time_range = (4.0, 8.0)

    # The PICO actor's HOME has both shoulder pitches at zero (the v12 HOME
    # revision says "shoulder_zero"), required of config/home_pose.yaml so the
    # robot's NEUTRAL_POSE and the teleop HOME can never differ.
    teleop_joint_pos = cfg.scene.entities["robot"].init_state.joint_pos
    if teleop_joint_pos is None:
        raise ValueError("Microban teleop requires an explicit initial joint pose")
    for name in ("left_shoulder_pitch", "right_shoulder_pitch"):
        if teleop_joint_pos[name] != 0.0:
            raise ValueError(
                f"Microban teleop requires HOME {name} = 0 (config/home_pose.yaml)"
            )

    # Head yaw + two neck axes are owned by the HMD controller.  Exact names make
    # accidental action-space growth fail loudly in the smoke test/exporter.
    action = cfg.actions["joint_pos"]
    if not isinstance(action, JointPositionActionCfg):
        raise TypeError("Expected velocity base task to use JointPositionActionCfg")
    # The six arm joints are overwritten by the robot's pico_arms move during
    # PICO teleoperation: the policy's arm outputs never reach the servos.
    # Train the same way (microban_teleop_mdp.PicoArmOverlayJointPositionAction).
    action = PicoArmOverlayJointPositionActionCfg(
        **{f.name: deepcopy(getattr(action, f.name)) for f in fields(action) if f.init}
    )
    action.actuator_names = MICROBAN_TELEOP_ACTION_JOINT_NAMES
    action.scale = 1.0
    cfg.actions["joint_pos"] = action

    if not play:
        # The real HMD controller owns these joints independently of the policy.
        # Move their simulated position targets throughout every training episode
        # so the actor learns under the same changing neck inertia it will observe
        # on hardware.  The stateful event clamps to Entity soft limits and applies
        # the runtime's 2.5 rad/s target slew limit on every 50 Hz policy step.
        cfg.events["hmd_neck_target_motion"] = EventTermCfg(
            func=HmdNeckTargetMotion,
            mode="step",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=MICROBAN_HMD_JOINT_NAMES,
                    preserve_order=True,
                ),
                "position_ranges_rad": MICROBAN_HMD_RUNTIME_LIMITS_RAD,
                "slew_rates_rad_s": MICROBAN_HMD_SLEW_RATES_RAD_S,
                "retarget_interval_s": MICROBAN_HMD_RETARGET_INTERVAL_S,
                # Learn the signed locomotion axes before adding an external
                # disturbance that sweeps a comparatively heavy head through
                # its full runtime range.  The first stage
                # (PICO_SCHEDULE["arm"]) switches the live stateful term to
                # the deployment-like distribution.
                "neutral_probability": (
                    MICROBAN_TELEOP_INITIAL_HMD_NEUTRAL_PROBABILITY
                ),
            },
        )
        # The arm goals the robot's pico_arms would follow: HOME until the
        # arm stage (PICO_SCHEDULE["arm"]), then the deployment distribution.
        cfg.events["pico_arm_target_motion"] = EventTermCfg(
            func=PicoArmTargetMotion,
            mode="step",
            params={
                "action_name": "joint_pos",
                "retarget_interval_s": PICO_ARM_RETARGET_INTERVAL_S,
                "home_probability": 1.0,
            },
        )

    # The encoders for all 21 joints exist on hardware.  Observing HMD-controlled
    # neck/head motion lets the actor compensate for its momentum while retaining
    # a strict 18-DoF action space.
    all_joint_cfg = SceneEntityCfg("robot", joint_names=(r".*",))
    cfg.observations["actor"].terms["joint_pos"] = ObservationTermCfg(
        func=velocity_mdp.joint_pos_rel,
        params={"asset_cfg": all_joint_cfg},
        noise=Unoise(n_min=-0.001, n_max=0.001),
        delay_min_lag=0,
        delay_max_lag=0,
    )
    cfg.observations["actor"].terms["joint_vel"] = ObservationTermCfg(
        func=velocity_mdp.joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",))},
        noise=Unoise(n_min=-0.25, n_max=0.25),
        delay_min_lag=0,
        delay_max_lag=1,
    )

    # Explicit safety contract: actor inference must not depend on quantities the
    # real robot cannot measure directly.
    actor_terms = cfg.observations["actor"].terms
    for forbidden_term in ("base_lin_vel", "root_pos", "root_position", "height_scan"):
        actor_terms.pop(forbidden_term, None)

    # The previous-action observation is the raw actor output (the v12
    # contract: the walking source actor's recurrence).
    raw_previous_action = ObservationTermCfg(
        func=velocity_mdp.last_action,
        params={"action_name": "joint_pos"},
    )
    actor_terms["actions"] = raw_previous_action
    cfg.observations["critic"].terms["actions"] = raw_previous_action

    actor_terms["foot_target"] = ObservationTermCfg(
        func=foot_target_offset_b,
        params={"command_name": "foot_target"},
    )
    # The arm targets of the last physics step minus HOME (the robot's
    # pico_arms output of the previous control cycle): no noise, no delay.
    actor_terms["arm_target"] = ObservationTermCfg(
        func=arm_target_rel,
        params={"action_name": "joint_pos"},
    )
    cfg.observations["critic"].terms["foot_target"] = ObservationTermCfg(
        func=foot_target_offset_b,
        params={"command_name": "foot_target"},
    )
    cfg.observations["critic"].terms["arm_target"] = ObservationTermCfg(
        func=arm_target_rel,
        params={"action_name": "joint_pos"},
    )

    # Exact-zero standing receives a small foot anchor from the first update.
    # Its 1 cm/s fade makes the reward exactly zero for every signed locomotion
    # sample (the smallest commanded translation is 6 cm/s and yaw is 0.4
    # rad/s), so it cannot reward the stationary local optimum on moving tasks.
    # On the rows single_support counts as one foot up (the same threshold) it
    # measures the lifted foot from the support foot (mdp docstring).
    cfg.rewards["foot_target_tracking"] = RewardTermCfg(
        func=foot_target_tracking_error_exp,
        weight=MICROBAN_TELEOP_NEUTRAL_FOOT_TRACKING_WEIGHT,
        params={
            "command_name": "foot_target",
            "std": 0.05,
            "lift_threshold": MICROBAN_TELEOP_SINGLE_SUPPORT_LIFT_THRESHOLD_M,
            "velocity_command_name": "twist",
            "velocity_fade_range": (0.0, 0.01),
        },
    )
    cfg.rewards["joint_soft_limit_guard"] = RewardTermCfg(
        func=normalized_joint_soft_limit_guard_l1_sum,
        weight=-5.0,
        params={
            "action_name": "joint_pos",
            "margin_ratio": MICROBAN_TELEOP_JOINT_LIMIT_GUARD_MARGIN_RATIO,
            "lookahead_s": MICROBAN_TELEOP_JOINT_LIMIT_GUARD_LOOKAHEAD_S,
        },
    )
    # Keep only a light smoothing prior.  At -0.1 this raw-coordinate term
    # dominated v1 and rewarded copying a saturated previous output forever.
    cfg.rewards["action_rate_l2"].weight = -0.02

    # Velocity: the terms of the earlier forward-lean PICO
    # (microban_teleop_velocity_rewards.py), with the final tracking stds from
    # the first update (the full envelope is sampled from the first update).
    velocity_params = {"command_name": "twist", "trunk_pitch": HOME_TRUNK_PITCH_RAD}
    cfg.rewards["track_linear_velocity"] = RewardTermCfg(
        func=planar_velocity_tracking_exp,
        weight=5.0,
        params={**velocity_params, "std": MICROBAN_TELEOP_LINEAR_TRACKING_STD_M_S},
    )
    cfg.rewards["commanded_planar_velocity_progress"] = RewardTermCfg(
        func=commanded_planar_velocity_progress,
        weight=2.0,
        params={**velocity_params, "command_threshold": 0.01},
    )
    cfg.rewards["track_angular_velocity"].params["std"] = MICROBAN_TELEOP_ANGULAR_TRACKING_STD_RAD_S
    cfg.rewards["linear_velocity_error_l1"] = RewardTermCfg(
        func=linear_velocity_tracking_error_l1, weight=-16.0, params=dict(velocity_params)
    )
    cfg.rewards["yaw_velocity_error_l1"] = RewardTermCfg(
        func=yaw_velocity_tracking_error_l1, weight=-1.0, params=dict(velocity_params)
    )
    cfg.rewards["air_time"].weight = 3.0
    cfg.rewards["air_time"].params["threshold_min"] = 0.02
    cfg.rewards["air_time"].params["threshold_max"] = 0.30

    # V2 converged to a wide static stance because the inherited term penalized
    # the 72 mm neutral foot spacing below an 80 mm threshold with weight -1000.
    # Keep a collision-avoidance margin, but do not make neutral stance itself a
    # dominant violation in this task.
    cfg.rewards["feet_distance"].weight = -100.0
    cfg.rewards["feet_distance"].params["min_dist"] = 0.07
    cfg.rewards["dof_pos_limits"].weight = -10.0
    # The walking task's no_stepping stays at 0 here: on a standing command
    # PICO penalizes the feet in the air that should be down instead (weight
    # in the arm stage).  The contact sensor lists body foot (right) before foot_2
    # (left); the foot target is (left, right).
    cfg.rewards["lifted_support_feet"] = RewardTermCfg(
        func=lifted_support_feet,
        weight=0.0,
        params={
            "sensor_name": cfg.rewards["no_stepping"].params["sensor_name"],
            "foot_target_command_name": "foot_target",
            "command_name": "twist",
            "command_threshold": cfg.rewards["no_stepping"].params["command_threshold"],
            "sensor_foot_ids": (1, 0),
        },
    )
    # Standing on the lower foot with the higher one up (weight in the foot
    # stage), the same rows and feet as lifted_support_feet.
    cfg.rewards["single_support"] = RewardTermCfg(
        func=single_support,
        weight=0.0,
        params={
            **cfg.rewards["lifted_support_feet"].params,
            "lift_threshold": MICROBAN_TELEOP_SINGLE_SUPPORT_LIFT_THRESHOLD_M,
        },
    )

    # Six foot XYZ offsets in metres, expressed in the HOME-levelled trunk
    # frame R_trunk * R_y(-HOME_TRUNK_PITCH_RAD): level at HOME (x forward,
    # y left, z up), the frame the PICO bridge sends and the twist uses, so a
    # world-vertical foot lift at HOME reads (0, 0, dz).  With a vertical
    # trunk at HOME it is the trunk frame.
    cfg.commands["foot_target"] = StationaryFootTargetCommandCfg(
        resampling_time_range=(3.0, 8.0),
        rel_single_support_envs=0.0,
        rel_both_feet_envs=0.0,
        lift_height_range=(MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M, 0.05),
        reach_xy_range=(-0.03, 0.03),
        both_feet_lift_height_range=(
            MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M,
            0.012,
        ),
        both_feet_reach_xy_range=(-0.01, 0.01),
        trunk_pitch=HOME_TRUNK_PITCH_RAD,
    )

    cfg.curriculum = {
        "staged_curriculum": CurriculumTermCfg(
            func=StagedCurriculum, params={"stages": TELEOP_STAGES}
        )
    }

    if play:
        # Deterministic neutral keypoints by default.  A live/specialized player
        # may write explicit PICO commands after reset.
        cfg.curriculum = {}
        cfg.commands["foot_target"].rel_single_support_envs = 0.0
        cfg.commands["foot_target"].rel_both_feet_envs = 0.0

    return cfg
