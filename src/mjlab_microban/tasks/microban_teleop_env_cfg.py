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
from dataclasses import MISSING, fields, replace

import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microban.robot.home_pose import HOME
from mjlab_microban.robot.microban_constants import (
    HOME_TRUNK_PITCH_RAD,
)
from mjlab_microban.tasks.curriculum import Setting, Stage, StagedCurriculum
from mjlab_microban.schedules import PICO_SCHEDULE
from mjlab_microban.tasks.mdp import (
    UniformVelocityCommandWithRotationCfg,
    extreme_joint_velocity,
    foot_target_offset_b,
    foot_target_tracking_error_exp,
    lifted_support_feet,
    support_zmp,
    upper_foot_lift,
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
    normalized_target_clip_excess_l1_sum,
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
# The foot stage's single-foot targets: 60 % of the foot targets, 85 % of them
# with a standing twist.  The command draws the single-foot rows first and
# rel_both_feet_envs then turns its share of all rows into both-feet targets,
# so 0.7 x 0.9 x 0.85 = 54 % of the samples train a standing single-foot
# target at full weight (46 % from the tighten stage's 0.1; it was 0.3 x 0.5:
# 15 %).  Resumed at update 4000 of a 1e-4 run, the unload reward grew 8 % in
# 200 updates with these shares and 33 % with the learning rate also tripled
# (MICROBAN_TELEOP_V12_FIXED_LEARNING_RATE).
PICO_SINGLE_SUPPORT_ENV_SHARE = 0.7
PICO_SINGLE_SUPPORT_STATIONARY_PROBABILITY = 0.85
# The lifted-support-feet penalty (mdp.lifted_support_feet) on a standing
# command, per foot that should be down and is in the air (mjlab multiplies by
# dt: weight * value is a rate per second).  Measured on standing rows: r =
# 6.32 /s of the other terms, n = 0.25 feet up (update 1000) to 0.44-0.63
# (stepping v13 run, update 3000).  Stepping costs w n: |w| = r / (2 x 0.44)
# = 7.2 -> 7 makes it 28-70 % of r.  A standing row stays above zero while n <
# r / |w| = 0.9, so a fall does not pay; a stop/walk switch steps the 2 s value
# by (6.32 - 7 x 0.25 - 2.45) x 2 = 4 (a bonus of 7 per foot down: 36).
# Those are the arm stage's numbers.  From the foot stage support_zmp pays
# the standing rows too (up to 10 /s, 8.8 on an unpushed stand of the v13h
# trial's model_7000): a row then stays above zero up to n = 2.2 (both
# feet), and the switch steps the 2 s value by about 22.
MICROBAN_TELEOP_LIFTED_SUPPORT_FEET_WEIGHT = -7.0
# The support reward (mdp.support_zmp), on every standing row: clamp(1 -
# |ZMP - P| / d0, 0, 1), the ZMP the centre of pressure of the floor's push
# on both feet, d0 half the HOME feet spacing.  P is the level point moved
# towards the lower (support) sole's centre by 1 - 2 s*, s* = 0.5 max(0, 1 -
# dz / threshold) the higher foot's share of the weight, dz the published
# targets' height difference.
# The level point is the soles' midpoint moved with the centre of mass the
# targets ask for: the foot reward measures each foot from the trunk below
# the threshold, so feet at the published targets o put the trunk -mean(o)
# from them, and a trunk shift over the feet moves the centre of mass by
# MICROBAN_TELEOP_TRUNK_SHIFT_COM_SHARE of its (x, y).  Without a foot target
# it is the midpoint: the soles stand as mirror images, so it is the centre
# of the area the two soles and the gap between them cover (their convex
# hull; 0.02 mm apart at HOME and in the judged crouches).  HOME's centre of
# mass stands over it (0.03 mm off), so standing still at HOME scores 1 and
# the ZMP over either sole's centre 0.  A two-foot target (dx, dy, dz) moves
# it by -0.8 (dx, dy), so this term and the foot reward ask for one pose.
# Left at the midpoint, the term paid more for dropping the shift than the
# foot reward took (at (8, 8, 16) mm, from the v13h trial's model_7000: about
# +1.6 /s against -0.37 /s).  A crouch's own forward shift (knees forward:
# 1.5 mm at dz = 16 mm) is not followed: straight down the level point stays
# the midpoint.
# From dz = threshold on P is the support sole's centre.  Threshold 10 mm:
# 4x the 2.5 mm floor band, reached by 84 % of the single-foot targets (z ~
# U(2.5, 50) mm).  On those rows the foot reward measures the lifted foot
# from the support foot (mdp.foot_target_tracking_error_exp), so how high it
# goes is the foot reward's, and taking it off the floor the lift reward's:
# this term reads only the floor's push.  Below the threshold the foot reward
# measures the feet in the trunk frame, so moving the trunk over the lower
# foot costs there: s* at dz = 5 mm moves it ~18 mm, -0.24 /s (foot stage),
# against +5 to +9 /s here.  In one-foot stance it pays 1 over the support
# sole's centre, 0.57 at its side edge (20 mm out) and 0.34 at its toe or
# heel (31 mm out): the distance is not scaled by the sole's half-length, and
# the stance the term is after (v13c's lifts, the centre of mass 7-16 mm
# inside the sole's edge) failed sideways.
# P follows the soles where they stand now.  Held where the support sole
# stood when the target began, it fines every move of that foot for the rest
# of the target, as v13d's support-foot penalty did, which stepped more on
# standing commands and fell more under pushes without fewer re-placements;
# the evaluation counts the re-placements.
# Weight (from the foot stage), per second: moving the ZMP from the midpoint
# to the support sole's centre pays w, each 10 % of the weight moved w / 5.
# What that move costs in the other terms on the one-foot rows (model_5000 of
# the +3 single-stance reward, a second or more after a push, the higher foot
# up against both feet down): 3.0-3.8 /s at 5-10 mm up, 5.7-6.2 at 10 mm or
# more, over 80 % of it the velocity terms (the trunk moving over the support
# foot on a standing command).  At 10 mm or more w = 10 pays that cost with
# the ZMP within 18-20 mm of the sole's centre (the side edge: 5.7); the
# lift reward (also 10, the same rows) pays it on its own once the foot is
# off the floor.  10 /s is the largest positive weight on these rows, shared
# with the lift and the foot reward's single-foot rows (track_linear 5 +
# track_angular 2 + upright 1 + pose 1 = 9).
# The same weight on the level rows (no target, a crouch) keeps one scale
# for the term.  Pushed there, stepping moves the ZMP onto one sole: on a
# pushed stand without a target (the judgment's J2 standing pushes, v13h
# trial and v13g model_7000, which did not train this term there; 16 x 2
# environments) r is 0.68-0.71 with both feet down and 0.06-0.11 with one up
# (44 % of the time), so each foot in the air costs about 6 /s here on top
# of lifted_support_feet's 7, and holding on over the toe or heel (31 mm
# off, r 0.34) instead of stepping gains about 2.5 /s.  A fall still costs
# more than any step (it ends the standing row's 15 /s); whether the robot
# still steps when it has to is the push judgment's (J2, standing and
# crouched).
MICROBAN_TELEOP_SINGLE_SUPPORT_LIFT_THRESHOLD_M = 0.010
MICROBAN_TELEOP_SUPPORT_ZMP_TOLERANCE_M = HOME.feet_lateral_m / 2.0
MICROBAN_TELEOP_SUPPORT_ZMP_WEIGHT = 10.0
# The share of a trunk shift over the feet that moves the centre of mass:
# the crouch pose of a two-foot target (robot/crouch_feedforward.py, both
# soles at HOME, the trunk moved by -(dx, dy, dz), the arms at HOME) puts the
# centre of mass -0.80 (dx, dy) from where the straight-down crouch puts it
# (robot.xml, |dx|, |dy| <= 10 mm, dz 4-20 mm: within 0.14 mm).
MICROBAN_TELEOP_TRUNK_SHIFT_COM_SHARE = 0.8
# The (left, right) soles' contact boxes: six nested boxes share one centre,
# the sole's.
MICROBAN_SOLE_GEOM_NAMES = ("left_foot_collision_1", "right_foot_collision_1")
# The lift reward (mdp.upper_foot_lift), on a standing row whose published
# foot targets differ in z by dz >= the threshold: the higher foot's height
# over the lower one / dz, clamped to [0, 1], while it touches nothing and the
# lower foot is down.  With the old share reward alone the policy took the weight
# off the higher foot and stopped there (lift 2 mm): the foot reward's exp
# form, far from a 40 mm target, pays about 0.05 /s for each 1 mm.  This term
# pays w / dz per metre all the way up (w = 10: 0.25 /s per mm at 40 mm).
# The evidence comes from one chain of resumed runs, not from this recipe
# trained from scratch: updates 0-4000 at 1e-4 with the old shares and no
# lift term, 4000-6000 with the shares above at 3e-4 (still no lift term),
# 6000-7000 with the lift paid on the height alone (trial C: lifting started,
# 11-13 mm at update 7000, but a heel raised with the toes down scored too,
# hence only off the floor), 7000-8999 with the lift paid only while the
# higher foot touches nothing (trial D, model_8999: the foot left the floor
# in every judged single-foot case, 22-39 mm up, no falls).  Here the term
# is on from the foot stage.  The lower-foot-down condition is not D's; on
# D's model_8999 the lower foot was never up while the higher one was off the
# floor, so it changes no row D trained on.  The same weight as the support
# reward: both pay the same rows up to w, the largest positive weight there,
# shared with the foot reward's single-foot rows.
MICROBAN_TELEOP_UPPER_FOOT_LIFT_WEIGHT = 10.0
# The foot reward's weight on those rows, as the lift's.  At weight 3 its exp
# paid 0.05-0.09 /s per mm far from the target (lift: 0.25), the lifted foot
# stayed 25 mm off over 14 cases; 1000 updates at 10 took it to 18.5 mm.
MICROBAN_TELEOP_SINGLE_FOOT_TRACKING_WEIGHT = MICROBAN_TELEOP_UPPER_FOOT_LIFT_WEIGHT
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


# A joint's raw output past the +-pi target clip moves nothing, so only the
# last-action observation feeds it back; with a gain above one it winds up.
# At update 1000 (arms and HMD moving) left_hip_yaw's raw output in one of
# 1024 envs grew 38x per step to 1e17 (step reward -1.6e33) with the joints,
# trunk and contacts at ordinary values, and two PICO runs broke (value loss
# inf, a non-finite actor, at updates 1065 and 1159).  Ordinary raw outputs
# reach 9.8 (1024 envs x 1500 steps).  Training ends an env whose raw output
# passes PICO_RAW_ACTION_LIMIT (the robot stops PICO when an output leaves its
# recorded envelope) and charges the action rate on outputs held to it, so
# the step that runs away costs an ordinary penalty.
PICO_RAW_ACTION_LIMIT = 15.0


def raw_action_runaway(env: ManagerBasedRlEnv, limit: float) -> torch.Tensor:
    """An env with a raw policy output past ``limit``."""
    return env.action_manager.action.abs().amax(dim=-1) > limit


def held_action_rate_l2(env: ManagerBasedRlEnv, limit: float) -> torch.Tensor:
    """mjlab's action_rate_l2 on the raw outputs held to +-``limit`` (unchanged below it)."""
    action = env.action_manager.action.clamp(-limit, limit)
    previous = env.action_manager.prev_action.clamp(-limit, limit)
    return torch.square(action - previous).sum(dim=1)


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
            Setting("reward", "support_zmp", "weight", MICROBAN_TELEOP_SUPPORT_ZMP_WEIGHT),
            Setting("reward", "upper_foot_lift", "weight", MICROBAN_TELEOP_UPPER_FOOT_LIFT_WEIGHT),
            Setting("reward", "foot_target_tracking", "weight", 2.0),
            Setting("reward", "foot_target_tracking", "params.single_foot_weight",
                    MICROBAN_TELEOP_SINGLE_FOOT_TRACKING_WEIGHT / 2.0),
            Setting("reward", "foot_target_tracking", "params.std", 0.05),
            Setting("command", "foot_target", "rel_single_support_envs", PICO_SINGLE_SUPPORT_ENV_SHARE),
            # Most single-foot targets come with a standing twist (the foot
            # reward is zero on a moving command; see
            # microban_teleop_foot_command.py).
            Setting("command", "foot_target", "single_support_stationary_probability",
                    PICO_SINGLE_SUPPORT_STATIONARY_PROBABILITY),
            Setting("command", "foot_target", "rel_both_feet_envs", 0.1),
        ),
    ),
    Stage(
        "tighten foot tracking",
        PICO_SCHEDULE["foot_tighten"],
        (
            Setting("reward", "foot_target_tracking", "weight", 3.0),
            Setting("reward", "foot_target_tracking", "params.single_foot_weight",
                    MICROBAN_TELEOP_SINGLE_FOOT_TRACKING_WEIGHT / 3.0),
            Setting(
                "reward", "foot_target_tracking", "params.std", MICROBAN_TELEOP_FOOT_TRACKING_FINAL_STD_M
            ),
            # Twice the crouches of the first runs (0.05, then 0.1; the single-foot
            # share 0.6 -> 0.7 keeps half the samples on one foot standing, the
            # rows without a foot target 36 -> 24 %): PICO with the clip barrier
            # (7de607ab) falls in 3.7-5.9 % of its crouched pushes, 0.2-0.7 %
            # standing or walking.
            Setting("command", "foot_target", "rel_both_feet_envs", 0.2),
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
    # On the plate (tasks/floor_plate.py: box-box contacts by the box collider, up to eight per sole) the feet's
    # contact sensor matched more than the default 64 (walking c65f9736: 104172 overflow warnings, 826 without the
    # plate), and support_zmp reads its forces, torques and centroids.  The get-up's 500.
    cfg.sim.contact_sensor_maxmatch = 500
    # The feet's floor contacts also report their centroid and the torque
    # about it (still one net force per foot): the support reward's centre
    # of pressure (mdp.feet_zmp).  Not observed; the walking task keeps its own.
    cfg.scene.sensors = tuple(
        replace(sensor, fields=("found", "force", "torque", "pos"))
        if sensor.name == "feet_ground_contact" else sensor
        for sensor in cfg.scene.sensors
    )

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
    # sample (the smallest commanded translation is 8 cm/s and yaw is 0.4
    # rad/s), so it cannot reward the stationary local optimum on moving tasks.
    # The fade stays at 1 cm/s in every stage, the standing threshold of
    # support_zmp and upper_foot_lift: with a 0.15 m/s fade from the
    # foot stage, the single-foot rows' weight (10) paid more for standing on
    # one foot than for following a slow walking command (0.1 m/s), and slow
    # walking broke down after the foot stage even with the arms down.
    # On the rows support_zmp asks the higher foot to carry nothing (the
    # same threshold) it measures the lifted foot from the support foot (mdp
    # docstring) and pays MICROBAN_TELEOP_SINGLE_FOOT_TRACKING_WEIGHT whatever
    # the term's weight (single_foot_weight = it / weight at every stage).
    cfg.rewards["foot_target_tracking"] = RewardTermCfg(
        func=foot_target_tracking_error_exp,
        weight=MICROBAN_TELEOP_NEUTRAL_FOOT_TRACKING_WEIGHT,
        params={
            "command_name": "foot_target",
            "std": 0.05,
            "lift_threshold": MICROBAN_TELEOP_SINGLE_SUPPORT_LIFT_THRESHOLD_M,
            "velocity_command_name": "twist",
            "velocity_fade_range": (0.0, 0.01),
            "single_foot_weight": (
                MICROBAN_TELEOP_SINGLE_FOOT_TRACKING_WEIGHT / MICROBAN_TELEOP_NEUTRAL_FOOT_TRACKING_WEIGHT
            ),
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
    # The servo goal range (+-pi) ends before a hip yaw's soft range
    # (-60..240 deg), so the soft-limit guard above never sees a hip-yaw target
    # past the clip, where the joint stops following and its raw output winds
    # up on the last-action observation (raw_action_runaway).  The get-up's
    # clip barrier, at the guard's weight, charges the target past +-pi.
    cfg.rewards["raw_target_clip_excess"] = RewardTermCfg(
        func=normalized_target_clip_excess_l1_sum,
        weight=-5.0,
        params={"action_name": "joint_pos"},
    )
    # The walker's action-rate weight.  The actor is the frozen walker plus a
    # residual, and the term sees their sum: at -0.02 (a fifth of the
    # walker's) the residual was free to change its output from step to step,
    # and PICO v13g2 shook standing still on the robot until it fell.
    cfg.rewards["action_rate_l2"].weight = -0.1
    # PICO keeps the foot-slip weight it was judged with (the walker's is -10).
    cfg.rewards["foot_slip"].weight = -1.0

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
    # dominant violation in this task.  60 mm, below the narrowest single-foot
    # target (HOME 92.8 mm less the 30 mm reach in): at 70 mm 8.3 % of them were
    # narrower, and the best place for the lifted foot was up to 7.2 mm off it.
    cfg.rewards["feet_distance"].weight = -100.0
    cfg.rewards["feet_distance"].params["min_dist"] = 0.06
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
    # The feet's centre of pressure where the target puts the weight (weight
    # in the foot stage), on every standing row of lifted_support_feet, its
    # feet (the foot reward's 1 cm/s fade): on a walking
    # command (8 cm/s and up) the walker steps on both feet in turn.  The
    # sole centres are the sole boxes' (the foot sites sit 3.5 mm ahead).
    cfg.rewards["support_zmp"] = RewardTermCfg(
        func=support_zmp,
        weight=0.0,
        params={
            **cfg.rewards["lifted_support_feet"].params,
            "lift_threshold": MICROBAN_TELEOP_SINGLE_SUPPORT_LIFT_THRESHOLD_M,
            "zmp_tolerance": MICROBAN_TELEOP_SUPPORT_ZMP_TOLERANCE_M,
            "com_share": MICROBAN_TELEOP_TRUNK_SHIFT_COM_SHARE,
            "sole_cfg": SceneEntityCfg(
                "robot", geom_names=MICROBAN_SOLE_GEOM_NAMES, preserve_order=True
            ),
        },
    )
    # The higher foot's lift over the lower one (weight in the foot stage), on
    # the standing rows and feet of support_zmp from its threshold up.
    cfg.rewards["upper_foot_lift"] = RewardTermCfg(
        func=upper_foot_lift,
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

    if not play:
        # The get-up's guards (an unphysical joint speed ends the env before it
        # poisons the batch, a NaN one is reset instead of aborting the run) and
        # the raw-output runaway guard (PICO_RAW_ACTION_LIMIT).
        cfg.terminations["nan_detection"] = TerminationTermCfg(func=envs_mdp.nan_detection)
        cfg.terminations["extreme_joint_velocity"] = TerminationTermCfg(
            func=extreme_joint_velocity, params={"max_joint_vel": 50.0}
        )
        cfg.terminations["raw_action_runaway"] = TerminationTermCfg(
            func=raw_action_runaway, params={"limit": PICO_RAW_ACTION_LIMIT}
        )
        cfg.rewards["action_rate_l2"].func = held_action_rate_l2
        cfg.rewards["action_rate_l2"].params = {"limit": PICO_RAW_ACTION_LIMIT}

    if play:
        # Deterministic neutral keypoints by default.  A live/specialized player
        # may write explicit PICO commands after reset.
        cfg.curriculum = {}
        cfg.commands["foot_target"].rel_single_support_envs = 0.0
        cfg.commands["foot_target"].rel_both_feet_envs = 0.0

    return cfg
