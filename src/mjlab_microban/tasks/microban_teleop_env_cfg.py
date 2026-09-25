# Copyright 2026 Marc Duclusaud

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
from dataclasses import MISSING, dataclass, fields

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microban.robot.microban_constants import (
    MICROBAN_ROBOT_CFG,
)
from mjlab_microban.tasks.mdp import (
    UniformVelocityCommandWithRotationCfg,
    foot_target_offset_b,
    foot_target_tracking_error_exp,
    hand_target_offset_b,
    hand_target_tracking_error_exp,
    set_command_velocity,
    set_stepping_parameters,
)
from mjlab_microban.tasks.microban_locomotion_prior import (
    MICROBAN_LOCOMOTION_PRIOR_FORWARD_VELOCITY_RANGE_M_S,
    MICROBAN_LOCOMOTION_PRIOR_PATH,
    MICROBAN_LOCOMOTION_PRIOR_SHA256,
    LocomotionPriorCommandCfg,
    locomotion_prior_action_target_error_exp,
    locomotion_prior_joint_position_error_exp,
)
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_HMD_JOINT_NAMES,
    MICROBAN_TELEOP_ACTION_JOINT_NAMES,
    MICROBAN_TELEOP_ACTOR_LATENT_ABS_MAX,
    MICROBAN_TELEOP_ACTOR_LATENT_MEAN_FRACTION,
    MICROBAN_TELEOP_ACTOR_LATENT_SCALE_MULTIPLIER,
    MICROBAN_TELEOP_ACTOR_STD_ABS_MAX,
    MICROBAN_TELEOP_ACTOR_STD_ENVELOPE_DIVISOR,
    MICROBAN_TELEOP_ACTOR_STD_MIN_ABS_MAX,
    MICROBAN_TELEOP_ACTOR_STD_MIN_ENVELOPE_DIVISOR,
    MICROBAN_TELEOP_FINAL_BOTH_FEET_LIFT_UPPER_M,
    MICROBAN_TELEOP_NUM_STEPS_PER_ENV,
    guarded_teleop_actor_raw_bounds,
)
from mjlab_microban.tasks.microban_safe_velocity_mdp import (
    MicrobanSafeVelocityBoundedGaussianDistribution,
)
from mjlab_microban.tasks.microban_teleop_mdp import (
    MICROBAN_HMD_RETARGET_INTERVAL_S,
    MICROBAN_HMD_RUNTIME_LIMITS_RAD,
    MICROBAN_HMD_SLEW_RATES_RAD_S,
    MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M,
    HmdNeckTargetMotion,
    ResetFixedFootTargetCommandCfg,
    ResetFixedHandTargetCommandCfg,
    ResumeSafeStepBasedStagedCurriculum,
    effective_action_after_target_clip,
    linear_velocity_tracking_error_l1,
    normalized_joint_soft_limit_guard_l1_sum,
    normalized_target_clip_excess_l1_sum,
    normalized_target_near_limit_l1_sum,
    raw_action_l2,
    yaw_velocity_tracking_error_l1,
)
from mjlab_microban.tasks.microban_tracking_env_cfg import (
    MICROBAN_BODY_JOINT_SOFT_LIMITS,
)
from mjlab_microban.tasks.microban_velocity_env_cfg import (
    make_microban_velocity_env_cfg,
)


@dataclass
class MicrobanTeleopPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO settings plus the finite privileged locomotion-teacher contract."""

    locomotion_prior_bc_forward_coefficient: float = 0.0
    locomotion_prior_bc_neutral_leg_coefficient: float = 0.0
    locomotion_prior_bc_arm_home_coefficient: float = 0.0
    locomotion_prior_bc_error_scale_rad: float = 0.15
    locomotion_prior_bc_chunks: int = 4
    locomotion_prior_bc_max_target_projection_rad: float = 1.0e-3


MICROBAN_TELEOP_LINEAR_TRACKING_STD_M_S = 0.5
MICROBAN_TELEOP_ANGULAR_TRACKING_STD_RAD_S = 1.25
MICROBAN_TELEOP_INITIAL_LINEAR_TRACKING_STD_M_S = 0.35
MICROBAN_TELEOP_INITIAL_ANGULAR_TRACKING_STD_RAD_S = 0.80
MICROBAN_TELEOP_INTERMEDIATE_LINEAR_TRACKING_STD_M_S = 0.35
MICROBAN_TELEOP_INTERMEDIATE_ANGULAR_TRACKING_STD_RAD_S = 0.80
MICROBAN_TELEOP_HAND_TRACKING_STD_M = 0.08
MICROBAN_TELEOP_HAND_TRACKING_FINAL_STD_M = 0.05
MICROBAN_TELEOP_NEUTRAL_FOOT_TRACKING_WEIGHT = 1.0
MICROBAN_TELEOP_FOOT_TRACKING_FINAL_STD_M = 0.03
MICROBAN_TELEOP_INITIAL_HMD_NEUTRAL_PROBABILITY = 1.0
MICROBAN_TELEOP_MOVING_HMD_NEUTRAL_PROBABILITY = 0.2
MICROBAN_TELEOP_JOINT_LIMIT_GUARD_MARGIN_RATIO = 0.05
MICROBAN_TELEOP_JOINT_LIMIT_GUARD_LOOKAHEAD_S = 0.12
MICROBAN_TELEOP_INITIAL_VELOCITY_ENVELOPE = {
    "lin_vel_x": (-0.35, 0.40),
    "lin_vel_y": (-0.25, 0.25),
    "ang_vel_z": (-1.20, 1.20),
    "rotation_ang_vel_z": (-1.20, 1.20),
}
MICROBAN_TELEOP_INTERMEDIATE_VELOCITY_ENVELOPE = {
    "lin_vel_x": (-0.4, 0.5),
    "lin_vel_y": (-0.2, 0.2),
    "ang_vel_z": (-1.5, 1.5),
    "rotation_ang_vel_z": (-1.5, 1.5),
}
MICROBAN_TELEOP_FINAL_TRANSLATION_VELOCITY_ENVELOPE = {
    "lin_vel_x": (-0.5, 0.7),
    "lin_vel_y": (-0.3, 0.3),
    "ang_vel_z": (-1.5, 1.5),
    "rotation_ang_vel_z": (-1.5, 1.5),
}
MICROBAN_TELEOP_FINAL_VELOCITY_ENVELOPE = {
    "lin_vel_x": (-0.5, 0.7),
    "lin_vel_y": (-0.3, 0.3),
    "ang_vel_z": (-1.5, 1.5),
    "rotation_ang_vel_z": (-3.0, 3.0),
}

MICROBAN_TELEOP_ISOLATED_AXIS_PROBABILITIES = {
    "standing": 0.10,
    "forward": 0.15,
    "backward": 0.15,
    "lateral_left": 0.15,
    "lateral_right": 0.15,
    "yaw_left": 0.15,
    "yaw_right": 0.15,
    "mixed": 0.0,
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
MICROBAN_TELEOP_INITIAL_SIGNED_AXIS_RANGES = {
    "forward": (0.25, 0.40),
    "backward": (-0.35, -0.20),
    "lateral_left": (0.15, 0.25),
    "lateral_right": (-0.25, -0.15),
    "yaw_left": (0.80, 1.20),
    "yaw_right": (-1.20, -0.80),
}
MICROBAN_TELEOP_INTERMEDIATE_SIGNED_AXIS_RANGES = {
    "forward": (0.10, 0.50),
    "backward": (-0.40, -0.10),
    "lateral_left": (0.08, 0.20),
    "lateral_right": (-0.20, -0.08),
    "yaw_left": (0.40, 1.50),
    "yaw_right": (-1.50, -0.40),
}
MICROBAN_TELEOP_FINAL_TRANSLATION_SIGNED_AXIS_RANGES = {
    "forward": (0.10, 0.70),
    "backward": (-0.50, -0.10),
    "lateral_left": (0.08, 0.30),
    "lateral_right": (-0.30, -0.08),
    "yaw_left": (0.40, 1.50),
    "yaw_right": (-1.50, -0.40),
}
MICROBAN_TELEOP_FINAL_SIGNED_AXIS_RANGES = {
    **MICROBAN_TELEOP_FINAL_TRANSLATION_SIGNED_AXIS_RANGES,
    "yaw_left": (0.40, 3.00),
    "yaw_right": (-3.00, -0.40),
}
MICROBAN_TELEOP_PRIOR_INITIAL_AXIS_PROBABILITIES = {
    "standing": 0.20,
    "forward": 0.80,
    "backward": 0.0,
    "lateral_left": 0.0,
    "lateral_right": 0.0,
    "yaw_left": 0.0,
    "yaw_right": 0.0,
    "mixed": 0.0,
}
MICROBAN_TELEOP_PRIOR_SIGNED_AXIS_RANGES = {
    **MICROBAN_TELEOP_INITIAL_SIGNED_AXIS_RANGES,
    "forward": MICROBAN_LOCOMOTION_PRIOR_FORWARD_VELOCITY_RANGE_M_S,
}
MICROBAN_TELEOP_PRIOR_ACTION_REWARD_WEIGHT = 0.0
MICROBAN_TELEOP_PRIOR_JOINT_REWARD_WEIGHT = 0.0
MICROBAN_TELEOP_PRIOR_REWARD_STD_RAD = 0.15


def microban_teleop_initial_action_std() -> tuple[float, ...]:
    """Match the dedicated bounded safe-velocity actor's initial widths."""

    lower, upper = microban_teleop_action_delta_bounds()
    values: list[float] = []
    for name, lower_value, upper_value in zip(
        MICROBAN_TELEOP_ACTION_JOINT_NAMES, lower, upper, strict=True
    ):
        if "shoulder_roll" in name:
            values.append(min(-lower_value, upper_value) * 1024.0 / 18.0)
        elif "shoulder" in name or "elbow" in name:
            values.append(0.05)
        else:
            values.append(0.08)
    return tuple(values)


def microban_teleop_action_delta_bounds() -> tuple[
    tuple[float, ...], tuple[float, ...]
]:
    """Return actor-output bounds in the exact 18-joint raw-delta order.

    ``JointPositionAction`` uses scale 1.0 and the task's configured default
    joint position as its offset, then applies the XML-derived absolute target
    clips.  The actor uses the shared five-percent guarded interval derived from
    those values; the environment/runtime hard clip remains at the wider soft
    limits.  Deployment metadata calls the same helper over resolved tensors.
    """

    defaults = dict(MICROBAN_ROBOT_CFG.init_state.joint_pos or {})
    ordered_defaults = tuple(
        float(defaults[name]) for name in MICROBAN_TELEOP_ACTION_JOINT_NAMES
    )
    ordered_lower = tuple(
        MICROBAN_BODY_JOINT_SOFT_LIMITS[name][0]
        for name in MICROBAN_TELEOP_ACTION_JOINT_NAMES
    )
    ordered_upper = tuple(
        MICROBAN_BODY_JOINT_SOFT_LIMITS[name][1]
        for name in MICROBAN_TELEOP_ACTION_JOINT_NAMES
    )
    return guarded_teleop_actor_raw_bounds(
        ordered_defaults,
        ordered_lower,
        ordered_upper,
        (1.0,) * len(MICROBAN_TELEOP_ACTION_JOINT_NAMES),
    )


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


def _set_teleop_locomotion_stage(
    env: object,
    *,
    envelope: dict[str, tuple[float, float]],
    signed_axis_ranges: dict[str, tuple[float, float]],
    signed_axis_probabilities: dict[str, float],
    linear_tracking_std: float,
    angular_tracking_std: float,
) -> None:
    """Apply one auditable command-acquisition stage atomically."""

    command = env.command_manager.get_term_cfg("twist")
    if not isinstance(command, UniformVelocityCommandWithRotationCfg):
        raise TypeError("Teleop locomotion stage requires the rotation command cfg")
    set_command_velocity(
        env,
        lin_vel_x=envelope["lin_vel_x"],
        lin_vel_y=envelope["lin_vel_y"],
        ang_vel_z=envelope["ang_vel_z"],
        rotation_env_ang_vel_z=envelope["rotation_ang_vel_z"],
    )
    command.signed_axis_ranges = deepcopy(signed_axis_ranges)
    command.signed_axis_probabilities = deepcopy(signed_axis_probabilities)
    env.reward_manager.get_term_cfg("track_linear_velocity").params["std"] = (
        linear_tracking_std
    )
    env.reward_manager.get_term_cfg("track_angular_velocity").params["std"] = (
        angular_tracking_std
    )


def _set_hmd_neck_neutral_probability(
    env: object, *, neutral_probability: float
) -> None:
    """Update the live stateful HMD disturbance and its auditable config.

    ``HmdNeckTargetMotion`` copies this value during construction, so mutating
    only ``EventTermCfg.params`` would silently leave the running event on its
    old probability.  Resume reconstruction applies this helper again through
    the step curriculum before collecting the next rollout.
    """

    if not 0.0 <= neutral_probability <= 1.0:
        raise ValueError("HMD neutral probability must be in [0, 1]")
    event_cfg = env.event_manager.get_term_cfg("hmd_neck_target_motion")
    motion = event_cfg.func
    if not isinstance(motion, HmdNeckTargetMotion):
        raise TypeError("HMD neck event did not build its stateful motion term")
    motion.neutral_probability = neutral_probability
    event_cfg.params["neutral_probability"] = neutral_probability


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
        initial_twist = cfg.commands["twist"]
        initial_twist.ranges.lin_vel_x = MICROBAN_TELEOP_INITIAL_VELOCITY_ENVELOPE[
            "lin_vel_x"
        ]
        initial_twist.ranges.lin_vel_y = MICROBAN_TELEOP_INITIAL_VELOCITY_ENVELOPE[
            "lin_vel_y"
        ]
        initial_twist.ranges.ang_vel_z = MICROBAN_TELEOP_INITIAL_VELOCITY_ENVELOPE[
            "ang_vel_z"
        ]
        initial_twist.rotation_env_ang_vel_range = (
            MICROBAN_TELEOP_INITIAL_VELOCITY_ENVELOPE["rotation_ang_vel_z"]
        )
        # Contract v9 starts inside the velocity range certified by the accepted
        # safe-source gate. At update 500 it expands to signed isolated axes.
        initial_twist.signed_axis_probabilities = deepcopy(
            MICROBAN_TELEOP_PRIOR_INITIAL_AXIS_PROBABILITIES
        )
        initial_twist.signed_axis_ranges = deepcopy(
            MICROBAN_TELEOP_PRIOR_SIGNED_AXIS_RANGES
        )
        initial_twist.rel_standing_envs = 0.0
        initial_twist.rel_forward_envs = 0.0
        initial_twist.rel_rotation_envs = 0.0
        initial_twist.rel_heading_envs = 0.0
        initial_twist.rel_world_envs = 0.0
        initial_twist.init_velocity_prob = 0.0

    # Use the shared robot HOME exactly: shoulder pitch 0 degrees, asymmetric
    # shoulder roll -10/+10 degrees, and elbows -20 degrees.  This is a software
    # command convention; it is not presented as a measured hardware zero.
    teleop_joint_pos = cfg.scene.entities["robot"].init_state.joint_pos
    if teleop_joint_pos is None:
        raise ValueError("Microban teleop requires an explicit initial joint pose")
    teleop_joint_pos["left_shoulder_pitch"] = 0.0
    teleop_joint_pos["right_shoulder_pitch"] = 0.0

    # Head yaw + two neck axes are owned by the HMD controller.  Exact names make
    # accidental action-space growth fail loudly in the smoke test/exporter.
    action = cfg.actions["joint_pos"]
    if not isinstance(action, JointPositionActionCfg):
        raise TypeError("Expected velocity base task to use JointPositionActionCfg")
    action.actuator_names = MICROBAN_TELEOP_ACTION_JOINT_NAMES
    action.scale = 1.0
    action.clip = MICROBAN_BODY_JOINT_SOFT_LIMITS

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
                # its full runtime range.  The 8,000-update stage switches the
                # live stateful term to the deployment-like distribution.
                "neutral_probability": (
                    MICROBAN_TELEOP_INITIAL_HMD_NEUTRAL_PROBABILITY
                ),
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

    # The v1 task fed back the unbounded network output even though the actuator
    # received a soft-clipped absolute target.  Together with a raw action-rate
    # reward that created an unstable hidden recurrence.  V2 feeds back the
    # actually effective target expressed in the original delta coordinates.
    actor_terms["actions"] = ObservationTermCfg(
        func=effective_action_after_target_clip,
        params={"action_name": "joint_pos"},
    )
    cfg.observations["critic"].terms["actions"] = ObservationTermCfg(
        func=effective_action_after_target_clip,
        params={"action_name": "joint_pos"},
    )

    actor_terms["foot_target"] = ObservationTermCfg(
        func=foot_target_offset_b,
        params={"command_name": "foot_target"},
    )
    actor_terms["hand_target"] = ObservationTermCfg(
        func=hand_target_offset_b,
        params={"command_name": "hand_target"},
    )
    cfg.observations["critic"].terms["foot_target"] = ObservationTermCfg(
        func=foot_target_offset_b,
        params={"command_name": "foot_target"},
    )
    cfg.observations["critic"].terms["hand_target"] = ObservationTermCfg(
        func=hand_target_offset_b,
        params={"command_name": "hand_target"},
    )

    # Exact-zero standing receives a small foot anchor from the first update.
    # Its 1 cm/s fade makes the reward exactly zero for every signed locomotion
    # sample (the smallest commanded translation is 6 cm/s and yaw is 0.4
    # rad/s), so it cannot reward the stationary local optimum on moving tasks.
    # Hand targets remain independent of walking and their two active flags
    # mask inactive hands.
    cfg.rewards["foot_target_tracking"] = RewardTermCfg(
        func=foot_target_tracking_error_exp,
        weight=MICROBAN_TELEOP_NEUTRAL_FOOT_TRACKING_WEIGHT,
        params={
            "command_name": "foot_target",
            "std": 0.05,
            "velocity_command_name": "twist",
            "velocity_fade_range": (0.0, 0.01),
        },
    )
    cfg.rewards["hand_target_tracking"] = RewardTermCfg(
        func=hand_target_tracking_error_exp,
        weight=0.0,
        params={
            "command_name": "hand_target",
            "std": MICROBAN_TELEOP_HAND_TRACKING_STD_M,
        },
    )
    cfg.rewards["target_clip_excess"] = RewardTermCfg(
        func=normalized_target_clip_excess_l1_sum,
        weight=-2.0,
        params={"action_name": "joint_pos"},
    )
    cfg.rewards["target_near_limit"] = RewardTermCfg(
        func=normalized_target_near_limit_l1_sum,
        weight=-1.0,
        params={
            "action_name": "joint_pos",
            "margin_ratio": 0.05,
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
    cfg.rewards["raw_action_l2"] = RewardTermCfg(
        func=raw_action_l2,
        weight=-0.01,
        params={"action_name": "joint_pos"},
    )
    # Keep only a light smoothing prior.  At -0.1 this raw-coordinate term
    # dominated v1 and rewarded copying a saturated previous output forever.
    cfg.rewards["action_rate_l2"].weight = -0.02

    # Wider kernels keep a useful learning signal at the final command extrema;
    # the old Gaussian rewards were effectively zero before the policy moved.
    cfg.rewards["track_linear_velocity"].params["std"] = (
        MICROBAN_TELEOP_INITIAL_LINEAR_TRACKING_STD_M_S
    )
    cfg.rewards["track_angular_velocity"].params["std"] = (
        MICROBAN_TELEOP_INITIAL_ANGULAR_TRACKING_STD_RAD_S
    )
    cfg.rewards["linear_velocity_error_l1"] = RewardTermCfg(
        func=linear_velocity_tracking_error_l1,
        weight=-4.0,
        params={"command_name": "twist"},
    )
    cfg.rewards["yaw_velocity_error_l1"] = RewardTermCfg(
        func=yaw_velocity_tracking_error_l1,
        weight=-1.0,
        params={"command_name": "twist"},
    )

    # V2 converged to a wide static stance because the inherited term penalized
    # the 72 mm neutral foot spacing below an 80 mm threshold with weight -1000.
    # Keep a collision-avoidance margin, but do not make neutral stance itself a
    # dominant violation in this task.
    cfg.rewards["feet_distance"].weight = -100.0
    cfg.rewards["feet_distance"].params["min_dist"] = 0.07
    cfg.rewards["dof_pos_limits"].weight = -10.0
    cfg.rewards["no_stepping"].params["foot_target_command_name"] = "foot_target"

    # Six foot XYZ offsets, and six hand XYZ offsets plus left/right active
    # flags.  All offsets are expressed in the trunk frame and measured in metres.
    cfg.commands["foot_target"] = ResetFixedFootTargetCommandCfg(
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
    )
    cfg.commands["hand_target"] = ResetFixedHandTargetCommandCfg(
        resampling_time_range=(3.0, 8.0),
        rel_active=0.0,
        reach_xy_range=(-0.08, 0.08),
        reach_z_range=(-0.08, 0.08),
    )
    # This command is privileged: it is appended only to the critic below and
    # never changes the actor's deployment-stable 83-value observation schema.
    # Keep the term and all reward/termination shapes in play mode too, but
    # disable it so every one of its 39 values and both rewards are exact zero.
    cfg.commands["locomotion_prior"] = LocomotionPriorCommandCfg(
        resampling_time_range=(1.0e9, 1.0e9),
        motion_file=str(MICROBAN_LOCOMOTION_PRIOR_PATH),
        expected_sha256=MICROBAN_LOCOMOTION_PRIOR_SHA256,
        # Retain the 39-wide critic term for checkpoint topology stability, but
        # never activate the dynamically rejected walk004 reference.
        enabled=False,
    )
    cfg.observations["critic"].terms["locomotion_prior"] = ObservationTermCfg(
        func=velocity_mdp.generated_commands,
        params={"command_name": "locomotion_prior"},
    )
    cfg.rewards["locomotion_prior_action_target"] = RewardTermCfg(
        func=locomotion_prior_action_target_error_exp,
        weight=MICROBAN_TELEOP_PRIOR_ACTION_REWARD_WEIGHT,
        params={
            "command_name": "locomotion_prior",
            "action_name": "joint_pos",
            "std": MICROBAN_TELEOP_PRIOR_REWARD_STD_RAD,
        },
    )
    cfg.rewards["locomotion_prior_joint_position"] = RewardTermCfg(
        func=locomotion_prior_joint_position_error_exp,
        weight=MICROBAN_TELEOP_PRIOR_JOINT_REWARD_WEIGHT,
        params={
            "command_name": "locomotion_prior",
            "std": MICROBAN_TELEOP_PRIOR_REWARD_STD_RAD,
        },
    )
    cfg.terminations.pop("locomotion_prior_clip_finished", None)

    # V10 never applies the dynamically rejected walk004 prior or direct BC.
    # The first stage changes only the command sampler, then acquires one signed
    # axis at a time before introducing mixed commands.
    # The v9 model_1499 state is migrated in full, then contract-v10 gates at
    # 3000/7000/10000/15000. Limb tracking stays behind the locomotion stages so
    # a stationary multi-objective optimum cannot win.
    cfg.curriculum = {
        "staged_curriculum": CurriculumTermCfg(
            func=ResumeSafeStepBasedStagedCurriculum,
            params={
                "stages": [
                    {
                        "name": "expand from safe-source forward to signed axes",
                        "step": 500 * 24,
                        "apply": lambda env: _set_teleop_locomotion_stage(
                            env,
                            envelope=MICROBAN_TELEOP_INITIAL_VELOCITY_ENVELOPE,
                            signed_axis_ranges=(
                                MICROBAN_TELEOP_INITIAL_SIGNED_AXIS_RANGES
                            ),
                            signed_axis_probabilities=(
                                MICROBAN_TELEOP_ISOLATED_AXIS_PROBABILITIES
                            ),
                            linear_tracking_std=(
                                MICROBAN_TELEOP_INITIAL_LINEAR_TRACKING_STD_M_S
                            ),
                            angular_tracking_std=(
                                MICROBAN_TELEOP_INITIAL_ANGULAR_TRACKING_STD_RAD_S
                            ),
                        ),
                    },
                    {
                        "name": "intermediate isolated signed axes",
                        "step": 1500 * 24,
                        "apply": lambda env: _set_teleop_locomotion_stage(
                            env,
                            envelope=(MICROBAN_TELEOP_INTERMEDIATE_VELOCITY_ENVELOPE),
                            signed_axis_ranges=(
                                MICROBAN_TELEOP_INTERMEDIATE_SIGNED_AXIS_RANGES
                            ),
                            signed_axis_probabilities=(
                                MICROBAN_TELEOP_ISOLATED_AXIS_PROBABILITIES
                            ),
                            linear_tracking_std=(
                                MICROBAN_TELEOP_INTERMEDIATE_LINEAR_TRACKING_STD_M_S
                            ),
                            angular_tracking_std=(
                                MICROBAN_TELEOP_INTERMEDIATE_ANGULAR_TRACKING_STD_RAD_S
                            ),
                        ),
                    },
                    {
                        "name": "final translation and moving-yaw isolated axes",
                        "step": 3000 * 24,
                        "apply": lambda env: _set_teleop_locomotion_stage(
                            env,
                            envelope=(
                                MICROBAN_TELEOP_FINAL_TRANSLATION_VELOCITY_ENVELOPE
                            ),
                            signed_axis_ranges=(
                                MICROBAN_TELEOP_FINAL_TRANSLATION_SIGNED_AXIS_RANGES
                            ),
                            signed_axis_probabilities=(
                                MICROBAN_TELEOP_ISOLATED_AXIS_PROBABILITIES
                            ),
                            linear_tracking_std=(
                                MICROBAN_TELEOP_LINEAR_TRACKING_STD_M_S
                            ),
                            angular_tracking_std=(
                                MICROBAN_TELEOP_ANGULAR_TRACKING_STD_RAD_S
                            ),
                        ),
                    },
                    {
                        "name": "final pure-yaw isolated axes",
                        "step": 4500 * 24,
                        "apply": lambda env: _set_teleop_locomotion_stage(
                            env,
                            envelope=MICROBAN_TELEOP_FINAL_VELOCITY_ENVELOPE,
                            signed_axis_ranges=(
                                MICROBAN_TELEOP_FINAL_SIGNED_AXIS_RANGES
                            ),
                            signed_axis_probabilities=(
                                MICROBAN_TELEOP_ISOLATED_AXIS_PROBABILITIES
                            ),
                            linear_tracking_std=(
                                MICROBAN_TELEOP_LINEAR_TRACKING_STD_M_S
                            ),
                            angular_tracking_std=(
                                MICROBAN_TELEOP_ANGULAR_TRACKING_STD_RAD_S
                            ),
                        ),
                    },
                    {
                        "name": "runtime envelope with mixed command replay",
                        "step": 6000 * 24,
                        "apply": lambda env: _set_teleop_locomotion_stage(
                            env,
                            envelope=MICROBAN_TELEOP_FINAL_VELOCITY_ENVELOPE,
                            signed_axis_ranges=(
                                MICROBAN_TELEOP_FINAL_SIGNED_AXIS_RANGES
                            ),
                            signed_axis_probabilities=(
                                MICROBAN_TELEOP_MIXED_AXIS_PROBABILITIES
                            ),
                            linear_tracking_std=(
                                MICROBAN_TELEOP_LINEAR_TRACKING_STD_M_S
                            ),
                            angular_tracking_std=(
                                MICROBAN_TELEOP_ANGULAR_TRACKING_STD_RAD_S
                            ),
                        ),
                    },
                    {
                        "name": (
                            "enable moving-HMD, stationary no-step guard, and broad "
                            "hand tracking"
                        ),
                        "step": 7000 * 24,
                        "apply": lambda env: (
                            set_stepping_parameters(
                                env,
                                air_time_weight=3.0,
                                no_stepping_penalty_weight=-1.0,
                            ),
                            _set_hmd_neck_neutral_probability(
                                env,
                                neutral_probability=(
                                    MICROBAN_TELEOP_MOVING_HMD_NEUTRAL_PROBABILITY
                                ),
                            ),
                            env.reward_manager.get_term_cfg(
                                "hand_target_tracking"
                            ).__setattr__("weight", 1.0),
                            env.reward_manager.get_term_cfg(
                                "hand_target_tracking"
                            ).params.__setitem__(
                                "std", MICROBAN_TELEOP_HAND_TRACKING_STD_M
                            ),
                            env.command_manager.get_term_cfg("hand_target").__setattr__(
                                "rel_active", 0.7
                            ),
                        ),
                    },
                    {
                        "name": "tighten hand tracking",
                        "step": 8500 * 24,
                        "apply": lambda env: (
                            env.reward_manager.get_term_cfg(
                                "hand_target_tracking"
                            ).__setattr__("weight", 2.0),
                            env.reward_manager.get_term_cfg(
                                "hand_target_tracking"
                            ).params.__setitem__(
                                "std", MICROBAN_TELEOP_HAND_TRACKING_FINAL_STD_M
                            ),
                        ),
                    },
                    {
                        "name": "enable broad stationary foot tracking",
                        "step": 10000 * 24,
                        "apply": lambda env: (
                            env.reward_manager.get_term_cfg(
                                "foot_target_tracking"
                            ).__setattr__("weight", 2.0),
                            env.reward_manager.get_term_cfg(
                                "foot_target_tracking"
                            ).params.__setitem__("std", 0.05),
                            env.reward_manager.get_term_cfg(
                                "foot_target_tracking"
                            ).params.__setitem__("velocity_fade_range", (0.0, 0.15)),
                            env.command_manager.get_term_cfg("foot_target").__setattr__(
                                "rel_single_support_envs", 0.3
                            ),
                            env.command_manager.get_term_cfg("foot_target").__setattr__(
                                "rel_both_feet_envs", 0.05
                            ),
                        ),
                    },
                    {
                        "name": "tighten foot tracking",
                        "step": 12000 * 24,
                        "apply": lambda env: (
                            env.reward_manager.get_term_cfg(
                                "foot_target_tracking"
                            ).__setattr__("weight", 3.0),
                            env.reward_manager.get_term_cfg(
                                "foot_target_tracking"
                            ).params.__setitem__(
                                "std", MICROBAN_TELEOP_FOOT_TRACKING_FINAL_STD_M
                            ),
                            env.command_manager.get_term_cfg("foot_target").__setattr__(
                                "rel_both_feet_envs", 0.1
                            ),
                            env.command_manager.get_term_cfg("foot_target").__setattr__(
                                "both_feet_lift_height_range",
                                (
                                    MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M,
                                    MICROBAN_TELEOP_FINAL_BOTH_FEET_LIFT_UPPER_M,
                                ),
                            ),
                        ),
                    },
                    {
                        "name": "materialize final deployment envelope",
                        "step": 15000 * 24,
                        "apply": lambda env: (
                            env.command_manager.get_term_cfg("foot_target").__setattr__(
                                "both_feet_lift_height_range",
                                (
                                    MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M,
                                    MICROBAN_TELEOP_FINAL_BOTH_FEET_LIFT_UPPER_M,
                                ),
                            ),
                            env.command_manager.get_term_cfg("hand_target").__setattr__(
                                "rel_active", 0.7
                            ),
                        ),
                    },
                ]
            },
        )
    }

    if play:
        # Deterministic neutral keypoints by default.  A live/specialized player
        # may write explicit PICO commands after reset.
        cfg.curriculum = {}
        cfg.commands["foot_target"].rel_single_support_envs = 0.0
        cfg.commands["foot_target"].rel_both_feet_envs = 0.0
        cfg.commands["hand_target"].rel_active = 0.0

    return cfg


@dataclass
class MicrobanTeleopRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Contract-v10 fixed-LR full-state migration/resume configuration."""

    # This is an intentionally separate constructor path for tools that create a
    # blank policy and immediately load a validated checkpoint actor.  It must
    # never be enabled by a training launch.
    checkpoint_consumer_mode: bool = False
    safe_velocity_checkpoint: str | None = None
    safe_velocity_checkpoint_sha256: str | None = None
    safe_velocity_acceptance_receipt: str | None = None
    save_pristine_checkpoint: bool = False


MicrobanTeleopRlCfg = MicrobanTeleopRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        # Match the source actor and avoid changing coordinates between PPO
        # rollout collection and old-log-probability evaluation.
        obs_normalization=False,
        distribution_cfg={
            "class_name": MicrobanSafeVelocityBoundedGaussianDistribution,
            "init_std": microban_teleop_initial_action_std(),
            "lower_bound": microban_teleop_action_delta_bounds()[0],
            "upper_bound": microban_teleop_action_delta_bounds()[1],
            "std_type": "log",
            "latent_scale_multiplier": (MICROBAN_TELEOP_ACTOR_LATENT_SCALE_MULTIPLIER),
            "latent_abs_max": MICROBAN_TELEOP_ACTOR_LATENT_ABS_MAX,
            "latent_mean_fraction": MICROBAN_TELEOP_ACTOR_LATENT_MEAN_FRACTION,
            "std_min_abs_max": MICROBAN_TELEOP_ACTOR_STD_MIN_ABS_MAX,
            "std_min_envelope_divisor": (
                MICROBAN_TELEOP_ACTOR_STD_MIN_ENVELOPE_DIVISOR
            ),
            "std_abs_max": MICROBAN_TELEOP_ACTOR_STD_ABS_MAX,
            "std_envelope_divisor": (MICROBAN_TELEOP_ACTOR_STD_ENVELOPE_DIVISOR),
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=MicrobanTeleopPpoAlgorithmCfg(
        class_name=("mjlab_microban.tasks.microban_teleop_mdp:LatentActionPPO"),
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        # Three passes retain the accepted contract-v9 PPO topology.
        num_learning_epochs=3,
        num_mini_batches=4,
        # V9's adaptive rate rose after the safe model_2500 region and degraded
        # deterministic yaw safety. V10 therefore pins both the algorithm scalar
        # and every optimizer param group at the observed 1e-5 floor.
        learning_rate=1.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    ),
    wandb_project="mjlab_microban_teleop",
    experiment_name="mjlab_microban_teleop",
    save_interval=100,
    num_steps_per_env=MICROBAN_TELEOP_NUM_STEPS_PER_ENV,
    max_iterations=15_000,
)
