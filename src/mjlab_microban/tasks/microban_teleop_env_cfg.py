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

import math
from copy import deepcopy
from dataclasses import dataclass, fields
from xml.etree import ElementTree

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
    MICROBAN_XML,
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
    guarded_teleop_actor_raw_bounds,
)
from mjlab_microban.tasks.microban_teleop_mdp import (
    MICROBAN_HMD_RETARGET_INTERVAL_S,
    MICROBAN_HMD_RUNTIME_LIMITS_RAD,
    MICROBAN_HMD_SLEW_RATES_RAD_S,
    MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M,
    AsymmetricBoundedGaussianDistribution,
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
from mjlab_microban.tasks.microban_velocity_env_cfg import (
    make_microban_velocity_env_cfg,
)


def _microban_soft_joint_position_clip() -> dict[str, tuple[float, float]]:
    """Derive action target clips with Entity's soft-limit formula.

    Keeping this derived from the model XML prevents the task and robot limits
    from drifting apart.  ``JointPositionAction`` applies the clip after adding
    the default-pose offset, so these are absolute target-position limits.
    """

    articulation = MICROBAN_ROBOT_CFG.articulation
    if articulation is None:
        raise ValueError("Microban must be configured as an articulation")
    factor = articulation.soft_joint_pos_limit_factor

    ranges: dict[str, tuple[float, float]] = {}
    root = ElementTree.parse(MICROBAN_XML).getroot()
    for joint in root.iter("joint"):
        name = joint.get("name")
        raw_range = joint.get("range")
        if name is None or raw_range is None:
            continue
        lower, upper = (float(value) for value in raw_range.split())
        midpoint = 0.5 * (lower + upper)
        half_range = 0.5 * (upper - lower) * factor
        ranges[name] = (midpoint - half_range, midpoint + half_range)

    missing = set(MICROBAN_TELEOP_ACTION_JOINT_NAMES) - ranges.keys()
    if missing:
        raise ValueError(f"Missing Microban joint ranges: {sorted(missing)}")
    return {name: ranges[name] for name in MICROBAN_TELEOP_ACTION_JOINT_NAMES}


MICROBAN_TELEOP_LINEAR_TRACKING_STD_M_S = 0.5
MICROBAN_TELEOP_ANGULAR_TRACKING_STD_RAD_S = 1.25
MICROBAN_TELEOP_HAND_TRACKING_STD_M = 0.08
MICROBAN_TELEOP_HAND_TRACKING_FINAL_STD_M = 0.05
MICROBAN_TELEOP_FOOT_TRACKING_FINAL_STD_M = 0.03
MICROBAN_TELEOP_JOINT_LIMIT_GUARD_MARGIN_RATIO = 0.05
MICROBAN_TELEOP_JOINT_LIMIT_GUARD_LOOKAHEAD_S = 0.12
MICROBAN_TELEOP_INITIAL_VELOCITY_ENVELOPE = {
    "lin_vel_x": (-0.2, 0.3),
    "lin_vel_y": (-0.1, 0.1),
    "ang_vel_z": (-0.4, 0.4),
    "rotation_ang_vel_z": (-0.8, 0.8),
}
MICROBAN_TELEOP_INTERMEDIATE_VELOCITY_ENVELOPE = {
    "lin_vel_x": (-0.4, 0.5),
    "lin_vel_y": (-0.2, 0.2),
    "ang_vel_z": (-0.8, 0.8),
    "rotation_ang_vel_z": (-1.5, 1.5),
}
MICROBAN_TELEOP_FINAL_VELOCITY_ENVELOPE = {
    "lin_vel_x": (-0.5, 0.7),
    "lin_vel_y": (-0.3, 0.3),
    "ang_vel_z": (-1.5, 1.5),
    "rotation_ang_vel_z": (-3.0, 3.0),
}


def microban_teleop_initial_action_std() -> tuple[float, ...]:
    """Return ordered 3-sigma-within-nearest-bound exploration widths.

    Values are capped at 0.15 rad.  Shoulder roll is intentionally much
    smaller: its 10-degree home is only one degree inside the 90% soft limit,
    so its exact value is one third of that one-degree headroom.
    """

    defaults = dict(MICROBAN_ROBOT_CFG.init_state.joint_pos or {})
    defaults["left_shoulder_pitch"] = math.radians(10.0)
    defaults["right_shoulder_pitch"] = math.radians(10.0)
    clips = _microban_soft_joint_position_clip()
    values: list[float] = []
    for name in MICROBAN_TELEOP_ACTION_JOINT_NAMES:
        lower, upper = clips[name]
        default = float(defaults[name])
        headroom = min(default - lower, upper - default)
        if headroom <= 0.0:
            raise ValueError(f"{name} home pose is outside its action clip")
        values.append(min(0.15, headroom / 3.0))
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
    defaults["left_shoulder_pitch"] = math.radians(10.0)
    defaults["right_shoulder_pitch"] = math.radians(10.0)
    clips = _microban_soft_joint_position_clip()
    ordered_defaults = tuple(
        float(defaults[name]) for name in MICROBAN_TELEOP_ACTION_JOINT_NAMES
    )
    ordered_lower = tuple(clips[name][0] for name in MICROBAN_TELEOP_ACTION_JOINT_NAMES)
    ordered_upper = tuple(clips[name][1] for name in MICROBAN_TELEOP_ACTION_JOINT_NAMES)
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

    init_fields = tuple(
        field.name
        for field in fields(UniformVelocityCommandWithRotationCfg)
        if field.init
    )
    missing = tuple(name for name in init_fields if not hasattr(command, name))
    if missing:
        raise TypeError(
            "Velocity command is missing fields required by "
            f"UniformVelocityCommandWithRotationCfg: {missing}"
        )
    return UniformVelocityCommandWithRotationCfg(
        **{name: deepcopy(getattr(command, name)) for name in init_fields}
    )


def make_microban_teleop_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build the independent PICO hybrid policy training environment."""

    # Reuse the validated locomotion dynamics/rewards without mutating the
    # deployed velocity configuration object.
    cfg = make_microban_velocity_env_cfg(play=play)

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
        initial_twist.rel_rotation_envs = 0.1

    # The current control runtime's NEUTRAL_POSE constant uses +10 degrees for
    # both shoulder-pitch joints.  This is a provisional software-contract match,
    # not a measured physical calibration or a claim about which task predates
    # another.  Keep the override local to this task so the deployed/get-up tasks
    # remain untouched.
    teleop_joint_pos = cfg.scene.entities["robot"].init_state.joint_pos
    if teleop_joint_pos is None:
        raise ValueError("Microban teleop requires an explicit initial joint pose")
    teleop_joint_pos["left_shoulder_pitch"] = math.radians(10.0)
    teleop_joint_pos["right_shoulder_pitch"] = math.radians(10.0)

    # Head yaw + two neck axes are owned by the HMD controller.  Exact names make
    # accidental action-space growth fail loudly in the smoke test/exporter.
    action = cfg.actions["joint_pos"]
    if not isinstance(action, JointPositionActionCfg):
        raise TypeError("Expected velocity base task to use JointPositionActionCfg")
    action.actuator_names = MICROBAN_TELEOP_ACTION_JOINT_NAMES
    action.scale = 1.0
    action.clip = _microban_soft_joint_position_clip()

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
                "neutral_probability": 0.2,
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

    # Foot targets fade out as locomotion demand grows.  Hand targets remain
    # independent of walking and their two active flags mask inactive hands.
    cfg.rewards["foot_target_tracking"] = RewardTermCfg(
        func=foot_target_tracking_error_exp,
        weight=0.0,
        params={
            "command_name": "foot_target",
            "std": 0.05,
            "velocity_command_name": "twist",
            "velocity_fade_range": (0.0, 0.15),
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
        MICROBAN_TELEOP_LINEAR_TRACKING_STD_M_S
    )
    cfg.rewards["track_angular_velocity"].params["std"] = (
        MICROBAN_TELEOP_ANGULAR_TRACKING_STD_RAD_S
    )
    cfg.rewards["linear_velocity_error_l1"] = RewardTermCfg(
        func=linear_velocity_tracking_error_l1,
        weight=-2.0,
        params={"command_name": "twist"},
    )
    cfg.rewards["yaw_velocity_error_l1"] = RewardTermCfg(
        func=yaw_velocity_tracking_error_l1,
        weight=-0.5,
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

    # V4 retains the capability order: establish locomotion over the complete
    # runtime envelope, add hands with a broad then precise kernel, and only then
    # introduce the more destabilizing single-/two-foot targets.  This avoids the
    # v2 failure mode where every objective arrived before the actor had learned
    # to respond to twist commands and a stable stationary policy won instead.
    cfg.curriculum = {
        "staged_curriculum": CurriculumTermCfg(
            func=ResumeSafeStepBasedStagedCurriculum,
            params={
                "stages": [
                    {
                        "name": "intermediate locomotion envelope",
                        "step": 1000 * 24,
                        "apply": lambda env: (
                            set_command_velocity(
                                env,
                                lin_vel_x=(
                                    MICROBAN_TELEOP_INTERMEDIATE_VELOCITY_ENVELOPE[
                                        "lin_vel_x"
                                    ]
                                ),
                                lin_vel_y=(
                                    MICROBAN_TELEOP_INTERMEDIATE_VELOCITY_ENVELOPE[
                                        "lin_vel_y"
                                    ]
                                ),
                                ang_vel_z=(
                                    MICROBAN_TELEOP_INTERMEDIATE_VELOCITY_ENVELOPE[
                                        "ang_vel_z"
                                    ]
                                ),
                                rotation_env_ang_vel_z=(
                                    MICROBAN_TELEOP_INTERMEDIATE_VELOCITY_ENVELOPE[
                                        "rotation_ang_vel_z"
                                    ]
                                ),
                            ),
                            set_stepping_parameters(env, rel_rotation_envs=0.15),
                        ),
                    },
                    {
                        "name": "final runtime locomotion envelope",
                        "step": 2500 * 24,
                        "apply": lambda env: (
                            set_command_velocity(
                                env,
                                lin_vel_x=MICROBAN_TELEOP_FINAL_VELOCITY_ENVELOPE[
                                    "lin_vel_x"
                                ],
                                lin_vel_y=MICROBAN_TELEOP_FINAL_VELOCITY_ENVELOPE[
                                    "lin_vel_y"
                                ],
                                ang_vel_z=MICROBAN_TELEOP_FINAL_VELOCITY_ENVELOPE[
                                    "ang_vel_z"
                                ],
                                rotation_env_ang_vel_z=(
                                    MICROBAN_TELEOP_FINAL_VELOCITY_ENVELOPE[
                                        "rotation_ang_vel_z"
                                    ]
                                ),
                            ),
                            set_stepping_parameters(
                                env,
                                air_time_weight=3.0,
                                no_stepping_penalty_weight=-1.0,
                                rel_standing_envs=0.1,
                                rel_rotation_envs=0.25,
                            ),
                        ),
                    },
                    {
                        "name": "enable broad hand tracking",
                        "step": 4500 * 24,
                        "apply": lambda env: (
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
                        "step": 6500 * 24,
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
                        "name": "enable broad foot tracking",
                        "step": 7500 * 24,
                        "apply": lambda env: (
                            env.reward_manager.get_term_cfg(
                                "foot_target_tracking"
                            ).__setattr__("weight", 2.0),
                            env.reward_manager.get_term_cfg(
                                "foot_target_tracking"
                            ).params.__setitem__("std", 0.05),
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
                        "step": 9500 * 24,
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
                                (MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M, 0.02),
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
    """Runner configuration with an explicit, pinned velocity bootstrap opt-in."""

    bootstrap_velocity_checkpoint: str | None = None
    bootstrap_velocity_checkpoint_sha256: str | None = None
    save_pristine_checkpoint: bool = False


MicrobanTeleopRlCfg = MicrobanTeleopRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": AsymmetricBoundedGaussianDistribution,
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
    algorithm=RslRlPpoAlgorithmCfg(
        class_name=("mjlab_microban.tasks.microban_teleop_mdp:LatentActionPPO"),
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=3,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    ),
    wandb_project="mjlab_microban_teleop",
    experiment_name="mjlab_microban_teleop",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=15_000,
)
