# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Instantiate and step the Microban PICO teleoperation task.

This is a bounded pre-training gate: it validates the deployment observation and
action contracts, target clipping and finite simulation outputs without starting
PPO training.

Example:
    uv run python -m mjlab_microban.scripts.smoke_teleop_task --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy

import torch
from mjlab.envs import ManagerBasedRlEnv

from mjlab_microban.tasks.mdp import (
    UniformVelocityCommandWithRotation,
    UniformVelocityCommandWithRotationCfg,
)
from mjlab_microban.tasks.microban_locomotion_prior import (
    MICROBAN_LOCOMOTION_PRIOR_COMMAND_WIDTH,
    LocomotionPriorCommand,
)
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_TELEOP_ACTION_JOINT_NAMES,
    MICROBAN_TELEOP_ACTION_WIDTH,
    MICROBAN_TELEOP_ACTOR_DEFAULT_EPSILON_RAD,
    MICROBAN_TELEOP_ACTOR_LATENT_ABS_MAX,
    MICROBAN_TELEOP_ACTOR_LATENT_MEAN_FRACTION,
    MICROBAN_TELEOP_ACTOR_LATENT_SCALE_MULTIPLIER,
    MICROBAN_TELEOP_ACTOR_LIMIT_MARGIN_RATIO,
    MICROBAN_TELEOP_ACTOR_STD_ABS_MAX,
    MICROBAN_TELEOP_ACTOR_STD_ENVELOPE_DIVISOR,
    MICROBAN_TELEOP_ACTOR_STD_MIN_ABS_MAX,
    MICROBAN_TELEOP_ACTOR_STD_MIN_ENVELOPE_DIVISOR,
    MICROBAN_TELEOP_FINAL_BOTH_FEET_LIFT_UPPER_M,
    MICROBAN_TELEOP_OBSERVATION_SCHEMA,
    MICROBAN_TELEOP_OBSERVATION_WIDTH,
    get_microban_teleop_metadata,
    validate_microban_teleop_observation_contract,
)
from mjlab_microban.tasks.microban_teleop_env_cfg import (
    MICROBAN_TELEOP_ANGULAR_TRACKING_STD_RAD_S,
    MICROBAN_TELEOP_FINAL_SIGNED_AXIS_RANGES,
    MICROBAN_TELEOP_FINAL_TRANSLATION_SIGNED_AXIS_RANGES,
    MICROBAN_TELEOP_FINAL_TRANSLATION_VELOCITY_ENVELOPE,
    MICROBAN_TELEOP_FINAL_VELOCITY_ENVELOPE,
    MICROBAN_TELEOP_FOOT_TRACKING_FINAL_STD_M,
    MICROBAN_TELEOP_HAND_TRACKING_FINAL_STD_M,
    MICROBAN_TELEOP_INITIAL_HMD_NEUTRAL_PROBABILITY,
    MICROBAN_TELEOP_INITIAL_SIGNED_AXIS_RANGES,
    MICROBAN_TELEOP_INITIAL_VELOCITY_ENVELOPE,
    MICROBAN_TELEOP_INTERMEDIATE_ANGULAR_TRACKING_STD_RAD_S,
    MICROBAN_TELEOP_INTERMEDIATE_LINEAR_TRACKING_STD_M_S,
    MICROBAN_TELEOP_INTERMEDIATE_SIGNED_AXIS_RANGES,
    MICROBAN_TELEOP_INTERMEDIATE_VELOCITY_ENVELOPE,
    MICROBAN_TELEOP_ISOLATED_AXIS_PROBABILITIES,
    MICROBAN_TELEOP_LINEAR_TRACKING_STD_M_S,
    MICROBAN_TELEOP_MIXED_AXIS_PROBABILITIES,
    MICROBAN_TELEOP_MOVING_HMD_NEUTRAL_PROBABILITY,
    MICROBAN_TELEOP_NEUTRAL_FOOT_TRACKING_WEIGHT,
    MICROBAN_TELEOP_PRIOR_INITIAL_AXIS_PROBABILITIES,
    MICROBAN_TELEOP_PRIOR_SIGNED_AXIS_RANGES,
    make_microban_teleop_env_cfg,
    microban_teleop_action_delta_bounds,
)
from mjlab_microban.tasks.microban_teleop_mdp import (
    MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M,
    HmdNeckTargetMotion,
    ResetFixedFootTargetCommand,
    ResetFixedHandTargetCommand,
    ResumeSafeStepBasedStagedCurriculum,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _assert_rotation_command_cfg(
    command_cfg: object,
    *,
    expected_rel_rotation_envs: float,
    expected_rotation_range: tuple[float, float],
    expected_signed_axis_probabilities: dict[str, float] | None = None,
    expected_signed_axis_ranges: dict[str, tuple[float, float]] | None = None,
) -> None:
    """Check rotation and optional v9 signed-axis config survive construction."""

    if not isinstance(command_cfg, UniformVelocityCommandWithRotationCfg):
        raise TypeError(
            "twist must be a formal UniformVelocityCommandWithRotationCfg, got "
            f"{type(command_cfg).__name__}"
        )
    if "build" in vars(command_cfg):
        raise AssertionError("twist still has a dynamically assigned build callback")
    if not math.isclose(
        command_cfg.rel_rotation_envs,
        expected_rel_rotation_envs,
        abs_tol=0.0,
    ):
        raise AssertionError(
            "Unexpected rotation-only environment fraction: "
            f"{command_cfg.rel_rotation_envs}"
        )
    if not math.isclose(command_cfg.rotation_min_ang_vel, 0.5, abs_tol=0.0):
        raise AssertionError(
            "Unexpected minimum rotation angular velocity: "
            f"{command_cfg.rotation_min_ang_vel}"
        )
    if command_cfg.rotation_env_ang_vel_range != expected_rotation_range:
        raise AssertionError(
            "Unexpected rotation-only angular velocity range: "
            f"{command_cfg.rotation_env_ang_vel_range}"
        )
    if command_cfg.signed_axis_probabilities != expected_signed_axis_probabilities:
        raise AssertionError(
            "Unexpected signed-axis probabilities: "
            f"{command_cfg.signed_axis_probabilities}"
        )
    if command_cfg.signed_axis_ranges != expected_signed_axis_ranges:
        raise AssertionError(
            f"Unexpected signed-axis ranges: {command_cfg.signed_axis_ranges}"
        )


def _assert_velocity_envelope(
    command_cfg: UniformVelocityCommandWithRotationCfg,
    expected: dict[str, tuple[float, float]],
) -> None:
    actual = {
        "lin_vel_x": command_cfg.ranges.lin_vel_x,
        "lin_vel_y": command_cfg.ranges.lin_vel_y,
        "ang_vel_z": command_cfg.ranges.ang_vel_z,
        "rotation_ang_vel_z": command_cfg.rotation_env_ang_vel_range,
    }
    if actual != expected:
        raise AssertionError(f"Velocity envelope drifted: {actual} != {expected}")


def _assert_runtime_signed_axis_sampling(
    twist: UniformVelocityCommandWithRotation,
    env_ids: torch.Tensor,
) -> None:
    """Exercise every v8 categorical mode through the real command term."""

    cfg = twist.cfg
    original_probabilities = deepcopy(cfg.signed_axis_probabilities)
    original_ranges = deepcopy(cfg.signed_axis_ranges)
    if original_probabilities is None or original_ranges is None:
        raise AssertionError("Training command did not enable signed-axis sampling")

    mode_names = tuple(original_probabilities)
    zero = torch.zeros(len(env_ids), device=env_ids.device)
    try:
        for mode in mode_names:
            cfg.signed_axis_probabilities = {
                name: float(name == mode) for name in mode_names
            }
            twist._resample_command(env_ids)
            command = twist.vel_command_b[env_ids]
            if not torch.equal(twist.vel_command_w[env_ids], command):
                raise AssertionError(
                    f"{mode} did not mirror body command to world storage"
                )

            nonzero_axes = torch.abs(command) > 0.0
            if mode == "standing":
                if torch.any(nonzero_axes) or not torch.all(
                    twist.is_standing_env[env_ids]
                ):
                    raise AssertionError("Standing mode emitted a non-zero command")
                continue

            if mode == "mixed":
                if not torch.all(nonzero_axes):
                    raise AssertionError("Mixed mode did not command all three axes")
                continue

            axis = 0 if mode in ("forward", "backward") else 1
            if mode in ("yaw_left", "yaw_right"):
                axis = 2
            if not torch.all(nonzero_axes[:, axis]):
                raise AssertionError(f"{mode} emitted a zero primary-axis command")
            other_axes = [candidate for candidate in range(3) if candidate != axis]
            if not torch.equal(command[:, other_axes], zero[:, None].expand(-1, 2)):
                raise AssertionError(f"{mode} leaked onto another command axis")
            lower, upper = original_ranges[mode]
            values = command[:, axis]
            if torch.any(values < lower) or torch.any(values > upper):
                raise AssertionError(f"{mode} escaped its signed range")
            if mode == "forward" and not torch.all(twist.is_forward_env[env_ids]):
                raise AssertionError("Forward mode flag was not materialized")
            if mode in ("yaw_left", "yaw_right") and not torch.all(
                twist.is_rotation_env[env_ids]
            ):
                raise AssertionError(f"{mode} rotation flag was not materialized")
    finally:
        cfg.signed_axis_probabilities = original_probabilities
        cfg.signed_axis_ranges = original_ranges
        twist._resample_command(env_ids)


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be at least 1")

    # The external HMD disturbance belongs to training only.  Play/live tools
    # must retain explicit ownership of the neck rather than fighting an event.
    play_cfg = make_microban_teleop_env_cfg(play=True)
    if "hmd_neck_target_motion" in play_cfg.events:
        raise AssertionError("HMD target randomization must be disabled in play mode")
    _assert_rotation_command_cfg(
        play_cfg.commands["twist"],
        expected_rel_rotation_envs=0.0,
        expected_rotation_range=(-1.5, 1.5),
    )

    cfg = make_microban_teleop_env_cfg(play=False)
    if cfg.sim.nconmax < 512 or cfg.sim.njmax < 2048:
        raise AssertionError(
            "Contract v9 requires nconmax>=512 and njmax>=2048"
        )
    _assert_rotation_command_cfg(
        cfg.commands["twist"],
        expected_rel_rotation_envs=0.0,
        expected_rotation_range=(
            MICROBAN_TELEOP_INITIAL_VELOCITY_ENVELOPE["rotation_ang_vel_z"]
        ),
        expected_signed_axis_probabilities=(
            MICROBAN_TELEOP_PRIOR_INITIAL_AXIS_PROBABILITIES
        ),
        expected_signed_axis_ranges=MICROBAN_TELEOP_PRIOR_SIGNED_AXIS_RANGES,
    )
    _assert_velocity_envelope(
        cfg.commands["twist"], MICROBAN_TELEOP_INITIAL_VELOCITY_ENVELOPE
    )
    hmd_event_cfg = cfg.events["hmd_neck_target_motion"]
    if hmd_event_cfg.params["neutral_probability"] != (
        MICROBAN_TELEOP_INITIAL_HMD_NEUTRAL_PROBABILITY
    ):
        raise AssertionError("Initial HMD neutral probability drifted")
    initial_foot_reward = cfg.rewards["foot_target_tracking"]
    if initial_foot_reward.weight != MICROBAN_TELEOP_NEUTRAL_FOOT_TRACKING_WEIGHT:
        raise AssertionError("Neutral foot tracking must be enabled from update zero")
    if initial_foot_reward.params["velocity_fade_range"] != (0.0, 0.01):
        raise AssertionError("Initial neutral-foot velocity fade range drifted")
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)

    try:
        twist = env.command_manager.get_term("twist")
        if not isinstance(twist, UniformVelocityCommandWithRotation):
            raise TypeError(
                f"twist built the wrong runtime command type: {type(twist).__name__}"
            )
        _assert_rotation_command_cfg(
            env.command_manager.get_term_cfg("twist"),
            expected_rel_rotation_envs=0.0,
            expected_rotation_range=(
                MICROBAN_TELEOP_INITIAL_VELOCITY_ENVELOPE["rotation_ang_vel_z"]
            ),
            expected_signed_axis_probabilities=(
                MICROBAN_TELEOP_PRIOR_INITIAL_AXIS_PROBABILITIES
            ),
            expected_signed_axis_ranges=MICROBAN_TELEOP_PRIOR_SIGNED_AXIS_RANGES,
        )
        env_ids = torch.arange(args.num_envs, device=args.device)
        _assert_runtime_signed_axis_sampling(twist, env_ids)

        action = env.action_manager.get_term("joint_pos")
        if tuple(action.target_names) != MICROBAN_TELEOP_ACTION_JOINT_NAMES:
            raise AssertionError(f"Unexpected action order: {action.target_names}")
        if env.action_manager.total_action_dim != MICROBAN_TELEOP_ACTION_WIDTH:
            raise AssertionError(
                f"Unexpected action dimension: {env.action_manager.total_action_dim}"
            )

        soft_limits = env.scene["robot"].data.soft_joint_pos_limits[
            0, action.target_ids
        ]
        if not torch.allclose(action._clip[0], soft_limits, atol=1.0e-6, rtol=0.0):
            raise AssertionError("Action clips do not match Microban soft joint limits")

        forbidden = {"base_lin_vel", "root_pos", "root_position", "height_scan"}
        validate_microban_teleop_observation_contract(env)
        actor_term_names = set(env.observation_manager.active_terms["actor"])
        leaked_terms = actor_term_names & forbidden
        if leaked_terms:
            raise AssertionError(f"Non-deployable actor observations: {leaked_terms}")
        if "locomotion_prior" in actor_term_names:
            raise AssertionError("Privileged locomotion prior leaked into actor")
        if "locomotion_prior" not in env.observation_manager.active_terms["critic"]:
            raise AssertionError("Critic is missing privileged locomotion prior")
        prior = env.command_manager.get_term("locomotion_prior")
        if not isinstance(prior, LocomotionPriorCommand):
            raise TypeError("Locomotion prior command built the wrong runtime type")
        if prior.command.shape != (
            args.num_envs,
            MICROBAN_LOCOMOTION_PRIOR_COMMAND_WIDTH,
        ):
            raise AssertionError("Locomotion prior command width drifted")
        if prior.cfg.enabled or prior.command.any():
            raise AssertionError("Contract v9 locomotion prior must start disabled")
        for name in (
            "locomotion_prior_action_target",
            "locomotion_prior_joint_position",
        ):
            if env.reward_manager.get_term_cfg(name).weight != 0.0:
                raise AssertionError(f"Contract v9 reward {name!r} must stay zero")
        if "locomotion_prior_clip_finished" in env.termination_manager.active_terms:
            raise AssertionError("Contract v9 retains a retired prior termination")

        event_cfg = env.event_manager.get_term_cfg("hmd_neck_target_motion")
        motion = event_cfg.func
        if not isinstance(motion, HmdNeckTargetMotion):
            raise TypeError("HMD target event did not build its stateful term")
        if tuple(motion.joint_names) != ("head", "neck_roll", "neck_pitch"):
            raise AssertionError(f"Unexpected HMD joint order: {motion.joint_names}")
        if motion.neutral_probability != (
            MICROBAN_TELEOP_INITIAL_HMD_NEUTRAL_PROBABILITY
        ) or event_cfg.params["neutral_probability"] != (
            MICROBAN_TELEOP_INITIAL_HMD_NEUTRAL_PROBABILITY
        ):
            raise AssertionError("Live/config HMD probability differs at update zero")

        actions = torch.zeros(
            (args.num_envs, env.action_manager.total_action_dim),
            device=args.device,
        )

        # Resetting with the same seed must reproduce the event's first random
        # waypoint.  The MuJoCo Warp trajectory itself is not promised bit-exact;
        # this check is specifically for our torch-vectorized command generator.
        env.reset(seed=args.seed)
        foot_target = env.command_manager.get_term("foot_target")
        hand_target = env.command_manager.get_term("hand_target")
        if not isinstance(foot_target, ResetFixedFootTargetCommand):
            raise TypeError(
                f"Unexpected foot target type: {type(foot_target).__name__}"
            )
        if not isinstance(hand_target, ResetFixedHandTargetCommand):
            raise TypeError(
                f"Unexpected hand target type: {type(hand_target).__name__}"
            )
        if foot_target._reference_pending.any() or hand_target._reference_pending.any():
            raise AssertionError(
                "Post-reset keypoint reference capture is still pending"
            )
        if not torch.allclose(
            foot_target._default_foot_pos_b,
            foot_target.current_foot_pos_b(),
            atol=1.0e-6,
            rtol=0.0,
        ):
            raise AssertionError("Foot zero reference was not captured after reset FK")
        if not torch.allclose(
            hand_target._default_hand_pos_b,
            hand_target.current_hand_pos_b(),
            atol=1.0e-6,
            rtol=0.0,
        ):
            raise AssertionError("Hand zero reference was not captured after reset FK")

        # Update zero deliberately holds the HMD command at neutral while the
        # actor acquires locomotion.  Check this behavior before temporarily
        # forcing random waypoints for the deterministic-seed contract below.
        initial_target = motion.current_target.clone()
        observations, rewards, terminated, truncated, _ = env.step(actions)
        if not torch.equal(motion.goal_target, initial_target) or not torch.equal(
            motion.current_target, initial_target
        ):
            raise AssertionError("Initial HMD-neutral stage emitted a moving waypoint")

        motion.neutral_probability = 0.0
        event_cfg.params["neutral_probability"] = 0.0
        env.reset(seed=args.seed)
        random_initial_target = motion.current_target.clone()
        observations, rewards, terminated, truncated, _ = env.step(actions)
        first_goal = motion.goal_target.clone()
        first_target = motion.current_target.clone()
        if torch.allclose(first_target, random_initial_target):
            raise AssertionError("HMD target did not move on the first training step")

        env.reset(seed=args.seed)
        observations, rewards, terminated, truncated, _ = env.step(actions)
        if not torch.allclose(motion.goal_target, first_goal, atol=0.0, rtol=0.0):
            raise AssertionError("HMD waypoint sampling is not seed-reproducible")

        # Restore the actual v8 stage-zero contract before testing curriculum
        # boundaries; the 8,000 boundary must be what first enables moving HMD.
        motion.neutral_probability = MICROBAN_TELEOP_INITIAL_HMD_NEUTRAL_PROBABILITY
        event_cfg.params["neutral_probability"] = (
            MICROBAN_TELEOP_INITIAL_HMD_NEUTRAL_PROBABILITY
        )
        env.reset(seed=args.seed)
        observations, rewards, terminated, truncated, _ = env.step(actions)

        # Periodic command sampling must not move the episode's reference.  Do
        # this after the seeded HMD comparison because command resampling also
        # consumes the process torch RNG.  Sentinels catch the shared command
        # implementation's former behavior even when the robot has barely moved.
        saved_foot_reference = foot_target._default_foot_pos_b.clone()
        saved_hand_reference = hand_target._default_hand_pos_b.clone()
        foot_sentinel = saved_foot_reference + 0.123
        hand_sentinel = saved_hand_reference - 0.123
        foot_target._default_foot_pos_b.copy_(foot_sentinel)
        hand_target._default_hand_pos_b.copy_(hand_sentinel)
        foot_target._resample(env_ids)
        hand_target._resample(env_ids)
        if not torch.equal(foot_target._default_foot_pos_b, foot_sentinel):
            raise AssertionError("Foot command resampling moved its reset reference")
        if not torch.equal(hand_target._default_hand_pos_b, hand_sentinel):
            raise AssertionError("Hand command resampling moved its reset reference")
        foot_target._default_foot_pos_b.copy_(saved_foot_reference)
        hand_target._default_hand_pos_b.copy_(saved_hand_reference)

        max_step = motion.slew_rates_rad_s * env.step_dt
        previous_target = motion.current_target.clone()
        for _ in range(max(0, args.steps - 1)):
            observations, rewards, terminated, truncated, _ = env.step(actions)
            target_delta = torch.abs(motion.current_target - previous_target)
            # Auto-reset legitimately snaps an environment's command back to its
            # neutral reset pose.  The slew contract applies within an episode.
            continuing = ~(terminated | truncated)
            if torch.any(target_delta[continuing] > max_step + 1.0e-6):
                raise AssertionError("HMD target exceeded its per-step slew limit")
            previous_target = motion.current_target.clone()
            if not torch.isfinite(observations["actor"]).all():
                raise AssertionError("Actor observation contains NaN/Inf")
            if not torch.isfinite(rewards).all():
                raise AssertionError("Reward contains NaN/Inf")

        if torch.any(motion.current_target < motion.position_lower - 1.0e-6):
            raise AssertionError("HMD target fell below its effective soft limit")
        if torch.any(motion.current_target > motion.position_upper + 1.0e-6):
            raise AssertionError("HMD target exceeded its effective soft limit")
        target_buffer = env.scene["robot"].data.joint_pos_target[:, motion.joint_ids]
        if not torch.allclose(target_buffer, motion.current_target):
            raise AssertionError("HMD target was not written to the neck actuators")

        expected_actor_shape = (
            args.num_envs,
            MICROBAN_TELEOP_OBSERVATION_WIDTH,
        )
        if observations["actor"].shape != expected_actor_shape:
            raise AssertionError(
                f"Unexpected actor shape: {tuple(observations['actor'].shape)}"
            )

        metadata = get_microban_teleop_metadata(env, run_path="smoke")
        final_deployment_metadata = get_microban_teleop_metadata(
            env,
            run_path="smoke",
            canonical_final_stage=True,
        )
        expected_observation_names = [
            name for name, _ in MICROBAN_TELEOP_OBSERVATION_SCHEMA
        ]
        if metadata["observation_names"] != expected_observation_names:
            raise AssertionError(
                "Metadata observation order differs from the actor contract: "
                f"{metadata['observation_names']}"
            )
        if metadata["observation_width"] != MICROBAN_TELEOP_OBSERVATION_WIDTH:
            raise AssertionError(
                "Unexpected metadata observation width: "
                f"{metadata['observation_width']}"
            )
        if list(json.loads(metadata["observation_schema_json"]).items()) != list(
            MICROBAN_TELEOP_OBSERVATION_SCHEMA
        ):
            raise AssertionError(
                "Metadata observation schema differs from actor contract"
            )
        if metadata["base_ang_vel_frame"] != "robot_body_xyz":
            raise AssertionError(
                f"Unexpected gyro frame: {metadata['base_ang_vel_frame']}"
            )
        if metadata["base_ang_vel_units"] != "rad_s":
            raise AssertionError(
                f"Unexpected gyro units: {metadata['base_ang_vel_units']}"
            )
        if metadata["locomotion_command_units"] != ["m_s", "m_s", "rad_s"]:
            raise AssertionError(
                "Unexpected locomotion command units: "
                f"{metadata['locomotion_command_units']}"
            )
        expected_target_frame = "robot_trunk_xyz_forward_left_up"
        for target_name in ("foot_target", "hand_target"):
            if metadata[f"{target_name}_frame"] != expected_target_frame:
                raise AssertionError(
                    f"Unexpected {target_name} frame: "
                    f"{metadata[f'{target_name}_frame']}"
                )
            for bound in ("lower", "upper"):
                if len(metadata[f"{target_name}_{bound}"]) != 6:
                    raise AssertionError(
                        f"{target_name}_{bound} is not a flattened two-target XYZ bound"
                    )
        if metadata["foot_target_lower"] != [-0.03, -0.03, 0.0] * 2:
            raise AssertionError(
                f"Unexpected foot target lower bounds: {metadata['foot_target_lower']}"
            )
        if metadata["foot_target_upper"] != [0.03, 0.03, 0.05] * 2:
            raise AssertionError(
                f"Unexpected foot target upper bounds: {metadata['foot_target_upper']}"
            )
        if metadata["simultaneous_both_feet_target_upper"] != (
            [0.01, 0.01, 0.012] * 2
        ):
            raise AssertionError(
                "Diagnostic metadata did not retain play simultaneous-foot support"
            )
        if final_deployment_metadata["simultaneous_both_feet_target_upper"] != (
            [0.01, 0.01, MICROBAN_TELEOP_FINAL_BOTH_FEET_LIFT_UPPER_M] * 2
        ):
            raise AssertionError(
                "Final deployment metadata did not materialize final-stage "
                "simultaneous-foot support"
            )
        if metadata["hand_target_lower"] != [-0.08, -0.08, -0.08] * 2:
            raise AssertionError(
                f"Unexpected hand target lower bounds: {metadata['hand_target_lower']}"
            )
        if metadata["hand_target_upper"] != [0.08, 0.08, 0.08] * 2:
            raise AssertionError(
                f"Unexpected hand target upper bounds: {metadata['hand_target_upper']}"
            )
        for target_name in ("foot_target", "hand_target"):
            semantics = metadata[f"{target_name}_semantics"]
            if "episode_reset_reference" not in semantics:
                raise AssertionError(
                    f"{target_name} metadata does not declare its reset reference"
                )
            if "resampling_does_not_move_reference" not in semantics:
                raise AssertionError(
                    f"{target_name} metadata does not freeze periodic resampling"
                )
        for key in (
            "action_joint_names",
            "joint_stiffness",
            "joint_damping",
            "default_joint_pos",
            "soft_joint_pos_lower",
            "soft_joint_pos_upper",
            "action_scale",
        ):
            if len(metadata[key]) != 18:
                raise AssertionError(f"Metadata field {key!r} is not action-aligned")
        if metadata["actor_target_guard_margin_ratio"] != (
            MICROBAN_TELEOP_ACTOR_LIMIT_MARGIN_RATIO
        ):
            raise AssertionError("Metadata actor target guard ratio drifted")
        if metadata["actor_default_interior_epsilon_rad"] != (
            MICROBAN_TELEOP_ACTOR_DEFAULT_EPSILON_RAD
        ):
            raise AssertionError("Metadata actor default epsilon drifted")
        if metadata["action_distribution_semantics"] != (
            "diagonal_normal_ppo_latent_stored_exactly_then_per_joint_"
            "asymmetric_zero_anchored_arctan_environment_transform_with_"
            "operational_envelope_v1"
        ):
            raise AssertionError("Metadata bounded action semantics drifted")
        expected_latent_metadata = {
            "actor_latent_operational_scale_multiplier": (
                MICROBAN_TELEOP_ACTOR_LATENT_SCALE_MULTIPLIER
            ),
            "actor_latent_operational_abs_max": MICROBAN_TELEOP_ACTOR_LATENT_ABS_MAX,
            "actor_latent_mean_fraction": MICROBAN_TELEOP_ACTOR_LATENT_MEAN_FRACTION,
            "actor_latent_std_min_abs_max": MICROBAN_TELEOP_ACTOR_STD_MIN_ABS_MAX,
            "actor_latent_std_min_envelope_divisor": (
                MICROBAN_TELEOP_ACTOR_STD_MIN_ENVELOPE_DIVISOR
            ),
            "actor_latent_std_abs_max": MICROBAN_TELEOP_ACTOR_STD_ABS_MAX,
            "actor_latent_std_envelope_divisor": (
                MICROBAN_TELEOP_ACTOR_STD_ENVELOPE_DIVISOR
            ),
        }
        for key, expected in expected_latent_metadata.items():
            if metadata[key] != expected or not math.isfinite(metadata[key]):
                raise AssertionError(f"Metadata field {key!r} drifted")
        exact_bounds = {
            name: json.loads(metadata[f"{name}_json"])
            for name in (
                "actor_raw_action_lower",
                "actor_raw_action_upper",
                "raw_action_soft_lower",
                "raw_action_soft_upper",
            )
        }
        if any(len(values) != 18 for values in exact_bounds.values()):
            raise AssertionError("Exact JSON action bounds are not action-aligned")
        expected_actor_lower, expected_actor_upper = (
            microban_teleop_action_delta_bounds()
        )
        for name, expected in (
            ("actor_raw_action_lower", expected_actor_lower),
            ("actor_raw_action_upper", expected_actor_upper),
        ):
            if not torch.equal(
                torch.tensor(exact_bounds[name], dtype=torch.float32),
                torch.tensor(expected, dtype=torch.float32),
            ):
                raise AssertionError(
                    f"Metadata {name!r} differs from bounded actor contract"
                )
        for actor_lower, actor_upper, soft_lower, soft_upper in zip(
            exact_bounds["actor_raw_action_lower"],
            exact_bounds["actor_raw_action_upper"],
            exact_bounds["raw_action_soft_lower"],
            exact_bounds["raw_action_soft_upper"],
            strict=True,
        ):
            if not soft_lower < actor_lower < 0.0 < actor_upper < soft_upper:
                raise AssertionError("Actor bounds are not guarded inside soft bounds")
        if len(metadata["observation_default_joint_pos"]) != 21:
            raise AssertionError(
                "Metadata observation_default_joint_pos is not observation-aligned"
            )
        if not math.isclose(metadata["control_hz"], 50.0, abs_tol=1.0e-9):
            raise AssertionError(
                f"Unexpected metadata control rate: {metadata['control_hz']}"
            )

        # Must match microban/src/constants.py:NEUTRAL_POSE.  A mismatch here
        # creates a target jump when the left trigger is released and re-enabled.
        defaults = dict(
            zip(metadata["action_joint_names"], metadata["default_joint_pos"])
        )
        expected_shoulder_pitch = 0.0
        for side in ("left", "right"):
            name = f"{side}_shoulder_pitch"
            if not math.isclose(
                defaults[name], expected_shoulder_pitch, abs_tol=1.0e-6
            ):
                raise AssertionError(
                    f"{name} default {defaults[name]} does not match the current "
                    f"runtime constant {expected_shoulder_pitch}"
                )

        # Walk every v9 boundary at the exact global step. This checks both that
        # no stage is applied one step early and that resuming on a boundary
        # materializes the complete command/reward state in one compute.
        curriculum_cfg = env.curriculum_manager.get_term_cfg("staged_curriculum")
        curriculum = curriculum_cfg.func
        if not isinstance(curriculum, ResumeSafeStepBasedStagedCurriculum):
            raise TypeError(f"Unexpected curriculum type: {type(curriculum).__name__}")
        stage_boundaries = (
            500,
            1500,
            3000,
            4500,
            6000,
            8000,
            12000,
            14000,
            16000,
            18000,
        )
        for expected_stage, boundary in enumerate(stage_boundaries):
            env.common_step_counter = boundary * 24 - 1
            env.curriculum_manager.compute()
            if curriculum.current_stage != expected_stage:
                raise AssertionError(
                    f"Curriculum stage {expected_stage + 1} applied before "
                    f"its v9 boundary {boundary}"
                )
            if boundary == 8000:
                if motion.neutral_probability != (
                    MICROBAN_TELEOP_INITIAL_HMD_NEUTRAL_PROBABILITY
                ) or event_cfg.params["neutral_probability"] != (
                    MICROBAN_TELEOP_INITIAL_HMD_NEUTRAL_PROBABILITY
                ):
                    raise AssertionError("Moving HMD was enabled before update 8000")
                if env.reward_manager.get_term_cfg("no_stepping").weight != 0.0:
                    raise AssertionError("No-step guard was enabled before update 8000")

            env.common_step_counter = boundary * 24
            env.curriculum_manager.compute()
            if curriculum.current_stage != expected_stage + 1:
                raise AssertionError(
                    f"V9 boundary {boundary} materialized stage "
                    f"{curriculum.current_stage}, expected {expected_stage + 1}"
                )

            twist_cfg = env.command_manager.get_term_cfg("twist")
            linear_reward = env.reward_manager.get_term_cfg("track_linear_velocity")
            angular_reward = env.reward_manager.get_term_cfg("track_angular_velocity")
            if boundary == 500:
                if (
                    twist_cfg.signed_axis_ranges
                    != MICROBAN_TELEOP_INITIAL_SIGNED_AXIS_RANGES
                    or twist_cfg.signed_axis_probabilities
                    != MICROBAN_TELEOP_ISOLATED_AXIS_PROBABILITIES
                    or prior.cfg.enabled
                    or prior.command.any()
                ):
                    raise AssertionError("Locomotion prior was not disabled exactly")
            elif boundary == 1500:
                _assert_velocity_envelope(
                    twist_cfg, MICROBAN_TELEOP_INTERMEDIATE_VELOCITY_ENVELOPE
                )
                if (
                    twist_cfg.signed_axis_ranges
                    != MICROBAN_TELEOP_INTERMEDIATE_SIGNED_AXIS_RANGES
                    or twist_cfg.signed_axis_probabilities
                    != MICROBAN_TELEOP_ISOLATED_AXIS_PROBABILITIES
                    or linear_reward.params["std"]
                    != MICROBAN_TELEOP_INTERMEDIATE_LINEAR_TRACKING_STD_M_S
                    or angular_reward.params["std"]
                    != MICROBAN_TELEOP_INTERMEDIATE_ANGULAR_TRACKING_STD_RAD_S
                ):
                    raise AssertionError("Intermediate isolated-axis stage drifted")
            elif boundary == 3000:
                _assert_velocity_envelope(
                    twist_cfg,
                    MICROBAN_TELEOP_FINAL_TRANSLATION_VELOCITY_ENVELOPE,
                )
                if (
                    twist_cfg.signed_axis_ranges
                    != MICROBAN_TELEOP_FINAL_TRANSLATION_SIGNED_AXIS_RANGES
                    or twist_cfg.signed_axis_probabilities
                    != MICROBAN_TELEOP_ISOLATED_AXIS_PROBABILITIES
                    or linear_reward.params["std"]
                    != MICROBAN_TELEOP_LINEAR_TRACKING_STD_M_S
                    or angular_reward.params["std"]
                    != MICROBAN_TELEOP_ANGULAR_TRACKING_STD_RAD_S
                ):
                    raise AssertionError(
                        "Final translation isolated-axis stage drifted"
                    )
            elif boundary == 4500:
                _assert_velocity_envelope(
                    twist_cfg, MICROBAN_TELEOP_FINAL_VELOCITY_ENVELOPE
                )
                if (
                    twist_cfg.signed_axis_ranges
                    != MICROBAN_TELEOP_FINAL_SIGNED_AXIS_RANGES
                    or twist_cfg.signed_axis_probabilities
                    != MICROBAN_TELEOP_ISOLATED_AXIS_PROBABILITIES
                ):
                    raise AssertionError("Final pure-yaw isolated-axis stage drifted")
            elif boundary == 6000 and (
                twist_cfg.signed_axis_probabilities
                != MICROBAN_TELEOP_MIXED_AXIS_PROBABILITIES
            ):
                raise AssertionError("Mixed-command replay was not enabled")

            hand_reward = env.reward_manager.get_term_cfg("hand_target_tracking")
            foot_reward = env.reward_manager.get_term_cfg("foot_target_tracking")
            hand_cfg = env.command_manager.get_term_cfg("hand_target")
            foot_cfg = env.command_manager.get_term_cfg("foot_target")
            if boundary <= 6000:
                if hand_reward.weight != 0.0 or foot_reward.weight != (
                    MICROBAN_TELEOP_NEUTRAL_FOOT_TRACKING_WEIGHT
                ):
                    raise AssertionError(
                        "Initial neutral-foot reward drifted during locomotion acquisition"
                    )
                if foot_reward.params["velocity_fade_range"] != (0.0, 0.01):
                    raise AssertionError("Initial neutral-foot fade range drifted")
                if (
                    hand_cfg.rel_active != 0.0
                    or foot_cfg.rel_single_support_envs != 0.0
                    or foot_cfg.rel_both_feet_envs != 0.0
                ):
                    raise AssertionError(
                        "Non-neutral limb targets were enabled during locomotion acquisition"
                    )
            elif boundary == 8000:
                if (
                    foot_reward.weight != MICROBAN_TELEOP_NEUTRAL_FOOT_TRACKING_WEIGHT
                    or hand_reward.weight != 0.0
                    or hand_cfg.rel_active != 0.0
                    or foot_cfg.rel_single_support_envs != 0.0
                    or foot_cfg.rel_both_feet_envs != 0.0
                ):
                    raise AssertionError(
                        "Neutral-foot anchor did not remain target-inactive"
                    )
                if motion.neutral_probability != (
                    MICROBAN_TELEOP_MOVING_HMD_NEUTRAL_PROBABILITY
                ) or event_cfg.params["neutral_probability"] != (
                    MICROBAN_TELEOP_MOVING_HMD_NEUTRAL_PROBABILITY
                ):
                    raise AssertionError(
                        "Update 8000 did not update both live/config HMD probability"
                    )
                if env.reward_manager.get_term_cfg("no_stepping").weight != -1.0:
                    raise AssertionError(
                        "Update 8000 did not enable the stationary no-step guard"
                    )
            elif boundary == 12000:
                if hand_reward.weight != 1.0 or hand_cfg.rel_active != 0.7:
                    raise AssertionError("Broad hand tracking stage drifted")
            elif boundary == 14000:
                if hand_reward.weight != 2.0 or (
                    hand_reward.params["std"]
                    != MICROBAN_TELEOP_HAND_TRACKING_FINAL_STD_M
                ):
                    raise AssertionError("Tight hand tracking stage drifted")
            elif boundary == 16000:
                if (
                    foot_reward.weight != 2.0
                    or foot_reward.params["std"] != 0.05
                    or foot_reward.params["velocity_fade_range"] != (0.0, 0.15)
                    or foot_cfg.rel_single_support_envs != 0.3
                    or foot_cfg.rel_both_feet_envs != 0.05
                ):
                    raise AssertionError("Broad stationary-foot stage drifted")

        twist_cfg = env.command_manager.get_term_cfg("twist")
        _assert_velocity_envelope(twist_cfg, MICROBAN_TELEOP_FINAL_VELOCITY_ENVELOPE)
        if twist_cfg.signed_axis_ranges != MICROBAN_TELEOP_FINAL_SIGNED_AXIS_RANGES:
            raise AssertionError("Final signed-axis ranges were not retained")

        hand_reward = env.reward_manager.get_term_cfg("hand_target_tracking")
        foot_reward = env.reward_manager.get_term_cfg("foot_target_tracking")
        foot_cfg = env.command_manager.get_term_cfg("foot_target")
        hand_cfg = env.command_manager.get_term_cfg("hand_target")
        if hand_reward.weight != 2.0 or (
            hand_reward.params["std"] != MICROBAN_TELEOP_HAND_TRACKING_FINAL_STD_M
        ):
            raise AssertionError("Final hand tracking stage was not materialized")
        if hand_cfg.rel_active != 0.7:
            raise AssertionError("Final hand activation was not materialized")
        if foot_reward.weight != 3.0 or (
            foot_reward.params["std"] != MICROBAN_TELEOP_FOOT_TRACKING_FINAL_STD_M
        ):
            raise AssertionError("Final foot tracking stage was not materialized")
        if foot_reward.params["velocity_fade_range"] != (0.0, 0.15):
            raise AssertionError("Final stationary-foot fade range drifted")
        if foot_cfg.rel_single_support_envs != 0.3:
            raise AssertionError("Final single-foot activation was not materialized")
        if foot_cfg.rel_both_feet_envs != 0.1:
            raise AssertionError("Final both-foot activation was not materialized")
        if foot_cfg.lift_height_range != (
            MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M,
            0.05,
        ):
            raise AssertionError("Single-foot floor-band support drifted")
        if foot_cfg.both_feet_lift_height_range != (
            MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M,
            MICROBAN_TELEOP_FINAL_BOTH_FEET_LIFT_UPPER_M,
        ):
            raise AssertionError("Final both-foot floor-band support drifted")
        if (
            env.reward_manager.get_term_cfg("no_stepping").weight != -1.0
            or MICROBAN_TELEOP_NEUTRAL_FOOT_TRACKING_WEIGHT <= 0.0
        ):
            raise AssertionError("Exact-stationary neutral-foot anchor drifted")

        env.curriculum_manager.compute()
        if curriculum.current_stage != len(stage_boundaries):
            raise AssertionError("Final curriculum stages were applied more than once")

        report = {
            "status": "pass",
            "device": args.device,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "seed": args.seed,
            "actor_observation_shape": list(observations["actor"].shape),
            "action_dim": env.action_manager.total_action_dim,
            "nconmax": cfg.sim.nconmax,
            "njmax": cfg.sim.njmax,
            "action_joint_names": action.target_names,
            "hmd_joint_names": list(motion.joint_names),
            "hmd_effective_limits_rad": [
                motion.position_lower[0].tolist(),
                motion.position_upper[0].tolist(),
            ],
            "hmd_slew_rates_rad_s": motion.slew_rates_rad_s[0].tolist(),
            "twist_command_type": type(twist).__name__,
            "rotation_env_ang_vel_range": list(twist.cfg.rotation_env_ang_vel_range),
            "signed_axis_modes": list(twist.cfg.signed_axis_probabilities),
            "v9_curriculum_boundaries": list(stage_boundaries),
            "keypoint_reference": "episode_reset_fixed",
            "resume_curriculum_stage": curriculum.current_stage,
            "terminated": int(terminated.sum().item()),
            "truncated": int(truncated.sum().item()),
        }
        print(json.dumps(report, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
