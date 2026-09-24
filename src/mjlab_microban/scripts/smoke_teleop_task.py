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

import torch
from mjlab.envs import ManagerBasedRlEnv

from mjlab_microban.tasks.mdp import (
    UniformVelocityCommandWithRotation,
    UniformVelocityCommandWithRotationCfg,
)
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_TELEOP_ACTION_JOINT_NAMES,
    MICROBAN_TELEOP_ACTION_WIDTH,
    MICROBAN_TELEOP_OBSERVATION_SCHEMA,
    MICROBAN_TELEOP_OBSERVATION_WIDTH,
    get_microban_teleop_metadata,
    validate_microban_teleop_observation_contract,
)
from mjlab_microban.tasks.microban_teleop_env_cfg import (
    make_microban_teleop_env_cfg,
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
    command_cfg: object, *, expected_rel_rotation_envs: float
) -> None:
    """Check the rotation-only command contract survives config construction."""

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
    if command_cfg.rotation_env_ang_vel_range != (-1.5, 1.5):
        raise AssertionError(
            "Unexpected rotation-only angular velocity range: "
            f"{command_cfg.rotation_env_ang_vel_range}"
        )


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
        play_cfg.commands["twist"], expected_rel_rotation_envs=0.0
    )

    cfg = make_microban_teleop_env_cfg(play=False)
    _assert_rotation_command_cfg(cfg.commands["twist"], expected_rel_rotation_envs=0.1)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    # Prevent the deliberately occasional neutral dwell from making the bounded
    # smoke check probabilistic; normal training keeps the configured 20% dwell.
    cfg.events["hmd_neck_target_motion"].params["neutral_probability"] = 0.0
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)

    try:
        twist = env.command_manager.get_term("twist")
        if not isinstance(twist, UniformVelocityCommandWithRotation):
            raise TypeError(
                f"twist built the wrong runtime command type: {type(twist).__name__}"
            )
        _assert_rotation_command_cfg(
            env.command_manager.get_term_cfg("twist"),
            expected_rel_rotation_envs=0.1,
        )

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

        event_cfg = env.event_manager.get_term_cfg("hmd_neck_target_motion")
        motion = event_cfg.func
        if not isinstance(motion, HmdNeckTargetMotion):
            raise TypeError("HMD target event did not build its stateful term")
        if tuple(motion.joint_names) != ("head", "neck_roll", "neck_pitch"):
            raise AssertionError(f"Unexpected HMD joint order: {motion.joint_names}")

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

        initial_target = motion.current_target.clone()
        observations, rewards, terminated, truncated, _ = env.step(actions)
        first_goal = motion.goal_target.clone()
        first_target = motion.current_target.clone()
        if torch.allclose(first_target, initial_target):
            raise AssertionError("HMD target did not move on the first training step")

        env.reset(seed=args.seed)
        observations, rewards, terminated, truncated, _ = env.step(actions)
        if not torch.allclose(motion.goal_target, first_goal, atol=0.0, rtol=0.0):
            raise AssertionError("HMD waypoint sampling is not seed-reproducible")

        # Periodic command sampling must not move the episode's reference.  Do
        # this after the seeded HMD comparison because command resampling also
        # consumes the process torch RNG.  Sentinels catch the shared command
        # implementation's former behavior even when the robot has barely moved.
        env_ids = torch.arange(args.num_envs, device=args.device)
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
        expected_shoulder_pitch = math.radians(10.0)
        for side in ("left", "right"):
            name = f"{side}_shoulder_pitch"
            if not math.isclose(
                defaults[name], expected_shoulder_pitch, abs_tol=1.0e-6
            ):
                raise AssertionError(
                    f"{name} default {defaults[name]} does not match the current "
                    f"runtime constant {expected_shoulder_pitch}"
                )

        # Simulate manager reconstruction followed by a checkpoint's restored
        # step count.  One compute must materialize every due stage before a
        # resumed rollout, rather than waiting for three future episode resets.
        curriculum_cfg = env.curriculum_manager.get_term_cfg("staged_curriculum")
        curriculum = curriculum_cfg.func
        if not isinstance(curriculum, ResumeSafeStepBasedStagedCurriculum):
            raise TypeError(f"Unexpected curriculum type: {type(curriculum).__name__}")
        env.common_step_counter = 3000 * 24
        env.curriculum_manager.compute()
        if curriculum.current_stage != 3:
            raise AssertionError(
                "Resume curriculum materialized stage "
                f"{curriculum.current_stage}, expected 3"
            )
        if env.reward_manager.get_term_cfg("hand_target_tracking").weight != 1.0:
            raise AssertionError("Hand curriculum stage was not materialized")
        if env.reward_manager.get_term_cfg("foot_target_tracking").weight != 2.0:
            raise AssertionError("Foot curriculum stage was not materialized")
        if env.command_manager.get_term_cfg("hand_target").rel_active != 0.7:
            raise AssertionError("Hand activation curriculum was not materialized")
        if (
            env.command_manager.get_term_cfg("foot_target").rel_single_support_envs
            != 0.3
        ):
            raise AssertionError("Foot activation curriculum was not materialized")
        if env.command_manager.get_term_cfg("twist").ranges.lin_vel_x != (-0.5, 0.6):
            raise AssertionError("Intermediate velocity stage was not materialized")
        if env.command_manager.get_term_cfg("twist").ranges.ang_vel_z != (-1.0, 1.0):
            raise AssertionError("Intermediate yaw stage was not materialized")
        if env.command_manager.get_term_cfg("foot_target").rel_both_feet_envs != 0.1:
            raise AssertionError("Both-foot curriculum was not materialized")
        foot_cfg = env.command_manager.get_term_cfg("foot_target")
        if foot_cfg.lift_height_range != (
            MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M,
            0.05,
        ):
            raise AssertionError("Single-foot floor-band support drifted")
        if foot_cfg.both_feet_lift_height_range != (
            MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M,
            0.012,
        ):
            raise AssertionError("Initial both-foot floor-band support drifted")
        env.curriculum_manager.compute()
        if curriculum.current_stage != 3:
            raise AssertionError("Curriculum stages were applied more than once")

        env.common_step_counter = 6000 * 24
        env.curriculum_manager.compute()
        if curriculum.current_stage != 5:
            raise AssertionError(
                "Resume curriculum materialized stage "
                f"{curriculum.current_stage}, expected 5"
            )
        twist_cfg = env.command_manager.get_term_cfg("twist")
        if twist_cfg.ranges.lin_vel_x != (-0.5, 0.7):
            raise AssertionError("Final asymmetric velocity stage was not materialized")
        if twist_cfg.ranges.lin_vel_y != (-0.3, 0.3):
            raise AssertionError("Final lateral velocity stage was not materialized")
        if twist_cfg.ranges.ang_vel_z != (-1.5, 1.5):
            raise AssertionError("Final moving-yaw stage was not materialized")
        if twist_cfg.rotation_env_ang_vel_range != (-3.0, 3.0):
            raise AssertionError("Final pure-yaw stage was not materialized")
        if env.command_manager.get_term_cfg(
            "foot_target"
        ).both_feet_lift_height_range != (
            MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M,
            0.02,
        ):
            raise AssertionError("Final both-foot floor-band support drifted")
        env.curriculum_manager.compute()
        if curriculum.current_stage != 5:
            raise AssertionError("Final curriculum stages were applied more than once")

        report = {
            "status": "pass",
            "device": args.device,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "seed": args.seed,
            "actor_observation_shape": list(observations["actor"].shape),
            "action_dim": env.action_manager.total_action_dim,
            "action_joint_names": action.target_names,
            "hmd_joint_names": list(motion.joint_names),
            "hmd_effective_limits_rad": [
                motion.position_lower[0].tolist(),
                motion.position_upper[0].tolist(),
            ],
            "hmd_slew_rates_rad_s": motion.slew_rates_rad_s[0].tolist(),
            "twist_command_type": type(twist).__name__,
            "rotation_env_ang_vel_range": list(twist.cfg.rotation_env_ang_vel_range),
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
