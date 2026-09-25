"""Evaluate a simulation-only direct-IK arm overlay on the legacy walk actor.

The legacy actor keeps its original 63-value observation and its own previous
raw action.  Only the six arm values sent to the teleoperation environment are
replaced by reachable Microban arm joint targets.  This mirrors the proposed
live simulation fallback without changing the live entry point.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.utils.nan_guard import NanGuard
from mjlab.utils.torch import configure_torch_backends
from tensordict import TensorDict

from mjlab_microban.legacy_velocity_diagnostics import (
    DEFAULT_LEGACY_VELOCITY_CHECKPOINT,
    DEFAULT_LEGACY_VELOCITY_SHA256,
    TwistScenario,
    checkpoint_sha256,
    default_scenarios,
    publish_json_atomic,
)
from mjlab_microban.robot.microban_hand_fk import (
    MICROBAN_ARM_HOME_JOINT_RAD,
    MICROBAN_REACHABLE_HAND_EVALUATION_JOINTS_DEG,
    microban_hand_offsets_from_arm_joints,
)
from mjlab_microban.scripts.probe_legacy_actor_in_teleop_env import (
    ActorLayout,
    _actor_layout,
    _assemble_legacy_observation,
    _indices_by_name,
    _source_cfg,
    _targets_are_neutral,
    _teleop_cfg,
    _tensor_parameter,
    _termination_names,
    _zero_neutral_targets,
)
from mjlab_microban.tasks.microban_velocity_env_cfg import MicrobanVelocityRlCfg
from mjlab_microban.teleop_v12_safety import (
    ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD,
    COMMANDED_TARGET_SOFT_LIMIT_EXCESS_MAX_RAD,
)

ARM_NAMES = (
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_elbow",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_elbow",
)
HAND_SLEW_M_S = 0.25


def _arm_pose_from_left_deg(left_deg: tuple[float, float, float]) -> torch.Tensor:
    """Create the bilateral pose used by the reachable-target contract."""

    left = torch.deg2rad(torch.tensor(left_deg, dtype=torch.float32))
    right = torch.stack((left[0], -left[1], left[2]))
    return torch.stack((left, right))


def _profiles() -> dict[str, tuple[torch.Tensor, ...]]:
    named = {
        name: _arm_pose_from_left_deg(pose)
        for name, pose in MICROBAN_REACHABLE_HAND_EVALUATION_JOINTS_DEG
    }
    home = torch.tensor(MICROBAN_ARM_HOME_JOINT_RAD, dtype=torch.float32)
    corner_a = _arm_pose_from_left_deg((-25.0, 30.0, -50.0))
    corner_b = _arm_pose_from_left_deg((25.0, 30.0, -10.0))
    return {
        "home": (home,),
        "reachable_F": (named["F"],),
        "reachable_B": (named["B"],),
        "opposed_FB": (torch.stack((named["F"][0], named["B"][1])),),
        "corner_sweep": (corner_a, corner_b, home),
    }


def _slew_arm_pose(
    current: torch.Tensor,
    goal: torch.Tensor,
    *,
    step_dt_s: float,
) -> torch.Tensor:
    """Match NativeControlMapper's FK-manifold Cartesian slew algorithm."""

    maximum_distance = HAND_SLEW_M_S * step_dt_s
    current_hands = microban_hand_offsets_from_arm_joints(current)
    goal_hands = microban_hand_offsets_from_arm_joints(goal)
    selected = current.clone()
    for side in range(2):
        distance = float(
            torch.linalg.vector_norm(goal_hands[side] - current_hands[side])
        )
        if distance <= maximum_distance or distance == 0.0:
            selected[side] = goal[side]
            continue
        low = 0.0
        high = 1.0
        delta = goal[side] - current[side]
        for _ in range(32):
            fraction = (low + high) * 0.5
            candidate = current[side] + delta * fraction
            pair = current.clone()
            pair[side] = candidate
            candidate_hand = microban_hand_offsets_from_arm_joints(pair)[side]
            if (
                float(torch.linalg.vector_norm(candidate_hand - current_hands[side]))
                <= maximum_distance
            ):
                low = fraction
            else:
                high = fraction
        selected[side] = current[side] + delta * low
    return selected


def _profile_goal(
    profile: tuple[torch.Tensor, ...], step: int, steps: int
) -> torch.Tensor:
    if len(profile) == 1:
        return profile[0]
    segment = max(1, steps // len(profile))
    return profile[min(step // segment, len(profile) - 1)]


def _evaluate_case(
    *,
    env: ManagerBasedRlEnv,
    wrapped: RslRlVecEnvWrapper,
    policy: Any,
    teleop_layout: ActorLayout,
    legacy_layout: ActorLayout,
    joint_pos_indices: tuple[int, ...],
    joint_vel_indices: tuple[int, ...],
    action_indices: tuple[int, ...],
    scenario: TwistScenario,
    profile_name: str,
    profile: tuple[torch.Tensor, ...],
    seed: int,
    steps: int,
    settle_steps: int,
) -> dict[str, Any]:
    env.reset(seed=seed)
    command = env.command_manager.get_term("twist")
    expected_twist = torch.tensor(
        scenario.twist, dtype=command.vel_command_b.dtype, device=env.device
    ).unsqueeze(0)
    command.vel_command_b[:] = expected_twist
    _zero_neutral_targets(env)
    observations = wrapped.get_observations()

    robot = env.scene["robot"]
    action = env.action_manager.get_term("joint_pos")
    scale = _tensor_parameter(action.scale, action.raw_action)
    offset = _tensor_parameter(action.offset, action.raw_action)
    if not bool(torch.equal(scale, torch.ones_like(scale))):
        raise ValueError("Direct q-HOME overlay requires exact unit action scale")
    action_by_name = {name: index for index, name in enumerate(action.target_names)}
    arm_action_indices = torch.tensor(
        [action_by_name[name] for name in ARM_NAMES],
        dtype=torch.long,
        device=env.device,
    )
    arm_joint_ids, resolved_names = robot.find_joints(ARM_NAMES, preserve_order=True)
    if tuple(resolved_names) != ARM_NAMES:
        raise ValueError(f"Arm joint order mismatch: {resolved_names}")
    arm_joint_ids_tensor = torch.tensor(
        arm_joint_ids, dtype=torch.long, device=env.device
    )
    arm_home = torch.tensor(
        MICROBAN_ARM_HOME_JOINT_RAD,
        dtype=action.raw_action.dtype,
        device=env.device,
    )
    expected_home = offset[0].index_select(0, arm_action_indices).reshape(2, 3)
    if not bool(torch.equal(arm_home, expected_home)):
        raise ValueError("Direct overlay HOME differs from action offset")

    all_soft_limits = robot.data.soft_joint_pos_limits
    arm_soft_limits = (
        all_soft_limits[0].index_select(0, arm_joint_ids_tensor).reshape(2, 3, 2)
    )
    current_target = arm_home.clone()
    actor_previous_action = torch.zeros_like(action.raw_action)
    maximum_actual_soft_limit_violation = 0.0
    maximum_arm_target_soft_limit_excess = 0.0
    maximum_arm_joint_error = 0.0
    hand_errors: list[torch.Tensor] = []
    measured_velocity: list[torch.Tensor] = []
    minimum_root_height = float(robot.data.root_link_pos_w[0, 2].item())
    fall_height = float(
        env.termination_manager.get_term_cfg("fell_over").params["minimum_height"]
    )
    fell = False
    fall_step: int | None = None
    nonfinite: dict[str, Any] | None = None
    termination_names: list[str] = []
    executed_steps = 0
    neutral_target_steps = 0

    for step in range(steps):
        actor_obs = observations["actor"]
        policy_obs = _assemble_legacy_observation(
            actor_obs,
            teleop=teleop_layout,
            legacy=legacy_layout,
            joint_pos_indices=joint_pos_indices,
            joint_vel_indices=joint_vel_indices,
            action_indices=action_indices,
            twist=expected_twist,
        )
        # Preserve the audited actor's internal recurrence.  The environment
        # sees the arm overlay, while the actor sees its own prior raw output.
        policy_obs[:, legacy_layout.terms["actions"]] = actor_previous_action
        with torch.inference_mode():
            actor_action = policy(TensorDict({"actor": policy_obs}, batch_size=[1]))
        if tuple(actor_action.shape) != (1, 18) or not bool(
            torch.isfinite(actor_action).all().item()
        ):
            nonfinite = {"phase": "actor", "step": step}
            break
        actor_previous_action.copy_(actor_action)

        goal = _profile_goal(profile, step, steps).to(
            dtype=current_target.dtype, device=current_target.device
        )
        current_target = _slew_arm_pose(
            current_target, goal, step_dt_s=float(env.step_dt)
        )
        overlay_action = actor_action.clone()
        overlay_action[0, arm_action_indices] = (current_target - arm_home).reshape(-1)
        target_excess = torch.maximum(
            torch.clamp(arm_soft_limits[..., 0] - current_target, min=0.0),
            torch.clamp(current_target - arm_soft_limits[..., 1], min=0.0),
        )
        maximum_arm_target_soft_limit_excess = max(
            maximum_arm_target_soft_limit_excess, float(target_excess.max().item())
        )

        observations, rewards, dones, _extras = wrapped.step(overlay_action)
        executed_steps += 1
        if _targets_are_neutral(env):
            neutral_target_steps += 1
        finite = all(
            bool(torch.isfinite(value).all().item())
            for value in (
                observations["actor"],
                rewards,
                robot.data.joint_pos,
                robot.data.joint_vel,
                robot.data.root_link_pose_w,
                robot.data.root_link_vel_w,
            )
        )
        if bool(NanGuard.detect_nans(env.sim.data)[0].item()) or not finite:
            nonfinite = {"phase": "physics", "step": step}
            break

        actual_violation = torch.maximum(
            torch.clamp(all_soft_limits[..., 0] - robot.data.joint_pos, min=0.0),
            torch.clamp(robot.data.joint_pos - all_soft_limits[..., 1], min=0.0),
        )
        maximum_actual_soft_limit_violation = max(
            maximum_actual_soft_limit_violation, float(actual_violation.max().item())
        )
        actual_arm = (
            robot.data.joint_pos[0].index_select(0, arm_joint_ids_tensor).reshape(2, 3)
        )
        maximum_arm_joint_error = max(
            maximum_arm_joint_error,
            float(torch.abs(actual_arm - current_target).max().item()),
        )
        desired_hands = microban_hand_offsets_from_arm_joints(current_target)
        actual_hands = microban_hand_offsets_from_arm_joints(actual_arm)
        hand_errors.append(
            torch.linalg.vector_norm(actual_hands - desired_hands, dim=-1).cpu()
        )

        root_height = float(robot.data.root_link_pos_w[0, 2].item())
        minimum_root_height = min(minimum_root_height, root_height)
        if (
            bool(env.termination_manager.get_term("fell_over")[0].item())
            or root_height < fall_height
        ) and not fell:
            fell = True
            fall_step = executed_steps
        if step >= settle_steps:
            measured_velocity.append(
                torch.stack(
                    (
                        robot.data.root_link_lin_vel_b[0, 0],
                        robot.data.root_link_lin_vel_b[0, 1],
                        robot.data.root_link_ang_vel_b[0, 2],
                    )
                )
                .detach()
                .cpu()
            )
        if bool(dones[0].item()):
            termination_names = _termination_names(env)
            break

    errors = torch.stack(hand_errors) if hand_errors else torch.empty((0, 2))
    velocity = (
        torch.stack(measured_velocity).mean(dim=0)
        if measured_velocity
        else torch.full((3,), math.nan)
    )
    return {
        "scenario": scenario.name,
        "arm_profile": profile_name,
        "command_twist": list(scenario.twist),
        "executed_steps": executed_steps,
        "completed": executed_steps == steps
        and not fell
        and nonfinite is None
        and not termination_names,
        "fell": fell,
        "fall_step": fall_step,
        "nonfinite": nonfinite,
        "termination_names": termination_names,
        "minimum_root_height_m": minimum_root_height,
        "maximum_actual_soft_limit_violation_rad": maximum_actual_soft_limit_violation,
        "maximum_arm_target_soft_limit_excess_rad": maximum_arm_target_soft_limit_excess,
        "maximum_arm_joint_tracking_error_rad": maximum_arm_joint_error,
        "hand_tracking_rms_m": (
            torch.sqrt(torch.mean(errors.square(), dim=0)).tolist()
            if errors.numel()
            else [None, None]
        ),
        "hand_tracking_p95_m": (
            torch.quantile(errors, 0.95, dim=0).tolist()
            if errors.numel()
            else [None, None]
        ),
        "measured_twist_mean": velocity.tolist(),
        "neutral_foot_hand_target_verified_steps": neutral_target_steps,
    }


def run_evaluation(
    *,
    checkpoint: Path,
    expected_sha256: str,
    device: str,
    seed: int,
    steps: int,
    settle_steps: int,
) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve()
    digest = checkpoint_sha256(checkpoint)
    if digest != expected_sha256:
        raise ValueError(f"Legacy checkpoint SHA-256 mismatch: {digest}")
    configure_torch_backends(allow_tf32=False, deterministic=True)
    torch.use_deterministic_algorithms(True, warn_only=True)

    source_env = ManagerBasedRlEnv(
        cfg=_source_cfg(seed=seed, steps=steps), device=device
    )
    source_wrapped = RslRlVecEnvWrapper(
        source_env, clip_actions=MicrobanVelocityRlCfg.clip_actions
    )
    try:
        legacy_layout = _actor_layout(source_env)
        runner = VelocityOnPolicyRunner(
            source_wrapped, asdict(MicrobanVelocityRlCfg), device=device
        )
        runner.load(
            str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
        )
        policy = runner.get_inference_policy(device=device)
    finally:
        source_wrapped.close()

    teleop_env = ManagerBasedRlEnv(
        cfg=_teleop_cfg(seed=seed, steps=steps), device=device
    )
    teleop_wrapped = RslRlVecEnvWrapper(teleop_env, clip_actions=None)
    try:
        teleop_layout = _actor_layout(teleop_env)
        joint_pos_indices = _indices_by_name(
            legacy_layout.joint_pos_names,
            teleop_layout.joint_pos_names,
            label="joint-position",
        )
        joint_vel_indices = _indices_by_name(
            legacy_layout.joint_vel_names,
            teleop_layout.joint_vel_names,
            label="joint-velocity",
        )
        action_indices = _indices_by_name(
            legacy_layout.action_names,
            teleop_layout.action_names,
            label="previous-action",
        )
        results = []
        for scenario in default_scenarios():
            for profile_name, profile in _profiles().items():
                print(f"[INFO] {scenario.name} / {profile_name}", flush=True)
                results.append(
                    _evaluate_case(
                        env=teleop_env,
                        wrapped=teleop_wrapped,
                        policy=policy,
                        teleop_layout=teleop_layout,
                        legacy_layout=legacy_layout,
                        joint_pos_indices=joint_pos_indices,
                        joint_vel_indices=joint_vel_indices,
                        action_indices=action_indices,
                        scenario=scenario,
                        profile_name=profile_name,
                        profile=profile,
                        seed=seed,
                        steps=steps,
                        settle_steps=settle_steps,
                    )
                )
        action = teleop_env.action_manager.get_term("joint_pos")
        robot = teleop_env.scene["robot"]
        action_by_name = {name: index for index, name in enumerate(action.target_names)}
        joint_by_name = {name: index for index, name in enumerate(robot.joint_names)}
        arm_margins = {}
        home = torch.tensor(
            MICROBAN_ARM_HOME_JOINT_RAD, device=teleop_env.device
        ).reshape(-1)
        for index, name in enumerate(ARM_NAMES):
            limits = robot.data.soft_joint_pos_limits[0, joint_by_name[name]]
            arm_margins[name] = {
                "action_index": action_by_name[name],
                "home_rad": float(home[index].item()),
                "soft_lower_rad": float(limits[0].item()),
                "soft_upper_rad": float(limits[1].item()),
            }
    finally:
        teleop_wrapped.close()
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "evaluation": "legacy_walk_direct_ik_arm_overlay_v1",
        "checkpoint": {"path": str(checkpoint), "sha256": digest},
        "settings": {
            "device": device,
            "seed": seed,
            "steps": steps,
            "settle_steps": settle_steps,
            "hand_slew_m_s": HAND_SLEW_M_S,
            "actor_previous_action": "original_actor_output_before_overlay",
            "environment_arm_action": "commanded_ik_q_rad_minus_HOME",
            "foot_target": "exact_zero_inactive",
        },
        "arm_contract": arm_margins,
        "thresholds": {
            "actual_dynamic_soft_limit_overshoot_rad_max": (
                ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD
            ),
            "commanded_arm_target_soft_limit_excess_rad_max": (
                COMMANDED_TARGET_SOFT_LIMIT_EXCESS_MAX_RAD
            ),
        },
        "results": results,
        "summary": {
            "case_count": len(results),
            "completed_case_count": sum(bool(item["completed"]) for item in results),
            "fall_case_count": sum(bool(item["fell"]) for item in results),
            "nonfinite_case_count": sum(
                item["nonfinite"] is not None for item in results
            ),
            "actual_soft_limit_violation_case_count": sum(
                float(item["maximum_actual_soft_limit_violation_rad"])
                > ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD
                for item in results
            ),
            "maximum_actual_soft_limit_violation_rad": max(
                float(item["maximum_actual_soft_limit_violation_rad"])
                for item in results
            ),
            "maximum_arm_target_soft_limit_excess_rad": max(
                float(item["maximum_arm_target_soft_limit_excess_rad"])
                for item in results
            ),
            "maximum_arm_joint_tracking_error_rad": max(
                float(item["maximum_arm_joint_tracking_error_rad"]) for item in results
            ),
            "maximum_hand_tracking_rms_m": max(
                max(float(value) for value in item["hand_tracking_rms_m"])
                for item in results
            ),
            "maximum_hand_tracking_p95_m": max(
                max(float(value) for value in item["hand_tracking_p95_m"])
                for item in results
            ),
            "neutral_foot_hand_target_all_steps": all(
                item["neutral_foot_hand_target_verified_steps"]
                == item["executed_steps"]
                for item in results
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_LEGACY_VELOCITY_CHECKPOINT
    )
    parser.add_argument("--expected-sha256", default=DEFAULT_LEGACY_VELOCITY_SHA256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--settle-steps", type=int, default=50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = run_evaluation(
        checkpoint=args.checkpoint,
        expected_sha256=args.expected_sha256,
        device=args.device,
        seed=args.seed,
        steps=args.steps,
        settle_steps=args.settle_steps,
    )
    if args.output is not None:
        output = publish_json_atomic(args.output, report)
        print(
            json.dumps(
                {"output": str(output), "summary": report["summary"]}, sort_keys=True
            )
        )
    else:
        print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
