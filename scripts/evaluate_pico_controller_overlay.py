#!/usr/bin/env python3
"""Deterministic simulation gate for legacy walking plus direct controller arm IK."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from mjlab_microban.legacy_velocity_diagnostics import publish_json_atomic
from mjlab_microban.robot.microban_hand_fk import (
    MICROBAN_ARM_HOME_JOINT_RAD,
    MICROBAN_ARM_JOINT_LOWER_RAD,
    MICROBAN_ARM_JOINT_UPPER_RAD,
    microban_hand_offsets_from_arm_joints,
)
from mjlab_microban.scripts.live_pico_teleop_sim import (
    AUDITED_LEGACY_WALK_SHA256,
    V12_PREVIEW_TASK,
    LivePicoSimulationPolicy,
    SimulationCommand,
    _configure_live_environment,
    _default_walk_checkpoint,
    _load_legacy_walk_actor,
    _sha256,
)
from mjlab_microban.teleop_v12_safety import (
    ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD,
    COMMANDED_TARGET_SOFT_LIMIT_EXCESS_MAX_RAD,
)


@dataclass(frozen=True)
class Scenario:
    name: str
    twist: tuple[float, float, float]
    arm_target: tuple[tuple[float, float, float], tuple[float, float, float]]


def _mirrored(
    left: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return left, (left[0], -left[1], left[2])


def scenarios() -> tuple[Scenario, ...]:
    lower = MICROBAN_ARM_JOINT_LOWER_RAD
    upper = MICROBAN_ARM_JOINT_UPPER_RAD
    return (
        Scenario("neutral_home", (0.0, 0.0, 0.0), MICROBAN_ARM_HOME_JOINT_RAD),
        Scenario("forward_upper", (0.2, 0.0, 0.0), upper),
        Scenario("backward_lower", (-0.2, 0.0, 0.0), lower),
        Scenario("left_mirrored", (0.0, 0.1, 0.0), _mirrored((0.25, 0.35, -0.5))),
        Scenario("right_mirrored", (0.0, -0.1, 0.0), _mirrored((-0.25, 0.35, -0.7))),
        Scenario("yaw_cross", (0.0, 0.0, 0.5), (upper[0], lower[1])),
    )


class _UnusedSource:
    def close(self) -> None:
        return None


class _UnusedMapper:
    def reset(self) -> None:
        return None

    @staticmethod
    def neutral() -> dict[str, object]:
        return {}


def _interpolate_arm_target(
    target: tuple[tuple[float, float, float], tuple[float, float, float]],
    step: int,
    ramp_steps: int,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    fraction = min(1.0, (step + 1) / max(1, ramp_steps))
    return tuple(
        tuple(
            home + fraction * (goal - home)
            for home, goal in zip(
                MICROBAN_ARM_HOME_JOINT_RAD[side], target[side], strict=True
            )
        )
        for side in range(2)
    )  # type: ignore[return-value]


def _command(
    scenario: Scenario,
    *,
    step: int,
    ramp_steps: int,
) -> SimulationCommand:
    arm_target = _interpolate_arm_target(scenario.arm_target, step, ramp_steps)
    hands = microban_hand_offsets_from_arm_joints(
        torch.tensor(arm_target, dtype=torch.float64)
    ).tolist()
    return SimulationCommand(
        enabled=True,
        twist=scenario.twist,
        foot_target=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        hand_target=(tuple(hands[0]), tuple(hands[1])),
        hand_active=(True, True),
        head_orientation=(0.0, 0.0, 0.0),
        head_yaw_front=False,
        locomotion_policy="pico_teleop",
        arm_joint_target=arm_target,
    )


def evaluate(
    *, device: str, steps: int, ramp_steps: int, settle_steps: int
) -> dict[str, object]:
    walk_checkpoint = _default_walk_checkpoint()
    if _sha256(walk_checkpoint) != AUDITED_LEGACY_WALK_SHA256:
        raise ValueError("Audited legacy walk checkpoint hash mismatch")

    env_cfg = load_env_cfg(V12_PREVIEW_TASK, play=True)
    _configure_live_environment(env_cfg)
    agent_cfg = load_rl_cfg(V12_PREVIEW_TASK)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    try:
        walk_actor = _load_legacy_walk_actor(walk_checkpoint, device)
        policy = LivePicoSimulationPolicy(
            env=env,
            actor=None,
            walk_actor=walk_actor,
            source=_UnusedSource(),
            mapper=_UnusedMapper(),
            controller_only_preview=True,
            status_period_s=1.0e9,
        )
        robot = env.scene["robot"]
        action_term = env.action_manager.get_term("joint_pos")
        arm_action_ids = torch.cat(
            (policy.arm_action_indices["left"], policy.arm_action_indices["right"])
        )
        arm_robot_ids = torch.cat(
            (policy.arm_joint_ids["left"], policy.arm_joint_ids["right"])
        )
        arm_action_limits = robot.data.soft_joint_pos_limits[
            :, action_term.target_ids[arm_action_ids]
        ]
        arm_actual_limits = robot.data.soft_joint_pos_limits[:, arm_robot_ids]
        results: list[dict[str, object]] = []
        for index, scenario in enumerate(scenarios()):
            env.reset(seed=42 + index)
            policy.walk_last_action.zero_()
            observations = wrapped.get_observations()
            settle_scenario = Scenario(
                "settle_home",
                (0.0, 0.0, 0.0),
                MICROBAN_ARM_HOME_JOINT_RAD,
            )
            settle_fell = False
            for step in range(settle_steps):
                settle_command = _command(
                    settle_scenario,
                    step=step,
                    ramp_steps=1,
                )
                policy._inject_command(settle_command)
                settle_base = policy._legacy_actor_action(observations)
                settle_action = policy._apply_controller_arm_overlay(
                    settle_base, settle_command
                )
                observations, _rewards, dones, _extras = wrapped.step(settle_action)
                if bool(dones.any().item()):
                    settle_fell = True
                    break
            maximum_actual_violation = 0.0
            maximum_target_violation = 0.0
            fell = False
            nonfinite = False
            executed = 0
            for step in range(steps):
                command = _command(
                    scenario,
                    step=step,
                    ramp_steps=ramp_steps,
                )
                policy._inject_command(command)
                base_action = policy._legacy_actor_action(observations)
                action = policy._apply_controller_arm_overlay(base_action, command)
                if not bool(torch.isfinite(action).all().item()):
                    nonfinite = True
                    break
                target = action * policy.main_action_scale + policy.main_action_offset
                arm_target = target[:, arm_action_ids]
                target_violation = torch.maximum(
                    arm_action_limits[..., 0] - arm_target,
                    arm_target - arm_action_limits[..., 1],
                ).clamp_min(0.0)
                maximum_target_violation = max(
                    maximum_target_violation,
                    float(target_violation.max().item()),
                )
                observations, _rewards, dones, _extras = wrapped.step(action)
                executed += 1
                actual = robot.data.joint_pos[:, arm_robot_ids]
                actual_violation = torch.maximum(
                    arm_actual_limits[..., 0] - actual,
                    actual - arm_actual_limits[..., 1],
                ).clamp_min(0.0)
                maximum_actual_violation = max(
                    maximum_actual_violation,
                    float(actual_violation.max().item()),
                )
                if bool(dones.any().item()):
                    fell = True
                    break
            passed = (
                executed == steps
                and not settle_fell
                and not fell
                and not nonfinite
                and maximum_target_violation
                <= COMMANDED_TARGET_SOFT_LIMIT_EXCESS_MAX_RAD
                and maximum_actual_violation
                <= ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD
            )
            results.append(
                {
                    "name": scenario.name,
                    "twist": list(scenario.twist),
                    "steps": executed,
                    "settle_fell": settle_fell,
                    "fell": fell,
                    "nonfinite": nonfinite,
                    "maximum_arm_target_soft_limit_violation_rad": (
                        maximum_target_violation
                    ),
                    "maximum_actual_arm_soft_limit_violation_rad": (
                        maximum_actual_violation
                    ),
                    "passed": passed,
                }
            )
        passed = all(bool(result["passed"]) for result in results)
        return {
            "schema_version": 1,
            "gate": "microban_pico_controller_direct_ik_overlay_simulation",
            "status": "pass" if passed else "fail",
            "simulation_only": True,
            "device": device,
            "steps_per_scenario": steps,
            "ramp_steps": ramp_steps,
            "settle_steps": settle_steps,
            "thresholds": {
                "maximum_arm_target_soft_limit_violation_rad": (
                    COMMANDED_TARGET_SOFT_LIMIT_EXCESS_MAX_RAD
                ),
                "maximum_actual_arm_soft_limit_transient_rad": (
                    ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD
                ),
            },
            "walk_checkpoint": {
                "path": str(walk_checkpoint.resolve()),
                "sha256": AUDITED_LEGACY_WALK_SHA256,
            },
            "foot_target": "exact_zero_inactive",
            "arm_base": "audited_legacy_walk_six_columns_replaced",
            "results": results,
        }
    finally:
        wrapped.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--ramp-steps", type=int, default=100)
    parser.add_argument("--settle-steps", type=int, default=50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        args.steps <= 0
        or args.ramp_steps <= 0
        or args.settle_steps <= 0
        or not math.isfinite(args.steps)
    ):
        parser.error("--steps, --ramp-steps and --settle-steps must be positive")
    report = evaluate(
        device=args.device,
        steps=args.steps,
        ramp_steps=args.ramp_steps,
        settle_steps=args.settle_steps,
    )
    if args.output is not None:
        publish_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
