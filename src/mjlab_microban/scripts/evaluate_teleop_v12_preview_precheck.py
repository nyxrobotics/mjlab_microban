"""Fast, non-deployable safety precheck for a contract-v12 preview.

This evaluator covers the observed worst-case both-feet scenario followed by
the two final-profile mixed-command scenarios.  It is a quick fail-first check
before the complete 9x300 + 8x300 preview acceptance suite; passing it is never
PICO-live or deployment authority and never replaces the full suite.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.nan_guard import NanGuard
from mjlab.utils.torch import configure_torch_backends
from tensordict import TensorDict

from mjlab_microban.legacy_velocity_diagnostics import publish_json_atomic
from mjlab_microban.robot.microban_hand_fk import microban_hand_fk_metadata
from mjlab_microban.scripts.evaluate_teleop_checkpoint import (
    HMD_ACTUAL_PEAK_TO_PEAK_MIN_RAD,
    HMD_TARGET_PEAK_TO_PEAK_MIN_RAD,
    HmdMotionStats,
    ScalarStats,
    _patch_initial_command_observation,
    _set_scenario,
)
from mjlab_microban.scripts.evaluate_teleop_v12_checkpoint import _load_actor
from mjlab_microban.scripts.evaluate_teleop_v12_tracking import (
    DIRECTIONAL_RESPONSE_MINIMUM,
    FINAL_PROFILE,
    TARGET_COLUMN_ABLATION_ACTION_DELTA_MIN,
    TARGET_COLUMN_ABLATION_METHOD,
    _scenarios,
    _target_column_ablation_evidence,
    _tracking_cfg,
    target_column_ablated_observation,
)
from mjlab_microban.tasks.microban_policy_export import MICROBAN_HMD_JOINT_NAMES
from mjlab_microban.tasks.microban_teleop_mdp import HmdNeckTargetMotion
from mjlab_microban.tasks.microban_teleop_v12_bootstrap import sha256_file
from mjlab_microban.tasks.microban_teleop_v12_preview import (
    TELEOP_V12_PREVIEW_INFO_KEY,
    TELEOP_V12_PREVIEW_PHASE2_LIFTED_ITERATION,
    TELEOP_V12_PREVIEW_PHASE_FULL_BODY,
    validate_preview_marker,
)
from mjlab_microban.tasks.microban_teleop_v12_runner import (
    validate_teleop_v12_environment_contract,
)
from mjlab_microban.teleop_v12_safety import (
    ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD,
)

TARGETED_SCENARIO_NAMES = (
    "bounded_both_feet",
    "mixed_forward_left",
    "mixed_backward_right",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_CHECKPOINT_RE = re.compile(r"model_([0-9]+)[.]pt")


def _preview_identity(
    checkpoint: Path,
    expected_sha256: str,
    *,
    diagnose_phase2_seed: bool = False,
) -> tuple[dict[str, Any], int, str]:
    """Authenticate exact preview bytes, marker, filename, and training clock."""

    checkpoint = checkpoint.expanduser().resolve()
    if _SHA256_RE.fullmatch(expected_sha256) is None:
        raise ValueError("Expected preview SHA-256 must be lowercase hexadecimal")
    match = _CHECKPOINT_RE.fullmatch(checkpoint.name)
    if match is None:
        raise ValueError("Preview checkpoint must be named model_<iteration>.pt")
    digest = sha256_file(checkpoint)
    if digest != expected_sha256:
        raise ValueError(f"Preview checkpoint SHA-256 mismatch: {digest}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("infos"), dict):
        raise TypeError("Preview checkpoint payload is malformed")
    iteration = payload.get("iter")
    if not isinstance(iteration, int) or isinstance(iteration, bool):
        raise TypeError("Preview checkpoint iteration is malformed")
    if iteration != int(match.group(1)):
        raise ValueError("Preview iteration does not match its filename")
    infos = payload["infos"]
    marker = validate_preview_marker(
        infos,
        iteration=iteration,
        required_phase=TELEOP_V12_PREVIEW_PHASE_FULL_BODY,
        require_live_candidate=not diagnose_phase2_seed,
    )
    if diagnose_phase2_seed and iteration != TELEOP_V12_PREVIEW_PHASE2_LIFTED_ITERATION:
        raise ValueError(
            "Phase-2 seed diagnosis requires exactly the lifted model_10000"
        )
    expected_env_state = {"common_step_counter": (iteration + 1) * 24}
    if infos.get("env_state") != expected_env_state:
        raise ValueError("Preview checkpoint clock is not exact")
    if sha256_file(checkpoint) != digest:
        raise ValueError("Preview checkpoint changed while reading identity")
    return marker, iteration, digest


def _action_observation_slice(env: ManagerBasedRlEnv) -> slice:
    terms = env.observation_manager.active_terms["actor"]
    shapes = env.observation_manager.group_obs_term_dim["actor"]
    offset = 0
    result: slice | None = None
    for name, shape in zip(terms, shapes, strict=True):
        width = math.prod(shape)
        if name == "actions":
            result = slice(offset, offset + width)
        offset += width
    if result is None:
        raise ValueError("Preview actor observation lacks previous actions")
    return result


def _joint_limit_evidence(
    *,
    names: tuple[str, ...],
    limits: torch.Tensor,
    maximum: torch.Tensor,
    maximum_step: list[int],
    maximum_position: list[float],
    maximum_side: list[str | None],
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        side = maximum_side[index]
        bound = None
        if side is not None:
            bound_index = 0 if side == "lower" else 1
            bound = float(limits[0, index, bound_index].item())
        evidence.append(
            {
                "joint": name,
                "maximum_violation_rad": float(maximum[index].item()),
                "step": maximum_step[index],
                "position_rad": (
                    maximum_position[index] if maximum_step[index] >= 0 else None
                ),
                "bound_side": side,
                "bound_rad": bound,
            }
        )
    return evidence


def _directional_response_passed(
    twist: tuple[float, float, float], response: dict[str, dict[str, Any]]
) -> bool:
    """Require every commanded twist axis, while accepting stationary cases."""

    axes = ("vx_m_s", "vy_m_s", "yaw_rad_s")
    expected = {
        axis for axis, command in zip(axes, twist, strict=True) if command != 0.0
    }
    return set(response) == expected and all(
        value.get("passed") is True for value in response.values()
    )


def _evaluate_targeted_scenario(
    *,
    env: ManagerBasedRlEnv,
    wrapped: RslRlVecEnvWrapper,
    policy: Any,
    scenario: Any,
    seed: int,
    steps: int,
    settle_steps: int,
) -> dict[str, Any]:
    env.reset(seed=seed)
    _set_scenario(env, scenario)
    observations = _patch_initial_command_observation(wrapped.get_observations(), env)
    robot = env.scene["robot"]
    action_term = env.action_manager.get_term("joint_pos")
    if action_term.cfg.clip is not None or wrapped.clip_actions is not None:
        raise ValueError("Preview precheck requires raw, unclipped actions")
    action_slice = _action_observation_slice(env)
    hmd_cfg = env.event_manager.get_term_cfg("hmd_neck_target_motion")
    hmd = hmd_cfg.func
    if not isinstance(hmd, HmdNeckTargetMotion) or tuple(hmd.joint_names) != (
        MICROBAN_HMD_JOINT_NAMES
    ):
        raise TypeError("Preview precheck requires the forced moving-HMD event")
    hmd_stats = HmdMotionStats.start(
        joint_names=tuple(hmd.joint_names),
        target=hmd.current_target[0],
        actual=robot.data.joint_pos[0, hmd.joint_ids],
    )

    joint_names = tuple(robot.joint_names)
    limits = robot.data.soft_joint_pos_limits
    maximum_violation = torch.zeros(len(joint_names), device="cpu")
    maximum_step = [-1] * len(joint_names)
    maximum_position = [0.0] * len(joint_names)
    maximum_side: list[str | None] = [None] * len(joint_names)
    velocity_stats = (ScalarStats(), ScalarStats(), ScalarStats())
    executed = 0
    raw_recurrence_steps = 0
    raw_recurrence_failure: dict[str, Any] | None = None
    nonfinite: dict[str, Any] | None = None
    fell = False
    termination_names: list[str] = []
    hmd_nonzero_steps = 0
    foot_nonzero_steps = 0
    hand_nonzero_steps = 0
    fall_height = float(
        env.termination_manager.get_term_cfg("fell_over").params["minimum_height"]
    )
    expects_foot = any(
        abs(value) > 0.0 for target in scenario.foot_target for value in target
    )
    expects_hand = any(scenario.hand_active)
    target_ablation_maximum: dict[str, float | None] = {
        "hand": None,
        "foot": None,
    }

    for step in range(steps):
        actor_obs = observations["actor"]
        if not bool(torch.isfinite(actor_obs).all().item()):
            nonfinite = {"step": step, "phase": "observation"}
            break
        hmd_nonzero_steps += int(
            bool(
                torch.count_nonzero(actor_obs[:, (*range(6, 9), *range(27, 30))]).item()
            )
        )
        foot_nonzero_steps += int(bool(torch.count_nonzero(actor_obs[:, 69:75]).item()))
        hand_nonzero_steps += int(bool(torch.count_nonzero(actor_obs[:, 75:83]).item()))
        with torch.inference_mode():
            actions = policy(TensorDict({"actor": actor_obs}, batch_size=[1]))
            ablated_actions: dict[str, torch.Tensor] = {}
            for target, expected in (
                ("hand", expects_hand),
                ("foot", expects_foot),
            ):
                if not expected:
                    continue
                ablated_obs = target_column_ablated_observation(actor_obs, target)
                ablated_actions[target] = policy(
                    TensorDict({"actor": ablated_obs}, batch_size=[1])
                )
        if not all(
            bool(value.isfinite().all().item())
            for value in (actions, *ablated_actions.values())
        ):
            nonfinite = {"step": step, "phase": "policy"}
            break
        for target, ablated in ablated_actions.items():
            value = float(torch.max(torch.abs(actions - ablated)).item())
            previous = target_ablation_maximum[target]
            target_ablation_maximum[target] = (
                value if previous is None else max(previous, value)
            )

        observations, rewards, dones, _extras = wrapped.step(actions)
        executed += 1
        action_manager_equal = torch.equal(action_term.raw_action, actions)
        observation_equal = torch.equal(observations["actor"][:, action_slice], actions)
        if action_manager_equal and observation_equal:
            raw_recurrence_steps += 1
        else:
            raw_recurrence_failure = {
                "step": step,
                "action_manager_equal": action_manager_equal,
                "observation_equal": observation_equal,
            }
            break

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
            nonfinite = {"step": step, "phase": "physics"}
            break

        hmd_stats.add(
            target=hmd.current_target[0],
            actual=robot.data.joint_pos[0, hmd.joint_ids],
        )
        position = robot.data.joint_pos[0]
        lower = torch.clamp(limits[0, :, 0] - position, min=0.0)
        upper = torch.clamp(position - limits[0, :, 1], min=0.0)
        violation = torch.maximum(lower, upper).detach().cpu()
        for index, value in enumerate(violation.tolist()):
            if value > float(maximum_violation[index].item()):
                maximum_violation[index] = value
                maximum_step[index] = step
                maximum_position[index] = float(position[index].item())
                maximum_side[index] = (
                    "lower" if float(lower[index].item()) > 0.0 else "upper"
                )

        root_height = float(robot.data.root_link_pos_w[0, 2].item())
        fell = (
            fell
            or bool(env.termination_manager.get_term("fell_over")[0].item())
            or root_height < fall_height
        )
        if step >= settle_steps:
            velocity_stats[0].add(robot.data.root_link_lin_vel_b[:, 0])
            velocity_stats[1].add(robot.data.root_link_lin_vel_b[:, 1])
            velocity_stats[2].add(robot.data.root_link_ang_vel_b[:, 2])
        if bool(dones[0].item()):
            termination_names = [
                name
                for name in env.termination_manager.active_terms
                if bool(env.termination_manager.get_term(name)[0].item())
            ]
            break

    axes = ("vx_m_s", "vy_m_s", "yaw_rad_s")
    measured_velocity = {
        name: stats.report(units="rad_s" if index == 2 else "m_s")
        for index, (name, stats) in enumerate(zip(axes, velocity_stats, strict=True))
    }
    directional_response: dict[str, dict[str, Any]] = {}
    for axis, command in zip(axes, scenario.twist, strict=True):
        if command == 0.0:
            continue
        mean = measured_velocity[axis]["mean"]
        signed = None if mean is None else float(mean) * (1.0 if command > 0 else -1.0)
        minimum = DIRECTIONAL_RESPONSE_MINIMUM[axis]
        directional_response[axis] = {
            "command": command,
            "measured_mean": mean,
            "signed_response": signed,
            "minimum_signed_response": minimum,
            "passed": signed is not None and signed >= minimum,
        }

    hmd_report = hmd_stats.report(step_dt=env.step_dt)
    hmd_passed = all(
        float(value["target_peak_to_peak_rad"]) >= HMD_TARGET_PEAK_TO_PEAK_MIN_RAD
        and float(value["actual_peak_to_peak_rad"]) >= HMD_ACTUAL_PEAK_TO_PEAK_MIN_RAD
        for value in hmd_report["per_axis"].values()
    )
    limit_evidence = _joint_limit_evidence(
        names=joint_names,
        limits=limits,
        maximum=maximum_violation,
        maximum_step=maximum_step,
        maximum_position=maximum_position,
        maximum_side=maximum_side,
    )
    maximum_actual = max(item["maximum_violation_rad"] for item in limit_evidence)
    target_column_ablation = _target_column_ablation_evidence(
        target_ablation_maximum,
        {"hand": expects_hand, "foot": expects_foot},
    )
    observation_coverage_passed = (
        hmd_nonzero_steps > 0
        and (not expects_foot or foot_nonzero_steps == executed)
        and (not expects_hand or hand_nonzero_steps == executed)
    )
    completed = (
        executed == steps
        and not fell
        and nonfinite is None
        and raw_recurrence_failure is None
        and not termination_names
    )
    checks = {
        "completed": completed,
        "no_fall": not fell,
        "finite": nonfinite is None,
        "actual_soft_limits": (
            maximum_actual <= ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD
        ),
        "raw_action_recurrence": raw_recurrence_steps == executed == steps,
        "directional_response": _directional_response_passed(
            scenario.twist, directional_response
        ),
        "forced_hmd_motion": hmd_passed,
        "nonzero_observation_coverage": (observation_coverage_passed),
        "target_column_ablation_response": all(
            value["passed"] is True for value in target_column_ablation.values()
        ),
    }
    return {
        "name": scenario.name,
        "status": "pass" if all(checks.values()) else "fail",
        "command": {"twist": list(scenario.twist)},
        "executed_steps": executed,
        "fell": fell,
        "nonfinite": nonfinite,
        "termination_names": termination_names,
        "raw_action_recurrence_verified_steps": raw_recurrence_steps,
        "raw_action_recurrence_failure": raw_recurrence_failure,
        "maximum_actual_soft_limit_violation_rad": maximum_actual,
        "joint_soft_limit_evidence": limit_evidence,
        "measured_velocity_body": measured_velocity,
        "directional_response": directional_response,
        "hmd_motion": hmd_report,
        "observation_coverage": {
            "hmd_nonzero_steps": hmd_nonzero_steps,
            "foot_nonzero_steps": foot_nonzero_steps,
            "hand_nonzero_steps": hand_nonzero_steps,
            "foot_target_expected": expects_foot,
            "hand_target_expected": expects_hand,
            "passed": observation_coverage_passed,
        },
        "target_column_ablation": target_column_ablation,
        "checks": checks,
    }


def run_preview_precheck(
    *,
    checkpoint: Path,
    expected_sha256: str,
    device: str = "cpu",
    diagnose_phase2_seed: bool = False,
) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve()
    marker, iteration, digest = _preview_identity(
        checkpoint,
        expected_sha256,
        diagnose_phase2_seed=diagnose_phase2_seed,
    )
    configure_torch_backends(allow_tf32=False, deterministic=True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    policy, loaded_iteration, _infos = _load_actor(
        checkpoint,
        device=device,
        allow_nondeployable_preview=True,
    )
    if loaded_iteration != iteration or sha256_file(checkpoint) != digest:
        raise ValueError("Preview checkpoint changed while loading actor")

    cfg = _tracking_cfg(seed=42, steps=300, perturbation=True)
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
    try:
        validate_teleop_v12_environment_contract(wrapped)
        by_name = {scenario.name: scenario for scenario in _scenarios(FINAL_PROFILE)}
        if tuple(name for name in TARGETED_SCENARIO_NAMES if name in by_name) != (
            TARGETED_SCENARIO_NAMES
        ):
            raise ValueError("Preview targeted scenario set drifted")
        results = [
            _evaluate_targeted_scenario(
                env=env,
                wrapped=wrapped,
                policy=policy,
                scenario=by_name[name],
                seed=42,
                steps=300,
                settle_steps=50,
            )
            for name in TARGETED_SCENARIO_NAMES
        ]
    finally:
        wrapped.close()
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    if sha256_file(checkpoint) != digest:
        raise ValueError("Preview checkpoint changed during precheck")

    checks = {
        "exact_scenarios_completed": all(
            result["checks"]["completed"] is True for result in results
        ),
        "no_falls": all(result["checks"]["no_fall"] is True for result in results),
        "finite": all(result["checks"]["finite"] is True for result in results),
        "actual_soft_limits": all(
            result["checks"]["actual_soft_limits"] is True for result in results
        ),
        "raw_action_recurrence": all(
            result["checks"]["raw_action_recurrence"] is True for result in results
        ),
        "directional_response": all(
            result["checks"]["directional_response"] is True for result in results
        ),
        "forced_hmd_motion": all(
            result["checks"]["forced_hmd_motion"] is True for result in results
        ),
        "nonzero_observation_coverage": all(
            result["checks"]["nonzero_observation_coverage"] is True
            for result in results
        ),
        "target_column_ablation_response": all(
            result["checks"]["target_column_ablation_response"] is True
            for result in results
        ),
    }
    all_checks_passed = all(checks.values())
    return {
        "schema_version": 1,
        "gate": "microban_teleop_v12_nondeployable_preview_targeted_precheck",
        "status": (
            "diagnostic"
            if diagnose_phase2_seed
            else ("pass" if all_checks_passed else "fail")
        ),
        "simulation_only": True,
        "preview_non_deployable": True,
        "canonical_deployment_accepted": False,
        "pico_live_accepted": False,
        "precheck_pass_is_go": False,
        "full_preview_acceptance_required": True,
        "scope": (
            "phase2_lifted_seed_diagnostic_never_authorizes_pico_live_or_deployment"
            if diagnose_phase2_seed
            else "fail_fast_only_pass_does_not_authorize_pico_live_or_deployment"
        ),
        "diagnose_phase2_seed": diagnose_phase2_seed,
        "diagnostic_checks_all_passed": (
            all_checks_passed if diagnose_phase2_seed else None
        ),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": digest,
            "iteration": iteration,
            "completed_updates": iteration + 1,
            "common_step_counter": (iteration + 1) * 24,
        },
        TELEOP_V12_PREVIEW_INFO_KEY: marker,
        "settings": {
            "device": device,
            "seed": 42,
            "steps": 300,
            "settle_steps": 50,
            "perturbation": True,
            "moving_hmd": "forced_non_neutral",
            "action_clip": None,
            "previous_action": "raw_actor_output",
            "target_column_ablation": TARGET_COLUMN_ABLATION_METHOD,
            "reachable_hand_target_fk": microban_hand_fk_metadata(),
            "scenario_names": list(TARGETED_SCENARIO_NAMES),
        },
        "thresholds": {
            "actual_soft_limit_violation_rad_max": (
                ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD
            ),
            "target_column_ablation_action_delta_min": (
                TARGET_COLUMN_ABLATION_ACTION_DELTA_MIN
            ),
        },
        "checks": checks,
        "results": results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--diagnose-phase2-seed",
        action="store_true",
        help=(
            "diagnose exact lifted model_10000 with the same scenarios; never emits "
            "PASS or PICO-live authority"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.expanduser().exists() and not args.force:
        raise FileExistsError(f"Output exists (pass --force): {args.output}")
    report = run_preview_precheck(
        checkpoint=args.checkpoint,
        expected_sha256=args.expected_sha256,
        device=args.device,
        diagnose_phase2_seed=args.diagnose_phase2_seed,
    )
    publish_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if report["status"] in {"pass", "diagnostic"} else 1


if __name__ == "__main__":
    sys.exit(main())
