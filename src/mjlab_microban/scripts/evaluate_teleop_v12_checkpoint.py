"""Evaluate a contract-v12 checkpoint on the nine legacy locomotion scenarios."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends
from tensordict import TensorDict

from mjlab_microban.legacy_velocity_diagnostics import (
    default_scenarios,
    publish_json_atomic,
)
from mjlab_microban.scripts.probe_legacy_actor_in_teleop_env import (
    _actor_layout,
    _evaluate_scenario,
    _indices_by_name,
    _source_cfg,
    _teleop_cfg,
    _tensor_parameter,
)
from mjlab_microban.tasks.microban_teleop_v12_actor import (
    TELEOP_V12_ADAPTER_GRADIENT_SCHEDULE_REVISION,
    LegacyAdapterTeleopActor,
    teleop_v12_active_adapter_columns,
)
from mjlab_microban.tasks.microban_teleop_v12_bootstrap import (
    assert_actor_frozen_against_source,
    sha256_file,
    validate_bootstrap_provenance,
)
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MICROBAN_TELEOP_V12_RECIPE_REVISION,
)
from mjlab_microban.tasks.microban_teleop_v12_preview import (
    TELEOP_V12_PREVIEW_PHASE_FULL_BODY,
    reject_preview_checkpoint,
    validate_preview_marker,
)
from mjlab_microban.tasks.microban_teleop_v12_runner import (
    TELEOP_V12_BOOTSTRAP_INFO_KEY,
)
from mjlab_microban.teleop_v12_safety import (
    ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD,
)

MINIMUM_SIGNED_RESPONSE = {
    "forward_0p1": 0.04,
    "forward_0p2": 0.08,
    "backward_0p1": 0.02,
    "backward_0p2": 0.04,
    "lateral_left_0p1": 0.02,
    "lateral_right_0p1": 0.02,
    "yaw_left_0p5": 0.20,
    "yaw_right_0p5": 0.20,
}


def _actor(device: str) -> LegacyAdapterTeleopActor:
    observations = TensorDict(
        {"actor": torch.zeros(1, 83, device=device)}, batch_size=[1]
    )
    return LegacyAdapterTeleopActor(
        obs=observations,
        obs_groups={"actor": ["actor"]},
        obs_set="actor",
        output_dim=18,
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ).to(device)


def _load_actor(
    checkpoint: Path,
    *,
    device: str,
    allow_nondeployable_preview: bool = False,
    allow_legacy_preview_v1: bool = False,
) -> tuple[LegacyAdapterTeleopActor, int, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("infos"), dict):
        raise TypeError("Contract-v12 checkpoint payload is malformed")
    infos = payload["infos"]
    iteration = payload.get("iter")
    if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < -1:
        raise ValueError("Checkpoint iteration is invalid")
    if allow_nondeployable_preview:
        preview_marker = validate_preview_marker(
            infos,
            iteration=iteration,
            allow_legacy_v1=allow_legacy_preview_v1,
        )
        if preview_marker.get("phase") == TELEOP_V12_PREVIEW_PHASE_FULL_BODY:
            from mjlab_microban.scripts.promote_teleop_v12_preview_visual import (
                validate_embedded_phase1_acceptance,
            )

            validate_embedded_phase1_acceptance(infos, fullbody_marker=preview_marker)
    else:
        reject_preview_checkpoint(infos)
    if infos.get("microban_teleop_training_contract_version") != "12":
        raise ValueError("Checkpoint is not contract-v12")
    if infos.get("microban_teleop_recipe_revision") != (
        MICROBAN_TELEOP_V12_RECIPE_REVISION
    ):
        raise ValueError("Checkpoint is not the safe contract-v12 recipe")
    if infos.get("previous_action_semantics") != "raw_actor_output":
        raise ValueError("Checkpoint previous-action semantics drifted")
    if infos.get("action_clip", object()) is not None:
        raise ValueError("Checkpoint action clip must be None")
    if infos.get("adapter_gradient_schedule_revision") != (
        TELEOP_V12_ADAPTER_GRADIENT_SCHEDULE_REVISION
    ):
        raise ValueError("Checkpoint adapter gradient schedule is not v12-safe")
    provenance = validate_bootstrap_provenance(
        infos.get(TELEOP_V12_BOOTSTRAP_INFO_KEY), verify_files=True
    )
    env_state = infos.get("env_state")
    expected_step = 0 if iteration == -1 else (iteration + 1) * 24
    if (
        not isinstance(env_state, dict)
        or env_state.get("common_step_counter") != expected_step
    ):
        raise ValueError("Checkpoint iteration/common-step relation drifted")
    actor = _actor(device)
    actor_state = payload.get("actor_state_dict")
    if not isinstance(actor_state, dict):
        raise TypeError("Checkpoint actor state is missing")
    actor.load_state_dict(actor_state, strict=True)
    actor.bind_common_step_provider(lambda: expected_step)
    if infos.get("active_actor_columns_at_save") != list(
        teleop_v12_active_adapter_columns(expected_step)
    ):
        raise ValueError("Checkpoint active adapter columns drifted from its clock")
    assert_actor_frozen_against_source(actor, provenance)
    actor.bind_frozen_legacy_reference()
    actor.assert_schedule_locked_weights_zero()
    actor.eval()
    return actor, iteration, infos


def _acceptance(results: list[dict[str, Any]]) -> tuple[dict[str, bool], str]:
    directional = {
        result["name"]: result["directional_response"] for result in results[1:]
    }
    checks = {
        "all_scenarios_completed": all(result["completed"] for result in results),
        "no_falls": not any(result["fell"] for result in results),
        "finite": all(result["nonfinite"] is None for result in results),
        "actual_soft_limits": all(
            0.0
            <= float(result["maximum_actual_soft_limit_violation_rad"])
            <= ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD
            for result in results
        ),
        "raw_action_recurrence": all(
            result["raw_action_recurrence_verified_steps"] == result["executed_steps"]
            for result in results
        ),
        "neutral_targets": all(
            result["neutral_foot_hand_target_verified_steps"]
            == result["executed_steps"]
            for result in results
        ),
        "directional_signs": all(
            response is not None and response["sign_matches"] is True
            for response in directional.values()
        ),
        "minimum_directional_response": all(
            response is not None
            and response["signed_response"] is not None
            and float(response["signed_response"]) >= MINIMUM_SIGNED_RESPONSE[name]
            for name, response in directional.items()
        ),
    }
    return checks, "pass" if all(checks.values()) else "fail"


def run_evaluation(
    *,
    checkpoint: Path,
    expected_sha256: str | None,
    device: str,
    seed: int,
    steps: int,
    settle_steps: int,
    allow_nondeployable_preview: bool = False,
    allow_legacy_preview_v1: bool = False,
) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve()
    digest = sha256_file(checkpoint)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"Checkpoint SHA-256 mismatch: {digest}")
    if steps != 300 or settle_steps != 50 or seed != 42:
        raise ValueError("Canonical v12 gate requires seed42, 300 steps, settle50")
    configure_torch_backends(allow_tf32=False, deterministic=True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    policy, iteration, _infos = _load_actor(
        checkpoint,
        device=device,
        allow_nondeployable_preview=allow_nondeployable_preview,
        allow_legacy_preview_v1=allow_legacy_preview_v1,
    )

    source_env = ManagerBasedRlEnv(
        cfg=_source_cfg(seed=seed, steps=steps), device=device
    )
    source_wrapped = RslRlVecEnvWrapper(source_env, clip_actions=None)
    try:
        legacy_layout = _actor_layout(source_env)
        source_action = source_env.action_manager.get_term("joint_pos")
        source_scale = _tensor_parameter(
            source_action.scale, source_action.raw_action
        ).clone()
        source_offset = _tensor_parameter(
            source_action.offset, source_action.raw_action
        ).clone()
    finally:
        source_wrapped.close()

    teleop_env = ManagerBasedRlEnv(
        cfg=_teleop_cfg(seed=seed, steps=steps), device=device
    )
    wrapped = RslRlVecEnvWrapper(teleop_env, clip_actions=None)
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
        teleop_action = teleop_env.action_manager.get_term("joint_pos")
        if not torch.equal(
            source_scale,
            _tensor_parameter(teleop_action.scale, teleop_action.raw_action),
        ) or not torch.equal(
            source_offset,
            _tensor_parameter(teleop_action.offset, teleop_action.raw_action),
        ):
            raise ValueError("Legacy and teleop action transforms differ")
        results = []
        for scenario in default_scenarios():
            print(f"[INFO] evaluating v12 {scenario.name}", flush=True)
            results.append(
                _evaluate_scenario(
                    env=teleop_env,
                    wrapped=wrapped,
                    policy=policy,
                    teleop_layout=teleop_layout,
                    legacy_layout=legacy_layout,
                    joint_pos_indices=joint_pos_indices,
                    joint_vel_indices=joint_vel_indices,
                    action_indices=action_indices,
                    scenario=scenario,
                    seed=seed,
                    steps=steps,
                    settle_steps=settle_steps,
                    policy_observation_mode="teleop83",
                )
            )
    finally:
        wrapped.close()
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    checks, status = _acceptance(results)
    directional = [
        result["directional_response"]
        for result in results
        if result["directional_response"] is not None
    ]
    return {
        "schema_version": 1,
        "gate": "microban_teleop_v12_neutral_locomotion_9x300",
        "status": status,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": digest,
            "iteration": iteration,
            "completed_updates": 0 if iteration == -1 else iteration + 1,
        },
        "settings": {
            "device": device,
            "seed": seed,
            "steps": steps,
            "settle_steps": settle_steps,
            "action_clip": None,
            "previous_action": "raw_actor_output",
            "policy_observation_width": 83,
        },
        "thresholds": {
            "actual_soft_limit_violation_rad_max": (
                ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD
            ),
            "minimum_signed_response": MINIMUM_SIGNED_RESPONSE,
        },
        "checks": checks,
        "results": results,
        "summary": {
            "scenario_count": len(results),
            "completed_scenario_count": sum(
                bool(result["completed"]) for result in results
            ),
            "fall_scenario_count": sum(bool(result["fell"]) for result in results),
            "nonfinite_scenario_count": sum(
                result["nonfinite"] is not None for result in results
            ),
            "directionally_correct_scenario_count": sum(
                bool(response["sign_matches"]) for response in directional
            ),
            "directional_scenario_count": len(directional),
            "minimum_signed_response": min(
                float(response["signed_response"])
                for response in directional
                if response["signed_response"] is not None
            ),
            "maximum_actual_soft_limit_violation_rad": max(
                float(result["maximum_actual_soft_limit_violation_rad"])
                for result in results
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--settle-steps", type=int, default=50)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_evaluation(
        checkpoint=args.checkpoint,
        expected_sha256=args.expected_sha256,
        device=args.device,
        seed=args.seed,
        steps=args.steps,
        settle_steps=args.settle_steps,
    )
    if args.output is not None:
        if args.output.expanduser().exists() and not args.force:
            raise FileExistsError(
                f"Output already exists (pass --force): {args.output}"
            )
        publish_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
