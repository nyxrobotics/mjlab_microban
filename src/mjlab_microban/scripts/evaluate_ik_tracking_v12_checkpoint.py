"""Evaluate a contract-v12 checkpoint on the nine legacy locomotion scenarios.

With ``--floor`` the scenarios run on a soft or slippery floor of the walking
judgment (pipeline/floors.py) in place of the play env's rigid one, with the
same lines; ``--twist-minimum-scale`` (a judged floor's
``lines.twist_minimum_scale``) multiplies the direction's minimums there.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends
from tensordict import TensorDict

from mjlab_microban.tasks.microban_ik_tracking_v12_env_cfg import (
    MICROBAN_IK_TRACKING_V12_ACTION_CLIP,
    MICROBAN_IK_TRACKING_V12_TRAINING_CONTRACT_VERSION,
    MICROBAN_IK_TRACKING_V13_ARM_OVERLAY_RECIPE_REVISION,
)
from mjlab_microban.tasks.microban_policy_export import MICROBAN_IK_TRACKING_OBSERVATION_WIDTH
from mjlab_microban.legacy_velocity_diagnostics import (
    default_scenarios,
    publish_json_atomic,
)
from mjlab_microban.scripts.probe_legacy_actor_in_ik_tracking_env import (
    _actor_layout,
    _evaluate_scenario,
    _indices_by_name,
    _source_cfg,
    _ik_tracking_cfg,
    _tensor_parameter,
)
from mjlab_microban.tasks.microban_ik_tracking_v12_actor import (
    IK_TRACKING_V12_ADAPTER_GRADIENT_SCHEDULE_REVISION,
    LegacyAdapterIkTrackingActor,
    ik_tracking_v12_active_adapter_columns,
)
from mjlab_microban.tasks.microban_ik_tracking_v12_bootstrap import (
    assert_actor_frozen_against_source,
    sha256_file,
    validate_bootstrap_provenance,
)
from mjlab_microban.tasks.microban_ik_tracking_v12_home_pose import (
    validate_ik_tracking_v12_home_pose,
)
from mjlab_microban.tasks.microban_ik_tracking_v12_runner import (
    IK_TRACKING_V12_BOOTSTRAP_INFO_KEY,
    require_bilateral_site_order,
)
from mjlab_microban.ik_tracking_v12_safety import (
    ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD,
)

from mjlab_microban.twist_pass_line import (
    twist_judgment,
    twist_pass_line_record,
)

_AXES = ("vx_m_s", "vy_m_s", "yaw_rad_s")
# The eight moving scenarios after "neutral" (legacy_velocity_diagnostics).
LOCOMOTION_MOVING_SCENARIOS = (
    "forward_0p1",
    "forward_0p2",
    "backward_0p1",
    "backward_0p2",
    "lateral_left_0p1",
    "lateral_right_0p1",
    "yaw_left_0p5",
    "yaw_right_0p5",
)


def locomotion_twist_judgments(results: list[dict[str, Any]], scale: float = 1.0) -> dict[str, Any]:
    """The walking reward's pass line on each moving scenario's mean body twist (its minimums times ``scale``)."""

    judgments = {}
    for result in results:
        command = [float(result["command"][axis]) for axis in _AXES]
        if all(value == 0.0 for value in command):
            continue
        twist = [float(result["measured_velocity_body"][axis]["mean"]) for axis in _AXES]
        judgments[result["name"]] = twist_judgment(command, twist, scale)
    return judgments


def _actor(device: str) -> LegacyAdapterIkTrackingActor:
    observations = TensorDict(
        {"actor": torch.zeros(1, MICROBAN_IK_TRACKING_OBSERVATION_WIDTH, device=device)},
        batch_size=[1],
    )
    return LegacyAdapterIkTrackingActor(
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


def recipe_report_settings(infos: dict[str, Any]) -> dict[str, Any]:
    """Report fields that label an arm-overlay checkpoint's evidence (or none)."""

    if infos.get("microban_teleop_recipe_revision") != (
        MICROBAN_IK_TRACKING_V13_ARM_OVERLAY_RECIPE_REVISION
    ):
        return {}
    return {"recipe_revision": infos.get("microban_teleop_recipe_revision")}


def _load_actor(
    checkpoint: Path,
    *,
    device: str,
) -> tuple[LegacyAdapterIkTrackingActor, int, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("infos"), dict):
        raise TypeError("Contract-v12 checkpoint payload is malformed")
    infos = payload["infos"]
    iteration = payload.get("iter")
    if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < -1:
        raise ValueError("Checkpoint iteration is invalid")
    expected_step = 0 if iteration == -1 else (iteration + 1) * 24
    if infos.get("microban_teleop_training_contract_version") != (
        MICROBAN_IK_TRACKING_V12_TRAINING_CONTRACT_VERSION
    ):
        raise ValueError("Checkpoint is not of the current IK tracking training contract")
    validate_ik_tracking_v12_home_pose(infos)
    require_bilateral_site_order(infos)
    expected_active_columns = list(ik_tracking_v12_active_adapter_columns(expected_step))
    if infos.get("previous_action_semantics") != "raw_actor_output":
        raise ValueError("Checkpoint previous-action semantics drifted")
    if infos.get("action_clip", object()) != MICROBAN_IK_TRACKING_V12_ACTION_CLIP:
        raise ValueError("Checkpoint action clip must be the servo goal range (+-pi)")
    if infos.get("adapter_gradient_schedule_revision") != (
        IK_TRACKING_V12_ADAPTER_GRADIENT_SCHEDULE_REVISION
    ):
        raise ValueError("Checkpoint adapter gradient schedule is not v12-safe")
    provenance = validate_bootstrap_provenance(
        infos.get(IK_TRACKING_V12_BOOTSTRAP_INFO_KEY), verify_files=True
    )
    env_state = infos.get("env_state")
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
    if infos.get("active_actor_columns_at_save") != expected_active_columns:
        raise ValueError("Checkpoint active adapter columns drifted from its clock")
    assert_actor_frozen_against_source(actor, provenance)
    actor.bind_frozen_legacy_reference()
    actor.assert_schedule_locked_weights_zero()
    actor.eval()
    return actor, iteration, infos


def _acceptance(results: list[dict[str, Any]], scale: float = 1.0) -> tuple[dict[str, bool], str]:
    judgments = locomotion_twist_judgments(results, scale)
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
            result["neutral_target_verified_steps"]
            == result["executed_steps"]
            for result in results
        ),
        "directional_signs": all(
            response is not None and response["sign_matches"] is True
            for response in directional.values()
        ),
        "twist_direction": tuple(judgments) == LOCOMOTION_MOVING_SCENARIOS
        and all(item["passed"] is True for item in judgments.values()),
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
    floor: tuple[float | None, float] | None = None,
    twist_minimum_scale: float = 1.0,
) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve()
    digest = sha256_file(checkpoint)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"Checkpoint SHA-256 mismatch: {digest}")
    if steps != 300 or settle_steps != 50:
        raise ValueError("The v12 locomotion evaluation runs 300 steps after a 50-step settle")
    if not 0.0 < twist_minimum_scale <= 1.0:
        raise ValueError(f"A minimum scale loosens the line: 0 < scale <= 1, got {twist_minimum_scale}")
    configure_torch_backends(allow_tf32=False, deterministic=True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    policy, iteration, _infos = _load_actor(checkpoint, device=device)

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

    ik_tracking_env = ManagerBasedRlEnv(
        cfg=_ik_tracking_cfg(seed=seed, steps=steps, floor=floor), device=device
    )
    wrapped = RslRlVecEnvWrapper(ik_tracking_env, clip_actions=None)
    try:
        sole_floor = None
        if floor is not None:
            from mjlab_microban.pipeline.floors import sole_floors

            sole_floor = sole_floors(ik_tracking_env)[0]
            if not math.isclose(sole_floor[1], floor[1], rel_tol=1e-5) or (
                floor[0] is not None and not math.isclose(sole_floor[0], floor[0], rel_tol=1e-5)
            ):
                raise RuntimeError(f"floor {floor} simulated as {sole_floor}")
        ik_tracking_layout = _actor_layout(ik_tracking_env)
        joint_pos_indices = _indices_by_name(
            legacy_layout.joint_pos_names,
            ik_tracking_layout.joint_pos_names,
            label="joint-position",
        )
        joint_vel_indices = _indices_by_name(
            legacy_layout.joint_vel_names,
            ik_tracking_layout.joint_vel_names,
            label="joint-velocity",
        )
        action_indices = _indices_by_name(
            legacy_layout.action_names,
            ik_tracking_layout.action_names,
            label="previous-action",
        )
        ik_tracking_action = ik_tracking_env.action_manager.get_term("joint_pos")
        if not torch.equal(
            source_scale,
            _tensor_parameter(ik_tracking_action.scale, ik_tracking_action.raw_action),
        ) or not torch.equal(
            source_offset,
            _tensor_parameter(ik_tracking_action.offset, ik_tracking_action.raw_action),
        ):
            raise ValueError("Legacy and IK tracking action transforms differ")
        results = []
        for scenario in default_scenarios():
            print(f"[INFO] evaluating v12 {scenario.name}", flush=True)
            results.append(
                _evaluate_scenario(
                    env=ik_tracking_env,
                    wrapped=wrapped,
                    policy=policy,
                    ik_tracking_layout=ik_tracking_layout,
                    legacy_layout=legacy_layout,
                    joint_pos_indices=joint_pos_indices,
                    joint_vel_indices=joint_vel_indices,
                    action_indices=action_indices,
                    scenario=scenario,
                    seed=seed,
                    steps=steps,
                    settle_steps=settle_steps,
                    policy_observation_mode="ik_tracking",
                )
            )
    finally:
        wrapped.close()
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    checks, status = _acceptance(results, twist_minimum_scale)
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
            "action_clip": list(MICROBAN_IK_TRACKING_V12_ACTION_CLIP),
            "previous_action": "raw_actor_output",
            "policy_observation_width": MICROBAN_IK_TRACKING_OBSERVATION_WIDTH,
            **recipe_report_settings(_infos),
            **({"floor": {"friction": floor[0], "time_constant_s": floor[1], "sole_floor": sole_floor}}
               if floor is not None else {}),
        },
        "thresholds": {
            "actual_soft_limit_violation_rad_max": (
                ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD
            ),
            "twist_pass_line": twist_pass_line_record(),
            # a judged floor's own line; the rigid floor's report stays as the stage gate reads it
            **({"twist_minimum_scale": twist_minimum_scale} if twist_minimum_scale != 1.0 else {}),
        },
        "checks": checks,
        "twist_judgment": locomotion_twist_judgments(results, twist_minimum_scale),
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
    parser.add_argument("--floor", help='a soft or slippery floor of the walking judgment, JSON '
                        '{"time_constant_s": s, "friction": mu or null} (pipeline/floors.py); default the play floor')
    parser.add_argument("--twist-minimum-scale", type=float, default=1.0,
                        help="multiplies the direction's minimums, 0 < scale <= 1 (a judged floor's "
                        "lines.twist_minimum_scale; default %(default)s)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    floor = None
    if args.floor:
        from mjlab_microban.pipeline.floors import parse

        floor = parse({"floor": json.loads(args.floor)})["floor"]
    report = run_evaluation(
        checkpoint=args.checkpoint,
        expected_sha256=args.expected_sha256,
        device=args.device,
        seed=args.seed,
        steps=args.steps,
        settle_steps=args.settle_steps,
        floor=floor,
        twist_minimum_scale=args.twist_minimum_scale,
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
