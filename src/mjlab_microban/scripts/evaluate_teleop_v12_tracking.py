"""Hash-bound staged HMD/hand/foot exposure and tracking gate for v12.

The raw legacy action contract intentionally has no target clip.  This gate
therefore rejects falls, non-finite values, broken raw-action recurrence, and
actual 21-joint soft-limit violations; hypothetical raw target excess is only
reported.  It also records per-joint source/v12/delta action envelopes for the
runtime finite-amplitude guard without inventing a bound before final evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.nan_guard import NanGuard
from mjlab.utils.torch import configure_torch_backends
from tensordict import TensorDict

from mjlab_microban.legacy_velocity_diagnostics import publish_json_atomic
from mjlab_microban.robot.microban_hand_fk import (
    microban_hand_fk_metadata,
    microban_reachable_hand_evaluation_offsets,
)
from mjlab_microban.scripts.evaluate_teleop_checkpoint import (
    HMD_ACTUAL_PEAK_TO_PEAK_MIN_RAD,
    HMD_TARGET_PEAK_TO_PEAK_MIN_RAD,
    EvaluationScenario,
    HmdMotionStats,
    ScalarStats,
    _configure_nominal_evaluation,
    _copy_forced_moving_hmd_neck_event,
    _patch_initial_command_observation,
    _set_scenario,
    default_scenarios,
)
from mjlab_microban.scripts.evaluate_teleop_v12_checkpoint import _load_actor
from mjlab_microban.scripts.teleop_v12_bootstrap_gate import (
    _legacy_model,
)
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_HMD_JOINT_NAMES,
    MICROBAN_TELEOP_ACTION_JOINT_NAMES,
    MICROBAN_TELEOP_OBSERVATION_WIDTH,
)
from mjlab_microban.tasks.microban_teleop_mdp import HmdNeckTargetMotion
from mjlab_microban.tasks.microban_teleop_v12_actor import (
    LEGACY_TO_TELEOP_OBSERVATION_INDEX,
    LEGACY_VELOCITY_CHECKPOINT_SHA256,
    TELEOP_V12_FOOT_OBSERVATION_COLUMNS,
    TELEOP_V12_HAND_ACTIVE_OBSERVATION_COLUMNS,
    TELEOP_V12_HAND_POSITION_OBSERVATION_COLUMNS,
)
from mjlab_microban.tasks.microban_teleop_v12_bootstrap import (
    inspect_legacy_velocity_checkpoint,
    sha256_file,
)
from mjlab_microban.tasks.microban_teleop_v12_deadline_fallback import (
    MICROBAN_TELEOP_V12_DEADLINE_CANARY_CHECKPOINT_SHA256,
    MICROBAN_TELEOP_V12_DEADLINE_CANARY_FALLBACK_PROFILE,
    MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_PROFILE,
    MICROBAN_TELEOP_V12_DEADLINE_FINAL_FOOT_P95_MAX_M,
    MICROBAN_TELEOP_V12_DEADLINE_FINAL_FOOT_RMS_MAX_M,
    MICROBAN_TELEOP_V12_DEADLINE_FINAL_FALLBACK_PROFILE,
    MICROBAN_TELEOP_V12_DEADLINE_FINAL_HAND_P95_MAX_M,
    MICROBAN_TELEOP_V12_DEADLINE_HAND_RMS_MAX_M,
    MICROBAN_TELEOP_V12_DEADLINE_POST_CANARY_INFO_KEY,
)
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    make_microban_teleop_v12_env_cfg,
)
from mjlab_microban.tasks.microban_teleop_v12_preview import (
    TELEOP_V12_PREVIEW_INFO_KEY,
    TELEOP_V12_PREVIEW_LEGACY_REVISION,
    TELEOP_V12_PREVIEW_PHASE_HMD_HAND,
)
from mjlab_microban.tasks.microban_teleop_v12_runner import (
    validate_teleop_v12_environment_contract,
)
from mjlab_microban.teleop_v12_safety import (
    ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD,
)

PRE_ACTIVATION_EXPOSURE_PROFILE = "pre_hmd_hand_foot_exposure_reachable_safety_v2"
EXPANDED_LOCOMOTION_PROFILE = "expanded_locomotion_pre_hmd_exposure_reachable_safety_v2"
HMD_HAND_PROFILE = "hmd_hand_reachable_performance_foot_exposure_v2"
HMD_HAND_ACTIVATION_CANARY_PROFILE = "hmd_hand_activation_canary_reachable_safety_v1"
FOOT_ACTIVATION_CANARY_PROFILE = "whole_body_foot_activation_canary_reachable_safety_v1"
WHOLE_BODY_PROFILE = "whole_body_reachable_performance_v2"
FINAL_PROFILE = "full_body_reachable_performance_perturbation_v2"
DEADLINE_FALLBACK_PROFILE = MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_PROFILE
DEADLINE_CANARY_FALLBACK_PROFILE = MICROBAN_TELEOP_V12_DEADLINE_CANARY_FALLBACK_PROFILE
DEADLINE_FINAL_FALLBACK_PROFILE = MICROBAN_TELEOP_V12_DEADLINE_FINAL_FALLBACK_PROFILE
TRACKING_PROFILES = (
    PRE_ACTIVATION_EXPOSURE_PROFILE,
    EXPANDED_LOCOMOTION_PROFILE,
    HMD_HAND_ACTIVATION_CANARY_PROFILE,
    HMD_HAND_PROFILE,
    FOOT_ACTIVATION_CANARY_PROFILE,
    WHOLE_BODY_PROFILE,
    FINAL_PROFILE,
    DEADLINE_FALLBACK_PROFILE,
    DEADLINE_CANARY_FALLBACK_PROFILE,
    DEADLINE_FINAL_FALLBACK_PROFILE,
)

HAND_RMS_MAX_M = 0.03
HAND_P95_MAX_M = 0.05
FOOT_RMS_MAX_M = 0.015
FOOT_P95_MAX_M = 0.025
DIRECTIONAL_RESPONSE_MINIMUM = {
    "vx_m_s": 0.04,
    "vy_m_s": 0.02,
    "yaw_rad_s": 0.20,
}
TARGET_COLUMN_ABLATION_ACTION_DELTA_MIN = 1.0e-4
TARGET_COLUMN_ABLATION_METHOD = (
    "same_observation_zero_target_position_columns_preserve_hand_active_flags_"
    "before_actor_forward_v2"
)


def hand_tracking_rms_max_m(profile: str) -> float:
    """Return the profile-specific hand RMS limit; all other limits are fixed."""

    required_tracking_scenario_names(profile)
    if profile in (
        DEADLINE_FALLBACK_PROFILE,
        DEADLINE_CANARY_FALLBACK_PROFILE,
        DEADLINE_FINAL_FALLBACK_PROFILE,
    ):
        return MICROBAN_TELEOP_V12_DEADLINE_HAND_RMS_MAX_M
    return HAND_RMS_MAX_M


def hand_tracking_p95_max_m(profile: str) -> float:
    """Return the profile-specific hand P95 limit."""

    required_tracking_scenario_names(profile)
    if profile == DEADLINE_FINAL_FALLBACK_PROFILE:
        return MICROBAN_TELEOP_V12_DEADLINE_FINAL_HAND_P95_MAX_M
    return HAND_P95_MAX_M


def foot_tracking_rms_max_m(profile: str) -> float:
    """Return the profile-specific foot RMS limit."""

    required_tracking_scenario_names(profile)
    if profile == DEADLINE_FINAL_FALLBACK_PROFILE:
        return MICROBAN_TELEOP_V12_DEADLINE_FINAL_FOOT_RMS_MAX_M
    return FOOT_RMS_MAX_M


def foot_tracking_p95_max_m(profile: str) -> float:
    """Return the profile-specific foot P95 limit."""

    required_tracking_scenario_names(profile)
    if profile == DEADLINE_FINAL_FALLBACK_PROFILE:
        return MICROBAN_TELEOP_V12_DEADLINE_FINAL_FOOT_P95_MAX_M
    return FOOT_P95_MAX_M


def target_column_ablation_observation_columns(
    target: str,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return zeroed and explicitly preserved actor columns for one target."""

    if target == "hand":
        return (
            TELEOP_V12_HAND_POSITION_OBSERVATION_COLUMNS,
            TELEOP_V12_HAND_ACTIVE_OBSERVATION_COLUMNS,
        )
    if target == "foot":
        return TELEOP_V12_FOOT_OBSERVATION_COLUMNS, ()
    raise ValueError(f"Unknown target-column ablation target: {target!r}")


def target_column_ablated_observation(
    actor_observation: torch.Tensor, target: str
) -> torch.Tensor:
    """Zero only target-position columns while preserving all other inputs."""

    if actor_observation.ndim != 2 or actor_observation.shape[1] != (
        MICROBAN_TELEOP_OBSERVATION_WIDTH
    ):
        raise ValueError("Target-column ablation requires a [batch, 83] observation")
    ablated_columns, preserved_columns = target_column_ablation_observation_columns(
        target
    )
    result = actor_observation.clone()
    preserved = result[:, preserved_columns].clone()
    result[:, ablated_columns] = 0.0
    if preserved_columns and not torch.equal(result[:, preserved_columns], preserved):
        raise RuntimeError("Target-column ablation modified preserved columns")
    return result


def required_tracking_scenario_names(profile: str) -> tuple[str, ...]:
    """Return the exact ordered scenario set authenticated by a stage gate."""

    names_by_profile = {
        PRE_ACTIVATION_EXPOSURE_PROFILE: (
            "low_forward",
            "max_hands_left",
            "max_keypoints_left",
        ),
        EXPANDED_LOCOMOTION_PROFILE: (
            "low_forward",
            "max_forward",
            "mixed_twist_forward_left",
            "max_hands_left",
            "max_keypoints_left",
        ),
        HMD_HAND_PROFILE: (
            "low_forward",
            "max_hands_left",
            "max_hands_right",
            "max_keypoints_left",
        ),
        DEADLINE_FALLBACK_PROFILE: (
            "low_forward",
            "max_hands_left",
            "max_hands_right",
            "max_keypoints_left",
        ),
        DEADLINE_CANARY_FALLBACK_PROFILE: (
            "low_forward",
            "max_hands_left",
            "max_hands_right",
            "max_keypoints_left",
            "max_keypoints_right",
            "bounded_both_feet",
        ),
        DEADLINE_FINAL_FALLBACK_PROFILE: (
            "low_forward",
            "max_hands_left",
            "max_hands_right",
            "max_keypoints_left",
            "max_keypoints_right",
            "bounded_both_feet",
            "mixed_forward_left",
            "mixed_backward_right",
        ),
        HMD_HAND_ACTIVATION_CANARY_PROFILE: (
            "low_forward",
            "max_hands_left",
            "max_hands_right",
        ),
        FOOT_ACTIVATION_CANARY_PROFILE: (
            "low_forward",
            "max_hands_left",
            "max_hands_right",
            "max_keypoints_left",
            "max_keypoints_right",
            "bounded_both_feet",
        ),
        WHOLE_BODY_PROFILE: (
            "low_forward",
            "max_hands_left",
            "max_hands_right",
            "max_keypoints_left",
            "max_keypoints_right",
            "bounded_both_feet",
        ),
        FINAL_PROFILE: (
            "low_forward",
            "max_hands_left",
            "max_hands_right",
            "max_keypoints_left",
            "max_keypoints_right",
            "bounded_both_feet",
            "mixed_forward_left",
            "mixed_backward_right",
        ),
    }
    try:
        return names_by_profile[profile]
    except KeyError as exc:
        raise ValueError(f"Unknown v12 tracking profile: {profile}") from exc


def required_tracking_check_names(profile: str) -> frozenset[str]:
    """Return the exact acceptance checks required for one stage profile."""

    required_tracking_scenario_names(profile)
    names = {
        "all_scenarios_completed",
        "no_falls",
        "finite",
        "actual_soft_limits",
        "raw_action_recurrence",
        "forced_hmd_motion",
        "nonzero_observation_coverage",
        "target_column_ablation_response",
        "twist_directional_response",
    }
    if profile in (
        HMD_HAND_PROFILE,
        DEADLINE_FALLBACK_PROFILE,
        DEADLINE_CANARY_FALLBACK_PROFILE,
        DEADLINE_FINAL_FALLBACK_PROFILE,
        FOOT_ACTIVATION_CANARY_PROFILE,
        WHOLE_BODY_PROFILE,
        FINAL_PROFILE,
    ):
        names.update(("hand_tracking_rms", "hand_tracking_p95"))
    if profile in (WHOLE_BODY_PROFILE, FINAL_PROFILE, DEADLINE_FINAL_FALLBACK_PROFILE):
        names.update(("foot_tracking_rms", "foot_tracking_p95"))
    return frozenset(names)


def required_target_column_ablation_targets(profile: str) -> frozenset[str]:
    """Return target inputs that this curriculum stage must have learned to use."""

    required_tracking_scenario_names(profile)
    if profile in (
        HMD_HAND_ACTIVATION_CANARY_PROFILE,
        HMD_HAND_PROFILE,
        DEADLINE_FALLBACK_PROFILE,
    ):
        return frozenset(("hand",))
    if profile in (
        FOOT_ACTIVATION_CANARY_PROFILE,
        DEADLINE_CANARY_FALLBACK_PROFILE,
        DEADLINE_FINAL_FALLBACK_PROFILE,
        WHOLE_BODY_PROFILE,
        FINAL_PROFILE,
    ):
        return frozenset(("hand", "foot"))
    return frozenset()


def required_tracking_profile(completed_updates: int) -> str:
    if isinstance(completed_updates, bool) or completed_updates <= 0:
        raise ValueError("completed_updates must be a positive integer")
    if completed_updates <= 3_000:
        return PRE_ACTIVATION_EXPOSURE_PROFILE
    if completed_updates <= 7_000:
        return EXPANDED_LOCOMOTION_PROFILE
    if completed_updates <= 7_100:
        return HMD_HAND_ACTIVATION_CANARY_PROFILE
    if completed_updates <= 10_000:
        return HMD_HAND_PROFILE
    if completed_updates <= 10_100:
        return FOOT_ACTIVATION_CANARY_PROFILE
    if completed_updates < 15_000:
        return WHOLE_BODY_PROFILE
    if completed_updates == 15_000:
        return FINAL_PROFILE
    raise ValueError("Contract-v12 training must not exceed 15000 updates")


def _scenarios(profile: str) -> tuple[EvaluationScenario, ...]:
    by_name = {scenario.name: scenario for scenario in default_scenarios()}
    reachable = dict(microban_reachable_hand_evaluation_offsets())
    forward = reachable["F"]
    backward = reachable["B"]
    forward_moderate = reachable["f"]
    backward_moderate = reachable["b"]
    hand_overrides = {
        "max_hands_left": (forward[0], backward[1]),
        "max_hands_right": (backward[0], forward[1]),
        "max_keypoints_left": (forward[0], backward[1]),
        "max_keypoints_right": (backward[0], forward[1]),
        "mixed_forward_left": (forward_moderate[0], backward_moderate[1]),
        "mixed_backward_right": (backward_moderate[0], forward_moderate[1]),
    }
    for name, hand_target in hand_overrides.items():
        by_name[name] = replace(by_name[name], hand_target=hand_target)
    names = required_tracking_scenario_names(profile)
    return tuple(by_name[name] for name in names)


def _tracking_cfg(*, seed: int, steps: int, perturbation: bool) -> Any:
    cfg = make_microban_teleop_v12_env_cfg(play=True)
    training = make_microban_teleop_v12_env_cfg(play=False)
    moving_hmd = _copy_forced_moving_hmd_neck_event(training)
    _configure_nominal_evaluation(
        cfg, steps=steps + 2, moving_hmd_neck_event=moving_hmd
    )
    cfg.scene.num_envs = 1
    cfg.seed = seed
    cfg.auto_reset = False
    cfg.episode_length_s = (steps + 2) * cfg.decimation * cfg.sim.mujoco.timestep
    if perturbation:
        push = deepcopy(training.events["push_robot"])
        push.interval_range_s = (1.0, 1.0)
        push.params["velocity_range"] = {
            "x": (0.35, 0.35),
            "y": (-0.20, -0.20),
        }
        cfg.events["push_robot"] = push
    return cfg


def _summary(values: torch.Tensor) -> dict[str, list[float]]:
    return {
        "minimum": values[0].tolist(),
        "maximum": values[1].tolist(),
        "absolute_maximum": torch.maximum(values[0].abs(), values[1].abs()).tolist(),
    }


def _active_foot_tracking_error(
    current: torch.Tensor, default: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Return errors only for feet commanded away from exact-zero inactive."""

    per_foot = torch.linalg.vector_norm(current - default - target, dim=-1)
    active = torch.linalg.vector_norm(target, dim=-1).gt(0.0)
    return per_foot[active]


def _target_column_ablation_evidence(
    maximum_by_target: dict[str, float | None],
    expected_by_target: dict[str, bool],
) -> dict[str, dict[str, Any]]:
    """Report actor sensitivity to target positions with hand flags preserved."""

    if set(maximum_by_target) != {"hand", "foot"} or set(expected_by_target) != {
        "hand",
        "foot",
    }:
        raise ValueError("Target-column ablation keys drifted")
    result: dict[str, dict[str, Any]] = {}
    for target in ("hand", "foot"):
        expected = expected_by_target[target]
        maximum = maximum_by_target[target]
        ablated_columns, preserved_columns = target_column_ablation_observation_columns(
            target
        )
        if type(expected) is not bool:
            raise TypeError("Target-column ablation expectation must be boolean")
        if maximum is not None and (
            isinstance(maximum, bool)
            or not isinstance(maximum, (int, float))
            or not math.isfinite(float(maximum))
            or float(maximum) < 0.0
        ):
            raise ValueError("Target-column ablation maximum is invalid")
        if not expected and maximum is not None:
            raise ValueError("Inactive target-column ablation maximum must be absent")
        result[target] = {
            "target_expected": expected,
            "ablated_observation_columns": list(ablated_columns),
            "preserved_observation_columns": list(preserved_columns),
            "maximum_absolute_action_delta": maximum,
            "minimum_required_action_delta": (
                TARGET_COLUMN_ABLATION_ACTION_DELTA_MIN if expected else None
            ),
            "passed": (
                not expected
                or (
                    maximum is not None
                    and maximum > TARGET_COLUMN_ABLATION_ACTION_DELTA_MIN
                )
            ),
        }
    return result


def _target_column_ablation_response_passes(
    results: list[dict[str, Any]], profile: str
) -> bool:
    """Validate exact evidence and enforce responses only after target activation."""

    required_targets = required_target_column_ablation_targets(profile)
    observed_required_targets: set[str] = set()
    evidence_fields = {
        "target_expected",
        "ablated_observation_columns",
        "preserved_observation_columns",
        "maximum_absolute_action_delta",
        "minimum_required_action_delta",
        "passed",
    }
    try:
        for result in results:
            command = result["command"]
            hand_active = command["hand_active"]
            foot_target = command["foot_target"]
            if (
                not isinstance(hand_active, (list, tuple))
                or len(hand_active) != 2
                or any(type(value) is not bool for value in hand_active)
                or not isinstance(foot_target, (list, tuple))
                or len(foot_target) != 2
                or any(
                    not isinstance(xyz, (list, tuple))
                    or len(xyz) != 3
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                        for value in xyz
                    )
                    for xyz in foot_target
                )
            ):
                return False
            expected_by_target = {
                "hand": any(hand_active),
                "foot": any(
                    float(value) != 0.0 for xyz in foot_target for value in xyz
                ),
            }
            ablation = result["target_column_ablation"]
            if not isinstance(ablation, dict) or set(ablation) != {"hand", "foot"}:
                return False
            for target, expected in expected_by_target.items():
                evidence = ablation[target]
                if not isinstance(evidence, dict) or set(evidence) != evidence_fields:
                    return False
                ablated_columns, preserved_columns = (
                    target_column_ablation_observation_columns(target)
                )
                if evidence["target_expected"] is not expected:
                    return False
                if evidence["ablated_observation_columns"] != list(ablated_columns):
                    return False
                if evidence["preserved_observation_columns"] != list(preserved_columns):
                    return False
                maximum = evidence["maximum_absolute_action_delta"]
                if expected:
                    if (
                        isinstance(maximum, bool)
                        or not isinstance(maximum, (int, float))
                        or not math.isfinite(float(maximum))
                        or float(maximum) < 0.0
                        or evidence["minimum_required_action_delta"]
                        != TARGET_COLUMN_ABLATION_ACTION_DELTA_MIN
                    ):
                        return False
                    response = float(maximum) > TARGET_COLUMN_ABLATION_ACTION_DELTA_MIN
                    if evidence["passed"] is not response:
                        return False
                    if target in required_targets:
                        observed_required_targets.add(target)
                        if not response:
                            return False
                elif (
                    maximum is not None
                    or evidence["minimum_required_action_delta"] is not None
                    or evidence["passed"] is not True
                ):
                    return False
    except (KeyError, TypeError):
        return False
    return observed_required_targets == set(required_targets)


def _evaluate_scenario(
    *,
    env: ManagerBasedRlEnv,
    wrapped: RslRlVecEnvWrapper,
    policy: Any,
    source_policy: Any,
    scenario: EvaluationScenario,
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
        raise ValueError("V12 tracking gate requires raw, unclipped actions")
    foot = env.command_manager.get_term("foot_target")
    hand = env.command_manager.get_term("hand_target")
    expects_foot = any(
        abs(value) > 0.0 for xyz in scenario.foot_target for value in xyz
    )
    expects_hand = any(scenario.hand_active)
    hmd_cfg = env.event_manager.get_term_cfg("hmd_neck_target_motion")
    hmd = hmd_cfg.func
    if not isinstance(hmd, HmdNeckTargetMotion):
        raise TypeError("V12 tracking gate requires HmdNeckTargetMotion")
    if tuple(hmd.joint_names) != MICROBAN_HMD_JOINT_NAMES:
        raise ValueError("HMD joint order drifted")
    hmd_stats = HmdMotionStats.start(
        joint_names=tuple(hmd.joint_names),
        target=hmd.current_target[0],
        actual=robot.data.joint_pos[0, hmd.joint_ids],
    )
    actor_terms = env.observation_manager.active_terms["actor"]
    actor_dims = env.observation_manager.group_obs_term_dim["actor"]
    offset = 0
    action_slice: slice | None = None
    for name, shape in zip(actor_terms, actor_dims, strict=True):
        width = math.prod(shape)
        if name == "actions":
            action_slice = slice(offset, offset + width)
        offset += width
    if action_slice is None:
        raise ValueError("Actor observation lacks previous actions")

    action_min = torch.full((3, 18), math.inf, device=env.device)
    action_max = torch.full((3, 18), -math.inf, device=env.device)
    foot_error = ScalarStats()
    hand_error = ScalarStats()
    velocity_stats = (ScalarStats(), ScalarStats(), ScalarStats())
    executed = 0
    raw_recurrence_steps = 0
    fell = False
    nonfinite: dict[str, Any] | None = None
    max_actual_violation = 0.0
    max_raw_target_excess = 0.0
    foot_observation_nonzero_steps = 0
    hand_observation_nonzero_steps = 0
    hmd_observation_nonzero_steps = 0
    target_ablation_maximum: dict[str, float | None] = {
        "hand": None,
        "foot": None,
    }
    termination_names: list[str] = []
    fall_height = float(
        env.termination_manager.get_term_cfg("fell_over").params["minimum_height"]
    )
    scale = torch.as_tensor(action_term.scale, device=env.device)
    offset_value = torch.as_tensor(action_term.offset, device=env.device)
    all_limits = robot.data.soft_joint_pos_limits
    action_limits = all_limits[:, action_term.target_ids]

    for step in range(steps):
        actor_obs = observations["actor"]
        if not bool(torch.isfinite(actor_obs).all().item()):
            nonfinite = {"step": step, "phase": "observation"}
            break
        foot_observation_nonzero_steps += int(
            bool(torch.count_nonzero(actor_obs[:, 69:75]).item())
        )
        hand_observation_nonzero_steps += int(
            bool(torch.count_nonzero(actor_obs[:, 75:83]).item())
        )
        hmd_observation_nonzero_steps += int(
            bool(
                torch.count_nonzero(actor_obs[:, (*range(6, 9), *range(27, 30))]).item()
            )
        )
        legacy_obs = actor_obs[
            :, [target for _source, target in LEGACY_TO_TELEOP_OBSERVATION_INDEX]
        ]
        with torch.inference_mode():
            actions = policy(TensorDict({"actor": actor_obs}, batch_size=[1]))
            source_actions = source_policy(
                TensorDict({"actor": legacy_obs}, batch_size=[1])
            )
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
        delta = actions - source_actions
        if not all(
            bool(torch.isfinite(value).all().item())
            for value in (actions, source_actions, delta, *ablated_actions.values())
        ):
            nonfinite = {"step": step, "phase": "policy"}
            break
        for target, ablated in ablated_actions.items():
            value = float(torch.max(torch.abs(actions - ablated)).item())
            previous = target_ablation_maximum[target]
            target_ablation_maximum[target] = (
                value if previous is None else max(previous, value)
            )
        for index, value in enumerate((actions, source_actions, delta)):
            action_min[index] = torch.minimum(action_min[index], value[0])
            action_max[index] = torch.maximum(action_max[index], value[0])
        raw_target = actions * scale + offset_value
        excess = torch.maximum(
            torch.clamp(action_limits[..., 0] - raw_target, min=0.0),
            torch.clamp(raw_target - action_limits[..., 1], min=0.0),
        )
        max_raw_target_excess = max(max_raw_target_excess, float(excess.max().item()))

        observations, rewards, dones, _extras = wrapped.step(actions)
        executed += 1
        if not torch.equal(action_term.raw_action, actions):
            raise RuntimeError("Action manager modified raw v12 action")
        if not torch.equal(observations["actor"][:, action_slice], actions):
            raise RuntimeError("Previous-action observation is not raw output")
        raw_recurrence_steps += 1
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
        violation = torch.maximum(
            torch.clamp(all_limits[..., 0] - robot.data.joint_pos, min=0.0),
            torch.clamp(robot.data.joint_pos - all_limits[..., 1], min=0.0),
        )
        max_actual_violation = max(max_actual_violation, float(violation.max().item()))
        root_height = float(robot.data.root_link_pos_w[0, 2].item())
        fell = (
            fell
            or bool(env.termination_manager.get_term("fell_over")[0].item())
            or root_height < fall_height
        )
        if step >= settle_steps:
            current_foot = foot.current_foot_pos_b()
            foot_error.add(
                _active_foot_tracking_error(
                    current_foot,
                    foot._default_foot_pos_b,
                    foot.foot_target_offset_b,
                )
            )
            current_hand = hand.current_hand_pos_b()
            per_hand = torch.linalg.vector_norm(
                current_hand - hand._default_hand_pos_b - hand.hand_target_offset_b,
                dim=-1,
            )
            hand_error.add(per_hand[hand.is_active])
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

    hmd_report = hmd_stats.report(step_dt=env.step_dt)
    hmd_motion = all(
        float(axis["target_peak_to_peak_rad"]) >= HMD_TARGET_PEAK_TO_PEAK_MIN_RAD
        and float(axis["actual_peak_to_peak_rad"]) >= HMD_ACTUAL_PEAK_TO_PEAK_MIN_RAD
        for axis in hmd_report["per_axis"].values()
    )
    completed = (
        executed == steps and not fell and nonfinite is None and not termination_names
    )
    axis_names = ("vx_m_s", "vy_m_s", "yaw_rad_s")
    measured_velocity = {
        name: stats.report(units="rad_s" if index == 2 else "m_s")
        for index, (name, stats) in enumerate(
            zip(axis_names, velocity_stats, strict=True)
        )
    }
    directional_response: dict[str, dict[str, float | bool]] = {}
    for axis, command in zip(axis_names, scenario.twist, strict=True):
        if command == 0.0:
            continue
        measured_mean = measured_velocity[axis]["mean"]
        signed = (
            None
            if measured_mean is None
            else float(measured_mean) * (1.0 if command > 0.0 else -1.0)
        )
        directional_response[axis] = {
            "command": command,
            "measured_mean": measured_mean,
            "signed_response": signed,
            "minimum_signed_response": DIRECTIONAL_RESPONSE_MINIMUM[axis],
            "passed": (
                signed is not None and signed >= DIRECTIONAL_RESPONSE_MINIMUM[axis]
            ),
        }
    directional_response_passed = all(
        value["passed"] is True for value in directional_response.values()
    )
    target_column_ablation = _target_column_ablation_evidence(
        target_ablation_maximum,
        {"hand": expects_hand, "foot": expects_foot},
    )
    return {
        "name": scenario.name,
        "command": {
            "twist": list(scenario.twist),
            "foot_target": [list(value) for value in scenario.foot_target],
            "hand_target": [list(value) for value in scenario.hand_target],
            "hand_active": list(scenario.hand_active),
        },
        "completed": completed,
        "executed_steps": executed,
        "fell": fell,
        "nonfinite": nonfinite,
        "termination_names": termination_names,
        "maximum_actual_soft_limit_violation_rad": max_actual_violation,
        "maximum_hypothetical_raw_target_soft_limit_excess_rad": (
            max_raw_target_excess
        ),
        "raw_action_recurrence_verified_steps": raw_recurrence_steps,
        "observation_coverage": {
            "hmd_nonzero_steps": hmd_observation_nonzero_steps,
            "foot_nonzero_steps": foot_observation_nonzero_steps,
            "hand_nonzero_steps": hand_observation_nonzero_steps,
            "foot_target_expected": expects_foot,
            "hand_target_expected": expects_hand,
            "passed": (
                hmd_observation_nonzero_steps > 0
                and (not expects_foot or foot_observation_nonzero_steps == executed)
                and (not expects_hand or hand_observation_nonzero_steps == executed)
            ),
        },
        "measured_velocity_body": measured_velocity,
        "directional_response": directional_response,
        "twist_directional_response_passed": directional_response_passed,
        "hmd_motion": hmd_report,
        "hmd_motion_evidence_passed": hmd_motion,
        "target_error": {
            "foot": foot_error.report(units="m"),
            "active_hand": hand_error.report(units="m"),
        },
        "target_column_ablation": target_column_ablation,
        "raw_action_envelope": {
            "joint_names": list(MICROBAN_TELEOP_ACTION_JOINT_NAMES),
            "v12": _summary(torch.stack((action_min[0], action_max[0]))),
            "legacy_source": _summary(torch.stack((action_min[1], action_max[1]))),
            "learned_minus_source": _summary(
                torch.stack((action_min[2], action_max[2]))
            ),
        },
    }


def _acceptance(
    results: list[dict[str, Any]], profile: str
) -> tuple[dict[str, bool], str]:
    checks = {
        "all_scenarios_completed": all(item["completed"] for item in results),
        "no_falls": not any(item["fell"] for item in results),
        "finite": all(item["nonfinite"] is None for item in results),
        "actual_soft_limits": all(
            0.0
            <= float(item["maximum_actual_soft_limit_violation_rad"])
            <= ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD
            for item in results
        ),
        "raw_action_recurrence": all(
            item["raw_action_recurrence_verified_steps"] == item["executed_steps"]
            for item in results
        ),
        "forced_hmd_motion": all(
            item["hmd_motion_evidence_passed"] for item in results
        ),
        "nonzero_observation_coverage": all(
            item["observation_coverage"]["passed"] for item in results
        ),
        "target_column_ablation_response": (
            _target_column_ablation_response_passes(results, profile)
        ),
        "twist_directional_response": all(
            item["twist_directional_response_passed"] for item in results
        ),
    }
    if profile in (
        HMD_HAND_PROFILE,
        DEADLINE_FALLBACK_PROFILE,
        DEADLINE_CANARY_FALLBACK_PROFILE,
        DEADLINE_FINAL_FALLBACK_PROFILE,
        FOOT_ACTIVATION_CANARY_PROFILE,
        WHOLE_BODY_PROFILE,
        FINAL_PROFILE,
    ):
        active_hand = [
            item["target_error"]["active_hand"]
            for item in results
            if item["target_error"]["active_hand"]["sample_count"]
        ]
        checks["hand_tracking_rms"] = bool(active_hand) and all(
            float(value["rms"]) <= hand_tracking_rms_max_m(profile)
            for value in active_hand
        )
        checks["hand_tracking_p95"] = bool(active_hand) and all(
            float(value["p95"]) <= hand_tracking_p95_max_m(profile)
            for value in active_hand
        )
    if profile in (WHOLE_BODY_PROFILE, FINAL_PROFILE, DEADLINE_FINAL_FALLBACK_PROFILE):
        active_foot = [
            item["target_error"]["foot"]
            for item in results
            if any(
                abs(value) > 0.0
                for xyz in item["command"]["foot_target"]
                for value in xyz
            )
        ]
        checks["foot_tracking_rms"] = bool(active_foot) and all(
            float(value["rms"]) <= foot_tracking_rms_max_m(profile)
            for value in active_foot
        )
        checks["foot_tracking_p95"] = bool(active_foot) and all(
            float(value["p95"]) <= foot_tracking_p95_max_m(profile)
            for value in active_foot
        )
    return checks, "pass" if all(checks.values()) else "fail"


def _aggregate_action_envelopes(results: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {
        "joint_names": list(MICROBAN_TELEOP_ACTION_JOINT_NAMES),
        "scenario_count": len(results),
        "step_count": sum(int(item["executed_steps"]) for item in results),
    }
    for policy_name in ("v12", "legacy_source", "learned_minus_source"):
        minimum = torch.tensor(
            [item["raw_action_envelope"][policy_name]["minimum"] for item in results]
        ).amin(dim=0)
        maximum = torch.tensor(
            [item["raw_action_envelope"][policy_name]["maximum"] for item in results]
        ).amax(dim=0)
        aggregate[policy_name] = _summary(torch.stack((minimum, maximum)))
    return aggregate


def run_evaluation(
    *,
    checkpoint: Path,
    expected_sha256: str | None,
    profile: str | None,
    device: str,
    seed: int,
    steps: int,
    settle_steps: int,
    allow_nondeployable_preview: bool = False,
    allow_legacy_preview_v1: bool = False,
    allow_corner_rescue: bool = False,
    allow_deadline_fallback: bool = False,
    allow_deadline_canary_fallback: bool = False,
) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve()
    digest = sha256_file(checkpoint)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"Checkpoint SHA-256 mismatch: {digest}")
    if seed != 42 or steps != 300 or settle_steps != 50:
        raise ValueError("Canonical tracking gate requires seed42/300/settle50")
    if allow_deadline_fallback and allow_deadline_canary_fallback:
        raise ValueError("Deadline source and canary fallback modes are exclusive")
    if allow_deadline_canary_fallback and digest != (
        MICROBAN_TELEOP_V12_DEADLINE_CANARY_CHECKPOINT_SHA256
    ):
        raise ValueError("Deadline canary checkpoint SHA-256 mismatch")
    configure_torch_backends(allow_tf32=False, deterministic=True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    policy, iteration, infos = _load_actor(
        checkpoint,
        device=device,
        allow_nondeployable_preview=allow_nondeployable_preview,
        allow_legacy_preview_v1=allow_legacy_preview_v1,
        allow_corner_rescue=allow_corner_rescue,
        allow_deadline_fallback=allow_deadline_fallback,
    )
    completed = iteration + 1
    if allow_deadline_fallback:
        required = DEADLINE_FALLBACK_PROFILE
    elif allow_deadline_canary_fallback:
        required = DEADLINE_CANARY_FALLBACK_PROFILE
    elif infos.get(MICROBAN_TELEOP_V12_DEADLINE_POST_CANARY_INFO_KEY) is not None:
        required = DEADLINE_FINAL_FALLBACK_PROFILE
    elif allow_corner_rescue:
        required = HMD_HAND_PROFILE
    elif allow_nondeployable_preview:
        marker = infos.get(TELEOP_V12_PREVIEW_INFO_KEY, {})
        required = (
            HMD_HAND_PROFILE
            if marker.get("phase") == TELEOP_V12_PREVIEW_PHASE_HMD_HAND
            else FINAL_PROFILE
        )
        if marker.get("revision") == TELEOP_V12_PREVIEW_LEGACY_REVISION:
            required = FINAL_PROFILE
    else:
        required = required_tracking_profile(completed)
    if profile is not None and profile != required:
        raise ValueError(f"Checkpoint requires tracking profile {required}")
    profile = required
    _source_identity, source_state = inspect_legacy_velocity_checkpoint(
        "repo://checkpoints/xc330_velocity/model_14999.pt",
        LEGACY_VELOCITY_CHECKPOINT_SHA256,
    )
    source_policy = _legacy_model().to(device)
    source_policy.load_state_dict(source_state, strict=True)
    source_policy.eval()

    perturbation = profile in (
        EXPANDED_LOCOMOTION_PROFILE,
        FINAL_PROFILE,
        DEADLINE_FINAL_FALLBACK_PROFILE,
    )
    cfg = _tracking_cfg(seed=seed, steps=steps, perturbation=perturbation)
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
    try:
        validate_teleop_v12_environment_contract(wrapped)
        results = []
        for scenario in _scenarios(profile):
            print(f"[INFO] v12 tracking {profile}: {scenario.name}", flush=True)
            results.append(
                _evaluate_scenario(
                    env=env,
                    wrapped=wrapped,
                    policy=policy,
                    source_policy=source_policy,
                    scenario=scenario,
                    seed=seed,
                    steps=steps,
                    settle_steps=settle_steps,
                )
            )
    finally:
        wrapped.close()
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    checks, status = _acceptance(results, profile)
    return {
        "schema_version": 1,
        "gate": "microban_teleop_v12_tracking",
        "profile": profile,
        "status": status,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": digest,
            "iteration": iteration,
            "completed_updates": completed,
        },
        "settings": {
            "device": device,
            "seed": seed,
            "steps": steps,
            "settle_steps": settle_steps,
            "moving_hmd": "forced_non_neutral",
            "perturbation": perturbation,
            "action_clip": None,
            "previous_action": "raw_actor_output",
            "target_column_ablation": TARGET_COLUMN_ABLATION_METHOD,
            "reachable_hand_target_fk": microban_hand_fk_metadata(),
        },
        "thresholds": {
            "actual_soft_limit_violation_rad_max": (
                ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD
            ),
            "hmd_target_peak_to_peak_rad_min": HMD_TARGET_PEAK_TO_PEAK_MIN_RAD,
            "hmd_actual_peak_to_peak_rad_min": HMD_ACTUAL_PEAK_TO_PEAK_MIN_RAD,
            "hand_rms_m_max": hand_tracking_rms_max_m(profile),
            "hand_p95_m_max": hand_tracking_p95_max_m(profile),
            "foot_rms_m_max": foot_tracking_rms_max_m(profile),
            "foot_p95_m_max": foot_tracking_p95_max_m(profile),
            "directional_response_minimum": DIRECTIONAL_RESPONSE_MINIMUM,
            "target_column_ablation_action_delta_min": (
                TARGET_COLUMN_ABLATION_ACTION_DELTA_MIN
            ),
            "raw_action_amplitude": "reported_finite_only_no_invented_threshold",
        },
        "checks": checks,
        "raw_action_envelope": _aggregate_action_envelopes(results),
        "results": results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--profile", choices=TRACKING_PROFILES)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--settle-steps", type=int, default=50)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--allow-corner-rescue",
        action="store_true",
        help="require the authenticated final model9999 corner-rescue checkpoint",
    )
    parser.add_argument(
        "--deadline-fallback",
        action="store_true",
        help=(
            "accept only the hash-pinned v1 checkpoint under the explicit "
            "35mm hand-RMS deadline profile"
        ),
    )
    parser.add_argument(
        "--deadline-canary-fallback",
        action="store_true",
        help=(
            "accept only the hash-pinned model10099 foot canary under the "
            "35mm hand-RMS profile"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_evaluation(
        checkpoint=args.checkpoint,
        expected_sha256=args.expected_sha256,
        profile=args.profile,
        device=args.device,
        seed=args.seed,
        steps=args.steps,
        settle_steps=args.settle_steps,
        allow_corner_rescue=args.allow_corner_rescue,
        allow_deadline_fallback=args.deadline_fallback,
        allow_deadline_canary_fallback=args.deadline_canary_fallback,
    )
    if args.output is not None:
        if args.output.expanduser().exists() and not args.force:
            raise FileExistsError(f"Output exists (pass --force): {args.output}")
        publish_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
