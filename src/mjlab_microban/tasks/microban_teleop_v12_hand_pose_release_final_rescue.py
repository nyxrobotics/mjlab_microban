"""Pose-release final-scenario rescue for the 15000 contract-v12 boundary.

The forward-lean active-hand arm pose-release chain (``lean_v12_pr_*``) trains
10100 -> 15000 as one segment.  Its model_14999 can fail the unchanged final
profile (``full_body_reachable_performance_perturbation_v2_completion_
allowance_v1``) on checks of individual evaluator scenarios: on 2026-10-05 two
retrains with the same seed failed only ``actual_soft_limits``
(bounded_both_feet 0.0921 rad, mixed_forward_left 0.0920 rad > 0.0873) and
``twist_directional_response`` (mixed_forward_left lateral -0.004 m/s < 0.02)
while hand/foot accuracy and locomotion passed.

This rescue is the canonical final rescue's 99-update replay
(``microban_teleop_v12_final_rescue``: model_14900 -> model_14999, completed
14901 -> 15000) for that lineage:

* parent: the unmarked pose-release ``model_14900`` of the same run as the
  failed model_14999 (a fresh chain, or a fresh chain through the pose-release
  model9900 corner rescue), exact clock, columns and Adam step;
* failed gate: that run's model_14999 tracking report under the unchanged
  final pose-release profile, failing at least one of the accuracy checks,
  ``actual_soft_limits`` or ``twist_directional_response`` and nothing else;
  the scenarios that failed are recorded and every one of them must be
  replayed by the selected mix;
* env: the pose-release env with only the twist, foot and hand samplers
  changed: per episode, a registered share of environments replays one failed
  evaluator scenario exactly (twist, both foot targets, both hand targets and
  hand-active flags), the rest keeps the ordinary samplers (30-70 %);
* saves keep the pose-release recipe revision and carry a pose-release final
  rescue marker under the final-rescue infos key, naming the parent, the
  failed gate and the inherited corner marker.  Every consumer rebuilds the
  marker from its recorded values, and only the rescue's model_14999 is
  consumable (``microban_teleop_v12_hand_pose_release_lineage``).

Every gate profile, threshold and check is unchanged: the rescue's model_14999
is judged by the ordinary 14999 stage gate under the same final profile.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from mjlab_microban.robot import home_contracts
from mjlab_microban.tasks.microban_teleop_env_cfg import (
    MICROBAN_TELEOP_FOOT_TRACKING_FINAL_STD_M,
    MICROBAN_TELEOP_HAND_TRACKING_FINAL_STD_M,
)
from mjlab_microban.tasks.microban_teleop_v12_actor import (
    TELEOP_V12_ADAPTER_GRADIENT_SCHEDULE_REVISION,
)
from mjlab_microban.tasks.microban_teleop_v12_corner_rescue import (
    _require_sha256,
)
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MICROBAN_TELEOP_V12_ACTION_CLIP,
    MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION,
)
from mjlab_microban.tasks.microban_teleop_v12_final_rescue import (
    MICROBAN_TELEOP_V12_FINAL_RESCUE_ACTIVE_COLUMNS,
    MICROBAN_TELEOP_V12_FINAL_RESCUE_INFO_KEY,
    MICROBAN_TELEOP_V12_FINAL_RESCUE_PARENT_COMMON_STEP,
    MICROBAN_TELEOP_V12_FINAL_RESCUE_PARENT_COMPLETED_UPDATES,
    MICROBAN_TELEOP_V12_FINAL_RESCUE_PARENT_ITERATION,
    MICROBAN_TELEOP_V12_FINAL_RESCUE_PARENT_OPTIMIZER_STEP,
    MICROBAN_TELEOP_V12_FINAL_RESCUE_PROCESS_UPDATES,
    MICROBAN_TELEOP_V12_FINAL_RESCUE_TARGET_COMMON_STEP,
    MICROBAN_TELEOP_V12_FINAL_RESCUE_TARGET_COMPLETED_UPDATES,
    MICROBAN_TELEOP_V12_FINAL_RESCUE_TARGET_ITERATION,
    MICROBAN_TELEOP_V12_FINAL_RESCUE_TARGET_OPTIMIZER_STEP,
    FinalRescueFootTargetCommandCfg,
    FinalRescueHandTargetCommandCfg,
    FinalRescueTwistCommandCfg,
    _rebased_cfg,
    evaluator_scenario_commands,
    final_rescue_pattern_state,
)

MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_TASK_ID = (
    "Mjlab-Teleop-V12-HandPoseRelease-Final-Rescue-Microban"
)
MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_MARKER_REVISION = (
    home_contracts.V12_PR_FINAL_RESCUE_MARKER_REVISION
)
MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_SAMPLER_REVISION = (
    home_contracts.V12_PR_FINAL_RESCUE_SAMPLER_REVISION
)
MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_MIX_ENV = (
    "MICROBAN_V12_PR_FINAL_RESCUE_MIX"
)
MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_DEFAULT_MIX = "pr_v1"
# The launcher stages the failed gate report next to the staged parent.
MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_FAILED_GATE_REPORT_FILENAME = (
    "failed_final_gate_tracking.json"
)
MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_SEED_PREFIX = (
    "pr_final_rescue_seed_"
)
# ... and the parent run's directory name (one line), so every load re-checks
# that the failed report's model_14999 is from the parent's own run.
MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_PARENT_RUN_FILENAME = (
    "parent_run.txt"
)

# Registered mixes: ordinary share first, then the replayed scenarios in
# sampler order.  pr_v1-pr_v3 replay the two scenarios that failed both
# 2026-10-05 lean final gates (mixed_forward_left: soft limits + lateral twist;
# bounded_both_feet: soft limits) with 50 / 70 / 30 % ordinary commands (the
# centered v1 rescue at 10 % ordinary lost push robustness).  pr_v4 adds
# mixed_backward_right for a failed gate that also fails there.
MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_MIXES: dict[
    str, dict[str, float]
] = {
    "pr_v1": {
        "ordinary": 0.50,
        "mixed_forward_left": 0.30,
        "bounded_both_feet": 0.20,
    },
    "pr_v2": {
        "ordinary": 0.70,
        "mixed_forward_left": 0.20,
        "bounded_both_feet": 0.10,
    },
    "pr_v3": {
        "ordinary": 0.30,
        "mixed_forward_left": 0.45,
        "bounded_both_feet": 0.25,
    },
    "pr_v4": {
        "ordinary": 0.50,
        "mixed_forward_left": 0.25,
        "bounded_both_feet": 0.15,
        "mixed_backward_right": 0.10,
    },
    # pr_v5/pr_v6: the pr_v1/pr_v2 shares, and every replayed episode also
    # gets the evaluator's perturbation instead of the ordinary random push
    # (see MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_SCENARIO_PUSH).
    # On 2026-10-06 pr_v1/pr_v2 rescues turned the lateral response positive
    # without that push (+0.155 / +0.026 m/s) but fell on mixed_forward_left
    # under it.
    "pr_v5": {
        "ordinary": 0.50,
        "mixed_forward_left": 0.30,
        "bounded_both_feet": 0.20,
    },
    "pr_v6": {
        "ordinary": 0.70,
        "mixed_forward_left": 0.20,
        "bounded_both_feet": 0.10,
    },
}
# Mixes whose replayed episodes take the evaluator's fixed push (ordinary
# episodes keep the ordinary random push of the staged curriculum).
MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_SCENARIO_PUSH_MIXES = frozenset(
    ("pr_v5", "pr_v6")
)
# The final profile's perturbation exactly as the tracking evaluator builds it
# (``evaluate_teleop_v12_tracking._tracking_cfg(perturbation=True)``: the
# training push_robot term with a 1.0 s interval and a fixed world-frame
# velocity kick); a test pins it to the evaluator.
MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_SCENARIO_PUSH: dict[str, Any] = {
    "interval_range_s": (1.0, 1.0),
    "velocity_range": {"x": (0.35, 0.35), "y": (-0.20, -0.20)},
}
MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_SCENARIO_PUSH_EVENT = (
    "final_rescue_scenario_push"
)
MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_MIN_ORDINARY_PROBABILITY = 0.30
# The failed final gate may fail only these checks (accuracy or the two
# per-scenario safety/response checks the 2026-10-05 lean gates failed); every
# other check (completion, falls, finiteness, recurrence, HMD motion, coverage,
# ablation) must pass.
MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_RESCUABLE_CHECKS = frozenset(
    (
        "hand_tracking_rms",
        "hand_tracking_p95",
        "foot_tracking_rms",
        "foot_tracking_p95",
        "actual_soft_limits",
        "twist_directional_response",
    )
)

for _name, _mix in MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_MIXES.items():
    if (
        not _name.startswith("pr_")
        or next(iter(_mix)) != "ordinary"
        or len(_mix) < 2
        or not math.isclose(sum(_mix.values()), 1.0)
        or any(value <= 0.0 for value in _mix.values())
        or _mix["ordinary"]
        < MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_MIN_ORDINARY_PROBABILITY
    ):
        raise RuntimeError(f"Pose-release final rescue mix {_name} is malformed")


def validate_hand_pose_release_final_rescue_mix(name: object) -> str:
    if (
        not isinstance(name, str)
        or name not in MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_MIXES
    ):
        raise ValueError(
            "Pose-release final rescue mix must be one of "
            f"{sorted(MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_MIXES)}: "
            f"{name!r}"
        )
    return name


def hand_pose_release_final_rescue_scenarios(name: str) -> tuple[str, ...]:
    mix = MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_MIXES[
        validate_hand_pose_release_final_rescue_mix(name)
    ]
    return tuple(key for key in mix if key != "ordinary")


def hand_pose_release_final_rescue_mix_probabilities(name: str) -> dict[str, float]:
    return dict(
        MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_MIXES[
            validate_hand_pose_release_final_rescue_mix(name)
        ]
    )


def hand_pose_release_final_rescue_uses_scenario_push(name: str) -> bool:
    return (
        validate_hand_pose_release_final_rescue_mix(name)
        in MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_SCENARIO_PUSH_MIXES
    )


def _scenario_push_record() -> dict[str, Any]:
    push = MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_SCENARIO_PUSH
    return {
        "source": "final_profile_tracking_evaluator_perturbation",
        "scope": "replayed_episodes_only",
        "interval_range_s": list(push["interval_range_s"]),
        "velocity_range": {
            axis: list(bounds) for axis, bounds in push["velocity_range"].items()
        },
        "ordinary_push": "staged_curriculum_random_push_on_ordinary_episodes_only",
    }


def push_ordinary_episodes(
    env: Any,
    env_ids: Any,
    velocity_range: dict[str, tuple[float, float]],
    final_rescue_mix: str,
) -> None:
    """The ordinary push_robot term, skipping replayed episodes."""

    from mjlab.envs.mdp import push_by_setting_velocity

    state = final_rescue_pattern_state(env, final_rescue_mix)
    ids = _as_env_ids(env, env_ids)
    ids = ids[state.pattern[ids] == 0]
    if len(ids) > 0:
        push_by_setting_velocity(env, ids, velocity_range)


def push_replayed_episodes(
    env: Any,
    env_ids: Any,
    velocity_range: dict[str, tuple[float, float]],
    final_rescue_mix: str,
) -> None:
    """The evaluator's fixed push, applied only to replayed episodes."""

    from mjlab.envs.mdp import push_by_setting_velocity

    state = final_rescue_pattern_state(env, final_rescue_mix)
    ids = _as_env_ids(env, env_ids)
    ids = ids[state.pattern[ids] != 0]
    if len(ids) > 0:
        push_by_setting_velocity(env, ids, velocity_range)


def _as_env_ids(env: Any, env_ids: Any) -> Any:
    import torch

    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(int(env.num_envs), device=env.device)
    return env_ids


def hand_pose_release_final_rescue_sampler_spec(
    name: str,
) -> tuple[tuple[str, ...], dict[str, float]]:
    scenarios = hand_pose_release_final_rescue_scenarios(name)
    # Builds (and checks against the evaluator) every replayed command.
    evaluator_scenario_commands(scenarios)
    return scenarios, hand_pose_release_final_rescue_mix_probabilities(name)


def final_gate_profile() -> str:
    """The final profile at 15000 (the same for every lineage)."""

    from mjlab_microban.scripts.evaluate_teleop_v12_tracking import (
        required_tracking_profile,
    )

    return required_tracking_profile(
        MICROBAN_TELEOP_V12_FINAL_RESCUE_TARGET_COMPLETED_UPDATES
    )


def failed_final_gate_scenarios(report: Mapping[str, Any]) -> list[str]:
    """Scenarios of a final tracking report that fail any per-scenario limit.

    Uses the report's own profile limits (soft-limit overshoot, twist
    directional response, hand/foot RMS and P95); the order is the evaluator's.
    """

    from mjlab_microban.scripts.evaluate_teleop_v12_tracking import (
        ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD,
        foot_tracking_p95_max_m,
        foot_tracking_rms_max_m,
        hand_tracking_p95_max_m,
        hand_tracking_rms_max_m,
    )

    profile = report["profile"]
    failed: list[str] = []
    for item in report["results"]:
        hand = item["target_error"]["active_hand"]
        foot = item["target_error"]["foot"]
        accuracy_failed = (
            bool(hand.get("sample_count"))
            and (
                float(hand["rms"]) > hand_tracking_rms_max_m(profile)
                or float(hand["p95"]) > hand_tracking_p95_max_m(profile)
            )
        ) or (
            bool(foot.get("sample_count"))
            and (
                float(foot["rms"]) > foot_tracking_rms_max_m(profile)
                or float(foot["p95"]) > foot_tracking_p95_max_m(profile)
            )
        )
        if (
            accuracy_failed
            or float(item["maximum_actual_soft_limit_violation_rad"])
            > ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD
            or item["twist_directional_response_passed"] is not True
        ):
            failed.append(str(item["name"]))
    return failed


def _failed_checks(value: object) -> list[str]:
    if (
        not isinstance(value, (list, tuple))
        or not value
        or any(not isinstance(item, str) for item in value)
        or list(value) != sorted(set(value))
        or not set(value)
        <= MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_RESCUABLE_CHECKS
    ):
        raise ValueError(
            "The failed final gate must fail a sorted non-empty subset of "
            f"{sorted(MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_RESCUABLE_CHECKS)}"
        )
    return list(value)


def _failed_scenarios(value: object, *, mix: str) -> list[str]:
    from mjlab_microban.scripts.evaluate_teleop_v12_tracking import (
        FINAL_PROFILE,
        required_tracking_scenario_names,
    )

    order = required_tracking_scenario_names(FINAL_PROFILE)
    if (
        not isinstance(value, (list, tuple))
        or not value
        or any(not isinstance(item, str) or item not in order for item in value)
        or list(value) != [name for name in order if name in value]
    ):
        raise ValueError("Failed final gate scenarios must be ordered evaluator names")
    replayed = hand_pose_release_final_rescue_scenarios(mix)
    missing = [name for name in value if name not in replayed]
    if missing:
        raise ValueError(
            f"Pose-release final rescue mix {mix} does not replay the failed "
            f"final-gate scenarios {missing}"
        )
    return list(value)


def _training_seed(value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value < 2**31
    ):
        raise ValueError("Pose-release final rescue training seed is malformed")
    return value


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Pose-release final rescue {label} is malformed")
    return value


def _parent_lineage_names() -> tuple[str, str]:
    from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release_lineage import (
        HAND_POSE_RELEASE_LINEAGE_FRESH,
        HAND_POSE_RELEASE_LINEAGE_FRESH_CORNER_RESCUE,
    )

    return HAND_POSE_RELEASE_LINEAGE_FRESH, HAND_POSE_RELEASE_LINEAGE_FRESH_CORNER_RESCUE


def hand_pose_release_final_rescue_marker(
    *,
    parent_checkpoint_sha256: str,
    failed_gate_checkpoint_sha256: str,
    failed_gate_tracking_report_sha256: str,
    failed_gate_failed_checks: Sequence[str],
    failed_gate_failed_scenarios: Sequence[str],
    inherited_corner_rescue_marker_sha256: str | None,
    sampler_mix: str,
    training_seed: int = 42,
    num_envs: int = 2_048,
) -> dict[str, Any]:
    """Return the marker embedded in every pose-release final-rescue save.

    Only the recorded hashes, failed checks/scenarios, the mix, the training
    seed (environment = agent seed, as the train CLI sets them) and the live
    environment count (2048 from the launcher) vary; every
    other field is fixed by code.  The parent lineage follows from the
    inherited corner marker (fresh chain, or fresh chain through the
    pose-release model9900 corner rescue).
    """

    fresh, fresh_corner = _parent_lineage_names()
    mix = validate_hand_pose_release_final_rescue_mix(sampler_mix)
    if parent_checkpoint_sha256 == failed_gate_checkpoint_sha256:
        raise ValueError("The failed final gate cannot name the parent itself")
    scenarios = hand_pose_release_final_rescue_scenarios(mix)
    return {
        "schema_version": 1,
        "revision": MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_MARKER_REVISION,
        "parent_identity": "recorded_and_rehashed_on_load",
        "parent_checkpoint_sha256": _require_sha256(
            parent_checkpoint_sha256, "Pose-release final rescue parent checkpoint"
        ),
        "parent_lineage": (
            fresh if inherited_corner_rescue_marker_sha256 is None else fresh_corner
        ),
        "failed_final_gate": {
            "checkpoint_sha256": _require_sha256(
                failed_gate_checkpoint_sha256, "Failed final gate checkpoint"
            ),
            "same_run_as_parent": True,
            "iteration": MICROBAN_TELEOP_V12_FINAL_RESCUE_TARGET_ITERATION,
            "completed_updates": (
                MICROBAN_TELEOP_V12_FINAL_RESCUE_TARGET_COMPLETED_UPDATES
            ),
            "tracking_profile": final_gate_profile(),
            "tracking_report_sha256": _require_sha256(
                failed_gate_tracking_report_sha256, "Failed final gate report"
            ),
            "failed_checks": _failed_checks(failed_gate_failed_checks),
            "failed_scenarios": _failed_scenarios(
                failed_gate_failed_scenarios, mix=mix
            ),
        },
        "inherited_corner_rescue_marker_sha256": (
            None
            if inherited_corner_rescue_marker_sha256 is None
            else _require_sha256(
                inherited_corner_rescue_marker_sha256, "Inherited corner marker"
            )
        ),
        "parent_iteration": MICROBAN_TELEOP_V12_FINAL_RESCUE_PARENT_ITERATION,
        "parent_completed_updates": (
            MICROBAN_TELEOP_V12_FINAL_RESCUE_PARENT_COMPLETED_UPDATES
        ),
        "parent_common_step_counter": (
            MICROBAN_TELEOP_V12_FINAL_RESCUE_PARENT_COMMON_STEP
        ),
        "target_iteration": MICROBAN_TELEOP_V12_FINAL_RESCUE_TARGET_ITERATION,
        "target_completed_updates": (
            MICROBAN_TELEOP_V12_FINAL_RESCUE_TARGET_COMPLETED_UPDATES
        ),
        "target_common_step_counter": (
            MICROBAN_TELEOP_V12_FINAL_RESCUE_TARGET_COMMON_STEP
        ),
        "process_updates": MICROBAN_TELEOP_V12_FINAL_RESCUE_PROCESS_UPDATES,
        "optimizer_step": {
            "parent": MICROBAN_TELEOP_V12_FINAL_RESCUE_PARENT_OPTIMIZER_STEP,
            "target": MICROBAN_TELEOP_V12_FINAL_RESCUE_TARGET_OPTIMIZER_STEP,
        },
        "training": {
            "environment_seed": _training_seed(training_seed),
            "agent_seed": _training_seed(training_seed),
            "num_envs": _positive_int(num_envs, "environment count"),
            "num_steps_per_env": 24,
        },
        "source_recipe_revision": MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION,
        # Saves keep the pose-release recipe; this marker names the replay.
        "rescue_recipe_revision": MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION,
        "rescue_task_id": MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_TASK_ID,
        "sampler_revision": (
            MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_SAMPLER_REVISION
        ),
        "sampler_mix": mix,
        "sampler_scenarios": list(scenarios),
        "sampler_probabilities": hand_pose_release_final_rescue_mix_probabilities(mix),
        "sampler_scope": "per_episode_shared_by_twist_foot_hand_commands",
        "scenario_commands": deepcopy(evaluator_scenario_commands(scenarios)),
        **(
            {"scenario_push": _scenario_push_record()}
            if hand_pose_release_final_rescue_uses_scenario_push(mix)
            else {}
        ),
        "adapter_gradient_schedule_revision": (
            TELEOP_V12_ADAPTER_GRADIENT_SCHEDULE_REVISION
        ),
        "active_actor_columns": list(MICROBAN_TELEOP_V12_FINAL_RESCUE_ACTIVE_COLUMNS),
        "gate_contract": "unchanged_15000_stage_gate_final_profile",
        "unchanged_contract": {
            "learning_rate": 1.0e-4,
            "action_semantics": "raw_actor_output",
            "action_clip": list(MICROBAN_TELEOP_V12_ACTION_CLIP),
            "hand_reward_weight": 2.0,
            "hand_reward_std_m": MICROBAN_TELEOP_HAND_TRACKING_FINAL_STD_M,
            "foot_reward_weight": 3.0,
            "foot_reward_std_m": MICROBAN_TELEOP_FOOT_TRACKING_FINAL_STD_M,
            "hand_rel_active": 0.7,
            "foot_rel_single_support_envs": 0.3,
            "foot_rel_both_feet_envs": 0.1,
            "normalizer": "unchanged_from_parent",
            "legacy_actor_tensors": "frozen",
            "pose_reward": "active_hand_arm_pose_release",
        },
    }


def is_hand_pose_release_final_rescue_marker(marker: object) -> bool:
    return isinstance(marker, Mapping) and marker.get("revision") == (
        MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_MARKER_REVISION
    )


def validate_hand_pose_release_final_rescue_marker(marker: object) -> dict[str, Any]:
    """Rebuild the marker from its recorded values and require equality."""

    if not is_hand_pose_release_final_rescue_marker(marker):
        raise ValueError("Pose-release final rescue lineage marker drifted")
    assert isinstance(marker, Mapping)
    failed_gate = marker.get("failed_final_gate")
    training = marker.get("training")
    if not isinstance(failed_gate, Mapping) or not isinstance(training, Mapping):
        raise ValueError("Pose-release final rescue lineage marker drifted")
    try:
        expected = hand_pose_release_final_rescue_marker(
            parent_checkpoint_sha256=marker.get("parent_checkpoint_sha256"),  # type: ignore[arg-type]
            failed_gate_checkpoint_sha256=failed_gate.get("checkpoint_sha256"),  # type: ignore[arg-type]
            failed_gate_tracking_report_sha256=failed_gate.get(  # type: ignore[arg-type]
                "tracking_report_sha256"
            ),
            failed_gate_failed_checks=failed_gate.get("failed_checks"),  # type: ignore[arg-type]
            failed_gate_failed_scenarios=failed_gate.get("failed_scenarios"),  # type: ignore[arg-type]
            inherited_corner_rescue_marker_sha256=marker.get(  # type: ignore[arg-type]
                "inherited_corner_rescue_marker_sha256"
            ),
            sampler_mix=marker.get("sampler_mix"),  # type: ignore[arg-type]
            training_seed=training.get("agent_seed"),  # type: ignore[arg-type]
            num_envs=training.get("num_envs"),  # type: ignore[arg-type]
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Pose-release final rescue lineage marker drifted") from exc
    if dict(marker) != expected:
        raise ValueError("Pose-release final rescue lineage marker drifted")
    return deepcopy(expected)


def make_microban_teleop_v12_hand_pose_release_final_rescue_env_cfg(
    play: bool = False, mix: str | None = None
):
    """Pose-release env with only the three command samplers changed."""

    from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release import (
        make_microban_teleop_v12_hand_pose_release_env_cfg,
    )

    cfg = make_microban_teleop_v12_hand_pose_release_env_cfg(play=play)
    if play:
        # Play/inspection keeps the canonical deterministic commands.
        return cfg
    mix = validate_hand_pose_release_final_rescue_mix(
        mix
        if mix is not None
        else os.environ.get(
            MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_MIX_ENV,
            MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_DEFAULT_MIX,
        )
    )
    for name, cls in (
        ("twist", FinalRescueTwistCommandCfg),
        ("foot_target", FinalRescueFootTargetCommandCfg),
        ("hand_target", FinalRescueHandTargetCommandCfg),
    ):
        cfg.commands[name] = _rebased_cfg(cfg.commands[name], cls, mix)
    if hand_pose_release_final_rescue_uses_scenario_push(mix):
        _install_scenario_push(cfg, mix)
    curriculum = cfg.curriculum.get("staged_curriculum")
    stages = None if curriculum is None else curriculum.params.get("stages")
    if not isinstance(stages, list):
        raise TypeError("Pose-release final rescue requires the v12 staged curriculum")
    # The process stops exactly at the 15000 boundary.
    stages[:] = [
        stage
        for stage in stages
        if int(stage.get("step", -1))
        < MICROBAN_TELEOP_V12_FINAL_RESCUE_TARGET_COMMON_STEP
    ]
    return cfg


def _install_scenario_push(cfg: Any, mix: str) -> None:
    """Ordinary push on ordinary episodes; the evaluator push on replayed ones.

    The ordinary term keeps its name, mode, interval and ``velocity_range``
    parameter, so the staged curriculum still sets its range.
    """

    ordinary = cfg.events.get("push_robot")
    if ordinary is None or ordinary.mode != "interval":
        raise TypeError("Pose-release final rescue requires the interval push_robot term")
    if set(ordinary.params) != {"velocity_range"}:
        raise TypeError("Pose-release final rescue push_robot params drifted")
    gated = deepcopy(ordinary)
    gated.func = push_ordinary_episodes
    gated.params["final_rescue_mix"] = mix
    cfg.events["push_robot"] = gated
    push = MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_SCENARIO_PUSH
    replay = deepcopy(ordinary)
    replay.func = push_replayed_episodes
    replay.interval_range_s = tuple(push["interval_range_s"])
    replay.params = {
        "velocity_range": deepcopy(push["velocity_range"]),
        "final_rescue_mix": mix,
    }
    name = MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_SCENARIO_PUSH_EVENT
    if name in cfg.events:
        raise ValueError(f"Event {name} already exists")
    cfg.events[name] = replay


def validate_hand_pose_release_final_rescue_infos(
    infos: Mapping[str, Any],
    *,
    iteration: int | None,
    corner_marker: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate the final-rescue part of a pose-release checkpoint's lineage.

    ``corner_marker`` is the checkpoint's (already validated) corner marker.
    Only model_14999 (completed 15000) is consumable; ``iteration=None`` is a
    structural check (HOME pose, packager recipe).
    """

    from mjlab_microban.tasks.microban_teleop_v12_corner_rescue import (
        canonical_json_sha256,
    )

    marker = validate_hand_pose_release_final_rescue_marker(
        infos.get(MICROBAN_TELEOP_V12_FINAL_RESCUE_INFO_KEY)
    )
    corner_sha = None if corner_marker is None else canonical_json_sha256(corner_marker)
    if marker["inherited_corner_rescue_marker_sha256"] != corner_sha:
        raise ValueError("Pose-release final rescue inherited corner lineage drifted")
    if iteration is not None:
        if (
            isinstance(iteration, bool)
            or not isinstance(iteration, int)
            or iteration != MICROBAN_TELEOP_V12_FINAL_RESCUE_TARGET_ITERATION
        ):
            raise ValueError(
                "Only model_14999 of a pose-release final rescue is consumable"
            )
        env_state = infos.get("env_state")
        if (
            isinstance(env_state, Mapping)
            and "common_step_counter" in env_state
            and env_state.get("common_step_counter")
            != MICROBAN_TELEOP_V12_FINAL_RESCUE_TARGET_COMMON_STEP
        ):
            raise ValueError("Pose-release final rescue clock drifted")
    # Recorded save fields (structural callers may pass partial infos).
    for key, expected in (
        (
            "active_actor_columns_at_save",
            list(MICROBAN_TELEOP_V12_FINAL_RESCUE_ACTIVE_COLUMNS),
        ),
        (
            "adapter_gradient_schedule_revision",
            TELEOP_V12_ADAPTER_GRADIENT_SCHEDULE_REVISION,
        ),
    ):
        if key in infos and infos[key] != expected:
            raise ValueError(f"Pose-release final rescue {key} drifted")
    return marker


def _hand_pose_release_final_rescue_rl_cfg():
    from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release import (
        MicrobanTeleopV12HandPoseReleaseRlCfg,
    )

    cfg = deepcopy(MicrobanTeleopV12HandPoseReleaseRlCfg)
    cfg.experiment_name = "mjlab_microban_teleop_v12"
    cfg.wandb_project = "mjlab_microban_teleop_v12_hand_pose_release_final_rescue"
    cfg.save_interval = MICROBAN_TELEOP_V12_FINAL_RESCUE_PROCESS_UPDATES
    cfg.max_iterations = MICROBAN_TELEOP_V12_FINAL_RESCUE_PROCESS_UPDATES
    return cfg


MicrobanTeleopV12HandPoseReleaseFinalRescueRlCfg = (
    _hand_pose_release_final_rescue_rl_cfg()
)
