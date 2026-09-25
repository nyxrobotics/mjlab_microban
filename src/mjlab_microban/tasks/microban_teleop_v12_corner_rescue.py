"""Hash-pinned corner-pair rescue for the pre-foot contract-v12 boundary.

The ordinary hand sampler remains active for 40% of resamples.  The failing
LF+RB bilateral extremum is replayed for 40%, while the already-passing LB+RF
extremum retains 20%.  This is a deliberately separate recipe: it changes command
sampling only, while retaining the v12 actor, optimizer, rewards, normalizer,
action semantics, and frozen legacy tensors.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch

from mjlab_microban.robot.microban_hand_fk import (
    MICROBAN_REACHABLE_HAND_EVALUATION_JOINTS_DEG,
    microban_hand_offsets_from_arm_joints,
)
from mjlab_microban.tasks.microban_teleop_env_cfg import (
    MICROBAN_TELEOP_HAND_TRACKING_FINAL_STD_M,
)
from mjlab_microban.tasks.microban_teleop_mdp import (
    ResetFixedHandTargetCommand,
    ResetFixedHandTargetCommandCfg,
)
from mjlab_microban.tasks.microban_teleop_v12_actor import (
    TELEOP_V12_ADAPTER_GRADIENT_SCHEDULE_REVISION,
    TELEOP_V12_FOOT_OBSERVATION_COLUMNS,
    TELEOP_V12_HAND_OBSERVATION_COLUMNS,
    TELEOP_V12_HMD_OBSERVATION_COLUMNS,
    TELEOP_V12_TARGET_POSITION_NORMALIZER_STORED_STD,
)
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MICROBAN_TELEOP_V12_RECIPE_REVISION,
    MicrobanTeleopV12RlCfg,
    make_microban_teleop_v12_env_cfg,
)

MICROBAN_TELEOP_V12_CORNER_RESCUE_TASK_ID = (
    "Mjlab-Teleop-V12-Corner-Rescue-Microban"
)
MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY = (
    "microban_teleop_v12_corner_pair_rescue"
)
MICROBAN_TELEOP_V12_CORNER_RESCUE_RECIPE_REVISION = (
    "model9900_targeted_bilateral_corner_pair_replay_to10000_v1"
)
MICROBAN_TELEOP_V12_CORNER_RESCUE_MARKER_REVISION = (
    "pinned_model9900_uniform40_lf_rb40_lb_rf20_99_updates_v1"
)
MICROBAN_TELEOP_V12_CORNER_RESCUE_SAMPLER_REVISION = (
    "uniform_joint_box40pct_lf_rb40pct_lb_rf20pct_v1"
)

MICROBAN_TELEOP_V12_CORNER_RESCUE_PARENT_SHA256 = (
    "063a8f65ebf9007d63395e9a5b98420eb025bd39416dab5727f9f4c06fc6e877"
)
MICROBAN_TELEOP_V12_CORNER_RESCUE_PARENT_TRACKING_SHA256 = (
    "399db0cee55c137d3d0226ebcb54b0fa3bb84c95496c5c206af55f2fb25837f4"
)
MICROBAN_TELEOP_V12_CORNER_RESCUE_PARENT_ITERATION = 9_900
MICROBAN_TELEOP_V12_CORNER_RESCUE_PARENT_COMPLETED_UPDATES = 9_901
MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_ITERATION = 9_999
MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_COMPLETED_UPDATES = 10_000
MICROBAN_TELEOP_V12_CORNER_RESCUE_PROCESS_UPDATES = 99
MICROBAN_TELEOP_V12_CORNER_RESCUE_NUM_STEPS_PER_ENV = 24
MICROBAN_TELEOP_V12_CORNER_RESCUE_PARENT_COMMON_STEP = (
    MICROBAN_TELEOP_V12_CORNER_RESCUE_PARENT_COMPLETED_UPDATES
    * MICROBAN_TELEOP_V12_CORNER_RESCUE_NUM_STEPS_PER_ENV
)
MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_COMMON_STEP = (
    MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_COMPLETED_UPDATES
    * MICROBAN_TELEOP_V12_CORNER_RESCUE_NUM_STEPS_PER_ENV
)
MICROBAN_TELEOP_V12_CORNER_RESCUE_PARENT_OPTIMIZER_STEP = 198_020
MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_OPTIMIZER_STEP = 200_000

MICROBAN_TELEOP_V12_LF_RB_PROBABILITY = 0.40
MICROBAN_TELEOP_V12_LB_RF_PROBABILITY = 0.20
MICROBAN_TELEOP_V12_UNIFORM_REMAINDER_PROBABILITY = 0.40
MICROBAN_TELEOP_V12_CORNER_RESCUE_ACTIVE_COLUMNS = (
    *TELEOP_V12_HMD_OBSERVATION_COLUMNS,
    *TELEOP_V12_HAND_OBSERVATION_COLUMNS,
)

if (
    MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_COMPLETED_UPDATES
    - MICROBAN_TELEOP_V12_CORNER_RESCUE_PARENT_COMPLETED_UPDATES
    != MICROBAN_TELEOP_V12_CORNER_RESCUE_PROCESS_UPDATES
):
    raise RuntimeError("Corner rescue must contain exactly 99 PPO updates")
if not math.isclose(
    MICROBAN_TELEOP_V12_LF_RB_PROBABILITY
    + MICROBAN_TELEOP_V12_LB_RF_PROBABILITY
    + MICROBAN_TELEOP_V12_UNIFORM_REMAINDER_PROBABILITY,
    1.0,
):
    raise RuntimeError("Corner rescue sampler probabilities must sum to one")


def assert_corner_rescue_optimizer_step(
    payload: Mapping[str, Any], *, expected_step: int
) -> None:
    """Require every initialized Adam state to share one exact update clock."""

    if isinstance(expected_step, bool) or expected_step < 0:
        raise ValueError("Expected optimizer step must be a non-negative integer")
    optimizer = payload.get("optimizer_state_dict")
    states = optimizer.get("state") if isinstance(optimizer, Mapping) else None
    if not isinstance(states, Mapping) or not states:
        raise TypeError("Corner rescue Adam state is missing")
    observed: list[int] = []
    for state in states.values():
        if not isinstance(state, Mapping) or "step" not in state:
            raise ValueError("Every corner rescue Adam state must expose a step")
        value = state["step"]
        if isinstance(value, torch.Tensor):
            if value.numel() != 1 or not bool(torch.isfinite(value).all().item()):
                raise ValueError("Corner rescue Adam step must be one finite scalar")
            scalar = float(value.item())
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            scalar = float(value)
        else:
            raise TypeError("Corner rescue Adam step has an unsupported type")
        if not scalar.is_integer():
            raise ValueError("Corner rescue Adam step must be integral")
        observed.append(int(scalar))
    if set(observed) != {expected_step}:
        raise ValueError(
            "Corner rescue optimizer clock drifted: "
            f"expected {expected_step}, observed {sorted(set(observed))}"
        )


def assert_corner_rescue_foot_adapter_zero(payload: Mapping[str, Any]) -> None:
    """Prove frozen foot normalizer, actor weights, and Adam state exactly."""

    actor = payload.get("actor_state_dict")
    optimizer = payload.get("optimizer_state_dict")
    if not isinstance(actor, Mapping) or not isinstance(optimizer, Mapping):
        raise TypeError("Corner rescue checkpoint state is incomplete")
    first = actor.get("mlp.0.weight")
    if not isinstance(first, torch.Tensor) or tuple(first.shape) != (512, 83):
        raise ValueError("Corner rescue actor W0 shape drifted")
    foot = first[:, TELEOP_V12_FOOT_OBSERVATION_COLUMNS]
    if not torch.equal(foot, torch.zeros_like(foot)):
        raise ValueError("Corner rescue foot actor columns are not exact zero")
    expected_std_values = TELEOP_V12_TARGET_POSITION_NORMALIZER_STORED_STD[:6]
    for name, expected_values in (
        ("obs_normalizer._mean", (0.0,) * 6),
        (
            "obs_normalizer._var",
            tuple(value * value for value in expected_std_values),
        ),
        ("obs_normalizer._std", expected_std_values),
    ):
        tensor = actor.get(name)
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != (1, 83):
            raise ValueError(f"Corner rescue actor {name} shape drifted")
        foot_tensor = tensor[:, TELEOP_V12_FOOT_OBSERVATION_COLUMNS]
        expected = foot_tensor.new_tensor(expected_values).unsqueeze(0)
        if not torch.equal(foot_tensor, expected):
            raise ValueError(f"Corner rescue foot {name} drifted")

    states = optimizer.get("state")
    if not isinstance(states, Mapping):
        raise TypeError("Corner rescue Adam state is missing")
    candidates = []
    for state in states.values():
        if not isinstance(state, Mapping):
            continue
        first_moment = state.get("exp_avg")
        second_moment = state.get("exp_avg_sq")
        if (
            isinstance(first_moment, torch.Tensor)
            and isinstance(second_moment, torch.Tensor)
            and tuple(first_moment.shape) == (512, 83)
            and tuple(second_moment.shape) == (512, 83)
        ):
            candidates.append(state)
    if len(candidates) != 1:
        raise ValueError("Corner rescue requires one unambiguous actor Adam state")
    for name in ("exp_avg", "exp_avg_sq"):
        moment = candidates[0][name][:, TELEOP_V12_FOOT_OBSERVATION_COLUMNS]
        if not torch.equal(moment, torch.zeros_like(moment)):
            raise ValueError(f"Corner rescue foot Adam {name} is not exact zero")


def _named_joint_pose(name: str) -> tuple[float, float, float]:
    try:
        return dict(MICROBAN_REACHABLE_HAND_EVALUATION_JOINTS_DEG)[name]
    except KeyError as exc:  # pragma: no cover - import-time contract guard
        raise RuntimeError(f"Missing reachable hand pose {name!r}") from exc


def corner_pair_joint_targets(
    *, device: torch.device | str, dtype: torch.dtype
) -> torch.Tensor:
    """Return ``[LF+RB, LB+RF]`` arm tuples in fixed left/right order."""

    forward = _named_joint_pose("F")
    backward = _named_joint_pose("B")
    values = (
        (forward, (backward[0], -backward[1], backward[2])),
        (backward, (forward[0], -forward[1], forward[2])),
    )
    return torch.deg2rad(torch.tensor(values, device=device, dtype=dtype))


def corner_pair_selection(selector: torch.Tensor) -> torch.Tensor:
    """Map uniform random values to uniform/LF+RB/LB+RF selection IDs.

    ``0`` preserves the ordinary sampler, ``1`` selects LF+RB, and ``2``
    selects LB+RF.
    """

    if not isinstance(selector, torch.Tensor) or not selector.is_floating_point():
        raise TypeError("Corner-pair selector must be a floating tensor")
    if not bool(torch.isfinite(selector).all().item()) or bool(
        torch.any((selector < 0.0) | (selector >= 1.0)).item()
    ):
        raise ValueError("Corner-pair selector values must be in [0, 1)")
    first_upper = MICROBAN_TELEOP_V12_LF_RB_PROBABILITY
    # Derive the upper boundary from the remainder so the exact float 0.6
    # belongs to the ordinary branch rather than 0.4 + 0.2 rounding upward.
    second_upper = 1.0 - MICROBAN_TELEOP_V12_UNIFORM_REMAINDER_PROBABILITY
    result = torch.zeros_like(selector, dtype=torch.long)
    result[selector < first_upper] = 1
    result[(selector >= first_upper) & (selector < second_upper)] = 2
    return result


class CornerPairHandTargetCommand(ResetFixedHandTargetCommand):
    """Ordinary reachable targets plus two explicitly replayed gate corners."""

    cfg: CornerPairHandTargetCommandCfg

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        # Preserve every ordinary independent/unilateral sample outside the
        # explicit 45% rescue mixture.
        super()._resample_command(env_ids)
        selector = torch.rand(
            len(env_ids),
            device=self.device,
            dtype=self.hand_target_offset_b.dtype,
        )
        choice = corner_pair_selection(selector)
        pair_ids = env_ids[choice != 0]
        if len(pair_ids) == 0:
            return
        pair_targets = corner_pair_joint_targets(
            device=self.device, dtype=self.hand_target_offset_b.dtype
        )
        selected_targets = pair_targets[choice[choice != 0] - 1]
        self.is_active[pair_ids] = True
        self.sampled_arm_joint_pos_rad[pair_ids] = selected_targets
        self.hand_target_offset_b[pair_ids] = microban_hand_offsets_from_arm_joints(
            selected_targets
        )


@dataclass(kw_only=True)
class CornerPairHandTargetCommandCfg(ResetFixedHandTargetCommandCfg):
    """Pinned uniform/LF+RB/LB+RF = 40/40/20 sampler configuration."""

    def build(self, env: Any) -> CornerPairHandTargetCommand:
        return CornerPairHandTargetCommand(self, env)


def corner_rescue_marker() -> dict[str, Any]:
    """Return the immutable marker embedded in every rescue checkpoint."""

    return {
        "schema_version": 1,
        "revision": MICROBAN_TELEOP_V12_CORNER_RESCUE_MARKER_REVISION,
        "parent_checkpoint_sha256": (
            MICROBAN_TELEOP_V12_CORNER_RESCUE_PARENT_SHA256
        ),
        "parent_strict_tracking_report_sha256": (
            MICROBAN_TELEOP_V12_CORNER_RESCUE_PARENT_TRACKING_SHA256
        ),
        "parent_iteration": MICROBAN_TELEOP_V12_CORNER_RESCUE_PARENT_ITERATION,
        "parent_completed_updates": (
            MICROBAN_TELEOP_V12_CORNER_RESCUE_PARENT_COMPLETED_UPDATES
        ),
        "parent_common_step_counter": (
            MICROBAN_TELEOP_V12_CORNER_RESCUE_PARENT_COMMON_STEP
        ),
        "target_iteration": MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_ITERATION,
        "target_completed_updates": (
            MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_COMPLETED_UPDATES
        ),
        "target_common_step_counter": (
            MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_COMMON_STEP
        ),
        "process_updates": MICROBAN_TELEOP_V12_CORNER_RESCUE_PROCESS_UPDATES,
        "optimizer_step": {
            "parent": MICROBAN_TELEOP_V12_CORNER_RESCUE_PARENT_OPTIMIZER_STEP,
            "target": MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_OPTIMIZER_STEP,
        },
        "training": {
            "environment_seed": 42,
            "agent_seed": 42,
            "num_envs": 2_048,
            "num_steps_per_env": 24,
        },
        "source_recipe_revision": MICROBAN_TELEOP_V12_RECIPE_REVISION,
        "rescue_recipe_revision": (
            MICROBAN_TELEOP_V12_CORNER_RESCUE_RECIPE_REVISION
        ),
        "sampler_revision": MICROBAN_TELEOP_V12_CORNER_RESCUE_SAMPLER_REVISION,
        "sampler_probabilities": {
            "ordinary_uniform_independent": (
                MICROBAN_TELEOP_V12_UNIFORM_REMAINDER_PROBABILITY
            ),
            "left_forward_right_backward": (
                MICROBAN_TELEOP_V12_LF_RB_PROBABILITY
            ),
            "left_backward_right_forward": (
                MICROBAN_TELEOP_V12_LB_RF_PROBABILITY
            ),
        },
        "corner_joint_tuples_deg": {
            "left_forward_right_backward": [
                list(_named_joint_pose("F")),
                [
                    _named_joint_pose("B")[0],
                    -_named_joint_pose("B")[1],
                    _named_joint_pose("B")[2],
                ],
            ],
            "left_backward_right_forward": [
                list(_named_joint_pose("B")),
                [
                    _named_joint_pose("F")[0],
                    -_named_joint_pose("F")[1],
                    _named_joint_pose("F")[2],
                ],
            ],
        },
        "adapter_gradient_schedule_revision": (
            TELEOP_V12_ADAPTER_GRADIENT_SCHEDULE_REVISION
        ),
        "active_actor_columns": list(
            MICROBAN_TELEOP_V12_CORNER_RESCUE_ACTIVE_COLUMNS
        ),
        "foot_activation": "held_inactive_through_completed_update_10000",
        "foot_observation_columns": list(TELEOP_V12_FOOT_OBSERVATION_COLUMNS),
        "foot_command_state_at_save": {
            "shape": [2_048, 2, 3],
            "target_offset_exact_zero": True,
            "single_support_active_count": 0,
            "both_feet_active_count": 0,
        },
        "unchanged_contract": {
            "learning_rate": 1.0e-4,
            "action_semantics": "raw_actor_output",
            "action_clip": None,
            "hand_reward_weight": 2.0,
            "hand_reward_std_m": MICROBAN_TELEOP_HAND_TRACKING_FINAL_STD_M,
            "normalizer": "unchanged_from_parent",
            "legacy_actor_tensors": "frozen",
        },
    }


def validate_corner_rescue_marker(
    infos: Mapping[str, Any], *, iteration: int
) -> dict[str, Any]:
    """Validate a saved rescue checkpoint's exact recipe and clock range."""

    if not isinstance(infos, Mapping):
        raise TypeError("Corner rescue checkpoint infos must be a mapping")
    if infos.get("microban_teleop_training_contract_version") != "12":
        raise ValueError("Corner rescue checkpoint is not contract-v12")
    if infos.get("microban_teleop_recipe_revision") != (
        MICROBAN_TELEOP_V12_CORNER_RESCUE_RECIPE_REVISION
    ):
        raise ValueError("Corner rescue recipe revision drifted")
    if infos.get("adapter_gradient_schedule_revision") != (
        TELEOP_V12_ADAPTER_GRADIENT_SCHEDULE_REVISION
    ):
        raise ValueError("Corner rescue adapter schedule drifted")
    if not (
        MICROBAN_TELEOP_V12_CORNER_RESCUE_PARENT_ITERATION
        < iteration
        <= MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_ITERATION
    ):
        raise ValueError("Corner rescue checkpoint clock is outside 9901..9999")
    env_state = infos.get("env_state")
    expected_step = (iteration + 1) * (
        MICROBAN_TELEOP_V12_CORNER_RESCUE_NUM_STEPS_PER_ENV
    )
    if (
        not isinstance(env_state, Mapping)
        or env_state.get("common_step_counter") != expected_step
    ):
        raise ValueError("Corner rescue iteration/common-step relation drifted")
    if infos.get("active_actor_columns_at_save") != list(
        MICROBAN_TELEOP_V12_CORNER_RESCUE_ACTIVE_COLUMNS
    ):
        raise ValueError("Corner rescue active actor columns drifted")
    return validate_corner_rescue_lineage_marker(
        infos.get(MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY)
    )


def validate_corner_rescue_lineage_marker(marker: object) -> dict[str, Any]:
    """Validate the immutable rescue lineage independent of current recipe."""

    expected = corner_rescue_marker()
    if marker != expected:
        raise ValueError("Corner rescue lineage marker drifted")
    return deepcopy(expected)


def validate_corner_rescue_canonical_lineage(
    infos: Mapping[str, Any], *, iteration: int
) -> dict[str, Any] | None:
    """Accept only the final rescue checkpoint or a marked canonical descendant.

    Intermediate rescue checkpoints are intentionally not consumable.  The first
    ordinary runner save after resuming model9999 returns to the canonical recipe
    while retaining the immutable historical marker.
    """

    if not isinstance(infos, Mapping):
        raise TypeError("Contract-v12 checkpoint infos must be a mapping")
    if isinstance(iteration, bool) or not isinstance(iteration, int):
        raise TypeError("Contract-v12 checkpoint iteration must be an integer")
    recipe = infos.get("microban_teleop_recipe_revision")
    marker = infos.get(MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY)
    if recipe == MICROBAN_TELEOP_V12_CORNER_RESCUE_RECIPE_REVISION:
        if iteration != MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_ITERATION:
            raise ValueError(
                "Only final model9999 from the corner rescue is consumable"
            )
        return validate_corner_rescue_marker(infos, iteration=iteration)
    if recipe != MICROBAN_TELEOP_V12_RECIPE_REVISION:
        raise ValueError("Checkpoint recipe is neither canonical nor corner rescue")
    if marker is None:
        return None
    if iteration <= MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_ITERATION:
        raise ValueError("Canonical corner-rescue descendant clock is invalid")
    return validate_corner_rescue_lineage_marker(marker)


def make_microban_teleop_v12_corner_rescue_env_cfg(play: bool = False):
    """Build the v12 task with only the pinned hand sampler changed."""

    cfg = make_microban_teleop_v12_env_cfg(play=play)
    existing = cfg.commands["hand_target"]
    cfg.commands["hand_target"] = CornerPairHandTargetCommandCfg(
        resampling_time_range=existing.resampling_time_range,
        rel_active=existing.rel_active,
    )
    if not play:
        curriculum = cfg.curriculum.get("staged_curriculum")
        stages = None if curriculum is None else curriculum.params.get("stages")
        if not isinstance(stages, list):
            raise TypeError("Corner rescue requires the v12 staged curriculum")
        # No foot stage may be reconstructed when model9900 is loaded.  The
        # process stops exactly at the original 10000 boundary.
        stages[:] = [
            stage
            for stage in stages
            if int(stage.get("step", -1))
            < MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_COMMON_STEP
        ]
    return cfg


MicrobanTeleopV12CornerRescueRlCfg = deepcopy(MicrobanTeleopV12RlCfg)
MicrobanTeleopV12CornerRescueRlCfg.experiment_name = (
    "mjlab_microban_teleop_v12"
)
MicrobanTeleopV12CornerRescueRlCfg.wandb_project = (
    "mjlab_microban_teleop_v12_corner_rescue"
)
MicrobanTeleopV12CornerRescueRlCfg.save_interval = (
    MICROBAN_TELEOP_V12_CORNER_RESCUE_PROCESS_UPDATES
)
MicrobanTeleopV12CornerRescueRlCfg.max_iterations = (
    MICROBAN_TELEOP_V12_CORNER_RESCUE_PROCESS_UPDATES
)
