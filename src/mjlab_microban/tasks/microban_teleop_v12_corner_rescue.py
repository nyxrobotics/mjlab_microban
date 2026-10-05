"""Recorded-parent corner-pair rescue for the pre-foot contract-v12 boundary.

The ordinary hand sampler remains active for 5% of resamples.  The failing
LF+RB bilateral extremum is replayed for 90%, while the already-passing LB+RF
extremum retains 5%.  This is a deliberately separate recipe: it changes command
sampling only, while retaining the v12 actor, optimizer, rewards, normalizer,
action semantics, and frozen legacy tensors.

The historical (old-HOME) chain pinned one model_9900 and its strict report by
literal SHA-256.  This revision records the parent checkpoint and its strict
tracking report (failing only ``hand_tracking_rms``) in the marker instead; the
runner re-hashes both when it loads the parent, and every consumer rebuilds the
marker from those recorded values.  Clocks, sampler, and contract are unchanged.

Active-hand arm pose-release variant (forward-lean HOME): the same 99-update
replay from a fresh pose-release chain's model_9900 whose strict HMD/hand report
fails only hand accuracy (RMS and/or P95).  Both bilateral corners failed there,
so its registered sampler mixes split the replay between LF+RB and LB+RF (5 %
ordinary): ``lf60`` = 60/35, ``lf65`` = 65/30, ``lf72`` = 72/23 and
``lf90`` = 90/5 (the canonical split), chosen
by the launcher through ``MICROBAN_V12_PR_CORNER_RESCUE_MIX`` and recorded in
the marker revision.
Its checkpoints keep the pose-release recipe revision (the env is the
pose-release env with only the hand sampler changed) and carry a pose-release
variant of the marker under the same infos key; only its model_9999 and that
model's ordinary pose-release descendants are consumable
(``microban_teleop_v12_hand_pose_release_lineage``).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch

from mjlab_microban.robot.microban_hand_fk import (
    MICROBAN_REACHABLE_HAND_EVALUATION_JOINTS_DEG,
    microban_hand_target_offsets_from_arm_joints,
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
    MICROBAN_TELEOP_V12_ACTION_CLIP,
    MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION,
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
    "model9900_targeted_bilateral_corner_pair_replay_to10000_receiver_box_f_v4"
)
MICROBAN_TELEOP_V12_CORNER_RESCUE_MARKER_REVISION = (
    "recorded_model9900_uniform5_lf_rb90_lb_rf5_99_updates_receiver_box_f_v4"
)
MICROBAN_TELEOP_V12_CORNER_RESCUE_SAMPLER_REVISION = (
    "uniform_joint_box5pct_lf_rb90pct_lb_rf5pct_receiver_box_f_v3"
)
# The launcher stages the parent's strict tracking report next to the staged
# parent checkpoint under this name; the runner re-validates and hashes it.
MICROBAN_TELEOP_V12_CORNER_RESCUE_PARENT_REPORT_FILENAME = (
    "parent_strict_tracking.json"
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

MICROBAN_TELEOP_V12_LF_RB_PROBABILITY = 0.90
MICROBAN_TELEOP_V12_LB_RF_PROBABILITY = 0.05
MICROBAN_TELEOP_V12_UNIFORM_REMAINDER_PROBABILITY = 0.05
# Pose-release variant (see module doc).
MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_CORNER_RESCUE_TASK_ID = (
    "Mjlab-Teleop-V12-HandPoseRelease-Corner-Rescue-Microban"
)
MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_CORNER_RESCUE_MARKER_REVISION = (
    "recorded_pose_release_model9900_uniform5_lf_rb60_lb_rf35_99_updates_"
    "receiver_box_f_v1"
)
MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_CORNER_RESCUE_SAMPLER_REVISION = (
    "uniform_joint_box5pct_lf_rb60pct_lb_rf35pct_receiver_box_f_v1"
)
MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_LF_RB_PROBABILITY = 0.60
MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_LB_RF_PROBABILITY = 0.35
# Registered pose-release sampler mixes (name -> marker/sampler revisions and
# LF+RB / LB+RF shares; the ordinary remainder is always 5 %).
MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_CORNER_RESCUE_MIX_ENV = (
    "MICROBAN_V12_PR_CORNER_RESCUE_MIX"
)
MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_CORNER_RESCUE_DEFAULT_MIX = "lf60"
MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_CORNER_RESCUE_MIXES: dict[str, dict[str, Any]] = {
    "lf60": {
        "marker_revision": (
            MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_CORNER_RESCUE_MARKER_REVISION
        ),
        "sampler_revision": (
            MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_CORNER_RESCUE_SAMPLER_REVISION
        ),
        "lf_rb": MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_LF_RB_PROBABILITY,
        "lb_rf": MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_LB_RF_PROBABILITY,
    },
    "lf65": {
        "marker_revision": (
            "recorded_pose_release_model9900_uniform5_lf_rb65_lb_rf30_99_updates_"
            "receiver_box_f_v1"
        ),
        "sampler_revision": (
            "uniform_joint_box5pct_lf_rb65pct_lb_rf30pct_receiver_box_f_pose_release_v1"
        ),
        "lf_rb": 0.65,
        "lb_rf": 0.30,
    },
    "lf72": {
        "marker_revision": (
            "recorded_pose_release_model9900_uniform5_lf_rb72_lb_rf23_99_updates_"
            "receiver_box_f_v1"
        ),
        "sampler_revision": (
            "uniform_joint_box5pct_lf_rb72pct_lb_rf23pct_receiver_box_f_pose_release_v1"
        ),
        "lf_rb": 0.72,
        "lb_rf": 0.23,
    },
    "lf90": {
        "marker_revision": (
            "recorded_pose_release_model9900_uniform5_lf_rb90_lb_rf5_99_updates_"
            "receiver_box_f_v1"
        ),
        "sampler_revision": (
            "uniform_joint_box5pct_lf_rb90pct_lb_rf5pct_receiver_box_f_pose_release_v1"
        ),
        "lf_rb": 0.90,
        "lb_rf": 0.05,
    },
}
# The pose-release parent may fail any non-empty subset of these strict checks.
MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_CORNER_RESCUE_PARENT_FAILED_CHECKS = (
    frozenset(("hand_tracking_rms", "hand_tracking_p95"))
)
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
for _mix in MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_CORNER_RESCUE_MIXES.values():
    if not math.isclose(
        _mix["lf_rb"] + _mix["lb_rf"] + MICROBAN_TELEOP_V12_UNIFORM_REMAINDER_PROBABILITY,
        1.0,
    ):
        raise RuntimeError("Pose-release corner rescue probabilities must sum to one")


def hand_pose_release_corner_rescue_mix(name: str) -> dict[str, Any]:
    try:
        return dict(MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_CORNER_RESCUE_MIXES[name])
    except KeyError as exc:
        raise ValueError(f"Unknown pose-release corner rescue mix {name!r}") from exc


def selected_hand_pose_release_corner_rescue_mix() -> str:
    """Mix named by the launcher's environment variable (default lf60)."""

    import os

    name = os.environ.get(
        MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_CORNER_RESCUE_MIX_ENV,
        MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_CORNER_RESCUE_DEFAULT_MIX,
    )
    hand_pose_release_corner_rescue_mix(name)
    return name


def hand_pose_release_corner_rescue_mix_for_lf_rb(lf_rb_probability: float) -> str:
    for name, mix in MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_CORNER_RESCUE_MIXES.items():
        if mix["lf_rb"] == lf_rb_probability:
            return name
    raise ValueError("Hand sampler share is not a registered pose-release mix")


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


def corner_pair_selection(
    selector: torch.Tensor,
    *,
    lf_rb_probability: float = MICROBAN_TELEOP_V12_LF_RB_PROBABILITY,
) -> torch.Tensor:
    """Map uniform random values to uniform/LF+RB/LB+RF selection IDs.

    ``0`` preserves the ordinary sampler, ``1`` selects LF+RB, and ``2``
    selects LB+RF.  The ordinary remainder is always 5 %; the pose-release
    variant passes its own LF+RB share (LB+RF takes the rest).
    """

    if not isinstance(selector, torch.Tensor) or not selector.is_floating_point():
        raise TypeError("Corner-pair selector must be a floating tensor")
    if not bool(torch.isfinite(selector).all().item()) or bool(
        torch.any((selector < 0.0) | (selector >= 1.0)).item()
    ):
        raise ValueError("Corner-pair selector values must be in [0, 1)")
    if lf_rb_probability not in (
        MICROBAN_TELEOP_V12_LF_RB_PROBABILITY,
        *(
            mix["lf_rb"]
            for mix in MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_CORNER_RESCUE_MIXES.values()
        ),
    ):
        raise ValueError("Corner-pair LF+RB share is not a registered mix")
    first_upper = lf_rb_probability
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
        choice = corner_pair_selection(
            selector, lf_rb_probability=self.cfg.lf_rb_probability
        )
        pair_ids = env_ids[choice != 0]
        if len(pair_ids) == 0:
            return
        pair_targets = corner_pair_joint_targets(
            device=self.device, dtype=self.hand_target_offset_b.dtype
        )
        selected_targets = pair_targets[choice[choice != 0] - 1]
        self.is_active[pair_ids] = True
        self.sampled_arm_joint_pos_rad[pair_ids] = selected_targets
        self.hand_target_offset_b[pair_ids] = (
            microban_hand_target_offsets_from_arm_joints(
                selected_targets, trunk_pitch=self.cfg.trunk_pitch
            )
        )


@dataclass(kw_only=True)
class CornerPairHandTargetCommandCfg(ResetFixedHandTargetCommandCfg):
    """Uniform/LF+RB/LB+RF = 5/90/5 (pose-release variant 5/60/35) sampler."""

    lf_rb_probability: float = MICROBAN_TELEOP_V12_LF_RB_PROBABILITY

    def build(self, env: Any) -> CornerPairHandTargetCommand:
        return CornerPairHandTargetCommand(self, env)


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def canonical_json_sha256(value: object) -> str:
    """Return the SHA-256 of a canonical JSON encoding (sorted keys)."""

    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _pose_release_parent_failed_checks(value: object) -> list[str]:
    if (
        not isinstance(value, (list, tuple))
        or not value
        or any(not isinstance(item, str) for item in value)
        or len(set(value)) != len(value)
        or not set(value)
        <= MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_CORNER_RESCUE_PARENT_FAILED_CHECKS
    ):
        raise ValueError(
            "Pose-release corner rescue parent may fail only hand accuracy checks"
        )
    return sorted(value)


def corner_rescue_marker(
    *,
    parent_checkpoint_sha256: str,
    parent_strict_tracking_report_sha256: str,
    hand_pose_release: bool = False,
    parent_strict_failed_checks: object = ("hand_tracking_rms",),
    pose_release_mix: str = MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_CORNER_RESCUE_DEFAULT_MIX,
) -> dict[str, Any]:
    """Return the marker embedded in every rescue checkpoint.

    Only the two recorded hashes vary (and, for the pose-release variant, the
    recorded strict failed checks); every other field is fixed by code.
    """

    marker = _canonical_corner_rescue_marker(
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        parent_strict_tracking_report_sha256=parent_strict_tracking_report_sha256,
    )
    if not hand_pose_release:
        return marker
    mix = hand_pose_release_corner_rescue_mix(pose_release_mix)
    marker["revision"] = mix["marker_revision"]
    marker["parent_strict_failed_checks"] = _pose_release_parent_failed_checks(
        parent_strict_failed_checks
    )
    marker["source_recipe_revision"] = (
        MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION
    )
    # Rescue saves keep the pose-release recipe; the marker names the replay.
    marker["rescue_recipe_revision"] = (
        MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION
    )
    marker["rescue_task_id"] = (
        MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_CORNER_RESCUE_TASK_ID
    )
    marker["parent_lineage"] = "fresh_pose_release_chain"
    marker["sampler_mix"] = pose_release_mix
    marker["sampler_revision"] = mix["sampler_revision"]
    marker["sampler_probabilities"] = {
        "ordinary_uniform_independent": (
            MICROBAN_TELEOP_V12_UNIFORM_REMAINDER_PROBABILITY
        ),
        "left_forward_right_backward": mix["lf_rb"],
        "left_backward_right_forward": mix["lb_rf"],
    }
    marker["unchanged_contract"] = {
        **marker["unchanged_contract"],
        "pose_reward": "active_hand_arm_pose_release",
    }
    return marker


def _hand_pose_release_mix_for_marker_revision(revision: object) -> str | None:
    for name, mix in MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_CORNER_RESCUE_MIXES.items():
        if mix["marker_revision"] == revision:
            return name
    return None


def is_hand_pose_release_corner_rescue_marker(marker: object) -> bool:
    return isinstance(marker, Mapping) and (
        _hand_pose_release_mix_for_marker_revision(marker.get("revision")) is not None
    )


def _canonical_corner_rescue_marker(
    *,
    parent_checkpoint_sha256: str,
    parent_strict_tracking_report_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "revision": MICROBAN_TELEOP_V12_CORNER_RESCUE_MARKER_REVISION,
        "parent_identity": "recorded_and_rehashed_on_load",
        "parent_checkpoint_sha256": _require_sha256(
            parent_checkpoint_sha256, "Corner rescue parent checkpoint"
        ),
        "parent_strict_tracking_report_sha256": _require_sha256(
            parent_strict_tracking_report_sha256, "Corner rescue parent report"
        ),
        "parent_strict_failed_checks": ["hand_tracking_rms"],
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
            "action_clip": list(MICROBAN_TELEOP_V12_ACTION_CLIP),
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
    """Rebuild the marker from its two recorded hashes and require equality."""

    if not isinstance(marker, Mapping):
        raise ValueError("Corner rescue lineage marker drifted")
    pose_release = is_hand_pose_release_corner_rescue_marker(marker)
    try:
        expected = corner_rescue_marker(
            parent_checkpoint_sha256=marker.get("parent_checkpoint_sha256"),  # type: ignore[arg-type]
            parent_strict_tracking_report_sha256=marker.get(  # type: ignore[arg-type]
                "parent_strict_tracking_report_sha256"
            ),
            hand_pose_release=pose_release,
            parent_strict_failed_checks=(
                marker.get("parent_strict_failed_checks")
                if pose_release
                else ("hand_tracking_rms",)
            ),
            pose_release_mix=(
                _hand_pose_release_mix_for_marker_revision(marker.get("revision"))
                if pose_release
                else MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_CORNER_RESCUE_DEFAULT_MIX
            ),
        )
    except ValueError as exc:
        raise ValueError("Corner rescue lineage marker drifted") from exc
    if dict(marker) != expected:
        raise ValueError("Corner rescue lineage marker drifted")
    return deepcopy(expected)


def validate_corner_rescue_canonical_lineage(
    infos: Mapping[str, Any],
    *,
    iteration: int,
    allow_hand_pose_release_recipe: bool = False,
) -> dict[str, Any] | None:
    """Accept only the final rescue checkpoint or a marked canonical descendant.

    Intermediate rescue checkpoints are intentionally not consumable.  The first
    ordinary runner save after resuming model9999 returns to the canonical recipe
    while retaining the immutable historical marker.  A checkpoint of the
    active-hand arm pose-release recipe is accepted when its lineage is
    release-eligible (fresh chain, or the recorded model_7099 switch whose
    parent checkpoint and stage gate are re-validated here); with
    ``allow_hand_pose_release_recipe`` the experimental switch is accepted too.
    It never carries a rescue, so the result is ``None``.
    """

    if not isinstance(infos, Mapping):
        raise TypeError("Contract-v12 checkpoint infos must be a mapping")
    if isinstance(iteration, bool) or not isinstance(iteration, int):
        raise TypeError("Contract-v12 checkpoint iteration must be an integer")
    recipe = infos.get("microban_teleop_recipe_revision")
    marker = infos.get(MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY)
    # Imported here: the final-rescue module imports this one.
    from mjlab_microban.tasks.microban_teleop_v12_final_rescue import (
        MICROBAN_TELEOP_V12_FINAL_RESCUE_INFO_KEY,
        MICROBAN_TELEOP_V12_FINAL_RESCUE_RECIPE_REVISION,
        validate_final_rescue_consumable,
    )
    from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release_final_rescue import (
        is_hand_pose_release_final_rescue_marker,
    )

    if recipe != MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION and (
        is_hand_pose_release_final_rescue_marker(
            infos.get(MICROBAN_TELEOP_V12_FINAL_RESCUE_INFO_KEY)
        )
    ):
        raise ValueError("Pose-release final rescue marker on a non-pose-release recipe")

    if recipe == MICROBAN_TELEOP_V12_FINAL_RESCUE_RECIPE_REVISION:
        # The final-scenario rescue (model14900 -> model14999) is consumable
        # only as its final model14999. Its marker re-validates the inherited
        # corner-rescue lineage, which stays the result for every caller.
        validate_final_rescue_consumable(infos, iteration=iteration)
        return None if marker is None else validate_corner_rescue_lineage_marker(marker)
    if recipe == MICROBAN_TELEOP_V12_CORNER_RESCUE_RECIPE_REVISION:
        if is_hand_pose_release_corner_rescue_marker(marker):
            raise ValueError("Pose-release corner marker on the canonical rescue recipe")
        if iteration != MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_ITERATION:
            raise ValueError(
                "Only final model9999 from the corner rescue is consumable"
            )
        return validate_corner_rescue_marker(infos, iteration=iteration)
    if recipe == MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION:
        from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release_lineage import (
            hand_pose_release_lineage,
        )

        hand_pose_release_lineage(
            infos,
            iteration=iteration,
            allow_experimental=allow_hand_pose_release_recipe,
        )
        # A pose-release corner rescue (its model_9999 or a descendant) carries
        # the validated pose-release marker forward; a plain chain has none.
        return None if marker is None else validate_corner_rescue_lineage_marker(marker)
    if recipe != MICROBAN_TELEOP_V12_RECIPE_REVISION:
        raise ValueError("Checkpoint recipe is neither canonical nor corner rescue")
    if marker is None:
        return None
    if is_hand_pose_release_corner_rescue_marker(marker):
        raise ValueError("Pose-release corner marker on a canonical checkpoint")
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
        trunk_pitch=existing.trunk_pitch,
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


def make_microban_teleop_v12_hand_pose_release_corner_rescue_env_cfg(
    play: bool = False,
):
    """Pose-release env with only the hand sampler changed (registered mix)."""

    from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release import (
        make_microban_teleop_v12_hand_pose_release_env_cfg,
    )

    cfg = make_microban_teleop_v12_hand_pose_release_env_cfg(play=play)
    existing = cfg.commands["hand_target"]
    cfg.commands["hand_target"] = CornerPairHandTargetCommandCfg(
        resampling_time_range=existing.resampling_time_range,
        rel_active=existing.rel_active,
        trunk_pitch=existing.trunk_pitch,
        lf_rb_probability=hand_pose_release_corner_rescue_mix(
            selected_hand_pose_release_corner_rescue_mix()
        )["lf_rb"],
    )
    if not play:
        curriculum = cfg.curriculum.get("staged_curriculum")
        stages = None if curriculum is None else curriculum.params.get("stages")
        if not isinstance(stages, list):
            raise TypeError("Corner rescue requires the v12 staged curriculum")
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


def _hand_pose_release_corner_rescue_rl_cfg():
    from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release import (
        MicrobanTeleopV12HandPoseReleaseRlCfg,
    )

    cfg = deepcopy(MicrobanTeleopV12HandPoseReleaseRlCfg)
    cfg.experiment_name = "mjlab_microban_teleop_v12"
    cfg.wandb_project = "mjlab_microban_teleop_v12_hand_pose_release_corner_rescue"
    cfg.save_interval = MICROBAN_TELEOP_V12_CORNER_RESCUE_PROCESS_UPDATES
    cfg.max_iterations = MICROBAN_TELEOP_V12_CORNER_RESCUE_PROCESS_UPDATES
    return cfg


MicrobanTeleopV12HandPoseReleaseCornerRescueRlCfg = (
    _hand_pose_release_corner_rescue_rl_cfg()
)
