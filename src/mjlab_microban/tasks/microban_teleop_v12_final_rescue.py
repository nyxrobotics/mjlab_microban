"""Recorded-parent final-scenario rescue for the 15000 contract-v12 boundary.

The corner rescue (``microban_teleop_v12_corner_rescue``) replays the failing
bilateral hand corner for 99 updates before the 10000 gate.  This module is the
same machinery for the final segment: 99 PPO updates from a canonical
``model_14900`` (completed update 14901) to ``model_14999`` (completed 15000),
with only the command samplers changed.

In a fixed fraction of episodes (90 % in total, split by a registered mix) each
environment replays one failing final-gate scenario exactly as the tracking
evaluator defines it (``evaluate_teleop_v12_tracking._scenarios`` under the
final profile): the twist, both foot targets and both hand targets of
``mixed_backward_right`` or ``max_keypoints_right``.  The remaining 10 % keep
the ordinary samplers.  A scenario is drawn once per episode and shared by the
twist, foot and hand commands, so every timer resample inside that episode
re-applies the same scenario (the evaluator holds it fixed for 300 steps).

Actor, optimizer, learning rate, rewards, normalizer, action semantics and the
frozen legacy tensors are unchanged.  The parent checkpoint, its final-profile
tracking report and the failed final-gate tracking report that triggered the
rescue are recorded in the marker; the runner re-hashes all three on load and
every consumer rebuilds the marker from the recorded values.  Only the final
``model_14999`` of the rescue is consumable, exactly like the corner rescue's
``model_9999``.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, fields
from functools import lru_cache
from typing import Any

import torch

from mjlab_microban.robot.microban_hand_fk import (
    MICROBAN_REACHABLE_HAND_EVALUATION_JOINTS_DEG,
    microban_hand_offsets_from_arm_joints,
)
from mjlab_microban.tasks.mdp import (
    UniformVelocityCommandWithRotation,
    UniformVelocityCommandWithRotationCfg,
)
from mjlab_microban.tasks.microban_teleop_env_cfg import (
    MICROBAN_TELEOP_FOOT_TRACKING_FINAL_STD_M,
    MICROBAN_TELEOP_HAND_TRACKING_FINAL_STD_M,
)
from mjlab_microban.tasks.microban_teleop_mdp import (
    ResetFixedFootTargetCommand,
    ResetFixedFootTargetCommandCfg,
    ResetFixedHandTargetCommand,
    ResetFixedHandTargetCommandCfg,
)
from mjlab_microban.tasks.microban_teleop_v12_actor import (
    TELEOP_V12_ADAPTER_GRADIENT_SCHEDULE_REVISION,
    TELEOP_V12_EXTRA_OBSERVATION_COLUMNS,
)
from mjlab_microban.tasks.microban_teleop_v12_corner_rescue import (
    _require_sha256,
    canonical_json_sha256,
    validate_corner_rescue_lineage_marker,
)
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MICROBAN_TELEOP_V12_ACTION_CLIP,
    MICROBAN_TELEOP_V12_RECIPE_REVISION,
    MicrobanTeleopV12RlCfg,
    make_microban_teleop_v12_env_cfg,
)

MICROBAN_TELEOP_V12_FINAL_RESCUE_TASK_ID = "Mjlab-Teleop-V12-Final-Rescue-Microban"
MICROBAN_TELEOP_V12_FINAL_RESCUE_INFO_KEY = "microban_teleop_v12_final_scenario_rescue"
MICROBAN_TELEOP_V12_FINAL_RESCUE_RECIPE_REVISION = (
    "model14900_targeted_final_scenario_replay_to15000_v1"
)
MICROBAN_TELEOP_V12_FINAL_RESCUE_MARKER_REVISION = (
    "recorded_model14900_ordinary10_final_scenarios90_99_updates_v1"
)
MICROBAN_TELEOP_V12_FINAL_RESCUE_SAMPLER_REVISION = (
    "episode_shared_twist_foot_hand_evaluator_scenario_replay_v1"
)
# The launcher stages the two reports next to the staged parent checkpoint.
MICROBAN_TELEOP_V12_FINAL_RESCUE_PARENT_REPORT_FILENAME = (
    "parent_final_profile_tracking.json"
)
MICROBAN_TELEOP_V12_FINAL_RESCUE_FAILED_GATE_REPORT_FILENAME = (
    "failed_final_gate_tracking.json"
)
MICROBAN_TELEOP_V12_FINAL_RESCUE_MIX_ENV = "MICROBAN_V12_FINAL_RESCUE_MIX"
MICROBAN_TELEOP_V12_FINAL_RESCUE_DEFAULT_MIX = "v1"

MICROBAN_TELEOP_V12_FINAL_RESCUE_PARENT_ITERATION = 14_900
MICROBAN_TELEOP_V12_FINAL_RESCUE_PARENT_COMPLETED_UPDATES = 14_901
MICROBAN_TELEOP_V12_FINAL_RESCUE_TARGET_ITERATION = 14_999
MICROBAN_TELEOP_V12_FINAL_RESCUE_TARGET_COMPLETED_UPDATES = 15_000
MICROBAN_TELEOP_V12_FINAL_RESCUE_PROCESS_UPDATES = 99
MICROBAN_TELEOP_V12_FINAL_RESCUE_NUM_STEPS_PER_ENV = 24
MICROBAN_TELEOP_V12_FINAL_RESCUE_OPTIMIZER_STEPS_PER_UPDATE = 20
MICROBAN_TELEOP_V12_FINAL_RESCUE_PARENT_COMMON_STEP = (
    MICROBAN_TELEOP_V12_FINAL_RESCUE_PARENT_COMPLETED_UPDATES
    * MICROBAN_TELEOP_V12_FINAL_RESCUE_NUM_STEPS_PER_ENV
)
MICROBAN_TELEOP_V12_FINAL_RESCUE_TARGET_COMMON_STEP = (
    MICROBAN_TELEOP_V12_FINAL_RESCUE_TARGET_COMPLETED_UPDATES
    * MICROBAN_TELEOP_V12_FINAL_RESCUE_NUM_STEPS_PER_ENV
)
MICROBAN_TELEOP_V12_FINAL_RESCUE_PARENT_OPTIMIZER_STEP = (
    MICROBAN_TELEOP_V12_FINAL_RESCUE_PARENT_COMPLETED_UPDATES
    * MICROBAN_TELEOP_V12_FINAL_RESCUE_OPTIMIZER_STEPS_PER_UPDATE
)
MICROBAN_TELEOP_V12_FINAL_RESCUE_TARGET_OPTIMIZER_STEP = (
    MICROBAN_TELEOP_V12_FINAL_RESCUE_TARGET_COMPLETED_UPDATES
    * MICROBAN_TELEOP_V12_FINAL_RESCUE_OPTIMIZER_STEPS_PER_UPDATE
)
MICROBAN_TELEOP_V12_FINAL_RESCUE_ACTIVE_COLUMNS = tuple(
    TELEOP_V12_EXTRA_OBSERVATION_COLUMNS
)

# Pattern 0 keeps the ordinary samplers; pattern i >= 1 replays SCENARIOS[i-1].
MICROBAN_TELEOP_V12_FINAL_RESCUE_SCENARIOS = (
    "mixed_backward_right",
    "max_keypoints_right",
)
# Left/right named reachable arm poses whose FK offsets the evaluator uses as the
# hand targets of each scenario (``_scenarios``: left from one name, right from
# the other, right roll mirrored).
_SCENARIO_HAND_POSE_NAMES = {
    "mixed_backward_right": ("b", "f"),
    "max_keypoints_right": ("B", "F"),
}
MICROBAN_TELEOP_V12_FINAL_RESCUE_ORDINARY_PROBABILITY = 0.10
# Registered mixes.  v1-v3 keep the corner rescue's 10 % ordinary share; the
# launcher selects one by name and the marker records its probabilities.
MICROBAN_TELEOP_V12_FINAL_RESCUE_MIXES: dict[str, dict[str, float]] = {
    "v1": {"mixed_backward_right": 0.50, "max_keypoints_right": 0.40},
    "v2": {"mixed_backward_right": 0.65, "max_keypoints_right": 0.25},
    "v3": {"mixed_backward_right": 0.35, "max_keypoints_right": 0.55},
    "v4": {"mixed_backward_right": 0.30, "max_keypoints_right": 0.20},
    "v5": {"mixed_backward_right": 0.20, "max_keypoints_right": 0.10},
}
# v1 (90 % replay) made mixed_backward_right worse (hand RMS 0.039 -> 0.058)
# and lost push robustness (falls), so v4/v5 keep half or more of the
# ordinary command distribution.
MICROBAN_TELEOP_V12_FINAL_RESCUE_MIX_ORDINARY_PROBABILITY: dict[str, float] = {
    "v4": 0.50,
    "v5": 0.70,
}


def _mix_ordinary_probability(name: str) -> float:
    return MICROBAN_TELEOP_V12_FINAL_RESCUE_MIX_ORDINARY_PROBABILITY.get(
        name, MICROBAN_TELEOP_V12_FINAL_RESCUE_ORDINARY_PROBABILITY
    )
# A parent/failed-gate report may fail only accuracy checks; every safety,
# coverage, ablation and locomotion check must pass.
MICROBAN_TELEOP_V12_FINAL_RESCUE_RESCUABLE_CHECKS = frozenset(
    (
        "hand_tracking_rms",
        "hand_tracking_p95",
        "foot_tracking_rms",
        "foot_tracking_p95",
    )
)

for _name, _mix in MICROBAN_TELEOP_V12_FINAL_RESCUE_MIXES.items():
    if tuple(_mix) != MICROBAN_TELEOP_V12_FINAL_RESCUE_SCENARIOS or not math.isclose(
        sum(_mix.values()) + _mix_ordinary_probability(_name),
        1.0,
    ):
        raise RuntimeError(f"Final rescue mix {_name} is malformed")
if (
    MICROBAN_TELEOP_V12_FINAL_RESCUE_TARGET_COMPLETED_UPDATES
    - MICROBAN_TELEOP_V12_FINAL_RESCUE_PARENT_COMPLETED_UPDATES
    != MICROBAN_TELEOP_V12_FINAL_RESCUE_PROCESS_UPDATES
):
    raise RuntimeError("Final rescue must contain exactly 99 PPO updates")


def validate_final_rescue_mix(name: object) -> str:
    if not isinstance(name, str) or name not in MICROBAN_TELEOP_V12_FINAL_RESCUE_MIXES:
        raise ValueError(
            "Final rescue mix must be one of "
            f"{sorted(MICROBAN_TELEOP_V12_FINAL_RESCUE_MIXES)}: {name!r}"
        )
    return name


def final_rescue_mix_probabilities(name: str) -> dict[str, float]:
    """Return ordinary + per-scenario probabilities of one registered mix."""

    mix = MICROBAN_TELEOP_V12_FINAL_RESCUE_MIXES[validate_final_rescue_mix(name)]
    return {
        "ordinary": _mix_ordinary_probability(name),
        **mix,
    }


@lru_cache(maxsize=1)
def final_rescue_scenario_commands() -> dict[str, dict[str, Any]]:
    """Return the evaluator's exact final-profile commands for both scenarios.

    Twist and foot targets are read from the tracking evaluator itself; the hand
    targets are trained through the named arm joint tuples whose FK offsets the
    evaluator uses, and they are checked against the evaluator's offsets.
    """

    # Lazy import: the evaluator imports the task package.
    from mjlab_microban.scripts.evaluate_teleop_v12_tracking import (
        FINAL_DEPLOYED_ACCURACY_PROFILE,
        _scenarios,
    )

    by_name = {item.name: item for item in _scenarios(FINAL_DEPLOYED_ACCURACY_PROFILE)}
    poses = dict(MICROBAN_REACHABLE_HAND_EVALUATION_JOINTS_DEG)
    result: dict[str, dict[str, Any]] = {}
    for name in MICROBAN_TELEOP_V12_FINAL_RESCUE_SCENARIOS:
        scenario = by_name[name]
        left_name, right_name = _SCENARIO_HAND_POSE_NAMES[name]
        left = poses[left_name]
        right = poses[right_name]
        joints_deg = [
            [float(left[0]), float(left[1]), float(left[2])],
            [float(right[0]), -float(right[1]), float(right[2])],
        ]
        offsets = microban_hand_offsets_from_arm_joints(
            torch.deg2rad(torch.tensor(joints_deg, dtype=torch.float64))
        )
        expected = torch.tensor(scenario.hand_target, dtype=torch.float64)
        if not torch.allclose(offsets, expected, atol=1.0e-9, rtol=0.0):
            raise RuntimeError(f"Final rescue hand joints drifted from {name}")
        if tuple(scenario.hand_active) != (True, True):
            raise RuntimeError(f"Final rescue scenario {name} must use both hands")
        result[name] = {
            "twist": [float(value) for value in scenario.twist],
            "foot_target": [
                [float(value) for value in xyz] for xyz in scenario.foot_target
            ],
            "hand_target": [
                [float(value) for value in xyz] for xyz in scenario.hand_target
            ],
            "hand_active": [True, True],
            "hand_joint_pose_names": [left_name, right_name],
            "hand_joint_deg": joints_deg,
        }
    return result


class _FinalRescuePatternState:
    """One scenario id per environment, redrawn once per episode reset.

    Every participating command term calls :meth:`on_reset` from its own
    ``reset``.  The first term of a reset event advances the environment's
    generation and redraws; the remaining terms see that generation and reuse
    the draw, so the order of terms inside the command manager is irrelevant.
    """

    def __init__(self, *, num_envs: int, device: torch.device | str, mix: str):
        self.mix = validate_final_rescue_mix(mix)
        probabilities = final_rescue_mix_probabilities(mix)
        values = torch.tensor(
            [
                probabilities["ordinary"],
                *(probabilities[name] for name in MICROBAN_TELEOP_V12_FINAL_RESCUE_SCENARIOS),
            ],
            dtype=torch.float64,
        )
        cumulative = torch.cumsum(values, dim=0)
        cumulative[-1] = 1.0
        self._cumulative = cumulative.to(device=device, dtype=torch.float64)
        self.device = device
        self.num_envs = num_envs
        self.generation = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._seen: dict[str, torch.Tensor] = {}
        self.pattern = self._draw(num_envs)

    def _draw(self, count: int) -> torch.Tensor:
        draws = torch.rand(count, device=self.device, dtype=torch.float64)
        return torch.searchsorted(self._cumulative, draws, right=True).clamp_(
            max=len(MICROBAN_TELEOP_V12_FINAL_RESCUE_SCENARIOS)
        )

    def on_reset(self, term_key: str, env_ids: torch.Tensor) -> None:
        if not isinstance(env_ids, torch.Tensor):
            raise TypeError("Final rescue reset requires explicit environment IDs")
        seen = self._seen.get(term_key)
        if seen is None:
            seen = torch.full_like(self.generation, -1)
            self._seen[term_key] = seen
        stale = seen[env_ids] == self.generation[env_ids]
        redraw = env_ids[stale]
        if len(redraw) > 0:
            self.generation[redraw] += 1
            self.pattern[redraw] = self._draw(len(redraw))
        seen[env_ids] = self.generation[env_ids]

    def replay(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (replayed env ids, zero-based scenario index per id)."""

        pattern = self.pattern[env_ids]
        selected = pattern != 0
        return env_ids[selected], pattern[selected] - 1


_PATTERN_STATE_ATTRIBUTE = "_microban_v12_final_rescue_pattern_state"


def final_rescue_pattern_state(env: Any, mix: str) -> _FinalRescuePatternState:
    state = getattr(env, _PATTERN_STATE_ATTRIBUTE, None)
    if state is None:
        state = _FinalRescuePatternState(
            num_envs=int(env.num_envs), device=env.device, mix=mix
        )
        setattr(env, _PATTERN_STATE_ATTRIBUTE, state)
    elif state.mix != mix:
        raise ValueError("Final rescue command terms disagree on the sampler mix")
    return state


def _scenario_tensor(key: str, *, device: Any, dtype: torch.dtype) -> torch.Tensor:
    commands = final_rescue_scenario_commands()
    return torch.tensor(
        [commands[name][key] for name in MICROBAN_TELEOP_V12_FINAL_RESCUE_SCENARIOS],
        device=device,
        dtype=dtype,
    )


class FinalRescueTwistCommand(UniformVelocityCommandWithRotation):
    """Ordinary twist sampler plus the evaluator's scenario twist replay."""

    cfg: FinalRescueTwistCommandCfg

    def __init__(self, cfg: FinalRescueTwistCommandCfg, env: Any):
        super().__init__(cfg, env)
        self._final_rescue = final_rescue_pattern_state(env, cfg.final_rescue_mix)
        self._scenario_twist = _scenario_tensor(
            "twist", device=self.device, dtype=self.vel_command_b.dtype
        )

    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        self._final_rescue.on_reset("twist", env_ids)  # type: ignore[arg-type]
        return super().reset(env_ids)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        ids, scenario = self._final_rescue.replay(env_ids)
        if len(ids) == 0:
            return
        value = self._scenario_twist[scenario]
        # Same writes as the evaluator's ``_set_scenario``.
        self.vel_command_b[ids] = value
        self.vel_command_w[ids] = value
        for flag_name in (
            "is_heading_env",
            "is_standing_env",
            "is_world_env",
            "is_forward_env",
            "is_rotation_env",
        ):
            getattr(self, flag_name)[ids] = False


class FinalRescueFootTargetCommand(ResetFixedFootTargetCommand):
    """Ordinary foot sampler plus the evaluator's scenario foot replay."""

    cfg: FinalRescueFootTargetCommandCfg

    def __init__(self, cfg: FinalRescueFootTargetCommandCfg, env: Any):
        super().__init__(cfg, env)
        self._final_rescue = final_rescue_pattern_state(env, cfg.final_rescue_mix)
        self._scenario_foot = _scenario_tensor(
            "foot_target", device=self.device, dtype=self.foot_target_offset_b.dtype
        )

    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        self._final_rescue.on_reset("foot_target", env_ids)  # type: ignore[arg-type]
        return super().reset(env_ids)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        ids, scenario = self._final_rescue.replay(env_ids)
        if len(ids) == 0:
            return
        value = self._scenario_foot[scenario]
        norms = value.norm(dim=-1)
        self.foot_target_offset_b[ids] = value
        self.is_single_support_env[ids] = norms.gt(0.0).any(dim=-1)
        self.lifted_foot_idx[ids] = norms.argmax(dim=-1)
        self.is_both_feet_env[ids] = False


class FinalRescueHandTargetCommand(ResetFixedHandTargetCommand):
    """Ordinary hand sampler plus the evaluator's scenario hand replay."""

    cfg: FinalRescueHandTargetCommandCfg

    def __init__(self, cfg: FinalRescueHandTargetCommandCfg, env: Any):
        super().__init__(cfg, env)
        self._final_rescue = final_rescue_pattern_state(env, cfg.final_rescue_mix)
        dtype = self.hand_target_offset_b.dtype
        self._scenario_joints = torch.deg2rad(
            _scenario_tensor("hand_joint_deg", device=self.device, dtype=dtype)
        )
        self._scenario_offsets = microban_hand_offsets_from_arm_joints(
            self._scenario_joints
        )

    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        self._final_rescue.on_reset("hand_target", env_ids)  # type: ignore[arg-type]
        return super().reset(env_ids)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        ids, scenario = self._final_rescue.replay(env_ids)
        if len(ids) == 0:
            return
        self.is_active[ids] = True
        self.sampled_arm_joint_pos_rad[ids] = self._scenario_joints[scenario]
        self.hand_target_offset_b[ids] = self._scenario_offsets[scenario]


@dataclass(kw_only=True)
class FinalRescueTwistCommandCfg(UniformVelocityCommandWithRotationCfg):
    final_rescue_mix: str = MICROBAN_TELEOP_V12_FINAL_RESCUE_DEFAULT_MIX

    def build(self, env: Any) -> FinalRescueTwistCommand:
        return FinalRescueTwistCommand(self, env)


@dataclass(kw_only=True)
class FinalRescueFootTargetCommandCfg(ResetFixedFootTargetCommandCfg):
    final_rescue_mix: str = MICROBAN_TELEOP_V12_FINAL_RESCUE_DEFAULT_MIX

    def build(self, env: Any) -> FinalRescueFootTargetCommand:
        return FinalRescueFootTargetCommand(self, env)


@dataclass(kw_only=True)
class FinalRescueHandTargetCommandCfg(ResetFixedHandTargetCommandCfg):
    final_rescue_mix: str = MICROBAN_TELEOP_V12_FINAL_RESCUE_DEFAULT_MIX

    def build(self, env: Any) -> FinalRescueHandTargetCommand:
        return FinalRescueHandTargetCommand(self, env)


def _rebased_cfg(existing: object, cls: type, mix: str) -> Any:
    """Copy every init field of ``existing`` into the rescue subclass."""

    values: dict[str, Any] = {}
    for field in fields(cls):
        if not field.init or field.name == "final_rescue_mix":
            continue
        if not hasattr(existing, field.name):
            raise TypeError(f"Final rescue cannot copy command field {field.name}")
        values[field.name] = deepcopy(getattr(existing, field.name))
    return cls(**values, final_rescue_mix=mix)


def make_microban_teleop_v12_final_rescue_env_cfg(
    play: bool = False, mix: str | None = None
):
    """Build the v12 task with only the three command samplers changed."""

    cfg = make_microban_teleop_v12_env_cfg(play=play)
    if play:
        # Play/inspection keeps the canonical deterministic commands.
        return cfg
    mix = validate_final_rescue_mix(
        mix
        if mix is not None
        else os.environ.get(
            MICROBAN_TELEOP_V12_FINAL_RESCUE_MIX_ENV,
            MICROBAN_TELEOP_V12_FINAL_RESCUE_DEFAULT_MIX,
        )
    )
    for name, cls in (
        ("twist", FinalRescueTwistCommandCfg),
        ("foot_target", FinalRescueFootTargetCommandCfg),
        ("hand_target", FinalRescueHandTargetCommandCfg),
    ):
        cfg.commands[name] = _rebased_cfg(cfg.commands[name], cls, mix)
    curriculum = cfg.curriculum.get("staged_curriculum")
    stages = None if curriculum is None else curriculum.params.get("stages")
    if not isinstance(stages, list):
        raise TypeError("Final rescue requires the v12 staged curriculum")
    # The process stops exactly at the 15000 boundary, like the ordinary
    # segment's last update; no later stage may be reconstructed.
    stages[:] = [
        stage
        for stage in stages
        if int(stage.get("step", -1))
        < MICROBAN_TELEOP_V12_FINAL_RESCUE_TARGET_COMMON_STEP
    ]
    return cfg


def _sorted_failed_checks(value: object, *, label: str) -> list[str]:
    if (
        not isinstance(value, (list, tuple))
        or any(not isinstance(item, str) for item in value)
        or list(value) != sorted(set(value))
        or not set(value).issubset(MICROBAN_TELEOP_V12_FINAL_RESCUE_RESCUABLE_CHECKS)
    ):
        raise ValueError(f"{label} failed checks must be sorted accuracy checks")
    return list(value)


def final_rescue_marker(
    *,
    parent_checkpoint_sha256: str,
    parent_tracking_report_sha256: str,
    parent_failed_checks: Sequence[str],
    failed_gate_checkpoint_sha256: str,
    failed_gate_tracking_report_sha256: str,
    failed_gate_failed_checks: Sequence[str],
    inherited_corner_rescue_marker_sha256: str | None,
    sampler_mix: str,
) -> dict[str, Any]:
    """Return the marker embedded in every final-rescue checkpoint.

    Only the recorded hashes, failed-check lists and the mix name vary; every
    other field is fixed by code.
    """

    from mjlab_microban.scripts.evaluate_teleop_v12_tracking import (
        FINAL_DEPLOYED_ACCURACY_PROFILE,
        required_tracking_profile,
    )

    failed_gate_checks = _sorted_failed_checks(
        failed_gate_failed_checks, label="Failed final gate"
    )
    if not failed_gate_checks:
        raise ValueError("The failed final gate must fail at least one check")
    return {
        "schema_version": 1,
        "revision": MICROBAN_TELEOP_V12_FINAL_RESCUE_MARKER_REVISION,
        "parent_identity": "recorded_and_rehashed_on_load",
        "parent_checkpoint_sha256": _require_sha256(
            parent_checkpoint_sha256, "Final rescue parent checkpoint"
        ),
        "parent_tracking_report_sha256": _require_sha256(
            parent_tracking_report_sha256, "Final rescue parent report"
        ),
        # Like the corner rescue: the parent's report is judged under the
        # canonical profile of the parent's own clock (whole body at 14901).
        "parent_tracking_profile": required_tracking_profile(
            MICROBAN_TELEOP_V12_FINAL_RESCUE_PARENT_COMPLETED_UPDATES
        ),
        "parent_failed_checks": _sorted_failed_checks(
            parent_failed_checks, label="Final rescue parent"
        ),
        "failed_final_gate": {
            "checkpoint_sha256": _require_sha256(
                failed_gate_checkpoint_sha256, "Failed final gate checkpoint"
            ),
            "iteration": MICROBAN_TELEOP_V12_FINAL_RESCUE_TARGET_ITERATION,
            "completed_updates": (
                MICROBAN_TELEOP_V12_FINAL_RESCUE_TARGET_COMPLETED_UPDATES
            ),
            "tracking_profile": FINAL_DEPLOYED_ACCURACY_PROFILE,
            "tracking_report_sha256": _require_sha256(
                failed_gate_tracking_report_sha256, "Failed final gate report"
            ),
            "failed_checks": failed_gate_checks,
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
            "environment_seed": 42,
            "agent_seed": 42,
            "num_envs": 2_048,
            "num_steps_per_env": 24,
        },
        "source_recipe_revision": MICROBAN_TELEOP_V12_RECIPE_REVISION,
        "rescue_recipe_revision": MICROBAN_TELEOP_V12_FINAL_RESCUE_RECIPE_REVISION,
        "sampler_revision": MICROBAN_TELEOP_V12_FINAL_RESCUE_SAMPLER_REVISION,
        "sampler_mix": validate_final_rescue_mix(sampler_mix),
        "sampler_probabilities": final_rescue_mix_probabilities(sampler_mix),
        "sampler_scope": "per_episode_shared_by_twist_foot_hand_commands",
        "scenario_commands": deepcopy(final_rescue_scenario_commands()),
        "adapter_gradient_schedule_revision": (
            TELEOP_V12_ADAPTER_GRADIENT_SCHEDULE_REVISION
        ),
        "active_actor_columns": list(MICROBAN_TELEOP_V12_FINAL_RESCUE_ACTIVE_COLUMNS),
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
        },
    }


def validate_final_rescue_lineage_marker(marker: object) -> dict[str, Any]:
    """Rebuild the marker from its recorded values and require equality."""

    if not isinstance(marker, Mapping):
        raise ValueError("Final rescue lineage marker drifted")
    failed_gate = marker.get("failed_final_gate")
    if not isinstance(failed_gate, Mapping):
        raise ValueError("Final rescue lineage marker drifted")
    try:
        expected = final_rescue_marker(
            parent_checkpoint_sha256=marker.get("parent_checkpoint_sha256"),  # type: ignore[arg-type]
            parent_tracking_report_sha256=marker.get(  # type: ignore[arg-type]
                "parent_tracking_report_sha256"
            ),
            parent_failed_checks=marker.get("parent_failed_checks"),  # type: ignore[arg-type]
            failed_gate_checkpoint_sha256=failed_gate.get("checkpoint_sha256"),  # type: ignore[arg-type]
            failed_gate_tracking_report_sha256=failed_gate.get(  # type: ignore[arg-type]
                "tracking_report_sha256"
            ),
            failed_gate_failed_checks=failed_gate.get("failed_checks"),  # type: ignore[arg-type]
            inherited_corner_rescue_marker_sha256=marker.get(  # type: ignore[arg-type]
                "inherited_corner_rescue_marker_sha256"
            ),
            sampler_mix=marker.get("sampler_mix"),  # type: ignore[arg-type]
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Final rescue lineage marker drifted") from exc
    if dict(marker) != expected:
        raise ValueError("Final rescue lineage marker drifted")
    return deepcopy(expected)


def assert_final_rescue_optimizer_step(
    payload: Mapping[str, Any], *, expected_step: int
) -> None:
    # Same exact single-clock check as the corner rescue.
    from mjlab_microban.tasks.microban_teleop_v12_corner_rescue import (
        assert_corner_rescue_optimizer_step,
    )

    assert_corner_rescue_optimizer_step(payload, expected_step=expected_step)


def validate_final_rescue_checkpoint(
    infos: Mapping[str, Any], *, iteration: int
) -> dict[str, Any]:
    """Validate a final-rescue save (any clock inside 14901..14999)."""

    if not isinstance(infos, Mapping):
        raise TypeError("Final rescue checkpoint infos must be a mapping")
    if infos.get("microban_teleop_training_contract_version") != "12":
        raise ValueError("Final rescue checkpoint is not contract-v12")
    if infos.get("microban_teleop_recipe_revision") != (
        MICROBAN_TELEOP_V12_FINAL_RESCUE_RECIPE_REVISION
    ):
        raise ValueError("Final rescue recipe revision drifted")
    if infos.get("adapter_gradient_schedule_revision") != (
        TELEOP_V12_ADAPTER_GRADIENT_SCHEDULE_REVISION
    ):
        raise ValueError("Final rescue adapter schedule drifted")
    if not (
        MICROBAN_TELEOP_V12_FINAL_RESCUE_PARENT_ITERATION
        < iteration
        <= MICROBAN_TELEOP_V12_FINAL_RESCUE_TARGET_ITERATION
    ):
        raise ValueError("Final rescue checkpoint clock is outside 14901..14999")
    env_state = infos.get("env_state")
    expected_step = (iteration + 1) * MICROBAN_TELEOP_V12_FINAL_RESCUE_NUM_STEPS_PER_ENV
    if (
        not isinstance(env_state, Mapping)
        or env_state.get("common_step_counter") != expected_step
    ):
        raise ValueError("Final rescue iteration/common-step relation drifted")
    if infos.get("active_actor_columns_at_save") != list(
        MICROBAN_TELEOP_V12_FINAL_RESCUE_ACTIVE_COLUMNS
    ):
        raise ValueError("Final rescue active actor columns drifted")
    from mjlab_microban.tasks.microban_teleop_v12_deadline_fallback import (
        MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_INFO_KEY,
    )

    if infos.get(MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_INFO_KEY) is not None:
        raise ValueError("Final rescue cannot carry deadline-fallback lineage")
    marker = validate_final_rescue_lineage_marker(
        infos.get(MICROBAN_TELEOP_V12_FINAL_RESCUE_INFO_KEY)
    )
    from mjlab_microban.tasks.microban_teleop_v12_corner_rescue import (
        MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY,
    )

    corner = infos.get(MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY)
    corner_sha = (
        None
        if corner is None
        else canonical_json_sha256(validate_corner_rescue_lineage_marker(corner))
    )
    if marker["inherited_corner_rescue_marker_sha256"] != corner_sha:
        raise ValueError("Final rescue inherited corner-rescue lineage drifted")
    return marker


def validate_final_rescue_consumable(
    infos: Mapping[str, Any], *, iteration: int
) -> dict[str, Any]:
    """Accept only the final ``model_14999`` of the rescue (15000 completed)."""

    if iteration != MICROBAN_TELEOP_V12_FINAL_RESCUE_TARGET_ITERATION:
        raise ValueError("Only final model14999 from the final rescue is consumable")
    return validate_final_rescue_checkpoint(infos, iteration=iteration)


MicrobanTeleopV12FinalRescueRlCfg = deepcopy(MicrobanTeleopV12RlCfg)
MicrobanTeleopV12FinalRescueRlCfg.experiment_name = "mjlab_microban_teleop_v12"
MicrobanTeleopV12FinalRescueRlCfg.wandb_project = (
    "mjlab_microban_teleop_v12_final_rescue"
)
MicrobanTeleopV12FinalRescueRlCfg.save_interval = (
    MICROBAN_TELEOP_V12_FINAL_RESCUE_PROCESS_UPDATES
)
MicrobanTeleopV12FinalRescueRlCfg.max_iterations = (
    MICROBAN_TELEOP_V12_FINAL_RESCUE_PROCESS_UPDATES
)
