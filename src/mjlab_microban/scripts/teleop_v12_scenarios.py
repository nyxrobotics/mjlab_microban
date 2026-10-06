"""Fixed teleop evaluation scenarios and rollout helpers of the v12 tracking gate.

Moved from the contract-v11 evaluator (``evaluate_teleop_checkpoint``) when that
evaluator was deleted; ``evaluate_teleop_v12_tracking`` is the only user.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import torch
from mjlab.envs import ManagerBasedRlEnv

from mjlab_microban.tasks.mdp import UniformVelocityCommandWithRotation
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_TELEOP_OBSERVATION_SCHEMA,
)
from mjlab_microban.tasks.microban_teleop_mdp import (
    HmdNeckTargetMotion,
    ResetFixedFootTargetCommand,
    ResetFixedHandTargetCommand,
)

# These are the physical command limits applied by microban's central input
# scaler and the final teleop-training curriculum.  Stationary yaw is wider
# than moving yaw, matching the runtime contract.
FORWARD_MAX_M_S = 0.7
BACKWARD_MAX_M_S = 0.5
LATERAL_MAX_M_S = 0.3
MOVING_YAW_MAX_RAD_S = 1.5
STATIONARY_YAW_MAX_RAD_S = 3.0

# Live PICO packets are limited to 80% of the training target envelope.  The
# receiver enforces the same values, so the evaluator exercises deployable
# maxima rather than unreachable wire commands.
TARGET_SAFETY_MARGIN = 0.8
FOOT_TARGET_LIMIT_M = (0.03, 0.03, 0.05)
SIMULTANEOUS_BOTH_FEET_TARGET_LIMIT_M = (0.01, 0.01, 0.02)
HAND_TARGET_LIMIT_M = (0.08, 0.08, 0.08)
CANONICAL_ACTIVE_FOOT_Z_LOWER_EDGE_M = 0.0026

# A moving-HMD stage report must demonstrate that the event was actually
# dispatched and that both its target and the physical neck moved on every
# axis.  These floors are deliberately small relative to the narrowest runtime
# range (46 degrees for neck roll), but large enough that numerical noise or a
# stale target cannot satisfy the gate.
HMD_TARGET_PEAK_TO_PEAK_MIN_RAD = 0.10
HMD_ACTUAL_PEAK_TO_PEAK_MIN_RAD = 0.05


@dataclass(frozen=True)
class EvaluationScenario:
    """One fixed command held for a complete evaluation rollout."""

    name: str
    twist: tuple[float, float, float]
    foot_target: tuple[tuple[float, float, float], tuple[float, float, float]]
    hand_target: tuple[tuple[float, float, float], tuple[float, float, float]]
    hand_active: tuple[bool, bool]


def default_scenarios() -> tuple[EvaluationScenario, ...]:
    """Return deterministic neutral, extrema and mixed deployment coverage."""

    zero = (0.0, 0.0, 0.0)
    foot_max = tuple(value * TARGET_SAFETY_MARGIN for value in FOOT_TARGET_LIMIT_M)
    both_feet_max = tuple(
        value * TARGET_SAFETY_MARGIN for value in SIMULTANEOUS_BOTH_FEET_TARGET_LIMIT_M
    )
    hand_max = tuple(value * TARGET_SAFETY_MARGIN for value in HAND_TARGET_LIMIT_M)
    foot_half = tuple(value * 0.5 for value in foot_max)
    hand_half = tuple(value * 0.5 for value in hand_max)

    return (
        EvaluationScenario("neutral", zero, (zero, zero), (zero, zero), (False, False)),
        EvaluationScenario(
            "low_forward", (0.1, 0.0, 0.0), (zero, zero), (zero, zero), (False, False)
        ),
        EvaluationScenario(
            "mid_forward", (0.2, 0.0, 0.0), (zero, zero), (zero, zero), (False, False)
        ),
        EvaluationScenario(
            "low_backward", (-0.1, 0.0, 0.0), (zero, zero), (zero, zero), (False, False)
        ),
        EvaluationScenario(
            "mid_backward", (-0.2, 0.0, 0.0), (zero, zero), (zero, zero), (False, False)
        ),
        EvaluationScenario(
            "low_lateral_left",
            (0.0, 0.1, 0.0),
            (zero, zero),
            (zero, zero),
            (False, False),
        ),
        EvaluationScenario(
            "mid_lateral_left",
            (0.0, 0.2, 0.0),
            (zero, zero),
            (zero, zero),
            (False, False),
        ),
        EvaluationScenario(
            "low_lateral_right",
            (0.0, -0.1, 0.0),
            (zero, zero),
            (zero, zero),
            (False, False),
        ),
        EvaluationScenario(
            "mid_lateral_right",
            (0.0, -0.2, 0.0),
            (zero, zero),
            (zero, zero),
            (False, False),
        ),
        EvaluationScenario(
            "low_yaw_left", (0.0, 0.0, 0.5), (zero, zero), (zero, zero), (False, False)
        ),
        EvaluationScenario(
            "mid_yaw_left", (0.0, 0.0, 1.0), (zero, zero), (zero, zero), (False, False)
        ),
        EvaluationScenario(
            "low_yaw_right",
            (0.0, 0.0, -0.5),
            (zero, zero),
            (zero, zero),
            (False, False),
        ),
        EvaluationScenario(
            "mid_yaw_right",
            (0.0, 0.0, -1.0),
            (zero, zero),
            (zero, zero),
            (False, False),
        ),
        # Exact zero is the only inactive foot representation.  These two
        # stationary cases exercise the first practical active value just above
        # the inclusive 2.5 mm support-foot floor band.
        EvaluationScenario(
            "floor_band_edge_single",
            zero,
            ((0.0, 0.0, CANONICAL_ACTIVE_FOOT_Z_LOWER_EDGE_M), zero),
            (zero, zero),
            (False, False),
        ),
        EvaluationScenario(
            "floor_band_edge_both",
            zero,
            (
                (0.0, 0.0, CANONICAL_ACTIVE_FOOT_Z_LOWER_EDGE_M),
                (0.0, 0.0, CANONICAL_ACTIVE_FOOT_Z_LOWER_EDGE_M),
            ),
            (zero, zero),
            (False, False),
        ),
        EvaluationScenario(
            "bounded_combined",
            (0.35, -0.15, 0.75),
            ((foot_half[0], -foot_half[1], foot_half[2]), zero),
            (
                (hand_half[0], -hand_half[1], hand_half[2]),
                (-hand_half[0], hand_half[1], -hand_half[2]),
            ),
            (True, True),
        ),
        EvaluationScenario(
            "max_forward",
            (FORWARD_MAX_M_S, 0.0, 0.0),
            (zero, zero),
            (zero, zero),
            (False, False),
        ),
        EvaluationScenario(
            "max_backward",
            (-BACKWARD_MAX_M_S, 0.0, 0.0),
            (zero, zero),
            (zero, zero),
            (False, False),
        ),
        EvaluationScenario(
            "max_lateral_left",
            (0.0, LATERAL_MAX_M_S, 0.0),
            (zero, zero),
            (zero, zero),
            (False, False),
        ),
        EvaluationScenario(
            "max_lateral_right",
            (0.0, -LATERAL_MAX_M_S, 0.0),
            (zero, zero),
            (zero, zero),
            (False, False),
        ),
        EvaluationScenario(
            "max_moving_yaw_left",
            (0.0, 0.0, MOVING_YAW_MAX_RAD_S),
            (zero, zero),
            (zero, zero),
            (False, False),
        ),
        EvaluationScenario(
            "max_moving_yaw_right",
            (0.0, 0.0, -MOVING_YAW_MAX_RAD_S),
            (zero, zero),
            (zero, zero),
            (False, False),
        ),
        EvaluationScenario(
            "max_stationary_yaw_left",
            (0.0, 0.0, STATIONARY_YAW_MAX_RAD_S),
            (zero, zero),
            (zero, zero),
            (False, False),
        ),
        EvaluationScenario(
            "max_stationary_yaw_right",
            (0.0, 0.0, -STATIONARY_YAW_MAX_RAD_S),
            (zero, zero),
            (zero, zero),
            (False, False),
        ),
        # Keep locomotion-only mixed commands separate from keypoint scenarios.
        # The v8 staged gate reaches mixed replay before hand/foot tracking is
        # enabled, so coupling these commands would reject a valid locomotion
        # stage for an objective that has not entered the curriculum yet.
        EvaluationScenario(
            "mixed_twist_forward_left",
            (FORWARD_MAX_M_S, LATERAL_MAX_M_S, MOVING_YAW_MAX_RAD_S),
            (zero, zero),
            (zero, zero),
            (False, False),
        ),
        EvaluationScenario(
            "mixed_twist_backward_right",
            (-BACKWARD_MAX_M_S, -LATERAL_MAX_M_S, -MOVING_YAW_MAX_RAD_S),
            (zero, zero),
            (zero, zero),
            (False, False),
        ),
        # Likewise, hand-only corners let the broad/tight hand stages be gated
        # before non-zero foot targets are introduced at iteration 16,000.
        EvaluationScenario(
            "max_hands_left",
            zero,
            (zero, zero),
            (
                (hand_max[0], -hand_max[1], hand_max[2]),
                (-hand_max[0], hand_max[1], -hand_max[2]),
            ),
            (True, True),
        ),
        EvaluationScenario(
            "max_hands_right",
            zero,
            (zero, zero),
            (
                (-hand_max[0], hand_max[1], -hand_max[2]),
                (hand_max[0], -hand_max[1], hand_max[2]),
            ),
            (True, True),
        ),
        EvaluationScenario(
            "max_keypoints_left",
            zero,
            ((foot_max[0], -foot_max[1], foot_max[2]), zero),
            (
                (hand_max[0], -hand_max[1], hand_max[2]),
                (-hand_max[0], hand_max[1], -hand_max[2]),
            ),
            (True, True),
        ),
        EvaluationScenario(
            "max_keypoints_right",
            zero,
            (zero, (-foot_max[0], foot_max[1], foot_max[2])),
            (
                (-hand_max[0], hand_max[1], -hand_max[2]),
                (hand_max[0], -hand_max[1], hand_max[2]),
            ),
            (True, True),
        ),
        # Simultaneous foot targets use the narrower stationary distribution
        # trained by v2, including the live 0.8 safety margin.
        EvaluationScenario(
            "bounded_both_feet",
            zero,
            (
                (both_feet_max[0], -both_feet_max[1], both_feet_max[2]),
                (-both_feet_max[0], both_feet_max[1], both_feet_max[2]),
            ),
            (zero, zero),
            (False, False),
        ),
        EvaluationScenario(
            "mixed_forward_left",
            (FORWARD_MAX_M_S, LATERAL_MAX_M_S, MOVING_YAW_MAX_RAD_S),
            ((foot_half[0], foot_half[1], foot_half[2]), zero),
            (
                (hand_half[0], hand_half[1], hand_half[2]),
                (-hand_half[0], -hand_half[1], -hand_half[2]),
            ),
            (True, True),
        ),
        EvaluationScenario(
            "mixed_backward_right",
            (-BACKWARD_MAX_M_S, -LATERAL_MAX_M_S, -MOVING_YAW_MAX_RAD_S),
            (zero, (-foot_half[0], -foot_half[1], foot_half[2])),
            (
                (-hand_half[0], -hand_half[1], hand_half[2]),
                (hand_half[0], hand_half[1], -hand_half[2]),
            ),
            (True, True),
        ),
    )


@dataclass
class ScalarStats:
    count: int = 0
    total: float = 0.0
    total_squared: float = 0.0
    minimum: float | None = None
    maximum: float | None = None
    samples: list[float] = field(default_factory=list, repr=False)

    def add(self, values: torch.Tensor) -> None:
        flat = values.detach().to(dtype=torch.float64).flatten()
        if flat.numel() == 0:
            return
        self.count += flat.numel()
        self.total += flat.sum().item()
        self.total_squared += torch.square(flat).sum().item()
        self.samples.extend(flat.cpu().tolist())
        value_min = flat.min().item()
        value_max = flat.max().item()
        self.minimum = (
            value_min if self.minimum is None else min(self.minimum, value_min)
        )
        self.maximum = (
            value_max if self.maximum is None else max(self.maximum, value_max)
        )

    def report(self, *, units: str) -> dict[str, float | int | str | None]:
        mean = self.total / self.count if self.count else None
        rms = math.sqrt(self.total_squared / self.count) if self.count else None
        p95 = _percentile(self.samples, 0.95)
        return {
            "sample_count": self.count,
            "mean": mean,
            "rms": rms,
            "p95": p95,
            "min": self.minimum,
            "max": self.maximum,
            "units": units,
        }


@dataclass
class HmdMotionStats:
    """Streaming target/actual motion evidence for the three HMD joints."""

    joint_names: tuple[str, ...]
    target_minimum: torch.Tensor
    target_maximum: torch.Tensor
    actual_minimum: torch.Tensor
    actual_maximum: torch.Tensor
    previous_target: torch.Tensor
    maximum_target_step: torch.Tensor
    tracking_error: tuple[ScalarStats, ...]
    sample_count: int = 1

    @classmethod
    def start(
        cls,
        *,
        joint_names: tuple[str, ...],
        target: torch.Tensor,
        actual: torch.Tensor,
    ) -> HmdMotionStats:
        target = target.detach().clone()
        actual = actual.detach().clone()
        return cls(
            joint_names=joint_names,
            target_minimum=target.clone(),
            target_maximum=target.clone(),
            actual_minimum=actual.clone(),
            actual_maximum=actual.clone(),
            previous_target=target.clone(),
            maximum_target_step=torch.zeros_like(target),
            tracking_error=tuple(ScalarStats() for _ in joint_names),
        )

    def add(self, *, target: torch.Tensor, actual: torch.Tensor) -> None:
        target = target.detach()
        actual = actual.detach()
        self.target_minimum = torch.minimum(self.target_minimum, target)
        self.target_maximum = torch.maximum(self.target_maximum, target)
        self.actual_minimum = torch.minimum(self.actual_minimum, actual)
        self.actual_maximum = torch.maximum(self.actual_maximum, actual)
        self.maximum_target_step = torch.maximum(
            self.maximum_target_step, torch.abs(target - self.previous_target)
        )
        self.previous_target = target.clone()
        for index, stats in enumerate(self.tracking_error):
            stats.add(torch.abs(target[index] - actual[index]).reshape(1))
        self.sample_count += 1

    def report(self, *, step_dt: float) -> dict[str, Any]:
        target_minimum = self.target_minimum.cpu().tolist()
        target_maximum = self.target_maximum.cpu().tolist()
        actual_minimum = self.actual_minimum.cpu().tolist()
        actual_maximum = self.actual_maximum.cpu().tolist()
        maximum_target_step = self.maximum_target_step.cpu().tolist()
        per_axis: dict[str, Any] = {}
        for index, name in enumerate(self.joint_names):
            per_axis[name] = {
                "target_min_rad": target_minimum[index],
                "target_max_rad": target_maximum[index],
                "target_peak_to_peak_rad": (
                    target_maximum[index] - target_minimum[index]
                ),
                "actual_min_rad": actual_minimum[index],
                "actual_max_rad": actual_maximum[index],
                "actual_peak_to_peak_rad": (
                    actual_maximum[index] - actual_minimum[index]
                ),
                "maximum_target_step_rad": maximum_target_step[index],
                "maximum_target_slew_rad_s": maximum_target_step[index] / step_dt,
                "absolute_tracking_error": self.tracking_error[index].report(
                    units="rad"
                ),
            }
        return {
            "active_event_member": True,
            "sample_count": self.sample_count,
            "joint_names": list(self.joint_names),
            "per_axis": per_axis,
        }


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = position - lower_index
    return ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight


def _copy_forced_moving_hmd_neck_event(training_cfg: Any) -> Any:
    """Copy only the training HMD step event and force every waypoint non-neutral."""

    source = training_cfg.events.get("hmd_neck_target_motion")
    if source is None:
        raise ValueError("Teleop training config is missing hmd_neck_target_motion")
    if source.mode != "step" or source.func is not HmdNeckTargetMotion:
        raise TypeError("Training HMD event must be the HmdNeckTargetMotion step term")
    copied = deepcopy(source)
    copied.params["neutral_probability"] = 0.0
    return copied


def _configure_nominal_evaluation(
    cfg: Any, *, steps: int, moving_hmd_neck_event: Any | None = None
) -> None:
    """Keep nominal reset events and optionally one forced moving-HMD step event."""

    cfg.scene.num_envs = 1
    cfg.auto_reset = False
    # Make the requested evaluation horizon the time limit.  A healthy scenario
    # therefore reaches ``time_out`` exactly on its final requested step.
    cfg.episode_length_s = steps * cfg.decimation * cfg.sim.mujoco.timestep

    reset_events = {
        name: term for name, term in cfg.events.items() if term.mode == "reset"
    }
    if moving_hmd_neck_event is not None:
        if (
            moving_hmd_neck_event.mode != "step"
            or moving_hmd_neck_event.func is not HmdNeckTargetMotion
            or moving_hmd_neck_event.params.get("neutral_probability") != 0.0
        ):
            raise ValueError(
                "Moving-HMD evaluation requires a non-neutral "
                "HmdNeckTargetMotion step event"
            )
        reset_events["hmd_neck_target_motion"] = moving_hmd_neck_event
    cfg.events = reset_events
    reset_base = cfg.events.get("reset_base")
    if reset_base is None:
        raise ValueError("Teleop task is missing its reset_base event")
    reset_base.params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    reset_base.params["velocity_range"] = {}

    # Play mode already disables these, but assign them explicitly so copying
    # the one training event can never bring training corruption/curriculum with it.
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    if moving_hmd_neck_event is None and "hmd_neck_target_motion" in cfg.events:
        raise AssertionError("Nominal evaluation must hold the HMD neck fixed")


def _set_scenario(env: ManagerBasedRlEnv, scenario: EvaluationScenario) -> None:
    twist = env.command_manager.get_term("twist")
    foot = env.command_manager.get_term("foot_target")
    hand = env.command_manager.get_term("hand_target")
    if not isinstance(twist, UniformVelocityCommandWithRotation):
        raise TypeError(f"Unexpected twist command type: {type(twist).__name__}")
    if not isinstance(foot, ResetFixedFootTargetCommand):
        raise TypeError(f"Unexpected foot command type: {type(foot).__name__}")
    if not isinstance(hand, ResetFixedHandTargetCommand):
        raise TypeError(f"Unexpected hand command type: {type(hand).__name__}")
    if foot._reference_pending.any() or hand._reference_pending.any():
        raise RuntimeError("Keypoint reset reference was not captured before injection")

    twist_value = torch.tensor(scenario.twist, device=env.device).unsqueeze(0)
    twist.vel_command_b.copy_(twist_value)
    twist.vel_command_w.copy_(twist_value)
    for flag_name in (
        "is_heading_env",
        "is_standing_env",
        "is_world_env",
        "is_forward_env",
        "is_rotation_env",
    ):
        getattr(twist, flag_name).fill_(False)
    twist.time_left.fill_(float("inf"))

    foot_value = torch.tensor(scenario.foot_target, device=env.device).unsqueeze(0)
    foot.foot_target_offset_b.copy_(foot_value)
    foot.is_single_support_env.copy_(foot_value.norm(dim=-1).gt(0.0).any(dim=-1))
    foot.lifted_foot_idx.copy_(foot_value.norm(dim=-1).argmax(dim=-1))
    foot.time_left.fill_(float("inf"))

    hand_value = torch.tensor(scenario.hand_target, device=env.device).unsqueeze(0)
    active_value = torch.tensor(scenario.hand_active, device=env.device).unsqueeze(0)
    hand.hand_target_offset_b.copy_(hand_value)
    hand.is_active.copy_(active_value)
    hand.time_left.fill_(float("inf"))


def _patch_initial_command_observation(
    observations: Any, env: ManagerBasedRlEnv
) -> Any:
    """Replace cached reset-time command slices without advancing delay buffers."""

    command_values = {
        "command": env.command_manager.get_term("twist").command,
        "foot_target": env.command_manager.get_term("foot_target").command,
        "hand_target": env.command_manager.get_term("hand_target").command,
    }
    patched = observations.clone()
    actor = patched["actor"].clone()
    offset = 0
    for name, width in MICROBAN_TELEOP_OBSERVATION_SCHEMA:
        if name in command_values:
            value = command_values[name]
            if value.shape != (env.num_envs, width):
                raise ValueError(
                    f"Unexpected {name} observation shape: {tuple(value.shape)}"
                )
            actor[:, offset : offset + width] = value
        offset += width
    patched["actor"] = actor
    return patched
