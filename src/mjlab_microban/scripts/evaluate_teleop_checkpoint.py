# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Headless deterministic safety evaluation for a Microban PICO checkpoint.

The evaluator never opens a viewer or a robot/network connection.  It loads one
saved ``Mjlab-Teleop-Microban`` checkpoint, disables training randomization and
runs a fixed set of deployment-envelope commands from the same nominal reset.

Example:
    uv run --locked python -m mjlab_microban.scripts.evaluate_teleop_checkpoint \
      --checkpoint logs/rsl_rl/mjlab_microban_teleop/<run>/model_<iteration>.pt \
      --output artifacts/teleop_evaluation.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.utils.nan_guard import NanGuard
from mjlab.utils.torch import configure_torch_backends

from mjlab_microban.tasks.mdp import UniformVelocityCommandWithRotation
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_HMD_JOINT_NAMES,
    MICROBAN_TELEOP_ACTION_WIDTH,
    MICROBAN_TELEOP_OBSERVATION_SCHEMA,
    MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION,
    TeleopCheckpointContract,
    validate_microban_teleop_observation_contract,
    validate_teleop_checkpoint_contract,
)
from mjlab_microban.tasks.microban_teleop_mdp import (
    MICROBAN_HMD_RETARGET_INTERVAL_S,
    MICROBAN_HMD_RUNTIME_LIMITS_RAD,
    MICROBAN_HMD_SLEW_RATES_RAD_S,
    MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M,
    HmdNeckTargetMotion,
    ResetFixedFootTargetCommand,
    ResetFixedHandTargetCommand,
)

TASK = "Mjlab-Teleop-Microban"
LOG_ROOT = Path("logs/rsl_rl/mjlab_microban_teleop")
_CHECKPOINT_RE = re.compile(r"^model_(?:(\d+)|(pristine))\.pt$")
TELEOP_EVALUATOR_REVISION = "microban_teleop_deterministic_evaluator_v9_1"
TELEOP_ACCEPTANCE_REVISION = "microban_teleop_acceptance_v9_1"

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

# Initial acceptance thresholds.  They are intentionally explicit in the JSON
# report so a result cannot be interpreted against an unknown gate.  Passing
# these simulation checks is necessary, but not sufficient, for deployment.
ACCEPTANCE_THRESHOLDS = {
    "timeout_fraction_min": 0.99,
    "hand_rms_m_max": 0.03,
    "hand_p95_m_max": 0.05,
    "foot_rms_m_max": 0.015,
    "foot_p95_m_max": 0.025,
    "linear_velocity_mae_m_s_max": 0.10,
    "low_linear_velocity_mae_m_s_max": 0.075,
    "low_linear_signed_response_m_s_min": 0.05,
    "yaw_velocity_mae_rad_s_max": 0.20,
    "self_collision_contacts_max": 0,
    "action_target_clip_fraction_max": 0.001,
    "actual_soft_limit_violation_rad_max": 1.0e-6,
}

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


def validate_scenarios(scenarios: tuple[EvaluationScenario, ...]) -> None:
    """Reject accidental coverage drift beyond the deployable command envelope."""

    names = [scenario.name for scenario in scenarios]
    if len(names) != len(set(names)):
        raise ValueError("Evaluation scenario names must be unique")

    foot_limit = torch.tensor(FOOT_TARGET_LIMIT_M) * TARGET_SAFETY_MARGIN
    both_feet_limit = (
        torch.tensor(SIMULTANEOUS_BOTH_FEET_TARGET_LIMIT_M) * TARGET_SAFETY_MARGIN
    )
    hand_limit = torch.tensor(HAND_TARGET_LIMIT_M) * TARGET_SAFETY_MARGIN
    for scenario in scenarios:
        vx, vy, yaw = scenario.twist
        yaw_limit = (
            STATIONARY_YAW_MAX_RAD_S
            if abs(vx) <= 1.0e-9 and abs(vy) <= 1.0e-9
            else MOVING_YAW_MAX_RAD_S
        )
        if not (-BACKWARD_MAX_M_S <= vx <= FORWARD_MAX_M_S):
            raise ValueError(f"{scenario.name}: vx exceeds the runtime envelope")
        if abs(vy) > LATERAL_MAX_M_S or abs(yaw) > yaw_limit:
            raise ValueError(f"{scenario.name}: twist exceeds the runtime envelope")

        foot = torch.tensor(scenario.foot_target, dtype=torch.float64)
        hand = torch.tensor(scenario.hand_target)
        if torch.any(torch.abs(foot[..., :2]) > foot_limit[:2] + 1.0e-9):
            raise ValueError(f"{scenario.name}: foot XY target exceeds the live bound")
        if torch.any(foot[..., 2] < 0.0) or torch.any(
            foot[..., 2] > foot_limit[2] + 1.0e-9
        ):
            raise ValueError(f"{scenario.name}: foot Z target exceeds the live bound")
        inactive_feet = torch.all(foot == 0.0, dim=-1)
        invalid_floor_band = (~inactive_feet) & (
            foot[..., 2] <= MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M
        )
        if bool(torch.any(invalid_floor_band).item()):
            raise ValueError(
                f"{scenario.name}: foot target inside the floor band must be "
                "exact XYZ zero"
            )
        active_feet = ~inactive_feet
        if bool(torch.all(active_feet).item()):
            if any(abs(value) > 1.0e-9 for value in scenario.twist):
                raise ValueError(
                    f"{scenario.name}: simultaneous foot targets require zero twist"
                )
            if torch.any(torch.abs(foot[..., :2]) > both_feet_limit[:2] + 1.0e-9):
                raise ValueError(
                    f"{scenario.name}: simultaneous foot XY exceeds the live bound"
                )
            if torch.any(foot[..., 2] > both_feet_limit[2] + 1.0e-9):
                raise ValueError(
                    f"{scenario.name}: simultaneous foot Z exceeds the live bound"
                )
        if torch.any(torch.abs(hand) > hand_limit + 1.0e-9):
            raise ValueError(f"{scenario.name}: hand target exceeds the live bound")
        for index, active in enumerate(scenario.hand_active):
            if not active and any(
                abs(value) > 1.0e-12 for value in scenario.hand_target[index]
            ):
                raise ValueError(
                    f"{scenario.name}: inactive hand {index} has a non-zero target"
                )


def resolve_checkpoint(
    checkpoint: Path | None,
    log_root: Path = LOG_ROOT,
    *,
    minimum_age_s: float = 10.0,
    now_s: float | None = None,
) -> Path:
    """Resolve an explicit checkpoint or the latest stable saved iteration."""

    if minimum_age_s < 0.0 or not math.isfinite(minimum_age_s):
        raise ValueError("minimum_age_s must be finite and non-negative")
    now_s = time.time() if now_s is None else now_s

    if checkpoint is not None:
        resolved = checkpoint.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {resolved}")
        if _CHECKPOINT_RE.fullmatch(resolved.name) is None:
            raise ValueError(
                "Checkpoint name must match model_<iteration>.pt or "
                f"model_pristine.pt: {resolved}"
            )
        age_s = now_s - resolved.stat().st_mtime
        if age_s < minimum_age_s:
            raise RuntimeError(
                f"Checkpoint may still be written ({age_s:.1f}s old; require "
                f"{minimum_age_s:.1f}s): {resolved}. Retry or select an older file."
            )
        return resolved

    root = log_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Teleop log directory not found: {root}")
    for run in sorted((path for path in root.iterdir() if path.is_dir()), reverse=True):
        candidates: list[tuple[int, Path]] = []
        for path in run.iterdir():
            match = _CHECKPOINT_RE.fullmatch(path.name)
            if (
                match is not None
                and match.group(1) is not None
                and path.is_file()
                and now_s - path.stat().st_mtime >= minimum_age_s
            ):
                candidates.append((int(match.group(1)), path))
        if candidates:
            return max(candidates, key=lambda item: item[0])[1].resolve()
    raise FileNotFoundError(
        f"No model_<iteration>.pt checkpoint at least {minimum_age_s:.1f}s old "
        f"found below {root}"
    )


def checkpoint_fingerprint(path: Path) -> tuple[int, int, int]:
    """Return fields that change if a direct ``torch.save`` is still in progress."""

    stat = path.stat()
    return stat.st_ino, stat.st_size, stat.st_mtime_ns


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish_json_report(output: Path, serialized: str, *, force: bool) -> None:
    """Publish a complete report atomically, without a no-force overwrite race."""

    output = output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as report_file:
            temporary = Path(report_file.name)
            report_file.write(serialized)
            report_file.write("\n")
            report_file.flush()
            os.fsync(report_file.fileno())

        if force:
            os.replace(temporary, output)
            temporary = None
            return

        # A same-filesystem hard link is an atomic create-if-absent operation.
        # Unlike Path.replace(), this cannot overwrite a report which appears
        # while the (potentially long-running) evaluation is in progress.
        try:
            os.link(temporary, output)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Output appeared during evaluation; refusing to replace: {output}"
            ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def select_scenarios(
    scenarios: tuple[EvaluationScenario, ...], requested: str | None
) -> tuple[EvaluationScenario, ...]:
    if requested is None:
        return scenarios
    names = [name.strip() for name in requested.split(",") if name.strip()]
    if not names:
        raise ValueError("--scenarios must contain at least one scenario name")
    by_name = {scenario.name: scenario for scenario in scenarios}
    unknown = sorted(set(names) - by_name.keys())
    if unknown:
        raise ValueError(f"Unknown scenarios {unknown}; available={sorted(by_name)}")
    if len(names) != len(set(names)):
        raise ValueError("--scenarios must not contain duplicates")
    return tuple(by_name[name] for name in names)


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


def command_axis_sign_diagnostic(
    scenario: EvaluationScenario,
    measured_base_velocity: dict[str, dict[str, float | int | str | None]],
) -> dict[str, Any]:
    """Check that every commanded velocity axis responds in its requested direction.

    A strictly correct sign is required for each nonzero command component. A
    tiny response in the right direction can satisfy this diagnostic, but it
    cannot hide a non-responsive policy because the independent velocity-MAE
    gates still apply.
    """

    axis_specs = (
        ("linear_x", scenario.twist[0], "m_s"),
        ("linear_y", scenario.twist[1], "m_s"),
        ("yaw", scenario.twist[2], "rad_s"),
    )
    axes: dict[str, dict[str, Any]] = {}
    for name, command, units in axis_specs:
        actual_mean = measured_base_velocity[name]["mean"]
        actual_finite = isinstance(actual_mean, (float, int)) and math.isfinite(
            actual_mean
        )
        applied = command != 0.0
        passed = not applied or (actual_finite and command * float(actual_mean) > 0.0)
        axes[name] = {
            "applied": applied,
            "command": command,
            "actual_mean": actual_mean,
            "units": units,
            "expected_sign": 1 if command > 0.0 else -1 if command < 0.0 else 0,
            "actual_sign": (
                1
                if actual_finite and float(actual_mean) > 0.0
                else -1
                if actual_finite and float(actual_mean) < 0.0
                else 0
                if actual_finite
                else None
            ),
            "passed": passed,
        }

    applied = any(axis["applied"] for axis in axes.values())
    return {
        "applied": applied,
        "passed": all(axis["passed"] for axis in axes.values()),
        "axes": axes,
    }


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


def _hmd_neck_motion_report(cfg: Any) -> dict[str, Any]:
    """Return JSON-safe evidence for the optional retained HMD event."""

    event = cfg.events.get("hmd_neck_target_motion")
    if event is None:
        return {"enabled": False, "params": None}
    if event.mode != "step" or event.func is not HmdNeckTargetMotion:
        raise TypeError("Evaluation HMD event is not HmdNeckTargetMotion")
    params = event.params
    asset_cfg = params.get("asset_cfg")
    joint_names = tuple(getattr(asset_cfg, "joint_names", ()) or ())
    if joint_names != MICROBAN_HMD_JOINT_NAMES:
        raise ValueError(
            f"Evaluation HMD joint order must be {MICROBAN_HMD_JOINT_NAMES}"
        )
    position_ranges = params.get("position_ranges_rad")
    slew_rates = params.get("slew_rates_rad_s")
    interval = params.get("retarget_interval_s")
    if position_ranges != MICROBAN_HMD_RUNTIME_LIMITS_RAD:
        raise ValueError("Evaluation HMD position ranges drifted from training")
    if slew_rates != MICROBAN_HMD_SLEW_RATES_RAD_S:
        raise ValueError("Evaluation HMD slew rates drifted from training")
    if tuple(interval or ()) != MICROBAN_HMD_RETARGET_INTERVAL_S:
        raise ValueError("Evaluation HMD retarget interval drifted from training")
    neutral_probability = params.get("neutral_probability")
    if neutral_probability != 0.0:
        raise ValueError("Moving-HMD evaluation requires neutral_probability=0.0")
    return {
        "enabled": True,
        "params": {
            "joint_names": list(joint_names),
            "position_ranges_rad": {
                name: [float(bounds[0]), float(bounds[1])]
                for name, bounds in position_ranges.items()
            },
            "slew_rates_rad_s": {
                name: float(value) for name, value in slew_rates.items()
            },
            "retarget_interval_s": [float(interval[0]), float(interval[1])],
            "neutral_probability": 0.0,
        },
    }


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


def _nonfinite_name(named_tensors: dict[str, torch.Tensor]) -> str | None:
    for name, value in named_tensors.items():
        if not bool(torch.isfinite(value).all().item()):
            return name
    return None


def _termination_names(env: ManagerBasedRlEnv) -> list[str]:
    return [
        name
        for name in env.termination_manager.active_terms
        if bool(env.termination_manager.get_term(name)[0].item())
    ]


def _contact_aligned_foot_site_ids(
    foot_contact: Any, foot: ResetFixedFootTargetCommand
) -> list[int]:
    """Match contact-body order to foot-site order for slip measurements."""

    primary_names = [
        slot.primary_name for slot in foot_contact._slots if slot.field_name == "found"
    ]
    # The Microban MJCF names the right/left terminal bodies foot/foot_2,
    # respectively, while the policy command stores sites in left/right order.
    site_for_primary = {"foot": "right_foot", "foot_2": "left_foot"}
    if set(primary_names) != set(site_for_primary):
        raise ValueError(
            "Unexpected Microban foot contact bodies: "
            f"{primary_names}; expected {sorted(site_for_primary)}"
        )
    site_ids_by_name = dict(
        zip(foot.cfg.foot_site_names, foot._foot_asset_cfg.site_ids, strict=True)
    )
    return [site_ids_by_name[site_for_primary[name]] for name in primary_names]


def _evaluate_scenario(
    *,
    env: ManagerBasedRlEnv,
    wrapped_env: RslRlVecEnvWrapper,
    policy: Any,
    scenario: EvaluationScenario,
    steps: int,
    settle_steps: int,
    seed: int,
    saturation_margin_ratio: float,
) -> dict[str, Any]:
    env.reset(seed=seed)
    _set_scenario(env, scenario)
    observations = _patch_initial_command_observation(
        wrapped_env.get_observations(), env
    )

    robot = env.scene["robot"]
    action_term = env.action_manager.get_term("joint_pos")
    foot = env.command_manager.get_term("foot_target")
    hand = env.command_manager.get_term("hand_target")
    twist = env.command_manager.get_term("twist")
    foot_contact = env.scene.sensors["feet_ground_contact"]
    self_collision = env.scene.sensors["self_collision"]
    contact_aligned_foot_site_ids = _contact_aligned_foot_site_ids(foot_contact, foot)
    fall_height_m = float(
        env.termination_manager.get_term_cfg("fell_over").params["minimum_height"]
    )

    all_soft_limits = robot.data.soft_joint_pos_limits
    clip_limits = action_term._clip
    spans = clip_limits[..., 1] - clip_limits[..., 0]
    near_margin = spans * saturation_margin_ratio

    raw_action_abs = ScalarStats()
    foot_slip = ScalarStats()
    foot_error = ScalarStats()
    hand_error = ScalarStats()
    linear_velocity_error = ScalarStats()
    yaw_velocity_error = ScalarStats()
    measured_linear_x = ScalarStats()
    measured_linear_y = ScalarStats()
    measured_yaw = ScalarStats()
    reward_stats = ScalarStats()
    root_height = ScalarStats()

    active_event_names = {
        name
        for names in env.event_manager.active_terms.values()
        for name in names
    }
    hmd_event_name = "hmd_neck_target_motion"
    hmd_motion_stats: HmdMotionStats | None = None
    hmd_motion: HmdNeckTargetMotion | None = None
    if hmd_event_name in active_event_names:
        runtime_hmd_cfg = env.event_manager.get_term_cfg(hmd_event_name)
        if not isinstance(runtime_hmd_cfg.func, HmdNeckTargetMotion):
            raise TypeError("Active HMD event is not HmdNeckTargetMotion")
        hmd_motion = runtime_hmd_cfg.func
        if tuple(hmd_motion.joint_names) != MICROBAN_HMD_JOINT_NAMES:
            raise ValueError("Active HMD event has an unexpected joint order")
        initial_target = hmd_motion.current_target[0]
        initial_actual = robot.data.joint_pos[0, hmd_motion.joint_ids]
        hmd_motion_stats = HmdMotionStats.start(
            joint_names=tuple(hmd_motion.joint_names),
            target=initial_target,
            actual=initial_actual,
        )

    action_value_count = 0
    wrapper_action_clip_count = 0
    action_clip_count = 0
    action_near_limit_count = 0
    actual_limit_value_count = 0
    actual_limit_violation_count = 0
    max_actual_limit_violation = 0.0
    max_actual_limit_violation_detail: dict[str, Any] | None = None
    minimum_actual_soft_margin = math.inf
    minimum_actual_soft_margin_detail: dict[str, Any] | None = None
    target_ids_value = action_term.target_ids
    if isinstance(target_ids_value, slice):
        action_target_ids = tuple(range(len(robot.joint_names))[target_ids_value])
    elif isinstance(target_ids_value, torch.Tensor):
        action_target_ids = tuple(int(value) for value in target_ids_value.tolist())
    else:
        action_target_ids = tuple(int(value) for value in target_ids_value)
    action_index_by_robot_joint = {
        robot_joint_index: action_index
        for action_index, robot_joint_index in enumerate(action_target_ids)
    }
    clip_count_by_joint = torch.zeros(
        len(action_term.target_names), dtype=torch.long, device=env.device
    )
    near_count_by_joint = torch.zeros_like(clip_count_by_joint)
    max_violation_by_joint = torch.zeros(
        len(robot.joint_names), dtype=torch.float64, device=env.device
    )
    min_margin_by_joint = torch.full(
        (len(robot.joint_names),),
        math.inf,
        dtype=torch.float64,
        device=env.device,
    )
    min_margin_detail_by_joint: list[dict[str, Any] | None] = [
        None for _ in robot.joint_names
    ]
    self_collision_steps = 0
    self_collision_contacts = 0
    self_collision_max_contacts = 0
    measured_steps = 0
    completed_steps = 0
    termination_names: list[str] = []
    nonfinite: dict[str, Any] | None = None

    for step in range(steps):
        actor_obs = observations["actor"]
        invalid = _nonfinite_name({"actor_observation": actor_obs})
        if invalid is not None:
            nonfinite = {"step": step, "tensor": invalid, "phase": "before_policy"}
            break

        with torch.inference_mode():
            actions = policy(observations)
        invalid = _nonfinite_name({"policy_action": actions})
        if invalid is not None:
            nonfinite = {"step": step, "tensor": invalid, "phase": "policy_output"}
            break
        if actions.shape != (1, MICROBAN_TELEOP_ACTION_WIDTH):
            raise ValueError(f"Unexpected policy action shape: {tuple(actions.shape)}")

        raw_action_abs.add(actions.abs())
        effective_actions = actions
        if wrapped_env.clip_actions is not None:
            effective_actions = torch.clamp(
                actions, -wrapped_env.clip_actions, wrapped_env.clip_actions
            )
            wrapper_action_clip_count += int(
                (torch.abs(effective_actions - actions) > 1.0e-7).sum().item()
            )
        scale = action_term.scale
        offset = action_term.offset
        unclipped_target = effective_actions * scale + offset
        clipped_target = torch.clamp(
            unclipped_target,
            min=clip_limits[..., 0],
            max=clip_limits[..., 1],
        )
        clipped = torch.abs(clipped_target - unclipped_target) > 1.0e-7
        distance_to_limit = torch.minimum(
            clipped_target - clip_limits[..., 0],
            clip_limits[..., 1] - clipped_target,
        )
        near_limit = distance_to_limit <= near_margin
        action_value_count += actions.numel()
        action_clip_count += int(clipped.sum().item())
        action_near_limit_count += int(near_limit.sum().item())
        clip_count_by_joint += clipped[0].to(dtype=torch.long)
        near_count_by_joint += near_limit[0].to(dtype=torch.long)

        observations, rewards, dones, _ = wrapped_env.step(actions)
        completed_steps += 1

        if bool(dones[0].item()):
            termination_names = _termination_names(env)
        if bool(NanGuard.detect_nans(env.sim.data)[0].item()):
            nonfinite = {
                "step": step,
                "tensor": "mujoco_warp_physics_state",
                "phase": "after_step",
            }
            break

        finite_tensors = {
            "actor_observation": observations["actor"],
            "reward": rewards,
            "processed_action_target": action_term._processed_actions,
            "joint_position_target": robot.data.joint_pos_target,
            "joint_position": robot.data.joint_pos,
            "joint_velocity": robot.data.joint_vel,
            "root_pose": robot.data.root_link_pose_w,
            "root_velocity": robot.data.root_link_vel_w,
        }
        invalid = _nonfinite_name(finite_tensors)
        if invalid is not None:
            nonfinite = {"step": step, "tensor": invalid, "phase": "after_step"}
            break

        if hmd_motion is not None and hmd_motion_stats is not None:
            hmd_motion_stats.add(
                target=hmd_motion.current_target[0],
                actual=robot.data.joint_pos[0, hmd_motion.joint_ids],
            )

        joint_pos = robot.data.joint_pos
        lower_violation = torch.clamp(all_soft_limits[..., 0] - joint_pos, min=0.0)
        upper_violation = torch.clamp(joint_pos - all_soft_limits[..., 1], min=0.0)
        violation = torch.maximum(lower_violation, upper_violation)
        soft_margin = torch.minimum(
            joint_pos - all_soft_limits[..., 0],
            all_soft_limits[..., 1] - joint_pos,
        )
        actual_limit_value_count += joint_pos.numel()
        actual_limit_violation_count += int((violation > 1.0e-6).sum().item())
        step_max_violation, step_max_joint = torch.max(violation[0], dim=0)
        step_max_value = float(step_max_violation.item())
        if step_max_value > max_actual_limit_violation:
            max_actual_limit_violation = step_max_value
            robot_joint_index = int(step_max_joint.item())
            action_index = action_index_by_robot_joint.get(robot_joint_index)
            position = float(joint_pos[0, robot_joint_index].item())
            lower = float(all_soft_limits[0, robot_joint_index, 0].item())
            upper = float(all_soft_limits[0, robot_joint_index, 1].item())
            max_actual_limit_violation_detail = {
                "step": step,
                "joint": robot.joint_names[robot_joint_index],
                "side": "lower" if position < lower else "upper",
                "position_rad": position,
                "velocity_rad_s": float(
                    robot.data.joint_vel[0, robot_joint_index].item()
                ),
                "soft_lower_rad": lower,
                "soft_upper_rad": upper,
                "joint_position_target_rad": float(
                    robot.data.joint_pos_target[0, robot_joint_index].item()
                ),
                "policy_action_index": action_index,
                "policy_action_raw": (
                    float(actions[0, action_index].item())
                    if action_index is not None
                    else None
                ),
                "unclipped_target_rad": (
                    float(unclipped_target[0, action_index].item())
                    if action_index is not None
                    else None
                ),
                "clipped_target_rad": (
                    float(clipped_target[0, action_index].item())
                    if action_index is not None
                    else None
                ),
            }
        max_violation_by_joint = torch.maximum(
            max_violation_by_joint, violation[0].to(dtype=torch.float64)
        )
        improved_margin_indices = torch.nonzero(
            soft_margin[0] < min_margin_by_joint, as_tuple=False
        ).flatten()
        for improved_joint_tensor in improved_margin_indices:
            robot_joint_index = int(improved_joint_tensor.item())
            action_index = action_index_by_robot_joint.get(robot_joint_index)
            position = float(joint_pos[0, robot_joint_index].item())
            lower = float(all_soft_limits[0, robot_joint_index, 0].item())
            upper = float(all_soft_limits[0, robot_joint_index, 1].item())
            min_margin_detail_by_joint[robot_joint_index] = {
                "step": step,
                "margin_rad": float(soft_margin[0, robot_joint_index].item()),
                "side": ("lower" if position - lower <= upper - position else "upper"),
                "position_rad": position,
                "velocity_rad_s": float(
                    robot.data.joint_vel[0, robot_joint_index].item()
                ),
                "soft_lower_rad": lower,
                "soft_upper_rad": upper,
                "joint_position_target_rad": float(
                    robot.data.joint_pos_target[0, robot_joint_index].item()
                ),
                "policy_action_raw": (
                    float(actions[0, action_index].item())
                    if action_index is not None
                    else None
                ),
            }
        min_margin_by_joint = torch.minimum(
            min_margin_by_joint, soft_margin[0].to(dtype=torch.float64)
        )
        step_min_margin, step_min_joint = torch.min(soft_margin[0], dim=0)
        step_min_margin_value = float(step_min_margin.item())
        if step_min_margin_value < minimum_actual_soft_margin:
            minimum_actual_soft_margin = step_min_margin_value
            robot_joint_index = int(step_min_joint.item())
            action_index = action_index_by_robot_joint.get(robot_joint_index)
            position = float(joint_pos[0, robot_joint_index].item())
            lower = float(all_soft_limits[0, robot_joint_index, 0].item())
            upper = float(all_soft_limits[0, robot_joint_index, 1].item())
            minimum_actual_soft_margin_detail = {
                "step": step,
                "joint": robot.joint_names[robot_joint_index],
                "side": ("lower" if position - lower <= upper - position else "upper"),
                "position_rad": position,
                "velocity_rad_s": float(
                    robot.data.joint_vel[0, robot_joint_index].item()
                ),
                "soft_lower_rad": lower,
                "soft_upper_rad": upper,
                "joint_position_target_rad": float(
                    robot.data.joint_pos_target[0, robot_joint_index].item()
                ),
                "policy_action_index": action_index,
                "policy_action_raw": (
                    float(actions[0, action_index].item())
                    if action_index is not None
                    else None
                ),
                "unclipped_target_rad": (
                    float(unclipped_target[0, action_index].item())
                    if action_index is not None
                    else None
                ),
                "clipped_target_rad": (
                    float(clipped_target[0, action_index].item())
                    if action_index is not None
                    else None
                ),
            }

        current_root_height = robot.data.root_link_pos_w[:, 2]
        root_height.add(current_root_height)
        if (
            bool((current_root_height < fall_height_m)[0].item())
            and "fell_over" not in termination_names
        ):
            termination_names.append("fell_over_current_state")
        collision_found = self_collision.data.found
        if collision_found is None:
            raise RuntimeError("self_collision sensor does not expose found")
        collision_count = int(collision_found[0].sum().item())
        self_collision_contacts += collision_count
        self_collision_max_contacts = max(self_collision_max_contacts, collision_count)
        self_collision_steps += int(collision_count > 0)

        if step >= settle_steps:
            measured_steps += 1
            reward_stats.add(rewards)

            contact_found = foot_contact.data.found
            if contact_found is None:
                raise RuntimeError("feet_ground_contact sensor does not expose found")
            contact_mask = contact_found[0] > 0
            foot_velocity = robot.data.site_lin_vel_w[
                0, contact_aligned_foot_site_ids, :2
            ].norm(dim=-1)
            foot_slip.add(foot_velocity[contact_mask])

            current_foot = foot.current_foot_pos_b()
            foot_error.add(
                torch.linalg.vector_norm(
                    current_foot - foot._default_foot_pos_b - foot.foot_target_offset_b,
                    dim=-1,
                )
            )
            current_hand = hand.current_hand_pos_b()
            per_hand_error = torch.linalg.vector_norm(
                current_hand - hand._default_hand_pos_b - hand.hand_target_offset_b,
                dim=-1,
            )
            hand_error.add(per_hand_error[hand.is_active])

            commanded = twist.vel_command_b
            actual_linear = robot.data.root_link_lin_vel_b[:, :2]
            actual_yaw = robot.data.root_link_ang_vel_b[:, 2]
            measured_linear_x.add(actual_linear[:, 0])
            measured_linear_y.add(actual_linear[:, 1])
            measured_yaw.add(actual_yaw)
            linear_velocity_error.add(
                torch.linalg.vector_norm(commanded[:, :2] - actual_linear, dim=-1)
            )
            yaw_velocity_error.add(torch.abs(commanded[:, 2] - actual_yaw))

        if termination_names:
            break

    def fraction(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    per_joint = {}
    clip_counts = clip_count_by_joint.cpu().tolist()
    near_counts = near_count_by_joint.cpu().tolist()
    for index, name in enumerate(action_term.target_names):
        per_joint[name] = {
            "clip_fraction": fraction(clip_counts[index], completed_steps),
            "near_soft_limit_fraction": fraction(near_counts[index], completed_steps),
        }
    actual_limit_per_joint = dict(
        zip(robot.joint_names, max_violation_by_joint.cpu().tolist(), strict=True)
    )
    actual_margin_per_joint = dict(
        zip(robot.joint_names, min_margin_by_joint.cpu().tolist(), strict=True)
    )
    actual_margin_detail_per_joint = dict(
        zip(robot.joint_names, min_margin_detail_by_joint, strict=True)
    )

    report = {
        "name": scenario.name,
        "command": asdict(scenario),
        "requested_steps": steps,
        "completed_steps": completed_steps,
        "measured_steps_after_settle": measured_steps,
        "duration_s": completed_steps * env.step_dt,
        "terminated": bool(termination_names),
        "termination_terms": termination_names,
        "reached_time_limit": "time_out" in termination_names,
        "fell_over": any(name.startswith("fell_over") for name in termination_names),
        "finite": nonfinite is None,
        "nonfinite_failure": nonfinite,
        "min_root_height_m": root_height.minimum,
        "reward": reward_stats.report(units="reward_per_step"),
        "action": {
            "joint_scope": "18_policy_controlled_joints",
            "raw_absolute": raw_action_abs.report(units="policy_output"),
            "wrapper_clip_fraction": fraction(
                wrapper_action_clip_count, action_value_count
            ),
            "target_clip_fraction": fraction(action_clip_count, action_value_count),
            "target_near_soft_limit_fraction": fraction(
                action_near_limit_count, action_value_count
            ),
            "near_limit_margin_ratio": saturation_margin_ratio,
            "per_joint": per_joint,
        },
        "joint_soft_limits": {
            "joint_scope": "all_21_joints_including_hmd_neck",
            "actual_violation_fraction": fraction(
                actual_limit_violation_count, actual_limit_value_count
            ),
            "max_actual_violation_rad": max_actual_limit_violation,
            "max_actual_violation_rad_by_joint": actual_limit_per_joint,
            "max_actual_violation_detail": max_actual_limit_violation_detail,
            "minimum_actual_soft_margin_rad": minimum_actual_soft_margin,
            "minimum_actual_soft_margin_rad_by_joint": actual_margin_per_joint,
            "minimum_actual_soft_margin_detail_by_joint": (
                actual_margin_detail_per_joint
            ),
            "minimum_actual_soft_margin_detail": (minimum_actual_soft_margin_detail),
        },
        "foot_slip": foot_slip.report(units="m_s_while_in_contact"),
        "self_collision": {
            "steps_with_contact": self_collision_steps,
            "step_fraction": fraction(self_collision_steps, completed_steps),
            "contact_count_total": self_collision_contacts,
            "max_contacts_in_step": self_collision_max_contacts,
        },
        "target_error": {
            "foot": foot_error.report(units="m"),
            "active_hand": hand_error.report(units="m"),
        },
        "velocity_tracking_error": {
            "linear_xy": linear_velocity_error.report(units="m_s"),
            "yaw": yaw_velocity_error.report(units="rad_s"),
        },
        "measured_base_velocity": {
            "linear_x": measured_linear_x.report(units="m_s"),
            "linear_y": measured_linear_y.report(units="m_s"),
            "yaw": measured_yaw.report(units="rad_s"),
        },
        "hmd_neck_motion": (
            hmd_motion_stats.report(step_dt=env.step_dt)
            if hmd_motion_stats is not None
            else {
                "active_event_member": False,
                "sample_count": 0,
                "joint_names": [],
                "per_axis": {},
            }
        ),
    }
    report["command_axis_sign"] = command_axis_sign_diagnostic(
        scenario, report["measured_base_velocity"]
    )
    report["acceptance"] = evaluate_scenario_acceptance(scenario, report)
    return report


def _maximum(values: list[float]) -> float | None:
    return max(values) if values else None


def summarize_hmd_motion_evidence(
    scenario_reports: list[dict[str, Any]], *, required: bool
) -> dict[str, Any]:
    """Aggregate fail-closed proof that the active HMD event moved every axis."""

    active_scenarios: list[str] = []
    inactive_scenarios: list[str] = []
    target_peak_to_peak_by_axis: dict[str, list[float]] = {
        name: [] for name in MICROBAN_HMD_JOINT_NAMES
    }
    actual_peak_to_peak_by_axis: dict[str, list[float]] = {
        name: [] for name in MICROBAN_HMD_JOINT_NAMES
    }
    malformed_scenarios: list[str] = []

    for scenario in scenario_reports:
        scenario_name = str(scenario.get("name", "<unnamed>"))
        motion = scenario.get("hmd_neck_motion")
        if not isinstance(motion, dict):
            inactive_scenarios.append(scenario_name)
            if required:
                malformed_scenarios.append(scenario_name)
            continue
        if motion.get("active_event_member") is not True:
            inactive_scenarios.append(scenario_name)
            continue
        active_scenarios.append(scenario_name)
        if motion.get("joint_names") != list(MICROBAN_HMD_JOINT_NAMES):
            malformed_scenarios.append(scenario_name)
            continue
        per_axis = motion.get("per_axis")
        if not isinstance(per_axis, dict):
            malformed_scenarios.append(scenario_name)
            continue
        for name in MICROBAN_HMD_JOINT_NAMES:
            axis = per_axis.get(name)
            if not isinstance(axis, dict):
                malformed_scenarios.append(scenario_name)
                break
            target = axis.get("target_peak_to_peak_rad")
            actual = axis.get("actual_peak_to_peak_rad")
            if (
                not isinstance(target, (int, float))
                or isinstance(target, bool)
                or not math.isfinite(float(target))
                or not isinstance(actual, (int, float))
                or isinstance(actual, bool)
                or not math.isfinite(float(actual))
            ):
                malformed_scenarios.append(scenario_name)
                break
            target_peak_to_peak_by_axis[name].append(float(target))
            actual_peak_to_peak_by_axis[name].append(float(actual))

    minimum_target_by_axis = {
        name: min(values) if values else None
        for name, values in target_peak_to_peak_by_axis.items()
    }
    minimum_actual_by_axis = {
        name: min(values) if values else None
        for name, values in actual_peak_to_peak_by_axis.items()
    }
    all_active = (
        len(active_scenarios) == len(scenario_reports)
        and not inactive_scenarios
        and not malformed_scenarios
    )
    no_active = not active_scenarios
    target_passed = all(
        value is not None and value >= HMD_TARGET_PEAK_TO_PEAK_MIN_RAD
        for value in minimum_target_by_axis.values()
    )
    actual_passed = all(
        value is not None and value >= HMD_ACTUAL_PEAK_TO_PEAK_MIN_RAD
        for value in minimum_actual_by_axis.values()
    )
    passed = (
        all_active and target_passed and actual_passed
        if required
        else no_active and not malformed_scenarios
    )
    return {
        "required": required,
        "passed": passed,
        "event_name": "hmd_neck_target_motion",
        "active_event_membership": {
            "all_scenarios": all_active,
            "active_scenarios": active_scenarios,
            "inactive_scenarios": inactive_scenarios,
            "malformed_scenarios": sorted(set(malformed_scenarios)),
        },
        "minimum_required_target_peak_to_peak_rad": (
            HMD_TARGET_PEAK_TO_PEAK_MIN_RAD if required else None
        ),
        "minimum_required_actual_peak_to_peak_rad": (
            HMD_ACTUAL_PEAK_TO_PEAK_MIN_RAD if required else None
        ),
        "minimum_observed_target_peak_to_peak_rad_by_axis": minimum_target_by_axis,
        "minimum_observed_actual_peak_to_peak_rad_by_axis": minimum_actual_by_axis,
        "target_excursion_passed": target_passed if required else None,
        "actual_excursion_passed": actual_passed if required else None,
    }


def evaluate_scenario_acceptance(
    scenario: EvaluationScenario, report: dict[str, Any]
) -> dict[str, Any]:
    """Apply the documented simulation gates to one deterministic rollout."""

    checks: dict[str, dict[str, Any]] = {}

    def add_maximum(name: str, value: float | None, limit: float) -> None:
        checks[name] = {
            "applied": value is not None,
            "value": value,
            "maximum": limit,
            "passed": value is not None and value <= limit,
        }

    def add_minimum(name: str, value: float | None, limit: float) -> None:
        checks[name] = {
            "applied": value is not None,
            "value": value,
            "minimum": limit,
            "passed": value is not None and value >= limit,
        }

    checks["finite"] = {
        "applied": True,
        "value": report["finite"],
        "required": True,
        "passed": report["finite"],
    }
    checks["fell_over"] = {
        "applied": True,
        "value": report["fell_over"],
        "required": False,
        "passed": not report["fell_over"],
    }
    checks["reached_time_limit"] = {
        "applied": True,
        "value": report["reached_time_limit"],
        "required": True,
        "passed": report["reached_time_limit"],
    }
    add_maximum(
        "self_collision_contacts",
        report["self_collision"]["contact_count_total"],
        ACCEPTANCE_THRESHOLDS["self_collision_contacts_max"],
    )
    add_maximum(
        "action_target_clip_fraction",
        report["action"]["target_clip_fraction"],
        ACCEPTANCE_THRESHOLDS["action_target_clip_fraction_max"],
    )
    add_maximum(
        "actual_soft_limit_violation_rad",
        report["joint_soft_limits"]["max_actual_violation_rad"],
        ACCEPTANCE_THRESHOLDS["actual_soft_limit_violation_rad_max"],
    )
    vx, vy, yaw = scenario.twist
    low_linear_command = yaw == 0.0 and 0.0 < math.hypot(vx, vy) <= 0.1000001
    add_maximum(
        "linear_velocity_mae_m_s",
        report["velocity_tracking_error"]["linear_xy"]["mean"],
        ACCEPTANCE_THRESHOLDS[
            "low_linear_velocity_mae_m_s_max"
            if low_linear_command
            else "linear_velocity_mae_m_s_max"
        ],
    )
    add_maximum(
        "yaw_velocity_mae_rad_s",
        report["velocity_tracking_error"]["yaw"]["mean"],
        ACCEPTANCE_THRESHOLDS["yaw_velocity_mae_rad_s_max"],
    )
    command_axis_sign = report["command_axis_sign"]
    checks["command_axis_sign"] = {
        "applied": command_axis_sign["applied"],
        "value": command_axis_sign["passed"],
        "required": True,
        "passed": command_axis_sign["passed"],
        "axes": command_axis_sign["axes"],
    }
    if low_linear_command:
        commanded_axis = "linear_x" if vx != 0.0 else "linear_y"
        actual_mean = report["measured_base_velocity"][commanded_axis]["mean"]
        add_minimum(
            "low_linear_signed_response_m_s",
            abs(float(actual_mean)) if actual_mean is not None else None,
            ACCEPTANCE_THRESHOLDS["low_linear_signed_response_m_s_min"],
        )

    if any(scenario.hand_active):
        add_maximum(
            "hand_rms_m",
            report["target_error"]["active_hand"]["rms"],
            ACCEPTANCE_THRESHOLDS["hand_rms_m_max"],
        )
        add_maximum(
            "hand_p95_m",
            report["target_error"]["active_hand"]["p95"],
            ACCEPTANCE_THRESHOLDS["hand_p95_m_max"],
        )

    # A fixed-foot gate is meaningful only for an exact stationary command.
    # Applying it to low-speed walking rewards the stationary local optimum.
    foot_tracking_active = vx == 0.0 and vy == 0.0 and yaw == 0.0
    if foot_tracking_active:
        add_maximum(
            "foot_rms_m",
            report["target_error"]["foot"]["rms"],
            ACCEPTANCE_THRESHOLDS["foot_rms_m_max"],
        )
        add_maximum(
            "foot_p95_m",
            report["target_error"]["foot"]["p95"],
            ACCEPTANCE_THRESHOLDS["foot_p95_m_max"],
        )

    failed = sorted(name for name, check in checks.items() if not check["passed"])
    return {
        "passed": not failed,
        "failed_checks": failed,
        "checks": checks,
        "foot_tracking_gate_applied": foot_tracking_active,
    }


def build_report(
    *,
    checkpoint: Path,
    checkpoint_digest: str,
    checkpoint_size_bytes: int,
    device: str,
    seed: int,
    steps: int,
    settle_steps: int,
    saturation_margin_ratio: float,
    scenario_reports: list[dict[str, Any]],
    control_hz: float,
    checkpoint_contract: TeleopCheckpointContract,
    hmd_neck_motion: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if hmd_neck_motion is None:
        hmd_neck_motion = {"enabled": False, "params": None}
    moving_hmd_neck = hmd_neck_motion.get("enabled") is True
    hmd_motion_evidence = summarize_hmd_motion_evidence(
        scenario_reports, required=moving_hmd_neck
    )
    hmd_neck_motion = {**hmd_neck_motion, "evidence": hmd_motion_evidence}
    falls = [item["name"] for item in scenario_reports if item["fell_over"]]
    nonfinite = [item["name"] for item in scenario_reports if not item["finite"]]
    incomplete = [
        item["name"]
        for item in scenario_reports
        if item["completed_steps"] != item["requested_steps"]
    ]
    unexpected_terminations = [
        item["name"]
        for item in scenario_reports
        if any(
            name != "time_out" and not name.startswith("fell_over")
            for name in item["termination_terms"]
        )
    ]
    hard_failures = sorted(set(falls + nonfinite + unexpected_terminations))
    acceptance_failures = [
        item["name"] for item in scenario_reports if not item["acceptance"]["passed"]
    ]
    timeout_count = sum(item["reached_time_limit"] for item in scenario_reports)
    timeout_fraction = (
        timeout_count / len(scenario_reports) if scenario_reports else 0.0
    )

    slip_maxima = [
        item["foot_slip"]["max"]
        for item in scenario_reports
        if item["foot_slip"]["max"] is not None
    ]
    target_error_maxima = [
        metric["max"]
        for item in scenario_reports
        for metric in item["target_error"].values()
        if metric["max"] is not None
    ]
    collision_fractions = [
        item["self_collision"]["step_fraction"]
        for item in scenario_reports
        if item["self_collision"]["step_fraction"] is not None
    ]
    soft_violation_maxima = [
        item["joint_soft_limits"]["max_actual_violation_rad"]
        for item in scenario_reports
    ]

    match = _CHECKPOINT_RE.fullmatch(checkpoint.name)
    checkpoint_iteration = (
        int(match.group(1))
        if match is not None and match.group(1) is not None
        else -1
        if match is not None
        else None
    )
    hard_pass = not hard_failures and not incomplete
    acceptance_pass = (
        hard_pass
        and not acceptance_failures
        and timeout_fraction >= ACCEPTANCE_THRESHOLDS["timeout_fraction_min"]
        and hmd_motion_evidence["passed"]
    )
    expected_names = [scenario.name for scenario in default_scenarios()]
    actual_names = [item["name"] for item in scenario_reports]
    canonical_coverage = (
        not moving_hmd_neck
        and actual_names == expected_names
        and steps >= 1000
        and settle_steps == 50
    )
    status = "fail" if not acceptance_pass else "diagnostic"
    if (
        acceptance_pass
        and canonical_coverage
        and not checkpoint_contract.diagnostic_legacy
        and not checkpoint_contract.pristine_pre_update
    ):
        status = "pass"
    limitations = [
        (
            "MuJoCo Warp seeded rollouts are reproducible in command generation "
            "but are not guaranteed bit-exact."
        ),
        (
            "The self-collision sensor is instantaneous at the end of each "
            "control step and can miss contacts within the four physics substeps."
        ),
        (
            "This nominal simulation does not certify physical IMU axes, camera "
            "transport, servo current, fall restraint or emergency stop behavior."
        ),
        (
            "bounded_both_feet uses the v2 simultaneous-foot live maximum and "
            "requires a stationary twist command."
        ),
        (
            "neutral covers exact-zero inactive feet; floor_band_edge_single "
            "and floor_band_edge_both use 2.6 mm, immediately above the "
            "inclusive 2.5 mm inactive floor band."
        ),
    ]
    if moving_hmd_neck:
        limitations.append(
            "This diagnostic forces every HMD waypoint non-neutral. It can gate "
            "moving-neck robustness but cannot be the nominal canonical pass."
        )
    else:
        limitations.append(
            "Nominal evaluation holds the HMD-owned neck at its default pose; "
            "use --moving-hmd-neck for the independent inertial-disturbance gate."
        )
    if checkpoint_contract.diagnostic_legacy:
        limitations.append(
            "This is an explicitly requested legacy-v1 diagnostic using raw "
            "previous-action feedback and the v1 scalar Gaussian actor. It can "
            "never pass the current deployment gate or be exported as current."
        )
    if checkpoint_contract.pristine_pre_update:
        limitations.append(
            "This is the pinned velocity bootstrap before any teleop PPO update. "
            "It is a safety baseline and can never pass the deployment gate."
        )
    return {
        "schema_version": 8,
        "evaluator_revision": TELEOP_EVALUATOR_REVISION,
        "acceptance_revision": TELEOP_ACCEPTANCE_REVISION,
        "status": status,
        "task": TASK,
        "checkpoint": str(checkpoint),
        "checkpoint_iteration": checkpoint_iteration,
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_size_bytes": checkpoint_size_bytes,
        "training_contract": {
            "version": checkpoint_contract.version,
            "previous_action_semantics": (
                checkpoint_contract.previous_action_semantics
            ),
            "diagnostic_legacy": checkpoint_contract.diagnostic_legacy,
            "pristine_pre_update": checkpoint_contract.pristine_pre_update,
            "training_provenance_sha256": (
                checkpoint_contract.training_provenance_sha256
            ),
            "canonical_training_stage": (
                checkpoint_contract.canonical_training_stage
            ),
            "deployment_compatible": (
                not checkpoint_contract.diagnostic_legacy
                and not checkpoint_contract.pristine_pre_update
                and checkpoint_contract.version
                == MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION
            ),
        },
        "device": device,
        "seed": seed,
        "control_hz": control_hz,
        "steps_per_scenario": steps,
        "settle_steps": settle_steps,
        "saturation_margin_ratio": saturation_margin_ratio,
        "target_safety_margin": TARGET_SAFETY_MARGIN,
        "acceptance_thresholds": ACCEPTANCE_THRESHOLDS,
        "hmd_neck_motion": hmd_neck_motion,
        "nominal_environment": {
            "viewer": False,
            "robot_network": False,
            "actor_corruption": False,
            "domain_randomization": False,
            "external_pushes": False,
            "auto_reset": False,
            "hmd_neck_motion": moving_hmd_neck,
        },
        "summary": {
            "scenario_count": len(scenario_reports),
            "canonical_coverage": canonical_coverage,
            "hard_safety_checks_passed": hard_pass,
            "acceptance_checks_passed": acceptance_pass,
            "hmd_motion_evidence_passed": hmd_motion_evidence["passed"],
            "training_contract_check_passed": (
                not checkpoint_contract.diagnostic_legacy
            ),
            "hard_failure_scenarios": hard_failures,
            "acceptance_failure_scenarios": acceptance_failures,
            "incomplete_scenarios": incomplete,
            "timeout_fraction": timeout_fraction,
            "fall_scenarios": falls,
            "nonfinite_scenarios": nonfinite,
            "unexpected_termination_scenarios": unexpected_terminations,
            "worst_foot_slip_m_s": _maximum(slip_maxima),
            "worst_target_error_m": _maximum(target_error_maxima),
            "worst_self_collision_step_fraction": _maximum(collision_fractions),
            "worst_actual_soft_limit_violation_rad": _maximum(soft_violation_maxima),
            "deployment_certified": False,
        },
        "limitations": limitations,
        "scenarios": scenario_reports,
    }


def evaluation_exit_code(report: dict[str, Any]) -> int:
    """Return zero only for a canonical pass; diagnostics are explicitly nonzero."""

    status = report.get("status")
    if status == "pass":
        return 0
    if status == "diagnostic":
        return 3
    return 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint .pt (default: highest numeric checkpoint in latest run)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional new path for a pure JSON report; the report is always printed",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow --output to replace an existing report",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Control steps per scenario (default: one 20 s episode at 50 Hz)",
    )
    parser.add_argument("--settle-steps", type=int, default=50)
    parser.add_argument(
        "--scenarios",
        default=None,
        help="Comma-separated subset; omit for the complete fixed suite",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="Print available scenario names and exit without loading a checkpoint",
    )
    parser.add_argument(
        "--saturation-margin-ratio",
        type=float,
        default=0.01,
        help="Fraction of joint soft-limit span counted as near saturation",
    )
    parser.add_argument(
        "--minimum-checkpoint-age-s",
        type=float,
        default=10.0,
        help="Ignore/reject files this recent to avoid an in-progress torch.save",
    )
    parser.add_argument(
        "--allow-legacy-teleop-contract",
        action="store_true",
        help=(
            "Diagnostics only: evaluate an unversioned v1 checkpoint with its "
            "original raw-action observation/distribution. The report can never pass "
            "and exits nonzero."
        ),
    )
    parser.add_argument(
        "--moving-hmd-neck",
        action="store_true",
        help=(
            "Diagnostics only: retain the training HmdNeckTargetMotion event with "
            "neutral_probability=0.0. Even full coverage cannot be canonical."
        ),
    )
    return parser.parse_args()


def _construct_checkpoint_consumer_runner(env, agent_cfg, device: str):
    """Construct the explicit actor-load-only runner used by this evaluator."""

    agent_cfg.checkpoint_consumer_mode = True
    runner_cls = load_runner_cls(TASK)
    if runner_cls is None:
        raise RuntimeError(f"No runner is registered for {TASK}")
    return runner_cls(env, asdict(agent_cfg), device=device)


def main() -> None:
    args = parse_args()
    scenarios = default_scenarios()
    validate_scenarios(scenarios)
    if args.list_scenarios:
        print("\n".join(scenario.name for scenario in scenarios))
        return
    if args.steps < 1:
        raise ValueError("--steps must be at least 1")
    if not 0 <= args.settle_steps < args.steps:
        raise ValueError("--settle-steps must be in [0, steps)")
    if not 0.0 < args.saturation_margin_ratio < 0.5:
        raise ValueError("--saturation-margin-ratio must be in (0, 0.5)")
    if args.output is not None and args.output.expanduser().exists() and not args.force:
        raise FileExistsError(
            f"Output already exists (pass --force to replace): {args.output}"
        )
    scenarios = select_scenarios(scenarios, args.scenarios)
    checkpoint = resolve_checkpoint(
        args.checkpoint,
        minimum_age_s=args.minimum_checkpoint_age_s,
    )
    checkpoint_contract = validate_teleop_checkpoint_contract(
        checkpoint,
        map_location="cpu",
        allow_legacy_diagnostic=args.allow_legacy_teleop_contract,
    )

    configure_torch_backends(allow_tf32=False, deterministic=True)
    torch.use_deterministic_algorithms(True, warn_only=True)

    env_cfg = load_env_cfg(TASK, play=True)
    moving_hmd_neck_event = None
    if args.moving_hmd_neck:
        training_cfg = load_env_cfg(TASK, play=False)
        moving_hmd_neck_event = _copy_forced_moving_hmd_neck_event(training_cfg)
    _configure_nominal_evaluation(
        env_cfg,
        steps=args.steps,
        moving_hmd_neck_event=moving_hmd_neck_event,
    )
    hmd_neck_motion = _hmd_neck_motion_report(env_cfg)
    env_cfg.seed = args.seed
    agent_cfg = load_rl_cfg(TASK)
    if checkpoint_contract.diagnostic_legacy:
        # Reconstruct the original v1 actor contract for comparison only.  V1
        # observed the raw network output and used a scalar-space Gaussian std;
        # loading those weights into the v2 log-std actor would either fail or,
        # worse, evaluate a different recurrence than the one that was trained.
        for group_name in ("actor", "critic"):
            env_cfg.observations[group_name].terms[
                "actions"
            ].func = velocity_mdp.last_action
            env_cfg.observations[group_name].terms["actions"].params = {
                "action_name": "joint_pos"
            }
        agent_cfg.actor.distribution_cfg = {
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        }
        # The current adapter requires and transforms latent actions.  A legacy-v1
        # checkpoint used RSL-RL's ordinary environment-action PPO path, so its
        # diagnostic reconstruction must restore that algorithm as well as the
        # original Gaussian.  This branch is already permanently barred from
        # saving/exporting by MicrobanTeleopOnPolicyRunner.
        agent_cfg.algorithm.class_name = "PPO"
    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    wrapped_env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)

    try:
        active_event_names = {
            name
            for names in raw_env.event_manager.active_terms.values()
            for name in names
        }
        if args.moving_hmd_neck:
            if "hmd_neck_target_motion" not in active_event_names:
                raise RuntimeError(
                    "Moving-HMD event is configured but is not an active event term"
                )
            runtime_hmd_cfg = raw_env.event_manager.get_term_cfg(
                "hmd_neck_target_motion"
            )
            runtime_hmd_motion = runtime_hmd_cfg.func
            if (
                not isinstance(runtime_hmd_motion, HmdNeckTargetMotion)
                or runtime_hmd_motion.neutral_probability != 0.0
            ):
                raise RuntimeError(
                    "Moving-HMD event did not materialize with neutral_probability=0.0"
                )
        elif "hmd_neck_target_motion" in active_event_names:
            raise RuntimeError("Nominal evaluator unexpectedly enabled HMD motion")
        validate_microban_teleop_observation_contract(raw_env)
        runner = _construct_checkpoint_consumer_runner(
            wrapped_env, agent_cfg, args.device
        )
        fingerprint_before = checkpoint_fingerprint(checkpoint)
        try:
            runner.load(
                str(checkpoint),
                load_cfg={"actor": True},
                strict=True,
                map_location=args.device,
                allow_legacy_teleop_contract=(args.allow_legacy_teleop_contract),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Checkpoint load failed; it may be incomplete: {checkpoint}. "
                "Retry or select an older explicit checkpoint."
            ) from exc
        if runner.loaded_checkpoint_contract != checkpoint_contract:
            raise RuntimeError(
                "Checkpoint contract changed between validation and actor load"
            )
        checkpoint_digest = checkpoint_sha256(checkpoint)
        fingerprint_after = checkpoint_fingerprint(checkpoint)
        if fingerprint_after != fingerprint_before:
            raise RuntimeError(
                f"Checkpoint changed while it was loaded: {checkpoint}. Retry."
            )
        policy = runner.get_inference_policy(device=args.device)

        scenario_reports = [
            _evaluate_scenario(
                env=raw_env,
                wrapped_env=wrapped_env,
                policy=policy,
                scenario=scenario,
                steps=args.steps,
                settle_steps=args.settle_steps,
                seed=args.seed,
                saturation_margin_ratio=args.saturation_margin_ratio,
            )
            for scenario in scenarios
        ]
        report = build_report(
            checkpoint=checkpoint,
            checkpoint_digest=checkpoint_digest,
            checkpoint_size_bytes=fingerprint_after[1],
            device=args.device,
            seed=args.seed,
            steps=args.steps,
            settle_steps=args.settle_steps,
            saturation_margin_ratio=args.saturation_margin_ratio,
            scenario_reports=scenario_reports,
            control_hz=1.0 / raw_env.step_dt,
            checkpoint_contract=checkpoint_contract,
            hmd_neck_motion=hmd_neck_motion,
        )
    finally:
        wrapped_env.close()

    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        _publish_json_report(args.output, serialized, force=args.force)
    print(serialized)
    exit_code = evaluation_exit_code(report)
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
