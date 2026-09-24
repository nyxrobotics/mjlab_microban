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
    MICROBAN_TELEOP_ACTION_WIDTH,
    MICROBAN_TELEOP_OBSERVATION_SCHEMA,
    TeleopCheckpointContract,
    validate_microban_teleop_observation_contract,
    validate_teleop_checkpoint_contract,
)
from mjlab_microban.tasks.microban_teleop_mdp import (
    MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M,
    ResetFixedFootTargetCommand,
    ResetFixedHandTargetCommand,
)

TASK = "Mjlab-Teleop-Microban"
LOG_ROOT = Path("logs/rsl_rl/mjlab_microban_teleop")
_CHECKPOINT_RE = re.compile(r"^model_(?:(\d+)|(pristine))\.pt$")

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
    "yaw_velocity_mae_rad_s_max": 0.20,
    "self_collision_contacts_max": 0,
    "action_target_clip_fraction_max": 0.001,
    "actual_soft_limit_violation_rad_max": 1.0e-6,
}


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


def _configure_nominal_evaluation(cfg: Any, *, steps: int) -> None:
    """Remove stochastic training events while preserving nominal reset events."""

    cfg.scene.num_envs = 1
    cfg.auto_reset = False
    # Make the requested evaluation horizon the time limit.  A healthy scenario
    # therefore reaches ``time_out`` exactly on its final requested step.
    cfg.episode_length_s = steps * cfg.decimation * cfg.sim.mujoco.timestep

    cfg.events = {
        name: term for name, term in cfg.events.items() if term.mode == "reset"
    }
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

    # Play mode already disables actor corruption and the random HMD motion.
    cfg.observations["actor"].enable_corruption = False
    if "hmd_neck_target_motion" in cfg.events:
        raise AssertionError("Headless evaluation must not randomize the HMD neck")


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
    reward_stats = ScalarStats()
    root_height = ScalarStats()

    action_value_count = 0
    wrapper_action_clip_count = 0
    action_clip_count = 0
    action_near_limit_count = 0
    actual_limit_value_count = 0
    actual_limit_violation_count = 0
    max_actual_limit_violation = 0.0
    clip_count_by_joint = torch.zeros(
        len(action_term.target_names), dtype=torch.long, device=env.device
    )
    near_count_by_joint = torch.zeros_like(clip_count_by_joint)
    max_violation_by_joint = torch.zeros(
        len(robot.joint_names), dtype=torch.float64, device=env.device
    )
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

        joint_pos = robot.data.joint_pos
        lower_violation = torch.clamp(all_soft_limits[..., 0] - joint_pos, min=0.0)
        upper_violation = torch.clamp(joint_pos - all_soft_limits[..., 1], min=0.0)
        violation = torch.maximum(lower_violation, upper_violation)
        actual_limit_value_count += joint_pos.numel()
        actual_limit_violation_count += int((violation > 1.0e-6).sum().item())
        max_actual_limit_violation = max(
            max_actual_limit_violation, float(violation.max().item())
        )
        max_violation_by_joint = torch.maximum(
            max_violation_by_joint, violation[0].to(dtype=torch.float64)
        )

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
    }
    report["acceptance"] = evaluate_scenario_acceptance(scenario, report)
    return report


def _maximum(values: list[float]) -> float | None:
    return max(values) if values else None


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
    add_maximum(
        "linear_velocity_mae_m_s",
        report["velocity_tracking_error"]["linear_xy"]["mean"],
        ACCEPTANCE_THRESHOLDS["linear_velocity_mae_m_s_max"],
    )
    add_maximum(
        "yaw_velocity_mae_rad_s",
        report["velocity_tracking_error"]["yaw"]["mean"],
        ACCEPTANCE_THRESHOLDS["yaw_velocity_mae_rad_s_max"],
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

    # The foot reward intentionally fades to zero as locomotion demand grows.
    # Only stationary/near-stationary scenarios are valid foot-tracking gates.
    vx, vy, yaw = scenario.twist
    foot_tracking_active = math.hypot(vx, vy) + abs(yaw) <= 0.15
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
) -> dict[str, Any]:
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
    )
    expected_names = [scenario.name for scenario in default_scenarios()]
    actual_names = [item["name"] for item in scenario_reports]
    canonical_coverage = (
        actual_names == expected_names and steps >= 1000 and settle_steps == 50
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
            "Play mode holds the HMD-owned neck at its default pose; moving-neck "
            "inertial disturbance is trained but is not covered by this evaluator."
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
    if checkpoint_contract.diagnostic_legacy:
        limitations.append(
            "This is an explicitly requested legacy-v1 diagnostic using raw "
            "previous-action feedback and the v1 scalar Gaussian actor. It can "
            "never pass the v5 deployment gate or be exported as v5."
        )
    if checkpoint_contract.pristine_pre_update:
        limitations.append(
            "This is the pinned velocity bootstrap before any teleop PPO update. "
            "It is a safety baseline and can never pass the deployment gate."
        )
    return {
        "schema_version": 5,
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
            "v5_deployment_compatible": (
                not checkpoint_contract.diagnostic_legacy
                and not checkpoint_contract.pristine_pre_update
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
        "nominal_environment": {
            "viewer": False,
            "robot_network": False,
            "actor_corruption": False,
            "domain_randomization": False,
            "external_pushes": False,
            "auto_reset": False,
        },
        "summary": {
            "scenario_count": len(scenario_reports),
            "canonical_coverage": canonical_coverage,
            "hard_safety_checks_passed": hard_pass,
            "acceptance_checks_passed": acceptance_pass,
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
    return parser.parse_args()


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
    _configure_nominal_evaluation(env_cfg, steps=args.steps)
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
    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    wrapped_env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)

    try:
        validate_microban_teleop_observation_contract(raw_env)
        runner_cls = load_runner_cls(TASK)
        runner = runner_cls(wrapped_env, asdict(agent_cfg), device=args.device)
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
