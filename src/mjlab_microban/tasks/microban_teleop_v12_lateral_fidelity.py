"""Opt-in lateral-fidelity variant of the forward-lean pose-release recipe.

Diagnosis (2026-10-06): the forward-lean pose-release chain loses the lateral
part of combined forward+lateral twist commands once hands activate (update
7100).  On held-out commands (0.6..0.7, +-0.3, +-1.0..1.2) the lean 7099 walks
+0.078 m/s to the commanded side, its 9500 only +0.018 and its 14999 about
+0.02 (with -0.06 under pushes) while forward speed rises from 0.11 to 0.29;
the centered line keeps +0.07 throughout.  The 15000 final gate's
``mixed_forward_left`` lateral minimum (0.02 m/s) failed in all 10 tries.
The inherited L1 velocity error barely separates the two behaviours because
neither reaches the commanded 0.7 m/s forward speed.

This variant keeps every pose-release term, curriculum stage, push and
command sampler and adds one reward term, ``mixed_command_lateral_deficit``
(``mixed_command_lateral_deficit_l1``), in the HOME-levelled trunk frame.  It is
non-zero only for commands with ``|c_x| >= 0.05`` and ``|c_y| >= 0.05``::

    p_x = v_x * sgn(c_x),  p_y = v_y * sgn(c_y),  r = |c_y| / |c_x|
    deficit = max(0, min(r * p_x, |c_y|) - p_y)

so forward progress without the proportional lateral progress costs, extra
lateral speed never does, and pure-axis commands are untouched.

The variant starts from a gated fresh-chain pose-release ``model_7099`` (the
first gated checkpoint after hand activation).  Every save carries the marker
below under ``microban_teleop_v12_lateral_fidelity``: the variant revision, the
term's weight and activation threshold, and the parent checkpoint and its
stage gate by path and SHA-256.  The recipe revision string stays the
pose-release one (the robot-side contract is unchanged); consumers re-validate
the marker (``microban_teleop_v12_hand_pose_release_lineage``) and the
pose-release runner refuses to resume a marked checkpoint without the term at
the recorded weight, or to add the term to an unmarked descendant.
"""

from __future__ import annotations

import os
import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microban.robot.microban_constants import HOME_TRUNK_PITCH_RAD
from mjlab_microban.tasks.mdp import home_levelled_root_lin_vel_b

MICROBAN_TELEOP_V12_LATERAL_FIDELITY_TASK_ID = (
    "Mjlab-Teleop-V12-HandPoseRelease-LateralFidelity-Microban"
)
MICROBAN_TELEOP_V12_LATERAL_FIDELITY_INFO_KEY = "microban_teleop_v12_lateral_fidelity"
MICROBAN_TELEOP_V12_LATERAL_FIDELITY_REVISION = "hand_pose_release_lateral_fidelity_v1"
MICROBAN_TELEOP_V12_LATERAL_FIDELITY_SCHEMA_VERSION = 1
MICROBAN_TELEOP_V12_LATERAL_FIDELITY_REWARD_NAME = "mixed_command_lateral_deficit"
MICROBAN_TELEOP_V12_LATERAL_FIDELITY_MIN_ABS_COMMAND_M_S = 0.05
MICROBAN_TELEOP_V12_LATERAL_FIDELITY_FRAME = "home_levelled_trunk_pitch_10deg"
# Revision v2: the vx-independent lateral shortfall while a hand is active.
MICROBAN_TELEOP_V12_LATERAL_FIDELITY_V2_REVISION = "hand_pose_release_lateral_fidelity_v2"
MICROBAN_TELEOP_V12_LATERAL_FIDELITY_V2_SCHEMA_VERSION = 2
MICROBAN_TELEOP_V12_LATERAL_FIDELITY_V2_REWARD_NAME = "hand_active_lateral_shortfall"
MICROBAN_TELEOP_V12_LATERAL_FIDELITY_V2_LATERAL_CAP_M_S = 0.10
MICROBAN_TELEOP_V12_LATERAL_FIDELITY_V2_HAND_COMMAND = "hand_target"
# Revision v3: the v2 shortfall plus the forward progress beyond a cap on
# mixed commands while a hand is active.
MICROBAN_TELEOP_V12_LATERAL_FIDELITY_V3_REVISION = "hand_pose_release_lateral_fidelity_v3"
MICROBAN_TELEOP_V12_LATERAL_FIDELITY_V3_SCHEMA_VERSION = 3
MICROBAN_TELEOP_V12_LATERAL_FIDELITY_V3_REWARD_NAME = (
    "hand_active_lateral_shortfall_forward_excess"
)
MICROBAN_TELEOP_V12_LATERAL_FIDELITY_V3_FORWARD_CAP_M_S = 0.15
# Registered weights by label (V1 default, V2 its fallback; s24 is the v2
# revision's first variant, s40 its fallback; x32 the v3 revision's first
# variant, x48 its fallback).  Every weight is unique.
MICROBAN_TELEOP_V12_LATERAL_FIDELITY_WEIGHTS: dict[str, float] = {
    "8": -8.0,
    "16": -16.0,
    "s24": -24.0,
    "s40": -40.0,
    "x32": -32.0,
    "x48": -48.0,
}
MICROBAN_TELEOP_V12_LATERAL_FIDELITY_V2_LABELS = frozenset(("s24", "s40"))
MICROBAN_TELEOP_V12_LATERAL_FIDELITY_V3_LABELS = frozenset(("x32", "x48"))
MICROBAN_TELEOP_V12_LATERAL_FIDELITY_DEFAULT_WEIGHT = "8"
# Read when the task package is imported (the launcher sets it).
MICROBAN_TELEOP_V12_LATERAL_FIDELITY_WEIGHT_ENV = "MICROBAN_V12_LATERAL_FIDELITY_WEIGHT"
MICROBAN_TELEOP_V12_LATERAL_FIDELITY_PARENT_ITERATION = 7_099
MICROBAN_TELEOP_V12_LATERAL_FIDELITY_PARENT_COMPLETED_UPDATES = 7_100
MICROBAN_TELEOP_V12_LATERAL_FIDELITY_NUM_STEPS_PER_ENV = 24
MICROBAN_TELEOP_V12_LATERAL_FIDELITY_REASON = (
    "forward-lean pose-release chain loses lateral velocity on combined "
    "forward+lateral commands after hand activation (held-out lean 7099 +0.078 "
    "m/s, 9500 +0.018 m/s); the 15000 final gate's mixed_forward_left lateral "
    "minimum failed in 10 tries; restart from the gated model_7099 with a "
    "lateral-deficit penalty on mixed commands"
)

MICROBAN_TELEOP_V12_LATERAL_FIDELITY_V2_REASON = (
    "forward-lean pose-release chain loses lateral velocity on forward+lateral "
    "commands once hands activate (held-out lean 7099 +0.07 m/s, 9999 about 0 "
    "with full targets); the v1 deficit scaled with forward progress and failed "
    "its 7500 probes at weights 8 and 16; a crossed probe tied the residual "
    "left/right gap to the command direction, not the target layout or HMD; "
    "restart from the gated model_7099 with a forward-speed-independent "
    "lateral shortfall penalty while a hand is active"
)

MICROBAN_TELEOP_V12_LATERAL_FIDELITY_V3_REASON = (
    "forward-lean pose-release chain trades lateral for forward speed on "
    "forward+lateral commands once hands activate and falls under pushes while "
    "walking fast (gate mixed_forward_left vx 0.41 m/s vs centered 0.06 m/s); "
    "the v2 shortfall alone failed its 8000 probe (vx rose to 0.23 m/s, left "
    "pushed lateral 0.00 m/s); restart from the gated model_7099 with the v2 "
    "shortfall plus the forward progress above 0.15 m/s on mixed commands "
    "while a hand is active"
)

_REPO_PREFIX = "repo://"


@dataclass(frozen=True)
class LateralFidelityVariant:
    """One registered (revision, term, weight) of the lateral-fidelity recipe."""

    label: str
    revision: str
    schema_version: int
    reward_term: str
    weight: float


def lateral_fidelity_variant(label: str) -> LateralFidelityVariant:
    weight = lateral_fidelity_weight(label)
    if label in MICROBAN_TELEOP_V12_LATERAL_FIDELITY_V3_LABELS:
        return LateralFidelityVariant(
            label,
            MICROBAN_TELEOP_V12_LATERAL_FIDELITY_V3_REVISION,
            MICROBAN_TELEOP_V12_LATERAL_FIDELITY_V3_SCHEMA_VERSION,
            MICROBAN_TELEOP_V12_LATERAL_FIDELITY_V3_REWARD_NAME,
            weight,
        )
    if label in MICROBAN_TELEOP_V12_LATERAL_FIDELITY_V2_LABELS:
        return LateralFidelityVariant(
            label,
            MICROBAN_TELEOP_V12_LATERAL_FIDELITY_V2_REVISION,
            MICROBAN_TELEOP_V12_LATERAL_FIDELITY_V2_SCHEMA_VERSION,
            MICROBAN_TELEOP_V12_LATERAL_FIDELITY_V2_REWARD_NAME,
            weight,
        )
    return LateralFidelityVariant(
        label,
        MICROBAN_TELEOP_V12_LATERAL_FIDELITY_REVISION,
        MICROBAN_TELEOP_V12_LATERAL_FIDELITY_SCHEMA_VERSION,
        MICROBAN_TELEOP_V12_LATERAL_FIDELITY_REWARD_NAME,
        weight,
    )


def _canonical(value: object) -> str:
    """Type-exact comparison form (True != 1, 1 != 1.0)."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def mixed_command_lateral_deficit_l1(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    trunk_pitch: float = HOME_TRUNK_PITCH_RAD,
    min_abs_command: float = MICROBAN_TELEOP_V12_LATERAL_FIDELITY_MIN_ABS_COMMAND_M_S,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Lateral progress missing for the forward progress made (see module doc)."""

    command = env.command_manager.get_command(command_name)
    actual = home_levelled_root_lin_vel_b(env, trunk_pitch, asset_cfg)
    return mixed_command_lateral_deficit(command[:, :2], actual[:, :2], min_abs_command)


def mixed_command_lateral_deficit(
    command_xy: torch.Tensor, velocity_xy: torch.Tensor, min_abs_command: float
) -> torch.Tensor:
    """Pure tensor form of the deficit (N, 2) x (N, 2) -> (N,)."""

    cx, cy = command_xy[:, 0], command_xy[:, 1]
    abs_cx, abs_cy = cx.abs(), cy.abs()
    active = (abs_cx >= min_abs_command) & (abs_cy >= min_abs_command)
    progress_x = velocity_xy[:, 0] * torch.sign(cx)
    progress_y = velocity_xy[:, 1] * torch.sign(cy)
    ratio = abs_cy / abs_cx.clamp(min=min_abs_command)
    expected_y = torch.minimum(ratio * progress_x, abs_cy)
    deficit = torch.clamp(expected_y - progress_y, min=0.0)
    return torch.where(active, deficit, torch.zeros_like(deficit))


def hand_active_lateral_shortfall_l1(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    hand_command_name: str = MICROBAN_TELEOP_V12_LATERAL_FIDELITY_V2_HAND_COMMAND,
    trunk_pitch: float = HOME_TRUNK_PITCH_RAD,
    min_abs_command: float = MICROBAN_TELEOP_V12_LATERAL_FIDELITY_MIN_ABS_COMMAND_M_S,
    lateral_cap: float = MICROBAN_TELEOP_V12_LATERAL_FIDELITY_V2_LATERAL_CAP_M_S,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Commanded lateral speed missing while a hand is active (see module doc)."""

    command = env.command_manager.get_command(command_name)
    actual = home_levelled_root_lin_vel_b(env, trunk_pitch, asset_cfg)
    hand = env.command_manager.get_term(hand_command_name)
    return hand_active_lateral_shortfall(
        command[:, :2],
        actual[:, :2],
        hand.is_active.any(dim=-1),
        min_abs_command,
        lateral_cap,
    )


def hand_active_lateral_shortfall(
    command_xy: torch.Tensor,
    velocity_xy: torch.Tensor,
    hand_active: torch.Tensor,
    min_abs_command: float,
    lateral_cap: float,
) -> torch.Tensor:
    """Pure tensor form (N, 2) x (N, 2) x (N,) -> (N,); never reads v_x or c_x."""

    cy = command_xy[:, 1]
    active = (cy.abs() >= min_abs_command) & hand_active.to(torch.bool)
    target = torch.clamp(cy.abs(), max=lateral_cap)
    progress_y = velocity_xy[:, 1] * torch.sign(cy)
    shortfall = torch.clamp(target - progress_y, min=0.0)
    return torch.where(active, shortfall, torch.zeros_like(shortfall))


def hand_active_lateral_shortfall_forward_excess_l1(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    hand_command_name: str = MICROBAN_TELEOP_V12_LATERAL_FIDELITY_V2_HAND_COMMAND,
    trunk_pitch: float = HOME_TRUNK_PITCH_RAD,
    min_abs_command: float = MICROBAN_TELEOP_V12_LATERAL_FIDELITY_MIN_ABS_COMMAND_M_S,
    lateral_cap: float = MICROBAN_TELEOP_V12_LATERAL_FIDELITY_V2_LATERAL_CAP_M_S,
    forward_cap: float = MICROBAN_TELEOP_V12_LATERAL_FIDELITY_V3_FORWARD_CAP_M_S,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """v2 shortfall plus mixed-command forward excess while a hand is active."""

    command = env.command_manager.get_command(command_name)
    actual = home_levelled_root_lin_vel_b(env, trunk_pitch, asset_cfg)
    hand = env.command_manager.get_term(hand_command_name)
    return hand_active_lateral_shortfall_forward_excess(
        command[:, :2],
        actual[:, :2],
        hand.is_active.any(dim=-1),
        min_abs_command,
        lateral_cap,
        forward_cap,
    )


def hand_active_lateral_shortfall_forward_excess(
    command_xy: torch.Tensor,
    velocity_xy: torch.Tensor,
    hand_active: torch.Tensor,
    min_abs_command: float,
    lateral_cap: float,
    forward_cap: float,
) -> torch.Tensor:
    """Shortfall (as v2) + max(0, v_x sgn(c_x) - forward_cap) on mixed commands.

    The forward part applies only when ``|c_x|`` and ``|c_y|`` are both at least
    ``min_abs_command`` and a hand is active: pure forward/backward commands and
    hands-off walking keep their speed.
    """

    shortfall = hand_active_lateral_shortfall(
        command_xy, velocity_xy, hand_active, min_abs_command, lateral_cap
    )
    cx, cy = command_xy[:, 0], command_xy[:, 1]
    mixed = (
        (cx.abs() >= min_abs_command)
        & (cy.abs() >= min_abs_command)
        & hand_active.to(torch.bool)
    )
    excess = torch.clamp(velocity_xy[:, 0] * torch.sign(cx) - forward_cap, min=0.0)
    return shortfall + torch.where(mixed, excess, torch.zeros_like(excess))


def _term_params(variant: LateralFidelityVariant) -> dict[str, Any]:
    if variant.reward_term == MICROBAN_TELEOP_V12_LATERAL_FIDELITY_V3_REWARD_NAME:
        return {
            "command_name": "twist",
            "hand_command_name": MICROBAN_TELEOP_V12_LATERAL_FIDELITY_V2_HAND_COMMAND,
            "trunk_pitch": HOME_TRUNK_PITCH_RAD,
            "min_abs_command": MICROBAN_TELEOP_V12_LATERAL_FIDELITY_MIN_ABS_COMMAND_M_S,
            "lateral_cap": MICROBAN_TELEOP_V12_LATERAL_FIDELITY_V2_LATERAL_CAP_M_S,
            "forward_cap": MICROBAN_TELEOP_V12_LATERAL_FIDELITY_V3_FORWARD_CAP_M_S,
        }
    if variant.reward_term == MICROBAN_TELEOP_V12_LATERAL_FIDELITY_V2_REWARD_NAME:
        return {
            "command_name": "twist",
            "hand_command_name": MICROBAN_TELEOP_V12_LATERAL_FIDELITY_V2_HAND_COMMAND,
            "trunk_pitch": HOME_TRUNK_PITCH_RAD,
            "min_abs_command": MICROBAN_TELEOP_V12_LATERAL_FIDELITY_MIN_ABS_COMMAND_M_S,
            "lateral_cap": MICROBAN_TELEOP_V12_LATERAL_FIDELITY_V2_LATERAL_CAP_M_S,
        }
    return {
        "command_name": "twist",
        "trunk_pitch": HOME_TRUNK_PITCH_RAD,
        "min_abs_command": MICROBAN_TELEOP_V12_LATERAL_FIDELITY_MIN_ABS_COMMAND_M_S,
    }


def _term_func(variant: LateralFidelityVariant):
    if variant.reward_term == MICROBAN_TELEOP_V12_LATERAL_FIDELITY_V3_REWARD_NAME:
        return hand_active_lateral_shortfall_forward_excess_l1
    if variant.reward_term == MICROBAN_TELEOP_V12_LATERAL_FIDELITY_V2_REWARD_NAME:
        return hand_active_lateral_shortfall_l1
    return mixed_command_lateral_deficit_l1


_TERM_NAMES = (
    MICROBAN_TELEOP_V12_LATERAL_FIDELITY_REWARD_NAME,
    MICROBAN_TELEOP_V12_LATERAL_FIDELITY_V2_REWARD_NAME,
    MICROBAN_TELEOP_V12_LATERAL_FIDELITY_V3_REWARD_NAME,
)


def lateral_fidelity_weight(label: str) -> float:
    if not isinstance(label, str):
        raise ValueError(f"Unregistered lateral-fidelity weight {label!r}")
    try:
        return MICROBAN_TELEOP_V12_LATERAL_FIDELITY_WEIGHTS[label]
    except KeyError as exc:
        raise ValueError(f"Unregistered lateral-fidelity weight {label!r}") from exc


def lateral_fidelity_weight_label(weight: float) -> str:
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        raise ValueError(f"Unregistered lateral-fidelity weight {weight!r}")
    for label, value in MICROBAN_TELEOP_V12_LATERAL_FIDELITY_WEIGHTS.items():
        if value == weight:
            return label
    raise ValueError(f"Unregistered lateral-fidelity weight {weight!r}")


def selected_lateral_fidelity_weight_label() -> str:
    label = os.environ.get(
        MICROBAN_TELEOP_V12_LATERAL_FIDELITY_WEIGHT_ENV,
        MICROBAN_TELEOP_V12_LATERAL_FIDELITY_DEFAULT_WEIGHT,
    )
    lateral_fidelity_weight(label)
    return label


def apply_lateral_fidelity(cfg: ManagerBasedRlEnvCfg, weight_label: str) -> None:
    """Add the lateral-deficit term (every other term is left as it is)."""

    variant = lateral_fidelity_variant(weight_label)
    if any(name in cfg.rewards for name in _TERM_NAMES):
        raise ValueError("Lateral-fidelity term is already installed")
    if "twist" not in cfg.commands:
        raise KeyError("Lateral fidelity requires the twist command")
    if (
        variant.reward_term != MICROBAN_TELEOP_V12_LATERAL_FIDELITY_REWARD_NAME
        and MICROBAN_TELEOP_V12_LATERAL_FIDELITY_V2_HAND_COMMAND not in cfg.commands
    ):
        raise KeyError("Lateral fidelity v2 requires the hand_target command")
    cfg.rewards[variant.reward_term] = RewardTermCfg(
        func=_term_func(variant),
        weight=variant.weight,
        params=_term_params(variant),
    )


def installed_lateral_fidelity_weight(env: Any) -> float | None:
    """Weight of the live term in an env's reward manager, or None."""

    manager = env.unwrapped.reward_manager if hasattr(env, "unwrapped") else env.reward_manager
    names = [name for name in _TERM_NAMES if name in manager.active_terms]
    if not names:
        return None
    if len(names) != 1:
        raise RuntimeError("More than one lateral-fidelity term is installed")
    term = manager.get_term_cfg(names[0])
    try:
        variant = lateral_fidelity_variant(lateral_fidelity_weight_label(term.weight))
    except ValueError as exc:
        raise RuntimeError("Lateral-fidelity reward weight is unregistered") from exc
    if (
        variant.reward_term != names[0]
        or term.func is not _term_func(variant)
        or term.params != _term_params(variant)
    ):
        raise RuntimeError("Lateral-fidelity reward term drifted")
    return float(term.weight)


def make_microban_teleop_v12_lateral_fidelity_env_cfg(
    play: bool = False, weight_label: str | None = None
) -> ManagerBasedRlEnvCfg:
    """Pose-release env plus the lateral-deficit term (weight from the launcher)."""

    from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release import (
        make_microban_teleop_v12_hand_pose_release_env_cfg,
    )

    cfg = make_microban_teleop_v12_hand_pose_release_env_cfg(play=play)
    apply_lateral_fidelity(
        cfg,
        weight_label
        if weight_label is not None
        else selected_lateral_fidelity_weight_label(),
    )
    return cfg


def lateral_fidelity_marker(
    *,
    parent_checkpoint_path: str,
    parent_checkpoint_sha256: str,
    parent_stage_gate_path: str,
    parent_stage_gate_sha256: str,
    weight: float,
) -> dict[str, Any]:
    """Build the exact marker (structure only; files are checked separately)."""

    from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
        MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION,
    )

    variant = lateral_fidelity_variant(lateral_fidelity_weight_label(weight))
    if not _is_sha256(parent_checkpoint_sha256) or not _is_sha256(
        parent_stage_gate_sha256
    ):
        raise ValueError("Lateral-fidelity parent SHA-256 is malformed")
    for name, path in (
        ("checkpoint", parent_checkpoint_path),
        ("stage gate", parent_stage_gate_path),
    ):
        if (
            not isinstance(path, str)
            or not path.startswith(_REPO_PREFIX)
            or ".." in path.split("/")
        ):
            raise ValueError(f"Lateral-fidelity parent {name} must be a repo:// path")
    if not parent_checkpoint_path.endswith(
        f"/model_{MICROBAN_TELEOP_V12_LATERAL_FIDELITY_PARENT_ITERATION}.pt"
    ):
        raise ValueError("Lateral-fidelity parent must be a model_7099.pt")
    v3 = variant.revision == MICROBAN_TELEOP_V12_LATERAL_FIDELITY_V3_REVISION
    v2 = v3 or variant.revision == MICROBAN_TELEOP_V12_LATERAL_FIDELITY_V2_REVISION
    marker: dict[str, Any] = {
        "schema_version": variant.schema_version,
        "revision": variant.revision,
        "recipe_revision": MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION,
        "reward_term": variant.reward_term,
        "reward_weight": variant.weight,
        "min_abs_command_m_s": MICROBAN_TELEOP_V12_LATERAL_FIDELITY_MIN_ABS_COMMAND_M_S,
    }
    if v2:
        marker["lateral_cap_m_s"] = MICROBAN_TELEOP_V12_LATERAL_FIDELITY_V2_LATERAL_CAP_M_S
        marker["requires_active_hand"] = True
    if v3:
        marker["forward_cap_m_s"] = MICROBAN_TELEOP_V12_LATERAL_FIDELITY_V3_FORWARD_CAP_M_S
    marker.update({
        "velocity_frame": MICROBAN_TELEOP_V12_LATERAL_FIDELITY_FRAME,
        "parent_lineage": "fresh_chain",
        "parent_checkpoint_path": parent_checkpoint_path,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "parent_iteration": MICROBAN_TELEOP_V12_LATERAL_FIDELITY_PARENT_ITERATION,
        "parent_completed_updates": (
            MICROBAN_TELEOP_V12_LATERAL_FIDELITY_PARENT_COMPLETED_UPDATES
        ),
        "parent_stage_gate_path": parent_stage_gate_path,
        "parent_stage_gate_sha256": parent_stage_gate_sha256,
        "reason": (
            MICROBAN_TELEOP_V12_LATERAL_FIDELITY_V3_REASON
            if v3
            else MICROBAN_TELEOP_V12_LATERAL_FIDELITY_V2_REASON
            if v2
            else MICROBAN_TELEOP_V12_LATERAL_FIDELITY_REASON
        ),
    })
    return marker


def validate_lateral_fidelity_marker(value: object) -> dict[str, Any]:
    """Rebuild the marker from its recorded values and require equality."""

    if not isinstance(value, Mapping):
        raise ValueError("Lateral-fidelity marker is malformed")
    try:
        expected = lateral_fidelity_marker(
            parent_checkpoint_path=value.get("parent_checkpoint_path"),  # type: ignore[arg-type]
            parent_checkpoint_sha256=value.get("parent_checkpoint_sha256"),  # type: ignore[arg-type]
            parent_stage_gate_path=value.get("parent_stage_gate_path"),  # type: ignore[arg-type]
            parent_stage_gate_sha256=value.get("parent_stage_gate_sha256"),  # type: ignore[arg-type]
            weight=value.get("reward_weight"),  # type: ignore[arg-type]
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Lateral-fidelity marker drifted") from exc
    try:
        same = _canonical(dict(value)) == _canonical(expected)
    except (TypeError, ValueError):
        same = False
    if not same:
        raise ValueError("Lateral-fidelity marker drifted")
    return deepcopy(expected)


def validate_lateral_fidelity_parent_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """The parent's own bytes: an unmarked fresh pose-release model_7099."""

    from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
        MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION,
    )
    from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release_lineage import (
        HAND_POSE_RELEASE_LINEAGE_FRESH,
        hand_pose_release_lineage,
    )

    if not isinstance(payload, Mapping):
        raise TypeError("Lateral-fidelity parent payload is malformed")
    infos = payload.get("infos")
    if not isinstance(infos, Mapping):
        raise TypeError("Lateral-fidelity parent infos are missing")
    if payload.get("iter") != MICROBAN_TELEOP_V12_LATERAL_FIDELITY_PARENT_ITERATION:
        raise ValueError("Lateral-fidelity parent must be iteration 7099")
    env_state = infos.get("env_state")
    if not isinstance(env_state, Mapping) or env_state.get("common_step_counter") != (
        MICROBAN_TELEOP_V12_LATERAL_FIDELITY_PARENT_COMPLETED_UPDATES
        * MICROBAN_TELEOP_V12_LATERAL_FIDELITY_NUM_STEPS_PER_ENV
    ):
        raise ValueError("Lateral-fidelity parent clock drifted")
    if infos.get("microban_teleop_recipe_revision") != (
        MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION
    ):
        raise ValueError("Lateral-fidelity parent is not the pose-release recipe")
    if infos.get(MICROBAN_TELEOP_V12_LATERAL_FIDELITY_INFO_KEY) is not None:
        raise ValueError("Lateral-fidelity parent already carries the marker")
    if (
        hand_pose_release_lineage(
            infos,
            iteration=MICROBAN_TELEOP_V12_LATERAL_FIDELITY_PARENT_ITERATION,
            verify_parent=False,
        )
        != HAND_POSE_RELEASE_LINEAGE_FRESH
    ):
        raise ValueError("Lateral-fidelity parent must be a fresh pose-release chain")
    return dict(infos)


def verify_lateral_fidelity_parent(marker: Mapping[str, Any]) -> dict[str, Any]:
    """Re-hash the parent files and re-run the stage-gate validator on them.

    Returns the parent's checkpoint infos.
    """

    from mjlab_microban.tasks.microban_teleop_v12_bootstrap import (
        resolve_bootstrap_artifact_path,
        sha256_file,
    )

    marker = validate_lateral_fidelity_marker(marker)
    checkpoint = resolve_bootstrap_artifact_path(marker["parent_checkpoint_path"])
    gate_path = resolve_bootstrap_artifact_path(marker["parent_stage_gate_path"])
    if not checkpoint.is_file() or not gate_path.is_file():
        raise ValueError("Lateral-fidelity parent checkpoint or gate is missing")
    if sha256_file(checkpoint) != marker["parent_checkpoint_sha256"]:
        raise ValueError("Lateral-fidelity parent checkpoint changed")
    if sha256_file(gate_path) != marker["parent_stage_gate_sha256"]:
        raise ValueError("Lateral-fidelity parent stage gate changed")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    parent_infos = validate_lateral_fidelity_parent_payload(payload)
    # Imported here: the stage module imports the lineage validators.
    from mjlab_microban.scripts.teleop_v12_stage import validate_gate

    gate = validate_gate(gate_path, checkpoint)
    if (
        gate.get("checkpoint_sha256") != marker["parent_checkpoint_sha256"]
        or gate.get("iteration") != MICROBAN_TELEOP_V12_LATERAL_FIDELITY_PARENT_ITERATION
        or gate.get("completed_updates")
        != MICROBAN_TELEOP_V12_LATERAL_FIDELITY_PARENT_COMPLETED_UPDATES
        or gate.get("status") != "pass"
    ):
        raise ValueError("Lateral-fidelity parent gate does not gate model_7099")
    if (
        sha256_file(checkpoint) != marker["parent_checkpoint_sha256"]
        or sha256_file(gate_path) != marker["parent_stage_gate_sha256"]
    ):
        raise ValueError("Lateral-fidelity parent changed while validating")
    return parent_infos


# Lineage markers a descendant must share with its parent model_7099.
LATERAL_FIDELITY_SHARED_PARENT_INFO_KEYS = (
    "legacy_velocity_actor_bootstrap_v12",
    "microban_teleop_v12_home_pose",
    "bilateral_site_order_revision",
    "microban_teleop_v12_lr_order_migration",
    "adapter_sanitization",
    "adapter_gradient_schedule_revision",
    "microban_teleop_training_contract_version",
    "previous_action_semantics",
    "action_clip",
)


def validate_lateral_fidelity_infos(
    infos: Mapping[str, Any],
    *,
    iteration: int | None = None,
    verify_parent: bool = True,
) -> dict[str, Any] | None:
    """Validate the marker of a checkpoint (None when it carries none)."""

    value = infos.get(MICROBAN_TELEOP_V12_LATERAL_FIDELITY_INFO_KEY)
    if value is None:
        return None
    marker = validate_lateral_fidelity_marker(value)
    if iteration is not None and (
        isinstance(iteration, bool)
        or not isinstance(iteration, int)
        or iteration <= MICROBAN_TELEOP_V12_LATERAL_FIDELITY_PARENT_ITERATION
    ):
        raise ValueError("Lateral-fidelity descendant clock is invalid")
    if verify_parent:
        parent_infos = verify_lateral_fidelity_parent(marker)
        for key in LATERAL_FIDELITY_SHARED_PARENT_INFO_KEYS:
            if parent_infos.get(key) != infos.get(key):
                raise ValueError(
                    f"Lateral-fidelity descendant {key} differs from its parent"
                )
    return marker
