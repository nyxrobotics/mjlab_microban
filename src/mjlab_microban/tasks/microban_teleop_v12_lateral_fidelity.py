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
from collections.abc import Mapping
from copy import deepcopy
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
# Registered weights (V1 default; V2 is the declared fallback).
MICROBAN_TELEOP_V12_LATERAL_FIDELITY_WEIGHTS: dict[str, float] = {
    "8": -8.0,
    "16": -16.0,
}
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

_REPO_PREFIX = "repo://"


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


def lateral_fidelity_weight(label: str) -> float:
    try:
        return MICROBAN_TELEOP_V12_LATERAL_FIDELITY_WEIGHTS[label]
    except KeyError as exc:
        raise ValueError(f"Unregistered lateral-fidelity weight {label!r}") from exc


def lateral_fidelity_weight_label(weight: float) -> str:
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

    if MICROBAN_TELEOP_V12_LATERAL_FIDELITY_REWARD_NAME in cfg.rewards:
        raise ValueError("Lateral-fidelity term is already installed")
    if "twist" not in cfg.commands:
        raise KeyError("Lateral fidelity requires the twist command")
    cfg.rewards[MICROBAN_TELEOP_V12_LATERAL_FIDELITY_REWARD_NAME] = RewardTermCfg(
        func=mixed_command_lateral_deficit_l1,
        weight=lateral_fidelity_weight(weight_label),
        params={
            "command_name": "twist",
            "trunk_pitch": HOME_TRUNK_PITCH_RAD,
            "min_abs_command": MICROBAN_TELEOP_V12_LATERAL_FIDELITY_MIN_ABS_COMMAND_M_S,
        },
    )


def installed_lateral_fidelity_weight(env: Any) -> float | None:
    """Weight of the live term in an env's reward manager, or None."""

    manager = env.unwrapped.reward_manager if hasattr(env, "unwrapped") else env.reward_manager
    if MICROBAN_TELEOP_V12_LATERAL_FIDELITY_REWARD_NAME not in manager.active_terms:
        return None
    term = manager.get_term_cfg(MICROBAN_TELEOP_V12_LATERAL_FIDELITY_REWARD_NAME)
    if term.func is not mixed_command_lateral_deficit_l1 or term.params != {
        "command_name": "twist",
        "trunk_pitch": HOME_TRUNK_PITCH_RAD,
        "min_abs_command": MICROBAN_TELEOP_V12_LATERAL_FIDELITY_MIN_ABS_COMMAND_M_S,
    }:
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

    label = lateral_fidelity_weight_label(weight)
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
    return {
        "schema_version": MICROBAN_TELEOP_V12_LATERAL_FIDELITY_SCHEMA_VERSION,
        "revision": MICROBAN_TELEOP_V12_LATERAL_FIDELITY_REVISION,
        "recipe_revision": MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION,
        "reward_term": MICROBAN_TELEOP_V12_LATERAL_FIDELITY_REWARD_NAME,
        "reward_weight": MICROBAN_TELEOP_V12_LATERAL_FIDELITY_WEIGHTS[label],
        "min_abs_command_m_s": MICROBAN_TELEOP_V12_LATERAL_FIDELITY_MIN_ABS_COMMAND_M_S,
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
        "reason": MICROBAN_TELEOP_V12_LATERAL_FIDELITY_REASON,
    }


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
    if dict(value) != expected:
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
