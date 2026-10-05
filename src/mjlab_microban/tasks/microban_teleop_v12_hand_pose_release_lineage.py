"""Release lineage of the active-hand arm pose-release recipe.

A pose-release checkpoint is release-eligible when its lineage is either

* a fresh pose-release chain (no switch marker at all; the recipe equals v11
  bit for bit before hand targets activate at update 7000), or
* the one recorded recipe switch at a pinned, gated canonical model_7099
  (completed 7100, just after the HMD/hand activation canary): the canonical
  v11 chain is resumed by the pose-release task from that exact checkpoint,
  and every later save carries the release-switch marker below.

Forward-lean HOME: no switch parent is pinned
(``HAND_POSE_RELEASE_RECIPE_SWITCH_PARENT_CHECKPOINT_SHA256`` is ``None``), so
every switch marker is refused and the lean pose-release recipe is trained as a
fresh chain (``scripts/train_microban_teleop_v12.sh start --hand-pose-release``).
The centered line's pin (its gated centered-HOME model_7099) is a different
HOME and recipe string and does not carry over.

The marker names the parent checkpoint and its stage gate by path and SHA-256.
Every consumer re-validates it on every load: the parent bytes must still hash
to the pinned SHA-256, the parent gate must still hash to the recorded value
and pass the stage-gate validator for exactly that checkpoint, and the parent
must be an unmarked canonical v11 checkpoint with the same frozen-source
bootstrap provenance as the descendant.

The older experimental switch (``release_eligible: False``, any parent) is
unchanged and stays evidence-only: stage gates and the exporter refuse it.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import torch

from mjlab_microban.tasks.microban_teleop_v12_bootstrap import (
    portable_bootstrap_artifact_path,
    resolve_bootstrap_artifact_path,
    sha256_file,
)
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION,
    MICROBAN_TELEOP_V12_RECIPE_REVISION,
)

# Checkpoint infos key of the release-eligible switch; distinct from the
# experimental key so the two can never be confused.
MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_SWITCH_INFO_KEY = (
    "microban_teleop_v12_hand_pose_release_recipe_switch"
)
# Same literal as microban_teleop_v12_hand_pose_release's experimental key
# (kept here so this module stays importable without the reward code).
MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_EXPERIMENTAL_SWITCH_INFO_KEY = (
    "microban_teleop_v12_experimental_recipe_switch"
)
HAND_POSE_RELEASE_RECIPE_SWITCH_SCHEMA_VERSION = 1
HAND_POSE_RELEASE_RECIPE_SWITCH_PARENT_ITERATION = 7_099
HAND_POSE_RELEASE_RECIPE_SWITCH_PARENT_COMPLETED_UPDATES = 7_100
HAND_POSE_RELEASE_RECIPE_SWITCH_NUM_STEPS_PER_ENV = 24
# Forward-lean HOME: no gated lean model_7099 is pinned, so the release switch
# is closed and the fresh pose-release chain is the release route.  (The
# centered line pinned its centered-HOME model_7099 here.)  Pinning a lean
# parent later re-opens the switch with every check below unchanged.
HAND_POSE_RELEASE_RECIPE_SWITCH_PARENT_CHECKPOINT_SHA256: str | None = None
_NO_SWITCH_PARENT_MESSAGE = (
    "No release-eligible pose-release switch parent is pinned at the "
    "forward-lean HOME; start a fresh pose-release chain "
    "(train_microban_teleop_v12.sh start --hand-pose-release)"
)
HAND_POSE_RELEASE_RECIPE_SWITCH_REASON = (
    "canonical v11 chain from this model_7099 failed the deployed-accuracy final "
    "gate on active-hand RMS (steady hand undershoot: the inherited HOME pose "
    "reward pulls active-hand arms back to HOME); switch to the pose-release "
    "recipe at the first gated checkpoint after hand activation, identical to "
    "v11 before update 7000"
)

HAND_POSE_RELEASE_LINEAGE_FRESH = "fresh_chain"
# A fresh chain whose model_9900 went through the pose-release corner rescue
# (microban_teleop_v12_corner_rescue, pose-release variant): its model_9999 and
# that model's ordinary pose-release descendants.
HAND_POSE_RELEASE_LINEAGE_FRESH_CORNER_RESCUE = "fresh_chain_model9900_corner_rescue"
HAND_POSE_RELEASE_LINEAGE_RELEASE_SWITCH = "release_eligible_recipe_switch"
HAND_POSE_RELEASE_LINEAGE_EXPERIMENTAL_SWITCH = "experimental_recipe_switch"

_REPO_PREFIX = "repo://"


def _rescue_info_keys() -> tuple[str, ...]:
    """Markers of every other v12 lineage extension (imported lazily)."""

    from mjlab_microban.tasks.microban_teleop_v12_corner_rescue import (
        MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY,
    )
    from mjlab_microban.tasks.microban_teleop_v12_deadline_fallback import (
        MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_INFO_KEY,
        MICROBAN_TELEOP_V12_DEADLINE_POST_CANARY_INFO_KEY,
    )
    from mjlab_microban.tasks.microban_teleop_v12_final_rescue import (
        MICROBAN_TELEOP_V12_FINAL_RESCUE_INFO_KEY,
    )

    return (
        MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY,
        MICROBAN_TELEOP_V12_FINAL_RESCUE_INFO_KEY,
        MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_INFO_KEY,
        MICROBAN_TELEOP_V12_DEADLINE_POST_CANARY_INFO_KEY,
    )


def _parent_forbidden_info_keys() -> tuple[str, ...]:
    from mjlab_microban.tasks.microban_teleop_v12_preview import (
        TELEOP_V12_PREVIEW_INFO_KEY,
    )

    return (
        MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_SWITCH_INFO_KEY,
        MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_EXPERIMENTAL_SWITCH_INFO_KEY,
        *_rescue_info_keys(),
        TELEOP_V12_PREVIEW_INFO_KEY,
        "preview_non_deployable",
    )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def hand_pose_release_recipe_switch_marker(
    *,
    parent_checkpoint_path: str,
    parent_checkpoint_sha256: str,
    parent_stage_gate_path: str,
    parent_stage_gate_sha256: str,
) -> dict[str, Any]:
    """Build the exact release-eligible switch marker (structure only)."""

    if HAND_POSE_RELEASE_RECIPE_SWITCH_PARENT_CHECKPOINT_SHA256 is None:
        raise ValueError(_NO_SWITCH_PARENT_MESSAGE)
    if parent_checkpoint_sha256 != (
        HAND_POSE_RELEASE_RECIPE_SWITCH_PARENT_CHECKPOINT_SHA256
    ):
        raise ValueError(
            "The release-eligible pose-release switch is pinned to the gated "
            "canonical model_7099"
        )
    if not _is_sha256(parent_stage_gate_sha256):
        raise ValueError("Pose-release switch parent gate SHA-256 is malformed")
    for label, path in (
        ("checkpoint", parent_checkpoint_path),
        ("stage gate", parent_stage_gate_path),
    ):
        if (
            not isinstance(path, str)
            or not path.startswith(_REPO_PREFIX)
            or ".." in path.split("/")
        ):
            raise ValueError(
                f"Pose-release switch parent {label} must be a repo:// path"
            )
    if not parent_checkpoint_path.endswith(
        f"/model_{HAND_POSE_RELEASE_RECIPE_SWITCH_PARENT_ITERATION}.pt"
    ):
        raise ValueError("Pose-release switch parent must be a model_7099.pt")
    return {
        "schema_version": HAND_POSE_RELEASE_RECIPE_SWITCH_SCHEMA_VERSION,
        "release_eligible": True,
        "parent_recipe_revision": MICROBAN_TELEOP_V12_RECIPE_REVISION,
        "recipe_revision": MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION,
        "parent_checkpoint_path": parent_checkpoint_path,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "parent_iteration": HAND_POSE_RELEASE_RECIPE_SWITCH_PARENT_ITERATION,
        "parent_completed_updates": (
            HAND_POSE_RELEASE_RECIPE_SWITCH_PARENT_COMPLETED_UPDATES
        ),
        "parent_stage_gate_path": parent_stage_gate_path,
        "parent_stage_gate_sha256": parent_stage_gate_sha256,
        "reason": HAND_POSE_RELEASE_RECIPE_SWITCH_REASON,
    }


def validate_hand_pose_release_recipe_switch_marker(value: object) -> dict[str, Any]:
    """Rebuild the marker from its recorded paths/hashes and require equality."""

    if not isinstance(value, Mapping):
        raise ValueError("Pose-release recipe switch marker is malformed")
    try:
        expected = hand_pose_release_recipe_switch_marker(
            parent_checkpoint_path=value.get("parent_checkpoint_path"),  # type: ignore[arg-type]
            parent_checkpoint_sha256=value.get("parent_checkpoint_sha256"),  # type: ignore[arg-type]
            parent_stage_gate_path=value.get("parent_stage_gate_path"),  # type: ignore[arg-type]
            parent_stage_gate_sha256=value.get("parent_stage_gate_sha256"),  # type: ignore[arg-type]
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Pose-release recipe switch marker drifted") from exc
    if dict(value) != expected:
        raise ValueError("Pose-release recipe switch marker drifted")
    return deepcopy(expected)


def validate_hand_pose_release_switch_parent_payload(
    payload: Mapping[str, Any], *, checkpoint_sha256: str
) -> dict[str, Any]:
    """Validate the parent's own bytes: pinned canonical v11 model_7099."""

    if HAND_POSE_RELEASE_RECIPE_SWITCH_PARENT_CHECKPOINT_SHA256 is None:
        raise ValueError(_NO_SWITCH_PARENT_MESSAGE)
    if checkpoint_sha256 != HAND_POSE_RELEASE_RECIPE_SWITCH_PARENT_CHECKPOINT_SHA256:
        raise ValueError(
            "The release-eligible pose-release switch may resume only the gated "
            "canonical model_7099 "
            f"({HAND_POSE_RELEASE_RECIPE_SWITCH_PARENT_CHECKPOINT_SHA256})"
        )
    if not isinstance(payload, Mapping):
        raise TypeError("Pose-release switch parent payload is malformed")
    infos = payload.get("infos")
    if not isinstance(infos, Mapping):
        raise TypeError("Pose-release switch parent infos are missing")
    if payload.get("iter") != HAND_POSE_RELEASE_RECIPE_SWITCH_PARENT_ITERATION:
        raise ValueError("Pose-release switch parent must be iteration 7099")
    env_state = infos.get("env_state")
    if not isinstance(env_state, Mapping) or env_state.get(
        "common_step_counter"
    ) != (
        HAND_POSE_RELEASE_RECIPE_SWITCH_PARENT_COMPLETED_UPDATES
        * HAND_POSE_RELEASE_RECIPE_SWITCH_NUM_STEPS_PER_ENV
    ):
        raise ValueError("Pose-release switch parent clock drifted")
    if infos.get("microban_teleop_recipe_revision") != (
        MICROBAN_TELEOP_V12_RECIPE_REVISION
    ):
        raise ValueError("Pose-release switch parent is not the canonical v11 recipe")
    present = [
        key for key in _parent_forbidden_info_keys() if infos.get(key) is not None
    ]
    if present:
        raise ValueError(
            "Pose-release switch parent must be an unmarked canonical checkpoint: "
            + ", ".join(present)
        )
    return dict(infos)


def verify_hand_pose_release_switch_parent(
    marker: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-hash the parent files and re-run the stage-gate validator on them.

    Returns the parent's checkpoint infos.
    """

    marker = validate_hand_pose_release_recipe_switch_marker(marker)
    checkpoint = resolve_bootstrap_artifact_path(marker["parent_checkpoint_path"])
    gate_path = resolve_bootstrap_artifact_path(marker["parent_stage_gate_path"])
    if not checkpoint.is_file() or not gate_path.is_file():
        raise ValueError("Pose-release switch parent checkpoint or gate is missing")
    if sha256_file(checkpoint) != marker["parent_checkpoint_sha256"]:
        raise ValueError("Pose-release switch parent checkpoint changed")
    if sha256_file(gate_path) != marker["parent_stage_gate_sha256"]:
        raise ValueError("Pose-release switch parent stage gate changed")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    parent_infos = validate_hand_pose_release_switch_parent_payload(
        payload, checkpoint_sha256=marker["parent_checkpoint_sha256"]
    )
    # Imported here: the stage module imports the lineage validators.
    from mjlab_microban.scripts.teleop_v12_stage import validate_gate

    gate = validate_gate(gate_path, checkpoint)
    if (
        gate.get("checkpoint_sha256") != marker["parent_checkpoint_sha256"]
        or gate.get("iteration") != HAND_POSE_RELEASE_RECIPE_SWITCH_PARENT_ITERATION
        or gate.get("completed_updates")
        != HAND_POSE_RELEASE_RECIPE_SWITCH_PARENT_COMPLETED_UPDATES
        or gate.get("status") != "pass"
        or any(gate.get(key) is not None for key in _rescue_info_keys())
    ):
        raise ValueError("Pose-release switch parent gate does not gate model_7099")
    if (
        sha256_file(checkpoint) != marker["parent_checkpoint_sha256"]
        or sha256_file(gate_path) != marker["parent_stage_gate_sha256"]
    ):
        raise ValueError("Pose-release switch parent changed while validating")
    return parent_infos


def hand_pose_release_lineage(
    infos: Mapping[str, Any],
    *,
    iteration: int | None = None,
    allow_experimental: bool = False,
    verify_parent: bool = True,
) -> str:
    """Classify (and validate) the lineage of a pose-release-recipe checkpoint.

    Returns ``fresh_chain``, ``release_eligible_recipe_switch`` or, only with
    ``allow_experimental``, ``experimental_recipe_switch``.  ``verify_parent``
    re-validates the release switch's parent files and gate (every real load
    does; structural callers such as the HOME-pose check skip the file I/O).
    """

    if not isinstance(infos, Mapping):
        raise TypeError("Contract-v12 checkpoint infos are malformed")
    if infos.get("microban_teleop_recipe_revision") != (
        MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION
    ):
        raise ValueError("Checkpoint is not the hand pose-release recipe")
    from mjlab_microban.tasks.microban_teleop_v12_corner_rescue import (
        MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY,
        MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_ITERATION,
        is_hand_pose_release_corner_rescue_marker,
        validate_corner_rescue_lineage_marker,
    )

    corner = infos.get(MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY)
    for key in _rescue_info_keys():
        if key == MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY:
            continue
        if infos.get(key) is not None:
            raise ValueError("Hand pose-release checkpoints cannot carry a rescue")
    if corner is not None:
        # Only the pose-release corner rescue is part of this lineage; its
        # intermediate saves (9901..9998) are not consumable.
        if not is_hand_pose_release_corner_rescue_marker(corner):
            raise ValueError("Hand pose-release checkpoints cannot carry a rescue")
        validate_corner_rescue_lineage_marker(corner)
        if iteration is not None and (
            isinstance(iteration, bool)
            or not isinstance(iteration, int)
            or iteration < MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_ITERATION
        ):
            raise ValueError(
                "Only model_9999 of a pose-release corner rescue (or a "
                "descendant) is consumable"
            )
    release = infos.get(MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_SWITCH_INFO_KEY)
    experimental = infos.get(
        MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_EXPERIMENTAL_SWITCH_INFO_KEY
    )
    if release is not None and experimental is not None:
        raise ValueError("A checkpoint cannot carry both pose-release switch markers")
    if corner is not None and (release is not None or experimental is not None):
        raise ValueError(
            "The pose-release corner rescue applies only to a fresh pose-release chain"
        )
    if experimental is not None:
        if not allow_experimental:
            raise ValueError(
                "Experimental pose-release recipe switch is not release-eligible"
            )
        from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release_runner import (
            validate_hand_pose_release_switch_marker,
        )

        validate_hand_pose_release_switch_marker(experimental)
        return HAND_POSE_RELEASE_LINEAGE_EXPERIMENTAL_SWITCH
    if release is None:
        if corner is not None:
            return HAND_POSE_RELEASE_LINEAGE_FRESH_CORNER_RESCUE
        return HAND_POSE_RELEASE_LINEAGE_FRESH
    marker = validate_hand_pose_release_recipe_switch_marker(release)
    if iteration is not None and (
        isinstance(iteration, bool)
        or not isinstance(iteration, int)
        or iteration <= HAND_POSE_RELEASE_RECIPE_SWITCH_PARENT_ITERATION
    ):
        raise ValueError("Pose-release switch descendant clock is invalid")
    if verify_parent:
        parent_infos = verify_hand_pose_release_switch_parent(marker)
        for key in (
            "legacy_velocity_actor_bootstrap_v12",
            "microban_teleop_v12_home_pose",
            "bilateral_site_order_revision",
            "microban_teleop_v12_lr_order_migration",
            "adapter_sanitization",
            "adapter_gradient_schedule_revision",
            "microban_teleop_training_contract_version",
            "previous_action_semantics",
            "action_clip",
        ):
            if parent_infos.get(key) != infos.get(key):
                raise ValueError(
                    f"Pose-release switch descendant {key} differs from its parent"
                )
    return HAND_POSE_RELEASE_LINEAGE_RELEASE_SWITCH


def hand_pose_release_switch_parent_paths(
    checkpoint: str, gate: str
) -> tuple[str, str]:
    """Portable repo:// forms of a parent checkpoint and its stage gate."""

    return (
        portable_bootstrap_artifact_path(checkpoint),
        portable_bootstrap_artifact_path(gate),
    )
