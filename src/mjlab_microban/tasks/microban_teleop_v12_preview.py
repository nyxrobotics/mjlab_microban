"""Fail-closed identity contract for simulation-only v12 preview checkpoints."""

from __future__ import annotations

from typing import Any

from mjlab_microban.tasks.microban_teleop_v12_actor import (
    TELEOP_V12_EXTRA_OBSERVATION_COLUMNS,
)

MICROBAN_TELEOP_V12_PREVIEW_TASK_ID = "Mjlab-Teleop-V12-Preview-Microban"
TELEOP_V12_PREVIEW_INFO_KEY = "teleop_v12_preview"
TELEOP_V12_PREVIEW_REVISION = "sanitized601_clock_lift_to10001_full20_sim_only_v1"
TELEOP_V12_PREVIEW_SOURCE_SHA256 = (
    "d31d5362dc6776a47395bcefc446bb4324d87361e423edc4a575b2b972d269a3"
)
TELEOP_V12_PREVIEW_SOURCE_ITERATION = 600
TELEOP_V12_PREVIEW_SOURCE_COMPLETED_UPDATES = 601
TELEOP_V12_PREVIEW_LIFTED_ITERATION = 10_000
TELEOP_V12_PREVIEW_LIFTED_COMPLETED_UPDATES = 10_001


def canonical_preview_info() -> dict[str, Any]:
    """Return the exact permanent lineage marker for the isolated preview."""

    return {
        "schema_version": 1,
        "revision": TELEOP_V12_PREVIEW_REVISION,
        "simulation_only": True,
        "source_checkpoint_sha256": TELEOP_V12_PREVIEW_SOURCE_SHA256,
        "source_iteration": TELEOP_V12_PREVIEW_SOURCE_ITERATION,
        "source_completed_updates": TELEOP_V12_PREVIEW_SOURCE_COMPLETED_UPDATES,
        "lifted_iteration": TELEOP_V12_PREVIEW_LIFTED_ITERATION,
        "lifted_completed_updates": TELEOP_V12_PREVIEW_LIFTED_COMPLETED_UPDATES,
        "active_actor_columns": list(TELEOP_V12_EXTRA_OBSERVATION_COLUMNS),
        "target_activation_asserted": True,
        "reward_activation_asserted": True,
    }


def validate_preview_marker(infos: object, *, iteration: int) -> dict[str, Any]:
    """Authenticate the non-deployable marker on a lifted or trained preview."""

    if not isinstance(infos, dict):
        raise TypeError("Preview checkpoint infos are malformed")
    if infos.get("preview_non_deployable") is not True:
        raise ValueError("Preview checkpoint lacks preview_non_deployable=true")
    marker = infos.get(TELEOP_V12_PREVIEW_INFO_KEY)
    expected = canonical_preview_info()
    if not isinstance(marker, dict) or marker != expected:
        raise ValueError("Preview checkpoint lineage marker drifted")
    if iteration < TELEOP_V12_PREVIEW_LIFTED_ITERATION:
        raise ValueError("Preview checkpoint iteration predates its clock lift")
    return marker


def reject_preview_checkpoint(infos: object) -> None:
    """Reject either preview marker independently, including partial forgery."""

    if not isinstance(infos, dict):
        return
    if infos.get("preview_non_deployable") is not None or (
        infos.get(TELEOP_V12_PREVIEW_INFO_KEY) is not None
    ):
        raise ValueError(
            "Simulation-only v12 preview checkpoint is forbidden in the "
            "canonical/deployment path"
        )
