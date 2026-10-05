"""Fail-closed runner for the pose-release 14900->14999 final-scenario rescue."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from mjlab_microban.tasks.microban_teleop_v12_corner_rescue import (
    MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY,
    canonical_json_sha256,
)
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION,
)
from mjlab_microban.tasks.microban_teleop_v12_final_rescue import (
    MICROBAN_TELEOP_V12_FINAL_RESCUE_ACTIVE_COLUMNS,
    MICROBAN_TELEOP_V12_FINAL_RESCUE_INFO_KEY,
    MICROBAN_TELEOP_V12_FINAL_RESCUE_PARENT_COMMON_STEP,
    MICROBAN_TELEOP_V12_FINAL_RESCUE_PARENT_ITERATION,
    MICROBAN_TELEOP_V12_FINAL_RESCUE_PARENT_OPTIMIZER_STEP,
    MICROBAN_TELEOP_V12_FINAL_RESCUE_TARGET_COMPLETED_UPDATES,
    MICROBAN_TELEOP_V12_FINAL_RESCUE_TARGET_ITERATION,
    assert_final_rescue_optimizer_step,
)
from mjlab_microban.tasks.microban_teleop_v12_final_rescue_runner import (
    MicrobanTeleopV12FinalRescueOnPolicyRunner,
)
from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release_final_rescue import (
    MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_FAILED_GATE_REPORT_FILENAME,
    MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_RESCUABLE_CHECKS,
    failed_final_gate_scenarios,
    final_gate_profile,
    hand_pose_release_final_rescue_marker,
    validate_hand_pose_release_final_rescue_marker,
)
from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release_runner import (
    MicrobanTeleopV12HandPoseReleaseOnPolicyRunner,
)


def validate_hand_pose_release_final_rescue_parent_payload(
    payload: dict[str, Any], *, checkpoint_sha256: str
) -> dict[str, Any] | None:
    """Validate the only checkpoint the pose-release final rescue resumes.

    An unmarked (no final rescue, no recipe switch) pose-release model_14900 of
    a fresh chain or of a fresh chain through the pose-release model9900
    corner rescue.  Returns its validated corner marker (or ``None``).
    """

    from mjlab_microban.tasks.microban_teleop_v12_corner_rescue import (
        validate_corner_rescue_lineage_marker,
    )
    from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release_lineage import (
        HAND_POSE_RELEASE_LINEAGE_FRESH,
        HAND_POSE_RELEASE_LINEAGE_FRESH_CORNER_RESCUE,
        hand_pose_release_lineage,
    )

    if (
        not isinstance(checkpoint_sha256, str)
        or len(checkpoint_sha256) != 64
        or any(c not in "0123456789abcdef" for c in checkpoint_sha256)
    ):
        raise ValueError("Pose-release final rescue model14900 SHA-256 is malformed")
    if payload.get("iter") != MICROBAN_TELEOP_V12_FINAL_RESCUE_PARENT_ITERATION:
        raise ValueError("Pose-release final rescue parent must be model_14900.pt")
    infos = payload.get("infos")
    if not isinstance(infos, dict):
        raise TypeError("Pose-release final rescue parent infos are missing")
    if infos.get("microban_teleop_training_contract_version") != "12":
        raise ValueError("Pose-release final rescue parent is not contract-v12")
    env_state = infos.get("env_state")
    if (
        not isinstance(env_state, dict)
        or env_state.get("common_step_counter")
        != MICROBAN_TELEOP_V12_FINAL_RESCUE_PARENT_COMMON_STEP
    ):
        raise ValueError("Pose-release final rescue parent clock drifted")
    if infos.get("microban_teleop_recipe_revision") != (
        MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION
    ):
        raise ValueError(
            "Pose-release final rescue parent is not the pose-release recipe"
        )
    if infos.get(MICROBAN_TELEOP_V12_FINAL_RESCUE_INFO_KEY) is not None:
        raise ValueError("Pose-release final rescue parent already carries a final rescue")
    if infos.get("active_actor_columns_at_save") != list(
        MICROBAN_TELEOP_V12_FINAL_RESCUE_ACTIVE_COLUMNS
    ):
        raise ValueError("Pose-release final rescue parent active columns drifted")
    lineage = hand_pose_release_lineage(
        infos,
        iteration=MICROBAN_TELEOP_V12_FINAL_RESCUE_PARENT_ITERATION,
        verify_parent=False,
    )
    if lineage not in (
        HAND_POSE_RELEASE_LINEAGE_FRESH,
        HAND_POSE_RELEASE_LINEAGE_FRESH_CORNER_RESCUE,
    ):
        raise ValueError(
            "Pose-release final rescue parent must be a fresh pose-release chain "
            "(optionally through the model9900 corner rescue)"
        )
    assert_final_rescue_optimizer_step(
        payload,
        expected_step=MICROBAN_TELEOP_V12_FINAL_RESCUE_PARENT_OPTIMIZER_STEP,
    )
    corner = infos.get(MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY)
    return None if corner is None else validate_corner_rescue_lineage_marker(corner)


def validate_hand_pose_release_failed_final_gate_report(
    report_path: Path, *, parent_checkpoint: Path | None = None
) -> dict[str, Any]:
    """The failed 15000 gate tracking report that triggers the rescue.

    The report must be the unchanged final pose-release profile of a
    model_14999, structurally valid, failing a non-empty subset of the
    rescuable checks and nothing else.  With ``parent_checkpoint`` the report's
    checkpoint must be that run's model_14999 (same directory) and its bytes
    must still hash to the report's identity.

    Returns report SHA-256, checkpoint SHA-256/path, failed checks and the
    failing scenarios.
    """

    from mjlab_microban.scripts.teleop_v12_stage import (
        _load_json,
        _validate_tracking_report,
    )
    from mjlab_microban.tasks.microban_teleop_v12_bootstrap import sha256_file

    resolved = Path(report_path).expanduser().resolve(strict=True)
    if not resolved.is_file() or Path(report_path).is_symlink():
        raise ValueError("Failed final gate report must be a regular file")
    before = sha256_file(resolved)
    report = _load_json(resolved)
    identity = report.get("checkpoint")
    if not isinstance(identity, dict):
        raise ValueError("Failed final gate report has no checkpoint identity")
    checkpoint_sha256 = identity.get("sha256")
    checkpoint_path = identity.get("path")
    if not isinstance(checkpoint_sha256, str) or not isinstance(checkpoint_path, str):
        raise ValueError("Failed final gate report has no checkpoint identity")
    profile = final_gate_profile()
    _validate_tracking_report(
        report,
        {
            "sha256": checkpoint_sha256,
            "iteration": MICROBAN_TELEOP_V12_FINAL_RESCUE_TARGET_ITERATION,
            "completed_updates": (
                MICROBAN_TELEOP_V12_FINAL_RESCUE_TARGET_COMPLETED_UPDATES
            ),
        },
        profile_override=profile,
        allowed_failed_checks=(
            MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_RESCUABLE_CHECKS
        ),
    )
    failed = sorted(name for name, passed in report["checks"].items() if not passed)
    if report.get("status") != "fail" or not failed:
        raise ValueError("The failed final gate report must fail a rescuable check")
    scenarios = failed_final_gate_scenarios(report)
    if not scenarios:
        raise ValueError("The failed final gate report names no failing scenario")
    if Path(checkpoint_path).name != (
        f"model_{MICROBAN_TELEOP_V12_FINAL_RESCUE_TARGET_ITERATION}.pt"
    ):
        raise ValueError("The failed final gate report is not of a model_14999")
    if parent_checkpoint is not None:
        parent = Path(parent_checkpoint).expanduser().resolve(strict=True)
        failed_checkpoint = Path(checkpoint_path).expanduser().resolve(strict=True)
        if failed_checkpoint.parent != parent.parent:
            raise ValueError("The failed final gate report is not from the parent's run")
        if sha256_file(failed_checkpoint) != checkpoint_sha256:
            raise ValueError("The failed final gate checkpoint changed since its report")
    if sha256_file(resolved) != before:
        raise ValueError("Failed final gate report changed while validating")
    return {
        "report_sha256": before,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_path": checkpoint_path,
        "profile": profile,
        "failed_checks": failed,
        "failed_scenarios": scenarios,
    }


class MicrobanTeleopV12HandPoseReleaseFinalRescueOnPolicyRunner(
    MicrobanTeleopV12FinalRescueOnPolicyRunner,
    MicrobanTeleopV12HandPoseReleaseOnPolicyRunner,
):
    """Resume one exact pose-release model_14900 for 99 updates.

    Clock, optimizer, save and environment checks are the canonical final
    rescue's (inherited); loading, the parent contract and the saved infos are
    the pose-release variant's: saves keep the pose-release recipe revision,
    the inherited corner marker, and add the pose-release final marker.
    """

    def load(
        self,
        path: str | bytes,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        if isinstance(path, bytes):
            raise TypeError("Final rescue training requires a filesystem checkpoint")
        from mjlab_microban.tasks.microban_teleop_v12_bootstrap import sha256_file

        resolved = Path(path).expanduser().resolve(strict=True)
        before = sha256_file(resolved)
        payload = torch.load(resolved, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError("Final rescue parent payload is malformed")
        corner = validate_hand_pose_release_final_rescue_parent_payload(
            payload, checkpoint_sha256=before
        )
        failed_report = resolved.parent / (
            MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_FAILED_GATE_REPORT_FILENAME
        )
        failed = validate_hand_pose_release_failed_final_gate_report(failed_report)
        if failed["checkpoint_sha256"] == before:
            raise ValueError("The failed final gate cannot name the parent itself")
        # Skip the canonical final-rescue load; the pose-release load (and the
        # base load beneath it) validates and binds every other lineage.
        loaded = MicrobanTeleopV12HandPoseReleaseOnPolicyRunner.load(
            self, str(resolved), load_cfg=load_cfg, strict=strict, map_location=map_location
        )
        if (
            sha256_file(resolved) != before
            or sha256_file(failed_report) != failed["report_sha256"]
        ):
            raise ValueError("Final rescue parent or its report changed while loading")
        if self.teleop_v12_corner_rescue != corner:
            raise RuntimeError("Final rescue inherited corner lineage drifted on load")
        if self.teleop_v12_hand_pose_release_switch is not None:
            raise RuntimeError("Pose-release final rescue parent carried a recipe switch")
        self._final_rescue_marker = hand_pose_release_final_rescue_marker(
            parent_checkpoint_sha256=before,
            failed_gate_checkpoint_sha256=failed["checkpoint_sha256"],
            failed_gate_tracking_report_sha256=failed["report_sha256"],
            failed_gate_failed_checks=failed["failed_checks"],
            failed_gate_failed_scenarios=failed["failed_scenarios"],
            inherited_corner_rescue_marker_sha256=(
                None if corner is None else canonical_json_sha256(corner)
            ),
            sampler_mix=self._live_mix(),
        )
        self._assert_final_rescue_environment()
        self._assert_live_optimizer_step(
            MICROBAN_TELEOP_V12_FINAL_RESCUE_PARENT_OPTIMIZER_STEP
        )
        return loaded

    def _contract_infos(self, infos: dict | None = None) -> dict:
        self._assert_final_rescue_environment()
        result = MicrobanTeleopV12HandPoseReleaseOnPolicyRunner._contract_infos(
            self, infos
        )
        marker = self._final_rescue_marker
        if marker is None:
            raise RuntimeError("Final rescue lineage marker is not bound")
        marker = validate_hand_pose_release_final_rescue_marker(marker)
        corner = result.get(MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY)
        if marker["inherited_corner_rescue_marker_sha256"] != (
            None if corner is None else canonical_json_sha256(corner)
        ):
            raise RuntimeError("Final rescue inherited corner lineage drifted")
        if result.get("microban_teleop_recipe_revision") != (
            MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION
        ):
            raise RuntimeError("Pose-release final rescue saves must keep the recipe")
        result[MICROBAN_TELEOP_V12_FINAL_RESCUE_INFO_KEY] = marker
        return result

    def _assert_final_rescue_environment(self) -> None:
        MicrobanTeleopV12FinalRescueOnPolicyRunner._assert_final_rescue_environment(
            self
        )
        self._assert_hand_pose_release_environment()
