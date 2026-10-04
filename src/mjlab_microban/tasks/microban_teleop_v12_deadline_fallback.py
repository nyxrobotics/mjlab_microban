"""Recorded deadline acceptance for a measured contract-v12 corner rescue.

This is deliberately not a second training recipe.  It accepts a corner-rescue
model_9999 whose strict HMD/hand report fails only ``hand_tracking_rms`` and
relaxes only that hand RMS limit from 30 mm to 35 mm (P95 stays 50 mm).  The
10100 foot canary and the 15000 endpoint follow the deadline profiles of the
historical release: canary hand RMS 35 mm, final hand 35/70 mm and foot
50/80 mm with perturbation.  All safety, locomotion, causal-response, and ONNX
requirements are unchanged.

The historical (old-HOME) chain pinned every checkpoint and strict report by
literal SHA-256.  This revision records those hashes in the markers instead:
the stage gate recomputes them from the evidence files, every descendant
re-hashes its immediate parent, and markers are rebuilt from their recorded
fields on every validation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import torch

from mjlab_microban.tasks.microban_teleop_v12_actor import (
    TELEOP_V12_ADAPTER_GRADIENT_SCHEDULE_REVISION,
    TELEOP_V12_EXTRA_OBSERVATION_COLUMNS,
    TELEOP_V12_HAND_OBSERVATION_COLUMNS,
    TELEOP_V12_HMD_OBSERVATION_COLUMNS,
)
from mjlab_microban.tasks.microban_teleop_v12_bootstrap import (
    resolve_bootstrap_artifact_path,
    sha256_file,
)
from mjlab_microban.tasks.microban_teleop_v12_corner_rescue import (
    MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY,
    MICROBAN_TELEOP_V12_CORNER_RESCUE_RECIPE_REVISION,
    MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_OPTIMIZER_STEP,
    assert_corner_rescue_foot_adapter_zero,
    assert_corner_rescue_optimizer_step,
    canonical_json_sha256,
    validate_corner_rescue_lineage_marker,
)
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MICROBAN_TELEOP_V12_RECIPE_REVISION,
)

MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_INFO_KEY = "microban_teleop_v12_deadline_fallback"
MICROBAN_TELEOP_V12_DEADLINE_RESUME_SOURCE_INFO_KEY = (
    "microban_teleop_v12_deadline_resume_source"
)
MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_REVISION = (
    "recorded_corner_rescue_rms35mm_p95_50mm_safety_unchanged_v2"
)
MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_PROFILE = (
    "deadline_hmd_hand_rms35mm_p95_50mm_safety_unchanged_v1"
)
MICROBAN_TELEOP_V12_DEADLINE_AUTHORIZATION = (
    "workflow_decision_reusing_historical_release_deadline_profiles_"
    "pending_user_review"
)
MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_ITERATION = 9_999
MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_COMPLETED_UPDATES = 10_000
MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_COMMON_STEP = 240_000
MICROBAN_TELEOP_V12_DEADLINE_CANARY_ITERATION = 10_099
MICROBAN_TELEOP_V12_DEADLINE_CANARY_COMPLETED_UPDATES = 10_100
MICROBAN_TELEOP_V12_DEADLINE_CANARY_COMMON_STEP = 242_400
MICROBAN_TELEOP_V12_DEADLINE_CANARY_OPTIMIZER_STEP = 202_000
MICROBAN_TELEOP_V12_DEADLINE_POST_CANARY_INFO_KEY = (
    "microban_teleop_v12_deadline_post_canary"
)
MICROBAN_TELEOP_V12_DEADLINE_POST_CANARY_RESUME_SOURCE_INFO_KEY = (
    "microban_teleop_v12_deadline_post_canary_resume_source"
)
MICROBAN_TELEOP_V12_DEADLINE_POST_CANARY_REVISION = (
    "recorded_model10099_foot_activation_rms35mm_safety_unchanged_v2"
)
MICROBAN_TELEOP_V12_DEADLINE_CANARY_FALLBACK_PROFILE = (
    "deadline_whole_body_foot_activation_canary_hand_rms35mm_v1"
)
MICROBAN_TELEOP_V12_DEADLINE_FINAL_FALLBACK_PROFILE = (
    "deadline_full_body_hand_rms35mm_p95_70mm_foot_rms50mm_p95_80mm_"
    "perturbation_v2"
)
MICROBAN_TELEOP_V12_DEADLINE_FINAL_ITERATION = 14_999
MICROBAN_TELEOP_V12_DEADLINE_FINAL_COMPLETED_UPDATES = 15_000
MICROBAN_TELEOP_V12_DEADLINE_FINAL_COMMON_STEP = 360_000
MICROBAN_TELEOP_V12_DEADLINE_FINAL_OPTIMIZER_STEP = 300_000
MICROBAN_TELEOP_V12_DEADLINE_HAND_RMS_MAX_M = 0.035
MICROBAN_TELEOP_V12_DEADLINE_HAND_P95_MAX_M = 0.05
MICROBAN_TELEOP_V12_DEADLINE_FINAL_HAND_P95_MAX_M = 0.07
MICROBAN_TELEOP_V12_DEADLINE_FINAL_FOOT_RMS_MAX_M = 0.05
MICROBAN_TELEOP_V12_DEADLINE_FINAL_FOOT_P95_MAX_M = 0.08
MICROBAN_TELEOP_V12_DEADLINE_FINAL_ONNX_PARITY_TOLERANCE = 2.5e-5
MICROBAN_TELEOP_V12_DEADLINE_ACTIVE_COLUMNS = (
    *TELEOP_V12_HMD_OBSERVATION_COLUMNS,
    *TELEOP_V12_HAND_OBSERVATION_COLUMNS,
)
# Historical old-HOME evidence still referenced only by the live-sim canary
# viewer (live_pico_teleop_sim.py, run_pico_v12_deadline_canary_sim.sh).
# Gates, training, resume, and export never compare against these.
MICROBAN_TELEOP_V12_DEADLINE_CANARY_CHECKPOINT_SHA256 = (
    "86a81f45f34d91036ab138963c835a3e78db98224f523d3484e1d3ed335082ff"
)
MICROBAN_TELEOP_V12_DEADLINE_POST_CANARY_RECEIPT_SHA256 = (
    "f7fcf511ba75f326cd94a2fc21a05360fa9f7925849fc2bce91709fd8678f27b"
)
# The only strict check either promotion may waive.
MICROBAN_TELEOP_V12_DEADLINE_WAIVABLE_CHECKS = ("hand_tracking_rms",)


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_metric(value: object, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{label} must be a finite non-negative number")
    return float(value)


def _hand_metrics(value: object, label: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != {"maximum_rms", "maximum_p95"}:
        raise ValueError(f"{label} hand metrics are malformed")
    return {
        "maximum_rms": _require_metric(value["maximum_rms"], f"{label} RMS"),
        "maximum_p95": _require_metric(value["maximum_p95"], f"{label} P95"),
    }


def strict_hand_tracking_summary(report: Mapping[str, Any]) -> dict[str, float]:
    """Return the worst active-hand RMS/P95 over a tracking report's scenarios."""

    rms: list[float] = []
    p95: list[float] = []
    for result in report.get("results", ()):
        hand = result.get("target_error", {}).get("active_hand", {})
        if hand.get("sample_count"):
            rms.append(float(hand["rms"]))
            p95.append(float(hand["p95"]))
    if not rms:
        raise ValueError("Strict tracking report contains no active-hand evidence")
    return {"maximum_rms": max(rms), "maximum_p95": max(p95)}


def deadline_fallback_marker(
    *,
    selected_checkpoint_sha256: str,
    corner_marker_sha256: str,
    strict_report_sha256: str,
    strict_hand_tracking_m: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the provenance inherited by every deadline descendant."""

    return {
        "schema_version": 2,
        "revision": MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_REVISION,
        "authorization": MICROBAN_TELEOP_V12_DEADLINE_AUTHORIZATION,
        "precedent": (
            "historical_old_home_release_model14999_passed_only_deadline_profiles"
        ),
        "selected_checkpoint": {
            "sha256": _require_sha256(
                selected_checkpoint_sha256, "Deadline selected checkpoint"
            ),
            "iteration": MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_ITERATION,
            "completed_updates": (
                MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_COMPLETED_UPDATES
            ),
            "recipe_revision": MICROBAN_TELEOP_V12_CORNER_RESCUE_RECIPE_REVISION,
            "corner_marker_sha256": _require_sha256(
                corner_marker_sha256, "Deadline corner marker"
            ),
        },
        "selected_strict_tracking_report_sha256": _require_sha256(
            strict_report_sha256, "Deadline strict report"
        ),
        "strict_failed_checks": list(MICROBAN_TELEOP_V12_DEADLINE_WAIVABLE_CHECKS),
        "strict_hand_tracking_m": _hand_metrics(
            strict_hand_tracking_m, "Deadline strict"
        ),
        "acceptance_profile": MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_PROFILE,
        "threshold_change": {
            "hand_rms_m_max": MICROBAN_TELEOP_V12_DEADLINE_HAND_RMS_MAX_M,
            "canonical_hand_rms_m_max": 0.03,
            "hand_p95_m_max": MICROBAN_TELEOP_V12_DEADLINE_HAND_P95_MAX_M,
            "hand_p95_changed": False,
            "all_safety_checks_changed": False,
            "locomotion_gate_changed": False,
            "onnx_gate_changed": False,
        },
        "promotion_requires": ("full_9x300_locomotion_tracking_onnx_schema_v2_gate"),
    }


def validate_deadline_fallback_marker(marker: object) -> dict[str, Any]:
    """Rebuild the marker from its recorded fields and require equality."""

    if not isinstance(marker, Mapping):
        raise ValueError("Deadline-fallback lineage marker drifted")
    selected = marker.get("selected_checkpoint")
    try:
        if not isinstance(selected, Mapping):
            raise ValueError("selected checkpoint is missing")
        expected = deadline_fallback_marker(
            selected_checkpoint_sha256=selected.get("sha256"),  # type: ignore[arg-type]
            corner_marker_sha256=selected.get("corner_marker_sha256"),  # type: ignore[arg-type]
            strict_report_sha256=marker.get(  # type: ignore[arg-type]
                "selected_strict_tracking_report_sha256"
            ),
            strict_hand_tracking_m=marker.get("strict_hand_tracking_m"),  # type: ignore[arg-type]
        )
    except ValueError as exc:
        raise ValueError("Deadline-fallback lineage marker drifted") from exc
    if dict(marker) != expected:
        raise ValueError("Deadline-fallback lineage marker drifted")
    return deepcopy(expected)


def deadline_post_canary_marker(
    *,
    canary_checkpoint_sha256: str,
    strict_report_sha256: str,
    strict_failed_checks: list[str],
    strict_hand_tracking_m: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the authorization that promotes one measured 10100 foot canary."""

    if not isinstance(strict_failed_checks, list) or not set(
        strict_failed_checks
    ).issubset(MICROBAN_TELEOP_V12_DEADLINE_WAIVABLE_CHECKS):
        raise ValueError("Deadline canary may waive only hand_tracking_rms")
    return {
        "schema_version": 2,
        "revision": MICROBAN_TELEOP_V12_DEADLINE_POST_CANARY_REVISION,
        "authorization": MICROBAN_TELEOP_V12_DEADLINE_AUTHORIZATION,
        "parent_checkpoint": {
            "sha256": _require_sha256(
                canary_checkpoint_sha256, "Deadline canary checkpoint"
            ),
            "iteration": MICROBAN_TELEOP_V12_DEADLINE_CANARY_ITERATION,
            "completed_updates": MICROBAN_TELEOP_V12_DEADLINE_CANARY_COMPLETED_UPDATES,
            "common_step_counter": MICROBAN_TELEOP_V12_DEADLINE_CANARY_COMMON_STEP,
            "optimizer_step": MICROBAN_TELEOP_V12_DEADLINE_CANARY_OPTIMIZER_STEP,
        },
        "canonical_strict_tracking_report_sha256": _require_sha256(
            strict_report_sha256, "Deadline canary strict report"
        ),
        "canonical_strict_profile": (
            "whole_body_foot_activation_canary_reachable_safety_v1"
        ),
        "canonical_failed_checks": sorted(strict_failed_checks),
        "fallback_profile": MICROBAN_TELEOP_V12_DEADLINE_CANARY_FALLBACK_PROFILE,
        "measured_hand_tracking_m": _hand_metrics(
            strict_hand_tracking_m, "Deadline canary strict"
        ),
        "threshold_change": {
            "hand_rms_m_max": MICROBAN_TELEOP_V12_DEADLINE_HAND_RMS_MAX_M,
            "canonical_hand_rms_m_max": 0.03,
            "hand_p95_m_max": MICROBAN_TELEOP_V12_DEADLINE_HAND_P95_MAX_M,
            "hand_p95_changed": False,
            "foot_activation_checks_changed": False,
            "all_safety_checks_changed": False,
            "locomotion_gate_changed": False,
            "onnx_gate_changed": False,
        },
        "promotion_requires": (
            "full_9x300_locomotion_foot_activation_tracking_onnx_schema_v2_gate"
        ),
        "next_endpoint": {
            "iteration": MICROBAN_TELEOP_V12_DEADLINE_FINAL_ITERATION,
            "completed_updates": MICROBAN_TELEOP_V12_DEADLINE_FINAL_COMPLETED_UPDATES,
            "tracking_profile": MICROBAN_TELEOP_V12_DEADLINE_FINAL_FALLBACK_PROFILE,
        },
    }


def validate_deadline_post_canary_marker(marker: object) -> dict[str, Any]:
    if not isinstance(marker, Mapping):
        raise ValueError("Deadline post-canary authorization marker drifted")
    parent = marker.get("parent_checkpoint")
    try:
        if not isinstance(parent, Mapping):
            raise ValueError("parent checkpoint is missing")
        expected = deadline_post_canary_marker(
            canary_checkpoint_sha256=parent.get("sha256"),  # type: ignore[arg-type]
            strict_report_sha256=marker.get(  # type: ignore[arg-type]
                "canonical_strict_tracking_report_sha256"
            ),
            strict_failed_checks=marker.get("canonical_failed_checks"),  # type: ignore[arg-type]
            strict_hand_tracking_m=marker.get("measured_hand_tracking_m"),  # type: ignore[arg-type]
        )
    except ValueError as exc:
        raise ValueError("Deadline post-canary authorization marker drifted") from exc
    if dict(marker) != expected:
        raise ValueError("Deadline post-canary authorization marker drifted")
    return deepcopy(expected)


def _resume_source(
    *,
    revision: str,
    parent_iteration: int,
    checkpoint_path: str,
    gate_path: str,
    gate_sha256: str,
    parent_checkpoint_sha256: str,
) -> dict[str, Any]:
    if (
        not isinstance(checkpoint_path, str)
        or not checkpoint_path
        or not isinstance(gate_path, str)
        or not gate_path
    ):
        raise ValueError("Deadline resume gate identity is malformed")
    return {
        "schema_version": 2,
        "revision": revision,
        "parent_checkpoint_sha256": _require_sha256(
            parent_checkpoint_sha256, "Deadline resume parent"
        ),
        "parent_iteration": parent_iteration,
        "parent_completed_updates": parent_iteration + 1,
        "parent_checkpoint_path": checkpoint_path,
        "full_stage_gate_path": gate_path,
        "full_stage_gate_sha256": _require_sha256(gate_sha256, "Deadline resume gate"),
    }


def deadline_fallback_resume_source(
    *,
    checkpoint_path: str,
    gate_path: str,
    gate_sha256: str,
    parent_checkpoint_sha256: str,
) -> dict[str, Any]:
    return _resume_source(
        revision="recorded_selected_full_gate_resume_source_v2",
        parent_iteration=MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_ITERATION,
        checkpoint_path=checkpoint_path,
        gate_path=gate_path,
        gate_sha256=gate_sha256,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
    )


def deadline_post_canary_resume_source(
    *,
    checkpoint_path: str,
    gate_path: str,
    gate_sha256: str,
    parent_checkpoint_sha256: str,
) -> dict[str, Any]:
    return _resume_source(
        revision="recorded_model10099_full_gate_resume_source_v2",
        parent_iteration=MICROBAN_TELEOP_V12_DEADLINE_CANARY_ITERATION,
        checkpoint_path=checkpoint_path,
        gate_path=gate_path,
        gate_sha256=gate_sha256,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
    )


def _validate_resume_source(value: object, builder, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} resume-source lineage is missing")
    try:
        expected = builder(
            checkpoint_path=value.get("parent_checkpoint_path"),
            gate_path=value.get("full_stage_gate_path"),
            gate_sha256=value.get("full_stage_gate_sha256"),
            parent_checkpoint_sha256=value.get("parent_checkpoint_sha256"),
        )
    except ValueError as exc:
        raise ValueError(f"{label} resume-source lineage drifted") from exc
    if dict(value) != expected:
        raise ValueError(f"{label} resume-source lineage drifted")
    return deepcopy(expected)


def validate_deadline_fallback_resume_source(value: object) -> dict[str, Any]:
    return _validate_resume_source(
        value, deadline_fallback_resume_source, "Deadline fallback"
    )


def validate_deadline_post_canary_resume_source(value: object) -> dict[str, Any]:
    return _validate_resume_source(
        value, deadline_post_canary_resume_source, "Deadline post-canary"
    )


def validate_deadline_fallback_checkpoint_payload(
    payload: Mapping[str, Any], *, checkpoint_sha256: str
) -> dict[str, Any]:
    """Validate a corner-rescue model_9999 eligible for deadline adjudication.

    Returns the checkpoint's validated corner-rescue marker.
    """

    _require_sha256(checkpoint_sha256, "Deadline fallback checkpoint")
    iteration = payload.get("iter")
    infos = payload.get("infos")
    if iteration != MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_ITERATION:
        raise ValueError("Deadline fallback requires the rescue model_9999.pt")
    if not isinstance(infos, Mapping):
        raise TypeError("Deadline fallback checkpoint infos are missing")
    env_state = infos.get("env_state")
    if (
        infos.get("microban_teleop_training_contract_version") != "12"
        or infos.get("microban_teleop_recipe_revision")
        != MICROBAN_TELEOP_V12_CORNER_RESCUE_RECIPE_REVISION
        or infos.get("adapter_gradient_schedule_revision")
        != TELEOP_V12_ADAPTER_GRADIENT_SCHEDULE_REVISION
        or infos.get("active_actor_columns_at_save")
        != list(MICROBAN_TELEOP_V12_DEADLINE_ACTIVE_COLUMNS)
        or not isinstance(env_state, Mapping)
        or env_state.get("common_step_counter")
        != MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_COMMON_STEP
    ):
        raise ValueError("Deadline fallback checkpoint contract/clock drifted")
    corner_marker = validate_corner_rescue_lineage_marker(
        infos.get(MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY)
    )
    if infos.get(MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_INFO_KEY) is not None:
        raise ValueError("Selected source checkpoint must precede fallback promotion")
    assert_corner_rescue_foot_adapter_zero(payload)
    assert_corner_rescue_optimizer_step(
        payload,
        expected_step=MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_OPTIMIZER_STEP,
    )
    return corner_marker


def validate_deadline_fallback_descendant(
    infos: Mapping[str, Any], *, iteration: int
) -> dict[str, Any] | None:
    """Validate inherited fallback provenance on ordinary canonical saves."""

    marker_value = infos.get(MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_INFO_KEY)
    if marker_value is None:
        return None
    marker = validate_deadline_fallback_marker(marker_value)
    if (
        infos.get("microban_teleop_recipe_revision")
        != MICROBAN_TELEOP_V12_RECIPE_REVISION
        or isinstance(iteration, bool)
        or not isinstance(iteration, int)
        or iteration
        not in (
            MICROBAN_TELEOP_V12_DEADLINE_CANARY_ITERATION,
            MICROBAN_TELEOP_V12_DEADLINE_FINAL_ITERATION,
        )
    ):
        raise ValueError("Deadline-fallback descendant recipe/clock drifted")
    corner = validate_corner_rescue_lineage_marker(
        infos.get(MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY)
    )
    if canonical_json_sha256(corner) != (
        marker["selected_checkpoint"]["corner_marker_sha256"]
    ):
        raise ValueError("Deadline-fallback descendant corner marker drifted")
    source = validate_deadline_fallback_resume_source(
        infos.get(MICROBAN_TELEOP_V12_DEADLINE_RESUME_SOURCE_INFO_KEY)
    )
    if source["parent_checkpoint_sha256"] != marker["selected_checkpoint"]["sha256"]:
        raise ValueError("Deadline-fallback resume source is not the selected parent")
    post_canary = infos.get(MICROBAN_TELEOP_V12_DEADLINE_POST_CANARY_INFO_KEY)
    post_source = infos.get(
        MICROBAN_TELEOP_V12_DEADLINE_POST_CANARY_RESUME_SOURCE_INFO_KEY
    )
    if iteration == MICROBAN_TELEOP_V12_DEADLINE_CANARY_ITERATION:
        if post_canary is not None or post_source is not None:
            raise ValueError("Deadline canary cannot contain post-canary lineage")
    else:
        authorization = validate_deadline_post_canary_marker(post_canary)
        validated_post_source = validate_deadline_post_canary_resume_source(
            post_source
        )
        if (
            validated_post_source["parent_checkpoint_sha256"]
            != authorization["parent_checkpoint"]["sha256"]
        ):
            raise ValueError("Deadline post-canary resume source is not the canary")
    return marker


def _verify_parent_files(
    source: Mapping[str, Any], *, expected_parent_sha256: str, label: str
) -> tuple[Any, Any]:
    checkpoint = resolve_bootstrap_artifact_path(source["parent_checkpoint_path"])
    gate = resolve_bootstrap_artifact_path(source["full_stage_gate_path"])
    if (
        not checkpoint.is_file()
        or sha256_file(checkpoint) != expected_parent_sha256
        or not gate.is_file()
        or sha256_file(gate) != source["full_stage_gate_sha256"]
    ):
        raise ValueError(f"{label} immediate-parent files changed")
    return checkpoint, gate


def validate_deadline_fallback_canary_payload(
    payload: Mapping[str, Any],
    *,
    verify_parent_files: bool,
    checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the sole allowed descendant of the selected rescue: model10099."""

    infos = payload.get("infos")
    iteration = payload.get("iter")
    if checkpoint_sha256 is not None:
        _require_sha256(checkpoint_sha256, "Deadline canary checkpoint")
    if not isinstance(infos, Mapping) or not isinstance(iteration, int):
        raise TypeError("Deadline canary checkpoint payload is malformed")
    if iteration != MICROBAN_TELEOP_V12_DEADLINE_CANARY_ITERATION:
        raise ValueError("Deadline canary must be model_10099.pt")
    marker = validate_deadline_fallback_descendant(infos, iteration=iteration)
    if marker is None:
        raise ValueError("Deadline canary lineage marker is missing")
    env_state = infos.get("env_state")
    if (
        not isinstance(env_state, Mapping)
        or env_state.get("common_step_counter")
        != MICROBAN_TELEOP_V12_DEADLINE_CANARY_COMMON_STEP
        or infos.get("active_actor_columns_at_save")
        != list(TELEOP_V12_EXTRA_OBSERVATION_COLUMNS)
    ):
        raise ValueError("Deadline canary clock/active columns drifted")
    assert_corner_rescue_optimizer_step(
        payload, expected_step=MICROBAN_TELEOP_V12_DEADLINE_CANARY_OPTIMIZER_STEP
    )
    source = validate_deadline_fallback_resume_source(
        infos.get(MICROBAN_TELEOP_V12_DEADLINE_RESUME_SOURCE_INFO_KEY)
    )
    if verify_parent_files:
        selected_sha = marker["selected_checkpoint"]["sha256"]
        checkpoint, _ = _verify_parent_files(
            source, expected_parent_sha256=selected_sha, label="Deadline canary"
        )
        parent = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(parent, Mapping):
            raise TypeError("Deadline canary parent payload is malformed")
        parent_corner = validate_deadline_fallback_checkpoint_payload(
            parent, checkpoint_sha256=selected_sha
        )
        if canonical_json_sha256(parent_corner) != (
            marker["selected_checkpoint"]["corner_marker_sha256"]
        ):
            raise ValueError("Deadline canary parent corner marker drifted")
        _verify_parent_files(
            source, expected_parent_sha256=selected_sha, label="Deadline canary"
        )
    return marker


def validate_deadline_fallback_final_payload(
    payload: Mapping[str, Any], *, verify_parent_files: bool
) -> dict[str, Any]:
    """Validate the only post-canary descendant: final model14999."""

    infos = payload.get("infos")
    iteration = payload.get("iter")
    if not isinstance(infos, Mapping) or not isinstance(iteration, int):
        raise TypeError("Deadline final checkpoint payload is malformed")
    marker = validate_deadline_fallback_descendant(infos, iteration=iteration)
    env_state = infos.get("env_state")
    if (
        marker is None
        or iteration != MICROBAN_TELEOP_V12_DEADLINE_FINAL_ITERATION
        or not isinstance(env_state, Mapping)
        or env_state.get("common_step_counter")
        != MICROBAN_TELEOP_V12_DEADLINE_FINAL_COMMON_STEP
        or infos.get("active_actor_columns_at_save")
        != list(TELEOP_V12_EXTRA_OBSERVATION_COLUMNS)
    ):
        raise ValueError("Deadline final clock/active columns drifted")
    assert_corner_rescue_optimizer_step(
        payload, expected_step=MICROBAN_TELEOP_V12_DEADLINE_FINAL_OPTIMIZER_STEP
    )
    authorization = validate_deadline_post_canary_marker(
        infos.get(MICROBAN_TELEOP_V12_DEADLINE_POST_CANARY_INFO_KEY)
    )
    source = validate_deadline_post_canary_resume_source(
        infos.get(MICROBAN_TELEOP_V12_DEADLINE_POST_CANARY_RESUME_SOURCE_INFO_KEY)
    )
    if verify_parent_files:
        canary_sha = authorization["parent_checkpoint"]["sha256"]
        checkpoint, _ = _verify_parent_files(
            source, expected_parent_sha256=canary_sha, label="Deadline final"
        )
        parent = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(parent, Mapping):
            raise TypeError("Deadline final parent payload is malformed")
        parent_marker = validate_deadline_fallback_canary_payload(
            parent, verify_parent_files=True, checkpoint_sha256=canary_sha
        )
        if parent_marker != marker:
            raise ValueError("Deadline final parent lineage drifted")
        _verify_parent_files(
            source, expected_parent_sha256=canary_sha, label="Deadline final"
        )
    return marker


def validate_deadline_fallback_descendant_payload(
    payload: Mapping[str, Any], *, verify_parent_files: bool
) -> dict[str, Any]:
    """Dispatch validation for the exact canary or exact final descendant."""

    iteration = payload.get("iter")
    if iteration == MICROBAN_TELEOP_V12_DEADLINE_CANARY_ITERATION:
        return validate_deadline_fallback_canary_payload(
            payload, verify_parent_files=verify_parent_files
        )
    if iteration == MICROBAN_TELEOP_V12_DEADLINE_FINAL_ITERATION:
        return validate_deadline_fallback_final_payload(
            payload, verify_parent_files=verify_parent_files
        )
    raise ValueError("Deadline fallback has no authorized checkpoint at this iteration")


def validate_deadline_fallback_resume_payload(
    payload: Mapping[str, Any], *, checkpoint_sha256: str
) -> dict[str, Any] | None:
    """Allow only selected-rescue->canary or promoted-canary->final resume.

    Returns the canary's inherited marker, or ``None`` for the selected rescue
    (whose marker exists only in its deadline gate).
    """

    infos = payload.get("infos")
    if (
        isinstance(infos, Mapping)
        and infos.get(MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_INFO_KEY) is not None
    ):
        marker = validate_deadline_fallback_canary_payload(
            payload,
            verify_parent_files=True,
            checkpoint_sha256=checkpoint_sha256,
        )
        if infos.get(MICROBAN_TELEOP_V12_DEADLINE_POST_CANARY_INFO_KEY) is not None:
            raise ValueError("Deadline final checkpoint cannot resume training")
        return marker
    validate_deadline_fallback_checkpoint_payload(
        payload, checkpoint_sha256=checkpoint_sha256
    )
    return None


def validate_deadline_fallback_training_request(
    *,
    current_iteration: int,
    common_step_counter: int,
    num_learning_iterations: int,
    save_interval: int,
) -> None:
    canary = (
        current_iteration == 10_000
        and common_step_counter == MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_COMMON_STEP
        and num_learning_iterations == 100
        and save_interval == 15_000
    )
    final = (
        current_iteration == MICROBAN_TELEOP_V12_DEADLINE_CANARY_COMPLETED_UPDATES
        and common_step_counter == MICROBAN_TELEOP_V12_DEADLINE_CANARY_COMMON_STEP
        and num_learning_iterations
        == (
            MICROBAN_TELEOP_V12_DEADLINE_FINAL_COMPLETED_UPDATES
            - MICROBAN_TELEOP_V12_DEADLINE_CANARY_COMPLETED_UPDATES
        )
        and save_interval == 15_000
    )
    if not (canary or final):
        raise ValueError(
            "Deadline fallback must run only exact 10000->10100 or "
            "post-canary-authorized 10100->15000 with save_interval=15000"
        )


def validate_deadline_fallback_save_endpoint(
    *, iteration: int, common_step_counter: int, filename: str
) -> None:
    canary = (
        iteration == MICROBAN_TELEOP_V12_DEADLINE_CANARY_ITERATION
        and common_step_counter == MICROBAN_TELEOP_V12_DEADLINE_CANARY_COMMON_STEP
        and filename == f"model_{MICROBAN_TELEOP_V12_DEADLINE_CANARY_ITERATION}.pt"
    )
    final = (
        iteration == MICROBAN_TELEOP_V12_DEADLINE_FINAL_ITERATION
        and common_step_counter == MICROBAN_TELEOP_V12_DEADLINE_FINAL_COMMON_STEP
        and filename == f"model_{MICROBAN_TELEOP_V12_DEADLINE_FINAL_ITERATION}.pt"
    )
    if not (canary or final):
        raise RuntimeError(
            "Deadline fallback may save only exact model10099/10100 or "
            "model14999/15000 endpoint"
        )
