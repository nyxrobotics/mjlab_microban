"""Hash-pinned deadline acceptance for one measured contract-v12 checkpoint.

This is deliberately not a second training recipe.  It authenticates the
selected v1 corner-rescue bytes and records the user's explicit deadline
decision to relax only hand RMS from 30 mm to 35 mm.  All safety, P95,
locomotion, causal-response, and ONNX requirements remain unchanged.
"""

from __future__ import annotations

import hashlib
import json
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
    MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_OPTIMIZER_STEP,
    assert_corner_rescue_foot_adapter_zero,
    assert_corner_rescue_optimizer_step,
)
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MICROBAN_TELEOP_V12_RECIPE_REVISION,
)

MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_INFO_KEY = "microban_teleop_v12_deadline_fallback"
MICROBAN_TELEOP_V12_DEADLINE_RESUME_SOURCE_INFO_KEY = (
    "microban_teleop_v12_deadline_resume_source"
)
MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_REVISION = (
    "selected_v1_rms35mm_p95_50mm_safety_unchanged_v1"
)
MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_PROFILE = (
    "deadline_hmd_hand_rms35mm_p95_50mm_safety_unchanged_v1"
)
MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_CHECKPOINT_SHA256 = (
    "393d35b4e7cc0453f5143c7f2be4d4a4658567ab6132dfb54d32e67eb26b62b7"
)
MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_STRICT_REPORT_SHA256 = (
    "c466d66cf5b5ac8603f558e0ae8612450ac74bbad78fd88719a2cdaa9673e4b7"
)
MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_REJECTED_V2_SHA256 = (
    "c6171ded2cffda43d04f0a1e85332975d459d9e02fda38546287332ca230b3a1"
)
MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_REJECTED_V2_REPORT_SHA256 = (
    "3c11b34e788e52b08cbfc35797a4128022224f459215117b7a8816b0c1c4f3eb"
)
MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_V1_RECIPE_REVISION = (
    "model9900_targeted_bilateral_corner_pair_replay_to10000_v1"
)
MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_V1_MARKER_SHA256 = (
    "17d36dc2afc6063087b3a8adda64989304aa9d5a83ad0faf31f17b44ba7d166c"
)
MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_ITERATION = 9_999
MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_COMPLETED_UPDATES = 10_000
MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_COMMON_STEP = 240_000
MICROBAN_TELEOP_V12_DEADLINE_CANARY_ITERATION = 10_099
MICROBAN_TELEOP_V12_DEADLINE_CANARY_COMPLETED_UPDATES = 10_100
MICROBAN_TELEOP_V12_DEADLINE_CANARY_COMMON_STEP = 242_400
MICROBAN_TELEOP_V12_DEADLINE_CANARY_OPTIMIZER_STEP = 202_000
MICROBAN_TELEOP_V12_DEADLINE_CANARY_CHECKPOINT_SHA256 = (
    "86a81f45f34d91036ab138963c835a3e78db98224f523d3484e1d3ed335082ff"
)
MICROBAN_TELEOP_V12_DEADLINE_CANARY_STRICT_REPORT_SHA256 = (
    "e8230cff4cd25e8af1d9931d1fcd459db112e5a34c88a20470e22c2019ec5161"
)
MICROBAN_TELEOP_V12_DEADLINE_POST_CANARY_RECEIPT_SHA256 = (
    "f7fcf511ba75f326cd94a2fc21a05360fa9f7925849fc2bce91709fd8678f27b"
)
MICROBAN_TELEOP_V12_DEADLINE_POST_CANARY_INFO_KEY = (
    "microban_teleop_v12_deadline_post_canary"
)
MICROBAN_TELEOP_V12_DEADLINE_POST_CANARY_RESUME_SOURCE_INFO_KEY = (
    "microban_teleop_v12_deadline_post_canary_resume_source"
)
MICROBAN_TELEOP_V12_DEADLINE_POST_CANARY_REVISION = (
    "pinned_model10099_foot_activation_rms35mm_safety_unchanged_v1"
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


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def deadline_fallback_v1_corner_marker() -> dict[str, Any]:
    """Return the exact historical 40/40/20 marker embedded in selected v1."""

    marker = {
        "schema_version": 1,
        "revision": "pinned_model9900_uniform40_lf_rb40_lb_rf20_99_updates_v1",
        "parent_checkpoint_sha256": (
            "063a8f65ebf9007d63395e9a5b98420eb025bd39416dab5727f9f4c06fc6e877"
        ),
        "parent_strict_tracking_report_sha256": (
            "399db0cee55c137d3d0226ebcb54b0fa3bb84c95496c5c206af55f2fb25837f4"
        ),
        "parent_iteration": 9_900,
        "parent_completed_updates": 9_901,
        "parent_common_step_counter": 237_624,
        "target_iteration": 9_999,
        "target_completed_updates": 10_000,
        "target_common_step_counter": 240_000,
        "process_updates": 99,
        "optimizer_step": {"parent": 198_020, "target": 200_000},
        "training": {
            "environment_seed": 42,
            "agent_seed": 42,
            "num_envs": 2_048,
            "num_steps_per_env": 24,
        },
        "source_recipe_revision": MICROBAN_TELEOP_V12_RECIPE_REVISION,
        "rescue_recipe_revision": (
            MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_V1_RECIPE_REVISION
        ),
        "sampler_revision": "uniform_joint_box40pct_lf_rb40pct_lb_rf20pct_v1",
        "sampler_probabilities": {
            "ordinary_uniform_independent": 0.4,
            "left_forward_right_backward": 0.4,
            "left_backward_right_forward": 0.2,
        },
        "corner_joint_tuples_deg": {
            "left_forward_right_backward": [
                [-25.0, 25.0, -50.0],
                [25.0, -20.0, -10.0],
            ],
            "left_backward_right_forward": [
                [25.0, 20.0, -10.0],
                [-25.0, -25.0, -50.0],
            ],
        },
        "adapter_gradient_schedule_revision": (
            TELEOP_V12_ADAPTER_GRADIENT_SCHEDULE_REVISION
        ),
        "active_actor_columns": list(MICROBAN_TELEOP_V12_DEADLINE_ACTIVE_COLUMNS),
        "foot_activation": "held_inactive_through_completed_update_10000",
        "foot_observation_columns": [69, 70, 71, 72, 73, 74],
        "foot_command_state_at_save": {
            "shape": [2_048, 2, 3],
            "target_offset_exact_zero": True,
            "single_support_active_count": 0,
            "both_feet_active_count": 0,
        },
        "unchanged_contract": {
            "learning_rate": 1.0e-4,
            "action_semantics": "raw_actor_output",
            "action_clip": None,
            "hand_reward_weight": 2.0,
            "hand_reward_std_m": 0.05,
            "normalizer": "unchanged_from_parent",
            "legacy_actor_tensors": "frozen",
        },
    }
    if _canonical_json_sha256(marker) != (
        MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_V1_MARKER_SHA256
    ):
        raise RuntimeError("Historical v1 corner marker literal drifted")
    return marker


def deadline_fallback_marker() -> dict[str, Any]:
    """Return immutable provenance inherited by every canonical descendant."""

    return {
        "schema_version": 1,
        "revision": MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_REVISION,
        "operator_authorization": (
            "user_explicitly_requested_relaxed_acceptance_to_finish_deadline"
        ),
        "selected_checkpoint": {
            "sha256": MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_CHECKPOINT_SHA256,
            "iteration": MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_ITERATION,
            "completed_updates": (
                MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_COMPLETED_UPDATES
            ),
            "recipe_revision": (
                MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_V1_RECIPE_REVISION
            ),
            "corner_marker_sha256": (
                MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_V1_MARKER_SHA256
            ),
        },
        "selected_strict_tracking_report_sha256": (
            MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_STRICT_REPORT_SHA256
        ),
        "explicitly_rejected_v2": {
            "checkpoint_sha256": (
                MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_REJECTED_V2_SHA256
            ),
            "strict_tracking_report_sha256": (
                MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_REJECTED_V2_REPORT_SHA256
            ),
            "reason": (
                "0.000163m_worst_rms_gain_did_not_justify_0.00152m_opposite_pair_"
                "rms_and_0.00192m_p95_regressions"
            ),
        },
        "selection_metrics_m": {
            "selected_v1": {
                "left_forward_right_backward_rms": 0.03348252665362493,
                "left_backward_right_forward_rms": 0.0235841066736868,
                "maximum_p95": 0.03766489420086145,
            },
            "rejected_v2": {
                "left_forward_right_backward_rms": 0.03332012060274224,
                "left_backward_right_forward_rms": 0.02510417169703905,
                "maximum_p95": 0.039579783566296094,
            },
        },
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
    expected = deadline_fallback_marker()
    if marker != expected:
        raise ValueError("Deadline-fallback lineage marker drifted")
    return deepcopy(expected)


def deadline_post_canary_marker() -> dict[str, Any]:
    """Return the one authorization that promotes the measured foot canary."""

    return {
        "schema_version": 1,
        "revision": MICROBAN_TELEOP_V12_DEADLINE_POST_CANARY_REVISION,
        "parent_checkpoint": {
            "sha256": MICROBAN_TELEOP_V12_DEADLINE_CANARY_CHECKPOINT_SHA256,
            "iteration": MICROBAN_TELEOP_V12_DEADLINE_CANARY_ITERATION,
            "completed_updates": MICROBAN_TELEOP_V12_DEADLINE_CANARY_COMPLETED_UPDATES,
            "common_step_counter": MICROBAN_TELEOP_V12_DEADLINE_CANARY_COMMON_STEP,
            "optimizer_step": MICROBAN_TELEOP_V12_DEADLINE_CANARY_OPTIMIZER_STEP,
        },
        "canonical_strict_tracking_report_sha256": (
            MICROBAN_TELEOP_V12_DEADLINE_CANARY_STRICT_REPORT_SHA256
        ),
        "canonical_strict_profile": (
            "whole_body_foot_activation_canary_reachable_safety_v1"
        ),
        "canonical_failed_checks": ["hand_tracking_rms"],
        "fallback_profile": MICROBAN_TELEOP_V12_DEADLINE_CANARY_FALLBACK_PROFILE,
        "measured_hand_tracking_m": {
            "maximum_rms": 0.03237772842406263,
            "maximum_p95": 0.03992060273885726,
        },
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
        },
    }


def validate_deadline_post_canary_marker(marker: object) -> dict[str, Any]:
    expected = deadline_post_canary_marker()
    if marker != expected:
        raise ValueError("Deadline post-canary authorization marker drifted")
    return deepcopy(expected)


def deadline_fallback_resume_source(
    *, checkpoint_path: str, gate_path: str, gate_sha256: str
) -> dict[str, Any]:
    if not checkpoint_path or not gate_path or len(gate_sha256) != 64:
        raise ValueError("Deadline fallback resume gate identity is malformed")
    return {
        "schema_version": 1,
        "revision": "selected_v1_full_gate_resume_source_v1",
        "parent_checkpoint_sha256": (
            MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_CHECKPOINT_SHA256
        ),
        "parent_iteration": MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_ITERATION,
        "parent_completed_updates": (
            MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_COMPLETED_UPDATES
        ),
        "parent_checkpoint_path": checkpoint_path,
        "full_stage_gate_path": gate_path,
        "full_stage_gate_sha256": gate_sha256,
    }


def validate_deadline_fallback_resume_source(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("Deadline fallback resume-source lineage is missing")
    gate_path = value.get("full_stage_gate_path")
    gate_sha = value.get("full_stage_gate_sha256")
    checkpoint_path = value.get("parent_checkpoint_path")
    if (
        not isinstance(checkpoint_path, str)
        or not isinstance(gate_path, str)
        or not isinstance(gate_sha, str)
    ):
        raise TypeError("Deadline fallback resume gate identity is malformed")
    expected = deadline_fallback_resume_source(
        checkpoint_path=checkpoint_path,
        gate_path=gate_path,
        gate_sha256=gate_sha,
    )
    if dict(value) != expected:
        raise ValueError("Deadline fallback resume-source lineage drifted")
    return deepcopy(expected)


def deadline_post_canary_resume_source(
    *, checkpoint_path: str, gate_path: str, gate_sha256: str
) -> dict[str, Any]:
    if not checkpoint_path or not gate_path or len(gate_sha256) != 64:
        raise ValueError("Deadline post-canary resume gate identity is malformed")
    return {
        "schema_version": 1,
        "revision": "pinned_model10099_full_gate_resume_source_v1",
        "parent_checkpoint_sha256": (
            MICROBAN_TELEOP_V12_DEADLINE_CANARY_CHECKPOINT_SHA256
        ),
        "parent_iteration": MICROBAN_TELEOP_V12_DEADLINE_CANARY_ITERATION,
        "parent_completed_updates": (
            MICROBAN_TELEOP_V12_DEADLINE_CANARY_COMPLETED_UPDATES
        ),
        "parent_checkpoint_path": checkpoint_path,
        "full_stage_gate_path": gate_path,
        "full_stage_gate_sha256": gate_sha256,
    }


def validate_deadline_post_canary_resume_source(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("Deadline post-canary resume-source lineage is missing")
    checkpoint_path = value.get("parent_checkpoint_path")
    gate_path = value.get("full_stage_gate_path")
    gate_sha = value.get("full_stage_gate_sha256")
    if (
        not isinstance(checkpoint_path, str)
        or not isinstance(gate_path, str)
        or not isinstance(gate_sha, str)
    ):
        raise TypeError("Deadline post-canary resume gate identity is malformed")
    expected = deadline_post_canary_resume_source(
        checkpoint_path=checkpoint_path,
        gate_path=gate_path,
        gate_sha256=gate_sha,
    )
    if dict(value) != expected:
        raise ValueError("Deadline post-canary resume-source lineage drifted")
    return deepcopy(expected)


def validate_deadline_fallback_checkpoint_payload(
    payload: Mapping[str, Any], *, checkpoint_sha256: str
) -> dict[str, Any]:
    """Authenticate the one v1 checkpoint eligible for deadline adjudication."""

    if checkpoint_sha256 == MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_REJECTED_V2_SHA256:
        raise ValueError("The v2 corner checkpoint is explicitly rejected")
    if checkpoint_sha256 != MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_CHECKPOINT_SHA256:
        raise ValueError("Deadline fallback checkpoint SHA-256 mismatch")
    iteration = payload.get("iter")
    infos = payload.get("infos")
    if iteration != MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_ITERATION:
        raise ValueError("Deadline fallback requires selected model_9999.pt")
    if not isinstance(infos, Mapping):
        raise TypeError("Deadline fallback checkpoint infos are missing")
    env_state = infos.get("env_state")
    if (
        infos.get("microban_teleop_training_contract_version") != "12"
        or infos.get("microban_teleop_recipe_revision")
        != MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_V1_RECIPE_REVISION
        or infos.get("adapter_gradient_schedule_revision")
        != TELEOP_V12_ADAPTER_GRADIENT_SCHEDULE_REVISION
        or infos.get("active_actor_columns_at_save")
        != list(MICROBAN_TELEOP_V12_DEADLINE_ACTIVE_COLUMNS)
        or not isinstance(env_state, Mapping)
        or env_state.get("common_step_counter")
        != MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_COMMON_STEP
    ):
        raise ValueError("Deadline fallback checkpoint contract/clock drifted")
    corner_marker = infos.get(MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY)
    if corner_marker != deadline_fallback_v1_corner_marker():
        raise ValueError("Selected v1 corner marker changed")
    if infos.get(MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_INFO_KEY) is not None:
        raise ValueError("Selected source checkpoint must precede fallback promotion")
    assert_corner_rescue_foot_adapter_zero(payload)
    assert_corner_rescue_optimizer_step(
        payload,
        expected_step=MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_OPTIMIZER_STEP,
    )
    return deadline_fallback_marker()


def validate_deadline_fallback_descendant(
    infos: Mapping[str, Any], *, iteration: int
) -> dict[str, Any] | None:
    """Validate inherited fallback provenance on ordinary canonical saves."""

    marker = infos.get(MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_INFO_KEY)
    if marker is None:
        return None
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
    if infos.get(MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY) != (
        deadline_fallback_v1_corner_marker()
    ):
        raise ValueError("Deadline-fallback descendant v1 corner marker drifted")
    validate_deadline_fallback_resume_source(
        infos.get(MICROBAN_TELEOP_V12_DEADLINE_RESUME_SOURCE_INFO_KEY)
    )
    post_canary = infos.get(MICROBAN_TELEOP_V12_DEADLINE_POST_CANARY_INFO_KEY)
    post_source = infos.get(
        MICROBAN_TELEOP_V12_DEADLINE_POST_CANARY_RESUME_SOURCE_INFO_KEY
    )
    if iteration == MICROBAN_TELEOP_V12_DEADLINE_CANARY_ITERATION:
        if post_canary is not None or post_source is not None:
            raise ValueError("Deadline canary cannot contain post-canary lineage")
    else:
        validate_deadline_post_canary_marker(post_canary)
        validate_deadline_post_canary_resume_source(post_source)
    return validate_deadline_fallback_marker(marker)


def validate_deadline_fallback_canary_payload(
    payload: Mapping[str, Any],
    *,
    verify_parent_files: bool,
    checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the sole allowed descendant: the 100-update foot canary."""

    infos = payload.get("infos")
    iteration = payload.get("iter")
    if (
        checkpoint_sha256 is not None
        and checkpoint_sha256 != MICROBAN_TELEOP_V12_DEADLINE_CANARY_CHECKPOINT_SHA256
    ):
        raise ValueError("Deadline canary checkpoint SHA-256 mismatch")
    if not isinstance(infos, Mapping) or not isinstance(iteration, int):
        raise TypeError("Deadline canary checkpoint payload is malformed")
    marker = validate_deadline_fallback_descendant(infos, iteration=iteration)
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
        checkpoint = resolve_bootstrap_artifact_path(source["parent_checkpoint_path"])
        gate = resolve_bootstrap_artifact_path(source["full_stage_gate_path"])
        if (
            not checkpoint.is_file()
            or sha256_file(checkpoint)
            != MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_CHECKPOINT_SHA256
            or not gate.is_file()
            or sha256_file(gate) != source["full_stage_gate_sha256"]
        ):
            raise ValueError("Deadline canary immediate-parent files changed")
        parent = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(parent, Mapping):
            raise TypeError("Deadline canary parent payload is malformed")
        validate_deadline_fallback_checkpoint_payload(
            parent,
            checkpoint_sha256=(MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_CHECKPOINT_SHA256),
        )
        if (
            sha256_file(checkpoint)
            != MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_CHECKPOINT_SHA256
            or sha256_file(gate) != source["full_stage_gate_sha256"]
        ):
            raise ValueError("Deadline canary parent files changed while validating")
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
        iteration != MICROBAN_TELEOP_V12_DEADLINE_FINAL_ITERATION
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
    validate_deadline_post_canary_marker(
        infos.get(MICROBAN_TELEOP_V12_DEADLINE_POST_CANARY_INFO_KEY)
    )
    source = validate_deadline_post_canary_resume_source(
        infos.get(MICROBAN_TELEOP_V12_DEADLINE_POST_CANARY_RESUME_SOURCE_INFO_KEY)
    )
    if verify_parent_files:
        checkpoint = resolve_bootstrap_artifact_path(source["parent_checkpoint_path"])
        gate = resolve_bootstrap_artifact_path(source["full_stage_gate_path"])
        if (
            not checkpoint.is_file()
            or sha256_file(checkpoint)
            != MICROBAN_TELEOP_V12_DEADLINE_CANARY_CHECKPOINT_SHA256
            or not gate.is_file()
            or sha256_file(gate) != source["full_stage_gate_sha256"]
        ):
            raise ValueError("Deadline final immediate-parent files changed")
        parent = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(parent, Mapping):
            raise TypeError("Deadline final parent payload is malformed")
        validate_deadline_fallback_canary_payload(
            parent,
            verify_parent_files=True,
            checkpoint_sha256=MICROBAN_TELEOP_V12_DEADLINE_CANARY_CHECKPOINT_SHA256,
        )
        if (
            sha256_file(checkpoint)
            != MICROBAN_TELEOP_V12_DEADLINE_CANARY_CHECKPOINT_SHA256
            or sha256_file(gate) != source["full_stage_gate_sha256"]
        ):
            raise ValueError("Deadline final parent files changed while validating")
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
) -> dict[str, Any]:
    """Allow only selected-v1->canary or pinned-canary->final resume."""

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
    return validate_deadline_fallback_checkpoint_payload(
        payload, checkpoint_sha256=checkpoint_sha256
    )


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
