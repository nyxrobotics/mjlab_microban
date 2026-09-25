"""Package one accepted final contract-v12 checkpoint for Microban hardware.

This is deliberately separate from the training-time ONNX gate.  The stage
gate proves a checkpoint and its evaluation evidence; this command turns that
evidence into the metadata contract consumed by Microban, checks the final
metadata-bearing graph with both ONNX implementations, invokes Microban's real
CPU-only loader, and only then atomically publishes the requested file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import onnx
import torch
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, list_to_csv_str
from onnx.reference import ReferenceEvaluator

from mjlab_microban.robot.microban_constants import HOME_FRAME
from mjlab_microban.robot.microban_hand_fk import (
    MICROBAN_HAND_TARGET_WIRE_ABS_BOUND_M,
    microban_hand_fk_metadata,
)
from mjlab_microban.scripts.evaluate_teleop_v12_checkpoint import _load_actor
from mjlab_microban.scripts.evaluate_teleop_v12_tracking import FINAL_PROFILE
from mjlab_microban.scripts.teleop_v12_bootstrap_gate import (
    ONNX_PARITY_TOLERANCE,
    _export_onnx_atomic,
)
from mjlab_microban.scripts.teleop_v12_lr_recovery import (
    PINNED_RAW_MODEL_9200_SHA256,
    PINNED_SOURCE_COMMON_STEP_COUNTER,
    PINNED_SOURCE_COMPLETED_UPDATES,
    PINNED_SOURCE_ITERATION,
)
from mjlab_microban.scripts.teleop_v12_stage import validate_gate
from mjlab_microban.tasks.mdp import MICROBAN_BILATERAL_SITE_ORDER_REVISION
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_HMD_JOINT_NAMES,
    MICROBAN_TELEOP_ACTION_JOINT_NAMES,
    MICROBAN_TELEOP_OBSERVATION_SCHEMA,
)
from mjlab_microban.tasks.microban_teleop_v12_actor import (
    TELEOP_V12_ADAPTER_GRADIENT_SCHEDULE_REVISION,
    TELEOP_V12_EXTRA_OBSERVATION_COLUMNS,
)
from mjlab_microban.tasks.microban_teleop_v12_bootstrap import (
    TeleopV12BootstrapProvenance,
    resolve_bootstrap_artifact_path,
    sha256_file,
    validate_bootstrap_provenance,
)
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MICROBAN_TELEOP_V12_RECIPE_REVISION,
    MICROBAN_TELEOP_V12_TRAINING_CONTRACT_VERSION,
)
from mjlab_microban.tasks.microban_teleop_v12_lr_order import (
    ACTOR_SWAP_BLOCKS,
    BILATERAL_SITE_ORDER_INFO_KEY,
    CRITIC_SWAP_BLOCKS,
    MIGRATION_REVISION,
    validate_bilateral_site_order_checkpoint,
    validate_lr_order_migration_marker,
)
from mjlab_microban.tasks.microban_teleop_v12_runner import (
    TELEOP_V12_BOOTSTRAP_INFO_KEY,
)
from mjlab_microban.teleop_v12_safety import (
    ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_DEG,
    ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD,
    COMMANDED_TARGET_SOFT_LIMIT_EXCESS_MAX_RAD,
)

FINAL_ITERATION = 14_999
FINAL_COMPLETED_UPDATES = 15_000
PACKAGER_REVISION = "microban_teleop_v12_final_deployment_packager_v2"
RUNTIME_GUARD_FORMULA = "max(v12_absmax,source_absmax+delta_absmax)*multiplier"
RUNTIME_GUARD_MULTIPLIER = 2.0
RUNTIME_GUARD_SEMANTICS = (
    "finite_float32_then_per_joint_absmax_else_same_cycle_legacy_fallback_v1"
)

_FOOT_LOWER = (-0.03, -0.03, 0.0) * 2
_FOOT_UPPER = (0.03, 0.03, 0.05) * 2
_BOTH_FEET_LOWER = (-0.01, -0.01, 0.0) * 2
_BOTH_FEET_UPPER = (0.01, 0.01, 0.02) * 2
_HAND_LOWER = tuple(-value for value in MICROBAN_HAND_TARGET_WIRE_ABS_BOUND_M) * 2
_HAND_UPPER = MICROBAN_HAND_TARGET_WIRE_ABS_BOUND_M * 2
MICROBAN_RUNTIME_IDENTITY_KEYS = frozenset(
    {
        "microban_runtime_validator_source_sha256",
        "microban_runtime_contract_source_sha256",
        "microban_runtime_selector_source_sha256",
        "microban_walk_runtime_source_sha256",
        "microban_walk_config_source_sha256",
        "microban_runtime_lock_sha256",
        "microban_walk_fallback_onnx_sha256",
    }
)

# Keep this explicit.  A runtime-side metadata addition must cause a reviewed
# packager/test change rather than silently producing an artifact that can only
# be rejected after copying it to the robot.
REQUIRED_V12_RUNTIME_METADATA_KEYS = frozenset(
    {
        "policy_type",
        "microban_teleop_training_contract_version",
        "microban_teleop_recipe_revision",
        "checkpoint_filename",
        "checkpoint_iteration",
        "checkpoint_iteration_semantics",
        "checkpoint_completed_updates",
        "checkpoint_sha256",
        "deployment_accepted",
        "v12_stage_gate_schema_version",
        "v12_stage_gate_name",
        "v12_stage_gate_status",
        "v12_stage_gate_canonical_boundary",
        "v12_stage_gate_sha256",
        "v12_stage_gate_checkpoint_sha256",
        "v12_stage_gate_checkpoint_iteration",
        "v12_stage_gate_completed_updates",
        "v12_locomotion_report_sha256",
        "v12_onnx_report_sha256",
        "v12_tracking_report_sha256",
        "v12_tracking_profile",
        "v12_bootstrap_provenance_schema_version",
        "v12_bootstrap_mapping_version",
        "v12_legacy_source_checkpoint_sha256",
        "v12_legacy_source_checkpoint_iteration",
        "v12_legacy_probe_sha256",
        "v12_legacy_probe_scenario_count",
        "v12_legacy_probe_steps_per_scenario",
        "v12_legacy_probe_settle_steps",
        "v12_legacy_probe_seed",
        "v12_bilateral_site_order_revision",
        "v12_lr_order_migration_schema_version",
        "v12_lr_order_migration_revision",
        "v12_lr_order_migration_strategy",
        "v12_lr_order_source_checkpoint_sha256",
        "v12_lr_order_source_checkpoint_iteration",
        "v12_lr_order_source_completed_updates",
        "v12_lr_order_source_common_step_counter",
        "v12_lr_order_actor_swap_blocks_json",
        "v12_lr_order_critic_swap_blocks_json",
        "v12_lr_order_foot_adapter_at_source",
        "v12_lr_order_migration_marker_sha256",
        "v12_source_to_target_columns_json",
        "v12_extra_observation_columns_json",
        "v12_actor_topology_json",
        "v12_normalizer_eps",
        "v12_normalizer_semantics",
        "v12_trainable_actor_parameters",
        "adapter_gradient_schedule_revision",
        "v12_active_actor_columns_at_save_json",
        "v12_frozen_legacy_tensors_verified",
        "v12_locomotion_gate",
        "v12_locomotion_status",
        "v12_locomotion_seed",
        "v12_locomotion_scenario_count",
        "v12_locomotion_steps_per_scenario",
        "v12_locomotion_settle_steps",
        "v12_locomotion_fall_scenario_count",
        "v12_locomotion_nonfinite_scenario_count",
        "v12_actual_dynamic_soft_limit_overshoot_max_deg",
        "v12_actual_dynamic_soft_limit_overshoot_max_rad",
        "v12_commanded_target_soft_limit_excess_max_rad",
        "v12_locomotion_actual_soft_limit_violation_scenario_count",
        "v12_locomotion_directionally_correct_scenario_count",
        "v12_locomotion_directional_scenario_count",
        "v12_locomotion_raw_action_recurrence_all_steps",
        "v12_onnx_gate",
        "v12_onnx_verified",
        "v12_onnx_parity_teleop_columns",
        "v12_onnx_parity_seed",
        "v12_onnx_parity_sample_count",
        "v12_onnx_parity_atol",
        "v12_onnx_reference_max_abs_error",
        "v12_onnxruntime_cpu_max_abs_error",
        "v12_neutral_legacy_parity_max_abs_error",
        "v12_neutral_legacy_parity_sample_count",
        "v12_raw_action_envelope_schema_version",
        "v12_raw_action_joint_names_json",
        "v12_raw_action_min_json",
        "v12_raw_action_max_json",
        "v12_raw_action_absmax_json",
        "v12_source_raw_action_min_json",
        "v12_source_raw_action_max_json",
        "v12_source_raw_action_absmax_json",
        "v12_learned_source_delta_min_json",
        "v12_learned_source_delta_max_json",
        "v12_learned_source_delta_absmax_json",
        "runtime_raw_action_guard_formula",
        "runtime_raw_action_guard_multiplier",
        "runtime_raw_action_guard_absmax_json",
        "runtime_raw_action_guard_semantics",
        "observation_schema_version",
        "observation_width",
        "action_width",
        "observation_schema_json",
        "observation_names",
        "observation_joint_names",
        "observation_default_joint_pos",
        "action_joint_names",
        "default_joint_pos",
        "action_scale",
        "base_ang_vel_frame",
        "base_ang_vel_units",
        "locomotion_command_order",
        "locomotion_command_units",
        "locomotion_command_frame",
        "previous_action_semantics",
        "action_target_semantics",
        "action_clip_semantics",
        "action_distribution_semantics",
        "runtime_action_semantics",
        "control_hz",
        "foot_target_lower",
        "foot_target_upper",
        "foot_target_frame",
        "foot_target_units",
        "foot_target_semantics",
        "simultaneous_both_feet_target_lower",
        "simultaneous_both_feet_target_upper",
        "simultaneous_both_feet_target_semantics",
        "simultaneous_both_feet_requires_zero_twist",
        "hand_target_lower",
        "hand_target_upper",
        "hand_target_fk",
        "hand_target_frame",
        "hand_target_units",
        "hand_target_semantics",
        "microban_runtime_validator_source_sha256",
        "microban_runtime_contract_source_sha256",
        "microban_runtime_selector_source_sha256",
        "microban_walk_runtime_source_sha256",
        "microban_walk_config_source_sha256",
        "microban_runtime_lock_sha256",
        "microban_walk_fallback_onnx_sha256",
    }
)


def _load_json(path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"JSON duplicates key {key!r}: {path}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"JSON contains non-finite value {value!r}: {path}")

    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"JSON SHA-256 mismatch for {path}: {digest}")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"JSON is not UTF-8: {path}") from exc
    value = json.loads(
        text,
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_deployable_lr_order_lineage(infos: Mapping[str, Any]) -> dict[str, Any]:
    """Require the exact authenticated recovery used by the canonical v12 run."""

    marker = validate_bilateral_site_order_checkpoint(infos)
    if infos.get(BILATERAL_SITE_ORDER_INFO_KEY) != (
        MICROBAN_BILATERAL_SITE_ORDER_REVISION
    ):
        raise ValueError(
            "Final checkpoint does not carry the corrected bilateral site-order revision"
        )
    if marker is None:
        raise ValueError(
            "Final checkpoint lacks the authenticated model-9200 bilateral migration"
        )
    marker = validate_lr_order_migration_marker(marker)
    expected = {
        "schema_version": 1,
        "revision": MIGRATION_REVISION,
        "site_order_revision": MICROBAN_BILATERAL_SITE_ORDER_REVISION,
        "strategy": "swap",
        "source_checkpoint_sha256": PINNED_RAW_MODEL_9200_SHA256,
        "actor_swap_blocks": [list(block) for block in ACTOR_SWAP_BLOCKS],
        "critic_swap_blocks": [list(block) for block in CRITIC_SWAP_BLOCKS],
        "zeroed_actor_columns": [],
        "foot_adapter_at_source": {
            "active": False,
            "maximum_absolute_w0": 0.0,
            "maximum_absolute_adam_moment": 0.0,
            "handling": "unlearned_exact_zero_left_untouched",
        },
    }
    mismatches = [name for name, value in expected.items() if marker.get(name) != value]
    expected_clock = {
        "iteration": PINNED_SOURCE_ITERATION,
        "completed_updates": PINNED_SOURCE_COMPLETED_UPDATES,
        "common_step_counter": PINNED_SOURCE_COMMON_STEP_COUNTER,
    }
    if marker.get("source_clock") != expected_clock:
        mismatches.append("source_clock")
    if mismatches:
        raise ValueError(
            "Final checkpoint bilateral migration lineage drifted: "
            + ", ".join(mismatches)
        )
    return marker


def _require_final_gate(
    gate: Mapping[str, Any], *, checkpoint: Path, checkpoint_sha256: str
) -> None:
    expected = {
        "schema_version": 2,
        "gate": "microban_teleop_v12_stage",
        "status": "pass",
        "checkpoint_sha256": checkpoint_sha256,
        "iteration": FINAL_ITERATION,
        "completed_updates": FINAL_COMPLETED_UPDATES,
        "canonical_boundary": True,
        "checkpoint_kind": "canonical_boundary",
        "tracking_profile": FINAL_PROFILE,
    }
    mismatches = [name for name, value in expected.items() if gate.get(name) != value]
    if mismatches:
        raise ValueError(
            "Contract-v12 deployment requires the exact accepted 15000-update "
            "gate; mismatched fields: " + ", ".join(mismatches)
        )
    if checkpoint.name != f"model_{FINAL_ITERATION}.pt":
        raise ValueError("Final contract-v12 checkpoint must be named model_14999.pt")


def _wire_metadata_value(value: list | str | float) -> str:
    return list_to_csv_str(value) if isinstance(value, list) else str(value)


def _read_onnx_metadata(path: Path) -> dict[str, str]:
    model = onnx.load(path)
    result: dict[str, str] = {}
    for item in model.metadata_props:
        if item.key in result:
            raise ValueError(f"Duplicate ONNX metadata key: {item.key}")
        result[item.key] = item.value
    return result


def _checked_envelope_vector(
    envelope: Mapping[str, Any], policy: str, statistic: str
) -> list[float]:
    policy_value = envelope.get(policy)
    if not isinstance(policy_value, Mapping):
        raise TypeError(f"Tracking envelope {policy!r} is missing")
    value = policy_value.get(statistic)
    if (
        not isinstance(value, list)
        or len(value) != len(MICROBAN_TELEOP_ACTION_JOINT_NAMES)
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in value
        )
    ):
        raise ValueError(f"Tracking envelope {policy}/{statistic} is malformed")
    return [float(item) for item in value]


def _runtime_guard(envelope: Mapping[str, Any]) -> list[float]:
    learned = _checked_envelope_vector(envelope, "v12", "absolute_maximum")
    source = _checked_envelope_vector(envelope, "legacy_source", "absolute_maximum")
    delta = _checked_envelope_vector(
        envelope, "learned_minus_source", "absolute_maximum"
    )
    result = [
        max(v12, legacy + difference) * RUNTIME_GUARD_MULTIPLIER
        for v12, legacy, difference in zip(learned, source, delta, strict=True)
    ]
    with np.errstate(over="ignore", invalid="ignore"):
        as_float32 = np.asarray(result, dtype=np.float32)
    if not np.isfinite(as_float32).all() or bool(np.any(as_float32 < 0.0)):
        raise ValueError("Tracking evidence produced an invalid float32 runtime guard")
    return result


def build_v12_deployment_metadata(
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
    gate_path: Path,
    gate: Mapping[str, Any],
    infos: Mapping[str, Any],
    bootstrap: TeleopV12BootstrapProvenance,
    locomotion: Mapping[str, Any],
    tracking: Mapping[str, Any],
    onnx_report: Mapping[str, Any],
    packager_parity: Mapping[str, float],
    microban_source_identity: Mapping[str, str],
) -> dict[str, list | str | float]:
    """Translate only already-validated gate evidence to the robot wire contract."""

    _require_final_gate(
        gate, checkpoint=checkpoint, checkpoint_sha256=checkpoint_sha256
    )
    if infos.get("trainable_actor_parameters") != ["mlp.0.weight"] or infos.get(
        "trainable_actor_columns"
    ) != list(TELEOP_V12_EXTRA_OBSERVATION_COLUMNS):
        raise ValueError("Final checkpoint trainable adapter declaration drifted")
    if infos.get("active_actor_columns_at_save") != list(
        TELEOP_V12_EXTRA_OBSERVATION_COLUMNS
    ):
        raise ValueError("Final checkpoint did not activate all v12 adapter columns")
    if set(microban_source_identity) != MICROBAN_RUNTIME_IDENTITY_KEYS or any(
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        for digest in microban_source_identity.values()
    ):
        raise ValueError("Microban runtime source identity is incomplete or malformed")

    report_hashes = gate.get("report_sha256")
    if not isinstance(report_hashes, Mapping):
        raise TypeError("V12 gate report hashes are malformed")
    locomotion_summary = locomotion.get("summary")
    locomotion_settings = locomotion.get("settings")
    locomotion_results = locomotion.get("results")
    tracking_envelope = tracking.get("raw_action_envelope")
    neutral = onnx_report.get("neutral_legacy_parity")
    onnx_evidence = onnx_report.get("onnx")
    if (
        not isinstance(locomotion_summary, Mapping)
        or not isinstance(locomotion_settings, Mapping)
        or not isinstance(locomotion_results, list)
        or not isinstance(tracking_envelope, Mapping)
        or not isinstance(neutral, Mapping)
        or not isinstance(onnx_evidence, Mapping)
    ):
        raise TypeError("Validated v12 gate evidence is incomplete")

    defaults = HOME_FRAME.joint_pos
    if not isinstance(defaults, Mapping):
        raise TypeError("Microban HOME_FRAME has no joint pose")
    observation_joints = [
        *MICROBAN_HMD_JOINT_NAMES,
        *MICROBAN_TELEOP_ACTION_JOINT_NAMES,
    ]
    try:
        observation_defaults = [float(defaults[name]) for name in observation_joints]
    except KeyError as exc:
        raise RuntimeError(f"Microban HOME_FRAME is missing {exc.args[0]!r}") from exc
    action_defaults = [
        float(defaults[name]) for name in MICROBAN_TELEOP_ACTION_JOINT_NAMES
    ]
    guard = _runtime_guard(tracking_envelope)
    source = bootstrap.source
    probe = bootstrap.probe
    lr_order_migration = _require_deployable_lr_order_lineage(infos)
    if dict(lr_order_migration) != dict(
        validate_lr_order_migration_marker(lr_order_migration)
    ):
        raise RuntimeError("Bilateral migration marker changed during validation")
    lr_source_clock = lr_order_migration["source_clock"]
    packager_source = Path(__file__).resolve()

    metadata: dict[str, list | str | float] = {
        "run_path": checkpoint.parent.name,
        "policy_type": "microban_pico_hybrid_teleop",
        "microban_teleop_training_contract_version": (
            MICROBAN_TELEOP_V12_TRAINING_CONTRACT_VERSION
        ),
        "microban_teleop_recipe_revision": MICROBAN_TELEOP_V12_RECIPE_REVISION,
        "checkpoint_filename": checkpoint.name,
        "checkpoint_iteration": str(FINAL_ITERATION),
        "checkpoint_iteration_semantics": (
            "zero_based_completed_update_index_from_model_filename"
        ),
        "checkpoint_completed_updates": str(FINAL_COMPLETED_UPDATES),
        "checkpoint_sha256": checkpoint_sha256,
        "deployment_accepted": "true",
        "v12_stage_gate_schema_version": str(gate["schema_version"]),
        "v12_stage_gate_name": str(gate["gate"]),
        "v12_stage_gate_status": str(gate["status"]),
        "v12_stage_gate_canonical_boundary": "true",
        "v12_stage_gate_sha256": sha256_file(gate_path),
        "v12_stage_gate_checkpoint_sha256": str(gate["checkpoint_sha256"]),
        "v12_stage_gate_checkpoint_iteration": str(gate["iteration"]),
        "v12_stage_gate_completed_updates": str(gate["completed_updates"]),
        "v12_locomotion_report_sha256": str(report_hashes["locomotion"]),
        "v12_tracking_report_sha256": str(report_hashes["tracking"]),
        "v12_onnx_report_sha256": str(report_hashes["onnx"]),
        "v12_tracking_profile": str(gate["tracking_profile"]),
        "v12_gate_onnx_artifact_sha256": str(gate["onnx"]["sha256"]),
        "v12_bootstrap_provenance_schema_version": str(bootstrap.schema_version),
        "v12_bootstrap_mapping_version": bootstrap.mapping_version,
        "v12_legacy_source_checkpoint_sha256": source.sha256,
        "v12_legacy_source_checkpoint_iteration": str(source.iteration),
        "v12_legacy_probe_sha256": probe.sha256,
        "v12_legacy_probe_scenario_count": str(probe.scenario_count),
        "v12_legacy_probe_steps_per_scenario": str(probe.steps),
        "v12_legacy_probe_settle_steps": str(probe.settle_steps),
        "v12_legacy_probe_seed": str(probe.seed),
        "v12_bilateral_site_order_revision": (
            MICROBAN_BILATERAL_SITE_ORDER_REVISION
        ),
        "v12_lr_order_migration_schema_version": str(
            lr_order_migration["schema_version"]
        ),
        "v12_lr_order_migration_revision": str(lr_order_migration["revision"]),
        "v12_lr_order_migration_strategy": str(lr_order_migration["strategy"]),
        "v12_lr_order_source_checkpoint_sha256": str(
            lr_order_migration["source_checkpoint_sha256"]
        ),
        "v12_lr_order_source_checkpoint_iteration": str(
            lr_source_clock["iteration"]
        ),
        "v12_lr_order_source_completed_updates": str(
            lr_source_clock["completed_updates"]
        ),
        "v12_lr_order_source_common_step_counter": str(
            lr_source_clock["common_step_counter"]
        ),
        "v12_lr_order_actor_swap_blocks_json": _json(
            lr_order_migration["actor_swap_blocks"]
        ),
        "v12_lr_order_critic_swap_blocks_json": _json(
            lr_order_migration["critic_swap_blocks"]
        ),
        "v12_lr_order_foot_adapter_at_source": (
            "inactive_exact_zero_left_untouched"
        ),
        "v12_lr_order_migration_marker_sha256": _canonical_json_sha256(
            lr_order_migration
        ),
        "v12_source_to_target_columns_json": _json(
            [list(pair) for pair in bootstrap.source_to_target_columns]
        ),
        "v12_extra_observation_columns_json": _json(
            list(bootstrap.new_trainable_columns)
        ),
        "v12_actor_topology_json": _json(list(bootstrap.target_actor_topology)),
        "v12_normalizer_eps": str(bootstrap.normalizer_eps),
        "v12_normalizer_semantics": (
            "frozen_source63_identity_hmd_flags_reachable_fk_target_scaling_v3"
        ),
        "v12_trainable_actor_parameters": "mlp.0.weight_extra_columns_only",
        "adapter_gradient_schedule_revision": (
            TELEOP_V12_ADAPTER_GRADIENT_SCHEDULE_REVISION
        ),
        "v12_active_actor_columns_at_save_json": _json(
            list(infos["active_actor_columns_at_save"])
        ),
        "v12_frozen_legacy_tensors_verified": "true",
        "v12_locomotion_gate": str(locomotion["gate"]),
        "v12_locomotion_status": str(locomotion["status"]),
        "v12_locomotion_seed": str(locomotion_settings["seed"]),
        "v12_locomotion_scenario_count": str(locomotion_summary["scenario_count"]),
        "v12_locomotion_steps_per_scenario": str(locomotion_settings["steps"]),
        "v12_locomotion_settle_steps": str(locomotion_settings["settle_steps"]),
        "v12_locomotion_fall_scenario_count": str(
            locomotion_summary["fall_scenario_count"]
        ),
        "v12_locomotion_nonfinite_scenario_count": str(
            locomotion_summary["nonfinite_scenario_count"]
        ),
        "v12_actual_dynamic_soft_limit_overshoot_max_deg": str(
            ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_DEG
        ),
        "v12_actual_dynamic_soft_limit_overshoot_max_rad": str(
            ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD
        ),
        "v12_commanded_target_soft_limit_excess_max_rad": str(
            COMMANDED_TARGET_SOFT_LIMIT_EXCESS_MAX_RAD
        ),
        "v12_locomotion_actual_soft_limit_violation_scenario_count": str(
            sum(
                float(result["maximum_actual_soft_limit_violation_rad"])
                > ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD
                for result in locomotion_results
            )
        ),
        "v12_locomotion_directionally_correct_scenario_count": str(
            locomotion_summary["directionally_correct_scenario_count"]
        ),
        "v12_locomotion_directional_scenario_count": str(
            locomotion_summary["directional_scenario_count"]
        ),
        "v12_locomotion_raw_action_recurrence_all_steps": "true",
        "v12_onnx_gate": str(onnx_report["gate"]),
        "v12_onnx_verified": "true",
        "v12_onnx_parity_teleop_columns": "random_finite_not_zeroed",
        "v12_onnx_parity_seed": "20260925",
        "v12_onnx_parity_sample_count": str(onnx_evidence["reference_samples"]),
        "v12_onnx_parity_atol": str(onnx_evidence["tolerance"]),
        "v12_onnx_reference_max_abs_error": str(
            onnx_evidence["reference_evaluator_maximum_absolute_error"]
        ),
        "v12_onnxruntime_cpu_max_abs_error": str(
            onnx_evidence["onnxruntime_cpu_maximum_absolute_error"]
        ),
        "v12_neutral_legacy_parity_max_abs_error": str(
            neutral["maximum_absolute_error"]
        ),
        "v12_neutral_legacy_parity_sample_count": str(neutral["samples"]),
        "v12_raw_action_envelope_schema_version": "1",
        "v12_raw_action_joint_names_json": _json(
            list(MICROBAN_TELEOP_ACTION_JOINT_NAMES)
        ),
        "v12_raw_action_min_json": _json(
            _checked_envelope_vector(tracking_envelope, "v12", "minimum")
        ),
        "v12_raw_action_max_json": _json(
            _checked_envelope_vector(tracking_envelope, "v12", "maximum")
        ),
        "v12_raw_action_absmax_json": _json(
            _checked_envelope_vector(tracking_envelope, "v12", "absolute_maximum")
        ),
        "v12_source_raw_action_min_json": _json(
            _checked_envelope_vector(tracking_envelope, "legacy_source", "minimum")
        ),
        "v12_source_raw_action_max_json": _json(
            _checked_envelope_vector(tracking_envelope, "legacy_source", "maximum")
        ),
        "v12_source_raw_action_absmax_json": _json(
            _checked_envelope_vector(
                tracking_envelope, "legacy_source", "absolute_maximum"
            )
        ),
        "v12_learned_source_delta_min_json": _json(
            _checked_envelope_vector(
                tracking_envelope, "learned_minus_source", "minimum"
            )
        ),
        "v12_learned_source_delta_max_json": _json(
            _checked_envelope_vector(
                tracking_envelope, "learned_minus_source", "maximum"
            )
        ),
        "v12_learned_source_delta_absmax_json": _json(
            _checked_envelope_vector(
                tracking_envelope, "learned_minus_source", "absolute_maximum"
            )
        ),
        "runtime_raw_action_guard_formula": RUNTIME_GUARD_FORMULA,
        "runtime_raw_action_guard_multiplier": str(RUNTIME_GUARD_MULTIPLIER),
        "runtime_raw_action_guard_absmax_json": _json(guard),
        "runtime_raw_action_guard_semantics": RUNTIME_GUARD_SEMANTICS,
        "observation_schema_version": "2",
        "observation_width": "83",
        "action_width": "18",
        "observation_schema_json": _json(
            [[name, width] for name, width in MICROBAN_TELEOP_OBSERVATION_SCHEMA]
        ),
        "observation_names": [
            name for name, _width in MICROBAN_TELEOP_OBSERVATION_SCHEMA
        ],
        "observation_joint_names": observation_joints,
        "observation_default_joint_pos": observation_defaults,
        "action_joint_names": list(MICROBAN_TELEOP_ACTION_JOINT_NAMES),
        "default_joint_pos": action_defaults,
        "action_scale": [1.0] * len(MICROBAN_TELEOP_ACTION_JOINT_NAMES),
        "base_ang_vel_frame": "robot_body_xyz",
        "base_ang_vel_units": "rad_s",
        "locomotion_command_order": [
            "linear_velocity_x",
            "linear_velocity_y",
            "angular_velocity_z",
        ],
        "locomotion_command_units": ["m_s", "m_s", "rad_s"],
        "locomotion_command_frame": "robot_body_forward_left_yaw_up",
        "previous_action_semantics": "raw_actor_output",
        "action_target_semantics": ("default_joint_pos_plus_raw_action_times_scale"),
        "action_clip_semantics": "none",
        "action_distribution_semantics": ("unbounded_gaussian_deterministic_mean_raw"),
        "runtime_action_semantics": (
            "raw_unbounded_default_plus_scale_no_target_clip_v1"
        ),
        "control_hz": 50.0,
        "foot_target_lower": list(_FOOT_LOWER),
        "foot_target_upper": list(_FOOT_UPPER),
        "foot_target_frame": "robot_trunk_xyz_forward_left_up",
        "foot_target_units": "metres",
        "foot_target_semantics": (
            "left_xyz_then_right_xyz_trunk_frame_offset_from_episode_reset_"
            "reference_metres_periodic_command_resampling_does_not_move_reference"
        ),
        "simultaneous_both_feet_target_lower": list(_BOTH_FEET_LOWER),
        "simultaneous_both_feet_target_upper": list(_BOTH_FEET_UPPER),
        "simultaneous_both_feet_target_semantics": (
            "left_and_right_nonzero_offsets_use_conservative_stationary_"
            "training_support"
        ),
        "simultaneous_both_feet_requires_zero_twist": "true",
        "hand_target_lower": list(_HAND_LOWER),
        "hand_target_upper": list(_HAND_UPPER),
        "hand_target_fk": _json(microban_hand_fk_metadata()),
        "hand_target_frame": "robot_trunk_xyz_forward_left_up",
        "hand_target_units": "metres",
        "hand_target_semantics": (
            "left_xyz_then_right_xyz_then_left_right_active_flags_"
            "trunk_frame_offset_from_episode_reset_reference_metres_"
            "periodic_command_resampling_does_not_move_reference"
        ),
        "v12_deployment_packager_revision": PACKAGER_REVISION,
        "v12_deployment_packager_source_sha256": sha256_file(packager_source),
        "v12_deployment_packager_reference_max_abs_error": str(
            packager_parity["reference_maximum_absolute_error"]
        ),
        "v12_deployment_packager_onnxruntime_cpu_max_abs_error": str(
            packager_parity["onnxruntime_cpu_maximum_absolute_error"]
        ),
        **microban_source_identity,
    }
    missing = REQUIRED_V12_RUNTIME_METADATA_KEYS.difference(metadata)
    if missing:
        raise RuntimeError(
            "Packager omitted required Microban v12 metadata: "
            + ", ".join(sorted(missing))
        )
    return metadata


def _validate_graph_contract(path: Path) -> None:
    model = onnx.load(path)
    onnx.checker.check_model(model, full_check=True)
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("Contract-v12 deployment ONNX must have one input/output")
    input_value = model.graph.input[0]
    output_value = model.graph.output[0]
    if input_value.name != "obs" or output_value.name != "actions":
        raise ValueError("Contract-v12 deployment tensors must be obs -> actions")
    for value, expected in ((input_value, [1, 83]), (output_value, [1, 18])):
        tensor = value.type.tensor_type
        shape = [dimension.dim_value for dimension in tensor.shape.dim]
        if tensor.elem_type != onnx.TensorProto.FLOAT or shape != expected:
            raise ValueError(
                f"Unsafe ONNX tensor {value.name}: type={tensor.elem_type}, "
                f"shape={shape}, expected float32 {expected}"
            )


def _validate_final_parity(actor: torch.nn.Module, path: Path) -> dict[str, float]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime CPU is required for v12 deployment") from exc

    generator = torch.Generator().manual_seed(20260925)
    # Advance the generator exactly as the hash-bound ONNX report did before
    # drawing its full-83-column corpus.
    torch.randn(10_000, 83, generator=generator)
    observations = torch.randn(64, 83, generator=generator)
    if not bool(
        torch.all(
            observations[:, TELEOP_V12_EXTRA_OBSERVATION_COLUMNS].abs().amax(dim=0)
            > 0.0
        ).item()
    ):
        raise AssertionError("Deployment parity corpus missed a teleop-only column")

    model = onnx.load(path)
    reference = ReferenceEvaluator(model)
    runtime = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    if runtime.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("Deployment parity did not use CPUExecutionProvider only")
    export_model = actor.as_onnx(verbose=False).cpu().eval()
    reference_max = 0.0
    runtime_max = 0.0
    with torch.inference_mode():
        for observation in observations:
            batch = observation.unsqueeze(0)
            expected = export_model(batch).detach().cpu().numpy()
            (reference_actual,) = reference.run(None, {"obs": batch.numpy()})
            (runtime_actual,) = runtime.run(None, {"obs": batch.numpy()})
            reference_error = float(np.max(np.abs(reference_actual - expected)))
            runtime_error = float(np.max(np.abs(runtime_actual - expected)))
            reference_max = max(reference_max, reference_error)
            runtime_max = max(runtime_max, runtime_error)
    if (
        not math.isfinite(reference_max)
        or not math.isfinite(runtime_max)
        or reference_max > ONNX_PARITY_TOLERANCE
        or runtime_max > ONNX_PARITY_TOLERANCE
    ):
        raise ValueError(
            "Final metadata-bearing ONNX parity failed: "
            f"reference={reference_max}, onnxruntime_cpu={runtime_max}, "
            f"tolerance={ONNX_PARITY_TOLERANCE}"
        )
    return {
        "reference_maximum_absolute_error": reference_max,
        "onnxruntime_cpu_maximum_absolute_error": runtime_max,
    }


def _run_microban_runtime_validator(
    path: Path, *, microban_repo: Path
) -> dict[str, Any]:
    microban_repo = microban_repo.expanduser().resolve()
    validator = microban_repo / "tools" / "validate_pico_policy.py"
    lock = microban_repo / "uv.lock"
    uv = shutil.which("uv")
    if uv is None or not validator.is_file() or not lock.is_file():
        raise FileNotFoundError(
            "Microban runtime validator, uv.lock, and uv executable are required"
        )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(microban_repo / "src")
    environment["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        [
            uv,
            "run",
            "--project",
            str(microban_repo),
            "--locked",
            "python",
            str(validator),
            str(path),
        ],
        cwd=microban_repo,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(
            "Microban runtime validator rejected the deployment ONNX"
            + (f": {detail[-2000:]}" if detail else "")
        )
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Microban runtime validator returned invalid JSON") from exc
    smoke = report.get("onnxruntime_compatibility_smoke")
    metadata = _read_onnx_metadata(path)
    runtime_source_identity = _microban_runtime_source_identity(microban_repo)
    reported_runtime_source_identity = report.get("runtime_source_identity")
    walk_fallback = report.get("walk_fallback")
    walk_smoke = walk_fallback.get("smoke") if isinstance(walk_fallback, Mapping) else None
    walk_input = walk_fallback.get("input") if isinstance(walk_fallback, Mapping) else None
    walk_output = walk_fallback.get("output") if isinstance(walk_fallback, Mapping) else None
    expected_walk_path = str((microban_repo / "src" / "agents" / "walk.onnx").resolve())
    maximum_walk_output = (
        walk_smoke.get("maximum_absolute_output")
        if isinstance(walk_smoke, Mapping)
        else None
    )
    if (
        report.get("status") != "pass"
        or report.get("policy") != str(path.resolve())
        or report.get("input_width") != 83
        or report.get("output_width") != 18
        or report.get("training_contract_version")
        != MICROBAN_TELEOP_V12_TRAINING_CONTRACT_VERSION
        or report.get("checkpoint_iteration") != FINAL_ITERATION
        or report.get("checkpoint_completed_updates") != FINAL_COMPLETED_UPDATES
        or report.get("checkpoint_sha256") != metadata.get("checkpoint_sha256")
        or report.get("v12_stage_gate_sha256") != metadata.get("v12_stage_gate_sha256")
        or not isinstance(smoke, dict)
        or smoke.get("status") != "pass"
        or smoke.get("sample_count") != 16
        or smoke.get("providers") != ["CPUExecutionProvider"]
        or reported_runtime_source_identity != runtime_source_identity
        or any(
            metadata.get(name) != digest
            for name, digest in runtime_source_identity.items()
        )
        or not isinstance(walk_fallback, Mapping)
        or walk_fallback.get("status") != "pass"
        or walk_fallback.get("policy") != expected_walk_path
        or walk_fallback.get("sha256")
        != runtime_source_identity["microban_walk_fallback_onnx_sha256"]
        or walk_fallback.get("providers") != ["CPUExecutionProvider"]
        or walk_input
        != {"name": "obs", "shape": [1, 63], "type": "tensor(float)"}
        or walk_output
        != {"name": "actions", "shape": [1, 18], "type": "tensor(float)"}
        or not isinstance(walk_smoke, Mapping)
        or walk_smoke.get("status") != "pass"
        or walk_smoke.get("sample_count") != 16
        or walk_smoke.get("corpus") != "deterministic_exact_float32_mod29_v1"
        or walk_smoke.get("all_outputs_finite") is not True
        or isinstance(maximum_walk_output, bool)
        or not isinstance(maximum_walk_output, (int, float))
        or not math.isfinite(float(maximum_walk_output))
        or float(maximum_walk_output) < 0.0
    ):
        raise RuntimeError(
            "Microban runtime validator report is not a complete CPU v12/fallback pass"
        )
    return report


def _capture_checkpoint(checkpoint: Path, destination: Path) -> str:
    digest = hashlib.sha256()
    with checkpoint.open("rb") as source, destination.open("xb") as target:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
            target.write(chunk)
        target.flush()
        os.fsync(target.fileno())
    return digest.hexdigest()


def _microban_runtime_source_identity(microban_repo: Path) -> dict[str, str]:
    repo = microban_repo.expanduser().resolve()
    paths = {
        "microban_runtime_validator_source_sha256": (
            repo / "tools" / "validate_pico_policy.py"
        ),
        "microban_runtime_contract_source_sha256": (
            repo / "src" / "moves" / "pico_hybrid.py"
        ),
        "microban_runtime_selector_source_sha256": (
            repo / "src" / "moves" / "policy_selector.py"
        ),
        "microban_walk_runtime_source_sha256": repo / "src" / "moves" / "walk.py",
        "microban_walk_config_source_sha256": repo / "src" / "constants.py",
        "microban_runtime_lock_sha256": repo / "uv.lock",
        "microban_walk_fallback_onnx_sha256": repo / "src" / "agents" / "walk.onnx",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Microban runtime source identity is incomplete: " + ", ".join(missing)
        )
    return {name: sha256_file(path) for name, path in paths.items()}


def _reject_protected_output(output: Path, protected: Mapping[str, Path]) -> None:
    conflicts = [name for name, path in protected.items() if output == path.resolve()]
    if conflicts:
        raise ValueError(
            "Deployment output would overwrite immutable input/source: "
            + ", ".join(conflicts)
        )


def package_v12_deployment(
    *,
    checkpoint: Path,
    gate_path: Path,
    output: Path,
    microban_repo: Path,
    force: bool = False,
) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve()
    gate_path = gate_path.expanduser().resolve()
    output = output.expanduser().absolute()
    output = output.parent.resolve() / output.name
    if not checkpoint.is_file() or not gate_path.is_file():
        raise FileNotFoundError("Checkpoint and v12 stage gate must both exist")
    if output.exists() and not force:
        raise FileExistsError(f"Deployment output exists (pass --force): {output}")

    gate_snapshot = _load_json(gate_path)
    gate_sha256 = sha256_file(gate_path)
    gate = validate_gate(gate_path, checkpoint)
    if gate != gate_snapshot or sha256_file(gate_path) != gate_sha256:
        raise RuntimeError("V12 stage gate changed while it was validated")
    checkpoint_sha256 = sha256_file(checkpoint)
    _require_final_gate(
        gate, checkpoint=checkpoint, checkpoint_sha256=checkpoint_sha256
    )
    reports = gate.get("reports")
    if not isinstance(reports, Mapping):
        raise TypeError("V12 stage gate report references are malformed")
    report_paths = {
        name: resolve_bootstrap_artifact_path(reports[name])
        for name in ("locomotion", "tracking", "onnx")
    }
    report_hashes = gate.get("report_sha256")
    if not isinstance(report_hashes, Mapping):
        raise TypeError("V12 stage gate report hashes are malformed")
    for name in ("locomotion", "tracking", "onnx"):
        expected = report_hashes.get(name)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"V12 stage gate {name} report SHA-256 is malformed")
    locomotion = _load_json(
        report_paths["locomotion"],
        expected_sha256=str(report_hashes["locomotion"]),
    )
    tracking = _load_json(
        report_paths["tracking"],
        expected_sha256=str(report_hashes["tracking"]),
    )
    onnx_report = _load_json(
        report_paths["onnx"], expected_sha256=str(report_hashes["onnx"])
    )
    microban_source_identity = _microban_runtime_source_identity(microban_repo)
    microban_repo_resolved = microban_repo.expanduser().resolve()
    _reject_protected_output(
        output,
        {
            "checkpoint": checkpoint,
            "stage_gate": gate_path,
            **{f"{name}_report": path for name, path in report_paths.items()},
            "stage_gate_onnx": resolve_bootstrap_artifact_path(gate["onnx"]["path"]),
            "packager_source": Path(__file__).resolve(),
            "microban_validator": (
                microban_repo_resolved / "tools" / "validate_pico_policy.py"
            ),
            "microban_runtime_contract": (
                microban_repo_resolved / "src" / "moves" / "pico_hybrid.py"
            ),
            "microban_runtime_selector": (
                microban_repo_resolved / "src" / "moves" / "policy_selector.py"
            ),
            "microban_walk_runtime": (
                microban_repo_resolved / "src" / "moves" / "walk.py"
            ),
            "microban_walk_config": microban_repo_resolved / "src" / "constants.py",
            "microban_walk_fallback": (
                microban_repo_resolved / "src" / "agents" / "walk.onnx"
            ),
            "microban_runtime_lock": microban_repo_resolved / "uv.lock",
        },
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    captured = output.with_name(f".{checkpoint.name}.{uuid4().hex}.captured")
    try:
        captured_sha256 = _capture_checkpoint(checkpoint, captured)
        if captured_sha256 != checkpoint_sha256:
            raise RuntimeError("Checkpoint changed while its bytes were captured")
        actor, iteration, infos = _load_actor(captured, device="cpu")
        if iteration != FINAL_ITERATION:
            raise ValueError("Captured checkpoint is not final model_14999.pt")
        bootstrap = validate_bootstrap_provenance(
            infos.get(TELEOP_V12_BOOTSTRAP_INFO_KEY), verify_files=True
        )
        _require_deployable_lr_order_lineage(infos)

        _export_onnx_atomic(actor, temporary)
        _validate_graph_contract(temporary)
        initial_parity = _validate_final_parity(actor, temporary)
        metadata = build_v12_deployment_metadata(
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            gate_path=gate_path,
            gate=gate,
            infos=infos,
            bootstrap=bootstrap,
            locomotion=locomotion,
            tracking=tracking,
            onnx_report=onnx_report,
            packager_parity=initial_parity,
            microban_source_identity=microban_source_identity,
        )
        existing = _read_onnx_metadata(temporary)
        overlap = set(existing).intersection(metadata)
        if overlap:
            raise ValueError(
                "Exported ONNX already claims deployment metadata: "
                + ", ".join(sorted(overlap))
            )
        attach_metadata_to_onnx(str(temporary), metadata)
        attached = _read_onnx_metadata(temporary)
        for key, expected in metadata.items():
            wire = _wire_metadata_value(expected)
            if attached.get(key) != wire:
                raise ValueError(
                    f"Attached ONNX metadata mismatch for {key}: "
                    f"{attached.get(key)!r} != {wire!r}"
                )
        _validate_graph_contract(temporary)
        final_parity = _validate_final_parity(actor, temporary)
        runtime_report = _run_microban_runtime_validator(
            temporary, microban_repo=microban_repo
        )

        # Revalidate every mutable source immediately before publication.
        if sha256_file(checkpoint) != checkpoint_sha256:
            raise RuntimeError("Checkpoint changed while deployment was packaged")
        if sha256_file(gate_path) != gate_sha256:
            raise RuntimeError("V12 stage gate changed while deployment was packaged")
        if validate_gate(gate_path, checkpoint) != gate:
            raise RuntimeError("V12 gate/report lineage changed before publication")
        if sha256_file(Path(__file__).resolve()) != metadata.get(
            "v12_deployment_packager_source_sha256"
        ):
            raise RuntimeError("V12 deployment packager changed before publication")
        if _microban_runtime_source_identity(microban_repo) != (
            microban_source_identity
        ):
            raise RuntimeError("Microban runtime validator changed before publication")
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, output)
        directory_fd = os.open(
            output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return {
            "schema_version": 1,
            "packager": PACKAGER_REVISION,
            "status": "pass",
            "output": str(output),
            "output_sha256": sha256_file(output),
            "checkpoint_sha256": checkpoint_sha256,
            "stage_gate_sha256": gate_sha256,
            "completed_updates": FINAL_COMPLETED_UPDATES,
            "parity": final_parity,
            "microban_runtime_source_identity": microban_source_identity,
            "microban_runtime_validator": runtime_report,
        }
    finally:
        captured.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stage-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--microban-repo",
        type=Path,
        default=Path(__file__).resolve().parents[4] / "microban",
        help="Physical Microban repository containing the real runtime validator",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = package_v12_deployment(
        checkpoint=args.checkpoint,
        gate_path=args.stage_gate,
        output=args.output,
        microban_repo=args.microban_repo,
        force=args.force,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
