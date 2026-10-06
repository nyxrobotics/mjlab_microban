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
import re
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

from mjlab_microban.robot import home_contracts
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
from mjlab_microban.scripts.teleop_v12_stage import validate_gate
from mjlab_microban.tasks.mdp import MICROBAN_BILATERAL_SITE_ORDER_REVISION
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_HMD_JOINT_NAMES,
    MICROBAN_TELEOP_ACTION_JOINT_NAMES,
    MICROBAN_TELEOP_OBSERVATION_SCHEMA,
    MICROBAN_TELEOP_TARGET_FRAME,
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
    MICROBAN_TELEOP_V12_ACTION_CLIP,
    MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION,
    MICROBAN_TELEOP_V12_TRAINING_CONTRACT_VERSION,
)
from mjlab_microban.tasks.microban_teleop_v12_home_pose import (
    TELEOP_V12_HOME_POSE_INFO_KEY,
    teleop_v12_home_pose_marker,
    validate_teleop_v12_home_pose,
)
from mjlab_microban.tasks.microban_teleop_v12_runner import (
    BILATERAL_SITE_ORDER_INFO_KEY,
    TELEOP_V12_BOOTSTRAP_INFO_KEY,
    require_bilateral_site_order,
)
from mjlab_microban.teleop_v12_safety import (
    ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_DEG,
    ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD,
    COMMANDED_TARGET_SOFT_LIMIT_EXCESS_MAX_RAD,
)

FINAL_ITERATION = 14_999
FINAL_COMPLETED_UPDATES = 15_000
# HOME-bound (robot/home_contracts.py): v6 at the centered HOME, v7 at the
# forward-lean HOME (HOME-levelled target frame), "<tag>" at any other.
PACKAGER_REVISION = home_contracts.V12_PACKAGER_REVISION
RUNTIME_GUARD_FORMULA = "max(v12_absmax,source_absmax+delta_absmax)*multiplier"
RUNTIME_GUARD_MULTIPLIER = 6.0
RUNTIME_GUARD_SEMANTICS = (
    "finite_float32_then_per_joint_absmax_else_hold_previous_targets_v1"
)
# Every Microban policy commands target = HOME + raw_action * scale with no
# software clip.  The only bound is the servo's one-turn goal range, which the
# robot applies where it writes goals and which training models as an absolute
# target saturation at action_clip_lower/upper = -/+pi on all 18 body joints.
PHYSICAL_MOTOR_TARGET_GUARD_SEMANTICS = (
    "finite_target_then_servo_goal_range_saturation_pi_v3"
)
ACTION_TARGET_SEMANTICS = (
    "default_joint_pos_plus_raw_action_times_scale_saturated_at_action_clip"
)
ACTION_CLIP_SEMANTICS = (
    "absolute_target_saturated_at_servo_goal_range_pi_no_software_clip_"
    "all_body_joints_radians"
)
RUNTIME_ACTION_SEMANTICS = (
    "raw_default_plus_scale_then_servo_goal_range_saturation_v3"
)
# Every chain is bootstrapped with the corrected bilateral site order (no
# checkpoint migration); the package records this marker.
LR_ORDER_NO_MIGRATION_REVISION = "none_corrected_site_order_from_bootstrap_v1"

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
        "microban_arm_runtime_source_sha256",
        "microban_arm_contract_source_sha256",
        "microban_network_input_source_sha256",
        "microban_input_contract_source_sha256",
        "microban_runtime_entrypoint_source_sha256",
        "microban_scheduler_source_sha256",
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
        "v12_home_pose_revision",
        "v12_training_home_pose_json",
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
        "v12_lr_order_migration_revision",
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
        "v12_runtime_smoke_corpus_semantics",
        "v12_runtime_smoke_observations_json",
        "v12_runtime_smoke_observations_sha256",
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
        "action_clip_lower",
        "action_clip_upper",
        "action_distribution_semantics",
        "runtime_action_semantics",
        "physical_motor_target_guard_semantics",
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
        "microban_arm_runtime_source_sha256",
        "microban_arm_contract_source_sha256",
        "microban_network_input_source_sha256",
        "microban_input_contract_source_sha256",
        "microban_runtime_entrypoint_source_sha256",
        "microban_scheduler_source_sha256",
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


RUNTIME_SMOKE_CORPUS_SEMANTICS = (
    "final_tracking_rollout_actor_observations_first_scored_and_last_step_v1"
)
RUNTIME_SMOKE_CORPUS_MIN_ROWS = 8
RUNTIME_SMOKE_CORPUS_MAX_ROWS = 64


def _runtime_smoke_corpus(tracking: Mapping[str, Any]) -> list[list[float]]:
    """Real actor observations from the final tracking rollouts.

    The robot's startup ONNX self-test runs these through ONNX Runtime and
    compares the outputs with the runtime guard, which comes from the raw
    actions of the same rollouts.
    """

    rows = tracking.get("runtime_smoke_observations")
    if (
        not isinstance(rows, list)
        or not RUNTIME_SMOKE_CORPUS_MIN_ROWS <= len(rows) <= RUNTIME_SMOKE_CORPUS_MAX_ROWS
    ):
        raise ValueError("Final tracking report lacks a runtime smoke corpus")
    corpus: list[list[float]] = []
    for row in rows:
        if (
            not isinstance(row, list)
            or len(row) != 83
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in row
            )
        ):
            raise ValueError("Final tracking runtime smoke corpus is malformed")
        corpus.append([float(value) for value in row])
    return corpus


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_final_gate(
    gate: Mapping[str, Any],
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
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
    }
    mismatches = [name for name, value in expected.items() if gate.get(name) != value]
    if gate.get("tracking_profile") != FINAL_PROFILE:
        mismatches.append("tracking_profile")
    if mismatches:
        raise ValueError(
            "Contract-v12 deployment requires the exact accepted 15000-update "
            "gate; mismatched fields: " + ", ".join(mismatches)
        )
    if checkpoint.name != f"model_{FINAL_ITERATION}.pt":
        raise ValueError("Final contract-v12 checkpoint must be named model_14999.pt")


BOUNDARY_STAGE_GATES_SEMANTICS = (
    "packager_validated_resume_ancestor_boundary_gates_sharing_the_final_"
    "checkpoint_carried_lineage_markers_v2"
)
# A pose-release final records the gates of its 10000 boundary and 10100
# canary (where hand and then foot accuracy are first judged), at every HOME.
# Missing explicit gates are discovered through the resume chain.
POSE_RELEASE_REQUIRED_BOUNDARY_COMPLETED_UPDATES = (10_000, 10_100)
_RESUME_LOAD_RUN_PATTERN = re.compile(r"\^([A-Za-z0-9][A-Za-z0-9_.-]*)\$")
_RESUME_LOAD_CHECKPOINT_PATTERN = re.compile(r"\^(model_[0-9]+)\[\.\]pt\$")
# Gate kinds (canonical_boundary flag, checkpoint_kind) the packager records:
# canonical boundaries (3000/7000/10000) and their activation canaries
# (3100/7100/10100), which the chain resumes from like a boundary.
_BOUNDARY_GATE_KINDS = (
    (True, "canonical_boundary"),
    (False, "activation_canary"),
)
# Lineage markers every descendant checkpoint carries forward unchanged.
_BOUNDARY_GATE_SHARED_INFO_KEYS = (
    "microban_teleop_training_contract_version",
    "microban_teleop_recipe_revision",
    TELEOP_V12_BOOTSTRAP_INFO_KEY,
    TELEOP_V12_HOME_POSE_INFO_KEY,
    BILATERAL_SITE_ORDER_INFO_KEY,
)
# Markers a boundary checkpoint may carry; a descendant carries them unchanged.
_BOUNDARY_GATE_INHERITED_INFO_KEYS: tuple[str, ...] = ()


def _agent_resume_fields(params: Path) -> dict[str, str]:
    """Top-level ``resume``/``load_run``/``load_checkpoint`` of a run's agent.yaml."""

    fields: dict[str, str] = {}
    for line in params.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition(":")
        if sep and key in ("resume", "load_run", "load_checkpoint"):
            fields[key] = value.strip().strip("'\"")
    return fields


def _resume_ancestry(checkpoint: Path) -> list[Path]:
    """Checkpoints the final checkpoint's run chain resumed from, nearest first.

    Each run directory records the exact checkpoint it resumed from in
    ``params/agent.yaml`` (``load_run: ^RUN$``, ``load_checkpoint:
    ^model_N[.]pt$``).  The walk ends at a run that did not resume, or whose
    parent directory has no params (a corner-rescue seed copy).  Checkpoint
    infos carry no parent hash, so this is the only record that separates a
    true ancestor boundary from a sibling sharing the same lineage markers.
    """

    current = checkpoint.expanduser().resolve()
    seen = {current}
    chain: list[Path] = []
    while True:
        params = current.parent / "params" / "agent.yaml"
        if not params.is_file():
            return chain
        fields = _agent_resume_fields(params)
        if fields.get("resume") != "true":
            return chain
        run_match = _RESUME_LOAD_RUN_PATTERN.fullmatch(fields.get("load_run", ""))
        checkpoint_match = _RESUME_LOAD_CHECKPOINT_PATTERN.fullmatch(
            fields.get("load_checkpoint", "")
        )
        if run_match is None or checkpoint_match is None:
            raise ValueError(f"Resume source in {params} is not one exact checkpoint")
        parent = current.parent.parent / run_match[1] / f"{checkpoint_match[1]}.pt"
        if not parent.is_file():
            raise ValueError(f"Resume parent recorded in {params} is missing: {parent}")
        parent = parent.resolve()
        if parent in seen:
            raise ValueError("Resume ancestry has a cycle")
        seen.add(parent)
        chain.append(parent)
        current = parent


def _discover_ancestor_boundary_gates(
    checkpoint: Path, *, gate_root: Path, explicit: tuple[Path, ...]
) -> tuple[Path, ...]:
    """Explicit gates plus the stage gates of the required pose-release ancestors.

    An ancestor ``model_{N-1}.pt`` for a required clock N whose gate
    ``{run}_model_{N-1}_gate.json`` exists in ``gate_root`` is added unless an
    explicit gate already covers that clock.
    """

    covered: set[int] = set()
    for gate_path in explicit:
        try:
            completed = _load_json(gate_path).get("completed_updates")
        except (OSError, ValueError, TypeError):
            continue
        if isinstance(completed, int):
            covered.add(completed)
    discovered: list[Path] = []
    for ancestor in _resume_ancestry(checkpoint):
        match = re.fullmatch(r"model_([0-9]+)\.pt", ancestor.name)
        if match is None:
            continue
        completed = int(match[1]) + 1
        if (
            completed not in POSE_RELEASE_REQUIRED_BOUNDARY_COMPLETED_UPDATES
            or completed in covered
        ):
            continue
        gate_path = gate_root / f"{ancestor.parent.name}_{ancestor.stem}_gate.json"
        if gate_path.is_file():
            discovered.append(gate_path.resolve())
            covered.add(completed)
    return (*explicit, *discovered)


def _checkpoint_recipe_revision(checkpoint: Path) -> str | None:
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        return payload["infos"].get("microban_teleop_recipe_revision")
    except Exception:  # noqa: BLE001  (unreadable here is refused later by the actor loader)
        return None


def _boundary_stage_gate_lineage(
    boundary_gates: tuple[Path, ...],
    *,
    final_infos: Mapping[str, Any],
    final_checkpoint: Path,
) -> list[dict[str, Any]]:
    """Validate earlier canonical-boundary gates and summarise their profiles.

    Each gate is fully revalidated (reports, ONNX, checkpoint identity) by
    ``validate_gate``; its checkpoint must be an earlier canonical boundary
    or activation canary in the final checkpoint's resume ancestry that
    shares every carried lineage marker with it.  A pose-release final must
    list its 10000 boundary and 10100 canary.  The package then records which
    tracking profile (and allowance, if any) judged each listed boundary so
    the robot can see it.
    """

    # At the centered HOME (no required clocks) a package without boundary
    # gates needs no resume record: its release archive keeps only the final
    # run (track-centered-home-clip 5b5a9d0 packaged it that way).
    ancestry = (
        set(_resume_ancestry(final_checkpoint))
        if boundary_gates or POSE_RELEASE_REQUIRED_BOUNDARY_COMPLETED_UPDATES
        else set()
    )
    entries: list[dict[str, Any]] = []
    seen: set[int] = set()
    for gate_path in boundary_gates:
        gate_path = gate_path.expanduser().resolve()
        gate_sha256 = sha256_file(gate_path)
        snapshot = _load_json(gate_path, expected_sha256=gate_sha256)
        checkpoint = resolve_bootstrap_artifact_path(str(snapshot.get("checkpoint", "")))
        if checkpoint.expanduser().resolve() not in ancestry:
            raise ValueError(
                "Boundary stage gate checkpoint is not in the final checkpoint's "
                f"resume ancestry: {checkpoint}"
            )
        gate = validate_gate(gate_path, checkpoint)
        if gate != snapshot or sha256_file(gate_path) != gate_sha256:
            raise RuntimeError("Boundary stage gate changed while it was validated")
        completed = gate.get("completed_updates")
        kind = (gate.get("canonical_boundary"), gate.get("checkpoint_kind"))
        if (
            kind not in _BOUNDARY_GATE_KINDS
            or not isinstance(completed, int)
            or isinstance(completed, bool)
            or completed >= FINAL_COMPLETED_UPDATES
            or completed in seen
        ):
            raise ValueError(
                "Boundary stage gates must be distinct earlier canonical "
                "boundaries or activation canaries"
            )
        seen.add(completed)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if sha256_file(checkpoint) != gate["checkpoint_sha256"]:
            raise RuntimeError("Boundary checkpoint changed while it was validated")
        infos = payload["infos"]
        for key in _BOUNDARY_GATE_SHARED_INFO_KEYS:
            if infos.get(key) != final_infos.get(key):
                raise ValueError(
                    f"Boundary gate checkpoint lineage marker {key} differs "
                    "from the final checkpoint"
                )
        for key in _BOUNDARY_GATE_INHERITED_INFO_KEYS:
            if infos.get(key) is not None and infos.get(key) != final_infos.get(key):
                raise ValueError(
                    f"Final checkpoint does not carry boundary marker {key}"
                )
        entries.append(
            {
                "completed_updates": completed,
                "checkpoint_kind": gate["checkpoint_kind"],
                "iteration": gate["iteration"],
                "checkpoint_sha256": gate["checkpoint_sha256"],
                "stage_gate_sha256": gate_sha256,
                "tracking_profile": gate["tracking_profile"],
            }
        )
    if final_infos.get("microban_teleop_recipe_revision") == (
        MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION
    ):
        missing = sorted(set(POSE_RELEASE_REQUIRED_BOUNDARY_COMPLETED_UPDATES) - seen)
        if missing:
            raise ValueError(
                "A pose-release final must record its 10000 boundary and 10100 "
                f"canary gates; missing clocks {missing} (pass --boundary-gate "
                "or keep the ancestors' gates in the gate directory)"
            )
    return sorted(entries, key=lambda entry: entry["completed_updates"])


def _deployment_recipe_revision(infos: Mapping[str, Any]) -> str:
    """Recipe string the package declares to the robot (the pose-release recipe)."""

    if infos.get("microban_teleop_recipe_revision") != (
        MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION
    ):
        raise ValueError("Only a pose-release checkpoint is packaged")
    return MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION


def _onnx_parity_rule_metadata(onnx_evidence: Mapping[str, Any]) -> dict[str, str]:
    """Ship the gate's norm-wise ONNX parity rule (already gate-validated).

    The full-83 parity bound is per sample ``atol + rtol * max|expected|``
    (teleop_v12_onnx_gate), so the absolute errors alone may exceed ``atol``.
    The robot needs the rule, rtol, output magnitude and bound ratios to apply
    the same cap the stage validator does.  A legacy absolute-only report
    ships nothing and the robot keeps the plain ``atol`` cap.
    """

    if "parity_rule" not in onnx_evidence and "relative_tolerance" not in onnx_evidence:
        return {}
    from mjlab_microban.scripts.teleop_v12_onnx_gate import (
        ONNX_PARITY_RELATIVE_TOLERANCE,
        ONNX_PARITY_RULE,
    )

    if (
        onnx_evidence.get("parity_rule") != ONNX_PARITY_RULE
        or onnx_evidence.get("relative_tolerance") != ONNX_PARITY_RELATIVE_TOLERANCE
    ):
        raise ValueError("V12 ONNX parity rule evidence drifted")
    values = {
        "v12_onnx_parity_max_abs_expected_output": onnx_evidence.get(
            "maximum_absolute_expected_output"
        ),
        "v12_onnx_reference_max_bound_ratio": onnx_evidence.get(
            "reference_evaluator_maximum_bound_ratio"
        ),
        "v12_onnxruntime_cpu_max_bound_ratio": onnx_evidence.get(
            "onnxruntime_cpu_maximum_bound_ratio"
        ),
    }
    for name, value in values.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            or (name.endswith("_bound_ratio") and float(value) > 1.0)
        ):
            raise ValueError(f"V12 ONNX parity rule evidence is invalid: {name}")
    return {
        "v12_onnx_parity_rule": ONNX_PARITY_RULE,
        "v12_onnx_parity_relative_tolerance": str(ONNX_PARITY_RELATIVE_TOLERANCE),
        **{name: str(float(value)) for name, value in values.items()},
    }


def _full_precision_csv(values: list[float]) -> str:
    """CSV of shortest round-trip floats (no 3-decimal rounding)."""

    return ",".join(repr(float(value)) for value in values)


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
    boundary_stage_gates: list[Mapping[str, Any]] | None = None,
) -> dict[str, list | str | float]:
    """Translate only already-validated gate evidence to the robot wire contract.

    ``boundary_stage_gates`` (from ``_boundary_stage_gate_lineage``) adds the
    profile that judged each listed earlier boundary of the lineage.
    """

    _require_final_gate(
        gate,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
    )
    home_pose = validate_teleop_v12_home_pose(infos)
    if gate.get(TELEOP_V12_HOME_POSE_INFO_KEY) != home_pose:
        raise ValueError("Final v12 gate HOME pose does not match its checkpoint")
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

    onnx_parity_rule_metadata = _onnx_parity_rule_metadata(onnx_evidence)

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
    if (
        home_pose["joint_names"] != observation_joints
        or home_pose["joint_pos_rad"] != observation_defaults
    ):
        raise ValueError("Final v12 checkpoint HOME pose differs from exporter defaults")
    guard = _runtime_guard(tracking_envelope)
    smoke_corpus_json = _json(_runtime_smoke_corpus(tracking))
    source = bootstrap.source
    probe = bootstrap.probe
    require_bilateral_site_order(infos)
    lr_order_metadata = {
        "v12_lr_order_migration_revision": LR_ORDER_NO_MIGRATION_REVISION,
    }
    packager_source = Path(__file__).resolve()

    metadata: dict[str, list | str | float] = {
        "run_path": checkpoint.parent.name,
        "policy_type": "microban_pico_hybrid_teleop",
        "microban_teleop_training_contract_version": (
            MICROBAN_TELEOP_V12_TRAINING_CONTRACT_VERSION
        ),
        "microban_teleop_recipe_revision": _deployment_recipe_revision(infos),
        "v12_home_pose_revision": home_pose["revision"],
        "v12_training_home_pose_json": _json(home_pose),
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
        "v12_bilateral_site_order_revision": (MICROBAN_BILATERAL_SITE_ORDER_REVISION),
        **lr_order_metadata,
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
        **onnx_parity_rule_metadata,
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
        "v12_runtime_smoke_corpus_semantics": RUNTIME_SMOKE_CORPUS_SEMANTICS,
        "v12_runtime_smoke_observations_json": smoke_corpus_json,
        "v12_runtime_smoke_observations_sha256": hashlib.sha256(
            smoke_corpus_json.encode("ascii")
        ).hexdigest(),
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
        # The v12 task inherits the MJLab `robot/imu_ang_vel` built-in sensor
        # observation. That sensor reports the IMU site's local XYZ axes.
        "base_ang_vel_frame": "imu_sensor_xyz",
        "base_ang_vel_units": "rad_s",
        "locomotion_command_order": [
            "linear_velocity_x",
            "linear_velocity_y",
            "angular_velocity_z",
        ],
        "locomotion_command_units": ["m_s", "m_s", "rad_s"],
        "locomotion_command_frame": "robot_body_forward_left_yaw_up",
        "previous_action_semantics": "raw_actor_output",
        "action_target_semantics": ACTION_TARGET_SEMANTICS,
        "action_clip_semantics": ACTION_CLIP_SEMANTICS,
        # Full precision: MjLab's 3-decimal CSV would write 3.142, wider than
        # pi, which a robot checking "never wider than the servo range" rejects.
        "action_clip_lower": _full_precision_csv(
            [MICROBAN_TELEOP_V12_ACTION_CLIP[0]]
            * len(MICROBAN_TELEOP_ACTION_JOINT_NAMES)
        ),
        "action_clip_upper": _full_precision_csv(
            [MICROBAN_TELEOP_V12_ACTION_CLIP[1]]
            * len(MICROBAN_TELEOP_ACTION_JOINT_NAMES)
        ),
        "action_distribution_semantics": ("unbounded_gaussian_deterministic_mean_raw"),
        "runtime_action_semantics": RUNTIME_ACTION_SEMANTICS,
        "physical_motor_target_guard_semantics": (
            PHYSICAL_MOTOR_TARGET_GUARD_SEMANTICS
        ),
        "control_hz": 50.0,
        "foot_target_lower": list(_FOOT_LOWER),
        "foot_target_upper": list(_FOOT_UPPER),
        "foot_target_frame": MICROBAN_TELEOP_TARGET_FRAME,
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
        "hand_target_frame": MICROBAN_TELEOP_TARGET_FRAME,
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
    if boundary_stage_gates:
        metadata["v12_boundary_stage_gates_semantics"] = BOUNDARY_STAGE_GATES_SEMANTICS
        metadata["v12_boundary_stage_gates_json"] = _json(
            [dict(entry) for entry in boundary_stage_gates]
        )
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


def _validate_final_parity(
    actor: torch.nn.Module, path: Path, *, tolerance: float
) -> dict[str, float]:
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
    from mjlab_microban.scripts.teleop_v12_onnx_gate import parity_bound_ratio

    reference_max = 0.0
    runtime_max = 0.0
    bound_ratio = 0.0
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
            bound_ratio = max(
                bound_ratio,
                parity_bound_ratio(reference_actual, expected, atol=tolerance),
                parity_bound_ratio(runtime_actual, expected, atol=tolerance),
            )
    # Same per-sample atol + rtol*max|expected| rule as teleop_v12_onnx_gate.
    if (
        not math.isfinite(reference_max)
        or not math.isfinite(runtime_max)
        or not math.isfinite(bound_ratio)
        or bound_ratio > 1.0
    ):
        raise ValueError(
            "Final metadata-bearing ONNX parity failed: "
            f"reference={reference_max}, onnxruntime_cpu={runtime_max}, "
            f"tolerance={tolerance}"
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
    walk_smoke = (
        walk_fallback.get("smoke") if isinstance(walk_fallback, Mapping) else None
    )
    walk_input = (
        walk_fallback.get("input") if isinstance(walk_fallback, Mapping) else None
    )
    walk_output = (
        walk_fallback.get("output") if isinstance(walk_fallback, Mapping) else None
    )
    expected_walk_path = str((microban_repo / "src" / "agents" / "walk.onnx").resolve())
    maximum_walk_output = (
        walk_smoke.get("maximum_absolute_output")
        if isinstance(walk_smoke, Mapping)
        else None
    )
    if (
        report.get("status") != "pass"
        or report.get("policy") != str(path.resolve())
        or metadata.get("base_ang_vel_frame") != "imu_sensor_xyz"
        or metadata.get("foot_target_frame") != MICROBAN_TELEOP_TARGET_FRAME
        or metadata.get("hand_target_frame") != MICROBAN_TELEOP_TARGET_FRAME
        or metadata.get("v12_home_pose_revision")
        != teleop_v12_home_pose_marker()["revision"]
        or metadata.get("v12_training_home_pose_json")
        != _json(teleop_v12_home_pose_marker())
        or metadata.get("runtime_raw_action_guard_semantics")
        != RUNTIME_GUARD_SEMANTICS
        or metadata.get("physical_motor_target_guard_semantics")
        != PHYSICAL_MOTOR_TARGET_GUARD_SEMANTICS
        or report.get("physical_motor_target_guard_semantics")
        != PHYSICAL_MOTOR_TARGET_GUARD_SEMANTICS
        or not isinstance(report.get("v12_raw_action_guard"), Mapping)
        or report["v12_raw_action_guard"].get("semantics")
        != RUNTIME_GUARD_SEMANTICS
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
        or walk_input != {"name": "obs", "shape": [1, 63], "type": "tensor(float)"}
        or walk_output != {"name": "actions", "shape": [1, 18], "type": "tensor(float)"}
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
        "microban_arm_runtime_source_sha256": (
            repo / "src" / "moves" / "pico_arms.py"
        ),
        "microban_arm_contract_source_sha256": repo / "src" / "pico_arm_contract.py",
        "microban_network_input_source_sha256": (
            repo / "src" / "input" / "network_input.py"
        ),
        "microban_input_contract_source_sha256": (
            repo / "src" / "input" / "input_source.py"
        ),
        "microban_runtime_entrypoint_source_sha256": repo / "src" / "main.py",
        "microban_scheduler_source_sha256": repo / "src" / "scheduler.py",
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


# Dry runs of scripts/retrain_all_for_home.py (scripts/home_pipeline/
# dry_run_tools.py package) pass dry_run=True: their package carries this
# metadata key, which the robot runtime refuses unless DRY_RUN_POLICY_ALLOW_ENV
# is "1" (set only by the dry run's own validator and test calls).  Without
# dry_run the packager refuses any dry-run evidence: a gate or checkpoint info
# key starting with "dry_run", or a forced-pass (DRYRUN_*) source probe receipt.
DRY_RUN_METADATA_KEY = "dry_run_not_deployable"
DRY_RUN_POLICY_ALLOW_ENV = "MICROBAN_ALLOW_DRYRUN_POLICY"


def _dry_run_evidence(
    *, gate: Mapping[str, Any], infos: Mapping[str, Any], probe_path: str
) -> list[str]:
    found = [f"gate {key}" for key in gate if str(key).startswith("dry_run")]
    found += [f"checkpoint info {key}" for key in infos if str(key).startswith("dry_run")]
    probe = resolve_bootstrap_artifact_path(probe_path)
    if probe.name.startswith("DRYRUN_"):
        found.append(f"source probe receipt {probe.name}")
    else:
        try:
            receipt = json.loads(probe.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            receipt = {}
        if isinstance(receipt, Mapping):
            found += [
                f"source probe receipt {key}"
                for key in receipt
                if str(key).startswith("dry_run")
            ]
    return found


def package_v12_deployment(
    *,
    checkpoint: Path,
    gate_path: Path,
    output: Path,
    microban_repo: Path,
    force: bool = False,
    boundary_gates: tuple[Path, ...] = (),
    dry_run: bool = False,
) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve()
    boundary_gates = tuple(path.expanduser().resolve() for path in boundary_gates)
    gate_path = gate_path.expanduser().resolve()
    if POSE_RELEASE_REQUIRED_BOUNDARY_COMPLETED_UPDATES and _checkpoint_recipe_revision(
        checkpoint
    ) == (MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION):
        # Discovery only; the requirement itself is enforced fail-closed on the
        # captured checkpoint's infos in _boundary_stage_gate_lineage.
        boundary_gates = _discover_ancestor_boundary_gates(
            checkpoint, gate_root=gate_path.parent, explicit=boundary_gates
        )
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
    onnx_evidence = onnx_report.get("onnx")
    if not isinstance(onnx_evidence, Mapping):
        raise TypeError("V12 ONNX parity evidence is malformed")
    parity_tolerance = float(onnx_evidence["tolerance"])
    microban_source_identity = _microban_runtime_source_identity(microban_repo)
    microban_repo_resolved = microban_repo.expanduser().resolve()
    _reject_protected_output(
        output,
        {
            "checkpoint": checkpoint,
            "stage_gate": gate_path,
            **{
                f"boundary_stage_gate_{index}": path
                for index, path in enumerate(boundary_gates)
            },
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
            "microban_arm_runtime": (
                microban_repo_resolved / "src" / "moves" / "pico_arms.py"
            ),
            "microban_arm_contract": (
                microban_repo_resolved / "src" / "pico_arm_contract.py"
            ),
            "microban_network_input": (
                microban_repo_resolved / "src" / "input" / "network_input.py"
            ),
            "microban_input_contract": (
                microban_repo_resolved / "src" / "input" / "input_source.py"
            ),
            "microban_runtime_entrypoint": microban_repo_resolved / "src" / "main.py",
            "microban_scheduler": microban_repo_resolved / "src" / "scheduler.py",
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
        require_bilateral_site_order(infos)
        boundary_stage_gates = _boundary_stage_gate_lineage(
            boundary_gates, final_infos=infos, final_checkpoint=checkpoint
        )
        if not dry_run:
            evidence = _dry_run_evidence(
                gate=gate, infos=infos, probe_path=bootstrap.probe.path
            )
            for boundary_gate in boundary_gates:
                evidence += [
                    f"boundary gate {boundary_gate.name} {key}"
                    for key in _load_json(boundary_gate)
                    if str(key).startswith("dry_run")
                ]
            if evidence:
                raise ValueError(
                    "Refusing to package dry-run evidence as deployable: "
                    + ", ".join(evidence)
                )

        _export_onnx_atomic(actor, temporary)
        _validate_graph_contract(temporary)
        initial_parity = _validate_final_parity(
            actor, temporary, tolerance=parity_tolerance
        )
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
            boundary_stage_gates=boundary_stage_gates,
        )
        if dry_run:
            metadata[DRY_RUN_METADATA_KEY] = "true"
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
        final_parity = _validate_final_parity(
            actor, temporary, tolerance=parity_tolerance
        )
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
        if (
            _boundary_stage_gate_lineage(
                boundary_gates, final_infos=infos, final_checkpoint=checkpoint
            )
            != boundary_stage_gates
        ):
            raise RuntimeError("Boundary stage gates changed before publication")
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
            "boundary_stage_gates": boundary_stage_gates,
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
    parser.add_argument(
        "--boundary-gate",
        type=Path,
        action="append",
        default=[],
        help=(
            "Earlier canonical-boundary or activation-canary stage gate in the "
            "final checkpoint's resume ancestry; repeatable.  Each is revalidated "
            "and its tracking profile is recorded in the package metadata.  A "
            "pose-release final requires its 10000 and 10100 gates; ones not "
            "given are discovered through the resume chain in the gate directory."
        ),
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
        boundary_gates=tuple(args.boundary_gate),
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
