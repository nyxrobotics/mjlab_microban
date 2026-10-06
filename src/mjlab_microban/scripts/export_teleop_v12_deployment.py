"""Package the accepted checkpoint a PICO run ended with for Microban hardware.

The gate (teleop_v12_stage) proves the checkpoint and its evaluation
evidence; this command turns that evidence into the metadata the robot reads
(docs/policies.md: the policy contract version, the servo gain, the HOME, the
PICO schedule, the runtime guard and the startup self-test observations),
checks the final metadata-bearing graph with both ONNX implementations and
only then atomically publishes the file.  The robot side is checked when the
release is installed (its tools/validate_policies.py and its tests), not
here: the package does not depend on the robot's sources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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
    validate_teleop_v12_home_pose,
)
from mjlab_microban.policy_contract import contract_metadata
from mjlab_microban.tasks.microban_teleop_v12_runner import (
    TELEOP_V12_BOOTSTRAP_INFO_KEY,
    require_bilateral_site_order,
)
from mjlab_microban.schedules import (
    PICO_MIN_FINAL_UPDATES,
    PICO_TOTAL_UPDATES,
    pico_schedule_record,
)
from mjlab_microban.teleop_v12_safety import (
    ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_DEG,
    ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD,
    COMMANDED_TARGET_SOFT_LIMIT_EXCESS_MAX_RAD,
)

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

_FOOT_LOWER = (-0.03, -0.03, 0.0) * 2
_FOOT_UPPER = (0.03, 0.03, 0.05) * 2
_BOTH_FEET_LOWER = (-0.01, -0.01, 0.0) * 2
_BOTH_FEET_UPPER = (0.01, 0.01, 0.02) * 2
_HAND_LOWER = tuple(-value for value in MICROBAN_HAND_TARGET_WIRE_ABS_BOUND_M) * 2
_HAND_UPPER = MICROBAN_HAND_TARGET_WIRE_ABS_BOUND_M * 2
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
        "policy_contract",
        "servo_kp",
        "home_tag",
        "home_joint_hash",
        "pico_schedule_json",
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
    from mjlab_microban.scripts.teleop_v12_stage import GATE_SCHEMA_VERSION

    expected = {
        "schema_version": GATE_SCHEMA_VERSION,
        "gate": "microban_teleop_v12_stage",
        "status": "pass",
        "checkpoint_sha256": checkpoint_sha256,
    }
    mismatches = [name for name, value in expected.items() if gate.get(name) != value]
    if gate.get("tracking_profile") != FINAL_PROFILE:
        mismatches.append("tracking_profile")
    completed = gate.get("completed_updates")
    if (
        not isinstance(completed, int)
        or isinstance(completed, bool)
        or not PICO_MIN_FINAL_UPDATES <= completed <= PICO_TOTAL_UPDATES
        or gate.get("iteration") != completed - 1
    ):
        mismatches.append("completed_updates")
    if mismatches:
        raise ValueError(
            "A PICO package requires the passing gate of the checkpoint its run "
            "ended with; mismatched fields: " + ", ".join(mismatches)
        )
    if checkpoint.name != f"model_{gate['iteration']}.pt":
        raise ValueError("The packaged checkpoint must be the gate's model_<iteration>.pt")


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
) -> dict[str, list | str | float]:
    """Translate only already-validated gate evidence to the robot wire contract."""

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
        "checkpoint_iteration": str(gate["iteration"]),
        "checkpoint_iteration_semantics": (
            "zero_based_completed_update_index_from_model_filename"
        ),
        "checkpoint_completed_updates": str(gate["completed_updates"]),
        "checkpoint_sha256": checkpoint_sha256,
        "deployment_accepted": "true",
        "v12_stage_gate_schema_version": str(gate["schema_version"]),
        "v12_stage_gate_name": str(gate["gate"]),
        "v12_stage_gate_status": str(gate["status"]),
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
        **contract_metadata(),
        "pico_schedule_json": _json(pico_schedule_record()),
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


def _capture_checkpoint(checkpoint: Path, destination: Path) -> str:
    digest = hashlib.sha256()
    with checkpoint.open("rb") as source, destination.open("xb") as target:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
            target.write(chunk)
        target.flush()
        os.fsync(target.fileno())
    return digest.hexdigest()


def _reject_protected_output(output: Path, protected: Mapping[str, Path]) -> None:
    conflicts = [name for name, path in protected.items() if output == path.resolve()]
    if conflicts:
        raise ValueError(
            "Deployment output would overwrite immutable input/source: "
            + ", ".join(conflicts)
        )


# Dry runs of scripts/retrain_all_for_home.py (pipeline/dry.py) pass
# dry_run=True: their package carries this metadata key, which the robot
# refuses unless DRY_RUN_POLICY_ALLOW_ENV is "1" (set only by the dry run's
# own validator and test calls).  Without dry_run the packager refuses any
# dry-run evidence: a gate or checkpoint info key starting with "dry_run", or
# a forced-pass (DRYRUN_*) source probe receipt.
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
    force: bool = False,
    dry_run: bool = False,
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
    onnx_evidence = onnx_report.get("onnx")
    if not isinstance(onnx_evidence, Mapping):
        raise TypeError("V12 ONNX parity evidence is malformed")
    parity_tolerance = float(onnx_evidence["tolerance"])
    _reject_protected_output(
        output,
        {
            "checkpoint": checkpoint,
            "stage_gate": gate_path,
            **{f"{name}_report": path for name, path in report_paths.items()},
            "stage_gate_onnx": resolve_bootstrap_artifact_path(gate["onnx"]["path"]),
            "packager_source": Path(__file__).resolve(),
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
        if iteration != gate["iteration"]:
            raise ValueError("Captured checkpoint is not the gate's checkpoint")
        bootstrap = validate_bootstrap_provenance(
            infos.get(TELEOP_V12_BOOTSTRAP_INFO_KEY), verify_files=True
        )
        require_bilateral_site_order(infos)
        if not dry_run:
            evidence = _dry_run_evidence(
                gate=gate, infos=infos, probe_path=bootstrap.probe.path
            )
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

        # Revalidate every mutable source immediately before publication.
        if sha256_file(checkpoint) != checkpoint_sha256:
            raise RuntimeError("Checkpoint changed while deployment was packaged")
        if sha256_file(gate_path) != gate_sha256:
            raise RuntimeError("V12 stage gate changed while deployment was packaged")
        if validate_gate(gate_path, checkpoint) != gate:
            raise RuntimeError("V12 gate/report evidence changed before publication")
        if sha256_file(Path(__file__).resolve()) != metadata.get(
            "v12_deployment_packager_source_sha256"
        ):
            raise RuntimeError("V12 deployment packager changed before publication")
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
            "schema_version": 2,
            "packager": PACKAGER_REVISION,
            "status": "pass",
            "output": str(output),
            "output_sha256": sha256_file(output),
            "checkpoint_sha256": checkpoint_sha256,
            "stage_gate_sha256": gate_sha256,
            "completed_updates": gate["completed_updates"],
            "parity": final_parity,
        }
    finally:
        captured.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stage-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = package_v12_deployment(
        checkpoint=args.checkpoint,
        gate_path=args.stage_gate,
        output=args.output,
        force=args.force,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
