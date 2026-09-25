# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Safe ONNX export helpers for Microban's 18-DoF teleoperation policy.

MJLab's generic velocity-policy metadata describes every actuated joint on the
robot.  That is unsafe for Microban because the PICO policy deliberately leaves
``head``, ``neck_roll`` and ``neck_pitch`` to the independent HMD controller.
This module derives every action-related metadata vector from the action term's
resolved 18-joint target order instead.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
import onnx
import torch
import wandb
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx
from mjlab.rl.runner import MjlabOnPolicyRunner
from onnx.reference import ReferenceEvaluator

from mjlab_microban.tasks.microban_teleop_provenance import (
    MICROBAN_TELEOP_CANONICAL_STAGE_MODE,
    MICROBAN_TELEOP_TRAINING_PROVENANCE_KEY,
    MICROBAN_TELEOP_TRAINING_PROVENANCE_SCHEMA_VERSION,
    MICROBAN_TELEOP_TRAINING_PROVENANCE_SHA256_KEY,
    collect_training_provenance,
    inherit_initial_stage_safe_velocity_identity,
    validate_canonical_stage_critical_config,
    validate_resume_source_checkpoint_file,
    validate_training_provenance,
)

# Entity.find_joints_by_actuator_names resolves in the model's natural joint
# order.  Keep this explicit contract next to the exporter and verify it at env
# construction/export time so an XML reorder cannot silently change deployment.
MICROBAN_TELEOP_ACTION_JOINT_NAMES: tuple[str, ...] = (
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_elbow",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_elbow",
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
)

MICROBAN_HMD_JOINT_NAMES: tuple[str, ...] = (
    "head",
    "neck_roll",
    "neck_pitch",
)

# This tuple is the wire format consumed by the deployment runtime.  The order
# is just as important as the widths: concatenating the same terms in a
# different order produces a valid-shaped policy input with incorrect meaning.
MICROBAN_TELEOP_OBSERVATION_SCHEMA: tuple[tuple[str, int], ...] = (
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("joint_pos", 21),
    ("joint_vel", 21),
    ("actions", 18),
    ("command", 3),
    ("foot_target", 6),
    ("hand_target", 8),
)
MICROBAN_TELEOP_OBSERVATION_WIDTH = sum(
    width for _, width in MICROBAN_TELEOP_OBSERVATION_SCHEMA
)
MICROBAN_TELEOP_ACTION_WIDTH = len(MICROBAN_TELEOP_ACTION_JOINT_NAMES)
MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION = "9"
MICROBAN_TELEOP_NUM_STEPS_PER_ENV = 24
MICROBAN_TELEOP_ACTOR_INITIALIZATION = (
    "bounded_raw_safe_velocity_actor_only_63_to_83_zero_new_columns_v1"
)
MICROBAN_TELEOP_RECIPE_REVISION = (
    "v9_accepted_safe_velocity_bootstrap_no_walk004_prior_full_pico_curriculum_v3"
)
MICROBAN_TELEOP_OBSERVATION_SCHEMA_VERSION = "2"
MICROBAN_TELEOP_PREVIOUS_ACTION_SEMANTICS = (
    "effective_action_after_absolute_target_soft_clip_in_raw_delta_coordinates"
)
# The final curriculum stage expands simultaneous-foot lift support from the
# conservative play/early-training ceiling (12 mm) to 20 mm. Deployment
# metadata for an acceptance-backed final checkpoint must describe that final
# support even though export constructs a play environment with curriculum
# disabled. The curriculum imports this wire-contract constant too, avoiding a
# duplicated final-stage limit.
MICROBAN_TELEOP_FINAL_BOTH_FEET_LIFT_UPPER_M = 0.02
MICROBAN_TELEOP_ACTOR_LIMIT_MARGIN_RATIO = 0.05
MICROBAN_TELEOP_ACTOR_DEFAULT_EPSILON_RAD = 1.0e-4
# Float32 action storage cannot reliably invert the arctangent transform once
# its latent input is millions of times wider than the selected asymmetric
# action side.  The bounded-action contract therefore keeps the stochastic latent inside this
# independently derivable operational envelope.  The mean uses three eighths
# of that envelope and the maximum standard deviation is one sixteenth, leaving
# ten sigmas to the closest side.  An envelope escape is therefore a fail-fast
# numerical fault rather than an ordinary clipped sample.  The minimum standard
# deviation keeps inverse quantization below 2.5% of one sigma throughout that
# reachable region.
MICROBAN_TELEOP_ACTOR_LATENT_SCALE_MULTIPLIER = 1024.0
MICROBAN_TELEOP_ACTOR_LATENT_ABS_MAX = 32.0
MICROBAN_TELEOP_ACTOR_LATENT_MEAN_FRACTION = 3.0 / 8.0
MICROBAN_TELEOP_ACTOR_STD_MIN_ABS_MAX = 0.025
MICROBAN_TELEOP_ACTOR_STD_MIN_ENVELOPE_DIVISOR = 64.0
MICROBAN_TELEOP_ACTOR_STD_ABS_MAX = 1.0
MICROBAN_TELEOP_ACTOR_STD_ENVELOPE_DIVISOR = 16.0

# A fixed, deterministic input corpus makes the checkpoint -> PyTorch -> ONNX
# comparison reproducible on every export host.  The tolerances allow ordinary
# float32 kernel reordering but are tight enough to catch wrong weights,
# normalization, activation functions, or observation ordering.
TELEOP_ONNX_GATE_VERSION = "1"
TELEOP_ONNX_PARITY_SEED = 20260924
TELEOP_ONNX_PARITY_SAMPLE_COUNT = 16
TELEOP_ONNX_PARITY_ATOL = 1e-5
TELEOP_ONNX_PARITY_RTOL = 1e-4
TELEOP_FINAL_CANONICAL_STAGE_START = 18_000
TELEOP_FINAL_CANONICAL_STAGE_TARGET = 20_000

_CHECKPOINT_NAME_RE = re.compile(r"model_(?:(\d+)|(pristine))\.pt")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_BOUNDED_ACTOR_BUFFER_KEYS = (
    "distribution.lower_bound",
    "distribution.upper_bound",
    "distribution.inward_lower_bound",
    "distribution.inward_upper_bound",
    "distribution.operational_lower_bound",
    "distribution.operational_upper_bound",
    "distribution.operational_action_lower",
    "distribution.operational_action_upper",
    "distribution.mean_lower_bound",
    "distribution.mean_upper_bound",
    "distribution.min_std",
    "distribution.max_std",
)

if MICROBAN_TELEOP_OBSERVATION_WIDTH != 83:
    raise RuntimeError(
        "Microban teleop observation schema must total 83 values, got "
        f"{MICROBAN_TELEOP_OBSERVATION_WIDTH}"
    )


def guarded_teleop_actor_raw_bounds(
    default_joint_pos: list[float] | tuple[float, ...],
    soft_joint_pos_lower: list[float] | tuple[float, ...],
    soft_joint_pos_upper: list[float] | tuple[float, ...],
    action_scale: list[float] | tuple[float, ...],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Derive zero-interior actor bounds inside the physical soft limits.

    The usual five-percent target guard can place a shoulder default exactly on
    its preferred interval boundary because that pose has only one degree of
    soft-limit headroom.  A bijection requires zero to be strictly interior, so
    the guarded interval expands by at most ``1e-4 rad`` around the default on
    such a side.  This preserves essentially the complete physical guard while
    keeping the environment/runtime hard clip independent and wider.
    """

    width = MICROBAN_TELEOP_ACTION_WIDTH
    vectors = {
        "default_joint_pos": tuple(default_joint_pos),
        "soft_joint_pos_lower": tuple(soft_joint_pos_lower),
        "soft_joint_pos_upper": tuple(soft_joint_pos_upper),
        "action_scale": tuple(action_scale),
    }
    for name, values in vectors.items():
        if len(values) != width:
            raise ValueError(f"{name} must contain exactly {width} values")
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError(f"{name} must contain only finite values")

    raw_lower: list[float] = []
    raw_upper: list[float] = []
    for default, lower, upper, scale in zip(
        vectors["default_joint_pos"],
        vectors["soft_joint_pos_lower"],
        vectors["soft_joint_pos_upper"],
        vectors["action_scale"],
        strict=True,
    ):
        default = float(default)
        lower = float(lower)
        upper = float(upper)
        scale = float(scale)
        if not lower < default < upper:
            raise ValueError("Every default joint position must be inside soft limits")
        if scale <= 0.0:
            raise ValueError("Every teleop action scale must be positive")
        span = upper - lower
        epsilon = min(
            MICROBAN_TELEOP_ACTOR_DEFAULT_EPSILON_RAD,
            0.5 * (default - lower),
            0.5 * (upper - default),
        )
        target_lower = min(
            default - epsilon,
            lower + MICROBAN_TELEOP_ACTOR_LIMIT_MARGIN_RATIO * span,
        )
        target_upper = max(
            default + epsilon,
            upper - MICROBAN_TELEOP_ACTOR_LIMIT_MARGIN_RATIO * span,
        )
        bounded_lower = (target_lower - default) / scale
        bounded_upper = (target_upper - default) / scale
        if not bounded_lower < 0.0 < bounded_upper:
            raise ValueError("Every guarded actor interval must strictly contain zero")
        if not (lower < target_lower < default < target_upper < upper):
            raise ValueError("Every guarded actor target must stay inside soft limits")
        raw_lower.append(bounded_lower)
        raw_upper.append(bounded_upper)
    return tuple(raw_lower), tuple(raw_upper)


@dataclass(frozen=True)
class TeleopTrainingProvenanceIdentity:
    """Validated, deployment-relevant identity from a checkpoint manifest."""

    schema_version: int
    sha256: str
    source_tree_sha256: str
    recipe_revision: str
    actor_initialization: str
    mode: str
    canonical_stage: bool
    stage_start_boundary: int | None
    stage_target_boundary: int | None
    parent_checkpoint_sha256: str | None
    parent_gate_sha256: str | None
    resume_source_checkpoint_path: str | None
    resume_source_checkpoint_sha256: str | None
    resume_source_checkpoint_iteration: int | None

    def metadata(self) -> dict[str, str]:
        """Return the exact ONNX wire representation of this identity."""

        if self.schema_version != MICROBAN_TELEOP_TRAINING_PROVENANCE_SCHEMA_VERSION:
            raise ValueError("Unsupported ONNX training provenance schema")
        for name, value in (
            ("training provenance", self.sha256),
            ("training source tree", self.source_tree_sha256),
        ):
            if _SHA256_RE.fullmatch(value) is None:
                raise ValueError(f"ONNX {name} SHA-256 is malformed")
        if (
            self.recipe_revision != MICROBAN_TELEOP_RECIPE_REVISION
            or self.actor_initialization != MICROBAN_TELEOP_ACTOR_INITIALIZATION
        ):
            raise ValueError("ONNX training recipe/actor identity is not current")
        lineage = (
            self.stage_start_boundary,
            self.stage_target_boundary,
            self.parent_checkpoint_sha256,
            self.parent_gate_sha256,
            self.resume_source_checkpoint_path,
            self.resume_source_checkpoint_sha256,
            self.resume_source_checkpoint_iteration,
        )
        if self.canonical_stage:
            if self.mode != MICROBAN_TELEOP_CANONICAL_STAGE_MODE:
                raise ValueError("Canonical ONNX training mode is inconsistent")
            for digest in (
                self.parent_checkpoint_sha256,
                self.parent_gate_sha256,
            ):
                if digest is not None and _SHA256_RE.fullmatch(digest) is None:
                    raise ValueError("Canonical ONNX parent SHA-256 is malformed")
        elif self.mode != "generic" or any(value is not None for value in lineage):
            raise ValueError("Generic ONNX training provenance has stage lineage")

        def optional(value: object | None) -> str:
            return "none" if value is None else str(value)

        return {
            "training_provenance_schema_version": str(self.schema_version),
            "training_provenance_sha256": self.sha256,
            "training_source_tree_sha256": self.source_tree_sha256,
            "training_recipe_revision": self.recipe_revision,
            "training_actor_initialization": self.actor_initialization,
            "training_provenance_mode": self.mode,
            "canonical_training_stage": str(self.canonical_stage).lower(),
            "training_stage_start_boundary": optional(self.stage_start_boundary),
            "training_stage_target_boundary": optional(self.stage_target_boundary),
            "training_parent_checkpoint_sha256": optional(
                self.parent_checkpoint_sha256
            ),
            "training_parent_gate_sha256": optional(self.parent_gate_sha256),
            "training_resume_source_checkpoint_sha256": optional(
                self.resume_source_checkpoint_sha256
            ),
            "training_resume_source_checkpoint_iteration": optional(
                self.resume_source_checkpoint_iteration
            ),
        }


@dataclass(frozen=True)
class SafeVelocityBootstrapIdentity:
    """Deployment-relevant identity of the actor-only safe source."""

    checkpoint_sha256: str
    checkpoint_iteration: int
    recipe_revision: str
    receipt_sha256: str
    receipt_schema_version: int
    receipt_gate: str
    mapping_version: str

    def metadata(self) -> dict[str, str]:
        """Expose the pinned source and receipt hashes in the ONNX itself."""

        from mjlab_microban.tasks.microban_safe_velocity_mdp import (
            MICROBAN_SAFE_VELOCITY_RECIPE_REVISION,
        )
        from mjlab_microban.tasks.microban_teleop_bootstrap import (
            SAFE_VELOCITY_ACCEPTANCE_GATE,
            SAFE_VELOCITY_ACCEPTANCE_RECEIPT_SCHEMA_VERSION,
            SAFE_VELOCITY_ACTOR_BOOTSTRAP_MAPPING_VERSION,
        )

        if _SHA256_RE.fullmatch(self.checkpoint_sha256) is None:
            raise ValueError("ONNX safe-velocity checkpoint SHA-256 is malformed")
        if _SHA256_RE.fullmatch(self.receipt_sha256) is None:
            raise ValueError("ONNX safe-velocity receipt SHA-256 is malformed")
        if self.checkpoint_iteration < 0:
            raise ValueError("ONNX safe-velocity checkpoint iteration is invalid")
        if (
            self.recipe_revision != MICROBAN_SAFE_VELOCITY_RECIPE_REVISION
            or self.receipt_schema_version
            != SAFE_VELOCITY_ACCEPTANCE_RECEIPT_SCHEMA_VERSION
            or self.receipt_gate != SAFE_VELOCITY_ACCEPTANCE_GATE
            or self.mapping_version
            != SAFE_VELOCITY_ACTOR_BOOTSTRAP_MAPPING_VERSION
        ):
            raise ValueError("ONNX safe-velocity bootstrap identity is not current")
        return {
            "safe_velocity_source_checkpoint_sha256": self.checkpoint_sha256,
            "safe_velocity_source_checkpoint_iteration": str(
                self.checkpoint_iteration
            ),
            "safe_velocity_source_recipe_revision": self.recipe_revision,
            "safe_velocity_acceptance_receipt_sha256": self.receipt_sha256,
            "safe_velocity_acceptance_receipt_schema_version": str(
                self.receipt_schema_version
            ),
            "safe_velocity_acceptance_gate": self.receipt_gate,
            "safe_velocity_bootstrap_mapping_version": self.mapping_version,
        }


@dataclass(frozen=True)
class TeleopAcceptanceReceiptIdentity:
    """Validated final simulation-acceptance receipt bound to one checkpoint."""

    path: Path
    schema_version: int
    sha256: str
    status: str
    completed_iterations: int
    evaluator_revision: str
    acceptance_revision: str
    evaluator_source_sha256: str
    checkpoint_sha256: str
    training_provenance_sha256: str
    recipe_revision: str
    nominal_report_count: int
    moving_hmd_report_count: int

    def metadata(self) -> dict[str, str]:
        return {
            "deployment_accepted": "true",
            "acceptance_receipt_schema_version": str(self.schema_version),
            "acceptance_receipt_sha256": self.sha256,
            "acceptance_status": self.status,
            "acceptance_boundary": str(self.completed_iterations),
            "acceptance_evaluator_revision": self.evaluator_revision,
            "acceptance_revision": self.acceptance_revision,
            "acceptance_evaluator_source_sha256": self.evaluator_source_sha256,
            "acceptance_checkpoint_sha256": self.checkpoint_sha256,
            "acceptance_training_provenance_sha256": (self.training_provenance_sha256),
            "acceptance_recipe_revision": self.recipe_revision,
            "acceptance_nominal_report_count": str(self.nominal_report_count),
            "acceptance_moving_hmd_report_count": str(self.moving_hmd_report_count),
        }


def _nonaccepted_onnx_metadata() -> dict[str, str]:
    """Return explicit non-deployment values for diagnostics/automatic exports."""

    return {
        "deployment_accepted": "false",
        "acceptance_receipt_schema_version": "none",
        "acceptance_receipt_sha256": "none",
        "acceptance_status": "none",
        "acceptance_boundary": "none",
        "acceptance_evaluator_revision": "none",
        "acceptance_revision": "none",
        "acceptance_evaluator_source_sha256": "none",
        "acceptance_checkpoint_sha256": "none",
        "acceptance_training_provenance_sha256": "none",
        "acceptance_recipe_revision": "none",
        "acceptance_nominal_report_count": "0",
        "acceptance_moving_hmd_report_count": "0",
    }


@dataclass(frozen=True)
class TeleopExportProvenance:
    """Immutable identity of the checkpoint, training, and ONNX exporter."""

    checkpoint_path: Path
    checkpoint_iteration: int
    checkpoint_sha256: str
    exporter_source_commit: str
    exporter_source_dirty: str
    exporter_source_sha256: str
    training: TeleopTrainingProvenanceIdentity
    safe_velocity_bootstrap: SafeVelocityBootstrapIdentity
    acceptance: TeleopAcceptanceReceiptIdentity | None = None

    def metadata(self, *, parity_verified: bool = True) -> dict[str, str]:
        """Return unambiguous string metadata for the deployment artifact."""

        if _SHA256_RE.fullmatch(self.checkpoint_sha256) is None:
            raise ValueError("ONNX checkpoint SHA-256 is malformed")
        if self.acceptance is not None:
            acceptance = self.acceptance
            if (
                self.checkpoint_iteration != TELEOP_FINAL_CANONICAL_STAGE_TARGET - 1
                or not self.training.canonical_stage
                or self.training.stage_start_boundary
                != TELEOP_FINAL_CANONICAL_STAGE_START
                or self.training.stage_target_boundary
                != TELEOP_FINAL_CANONICAL_STAGE_TARGET
                or acceptance.schema_version != 3
                or acceptance.status != "pass"
                or acceptance.completed_iterations
                != TELEOP_FINAL_CANONICAL_STAGE_TARGET
                or acceptance.checkpoint_sha256 != self.checkpoint_sha256
                or acceptance.training_provenance_sha256 != self.training.sha256
                or acceptance.recipe_revision != self.training.recipe_revision
            ):
                raise ValueError(
                    "Accepted ONNX metadata is not bound to the final checkpoint"
                )
            for name, value in (
                ("acceptance receipt", acceptance.sha256),
                ("acceptance evaluator source", acceptance.evaluator_source_sha256),
            ):
                if _SHA256_RE.fullmatch(value) is None:
                    raise ValueError(f"ONNX {name} SHA-256 is malformed")

        iteration_semantics = (
            "pristine_accepted_safe_velocity_bootstrap_before_first_ppo_update"
            if self.checkpoint_iteration == -1
            else "zero_based_completed_update_index_from_model_filename"
        )
        metadata = {
            "checkpoint_filename": self.checkpoint_path.name,
            "checkpoint_iteration": str(self.checkpoint_iteration),
            "checkpoint_iteration_semantics": iteration_semantics,
            "checkpoint_completed_updates": str(self.checkpoint_iteration + 1),
            "checkpoint_sha256": self.checkpoint_sha256,
            "exporter_source_commit": self.exporter_source_commit,
            "exporter_source_dirty": self.exporter_source_dirty,
            "exporter_source_sha256": self.exporter_source_sha256,
            "onnx_parity_gate_version": TELEOP_ONNX_GATE_VERSION,
            "onnx_parity_runtime": "onnx.reference.ReferenceEvaluator",
            "onnx_parity_seed": str(TELEOP_ONNX_PARITY_SEED),
            "onnx_parity_sample_count": str(TELEOP_ONNX_PARITY_SAMPLE_COUNT),
            "onnx_parity_atol": str(TELEOP_ONNX_PARITY_ATOL),
            "onnx_parity_rtol": str(TELEOP_ONNX_PARITY_RTOL),
            **self.training.metadata(),
            **self.safe_velocity_bootstrap.metadata(),
            **(
                self.acceptance.metadata()
                if self.acceptance is not None
                else _nonaccepted_onnx_metadata()
            ),
        }
        if parity_verified:
            metadata["onnx_parity_verified"] = "true"
        return metadata


@dataclass(frozen=True)
class TeleopCheckpointContract:
    """Validated identity of the training semantics stored in a checkpoint."""

    version: str
    previous_action_semantics: str
    iteration: int
    common_step_counter: int
    diagnostic_legacy: bool = False
    pristine_pre_update: bool = False
    training_provenance_sha256: str | None = None
    canonical_training_stage: bool = False
    training_provenance_identity: TeleopTrainingProvenanceIdentity | None = None
    safe_velocity_bootstrap_identity: SafeVelocityBootstrapIdentity | None = None
    training_provenance: dict[str, object] | None = None
    safe_velocity_bootstrap_info: dict[str, object] | None = None


@dataclass(frozen=True)
class TeleopOnnxParityResult:
    """Numerical error observed across the deterministic parity corpus."""

    max_absolute_error: float
    max_relative_error: float
    sample_count: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_teleop_onnx_temporary_path(output_path: str | Path) -> Path:
    """Return a collision-resistant temporary path beside an ONNX output."""

    output = Path(output_path)
    return output.with_name(f".{output.name}.{uuid4().hex}.tmp")


def _git_source_state(source_path: Path) -> tuple[str, str]:
    """Return ``(HEAD, dirty)`` without making Git a deployment dependency."""

    try:
        root_result = subprocess.run(
            ["git", "-C", str(source_path.parent), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        repository = Path(root_result.stdout.strip())
        commit_result = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        status_result = subprocess.run(
            ["git", "-C", str(repository), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return "unavailable", "unknown"
    return commit_result.stdout.strip(), str(bool(status_result.stdout.strip())).lower()


def _validated_training_provenance_identity(
    manifest: object,
    digest: object,
) -> TeleopTrainingProvenanceIdentity:
    """Validate and reduce a checkpoint manifest to deployment wire fields."""

    validated = validate_training_provenance(
        manifest,
        digest,
        expected_contract_version=MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION,
        expected_recipe_revision=MICROBAN_TELEOP_RECIPE_REVISION,
        expected_actor_initialization=MICROBAN_TELEOP_ACTOR_INITIALIZATION,
    )
    assert isinstance(digest, str)  # Enforced by validate_training_provenance.

    canonical_stage = validated.get("canonical_stage")
    if not isinstance(canonical_stage, bool):
        raise TypeError(
            "Checkpoint training provenance canonical_stage must be boolean"
        )
    source = validated.get("source")
    if not isinstance(source, dict):
        raise TypeError("Checkpoint training provenance source is malformed")
    if source.get("algorithm") != "sha256(canonical_json_path_to_sha256_v1)":
        raise ValueError("Checkpoint training source algorithm is unsupported")
    source_tree_sha256 = source.get("tree_sha256")
    if (
        not isinstance(source_tree_sha256, str)
        or _SHA256_RE.fullmatch(source_tree_sha256) is None
    ):
        raise ValueError("Checkpoint training source tree SHA-256 is malformed")

    recipe_revision = validated.get("recipe_revision")
    actor_initialization = validated.get("actor_initialization")
    if not isinstance(recipe_revision, str) or not recipe_revision:
        raise TypeError("Checkpoint training recipe revision must be non-empty")
    if not isinstance(actor_initialization, str) or not actor_initialization:
        raise TypeError("Checkpoint training actor initialization must be non-empty")

    invocation = validated.get("invocation")
    if not isinstance(invocation, dict):
        raise TypeError("Checkpoint training provenance invocation is malformed")
    mode = invocation.get("mode")
    if mode not in ("generic", MICROBAN_TELEOP_CANONICAL_STAGE_MODE):
        raise ValueError("Checkpoint training provenance mode is unsupported")
    lineage = {
        "stage_start_boundary": invocation.get("stage_start_boundary"),
        "stage_target_boundary": invocation.get("stage_target_boundary"),
        "parent_checkpoint_sha256": invocation.get("parent_checkpoint_sha256"),
        "parent_gate_sha256": invocation.get("parent_gate_sha256"),
        "resume_source_checkpoint_path": invocation.get(
            "resume_source_checkpoint_path"
        ),
        "resume_source_checkpoint_sha256": invocation.get(
            "resume_source_checkpoint_sha256"
        ),
        "resume_source_checkpoint_iteration": invocation.get(
            "resume_source_checkpoint_iteration"
        ),
    }
    if canonical_stage:
        if mode != MICROBAN_TELEOP_CANONICAL_STAGE_MODE:
            raise ValueError(
                "Canonical checkpoint training provenance has a noncanonical mode"
            )
        # This additionally validates the exact 2,048-env/seed/config contract
        # and the adjacent stage/parent lineage before ONNX may claim canonical.
        validate_canonical_stage_critical_config(validated)
    else:
        if mode != "generic":
            raise ValueError(
                "Noncanonical checkpoint training provenance must use generic mode"
            )
        if any(value is not None for value in lineage.values()):
            raise ValueError(
                "Generic checkpoint training provenance cannot claim stage lineage"
            )

    return TeleopTrainingProvenanceIdentity(
        schema_version=MICROBAN_TELEOP_TRAINING_PROVENANCE_SCHEMA_VERSION,
        sha256=digest,
        source_tree_sha256=source_tree_sha256,
        recipe_revision=recipe_revision,
        actor_initialization=actor_initialization,
        mode=mode,
        canonical_stage=canonical_stage,
        stage_start_boundary=lineage["stage_start_boundary"],
        stage_target_boundary=lineage["stage_target_boundary"],
        parent_checkpoint_sha256=lineage["parent_checkpoint_sha256"],
        parent_gate_sha256=lineage["parent_gate_sha256"],
        resume_source_checkpoint_path=lineage["resume_source_checkpoint_path"],
        resume_source_checkpoint_sha256=lineage[
            "resume_source_checkpoint_sha256"
        ],
        resume_source_checkpoint_iteration=lineage[
            "resume_source_checkpoint_iteration"
        ],
    )


def _load_hashed_json(path: Path, *, description: str) -> tuple[dict, str]:
    """Read one immutable JSON snapshot and return its exact byte digest."""

    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Cannot read {description}: {path}") from exc
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{description} is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(decoded, dict):
        raise TypeError(f"{description} must contain one JSON object")
    return decoded, hashlib.sha256(payload).hexdigest()


def _validate_acceptance_report(
    report: Mapping[str, object],
    *,
    path: Path,
    expected_seed: int,
    moving_hmd: bool,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    training_provenance_sha256: str,
    evaluator_revision: str,
    acceptance_revision: str,
    expected_scenarios: tuple[str, ...],
) -> None:
    """Recheck the receipt-critical claims of one hashed evaluator report."""

    expected = {
        "schema_version": 8,
        "seed": expected_seed,
        "checkpoint_iteration": TELEOP_FINAL_CANONICAL_STAGE_TARGET - 1,
        "checkpoint_sha256": checkpoint_sha256,
        "evaluator_revision": evaluator_revision,
        "acceptance_revision": acceptance_revision,
        "steps_per_scenario": 1000,
        "settle_steps": 50,
        "status": "diagnostic" if moving_hmd else "pass",
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(
                f"Acceptance report {path.name} has invalid {key}: "
                f"{report.get(key)!r} != {value!r}"
            )
    checkpoint_value = report.get("checkpoint")
    if not isinstance(checkpoint_value, str) or (
        Path(checkpoint_value).resolve() != checkpoint_path
    ):
        raise ValueError(f"Acceptance report {path.name} checkpoint path mismatch")
    training_contract = report.get("training_contract")
    if not isinstance(training_contract, dict):
        raise TypeError(f"Acceptance report {path.name} training_contract is malformed")
    if (
        training_contract.get("version") != MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION
        or training_contract.get("training_provenance_sha256")
        != training_provenance_sha256
        or training_contract.get("canonical_training_stage") is not True
        or training_contract.get("deployment_compatible") is not True
    ):
        raise ValueError(
            f"Acceptance report {path.name} training contract is not deployable"
        )
    summary = report.get("summary")
    if not isinstance(summary, dict):
        raise TypeError(f"Acceptance report {path.name} summary is malformed")
    if (
        summary.get("hard_safety_checks_passed") is not True
        or summary.get("acceptance_checks_passed") is not True
    ):
        raise ValueError(f"Acceptance report {path.name} did not pass acceptance")
    expected_canonical_coverage = not moving_hmd
    if summary.get("canonical_coverage") is not expected_canonical_coverage:
        # Nominal final reports must be canonical; moving-HMD reports are the
        # same full suite but intentionally diagnostic/noncanonical.
        raise ValueError(f"Acceptance report {path.name} canonical coverage is invalid")
    scenario_reports = report.get("scenarios")
    if (
        not isinstance(scenario_reports, list)
        or any(not isinstance(item, dict) for item in scenario_reports)
        or tuple(item.get("name") for item in scenario_reports) != expected_scenarios
    ):
        raise ValueError(
            f"Acceptance report {path.name} canonical scenario order is invalid"
        )
    hmd_motion = report.get("hmd_neck_motion")
    nominal_environment = report.get("nominal_environment")
    if not isinstance(hmd_motion, dict) or not isinstance(nominal_environment, dict):
        raise TypeError(f"Acceptance report {path.name} HMD metadata is malformed")
    if (
        hmd_motion.get("enabled") is not moving_hmd
        or nominal_environment.get("hmd_neck_motion") is not moving_hmd
    ):
        raise ValueError(f"Acceptance report {path.name} HMD mode mismatch")
    if not moving_hmd:
        if hmd_motion.get("params") is not None:
            raise ValueError(
                f"Acceptance report {path.name} nominal HMD params must be null"
            )
        return

    params = hmd_motion.get("params")
    evidence = hmd_motion.get("evidence")
    if (
        not isinstance(params, dict)
        or params.get("neutral_probability") != 0.0
        or not isinstance(evidence, dict)
        or evidence.get("passed") is not True
    ):
        raise ValueError(
            f"Acceptance report {path.name} moving-HMD evidence did not pass"
        )
    membership = evidence.get("active_event_membership")
    if (
        not isinstance(membership, dict)
        or membership.get("all_scenarios") is not True
        or membership.get("inactive_scenarios") != []
        or membership.get("malformed_scenarios") != []
    ):
        raise ValueError(
            f"Acceptance report {path.name} moving-HMD membership is invalid"
        )
    if (
        evidence.get("minimum_required_target_peak_to_peak_rad") != 0.10
        or evidence.get("minimum_required_actual_peak_to_peak_rad") != 0.05
    ):
        raise ValueError(f"Acceptance report {path.name} moving-HMD thresholds drifted")
    expected_axes = {"head", "neck_roll", "neck_pitch"}
    for key, minimum in (
        ("minimum_observed_target_peak_to_peak_rad_by_axis", 0.10),
        ("minimum_observed_actual_peak_to_peak_rad_by_axis", 0.05),
    ):
        values = evidence.get(key)
        if (
            not isinstance(values, dict)
            or set(values) != expected_axes
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < minimum
                for value in values.values()
            )
        ):
            raise ValueError(
                f"Acceptance report {path.name} moving-HMD {key} is invalid"
            )


def validate_final_teleop_acceptance_receipt(
    receipt_path: str | Path,
    *,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    training: TeleopTrainingProvenanceIdentity,
) -> TeleopAcceptanceReceiptIdentity:
    """Validate an exact schema-3 final gate and every report it hashes."""

    from mjlab_microban.scripts import evaluate_teleop_checkpoint as evaluator

    receipt_file = Path(receipt_path).resolve()
    checkpoint = Path(checkpoint_path).resolve()
    receipt, receipt_sha256 = _load_hashed_json(
        receipt_file, description="Teleop acceptance receipt"
    )
    expected_keys = {
        "schema_version",
        "status",
        "training_contract_version",
        "recipe_revision",
        "training_provenance_sha256",
        "evaluator_revision",
        "acceptance_revision",
        "evaluator_source_sha256",
        "run_name",
        "completed_iterations",
        "checkpoint",
        "checkpoint_sha256",
        "evaluation_seeds",
        "scenarios",
        "reports",
        "report_sha256",
        "moving_hmd_reports",
        "moving_hmd_report_sha256",
    }
    if set(receipt) != expected_keys:
        raise ValueError(
            "Teleop acceptance receipt does not match the exact schema-3 fields"
        )
    if receipt.get("schema_version") != 3 or receipt.get("status") != "pass":
        raise ValueError("Teleop acceptance receipt is not a schema-3 pass")
    if (
        receipt.get("training_contract_version")
        != MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION
        or receipt.get("recipe_revision") != training.recipe_revision
        or receipt.get("training_provenance_sha256") != training.sha256
    ):
        raise ValueError("Teleop acceptance receipt training identity mismatch")
    if (
        not training.canonical_stage
        or training.stage_start_boundary != TELEOP_FINAL_CANONICAL_STAGE_START
        or training.stage_target_boundary != TELEOP_FINAL_CANONICAL_STAGE_TARGET
    ):
        raise ValueError(
            "A final acceptance receipt requires canonical training stage "
            f"{TELEOP_FINAL_CANONICAL_STAGE_START}->"
            f"{TELEOP_FINAL_CANONICAL_STAGE_TARGET}"
        )

    evaluator_revision = evaluator.TELEOP_EVALUATOR_REVISION
    acceptance_revision = evaluator.TELEOP_ACCEPTANCE_REVISION
    expected_scenarios = tuple(
        scenario.name for scenario in evaluator.default_scenarios()
    )
    evaluator_source = Path(evaluator.__file__).resolve()
    evaluator_source_sha256 = _sha256_file(evaluator_source)
    if (
        receipt.get("evaluator_revision") != evaluator_revision
        or receipt.get("acceptance_revision") != acceptance_revision
        or receipt.get("evaluator_source_sha256") != evaluator_source_sha256
    ):
        raise ValueError(
            "Teleop acceptance receipt evaluator identity does not match current code"
        )
    if (
        receipt.get("completed_iterations") != TELEOP_FINAL_CANONICAL_STAGE_TARGET
        or receipt.get("checkpoint_sha256") != checkpoint_sha256
        or receipt.get("evaluation_seeds") != [42, 43, 44]
        or receipt.get("scenarios") != "canonical"
    ):
        raise ValueError("Teleop acceptance receipt final checkpoint/suite mismatch")
    receipt_checkpoint = receipt.get("checkpoint")
    run_name = receipt.get("run_name")
    if (
        not isinstance(receipt_checkpoint, str)
        or Path(receipt_checkpoint).resolve() != checkpoint
        or not isinstance(run_name, str)
        or not run_name
        or checkpoint.parent.name != run_name
    ):
        raise ValueError("Teleop acceptance receipt checkpoint/run path mismatch")

    report_groups = (
        ("reports", "report_sha256", False),
        ("moving_hmd_reports", "moving_hmd_report_sha256", True),
    )
    all_paths: list[Path] = []
    for paths_key, digest_key, moving_hmd in report_groups:
        path_values = receipt.get(paths_key)
        digest_values = receipt.get(digest_key)
        if (
            not isinstance(path_values, list)
            or len(path_values) != 3
            or not all(isinstance(value, str) for value in path_values)
            or not isinstance(digest_values, dict)
            or set(digest_values) != set(path_values)
        ):
            raise ValueError(
                f"Teleop acceptance receipt {paths_key} digest set is malformed"
            )
        for expected_seed, path_value in zip((42, 43, 44), path_values, strict=True):
            report_path = Path(path_value).resolve()
            report, actual_sha256 = _load_hashed_json(
                report_path, description="Teleop acceptance report"
            )
            expected_sha256 = digest_values.get(path_value)
            if (
                not isinstance(expected_sha256, str)
                or _SHA256_RE.fullmatch(expected_sha256) is None
                or actual_sha256 != expected_sha256
            ):
                raise ValueError(
                    f"Teleop acceptance report SHA-256 mismatch: {report_path}"
                )
            _validate_acceptance_report(
                report,
                path=report_path,
                expected_seed=expected_seed,
                moving_hmd=moving_hmd,
                checkpoint_path=checkpoint,
                checkpoint_sha256=checkpoint_sha256,
                training_provenance_sha256=training.sha256,
                evaluator_revision=evaluator_revision,
                acceptance_revision=acceptance_revision,
                expected_scenarios=expected_scenarios,
            )
            all_paths.append(report_path)
    if len(set(all_paths)) != 6:
        raise ValueError("Teleop acceptance receipt report paths must be unique")

    return TeleopAcceptanceReceiptIdentity(
        path=receipt_file,
        schema_version=3,
        sha256=receipt_sha256,
        status="pass",
        completed_iterations=TELEOP_FINAL_CANONICAL_STAGE_TARGET,
        evaluator_revision=evaluator_revision,
        acceptance_revision=acceptance_revision,
        evaluator_source_sha256=evaluator_source_sha256,
        checkpoint_sha256=checkpoint_sha256,
        training_provenance_sha256=training.sha256,
        recipe_revision=training.recipe_revision,
        nominal_report_count=3,
        moving_hmd_report_count=3,
    )


def collect_teleop_export_provenance(
    checkpoint_path: str | Path,
    *,
    require_canonical_stage: bool = False,
    require_final_canonical_stage: bool = False,
    acceptance_receipt: str | Path | None = None,
    require_final_acceptance: bool = False,
) -> TeleopExportProvenance:
    """Capture validated checkpoint, training, and exporter source identity.

    Checkpoints must retain RSL-RL's ``model_<N>.pt`` name or the explicit
    ``model_pristine.pt`` pre-update name. Treating an arbitrary filename as an
    iteration would make later audit/reproduction ambiguous.
    """

    checkpoint = Path(checkpoint_path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    match = _CHECKPOINT_NAME_RE.fullmatch(checkpoint.name)
    if match is None:
        raise ValueError(
            "Checkpoint filename must be model_<iteration>.pt or "
            "model_pristine.pt for provenance, "
            f"got {checkpoint.name!r}"
        )
    iteration = int(match.group(1)) if match.group(1) is not None else -1
    checkpoint_sha256 = _sha256_file(checkpoint)
    contract = validate_teleop_checkpoint_contract(checkpoint)
    training = contract.training_provenance_identity
    if training is None:
        raise ValueError("Teleop ONNX export requires checkpoint training provenance")
    if (
        require_canonical_stage
        or require_final_canonical_stage
        or require_final_acceptance
        or acceptance_receipt is not None
    ) and not (training.canonical_stage):
        raise ValueError(
            "Teleop ONNX deployment export requires a canonical training stage"
        )
    if (
        require_final_canonical_stage
        or require_final_acceptance
        or acceptance_receipt is not None
    ) and (
        training.stage_start_boundary != TELEOP_FINAL_CANONICAL_STAGE_START
        or training.stage_target_boundary != TELEOP_FINAL_CANONICAL_STAGE_TARGET
    ):
        raise ValueError(
            "Teleop ONNX final deployment export requires canonical stage "
            f"{TELEOP_FINAL_CANONICAL_STAGE_START}->"
            f"{TELEOP_FINAL_CANONICAL_STAGE_TARGET}"
        )
    if _sha256_file(checkpoint) != checkpoint_sha256:
        raise RuntimeError(
            "Checkpoint changed while export provenance was being collected"
        )

    acceptance = None
    if acceptance_receipt is not None:
        acceptance = validate_final_teleop_acceptance_receipt(
            acceptance_receipt,
            checkpoint_path=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            training=training,
        )
    elif require_final_acceptance:
        raise ValueError(
            "Teleop ONNX deployment export requires a final acceptance receipt"
        )

    source_path = Path(__file__).resolve()
    source_commit, source_dirty = _git_source_state(source_path)
    return TeleopExportProvenance(
        checkpoint_path=checkpoint,
        checkpoint_iteration=iteration,
        checkpoint_sha256=checkpoint_sha256,
        exporter_source_commit=source_commit,
        exporter_source_dirty=source_dirty,
        exporter_source_sha256=_sha256_file(source_path),
        training=training,
        safe_velocity_bootstrap=(
            contract.safe_velocity_bootstrap_identity
            if contract.safe_velocity_bootstrap_identity is not None
            else (_raise_missing_safe_velocity_bootstrap_identity())
        ),
        acceptance=acceptance,
    )


def validate_safe_velocity_actor_bootstrap_info(
    info: object,
    *,
    verify_source_checkpoint: bool = True,
) -> dict[str, object]:
    """Validate the mandatory bounded/raw safe-velocity source provenance.

    Resume and export call this with source verification enabled. This makes a
    missing, changed, legacy-unbounded, or shape-compatible wrong source fail
    before checkpoint tensors are loaded or deployment metadata is emitted.
    """

    if not isinstance(info, dict):
        raise TypeError(
            "Checkpoint safe_velocity_actor_bootstrap must be a dictionary"
        )

    from mjlab_microban.tasks.microban_safe_velocity_checkpoint import (
        MICROBAN_SAFE_VELOCITY_ACTOR_TOPOLOGY,
        MICROBAN_SAFE_VELOCITY_CHECKPOINT_SCHEMA_VERSION,
        MICROBAN_SAFE_VELOCITY_RECIPE_REVISION,
        inspect_safe_velocity_checkpoint,
    )
    from mjlab_microban.tasks.microban_safe_velocity_env_cfg import (
        MICROBAN_SAFE_VELOCITY_JOINT_NAMES,
        MICROBAN_SAFE_VELOCITY_OBSERVATION_SCHEMA,
    )
    from mjlab_microban.tasks.microban_teleop_bootstrap import (
        SAFE_VELOCITY_ACCEPTANCE_GATE,
        SAFE_VELOCITY_ACCEPTANCE_RECEIPT_SCHEMA_VERSION,
        SAFE_VELOCITY_ACTOR_BOOTSTRAP_MAPPING_VERSION,
        SAFE_VELOCITY_COPIED_DISTRIBUTION_KEYS,
        SAFE_VELOCITY_NEW_TELEOP_OBSERVATION_COLUMNS,
        SAFE_VELOCITY_TARGET_ACTOR_TOPOLOGY,
        VELOCITY_TO_TELEOP_OBSERVATION_INDEX,
        validate_safe_velocity_acceptance_receipt,
    )

    exact = {
        "mapping_version": SAFE_VELOCITY_ACTOR_BOOTSTRAP_MAPPING_VERSION,
        "source_checkpoint_schema_version": (
            MICROBAN_SAFE_VELOCITY_CHECKPOINT_SCHEMA_VERSION
        ),
        "source_recipe_revision": MICROBAN_SAFE_VELOCITY_RECIPE_REVISION,
        "source_actor_topology": list(MICROBAN_SAFE_VELOCITY_ACTOR_TOPOLOGY),
        "source_observation_schema": [
            [name, width] for name, width in MICROBAN_SAFE_VELOCITY_OBSERVATION_SCHEMA
        ],
        "source_action_joint_names": list(MICROBAN_SAFE_VELOCITY_JOINT_NAMES),
        "source_actor_observation_normalization": False,
        "source_acceptance_receipt_schema_version": (
            SAFE_VELOCITY_ACCEPTANCE_RECEIPT_SCHEMA_VERSION
        ),
        "source_acceptance_gate": SAFE_VELOCITY_ACCEPTANCE_GATE,
        "target_actor_topology": list(SAFE_VELOCITY_TARGET_ACTOR_TOPOLOGY),
        "observation_index_mapping": [
            [source, target]
            for source, target in VELOCITY_TO_TELEOP_OBSERVATION_INDEX
        ],
        "new_teleop_observation_columns": list(
            SAFE_VELOCITY_NEW_TELEOP_OBSERVATION_COLUMNS
        ),
        "copied_distribution_keys": list(SAFE_VELOCITY_COPIED_DISTRIBUTION_KEYS),
        "copied_state": "actor_mlp_and_complete_bounded_distribution_only",
        "distribution_contract_equal": True,
        "actor_normalizer_copied": False,
        "critic_copied": False,
        "optimizer_copied": False,
        "iteration_copied": False,
    }
    identity_keys = {
        "source_checkpoint_path",
        "source_checkpoint_sha256",
        "source_checkpoint_iteration",
        "source_acceptance_receipt_path",
        "source_acceptance_receipt_sha256",
    }
    expected_keys = set(exact).union(identity_keys)
    if set(info) != expected_keys:
        raise ValueError(
            "Checkpoint safe-velocity bootstrap provenance key set differs from "
            "contract v9: "
            f"missing={sorted(expected_keys - set(info))}, "
            f"unexpected={sorted(set(info) - expected_keys)}"
        )
    for key, expected in exact.items():
        if info.get(key) != expected:
            raise ValueError(
                "Checkpoint safe-velocity bootstrap provenance does not match "
                f"contract v9 for {key!r}"
            )

    source_path = info.get("source_checkpoint_path")
    source_sha256 = info.get("source_checkpoint_sha256")
    source_iteration = info.get("source_checkpoint_iteration")
    receipt_path = info.get("source_acceptance_receipt_path")
    receipt_sha256 = info.get("source_acceptance_receipt_sha256")
    if not isinstance(source_path, str) or not source_path:
        raise TypeError("Safe-velocity source path must be a non-empty string")
    if not Path(source_path).is_absolute():
        raise ValueError("Safe-velocity source path must be absolute")
    if (
        not isinstance(source_sha256, str)
        or _SHA256_RE.fullmatch(source_sha256) is None
    ):
        raise ValueError("Safe-velocity source SHA-256 must be lowercase hex")
    if (
        not isinstance(source_iteration, int)
        or isinstance(source_iteration, bool)
        or source_iteration < 0
    ):
        raise ValueError("Safe-velocity source iteration must be an integer >= 0")
    if not isinstance(receipt_path, str) or not receipt_path:
        raise TypeError("Safe-velocity acceptance receipt path must be non-empty")
    if not Path(receipt_path).is_absolute():
        raise ValueError("Safe-velocity acceptance receipt path must be absolute")
    if (
        not isinstance(receipt_sha256, str)
        or _SHA256_RE.fullmatch(receipt_sha256) is None
    ):
        raise ValueError("Safe-velocity acceptance receipt SHA-256 is malformed")

    if verify_source_checkpoint:
        identity = inspect_safe_velocity_checkpoint(
            source_path, expected_sha256=source_sha256
        )
        observed = {
            "source_checkpoint_iteration": identity.iteration,
            "source_checkpoint_schema_version": identity.schema_version,
            "source_recipe_revision": identity.recipe_revision,
            "source_actor_topology": list(identity.actor_topology),
            "source_observation_schema": [
                [name, width] for name, width in identity.observation_schema
            ],
            "source_action_joint_names": list(identity.action_joint_names),
            "source_actor_observation_normalization": (
                identity.actor_obs_normalization
            ),
        }
        for key, value in observed.items():
            if info.get(key) != value:
                raise ValueError(
                    "Safe-velocity source checkpoint identity changed for "
                    f"{key!r}"
                )
        receipt = validate_safe_velocity_acceptance_receipt(
            receipt_path,
            identity,
            expected_receipt_sha256=receipt_sha256,
        )
        if (
            info.get("source_acceptance_receipt_path") != str(receipt.path)
            or info.get("source_acceptance_receipt_sha256") != receipt.sha256
        ):
            raise ValueError(
                "Safe-velocity acceptance receipt identity changed"
            )
    return info


def _raise_missing_safe_velocity_bootstrap_identity() -> SafeVelocityBootstrapIdentity:
    raise ValueError("Teleop ONNX export requires safe-velocity bootstrap identity")


def _safe_velocity_bootstrap_identity(
    info: Mapping[str, object],
) -> SafeVelocityBootstrapIdentity:
    """Project already-validated checkpoint provenance into ONNX identity."""

    return SafeVelocityBootstrapIdentity(
        checkpoint_sha256=str(info["source_checkpoint_sha256"]),
        checkpoint_iteration=int(info["source_checkpoint_iteration"]),
        recipe_revision=str(info["source_recipe_revision"]),
        receipt_sha256=str(info["source_acceptance_receipt_sha256"]),
        receipt_schema_version=int(
            info["source_acceptance_receipt_schema_version"]
        ),
        receipt_gate=str(info["source_acceptance_gate"]),
        mapping_version=str(info["mapping_version"]),
    )


def validate_velocity_actor_bootstrap_info(info: object) -> None:
    """Reject retired provenance through the current mandatory v9 validator.

    The name remains importable for older audit code, but only the v9
    ``safe_velocity_actor_bootstrap`` schema can pass.
    """

    validate_safe_velocity_actor_bootstrap_info(
        info,
        verify_source_checkpoint=True,
    )


def validate_teleop_checkpoint_contract(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device | None = "cpu",
    allow_legacy_diagnostic: bool = False,
) -> TeleopCheckpointContract:
    """Reject checkpoints trained with a different observation/action contract.

    Older checkpoints can have the same tensor widths as the current contract,
    so PyTorch can load them without an error and an exporter could otherwise
    attach current metadata to incompatible training semantics. Validation
    happens before any actor state is loaded. Contract v9 has no legacy escape
    hatch: every v1-v8, unversioned, unbounded, or normalized checkpoint is
    rejected even when ``allow_legacy_diagnostic`` is passed by old callers.
    """

    checkpoint = Path(checkpoint_path).resolve()
    match = _CHECKPOINT_NAME_RE.fullmatch(checkpoint.name)
    if match is None:
        raise ValueError(
            "Checkpoint filename must be model_<iteration>.pt or "
            "model_pristine.pt for contract "
            f"validation, got {checkpoint.name!r}"
        )
    expected_iteration = int(match.group(1)) if match.group(1) is not None else -1
    loaded = torch.load(checkpoint, map_location=map_location, weights_only=False)
    if not isinstance(loaded, dict):
        raise TypeError("Teleop checkpoint must contain a dictionary")
    iteration = loaded.get("iter")
    if (
        not isinstance(iteration, int)
        or isinstance(iteration, bool)
        or iteration != expected_iteration
    ):
        raise ValueError(
            "Checkpoint internal iteration must match its canonical filename "
            f"({iteration!r} != {expected_iteration})"
        )

    infos = loaded.get("infos")
    if not isinstance(infos, dict):
        raise TypeError("Teleop checkpoint infos must be a dictionary")
    env_state = infos.get("env_state")
    common_step_counter = (
        env_state.get("common_step_counter") if isinstance(env_state, dict) else None
    )
    if (
        not isinstance(common_step_counter, int)
        or isinstance(common_step_counter, bool)
        or common_step_counter < 0
    ):
        raise ValueError(
            "Teleop checkpoint common_step_counter must be a non-negative integer"
        )
    pristine_pre_update = infos.get("pristine_pre_update")
    if expected_iteration == -1:
        if pristine_pre_update is not True or common_step_counter != 0:
            raise ValueError(
                "model_pristine.pt requires pristine_pre_update=true and "
                "common_step_counter=0"
            )
    elif pristine_pre_update not in (None, False):
        raise ValueError(
            "Only model_pristine.pt may carry the pristine_pre_update marker"
        )

    version = infos.get("microban_teleop_training_contract_version")
    semantics = infos.get("previous_action_semantics")
    del allow_legacy_diagnostic
    if version != MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION:
        raise ValueError(
            "Checkpoint is not the current Microban teleop training contract; "
            "v1-v8 and legacy velocity checkpoints are rejected; start contract "
            "v9 from a verified bounded safe-velocity checkpoint"
        )
    if semantics != MICROBAN_TELEOP_PREVIOUS_ACTION_SEMANTICS:
        raise ValueError(
            "Checkpoint previous-action semantics do not match the current contract"
        )
    actor_initialization = infos.get("microban_teleop_actor_initialization")
    if actor_initialization != MICROBAN_TELEOP_ACTOR_INITIALIZATION:
        raise ValueError(
            "Checkpoint actor initialization does not match the current safe-source "
            "contract; start a fresh run"
        )
    recipe_revision = infos.get("microban_teleop_recipe_revision")
    if recipe_revision != MICROBAN_TELEOP_RECIPE_REVISION:
        raise ValueError(
            "Checkpoint recipe revision does not match the current training "
            "contract; start a fresh run"
        )
    training_provenance = infos.get(MICROBAN_TELEOP_TRAINING_PROVENANCE_KEY)
    training_provenance_sha256 = infos.get(
        MICROBAN_TELEOP_TRAINING_PROVENANCE_SHA256_KEY
    )
    if training_provenance is None or training_provenance_sha256 is None:
        raise ValueError(
            "Contract-v9 checkpoint requires training provenance and its SHA-256"
        )
    training_provenance_identity = _validated_training_provenance_identity(
        training_provenance,
        training_provenance_sha256,
    )
    canonical_training_stage = training_provenance_identity.canonical_stage
    if expected_iteration >= 0:
        expected_common_step_counter = (
            iteration + 1
        ) * MICROBAN_TELEOP_NUM_STEPS_PER_ENV
        if common_step_counter != expected_common_step_counter:
            raise ValueError(
                "Contract-v9 checkpoint common_step_counter must equal "
                f"(iteration + 1) * {MICROBAN_TELEOP_NUM_STEPS_PER_ENV} "
                f"({common_step_counter} != {expected_common_step_counter})"
            )
    if "velocity_actor_bootstrap" in infos:
        raise ValueError(
            "Contract v9 rejects the retired legacy velocity_actor_bootstrap field"
        )
    bootstrap_info = validate_safe_velocity_actor_bootstrap_info(
        infos.get("safe_velocity_actor_bootstrap"),
        verify_source_checkpoint=True,
    )
    return TeleopCheckpointContract(
        version=version,
        previous_action_semantics=semantics,
        iteration=iteration,
        common_step_counter=common_step_counter,
        pristine_pre_update=expected_iteration == -1,
        training_provenance_sha256=training_provenance_sha256,
        canonical_training_stage=canonical_training_stage,
        training_provenance_identity=training_provenance_identity,
        safe_velocity_bootstrap_identity=_safe_velocity_bootstrap_identity(
            bootstrap_info
        ),
        training_provenance=deepcopy(training_provenance),
        safe_velocity_bootstrap_info=deepcopy(bootstrap_info),
    )


def validate_bounded_actor_checkpoint_buffers(
    checkpoint_path: str | Path,
    expected_actor_state: Mapping[str, object],
    *,
    map_location: str | torch.device | None = "cpu",
) -> None:
    """Reject a checkpoint whose persisted transform bounds can drift.

    RSL-RL includes registered distribution buffers in ``actor_state_dict`` and
    ``load_state_dict`` would otherwise overwrite the bounds derived from the
    current robot/default/guard contract.  Compare every transform buffer
    exactly and require the learned log-standard-deviation parameter to lie in
    the current contract's projected interval before any actor state is loaded.
    """

    loaded = torch.load(
        Path(checkpoint_path).resolve(),
        map_location=map_location,
        weights_only=False,
    )
    if not isinstance(loaded, dict):
        raise TypeError("Teleop checkpoint must contain a dictionary")
    checkpoint_actor_state = loaded.get("actor_state_dict")
    if not isinstance(checkpoint_actor_state, Mapping):
        raise TypeError("Teleop checkpoint has no actor_state_dict mapping")

    validate_finite_actor_state(
        checkpoint_actor_state, context="Checkpoint actor_state_dict"
    )
    validate_finite_actor_state(
        expected_actor_state, context="Current actor state_dict"
    )

    for key in _BOUNDED_ACTOR_BUFFER_KEYS:
        candidate = checkpoint_actor_state.get(key)
        expected = expected_actor_state.get(key)
        if not isinstance(candidate, torch.Tensor):
            raise TypeError(f"Checkpoint actor is missing Tensor {key!r}")
        if not isinstance(expected, torch.Tensor):
            raise TypeError(f"Current bounded actor is missing Tensor {key!r}")
        if candidate.shape != expected.shape:
            raise ValueError(
                f"Checkpoint actor buffer {key!r} shape differs from current "
                f"contract: {tuple(candidate.shape)} != {tuple(expected.shape)}"
            )
        if candidate.dtype != expected.dtype:
            raise ValueError(
                f"Checkpoint actor buffer {key!r} dtype differs from current "
                f"contract: {candidate.dtype} != {expected.dtype}"
            )
        if not bool(torch.isfinite(expected).all().item()):
            raise ValueError(f"Current bounded actor buffer {key!r} is non-finite")
        if not bool(torch.isfinite(candidate).all().item()):
            raise ValueError(f"Checkpoint actor buffer {key!r} is non-finite")
        if not torch.equal(candidate.detach().cpu(), expected.detach().cpu()):
            raise ValueError(
                f"Checkpoint actor buffer {key!r} differs from the current "
                "guarded action-bound contract"
            )

    log_std_key = "distribution.log_std_param"
    candidate_log_std = checkpoint_actor_state.get(log_std_key)
    expected_log_std = expected_actor_state.get(log_std_key)
    expected_min_std = expected_actor_state.get("distribution.min_std")
    expected_max_std = expected_actor_state.get("distribution.max_std")
    for name, value in (
        ("checkpoint log_std_param", candidate_log_std),
        ("current log_std_param", expected_log_std),
        ("current min_std", expected_min_std),
        ("current max_std", expected_max_std),
    ):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Bounded actor is missing Tensor {name!r}")
    assert isinstance(candidate_log_std, torch.Tensor)
    assert isinstance(expected_log_std, torch.Tensor)
    assert isinstance(expected_min_std, torch.Tensor)
    assert isinstance(expected_max_std, torch.Tensor)
    if candidate_log_std.shape != expected_log_std.shape:
        raise ValueError(
            "Checkpoint actor log_std_param shape differs from current contract: "
            f"{tuple(candidate_log_std.shape)} != {tuple(expected_log_std.shape)}"
        )
    if candidate_log_std.dtype != expected_log_std.dtype:
        raise ValueError(
            "Checkpoint actor log_std_param dtype differs from current contract: "
            f"{candidate_log_std.dtype} != {expected_log_std.dtype}"
        )
    if (
        expected_min_std.shape != expected_log_std.shape
        or expected_max_std.shape != expected_log_std.shape
    ):
        raise ValueError("Current bounded actor std buffers have inconsistent shapes")
    if not bool(
        torch.all(expected_min_std > 0.0).item()
        and torch.all(expected_min_std < expected_max_std).item()
    ):
        raise ValueError("Current bounded actor std interval is invalid")
    candidate_cpu = candidate_log_std.detach().cpu()
    lower_cpu = torch.log(expected_min_std.detach().cpu())
    upper_cpu = torch.log(expected_max_std.detach().cpu())
    if not bool(
        torch.all((candidate_cpu >= lower_cpu) & (candidate_cpu <= upper_cpu)).item()
    ):
        raise ValueError(
            "Checkpoint actor distribution.log_std_param is outside the current "
            "projected std contract"
        )


def validate_finite_actor_state(
    actor_state: Mapping[str, object],
    *,
    context: str,
) -> None:
    """Fail closed if any floating/complex actor state tensor is non-finite."""

    if not isinstance(actor_state, Mapping):
        raise TypeError(f"{context} must be a mapping")
    for key, value in actor_state.items():
        if not isinstance(key, str):
            raise TypeError(f"{context} keys must be strings")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{context} entry {key!r} must be a Tensor")
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all().item()
        ):
            raise FloatingPointError(f"{context} entry {key!r} is non-finite")


def validate_finite_checkpoint_actor_state(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device | None = "cpu",
) -> None:
    """Validate every floating actor tensor before any checkpoint load."""

    loaded = torch.load(
        Path(checkpoint_path).resolve(),
        map_location=map_location,
        weights_only=False,
    )
    if not isinstance(loaded, dict):
        raise TypeError("Teleop checkpoint must contain a dictionary")
    actor_state = loaded.get("actor_state_dict")
    if not isinstance(actor_state, Mapping):
        raise TypeError("Teleop checkpoint has no actor_state_dict mapping")
    validate_finite_actor_state(actor_state, context="Checkpoint actor_state_dict")


def _representative_observation_bounds() -> tuple[np.ndarray, np.ndarray]:
    """Return conservative finite bounds in the exact 83-value schema order."""

    lower = np.asarray(
        [-4.0] * 3
        + [-1.0] * 3
        + [-math.pi] * 21
        + [-12.0] * 21
        + [-1.0] * 18
        + [-1.0, -1.0, -2.0]
        + [-0.03, -0.03, 0.0] * 2
        + [-0.08] * 6
        + [0.0, 0.0],
        dtype=np.float32,
    )
    upper = np.asarray(
        [4.0] * 3
        + [1.0] * 3
        + [math.pi] * 21
        + [12.0] * 21
        + [1.0] * 18
        + [1.0, 1.0, 2.0]
        + [0.03, 0.03, 0.05] * 2
        + [0.08] * 6
        + [1.0, 1.0],
        dtype=np.float32,
    )
    if lower.shape != (MICROBAN_TELEOP_OBSERVATION_WIDTH,) or upper.shape != (
        MICROBAN_TELEOP_OBSERVATION_WIDTH,
    ):
        raise RuntimeError("Parity observation bounds do not match the 83-value schema")
    return lower, upper


def deterministic_teleop_parity_inputs(
    *,
    seed: int = TELEOP_ONNX_PARITY_SEED,
    sample_count: int = TELEOP_ONNX_PARITY_SAMPLE_COUNT,
) -> np.ndarray:
    """Build repeatable neutral, boundary, and seeded finite observations."""

    if sample_count < 4:
        raise ValueError("Parity corpus requires at least four samples")
    lower, upper = _representative_observation_bounds()
    midpoint = (lower + upper) * np.float32(0.5)
    neutral = np.zeros(MICROBAN_TELEOP_OBSERVATION_WIDTH, dtype=np.float32)
    # A level robot observes gravity along -Z in the body frame.
    neutral[5] = -1.0
    rows = [neutral, lower, upper, midpoint]
    rng = np.random.default_rng(seed)
    if sample_count > len(rows):
        random_rows = rng.uniform(
            lower,
            upper,
            size=(sample_count - len(rows), MICROBAN_TELEOP_OBSERVATION_WIDTH),
        ).astype(np.float32)
        # The final two hand target values are activation flags, not positions.
        random_rows[:, -2:] = rng.integers(0, 2, size=(random_rows.shape[0], 2)).astype(
            np.float32
        )
        rows.extend(random_rows)
    observations = np.stack(rows, axis=0).reshape(
        sample_count, 1, MICROBAN_TELEOP_OBSERVATION_WIDTH
    )
    if not np.isfinite(observations).all():
        raise RuntimeError("Parity corpus contains non-finite observations")
    return observations


def validate_pytorch_onnx_parity(
    pytorch_policy: torch.nn.Module,
    onnx_path: str | Path,
    *,
    seed: int = TELEOP_ONNX_PARITY_SEED,
    sample_count: int = TELEOP_ONNX_PARITY_SAMPLE_COUNT,
    atol: float = TELEOP_ONNX_PARITY_ATOL,
    rtol: float = TELEOP_ONNX_PARITY_RTOL,
) -> TeleopOnnxParityResult:
    """Compare deterministic PyTorch and ONNX inference sample by sample."""

    if atol < 0.0 or rtol < 0.0:
        raise ValueError("Parity tolerances must be non-negative")
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    evaluator = ReferenceEvaluator(model)
    input_name = model.graph.input[0].name
    output_name = model.graph.output[0].name
    observations = deterministic_teleop_parity_inputs(
        seed=seed, sample_count=sample_count
    )

    pytorch_policy.to("cpu")
    pytorch_policy.eval()
    max_absolute_error = 0.0
    max_relative_error = 0.0
    with torch.inference_mode():
        for sample_index, observation in enumerate(observations):
            torch_output = pytorch_policy(torch.from_numpy(observation))
            if not isinstance(torch_output, torch.Tensor):
                raise TypeError("PyTorch export policy must return one Tensor")
            expected = torch_output.detach().cpu().numpy()
            actual = np.asarray(
                evaluator.run([output_name], {input_name: observation})[0]
            )
            expected_shape = (1, MICROBAN_TELEOP_ACTION_WIDTH)
            if expected.shape != expected_shape or actual.shape != expected_shape:
                raise ValueError(
                    "Parity output shape mismatch at sample "
                    f"{sample_index}: PyTorch {expected.shape}, ONNX {actual.shape}"
                )
            if not np.isfinite(expected).all() or not np.isfinite(actual).all():
                raise ValueError(
                    f"Non-finite policy output in parity sample {sample_index}"
                )
            absolute_error = np.abs(expected - actual)
            relative_error = absolute_error / np.maximum(np.abs(expected), atol)
            max_absolute_error = max(max_absolute_error, float(absolute_error.max()))
            max_relative_error = max(max_relative_error, float(relative_error.max()))
            if not np.allclose(expected, actual, atol=atol, rtol=rtol):
                raise ValueError(
                    "PyTorch/ONNX parity failed at sample "
                    f"{sample_index}: max_abs={float(absolute_error.max()):.8g}, "
                    f"max_rel={float(relative_error.max()):.8g}, "
                    f"atol={atol}, rtol={rtol}"
                )
    return TeleopOnnxParityResult(
        max_absolute_error=max_absolute_error,
        max_relative_error=max_relative_error,
        sample_count=sample_count,
    )


def _onnx_metadata(path: Path) -> dict[str, str]:
    model = onnx.load(str(path))
    values: dict[str, str] = {}
    for entry in model.metadata_props:
        if entry.key in values:
            raise ValueError(f"Duplicate ONNX metadata key: {entry.key}")
        values[entry.key] = entry.value
    return values


def publish_gated_teleop_onnx(
    temporary_path: str | Path,
    output_path: str | Path,
    *,
    pytorch_policy: torch.nn.Module,
    policy_metadata: Mapping[str, list | str | float],
    provenance: TeleopExportProvenance,
) -> TeleopOnnxParityResult:
    """Validate, provenance-tag, parity-check, and atomically publish an ONNX.

    The temporary and final files must share a directory, which makes
    ``Path.replace`` an atomic same-filesystem publication.  Any exception is
    raised before replacement, preserving the last known-good output.
    """

    temporary = Path(temporary_path).resolve()
    output = Path(output_path).resolve()
    if temporary.parent != output.parent:
        raise ValueError("Temporary and output ONNX must share a directory")
    if temporary == output:
        raise ValueError("Temporary and output ONNX paths must differ")

    try:
        validate_action_only_onnx(temporary)
        pending_provenance_metadata = provenance.metadata(parity_verified=False)
        verified_provenance_metadata = provenance.metadata()
        overlap = set(policy_metadata).intersection(verified_provenance_metadata)
        if overlap:
            raise ValueError(
                "Policy metadata collides with export provenance: "
                + ", ".join(sorted(overlap))
            )
        existing = _onnx_metadata(temporary)
        requested_keys = set(policy_metadata).union(verified_provenance_metadata)
        existing_overlap = set(existing).intersection(requested_keys)
        if existing_overlap:
            raise ValueError(
                "Exported ONNX already contains requested metadata keys: "
                + ", ".join(sorted(existing_overlap))
            )

        attach_metadata_to_onnx(
            str(temporary),
            {**dict(policy_metadata), **pending_provenance_metadata},
        )
        validate_action_only_onnx(temporary)
        attached = _onnx_metadata(temporary)
        for key, expected in pending_provenance_metadata.items():
            if attached.get(key) != expected:
                raise ValueError(
                    f"ONNX provenance metadata mismatch for {key}: "
                    f"{attached.get(key)!r} != {expected!r}"
                )
        if "onnx_parity_verified" in attached:
            raise ValueError("ONNX claimed parity verification before parity ran")

        # Do not assert verification in the artifact until the graph has passed.
        validate_pytorch_onnx_parity(pytorch_policy, temporary)
        attach_metadata_to_onnx(str(temporary), {"onnx_parity_verified": "true"})

        # Validate the metadata-bearing final bytes, including a second numerical
        # pass after the metadata rewrite, before atomic publication.
        validate_action_only_onnx(temporary)
        attached = _onnx_metadata(temporary)
        for key, expected in verified_provenance_metadata.items():
            if attached.get(key) != expected:
                raise ValueError(
                    f"ONNX provenance metadata mismatch for {key}: "
                    f"{attached.get(key)!r} != {expected!r}"
                )
        result = validate_pytorch_onnx_parity(pytorch_policy, temporary)

        current_sha256 = _sha256_file(provenance.checkpoint_path)
        if current_sha256 != provenance.checkpoint_sha256:
            raise RuntimeError(
                "Checkpoint changed while ONNX was being exported; refusing publication"
            )
        temporary.replace(output)
        return result
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def validate_microban_teleop_observation_contract(
    env: ManagerBasedRlEnv,
) -> None:
    """Reject an environment whose actor vector differs from the wire schema."""

    manager = env.observation_manager
    expected_names = tuple(name for name, _ in MICROBAN_TELEOP_OBSERVATION_SCHEMA)
    actor_names = tuple(manager.active_terms["actor"])
    if actor_names != expected_names:
        raise ValueError(
            "Unsafe Microban actor observation order: "
            f"resolved {actor_names}, expected {expected_names}"
        )

    if not manager.group_obs_concatenate["actor"]:
        raise ValueError("Microban actor observations must be concatenated")

    actor_term_dims = manager.group_obs_term_dim["actor"]
    actor_widths = tuple(math.prod(dims) for dims in actor_term_dims)
    expected_widths = tuple(width for _, width in MICROBAN_TELEOP_OBSERVATION_SCHEMA)
    if actor_widths != expected_widths:
        raise ValueError(
            "Unsafe Microban actor observation widths: "
            f"resolved {actor_widths}, expected {expected_widths}"
        )

    actor_group_dim = manager.group_obs_dim["actor"]
    if actor_group_dim != (MICROBAN_TELEOP_OBSERVATION_WIDTH,):
        raise ValueError(
            "Unsafe Microban actor observation shape: "
            f"resolved {actor_group_dim}, expected "
            f"({MICROBAN_TELEOP_OBSERVATION_WIDTH},)"
        )


def _as_action_vector(value: float | torch.Tensor, count: int) -> list[float]:
    if isinstance(value, torch.Tensor):
        values = (
            value[0].detach().cpu().tolist()
            if value.ndim == 2
            else value.detach().cpu().tolist()
        )
        if len(values) != count:
            raise ValueError(f"Expected {count} action values, got {len(values)}")
        return [float(item) for item in values]
    return [float(value)] * count


def _command_target_bounds(
    env: ManagerBasedRlEnv,
    *,
    canonical_final_stage: bool = False,
) -> tuple[
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
]:
    """Derive keypoint clamps from the registered command configuration.

    A normal/diagnostic export reports the environment exactly as constructed.
    An acceptance-backed final export sets ``canonical_final_stage`` because its
    play environment has no curriculum manager to materialize the final 20 mm
    simultaneous-foot lift range used by the accepted checkpoint.
    """

    foot_cfg = env.command_manager.get_term_cfg("foot_target")
    hand_cfg = env.command_manager.get_term_cfg("hand_target")

    foot_xy_lower, foot_xy_upper = map(float, foot_cfg.reach_xy_range)
    foot_z_lower, foot_z_upper = map(float, foot_cfg.lift_height_range)
    # Exact XYZ zero is the inactive-foot command, so metadata keeps a zero Z
    # lower bound even though active lift samples start at the floor-band edge.
    # The contract is disjoint: exact-zero inactive, or an active positive lift.
    foot_xyz_lower = [foot_xy_lower, foot_xy_lower, min(0.0, foot_z_lower)]
    foot_xyz_upper = [foot_xy_upper, foot_xy_upper, max(0.0, foot_z_upper)]

    both_xy_lower, both_xy_upper = map(float, foot_cfg.both_feet_reach_xy_range)
    _both_z_lower, both_z_upper = map(float, foot_cfg.both_feet_lift_height_range)
    if canonical_final_stage:
        both_z_upper = MICROBAN_TELEOP_FINAL_BOTH_FEET_LIFT_UPPER_M
    if not 0.0 <= _both_z_lower <= both_z_upper <= foot_z_upper:
        raise ValueError(
            "Simultaneous-foot lift support must remain inside single-foot support"
        )
    # Exact zero likewise represents an inactive ordinary stance for both feet;
    # it is not an active sample inside the positive floor band.
    both_xyz_lower = [both_xy_lower, both_xy_lower, 0.0]
    both_xyz_upper = [both_xy_upper, both_xy_upper, both_z_upper]

    hand_xy_lower, hand_xy_upper = map(float, hand_cfg.reach_xy_range)
    hand_z_lower, hand_z_upper = map(float, hand_cfg.reach_z_range)
    hand_xyz_lower = [hand_xy_lower, hand_xy_lower, hand_z_lower]
    hand_xyz_upper = [hand_xy_upper, hand_xy_upper, hand_z_upper]

    return (
        foot_xyz_lower * 2,
        foot_xyz_upper * 2,
        both_xyz_lower * 2,
        both_xyz_upper * 2,
        hand_xyz_lower * 2,
        hand_xyz_upper * 2,
    )


def get_microban_teleop_metadata(
    env: ManagerBasedRlEnv,
    run_path: str,
    *,
    canonical_final_stage: bool = False,
) -> dict[str, list | str | float]:
    """Return metadata aligned exactly with the policy's 18 action outputs.

    ``joint_names`` remains as a backwards-compatible alias for existing Microban
    deployment code.  New code should prefer the unambiguous
    ``action_joint_names`` field. ``canonical_final_stage`` is reserved for an
    acceptance-backed final deployment artifact; ordinary and automatic
    diagnostic exports must leave it false.
    """

    validate_microban_teleop_observation_contract(env)

    robot: Entity = env.scene["robot"]
    action = env.action_manager.get_term("joint_pos")
    if not isinstance(action, JointPositionAction):
        raise TypeError(
            "Microban teleop requires a JointPositionAction named 'joint_pos'"
        )

    action_joint_names = list(action.target_names)
    expected = list(MICROBAN_TELEOP_ACTION_JOINT_NAMES)
    if action_joint_names != expected:
        raise ValueError(
            "Unsafe Microban action order: "
            f"resolved {action_joint_names}, expected {expected}"
        )

    action_joint_ids = action.target_ids.detach().cpu().tolist()

    # Each Microban actuator targets exactly one like-named joint.  Resolve gain
    # rows in action order rather than global (21-joint) order.
    joint_name_to_ctrl_id: dict[str, int] = {}
    for actuator in robot.spec.actuators:
        joint_name_to_ctrl_id[actuator.target.split("/")[-1]] = actuator.id
    ctrl_ids = [joint_name_to_ctrl_id[name] for name in action_joint_names]

    joint_stiffness = env.sim.mj_model.actuator_gainprm[ctrl_ids, 0].tolist()
    joint_damping = (-env.sim.mj_model.actuator_biasprm[ctrl_ids, 2]).tolist()
    default_joint_pos = (
        robot.data.default_joint_pos[0, action_joint_ids].detach().cpu().tolist()
    )
    observation_default_joint_pos = (
        robot.data.default_joint_pos[0].detach().cpu().tolist()
    )
    entity_soft_limits = (
        robot.data.soft_joint_pos_limits[0, action_joint_ids].detach().cpu().tolist()
    )

    scale = torch.as_tensor(
        action.scale,
        dtype=robot.data.default_joint_pos.dtype,
        device=robot.data.default_joint_pos.device,
    )
    offset = torch.as_tensor(
        action.offset,
        dtype=robot.data.default_joint_pos.dtype,
        device=robot.data.default_joint_pos.device,
    )
    if scale.ndim == 0:
        scale = scale.expand(MICROBAN_TELEOP_ACTION_WIDTH)
    else:
        scale = scale[0]
    if offset.ndim == 0:
        offset = offset.expand(MICROBAN_TELEOP_ACTION_WIDTH)
    else:
        offset = offset[0]
    if tuple(scale.shape) != (MICROBAN_TELEOP_ACTION_WIDTH,) or not bool(
        torch.all(torch.isfinite(scale) & (scale > 0.0)).item()
    ):
        raise ValueError("Microban teleop action scale must be finite and positive")
    if tuple(offset.shape) != (MICROBAN_TELEOP_ACTION_WIDTH,) or not bool(
        torch.isfinite(offset).all().item()
    ):
        raise ValueError("Microban teleop action offset must be finite")
    if not torch.allclose(
        offset,
        robot.data.default_joint_pos[0, action_joint_ids],
        rtol=0.0,
        atol=1e-7,
    ):
        raise ValueError("Microban teleop action offset must equal its default pose")
    action_clip = torch.as_tensor(
        action._clip,
        dtype=offset.dtype,
        device=offset.device,
    )
    if action_clip.ndim == 3:
        if tuple(action_clip.shape[1:]) != (MICROBAN_TELEOP_ACTION_WIDTH, 2):
            raise ValueError("Microban teleop action clip has an invalid shape")
        if not torch.equal(action_clip, action_clip[0:1].expand_as(action_clip)):
            raise ValueError("Microban teleop action clip must be environment-invariant")
        action_clip = action_clip[0]
    if tuple(action_clip.shape) != (MICROBAN_TELEOP_ACTION_WIDTH, 2):
        raise ValueError("Microban teleop action clip has an invalid shape")
    if not bool(
        torch.isfinite(action_clip).all()
        and torch.all(action_clip[:, 0] < action_clip[:, 1])
    ):
        raise ValueError("Microban teleop action clip must be finite and ordered")
    entity_soft_limits_tensor = torch.as_tensor(
        entity_soft_limits,
        dtype=offset.dtype,
        device=offset.device,
    )
    if bool(
        torch.any(action_clip[:, 0] < entity_soft_limits_tensor[:, 0] - 1.0e-6)
        or torch.any(action_clip[:, 1] > entity_soft_limits_tensor[:, 1] + 1.0e-6)
    ):
        raise ValueError("Microban teleop action clip exceeds entity soft limits")
    soft_lower = [float(value) for value in action_clip[:, 0].tolist()]
    soft_upper = [float(value) for value in action_clip[:, 1].tolist()]
    raw_soft_lower = ((action_clip[:, 0] - offset) / scale).tolist()
    raw_soft_upper = ((action_clip[:, 1] - offset) / scale).tolist()
    if not all(
        math.isfinite(lower) and math.isfinite(upper) and lower < 0.0 < upper
        for lower, upper in zip(raw_soft_lower, raw_soft_upper, strict=True)
    ):
        raise ValueError(
            "Microban teleop defaults must lie strictly inside every raw action bound"
        )
    # Export the same float32 contract installed in the actor distribution.
    # Re-deriving the guarded interval from runtime float32 offsets can cross a
    # final rounding boundary relative to the Python constants used to build
    # the actor, even though both absolute clips describe the same joint limit.
    from mjlab_microban.tasks.microban_teleop_env_cfg import (
        microban_teleop_action_delta_bounds,
    )

    configured_actor_lower, configured_actor_upper = (
        microban_teleop_action_delta_bounds()
    )
    actor_raw_lower = tuple(
        float(value)
        for value in torch.as_tensor(
            configured_actor_lower, dtype=offset.dtype, device="cpu"
        ).tolist()
    )
    actor_raw_upper = tuple(
        float(value)
        for value in torch.as_tensor(
            configured_actor_upper, dtype=offset.dtype, device="cpu"
        ).tolist()
    )

    actor_terms = list(env.observation_manager.active_terms["actor"])
    input_schema = dict(MICROBAN_TELEOP_OBSERVATION_SCHEMA)
    (
        foot_target_lower,
        foot_target_upper,
        simultaneous_both_feet_target_lower,
        simultaneous_both_feet_target_upper,
        hand_target_lower,
        hand_target_upper,
    ) = _command_target_bounds(
        env,
        canonical_final_stage=canonical_final_stage,
    )

    return {
        "run_path": run_path,
        "policy_type": "microban_pico_hybrid_teleop",
        "microban_teleop_training_contract_version": (
            MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION
        ),
        "observation_schema_version": MICROBAN_TELEOP_OBSERVATION_SCHEMA_VERSION,
        "control_hz": float(1.0 / env.step_dt),
        "joint_names": action_joint_names,
        "action_joint_names": action_joint_names,
        "hmd_joint_names": list(MICROBAN_HMD_JOINT_NAMES),
        "observation_joint_names": list(robot.joint_names),
        "observation_default_joint_pos": observation_default_joint_pos,
        "joint_stiffness": joint_stiffness,
        "joint_damping": joint_damping,
        "gain_metadata_scope": "simulation_model_not_hardware_servo_registers",
        "default_joint_pos": default_joint_pos,
        "soft_joint_pos_lower": soft_lower,
        "soft_joint_pos_upper": soft_upper,
        "entity_soft_joint_pos_lower": [
            float(bounds[0]) for bounds in entity_soft_limits
        ],
        "entity_soft_joint_pos_upper": [
            float(bounds[1]) for bounds in entity_soft_limits
        ],
        # MJLab formats list metadata to only three decimals, which would turn
        # the shoulder's 1e-4-rad interior allowance into signed zero.  JSON
        # strings preserve the exact safety bounds through ONNX metadata.
        "raw_action_soft_lower_json": json.dumps(
            [float(value) for value in raw_soft_lower], separators=(",", ":")
        ),
        "raw_action_soft_upper_json": json.dumps(
            [float(value) for value in raw_soft_upper], separators=(",", ":")
        ),
        "actor_raw_action_lower_json": json.dumps(
            list(actor_raw_lower), separators=(",", ":")
        ),
        "actor_raw_action_upper_json": json.dumps(
            list(actor_raw_upper), separators=(",", ":")
        ),
        "actor_target_guard_margin_ratio": (MICROBAN_TELEOP_ACTOR_LIMIT_MARGIN_RATIO),
        "actor_default_interior_epsilon_rad": (
            MICROBAN_TELEOP_ACTOR_DEFAULT_EPSILON_RAD
        ),
        "command_names": list(env.command_manager.active_terms),
        "observation_names": actor_terms,
        "observation_width": MICROBAN_TELEOP_OBSERVATION_WIDTH,
        "observation_schema_json": json.dumps(input_schema, separators=(",", ":")),
        "base_ang_vel_frame": "robot_body_xyz",
        "base_ang_vel_units": "rad_s",
        "locomotion_command_order": [
            "linear_velocity_x",
            "linear_velocity_y",
            "angular_velocity_z",
        ],
        "locomotion_command_units": ["m_s", "m_s", "rad_s"],
        "locomotion_command_frame": "robot_body_forward_left_yaw_up",
        "previous_action_semantics": MICROBAN_TELEOP_PREVIOUS_ACTION_SEMANTICS,
        "action_target_semantics": "default_joint_pos_plus_raw_action_times_scale",
        "action_clip_semantics": "absolute_joint_position_radians",
        "action_distribution_semantics": (
            "diagonal_normal_ppo_latent_stored_exactly_then_per_joint_"
            "asymmetric_zero_anchored_arctan_environment_transform_with_"
            "operational_envelope_v1"
        ),
        "actor_latent_operational_scale_multiplier": (
            MICROBAN_TELEOP_ACTOR_LATENT_SCALE_MULTIPLIER
        ),
        "actor_latent_operational_abs_max": (MICROBAN_TELEOP_ACTOR_LATENT_ABS_MAX),
        "actor_latent_mean_fraction": MICROBAN_TELEOP_ACTOR_LATENT_MEAN_FRACTION,
        "actor_latent_std_min_abs_max": MICROBAN_TELEOP_ACTOR_STD_MIN_ABS_MAX,
        "actor_latent_std_min_envelope_divisor": (
            MICROBAN_TELEOP_ACTOR_STD_MIN_ENVELOPE_DIVISOR
        ),
        "actor_latent_std_abs_max": MICROBAN_TELEOP_ACTOR_STD_ABS_MAX,
        "microban_teleop_actor_initialization": (MICROBAN_TELEOP_ACTOR_INITIALIZATION),
        "microban_teleop_recipe_revision": MICROBAN_TELEOP_RECIPE_REVISION,
        "actor_latent_std_envelope_divisor": (
            MICROBAN_TELEOP_ACTOR_STD_ENVELOPE_DIVISOR
        ),
        "foot_target_semantics": (
            "left_xyz_then_right_xyz_trunk_frame_offset_from_episode_reset_"
            "reference_metres_periodic_command_resampling_does_not_move_reference"
        ),
        "foot_target_frame": "robot_trunk_xyz_forward_left_up",
        "foot_target_units": "metres",
        "foot_target_lower": foot_target_lower,
        "foot_target_upper": foot_target_upper,
        "simultaneous_both_feet_target_lower": (simultaneous_both_feet_target_lower),
        "simultaneous_both_feet_target_upper": (simultaneous_both_feet_target_upper),
        "simultaneous_both_feet_target_semantics": (
            "left_and_right_nonzero_offsets_use_conservative_stationary_"
            "training_support"
        ),
        "simultaneous_both_feet_requires_zero_twist": "true",
        "hand_target_semantics": (
            "left_xyz_then_right_xyz_then_left_right_active_flags_"
            "trunk_frame_offset_from_episode_reset_reference_metres_"
            "periodic_command_resampling_does_not_move_reference"
        ),
        "hand_target_frame": "robot_trunk_xyz_forward_left_up",
        "hand_target_units": "metres",
        "hand_target_lower": hand_target_lower,
        "hand_target_upper": hand_target_upper,
        "action_scale": _as_action_vector(action.scale, len(action_joint_names)),
    }


def _tensor_shape(value: onnx.ValueInfoProto) -> tuple[int | str, ...]:
    """Return an ONNX tensor shape without accepting unknown dimensions."""

    shape: list[int | str] = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            shape.append(dim.dim_param)
        else:
            shape.append("?")
    return tuple(shape)


def validate_action_only_onnx(
    onnx_path: str | Path,
    expected_input_width: int = MICROBAN_TELEOP_OBSERVATION_WIDTH,
    expected_action_count: int = MICROBAN_TELEOP_ACTION_WIDTH,
) -> None:
    """Reject an export that is not exactly ``[1, 83] -> [1, 18]``."""

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    if len(model.graph.input) != 1:
        raise ValueError(
            f"Expected action-only ONNX with one input, got {len(model.graph.input)}"
        )
    model_input = model.graph.input[0]
    expected_input_shape = (1, expected_input_width)
    input_shape = _tensor_shape(model_input)
    if input_shape != expected_input_shape:
        raise ValueError(
            f"Expected ONNX input shape {expected_input_shape}, got {input_shape}"
        )
    if len(model.graph.output) != 1:
        raise ValueError(
            f"Expected action-only ONNX with one output, got {len(model.graph.output)}"
        )
    output = model.graph.output[0]
    expected_output_shape = (1, expected_action_count)
    output_shape = _tensor_shape(output)
    if output_shape != expected_output_shape:
        raise ValueError(
            f"Expected ONNX output shape {expected_output_shape}, got {output_shape}"
        )


class MicrobanTeleopOnPolicyRunner(MjlabOnPolicyRunner):
    """Velocity-style PPO runner with Microban-safe automatic ONNX export."""

    env: RslRlVecEnvWrapper
    loaded_checkpoint_contract: TeleopCheckpointContract | None = None

    def __init__(
        self,
        env,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        """Construct a fresh safe-source runner or a v9-only resume runner."""

        if not hasattr(env, "clip_actions") or env.clip_actions is not None:
            raise ValueError(
                "Microban teleop contract requires wrapper clip_actions=None"
            )
        if getattr(env, "num_actions", None) != MICROBAN_TELEOP_ACTION_WIDTH:
            raise ValueError(
                "Microban teleop contract requires exactly 18 environment actions"
            )

        runner_cfg = deepcopy(train_cfg)
        retired_path = runner_cfg.pop("bootstrap_velocity_checkpoint", None)
        retired_sha256 = runner_cfg.pop(
            "bootstrap_velocity_checkpoint_sha256", None
        )
        if retired_path is not None or retired_sha256 is not None:
            raise ValueError(
                "Contract v9 rejects legacy bootstrap_velocity_checkpoint options; "
                "use the dedicated safe_velocity_checkpoint pair"
            )
        safe_source_path = runner_cfg.pop("safe_velocity_checkpoint", None)
        safe_source_sha256 = runner_cfg.pop(
            "safe_velocity_checkpoint_sha256", None
        )
        safe_acceptance_receipt = runner_cfg.pop(
            "safe_velocity_acceptance_receipt", None
        )
        checkpoint_consumer_mode = runner_cfg.pop(
            "checkpoint_consumer_mode", False
        )
        if not isinstance(checkpoint_consumer_mode, bool):
            raise TypeError("checkpoint_consumer_mode must be a boolean")
        save_pristine_checkpoint = bool(
            runner_cfg.pop("save_pristine_checkpoint", False)
        )
        fresh_source_values = (
            safe_source_path,
            safe_source_sha256,
            safe_acceptance_receipt,
        )
        if any(value is None for value in fresh_source_values) and any(
            value is not None for value in fresh_source_values
        ):
            raise ValueError(
                "Safe-velocity bootstrap requires checkpoint path, SHA-256, and "
                "a passing acceptance receipt together"
            )
        resume = bool(runner_cfg.get("resume", False))
        if checkpoint_consumer_mode and resume:
            raise ValueError(
                "Checkpoint consumer mode is actor-load only and cannot resume "
                "training"
            )
        if checkpoint_consumer_mode and safe_source_path is not None:
            raise ValueError(
                "Checkpoint consumer mode cannot accept fresh safe-velocity "
                "bootstrap options"
            )
        if checkpoint_consumer_mode and save_pristine_checkpoint:
            raise ValueError(
                "Checkpoint consumer mode cannot save a pristine checkpoint"
            )
        if checkpoint_consumer_mode and log_dir is not None:
            raise ValueError("Checkpoint consumer mode cannot have a training log_dir")
        if resume and safe_source_path is not None:
            raise ValueError(
                "Safe-velocity source options are fresh-run only; v9 resume reads "
                "the pinned source identity from the teleop checkpoint"
            )
        if not resume and not checkpoint_consumer_mode and safe_source_path is None:
            raise ValueError(
                "A fresh contract-v9 run requires --agent.safe-velocity-checkpoint "
                "--agent.safe-velocity-checkpoint-sha256, and "
                "--agent.safe-velocity-acceptance-receipt"
            )
        if resume and save_pristine_checkpoint:
            raise ValueError(
                "A pristine checkpoint can only be saved by a fresh v9 bootstrap"
            )
        if save_pristine_checkpoint and log_dir is None:
            raise ValueError("A pristine teleop checkpoint requires a runner log_dir")

        self.checkpoint_consumer_mode = checkpoint_consumer_mode
        self.teleop_training_resume = resume
        if checkpoint_consumer_mode:
            self.teleop_training_provenance = None
            self.teleop_training_provenance_sha256 = None
        else:
            (
                self.teleop_training_provenance,
                self.teleop_training_provenance_sha256,
            ) = collect_training_provenance(
                env,
                train_cfg,
                training_contract_version=MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION,
                recipe_revision=MICROBAN_TELEOP_RECIPE_REVISION,
                actor_initialization=MICROBAN_TELEOP_ACTOR_INITIALIZATION,
            )
            if resume and not self.teleop_training_provenance.get(
                "canonical_stage", False
            ):
                raise ValueError(
                    "Contract-v9 training resume requires the canonical stage "
                    "wrapper and parent gate lineage"
                )
        super().__init__(env, runner_cfg, log_dir=log_dir, device=device)
        self.loaded_checkpoint_contract = None
        self.safe_velocity_bootstrap_provenance = None
        self.safe_velocity_actor_bootstrap_info = None
        validate_finite_actor_state(
            self.alg.get_policy().state_dict(), context="Fresh actor state_dict"
        )
        if resume or checkpoint_consumer_mode:
            return

        assert safe_source_path is not None and safe_source_sha256 is not None
        assert safe_acceptance_receipt is not None

        expected_joint_names = (
            *MICROBAN_HMD_JOINT_NAMES,
            *MICROBAN_TELEOP_ACTION_JOINT_NAMES,
        )
        joint_names = tuple(self.env.unwrapped.scene["robot"].joint_names)
        if joint_names != expected_joint_names:
            raise ValueError(
                "Safe-velocity bootstrap requires teleop natural joint order "
                f"{expected_joint_names}, got {joint_names}"
            )

        from mjlab_microban.tasks.microban_teleop_bootstrap import (
            load_safe_velocity_actor_bootstrap,
            serialize_safe_velocity_actor_bootstrap_provenance,
        )

        self.safe_velocity_bootstrap_provenance = load_safe_velocity_actor_bootstrap(
            self.alg.get_policy(),
            safe_source_path,
            safe_source_sha256,
            safe_acceptance_receipt,
        )
        validate_finite_actor_state(
            self.alg.get_policy().state_dict(), context="Bootstrapped actor state_dict"
        )
        bootstrap = self.safe_velocity_bootstrap_provenance
        source = bootstrap.source
        self.safe_velocity_actor_bootstrap_info = (
            serialize_safe_velocity_actor_bootstrap_provenance(bootstrap)
        )
        validate_safe_velocity_actor_bootstrap_info(
            self.safe_velocity_actor_bootstrap_info,
            verify_source_checkpoint=True,
        )
        print(
            "Initialized contract-v9 teleop actor from bounded safe-velocity "
            f"checkpoint {source.path} (sha256={source.sha256}; actor MLP and "
            "distribution copied, critic/optimizer/iteration not copied)"
        )
        if save_pristine_checkpoint and self.gpu_global_rank == 0:
            assert log_dir is not None
            self._save_pristine_checkpoint(Path(log_dir) / "model_pristine.pt")

    def _save_pristine_checkpoint(self, path: Path) -> None:
        """Atomically save the mapped actor before the first PPO rollout/update."""

        if self.safe_velocity_actor_bootstrap_info is None:
            raise RuntimeError(
                "Pristine checkpoint requires safe-velocity bootstrap info"
            )
        validate_safe_velocity_actor_bootstrap_info(
            self.safe_velocity_actor_bootstrap_info,
            verify_source_checkpoint=True,
        )
        if self.env.unwrapped.common_step_counter != 0:
            raise ValueError("Pristine checkpoint requires common_step_counter == 0")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"Refusing to replace pristine checkpoint: {path}")
        validate_finite_actor_state(
            self.alg.get_policy().state_dict(), context="Pristine actor state_dict"
        )
        payload = self.alg.save()
        payload["iter"] = -1
        payload["infos"] = {
            "microban_teleop_training_contract_version": (
                MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION
            ),
            "previous_action_semantics": MICROBAN_TELEOP_PREVIOUS_ACTION_SEMANTICS,
            "microban_teleop_actor_initialization": (
                MICROBAN_TELEOP_ACTOR_INITIALIZATION
            ),
            "microban_teleop_recipe_revision": MICROBAN_TELEOP_RECIPE_REVISION,
            MICROBAN_TELEOP_TRAINING_PROVENANCE_KEY: deepcopy(
                self.teleop_training_provenance
            ),
            MICROBAN_TELEOP_TRAINING_PROVENANCE_SHA256_KEY: (
                self.teleop_training_provenance_sha256
            ),
            "safe_velocity_actor_bootstrap": deepcopy(
                self.safe_velocity_actor_bootstrap_info
            ),
            "env_state": {"common_step_counter": 0},
            "pristine_pre_update": True,
        }
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            torch.save(payload, temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"[INFO] Saved pristine pre-update checkpoint: {path}")

    def _require_nonlegacy_deployment_contract(self) -> None:
        """Prevent diagnostic weights from being saved or exported."""

        contract = self.loaded_checkpoint_contract
        if contract is not None and contract.diagnostic_legacy:
            raise ValueError(
                "Legacy teleop checkpoints are diagnostics-only and cannot be "
                "saved, exported, or tagged with current-contract metadata"
            )
        if contract is not None and contract.pristine_pre_update:
            raise ValueError(
                "Pristine pre-update checkpoints are safety diagnostics-only and "
                "cannot be saved or exported"
            )

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
        allow_legacy_teleop_contract: bool = False,
    ) -> dict:
        """Validate current semantics, then load at the next PPO iteration.

        RSL-RL stores the zero-based iteration that has just completed.  Its
        default loader resumes *at* that index, repeating one PPO update.  This
        task treats a checkpoint named ``model_N.pt`` as ``N + 1`` completed
        iterations and starts at ``N + 1`` instead. Only contract-v9 checkpoints
        carrying the verified safe-source identity are accepted.
        """

        if getattr(self, "safe_velocity_bootstrap_provenance", None) is not None:
            raise ValueError(
                "A safe-velocity-bootstrapped fresh runner cannot also load a "
                "teleop checkpoint; construct a v9 resume runner"
            )
        if getattr(self, "checkpoint_consumer_mode", False):
            actor_only = (
                load_cfg is not None
                and load_cfg.get("actor") is True
                and not any(
                    bool(load_cfg.get(name, False))
                    for name in ("critic", "optimizer", "iteration", "rnd")
                )
            )
            if not actor_only:
                raise ValueError(
                    "Checkpoint consumer mode requires an explicit actor-only load"
                )
        loads_iteration = load_cfg is None or bool(load_cfg.get("iteration", False))
        contract = validate_teleop_checkpoint_contract(
            path,
            map_location=map_location or "cpu",
            allow_legacy_diagnostic=allow_legacy_teleop_contract,
        )
        if getattr(self, "teleop_training_resume", False):
            self._inherit_interrupted_initial_stage_safe_source(contract)
            self._validate_canonical_resume_lineage(path, contract)
        if contract.pristine_pre_update:
            actor_only = (
                load_cfg is not None
                and load_cfg.get("actor") is True
                and not any(
                    bool(load_cfg.get(name, False))
                    for name in ("critic", "optimizer", "iteration", "rnd")
                )
            )
            if not actor_only:
                raise ValueError(
                    "Pristine pre-update checkpoints permit only an explicit "
                    "actor-only diagnostic load and cannot resume training"
                )
        validate_finite_checkpoint_actor_state(path, map_location=map_location or "cpu")
        validate_bounded_actor_checkpoint_buffers(
            path,
            self.alg.get_policy().state_dict(),
            map_location=map_location or "cpu",
        )
        self.loaded_checkpoint_contract = contract
        infos = super().load(
            path,
            load_cfg=load_cfg,
            strict=strict,
            map_location=map_location,
        )
        validate_finite_actor_state(
            self.alg.get_policy().state_dict(), context="Loaded actor state_dict"
        )
        saved_bootstrap_info = infos.get("safe_velocity_actor_bootstrap")
        validate_safe_velocity_actor_bootstrap_info(
            saved_bootstrap_info,
            verify_source_checkpoint=True,
        )
        self.safe_velocity_actor_bootstrap_info = deepcopy(saved_bootstrap_info)
        if not loads_iteration:
            return infos

        self.current_learning_iteration += 1
        env = self.env.unwrapped
        if (
            not isinstance(env.common_step_counter, int)
            or isinstance(env.common_step_counter, bool)
            or env.common_step_counter < 0
        ):
            raise ValueError(
                "Checkpoint common_step_counter must be a non-negative integer"
            )

        # Manager objects are constructed before the checkpoint is loaded.  Run
        # the resume-safe curriculum once at the restored global step so all due
        # reward weights and command ranges are active before the first rollout.
        # The constructor already performed a provisional reset using step-zero
        # config; discard it after applying the restored config.  Checkpoints do
        # not preserve simulator/RNG state, so this is not throwing away resumed
        # episode state.  In particular, a 1,000+ resume must not retain the
        # now-disabled locomotion-prior pose or its low-forward command sample.
        restored_common_step_counter = env.common_step_counter
        env.curriculum_manager.compute()
        env.reset()
        if env.common_step_counter != restored_common_step_counter:
            raise RuntimeError("Environment reset changed the restored global step")
        return infos

    def _inherit_interrupted_initial_stage_safe_source(
        self,
        contract: TeleopCheckpointContract,
    ) -> None:
        """Bind a 0->1500 resume manifest to the checkpoint's safe source."""

        current = self.teleop_training_provenance
        invocation = current.get("invocation") if isinstance(current, dict) else None
        if not isinstance(invocation, dict):
            raise TypeError("Current canonical resume invocation is malformed")
        if invocation.get("stage_start_boundary") != 0:
            return
        if (
            contract.training_provenance is None
            or contract.safe_velocity_bootstrap_info is None
        ):
            raise ValueError(
                "Interrupted initial-stage checkpoint is missing pinned safe-source "
                "provenance"
            )
        (
            self.teleop_training_provenance,
            self.teleop_training_provenance_sha256,
        ) = inherit_initial_stage_safe_velocity_identity(
            current,
            contract.training_provenance,
            contract.safe_velocity_bootstrap_info,
        )
        validate_canonical_stage_critical_config(
            self.teleop_training_provenance
        )

    def _validate_canonical_resume_lineage(
        self,
        checkpoint_path: str | Path,
        contract: TeleopCheckpointContract,
    ) -> None:
        """Bind this process's stage invocation to its exact parent checkpoint."""

        current = _validated_training_provenance_identity(
            self.teleop_training_provenance,
            self.teleop_training_provenance_sha256,
        )
        loaded = contract.training_provenance_identity
        if not current.canonical_stage or loaded is None or not loaded.canonical_stage:
            raise ValueError("A v9 training resume requires canonical stage provenance")
        resolved_checkpoint = Path(checkpoint_path).resolve()
        observed_checkpoint_sha256 = _sha256_file(resolved_checkpoint)
        resume_source_mismatches = []
        if current.resume_source_checkpoint_path != str(resolved_checkpoint):
            resume_source_mismatches.append("path")
        if current.resume_source_checkpoint_sha256 != observed_checkpoint_sha256:
            resume_source_mismatches.append("SHA-256")
        if current.resume_source_checkpoint_iteration != contract.iteration:
            resume_source_mismatches.append("iteration")
        if resume_source_mismatches:
            raise ValueError(
                "Canonical resume source does not match the checkpoint being "
                "loaded: " + ", ".join(resume_source_mismatches)
            )
        start = current.stage_start_boundary
        target = current.stage_target_boundary
        completed = contract.iteration + 1
        if start is None or target is None:
            raise ValueError("Canonical resume stage boundaries are missing")
        if completed == start:
            if current.parent_checkpoint_sha256 != _sha256_file(checkpoint_path):
                raise ValueError(
                    "Canonical resume parent checkpoint SHA-256 does not match "
                    "the loaded checkpoint"
                )
            if loaded.stage_target_boundary != start:
                raise ValueError(
                    "Canonical resume checkpoint is not the preceding stage target"
                )
            self._validate_canonical_parent_gate(
                boundary=start,
                parent_checkpoint_sha256=current.parent_checkpoint_sha256,
                parent_gate_sha256=current.parent_gate_sha256,
            )
            return
        if start < completed < target:
            lineage = (
                "stage_start_boundary",
                "stage_target_boundary",
                "parent_checkpoint_sha256",
                "parent_gate_sha256",
            )
            mismatches = [
                name
                for name in lineage
                if getattr(current, name) != getattr(loaded, name)
            ]
            if mismatches:
                raise ValueError(
                    "Interrupted canonical resume changed pinned lineage: "
                    + ", ".join(mismatches)
                )
            if start > 0:
                self._validate_canonical_parent_gate(
                    boundary=start,
                    parent_checkpoint_sha256=current.parent_checkpoint_sha256,
                    parent_gate_sha256=current.parent_gate_sha256,
                )
            return
        raise ValueError(
            "Loaded checkpoint iteration is outside the canonical resume stage"
        )

    @staticmethod
    def _validate_canonical_parent_gate(
        *,
        boundary: int,
        parent_checkpoint_sha256: str | None,
        parent_gate_sha256: str | None,
    ) -> None:
        """Revalidate the exact on-disk parent gate used to unlock a stage."""

        if (
            parent_checkpoint_sha256 is None
            or parent_gate_sha256 is None
            or _SHA256_RE.fullmatch(parent_checkpoint_sha256) is None
            or _SHA256_RE.fullmatch(parent_gate_sha256) is None
        ):
            raise ValueError("Canonical resume parent SHA-256 values are missing")
        from mjlab_microban.scripts import evaluate_teleop_checkpoint as evaluator

        gate_root = Path(__file__).resolve().parents[3] / "artifacts/teleop_v9_gates"
        matches: list[Path] = []
        for candidate in gate_root.glob(f"*_boundary_{boundary}_gate.json"):
            if not candidate.is_file() or _sha256_file(candidate) != parent_gate_sha256:
                continue
            gate, _digest = _load_hashed_json(
                candidate, description="Canonical parent gate"
            )
            parent_checkpoint = Path(gate.get("checkpoint", "")).resolve()
            if not parent_checkpoint.is_file():
                continue
            try:
                parent_contract = validate_teleop_checkpoint_contract(
                    parent_checkpoint
                )
            except (OSError, TypeError, ValueError):
                continue
            parent_training = parent_contract.training_provenance_identity
            reports = gate.get("reports")
            report_sha256 = gate.get("report_sha256")
            moving_reports = gate.get("moving_hmd_reports")
            moving_sha256 = gate.get("moving_hmd_report_sha256")
            expected_moving_count = 3 if boundary >= 12000 else 0
            identity_matches = (
                gate.get("schema_version") == 3
                and gate.get("status") == "pass"
                and gate.get("training_contract_version")
                == MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION
                and gate.get("recipe_revision") == MICROBAN_TELEOP_RECIPE_REVISION
                and gate.get("completed_iterations") == boundary
                and gate.get("checkpoint_sha256") == parent_checkpoint_sha256
                and _sha256_file(parent_checkpoint) == parent_checkpoint_sha256
                and parent_training is not None
                and gate.get("training_provenance_sha256")
                == parent_training.sha256
                and gate.get("evaluator_revision")
                == evaluator.TELEOP_EVALUATOR_REVISION
                and gate.get("acceptance_revision")
                == evaluator.TELEOP_ACCEPTANCE_REVISION
                and gate.get("evaluator_source_sha256")
                == _sha256_file(Path(evaluator.__file__))
                and gate.get("evaluation_seeds") == [42, 43, 44]
                and isinstance(reports, list)
                and len(reports) == 3
                and isinstance(report_sha256, dict)
                and set(report_sha256) == set(reports)
                and isinstance(moving_reports, list)
                and len(moving_reports) == expected_moving_count
                and isinstance(moving_sha256, dict)
                and set(moving_sha256) == set(moving_reports)
            )
            if not identity_matches:
                continue
            report_pairs = tuple(report_sha256.items()) + tuple(
                moving_sha256.items()
            )
            if all(
                Path(report).is_file()
                and _sha256_file(Path(report)) == expected
                for report, expected in report_pairs
            ):
                matches.append(candidate.resolve())
        if len(matches) != 1:
            raise ValueError(
                "Canonical resume parent gate was not found uniquely by its "
                f"pinned SHA-256 ({len(matches)} matches)"
            )

    def learn(
        self,
        num_learning_iterations: int,
        init_at_random_ep_len: bool = False,
    ) -> None:
        """Run PPO only for a training/resume construction, never a consumer."""

        if getattr(self, "checkpoint_consumer_mode", False):
            raise RuntimeError("Checkpoint consumer mode cannot train")
        return super().learn(num_learning_iterations, init_at_random_ep_len)

    def save(self, path: str, infos=None) -> None:
        if getattr(self, "checkpoint_consumer_mode", False):
            raise RuntimeError("Checkpoint consumer mode cannot save checkpoints")
        self._require_nonlegacy_deployment_contract()
        validate_finite_actor_state(
            self.alg.get_policy().state_dict(), context="Actor state_dict before save"
        )
        training_provenance = getattr(self, "teleop_training_provenance", None)
        training_provenance_sha256 = getattr(
            self, "teleop_training_provenance_sha256", None
        )
        validated_training_provenance = validate_training_provenance(
            training_provenance,
            training_provenance_sha256,
            expected_contract_version=MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION,
            expected_recipe_revision=MICROBAN_TELEOP_RECIPE_REVISION,
            expected_actor_initialization=MICROBAN_TELEOP_ACTOR_INITIALIZATION,
        )
        if validated_training_provenance.get("canonical_stage") is True:
            validate_canonical_stage_critical_config(
                validated_training_provenance
            )
            validate_resume_source_checkpoint_file(
                validated_training_provenance
            )
        bootstrap_info = getattr(self, "safe_velocity_actor_bootstrap_info", None)
        validate_safe_velocity_actor_bootstrap_info(
            bootstrap_info,
            verify_source_checkpoint=True,
        )
        contract_infos = {
            **(infos or {}),
            "microban_teleop_training_contract_version": (
                MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION
            ),
            "previous_action_semantics": (MICROBAN_TELEOP_PREVIOUS_ACTION_SEMANTICS),
            "microban_teleop_actor_initialization": (
                MICROBAN_TELEOP_ACTOR_INITIALIZATION
            ),
            "microban_teleop_recipe_revision": MICROBAN_TELEOP_RECIPE_REVISION,
            MICROBAN_TELEOP_TRAINING_PROVENANCE_KEY: deepcopy(training_provenance),
            MICROBAN_TELEOP_TRAINING_PROVENANCE_SHA256_KEY: (
                training_provenance_sha256
            ),
            "safe_velocity_actor_bootstrap": deepcopy(bootstrap_info),
        }
        contract_infos.pop("velocity_actor_bootstrap", None)
        super().save(path, contract_infos)
        policy_dir, _filename, onnx_path = self._get_export_paths(path)
        temporary_path = unique_teleop_onnx_temporary_path(onnx_path)
        temporary_filename = temporary_path.name
        try:
            provenance = collect_teleop_export_provenance(path)
            # Publish only a fully validated, metadata-complete model.  Failed
            # exports must not replace the last known-good deployment artifact.
            self.export_policy_to_onnx(str(policy_dir), temporary_filename)
            run_name: str = (
                wandb.run.name
                if self.logger.logger_type == "wandb" and wandb.run
                else "local"
            )
            metadata = get_microban_teleop_metadata(self.env.unwrapped, run_name)
            parity = publish_gated_teleop_onnx(
                temporary_path,
                onnx_path,
                pytorch_policy=self.alg.get_policy().as_onnx(verbose=False),
                policy_metadata=metadata,
                provenance=provenance,
            )
            print(
                "[INFO] Published parity-gated Microban teleop ONNX "
                f"(max_abs={parity.max_absolute_error:.3g})"
            )
            if self.logger.logger_type == "wandb" and self.cfg["upload_model"]:
                wandb.save(str(onnx_path), base_path=str(policy_dir))
        except Exception as exc:  # noqa: BLE001 - export failure must not stop PPO.
            temporary_path.unlink(missing_ok=True)
            print(
                f"[WARN] Microban teleop ONNX export failed (training continues): {exc}"
            )

    def export_policy_to_onnx(
        self,
        path: str,
        filename: str = "policy.onnx",
        verbose: bool = False,
    ) -> None:
        """Export only current-contract weights, never legacy diagnostic weights."""

        self._require_nonlegacy_deployment_contract()
        validate_finite_actor_state(
            self.alg.get_policy().state_dict(), context="Actor state_dict before export"
        )
        super().export_policy_to_onnx(path, filename, verbose)
