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
MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION = "3"
MICROBAN_TELEOP_OBSERVATION_SCHEMA_VERSION = "2"
MICROBAN_TELEOP_PREVIOUS_ACTION_SEMANTICS = (
    "effective_action_after_absolute_target_soft_clip_in_raw_delta_coordinates"
)

# A fixed, deterministic input corpus makes the checkpoint -> PyTorch -> ONNX
# comparison reproducible on every export host.  The tolerances allow ordinary
# float32 kernel reordering but are tight enough to catch wrong weights,
# normalization, activation functions, or observation ordering.
TELEOP_ONNX_GATE_VERSION = "1"
TELEOP_ONNX_PARITY_SEED = 20260924
TELEOP_ONNX_PARITY_SAMPLE_COUNT = 16
TELEOP_ONNX_PARITY_ATOL = 1e-5
TELEOP_ONNX_PARITY_RTOL = 1e-4

_CHECKPOINT_NAME_RE = re.compile(r"model_(\d+)\.pt")

if MICROBAN_TELEOP_OBSERVATION_WIDTH != 83:
    raise RuntimeError(
        "Microban teleop observation schema must total 83 values, got "
        f"{MICROBAN_TELEOP_OBSERVATION_WIDTH}"
    )


@dataclass(frozen=True)
class TeleopExportProvenance:
    """Immutable identity of the checkpoint and exporter used for an ONNX."""

    checkpoint_path: Path
    checkpoint_iteration: int
    checkpoint_sha256: str
    exporter_source_commit: str
    exporter_source_dirty: str
    exporter_source_sha256: str

    def metadata(self, *, parity_verified: bool = True) -> dict[str, str]:
        """Return unambiguous string metadata for the deployment artifact."""

        metadata = {
            "checkpoint_filename": self.checkpoint_path.name,
            "checkpoint_iteration": str(self.checkpoint_iteration),
            "checkpoint_iteration_semantics": (
                "zero_based_completed_update_index_from_model_filename"
            ),
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


def collect_teleop_export_provenance(
    checkpoint_path: str | Path,
) -> TeleopExportProvenance:
    """Capture checkpoint bytes, iteration, and exporter source identity.

    Checkpoints must retain RSL-RL's ``model_<N>.pt`` name.  Treating an
    arbitrary filename as an iteration would make later audit/reproduction
    ambiguous.
    """

    checkpoint = Path(checkpoint_path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    match = _CHECKPOINT_NAME_RE.fullmatch(checkpoint.name)
    if match is None:
        raise ValueError(
            "Checkpoint filename must be model_<iteration>.pt for provenance, "
            f"got {checkpoint.name!r}"
        )
    iteration = int(match.group(1))
    source_path = Path(__file__).resolve()
    source_commit, source_dirty = _git_source_state(source_path)
    return TeleopExportProvenance(
        checkpoint_path=checkpoint,
        checkpoint_iteration=iteration,
        checkpoint_sha256=_sha256_file(checkpoint),
        exporter_source_commit=source_commit,
        exporter_source_dirty=source_dirty,
        exporter_source_sha256=_sha256_file(source_path),
    )


def validate_teleop_checkpoint_contract(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device | None = "cpu",
    allow_legacy_diagnostic: bool = False,
) -> TeleopCheckpointContract:
    """Reject checkpoints trained with a different observation/action contract.

    Older checkpoints can have the same tensor widths as v3, so PyTorch can load
    them without an error and an exporter could otherwise attach current v3
    metadata to incompatible training semantics. Validation happens before any
    actor state is loaded. The only legacy escape hatch is explicitly marked
    diagnostic use for an unversioned v1 checkpoint; v2 must never resume into
    v3, and the runner forbids optimizer/iteration resume in diagnostic mode.
    """

    checkpoint = Path(checkpoint_path).resolve()
    match = _CHECKPOINT_NAME_RE.fullmatch(checkpoint.name)
    if match is None:
        raise ValueError(
            "Checkpoint filename must be model_<iteration>.pt for contract "
            f"validation, got {checkpoint.name!r}"
        )
    expected_iteration = int(match.group(1))
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
            "Checkpoint internal iteration must match its model_<iteration>.pt "
            f"filename ({iteration!r} != {expected_iteration})"
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

    version = infos.get("microban_teleop_training_contract_version")
    semantics = infos.get("previous_action_semantics")
    if version is None and semantics is None and allow_legacy_diagnostic:
        return TeleopCheckpointContract(
            version="legacy_unversioned_v1",
            previous_action_semantics="raw_policy_output_before_target_clip",
            iteration=iteration,
            common_step_counter=common_step_counter,
            diagnostic_legacy=True,
        )
    if version != MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION:
        raise ValueError(
            "Checkpoint is not Microban teleop training contract v3; v1/v2 "
            "checkpoints require a clean retrain and cannot be resumed/exported"
        )
    if semantics != MICROBAN_TELEOP_PREVIOUS_ACTION_SEMANTICS:
        raise ValueError("Checkpoint previous-action semantics do not match v3")
    return TeleopCheckpointContract(
        version=version,
        previous_action_semantics=semantics,
        iteration=iteration,
        common_step_counter=common_step_counter,
    )


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
) -> tuple[
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
]:
    """Derive deploy-time keypoint clamps from the registered command config."""

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
    env: ManagerBasedRlEnv, run_path: str
) -> dict[str, list | str | float]:
    """Return metadata aligned exactly with the policy's 18 action outputs.

    ``joint_names`` remains as a backwards-compatible alias for existing Microban
    deployment code.  New code should prefer the unambiguous
    ``action_joint_names`` field.
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
    soft_limits = (
        robot.data.soft_joint_pos_limits[0, action_joint_ids].detach().cpu().tolist()
    )
    soft_lower = [float(bounds[0]) for bounds in soft_limits]
    soft_upper = [float(bounds[1]) for bounds in soft_limits]

    actor_terms = list(env.observation_manager.active_terms["actor"])
    input_schema = dict(MICROBAN_TELEOP_OBSERVATION_SCHEMA)
    (
        foot_target_lower,
        foot_target_upper,
        simultaneous_both_feet_target_lower,
        simultaneous_both_feet_target_upper,
        hand_target_lower,
        hand_target_upper,
    ) = _command_target_bounds(env)

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
        """Construct a fresh v3 runner and optionally bootstrap its actor only.

        Bootstrap options are removed before RSL-RL sees the config.  Requiring
        both an explicit path and SHA-256 prevents a similarly named or replaced
        velocity checkpoint from silently becoming the initialization source.
        """

        runner_cfg = deepcopy(train_cfg)
        bootstrap_path = runner_cfg.pop("bootstrap_velocity_checkpoint", None)
        bootstrap_sha256 = runner_cfg.pop("bootstrap_velocity_checkpoint_sha256", None)
        if (bootstrap_path is None) != (bootstrap_sha256 is None):
            raise ValueError(
                "Velocity actor bootstrap requires both checkpoint path and SHA-256"
            )
        if bootstrap_path is not None and bool(runner_cfg.get("resume", False)):
            raise ValueError(
                "Velocity actor bootstrap is only valid for a fresh run and cannot "
                "be combined with teleop checkpoint resume"
            )

        super().__init__(env, runner_cfg, log_dir=log_dir, device=device)
        self.loaded_checkpoint_contract = None
        self.bootstrap_velocity_provenance = None
        self.velocity_actor_bootstrap_info = None
        if bootstrap_path is None:
            return

        expected_joint_names = (
            *MICROBAN_HMD_JOINT_NAMES,
            *MICROBAN_TELEOP_ACTION_JOINT_NAMES,
        )
        joint_names = tuple(self.env.unwrapped.scene["robot"].joint_names)
        if joint_names != expected_joint_names:
            raise ValueError(
                "Velocity bootstrap requires teleop natural joint order "
                f"{expected_joint_names}, got {joint_names}"
            )

        from mjlab_microban.tasks.microban_teleop_bootstrap import (
            VELOCITY_ACTOR_BOOTSTRAP_MAPPING_VERSION,
            load_velocity_actor_bootstrap,
        )

        self.bootstrap_velocity_provenance = load_velocity_actor_bootstrap(
            self.alg.get_policy(),
            bootstrap_path,
            bootstrap_sha256,
        )
        self.velocity_actor_bootstrap_info = {
            "mapping_version": VELOCITY_ACTOR_BOOTSTRAP_MAPPING_VERSION,
            "source_checkpoint_path": str(
                self.bootstrap_velocity_provenance.checkpoint_path
            ),
            "source_checkpoint_sha256": (
                self.bootstrap_velocity_provenance.checkpoint_sha256
            ),
            "source_normalizer_count": (
                self.bootstrap_velocity_provenance.source_normalizer_count
            ),
            "installed_normalizer_count": (
                self.bootstrap_velocity_provenance.installed_normalizer_count
            ),
            "copied_state": "actor_normalizer_and_mlp_only",
            "distribution_copied": False,
            "critic_copied": False,
            "optimizer_copied": False,
        }
        print(
            "Initialized teleop actor shared inputs from pinned velocity checkpoint "
            f"{self.bootstrap_velocity_provenance.checkpoint_path} "
            f"(sha256={self.bootstrap_velocity_provenance.checkpoint_sha256}, "
            "critic/distribution/optimizer not copied)"
        )

    def _require_nonlegacy_deployment_contract(self) -> None:
        """Prevent diagnostic v1 weights from ever being saved or exported.

        ``allow_legacy_teleop_contract`` exists solely so the deterministic
        evaluator can characterize the final v1 run.  Once those weights are
        resident in a runner, every path that could create a checkpoint or ONNX
        must fail before writing anything; otherwise current v3 metadata could
        be attached to an observation-incompatible actor.
        """

        contract = self.loaded_checkpoint_contract
        if contract is not None and contract.diagnostic_legacy:
            raise ValueError(
                "Legacy teleop checkpoints are diagnostics-only and cannot be "
                "saved, exported, or tagged with v3 metadata"
            )

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
        allow_legacy_teleop_contract: bool = False,
    ) -> dict:
        """Validate v3 semantics, then load at the next PPO iteration.

        RSL-RL stores the zero-based iteration that has just completed.  Its
        default loader resumes *at* that index, repeating one PPO update.  This
        task treats a checkpoint named ``model_N.pt`` as ``N + 1`` completed
        iterations and starts at ``N + 1`` instead.  V1 weights have identical
        tensor widths but incompatible previous-action semantics, so they are
        rejected before state loading.  An explicit legacy mode exists only for
        actor-only diagnostics and can never resume training.
        """

        if getattr(self, "bootstrap_velocity_provenance", None) is not None:
            raise ValueError(
                "A velocity-bootstrapped fresh runner cannot also load a teleop "
                "checkpoint; construct a non-bootstrap runner to resume v3"
            )
        loads_iteration = load_cfg is None or bool(load_cfg.get("iteration", False))
        contract = validate_teleop_checkpoint_contract(
            path,
            map_location=map_location or "cpu",
            allow_legacy_diagnostic=allow_legacy_teleop_contract,
        )
        if contract.diagnostic_legacy:
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
                    "Legacy teleop checkpoints permit only an explicit actor-only "
                    "diagnostic load and cannot resume training"
                )
        self.loaded_checkpoint_contract = contract
        infos = super().load(
            path,
            load_cfg=load_cfg,
            strict=strict,
            map_location=map_location,
        )
        saved_bootstrap_info = infos.get("velocity_actor_bootstrap")
        if saved_bootstrap_info is not None and not isinstance(
            saved_bootstrap_info, dict
        ):
            raise TypeError("Checkpoint velocity_actor_bootstrap must be a dictionary")
        self.velocity_actor_bootstrap_info = deepcopy(saved_bootstrap_info)
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
        env.curriculum_manager.compute()
        return infos

    def save(self, path: str, infos=None) -> None:
        self._require_nonlegacy_deployment_contract()
        contract_infos = {
            **(infos or {}),
            "microban_teleop_training_contract_version": (
                MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION
            ),
            "previous_action_semantics": (MICROBAN_TELEOP_PREVIOUS_ACTION_SEMANTICS),
        }
        contract_infos.pop("velocity_actor_bootstrap", None)
        bootstrap_info = getattr(self, "velocity_actor_bootstrap_info", None)
        if bootstrap_info is not None:
            contract_infos["velocity_actor_bootstrap"] = deepcopy(bootstrap_info)
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
        """Export only fresh/resumed v3 weights, never legacy diagnostic weights."""

        self._require_nonlegacy_deployment_contract()
        super().export_policy_to_onnx(path, filename, verbose)
