# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Fail-closed bounded safe-velocity to PICO-teleop actor bootstrap.

The dedicated safe-velocity actor consumes 63 raw observations. The teleop
actor consumes the same fields plus the three HMD-owned joints and the PICO
foot/hand targets (83 values total). Contract v9 embeds the 63 shared columns,
leaves every new column at exact zero, and copies the remaining actor MLP and
bounded distribution exactly. Critic, optimizer, and PPO iteration state are
never read from the source checkpoint.

Historical ``Mjlab-Velocity-Microban`` actors are unbounded and normalized.
They cannot pass :func:`load_safe_velocity_actor_bootstrap`, because the loader
first applies the dedicated safe-velocity checkpoint inspector and strict
bounded-actor reconstruction.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from mjlab_microban.tasks.microban_safe_velocity_checkpoint import (
    MICROBAN_SAFE_VELOCITY_ACTOR_TOPOLOGY,
    MICROBAN_SAFE_VELOCITY_CHECKPOINT_SCHEMA_VERSION,
    MICROBAN_SAFE_VELOCITY_RECIPE_REVISION,
    SafeVelocityCheckpointIdentity,
    load_frozen_safe_velocity_actor,
)
from mjlab_microban.tasks.microban_safe_velocity_env_cfg import (
    MICROBAN_SAFE_VELOCITY_JOINT_NAMES,
    MICROBAN_SAFE_VELOCITY_OBSERVATION_SCHEMA,
)
from mjlab_microban.tasks.microban_teleop_mdp import (
    AsymmetricBoundedGaussianDistribution,
)

VELOCITY_ACTOR_OBSERVATION_WIDTH = 63
TELEOP_ACTOR_OBSERVATION_WIDTH = 83
SAFE_VELOCITY_ACTOR_BOOTSTRAP_MAPPING_VERSION = (
    "bounded_raw_safe_velocity_63_to_teleop_83_v1"
)
# Import compatibility only. The value names the v9 safe mapping; it does not
# re-enable the retired unbounded/normalized velocity loader.
VELOCITY_ACTOR_BOOTSTRAP_MAPPING_VERSION = SAFE_VELOCITY_ACTOR_BOOTSTRAP_MAPPING_VERSION
SAFE_VELOCITY_TARGET_ACTOR_TOPOLOGY = (83, 512, 256, 128, 18)
SAFE_VELOCITY_ACCEPTANCE_RECEIPT_SCHEMA_VERSION = 2
SAFE_VELOCITY_ACCEPTANCE_GATE = "microban_safe_velocity_fixed_forward_v3"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SAFE_VELOCITY_GATE_CONFIGURATION = {
    "device": "cuda:0",
    "num_envs": 64,
    "steps_requested": 200,
    "steps_executed": 200,
    "seed": 42,
    "command_vx_m_s": 0.08,
    "actual_command_exactly_verified_each_step": True,
    "step_dt_s": 0.02,
    "guard_lookahead_s": 0.12,
    "guard_margin_ratio": 0.05,
}
_SAFE_VELOCITY_GATE_THRESHOLDS = {
    "completion_fraction_min": 0.95,
    "fall_fraction_max": 0.0,
    "nonfinite_fraction_max": 0.0,
    "forward_velocity_p05_m_s_min": 0.01,
    "forward_displacement_p05_m_min": 0.02,
    "actual_soft_limit_violation_rad_max": 1.0e-6,
    "actual_lookahead_soft_limit_violation_rad_max": 1.0e-6,
    "target_clip_rad_max": 1.0e-7,
}

# The safe-velocity actor initializer and teleop actor construction share these
# constants. Contract v9 copies the learned source head verbatim and applies no
# second shoulder override during mapping.
TELEOP_SHOULDER_ROLL_ACTION_INDICES: tuple[int, int] = (1, 10)
TELEOP_SHOULDER_ROLL_INITIAL_LATENT_BIASES: tuple[float, float] = (-0.25, 0.25)
TELEOP_SHOULDER_ROLL_INITIALIZATION = (
    "zero_final_weights_constant_inward_latent_bias_v1"
)

# safe velocity: base_ang_vel(3), gravity(3), body joint_pos(18),
# body joint_vel(18), actions(18), command(3)
# teleop: base_ang_vel(3), gravity(3), joint_pos(21; HMD first),
# joint_vel(21; HMD first), actions(18), command(3), feet(6), hands(8)
VELOCITY_TO_TELEOP_OBSERVATION_INDEX: tuple[tuple[int, int], ...] = (
    *((index, index) for index in range(6)),
    *((index, index + 3) for index in range(6, 24)),
    *((index, index + 6) for index in range(24, 42)),
    *((index, index + 6) for index in range(42, 60)),
    *((index, index + 6) for index in range(60, 63)),
)
if tuple(source for source, _target in VELOCITY_TO_TELEOP_OBSERVATION_INDEX) != (
    tuple(range(VELOCITY_ACTOR_OBSERVATION_WIDTH))
):
    raise RuntimeError("Safe-velocity mapping must cover every source column once")
if len({target for _source, target in VELOCITY_TO_TELEOP_OBSERVATION_INDEX}) != (
    VELOCITY_ACTOR_OBSERVATION_WIDTH
):
    raise RuntimeError("Safe-velocity mapping target columns must be unique")

SAFE_VELOCITY_NEW_TELEOP_OBSERVATION_COLUMNS: tuple[int, ...] = tuple(
    sorted(
        set(range(TELEOP_ACTOR_OBSERVATION_WIDTH))
        - {target for _source, target in VELOCITY_TO_TELEOP_OBSERVATION_INDEX}
    )
)
if len(SAFE_VELOCITY_NEW_TELEOP_OBSERVATION_COLUMNS) != 20:
    raise RuntimeError("Safe-velocity bootstrap must introduce exactly 20 columns")

_MLP_KEYS_AFTER_FIRST_WEIGHT = (
    "mlp.0.bias",
    "mlp.2.weight",
    "mlp.2.bias",
    "mlp.4.weight",
    "mlp.4.bias",
    "mlp.6.weight",
    "mlp.6.bias",
)
_DISTRIBUTION_PARAMETER_KEYS = ("distribution.log_std_param",)
_DISTRIBUTION_CONTRACT_KEYS = (
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
SAFE_VELOCITY_COPIED_DISTRIBUTION_KEYS = (
    *_DISTRIBUTION_PARAMETER_KEYS,
    *_DISTRIBUTION_CONTRACT_KEYS,
)


@dataclass(frozen=True)
class SafeVelocityActorBootstrapProvenance:
    """Verified source identity installed into a fresh contract-v9 actor."""

    source: SafeVelocityCheckpointIdentity
    acceptance_receipt: SafeVelocityAcceptanceReceiptIdentity
    mapping_version: str
    target_actor_topology: tuple[int, ...]
    new_teleop_observation_columns: tuple[int, ...]
    copied_distribution_keys: tuple[str, ...]


@dataclass(frozen=True)
class SafeVelocityAcceptanceReceiptIdentity:
    """Raw-byte identity of a passing fixed-forward safety evaluation."""

    path: Path
    sha256: str
    schema_version: int
    gate: str


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Safe-velocity acceptance receipt duplicates {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> object:
    raise ValueError(
        f"Safe-velocity acceptance receipt contains non-finite JSON {value!r}"
    )


def validate_safe_velocity_acceptance_receipt(
    receipt_path: str | Path,
    source: SafeVelocityCheckpointIdentity,
    *,
    expected_receipt_sha256: str | None = None,
) -> SafeVelocityAcceptanceReceiptIdentity:
    """Validate one evaluator receipt against the already-inspected source.

    The evaluator JSON is not trusted merely because it says ``pass``: its raw
    bytes are pinned, its checkpoint identity must match the source exactly,
    every emitted safety check must be the JSON boolean ``true``, and the
    evaluator must attest that it verified the commanded velocity on every
    simulator step.
    """

    path = Path(receipt_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Safe-velocity acceptance receipt not found: {path}")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_receipt_sha256 is not None:
        if _SHA256_RE.fullmatch(expected_receipt_sha256) is None:
            raise ValueError("Expected receipt SHA-256 must be lowercase hex")
        if digest != expected_receipt_sha256:
            raise ValueError(
                "Safe-velocity acceptance receipt SHA-256 mismatch: "
                f"{digest} != {expected_receipt_sha256}"
            )
    try:
        report = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_nonfinite_json,
        )
    except UnicodeDecodeError as exc:
        raise ValueError("Safe-velocity acceptance receipt must be UTF-8 JSON") from exc
    if not isinstance(report, dict):
        raise TypeError("Safe-velocity acceptance receipt root must be an object")
    expected_report_keys = {
        "schema_version",
        "gate",
        "checkpoint",
        "configuration",
        "metrics",
        "thresholds",
        "checks",
        "status",
        "summary",
    }
    if set(report) != expected_report_keys:
        raise ValueError("Safe-velocity acceptance receipt fields drifted")
    if report.get("schema_version") != SAFE_VELOCITY_ACCEPTANCE_RECEIPT_SCHEMA_VERSION:
        raise ValueError("Safe-velocity acceptance receipt schema is unsupported")
    if report.get("gate") != SAFE_VELOCITY_ACCEPTANCE_GATE:
        raise ValueError("Safe-velocity acceptance receipt gate is not current")
    checks = report.get("checks")
    required_checks = {
        "completion",
        "no_falls",
        "finite",
        "forward_velocity",
        "forward_displacement",
        "actual_soft_limits",
        "actual_lookahead_soft_limits",
        "absolute_target_clip",
    }
    if not isinstance(checks, dict) or set(checks) != required_checks:
        raise ValueError("Safe-velocity acceptance receipt checks are incomplete")

    configuration = report.get("configuration")
    if not isinstance(configuration, dict):
        raise TypeError("Safe-velocity acceptance configuration is malformed")
    if set(configuration) != set(_SAFE_VELOCITY_GATE_CONFIGURATION).union(
        {"absolute_clip_max_tensor_error_rad"}
    ):
        raise ValueError("Safe-velocity acceptance configuration fields drifted")
    for key, expected in _SAFE_VELOCITY_GATE_CONFIGURATION.items():
        if configuration.get(key) != expected:
            raise ValueError(
                f"Safe-velocity acceptance configuration is not canonical for {key!r}"
            )
    clip_tensor_error = configuration["absolute_clip_max_tensor_error_rad"]
    if (
        isinstance(clip_tensor_error, bool)
        or not isinstance(clip_tensor_error, (int, float))
        or not math.isfinite(float(clip_tensor_error))
        or not 0.0 <= float(clip_tensor_error) <= 1.0e-6
    ):
        raise ValueError("Safe-velocity action-clip tensor check is invalid")

    thresholds = report.get("thresholds")
    if thresholds != _SAFE_VELOCITY_GATE_THRESHOLDS:
        raise ValueError("Safe-velocity acceptance thresholds are not canonical")
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        raise TypeError("Safe-velocity acceptance metrics are malformed")

    required_metrics = {
        "completion_fraction",
        "fall_fraction",
        "nonfinite_fraction",
        "forward_velocity_median_m_s",
        "forward_velocity_p05_m_s",
        "forward_displacement_median_m",
        "forward_displacement_p05_m",
        "maximum_actual_soft_limit_violation_rad",
        "maximum_actual_lookahead_soft_limit_violation_rad",
        "maximum_preferred_margin_lookahead_excess_rad",
        "maximum_target_clip_rad",
        "minimum_root_height_m",
    }
    if set(metrics) != required_metrics:
        raise ValueError("Safe-velocity acceptance metric fields drifted")
    metric_values: dict[str, float] = {}
    for key in required_metrics:
        value = metrics.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"Safe-velocity acceptance metric {key!r} must be finite")
        metric_values[key] = float(value)
    derived_checks = {
        "completion": metric_values["completion_fraction"]
        >= _SAFE_VELOCITY_GATE_THRESHOLDS["completion_fraction_min"],
        "no_falls": metric_values["fall_fraction"]
        <= _SAFE_VELOCITY_GATE_THRESHOLDS["fall_fraction_max"],
        "finite": metric_values["nonfinite_fraction"]
        <= _SAFE_VELOCITY_GATE_THRESHOLDS["nonfinite_fraction_max"],
        "forward_velocity": metric_values["forward_velocity_p05_m_s"]
        >= _SAFE_VELOCITY_GATE_THRESHOLDS["forward_velocity_p05_m_s_min"],
        "forward_displacement": metric_values["forward_displacement_p05_m"]
        >= _SAFE_VELOCITY_GATE_THRESHOLDS["forward_displacement_p05_m_min"],
        "actual_soft_limits": metric_values["maximum_actual_soft_limit_violation_rad"]
        <= _SAFE_VELOCITY_GATE_THRESHOLDS["actual_soft_limit_violation_rad_max"],
        "actual_lookahead_soft_limits": metric_values[
            "maximum_actual_lookahead_soft_limit_violation_rad"
        ]
        <= _SAFE_VELOCITY_GATE_THRESHOLDS[
            "actual_lookahead_soft_limit_violation_rad_max"
        ],
        "absolute_target_clip": metric_values["maximum_target_clip_rad"]
        <= _SAFE_VELOCITY_GATE_THRESHOLDS["target_clip_rad_max"],
    }
    if checks != derived_checks:
        raise ValueError(
            "Safe-velocity acceptance checks do not match metrics and thresholds"
        )
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    summary = report.get("summary")
    if not isinstance(summary, dict):
        raise TypeError("Safe-velocity acceptance receipt summary is malformed")
    if set(summary) != {"passed", "failed_checks"}:
        raise ValueError("Safe-velocity acceptance summary fields drifted")
    if (
        summary.get("passed") is not (not failed_checks)
        or summary.get("failed_checks") != failed_checks
    ):
        raise ValueError("Safe-velocity acceptance summary is inconsistent")
    if report.get("status") != ("pass" if not failed_checks else "fail"):
        raise ValueError("Safe-velocity acceptance status is inconsistent")
    if failed_checks:
        raise ValueError("Safe-velocity acceptance receipt did not pass all checks")

    checkpoint = report.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise TypeError("Safe-velocity acceptance checkpoint identity is malformed")
    expected_checkpoint = {
        "path": str(source.path),
        "sha256": source.sha256,
        "iteration": source.iteration,
        "checkpoint_schema_version": source.schema_version,
        "recipe_revision": source.recipe_revision,
        "actor_topology": list(source.actor_topology),
        "actor_obs_normalization": source.actor_obs_normalization,
        "observation_schema": [list(item) for item in source.observation_schema],
        "action_joint_names": list(source.action_joint_names),
    }
    if set(checkpoint) != set(expected_checkpoint):
        raise ValueError("Safe-velocity acceptance checkpoint fields drifted")
    for key, expected in expected_checkpoint.items():
        if checkpoint.get(key) != expected:
            raise ValueError(
                f"Safe-velocity acceptance receipt checkpoint mismatch for {key!r}"
            )

    # Detect a replacement between the read/hash above and returning the pin.
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise RuntimeError("Safe-velocity acceptance receipt changed while reading")
    return SafeVelocityAcceptanceReceiptIdentity(
        path=path,
        sha256=digest,
        schema_version=SAFE_VELOCITY_ACCEPTANCE_RECEIPT_SCHEMA_VERSION,
        gate=SAFE_VELOCITY_ACCEPTANCE_GATE,
    )


def serialize_safe_velocity_actor_bootstrap_provenance(
    provenance: SafeVelocityActorBootstrapProvenance,
) -> dict[str, object]:
    """Return the canonical checkpoint ``infos`` representation."""

    source = provenance.source
    receipt = provenance.acceptance_receipt
    return {
        "mapping_version": provenance.mapping_version,
        "source_checkpoint_path": str(source.path),
        "source_checkpoint_sha256": source.sha256,
        "source_checkpoint_iteration": source.iteration,
        "source_checkpoint_schema_version": source.schema_version,
        "source_recipe_revision": source.recipe_revision,
        "source_actor_topology": list(source.actor_topology),
        "source_observation_schema": [
            [name, width] for name, width in source.observation_schema
        ],
        "source_action_joint_names": list(source.action_joint_names),
        "source_actor_observation_normalization": source.actor_obs_normalization,
        "source_acceptance_receipt_path": str(receipt.path),
        "source_acceptance_receipt_sha256": receipt.sha256,
        "source_acceptance_receipt_schema_version": receipt.schema_version,
        "source_acceptance_gate": receipt.gate,
        "target_actor_topology": list(provenance.target_actor_topology),
        "observation_index_mapping": [
            [source_index, target_index]
            for source_index, target_index in VELOCITY_TO_TELEOP_OBSERVATION_INDEX
        ],
        "new_teleop_observation_columns": list(
            provenance.new_teleop_observation_columns
        ),
        "copied_distribution_keys": list(provenance.copied_distribution_keys),
        "copied_state": "actor_mlp_and_complete_bounded_distribution_only",
        "distribution_contract_equal": True,
        "actor_normalizer_copied": False,
        "critic_copied": False,
        "optimizer_copied": False,
        "iteration_copied": False,
    }


def _require_tensor(
    state: Mapping[str, object], key: str, expected_shape: tuple[int, ...]
) -> torch.Tensor:
    value = state.get(key)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Actor state is missing Tensor {key!r}")
    if tuple(value.shape) != expected_shape:
        raise ValueError(
            f"Actor state {key!r} must have shape {expected_shape}, "
            f"got {tuple(value.shape)}"
        )
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"Actor state {key!r} contains non-finite values")
    return value


def initialize_teleop_shoulder_roll_mlp_head(mlp: torch.nn.Module) -> None:
    """Seed a newly constructed bounded actor's narrow shoulder-roll rows."""

    linear_layers = [
        module for module in mlp.modules() if isinstance(module, torch.nn.Linear)
    ]
    if not linear_layers:
        raise ValueError("Fresh actor MLP must contain a linear output layer")
    final_layer = linear_layers[-1]
    if final_layer.out_features != 18 or final_layer.bias is None:
        raise ValueError("Fresh actor final layer must contain 18 biased action rows")
    shoulder_indices = list(TELEOP_SHOULDER_ROLL_ACTION_INDICES)
    with torch.no_grad():
        final_layer.weight[shoulder_indices] = 0.0
        final_layer.bias[shoulder_indices] = final_layer.bias.new_tensor(
            TELEOP_SHOULDER_ROLL_INITIAL_LATENT_BIASES
        )


def expand_velocity_observation_to_teleop(
    velocity_observation: torch.Tensor,
) -> torch.Tensor:
    """Embed raw safe-velocity observations with all 20 new fields zeroed."""

    if velocity_observation.ndim < 1 or velocity_observation.shape[-1] != (
        VELOCITY_ACTOR_OBSERVATION_WIDTH
    ):
        raise ValueError(
            f"Velocity observation must end in width {VELOCITY_ACTOR_OBSERVATION_WIDTH}"
        )
    teleop = velocity_observation.new_zeros(
        (*velocity_observation.shape[:-1], TELEOP_ACTOR_OBSERVATION_WIDTH)
    )
    source_indices, target_indices = zip(
        *VELOCITY_TO_TELEOP_OBSERVATION_INDEX, strict=True
    )
    teleop[..., list(target_indices)] = velocity_observation[..., list(source_indices)]
    return teleop


def _validate_distribution_contract(
    target_actor_state: Mapping[str, object],
    source_actor_state: Mapping[str, object],
) -> None:
    """Require identical complete bounded-transform state before copying."""

    target_distribution_keys = {
        key for key in target_actor_state if key.startswith("distribution.")
    }
    source_distribution_keys = {
        key for key in source_actor_state if key.startswith("distribution.")
    }
    expected_keys = set(SAFE_VELOCITY_COPIED_DISTRIBUTION_KEYS)
    if target_distribution_keys != expected_keys:
        raise ValueError(
            "Teleop bounded distribution state keys differ from contract: "
            f"{sorted(target_distribution_keys)}"
        )
    if source_distribution_keys != expected_keys:
        raise ValueError(
            "Safe-velocity bounded distribution state keys differ from contract: "
            f"{sorted(source_distribution_keys)}"
        )

    # log_std_param is learned and copied below. Every registered transform
    # buffer must be exactly equal before the source parameter may be accepted.
    for key in _DISTRIBUTION_CONTRACT_KEYS:
        target = _require_tensor(target_actor_state, key, (18,))
        source = _require_tensor(source_actor_state, key, (18,))
        if target.dtype != source.dtype:
            raise ValueError(f"Bounded distribution dtype mismatch for {key!r}")
        if not torch.equal(target.detach().cpu(), source.detach().cpu()):
            raise ValueError(
                "Safe-velocity and teleop bounded distribution contracts differ "
                f"for {key!r}"
            )


def bootstrap_teleop_actor_state(
    target_actor_state: Mapping[str, object],
    safe_velocity_actor_state: Mapping[str, object],
) -> dict[str, torch.Tensor]:
    """Map one already-verified bounded/raw 63-input actor into 83 inputs."""

    if any(key.startswith("obs_normalizer.") for key in target_actor_state):
        raise ValueError("Contract-v9 teleop actor must use raw observations")
    if any(key.startswith("obs_normalizer.") for key in safe_velocity_actor_state):
        raise ValueError("Safe-velocity source actor must use raw observations")

    expected_keys = {
        "mlp.0.weight",
        *_MLP_KEYS_AFTER_FIRST_WEIGHT,
        *SAFE_VELOCITY_COPIED_DISTRIBUTION_KEYS,
    }
    for name, state in (
        ("teleop target", target_actor_state),
        ("safe-velocity source", safe_velocity_actor_state),
    ):
        actual_keys = set(state)
        if actual_keys != expected_keys:
            raise ValueError(
                f"{name} actor state key set differs from the exact topology: "
                f"missing={sorted(expected_keys - actual_keys)}, "
                f"unexpected={sorted(actual_keys - expected_keys)}"
            )

    target_first = _require_tensor(
        target_actor_state, "mlp.0.weight", (512, TELEOP_ACTOR_OBSERVATION_WIDTH)
    )
    source_first = _require_tensor(
        safe_velocity_actor_state,
        "mlp.0.weight",
        (512, VELOCITY_ACTOR_OBSERVATION_WIDTH),
    )
    _validate_distribution_contract(target_actor_state, safe_velocity_actor_state)
    expected_shared_shapes = {
        "mlp.0.bias": (512,),
        "mlp.2.weight": (256, 512),
        "mlp.2.bias": (256,),
        "mlp.4.weight": (128, 256),
        "mlp.4.bias": (128,),
        "mlp.6.weight": (18, 128),
        "mlp.6.bias": (18,),
    }
    for key, shape in expected_shared_shapes.items():
        _require_tensor(target_actor_state, key, shape)
        _require_tensor(safe_velocity_actor_state, key, shape)

    result: dict[str, torch.Tensor] = {}
    for key, value in target_actor_state.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Target actor state {key!r} is not a Tensor")
        result[key] = value.detach().clone()

    source_indices, target_indices = zip(
        *VELOCITY_TO_TELEOP_OBSERVATION_INDEX, strict=True
    )
    first = torch.zeros_like(target_first)
    first[:, list(target_indices)] = source_first[:, list(source_indices)].to(
        device=first.device, dtype=first.dtype
    )
    result["mlp.0.weight"] = first

    copied_keys = (
        *_MLP_KEYS_AFTER_FIRST_WEIGHT,
        *SAFE_VELOCITY_COPIED_DISTRIBUTION_KEYS,
    )
    for key in copied_keys:
        target_value = result.get(key)
        if target_value is None:
            raise ValueError(f"Target actor state is missing Tensor {key!r}")
        source_value = _require_tensor(
            safe_velocity_actor_state, key, tuple(target_value.shape)
        )
        if source_value.dtype != target_value.dtype:
            raise ValueError(f"Actor state dtype mismatch for {key!r}")
        result[key] = (
            source_value.to(device=target_value.device, dtype=target_value.dtype)
            .detach()
            .clone()
        )

    new_columns = list(SAFE_VELOCITY_NEW_TELEOP_OBSERVATION_COLUMNS)
    if not torch.equal(
        result["mlp.0.weight"][:, new_columns],
        torch.zeros_like(result["mlp.0.weight"][:, new_columns]),
    ):
        raise RuntimeError("New teleop first-layer columns are not exact zero")
    return result


def load_safe_velocity_actor_bootstrap(
    target_actor: torch.nn.Module,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    acceptance_receipt_path: str | Path,
) -> SafeVelocityActorBootstrapProvenance:
    """Strictly verify and install only a dedicated bounded safe actor."""

    if not isinstance(
        getattr(target_actor, "distribution", None),
        AsymmetricBoundedGaussianDistribution,
    ):
        raise TypeError("Contract-v9 target actor must use the bounded distribution")
    source_actor, identity = load_frozen_safe_velocity_actor(
        checkpoint_path,
        device="cpu",
        expected_sha256=checkpoint_sha256,
    )
    if identity.schema_version != MICROBAN_SAFE_VELOCITY_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Safe-velocity checkpoint schema version is not supported")
    if identity.recipe_revision != MICROBAN_SAFE_VELOCITY_RECIPE_REVISION:
        raise ValueError("Safe-velocity checkpoint recipe revision is not current")
    if identity.actor_topology != MICROBAN_SAFE_VELOCITY_ACTOR_TOPOLOGY:
        raise ValueError("Safe-velocity checkpoint actor topology is not current")
    if identity.observation_schema != MICROBAN_SAFE_VELOCITY_OBSERVATION_SCHEMA:
        raise ValueError("Safe-velocity observation schema is not current")
    if identity.action_joint_names != MICROBAN_SAFE_VELOCITY_JOINT_NAMES:
        raise ValueError("Safe-velocity action order is not current")
    if identity.actor_obs_normalization is not False:
        raise ValueError("Safe-velocity actor must consume raw observations")
    acceptance_receipt = validate_safe_velocity_acceptance_receipt(
        acceptance_receipt_path,
        identity,
    )

    mapped = bootstrap_teleop_actor_state(
        target_actor.state_dict(), source_actor.state_dict()
    )
    target_actor.load_state_dict(mapped, strict=True)
    return SafeVelocityActorBootstrapProvenance(
        source=identity,
        acceptance_receipt=acceptance_receipt,
        mapping_version=SAFE_VELOCITY_ACTOR_BOOTSTRAP_MAPPING_VERSION,
        target_actor_topology=SAFE_VELOCITY_TARGET_ACTOR_TOPOLOGY,
        new_teleop_observation_columns=(SAFE_VELOCITY_NEW_TELEOP_OBSERVATION_COLUMNS),
        copied_distribution_keys=SAFE_VELOCITY_COPIED_DISTRIBUTION_KEYS,
    )
