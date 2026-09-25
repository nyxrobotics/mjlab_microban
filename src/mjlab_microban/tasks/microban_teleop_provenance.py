# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Deterministic provenance for canonical Microban teleop training stages.

The human-readable recipe revision identifies the intended algorithm.  It is
not, by itself, evidence that a checkpoint was produced with that recipe: a
generic Tyro invocation can override the resolved environment or PPO config.
This module records the *resolved* configs, critical launch values and exact
training-source bytes, then binds them with one canonical-JSON SHA-256.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from copy import deepcopy
from enum import Enum
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Schema 3 deliberately separates the current fresh actor-only bootstrap recipe
# from the historical v9 -> v10 full-state migration ledger.  In particular, a
# v10 manifest cannot be relabelled as v11 merely because its tensor shapes are
# compatible.
MICROBAN_TELEOP_TRAINING_PROVENANCE_SCHEMA_VERSION = 3
MICROBAN_TELEOP_CANONICAL_STAGE_MODE = "canonical_v11_stage"
# Retained only so the byte-pinned v10 migration source remains auditable.  It
# is not accepted by collect_training_provenance as a current canonical mode.
MICROBAN_TELEOP_CANONICAL_MIGRATION_STAGE_MODE = "canonical_v10_migration_stage"
MICROBAN_TELEOP_TRAINING_PROVENANCE_KEY = "microban_teleop_training_provenance"
MICROBAN_TELEOP_TRAINING_PROVENANCE_SHA256_KEY = (
    "microban_teleop_training_provenance_sha256"
)

_SHA256_HEX_LENGTH = 64
MICROBAN_TELEOP_CANONICAL_STAGE_BOUNDARIES = (
    0,
    3000,
    7000,
    10000,
    15000,
)
MICROBAN_TELEOP_V10_MIGRATION_START_BOUNDARY = 1500
MICROBAN_TELEOP_V10_MIGRATION_TARGET_BOUNDARY = 3000
MICROBAN_TELEOP_V10_LEGACY_CHECKPOINT_SHA256 = (
    "de8b6139872179679a16d72f3007f6d96cf65c97fa88565841eaa5f89511a65f"
)
MICROBAN_TELEOP_V10_LEGACY_CHECKPOINT_ITERATION = 1499
MICROBAN_TELEOP_V10_LEGACY_COMMON_STEP_COUNTER = 36_000
MICROBAN_TELEOP_V10_LEGACY_TRAINING_PROVENANCE_SHA256 = (
    "f09f5580f03d3e38deef4916db7aea3bd8b1f683dc02a75079d24dd17923fce9"
)
MICROBAN_TELEOP_V10_LEGACY_SOURCE_TREE_SHA256 = (
    "61a9fc7b1fe10436c0f033f89710e33e9e5470716d94794f110731c18e7d792a"
)
MICROBAN_TELEOP_V10_LEGACY_GATE_SHA256 = (
    "acb2e39411155d70aed2b18561a243bd8ad20eef56e0942d2dafbb4f96f39b7c"
)
MICROBAN_TELEOP_V10_LEGACY_SAFE_VELOCITY_CHECKPOINT_SHA256 = (
    "416a8b16f7f7980822e4e1df81ffaf9515bc18a246e6fc257405a2c46ceece93"
)
MICROBAN_TELEOP_V10_LEGACY_SAFE_VELOCITY_ACCEPTANCE_RECEIPT_SHA256 = (
    "e68701b11774dd30c8e45a2fd89614a2e4423a9486d01a0d936f0fa6fb760492"
)
MICROBAN_TELEOP_V10_LEGACY_OPTIMIZER_LEARNING_RATE = 7.593750000000002e-05
MICROBAN_TELEOP_V10_FIXED_LEARNING_RATE = 1.0e-5
MICROBAN_TELEOP_V10_MIGRATION_SOURCE_SCHEMA_VERSION = 1
MICROBAN_TELEOP_V10_TRAINING_PROVENANCE_SCHEMA_VERSION = 2
MICROBAN_TELEOP_V11_FIXED_LEARNING_RATE = 1.0e-4
MICROBAN_TELEOP_V11_ENTROPY_COEF = 0.005
MICROBAN_TELEOP_V11_NUM_LEARNING_EPOCHS = 5
MICROBAN_TELEOP_V9_CANONICAL_STAGE_MODE = "canonical_v9_stage"
MICROBAN_TELEOP_V9_TRAINING_PROVENANCE_SCHEMA_VERSION = 1
MICROBAN_TELEOP_V9_TRAINING_CONTRACT_VERSION = "9"
MICROBAN_TELEOP_V9_RECIPE_REVISION = (
    "v9_accepted_safe_velocity_bootstrap_no_walk004_prior_full_pico_curriculum_v3"
)
MICROBAN_TELEOP_V9_ACTOR_INITIALIZATION = (
    "bounded_raw_safe_velocity_actor_only_63_to_83_zero_new_columns_v1"
)
_CANONICAL_STAGE_ENV_NAMES = (
    "MICROBAN_TELEOP_STAGE_START_BOUNDARY",
    "MICROBAN_TELEOP_STAGE_TARGET_BOUNDARY",
    "MICROBAN_TELEOP_PARENT_CHECKPOINT_SHA256",
    "MICROBAN_TELEOP_PARENT_GATE_SHA256",
    "MICROBAN_TELEOP_RESUME_SOURCE_CHECKPOINT_PATH",
    "MICROBAN_TELEOP_RESUME_SOURCE_CHECKPOINT_SHA256",
    "MICROBAN_TELEOP_RESUME_SOURCE_CHECKPOINT_ITERATION",
)
_SAFE_VELOCITY_IDENTITY_FIELDS = (
    "safe_velocity_checkpoint",
    "safe_velocity_checkpoint_sha256",
    "safe_velocity_acceptance_receipt",
)


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 of one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _qualified_name(value: object) -> str:
    module = getattr(value, "__module__", None)
    name = getattr(value, "__qualname__", None) or getattr(value, "__name__", None)
    if not isinstance(module, str) or not isinstance(name, str):
        raise TypeError(f"Cannot derive a stable qualified name for {value!r}")
    return f"{module}:{name}"


def canonicalize_training_config(value: Any, *, _seen: set[int] | None = None) -> Any:
    """Convert a resolved MjLab config into deterministic JSON values.

    Unsupported opaque objects fail closed instead of falling back to ``repr``;
    a repr can contain process-specific memory addresses and would make the
    provenance digest look meaningful while being irreproducible.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Training provenance cannot encode non-finite floats")
        return value
    if isinstance(value, np.generic):
        return canonicalize_training_config(value.item(), _seen=_seen)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, slice):
        return {
            "__slice__": [
                canonicalize_training_config(value.start, _seen=_seen),
                canonicalize_training_config(value.stop, _seen=_seen),
                canonicalize_training_config(value.step, _seen=_seen),
            ]
        }
    if isinstance(value, Enum):
        return {
            "__enum__": _qualified_name(type(value)),
            "value": canonicalize_training_config(value.value, _seen=_seen),
        }
    if isinstance(value, type) or callable(value):
        return {"__callable__": _qualified_name(value)}
    if isinstance(value, torch.Tensor):
        return {
            "__tensor_dtype__": str(value.dtype),
            "value": canonicalize_training_config(
                value.detach().cpu().tolist(), _seen=_seen
            ),
        }
    if isinstance(value, np.ndarray):
        return {
            "__ndarray_dtype__": str(value.dtype),
            "value": canonicalize_training_config(value.tolist(), _seen=_seen),
        }

    if _seen is None:
        _seen = set()
    identity = id(value)
    if identity in _seen:
        raise ValueError("Training config contains a reference cycle")
    _seen.add(identity)
    try:
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {
                "__dataclass__": _qualified_name(type(value)),
                "fields": {
                    field.name: canonicalize_training_config(
                        getattr(value, field.name), _seen=_seen
                    )
                    for field in dataclasses.fields(value)
                },
            }
        if isinstance(value, Mapping):
            result: dict[str, Any] = {}
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
                if not isinstance(key, (str, int, float, bool)):
                    raise TypeError(
                        "Training config mapping keys must be scalar JSON values"
                    )
                canonical_key = str(key)
                if canonical_key in result:
                    raise ValueError(
                        f"Training config keys collide after normalization: {key!r}"
                    )
                result[canonical_key] = canonicalize_training_config(item, _seen=_seen)
            return result
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return [canonicalize_training_config(item, _seen=_seen) for item in value]
        if isinstance(value, (set, frozenset)):
            normalized = [
                canonicalize_training_config(item, _seen=_seen) for item in value
            ]
            return sorted(
                normalized,
                key=lambda item: json.dumps(
                    item, sort_keys=True, separators=(",", ":"), ensure_ascii=True
                ),
            )
        if hasattr(value, "__dict__"):
            public_fields = {
                key: item
                for key, item in vars(value).items()
                if not key.startswith("_")
            }
            return {
                "__object__": _qualified_name(type(value)),
                "fields": canonicalize_training_config(public_fields, _seen=_seen),
            }
    finally:
        _seen.remove(identity)
    raise TypeError(
        "Unsupported resolved training-config value "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def canonical_json_sha256(value: Any) -> str:
    """Hash one already-JSON-compatible value with a fixed encoding."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_lowercase_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _v10_migration_source_identity(checkpoint_path: str) -> dict[str, object]:
    """Build the immutable contract-v9 -> contract-v10 migration ledger."""

    return {
        "schema_version": MICROBAN_TELEOP_V10_MIGRATION_SOURCE_SCHEMA_VERSION,
        "source_contract_version": MICROBAN_TELEOP_V9_TRAINING_CONTRACT_VERSION,
        "source_checkpoint_path": checkpoint_path,
        "source_checkpoint_sha256": MICROBAN_TELEOP_V10_LEGACY_CHECKPOINT_SHA256,
        "source_checkpoint_iteration": (
            MICROBAN_TELEOP_V10_LEGACY_CHECKPOINT_ITERATION
        ),
        "source_common_step_counter": (MICROBAN_TELEOP_V10_LEGACY_COMMON_STEP_COUNTER),
        "source_training_provenance_schema_version": (
            MICROBAN_TELEOP_V9_TRAINING_PROVENANCE_SCHEMA_VERSION
        ),
        "source_training_provenance_sha256": (
            MICROBAN_TELEOP_V10_LEGACY_TRAINING_PROVENANCE_SHA256
        ),
        "source_training_source_tree_sha256": (
            MICROBAN_TELEOP_V10_LEGACY_SOURCE_TREE_SHA256
        ),
        "source_recipe_revision": MICROBAN_TELEOP_V9_RECIPE_REVISION,
        "source_actor_initialization": MICROBAN_TELEOP_V9_ACTOR_INITIALIZATION,
        "source_provenance_mode": MICROBAN_TELEOP_V9_CANONICAL_STAGE_MODE,
        "source_stage_start_boundary": 0,
        "source_stage_target_boundary": (MICROBAN_TELEOP_V10_MIGRATION_START_BOUNDARY),
        "source_parent_gate_sha256": MICROBAN_TELEOP_V10_LEGACY_GATE_SHA256,
        "source_optimizer_learning_rate": (
            MICROBAN_TELEOP_V10_LEGACY_OPTIMIZER_LEARNING_RATE
        ),
        "destination_optimizer_learning_rate": (
            MICROBAN_TELEOP_V10_FIXED_LEARNING_RATE
        ),
        "state_transfer": ("actor_critic_optimizer_moments_iteration_common_step_v1"),
        "safe_velocity_checkpoint_sha256": (
            MICROBAN_TELEOP_V10_LEGACY_SAFE_VELOCITY_CHECKPOINT_SHA256
        ),
        "safe_velocity_acceptance_receipt_sha256": (
            MICROBAN_TELEOP_V10_LEGACY_SAFE_VELOCITY_ACCEPTANCE_RECEIPT_SHA256
        ),
    }


def validate_v10_migration_source_identity(
    migration_source: object,
    *,
    verify_source_checkpoint: bool = True,
) -> dict[str, object]:
    """Validate the exact legacy source inherited by every canonical v10 stage."""

    if not isinstance(migration_source, dict):
        raise TypeError("Contract-v10 provenance migration_source must be a dictionary")
    source_path = migration_source.get("source_checkpoint_path")
    if not isinstance(source_path, str) or not Path(source_path).is_absolute():
        raise ValueError("Migration source checkpoint path must be absolute")
    expected = _v10_migration_source_identity(source_path)
    if set(migration_source) != set(expected):
        raise ValueError(
            "Contract-v10 migration_source key set is not canonical: "
            f"missing={sorted(set(expected) - set(migration_source))}, "
            f"unexpected={sorted(set(migration_source) - set(expected))}"
        )
    mismatches = [
        name for name, value in expected.items() if migration_source.get(name) != value
    ]
    if mismatches:
        raise ValueError(
            "Contract-v10 migration_source identity mismatch: " + ", ".join(mismatches)
        )
    if verify_source_checkpoint:
        try:
            resolved = Path(source_path).resolve(strict=True)
        except OSError as exc:
            raise ValueError("Migration source checkpoint file is missing") from exc
        if not resolved.is_file() or str(resolved) != source_path:
            raise ValueError("Migration source checkpoint path is not canonical")
        if sha256_file(resolved) != MICROBAN_TELEOP_V10_LEGACY_CHECKPOINT_SHA256:
            raise ValueError("Migration source checkpoint SHA-256 mismatch")
    return migration_source


def inherit_v10_migration_source_identity(
    current_manifest: object,
    loaded_manifest: object,
) -> tuple[dict, str]:
    """Carry the immutable migration ledger into an interrupted/later v10 stage."""

    manifests: list[tuple[str, dict]] = []
    for label, value in (("current", current_manifest), ("loaded", loaded_manifest)):
        if not isinstance(value, dict):
            raise TypeError(f"{label.title()} training provenance must be a dictionary")
        if (
            value.get("schema_version")
            != MICROBAN_TELEOP_V10_TRAINING_PROVENANCE_SCHEMA_VERSION
            or value.get("canonical_stage") is not True
        ):
            raise ValueError(
                f"{label.title()} training provenance is not canonical contract-v10"
            )
        manifests.append((label, value))

    current = manifests[0][1]
    loaded = manifests[1][1]
    if current.get("migration_source") is not None:
        raise ValueError(
            "A normal v10 resume must inherit migration_source from its authenticated "
            "checkpoint"
        )
    inherited_source = deepcopy(loaded.get("migration_source"))
    validate_v10_migration_source_identity(inherited_source)
    inherited = deepcopy(current)
    inherited["migration_source"] = inherited_source
    return inherited, canonical_json_sha256(inherited)


def inherit_initial_stage_safe_velocity_identity(
    current_manifest: object,
    loaded_manifest: object,
    bootstrap_info: object,
) -> tuple[dict, str]:
    """Restore a pinned safe-source identity on an interrupted initial resume.

    The canonical wrapper intentionally does not put mutable bootstrap paths back
    on a resume command line.  Consequently, the runner's provisional manifest
    has ``None`` for the three safe-source fields until it has authenticated the
    teleop checkpoint being resumed.  Copy only those fields from that checkpoint
    and require them to agree exactly with its independently validated bootstrap
    record.  All other resolved current-process config remains untouched.
    """

    manifests: list[tuple[str, dict]] = []
    for label, value in (
        ("current", current_manifest),
        ("loaded", loaded_manifest),
    ):
        if not isinstance(value, dict):
            raise TypeError(f"{label.title()} training provenance must be a dictionary")
        if value.get("canonical_stage") is not True:
            raise ValueError(
                f"{label.title()} training provenance is not a canonical stage"
            )
        invocation = value.get("invocation")
        if not isinstance(invocation, dict):
            raise TypeError(
                f"{label.title()} training provenance invocation is malformed"
            )
        if (
            invocation.get("stage_start_boundary") != 0
            or invocation.get("stage_target_boundary") != 3000
            or invocation.get("parent_checkpoint_sha256") is not None
            or invocation.get("parent_gate_sha256") is not None
        ):
            raise ValueError(
                "Safe-source inheritance is valid only within canonical stage "
                "0->3000 with null parents"
            )
        resolved = value.get("resolved_config")
        critical = resolved.get("critical") if isinstance(resolved, dict) else None
        if not isinstance(critical, dict):
            raise TypeError(
                f"{label.title()} training provenance critical config is malformed"
            )
        manifests.append((label, critical))

    if not isinstance(bootstrap_info, dict):
        raise TypeError("Loaded safe-velocity bootstrap info must be a dictionary")
    expected = {
        "safe_velocity_checkpoint": bootstrap_info.get("source_checkpoint_path"),
        "safe_velocity_checkpoint_sha256": bootstrap_info.get(
            "source_checkpoint_sha256"
        ),
        "safe_velocity_acceptance_receipt": bootstrap_info.get(
            "source_acceptance_receipt_path"
        ),
    }
    if any(not isinstance(value, str) or not value for value in expected.values()):
        raise ValueError("Loaded safe-velocity bootstrap identity is incomplete")

    current_critical = manifests[0][1]
    loaded_critical = manifests[1][1]
    current_values = {
        name: current_critical.get(name) for name in _SAFE_VELOCITY_IDENTITY_FIELDS
    }
    if any(value is not None for value in current_values.values()):
        raise ValueError(
            "Interrupted initial-stage resume must inherit its safe source from "
            "the checkpoint, not from mutable runner arguments"
        )
    loaded_values = {
        name: loaded_critical.get(name) for name in _SAFE_VELOCITY_IDENTITY_FIELDS
    }
    if loaded_values != expected:
        mismatches = [
            name
            for name in _SAFE_VELOCITY_IDENTITY_FIELDS
            if loaded_values[name] != expected[name]
        ]
        raise ValueError(
            "Loaded initial-stage safe-source provenance disagrees with its "
            "validated bootstrap record: " + ", ".join(mismatches)
        )

    inherited = deepcopy(current_manifest)
    inherited["resolved_config"]["critical"].update(expected)
    return inherited, canonical_json_sha256(inherited)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def collect_training_source_manifest(project_root: str | Path | None = None) -> dict:
    """Hash all local sources that can affect a teleop training rollout."""

    root = Path(project_root).resolve() if project_root is not None else _project_root()
    candidates = set((root / "src" / "mjlab_microban").rglob("*.py"))
    candidates.update((root / "src" / "mjlab_microban").rglob("*.xml"))
    candidates.update((root / "src" / "mjlab_microban").rglob("*.json"))
    candidates.update(
        root / relative
        for relative in (
            "data/motions/microban_twist2_walk004_locomotion_prior.npz",
            "pyproject.toml",
            "uv.lock",
            "scripts/train_microban_teleop.sh",
            "scripts/train_microban_teleop_v9.sh",
            "scripts/train_microban_teleop_v10.sh",
            "scripts/train_microban_teleop_v11.sh",
            "scripts/evaluate_microban_teleop_v8_stage.sh",
            "scripts/evaluate_microban_teleop_v9_stage.sh",
            "scripts/evaluate_microban_teleop_v10_stage.sh",
            "scripts/evaluate_microban_teleop_v11_stage.sh",
        )
    )
    files = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(candidates)
        if path.is_file()
    }
    if not files:
        raise ValueError(f"No Microban training source files found below {root}")
    return {
        "algorithm": "sha256(canonical_json_path_to_sha256_v1)",
        "tree_sha256": canonical_json_sha256(files),
        "files": files,
    }


def _optional_sha256_env(name: str) -> str | None:
    value = os.environ.get(name, "")
    if not value:
        return None
    if len(value) != _SHA256_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be empty or one lowercase SHA-256")
    return value


def _optional_existing_file_env(name: str) -> str | None:
    value = os.environ.get(name, "")
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{name} does not identify an existing file") from exc
    if not resolved.is_file():
        raise ValueError(f"{name} does not identify a file")
    return str(resolved)


def _optional_nonnegative_int_env(name: str) -> int | None:
    value = os.environ.get(name, "")
    if not value:
        return None
    if not value.isdecimal():
        raise ValueError(f"{name} must be empty or a non-negative decimal integer")
    return int(value)


def _stage_boundary_env(name: str, *, required: bool) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        if required:
            raise ValueError(f"Canonical stage launch requires {name}")
        return None
    if not value.isdecimal():
        raise ValueError(f"{name} must be a non-negative decimal integer")
    return int(value)


def _validate_canonical_stage_lineage(
    *,
    mode: object,
    start_boundary: object,
    target_boundary: object,
    parent_checkpoint_sha256: object,
    parent_gate_sha256: object,
) -> None:
    if (
        isinstance(start_boundary, bool)
        or not isinstance(start_boundary, int)
        or isinstance(target_boundary, bool)
        or not isinstance(target_boundary, int)
    ):
        raise TypeError("Canonical stage boundaries must be integers")
    if mode != MICROBAN_TELEOP_CANONICAL_STAGE_MODE:
        raise ValueError("Canonical stage mode is unsupported")
    allowed_intervals = set(pairwise(MICROBAN_TELEOP_CANONICAL_STAGE_BOUNDARIES))
    if (start_boundary, target_boundary) not in allowed_intervals:
        raise ValueError(
            "Canonical stage interval is not one adjacent contract-v11 boundary pair: "
            f"{start_boundary}->{target_boundary}"
        )
    parents = (parent_checkpoint_sha256, parent_gate_sha256)
    if start_boundary == 0:
        if parents != (None, None):
            raise ValueError("Canonical v11 fresh stage must have null parents")
        return
    for name, value in zip(("parent checkpoint", "parent gate"), parents, strict=True):
        if (
            not isinstance(value, str)
            or len(value) != _SHA256_HEX_LENGTH
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"Canonical stage {name} must be one lowercase SHA-256")


def collect_training_provenance(
    env: Any,
    runner_cfg: Mapping[str, Any],
    *,
    training_contract_version: str,
    recipe_revision: str,
    actor_initialization: str,
    project_root: str | Path | None = None,
) -> tuple[dict, str]:
    """Collect the resolved config/source manifest and its canonical digest."""

    mode = os.environ.get("MICROBAN_TELEOP_PROVENANCE_MODE", "generic")
    canonical_modes = (MICROBAN_TELEOP_CANONICAL_STAGE_MODE,)
    canonical_stage = mode in canonical_modes
    if mode not in ("generic", *canonical_modes):
        raise ValueError(
            "MICROBAN_TELEOP_PROVENANCE_MODE must be 'generic', "
            f"'{MICROBAN_TELEOP_CANONICAL_STAGE_MODE}'"
        )
    for name in _CANONICAL_STAGE_ENV_NAMES:
        if not canonical_stage and os.environ.get(name):
            raise ValueError(f"{name} is valid only for a canonical stage launch")

    start_boundary = _stage_boundary_env(
        "MICROBAN_TELEOP_STAGE_START_BOUNDARY", required=canonical_stage
    )
    target_boundary = _stage_boundary_env(
        "MICROBAN_TELEOP_STAGE_TARGET_BOUNDARY", required=canonical_stage
    )
    parent_checkpoint_sha256 = _optional_sha256_env(
        "MICROBAN_TELEOP_PARENT_CHECKPOINT_SHA256"
    )
    parent_gate_sha256 = _optional_sha256_env("MICROBAN_TELEOP_PARENT_GATE_SHA256")
    resume_source_checkpoint_path = _optional_existing_file_env(
        "MICROBAN_TELEOP_RESUME_SOURCE_CHECKPOINT_PATH"
    )
    resume_source_checkpoint_sha256 = _optional_sha256_env(
        "MICROBAN_TELEOP_RESUME_SOURCE_CHECKPOINT_SHA256"
    )
    resume_source_checkpoint_iteration = _optional_nonnegative_int_env(
        "MICROBAN_TELEOP_RESUME_SOURCE_CHECKPOINT_ITERATION"
    )
    resume_source_values = (
        resume_source_checkpoint_path,
        resume_source_checkpoint_sha256,
        resume_source_checkpoint_iteration,
    )
    if any(value is not None for value in resume_source_values) and any(
        value is None for value in resume_source_values
    ):
        raise ValueError(
            "Canonical resume source requires checkpoint path, SHA-256, and "
            "iteration together"
        )
    if (
        resume_source_checkpoint_path is not None
        and resume_source_checkpoint_sha256 is not None
        and sha256_file(resume_source_checkpoint_path)
        != resume_source_checkpoint_sha256
    ):
        raise ValueError("Canonical resume source checkpoint SHA-256 mismatch")
    if canonical_stage:
        assert start_boundary is not None and target_boundary is not None
        _validate_canonical_stage_lineage(
            mode=mode,
            start_boundary=start_boundary,
            target_boundary=target_boundary,
            parent_checkpoint_sha256=parent_checkpoint_sha256,
            parent_gate_sha256=parent_gate_sha256,
        )
        resume = bool(runner_cfg.get("resume", False))
        if start_boundary == 0 and not resume:
            if any(value is not None for value in resume_source_values):
                raise ValueError(
                    "Canonical v11 fresh stage cannot claim a resume source"
                )
        elif any(value is None for value in resume_source_values):
            raise ValueError("Canonical v11 resume requires an exact resume source")

    unwrapped = getattr(env, "unwrapped", env)
    env_cfg = getattr(unwrapped, "cfg", None)
    if env_cfg is None:
        raise ValueError("Teleop training provenance requires env.unwrapped.cfg")
    num_envs = getattr(unwrapped, "num_envs", None)
    if isinstance(num_envs, bool) or not isinstance(num_envs, int):
        raise TypeError("Teleop training provenance requires an integer num_envs")

    resolved_runner = canonicalize_training_config(dict(runner_cfg))
    resolved_environment = canonicalize_training_config(env_cfg)
    critical = {
        "num_envs": num_envs,
        "environment_seed": getattr(env_cfg, "seed", None),
        "runner_seed": runner_cfg.get("seed"),
        "num_steps_per_env": runner_cfg.get("num_steps_per_env"),
        "save_interval": runner_cfg.get("save_interval"),
        "max_iterations_for_process": runner_cfg.get("max_iterations"),
        "resume": bool(runner_cfg.get("resume", False)),
        "logger": runner_cfg.get("logger"),
        "upload_model": runner_cfg.get("upload_model"),
        "wrapper_clip_actions": canonicalize_training_config(
            getattr(env, "clip_actions", None)
        ),
        "checkpoint_consumer_mode": runner_cfg.get("checkpoint_consumer_mode", False),
        "safe_velocity_checkpoint": runner_cfg.get("safe_velocity_checkpoint"),
        "safe_velocity_checkpoint_sha256": runner_cfg.get(
            "safe_velocity_checkpoint_sha256"
        ),
        "safe_velocity_acceptance_receipt": runner_cfg.get(
            "safe_velocity_acceptance_receipt"
        ),
        "save_pristine_checkpoint": bool(
            runner_cfg.get("save_pristine_checkpoint", False)
        ),
    }
    manifest = {
        "schema_version": MICROBAN_TELEOP_TRAINING_PROVENANCE_SCHEMA_VERSION,
        "canonical_stage": canonical_stage,
        "training_contract_version": training_contract_version,
        "recipe_revision": recipe_revision,
        "actor_initialization": actor_initialization,
        "resolved_config": {
            "critical": canonicalize_training_config(critical),
            "environment": resolved_environment,
            "runner": resolved_runner,
        },
        "source": collect_training_source_manifest(project_root),
        # Kept as an explicit null field so schema-3 validators can reject any
        # attempt to smuggle the retired v10 full-state migration into v11.
        "migration_source": None,
        "invocation": {
            "mode": mode,
            "stage_start_boundary": start_boundary,
            "stage_target_boundary": target_boundary,
            "parent_checkpoint_sha256": parent_checkpoint_sha256,
            "parent_gate_sha256": parent_gate_sha256,
            "resume_source_checkpoint_path": resume_source_checkpoint_path,
            "resume_source_checkpoint_sha256": resume_source_checkpoint_sha256,
            "resume_source_checkpoint_iteration": (resume_source_checkpoint_iteration),
        },
    }
    return manifest, canonical_json_sha256(manifest)


def validate_training_provenance(
    manifest: object,
    digest: object,
    *,
    require_canonical_stage: bool = False,
    expected_contract_version: str | None = None,
    expected_recipe_revision: str | None = None,
    expected_actor_initialization: str | None = None,
    require_current_source: bool = False,
    project_root: str | Path | None = None,
) -> dict:
    """Validate a checkpoint provenance pair and return the typed manifest."""

    if not isinstance(manifest, dict):
        raise TypeError("Checkpoint training provenance must be a dictionary")
    if (
        not isinstance(digest, str)
        or len(digest) != _SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("Checkpoint training provenance SHA-256 is malformed")
    actual_digest = canonical_json_sha256(manifest)
    if digest != actual_digest:
        raise ValueError("Checkpoint training provenance SHA-256 mismatch")
    if (
        manifest.get("schema_version")
        != MICROBAN_TELEOP_TRAINING_PROVENANCE_SCHEMA_VERSION
    ):
        raise ValueError("Checkpoint training provenance schema is unsupported")
    if require_canonical_stage and manifest.get("canonical_stage") is not True:
        raise ValueError("Checkpoint was not produced by the canonical stage driver")
    expected_values = {
        "training_contract_version": expected_contract_version,
        "recipe_revision": expected_recipe_revision,
        "actor_initialization": expected_actor_initialization,
    }
    for key, expected in expected_values.items():
        if expected is not None and manifest.get(key) != expected:
            raise ValueError(f"Checkpoint training provenance {key} mismatch")

    source = manifest.get("source")
    if not isinstance(source, dict):
        raise TypeError("Checkpoint training provenance source is malformed")
    files = source.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Checkpoint training provenance source files are missing")
    if source.get("tree_sha256") != canonical_json_sha256(files):
        raise ValueError("Checkpoint training source tree SHA-256 mismatch")
    if require_current_source:
        current = collect_training_source_manifest(project_root)
        if source != current:
            raise ValueError(
                "Checkpoint training source does not match the current worktree"
            )
    return manifest


def validate_resume_source_checkpoint_file(
    manifest: Mapping[str, Any],
) -> Path | None:
    """Re-hash the exact intermediate checkpoint pinned by a resume process."""

    invocation = manifest.get("invocation")
    if not isinstance(invocation, dict):
        raise TypeError("Training provenance invocation is malformed")
    path_value = invocation.get("resume_source_checkpoint_path")
    sha256 = invocation.get("resume_source_checkpoint_sha256")
    iteration = invocation.get("resume_source_checkpoint_iteration")
    values = (path_value, sha256, iteration)
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("Resume source checkpoint identity is incomplete")
    if not isinstance(path_value, str) or not Path(path_value).is_absolute():
        raise ValueError("Resume source checkpoint path must be absolute")
    try:
        path = Path(path_value).resolve(strict=True)
    except OSError as exc:
        raise ValueError("Resume source checkpoint file is missing") from exc
    if str(path) != path_value or not path.is_file():
        raise ValueError("Resume source checkpoint path is not canonical")
    if (
        not isinstance(iteration, int)
        or isinstance(iteration, bool)
        or iteration < 0
        or path.name != f"model_{iteration}.pt"
    ):
        raise ValueError("Resume source checkpoint filename/iteration mismatch")
    if not isinstance(sha256, str) or sha256_file(path) != sha256:
        raise ValueError("Resume source checkpoint SHA-256 mismatch")
    return path


def validate_canonical_stage_critical_config(manifest: Mapping[str, Any]) -> None:
    """Reject a provenance-tagged stage whose actual critical config drifted."""

    resolved = manifest.get("resolved_config")
    critical = resolved.get("critical") if isinstance(resolved, dict) else None
    invocation = manifest.get("invocation")
    if not isinstance(critical, dict) or not isinstance(invocation, dict):
        raise TypeError("Canonical training provenance config is malformed")
    expected = {
        "num_envs": 2048,
        "environment_seed": 42,
        "runner_seed": 42,
        "num_steps_per_env": 24,
        "save_interval": 100,
        "logger": "tensorboard",
        "upload_model": False,
        "wrapper_clip_actions": None,
        "checkpoint_consumer_mode": False,
    }
    mismatches = [
        f"{key}={critical.get(key)!r} (expected {value!r})"
        for key, value in expected.items()
        if critical.get(key) != value
    ]
    if (
        manifest.get("schema_version")
        != MICROBAN_TELEOP_TRAINING_PROVENANCE_SCHEMA_VERSION
    ):
        mismatches.append("training provenance schema is not contract-v11 schema 3")
    if manifest.get("training_contract_version") != "11":
        mismatches.append("training contract version is not 11")

    mode = invocation.get("mode")
    start_boundary = invocation.get("stage_start_boundary")
    target_boundary = invocation.get("stage_target_boundary")
    safe_path = critical.get("safe_velocity_checkpoint")
    safe_sha256 = critical.get("safe_velocity_checkpoint_sha256")
    safe_receipt = critical.get("safe_velocity_acceptance_receipt")
    safe_values = (safe_path, safe_sha256, safe_receipt)
    if any(value is None for value in safe_values) != all(
        value is None for value in safe_values
    ):
        mismatches.append("safe-velocity source identity must be complete or absent")

    runner = resolved.get("runner") if isinstance(resolved, dict) else None
    algorithm = runner.get("algorithm") if isinstance(runner, dict) else None
    if not isinstance(algorithm, dict):
        mismatches.append("resolved runner algorithm config is malformed")
    else:
        if algorithm.get("learning_rate") != MICROBAN_TELEOP_V11_FIXED_LEARNING_RATE:
            mismatches.append(
                "algorithm learning_rate must be fixed at "
                f"{MICROBAN_TELEOP_V11_FIXED_LEARNING_RATE!r}"
            )
        if algorithm.get("schedule") != "fixed":
            mismatches.append("algorithm schedule must be 'fixed'")
        if algorithm.get("entropy_coef") != MICROBAN_TELEOP_V11_ENTROPY_COEF:
            mismatches.append(
                f"algorithm entropy_coef must be {MICROBAN_TELEOP_V11_ENTROPY_COEF!r}"
            )
        if (
            algorithm.get("num_learning_epochs")
            != MICROBAN_TELEOP_V11_NUM_LEARNING_EPOCHS
        ):
            mismatches.append(
                "algorithm num_learning_epochs must be "
                f"{MICROBAN_TELEOP_V11_NUM_LEARNING_EPOCHS}"
            )
    resume_source_path = invocation.get("resume_source_checkpoint_path")
    resume_source_sha256 = invocation.get("resume_source_checkpoint_sha256")
    resume_source_iteration = invocation.get("resume_source_checkpoint_iteration")
    resume_source_values = (
        resume_source_path,
        resume_source_sha256,
        resume_source_iteration,
    )
    resume = critical.get("resume")
    if not isinstance(resume, bool):
        mismatches.append("resume must be boolean")
    if isinstance(start_boundary, int) and start_boundary > 0 and resume is not True:
        mismatches.append("noninitial canonical v11 stages must resume full state")
    if resume is False and start_boundary == 0:
        if any(value is None for value in safe_values):
            mismatches.append(
                "fresh canonical v11 stage requires the complete safe-velocity source"
            )
        if critical.get("save_pristine_checkpoint") not in (True, False):
            mismatches.append("save_pristine_checkpoint must be boolean")
    elif critical.get("save_pristine_checkpoint") is not False:
        mismatches.append("only a fresh v11 stage may save a pristine checkpoint")
    if (
        resume is True
        and start_boundary == 0
        and any(value is None for value in safe_values)
    ):
        # The current invocation omits mutable safe-source arguments. The runner
        # fills these three values from the authenticated checkpoint before this
        # validator is called.
        mismatches.append(
            "interrupted initial-stage resume did not inherit safe-source identity"
        )
    if (
        isinstance(start_boundary, int)
        and start_boundary > 0
        and any(value is not None for value in safe_values)
    ):
        mismatches.append(
            "post-3000 stages must carry safe-source identity in checkpoint infos only"
        )
    if resume is True:
        if (
            not isinstance(resume_source_path, str)
            or not Path(resume_source_path).is_absolute()
        ):
            mismatches.append("resume source checkpoint path must be absolute")
        if (
            not isinstance(resume_source_sha256, str)
            or len(resume_source_sha256) != _SHA256_HEX_LENGTH
            or any(
                character not in "0123456789abcdef"
                for character in resume_source_sha256
            )
        ):
            mismatches.append("resume source checkpoint SHA-256 is malformed")
        if (
            not isinstance(resume_source_iteration, int)
            or isinstance(resume_source_iteration, bool)
            or resume_source_iteration < 0
        ):
            mismatches.append("resume source checkpoint iteration is invalid")
        if (
            isinstance(start_boundary, int)
            and isinstance(target_boundary, int)
            and isinstance(resume_source_iteration, int)
            and not isinstance(resume_source_iteration, bool)
        ):
            completed = resume_source_iteration + 1
            if not start_boundary <= completed < target_boundary:
                mismatches.append(
                    "resume source checkpoint iteration is outside the stage"
                )
            if completed == start_boundary and resume_source_sha256 != invocation.get(
                "parent_checkpoint_sha256"
            ):
                mismatches.append(
                    "boundary resume source SHA-256 differs from stage parent"
                )
    elif any(value is not None for value in resume_source_values):
        mismatches.append("fresh stages cannot claim a resume source checkpoint")
    max_iterations = critical.get("max_iterations_for_process")
    if (
        not isinstance(max_iterations, int)
        or isinstance(max_iterations, bool)
        or max_iterations <= 0
    ):
        mismatches.append("max_iterations_for_process must be a positive integer")
    elif resume is False and start_boundary == 0 and isinstance(target_boundary, int):
        allowed_iterations = {target_boundary}
        if target_boundary > 100:
            allowed_iterations.add(100)
        if max_iterations not in allowed_iterations:
            mismatches.append(
                "fresh max_iterations_for_process must be the complete initial "
                "stage or the canonical 100-update canary"
            )
    elif (
        isinstance(target_boundary, int)
        and isinstance(resume_source_iteration, int)
        and not isinstance(resume_source_iteration, bool)
        and max_iterations > target_boundary - (resume_source_iteration + 1)
    ):
        mismatches.append("max_iterations_for_process crosses the stage boundary")
    elif (
        isinstance(target_boundary, int)
        and isinstance(resume_source_iteration, int)
        and not isinstance(resume_source_iteration, bool)
    ):
        remaining_iterations = target_boundary - (resume_source_iteration + 1)
        allowed_iterations = {remaining_iterations}
        if remaining_iterations > 100:
            allowed_iterations.add(100)
        if max_iterations not in allowed_iterations:
            mismatches.append(
                "max_iterations_for_process must be either the complete remaining "
                "stage or the canonical 100-update canary"
            )

    if mode != MICROBAN_TELEOP_CANONICAL_STAGE_MODE:
        mismatches.append(f"mode={mode!r}")
    try:
        _validate_canonical_stage_lineage(
            mode=mode,
            start_boundary=invocation.get("stage_start_boundary"),
            target_boundary=invocation.get("stage_target_boundary"),
            parent_checkpoint_sha256=invocation.get("parent_checkpoint_sha256"),
            parent_gate_sha256=invocation.get("parent_gate_sha256"),
        )
    except (TypeError, ValueError) as exc:
        mismatches.append(str(exc))

    if manifest.get("migration_source") is not None:
        mismatches.append("contract-v11 forbids the retired v10 migration_source")
    if mismatches:
        raise ValueError(
            "Checkpoint is not a canonical v11 stage config: " + "; ".join(mismatches)
        )
