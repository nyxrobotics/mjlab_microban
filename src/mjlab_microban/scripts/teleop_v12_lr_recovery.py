"""Authenticate the one-time model9200 L/R-order recovery route.

The ordinary contract-v12 stage driver intentionally accepts only checkpoints
that already passed the preceding canonical gate.  The bilateral site-order
bug was discovered while the 7100->10000 hand stage was running, so its repair
needs one deliberately narrow exception: migrate the hash-pinned raw
``model_9200.pt``, preserve its training clock, and replay exactly 799 updates
to the unchanged 10000-update boundary.

This module is CPU-only.  It independently reconstructs the migration from the
pinned source, compares the complete checkpoint payload, authenticates the
migration receipt, and emits a second receipt binding that migrated checkpoint
to the fixed recovery route.  It does not relax any ordinary stage gate or
deployment/export check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from mjlab_microban.scripts.migrate_teleop_v12_lr_order import (
    ACTOR_PERMUTATION,
    CRITIC_PERMUTATION,
    MIGRATION_INFO_KEY,
    MIGRATION_REVISION,
    migrate_checkpoint_payload,
)

from mjlab_microban.legacy_velocity_diagnostics import (
    checkpoint_sha256,
    publish_json_atomic,
)
from mjlab_microban.tasks.mdp import MICROBAN_BILATERAL_SITE_ORDER_REVISION
from mjlab_microban.tasks.microban_teleop_v12_actor import (
    TELEOP_V12_FOOT_OBSERVATION_COLUMNS,
    TELEOP_V12_HAND_OBSERVATION_COLUMNS,
    TELEOP_V12_HMD_OBSERVATION_COLUMNS,
)

RECOVERY_SCHEMA_VERSION = 1
RECOVERY_GATE = "microban_teleop_v12_lr_order_recovery_seed"
RECOVERY_REVISION = "pinned_model9200_lr_swap_replay_to10000_v1"

PINNED_RAW_MODEL_9200_SHA256 = (
    "16c9b9d19df6513851b3da26228ae612512fdb2d894542741f0474e4762691c7"
)
PINNED_SOURCE_ITERATION = 9_200
PINNED_SOURCE_COMPLETED_UPDATES = PINNED_SOURCE_ITERATION + 1
PINNED_SOURCE_COMMON_STEP_COUNTER = PINNED_SOURCE_COMPLETED_UPDATES * 24
RECOVERY_TARGET_ITERATION = 9_999
RECOVERY_TARGET_COMPLETED_UPDATES = RECOVERY_TARGET_ITERATION + 1
RECOVERY_PROCESS_UPDATES = (
    RECOVERY_TARGET_COMPLETED_UPDATES - PINNED_SOURCE_COMPLETED_UPDATES
)
if RECOVERY_PROCESS_UPDATES != 799:
    raise RuntimeError("The model9200 recovery route must contain exactly 799 updates")

EXPECTED_ACTIVE_ACTOR_COLUMNS = (
    *TELEOP_V12_HMD_OBSERVATION_COLUMNS,
    *TELEOP_V12_HAND_OBSERVATION_COLUMNS,
)
EXPECTED_MIGRATED_FILENAME = f"model_{PINNED_SOURCE_ITERATION}.pt"


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object contains duplicate key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"JSON contains non-finite numeric constant: {value}")


def _load_json_object(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    with resolved.open("r", encoding="utf-8") as stream:
        value = json.load(
            stream,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {resolved}")
    return value


def _load_checkpoint(path: Path) -> dict[str, Any]:
    value = torch.load(
        path.expanduser().resolve(strict=True),
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(value, dict):
        raise TypeError(f"Checkpoint root must be a dictionary: {path}")
    return value


def _assert_deep_equal(expected: Any, actual: Any, *, path: str = "root") -> None:
    """Compare a migrated payload without trusting its serialized receipt."""

    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            raise TypeError(f"{path} changed from tensor to {type(actual).__name__}")
        if (
            expected.dtype != actual.dtype
            or expected.shape != actual.shape
            or not torch.equal(expected, actual)
        ):
            raise ValueError(f"{path} differs from the independently migrated tensor")
        return
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"{path} dictionary keys differ from the expected payload")
        for key, value in expected.items():
            _assert_deep_equal(value, actual[key], path=f"{path}.{key}")
        return
    if isinstance(expected, (list, tuple)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise ValueError(f"{path} sequence shape/type differs")
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            _assert_deep_equal(left, right, path=f"{path}[{index}]")
        return
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{path} differs: {actual!r} != {expected!r}")


def _clock(payload: dict[str, Any]) -> dict[str, int]:
    iteration = payload.get("iter")
    infos = payload.get("infos")
    if not isinstance(iteration, int) or isinstance(iteration, bool):
        raise TypeError("Checkpoint iteration is malformed")
    if not isinstance(infos, dict) or not isinstance(infos.get("env_state"), dict):
        raise TypeError("Checkpoint environment state is missing")
    common_step_counter = infos["env_state"].get("common_step_counter")
    if not isinstance(common_step_counter, int) or isinstance(
        common_step_counter, bool
    ):
        raise TypeError("Checkpoint common_step_counter is malformed")
    return {
        "iteration": iteration,
        "completed_updates": iteration + 1,
        "common_step_counter": common_step_counter,
    }


def _optimizer_state_for_shape(
    payload: dict[str, Any], shape: tuple[int, int]
) -> tuple[int, dict[str, Any]]:
    optimizer = payload.get("optimizer_state_dict")
    states = None if not isinstance(optimizer, dict) else optimizer.get("state")
    if not isinstance(states, dict):
        raise TypeError("Checkpoint optimizer state is missing")
    candidates: list[tuple[int, dict[str, Any]]] = []
    for parameter_id, state in states.items():
        if not isinstance(parameter_id, int) or not isinstance(state, dict):
            continue
        first = state.get("exp_avg")
        second = state.get("exp_avg_sq")
        if (
            isinstance(first, torch.Tensor)
            and isinstance(second, torch.Tensor)
            and tuple(first.shape) == shape
            and tuple(second.shape) == shape
        ):
            candidates.append((parameter_id, state))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one Adam state with shape {shape}, got {len(candidates)}"
        )
    return candidates[0]


def _assert_foot_adapter_exact_zero(payload: dict[str, Any]) -> int:
    actor = payload.get("actor_state_dict")
    if not isinstance(actor, dict):
        raise TypeError("Checkpoint actor state is missing")
    first = actor.get("mlp.0.weight")
    if not isinstance(first, torch.Tensor) or tuple(first.shape) != (512, 83):
        raise ValueError("Checkpoint actor W0 must have shape [512, 83]")
    foot = first[:, TELEOP_V12_FOOT_OBSERVATION_COLUMNS]
    if not torch.equal(foot, torch.zeros_like(foot)):
        raise ValueError("Pre-foot recovery actor columns are not exact zero")
    parameter_id, state = _optimizer_state_for_shape(payload, (512, 83))
    for name in ("exp_avg", "exp_avg_sq"):
        moment = state.get(name)
        if not isinstance(moment, torch.Tensor):
            raise TypeError(f"Actor Adam {name} is missing")
        foot_moment = moment[:, TELEOP_V12_FOOT_OBSERVATION_COLUMNS]
        if not torch.equal(foot_moment, torch.zeros_like(foot_moment)):
            raise ValueError(f"Pre-foot recovery actor Adam {name} is not exact zero")
    return parameter_id


def _expected_migration_receipt(
    *,
    source: Path,
    migrated: Path,
    source_sha256: str,
    migrated_sha256: str,
    source_payload: dict[str, Any],
    migrated_payload: dict[str, Any],
    transform: dict[str, Any],
) -> dict[str, Any]:
    source_clock = _clock(source_payload)
    migrated_clock = _clock(migrated_payload)
    serialized_source_clock = {
        "iteration": source_clock["iteration"],
        "common_step_counter": source_clock["common_step_counter"],
    }
    serialized_migrated_clock = {
        "iteration": migrated_clock["iteration"],
        "common_step_counter": migrated_clock["common_step_counter"],
    }
    return {
        "schema_version": 1,
        "gate": "microban_teleop_v12_lr_order_migration",
        "status": "pass",
        "migration_revision": MIGRATION_REVISION,
        "strategy": "swap",
        "source_checkpoint": {
            "path": str(source),
            "sha256": source_sha256,
            "size_bytes": source.stat().st_size,
            **serialized_source_clock,
        },
        "output_checkpoint": {
            "path": str(migrated),
            "sha256": migrated_sha256,
            "size_bytes": migrated.stat().st_size,
            **serialized_migrated_clock,
        },
        "clock_preservation": {
            "source": serialized_source_clock,
            "output": serialized_migrated_clock,
            "passed": True,
        },
        "transform": transform,
    }


def validate_recovery_seed(
    *,
    source_checkpoint: Path,
    migrated_checkpoint: Path,
    migration_receipt: Path,
    expected_source_sha256: str | None = None,
) -> dict[str, Any]:
    """Return a route receipt only after independently proving every transform."""

    expected_source_sha256 = (
        PINNED_RAW_MODEL_9200_SHA256
        if expected_source_sha256 is None
        else expected_source_sha256
    )
    source = source_checkpoint.expanduser().resolve(strict=True)
    migrated = migrated_checkpoint.expanduser().resolve(strict=True)
    receipt_path = migration_receipt.expanduser().resolve(strict=True)
    if len({source, migrated, receipt_path}) != 3:
        raise ValueError("Source, migrated checkpoint, and receipt must be distinct")
    if migrated.name != EXPECTED_MIGRATED_FILENAME:
        raise ValueError(
            f"Recovery seed must be named {EXPECTED_MIGRATED_FILENAME!r}"
        )

    source_sha256 = checkpoint_sha256(source)
    if source_sha256 != expected_source_sha256:
        raise ValueError(f"Pinned raw model9200 SHA-256 mismatch: {source_sha256}")
    migrated_sha256 = checkpoint_sha256(migrated)
    source_payload = _load_checkpoint(source)
    migrated_payload = _load_checkpoint(migrated)
    source_clock = _clock(source_payload)
    migrated_clock = _clock(migrated_payload)
    expected_clock = {
        "iteration": PINNED_SOURCE_ITERATION,
        "completed_updates": PINNED_SOURCE_COMPLETED_UPDATES,
        "common_step_counter": PINNED_SOURCE_COMMON_STEP_COUNTER,
    }
    if source_clock != expected_clock or migrated_clock != expected_clock:
        raise ValueError(
            f"Recovery clock must remain {expected_clock}; "
            f"source={source_clock}, migrated={migrated_clock}"
        )

    independently_migrated, transform = migrate_checkpoint_payload(
        source_payload,
        source_path=source,
        source_sha256=source_sha256,
        strategy="swap",
    )
    _assert_deep_equal(
        independently_migrated,
        migrated_payload,
        path="migrated_checkpoint",
    )
    actual_migration_receipt = _load_json_object(receipt_path)
    expected_migration_receipt = _expected_migration_receipt(
        source=source,
        migrated=migrated,
        source_sha256=source_sha256,
        migrated_sha256=migrated_sha256,
        source_payload=source_payload,
        migrated_payload=migrated_payload,
        transform=transform,
    )
    _assert_deep_equal(
        expected_migration_receipt,
        actual_migration_receipt,
        path="migration_receipt",
    )

    infos = migrated_payload.get("infos")
    if not isinstance(infos, dict):
        raise TypeError("Migrated checkpoint infos are missing")
    marker = infos.get(MIGRATION_INFO_KEY)
    if not isinstance(marker, dict):
        raise TypeError("Migrated checkpoint lacks the L/R migration marker")
    if (
        marker.get("revision") != MIGRATION_REVISION
        or marker.get("site_order_revision")
        != MICROBAN_BILATERAL_SITE_ORDER_REVISION
        or marker.get("strategy") != "swap"
    ):
        raise ValueError("Migrated checkpoint marker is not the required L/R swap")
    active_columns = infos.get("active_actor_columns_at_save")
    if active_columns != list(EXPECTED_ACTIVE_ACTOR_COLUMNS):
        raise ValueError("model9200 active adapter columns drifted")
    actor_optimizer_id = _assert_foot_adapter_exact_zero(migrated_payload)
    if marker.get("actor_w0_optimizer_parameter_id") != actor_optimizer_id:
        raise ValueError("Actor Adam parameter identity drifted from migration marker")

    checks = {
        "pinned_source_sha256": True,
        "complete_payload_matches_independent_swap": True,
        "migration_receipt_matches": True,
        "clock_preserved_at_completed_9201": True,
        "actor_critic_and_adam_permutations_authenticated": True,
        "foot_actor_and_adam_exact_zero": True,
        "corrected_site_order_revision": True,
        "canonical_replay_length_799": True,
    }
    return {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "gate": RECOVERY_GATE,
        "status": "pass",
        "recovery_revision": RECOVERY_REVISION,
        "site_order_revision": MICROBAN_BILATERAL_SITE_ORDER_REVISION,
        "source_checkpoint": {
            "path": str(source),
            "sha256": source_sha256,
            "size_bytes": source.stat().st_size,
            **source_clock,
        },
        "migrated_checkpoint": {
            "path": str(migrated),
            "sha256": migrated_sha256,
            "size_bytes": migrated.stat().st_size,
            **migrated_clock,
        },
        "migration_receipt": {
            "path": str(receipt_path),
            "sha256": checkpoint_sha256(receipt_path),
            "migration_revision": MIGRATION_REVISION,
            "strategy": "swap",
        },
        "permutations": {
            "actor": list(ACTOR_PERMUTATION),
            "critic": list(CRITIC_PERMUTATION),
            "actor_optimizer_parameter_id": actor_optimizer_id,
        },
        "route": {
            "task": "Mjlab-Teleop-V12-Microban",
            "source_iteration": PINNED_SOURCE_ITERATION,
            "source_completed_updates": PINNED_SOURCE_COMPLETED_UPDATES,
            "source_common_step_counter": PINNED_SOURCE_COMMON_STEP_COUNTER,
            "target_iteration": RECOVERY_TARGET_ITERATION,
            "target_completed_updates": RECOVERY_TARGET_COMPLETED_UPDATES,
            "process_updates": RECOVERY_PROCESS_UPDATES,
            "num_envs": 2_048,
            "environment_seed": 42,
            "agent_seed": 42,
            "num_steps_per_env": 24,
            "save_interval": 100,
            "logger": "tensorboard",
            "upload_model": False,
            "nan_guard": True,
            "foot_adapter_columns": list(TELEOP_V12_FOOT_OBSERVATION_COLUMNS),
            "foot_state_at_source": "inactive_exact_zero",
            "foot_activation": "only_after_completed_update_10000",
        },
        "checks": checks,
    }


def create_recovery_receipt(
    *,
    source_checkpoint: Path,
    migrated_checkpoint: Path,
    migration_receipt: Path,
    output: Path,
    force: bool,
    expected_source_sha256: str | None = None,
) -> dict[str, Any]:
    destination = output.expanduser().resolve()
    if destination.exists() and not force:
        raise FileExistsError(f"Recovery receipt already exists: {destination}")
    if destination.is_symlink():
        raise ValueError("Recovery receipt output must not be a symlink")
    report = validate_recovery_seed(
        source_checkpoint=source_checkpoint,
        migrated_checkpoint=migrated_checkpoint,
        migration_receipt=migration_receipt,
        expected_source_sha256=expected_source_sha256,
    )
    publish_json_atomic(destination, report)
    return report


def validate_recovery_receipt(
    *,
    recovery_receipt: Path,
    source_checkpoint: Path,
    migrated_checkpoint: Path,
    migration_receipt: Path,
    expected_source_sha256: str | None = None,
) -> dict[str, Any]:
    actual = _load_json_object(recovery_receipt)
    expected = validate_recovery_seed(
        source_checkpoint=source_checkpoint,
        migrated_checkpoint=migrated_checkpoint,
        migration_receipt=migration_receipt,
        expected_source_sha256=expected_source_sha256,
    )
    _assert_deep_equal(expected, actual, path="recovery_receipt")
    return actual


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("create", "validate"):
        command = subparsers.add_parser(name)
        if name == "validate":
            command.add_argument("recovery_receipt", type=Path)
        command.add_argument("--source", type=Path, required=True)
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--migration-receipt", type=Path, required=True)
        if name == "create":
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    common = {
        "source_checkpoint": args.source,
        "migrated_checkpoint": args.checkpoint,
        "migration_receipt": args.migration_receipt,
    }
    if args.command == "create":
        report = create_recovery_receipt(
            **common,
            output=args.output,
            force=args.force,
        )
    else:
        report = validate_recovery_receipt(
            **common,
            recovery_receipt=args.recovery_receipt,
        )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
