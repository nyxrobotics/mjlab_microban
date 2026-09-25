#!/usr/bin/env python3
"""Rebuild the archived v12 bilateral migration in a relocated checkout."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType


def _load_relocator_module() -> ModuleType:
    path = Path(__file__).with_name("validate_microban_teleop_v12_relocated_gate.py")
    spec = importlib.util.spec_from_file_location("microban_v12_relocator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load relocation support: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _top_level_yaml_scalars(path: Path) -> dict[str, object]:
    result: dict[str, object] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line[0].isspace() or ":" not in line:
            continue
        key, raw = line.split(":", 1)
        value = raw.strip()
        if not value:
            continue
        if value == "true":
            parsed: object = True
        elif value == "false":
            parsed = False
        elif value == "null":
            parsed = None
        else:
            try:
                parsed = int(value)
            except ValueError:
                parsed = value
        result[key] = parsed
    return result


def _require_run_context(path: Path, expected: Mapping[str, object]) -> None:
    actual = _top_level_yaml_scalars(path)
    mismatches = {
        key: (actual.get(key), value)
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Training run context drifted: {path}: {mismatches}")


def _checkpoint_clock(payload: Mapping[str, object]) -> tuple[int, int]:
    iteration = payload.get("iter")
    infos = payload.get("infos")
    env_state = infos.get("env_state") if isinstance(infos, Mapping) else None
    common_step = (
        env_state.get("common_step_counter") if isinstance(env_state, Mapping) else None
    )
    if (
        not isinstance(iteration, int)
        or isinstance(iteration, bool)
        or not isinstance(common_step, int)
        or isinstance(common_step, bool)
        or common_step != (iteration + 1) * 24
    ):
        raise ValueError("Checkpoint iteration/common-step clock is malformed")
    return iteration, common_step


def _optimizer_step(payload: Mapping[str, object]) -> int:
    optimizer = payload.get("optimizer_state_dict")
    states = optimizer.get("state") if isinstance(optimizer, Mapping) else None
    if not isinstance(states, Mapping) or not states:
        raise ValueError("Checkpoint optimizer state is missing")
    steps: set[int] = set()
    for state in states.values():
        step = state.get("step") if isinstance(state, Mapping) else None
        if hasattr(step, "item"):
            step = step.item()
        if (
            not isinstance(step, (int, float))
            or isinstance(step, bool)
            or not math.isfinite(float(step))
            or int(step) != step
        ):
            raise ValueError("Checkpoint optimizer step is malformed")
        steps.add(int(step))
    if len(steps) != 1:
        raise ValueError(f"Checkpoint optimizer steps disagree: {sorted(steps)}")
    return steps.pop()


def _require_training_transition(
    before: Mapping[str, object],
    after: Mapping[str, object],
    *,
    updates: int,
    label: str,
) -> None:
    import torch

    active_columns = {6, 7, 8, 27, 28, 29, 75, 76, 77, 78, 79, 80, 81, 82}
    foot_columns = tuple(range(69, 75))

    before_iteration, before_common = _checkpoint_clock(before)
    after_iteration, after_common = _checkpoint_clock(after)
    if (
        after_iteration - before_iteration != updates
        or after_common - before_common != updates * 24
        or _optimizer_step(after) - _optimizer_step(before) != updates * 20
    ):
        raise ValueError(f"{label} training clock/optimizer delta drifted")
    before_actor = before.get("actor_state_dict")
    after_actor = after.get("actor_state_dict")
    if (
        not isinstance(before_actor, Mapping)
        or not isinstance(after_actor, Mapping)
        or set(before_actor) != set(after_actor)
    ):
        raise ValueError(f"{label} actor tensor inventory drifted")
    changed: set[str] = set()
    for name in before_actor:
        left = before_actor[name]
        right = after_actor[name]
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise TypeError(f"{label} actor value {name!r} is not a tensor")
        if not torch.equal(left, right):
            changed.add(name)
    if changed != {"mlp.0.weight"}:
        raise ValueError(f"{label} changed unexpected actor tensors: {sorted(changed)}")
    before_w0 = before_actor["mlp.0.weight"]
    after_w0 = after_actor["mlp.0.weight"]
    changed_columns = set(
        torch.nonzero(torch.any(before_w0 != after_w0, dim=0), as_tuple=False)
        .flatten()
        .tolist()
    )
    if changed_columns != active_columns:
        raise ValueError(f"{label} actor W0 changed-column set drifted")
    for value in (before_w0[:, foot_columns], after_w0[:, foot_columns]):
        if not torch.equal(value, torch.zeros_like(value)):
            raise ValueError(f"{label} foot actor columns are not exact zero")

    before_optimizer = before["optimizer_state_dict"]
    after_optimizer = after["optimizer_state_dict"]
    if before_optimizer.get("param_groups") != after_optimizer.get("param_groups"):
        raise ValueError(f"{label} optimizer parameter groups drifted")
    before_states = before_optimizer.get("state")
    after_states = after_optimizer.get("state")
    expected_ids = {1, *range(9, 17)}
    if (
        not isinstance(before_states, Mapping)
        or not isinstance(after_states, Mapping)
        or set(before_states) != expected_ids
        or set(after_states) != expected_ids
    ):
        raise ValueError(f"{label} optimizer state inventory drifted")
    for moment_name in ("exp_avg", "exp_avg_sq"):
        before_moment = before_states[1].get(moment_name)
        after_moment = after_states[1].get(moment_name)
        if not isinstance(before_moment, torch.Tensor) or not isinstance(
            after_moment, torch.Tensor
        ):
            raise TypeError(f"{label} actor Adam moment is missing")
        moment_changed = set(
            torch.nonzero(
                torch.any(before_moment != after_moment, dim=0), as_tuple=False
            )
            .flatten()
            .tolist()
        )
        if moment_changed != active_columns:
            raise ValueError(f"{label} actor Adam changed-column set drifted")
        inactive = sorted(set(range(before_moment.shape[1])) - active_columns)
        for value in (before_moment[:, inactive], after_moment[:, inactive]):
            if not torch.equal(value, torch.zeros_like(value)):
                raise ValueError(f"{label} inactive actor Adam moments are not zero")
    for parameter_id in range(9, 17):
        before_state = before_states[parameter_id]
        after_state = after_states[parameter_id]
        if set(before_state) != set(after_state):
            raise ValueError(f"{label} critic Adam state keys drifted")
        for name in ("exp_avg", "exp_avg_sq"):
            left = before_state[name]
            right = after_state[name]
            if (
                not isinstance(left, torch.Tensor)
                or not isinstance(right, torch.Tensor)
                or left.shape != right.shape
                or not bool(torch.isfinite(left).all().item())
                or not bool(torch.isfinite(right).all().item())
            ):
                raise ValueError(f"{label} critic Adam tensor contract drifted")


def _migration_marker(payload: Mapping[str, object]) -> object:
    infos = payload.get("infos")
    if not isinstance(infos, Mapping):
        raise TypeError("Checkpoint infos are missing")
    return infos.get("microban_teleop_v12_lr_order_migration")


def _require_fixed_infos(
    before: Mapping[str, object],
    after: Mapping[str, object],
    *,
    allow_recipe_change: bool,
    label: str,
) -> None:
    before_infos = before.get("infos")
    after_infos = after.get("infos")
    if not isinstance(before_infos, Mapping) or not isinstance(after_infos, Mapping):
        raise TypeError(f"{label} checkpoint infos are missing")
    names = {
        "microban_teleop_training_contract_version",
        "previous_action_semantics",
        "action_clip",
        "trainable_actor_parameters",
        "trainable_actor_columns",
        "adapter_gradient_schedule_revision",
        "active_actor_columns_at_save",
        "legacy_velocity_actor_bootstrap_v12",
        "adapter_sanitization",
    }
    if not allow_recipe_change:
        names.add("microban_teleop_recipe_revision")
    drifted = sorted(
        name for name in names if before_infos.get(name) != after_infos.get(name)
    )
    if drifted:
        raise ValueError(f"{label} fixed checkpoint infos drifted: {drifted}")


def _portable_receipt_paths(
    value: object, relocator: object, *, parent_key: str | None = None
) -> object:
    if isinstance(value, Mapping):
        return {
            key: _portable_receipt_paths(nested, relocator, parent_key=key)
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [
            _portable_receipt_paths(nested, relocator, parent_key=parent_key)
            for nested in value
        ]
    if (
        isinstance(parent_key, str)
        and (parent_key == "path" or parent_key.endswith("_path"))
        and isinstance(value, str)
    ):
        return relocator.portable(value)
    return value


def _relocated_receipt_paths(
    value: object, relocator: object, *, parent_key: str | None = None
) -> object:
    if isinstance(value, Mapping):
        return {
            key: _relocated_receipt_paths(nested, relocator, parent_key=key)
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [
            _relocated_receipt_paths(nested, relocator, parent_key=parent_key)
            for nested in value
        ]
    if parent_key == "path" and isinstance(value, str):
        return str(relocator.resolve(value))
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout-root", type=Path, required=True)
    parser.add_argument("--recorded-root", action="append", default=[], type=Path)
    parser.add_argument("--source", required=True)
    parser.add_argument("--migrated", required=True)
    parser.add_argument("--migration-receipt", required=True)
    parser.add_argument("--recovery-receipt", required=True)
    parser.add_argument("--corner-parent", required=True)
    parser.add_argument("--corner-parent-tracking", required=True)
    parser.add_argument("--corner-superseded-tracking", required=True)
    parser.add_argument("--stage-7100", required=True)
    parser.add_argument("--selected-10000", required=True)
    parser.add_argument("--raw-agent-params", required=True)
    parser.add_argument("--replay-agent-params", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    support = _load_relocator_module()
    relocator = support.ArtifactRelocator(args.checkout_root, args.recorded_root)
    support.install_relocated_resolvers(relocator)

    recovery = importlib.import_module("mjlab_microban.scripts.teleop_v12_lr_recovery")
    corner = importlib.import_module("mjlab_microban.scripts.teleop_v12_corner_rescue")
    support._patch_loaded_modules(relocator.resolve, relocator.portable)
    for name, module in tuple(sys.modules.items()):
        if module is None or not (
            name == "mjlab_microban" or name.startswith("mjlab_microban.")
        ):
            continue
        if not support._module_is_from_checkout(module, relocator.checkout_root):
            raise support.RelocationError(
                f"Module was not imported from pinned checkout: {name}"
            )

    source = relocator.resolve(args.source)
    migrated = relocator.resolve(args.migrated)
    migration_receipt = relocator.resolve(args.migration_receipt)
    recovery_receipt = relocator.resolve(args.recovery_receipt)
    stage_7100 = relocator.resolve(args.stage_7100)
    selected_10000 = relocator.resolve(args.selected_10000)
    raw_agent_params = relocator.resolve(args.raw_agent_params)
    replay_agent_params = relocator.resolve(args.replay_agent_params)
    archived_migration = _load_json_object(migration_receipt)
    relocated_migration = _relocated_receipt_paths(archived_migration, relocator)
    original_loader = recovery._load_json_object
    original_migrate = recovery.migrate_checkpoint_payload
    source_identity = archived_migration.get("source_checkpoint")
    if not isinstance(source_identity, dict) or not isinstance(
        source_identity.get("path"), str
    ):
        raise TypeError("Archived migration source path is missing")
    recorded_source = Path(source_identity["path"])
    if relocator.resolve(recorded_source) != source:
        raise ValueError("Archived migration source path does not map to source bytes")

    def relocated_loader(path: Path) -> dict[str, object]:
        if path.expanduser().resolve(strict=True) == migration_receipt:
            assert isinstance(relocated_migration, dict)
            return relocated_migration
        return original_loader(path)

    def relocated_migrate(
        source_payload: dict[str, object],
        *,
        source_path: Path,
        source_sha256: str,
        strategy: str,
    ) -> tuple[dict[str, object], dict[str, object]]:
        if source_path != source:
            raise ValueError("Recovery validator used an unexpected source path")
        return original_migrate(
            source_payload,
            source_path=recorded_source,
            source_sha256=source_sha256,
            strategy=strategy,
        )

    recovery._load_json_object = relocated_loader
    recovery.migrate_checkpoint_payload = relocated_migrate
    try:
        rebuilt = recovery.validate_recovery_seed(
            source_checkpoint=source,
            migrated_checkpoint=migrated,
            migration_receipt=migration_receipt,
        )
    finally:
        recovery._load_json_object = original_loader
        recovery.migrate_checkpoint_payload = original_migrate
    archived = _load_json_object(recovery_receipt)
    if _portable_receipt_paths(rebuilt, relocator) != _portable_receipt_paths(
        archived, relocator
    ):
        raise ValueError("Relocated bilateral recovery receipt is not canonical")

    _require_run_context(
        raw_agent_params,
        {
            "seed": 42,
            "num_steps_per_env": 24,
            "max_iterations": 2900,
            "save_interval": 100,
            "experiment_name": "mjlab_microban_teleop_v12",
            "run_name": "v12_canonical_7100_to10000",
            "resume": True,
            "load_run": "^2026-09-25_22-14-43_v12_canonical_7000_to7100$",
            "load_checkpoint": "^model_7099[.]pt$",
            "upload_model": False,
            "simulation_preview_mode": False,
        },
    )
    _require_run_context(
        replay_agent_params,
        {
            "seed": 42,
            "num_steps_per_env": 24,
            "max_iterations": 799,
            "save_interval": 100,
            "experiment_name": "mjlab_microban_teleop_v12",
            "run_name": "v12_lrfix_9201_to10000",
            "resume": True,
            "load_run": "^2026-09-25_23-17-45_v12_lrfix_9200_replay$",
            "load_checkpoint": "^model_9200[.]pt$",
            "upload_model": False,
            "simulation_preview_mode": False,
        },
    )
    stage_7100_payload = recovery._load_checkpoint(stage_7100)
    raw_payload = recovery._load_checkpoint(source)
    migrated_payload = recovery._load_checkpoint(migrated)
    corner_parent_path = relocator.resolve(args.corner_parent)
    corner_parent_payload = recovery._load_checkpoint(corner_parent_path)
    selected_payload = recovery._load_checkpoint(selected_10000)
    _require_training_transition(
        stage_7100_payload,
        raw_payload,
        updates=2101,
        label="stage7100_to_raw9200",
    )
    _require_training_transition(
        migrated_payload,
        corner_parent_payload,
        updates=700,
        label="migrated9200_to_corner_parent9900",
    )
    _require_training_transition(
        corner_parent_payload,
        selected_payload,
        updates=99,
        label="corner_parent9900_to_selected9999",
    )
    _require_fixed_infos(
        stage_7100_payload,
        raw_payload,
        allow_recipe_change=False,
        label="stage7100_to_raw9200",
    )
    _require_fixed_infos(
        migrated_payload,
        corner_parent_payload,
        allow_recipe_change=False,
        label="migrated9200_to_corner_parent9900",
    )
    _require_fixed_infos(
        corner_parent_payload,
        selected_payload,
        allow_recipe_change=True,
        label="corner_parent9900_to_selected9999",
    )
    marker = _migration_marker(migrated_payload)
    if marker is None or marker != _migration_marker(corner_parent_payload):
        raise ValueError("Migration marker drifted before the corner parent")
    if marker != _migration_marker(selected_payload):
        raise ValueError("Migration marker drifted during corner rescue")

    parent = corner.validate_parent_checkpoint(
        corner_parent_path,
        relocator.resolve(args.corner_parent_tracking),
        relocator.resolve(args.corner_superseded_tracking),
    )
    if not isinstance(parent, dict) or parent.get("status") != "pass":
        raise ValueError("Relocated corner-rescue parent validation did not pass")
    print(
        "MICROBAN_TELEOP_V12_RELOCATED_MIGRATION=PASS "
        f"source={relocator.portable(source)} migrated={relocator.portable(migrated)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
