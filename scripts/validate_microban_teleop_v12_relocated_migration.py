#!/usr/bin/env python3
"""Rebuild the archived v12 bilateral migration in a relocated checkout."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
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

    parent = corner.validate_parent_checkpoint(
        relocator.resolve(args.corner_parent),
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
