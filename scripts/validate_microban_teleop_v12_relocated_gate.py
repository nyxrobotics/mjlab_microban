#!/usr/bin/env python3
"""Validate an archived v12 gate in a relocated pinned checkout.

Historical v12 reports recorded absolute paths from the checkout that created
them.  This wrapper leaves those authenticated JSON bytes untouched while
mapping data-only artifact paths into the checkout that contains the pinned
validator implementation.
"""

from __future__ import annotations

import argparse
import importlib
import os
import stat
import sys
from collections.abc import Callable, Sequence
from pathlib import Path, PurePosixPath
from types import ModuleType

_REPO_PREFIX = "repo://"
_RESOLVER_NAME = "resolve_bootstrap_artifact_path"
_PORTABLE_NAME = "portable_bootstrap_artifact_path"


class RelocationError(ValueError):
    """A recorded artifact path cannot be safely relocated."""


def _has_parent_reference(path: Path) -> bool:
    return any(part == ".." for part in path.parts)


def _reject_existing_symlink_components(path: Path) -> None:
    """Reject symlinks without requiring a possibly stale path to exist."""

    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            return
        if stat.S_ISLNK(mode):
            raise RelocationError(f"Symlink path component is forbidden: {current}")


def _absolute_root(value: str | Path, *, must_exist: bool) -> Path:
    raw = Path(value).expanduser()
    if not raw.is_absolute() or _has_parent_reference(raw):
        raise RelocationError(f"Root must be an absolute path without '..': {value}")
    _reject_existing_symlink_components(raw)
    normalized = Path(os.path.normpath(str(raw)))
    if must_exist:
        try:
            mode = normalized.lstat().st_mode
        except FileNotFoundError as exc:
            raise RelocationError(
                f"Checkout root does not exist: {normalized}"
            ) from exc
        if not stat.S_ISDIR(mode):
            raise RelocationError(f"Checkout root is not a directory: {normalized}")
    return normalized


def _repo_relative(value: str) -> Path:
    encoded = value.removeprefix(_REPO_PREFIX)
    pure = PurePosixPath(encoded)
    if (
        not encoded
        or encoded.startswith("/")
        or "\\" in encoded
        or pure.is_absolute()
        or any(part in ("", ".", "..") for part in encoded.split("/"))
        or pure.as_posix() != encoded
    ):
        raise RelocationError(f"Invalid repo-relative artifact path: {value}")
    return Path(*pure.parts)


def _is_allowed_data_path(relative: Path) -> bool:
    parts = relative.parts
    pinned_bootstrap = parts in {
        ("checkpoints", "xc330_velocity", "model_14999.pt"),
        (
            "artifacts",
            "legacy_teleop_probe",
            "model_14999_teleop83_raw_9x300.json",
        ),
    }
    artifacts = (
        len(parts) >= 3
        and parts[0] == "artifacts"
        and parts[1].startswith("teleop_v12_")
        and parts[1] != "teleop_v12_"
    )
    training_log = len(parts) >= 4 and parts[:3] == (
        "logs",
        "rsl_rl",
        "mjlab_microban_teleop_v12",
    )
    return pinned_bootstrap or artifacts or training_log


class ArtifactRelocator:
    """Fail-closed mapper from authenticated recorded paths to one checkout."""

    def __init__(
        self,
        checkout_root: str | Path,
        recorded_roots: Sequence[str | Path] = (),
    ) -> None:
        self.checkout_root = _absolute_root(checkout_root, must_exist=True)
        roots = [_absolute_root(root, must_exist=False) for root in recorded_roots]
        self.recorded_roots = tuple(dict.fromkeys(roots))

    def _checked_file(self, relative: Path) -> Path:
        if relative.is_absolute() or _has_parent_reference(relative):
            raise RelocationError(f"Artifact path escapes checkout: {relative}")
        if not _is_allowed_data_path(relative):
            raise RelocationError(
                f"Artifact path is outside data allowlist: {relative}"
            )

        candidate = self.checkout_root / relative
        current = self.checkout_root
        for part in relative.parts:
            current /= part
            try:
                mode = current.lstat().st_mode
            except FileNotFoundError as exc:
                raise RelocationError(
                    f"Relocated artifact does not exist: {candidate}"
                ) from exc
            if stat.S_ISLNK(mode):
                raise RelocationError(
                    f"Relocated artifact traverses a symlink: {current}"
                )
        if not stat.S_ISREG(mode):
            raise RelocationError(
                f"Relocated artifact is not a regular file: {candidate}"
            )

        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(self.checkout_root)
        except ValueError as exc:
            raise RelocationError(
                f"Relocated artifact escapes checkout: {candidate}"
            ) from exc
        if resolved != candidate:
            raise RelocationError(f"Relocated artifact is not canonical: {candidate}")
        return candidate

    def _absolute_relative(self, path: Path) -> Path:
        if _has_parent_reference(path):
            raise RelocationError(f"Absolute artifact path contains '..': {path}")
        normalized = Path(os.path.normpath(str(path)))
        roots = (self.checkout_root, *self.recorded_roots)
        matches: list[Path] = []
        for root in roots:
            try:
                matches.append(normalized.relative_to(root))
            except ValueError:
                continue
        if not matches:
            raise RelocationError(
                f"Absolute artifact path is outside checkout and recorded roots: {path}"
            )
        distinct = {match.as_posix() for match in matches}
        if len(distinct) != 1:
            raise RelocationError(f"Absolute artifact path has ambiguous roots: {path}")
        return matches[0]

    def resolve(self, value: str | Path) -> Path:
        text = str(value)
        if text.startswith(_REPO_PREFIX):
            relative = _repo_relative(text)
        else:
            path = Path(value).expanduser()
            if not path.is_absolute():
                raise RelocationError(
                    "Artifact path must be repo:// or an approved absolute path: "
                    f"{value}"
                )
            relative = self._absolute_relative(path)
        return self._checked_file(relative)

    def portable(self, value: str | Path) -> str:
        resolved = self.resolve(value)
        relative = resolved.relative_to(self.checkout_root)
        return f"{_REPO_PREFIX}{relative.as_posix()}"


def _module_is_from_checkout(module: ModuleType, checkout_root: Path) -> bool:
    filename = getattr(module, "__file__", None)
    if filename is None:
        return True
    source = Path(filename)
    if not source.is_absolute():
        return False
    try:
        source_root = (checkout_root / "src").resolve(strict=True)
        resolved = source.resolve(strict=True)
        resolved.relative_to(source_root)
    except (FileNotFoundError, ValueError):
        return False
    return resolved == source


def _patch_loaded_modules(
    resolver: Callable[[str | Path], Path],
    portable: Callable[[str | Path], str],
) -> None:
    for name, module in tuple(sys.modules.items()):
        if module is None or not (
            name == "mjlab_microban" or name.startswith("mjlab_microban.")
        ):
            continue
        if hasattr(module, _RESOLVER_NAME):
            setattr(module, _RESOLVER_NAME, resolver)
        if hasattr(module, _PORTABLE_NAME):
            setattr(module, _PORTABLE_NAME, portable)


def install_relocated_resolvers(
    relocator: ArtifactRelocator,
) -> ModuleType:
    """Patch bootstrap before stage import, then patch every copied alias."""

    resolver = relocator.resolve
    portable = relocator.portable
    bootstrap = importlib.import_module(
        "mjlab_microban.tasks.microban_teleop_v12_bootstrap"
    )
    if not _module_is_from_checkout(bootstrap, relocator.checkout_root):
        raise RelocationError(
            "Bootstrap resolver was not imported from pinned checkout"
        )
    setattr(bootstrap, _RESOLVER_NAME, resolver)
    setattr(bootstrap, _PORTABLE_NAME, portable)

    stage = importlib.import_module("mjlab_microban.scripts.teleop_v12_stage")
    _patch_loaded_modules(resolver, portable)
    for name, module in tuple(sys.modules.items()):
        if module is None or not (
            name == "mjlab_microban" or name.startswith("mjlab_microban.")
        ):
            continue
        if not _module_is_from_checkout(module, relocator.checkout_root):
            raise RelocationError(
                f"Module was not imported from pinned checkout: {name}"
            )
    return stage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout-root", type=Path, required=True)
    parser.add_argument(
        "--recorded-root",
        action="append",
        default=[],
        type=Path,
        help="historical checkout root recorded in archived JSON (repeatable)",
    )
    # Keep these as strings so argparse does not collapse ``repo://`` to
    # ``repo:/`` before the fail-closed resolver sees it.
    parser.add_argument("gate")
    parser.add_argument("checkpoint")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    relocator = ArtifactRelocator(args.checkout_root, args.recorded_root)
    gate = relocator.resolve(args.gate)
    checkpoint = relocator.resolve(args.checkpoint)
    stage = install_relocated_resolvers(relocator)
    result = stage.validate_gate(gate, checkpoint)
    if not isinstance(result, dict) or result.get("status") != "pass":
        raise RelocationError("Pinned v12 validator did not return a pass gate")
    print(f"MICROBAN_TELEOP_V12_RELOCATED_GATE=PASS gate={relocator.portable(gate)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
