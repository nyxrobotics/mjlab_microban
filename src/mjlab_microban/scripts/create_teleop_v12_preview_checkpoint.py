"""Create a non-deployable full-body preview by clock-lifting sanitized601."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import torch

from mjlab_microban.legacy_velocity_diagnostics import publish_json_atomic
from mjlab_microban.tasks.microban_teleop_v12_actor import (
    TELEOP_V12_EXTRA_OBSERVATION_COLUMNS,
)
from mjlab_microban.tasks.microban_teleop_v12_bootstrap import sha256_file
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MICROBAN_TELEOP_V12_RECIPE_REVISION,
)
from mjlab_microban.tasks.microban_teleop_v12_preview import (
    TELEOP_V12_PREVIEW_INFO_KEY,
    TELEOP_V12_PREVIEW_LIFTED_COMPLETED_UPDATES,
    TELEOP_V12_PREVIEW_LIFTED_ITERATION,
    TELEOP_V12_PREVIEW_SOURCE_COMPLETED_UPDATES,
    TELEOP_V12_PREVIEW_SOURCE_ITERATION,
    TELEOP_V12_PREVIEW_SOURCE_SHA256,
    canonical_preview_info,
    validate_preview_marker,
)
from mjlab_microban.tasks.microban_teleop_v12_runner import _atomic_torch_save


def create_preview_checkpoint(source: Path, destination: Path) -> dict[str, Any]:
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if source == destination:
        raise ValueError("Preview creator refuses to overwrite its source")
    if destination.exists():
        raise FileExistsError(f"Preview destination already exists: {destination}")
    source_sha = sha256_file(source)
    if source_sha != TELEOP_V12_PREVIEW_SOURCE_SHA256:
        raise ValueError(f"Preview source SHA-256 mismatch: {source_sha}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("infos"), dict):
        raise TypeError("Preview source checkpoint is malformed")
    infos = payload["infos"]
    if (
        payload.get("iter") != TELEOP_V12_PREVIEW_SOURCE_ITERATION
        or infos.get("env_state", {}).get("common_step_counter")
        != TELEOP_V12_PREVIEW_SOURCE_COMPLETED_UPDATES * 24
        or infos.get("microban_teleop_recipe_revision")
        != MICROBAN_TELEOP_V12_RECIPE_REVISION
        or infos.get("active_actor_columns_at_save") != []
        or not isinstance(infos.get("adapter_sanitization"), dict)
    ):
        raise ValueError("Preview source is not canonical sanitized601")
    first = payload.get("actor_state_dict", {}).get("mlp.0.weight")
    if not isinstance(first, torch.Tensor) or not torch.equal(
        first[:, TELEOP_V12_EXTRA_OBSERVATION_COLUMNS],
        torch.zeros_like(first[:, TELEOP_V12_EXTRA_OBSERVATION_COLUMNS]),
    ):
        raise ValueError("Preview source adapter is not exact zero")

    lifted = copy.deepcopy(payload)
    lifted["iter"] = TELEOP_V12_PREVIEW_LIFTED_ITERATION
    lifted_infos = lifted["infos"]
    lifted_infos["env_state"] = {
        "common_step_counter": TELEOP_V12_PREVIEW_LIFTED_COMPLETED_UPDATES * 24
    }
    lifted_infos["active_actor_columns_at_save"] = list(
        TELEOP_V12_EXTRA_OBSERVATION_COLUMNS
    )
    lifted_infos["preview_non_deployable"] = True
    lifted_infos[TELEOP_V12_PREVIEW_INFO_KEY] = canonical_preview_info()

    # The preview starts from the exact sanitized policy/critic/optimizer; only
    # its explicit clock and non-deployable lineage may differ.
    for key in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict"):
        if not _nested_equal(payload[key], lifted[key]):
            raise RuntimeError(f"Preview clock lift changed {key}")
    _atomic_torch_save(lifted, destination)
    if sha256_file(source) != source_sha:
        destination.unlink(missing_ok=True)
        raise RuntimeError("Preview source changed during clock lift")
    verified = torch.load(destination, map_location="cpu", weights_only=False)
    validate_preview_marker(verified.get("infos"), iteration=verified.get("iter", -1))
    return {
        "schema_version": 1,
        "creator": "microban_teleop_v12_fullbody_preview_clock_lift",
        "status": "pass",
        "preview_non_deployable": True,
        "source": {
            "path": str(source),
            "sha256": source_sha,
            "iteration": TELEOP_V12_PREVIEW_SOURCE_ITERATION,
            "completed_updates": TELEOP_V12_PREVIEW_SOURCE_COMPLETED_UPDATES,
        },
        "output": {
            "path": str(destination),
            "sha256": sha256_file(destination),
            "iteration": TELEOP_V12_PREVIEW_LIFTED_ITERATION,
            "completed_updates": TELEOP_V12_PREVIEW_LIFTED_COMPLETED_UPDATES,
        },
        "checks": {
            "source_unchanged": True,
            "actor_unchanged": True,
            "critic_unchanged": True,
            "optimizer_unchanged": True,
            "full20_adapter_active_after_resume": True,
            "canonical_deployment_forbidden": True,
        },
    }


def _nested_equal(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _nested_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _nested_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = create_preview_checkpoint(args.source, args.destination)
    if args.output is not None:
        if args.output.expanduser().exists():
            raise FileExistsError(f"Receipt exists: {args.output}")
        publish_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
