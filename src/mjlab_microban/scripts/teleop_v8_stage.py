# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Fail-closed planning and provenance checks for canonical teleop v9 stages."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_TELEOP_ACTOR_INITIALIZATION,
    MICROBAN_TELEOP_RECIPE_REVISION,
    MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION,
    validate_teleop_checkpoint_contract,
)
from mjlab_microban.tasks.microban_teleop_provenance import (
    MICROBAN_TELEOP_TRAINING_PROVENANCE_KEY,
    MICROBAN_TELEOP_TRAINING_PROVENANCE_SHA256_KEY,
    sha256_file,
    validate_canonical_stage_critical_config,
    validate_training_provenance,
)

# Historical v9 tooling must retain its own frozen stage plan.  Importing the
# current contract's boundaries silently changed v9 audit results when v10
# introduced a new curriculum.
V9_STAGE_BOUNDARIES = (
    1_500,
    3_000,
    4_500,
    6_000,
    8_000,
    12_000,
    14_000,
    16_000,
    18_000,
    20_000,
)
# Import compatibility for the historical module/test name.  The accepted
# checkpoint contract itself is v9-only.
V8_STAGE_BOUNDARIES = V9_STAGE_BOUNDARIES


@dataclass(frozen=True)
class StageInterval:
    """The canonical interval containing one completed-update count."""

    start_boundary: int
    target_boundary: int
    interrupted: bool


def resolve_stage_interval(completed_iterations: int) -> StageInterval:
    """Resolve a boundary or interrupted checkpoint to exactly one next target."""

    if isinstance(completed_iterations, bool) or not isinstance(
        completed_iterations, int
    ):
        raise TypeError("completed_iterations must be an integer")
    if completed_iterations < 1:
        raise ValueError("A resumable checkpoint must contain at least one update")
    if completed_iterations >= V9_STAGE_BOUNDARIES[-1]:
        if completed_iterations == V9_STAGE_BOUNDARIES[-1]:
            raise ValueError(
                "The final 20,000-update boundary is already reached; evaluate/export it"
            )
        raise ValueError("Checkpoint exceeds the final v9 stage boundary")

    previous = 0
    for boundary in V9_STAGE_BOUNDARIES:
        if completed_iterations == boundary:
            boundary_index = V9_STAGE_BOUNDARIES.index(boundary)
            return StageInterval(
                start_boundary=boundary,
                target_boundary=V9_STAGE_BOUNDARIES[boundary_index + 1],
                interrupted=False,
            )
        if completed_iterations < boundary:
            return StageInterval(
                start_boundary=previous,
                target_boundary=boundary,
                interrupted=True,
            )
        previous = boundary
    raise AssertionError("Unreachable v9 stage interval")


def _checkpoint_infos(checkpoint_path: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("infos"), dict):
        raise TypeError("Checkpoint does not contain an infos dictionary")
    return payload["infos"]


def _validate_parent_gate(
    *,
    gate_root: Path,
    start_boundary: int,
    parent_checkpoint_sha256: str,
    parent_gate_sha256: str,
) -> Path:
    from mjlab_microban.scripts import evaluate_teleop_checkpoint as evaluator

    matches: list[Path] = []
    for candidate in sorted(
        gate_root.glob(f"*_boundary_{start_boundary}_gate.json")
    ):
        if not candidate.is_file() or sha256_file(candidate) != parent_gate_sha256:
            continue
        gate = json.loads(candidate.read_text(encoding="utf-8"))
        parent_checkpoint = Path(gate.get("checkpoint", "")).resolve()
        if (
            gate.get("status") == "pass"
            and gate.get("training_contract_version")
            == MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION
            and gate.get("completed_iterations") == start_boundary
            and gate.get("evaluator_revision") == evaluator.TELEOP_EVALUATOR_REVISION
            and gate.get("acceptance_revision")
            == evaluator.TELEOP_ACCEPTANCE_REVISION
            and gate.get("acceptance_profile")
            == evaluator.INTERMEDIATE_HARD_SAFETY_PROFILE
            and gate.get("checkpoint_sha256") == parent_checkpoint_sha256
            and parent_checkpoint.is_file()
            and sha256_file(parent_checkpoint) == parent_checkpoint_sha256
        ):
            matches.append(candidate.resolve())
    if len(matches) != 1:
        raise ValueError(
            "Interrupted stage parent gate was not found uniquely by its pinned "
            f"SHA-256 ({len(matches)} matches)"
        )
    return matches[0]


def validate_interrupted_stage_checkpoint(
    checkpoint_path: str | Path,
    *,
    completed_iterations: int,
    interval: StageInterval,
    gate_root: str | Path,
) -> dict[str, Any]:
    """Validate and return the pinned invocation for an in-stage checkpoint."""

    if not interval.interrupted:
        raise ValueError("Interrupted-stage validation requires an in-stage interval")
    checkpoint = Path(checkpoint_path).resolve()
    contract = validate_teleop_checkpoint_contract(checkpoint, map_location="cpu")
    if contract.iteration != completed_iterations - 1:
        raise ValueError("Checkpoint iteration does not match its completed-update count")

    infos = _checkpoint_infos(checkpoint)
    manifest = infos.get(MICROBAN_TELEOP_TRAINING_PROVENANCE_KEY)
    digest = infos.get(MICROBAN_TELEOP_TRAINING_PROVENANCE_SHA256_KEY)
    manifest = validate_training_provenance(
        manifest,
        digest,
        require_canonical_stage=True,
        expected_contract_version=MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION,
        expected_recipe_revision=MICROBAN_TELEOP_RECIPE_REVISION,
        expected_actor_initialization=MICROBAN_TELEOP_ACTOR_INITIALIZATION,
        require_current_source=True,
    )
    validate_canonical_stage_critical_config(manifest)
    invocation = manifest.get("invocation")
    if not isinstance(invocation, dict):
        raise TypeError("Checkpoint canonical invocation is malformed")
    if invocation.get("stage_start_boundary") != interval.start_boundary:
        raise ValueError("Interrupted checkpoint stage-start provenance mismatch")
    if invocation.get("stage_target_boundary") != interval.target_boundary:
        raise ValueError("Interrupted checkpoint stage-target provenance mismatch")

    parent_checkpoint_sha256 = invocation.get("parent_checkpoint_sha256")
    parent_gate_sha256 = invocation.get("parent_gate_sha256")
    if interval.start_boundary == 0:
        if parent_checkpoint_sha256 is not None or parent_gate_sha256 is not None:
            raise ValueError("Initial interrupted stage must not have parent SHAs")
        parent_gate = None
    else:
        if not isinstance(parent_checkpoint_sha256, str) or not isinstance(
            parent_gate_sha256, str
        ):
            raise ValueError("Interrupted resumed stage is missing parent SHAs")
        parent_gate = _validate_parent_gate(
            gate_root=Path(gate_root).resolve(),
            start_boundary=interval.start_boundary,
            parent_checkpoint_sha256=parent_checkpoint_sha256,
            parent_gate_sha256=parent_gate_sha256,
        )
    return {
        "stage_start_boundary": interval.start_boundary,
        "stage_target_boundary": interval.target_boundary,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "parent_gate_sha256": parent_gate_sha256,
        "parent_gate": str(parent_gate) if parent_gate is not None else None,
        "training_provenance_sha256": digest,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--completed-iterations", type=int, required=True)
    parser.add_argument("--stage-start-boundary", type=int, required=True)
    parser.add_argument("--stage-target-boundary", type=int, required=True)
    parser.add_argument("--gate-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    interval = StageInterval(
        start_boundary=args.stage_start_boundary,
        target_boundary=args.stage_target_boundary,
        interrupted=True,
    )
    result = validate_interrupted_stage_checkpoint(
        args.checkpoint,
        completed_iterations=args.completed_iterations,
        interval=interval,
        gate_root=args.gate_root,
    )
    # Exactly four machine-readable lines keep the Bash wrapper free from eval
    # and preserve empty initial-parent values.
    print(result["stage_start_boundary"])
    print(result["stage_target_boundary"])
    print(result["parent_checkpoint_sha256"] or "")
    print(result["parent_gate_sha256"] or "")


if __name__ == "__main__":
    main()
