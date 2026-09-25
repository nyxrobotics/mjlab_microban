"""Evaluate a non-deployable preview without relaxing canonical v12 gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

from mjlab_microban.legacy_velocity_diagnostics import publish_json_atomic
from mjlab_microban.scripts.evaluate_teleop_v12_checkpoint import (
    run_evaluation as run_locomotion_evaluation,
)
from mjlab_microban.scripts.evaluate_teleop_v12_tracking import (
    FINAL_PROFILE,
    HMD_HAND_PROFILE,
)
from mjlab_microban.scripts.evaluate_teleop_v12_tracking import (
    run_evaluation as run_tracking_evaluation,
)
from mjlab_microban.scripts.promote_teleop_v12_preview_visual import (
    validate_embedded_phase1_acceptance,
)
from mjlab_microban.tasks.microban_teleop_v12_bootstrap import sha256_file
from mjlab_microban.tasks.microban_teleop_v12_preview import (
    TELEOP_V12_PREVIEW_FULLBODY_STRICT_QUALITY,
    TELEOP_V12_PREVIEW_INFO_KEY,
    TELEOP_V12_PREVIEW_LEGACY_REVISION,
    TELEOP_V12_PREVIEW_PHASE1_STRICT_QUALITY,
    TELEOP_V12_PREVIEW_PHASE1_TRAINED_ITERATION,
    TELEOP_V12_PREVIEW_PHASE_FULL_BODY,
    TELEOP_V12_PREVIEW_PHASE_HMD_HAND,
    preview_evaluator_source_manifest,
    validate_preview_evaluation_report,
    validate_preview_marker,
)


def run_preview_evaluation(
    *,
    checkpoint: Path,
    expected_sha256: str | None,
    device: str,
    allow_legacy_v1: bool = False,
) -> dict[str, Any]:
    source_manifest = preview_evaluator_source_manifest()
    checkpoint = checkpoint.expanduser().resolve()
    digest = sha256_file(checkpoint)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"Preview checkpoint SHA-256 mismatch: {digest}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("infos"), dict):
        raise TypeError("Preview checkpoint payload is malformed")
    iteration = payload.get("iter")
    if not isinstance(iteration, int) or isinstance(iteration, bool):
        raise TypeError("Preview checkpoint iteration is malformed")
    marker = validate_preview_marker(
        payload["infos"],
        iteration=iteration,
        allow_legacy_v1=allow_legacy_v1,
    )
    phase = marker.get("phase")
    legacy = marker.get("revision") == TELEOP_V12_PREVIEW_LEGACY_REVISION
    if phase == TELEOP_V12_PREVIEW_PHASE_HMD_HAND:
        if iteration != TELEOP_V12_PREVIEW_PHASE1_TRAINED_ITERATION:
            raise ValueError("Phase-1 acceptance requires exactly model_7100")
        tracking_profile = HMD_HAND_PROFILE
    elif phase == TELEOP_V12_PREVIEW_PHASE_FULL_BODY:
        validate_preview_marker(
            payload["infos"], iteration=iteration, require_live_candidate=True
        )
        validate_embedded_phase1_acceptance(
            payload["infos"], fullbody_marker=marker
        )
        tracking_profile = FINAL_PROFILE
    elif legacy:
        tracking_profile = FINAL_PROFILE
    else:
        raise ValueError("Unsupported preview phase")

    locomotion = run_locomotion_evaluation(
        checkpoint=checkpoint,
        expected_sha256=digest,
        device=device,
        seed=42,
        steps=300,
        settle_steps=50,
        allow_nondeployable_preview=True,
        allow_legacy_preview_v1=legacy,
    )
    tracking = run_tracking_evaluation(
        checkpoint=checkpoint,
        expected_sha256=digest,
        profile=tracking_profile,
        device=device,
        seed=42,
        steps=300,
        settle_steps=50,
        allow_nondeployable_preview=True,
        allow_legacy_preview_v1=legacy,
    )
    tracking_check_name = (
        "hmd_hand_performance_foot_exposure"
        if tracking_profile == HMD_HAND_PROFILE
        else "full_body_perturbation_8x300"
    )
    checks = {
        "legacy_locomotion_9x300": locomotion.get("status") == "pass"
        and bool(locomotion.get("checks"))
        and all(value is True for value in locomotion["checks"].values()),
        tracking_check_name: tracking.get("status") == "pass"
        and tracking.get("profile") == tracking_profile
        and bool(tracking.get("checks"))
        and all(value is True for value in tracking["checks"].values()),
    }
    if sha256_file(checkpoint) != digest:
        raise ValueError("Preview checkpoint changed during evaluation")
    if preview_evaluator_source_manifest() != source_manifest:
        raise ValueError("Preview evaluator sources changed during evaluation")
    passed = all(checks.values())
    live_candidate = (
        passed and phase == TELEOP_V12_PREVIEW_PHASE_FULL_BODY and not legacy
    )
    quality_class = (
        "legacy_v1_read_only"
        if legacy
        else (
            TELEOP_V12_PREVIEW_PHASE1_STRICT_QUALITY
            if phase == TELEOP_V12_PREVIEW_PHASE_HMD_HAND
            else TELEOP_V12_PREVIEW_FULLBODY_STRICT_QUALITY
        )
    )
    report = {
        "schema_version": 2,
        "gate": "microban_teleop_v12_nondeployable_preview_acceptance",
        "status": "pass" if passed else "fail",
        "simulation_only": True,
        "preview_non_deployable": True,
        "live_simulation_candidate": live_candidate,
        "legacy_preview_read_only": legacy,
        "quality_class": quality_class,
        "source_manifest": source_manifest,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": digest,
            "iteration": iteration,
            "completed_updates": iteration + 1,
        },
        TELEOP_V12_PREVIEW_INFO_KEY: marker,
        "settings": {
            "device": device,
            "seed": 42,
            "steps_per_scenario": 300,
            "settle_steps": 50,
            "locomotion_scenario_count": 9,
            "tracking_profile": tracking_profile,
            "tracking_scenario_count": len(tracking.get("results", [])),
            "canonical_deployment_accepted": False,
        },
        "checks": checks,
        "locomotion": locomotion,
        "tracking": tracking,
    }
    if not legacy:
        validate_preview_evaluation_report(
            report,
            checkpoint_sha256=digest,
            iteration=iteration,
            marker=marker,
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-legacy-v1", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.expanduser().exists() and not args.force:
        raise FileExistsError(f"Output exists (pass --force): {args.output}")
    report = run_preview_evaluation(
        checkpoint=args.checkpoint,
        expected_sha256=args.expected_sha256,
        device=args.device,
        allow_legacy_v1=args.allow_legacy_v1,
    )
    publish_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
