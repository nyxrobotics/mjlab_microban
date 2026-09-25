"""Promote a strict-safe preview with explicitly relaxed visual tracking quality."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import torch

from mjlab_microban.legacy_velocity_diagnostics import publish_json_atomic
from mjlab_microban.tasks.microban_teleop_v12_bootstrap import sha256_file
from mjlab_microban.tasks.microban_teleop_v12_preview import (
    TELEOP_V12_PREVIEW_FULLBODY_VISUAL_QUALITY,
    TELEOP_V12_PREVIEW_INFO_KEY,
    TELEOP_V12_PREVIEW_PHASE1_ACCEPTANCE_INFO_KEY,
    TELEOP_V12_PREVIEW_PHASE1_STRICT_QUALITY,
    TELEOP_V12_PREVIEW_PHASE1_TRAINED_ITERATION,
    TELEOP_V12_PREVIEW_PHASE1_VISUAL_QUALITY,
    TELEOP_V12_PREVIEW_PHASE_FULL_BODY,
    TELEOP_V12_PREVIEW_PHASE_HMD_HAND,
    preview_evaluator_source_manifest,
    validate_preview_evaluation_report,
    validate_preview_marker,
)

PHASE1_VISUAL_GATE = "microban_teleop_v12_preview_phase1_visual_promotion"
FULLBODY_VISUAL_GATE = "microban_teleop_v12_preview_fullbody_visual_acceptance"
HAND_RMS_MAX_M = 0.13
HAND_P95_MAX_M = 0.15
FOOT_RMS_MAX_M = 0.08
FOOT_P95_MAX_M = 0.13
LEARNED_SOURCE_DELTA_MIN = 1.0e-4
_LOWER_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def visual_promotion_source_manifest() -> dict[str, Any]:
    """Extend the strict evaluator manifest with this promotion decision code."""

    base = preview_evaluator_source_manifest()
    files = dict(base["files"])
    root = Path(__file__).resolve().parents[3]
    name = "src/mjlab_microban/scripts/promote_teleop_v12_preview_visual.py"
    files[name] = sha256_file(root / name)
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema_version": 1,
        "files": files,
        "aggregate_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _embedded_report_sha256(report: dict[str, Any]) -> str:
    """Match the durable JSON encoding used for the source receipt on disk."""

    encoded = (
        json.dumps(
            report,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _finite_max(results: list[dict[str, Any]], target: str, metric: str) -> float:
    values = [
        result["target_error"][target][metric]
        for result in results
        if result["target_error"][target][metric] is not None
    ]
    if not values or any(
        type(value) not in (int, float) or not math.isfinite(value)
        for value in values
    ):
        raise ValueError(f"Strict receipt has invalid {target} {metric} evidence")
    return float(max(values))


def _learned_delta_evidence(
    results: list[dict[str, Any]], target: str
) -> dict[str, float]:
    evidence: dict[str, float] = {}
    for result in results:
        if result["target_error"][target]["sample_count"] <= 0:
            continue
        values = result["raw_action_envelope"]["learned_minus_source"][
            "absolute_maximum"
        ]
        if (
            not isinstance(values, list)
            or len(values) != 18
            or any(
                type(value) not in (int, float) or not math.isfinite(value)
                for value in values
            )
        ):
            raise ValueError("Strict receipt learned-source evidence is malformed")
        evidence[result["name"]] = float(max(values))
    if not evidence or any(value <= LEARNED_SOURCE_DELTA_MIN for value in evidence.values()):
        raise ValueError(f"{target} scenarios lack learned-source action response")
    return evidence


def _target_column_ablation_evidence(
    results: list[dict[str, Any]], error_target: str, ablation_target: str
) -> dict[str, float]:
    """Require target-attributable same-observation action sensitivity."""

    evidence: dict[str, float] = {}
    for result in results:
        if result["target_error"][error_target]["sample_count"] <= 0:
            continue
        item = result.get("target_column_ablation", {}).get(ablation_target)
        if not isinstance(item, dict):
            raise TypeError("Target-column ablation evidence is missing")
        maximum = item.get("maximum_absolute_action_delta")
        if (
            item.get("target_expected") is not True
            or item.get("minimum_required_action_delta")
            != LEARNED_SOURCE_DELTA_MIN
            or item.get("passed") is not True
            or type(maximum) not in (int, float)
            or not math.isfinite(maximum)
            or maximum <= LEARNED_SOURCE_DELTA_MIN
        ):
            raise ValueError("Target-column ablation response did not pass")
        evidence[result["name"]] = float(maximum)
    if not evidence:
        raise ValueError("No active target-column ablation evidence was found")
    return evidence


def create_visual_promotion(
    checkpoint: Path,
    *,
    expected_checkpoint_sha256: str,
    strict_evaluation_receipt: Path,
    expected_receipt_sha256: str,
) -> dict[str, Any]:
    """Recompute the only allowed relaxed quality decision from strict evidence."""

    source_manifest = visual_promotion_source_manifest()
    for value, label in (
        (expected_checkpoint_sha256, "checkpoint"),
        (expected_receipt_sha256, "strict receipt"),
    ):
        if not _LOWER_SHA256.fullmatch(value):
            raise ValueError(f"Expected {label} SHA-256 must be lowercase hex")
    checkpoint = checkpoint.expanduser().resolve()
    if sha256_file(checkpoint) != expected_checkpoint_sha256:
        raise ValueError("Preview checkpoint SHA-256 mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("infos"), dict):
        raise TypeError("Preview checkpoint payload is malformed")
    iteration = payload.get("iter")
    if not isinstance(iteration, int) or isinstance(iteration, bool):
        raise TypeError("Preview checkpoint iteration is malformed")
    marker = validate_preview_marker(payload["infos"], iteration=iteration)

    receipt_path = strict_evaluation_receipt.expanduser()
    if receipt_path.is_symlink():
        raise ValueError("Strict evaluation receipt cannot be a symlink")
    receipt_path = receipt_path.resolve()
    if sha256_file(receipt_path) != expected_receipt_sha256:
        raise ValueError("Strict evaluation receipt SHA-256 mismatch")
    strict = json.loads(receipt_path.read_text())
    validate_preview_evaluation_report(
        strict,
        checkpoint_sha256=expected_checkpoint_sha256,
        iteration=iteration,
        marker=marker,
    )
    if strict.get("status") != "fail":
        raise ValueError("Visual promotion requires a strict quality failure")
    locomotion_checks = strict["locomotion"]["checks"]
    tracking_checks = strict["tracking"]["checks"]
    if not all(locomotion_checks.values()):
        raise ValueError("Visual promotion cannot relax locomotion")
    false_tracking = {name for name, passed in tracking_checks.items() if not passed}
    results = strict["tracking"]["results"]
    hand_rms = _finite_max(results, "active_hand", "rms")
    hand_p95 = _finite_max(results, "active_hand", "p95")
    if marker["phase"] == TELEOP_V12_PREVIEW_PHASE_HMD_HAND:
        if iteration != TELEOP_V12_PREVIEW_PHASE1_TRAINED_ITERATION:
            raise ValueError("Phase1 visual promotion requires model_7100")
        if not false_tracking or not false_tracking <= {
            "hand_tracking_rms",
            "hand_tracking_p95",
        }:
            raise ValueError("Phase1 promotion may relax only hand quality checks")
        if hand_rms > HAND_RMS_MAX_M or hand_p95 > HAND_P95_MAX_M:
            raise ValueError("Phase1 hand error exceeds visual-only limits")
        gate = PHASE1_VISUAL_GATE
        quality_class = TELEOP_V12_PREVIEW_PHASE1_VISUAL_QUALITY
        foot_rms = None
        foot_p95 = None
        delta_evidence: dict[str, Any] = {
            "hand_scenario_max_abs": _learned_delta_evidence(
                results, "active_hand"
            )
        }
        target_ablation_evidence = {
            "hand_scenario_max_abs": _target_column_ablation_evidence(
                results, "active_hand", "hand"
            )
        }
    elif marker["phase"] == TELEOP_V12_PREVIEW_PHASE_FULL_BODY:
        allowed = {
            "hand_tracking_rms",
            "hand_tracking_p95",
            "foot_tracking_rms",
            "foot_tracking_p95",
        }
        if not false_tracking or not false_tracking <= allowed:
            raise ValueError("Fullbody promotion may relax only hand/foot quality")
        foot_rms = _finite_max(results, "foot", "rms")
        foot_p95 = _finite_max(results, "foot", "p95")
        if (
            hand_rms > HAND_RMS_MAX_M
            or hand_p95 > HAND_P95_MAX_M
            or foot_rms > FOOT_RMS_MAX_M
            or foot_p95 > FOOT_P95_MAX_M
        ):
            raise ValueError("Fullbody error exceeds visual-only limits")
        gate = FULLBODY_VISUAL_GATE
        quality_class = TELEOP_V12_PREVIEW_FULLBODY_VISUAL_QUALITY
        delta_evidence = {
            "hand_scenario_max_abs": _learned_delta_evidence(results, "active_hand"),
            "foot_scenario_max_abs": _learned_delta_evidence(results, "foot"),
        }
        target_ablation_evidence = {
            "hand_scenario_max_abs": _target_column_ablation_evidence(
                results, "active_hand", "hand"
            ),
            "foot_scenario_max_abs": _target_column_ablation_evidence(
                results, "foot", "foot"
            ),
        }
    else:
        raise ValueError("Visual promotion requires a staged-v2 phase")
    if sha256_file(checkpoint) != expected_checkpoint_sha256 or sha256_file(
        receipt_path
    ) != expected_receipt_sha256:
        raise ValueError("Promotion inputs changed while validating")
    if visual_promotion_source_manifest() != source_manifest:
        raise ValueError("Visual promotion sources changed while validating")

    checks = {
        "source_hash_bound": True,
        "checkpoint_hash_bound": True,
        "strict_hard_safety_passed": True,
        "strict_locomotion_passed": True,
        "only_tracking_quality_failed": True,
        "relaxed_hand_rms_passed": True,
        "relaxed_hand_p95_passed": True,
        "learned_source_delta_coverage_passed": True,
        "canonical_deployment_forbidden": True,
    }
    checks["target_column_ablation_response_passed"] = True
    if marker["phase"] == TELEOP_V12_PREVIEW_PHASE_FULL_BODY:
        checks.update(
            {
                "relaxed_foot_rms_passed": True,
                "relaxed_foot_p95_passed": True,
            }
        )
    report = {
        "schema_version": 1,
        "gate": gate,
        "status": "pass",
        "simulation_only": True,
        "preview_non_deployable": True,
        "canonical_deployment_accepted": False,
        "quality_class": quality_class,
        "source_manifest": source_manifest,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": expected_checkpoint_sha256,
            "iteration": iteration,
            "completed_updates": iteration + 1,
        },
        TELEOP_V12_PREVIEW_INFO_KEY: marker,
        "source_evaluation_receipt": {
            "path": str(receipt_path),
            "sha256": expected_receipt_sha256,
            "schema_version": 2,
            "gate": "microban_teleop_v12_nondeployable_preview_acceptance",
            "status": "fail",
        },
        "thresholds": {
            "hand_rms_m_max": HAND_RMS_MAX_M,
            "hand_p95_m_max": HAND_P95_MAX_M,
            "foot_rms_m_max": FOOT_RMS_MAX_M if foot_rms is not None else None,
            "foot_p95_m_max": FOOT_P95_MAX_M if foot_p95 is not None else None,
            "learned_source_delta_min": (
                LEARNED_SOURCE_DELTA_MIN if delta_evidence else None
            ),
        },
        "evidence": {
            "maximum_hand_rms_m": hand_rms,
            "maximum_hand_p95_m": hand_p95,
            "maximum_foot_rms_m": foot_rms,
            "maximum_foot_p95_m": foot_p95,
            "strict_failed_tracking_checks": sorted(false_tracking),
            "learned_source_delta_coverage": delta_evidence,
            "target_column_ablation": target_ablation_evidence,
        },
        "checks": checks,
        "strict_evaluation": strict,
    }
    validate_visual_promotion_report(
        report,
        checkpoint_sha256=expected_checkpoint_sha256,
        iteration=iteration,
        marker=marker,
    )
    return report


def validate_visual_promotion_report(
    report: object,
    *,
    checkpoint_sha256: str,
    iteration: int,
    marker: dict[str, Any],
) -> dict[str, Any]:
    """Recompute a self-contained visual decision without its source file."""

    if not isinstance(report, dict):
        raise TypeError("Visual promotion report is malformed")
    strict = report.get("strict_evaluation")
    validate_preview_evaluation_report(
        strict,
        checkpoint_sha256=checkpoint_sha256,
        iteration=iteration,
        marker=marker,
    )
    if not isinstance(strict, dict) or strict.get("status") != "fail":
        raise ValueError("Visual promotion strict evidence must fail quality")
    if not all(strict["locomotion"]["checks"].values()):
        raise ValueError("Visual promotion strict locomotion evidence failed")
    tracking_checks = strict["tracking"]["checks"]
    false_tracking = {name for name, passed in tracking_checks.items() if not passed}
    results = strict["tracking"]["results"]
    hand_rms = _finite_max(results, "active_hand", "rms")
    hand_p95 = _finite_max(results, "active_hand", "p95")
    if marker.get("phase") == TELEOP_V12_PREVIEW_PHASE_HMD_HAND:
        allowed = {"hand_tracking_rms", "hand_tracking_p95"}
        if iteration != TELEOP_V12_PREVIEW_PHASE1_TRAINED_ITERATION:
            raise ValueError("Phase1 visual evidence requires model_7100")
        gate = PHASE1_VISUAL_GATE
        quality_class = TELEOP_V12_PREVIEW_PHASE1_VISUAL_QUALITY
        foot_rms = None
        foot_p95 = None
        delta_evidence: dict[str, Any] = {
            "hand_scenario_max_abs": _learned_delta_evidence(
                results, "active_hand"
            )
        }
        target_ablation_evidence = {
            "hand_scenario_max_abs": _target_column_ablation_evidence(
                results, "active_hand", "hand"
            )
        }
    elif marker.get("phase") == TELEOP_V12_PREVIEW_PHASE_FULL_BODY:
        allowed = {
            "hand_tracking_rms",
            "hand_tracking_p95",
            "foot_tracking_rms",
            "foot_tracking_p95",
        }
        gate = FULLBODY_VISUAL_GATE
        quality_class = TELEOP_V12_PREVIEW_FULLBODY_VISUAL_QUALITY
        foot_rms = _finite_max(results, "foot", "rms")
        foot_p95 = _finite_max(results, "foot", "p95")
        delta_evidence = {
            "hand_scenario_max_abs": _learned_delta_evidence(
                results, "active_hand"
            ),
            "foot_scenario_max_abs": _learned_delta_evidence(results, "foot"),
        }
        target_ablation_evidence = {
            "hand_scenario_max_abs": _target_column_ablation_evidence(
                results, "active_hand", "hand"
            ),
            "foot_scenario_max_abs": _target_column_ablation_evidence(
                results, "foot", "foot"
            ),
        }
    else:
        raise ValueError("Visual promotion requires a staged-v2 marker")
    if not false_tracking or not false_tracking <= allowed:
        raise ValueError("Visual promotion attempted to relax a hard check")
    if (
        hand_rms > HAND_RMS_MAX_M
        or hand_p95 > HAND_P95_MAX_M
        or (foot_rms is not None and foot_rms > FOOT_RMS_MAX_M)
        or (foot_p95 is not None and foot_p95 > FOOT_P95_MAX_M)
    ):
        raise ValueError("Visual promotion evidence exceeds relaxed limits")

    expected_checks = {
        "source_hash_bound": True,
        "checkpoint_hash_bound": True,
        "strict_hard_safety_passed": True,
        "strict_locomotion_passed": True,
        "only_tracking_quality_failed": True,
        "relaxed_hand_rms_passed": True,
        "relaxed_hand_p95_passed": True,
        "learned_source_delta_coverage_passed": True,
        "target_column_ablation_response_passed": True,
        "canonical_deployment_forbidden": True,
    }
    if foot_rms is not None:
        expected_checks.update(
            {
                "relaxed_foot_rms_passed": True,
                "relaxed_foot_p95_passed": True,
            }
        )
    expected_thresholds = {
        "hand_rms_m_max": HAND_RMS_MAX_M,
        "hand_p95_m_max": HAND_P95_MAX_M,
        "foot_rms_m_max": FOOT_RMS_MAX_M if foot_rms is not None else None,
        "foot_p95_m_max": FOOT_P95_MAX_M if foot_p95 is not None else None,
        "learned_source_delta_min": LEARNED_SOURCE_DELTA_MIN,
    }
    expected_evidence = {
        "maximum_hand_rms_m": hand_rms,
        "maximum_hand_p95_m": hand_p95,
        "maximum_foot_rms_m": foot_rms,
        "maximum_foot_p95_m": foot_p95,
        "strict_failed_tracking_checks": sorted(false_tracking),
        "learned_source_delta_coverage": delta_evidence,
        "target_column_ablation": target_ablation_evidence,
    }
    checkpoint = report.get("checkpoint")
    source = report.get("source_evaluation_receipt")
    expected_keys = {
        "schema_version",
        "gate",
        "status",
        "simulation_only",
        "preview_non_deployable",
        "canonical_deployment_accepted",
        "quality_class",
        "source_manifest",
        "checkpoint",
        TELEOP_V12_PREVIEW_INFO_KEY,
        "source_evaluation_receipt",
        "thresholds",
        "evidence",
        "checks",
        "strict_evaluation",
    }
    if (
        set(report) != expected_keys
        or report.get("schema_version") != 1
        or report.get("gate") != gate
        or report.get("status") != "pass"
        or report.get("simulation_only") is not True
        or report.get("preview_non_deployable") is not True
        or report.get("canonical_deployment_accepted") is not False
        or report.get("quality_class") != quality_class
        or report.get("source_manifest") != visual_promotion_source_manifest()
        or report.get(TELEOP_V12_PREVIEW_INFO_KEY) != marker
        or report.get("checks") != expected_checks
        or report.get("thresholds") != expected_thresholds
        or report.get("evidence") != expected_evidence
        or not isinstance(checkpoint, dict)
        or checkpoint.get("sha256") != checkpoint_sha256
        or checkpoint.get("iteration") != iteration
        or checkpoint.get("completed_updates") != iteration + 1
        or not isinstance(source, dict)
        or set(source) != {"path", "sha256", "schema_version", "gate", "status"}
        or not isinstance(source.get("path"), str)
        or not _LOWER_SHA256.fullmatch(source.get("sha256", ""))
        or source.get("sha256") != _embedded_report_sha256(strict)
        or source.get("schema_version") != 2
        or source.get("gate")
        != "microban_teleop_v12_nondeployable_preview_acceptance"
        or source.get("status") != "fail"
    ):
        raise ValueError("Visual promotion receipt failed strict validation")
    return report


def validate_embedded_phase1_acceptance(
    infos: object, *, fullbody_marker: dict[str, Any]
) -> dict[str, Any]:
    """Authenticate the portable phase-1 evidence embedded by the phase lift."""

    if not isinstance(infos, dict):
        raise TypeError("Full-body preview checkpoint infos are malformed")
    if fullbody_marker.get("phase") != TELEOP_V12_PREVIEW_PHASE_FULL_BODY:
        raise ValueError("Embedded phase-1 evidence requires a full-body marker")
    report = infos.get(TELEOP_V12_PREVIEW_PHASE1_ACCEPTANCE_INFO_KEY)
    if not isinstance(report, dict):
        raise TypeError("Full-body preview lacks embedded phase-1 acceptance")
    expected_digest = fullbody_marker.get("phase1_acceptance_receipt_sha256")
    if _embedded_report_sha256(report) != expected_digest:
        raise ValueError("Embedded phase-1 acceptance digest drifted")
    phase1_marker = report.get(TELEOP_V12_PREVIEW_INFO_KEY)
    if not isinstance(phase1_marker, dict):
        raise TypeError("Embedded phase-1 marker is missing")
    validate_preview_marker(
        {
            "preview_non_deployable": True,
            TELEOP_V12_PREVIEW_INFO_KEY: phase1_marker,
        },
        iteration=TELEOP_V12_PREVIEW_PHASE1_TRAINED_ITERATION,
        required_phase=TELEOP_V12_PREVIEW_PHASE_HMD_HAND,
    )
    phase1_checkpoint_sha = fullbody_marker.get("phase_source_checkpoint_sha256")
    gate = report.get("gate")
    if gate == "microban_teleop_v12_nondeployable_preview_acceptance":
        validate_preview_evaluation_report(
            report,
            checkpoint_sha256=phase1_checkpoint_sha,
            iteration=TELEOP_V12_PREVIEW_PHASE1_TRAINED_ITERATION,
            marker=phase1_marker,
        )
        if (
            report.get("status") != "pass"
            or report.get("quality_class")
            != TELEOP_V12_PREVIEW_PHASE1_STRICT_QUALITY
        ):
            raise ValueError("Embedded strict phase-1 acceptance did not pass")
    elif gate == PHASE1_VISUAL_GATE:
        validate_visual_promotion_report(
            report,
            checkpoint_sha256=phase1_checkpoint_sha,
            iteration=TELEOP_V12_PREVIEW_PHASE1_TRAINED_ITERATION,
            marker=phase1_marker,
        )
    else:
        raise ValueError("Embedded phase-1 acceptance gate is not allowed")
    if report.get("quality_class") != fullbody_marker.get("phase1_quality_class"):
        raise ValueError("Embedded phase-1 quality class disagrees with marker")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--strict-evaluation-receipt", type=Path, required=True)
    parser.add_argument("--expected-receipt-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.expanduser().exists() and not args.force:
        raise FileExistsError(f"Output exists (pass --force): {args.output}")
    report = create_visual_promotion(
        args.checkpoint,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        strict_evaluation_receipt=args.strict_evaluation_receipt,
        expected_receipt_sha256=args.expected_receipt_sha256,
    )
    publish_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
