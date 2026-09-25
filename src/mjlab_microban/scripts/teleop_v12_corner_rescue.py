"""Authenticate the pinned model9900 targeted corner-pair rescue route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from mjlab_microban.legacy_velocity_diagnostics import publish_json_atomic
from mjlab_microban.scripts.evaluate_teleop_v12_tracking import HMD_HAND_PROFILE
from mjlab_microban.scripts.teleop_v12_stage import (
    _load_json,
    _validate_tracking_report,
    validate_gate,
)
from mjlab_microban.tasks.microban_teleop_v12_bootstrap import (
    resolve_bootstrap_artifact_path,
    sha256_file,
    validate_bootstrap_provenance,
)
from mjlab_microban.tasks.microban_teleop_v12_corner_rescue import (
    MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY,
    MICROBAN_TELEOP_V12_CORNER_RESCUE_PARENT_TRACKING_SHA256,
    MICROBAN_TELEOP_V12_CORNER_RESCUE_RECIPE_REVISION,
    MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_COMPLETED_UPDATES,
    MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_ITERATION,
    MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_OPTIMIZER_STEP,
    MICROBAN_TELEOP_V12_CORNER_RESCUE_V1_CHECKPOINT_SHA256,
    MICROBAN_TELEOP_V12_CORNER_RESCUE_V1_TRACKING_SHA256,
    corner_rescue_marker,
    validate_corner_rescue_marker,
)
from mjlab_microban.tasks.microban_teleop_v12_corner_rescue_runner import (
    assert_corner_rescue_foot_adapter_zero,
    assert_corner_rescue_optimizer_step,
    validate_corner_rescue_parent_payload,
)
from mjlab_microban.tasks.microban_teleop_v12_lr_order import (
    validate_bilateral_site_order_checkpoint,
)
from mjlab_microban.tasks.microban_teleop_v12_preview import (
    reject_preview_checkpoint,
)
from mjlab_microban.tasks.microban_teleop_v12_runner import (
    TELEOP_V12_BOOTSTRAP_INFO_KEY,
)

CORNER_RESCUE_RECEIPT_SCHEMA_VERSION = 1
CORNER_RESCUE_RECEIPT_GATE = "microban_teleop_v12_corner_pair_rescue"
def _load_checkpoint(path: Path) -> tuple[Path, str, dict[str, Any]]:
    resolved = path.expanduser().resolve(strict=True)
    digest = sha256_file(resolved)
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("Corner rescue checkpoint root must be a dictionary")
    return resolved, digest, payload


def validate_parent_checkpoint(
    path: Path, tracking_report: Path, superseded_v1_tracking_report: Path
) -> dict[str, Any]:
    """Authenticate the exact corrected model9900 before simulator startup."""

    resolved, digest, payload = _load_checkpoint(path)
    validate_corner_rescue_parent_payload(payload, checkpoint_sha256=digest)
    infos = payload.get("infos")
    assert isinstance(infos, dict)
    validate_bilateral_site_order_checkpoint(infos)
    reject_preview_checkpoint(infos)
    validate_bootstrap_provenance(
        infos.get(TELEOP_V12_BOOTSTRAP_INFO_KEY), verify_files=True
    )
    report_path = tracking_report.expanduser().resolve(strict=True)
    report_digest = sha256_file(report_path)
    if report_digest != MICROBAN_TELEOP_V12_CORNER_RESCUE_PARENT_TRACKING_SHA256:
        raise ValueError("Pinned parent strict tracking report SHA-256 mismatch")
    report = _load_json(report_path)
    expected_identity = {
        "sha256": digest,
        "iteration": payload["iter"],
        "completed_updates": payload["iter"] + 1,
    }
    _validate_tracking_report(
        report,
        expected_identity,
        profile_override=HMD_HAND_PROFILE,
        allowed_failed_checks=frozenset(("hand_tracking_rms",)),
    )
    failed = sorted(name for name, passed in report["checks"].items() if not passed)
    if report.get("status") != "fail" or failed != ["hand_tracking_rms"]:
        raise ValueError(
            "Pinned parent report must fail only the strict hand RMS check"
        )
    v1_report_path = superseded_v1_tracking_report.expanduser().resolve(strict=True)
    v1_report_digest = sha256_file(v1_report_path)
    if v1_report_digest != MICROBAN_TELEOP_V12_CORNER_RESCUE_V1_TRACKING_SHA256:
        raise ValueError("Pinned superseded-v1 tracking report SHA-256 mismatch")
    v1_report = _load_json(v1_report_path)
    _validate_tracking_report(
        v1_report,
        {
            "sha256": MICROBAN_TELEOP_V12_CORNER_RESCUE_V1_CHECKPOINT_SHA256,
            "iteration": MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_ITERATION,
            "completed_updates": (
                MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_COMPLETED_UPDATES
            ),
        },
        profile_override=HMD_HAND_PROFILE,
        allowed_failed_checks=frozenset(("hand_tracking_rms",)),
    )
    v1_failed = sorted(
        name for name, passed in v1_report["checks"].items() if not passed
    )
    if v1_report.get("status") != "fail" or v1_failed != ["hand_tracking_rms"]:
        raise ValueError("Superseded v1 report must fail only strict hand RMS")
    return {
        "schema_version": 1,
        "gate": "microban_teleop_v12_corner_pair_rescue_parent",
        "status": "pass",
        "checkpoint": {
            "path": str(resolved),
            "sha256": digest,
            "iteration": payload["iter"],
            "completed_updates": payload["iter"] + 1,
        },
        "strict_tracking_report": {
            "path": str(report_path),
            "sha256": report_digest,
            "status": report["status"],
            "failed_checks": failed,
        },
        "superseded_v1_tracking_report": {
            "path": str(v1_report_path),
            "sha256": v1_report_digest,
            "checkpoint_sha256": (
                MICROBAN_TELEOP_V12_CORNER_RESCUE_V1_CHECKPOINT_SHA256
            ),
            "status": v1_report["status"],
            "failed_checks": v1_failed,
        },
        "target": corner_rescue_marker(),
    }


def validate_rescue_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the final marked checkpoint and return identity plus payload."""

    resolved, digest, payload = _load_checkpoint(path)
    iteration = payload.get("iter")
    infos = payload.get("infos")
    if not isinstance(iteration, int) or isinstance(iteration, bool):
        raise TypeError("Corner rescue checkpoint iteration is malformed")
    if not isinstance(infos, dict):
        raise TypeError("Corner rescue checkpoint infos are missing")
    validate_corner_rescue_marker(infos, iteration=iteration)
    validate_bilateral_site_order_checkpoint(infos)
    reject_preview_checkpoint(infos)
    validate_bootstrap_provenance(
        infos.get(TELEOP_V12_BOOTSTRAP_INFO_KEY), verify_files=True
    )
    assert_corner_rescue_foot_adapter_zero(payload)
    assert_corner_rescue_optimizer_step(
        payload,
        expected_step=MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_OPTIMIZER_STEP,
    )
    if iteration != MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_ITERATION:
        raise ValueError("Corner rescue acceptance is valid only at model_9999.pt")
    identity = {
        "path": str(resolved),
        "sha256": digest,
        "iteration": iteration,
        "completed_updates": iteration + 1,
    }
    return identity, payload


def build_receipt(
    *, checkpoint: Path, tracking_report: Path, stage_gate: Path
) -> dict[str, Any]:
    """Bind the unchanged full stage gate to the authenticated rescue endpoint."""

    identity, payload = validate_rescue_checkpoint(checkpoint)
    report_path = tracking_report.expanduser().resolve(strict=True)
    report = _load_json(report_path)
    _validate_tracking_report(
        report,
        {
            "sha256": identity["sha256"],
            "iteration": identity["iteration"],
            "completed_updates": identity["completed_updates"],
        },
        profile_override=HMD_HAND_PROFILE,
    )
    checks = report["checks"]
    failed = sorted(name for name, value in checks.items() if value is not True)
    if failed or report.get("status") != "pass":
        raise ValueError("Corner rescue promotion requires strict tracking PASS")
    gate_path = stage_gate.expanduser().resolve(strict=True)
    gate = validate_gate(gate_path, Path(identity["path"]))
    gate_tracking_path = gate.get("reports", {}).get("tracking")
    if (
        resolve_bootstrap_artifact_path(str(gate_tracking_path)) != report_path
        or gate.get("report_sha256", {}).get("tracking") != sha256_file(report_path)
    ):
        raise ValueError("Full stage gate is not bound to the supplied tracking report")
    infos = payload["infos"]
    return {
        "schema_version": CORNER_RESCUE_RECEIPT_SCHEMA_VERSION,
        "gate": CORNER_RESCUE_RECEIPT_GATE,
        "status": "pass",
        "recipe_revision": MICROBAN_TELEOP_V12_CORNER_RESCUE_RECIPE_REVISION,
        "checkpoint": identity,
        "lineage": infos[MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY],
        "tracking_report": {
            "path": str(report_path),
            "sha256": sha256_file(report_path),
            "profile": report["profile"],
            "status": report["status"],
            "checks": checks,
            "failed_checks": failed,
        },
        "full_stage_gate": {
            "path": str(gate_path),
            "sha256": sha256_file(gate_path),
            "schema_version": gate["schema_version"],
            "status": gate["status"],
            "report_sha256": gate["report_sha256"],
        },
        "promotion": {
            "eligible": True,
            "completed_updates": (
                MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_COMPLETED_UPDATES
            ),
            "strict_thresholds_relaxed": False,
            "foot_activation_during_rescue": False,
            "requires_schema_v2_full_stage_gate": True,
        },
    }


def create_receipt(
    *,
    checkpoint: Path,
    tracking_report: Path,
    stage_gate: Path,
    output: Path,
    force: bool,
) -> dict[str, Any]:
    destination = output.expanduser().resolve()
    if destination.exists() and not force:
        raise FileExistsError(f"Corner rescue receipt exists: {destination}")
    if destination.is_symlink():
        raise ValueError("Corner rescue receipt output must not be a symlink")
    receipt = build_receipt(
        checkpoint=checkpoint,
        tracking_report=tracking_report,
        stage_gate=stage_gate,
    )
    publish_json_atomic(destination, receipt)
    return receipt


def validate_receipt(
    *, receipt: Path, checkpoint: Path, tracking_report: Path, stage_gate: Path
) -> dict[str, Any]:
    actual_path = receipt.expanduser().resolve(strict=True)
    actual = _load_json(actual_path)
    expected = build_receipt(
        checkpoint=checkpoint,
        tracking_report=tracking_report,
        stage_gate=stage_gate,
    )
    if actual != expected:
        raise ValueError("Corner rescue receipt content/hash binding drifted")
    return actual


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    parent = commands.add_parser("validate-parent")
    parent.add_argument("checkpoint", type=Path)
    parent.add_argument("tracking_report", type=Path)
    parent.add_argument("superseded_v1_tracking_report", type=Path)
    for name in ("create-receipt", "validate-receipt"):
        command = commands.add_parser(name)
        if name == "validate-receipt":
            command.add_argument("receipt", type=Path)
        command.add_argument("checkpoint", type=Path)
        command.add_argument("tracking_report", type=Path)
        command.add_argument("stage_gate", type=Path)
        if name == "create-receipt":
            command.add_argument("output", type=Path)
            command.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-parent":
        result = validate_parent_checkpoint(
            args.checkpoint,
            args.tracking_report,
            args.superseded_v1_tracking_report,
        )
    elif args.command == "create-receipt":
        result = create_receipt(
            checkpoint=args.checkpoint,
            tracking_report=args.tracking_report,
            stage_gate=args.stage_gate,
            output=args.output,
            force=args.force,
        )
    else:
        result = validate_receipt(
            receipt=args.receipt,
            checkpoint=args.checkpoint,
            tracking_report=args.tracking_report,
            stage_gate=args.stage_gate,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if result.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
