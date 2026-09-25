"""Create a hash-bound promotion receipt for the one deadline fallback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from mjlab_microban.legacy_velocity_diagnostics import publish_json_atomic
from mjlab_microban.scripts.evaluate_teleop_v12_tracking import (
    DEADLINE_FALLBACK_PROFILE,
)
from mjlab_microban.scripts.teleop_v12_stage import (
    _load_json,
    _validate_deadline_strict_report,
    _validate_tracking_report,
    validate_gate,
)
from mjlab_microban.tasks.microban_teleop_v12_bootstrap import (
    resolve_bootstrap_artifact_path,
    sha256_file,
    validate_bootstrap_provenance,
)
from mjlab_microban.tasks.microban_teleop_v12_deadline_fallback import (
    MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_INFO_KEY,
    MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_REVISION,
    validate_deadline_fallback_checkpoint_payload,
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

DEADLINE_FALLBACK_RECEIPT_SCHEMA_VERSION = 1
DEADLINE_FALLBACK_RECEIPT_GATE = "microban_teleop_v12_deadline_fallback"


def _checkpoint_identity(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.expanduser().resolve(strict=True)
    digest = sha256_file(resolved)
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("infos"), dict):
        raise TypeError("Deadline fallback checkpoint payload is malformed")
    marker = validate_deadline_fallback_checkpoint_payload(
        payload, checkpoint_sha256=digest
    )
    infos = payload["infos"]
    validate_bilateral_site_order_checkpoint(infos)
    reject_preview_checkpoint(infos)
    validate_bootstrap_provenance(
        infos.get(TELEOP_V12_BOOTSTRAP_INFO_KEY), verify_files=True
    )
    iteration = payload["iter"]
    return (
        {
            "path": str(resolved),
            "sha256": digest,
            "iteration": iteration,
            "completed_updates": iteration + 1,
        },
        marker,
    )


def build_receipt(
    *,
    checkpoint: Path,
    strict_tracking_report: Path,
    fallback_tracking_report: Path,
    stage_gate: Path,
) -> dict[str, Any]:
    """Rebuild all evidence and return the only promotable fallback receipt."""

    identity, marker = _checkpoint_identity(checkpoint)
    expected_identity = {
        name: identity[name] for name in ("sha256", "iteration", "completed_updates")
    }
    strict_path = strict_tracking_report.expanduser().resolve(strict=True)
    _validate_deadline_strict_report(strict_path, expected_identity)
    tracking_path = fallback_tracking_report.expanduser().resolve(strict=True)
    tracking = _load_json(tracking_path)
    _validate_tracking_report(
        tracking,
        expected_identity,
        profile_override=DEADLINE_FALLBACK_PROFILE,
    )
    if tracking.get("status") != "pass":
        raise ValueError("Deadline fallback tracking evidence did not pass")

    gate_path = stage_gate.expanduser().resolve(strict=True)
    gate = validate_gate(gate_path, Path(identity["path"]))
    if gate.get(MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_INFO_KEY) != marker:
        raise ValueError("Stage gate deadline authorization marker drifted")
    reports = gate.get("reports")
    report_hashes = gate.get("report_sha256")
    if (
        not isinstance(reports, dict)
        or not isinstance(report_hashes, dict)
        or resolve_bootstrap_artifact_path(reports.get("tracking", "")) != tracking_path
        or report_hashes.get("tracking") != sha256_file(tracking_path)
        or resolve_bootstrap_artifact_path(
            gate.get("deadline_fallback_strict_report", "")
        )
        != strict_path
        or gate.get("deadline_fallback_strict_report_sha256")
        != sha256_file(strict_path)
    ):
        raise ValueError("Stage gate is not bound to both tracking reports")

    return {
        "schema_version": DEADLINE_FALLBACK_RECEIPT_SCHEMA_VERSION,
        "gate": DEADLINE_FALLBACK_RECEIPT_GATE,
        "status": "pass",
        "revision": MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_REVISION,
        "checkpoint": identity,
        "lineage": marker,
        "strict_failure_report": {
            "path": str(strict_path),
            "sha256": sha256_file(strict_path),
            "status": "fail",
            "failed_checks": ["hand_tracking_rms"],
        },
        "fallback_tracking_report": {
            "path": str(tracking_path),
            "sha256": sha256_file(tracking_path),
            "profile": DEADLINE_FALLBACK_PROFILE,
            "status": "pass",
            "checks": tracking["checks"],
        },
        "full_stage_gate": {
            "path": str(gate_path),
            "sha256": sha256_file(gate_path),
            "schema_version": 2,
            "status": "pass",
            "report_sha256": gate["report_sha256"],
        },
        "promotion": {
            "eligible": True,
            "completed_updates": identity["completed_updates"],
            "only_threshold_change": "hand_rms_m_max_0.030_to_0.035",
            "hand_p95_m_max": 0.05,
            "safety_thresholds_changed": False,
            "locomotion_gate_changed": False,
            "onnx_gate_changed": False,
            "requires_schema_v2_full_stage_gate": True,
        },
    }


def create_receipt(
    *,
    checkpoint: Path,
    strict_tracking_report: Path,
    fallback_tracking_report: Path,
    stage_gate: Path,
    output: Path,
    force: bool,
) -> dict[str, Any]:
    destination = output.expanduser().resolve()
    if destination.exists() and not force:
        raise FileExistsError(f"Deadline fallback receipt exists: {destination}")
    if destination.is_symlink():
        raise ValueError("Deadline fallback receipt output must not be a symlink")
    receipt = build_receipt(
        checkpoint=checkpoint,
        strict_tracking_report=strict_tracking_report,
        fallback_tracking_report=fallback_tracking_report,
        stage_gate=stage_gate,
    )
    publish_json_atomic(destination, receipt)
    return receipt


def validate_receipt(
    *,
    receipt: Path,
    checkpoint: Path,
    strict_tracking_report: Path,
    fallback_tracking_report: Path,
    stage_gate: Path,
) -> dict[str, Any]:
    actual = _load_json(receipt.expanduser().resolve(strict=True))
    expected = build_receipt(
        checkpoint=checkpoint,
        strict_tracking_report=strict_tracking_report,
        fallback_tracking_report=fallback_tracking_report,
        stage_gate=stage_gate,
    )
    if actual != expected:
        raise ValueError("Deadline fallback receipt content/hash binding drifted")
    return actual


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("create-receipt", "validate-receipt"):
        command = commands.add_parser(name)
        if name == "validate-receipt":
            command.add_argument("receipt", type=Path)
        command.add_argument("checkpoint", type=Path)
        command.add_argument("strict_tracking_report", type=Path)
        command.add_argument("fallback_tracking_report", type=Path)
        command.add_argument("stage_gate", type=Path)
        if name == "create-receipt":
            command.add_argument("output", type=Path)
            command.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    common = {
        "checkpoint": args.checkpoint,
        "strict_tracking_report": args.strict_tracking_report,
        "fallback_tracking_report": args.fallback_tracking_report,
        "stage_gate": args.stage_gate,
    }
    if args.command == "create-receipt":
        result = create_receipt(
            **common,
            output=args.output,
            force=args.force,
        )
    else:
        result = validate_receipt(receipt=args.receipt, **common)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
