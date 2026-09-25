"""Create a hash-bound promotion receipt for the one deadline fallback."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from mjlab_microban.legacy_velocity_diagnostics import publish_json_atomic
from mjlab_microban.scripts.evaluate_teleop_v12_tracking import (
    DEADLINE_CANARY_FALLBACK_PROFILE,
    DEADLINE_FALLBACK_PROFILE,
    FOOT_ACTIVATION_CANARY_PROFILE,
    required_tracking_check_names,
)
from mjlab_microban.scripts.teleop_v12_stage import (
    _load_json,
    _validate_deadline_canary_strict_report,
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
    MICROBAN_TELEOP_V12_DEADLINE_CANARY_CHECKPOINT_SHA256,
    MICROBAN_TELEOP_V12_DEADLINE_CANARY_STRICT_REPORT_SHA256,
    MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_INFO_KEY,
    MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_REVISION,
    MICROBAN_TELEOP_V12_DEADLINE_POST_CANARY_INFO_KEY,
    MICROBAN_TELEOP_V12_DEADLINE_POST_CANARY_RECEIPT_SHA256,
    MICROBAN_TELEOP_V12_DEADLINE_POST_CANARY_REVISION,
    deadline_fallback_marker,
    deadline_post_canary_marker,
    validate_deadline_fallback_canary_payload,
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
DEADLINE_POST_CANARY_RECEIPT_GATE = "microban_teleop_v12_deadline_post_canary_fallback"


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


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


def _canary_checkpoint_identity(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.expanduser().resolve(strict=True)
    digest = sha256_file(resolved)
    if digest != MICROBAN_TELEOP_V12_DEADLINE_CANARY_CHECKPOINT_SHA256:
        raise ValueError("Deadline canary checkpoint SHA-256 mismatch")
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("infos"), dict):
        raise TypeError("Deadline canary checkpoint payload is malformed")
    marker = validate_deadline_fallback_canary_payload(
        payload, verify_parent_files=True, checkpoint_sha256=digest
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


def _validate_deadline_post_canary_receipt_structure(
    receipt: Mapping[str, Any], *, checkpoint_sha256: str
) -> dict[str, Any]:
    """Validate exact receipt content independent of its serialized bytes."""

    checkpoint = receipt.get("checkpoint")
    strict = receipt.get("strict_failure_report")
    fallback = receipt.get("fallback_tracking_report")
    stage = receipt.get("full_stage_gate")
    promotion = receipt.get("promotion")
    if (
        checkpoint_sha256 != MICROBAN_TELEOP_V12_DEADLINE_CANARY_CHECKPOINT_SHA256
        or receipt.get("schema_version") != DEADLINE_FALLBACK_RECEIPT_SCHEMA_VERSION
        or receipt.get("gate") != DEADLINE_POST_CANARY_RECEIPT_GATE
        or receipt.get("status") != "pass"
        or receipt.get("revision") != MICROBAN_TELEOP_V12_DEADLINE_POST_CANARY_REVISION
        or receipt.get("lineage") != deadline_fallback_marker()
        or receipt.get("post_canary_authorization") != deadline_post_canary_marker()
        or not isinstance(checkpoint, Mapping)
        or checkpoint.get("sha256") != checkpoint_sha256
        or checkpoint.get("iteration") != 10_099
        or checkpoint.get("completed_updates") != 10_100
        or not isinstance(checkpoint.get("path"), str)
        or not checkpoint["path"]
    ):
        raise ValueError("Deadline post-canary receipt identity drifted")
    if (
        not isinstance(strict, Mapping)
        or strict.get("sha256")
        != MICROBAN_TELEOP_V12_DEADLINE_CANARY_STRICT_REPORT_SHA256
        or strict.get("profile") != FOOT_ACTIVATION_CANARY_PROFILE
        or strict.get("status") != "fail"
        or strict.get("failed_checks") != ["hand_tracking_rms"]
        or not isinstance(strict.get("path"), str)
        or not strict["path"]
    ):
        raise ValueError("Deadline post-canary strict evidence drifted")
    expected_checks = required_tracking_check_names(DEADLINE_CANARY_FALLBACK_PROFILE)
    if (
        not isinstance(fallback, Mapping)
        or fallback.get("profile") != DEADLINE_CANARY_FALLBACK_PROFILE
        or fallback.get("status") != "pass"
        or not _is_sha256(fallback.get("sha256"))
        or not isinstance(fallback.get("path"), str)
        or not fallback["path"]
        or not isinstance(fallback.get("checks"), Mapping)
        or set(fallback["checks"]) != set(expected_checks)
        or not all(value is True for value in fallback["checks"].values())
    ):
        raise ValueError("Deadline post-canary fallback evidence drifted")
    report_hashes = stage.get("report_sha256") if isinstance(stage, Mapping) else None
    if (
        not isinstance(stage, Mapping)
        or stage.get("schema_version") != 2
        or stage.get("status") != "pass"
        or stage.get("checkpoint_sha256") != checkpoint_sha256
        or stage.get("tracking_profile") != DEADLINE_CANARY_FALLBACK_PROFILE
        or not isinstance(stage.get("path"), str)
        or not stage["path"]
        or not _is_sha256(stage.get("sha256"))
        or not isinstance(report_hashes, Mapping)
        or set(report_hashes) != {"locomotion", "tracking", "onnx"}
        or any(not _is_sha256(value) for value in report_hashes.values())
        or report_hashes.get("tracking") != fallback.get("sha256")
    ):
        raise ValueError("Deadline post-canary full stage gate drifted")
    if promotion != {
        "eligible": True,
        "completed_updates": 10_100,
        "next_completed_updates": 15_000,
        "only_threshold_change": "hand_rms_m_max_0.030_to_0.035",
        "hand_p95_m_max": 0.05,
        "foot_activation_checks_changed": False,
        "safety_thresholds_changed": False,
        "locomotion_gate_changed": False,
        "onnx_gate_changed": False,
        "requires_schema_v2_full_stage_gate": True,
    }:
        raise ValueError("Deadline post-canary promotion contract drifted")
    return dict(receipt)


def validate_deadline_post_canary_receipt_payload(
    receipt: Mapping[str, Any], *, checkpoint_sha256: str, receipt_sha256: str
) -> dict[str, Any]:
    """Validate a receipt behind the pinned serialized receipt hash."""

    if receipt_sha256 != MICROBAN_TELEOP_V12_DEADLINE_POST_CANARY_RECEIPT_SHA256:
        raise ValueError("Deadline post-canary receipt SHA-256 mismatch")
    return _validate_deadline_post_canary_receipt_structure(
        receipt, checkpoint_sha256=checkpoint_sha256
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


def build_post_canary_receipt(
    *,
    checkpoint: Path,
    strict_tracking_report: Path,
    fallback_tracking_report: Path,
    stage_gate: Path,
) -> dict[str, Any]:
    """Rebuild all evidence for the one promotable model10099 canary."""

    identity, lineage = _canary_checkpoint_identity(checkpoint)
    expected_identity = {
        name: identity[name] for name in ("sha256", "iteration", "completed_updates")
    }
    strict_path = strict_tracking_report.expanduser().resolve(strict=True)
    _validate_deadline_canary_strict_report(strict_path, expected_identity)
    fallback_path = fallback_tracking_report.expanduser().resolve(strict=True)
    tracking = _load_json(fallback_path)
    _validate_tracking_report(
        tracking,
        expected_identity,
        profile_override=DEADLINE_CANARY_FALLBACK_PROFILE,
    )
    if tracking.get("status") != "pass":
        raise ValueError("Deadline canary fallback tracking evidence did not pass")

    gate_path = stage_gate.expanduser().resolve(strict=True)
    gate = validate_gate(gate_path, Path(identity["path"]))
    authorization = deadline_post_canary_marker()
    reports = gate.get("reports")
    report_hashes = gate.get("report_sha256")
    if (
        gate.get(MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_INFO_KEY) != lineage
        or gate.get(MICROBAN_TELEOP_V12_DEADLINE_POST_CANARY_INFO_KEY) != authorization
        or gate.get("checkpoint_sha256") != identity["sha256"]
        or gate.get("tracking_profile") != DEADLINE_CANARY_FALLBACK_PROFILE
        or not isinstance(reports, dict)
        or not isinstance(report_hashes, dict)
        or resolve_bootstrap_artifact_path(reports.get("tracking", "")) != fallback_path
        or report_hashes.get("tracking") != sha256_file(fallback_path)
        or resolve_bootstrap_artifact_path(
            gate.get("deadline_canary_strict_report", "")
        )
        != strict_path
        or gate.get("deadline_canary_strict_report_sha256")
        != MICROBAN_TELEOP_V12_DEADLINE_CANARY_STRICT_REPORT_SHA256
    ):
        raise ValueError("Stage gate is not bound to the canary promotion evidence")

    receipt = {
        "schema_version": DEADLINE_FALLBACK_RECEIPT_SCHEMA_VERSION,
        "gate": DEADLINE_POST_CANARY_RECEIPT_GATE,
        "status": "pass",
        "revision": MICROBAN_TELEOP_V12_DEADLINE_POST_CANARY_REVISION,
        "checkpoint": identity,
        "lineage": lineage,
        "post_canary_authorization": authorization,
        "strict_failure_report": {
            "path": str(strict_path),
            "sha256": sha256_file(strict_path),
            "profile": FOOT_ACTIVATION_CANARY_PROFILE,
            "status": "fail",
            "failed_checks": ["hand_tracking_rms"],
        },
        "fallback_tracking_report": {
            "path": str(fallback_path),
            "sha256": sha256_file(fallback_path),
            "profile": DEADLINE_CANARY_FALLBACK_PROFILE,
            "status": "pass",
            "checks": tracking["checks"],
        },
        "full_stage_gate": {
            "path": str(gate_path),
            "sha256": sha256_file(gate_path),
            "schema_version": 2,
            "status": "pass",
            "checkpoint_sha256": identity["sha256"],
            "tracking_profile": DEADLINE_CANARY_FALLBACK_PROFILE,
            "report_sha256": gate["report_sha256"],
        },
        "promotion": {
            "eligible": True,
            "completed_updates": identity["completed_updates"],
            "next_completed_updates": 15_000,
            "only_threshold_change": "hand_rms_m_max_0.030_to_0.035",
            "hand_p95_m_max": 0.05,
            "foot_activation_checks_changed": False,
            "safety_thresholds_changed": False,
            "locomotion_gate_changed": False,
            "onnx_gate_changed": False,
            "requires_schema_v2_full_stage_gate": True,
        },
    }
    return _validate_deadline_post_canary_receipt_structure(
        receipt, checkpoint_sha256=identity["sha256"]
    )


def create_post_canary_receipt(
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
        raise FileExistsError(f"Deadline post-canary receipt exists: {destination}")
    if destination.is_symlink():
        raise ValueError("Deadline post-canary receipt output must not be a symlink")
    receipt = build_post_canary_receipt(
        checkpoint=checkpoint,
        strict_tracking_report=strict_tracking_report,
        fallback_tracking_report=fallback_tracking_report,
        stage_gate=stage_gate,
    )
    publish_json_atomic(destination, receipt)
    return receipt


def validate_post_canary_receipt(
    *,
    receipt: Path,
    checkpoint: Path,
    strict_tracking_report: Path,
    fallback_tracking_report: Path,
    stage_gate: Path,
) -> dict[str, Any]:
    actual = _load_json(receipt.expanduser().resolve(strict=True))
    expected = build_post_canary_receipt(
        checkpoint=checkpoint,
        strict_tracking_report=strict_tracking_report,
        fallback_tracking_report=fallback_tracking_report,
        stage_gate=stage_gate,
    )
    if actual != expected:
        raise ValueError("Deadline post-canary receipt content/hash binding drifted")
    return actual


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in (
        "create-receipt",
        "validate-receipt",
        "create-canary-receipt",
        "validate-canary-receipt",
    ):
        command = commands.add_parser(name)
        if name.startswith("validate-"):
            command.add_argument("receipt", type=Path)
        command.add_argument("checkpoint", type=Path)
        command.add_argument("strict_tracking_report", type=Path)
        command.add_argument("fallback_tracking_report", type=Path)
        command.add_argument("stage_gate", type=Path)
        if name.startswith("create-"):
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
    elif args.command == "validate-receipt":
        result = validate_receipt(receipt=args.receipt, **common)
    elif args.command == "create-canary-receipt":
        result = create_post_canary_receipt(
            **common,
            output=args.output,
            force=args.force,
        )
    else:
        result = validate_post_canary_receipt(receipt=args.receipt, **common)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
