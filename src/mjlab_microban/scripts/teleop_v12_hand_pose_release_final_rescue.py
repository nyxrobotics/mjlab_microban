"""Authenticate the pose-release model14900 final-scenario rescue route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from mjlab_microban.tasks.microban_teleop_v12_bootstrap import (
    sha256_file,
    validate_bootstrap_provenance,
)
from mjlab_microban.tasks.microban_teleop_v12_corner_rescue import (
    canonical_json_sha256,
)
from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release_final_rescue import (
    hand_pose_release_final_rescue_marker,
    validate_hand_pose_release_final_rescue_mix,
)
from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release_final_rescue_runner import (
    validate_hand_pose_release_failed_final_gate_report,
    validate_hand_pose_release_final_rescue_parent_payload,
)
from mjlab_microban.tasks.microban_teleop_v12_home_pose import (
    validate_teleop_v12_home_pose,
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


def validate_parent(
    checkpoint: Path, failed_gate_report: Path, mix: str
) -> dict[str, Any]:
    """Authenticate model14900 and the same run's failed 15000 gate report."""

    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise ValueError("Pose-release final rescue parent must be a regular file")
    resolved = checkpoint.expanduser().resolve(strict=True)
    digest = sha256_file(resolved)
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("Final rescue parent root must be a dictionary")
    corner = validate_hand_pose_release_final_rescue_parent_payload(
        payload, checkpoint_sha256=digest
    )
    infos = payload["infos"]
    validate_teleop_v12_home_pose(infos)
    validate_bilateral_site_order_checkpoint(infos)
    reject_preview_checkpoint(infos)
    validate_bootstrap_provenance(
        infos.get(TELEOP_V12_BOOTSTRAP_INFO_KEY), verify_files=True
    )
    failed = validate_hand_pose_release_failed_final_gate_report(
        failed_gate_report, parent_checkpoint=resolved
    )
    if failed["checkpoint_sha256"] == digest:
        raise ValueError("The failed final gate cannot name the parent itself")
    target = hand_pose_release_final_rescue_marker(
        parent_checkpoint_sha256=digest,
        failed_gate_checkpoint_sha256=failed["checkpoint_sha256"],
        failed_gate_tracking_report_sha256=failed["report_sha256"],
        failed_gate_failed_checks=failed["failed_checks"],
        failed_gate_failed_scenarios=failed["failed_scenarios"],
        inherited_corner_rescue_marker_sha256=(
            None if corner is None else canonical_json_sha256(corner)
        ),
        sampler_mix=validate_hand_pose_release_final_rescue_mix(mix),
    )
    if sha256_file(resolved) != digest:
        raise ValueError("Final rescue parent changed while validating")
    return {
        "schema_version": 1,
        "gate": "microban_teleop_v12_hand_pose_release_final_rescue_parent",
        "status": "pass",
        "checkpoint": {
            "path": str(resolved),
            "sha256": digest,
            "iteration": payload["iter"],
            "completed_updates": payload["iter"] + 1,
        },
        "failed_final_gate_tracking_report": {
            "path": str(failed_gate_report.expanduser().resolve(strict=True)),
            **failed,
        },
        "target": target,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    parent = commands.add_parser("validate-parent")
    parent.add_argument("checkpoint", type=Path)
    parent.add_argument("failed_final_gate_tracking_report", type=Path)
    parent.add_argument("--mix", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = validate_parent(
        args.checkpoint, args.failed_final_gate_tracking_report, args.mix
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if result.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
