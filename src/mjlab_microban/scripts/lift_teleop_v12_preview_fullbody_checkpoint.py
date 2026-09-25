"""Lift an accepted staged-v2 HMD/hand preview into the full-body phase."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch

from mjlab_microban.legacy_velocity_diagnostics import publish_json_atomic
from mjlab_microban.scripts.create_teleop_v12_preview_checkpoint import _nested_equal
from mjlab_microban.scripts.evaluate_teleop_v12_tracking import HMD_HAND_PROFILE
from mjlab_microban.scripts.promote_teleop_v12_preview_visual import (
    PHASE1_VISUAL_GATE,
    validate_embedded_phase1_acceptance,
    validate_visual_promotion_report,
)
from mjlab_microban.tasks.microban_teleop_v12_actor import (
    TELEOP_V12_EXTRA_OBSERVATION_COLUMNS,
    TELEOP_V12_FOOT_OBSERVATION_COLUMNS,
    TELEOP_V12_HAND_OBSERVATION_COLUMNS,
    TELEOP_V12_HMD_OBSERVATION_COLUMNS,
)
from mjlab_microban.tasks.microban_teleop_v12_bootstrap import sha256_file
from mjlab_microban.tasks.microban_teleop_v12_preview import (
    TELEOP_V12_PREVIEW_HMD_HAND_COLUMNS,
    TELEOP_V12_PREVIEW_INFO_KEY,
    TELEOP_V12_PREVIEW_PHASE1_ACCEPTANCE_INFO_KEY,
    TELEOP_V12_PREVIEW_PHASE1_STRICT_QUALITY,
    TELEOP_V12_PREVIEW_PHASE1_TRAINED_COMPLETED_UPDATES,
    TELEOP_V12_PREVIEW_PHASE1_TRAINED_ITERATION,
    TELEOP_V12_PREVIEW_PHASE2_LIFTED_COMPLETED_UPDATES,
    TELEOP_V12_PREVIEW_PHASE2_LIFTED_ITERATION,
    TELEOP_V12_PREVIEW_PHASE_FULL_BODY,
    TELEOP_V12_PREVIEW_PHASE_HMD_HAND,
    staged_preview_info,
    validate_preview_evaluation_report,
    validate_preview_marker,
)
from mjlab_microban.tasks.microban_teleop_v12_runner import _atomic_torch_save

_LOWER_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _require_expected_sha(value: str, *, label: str) -> None:
    if not _LOWER_SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")


def _require_foot_locked_and_phase1_learned(payload: dict[str, Any]) -> None:
    actor = payload.get("actor_state_dict")
    if not isinstance(actor, dict):
        raise TypeError("Phase-1 actor state is malformed")
    first = actor.get("mlp.0.weight")
    if not isinstance(first, torch.Tensor) or tuple(first.shape) != (512, 83):
        raise ValueError("Phase-1 actor W0 shape drifted")
    if not bool(torch.isfinite(first).all().item()):
        raise ValueError("Phase-1 actor W0 is non-finite")
    foot = first[:, TELEOP_V12_FOOT_OBSERVATION_COLUMNS]
    if not torch.equal(foot, torch.zeros_like(foot)):
        raise ValueError("Phase-1 foot adapter columns are not exact zero")
    for label, columns in (
        ("HMD", TELEOP_V12_HMD_OBSERVATION_COLUMNS),
        ("hand", TELEOP_V12_HAND_OBSERVATION_COLUMNS),
    ):
        if not bool(torch.any(first[:, columns] != 0).item()):
            raise ValueError(f"Phase-1 {label} adapter columns were not learned")

    optimizer = payload.get("optimizer_state_dict")
    if not isinstance(optimizer, dict) or not isinstance(optimizer.get("state"), dict):
        raise TypeError("Phase-1 optimizer state is malformed")
    candidates = []
    for state in optimizer["state"].values():
        if isinstance(state, dict) and any(
            isinstance(value, torch.Tensor) and tuple(value.shape) == (512, 83)
            for value in state.values()
        ):
            candidates.append(state)
    if len(candidates) != 1:
        raise ValueError("Could not identify phase-1 actor W0 optimizer state")
    moments = candidates[0]
    for name in ("exp_avg", "exp_avg_sq"):
        value = moments.get(name)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != (512, 83):
            raise ValueError(f"Phase-1 optimizer {name} shape drifted")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"Phase-1 optimizer {name} is non-finite")
        locked = value[:, TELEOP_V12_FOOT_OBSERVATION_COLUMNS]
        if not torch.equal(locked, torch.zeros_like(locked)):
            raise ValueError(f"Phase-1 foot optimizer {name} is not exact zero")
        active = value[:, TELEOP_V12_PREVIEW_HMD_HAND_COLUMNS]
        if not bool(torch.any(active != 0).item()):
            raise ValueError(f"Phase-1 active optimizer {name} was not learned")


def _validate_phase1_receipt(
    path: Path,
    *,
    expected_sha256: str,
    checkpoint_sha256: str,
    marker: dict[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    _require_expected_sha(expected_sha256, label="Phase-1 receipt hash")
    path = path.expanduser()
    if path.is_symlink():
        raise ValueError("Phase-1 acceptance receipt cannot be a symlink")
    path = path.resolve()
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"Phase-1 receipt SHA-256 mismatch: {actual}")
    report = json.loads(path.read_text())
    if not isinstance(report, dict):
        raise TypeError("Phase-1 acceptance receipt is malformed")
    gate = report.get("gate")
    if gate == "microban_teleop_v12_nondeployable_preview_acceptance":
        validate_preview_evaluation_report(
            report,
            checkpoint_sha256=checkpoint_sha256,
            iteration=TELEOP_V12_PREVIEW_PHASE1_TRAINED_ITERATION,
            marker=marker,
        )
        if (
            report.get("status") != "pass"
            or report.get("quality_class")
            != TELEOP_V12_PREVIEW_PHASE1_STRICT_QUALITY
            or report.get("settings", {}).get("tracking_profile") != HMD_HAND_PROFILE
        ):
            raise ValueError("Phase-1 strict acceptance receipt did not pass")
        quality_class = TELEOP_V12_PREVIEW_PHASE1_STRICT_QUALITY
    elif gate == PHASE1_VISUAL_GATE:
        recomputed = validate_visual_promotion_report(
            report,
            checkpoint_sha256=checkpoint_sha256,
            iteration=TELEOP_V12_PREVIEW_PHASE1_TRAINED_ITERATION,
            marker=marker,
        )
        quality_class = recomputed["quality_class"]
    else:
        raise ValueError("Phase-1 receipt gate is not accepted")
    return actual, quality_class, report


def lift_fullbody_preview_checkpoint(
    source: Path,
    destination: Path,
    *,
    expected_source_sha256: str,
    phase1_acceptance_receipt: Path,
    expected_receipt_sha256: str,
) -> dict[str, Any]:
    """Clock-lift an accepted model_7100 without changing learned state."""

    _require_expected_sha(expected_source_sha256, label="Phase-1 checkpoint hash")
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if source == destination:
        raise ValueError("Phase-2 lift refuses to overwrite its source")
    if destination.exists():
        raise FileExistsError(f"Phase-2 destination already exists: {destination}")
    source_sha = sha256_file(source)
    if source_sha != expected_source_sha256:
        raise ValueError(f"Phase-1 checkpoint SHA-256 mismatch: {source_sha}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("infos"), dict):
        raise TypeError("Phase-1 checkpoint payload is malformed")
    infos = payload["infos"]
    if (
        payload.get("iter") != TELEOP_V12_PREVIEW_PHASE1_TRAINED_ITERATION
        or infos.get("env_state", {}).get("common_step_counter")
        != TELEOP_V12_PREVIEW_PHASE1_TRAINED_COMPLETED_UPDATES * 24
        or infos.get("active_actor_columns_at_save")
        != list(TELEOP_V12_PREVIEW_HMD_HAND_COLUMNS)
    ):
        raise ValueError("Phase-1 checkpoint clock/active columns drifted")
    marker = validate_preview_marker(
        infos,
        iteration=TELEOP_V12_PREVIEW_PHASE1_TRAINED_ITERATION,
        required_phase=TELEOP_V12_PREVIEW_PHASE_HMD_HAND,
    )
    _require_foot_locked_and_phase1_learned(payload)
    receipt_sha, quality_class, phase1_acceptance = _validate_phase1_receipt(
        phase1_acceptance_receipt,
        expected_sha256=expected_receipt_sha256,
        checkpoint_sha256=source_sha,
        marker=marker,
    )

    lifted = copy.deepcopy(payload)
    lifted["iter"] = TELEOP_V12_PREVIEW_PHASE2_LIFTED_ITERATION
    lifted_infos = lifted["infos"]
    lifted_infos["env_state"] = {
        "common_step_counter": TELEOP_V12_PREVIEW_PHASE2_LIFTED_COMPLETED_UPDATES * 24
    }
    lifted_infos["active_actor_columns_at_save"] = list(
        TELEOP_V12_EXTRA_OBSERVATION_COLUMNS
    )
    lifted_infos[TELEOP_V12_PREVIEW_INFO_KEY] = staged_preview_info(
        phase=TELEOP_V12_PREVIEW_PHASE_FULL_BODY,
        phase_source_checkpoint_sha256=source_sha,
        phase1_acceptance_receipt_sha256=receipt_sha,
        phase1_quality_class=quality_class,
    )
    lifted_infos[TELEOP_V12_PREVIEW_PHASE1_ACCEPTANCE_INFO_KEY] = copy.deepcopy(
        phase1_acceptance
    )
    validate_embedded_phase1_acceptance(
        lifted_infos,
        fullbody_marker=lifted_infos[TELEOP_V12_PREVIEW_INFO_KEY],
    )
    for key in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict"):
        if not _nested_equal(payload.get(key), lifted.get(key)):
            raise RuntimeError(f"Phase-2 clock lift changed {key}")

    _atomic_torch_save(lifted, destination)
    if sha256_file(source) != source_sha or sha256_file(
        phase1_acceptance_receipt.expanduser().resolve()
    ) != receipt_sha:
        destination.unlink(missing_ok=True)
        raise RuntimeError("Phase-2 source or receipt changed during clock lift")
    verified = torch.load(destination, map_location="cpu", weights_only=False)
    verified_marker = validate_preview_marker(
        verified.get("infos"),
        iteration=verified.get("iter", -1),
        required_phase=TELEOP_V12_PREVIEW_PHASE_FULL_BODY,
    )
    validate_embedded_phase1_acceptance(
        verified.get("infos"), fullbody_marker=verified_marker
    )
    _require_foot_locked_and_phase1_learned(verified)
    return {
        "schema_version": 1,
        "creator": "microban_teleop_v12_fullbody_preview_phase_lift_v2",
        "status": "pass",
        "preview_non_deployable": True,
        "source": {
            "path": str(source),
            "sha256": source_sha,
            "iteration": TELEOP_V12_PREVIEW_PHASE1_TRAINED_ITERATION,
            "completed_updates": TELEOP_V12_PREVIEW_PHASE1_TRAINED_COMPLETED_UPDATES,
        },
        "phase1_acceptance_receipt": {
            "path": str(phase1_acceptance_receipt.expanduser().resolve()),
            "sha256": receipt_sha,
            "quality_class": quality_class,
        },
        "output": {
            "path": str(destination),
            "sha256": sha256_file(destination),
            "iteration": TELEOP_V12_PREVIEW_PHASE2_LIFTED_ITERATION,
            "completed_updates": TELEOP_V12_PREVIEW_PHASE2_LIFTED_COMPLETED_UPDATES,
        },
        "checks": {
            "phase1_acceptance_passed": True,
            "actor_unchanged": True,
            "critic_unchanged": True,
            "optimizer_unchanged": True,
            "learned_hmd_hand_preserved": True,
            "foot6_w0_and_adam_exact_zero": True,
            "full20_active_after_resume": True,
            "canonical_deployment_forbidden": True,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--phase1-acceptance-receipt", type=Path, required=True)
    parser.add_argument("--expected-receipt-sha256", required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = lift_fullbody_preview_checkpoint(
        args.source,
        args.destination,
        expected_source_sha256=args.expected_source_sha256,
        phase1_acceptance_receipt=args.phase1_acceptance_receipt,
        expected_receipt_sha256=args.expected_receipt_sha256,
    )
    if args.output is not None:
        if args.output.expanduser().exists():
            raise FileExistsError(f"Receipt exists: {args.output}")
        publish_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
