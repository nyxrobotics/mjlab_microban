#!/usr/bin/env python3
"""Adjudicate one immutable direct-IK overlay simulation report.

This script performs no simulation and imports no MJLab task code.  It binds a
raw 45-case report by SHA-256, independently recomputes every safety check, and
writes a simulation-only acceptance receipt.  The 5 degree allowance applies
only to measured dynamic joint overshoot; commanded arm targets must still stay
within their soft limits to numerical tolerance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mjlab_microban.teleop_v12_safety import (
    ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_DEG,
    ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD,
    COMMANDED_TARGET_SOFT_LIMIT_EXCESS_MAX_RAD,
)

GATE_REVISION = "microban_simulation_only_direct_ik_overlay_acceptance_v1"
EXPECTED_CASE_COUNT = 45


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{label} must be a finite number") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def _read_bound_report(
    path: Path, expected_sha256: str, *, label: str = "Raw report"
) -> tuple[bytes, dict[str, Any]]:
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise ValueError(
            f"Expected {label.lower()} SHA-256 must be 64 lowercase hex characters"
        )
    # Check the supplied leaf itself before any resolution so a symlink cannot
    # inherit the regular-file status of its target.
    path = path.expanduser().absolute()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch: {digest}")
    try:
        report = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(report, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return payload, report


def adjudicate_report(
    report: dict[str, Any], *, parent_path: Path, parent_sha256: str
) -> dict[str, Any]:
    """Recompute the complete 45-case simulation-only acceptance decision."""

    results = report.get("results")
    settings = report.get("settings")
    if not isinstance(results, list) or not all(
        isinstance(item, dict) for item in results
    ):
        raise TypeError("Raw report results must be a list of objects")
    if not isinstance(settings, dict):
        raise TypeError("Raw report settings must be an object")

    configured_steps = settings.get("steps")
    if (
        not isinstance(configured_steps, int)
        or isinstance(configured_steps, bool)
        or configured_steps <= 0
    ):
        raise TypeError("Raw report settings.steps must be a positive integer")

    identities: set[tuple[str, str]] = set()
    completed = True
    no_falls = True
    no_nonfinite = True
    all_steps_executed = True
    neutral_targets_all_steps = True
    maximum_actual_overshoot = 0.0
    maximum_target_excess = 0.0
    worst_actual_case: dict[str, Any] | None = None
    worst_target_case: dict[str, Any] | None = None

    for index, item in enumerate(results):
        scenario = item.get("scenario")
        profile = item.get("arm_profile")
        if not isinstance(scenario, str) or not scenario:
            raise TypeError(f"results[{index}].scenario must be a non-empty string")
        if not isinstance(profile, str) or not profile:
            raise TypeError(f"results[{index}].arm_profile must be a non-empty string")
        identity = (scenario, profile)
        if identity in identities:
            raise ValueError(f"Duplicate case identity: {identity}")
        identities.add(identity)

        executed_steps = item.get("executed_steps")
        neutral_steps = item.get("neutral_foot_hand_target_verified_steps")
        if not isinstance(executed_steps, int) or isinstance(executed_steps, bool):
            raise TypeError(f"results[{index}].executed_steps must be an integer")
        if not isinstance(neutral_steps, int) or isinstance(neutral_steps, bool):
            raise TypeError(
                f"results[{index}].neutral_foot_hand_target_verified_steps must be an integer"
            )

        actual = _finite_number(
            item.get("maximum_actual_soft_limit_violation_rad"),
            label=f"results[{index}].maximum_actual_soft_limit_violation_rad",
        )
        target = _finite_number(
            item.get("maximum_arm_target_soft_limit_excess_rad"),
            label=f"results[{index}].maximum_arm_target_soft_limit_excess_rad",
        )
        if worst_actual_case is None or actual > maximum_actual_overshoot:
            maximum_actual_overshoot = actual
            worst_actual_case = {
                "scenario": scenario,
                "arm_profile": profile,
                "value_rad": actual,
            }
        if worst_target_case is None or target > maximum_target_excess:
            maximum_target_excess = target
            worst_target_case = {
                "scenario": scenario,
                "arm_profile": profile,
                "value_rad": target,
            }

        completed = completed and item.get("completed") is True
        no_falls = no_falls and item.get("fell") is False
        no_nonfinite = no_nonfinite and item.get("nonfinite") is None
        all_steps_executed = all_steps_executed and executed_steps == configured_steps
        neutral_targets_all_steps = (
            neutral_targets_all_steps and neutral_steps == executed_steps
        )

    checks = {
        "case_count_is_45": len(results) == EXPECTED_CASE_COUNT,
        "case_identities_are_unique": len(identities) == len(results),
        "all_cases_completed": completed,
        "all_configured_steps_executed": all_steps_executed,
        "fall_count_is_zero": no_falls,
        "nonfinite_count_is_zero": no_nonfinite,
        "commanded_arm_target_soft_limit_excess_within_1e_7_rad": (
            maximum_target_excess <= COMMANDED_TARGET_SOFT_LIMIT_EXCESS_MAX_RAD
        ),
        "neutral_foot_hand_targets_verified_all_steps": neutral_targets_all_steps,
        "actual_dynamic_soft_limit_overshoot_within_5deg": (
            maximum_actual_overshoot <= ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD
        ),
    }
    status = "pass" if all(checks.values()) else "fail"
    return {
        "schema_version": 1,
        "gate": GATE_REVISION,
        "status": status,
        "simulation_only": True,
        "physical_deployment_accepted": False,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "parent_raw_report": {
            "path": str(parent_path.expanduser().resolve()),
            "sha256": parent_sha256,
        },
        "policy": {
            "actual_dynamic_soft_limit_overshoot_tolerance_deg": (
                ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_DEG
            ),
            "actual_dynamic_soft_limit_overshoot_tolerance_rad": (
                ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD
            ),
            "commanded_arm_target_soft_limit_excess_tolerance_rad": (
                COMMANDED_TARGET_SOFT_LIMIT_EXCESS_MAX_RAD
            ),
            "scope": "simulation_only",
        },
        "evidence": {
            "case_count": len(results),
            "configured_steps_per_case": configured_steps,
            "maximum_actual_dynamic_soft_limit_overshoot_rad": (
                maximum_actual_overshoot
            ),
            "maximum_actual_dynamic_soft_limit_overshoot_deg": math.degrees(
                maximum_actual_overshoot
            ),
            "worst_actual_dynamic_soft_limit_case": worst_actual_case,
            "maximum_commanded_arm_target_soft_limit_excess_rad": (
                maximum_target_excess
            ),
            "worst_commanded_arm_target_soft_limit_case": worst_target_case,
        },
        "checks": checks,
    }


def verify_receipt_payload(receipt: dict[str, Any]) -> dict[str, Any]:
    """Re-hash the parent raw report and independently reproduce a receipt."""

    if receipt.get("schema_version") != 1:
        raise ValueError("Receipt schema_version must be exactly 1")
    parent = receipt.get("parent_raw_report")
    if not isinstance(parent, dict):
        raise TypeError("Receipt parent_raw_report must be an object")
    parent_path = parent.get("path")
    parent_sha256 = parent.get("sha256")
    if not isinstance(parent_path, str) or not parent_path:
        raise TypeError("Receipt parent raw report path must be a non-empty string")
    if not isinstance(parent_sha256, str):
        raise TypeError("Receipt parent raw report SHA-256 must be a string")

    _raw_bytes, raw_report = _read_bound_report(
        Path(parent_path), parent_sha256, label="Parent raw report"
    )
    recomputed = adjudicate_report(
        raw_report,
        parent_path=Path(parent_path),
        parent_sha256=parent_sha256,
    )
    compared_fields = (
        "schema_version",
        "gate",
        "status",
        "simulation_only",
        "physical_deployment_accepted",
        "parent_raw_report",
        "policy",
        "evidence",
        "checks",
    )
    for field in compared_fields:
        if receipt.get(field) != recomputed[field]:
            raise ValueError(
                f"Receipt field does not match recomputed evidence: {field}"
            )
    if receipt["status"] != "pass" or not all(
        value is True for value in receipt["checks"].values()
    ):
        raise ValueError("Receipt is not a passing all-checks acceptance")
    return recomputed


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> Path:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Output already exists: {destination}")
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _self_test() -> None:
    def case(index: int) -> dict[str, Any]:
        return {
            "scenario": f"scenario_{index // 5}",
            "arm_profile": f"profile_{index % 5}",
            "completed": True,
            "executed_steps": 300,
            "fell": False,
            "nonfinite": None,
            "maximum_actual_soft_limit_violation_rad": math.radians(2.01),
            "maximum_arm_target_soft_limit_excess_rad": 0.0,
            "neutral_foot_hand_target_verified_steps": 300,
        }

    good = {"settings": {"steps": 300}, "results": [case(i) for i in range(45)]}
    accepted = adjudicate_report(
        good, parent_path=Path("raw.json"), parent_sha256="0" * 64
    )
    assert accepted["status"] == "pass"
    assert all(accepted["checks"].values())

    excessive_actual = json.loads(json.dumps(good))
    excessive_actual["results"][0]["maximum_actual_soft_limit_violation_rad"] = (
        math.radians(5.01)
    )
    assert (
        adjudicate_report(
            excessive_actual, parent_path=Path("raw.json"), parent_sha256="0" * 64
        )["status"]
        == "fail"
    )

    target_excess = json.loads(json.dumps(good))
    target_excess["results"][0]["maximum_arm_target_soft_limit_excess_rad"] = 1.1e-7
    assert (
        adjudicate_report(
            target_excess, parent_path=Path("raw.json"), parent_sha256="0" * 64
        )["status"]
        == "fail"
    )

    fallen = json.loads(json.dumps(good))
    fallen["results"][0]["fell"] = True
    assert (
        adjudicate_report(fallen, parent_path=Path("raw.json"), parent_sha256="0" * 64)[
            "status"
        ]
        == "fail"
    )

    with tempfile.TemporaryDirectory(prefix="direct-ik-adjudicator-test-") as directory:
        raw_path = Path(directory) / "raw.json"
        raw_bytes = (
            json.dumps(good, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode()
        raw_path.write_bytes(raw_bytes)
        raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        verifiable = adjudicate_report(
            good, parent_path=raw_path, parent_sha256=raw_sha256
        )
        reproduced = verify_receipt_payload(verifiable)
        assert reproduced["checks"] == verifiable["checks"]

        receipt_path = Path(directory) / "receipt.json"
        receipt_bytes = (
            json.dumps(verifiable, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode()
        receipt_path.write_bytes(receipt_bytes)
        receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
        _bytes, loaded_receipt = _read_bound_report(
            receipt_path, receipt_sha256, label="Receipt"
        )
        verify_receipt_payload(loaded_receipt)

        receipt_symlink = Path(directory) / "receipt-link.json"
        receipt_symlink.symlink_to(receipt_path)
        try:
            _read_bound_report(receipt_symlink, receipt_sha256, label="Receipt")
        except ValueError:
            pass
        else:
            raise AssertionError("Symlink receipt unexpectedly passed")

        tampered = json.loads(json.dumps(verifiable))
        tampered["evidence"]["maximum_actual_dynamic_soft_limit_overshoot_rad"] = 0.0
        try:
            verify_receipt_payload(tampered)
        except ValueError:
            pass
        else:
            raise AssertionError("Tampered receipt unexpectedly verified")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_report", type=Path, nargs="?")
    parser.add_argument("expected_raw_report_sha256", nargs="?")
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--verify-receipt",
        nargs=2,
        metavar=("RECEIPT", "EXPECTED_RECEIPT_SHA256"),
        help="Re-hash a receipt and its parent raw report, then recompute every check",
    )
    args = parser.parse_args()
    if args.self_test:
        if args.verify_receipt is not None:
            parser.error("--self-test cannot be combined with --verify-receipt")
        _self_test()
        print("direct-IK overlay adjudicator self-test: PASS")
        return 0
    if args.verify_receipt is not None:
        if any(
            value is not None
            for value in (
                args.raw_report,
                args.expected_raw_report_sha256,
                args.output,
            )
        ):
            parser.error("--verify-receipt cannot be combined with receipt creation")
        receipt_path = Path(args.verify_receipt[0])
        expected_receipt_sha256 = args.verify_receipt[1]
        _receipt_bytes, receipt = _read_bound_report(
            receipt_path, expected_receipt_sha256, label="Receipt"
        )
        recomputed = verify_receipt_payload(receipt)
        print(
            json.dumps(
                {
                    "receipt": str(receipt_path.expanduser().resolve()),
                    "sha256": expected_receipt_sha256,
                    "status": "PASS",
                    "gate": recomputed["gate"],
                    "checks": recomputed["checks"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if (
        args.raw_report is None
        or args.expected_raw_report_sha256 is None
        or args.output is None
    ):
        parser.error("raw_report, expected_raw_report_sha256, and output are required")
    _payload, report = _read_bound_report(
        args.raw_report, args.expected_raw_report_sha256
    )
    receipt = adjudicate_report(
        report,
        parent_path=args.raw_report,
        parent_sha256=args.expected_raw_report_sha256,
    )
    destination = _write_json_exclusive(args.output, receipt)
    output_sha256 = hashlib.sha256(destination.read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "output": str(destination),
                "sha256": output_sha256,
                "status": receipt["status"],
                "checks": receipt["checks"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if receipt["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
