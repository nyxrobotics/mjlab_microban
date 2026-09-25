# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Write the CPU-only Microban locomotion-prior suitability receipt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mjlab_microban.locomotion_prior_suitability import (
    DEFAULT_LOCOMOTION_PRIOR_PATH,
    DEFAULT_LOCOMOTION_PRIOR_SHA256,
    DEFAULT_ROBOT_XML_PATH,
    DEFAULT_ROBOT_XML_SHA256,
    audit_locomotion_prior_suitability,
    publish_suitability_receipt,
)

DEFAULT_OUTPUT_PATH = Path("artifacts/microban_locomotion_prior_suitability.json")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit exact Microban foot-box FK without simulation, GPU, or robot I/O"
        )
    )
    parser.add_argument(
        "--prior",
        type=Path,
        default=DEFAULT_LOCOMOTION_PRIOR_PATH,
        help="retargeted locomotion-prior NPZ",
    )
    parser.add_argument(
        "--robot-xml",
        type=Path,
        default=DEFAULT_ROBOT_XML_PATH,
        help="Microban MuJoCo XML used for exact foot-box FK",
    )
    parser.add_argument(
        "--expected-prior-sha256",
        default=DEFAULT_LOCOMOTION_PRIOR_SHA256,
        help="required SHA-256 for --prior",
    )
    parser.add_argument(
        "--expected-robot-xml-sha256",
        default=DEFAULT_ROBOT_XML_SHA256,
        help="required SHA-256 for --robot-xml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="deterministic JSON receipt path",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="atomically replace an existing receipt",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        report = audit_locomotion_prior_suitability(
            args.prior,
            args.robot_xml,
            expected_prior_sha256=args.expected_prior_sha256,
            expected_robot_xml_sha256=args.expected_robot_xml_sha256,
        )
        publish_suitability_receipt(args.output, report, force=args.force)
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    checks = report["aggregate"]["checks"]
    summary = {
        "output": str(args.output),
        "receipt_payload_sha256": report["receipt_payload_sha256"],
        "status": report["status"],
        "values": {name: check["value"] for name, check in sorted(checks.items())},
    }
    print(json.dumps(summary, sort_keys=True))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
