"""DRY-RUN ONLY stand-ins of scripts/retrain_all_for_home.py --dry-run.

A dry run trains every policy for a few updates (schedules scaled by
``MICROBAN_SCHEDULE_SCALE``), so its walker cannot pass the PICO source probe
and its PICO checkpoint cannot pass the judgment.  Where a release would stop,
a dry run continues with these stand-ins, so every later step runs on real
files.  Everything they produce is marked: file names start with
``DRYRUN_FORCED_PASS_``, JSON carries ``dry_run_*`` keys with the measured
values, and the PICO package carries ``dry_run_not_deployable`` (the packager
refuses all of this without ``dry_run=True``; the robot refuses such a
package unless MICROBAN_ALLOW_DRYRUN_POLICY=1).

usage (subprocesses of the pipeline):
  python -m mjlab_microban.pipeline.dry force-probe RECEIPT.json OUT_DIR/DRYRUN_FORCED_PASS_<name>.json
  python -m mjlab_microban.pipeline.dry package CHECKPOINT REPORT_PREFIX GATE_OUT ONNX_OUT
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

FORCED_PREFIX = "DRYRUN_FORCED_PASS_"
# The robot's startup self-test needs at least 8 possible-state rows; a
# fallen DRYRUN policy records fewer, so its possible ones are cycled to this
# many.
SMOKE_ROWS = 16
# Fields the v12 source-probe validator requires of a passing receipt
# (microban_teleop_v12_bootstrap.validate_legacy_teleop_probe_receipt).
PROBE_PASS_SUMMARY = {
    "scenario_count": 9,
    "completed_scenario_count": 9,
    "fall_scenario_count": 0,
    "nonfinite_scenario_count": 0,
    "directionally_correct_scenario_count": 8,
    "directional_scenario_count": 8,
    "neutral_target_contract_all_steps": True,
    "raw_action_recurrence_all_steps": True,
}
PROBE_PASS_RESULT = {
    "completed": True,
    "fell": False,
    "nonfinite": None,
    "executed_steps": 300,
    "raw_action_recurrence_verified_steps": 300,
    "neutral_foot_hand_target_verified_steps": 300,
}


def forced_probe_report(report: dict, max_overshoot_rad: float) -> dict:
    """A 9x300 receipt with its pass/fail fields forced (measured ones kept)."""

    out = copy.deepcopy(report)
    results = out.get("results")
    if not isinstance(results, list) or len(results) != 9 or not isinstance(out.get("summary"), dict):
        raise ValueError("force-probe needs a complete nine-scenario probe receipt")
    out["dry_run_original_summary"] = copy.deepcopy(out["summary"])
    out["summary"].update(PROBE_PASS_SUMMARY)
    original_results = []
    for result in results:
        original_results.append({key: result.get(key) for key in (*PROBE_PASS_RESULT,
                                                                "maximum_actual_soft_limit_violation_rad")})
        result.update(PROBE_PASS_RESULT)
        overshoot = result.get("maximum_actual_soft_limit_violation_rad")
        if not isinstance(overshoot, (int, float)) or not 0.0 <= float(overshoot) <= max_overshoot_rad:
            result["maximum_actual_soft_limit_violation_rad"] = 0.0
    out["dry_run_original_results"] = original_results
    out["dry_run_forced_pass_not_deployable"] = True
    return out


def force_probe(source: str, output: str) -> int:
    from mjlab_microban.tasks.microban_teleop_v12_bootstrap import (
        ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD,
    )

    destination = Path(output)
    if not destination.name.startswith(FORCED_PREFIX):
        raise SystemExit(f"force-probe output must be named {FORCED_PREFIX}*")
    report = json.loads(Path(source).read_text(encoding="utf-8"))
    forced = forced_probe_report(report, ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD)
    forced["dry_run_forced_from"] = str(Path(source).resolve())
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(forced, indent=1, sort_keys=True), encoding="utf-8")
    print(destination)
    return 0


def forced_reports(report_prefix: str, out_dir: Path, max_overshoot_rad: float) -> tuple[Path, Path]:
    """DRYRUN_FORCED_PASS_ copies of a judgment's locomotion and tracking reports."""

    forced = []
    for suffix in ("_9x300.json", "_tracking.json"):
        source = Path(report_prefix + suffix)
        report = json.loads(source.read_text(encoding="utf-8"))
        if suffix == "_9x300.json":
            report = forced_probe_report(report, max_overshoot_rad)
        report["dry_run_original_status"] = report.get("status")
        report["status"] = "pass"
        rows = report.get("runtime_smoke_observations")
        if suffix == "_tracking.json" and isinstance(rows, list) and rows:
            from mjlab_microban.policy_contract import physical_row_problem

            possible = [row for row in rows if physical_row_problem("pico", row) is None]
            if not possible:
                raise ValueError("the dry PICO policy recorded no possible-state observation")
            report["dry_run_smoke_rows_original_count"] = len(rows)
            report["runtime_smoke_observations"] = [possible[i % len(possible)] for i in range(SMOKE_ROWS)]
        copy_path = out_dir / f"{FORCED_PREFIX}{source.name}"
        copy_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
        forced.append(copy_path)
    return forced[0], forced[1]


def package(checkpoint: str, report_prefix: str, gate_out: str, onnx_out: str) -> int:
    """The real PICO packager on a dry checkpoint whose judgment failed on performance.

    The locomotion/tracking reports are copied with status forced to "pass"
    (their threshold validation is skipped), the gate carries
    ``dry_run_status_forced_not_deployable`` and the package
    ``dry_run_not_deployable``.  Prints the packager receipt (JSON).
    """

    from mjlab_microban.scripts import export_teleop_v12_deployment as deployment
    from mjlab_microban.scripts import teleop_v12_stage as stage
    from mjlab_microban.tasks.microban_teleop_v12_bootstrap import (
        ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD,
    )

    gate_path = Path(gate_out)
    checkpoint_path = Path(checkpoint)
    stage._validate_locomotion_report = lambda report, identity: None
    stage._validate_tracking_report = lambda report, identity: report.get("profile")
    locomotion, tracking = forced_reports(report_prefix, gate_path.parent,
                                          ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD)

    def build() -> dict:
        gate = stage.create_gate(
            checkpoint=checkpoint_path,
            locomotion_report=locomotion,
            tracking_report=tracking,
            onnx_report=Path(report_prefix + "_onnx.json"),
        )
        gate["dry_run_status_forced_not_deployable"] = True
        return json.loads(json.dumps(gate))

    gate_path.write_text(json.dumps(build(), indent=2, sort_keys=True), encoding="utf-8")

    def validate_gate(path: Path, ckpt: Path) -> dict:
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
        if Path(path).resolve() != gate_path.resolve() or Path(ckpt).resolve() != checkpoint_path.resolve() \
                or loaded != build():
            raise ValueError("dry-run gate drifted")
        return loaded

    deployment.validate_gate = validate_gate
    receipt = deployment.package_v12_deployment(
        checkpoint=checkpoint_path,
        gate_path=gate_path,
        output=Path(onnx_out),
        force=True,
        dry_run=True,
    )
    receipt["dry_run_status_forced_not_deployable"] = True
    print(json.dumps(receipt, sort_keys=True, default=str))
    return 0


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[0] == "force-probe":
        return force_probe(*argv[1:])
    if len(argv) == 5 and argv[0] == "package":
        return package(*argv[1:])
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
