"""DRY-RUN ONLY helpers of scripts/retrain_all_for_home.py --dry-run.

Nothing produced here is deployable; every artifact is labelled DRYRUN.

usage:
  uv run --locked python scripts/home_pipeline/dry_run_tools.py lift-clock \
      SOURCE_MODEL.pt DEST_RUN_DIR ITERATION
  uv run --locked --with onnxruntime --with 'protobuf<7' python \
      scripts/home_pipeline/dry_run_tools.py package \
      CHECKPOINT REPORT_PREFIX GATE_OUT ONNX_OUT MICROBAN_REPO

lift-clock: copy a v12 checkpoint into a new run directory with its update
clock lifted to ITERATION, so a dry run can cross every stage boundary
(3000/3100/7000/7100/10000/10100/15000) with a few updates per segment.

package: run the real v12 deployment packager (ONNX export, metadata, parity,
robot runtime validator on MICROBAN_REPO, identity hashes of the robot tree)
on a dry-run checkpoint whose stage reports fail on performance.  The
locomotion/tracking reports are copied next to GATE_OUT with status forced
to "pass" (marked ``dry_run_original_status``), their status/threshold
validation is skipped, a short runtime smoke corpus (a DRYRUN checkpoint
that fell records fewer than 16 rows) is cycled to 16 rows, and the gate
carries ``dry_run_status_forced_not_deployable``.  Prints the packager receipt (JSON).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


# Runtime smoke rows of a complete final tracking report (the packager's
# ONNX Runtime smoke on the robot requires exactly this many samples).
SMOKE_ROWS = 16


def lift_clock(source: str, destination_dir: str, iteration: str) -> int:
    import torch

    from mjlab_microban.tasks.microban_teleop_v12_actor import (
        teleop_v12_active_adapter_columns,
    )

    payload = torch.load(source, map_location="cpu", weights_only=False)
    target = int(iteration)
    step = (target + 1) * 24
    old_active = payload["infos"]["active_actor_columns_at_save"]
    new_active = list(teleop_v12_active_adapter_columns(step))
    if not set(old_active) <= set(new_active):
        raise SystemExit("cannot lift backwards across an adapter schedule boundary")
    payload["iter"] = target
    payload["infos"]["env_state"]["common_step_counter"] = step
    payload["infos"]["active_actor_columns_at_save"] = new_active
    payload["infos"]["dry_run_clock_lift_from"] = str(source)
    out_dir = Path(destination_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    out = out_dir / f"model_{target}.pt"
    torch.save(payload, out)
    print(out)
    return 0


def package(checkpoint: str, report_prefix: str, gate_out: str, onnx_out: str, repo: str) -> int:
    from mjlab_microban.scripts import export_teleop_v12_deployment as deployment
    from mjlab_microban.scripts import teleop_v12_stage as stage

    checkpoint_path = Path(checkpoint)
    gate_path = Path(gate_out)
    onnx_path = Path(onnx_out)
    forced = []
    for suffix in ("_9x300.json", "_tracking.json"):
        source = Path(report_prefix + suffix)
        report = json.loads(source.read_text(encoding="utf-8"))
        report["dry_run_original_status"] = report.get("status")
        report["status"] = "pass"
        rows = report.get("runtime_smoke_observations")
        if suffix == "_tracking.json" and isinstance(rows, list) and rows and len(rows) != SMOKE_ROWS:
            # A passing final gate completes every scenario and records exactly
            # 16 rows, which the packager's robot smoke check requires; a
            # DRYRUN checkpoint can fall early and record fewer, so cycle them.
            report["dry_run_smoke_rows_original_count"] = len(rows)
            report["runtime_smoke_observations"] = [rows[i % len(rows)] for i in range(SMOKE_ROWS)]
        copy = gate_path.parent / f"DRYRUN_FORCED_PASS_{source.name}"
        copy.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
        forced.append(copy)
    locomotion, tracking = forced
    onnx_report = Path(report_prefix + "_onnx.json")
    stage._validate_locomotion_report = lambda report, identity: None
    # The real validator returns the profile the report was judged under.
    stage._validate_tracking_report = lambda report, identity, **_kwargs: report.get("profile")

    def build() -> dict:
        gate = stage.create_gate(
            checkpoint=checkpoint_path,
            locomotion_report=locomotion,
            tracking_report=tracking,
            onnx_report=onnx_report,
        )
        gate["dry_run_status_forced_not_deployable"] = True
        return json.loads(json.dumps(gate))

    gate_path.write_text(json.dumps(build(), indent=2, sort_keys=True), encoding="utf-8")

    def validate_gate(path: Path, ckpt: Path) -> dict:
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
        if Path(ckpt).resolve() != checkpoint_path.resolve() or loaded != build():
            raise ValueError("dry-run gate drifted")
        return loaded

    deployment.validate_gate = validate_gate
    receipt = deployment.package_v12_deployment(
        checkpoint=checkpoint_path,
        gate_path=gate_path,
        output=onnx_path,
        microban_repo=Path(repo),
        force=True,
    )
    receipt["dry_run_status_forced_not_deployable"] = True
    print(json.dumps(receipt, sort_keys=True, default=str))
    return 0


def main(argv: list[str]) -> int:
    if len(argv) >= 1 and argv[0] == "lift-clock" and len(argv) == 4:
        return lift_clock(*argv[1:])
    if len(argv) >= 1 and argv[0] == "package" and len(argv) == 6:
        return package(*argv[1:])
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
