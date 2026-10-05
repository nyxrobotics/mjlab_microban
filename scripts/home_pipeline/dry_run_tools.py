"""DRY-RUN ONLY helpers of scripts/retrain_all_for_home.py --dry-run.

Nothing produced here is deployable; every artifact is labelled DRYRUN.

usage:
  uv run --locked python scripts/home_pipeline/dry_run_tools.py lift-clock \
      SOURCE_MODEL.pt DEST_RUN_DIR ITERATION
  uv run --locked --with onnxruntime --with 'protobuf<7' python \
      scripts/home_pipeline/dry_run_tools.py package \
      CHECKPOINT REPORT_PREFIX GATE_OUT ONNX_OUT MICROBAN_REPO \
      [BOUNDARY_CHECKPOINT BOUNDARY_REPORT_PREFIX]...
  uv run --locked python scripts/home_pipeline/dry_run_tools.py force-probe \
      PROBE_RECEIPT.json OUTPUT_DIR/DRYRUN_FORCED_PASS_<name>.json
  uv run --locked python scripts/home_pipeline/dry_run_tools.py \
      stamp-corner-rescue PARENT_MODEL_9900.pt PARENT_REPORT.json \
      SOURCE_MODEL_9999.pt DEST_RUN_DIR MIX

lift-clock: copy a v12 checkpoint into a new run directory with its update
clock lifted to ITERATION, so a dry run can cross every stage boundary
(3000/3100/7000/7100/10000/10100/15000) with a few updates per segment.  A
source in a sibling run directory is recorded as the copy's resume parent in
params/agent.yaml (load_run/load_checkpoint), as a resumed run records it.

package: run the real v12 deployment packager (ONNX export, metadata, parity,
robot runtime validator on MICROBAN_REPO, identity hashes of the robot tree)
on a dry-run checkpoint whose stage reports fail on performance.  The
locomotion/tracking reports are copied next to GATE_OUT with status forced
to "pass" (marked ``dry_run_original_status``), their status/threshold
validation is skipped, the package carries ``dry_run_not_deployable`` (the
robot runtime refuses it unless MICROBAN_ALLOW_DRYRUN_POLICY=1, which only
the dry run sets for its own checks), a short runtime smoke corpus (a DRYRUN checkpoint
that fell records fewer than 16 rows) is cycled to 16 rows, and the gate
carries ``dry_run_status_forced_not_deployable``.  The locomotion summary
and per-scenario pass fields the robot validator re-checks in the package
metadata (falls, completion, directional count, soft-limit overshoot) are
forced like force-probe (measured values kept in ``dry_run_original_*``), so
a from-scratch ``--dry-run-plumbing`` policy that falls still reaches the
robot validator.  Each BOUNDARY_CHECKPOINT/BOUNDARY_REPORT_PREFIX pair (the
model_9999 the chain passed the 10000 boundary with, e.g. a stamped corner
rescue) gets the same forced gate (DRYRUN_boundary_gate_model_<N>.json next
to GATE_OUT) and is passed to the packager as --boundary-gate, so its real
boundary lineage check (shared and inherited lineage markers, e.g. the
pose-release corner-rescue marker the final checkpoint must carry) runs.
Prints the packager receipt (JSON).

force-probe (``--dry-run-plumbing`` only): copy a failing 9x300 v12 source
probe receipt of a from-scratch dry-run walker with its pass/fail fields
forced to the source-gate values (``dry_run_original_*`` keep the measured
ones, ``dry_run_forced_pass_not_deployable`` marks it), so the v12 start
(train_microban_teleop_v12.sh start --dry-run-probe-receipt) can bootstrap a
plumbing chain from a walker that cannot walk yet.

stamp-corner-rescue: the dry-run stand-in for the pose-release model_9900
corner rescue (whose real launcher needs 2048 envs and a canonical parent):
copy the dry chain's model_9999 into DEST_RUN_DIR with the exact pose-release
corner-rescue lineage marker of MIX bound to the parent's and its strict
report's SHA-256 (``dry_run_synthetic_corner_rescue`` marks it).  Every
downstream consumer (ordinary pose-release resume, stage gates, packager)
then validates a real corner-rescue lineage.
"""

from __future__ import annotations

import json
import os
import re
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
    source_path = Path(source).resolve()
    match = re.fullmatch(r"model_([0-9]+)\.pt", source_path.name)
    if match and source_path.parent.parent == out_dir.resolve().parent:
        # Record the lift's parent the way a resumed run does, so the
        # packager's resume-ancestry walk (export_teleop_v12_deployment
        # _resume_ancestry) follows a dry chain through its lifted copies.
        (out_dir / "params").mkdir()
        (out_dir / "params" / "agent.yaml").write_text(
            "resume: true\n"
            f"load_run: ^{source_path.parent.name}$\n"
            f"load_checkpoint: ^model_{match[1]}[.]pt$\n"
            "dry_run_clock_lift: true\n",
            encoding="utf-8",
        )
    print(out)
    return 0


def forced_stage_reports(report_prefix: str, out_dir: Path, max_overshoot_rad: float) -> tuple[Path, Path]:
    """DRYRUN_FORCED_PASS_ copies of a stage gate's locomotion and tracking reports."""

    forced = []
    for suffix in ("_9x300.json", "_tracking.json"):
        source = Path(report_prefix + suffix)
        report = json.loads(source.read_text(encoding="utf-8"))
        if suffix == "_9x300.json":
            report = forced_probe_report(report, max_overshoot_rad)
        report["dry_run_original_status"] = report.get("status")
        report["status"] = "pass"
        rows = report.get("runtime_smoke_observations")
        if suffix == "_tracking.json" and isinstance(rows, list) and rows and len(rows) != SMOKE_ROWS:
            # A passing final gate completes every scenario and records exactly
            # 16 rows, which the packager's robot smoke check requires; a
            # DRYRUN checkpoint can fall early and record fewer, so cycle them.
            report["dry_run_smoke_rows_original_count"] = len(rows)
            report["runtime_smoke_observations"] = [rows[i % len(rows)] for i in range(SMOKE_ROWS)]
        copy = out_dir / f"DRYRUN_FORCED_PASS_{source.name}"
        copy.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
        forced.append(copy)
    return forced[0], forced[1]


def package(checkpoint: str, report_prefix: str, gate_out: str, onnx_out: str, repo: str,
            *boundaries: str) -> int:
    from mjlab_microban.scripts import export_teleop_v12_deployment as deployment
    from mjlab_microban.scripts import teleop_v12_stage as stage
    from mjlab_microban.tasks.microban_teleop_v12_bootstrap import (
        ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD,
    )

    if len(boundaries) % 2:
        raise SystemExit("package: boundary arguments come in CHECKPOINT REPORT_PREFIX pairs")
    gate_path = Path(gate_out)
    onnx_path = Path(onnx_out)
    stage._validate_locomotion_report = lambda report, identity: None
    # The real validator returns the profile the report was judged under.
    stage._validate_tracking_report = lambda report, identity, **_kwargs: report.get("profile")

    def builder(ckpt: Path, prefix: str):
        locomotion, tracking = forced_stage_reports(prefix, gate_path.parent,
                                                    ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD)

        def build() -> dict:
            gate = stage.create_gate(
                checkpoint=ckpt,
                locomotion_report=locomotion,
                tracking_report=tracking,
                onnx_report=Path(prefix + "_onnx.json"),
            )
            gate["dry_run_status_forced_not_deployable"] = True
            return json.loads(json.dumps(gate))

        return build

    # gate path -> (checkpoint, builder); the packager's validate_gate accepts exactly these.
    gates: dict[Path, tuple[Path, object]] = {}
    checkpoint_path = Path(checkpoint)
    gates[gate_path.resolve()] = (checkpoint_path, builder(checkpoint_path, report_prefix))
    boundary_paths = []
    for ckpt, prefix in zip(boundaries[0::2], boundaries[1::2]):
        ckpt_path = Path(ckpt)
        out = gate_path.parent / f"DRYRUN_boundary_gate_{ckpt_path.parent.name}_{ckpt_path.stem}.json"
        gates[out.resolve()] = (ckpt_path, builder(ckpt_path, prefix))
        boundary_paths.append(out)
    for path, (_ckpt, build) in gates.items():
        path.write_text(json.dumps(build(), indent=2, sort_keys=True), encoding="utf-8")

    def validate_gate(path: Path, ckpt: Path) -> dict:
        entry = gates.get(Path(path).resolve())
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
        if entry is None or Path(ckpt).resolve() != entry[0].resolve() or loaded != entry[1]():
            raise ValueError("dry-run gate drifted")
        return loaded

    deployment.validate_gate = validate_gate
    # The package is marked dry_run_not_deployable; the robot runtime refuses
    # it unless this variable is "1" (only for the dry run's own validator).
    os.environ[deployment.DRY_RUN_POLICY_ALLOW_ENV] = "1"
    receipt = deployment.package_v12_deployment(
        checkpoint=checkpoint_path,
        gate_path=gate_path,
        output=onnx_path,
        microban_repo=Path(repo),
        force=True,
        boundary_gates=tuple(boundary_paths),
        dry_run=True,
    )
    receipt["dry_run_status_forced_not_deployable"] = True
    receipt["dry_run_boundary_gates"] = [str(p) for p in boundary_paths]
    print(json.dumps(receipt, sort_keys=True, default=str))
    return 0


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
FORCED_PREFIX = "DRYRUN_FORCED_PASS_"


def forced_probe_report(report: dict, max_overshoot_rad: float) -> dict:
    """The receipt with its pass/fail fields forced (pure; see force-probe)."""

    import copy

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
    destination.write_text(json.dumps(forced, indent=1, sort_keys=True), encoding="utf-8")
    print(destination)
    return 0


def corner_rescue_infos(infos: dict, *, parent_sha256: str, report_sha256: str, mix: str,
                        provenance: dict) -> dict:
    """Infos of a dry model_9999 stamped with the pose-release corner marker (pure)."""

    from mjlab_microban.tasks.microban_teleop_v12_corner_rescue import (
        MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY,
        corner_rescue_marker,
    )
    from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release_lineage import (
        HAND_POSE_RELEASE_LINEAGE_FRESH,
        HAND_POSE_RELEASE_LINEAGE_FRESH_CORNER_RESCUE,
        hand_pose_release_lineage,
    )

    if hand_pose_release_lineage(infos, iteration=9999, verify_parent=False) != (
            HAND_POSE_RELEASE_LINEAGE_FRESH):
        raise ValueError("stamp-corner-rescue needs a fresh pose-release chain's model_9999")
    out = dict(infos)
    out[MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY] = corner_rescue_marker(
        parent_checkpoint_sha256=parent_sha256,
        parent_strict_tracking_report_sha256=report_sha256,
        hand_pose_release=True,
        parent_strict_failed_checks=("hand_tracking_rms",),
        pose_release_mix=mix,
    )
    out["dry_run_synthetic_corner_rescue"] = {**provenance, "mix": mix, "not_deployable": True}
    if hand_pose_release_lineage(out, iteration=9999, verify_parent=False) != (
            HAND_POSE_RELEASE_LINEAGE_FRESH_CORNER_RESCUE):
        raise AssertionError("stamped lineage is not the pose-release corner rescue")
    return out


def stamp_corner_rescue(parent: str, report: str, source: str, destination_dir: str, mix: str) -> int:
    import hashlib

    import torch

    def digest(path: str) -> str:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    payload = torch.load(source, map_location="cpu", weights_only=False)
    if payload.get("iter") != 9999 or Path(parent).name != "model_9900.pt":
        raise SystemExit("stamp-corner-rescue needs a model_9900 parent and a model_9999 source")
    payload["infos"] = corner_rescue_infos(
        payload["infos"], parent_sha256=digest(parent), report_sha256=digest(report), mix=mix,
        provenance={"parent": str(Path(parent).resolve()), "parent_report": str(Path(report).resolve()),
                    "source": str(Path(source).resolve())})
    out_dir = Path(destination_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    out = out_dir / "model_9999.pt"
    torch.save(payload, out)
    print(out)
    return 0


def main(argv: list[str]) -> int:
    if len(argv) >= 1 and argv[0] == "lift-clock" and len(argv) == 4:
        return lift_clock(*argv[1:])
    if len(argv) >= 1 and argv[0] == "package" and len(argv) >= 6 and len(argv) % 2 == 0:
        return package(*argv[1:])
    if len(argv) >= 1 and argv[0] == "force-probe" and len(argv) == 3:
        return force_probe(*argv[1:])
    if len(argv) >= 1 and argv[0] == "stamp-corner-rescue" and len(argv) == 6:
        return stamp_corner_rescue(*argv[1:])
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
