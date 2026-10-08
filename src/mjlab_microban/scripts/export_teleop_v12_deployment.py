"""Package the accepted checkpoint a PICO run ended with for Microban hardware.

The gate (teleop_v12_stage) proves the checkpoint and its evaluation
evidence; this command turns that evidence into the robot's contract
microban-policy-1 (policy_contract.py, docs/policies.md: the HOME stamp, the
layout, the PICO foot and arm targets and curriculum, the raw-action guard and the startup
self-test: the final tracking rollouts' actor observations with the actor's
deterministic output for each), checks the final metadata-bearing graph with
both ONNX implementations and the robot's self-test rule and only then
atomically publishes the file.  The robot side is checked when the release is
installed (its tools/validate_policies.py and its tests): the package does not
depend on the robot's sources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import onnx
import torch
from mjlab.rl.exporter_utils import attach_metadata_to_onnx
from onnx.reference import ReferenceEvaluator

from mjlab_microban.tasks.microban_policy_export import MICROBAN_TELEOP_OBSERVATION_WIDTH
from mjlab_microban.robot import home_contracts
from mjlab_microban.robot.microban_constants import HOME_FRAME
from mjlab_microban.scripts.evaluate_teleop_v12_checkpoint import _load_actor
from mjlab_microban.scripts.evaluate_teleop_v12_tracking import FINAL_PROFILE
from mjlab_microban.scripts.teleop_v12_bootstrap_gate import (
    _export_onnx_atomic,
)
from mjlab_microban.scripts.teleop_v12_stage import validate_gate
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_HMD_JOINT_NAMES,
    MICROBAN_TELEOP_ACTION_JOINT_NAMES,
    MICROBAN_TELEOP_OBSERVATION_SCHEMA,
    MICROBAN_TELEOP_TARGET_FRAME,
)
from mjlab_microban.tasks.microban_teleop_v12_actor import (
    TELEOP_V12_EXTRA_OBSERVATION_COLUMNS,
)
from mjlab_microban.tasks.microban_teleop_v12_bootstrap import (
    TeleopV12BootstrapProvenance,
    resolve_bootstrap_artifact_path,
    sha256_file,
    validate_bootstrap_provenance,
)
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MICROBAN_TELEOP_V12_ACTION_CLIP,
    MICROBAN_TELEOP_V13_ARM_OVERLAY_RECIPE_REVISION,
)
from mjlab_microban.tasks.microban_teleop_v12_home_pose import (
    TELEOP_V12_HOME_POSE_INFO_KEY,
    validate_teleop_v12_home_pose,
)
from mjlab_microban import policy_contract
from mjlab_microban.tasks.microban_teleop_v12_runner import (
    TELEOP_V12_BOOTSTRAP_INFO_KEY,
    require_bilateral_site_order,
)
from mjlab_microban.schedules import (
    PICO_TOTAL_UPDATES,
    pico_schedule_record,
)

# HOME-bound (robot/home_contracts.py): recorded in the packaging receipt.
PACKAGER_REVISION = home_contracts.V12_PACKAGER_REVISION
# Raw-action guard: max(v12_absmax, source_absmax + delta_absmax) * 6 over the
# final tracking rollouts; the robot stops when an output exceeds it.
RUNTIME_GUARD_MULTIPLIER = 6.0


def _load_json(path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"JSON duplicates key {key!r}: {path}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"JSON contains non-finite value {value!r}: {path}")

    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"JSON SHA-256 mismatch for {path}: {digest}")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"JSON is not UTF-8: {path}") from exc
    value = json.loads(
        text,
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _self_test_rows(tracking: Mapping[str, Any]) -> list[list[float]]:
    """Real actor observations of the final tracking rollouts (possible states only)."""

    rows = tracking.get("runtime_smoke_observations")
    if not isinstance(rows, list) or any(
        not isinstance(row, list)
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in row)
        for row in rows
    ):
        raise ValueError("Final tracking report lacks its actor observations")
    try:
        return policy_contract.select_self_test_rows("pico", rows)
    except policy_contract.PolicyContractError as exc:
        raise ValueError(str(exc)) from exc


def _require_final_gate(
    gate: Mapping[str, Any],
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
) -> None:
    from mjlab_microban.scripts.teleop_v12_stage import GATE_SCHEMA_VERSION

    expected = {
        "schema_version": GATE_SCHEMA_VERSION,
        "gate": "microban_teleop_v12_stage",
        "status": "pass",
        "checkpoint_sha256": checkpoint_sha256,
    }
    mismatches = [name for name, value in expected.items() if gate.get(name) != value]
    if gate.get("tracking_profile") != FINAL_PROFILE:
        mismatches.append("tracking_profile")
    completed = gate.get("completed_updates")
    if (
        not isinstance(completed, int)
        or isinstance(completed, bool)
        or completed != PICO_TOTAL_UPDATES
        or gate.get("iteration") != completed - 1
    ):
        mismatches.append("completed_updates")
    if mismatches:
        raise ValueError(
            "A PICO package requires the passing gate of the checkpoint its run "
            "ended with; mismatched fields: " + ", ".join(mismatches)
        )
    if checkpoint.name != f"model_{gate['iteration']}.pt":
        raise ValueError("The packaged checkpoint must be the gate's model_<iteration>.pt")


def _deployment_recipe_revision(infos: Mapping[str, Any]) -> str:
    """Recipe string the package declares to the robot (the arm-overlay recipe)."""

    if infos.get("microban_teleop_recipe_revision") != (
        MICROBAN_TELEOP_V13_ARM_OVERLAY_RECIPE_REVISION
    ):
        raise ValueError("Only an arm-overlay checkpoint is packaged")
    return MICROBAN_TELEOP_V13_ARM_OVERLAY_RECIPE_REVISION


def _read_onnx_metadata(path: Path) -> dict[str, str]:
    model = onnx.load(path)
    result: dict[str, str] = {}
    for item in model.metadata_props:
        if item.key in result:
            raise ValueError(f"Duplicate ONNX metadata key: {item.key}")
        result[item.key] = item.value
    return result


def _checked_envelope_vector(
    envelope: Mapping[str, Any], policy: str, statistic: str
) -> list[float]:
    policy_value = envelope.get(policy)
    if not isinstance(policy_value, Mapping):
        raise TypeError(f"Tracking envelope {policy!r} is missing")
    value = policy_value.get(statistic)
    if (
        not isinstance(value, list)
        or len(value) != len(MICROBAN_TELEOP_ACTION_JOINT_NAMES)
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in value
        )
    ):
        raise ValueError(f"Tracking envelope {policy}/{statistic} is malformed")
    return [float(item) for item in value]


def _runtime_guard(envelope: Mapping[str, Any]) -> list[float]:
    learned = _checked_envelope_vector(envelope, "v12", "absolute_maximum")
    source = _checked_envelope_vector(envelope, "legacy_source", "absolute_maximum")
    delta = _checked_envelope_vector(
        envelope, "learned_minus_source", "absolute_maximum"
    )
    result = [
        max(v12, legacy + difference) * RUNTIME_GUARD_MULTIPLIER
        for v12, legacy, difference in zip(learned, source, delta, strict=True)
    ]
    with np.errstate(over="ignore", invalid="ignore"):
        as_float32 = np.asarray(result, dtype=np.float32)
    if not np.isfinite(as_float32).all() or bool(np.any(as_float32 < 0.0)):
        raise ValueError("Tracking evidence produced an invalid float32 runtime guard")
    return result


def build_v12_deployment_metadata(
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
    gate_path: Path,
    gate: Mapping[str, Any],
    infos: Mapping[str, Any],
    bootstrap: TeleopV12BootstrapProvenance,
    tracking: Mapping[str, Any],
    self_test_observations: list[list[float]],
    self_test_actions: list[list[float]],
    dry_run: bool,
) -> dict[str, str]:
    """Translate the already-validated gate evidence to microban-policy-1."""

    _require_final_gate(
        gate,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
    )
    _deployment_recipe_revision(infos)
    home_pose = validate_teleop_v12_home_pose(infos)
    if gate.get(TELEOP_V12_HOME_POSE_INFO_KEY) != home_pose:
        raise ValueError("Final v12 gate HOME pose does not match its checkpoint")
    if infos.get("trainable_actor_parameters") != ["mlp.0.weight"] or infos.get(
        "trainable_actor_columns"
    ) != list(TELEOP_V12_EXTRA_OBSERVATION_COLUMNS):
        raise ValueError("Final checkpoint trainable adapter declaration drifted")
    if infos.get("active_actor_columns_at_save") != list(
        TELEOP_V12_EXTRA_OBSERVATION_COLUMNS
    ):
        raise ValueError("Final checkpoint did not activate all v12 adapter columns")
    observation_joints = [*MICROBAN_HMD_JOINT_NAMES, *MICROBAN_TELEOP_ACTION_JOINT_NAMES]
    if (
        tuple(observation_joints) != policy_contract.OBSERVATION_JOINT_NAMES["pico"]
        or tuple(MICROBAN_TELEOP_OBSERVATION_SCHEMA) != policy_contract.OBSERVATION_SCHEMAS["pico"]
        or home_pose["joint_names"] != observation_joints
        or home_pose["joint_pos_rad"] != [float(HOME_FRAME.joint_pos[name]) for name in observation_joints]
    ):
        raise ValueError("Final v12 checkpoint HOME pose or layout differs from the contract")
    if list(MICROBAN_TELEOP_V12_ACTION_CLIP) != [-policy_contract.SERVO_TARGET_RANGE_RAD,
                                                  policy_contract.SERVO_TARGET_RANGE_RAD]:
        raise ValueError("PICO action clip is not the servo goal range (+-pi)")
    if bootstrap.previous_action_semantics != "raw_actor_output":
        raise ValueError("PICO does not feed back its raw output")
    tracking_envelope = tracking.get("raw_action_envelope")
    if not isinstance(tracking_envelope, Mapping):
        raise TypeError("Validated v12 tracking evidence lacks its raw-action envelope")
    require_bilateral_site_order(infos)
    try:
        metadata = policy_contract.contract_metadata(
            "pico",
            joint_names=observation_joints,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            gate_report_sha256=sha256_file(gate_path),
            self_test_observations=self_test_observations,
            self_test_actions=self_test_actions,
            dry_run=dry_run,
        )
        metadata.update(policy_contract.pico_metadata(
            walk_checkpoint_sha256=bootstrap.source.sha256,
            target_frame=MICROBAN_TELEOP_TARGET_FRAME,
            raw_action_guard=_runtime_guard(tracking_envelope),
            curriculum=pico_schedule_record(),
            active_adapter_columns=infos["active_actor_columns_at_save"],
            checkpoint_iteration=int(gate["iteration"]),
        ))
    except policy_contract.PolicyContractError as exc:
        raise ValueError(str(exc)) from exc
    metadata["run_path"] = checkpoint.parent.name
    return metadata


def _actor_outputs(actor: torch.nn.Module, rows: list[list[float]]) -> list[list[float]]:
    """The deployed (deterministic, normalized) actor's output for each row."""

    export_model = actor.as_onnx(verbose=False).cpu().eval()
    with torch.inference_mode():
        batch = torch.tensor(rows, dtype=torch.float32)
        return [export_model(row.unsqueeze(0))[0].double().tolist() for row in batch]


def _validate_graph_contract(path: Path) -> None:
    model = onnx.load(path)
    onnx.checker.check_model(model, full_check=True)
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("Contract-v12 deployment ONNX must have one input/output")
    input_value = model.graph.input[0]
    output_value = model.graph.output[0]
    if input_value.name != "obs" or output_value.name != "actions":
        raise ValueError("Contract-v12 deployment tensors must be obs -> actions")
    for value, expected in ((input_value, [1, MICROBAN_TELEOP_OBSERVATION_WIDTH]), (output_value, [1, 18])):
        tensor = value.type.tensor_type
        shape = [dimension.dim_value for dimension in tensor.shape.dim]
        if tensor.elem_type != onnx.TensorProto.FLOAT or shape != expected:
            raise ValueError(
                f"Unsafe ONNX tensor {value.name}: type={tensor.elem_type}, "
                f"shape={shape}, expected float32 {expected}"
            )


def _validate_final_parity(
    actor: torch.nn.Module, path: Path, *, tolerance: float
) -> dict[str, float]:
    import onnxruntime as ort

    generator = torch.Generator().manual_seed(20260925)
    # Advance the generator exactly as the hash-bound ONNX report did before
    # drawing its full-width corpus.
    torch.randn(10_000, MICROBAN_TELEOP_OBSERVATION_WIDTH, generator=generator)
    observations = torch.randn(64, MICROBAN_TELEOP_OBSERVATION_WIDTH, generator=generator)
    if not bool(
        torch.all(
            observations[:, TELEOP_V12_EXTRA_OBSERVATION_COLUMNS].abs().amax(dim=0)
            > 0.0
        ).item()
    ):
        raise AssertionError("Deployment parity corpus missed a teleop-only column")

    model = onnx.load(path)
    reference = ReferenceEvaluator(model)
    runtime = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    if runtime.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("Deployment parity did not use CPUExecutionProvider only")
    export_model = actor.as_onnx(verbose=False).cpu().eval()
    from mjlab_microban.scripts.teleop_v12_onnx_gate import parity_bound_ratio

    reference_max = 0.0
    runtime_max = 0.0
    bound_ratio = 0.0
    with torch.inference_mode():
        for observation in observations:
            batch = observation.unsqueeze(0)
            expected = export_model(batch).detach().cpu().numpy()
            (reference_actual,) = reference.run(None, {"obs": batch.numpy()})
            (runtime_actual,) = runtime.run(None, {"obs": batch.numpy()})
            reference_error = float(np.max(np.abs(reference_actual - expected)))
            runtime_error = float(np.max(np.abs(runtime_actual - expected)))
            reference_max = max(reference_max, reference_error)
            runtime_max = max(runtime_max, runtime_error)
            bound_ratio = max(
                bound_ratio,
                parity_bound_ratio(reference_actual, expected, atol=tolerance),
                parity_bound_ratio(runtime_actual, expected, atol=tolerance),
            )
    # Same per-sample atol + rtol*max|expected| rule as teleop_v12_onnx_gate.
    if (
        not math.isfinite(reference_max)
        or not math.isfinite(runtime_max)
        or not math.isfinite(bound_ratio)
        or bound_ratio > 1.0
    ):
        raise ValueError(
            "Final metadata-bearing ONNX parity failed: "
            f"reference={reference_max}, onnxruntime_cpu={runtime_max}, "
            f"tolerance={tolerance}"
        )
    return {
        "reference_maximum_absolute_error": reference_max,
        "onnxruntime_cpu_maximum_absolute_error": runtime_max,
    }


def _capture_checkpoint(checkpoint: Path, destination: Path) -> str:
    digest = hashlib.sha256()
    with checkpoint.open("rb") as source, destination.open("xb") as target:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
            target.write(chunk)
        target.flush()
        os.fsync(target.fileno())
    return digest.hexdigest()


def _reject_protected_output(output: Path, protected: Mapping[str, Path]) -> None:
    conflicts = [name for name, path in protected.items() if output == path.resolve()]
    if conflicts:
        raise ValueError(
            "Deployment output would overwrite immutable input/source: "
            + ", ".join(conflicts)
        )


# Dry runs of scripts/retrain_all_for_home.py (pipeline/dry.py) pass
# dry_run=True: their package carries dry_run_not_deployable,
# which the robot refuses unless MICROBAN_ALLOW_DRYRUN_POLICY is "1" (set only
# by the dry run's own validator and test calls).  Without dry_run the packager
# refuses any dry-run evidence: a gate or checkpoint info key starting with
# "dry_run", or a forced-pass (DRYRUN_*) source probe receipt.


def _dry_run_evidence(
    *, gate: Mapping[str, Any], infos: Mapping[str, Any], probe_path: str
) -> list[str]:
    found = [f"gate {key}" for key in gate if str(key).startswith("dry_run")]
    found += [f"checkpoint info {key}" for key in infos if str(key).startswith("dry_run")]
    probe = resolve_bootstrap_artifact_path(probe_path)
    if probe.name.startswith("DRYRUN_"):
        found.append(f"source probe receipt {probe.name}")
    else:
        try:
            receipt = json.loads(probe.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            receipt = {}
        if isinstance(receipt, Mapping):
            found += [
                f"source probe receipt {key}"
                for key in receipt
                if str(key).startswith("dry_run")
            ]
    return found


def package_v12_deployment(
    *,
    checkpoint: Path,
    gate_path: Path,
    output: Path,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve()
    gate_path = gate_path.expanduser().resolve()
    output = output.expanduser().absolute()
    output = output.parent.resolve() / output.name
    if not checkpoint.is_file() or not gate_path.is_file():
        raise FileNotFoundError("Checkpoint and v12 stage gate must both exist")
    if output.exists() and not force:
        raise FileExistsError(f"Deployment output exists (pass --force): {output}")

    gate_snapshot = _load_json(gate_path)
    gate_sha256 = sha256_file(gate_path)
    gate = validate_gate(gate_path, checkpoint)
    if gate != gate_snapshot or sha256_file(gate_path) != gate_sha256:
        raise RuntimeError("V12 stage gate changed while it was validated")
    checkpoint_sha256 = sha256_file(checkpoint)
    _require_final_gate(
        gate, checkpoint=checkpoint, checkpoint_sha256=checkpoint_sha256
    )
    reports = gate.get("reports")
    if not isinstance(reports, Mapping):
        raise TypeError("V12 stage gate report references are malformed")
    report_paths = {
        name: resolve_bootstrap_artifact_path(reports[name])
        for name in ("locomotion", "tracking", "onnx")
    }
    report_hashes = gate.get("report_sha256")
    if not isinstance(report_hashes, Mapping):
        raise TypeError("V12 stage gate report hashes are malformed")
    for name in ("locomotion", "tracking", "onnx"):
        expected = report_hashes.get(name)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"V12 stage gate {name} report SHA-256 is malformed")
    for name in ("locomotion", "onnx"):
        _load_json(report_paths[name], expected_sha256=str(report_hashes[name]))
    tracking = _load_json(
        report_paths["tracking"],
        expected_sha256=str(report_hashes["tracking"]),
    )
    onnx_report = _load_json(
        report_paths["onnx"], expected_sha256=str(report_hashes["onnx"])
    )
    onnx_evidence = onnx_report.get("onnx")
    if not isinstance(onnx_evidence, Mapping):
        raise TypeError("V12 ONNX parity evidence is malformed")
    parity_tolerance = float(onnx_evidence["tolerance"])
    _reject_protected_output(
        output,
        {
            "checkpoint": checkpoint,
            "stage_gate": gate_path,
            **{f"{name}_report": path for name, path in report_paths.items()},
            "stage_gate_onnx": resolve_bootstrap_artifact_path(gate["onnx"]["path"]),
        },
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    captured = output.with_name(f".{checkpoint.name}.{uuid4().hex}.captured")
    try:
        captured_sha256 = _capture_checkpoint(checkpoint, captured)
        if captured_sha256 != checkpoint_sha256:
            raise RuntimeError("Checkpoint changed while its bytes were captured")
        actor, iteration, infos = _load_actor(captured, device="cpu")
        if iteration != gate["iteration"]:
            raise ValueError("Captured checkpoint is not the gate's checkpoint")
        bootstrap = validate_bootstrap_provenance(
            infos.get(TELEOP_V12_BOOTSTRAP_INFO_KEY), verify_files=True
        )
        require_bilateral_site_order(infos)
        if not dry_run:
            evidence = _dry_run_evidence(
                gate=gate, infos=infos, probe_path=bootstrap.probe.path
            )
            if evidence:
                raise ValueError(
                    "Refusing to package dry-run evidence as deployable: "
                    + ", ".join(evidence)
                )

        _export_onnx_atomic(actor, temporary)
        _validate_graph_contract(temporary)
        _validate_final_parity(actor, temporary, tolerance=parity_tolerance)
        rows = _self_test_rows(tracking)
        metadata = build_v12_deployment_metadata(
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            gate_path=gate_path,
            gate=gate,
            infos=infos,
            bootstrap=bootstrap,
            tracking=tracking,
            self_test_observations=rows,
            self_test_actions=_actor_outputs(actor, rows),
            dry_run=dry_run,
        )
        existing = _read_onnx_metadata(temporary)
        overlap = set(existing).intersection(metadata)
        if overlap:
            raise ValueError(
                "Exported ONNX already claims deployment metadata: "
                + ", ".join(sorted(overlap))
            )
        attach_metadata_to_onnx(str(temporary), metadata)
        attached = _read_onnx_metadata(temporary)
        for key, expected in metadata.items():
            if attached.get(key) != expected:
                raise ValueError(
                    f"Attached ONNX metadata mismatch for {key}: "
                    f"{attached.get(key)!r} != {expected!r}"
                )
        _validate_graph_contract(temporary)
        final_parity = _validate_final_parity(
            actor, temporary, tolerance=parity_tolerance
        )
        try:
            self_test = policy_contract.check_self_test(temporary, metadata)
        except policy_contract.PolicyContractError as exc:
            raise ValueError(str(exc)) from exc

        # Revalidate every mutable source immediately before publication.
        if sha256_file(checkpoint) != checkpoint_sha256:
            raise RuntimeError("Checkpoint changed while deployment was packaged")
        if sha256_file(gate_path) != gate_sha256:
            raise RuntimeError("V12 stage gate changed while deployment was packaged")
        if validate_gate(gate_path, checkpoint) != gate:
            raise RuntimeError("V12 gate/report evidence changed before publication")
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, output)
        directory_fd = os.open(
            output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return {
            "schema_version": 2,
            "packager": PACKAGER_REVISION,
            "status": "pass",
            "output": str(output),
            "output_sha256": sha256_file(output),
            "checkpoint_sha256": checkpoint_sha256,
            "stage_gate_sha256": gate_sha256,
            "completed_updates": gate["completed_updates"],
            "parity": final_parity,
            "self_test_rows": len(rows),
            "self_test_bound_ratio": self_test,
        }
    finally:
        captured.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stage-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = package_v12_deployment(
        checkpoint=args.checkpoint,
        gate_path=args.stage_gate,
        output=args.output,
        force=args.force,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
