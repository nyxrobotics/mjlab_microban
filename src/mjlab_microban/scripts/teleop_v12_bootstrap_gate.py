"""Publish the contract-v12 pristine parity and ONNX bootstrap receipt."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
import onnx
import torch
from onnx.reference import ReferenceEvaluator
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from mjlab_microban.tasks.microban_policy_export import MICROBAN_TELEOP_OBSERVATION_WIDTH
from mjlab_microban.legacy_velocity_diagnostics import publish_json_atomic
from mjlab_microban.tasks.microban_teleop_v12_actor import (
    LEGACY_TO_TELEOP_OBSERVATION_INDEX,
    TELEOP_V12_EXTRA_OBSERVATION_COLUMNS,
    LegacyAdapterTeleopActor,
)
from mjlab_microban.tasks.microban_teleop_v12_bootstrap import (
    bootstrap_legacy_actor,
    inspect_legacy_velocity_checkpoint,
    serialize_bootstrap_provenance,
    sha256_file,
)
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MICROBAN_TELEOP_V12_ACTION_CLIP,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = ROOT / "artifacts/teleop_v12_bootstrap"
PRISTINE_PARITY_TOLERANCE = 2.0e-5
ONNX_PARITY_TOLERANCE = 2.0e-5


def _observations(width: int) -> TensorDict:
    return TensorDict({"actor": torch.zeros(1, width)}, batch_size=[1])


def _legacy_model() -> MLPModel:
    return MLPModel(
        obs=_observations(63),
        obs_groups={"actor": ["actor"]},
        obs_set="actor",
        output_dim=18,
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    )


def _teleop_model() -> LegacyAdapterTeleopActor:
    return LegacyAdapterTeleopActor(
        obs=_observations(MICROBAN_TELEOP_OBSERVATION_WIDTH),
        obs_groups={"actor": ["actor"]},
        obs_set="actor",
        output_dim=18,
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    )


def _export_onnx_atomic(actor: LegacyAdapterTeleopActor, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    model = actor.as_onnx(verbose=False).cpu().eval()
    try:
        torch.onnx.export(
            model,
            model.get_dummy_inputs(),  # type: ignore[operator]
            temporary,
            export_params=True,
            opset_version=18,
            input_names=model.input_names,
            output_names=model.output_names,
            dynamic_axes={},
            dynamo=False,
        )
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def run_gate(
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
    probe_receipt: Path,
    probe_receipt_sha256: str,
    onnx_path: Path,
) -> dict[str, object]:
    import onnxruntime as ort
    source_identity, source_state = inspect_legacy_velocity_checkpoint(
        checkpoint, checkpoint_sha256
    )
    source = _legacy_model()
    source.load_state_dict(source_state, strict=True)
    source.eval()
    target = _teleop_model()
    provenance = bootstrap_legacy_actor(
        target,
        checkpoint,
        checkpoint_sha256,
        probe_receipt,
        probe_receipt_sha256,
    )
    target.eval()

    generator = torch.Generator().manual_seed(20260925)
    teleop_observations = torch.randn(10_000, MICROBAN_TELEOP_OBSERVATION_WIDTH, generator=generator)
    legacy_observations = teleop_observations[
        :, [target for _source, target in LEGACY_TO_TELEOP_OBSERVATION_INDEX]
    ]
    # Legacy parity checks the 63->81 weight mapping, so it is evaluated in
    # float64 on copies: in float32 the two widths accumulate in different
    # orders and randn inputs drive raw actions to hundreds of radians, where
    # one float32 ulp alone exceeds the 2e-5 bound.  A mapping defect still
    # shows as O(1e-3) or more; the deployed float32 export is checked by the
    # ONNX parity below.
    with torch.inference_mode():
        source_actions = copy.deepcopy(source).double()(
            TensorDict({"actor": legacy_observations.double()}, batch_size=[10_000])
        )
        target_actions = copy.deepcopy(target).double()(
            TensorDict({"actor": teleop_observations.double()}, batch_size=[10_000])
        )
    parity_max = float(torch.max(torch.abs(source_actions - target_actions)).item())
    if parity_max > PRISTINE_PARITY_TOLERANCE:
        raise ValueError(
            f"Pristine legacy parity failed: {parity_max} > {PRISTINE_PARITY_TOLERANCE}"
        )
    first = target.mlp[0]
    assert isinstance(first, torch.nn.Linear)
    if torch.count_nonzero(
        first.weight[:, TELEOP_V12_EXTRA_OBSERVATION_COLUMNS]
    ).item() != 0:
        raise ValueError("Pristine teleop-only W0 columns are not exact zero")

    _export_onnx_atomic(target, onnx_path)
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model, full_check=True)
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("Contract-v12 ONNX must have one input and one output")
    evaluator = ReferenceEvaluator(model)
    runtime = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    if runtime.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("ONNX Runtime did not select only CPUExecutionProvider")
    # Same norm-wise rule as the stage ONNX gate (teleop_v12_onnx_gate): the
    # randn corpus drives raw actions to hundreds of radians, where float32
    # accumulation-order differences alone exceed a pure 2e-5 bound.
    from mjlab_microban.scripts.teleop_v12_onnx_gate import (
        ONNX_PARITY_RELATIVE_TOLERANCE,
        ONNX_PARITY_RULE,
        parity_bound_ratio,
    )

    reference_observations = teleop_observations[:64]
    reference_errors: list[float] = []
    runtime_errors: list[float] = []
    reference_ratios: list[float] = []
    runtime_ratios: list[float] = []
    expected_magnitudes: list[float] = []
    export_model = target.as_onnx(verbose=False).cpu().eval()
    with torch.inference_mode():
        for observation in reference_observations:
            batch = observation.unsqueeze(0)
            expected = export_model(batch).cpu().numpy()
            (actual,) = evaluator.run(None, {"obs": batch.cpu().numpy()})
            reference_errors.append(float(np.max(np.abs(actual - expected))))
            (runtime_actual,) = runtime.run(None, {"obs": batch.cpu().numpy()})
            runtime_errors.append(
                float(np.max(np.abs(runtime_actual - expected)))
            )
            reference_ratios.append(
                parity_bound_ratio(actual, expected, atol=ONNX_PARITY_TOLERANCE)
            )
            runtime_ratios.append(
                parity_bound_ratio(
                    runtime_actual, expected, atol=ONNX_PARITY_TOLERANCE
                )
            )
            expected_magnitudes.append(float(np.max(np.abs(expected))))
    reference_max = max(reference_errors)
    runtime_max = max(runtime_errors)
    reference_ratio = max(reference_ratios)
    runtime_ratio = max(runtime_ratios)
    if not np.isfinite(reference_ratio) or reference_ratio > 1.0:
        raise ValueError(
            "ONNX ReferenceEvaluator parity failed: "
            f"{reference_max} (bound ratio {reference_ratio})"
        )
    if not np.isfinite(runtime_ratio) or runtime_ratio > 1.0:
        raise ValueError(
            f"ONNX Runtime CPU parity failed: {runtime_max} "
            f"(bound ratio {runtime_ratio})"
        )

    return {
        "schema_version": 1,
        "gate": "microban_teleop_v12_pristine_bootstrap",
        "status": "pass",
        "bootstrap": serialize_bootstrap_provenance(provenance),
        "source_identity": {
            "path": source_identity.path,
            "sha256": source_identity.sha256,
            "iteration": source_identity.iteration,
        },
        "contract": {
            "observation_width": MICROBAN_TELEOP_OBSERVATION_WIDTH,
            "action_width": 18,
            "action_clip": list(MICROBAN_TELEOP_V12_ACTION_CLIP),
            "previous_action": "raw_actor_output",
            "normalizer": "frozen_empirical",
            "trainable_actor_parameter": "mlp.0.weight",
            "trainable_actor_columns": list(TELEOP_V12_EXTRA_OBSERVATION_COLUMNS),
        },
        "pristine_parity": {
            "seed": 20260925,
            "samples": 10_000,
            "maximum_absolute_error": parity_max,
            "tolerance": PRISTINE_PARITY_TOLERANCE,
        },
        "closed_loop_evidence": {
            "receipt": str(probe_receipt.resolve()),
            "sha256": probe_receipt_sha256,
            "scenarios": 9,
            "steps_per_scenario": 300,
        },
        "onnx": {
            "path": str(onnx_path.resolve()),
            "sha256": sha256_file(onnx_path),
            "opset": 18,
            "reference_samples": len(reference_observations),
            "reference_evaluator_maximum_absolute_error": reference_max,
            "onnxruntime_cpu_maximum_absolute_error": runtime_max,
            "onnxruntime_version": ort.__version__,
            "onnxruntime_providers": runtime.get_providers(),
            "tolerance": ONNX_PARITY_TOLERANCE,
            "relative_tolerance": ONNX_PARITY_RELATIVE_TOLERANCE,
            "parity_rule": ONNX_PARITY_RULE,
            "maximum_absolute_expected_output": max(expected_magnitudes),
            "reference_evaluator_maximum_bound_ratio": reference_ratio,
            "onnxruntime_cpu_maximum_bound_ratio": runtime_ratio,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--probe-receipt", type=Path, required=True)
    parser.add_argument("--probe-receipt-sha256", required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    onnx_path = output_dir / "pristine_actor.onnx"
    receipt_path = output_dir / "pristine_bootstrap_gate.json"
    if not args.force and (onnx_path.exists() or receipt_path.exists()):
        raise FileExistsError("Bootstrap outputs already exist (pass --force)")
    report = run_gate(
        checkpoint=args.checkpoint,
        checkpoint_sha256=args.checkpoint_sha256,
        probe_receipt=args.probe_receipt,
        probe_receipt_sha256=args.probe_receipt_sha256,
        onnx_path=onnx_path,
    )
    publish_json_atomic(receipt_path, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
