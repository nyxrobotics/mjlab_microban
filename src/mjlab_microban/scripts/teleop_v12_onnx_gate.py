"""Export and parity-check any contract-v12 checkpoint with ONNX Runtime CPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx.reference import ReferenceEvaluator
from tensordict import TensorDict

from mjlab_microban.legacy_velocity_diagnostics import publish_json_atomic
from mjlab_microban.scripts.evaluate_teleop_v12_checkpoint import _load_actor
from mjlab_microban.scripts.teleop_v12_bootstrap_gate import (
    ONNX_PARITY_TOLERANCE,
    PRISTINE_PARITY_TOLERANCE,
    _export_onnx_atomic,
    _legacy_model,
)
from mjlab_microban.tasks.microban_teleop_v12_actor import (
    LEGACY_TO_TELEOP_OBSERVATION_INDEX,
    LEGACY_VELOCITY_CHECKPOINT_SHA256,
    TELEOP_V12_EXTRA_OBSERVATION_COLUMNS,
)
from mjlab_microban.tasks.microban_teleop_v12_bootstrap import (
    inspect_legacy_velocity_checkpoint,
    sha256_file,
)


def run_gate(
    *,
    checkpoint: Path,
    expected_sha256: str | None,
    onnx_path: Path,
    allow_deadline_fallback: bool = False,
) -> dict[str, object]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime CPU is required; run with `uv run --locked --with "
            "onnxruntime --with 'protobuf<7'`"
        ) from exc

    checkpoint = checkpoint.expanduser().resolve()
    checkpoint_digest = sha256_file(checkpoint)
    if expected_sha256 is not None and checkpoint_digest != expected_sha256:
        raise ValueError(f"Checkpoint SHA-256 mismatch: {checkpoint_digest}")
    target, iteration, _infos = _load_actor(
        checkpoint,
        device="cpu",
        allow_deadline_fallback=allow_deadline_fallback,
    )
    _source_identity, source_state = inspect_legacy_velocity_checkpoint(
        "repo://checkpoints/xc330_velocity/model_14999.pt",
        LEGACY_VELOCITY_CHECKPOINT_SHA256,
    )
    source = _legacy_model()
    source.load_state_dict(source_state, strict=True)
    source.eval()

    generator = torch.Generator().manual_seed(20260925)
    neutral_observations = torch.randn(10_000, 83, generator=generator)
    neutral_observations[:, TELEOP_V12_EXTRA_OBSERVATION_COLUMNS] = 0.0
    legacy_observations = neutral_observations[
        :, [target for _source, target in LEGACY_TO_TELEOP_OBSERVATION_INDEX]
    ]
    with torch.inference_mode():
        expected_actions = source(
            TensorDict({"actor": legacy_observations}, batch_size=[10_000])
        )
        actual_actions = target(
            TensorDict({"actor": neutral_observations}, batch_size=[10_000])
        )
    neutral_max = float(torch.max(torch.abs(actual_actions - expected_actions)).item())
    if neutral_max > PRISTINE_PARITY_TOLERANCE:
        raise ValueError(
            f"Neutral legacy parity failed: {neutral_max} > {PRISTINE_PARITY_TOLERANCE}"
        )

    _export_onnx_atomic(target, onnx_path)
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model, full_check=True)
    reference = ReferenceEvaluator(model)
    runtime = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    if runtime.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("ONNX Runtime did not select CPUExecutionProvider only")
    export_model = target.as_onnx(verbose=False).cpu().eval()
    # Keep neutral legacy equivalence and export compatibility as independent
    # checks.  An adapter could pass the former while ONNX silently drops or
    # misorders every learned teleop-only column, so exercise all 83 inputs here.
    onnx_observations = torch.randn(64, 83, generator=generator)
    if not bool(
        torch.all(
            onnx_observations[:, TELEOP_V12_EXTRA_OBSERVATION_COLUMNS].abs().amax(dim=0)
            > 0.0
        ).item()
    ):
        raise AssertionError("ONNX parity corpus did not cover every extra column")
    reference_errors: list[float] = []
    runtime_errors: list[float] = []
    with torch.inference_mode():
        for observation in onnx_observations:
            batch = observation.unsqueeze(0)
            expected = export_model(batch).numpy()
            (reference_actual,) = reference.run(None, {"obs": batch.numpy()})
            (runtime_actual,) = runtime.run(None, {"obs": batch.numpy()})
            reference_errors.append(float(np.max(np.abs(reference_actual - expected))))
            runtime_errors.append(float(np.max(np.abs(runtime_actual - expected))))
    reference_max = max(reference_errors)
    runtime_max = max(runtime_errors)
    if reference_max > ONNX_PARITY_TOLERANCE or runtime_max > (ONNX_PARITY_TOLERANCE):
        raise ValueError(
            "ONNX parity failed: "
            f"reference={reference_max}, onnxruntime_cpu={runtime_max}"
        )
    return {
        "schema_version": 1,
        "gate": "microban_teleop_v12_checkpoint_onnx",
        "status": "pass",
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_digest,
            "iteration": iteration,
            "completed_updates": 0 if iteration == -1 else iteration + 1,
        },
        "neutral_legacy_parity": {
            "samples": 10_000,
            "maximum_absolute_error": neutral_max,
            "tolerance": PRISTINE_PARITY_TOLERANCE,
            "teleop_only_columns": "exact_zero",
        },
        "onnx": {
            "path": str(onnx_path.resolve()),
            "sha256": sha256_file(onnx_path),
            "opset": 18,
            "input_shape": [1, 83],
            "output_shape": [1, 18],
            "reference_samples": 64,
            "input_coverage": "deterministic_nonzero_all_83_columns",
            "teleop_only_columns_nonzero": True,
            "reference_evaluator_maximum_absolute_error": reference_max,
            "onnxruntime_cpu_maximum_absolute_error": runtime_max,
            "onnxruntime_version": ort.__version__,
            "onnxruntime_providers": runtime.get_providers(),
            "tolerance": ONNX_PARITY_TOLERANCE,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--deadline-fallback",
        action="store_true",
        help="accept only the hash-pinned v1 deadline-fallback checkpoint",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.force and (
        args.onnx.expanduser().exists()
        or (args.output is not None and args.output.expanduser().exists())
    ):
        raise FileExistsError("Gate output exists (pass --force)")
    report = run_gate(
        checkpoint=args.checkpoint,
        expected_sha256=args.expected_sha256,
        onnx_path=args.onnx.expanduser().resolve(),
        allow_deadline_fallback=args.deadline_fallback,
    )
    if args.output is not None:
        publish_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
