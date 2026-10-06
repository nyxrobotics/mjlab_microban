"""Export and parity-check any contract-v12 checkpoint with ONNX Runtime CPU."""

from __future__ import annotations

import argparse
import copy
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
    TELEOP_V12_EXTRA_OBSERVATION_COLUMNS,
)
from mjlab_microban.tasks.microban_teleop_v12_runner import (
    TELEOP_V12_BOOTSTRAP_INFO_KEY,
)
from mjlab_microban.tasks.microban_teleop_v12_bootstrap import (
    load_bootstrap_source_state,
    validate_bootstrap_provenance,
    sha256_file,
)

# The 64-sample corpus feeds randn into raw observation columns, so the
# normalized teleop-target columns see ~20-sigma inputs and raw actions reach
# tens of radians.  Float32 accumulation-order differences between torch and
# ONNX Runtime then scale with the magnitude of the hidden activations, and
# they land on every output of that sample, including outputs that cancel to
# a small value.  A pure absolute bound of 2e-5 fails at ~5e-7 of the
# sample's output scale (a few ulps).  The bound is therefore norm-wise per
# sample: max|onnx - torch| <= atol + rtol * max|torch|.  rtol = 1e-6 is ~8
# float32 ulps, far below any real export defect (O(1e-3) and above).
ONNX_PARITY_RELATIVE_TOLERANCE = 1.0e-6
ONNX_PARITY_RULE = "max_abs_error_le_atol_plus_rtol_times_max_abs_expected_per_sample_v1"


def parity_bound_ratio(
    actual: np.ndarray, expected: np.ndarray, *, atol: float
) -> float:
    """Return max|actual - expected| / (atol + rtol * max|expected|) per sample."""

    bound = atol + ONNX_PARITY_RELATIVE_TOLERANCE * float(np.max(np.abs(expected)))
    return float(np.max(np.abs(actual - expected)) / bound)


def run_gate(
    *,
    checkpoint: Path,
    expected_sha256: str | None,
    onnx_path: Path,
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
    target, iteration, infos = _load_actor(checkpoint, device="cpu")
    parity_tolerance = ONNX_PARITY_TOLERANCE
    # The frozen velocity source is whatever this checkpoint was bootstrapped
    # from; its recorded SHA-256 is re-verified before the tensors are used.
    source_state = load_bootstrap_source_state(
        validate_bootstrap_provenance(
            infos.get(TELEOP_V12_BOOTSTRAP_INFO_KEY), verify_files=True
        )
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
    # Float64 copies: this checks the frozen 63->83 mapping exactly; float32
    # accumulation order differs between the two widths and alone exceeds 2e-5
    # once randn inputs drive raw actions to hundreds of radians (see
    # teleop_v12_bootstrap_gate).  The float32 export is checked below.
    with torch.inference_mode():
        expected_actions = copy.deepcopy(source).double()(
            TensorDict({"actor": legacy_observations.double()}, batch_size=[10_000])
        )
        actual_actions = copy.deepcopy(target).double()(
            TensorDict({"actor": neutral_observations.double()}, batch_size=[10_000])
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
    reference_ratios: list[float] = []
    runtime_ratios: list[float] = []
    expected_magnitudes: list[float] = []
    with torch.inference_mode():
        for observation in onnx_observations:
            batch = observation.unsqueeze(0)
            expected = export_model(batch).numpy()
            (reference_actual,) = reference.run(None, {"obs": batch.numpy()})
            (runtime_actual,) = runtime.run(None, {"obs": batch.numpy()})
            reference_errors.append(float(np.max(np.abs(reference_actual - expected))))
            runtime_errors.append(float(np.max(np.abs(runtime_actual - expected))))
            reference_ratios.append(
                parity_bound_ratio(reference_actual, expected, atol=parity_tolerance)
            )
            runtime_ratios.append(
                parity_bound_ratio(runtime_actual, expected, atol=parity_tolerance)
            )
            expected_magnitudes.append(float(np.max(np.abs(expected))))
    reference_max = max(reference_errors)
    runtime_max = max(runtime_errors)
    reference_ratio = max(reference_ratios)
    runtime_ratio = max(runtime_ratios)
    expected_max = max(expected_magnitudes)
    if not all(
        np.isfinite(value)
        for value in (reference_max, runtime_max, reference_ratio, runtime_ratio)
    ) or (reference_ratio > 1.0 or runtime_ratio > 1.0):
        raise ValueError(
            "ONNX parity failed: "
            f"reference={reference_max} (bound ratio {reference_ratio}), "
            f"onnxruntime_cpu={runtime_max} (bound ratio {runtime_ratio}), "
            f"max |expected|={expected_max}"
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
            "tolerance": parity_tolerance,
            "relative_tolerance": ONNX_PARITY_RELATIVE_TOLERANCE,
            "parity_rule": ONNX_PARITY_RULE,
            "maximum_absolute_expected_output": expected_max,
            "reference_evaluator_maximum_bound_ratio": reference_ratio,
            "onnxruntime_cpu_maximum_bound_ratio": runtime_ratio,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
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
    )
    if args.output is not None:
        publish_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
