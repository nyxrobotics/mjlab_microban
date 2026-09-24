#!/usr/bin/env bash
# API/contract preflight before starting the first v8 stage.
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd -- "${PROJECT_ROOT}"
uv run --locked --with pytest python -m pytest -q \
    tests/test_bounded_std_projection.py \
    tests/test_microban_policy_export.py \
    tests/test_signed_axis_velocity_command.py \
    tests/test_teleop_evaluation.py \
    tests/test_teleop_onnx_parity_gate.py \
    tests/test_teleop_training_provenance.py \
    tests/test_teleop_v2_contract.py \
    tests/test_teleop_v8_stage.py \
    tests/test_teleop_velocity_bootstrap.py
"${SCRIPT_DIR}/train_microban_teleop.sh" smoke
echo "[PASS] contract v8 unit and CUDA environment smoke preflight"
