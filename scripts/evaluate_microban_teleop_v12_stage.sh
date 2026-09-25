#!/usr/bin/env bash
# Evaluate one v12 checkpoint and publish a hash-bound resume gate.
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly LOG_ROOT="${PROJECT_ROOT}/logs/rsl_rl/mjlab_microban_teleop_v12"
readonly GATE_ROOT="${PROJECT_ROOT}/artifacts/teleop_v12_gates"

fail() { echo "$*" >&2; exit 2; }
[[ $# -ge 1 && $# -le 2 ]] || fail "Usage: $0 RUN_NAME [ITERATION]"
run_name="$1"
[[ "${run_name}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] \
    || fail "RUN_NAME must be one safe literal directory name."
run_dir="${LOG_ROOT}/${run_name}"
[[ -d "${run_dir}" ]] || fail "Run directory not found: ${run_dir}"

if (( $# == 2 )); then
    iteration="$2"
    [[ "${iteration}" =~ ^[0-9]+$ ]] || fail "ITERATION must be numeric."
else
    iteration=-1
    shopt -s nullglob
    candidates=("${run_dir}"/model_*.pt)
    shopt -u nullglob
    for path in "${candidates[@]}"; do
        name="${path##*/}"
        if [[ "${name}" =~ ^model_([0-9]+)[.]pt$ ]]; then
            value=$((10#${BASH_REMATCH[1]}))
            (( value > iteration )) && iteration="${value}"
        fi
    done
    (( iteration >= 0 )) || fail "No numeric checkpoint found in ${run_dir}."
fi

checkpoint="${run_dir}/model_${iteration}.pt"
[[ -f "${checkpoint}" ]] || fail "Checkpoint not found: ${checkpoint}"
checkpoint_sha="$(sha256sum -- "${checkpoint}" | awk '{print $1}')"
mkdir -p -- "${GATE_ROOT}"
prefix="${GATE_ROOT}/${run_name}_model_${iteration}"
locomotion_report="${prefix}_9x300.json"
tracking_report="${prefix}_tracking.json"
onnx_report="${prefix}_onnx.json"
onnx_path="${prefix}.onnx"
gate_path="${prefix}_gate.json"

cd -- "${PROJECT_ROOT}"
uv run --locked python -m mjlab_microban.scripts.evaluate_teleop_v12_checkpoint \
    "${checkpoint}" --expected-sha256 "${checkpoint_sha}" \
    --output "${locomotion_report}" --force
uv run --locked python -m mjlab_microban.scripts.evaluate_teleop_v12_tracking \
    "${checkpoint}" --expected-sha256 "${checkpoint_sha}" \
    --output "${tracking_report}" --force
uv run --locked --with onnxruntime --with 'protobuf<7' python -m \
    mjlab_microban.scripts.teleop_v12_onnx_gate \
    "${checkpoint}" --expected-sha256 "${checkpoint_sha}" \
    --onnx "${onnx_path}" --output "${onnx_report}" --force
uv run --locked python -m mjlab_microban.scripts.teleop_v12_stage create \
    "${checkpoint}" "${locomotion_report}" "${tracking_report}" \
    "${onnx_report}" "${gate_path}" --force
echo "[PASS] v12 gate: ${gate_path}"
