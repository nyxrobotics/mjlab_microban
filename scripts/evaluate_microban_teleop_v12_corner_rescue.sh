#!/usr/bin/env bash
# Full unchanged schema-v2 gate and promotion receipt for the targeted rescue.
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly LOG_ROOT="${PROJECT_ROOT}/logs/rsl_rl/mjlab_microban_teleop_v12"
readonly GATE_ROOT="${PROJECT_ROOT}/artifacts/teleop_v12_gates"
readonly ARTIFACT_ROOT="${PROJECT_ROOT}/artifacts/teleop_v12_corner_rescue"

fail() { echo "$*" >&2; exit 2; }
[[ $# == 1 ]] || fail "Usage: $0 RUN_NAME"
run_name="$1"
[[ "${run_name}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] \
    || fail "RUN_NAME must be one safe literal directory name."
checkpoint="${LOG_ROOT}/${run_name}/model_9999.pt"
[[ -f "${checkpoint}" && ! -L "${checkpoint}" ]] \
    || fail "Corner rescue model_9999.pt not found."
checkpoint_sha="$(sha256sum -- "${checkpoint}" | awk '{print $1}')"
mkdir -p -- "${ARTIFACT_ROOT}"
mkdir -p -- "${GATE_ROOT}"
prefix="${GATE_ROOT}/${run_name}_model_9999"
locomotion="${prefix}_9x300.json"
tracking="${prefix}_tracking.json"
onnx_report="${prefix}_onnx.json"
onnx_path="${prefix}.onnx"
gate="${prefix}_gate.json"
receipt="${ARTIFACT_ROOT}/${run_name}_model_9999_receipt.json"

cd -- "${PROJECT_ROOT}"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] \
    || fail "Corner rescue evaluation requires a clean committed source tree."

uv run --locked python -m mjlab_microban.scripts.evaluate_teleop_v12_checkpoint \
    "${checkpoint}" --expected-sha256 "${checkpoint_sha}" \
    --output "${locomotion}" --force
uv run --locked python -m mjlab_microban.scripts.evaluate_teleop_v12_tracking \
    "${checkpoint}" --expected-sha256 "${checkpoint_sha}" \
    --allow-corner-rescue --output "${tracking}" --force
uv run --locked --with onnxruntime --with 'protobuf<7' python -m \
    mjlab_microban.scripts.teleop_v12_onnx_gate \
    "${checkpoint}" --expected-sha256 "${checkpoint_sha}" \
    --onnx "${onnx_path}" --output "${onnx_report}" --force
uv run --locked python -m mjlab_microban.scripts.teleop_v12_stage create \
    "${checkpoint}" "${locomotion}" "${tracking}" "${onnx_report}" \
    "${gate}" --force
uv run --locked python -m mjlab_microban.scripts.teleop_v12_stage validate \
    "${gate}" "${checkpoint}" >/dev/null
uv run --locked python -m mjlab_microban.scripts.teleop_v12_corner_rescue \
    create-receipt "${checkpoint}" "${tracking}" "${gate}" \
    "${receipt}" --force >/dev/null
uv run --locked python -m mjlab_microban.scripts.teleop_v12_corner_rescue \
    validate-receipt "${receipt}" "${checkpoint}" "${tracking}" "${gate}" \
    >/dev/null
echo "[PASS] full schema-v2 gate: ${gate}"
echo "[PASS] authenticated promotion receipt: ${receipt}"
