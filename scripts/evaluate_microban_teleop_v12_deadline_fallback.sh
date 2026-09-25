#!/usr/bin/env bash
# Full v1-only deadline adjudication: 9x300 + tracking + ONNX + schema-v2.
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly LOG_ROOT="${PROJECT_ROOT}/logs/rsl_rl/mjlab_microban_teleop_v12"
readonly GATE_ROOT="${PROJECT_ROOT}/artifacts/teleop_v12_gates"
readonly ARTIFACT_ROOT="${PROJECT_ROOT}/artifacts/teleop_v12_deadline_fallback"
readonly SELECTED_SHA="393d35b4e7cc0453f5143c7f2be4d4a4658567ab6132dfb54d32e67eb26b62b7"
readonly STRICT_REPORT_SHA="c466d66cf5b5ac8603f558e0ae8612450ac74bbad78fd88719a2cdaa9673e4b7"

fail() { echo "$*" >&2; exit 2; }
[[ $# == 2 ]] || fail "Usage: $0 RUN_NAME STRICT_TRACKING_REPORT"
run_name="$1"
strict_tracking_input="$2"
[[ "${run_name}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] \
    || fail "RUN_NAME must be one safe literal directory name."
checkpoint="${LOG_ROOT}/${run_name}/model_9999.pt"
[[ -f "${checkpoint}" && ! -L "${checkpoint}" ]] \
    || fail "Selected model_9999.pt not found."
[[ -f "${strict_tracking_input}" && ! -L "${strict_tracking_input}" ]] \
    || fail "Pinned strict tracking report not found."
strict_tracking="$(realpath -- "${strict_tracking_input}")"
[[ "$(sha256sum -- "${checkpoint}" | awk '{print $1}')" == "${SELECTED_SHA}" ]] \
    || fail "Only the selected v1 checkpoint is eligible."
[[ "$(sha256sum -- "${strict_tracking}" | awk '{print $1}')" == "${STRICT_REPORT_SHA}" ]] \
    || fail "Pinned strict tracking report SHA-256 mismatch."

mkdir -p -- "${ARTIFACT_ROOT}" "${GATE_ROOT}"
prefix="${GATE_ROOT}/${run_name}_model_9999"
locomotion="${prefix}_9x300.json"
tracking="${prefix}_deadline_tracking.json"
onnx_report="${prefix}_deadline_onnx.json"
onnx_path="${prefix}_deadline.onnx"
gate="${prefix}_gate.json"
receipt="${ARTIFACT_ROOT}/${run_name}_model_9999_receipt.json"

cd -- "${PROJECT_ROOT}"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] \
    || fail "Deadline fallback evaluation requires a clean committed source tree."

uv run --locked python -m mjlab_microban.scripts.evaluate_teleop_v12_checkpoint \
    "${checkpoint}" --expected-sha256 "${SELECTED_SHA}" --deadline-fallback \
    --output "${locomotion}" --force
uv run --locked python -m mjlab_microban.scripts.evaluate_teleop_v12_tracking \
    "${checkpoint}" --expected-sha256 "${SELECTED_SHA}" --deadline-fallback \
    --output "${tracking}" --force
uv run --locked --with onnxruntime --with 'protobuf<7' python -m \
    mjlab_microban.scripts.teleop_v12_onnx_gate \
    "${checkpoint}" --expected-sha256 "${SELECTED_SHA}" --deadline-fallback \
    --onnx "${onnx_path}" --output "${onnx_report}" --force
uv run --locked python -m mjlab_microban.scripts.teleop_v12_stage \
    create-deadline-fallback "${checkpoint}" "${strict_tracking}" \
    "${locomotion}" "${tracking}" "${onnx_report}" "${gate}" --force
uv run --locked python -m mjlab_microban.scripts.teleop_v12_stage \
    validate "${gate}" "${checkpoint}" >/dev/null
uv run --locked python -m mjlab_microban.scripts.teleop_v12_deadline_fallback \
    create-receipt "${checkpoint}" "${strict_tracking}" "${tracking}" \
    "${gate}" "${receipt}" --force >/dev/null
uv run --locked python -m mjlab_microban.scripts.teleop_v12_deadline_fallback \
    validate-receipt "${receipt}" "${checkpoint}" "${strict_tracking}" \
    "${tracking}" "${gate}" >/dev/null

echo "[PASS] v1-only deadline fallback gate: ${gate}"
echo "[PASS] deadline promotion receipt: ${receipt}"
echo "[NEXT] scripts/train_microban_teleop_v12.sh resume ${run_name}"
