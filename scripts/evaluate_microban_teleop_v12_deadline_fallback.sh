#!/usr/bin/env bash
# Deadline adjudication of a corner-rescue model_9999 whose strict report fails
# only hand RMS: 9x300 + 35 mm tracking + ONNX + schema-v2 gate.  The rescue
# checkpoint and strict report are recorded in the gate marker (no pins).
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly LOG_ROOT="${PROJECT_ROOT}/logs/rsl_rl/mjlab_microban_teleop_v12"
readonly GATE_ROOT="${PROJECT_ROOT}/artifacts/teleop_v12_gates"
readonly ARTIFACT_ROOT="${PROJECT_ROOT}/artifacts/teleop_v12_deadline_fallback"

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
    || fail "Strict tracking report not found."
strict_tracking="$(realpath -- "${strict_tracking_input}")"
SELECTED_SHA="$(sha256sum -- "${checkpoint}" | awk '{print $1}')"

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

echo "[PASS] deadline fallback gate: ${gate}"
echo "[PASS] deadline promotion receipt: ${receipt}"
echo "[NEXT] scripts/train_microban_teleop_v12.sh resume ${run_name}"
