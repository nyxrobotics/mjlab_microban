#!/usr/bin/env bash
# Exact model10099 post-canary adjudication: 9x300 + foot canary + ONNX + gate.
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly LOG_ROOT="${PROJECT_ROOT}/logs/rsl_rl/mjlab_microban_teleop_v12"
readonly GATE_ROOT="${PROJECT_ROOT}/artifacts/teleop_v12_gates"
readonly ARTIFACT_ROOT="${PROJECT_ROOT}/artifacts/teleop_v12_deadline_fallback"
readonly CANARY_SHA="86a81f45f34d91036ab138963c835a3e78db98224f523d3484e1d3ed335082ff"
readonly STRICT_REPORT_SHA="e8230cff4cd25e8af1d9931d1fcd459db112e5a34c88a20470e22c2019ec5161"

fail() { echo "$*" >&2; exit 2; }
[[ $# == 2 ]] || fail "Usage: $0 RUN_NAME STRICT_TRACKING_REPORT"
run_name="$1"
strict_tracking_input="$2"
[[ "${run_name}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] \
    || fail "RUN_NAME must be one safe literal directory name."
checkpoint="${LOG_ROOT}/${run_name}/model_10099.pt"
[[ -f "${checkpoint}" && ! -L "${checkpoint}" ]] \
    || fail "Pinned model_10099.pt not found."
[[ -f "${strict_tracking_input}" && ! -L "${strict_tracking_input}" ]] \
    || fail "Pinned canonical tracking report not found."
strict_tracking="$(realpath -- "${strict_tracking_input}")"
[[ "$(sha256sum -- "${checkpoint}" | awk '{print $1}')" == "${CANARY_SHA}" ]] \
    || fail "Only the measured model10099 canary is eligible."
[[ "$(sha256sum -- "${strict_tracking}" | awk '{print $1}')" == "${STRICT_REPORT_SHA}" ]] \
    || fail "Pinned canary strict-report SHA-256 mismatch."

mkdir -p -- "${ARTIFACT_ROOT}" "${GATE_ROOT}"
prefix="${GATE_ROOT}/${run_name}_model_10099"
locomotion="${prefix}_9x300.json"
tracking="${prefix}_deadline_canary_tracking.json"
onnx_report="${prefix}_onnx.json"
onnx_path="${prefix}.onnx"
gate="${prefix}_gate.json"
receipt="${ARTIFACT_ROOT}/${run_name}_model_10099_post_canary_receipt.json"

cd -- "${PROJECT_ROOT}"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] \
    || fail "Deadline canary evaluation requires a clean committed source tree."

uv run --locked python -m mjlab_microban.scripts.evaluate_teleop_v12_checkpoint \
    "${checkpoint}" --expected-sha256 "${CANARY_SHA}" \
    --output "${locomotion}" --force
uv run --locked python -m mjlab_microban.scripts.evaluate_teleop_v12_tracking \
    "${checkpoint}" --expected-sha256 "${CANARY_SHA}" \
    --deadline-canary-fallback --output "${tracking}" --force
uv run --locked --with onnxruntime --with 'protobuf<7' python -m \
    mjlab_microban.scripts.teleop_v12_onnx_gate \
    "${checkpoint}" --expected-sha256 "${CANARY_SHA}" \
    --onnx "${onnx_path}" --output "${onnx_report}" --force
uv run --locked python -m mjlab_microban.scripts.teleop_v12_stage \
    create-deadline-canary-fallback "${checkpoint}" "${strict_tracking}" \
    "${locomotion}" "${tracking}" "${onnx_report}" "${gate}" --force
uv run --locked python -m mjlab_microban.scripts.teleop_v12_stage \
    validate "${gate}" "${checkpoint}" >/dev/null
uv run --locked python -m mjlab_microban.scripts.teleop_v12_deadline_fallback \
    create-canary-receipt "${checkpoint}" "${strict_tracking}" "${tracking}" \
    "${gate}" "${receipt}" --force >/dev/null
uv run --locked python -m mjlab_microban.scripts.teleop_v12_deadline_fallback \
    validate-canary-receipt "${receipt}" "${checkpoint}" "${strict_tracking}" \
    "${tracking}" "${gate}" >/dev/null

echo "[PASS] exact model10099 post-canary gate: ${gate}"
echo "[PASS] post-canary promotion receipt: ${receipt}"
echo "[NEXT] scripts/train_microban_teleop_v12.sh resume ${run_name}"
