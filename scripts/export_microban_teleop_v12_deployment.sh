#!/usr/bin/env bash
# Package a final, accepted contract-v12 checkpoint with the real robot parser.
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly LOG_ROOT="${PROJECT_ROOT}/logs/rsl_rl/mjlab_microban_teleop_v12"
readonly GATE_ROOT="${PROJECT_ROOT}/artifacts/teleop_v12_gates"
readonly DEFAULT_MICROBAN_REPO="$(cd -- "${PROJECT_ROOT}/.." && pwd)/microban"

fail() { echo "$*" >&2; exit 2; }
[[ $# -ge 1 && $# -le 3 ]] \
    || fail "Usage: $0 RUN_NAME [OUTPUT_ONNX] [--force]"
run_name="$1"
[[ "${run_name}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] \
    || fail "RUN_NAME must be one safe literal directory name."
output="${2:-${PROJECT_ROOT}/artifacts/microban_teleop_v12.onnx}"
force=()
if (( $# == 3 )); then
    [[ "$3" == "--force" ]] || fail "Third argument must be --force."
    force=(--force)
fi

checkpoint="${LOG_ROOT}/${run_name}/model_14999.pt"
gate="${GATE_ROOT}/${run_name}_model_14999_gate.json"
[[ -f "${checkpoint}" ]] || fail "Final checkpoint not found: ${checkpoint}"
[[ -f "${gate}" ]] || fail "Final stage gate not found: ${gate}"
[[ -f "${DEFAULT_MICROBAN_REPO}/tools/validate_pico_policy.py" ]] \
    || fail "Microban runtime validator not found: ${DEFAULT_MICROBAN_REPO}"

cd -- "${PROJECT_ROOT}"
CUDA_VISIBLE_DEVICES="" uv run --locked --with onnxruntime --with 'protobuf<7' \
    python -m mjlab_microban.scripts.export_teleop_v12_deployment \
    --checkpoint "${checkpoint}" \
    --stage-gate "${gate}" \
    --microban-repo "${DEFAULT_MICROBAN_REPO}" \
    --output "${output}" \
    "${force[@]}"
