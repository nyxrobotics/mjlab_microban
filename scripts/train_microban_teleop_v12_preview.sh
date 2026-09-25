#!/usr/bin/env bash
# Explicit simulation-only hand/foot preview. Never produces a deployable model.
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly LOG_ROOT="${PROJECT_ROOT}/logs/rsl_rl/mjlab_microban_teleop_v12_preview"
readonly SOURCE="${PROJECT_ROOT}/logs/rsl_rl/mjlab_microban_teleop_v12/2026-09-25_16-30-00_v12_sanitized601/model_600.pt"
readonly SOURCE_SHA="d31d5362dc6776a47395bcefc446bb4324d87361e423edc4a575b2b972d269a3"

fail() { echo "$*" >&2; exit 2; }
run_name="${1:-}"
[[ "${run_name}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] \
    || fail "Usage: $0 SAFE_RUN_NAME [--updates 1..1000]"
shift
updates=100
if (( $# > 0 )); then
    [[ "$1" == "--updates" && $# == 2 && "$2" =~ ^[0-9]+$ ]] \
        || fail "Only --updates 1..1000 is supported."
    updates=$((10#$2))
fi
(( updates >= 1 && updates <= 1000 )) \
    || fail "Preview updates must be in 1..1000."

cd -- "${PROJECT_ROOT}"
[[ "$(sha256sum -- "${SOURCE}" | awk '{print $1}')" == "${SOURCE_SHA}" ]] \
    || fail "Canonical sanitized601 source SHA-256 mismatch."
seed_run="${run_name}_clocklift_seed"
seed_dir="${LOG_ROOT}/${seed_run}"
checkpoint="${seed_dir}/model_10000.pt"
receipt="${PROJECT_ROOT}/artifacts/teleop_v12_preview/${run_name}_clocklift.json"
[[ ! -e "${seed_dir}" && ! -e "${receipt}" ]] \
    || fail "Preview seed/receipt already exists; choose a new run name."
mkdir -p -- "${seed_dir}"
uv run --locked python -m \
    mjlab_microban.scripts.create_teleop_v12_preview_checkpoint \
    "${SOURCE}" "${checkpoint}" --output "${receipt}" >/dev/null

echo "[WARNING] NON-DEPLOYABLE SIMULATION PREVIEW: ${run_name}" >&2
echo "[INFO] clocklift=${checkpoint} updates=${updates}" >&2
exec uv run --locked train Mjlab-Teleop-V12-Preview-Microban \
    --env.scene.num-envs 2048 --env.seed 42 --agent.seed 42 \
    --agent.resume True --agent.load-run "^${seed_run}$" \
    --agent.load-checkpoint '^model_10000[.]pt$' \
    --agent.max-iterations "${updates}" --agent.save-interval "${updates}" \
    --agent.run-name "${run_name}" --agent.logger tensorboard \
    --agent.upload-model False --enable-nan-guard True
