#!/usr/bin/env bash
# Staged-v2 phase 2: add foot learning after an accepted HMD/hand phase.
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly LOG_ROOT="${PROJECT_ROOT}/logs/rsl_rl/mjlab_microban_teleop_v12_preview"

fail() { echo "$*" >&2; exit 2; }
(( $# >= 5 )) || fail "Usage: $0 RUN_NAME PHASE1_MODEL_7100 PHASE1_SHA PHASE1_PASS_RECEIPT RECEIPT_SHA [--updates N] [--save-interval N]"
run_name=$1
phase1_checkpoint=$2
phase1_sha=$3
phase1_receipt=$4
receipt_sha=$5
shift 5
updates=100
save_interval=100
while (( $# )); do
    case "$1" in
        --updates)
            (( $# >= 2 )) || fail "--updates requires an integer."
            updates=$2
            shift 2
            ;;
        --save-interval)
            (( $# >= 2 )) || fail "--save-interval requires an integer."
            save_interval=$2
            shift 2
            ;;
        *) fail "Unknown argument: $1" ;;
    esac
done
[[ "${updates}" =~ ^[1-9][0-9]*$ ]] || fail "--updates must be a positive integer."
[[ "${save_interval}" =~ ^[1-9][0-9]*$ ]] || fail "--save-interval must be a positive integer."
(( updates <= 1000 )) || fail "Preview phase2 is capped at 1000 updates."
(( save_interval <= updates )) || fail "--save-interval cannot exceed --updates."
[[ "${run_name}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || fail "Invalid run name."
[[ "${phase1_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "Invalid phase1 SHA-256."
[[ "${receipt_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "Invalid receipt SHA-256."
[[ -f "${phase1_checkpoint}" && ! -L "${phase1_checkpoint}" ]] || fail "Phase1 checkpoint must be a regular file."
[[ -f "${phase1_receipt}" && ! -L "${phase1_receipt}" ]] || fail "Phase1 receipt must be a regular file."
phase1_checkpoint="$(realpath -- "${phase1_checkpoint}")"
phase1_receipt="$(realpath -- "${phase1_receipt}")"

cd -- "${PROJECT_ROOT}"
[[ "$(sha256sum -- "${phase1_checkpoint}" | awk '{print $1}')" == "${phase1_sha}" ]] || fail "Phase1 checkpoint SHA mismatch."
[[ "$(sha256sum -- "${phase1_receipt}" | awk '{print $1}')" == "${receipt_sha}" ]] || fail "Phase1 receipt SHA mismatch."
seed_run="${run_name}_fullbody_clocklift_seed"
seed_dir="${LOG_ROOT}/${seed_run}"
checkpoint="${seed_dir}/model_10000.pt"
lift_receipt="${PROJECT_ROOT}/artifacts/teleop_v12_preview/${run_name}_fullbody_clocklift.json"
[[ ! -e "${seed_dir}" && ! -e "${lift_receipt}" ]] || fail "Phase2 seed/receipt exists; choose a new run name."
mkdir -p -- "${seed_dir}"
uv run --locked python -m \
    mjlab_microban.scripts.lift_teleop_v12_preview_fullbody_checkpoint \
    "${phase1_checkpoint}" "${checkpoint}" \
    --expected-source-sha256 "${phase1_sha}" \
    --phase1-acceptance-receipt "${phase1_receipt}" \
    --expected-receipt-sha256 "${receipt_sha}" \
    --output "${lift_receipt}" >/dev/null

echo "[WARNING] NON-DEPLOYABLE STAGED-V2 FULL-BODY PREVIEW: ${run_name}" >&2
echo "[INFO] clocklift=${checkpoint} updates=${updates} save_interval=${save_interval} foot=active" >&2
exec uv run --locked train Mjlab-Teleop-V12-Preview-Microban \
    --env.scene.num-envs 2048 --env.seed 42 --agent.seed 42 \
    --agent.resume True --agent.load-run "^${seed_run}$" \
    --agent.load-checkpoint '^model_10000[.]pt$' \
    --agent.max-iterations "${updates}" --agent.save-interval "${save_interval}" \
    --agent.run-name "${run_name}" --agent.logger tensorboard \
    --agent.upload-model False --enable-nan-guard True
