#!/usr/bin/env bash
# One-time, hash-pinned 99-update corner replay from corrected model9900.
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly LOG_ROOT="${PROJECT_ROOT}/logs/rsl_rl/mjlab_microban_teleop_v12"
readonly SOURCE_SHA="063a8f65ebf9007d63395e9a5b98420eb025bd39416dab5727f9f4c06fc6e877"
readonly PARENT_TRACKING_SHA="399db0cee55c137d3d0226ebcb54b0fa3bb84c95496c5c206af55f2fb25837f4"
readonly SOURCE_ITERATION=9900
readonly PROCESS_UPDATES=99
readonly SEED_RUN="corner_rescue_seed_model9900"

usage() {
    cat <<'EOF'
Usage:
  scripts/train_microban_teleop_v12_corner_rescue.sh MODEL_9900 PARENT_TRACKING_REPORT \
    [--agent.run-name NAME]

MODEL_9900 must have the pinned corrected-replay SHA-256. The launcher stages
those immutable bytes under the dedicated experiment. PARENT_TRACKING_REPORT
must be the pinned strict report whose only failure is hand RMS. The launcher
validates both inputs and full-state lineage on CPU, then runs exactly 99
updates to model_9999. No training override other than the output run name is
accepted.
EOF
}

fail() { echo "$*" >&2; exit 2; }

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
(( $# >= 2 )) || { usage >&2; exit 2; }
source_checkpoint="$1"
parent_tracking_report="$2"
shift 2
output_run_name="v12_corner_rescue_9901_to10000"
if (( $# > 0 )); then
    [[ "$1" == "--agent.run-name" && $# == 2 ]] \
        || fail "Only one optional --agent.run-name NAME is supported."
    [[ "$2" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] \
        || fail "Unsafe output run name."
    output_run_name="$2"
fi
[[ "${output_run_name}" != "${SEED_RUN}" ]] \
    || fail "Output run name is reserved for the immutable seed."

source_checkpoint="$(realpath -e -- "${source_checkpoint}")"
parent_tracking_report="$(realpath -e -- "${parent_tracking_report}")"
[[ -f "${source_checkpoint}" ]] || fail "model9900 not found."
[[ -f "${parent_tracking_report}" ]] || fail "Parent tracking report not found."
[[ "$(sha256sum -- "${source_checkpoint}" | awk '{print $1}')" == "${SOURCE_SHA}" ]] \
    || fail "Pinned corrected model9900 SHA-256 mismatch."
[[ "$(sha256sum -- "${parent_tracking_report}" | awk '{print $1}')" == "${PARENT_TRACKING_SHA}" ]] \
    || fail "Pinned parent tracking report SHA-256 mismatch."

cd -- "${PROJECT_ROOT}"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] \
    || fail "Corner rescue training requires a clean committed source tree."
uv run --locked python -m mjlab_microban.scripts.teleop_v12_corner_rescue \
    validate-parent "${source_checkpoint}" "${parent_tracking_report}" >/dev/null

seed_dir="${LOG_ROOT}/${SEED_RUN}"
seed_checkpoint="${seed_dir}/model_${SOURCE_ITERATION}.pt"
mkdir -p -- "${seed_dir}"
if [[ -e "${seed_checkpoint}" ]]; then
    [[ -f "${seed_checkpoint}" && ! -L "${seed_checkpoint}" ]] \
        || fail "Existing rescue seed is not a regular file."
    [[ "$(sha256sum -- "${seed_checkpoint}" | awk '{print $1}')" == "${SOURCE_SHA}" ]] \
        || fail "Existing rescue seed has the wrong SHA-256."
else
    cp --reflink=auto --no-clobber -- "${source_checkpoint}" "${seed_checkpoint}"
    [[ "$(sha256sum -- "${seed_checkpoint}" | awk '{print $1}')" == "${SOURCE_SHA}" ]] \
        || fail "Staged rescue seed changed during copy."
fi

echo "[INFO] authenticated 40/40/20 corner rescue completed=9901 target=10000 process_updates=${PROCESS_UPDATES}"
exec uv run --locked train Mjlab-Teleop-V12-Corner-Rescue-Microban \
    --env.scene.num-envs 2048 --env.seed 42 --agent.seed 42 \
    --agent.num-steps-per-env 24 --agent.max-iterations "${PROCESS_UPDATES}" \
    --agent.save-interval "${PROCESS_UPDATES}" --agent.logger tensorboard \
    --agent.upload-model False --enable-nan-guard True \
    --agent.resume True \
    --agent.load-run "^${SEED_RUN}$" \
    --agent.load-checkpoint "^model_${SOURCE_ITERATION}[.]pt$" \
    --agent.run-name "${output_run_name}"
