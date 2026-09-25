#!/usr/bin/env bash
# One-time, hash-pinned replay of corrected model9200 to the canonical 10000 gate.
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly LOG_ROOT="${PROJECT_ROOT}/logs/rsl_rl/mjlab_microban_teleop_v12"
readonly SOURCE_SHA="16c9b9d19df6513851b3da26228ae612512fdb2d894542741f0474e4762691c7"
readonly SOURCE_ITERATION=9200
readonly PROCESS_UPDATES=799

usage() {
    cat <<'EOF'
Usage:
  scripts/train_microban_teleop_v12_lr_recovery.sh \
    SEED_RUN RAW_MODEL_9200 MIGRATION_RECEIPT RECOVERY_RECEIPT \
    [--agent.run-name NAME]

SEED_RUN must contain the authenticated migrated checkpoint as model_9200.pt.
The launcher revalidates the pinned raw source, both receipts, the complete
checkpoint permutation, the preserved canonical clock, and exact-zero inactive
foot state before starting exactly 799 updates. It cannot start, canary, extend,
or resume any other interval. The output endpoint is canonical model_9999.pt;
run the ordinary strict v12 stage gate there before activating feet.
EOF
}

fail() { echo "$*" >&2; exit 2; }

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
(( $# >= 4 )) || { usage >&2; exit 2; }

seed_run="$1"
raw_source="$2"
migration_receipt="$3"
recovery_receipt="$4"
shift 4

[[ "${seed_run}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] \
    || fail "SEED_RUN must be one safe literal directory name."
output_run_name="v12_lr_recovery_9201_to10000"
if (( $# > 0 )); then
    [[ "$1" == "--agent.run-name" && $# == 2 ]] \
        || fail "Only one optional --agent.run-name NAME is supported."
    [[ "$2" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] \
        || fail "Unsafe output run name."
    output_run_name="$2"
fi

seed_dir="${LOG_ROOT}/${seed_run}"
migrated_checkpoint="${seed_dir}/model_${SOURCE_ITERATION}.pt"
[[ -d "${seed_dir}" ]] || fail "Recovery seed run not found: ${seed_dir}"
[[ -f "${migrated_checkpoint}" ]] \
    || fail "Migrated recovery checkpoint not found: ${migrated_checkpoint}"
[[ -f "${raw_source}" ]] || fail "Raw model9200 not found: ${raw_source}"
[[ -f "${migration_receipt}" ]] \
    || fail "Migration receipt not found: ${migration_receipt}"
[[ -f "${recovery_receipt}" ]] \
    || fail "Recovery receipt not found: ${recovery_receipt}"

cd -- "${PROJECT_ROOT}"
[[ "$(sha256sum -- "${raw_source}" | awk '{print $1}')" == "${SOURCE_SHA}" ]] \
    || fail "Pinned raw model9200 SHA-256 mismatch."

# This validator reloads both checkpoints on CPU and independently reconstructs
# every migrated tensor. The runner performs its own marker validation again
# while loading, closing the path between this preflight and PPO resume.
uv run --locked python -m \
    mjlab_microban.scripts.teleop_v12_lr_recovery validate \
    "${recovery_receipt}" \
    --source "${raw_source}" \
    --checkpoint "${migrated_checkpoint}" \
    --migration-receipt "${migration_receipt}" >/dev/null

echo "[INFO] authenticated v12 L/R recovery completed=9201 target=10000 process_updates=${PROCESS_UPDATES}"
exec uv run --locked train Mjlab-Teleop-V12-Microban \
    --env.scene.num-envs 2048 --env.seed 42 --agent.seed 42 \
    --agent.num-steps-per-env 24 --agent.max-iterations "${PROCESS_UPDATES}" \
    --agent.save-interval 100 --agent.logger tensorboard \
    --agent.upload-model False --enable-nan-guard True \
    --agent.resume True \
    --agent.load-run "^${seed_run}$" \
    --agent.load-checkpoint "^model_${SOURCE_ITERATION}[.]pt$" \
    --agent.run-name "${output_run_name}"
