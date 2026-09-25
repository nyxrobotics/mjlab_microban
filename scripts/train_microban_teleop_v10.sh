#!/usr/bin/env bash
# Fail-closed canonical contract-v10 migration/stage driver.
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly LOG_ROOT="${PROJECT_ROOT}/logs/rsl_rl/mjlab_microban_teleop"
readonly GATE_ROOT="${PROJECT_ROOT}/artifacts/teleop_v10_gates"
readonly LEGACY_GATE_ROOT="${PROJECT_ROOT}/artifacts/teleop_v9_gates"
readonly CANARY_ROOT="${PROJECT_ROOT}/artifacts/teleop_v10_canaries"
readonly STAGE_BOUNDARIES=(3000 7000 10000 15000)

usage() {
    cat <<'EOF'
Usage:
  scripts/train_microban_teleop_v10.sh migrate LEGACY_MODEL_1499_PT LEGACY_GATE_JSON [--canary] [--agent.run-name NAME]
  scripts/train_microban_teleop_v10.sh resume RUN_NAME [--canary] [--agent.run-name NAME]

The only v10 entry is the exact accepted contract-v9 model_1499.pt/gate pair.
Migration advances 1500->3000; later stages are 3000->7000->10000->15000.
--canary runs exactly 100 PPO updates, writes an aligned checkpoint, and stops.
Evaluate that checkpoint with evaluate_microban_teleop_v10_stage.sh --canary
before resuming it. Without --canary, the process runs to the stage boundary.
Every invocation writes a new timestamp-prefixed run directory; pass that full
directory name to the evaluator and to the next resume invocation.

Canonical settings are fixed at 2048 environments, seed 42, 24 rollout steps,
save interval 100, PPO learning rate 1e-5, and schedule=fixed.
EOF
}

fail() {
    echo "$*" >&2
    exit 2
}

command -v uv >/dev/null 2>&1 || fail "Required command not found: uv"
command -v sha256sum >/dev/null 2>&1 || fail "Required command not found: sha256sum"

for owned_name in \
    MICROBAN_TELEOP_TARGET_ITERS \
    MICROBAN_TELEOP_MAX_ITERS \
    MICROBAN_TELEOP_PROVENANCE_MODE \
    MICROBAN_TELEOP_STAGE_START_BOUNDARY \
    MICROBAN_TELEOP_STAGE_TARGET_BOUNDARY \
    MICROBAN_TELEOP_PARENT_CHECKPOINT_SHA256 \
    MICROBAN_TELEOP_PARENT_GATE_SHA256 \
    MICROBAN_TELEOP_RESUME_SOURCE_CHECKPOINT_PATH \
    MICROBAN_TELEOP_RESUME_SOURCE_CHECKPOINT_SHA256 \
    MICROBAN_TELEOP_RESUME_SOURCE_CHECKPOINT_ITERATION; do
    [[ -z "${!owned_name+x}" ]] \
        || fail "The canonical v10 stage driver owns ${owned_name}; unset it."
done
[[ "${MICROBAN_TELEOP_NUM_ENVS:-2048}" == "2048" ]] \
    || fail "Canonical v10 requires MICROBAN_TELEOP_NUM_ENVS=2048."
[[ "${MICROBAN_TELEOP_SEED:-42}" == "42" ]] \
    || fail "Canonical v10 requires MICROBAN_TELEOP_SEED=42."
[[ "${MICROBAN_TELEOP_SAVE_INTERVAL:-100}" == "100" ]] \
    || fail "Canonical v10 requires MICROBAN_TELEOP_SAVE_INTERVAL=100."

mode="${1:-}"
source_checkpoint=""
source_gate=""
run_name=""
case "${mode}" in
    -h|--help|"") usage; exit 0 ;;
    migrate)
        (( $# >= 3 )) || { usage >&2; exit 2; }
        source_checkpoint="$2"
        source_gate="$3"
        shift 3
        ;;
    resume)
        (( $# >= 2 )) || { usage >&2; exit 2; }
        run_name="$2"
        [[ "${run_name}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] \
            || fail "RUN_NAME must be one literal run directory name."
        shift 2
        ;;
    *) fail "Unknown mode: ${mode}" ;;
esac

canary=0
output_run_name_seen=0
additional_args=()
while (( $# > 0 )); do
    case "$1" in
        --canary)
            (( canary == 0 )) || fail "--canary may be specified only once."
            canary=1
            shift
            ;;
        --agent.run-name)
            (( $# >= 2 )) || fail "--agent.run-name requires a value."
            [[ -n "$2" && "$2" != --* ]] \
                || fail "--agent.run-name requires a value."
            (( output_run_name_seen == 0 )) \
                || fail "--agent.run-name may be specified only once."
            [[ "$2" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] \
                || fail "--agent.run-name must be a safe literal suffix."
            output_run_name_seen=1
            additional_args+=("$1" "$2")
            shift 2
            ;;
        --agent.run-name=*)
            output_run_name="${1#*=}"
            (( output_run_name_seen == 0 )) \
                || fail "--agent.run-name may be specified only once."
            [[ "${output_run_name}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] \
                || fail "--agent.run-name must be a safe literal suffix."
            output_run_name_seen=1
            additional_args+=("$1")
            shift
            ;;
        *)
            fail "Unsupported canonical v10 override: $1 (only --canary and --agent.run-name are allowed)."
            ;;
    esac
done

target_boundary=3000
stage_start_boundary=1500
completed_iterations=1500
parent_checkpoint_sha256=""
parent_gate_sha256=""
resume_source_checkpoint_path=""
resume_source_checkpoint_sha256=""
resume_source_checkpoint_iteration=""
runner_mode_args=()

cd -- "${PROJECT_ROOT}"
if [[ "${mode}" == "migrate" ]]; then
    [[ -f "${source_checkpoint}" ]] \
        || fail "Legacy migration checkpoint not found: ${source_checkpoint}"
    [[ -f "${source_gate}" ]] \
        || fail "Legacy migration gate not found: ${source_gate}"
    source_checkpoint="$(realpath --canonicalize-existing -- "${source_checkpoint}")"
    source_gate="$(realpath --canonicalize-existing -- "${source_gate}")"
    mapfile -t migration_metadata < <(
        uv run --locked python -m mjlab_microban.scripts.teleop_v10_stage \
            validate-migration "${source_checkpoint}" "${source_gate}"
    )
    (( ${#migration_metadata[@]} == 4 )) \
        || fail "Migration validator returned malformed metadata."
    [[ "${migration_metadata[0]}" == "${source_checkpoint}" ]] \
        || fail "Migration checkpoint changed during validation."
    [[ "${migration_metadata[2]}" == "${source_gate}" ]] \
        || fail "Migration gate changed during validation."
    parent_checkpoint_sha256="${migration_metadata[1]}"
    parent_gate_sha256="${migration_metadata[3]}"
    resume_source_checkpoint_path="${source_checkpoint}"
    resume_source_checkpoint_sha256="${parent_checkpoint_sha256}"
    resume_source_checkpoint_iteration=1499

    source_run_dir="$(dirname -- "${source_checkpoint}")"
    [[ "$(dirname -- "${source_run_dir}")" == "${LOG_ROOT}" ]] \
        || fail "Migration checkpoint must be directly under ${LOG_ROOT}/RUN_NAME."
    source_run_name="$(basename -- "${source_run_dir}")"
    [[ "${source_run_name}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] \
        || fail "Legacy source run directory name is not loadable."
    runner_mode_args=(
        --agent.resume True
        --agent.load-run "^${source_run_name}$"
        --agent.load-checkpoint '^model_1499[.]pt$'
    )
else
    readonly RUN_DIR="${LOG_ROOT}/${run_name}"
    [[ -d "${RUN_DIR}" ]] || fail "Resume run does not exist: ${RUN_DIR}"
    checkpoint_iteration=-1
    shopt -s nullglob
    checkpoint_paths=("${RUN_DIR}"/model_*.pt)
    shopt -u nullglob
    for checkpoint_path in "${checkpoint_paths[@]}"; do
        checkpoint_name="${checkpoint_path##*/}"
        if [[ "${checkpoint_name}" =~ ^model_([0-9]+)[.]pt$ ]]; then
            candidate_iteration=$((10#${BASH_REMATCH[1]}))
            if (( candidate_iteration > checkpoint_iteration )); then
                checkpoint_iteration="${candidate_iteration}"
            fi
        fi
    done
    (( checkpoint_iteration >= 0 )) \
        || fail "No numeric model_<iteration>.pt checkpoint found in ${RUN_DIR}."
    completed_iterations=$((checkpoint_iteration + 1))
    (( completed_iterations < 15000 )) \
        || fail "The final 15,000-update boundary is already reached; evaluate/export it."
    readonly CHECKPOINT_PATH="${RUN_DIR}/model_${checkpoint_iteration}.pt"

    boundary_index=-1
    interrupted_stage=0
    previous_boundary=1500
    for index in "${!STAGE_BOUNDARIES[@]}"; do
        if (( STAGE_BOUNDARIES[index] == completed_iterations )); then
            boundary_index="${index}"
            break
        fi
        if (( completed_iterations < STAGE_BOUNDARIES[index] )); then
            interrupted_stage=1
            stage_start_boundary="${previous_boundary}"
            target_boundary="${STAGE_BOUNDARIES[index]}"
            break
        fi
        previous_boundary="${STAGE_BOUNDARIES[index]}"
    done

    if (( interrupted_stage == 1 )); then
        set +e
        metadata_output="$(
            uv run --locked python -m mjlab_microban.scripts.teleop_v10_stage \
                validate-interrupted "${CHECKPOINT_PATH}" \
                --completed-iterations "${completed_iterations}" \
                --stage-start-boundary "${stage_start_boundary}" \
                --stage-target-boundary "${target_boundary}" \
                --gate-root "${GATE_ROOT}" \
                --legacy-gate-root "${LEGACY_GATE_ROOT}" \
                --canary-root "${CANARY_ROOT}"
        )"
        metadata_status=$?
        set -e
        (( metadata_status == 0 )) \
            || fail "Interrupted v10 stage/canary validation failed."
        mapfile -t stage_metadata <<<"${metadata_output}"
        (( ${#stage_metadata[@]} == 4 )) \
            || fail "Interrupted v10 stage validator returned malformed metadata."
        [[ "${stage_metadata[0]}" == "${stage_start_boundary}" ]] \
            || fail "Interrupted v10 stage start changed during validation."
        [[ "${stage_metadata[1]}" == "${target_boundary}" ]] \
            || fail "Interrupted v10 stage target changed during validation."
        parent_checkpoint_sha256="${stage_metadata[2]}"
        parent_gate_sha256="${stage_metadata[3]}"
    else
        (( boundary_index >= 0 && boundary_index + 1 < ${#STAGE_BOUNDARIES[@]} )) \
            || fail "Latest checkpoint does not identify a resumable v10 boundary."
        readonly GATE_PATH="${GATE_ROOT}/${run_name}_boundary_${completed_iterations}_gate.json"
        [[ -f "${GATE_PATH}" ]] \
            || fail "Missing gate receipt: ${GATE_PATH}. Run evaluate_microban_teleop_v10_stage.sh ${run_name} first."
        uv run --locked python - "${GATE_PATH}" "${CHECKPOINT_PATH}" \
            "${completed_iterations}" <<'PY'
import sys
from pathlib import Path

from mjlab_microban.scripts.teleop_v10_stage import validate_stage_gate
from mjlab_microban.tasks.microban_teleop_provenance import sha256_file

gate = Path(sys.argv[1]).resolve()
checkpoint = Path(sys.argv[2]).resolve()
boundary = int(sys.argv[3])
validate_stage_gate(
    gate,
    expected_boundary=boundary,
    expected_checkpoint_sha256=sha256_file(checkpoint),
)
print(f"[PASS] verified v10 boundary {boundary}: {gate}")
PY
        stage_start_boundary="${completed_iterations}"
        target_boundary="${STAGE_BOUNDARIES[boundary_index + 1]}"
        parent_checkpoint_sha256="$(sha256sum -- "${CHECKPOINT_PATH}" | awk '{print $1}')"
        parent_gate_sha256="$(sha256sum -- "${GATE_PATH}" | awk '{print $1}')"
    fi
    resume_source_checkpoint_path="$(realpath --canonicalize-existing -- "${CHECKPOINT_PATH}")"
    resume_source_checkpoint_sha256="$(sha256sum -- "${resume_source_checkpoint_path}" | awk '{print $1}')"
    resume_source_checkpoint_iteration="${checkpoint_iteration}"
    runner_mode_args=(
        --agent.resume True
        --agent.load-run "^${run_name}$"
        --agent.load-checkpoint "^model_${checkpoint_iteration}[.]pt$"
    )
fi

remaining_iterations=$((target_boundary - completed_iterations))
(( remaining_iterations > 0 )) || fail "Stage has no remaining updates."
iterations_to_run="${remaining_iterations}"
if (( canary == 1 && remaining_iterations <= 100 )); then
    fail "At most 100 updates remain; use a normal boundary run, not --canary."
fi
if (( canary == 1 )); then
    iterations_to_run=100
fi

if (( stage_start_boundary == 1500 )); then
    provenance_mode="canonical_v10_migration_stage"
else
    provenance_mode="canonical_v10_stage"
fi
export MICROBAN_TELEOP_TARGET_ITERS="${target_boundary}"
export MICROBAN_TELEOP_NUM_ENVS=2048
export MICROBAN_TELEOP_SEED=42
export MICROBAN_TELEOP_SAVE_INTERVAL=100
export MICROBAN_TELEOP_PROVENANCE_MODE="${provenance_mode}"
export MICROBAN_TELEOP_STAGE_START_BOUNDARY="${stage_start_boundary}"
export MICROBAN_TELEOP_STAGE_TARGET_BOUNDARY="${target_boundary}"
export MICROBAN_TELEOP_PARENT_CHECKPOINT_SHA256="${parent_checkpoint_sha256}"
export MICROBAN_TELEOP_PARENT_GATE_SHA256="${parent_gate_sha256}"
export MICROBAN_TELEOP_RESUME_SOURCE_CHECKPOINT_PATH="${resume_source_checkpoint_path}"
export MICROBAN_TELEOP_RESUME_SOURCE_CHECKPOINT_SHA256="${resume_source_checkpoint_sha256}"
export MICROBAN_TELEOP_RESUME_SOURCE_CHECKPOINT_ITERATION="${resume_source_checkpoint_iteration}"
echo "[INFO] Contract-v10 ${provenance_mode} ${stage_start_boundary}->${target_boundary}; completed=${completed_iterations}; process_updates=${iterations_to_run}"

exec uv run --locked train Mjlab-Teleop-Microban \
    --env.scene.num-envs 2048 \
    --env.seed 42 \
    --agent.seed 42 \
    --agent.num-steps-per-env 24 \
    --agent.max-iterations "${iterations_to_run}" \
    --agent.save-interval 100 \
    --agent.logger tensorboard \
    --agent.upload-model False \
    --agent.algorithm.learning-rate 0.00001 \
    --agent.algorithm.schedule fixed \
    --enable-nan-guard True \
    "${runner_mode_args[@]}" \
    "${additional_args[@]}"
