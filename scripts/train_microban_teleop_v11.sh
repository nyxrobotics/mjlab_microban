#!/usr/bin/env bash
# Fail-closed canonical contract-v11 fresh-bootstrap/stage driver.
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly LOG_ROOT="${PROJECT_ROOT}/logs/rsl_rl/mjlab_microban_teleop"
readonly GATE_ROOT="${PROJECT_ROOT}/artifacts/teleop_v11_gates"
readonly CANARY_ROOT="${PROJECT_ROOT}/artifacts/teleop_v11_canaries"
readonly SAFE_CHECKPOINT="${PROJECT_ROOT}/logs/rsl_rl/mjlab_microban_safe_velocity/2026-09-25_09-27-27_safe_velocity_v9_2048_501canary/model_500.pt"
readonly SAFE_CHECKPOINT_SHA256="416a8b16f7f7980822e4e1df81ffaf9515bc18a246e6fc257405a2c46ceece93"
readonly SAFE_RECEIPT="${PROJECT_ROOT}/artifacts/microban_safe_velocity_v9_model500_v3_pass.json"
readonly SAFE_RECEIPT_SHA256="e68701b11774dd30c8e45a2fd89614a2e4423a9486d01a0d936f0fa6fb760492"
readonly STAGE_BOUNDARIES=(3000 7000 10000 15000)

usage() {
    cat <<'EOF'
Usage:
  scripts/train_microban_teleop_v11.sh start [--canary] [--agent.run-name NAME]
  scripts/train_microban_teleop_v11.sh resume RUN_NAME [--canary] [--agent.run-name NAME]

start creates a fresh contract-v11 PPO state, maps only the actor/distribution
from the pinned accepted safe-velocity model_500.pt, and saves
model_pristine.pt before update zero. It never imports a teleop critic, Adam
state, iteration, or environment step counter.

Stages are 0->3000->7000->10000->15000. --canary runs exactly 100 updates and
stops at an aligned checkpoint. Before resuming an interrupted run, evaluate it
with evaluate_microban_teleop_v11_stage.sh --canary RUN_NAME. Exact boundary
resumes require the corresponding three-seed boundary gate.

Canonical settings are fixed at 2048 environments, seed 42, 24 rollout steps,
save interval 100, PPO learning rate 1e-4, fixed schedule, entropy coefficient
0.005, and five learning epochs.
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
        || fail "The canonical v11 stage driver owns ${owned_name}; unset it."
done
[[ "${MICROBAN_TELEOP_NUM_ENVS:-2048}" == "2048" ]] \
    || fail "Canonical v11 requires MICROBAN_TELEOP_NUM_ENVS=2048."
[[ "${MICROBAN_TELEOP_SEED:-42}" == "42" ]] \
    || fail "Canonical v11 requires MICROBAN_TELEOP_SEED=42."
[[ "${MICROBAN_TELEOP_SAVE_INTERVAL:-100}" == "100" ]] \
    || fail "Canonical v11 requires MICROBAN_TELEOP_SAVE_INTERVAL=100."

mode="${1:-}"
run_name=""
case "${mode}" in
    -h|--help|"") usage; exit 0 ;;
    start) shift ;;
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
            fail "Unsupported canonical v11 override: $1 (only --canary and --agent.run-name are allowed)."
            ;;
    esac
done

cd -- "${PROJECT_ROOT}"

stage_start_boundary=0
target_boundary=3000
completed_iterations=0
parent_checkpoint_sha256=""
parent_gate_sha256=""
resume_source_checkpoint_path=""
resume_source_checkpoint_sha256=""
resume_source_checkpoint_iteration=""
runner_mode_args=()

if [[ "${mode}" == "start" ]]; then
    [[ -f "${SAFE_CHECKPOINT}" ]] \
        || fail "Pinned safe-velocity checkpoint not found: ${SAFE_CHECKPOINT}"
    [[ -f "${SAFE_RECEIPT}" ]] \
        || fail "Pinned safe-velocity acceptance receipt not found: ${SAFE_RECEIPT}"
    [[ "$(sha256sum -- "${SAFE_CHECKPOINT}" | awk '{print $1}')" == "${SAFE_CHECKPOINT_SHA256}" ]] \
        || fail "Pinned safe-velocity checkpoint SHA-256 mismatch."
    [[ "$(sha256sum -- "${SAFE_RECEIPT}" | awk '{print $1}')" == "${SAFE_RECEIPT_SHA256}" ]] \
        || fail "Pinned safe-velocity acceptance receipt SHA-256 mismatch."
    uv run --locked python -m mjlab_microban.scripts.teleop_v11_stage \
        validate-bootstrap "${SAFE_CHECKPOINT}" "${SAFE_RECEIPT}"
    runner_mode_args=(
        --agent.safe-velocity-checkpoint "${SAFE_CHECKPOINT}"
        --agent.safe-velocity-checkpoint-sha256 "${SAFE_CHECKPOINT_SHA256}"
        --agent.safe-velocity-acceptance-receipt "${SAFE_RECEIPT}"
        --agent.save-pristine-checkpoint True
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

    mapfile -t interval_metadata < <(
        uv run --locked python -m mjlab_microban.scripts.teleop_v11_stage \
            resolve-interval "${completed_iterations}"
    )
    (( ${#interval_metadata[@]} == 3 )) \
        || fail "Stage interval resolver returned malformed metadata."
    stage_start_boundary="${interval_metadata[0]}"
    target_boundary="${interval_metadata[1]}"
    interrupted_stage="${interval_metadata[2]}"

    if [[ "${interrupted_stage}" == "1" ]]; then
        mapfile -t stage_metadata < <(
            uv run --locked python -m mjlab_microban.scripts.teleop_v11_stage \
                validate-interrupted "${CHECKPOINT_PATH}" \
                --completed-iterations "${completed_iterations}" \
                --stage-start-boundary "${stage_start_boundary}" \
                --stage-target-boundary "${target_boundary}" \
                --gate-root "${GATE_ROOT}" \
                --canary-root "${CANARY_ROOT}"
        )
        (( ${#stage_metadata[@]} == 4 )) \
            || fail "Interrupted v11 stage validator returned malformed metadata."
        [[ "${stage_metadata[0]}" == "${stage_start_boundary}" ]] \
            || fail "Interrupted v11 stage start changed during validation."
        [[ "${stage_metadata[1]}" == "${target_boundary}" ]] \
            || fail "Interrupted v11 stage target changed during validation."
        if (( stage_start_boundary == 0 )); then
            [[ "${stage_metadata[2]}" == "none" && "${stage_metadata[3]}" == "none" ]] \
                || fail "Initial v11 stage unexpectedly claims parent lineage."
            parent_checkpoint_sha256=""
            parent_gate_sha256=""
        else
            parent_checkpoint_sha256="${stage_metadata[2]}"
            parent_gate_sha256="${stage_metadata[3]}"
        fi
    else
        readonly GATE_PATH="${GATE_ROOT}/${run_name}_boundary_${completed_iterations}_gate.json"
        [[ -f "${GATE_PATH}" ]] \
            || fail "Missing gate receipt: ${GATE_PATH}. Run evaluate_microban_teleop_v11_stage.sh ${run_name} first."
        mapfile -t gate_metadata < <(
            uv run --locked python -m mjlab_microban.scripts.teleop_v11_stage \
                validate-boundary "${GATE_PATH}" "${CHECKPOINT_PATH}" \
                --boundary "${completed_iterations}"
        )
        (( ${#gate_metadata[@]} == 2 )) \
            || fail "Boundary gate validator returned malformed metadata."
        parent_checkpoint_sha256="${gate_metadata[0]}"
        parent_gate_sha256="${gate_metadata[1]}"
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

export MICROBAN_TELEOP_TARGET_ITERS="${target_boundary}"
export MICROBAN_TELEOP_NUM_ENVS=2048
export MICROBAN_TELEOP_SEED=42
export MICROBAN_TELEOP_SAVE_INTERVAL=100
export MICROBAN_TELEOP_PROVENANCE_MODE=canonical_v11_stage
export MICROBAN_TELEOP_STAGE_START_BOUNDARY="${stage_start_boundary}"
export MICROBAN_TELEOP_STAGE_TARGET_BOUNDARY="${target_boundary}"
export MICROBAN_TELEOP_PARENT_CHECKPOINT_SHA256="${parent_checkpoint_sha256}"
export MICROBAN_TELEOP_PARENT_GATE_SHA256="${parent_gate_sha256}"
export MICROBAN_TELEOP_RESUME_SOURCE_CHECKPOINT_PATH="${resume_source_checkpoint_path}"
export MICROBAN_TELEOP_RESUME_SOURCE_CHECKPOINT_SHA256="${resume_source_checkpoint_sha256}"
export MICROBAN_TELEOP_RESUME_SOURCE_CHECKPOINT_ITERATION="${resume_source_checkpoint_iteration}"

echo "[INFO] Contract-v11 canonical_v11_stage ${stage_start_boundary}->${target_boundary}; completed=${completed_iterations}; process_updates=${iterations_to_run}"

exec uv run --locked train Mjlab-Teleop-Microban \
    --env.scene.num-envs 2048 \
    --env.seed 42 \
    --agent.seed 42 \
    --agent.num-steps-per-env 24 \
    --agent.max-iterations "${iterations_to_run}" \
    --agent.save-interval 100 \
    --agent.logger tensorboard \
    --agent.upload-model False \
    --agent.algorithm.learning-rate 0.0001 \
    --agent.algorithm.schedule fixed \
    --agent.algorithm.entropy-coef 0.005 \
    --agent.algorithm.num-learning-epochs 5 \
    --enable-nan-guard True \
    "${runner_mode_args[@]}" \
    "${additional_args[@]}"
