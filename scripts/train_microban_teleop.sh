#!/usr/bin/env bash
# Reproducible smoke/train/resume entry point for the PICO hybrid policy.
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

usage() {
    cat <<'EOF'
Usage:
  scripts/train_microban_teleop.sh smoke
  scripts/train_microban_teleop.sh train [additional train options]
  scripts/train_microban_teleop.sh resume RUN_NAME [additional train options]

Environment overrides:
  MICROBAN_TELEOP_NUM_ENVS      Parallel environments (default: 4096)
  MICROBAN_TELEOP_TARGET_ITERS  Target total completed PPO iterations (default: 15000)
  MICROBAN_TELEOP_SAVE_INTERVAL Checkpoint interval in iterations (default: 500)
  MICROBAN_TELEOP_SEED          Environment/agent seed (default: 42)

RUN_NAME is the timestamp directory below
logs/rsl_rl/mjlab_microban_teleop/, for example 2026-09-24_12-34-56.
EOF
}

if ! command -v uv >/dev/null 2>&1; then
    echo "Required command not found: uv" >&2
    exit 1
fi

mode="${1:-}"
run_name=""
case "${mode}" in
    -h|--help|"")
        usage
        exit 0
        ;;
    smoke)
        if (( $# != 1 )); then
            usage >&2
            exit 2
        fi
        cd -- "${PROJECT_ROOT}"
        exec uv run --locked python -m mjlab_microban.scripts.smoke_teleop_task \
            --device cuda:0 --num-envs 4 --steps 5
        ;;
    train)
        shift
        ;;
    resume)
        if (( $# < 2 )); then
            usage >&2
            exit 2
        fi
        run_name="$2"
        if [[ ! "${run_name}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
            echo "RUN_NAME must be one literal run directory name using only letters, digits, '_' and '-'." >&2
            exit 2
        fi
        shift 2
        ;;
    *)
        echo "Unknown mode: ${mode}" >&2
        usage >&2
        exit 2
        ;;
esac

readonly NUM_ENVS="${MICROBAN_TELEOP_NUM_ENVS:-4096}"
if [[ -n "${MICROBAN_TELEOP_MAX_ITERS+x}" ]]; then
    if [[ -n "${MICROBAN_TELEOP_TARGET_ITERS+x}" ]]; then
        echo "Set only MICROBAN_TELEOP_TARGET_ITERS; do not also set the deprecated MICROBAN_TELEOP_MAX_ITERS." >&2
        exit 2
    fi
    echo "[WARN] MICROBAN_TELEOP_MAX_ITERS is deprecated; use MICROBAN_TELEOP_TARGET_ITERS." >&2
fi
readonly TARGET_ITERS="${MICROBAN_TELEOP_TARGET_ITERS:-${MICROBAN_TELEOP_MAX_ITERS:-15000}}"
readonly SAVE_INTERVAL="${MICROBAN_TELEOP_SAVE_INTERVAL:-500}"
readonly SEED="${MICROBAN_TELEOP_SEED:-42}"

for numeric_value in "${NUM_ENVS}" "${TARGET_ITERS}" "${SAVE_INTERVAL}" "${SEED}"; do
    if [[ ! "${numeric_value}" =~ ^[0-9]+$ ]]; then
        echo "Environment overrides must be non-negative integers." >&2
        exit 2
    fi
done
if (( NUM_ENVS < 1 || TARGET_ITERS < 1 || SAVE_INTERVAL < 1 )); then
    echo "NUM_ENVS, TARGET_ITERS and SAVE_INTERVAL must be positive." >&2
    exit 2
fi
if ((
    NUM_ENVS > 2147483647
    || TARGET_ITERS > 2147483647
    || SAVE_INTERVAL > 2147483647
    || SEED > 2147483647
)); then
    echo "Environment overrides must not exceed 2147483647." >&2
    exit 2
fi

# These options define the wrapper's reproducibility contract.  Accepting a
# second copy through additional arguments would make the effective resume
# target depend on Tyro's duplicate-option behavior.
for option in "$@"; do
    case "${option}" in
        --agent.max-iterations|--agent.max-iterations=*|\
        --agent.save-interval|--agent.save-interval=*|\
        --agent.resume|--agent.resume=*|\
        --agent.load-run|--agent.load-run=*|\
        --agent.load-checkpoint|--agent.load-checkpoint=*)
            echo "${option} is controlled by this wrapper; use its mode/environment overrides instead." >&2
            exit 2
            ;;
    esac
done

if [[ "${mode}" == "resume" ]]; then
    for option in "$@"; do
        case "${option}" in
            --agent.bootstrap-velocity-checkpoint|--agent.bootstrap-velocity-checkpoint=*|\
            --agent.bootstrap-velocity-checkpoint-sha256|--agent.bootstrap-velocity-checkpoint-sha256=*)
                echo "Velocity actor bootstrap is fresh-run-only and cannot be used with resume." >&2
                exit 2
                ;;
        esac
    done
fi

iterations_to_run="${TARGET_ITERS}"
resume_args=()
if [[ "${mode}" == "resume" ]]; then
    readonly LOG_ROOT="${PROJECT_ROOT}/logs/rsl_rl/mjlab_microban_teleop"
    readonly RUN_DIR="${LOG_ROOT}/${run_name}"
    if [[ ! -d "${RUN_DIR}" ]]; then
        echo "Resume run does not exist: ${RUN_DIR}" >&2
        exit 2
    fi

    checkpoint_iteration=-1
    checkpoint_name=""
    shopt -s nullglob
    checkpoint_paths=("${RUN_DIR}"/model_*.pt)
    shopt -u nullglob
    for checkpoint_path in "${checkpoint_paths[@]}"; do
        candidate_name="${checkpoint_path##*/}"
        if [[ "${candidate_name}" =~ ^model_([0-9]+)\.pt$ ]]; then
            candidate_iteration="${BASH_REMATCH[1]}"
            # Force base ten so zero-padded names cannot be parsed as octal.
            candidate_iteration=$((10#${candidate_iteration}))
            if (( candidate_iteration > checkpoint_iteration )); then
                checkpoint_iteration="${candidate_iteration}"
                checkpoint_name="${candidate_name}"
            fi
        fi
    done
    if (( checkpoint_iteration < 0 )); then
        echo "No numeric model_<iteration>.pt checkpoint found in ${RUN_DIR}." >&2
        exit 2
    fi

    # Checkpoint suffixes are zero-based completed iteration indices: model_0
    # exists only after the first PPO update has completed.
    completed_iterations=$((checkpoint_iteration + 1))
    if (( completed_iterations >= TARGET_ITERS )); then
        echo "Resume target already reached: ${completed_iterations} completed, target ${TARGET_ITERS}." >&2
        exit 3
    fi
    iterations_to_run=$((TARGET_ITERS - completed_iterations))
    resume_args=(
        --agent.resume True
        --agent.load-run "^${run_name}$"
        --agent.load-checkpoint "^model_${checkpoint_iteration}[.]pt$"
    )
    echo "[INFO] Resume source: ${RUN_DIR}/${checkpoint_name}"
    echo "[INFO] Completed: ${completed_iterations}; target: ${TARGET_ITERS}; remaining: ${iterations_to_run}"
else
    echo "[INFO] New run target: ${TARGET_ITERS} completed PPO iterations"
fi

cd -- "${PROJECT_ROOT}"
exec uv run --locked train Mjlab-Teleop-Microban \
    --env.scene.num-envs "${NUM_ENVS}" \
    --env.seed "${SEED}" \
    --agent.seed "${SEED}" \
    --agent.max-iterations "${iterations_to_run}" \
    --agent.save-interval "${SAVE_INTERVAL}" \
    --agent.logger tensorboard \
    --agent.upload-model False \
    --enable-nan-guard True \
    "${resume_args[@]}" \
    "$@"
