#!/usr/bin/env bash
# Reproducible launcher for the isolated bounded Microban velocity policy.
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly EXPERIMENT_NAME="mjlab_microban_safe_velocity"

usage() {
    cat <<'EOF'
Usage:
  scripts/train_microban_safe_velocity.sh smoke
  scripts/train_microban_safe_velocity.sh canary
  scripts/train_microban_safe_velocity.sh reward-canary
  scripts/train_microban_safe_velocity.sh resume RUN_NAME [UPDATES]
  scripts/train_microban_safe_velocity.sh evaluate CHECKPOINT [OUTPUT_JSON]

smoke    64 environments, one PPO update.
canary   2048 environments, 51 PPO updates, checkpoints 0/25/50.
reward-canary  2048 environments, 501 PPO updates for the v9 body-progress signal.
resume   Resume the latest numeric checkpoint (default 51 additional updates).
evaluate Run the deterministic 64-env, 200-step, +0.08 m/s gate.

The canary intentionally uses 2048 rather than 4096 environments.  With the
required nconmax=512/njmax=2048 safety capacity, 4096 environments exceed a
24 GiB RTX 4090 during MuJoCo-Warp graph construction.
EOF
}

fail() {
    echo "$*" >&2
    exit 2
}

run_train() {
    local num_envs="$1"
    local iterations="$2"
    local save_interval="$3"
    local run_name="$4"
    shift 4
    cd -- "${PROJECT_ROOT}"
    exec uv run --locked train Mjlab-SafeVelocity-Microban \
        --env.scene.num-envs "${num_envs}" \
        --env.seed 42 \
        --agent.seed 42 \
        --agent.max-iterations "${iterations}" \
        --agent.save-interval "${save_interval}" \
        --agent.run-name "${run_name}" \
        --agent.logger tensorboard \
        --agent.upload-model False \
        --enable-nan-guard True \
        "$@"
}

mode="${1:-}"
case "${mode}" in
    smoke)
        (( $# == 1 )) || fail "smoke takes no additional arguments"
        run_train 64 1 1 safe_velocity_64x1_smoke
        ;;
    canary)
        (( $# == 1 )) || fail "canary takes no additional arguments"
        run_train 2048 51 25 safe_velocity_2048_51canary
        ;;
    reward-canary)
        (( $# == 1 )) || fail "reward-canary takes no additional arguments"
        run_train 2048 501 25 safe_velocity_v9_2048_501canary
        ;;
    resume)
        (( $# == 2 || $# == 3 )) \
            || fail "resume requires RUN_NAME [UPDATES]"
        run_name="$2"
        updates="${3:-51}"
        [[ "${run_name}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] \
            || fail "RUN_NAME contains unsafe characters"
        [[ "${updates}" =~ ^[1-9][0-9]*$ ]] \
            || fail "UPDATES must be a positive integer"
        (( updates <= 2000 )) || fail "UPDATES must be <= 2000"
        run_dir="${PROJECT_ROOT}/logs/rsl_rl/${EXPERIMENT_NAME}/${run_name}"
        [[ -d "${run_dir}" ]] || fail "Run directory does not exist: ${run_dir}"
        latest_iteration=-1
        latest_name=""
        shopt -s nullglob
        candidates=("${run_dir}"/model_*.pt)
        shopt -u nullglob
        for checkpoint in "${candidates[@]}"; do
            name="${checkpoint##*/}"
            if [[ "${name}" =~ ^model_([0-9]+)[.]pt$ ]]; then
                iteration=$((10#${BASH_REMATCH[1]}))
                if (( iteration > latest_iteration )); then
                    latest_iteration="${iteration}"
                    latest_name="${name}"
                fi
            fi
        done
        (( latest_iteration >= 0 )) || fail "No numeric checkpoint in ${run_dir}"
        echo "[INFO] Resume source: ${run_dir}/${latest_name}"
        echo "[INFO] Adds ${updates} PPO updates; checkpoint env_state restores curriculum steps."
        run_train 2048 "${updates}" 25 "safe_velocity_2048_resume_${updates}" \
            --agent.resume True \
            --agent.load-run "^${run_name}$" \
            --agent.load-checkpoint "^model_${latest_iteration}[.]pt$"
        ;;
    evaluate)
        (( $# == 2 || $# == 3 )) || fail "evaluate requires CHECKPOINT [OUTPUT_JSON]"
        checkpoint="$2"
        output="${3:-${PROJECT_ROOT}/artifacts/microban_safe_velocity_checkpoint_gate.json}"
        cd -- "${PROJECT_ROOT}"
        exec uv run --locked python -m \
            mjlab_microban.scripts.evaluate_safe_velocity_checkpoint \
            --checkpoint "${checkpoint}" \
            --device cuda:0 \
            --num-envs 64 \
            --steps 200 \
            --command-vx 0.08 \
            --seed 42 \
            --output "${output}"
        ;;
    -h|--help|"")
        usage
        ;;
    *)
        fail "Unknown mode: ${mode}"
        ;;
esac
