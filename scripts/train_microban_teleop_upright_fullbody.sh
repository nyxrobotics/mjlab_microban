#!/usr/bin/env bash
# Fresh, independent 83-input/18-output teleop training at physical neutral HOME.
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly EXPERIMENT="mjlab_microban_teleop_upright_fullbody"
readonly TASK="Mjlab-Teleop-Upright-Fullbody-Microban"
readonly LOG_ROOT="${PROJECT_ROOT}/logs/rsl_rl/${EXPERIMENT}"

usage() {
    cat <<'EOF'
Usage:
  scripts/train_microban_teleop_upright_fullbody.sh start RUN_LABEL [UPDATES]
  scripts/train_microban_teleop_upright_fullbody.sh resume SOURCE_RUN CHECKPOINT_ITERATION UPDATES RUN_LABEL

This is a new policy trained from scratch at the physical A-button neutral:
bilateral hip pitch +1.198384259489 degrees, ankle pitch -1.198384259489
degrees, shoulder pitch 0 degrees, and a vertical trunk. It does not
load the old v12 checkpoint or walk004 motion prior.
UPDATES defaults to 15000 for start. Use multiples of 100 so the last update
is written to a checkpoint. Resume names an existing run directory under the
separate upright_fullbody experiment and creates a newly named output run.
EOF
}
fail() { echo "$*" >&2; exit 2; }

[[ $# -ge 1 ]] || { usage; exit 2; }
mode=$1
case "$mode" in
    -h|--help) usage; exit 0 ;;
    start)
        [[ $# -eq 2 || $# -eq 3 ]] || fail "Usage: $0 start RUN_LABEL [UPDATES]"
        run_label=$2
        updates=${3:-15000}
        runner_args=()
        ;;
    resume)
        [[ $# -eq 5 ]] || fail "Usage: $0 resume SOURCE_RUN CHECKPOINT_ITERATION UPDATES RUN_LABEL"
        source_run=$2
        iteration=$3
        updates=$4
        run_label=$5
        [[ "$source_run" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] \
            || fail "SOURCE_RUN must be one safe run-directory name."
        [[ "$iteration" =~ ^(0|[1-9][0-9]*)$ ]] \
            || fail "CHECKPOINT_ITERATION must be a non-negative integer."
        checkpoint="${LOG_ROOT}/${source_run}/model_${iteration}.pt"
        [[ -f "$checkpoint" && ! -L "$checkpoint" ]] \
            || fail "Upright full-body checkpoint is missing: $checkpoint"
        runner_args=(
            --agent.resume True
            --agent.load-run "^${source_run}$"
            --agent.load-checkpoint "^model_${iteration}[.]pt$"
        )
        ;;
    *) usage; fail "Unknown mode: $mode" ;;
esac

[[ "$run_label" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] \
    || fail "RUN_LABEL must contain only letters, digits, underscores or hyphens."
[[ "$updates" =~ ^[1-9][0-9]*$ ]] || fail "UPDATES must be a positive integer."
(( updates % 100 == 0 )) || fail "UPDATES must be a multiple of 100."
completed=0
if [[ "$mode" == resume ]]; then
    completed=$((iteration + 1))
fi
(( completed + updates <= 15000 )) \
    || fail "Requested training would exceed 15000 total updates."

cd -- "$PROJECT_ROOT"
echo "[INFO] task=$TASK HOME hip pitch=+1.198384259489 deg ankle pitch=-1.198384259489 deg shoulder pitch=0 deg trunk pitch=0 deg mode=$mode" >&2
echo "[INFO] completed=$completed updates=$updates output_label=$run_label" >&2
echo "[INFO] This training route has no release gate; checkpoints are simulation-only." >&2
exec uv run --locked train "$TASK" \
    --env.scene.num-envs 2048 --env.seed 42 --agent.seed 42 \
    --agent.num-steps-per-env 24 --agent.max-iterations "$updates" \
    --agent.save-interval 100 --agent.logger tensorboard \
    --agent.upload-model False --enable-nan-guard True \
    "${runner_args[@]}" --agent.run-name "$run_label"
