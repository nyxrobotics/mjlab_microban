#!/usr/bin/env bash
# Contract-v12 3000/7000/10000/15000 stage driver at the centered HOME with the
# shared +-1.57 rad target clip.  The frozen locomotion actor is a centered-HOME
# Mjlab-Velocity-Microban checkpoint chosen with --source on a fresh start.
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly LOG_ROOT="${PROJECT_ROOT}/logs/rsl_rl/mjlab_microban_teleop_v12"
readonly GATE_ROOT="${PROJECT_ROOT}/artifacts/teleop_v12_gates"
readonly PROBE_ROOT="${PROJECT_ROOT}/artifacts/legacy_teleop_probe"
readonly BOOTSTRAP_ROOT="${PROJECT_ROOT}/artifacts/teleop_v12_bootstrap"

usage() {
    cat <<'EOF'
Usage:
  scripts/train_microban_teleop_v12.sh start --source VELOCITY_MODEL.pt [--source-sha256 SHA]
      [--canary] [--agent.run-name NAME] [--num-envs N] [--max-updates N]
  scripts/train_microban_teleop_v12.sh resume RUN_NAME [--canary] [--agent.run-name NAME]
      [--num-envs N] [--max-updates N] [--dry-run-skip-gate]

Fresh start hashes the centered-HOME velocity checkpoint, runs the 9x300 raw
recurrence probe of it in the teleop task (must pass), publishes the pristine
parity/ONNX bootstrap receipt, and writes model_pristine.pt.  The source path,
its SHA-256 and the probe receipt are recorded in every checkpoint and
re-hashed on every save/resume, so keep the source file in place.
--num-envs (default 2048) and --max-updates (cap on this process's updates)
exist for plumbing dry runs; --dry-run-skip-gate resumes without a stage gate
and must never be used for a release chain. Resume requires a schema-v2 hash-bound locomotion + tracking
+ ONNX gate produced by scripts/evaluate_microban_teleop_v12_stage.sh. Normal
runs stop at 3000/7000/10000/15000. A boundary that activates push, HMD/hand,
or feet is followed by a mandatory 100-update canary and a second gate before
the remaining stage may run. --canary also limits any other segment to 100.
EOF
}
fail() { echo "$*" >&2; exit 2; }

mode="${1:-}"
run_name=""
case "${mode}" in
    -h|--help|"") usage; exit 0 ;;
    start) shift ;;
    resume)
        (( $# >= 2 )) || fail "resume requires RUN_NAME"
        run_name="$2"
        [[ "${run_name}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] \
            || fail "RUN_NAME must be one safe literal directory name."
        shift 2
        ;;
    *) fail "Unknown mode: ${mode}" ;;
esac

canary=0
extra_args=()
source_path=""
source_sha=""
num_envs=2048
max_updates=0
skip_gate=0
while (( $# > 0 )); do
    case "$1" in
        --canary) (( canary == 0 )) || fail "Duplicate --canary"; canary=1; shift ;;
        --source)
            (( $# >= 2 )) || fail "--source requires a checkpoint path"
            source_path="$(realpath -e -- "$2")" || fail "Source not found: $2"
            shift 2
            ;;
        --source-sha256)
            (( $# >= 2 )) && [[ "$2" =~ ^[0-9a-f]{64}$ ]] \
                || fail "--source-sha256 requires a lowercase SHA-256"
            source_sha="$2"; shift 2
            ;;
        --num-envs)
            (( $# >= 2 )) && [[ "$2" =~ ^[1-9][0-9]*$ ]] \
                || fail "--num-envs requires a positive integer"
            num_envs="$2"; shift 2
            ;;
        --max-updates)
            (( $# >= 2 )) && [[ "$2" =~ ^[1-9][0-9]*$ ]] \
                || fail "--max-updates requires a positive integer"
            max_updates="$2"; shift 2
            ;;
        --dry-run-skip-gate) skip_gate=1; shift ;;
        --agent.run-name)
            (( $# >= 2 )) || fail "--agent.run-name requires NAME"
            [[ "$2" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] \
                || fail "Unsafe output run name."
            extra_args+=("$1" "$2"); shift 2
            ;;
        *) fail "Unsupported override: $1" ;;
    esac
done

cd -- "${PROJECT_ROOT}"
completed=0
target=3000
mandatory_activation_canary=0
runner_args=()
save_interval=100
if [[ "${mode}" == "start" ]]; then
    [[ -n "${source_path}" ]] || fail "start requires --source VELOCITY_MODEL.pt"
    (( skip_gate == 0 )) || fail "--dry-run-skip-gate is resume-only"
    actual_sha="$(sha256sum -- "${source_path}" | awk '{print $1}')"
    if [[ -n "${source_sha}" ]]; then
        [[ "${actual_sha}" == "${source_sha}" ]] \
            || fail "Velocity source SHA-256 mismatch: ${actual_sha}"
    fi
    source_sha="${actual_sha}"
    probe="${PROBE_ROOT}/velocity_${source_sha:0:16}_teleop83_raw_9x300.json"
    mkdir -p -- "${PROBE_ROOT}" "${BOOTSTRAP_ROOT}"
    uv run --locked python -m mjlab_microban.scripts.probe_legacy_actor_in_teleop_env \
        --checkpoint "${source_path}" --expected-sha256 "${source_sha}" \
        --output "${probe}" --force
    probe_sha="$(sha256sum -- "${probe}" | awk '{print $1}')"
    # Fails unless the probe passed (9/9 completed, no fall, 8/8 directional).
    uv run --locked --with onnxruntime --with 'protobuf<7' python -m \
        mjlab_microban.scripts.teleop_v12_bootstrap_gate \
        --checkpoint "${source_path}" --checkpoint-sha256 "${source_sha}" \
        --probe-receipt "${probe}" --probe-receipt-sha256 "${probe_sha}" \
        --output-dir "${BOOTSTRAP_ROOT}/velocity_${source_sha:0:16}" \
        --force >/dev/null
    runner_args=(
        --agent.legacy-velocity-checkpoint "${source_path}"
        --agent.legacy-velocity-checkpoint-sha256 "${source_sha}"
        --agent.legacy-teleop-probe-receipt "${probe}"
        --agent.legacy-teleop-probe-receipt-sha256 "${probe_sha}"
        --agent.save-pristine-checkpoint True
    )
else
    [[ -z "${source_path}${source_sha}" ]] \
        || fail "resume reads its velocity source from the checkpoint"
    run_dir="${LOG_ROOT}/${run_name}"
    [[ -d "${run_dir}" ]] || fail "Resume run not found: ${run_dir}"
    iteration=-1
    shopt -s nullglob
    candidates=("${run_dir}"/model_*.pt)
    shopt -u nullglob
    for path in "${candidates[@]}"; do
        name="${path##*/}"
        if [[ "${name}" =~ ^model_([0-9]+)[.]pt$ ]]; then
            value=$((10#${BASH_REMATCH[1]}))
            (( value > iteration )) && iteration="${value}"
        fi
    done
    (( iteration >= 0 )) || fail "No numeric checkpoint found."
    completed=$((iteration + 1))
    (( completed < 15000 )) || fail "Final 15000-update boundary reached."
    checkpoint="${run_dir}/model_${iteration}.pt"
    gate="${GATE_ROOT}/${run_name}_model_${iteration}_gate.json"
    if (( skip_gate == 1 )); then
        echo "[WARN] --dry-run-skip-gate: resuming WITHOUT a stage gate" >&2
        resume_mode=canonical
    else
        [[ -f "${gate}" ]] \
            || fail "Missing gate; run scripts/evaluate_microban_teleop_v12_stage.sh ${run_name} ${iteration}"
        uv run --locked python -m mjlab_microban.scripts.teleop_v12_stage validate \
            "${gate}" "${checkpoint}" >/dev/null
        resume_mode="$(uv run --locked python -m \
            mjlab_microban.scripts.teleop_v12_stage resume-mode \
            "${gate}" "${checkpoint}" --shell)"
    fi
    case "${resume_mode}" in
        canonical) ;;
        deadline_fallback)
            gate_sha="$(sha256sum -- "${gate}" | awk '{print $1}')"
            runner_args+=(
                --agent.deadline-fallback-resume True
                --agent.deadline-fallback-resume-gate "${gate}"
                --agent.deadline-fallback-resume-gate-sha256 "${gate_sha}"
            )
            save_interval=15000
            ;;
        deadline_fallback_post_canary)
            gate_sha="$(sha256sum -- "${gate}" | awk '{print $1}')"
            runner_args+=(
                --agent.deadline-fallback-resume True
                --agent.deadline-fallback-resume-gate "${gate}"
                --agent.deadline-fallback-resume-gate-sha256 "${gate_sha}"
            )
            save_interval=15000
            ;;
        deadline_fallback_canary_complete)
            fail "Deadline fallback canary reached 10100; explicit post-canary promotion is required."
            ;;
        *) fail "Unknown validated resume mode: ${resume_mode}" ;;
    esac
    read -r target mandatory_activation_canary < <(
        uv run --locked python -m mjlab_microban.scripts.teleop_v12_stage \
            route "${completed}" --shell
    )
    runner_args+=(
        --agent.resume True
        --agent.load-run "^${run_name}$"
        --agent.load-checkpoint "^model_${iteration}[.]pt$"
    )
fi

iterations=$((target - completed))
if (( canary == 1 && mandatory_activation_canary == 0 )); then
    (( iterations > 100 )) || fail "At most 100 updates remain; run to boundary."
    iterations=100
fi
if (( max_updates > 0 && iterations > max_updates )); then
    iterations="${max_updates}"
fi
if (( mandatory_activation_canary == 1 )); then
    echo "[INFO] mandatory activation canary endpoint=${target}"
fi
echo "[INFO] v12 completed=${completed} target=${target} process_updates=${iterations}"

exec uv run --locked train Mjlab-Teleop-V12-Microban \
    --env.scene.num-envs "${num_envs}" --env.seed 42 --agent.seed 42 \
    --agent.num-steps-per-env 24 --agent.max-iterations "${iterations}" \
    --agent.save-interval "${save_interval}" --agent.logger tensorboard \
    --agent.upload-model False --enable-nan-guard True \
    "${runner_args[@]}" "${extra_args[@]}"
