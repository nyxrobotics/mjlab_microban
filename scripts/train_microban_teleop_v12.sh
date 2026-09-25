#!/usr/bin/env bash
# Canonical contract-v12 3000/7000/10000/15000 stage driver.
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly LOG_ROOT="${PROJECT_ROOT}/logs/rsl_rl/mjlab_microban_teleop_v12"
readonly GATE_ROOT="${PROJECT_ROOT}/artifacts/teleop_v12_gates"
readonly SOURCE="${PROJECT_ROOT}/checkpoints/xc330_velocity/model_14999.pt"
readonly SOURCE_SHA="b0bcdadac39716be784207dd6b2b93157162a3e80650e23c05f490c400b9e141"
readonly PROBE="${PROJECT_ROOT}/artifacts/legacy_teleop_probe/model_14999_teleop83_raw_9x300.json"
readonly PROBE_SHA="f51378d59ff4d68fb1185a91eb2a863749e5c7be6ec4cd0ab4a0b08f1565e69d"

usage() {
    cat <<'EOF'
Usage:
  scripts/train_microban_teleop_v12.sh start [--canary] [--agent.run-name NAME]
  scripts/train_microban_teleop_v12.sh resume RUN_NAME [--canary] [--agent.run-name NAME]

Fresh start authenticates model_14999 plus its 9x300 receipt and writes
model_pristine.pt. Resume requires a schema-v2 hash-bound locomotion + tracking
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
while (( $# > 0 )); do
    case "$1" in
        --canary) (( canary == 0 )) || fail "Duplicate --canary"; canary=1; shift ;;
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
    [[ "$(sha256sum -- "${SOURCE}" | awk '{print $1}')" == "${SOURCE_SHA}" ]] \
        || fail "Pinned legacy source SHA-256 mismatch."
    [[ "$(sha256sum -- "${PROBE}" | awk '{print $1}')" == "${PROBE_SHA}" ]] \
        || fail "Pinned legacy probe SHA-256 mismatch."
    uv run --locked --with onnxruntime --with 'protobuf<7' python -m \
        mjlab_microban.scripts.teleop_v12_bootstrap_gate --force >/dev/null
    runner_args=(
        --agent.legacy-velocity-checkpoint "${SOURCE}"
        --agent.legacy-velocity-checkpoint-sha256 "${SOURCE_SHA}"
        --agent.legacy-teleop-probe-receipt "${PROBE}"
        --agent.legacy-teleop-probe-receipt-sha256 "${PROBE_SHA}"
        --agent.save-pristine-checkpoint True
    )
else
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
    [[ -f "${gate}" ]] \
        || fail "Missing gate; run scripts/evaluate_microban_teleop_v12_stage.sh ${run_name} ${iteration}"
    uv run --locked python -m mjlab_microban.scripts.teleop_v12_stage validate \
        "${gate}" "${checkpoint}" >/dev/null
    resume_mode="$(uv run --locked python -m \
        mjlab_microban.scripts.teleop_v12_stage resume-mode \
        "${gate}" "${checkpoint}" --shell)"
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
if (( mandatory_activation_canary == 1 )); then
    echo "[INFO] mandatory activation canary endpoint=${target}"
fi
echo "[INFO] v12 completed=${completed} target=${target} process_updates=${iterations}"

exec uv run --locked train Mjlab-Teleop-V12-Microban \
    --env.scene.num-envs 2048 --env.seed 42 --agent.seed 42 \
    --agent.num-steps-per-env 24 --agent.max-iterations "${iterations}" \
    --agent.save-interval "${save_interval}" --agent.logger tensorboard \
    --agent.upload-model False --enable-nan-guard True \
    "${runner_args[@]}" "${extra_args[@]}"
