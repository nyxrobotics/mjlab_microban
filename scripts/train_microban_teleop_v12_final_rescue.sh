#!/usr/bin/env bash
# 99-update final-scenario replay (14901->15000) from a canonical model_14900
# whose canonical (whole-body) report fails only accuracy checks, after the
# same run's 15000 gate failed only accuracy checks.  The parent and both
# reports are recorded in the rescue marker and re-hashed on load.
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly LOG_ROOT="${PROJECT_ROOT}/logs/rsl_rl/mjlab_microban_teleop_v12"
readonly SOURCE_ITERATION=14900
readonly PROCESS_UPDATES=99
readonly PARENT_REPORT_NAME="parent_final_profile_tracking.json"
readonly FAILED_REPORT_NAME="failed_final_gate_tracking.json"

usage() {
    cat <<'EOF_USAGE'
Usage:
  scripts/train_microban_teleop_v12_final_rescue.sh MODEL_14900 \
    PARENT_TRACKING_REPORT FAILED_FINAL_GATE_TRACKING_REPORT \
    [--mix v1|v2|v3] [--agent.run-name NAME]

MODEL_14900 must be a canonical contract-v12 model_14900.pt (all 20 adapter
columns active, Adam step 298020). PARENT_TRACKING_REPORT must be its canonical
whole-body tracking report failing only accuracy checks (or passing).
FAILED_FINAL_GATE_TRACKING_REPORT must be the final-profile tracking report of
model_14999 of the same run, failing only accuracy checks. The launcher
validates all three on CPU, stages the immutable bytes under
final_rescue_seed_<sha16>/ and runs exactly 99 updates to model_14999 with the
10 % ordinary / 90 % evaluator-scenario (mixed_backward_right,
max_keypoints_right) command replay of the selected mix.
EOF_USAGE
}

fail() { echo "$*" >&2; exit 2; }

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
(( $# >= 3 )) || { usage >&2; exit 2; }
source_checkpoint="$1"
parent_report="$2"
failed_report="$3"
shift 3
mix="v1"
output_run_name=""
while (( $# > 0 )); do
    case "$1" in
        --mix)
            (( $# >= 2 )) || fail "--mix requires NAME"
            [[ "$2" =~ ^v[0-9]+$ ]] || fail "Unsafe mix name."
            mix="$2"; shift 2 ;;
        --agent.run-name)
            (( $# >= 2 )) || fail "--agent.run-name requires NAME"
            [[ "$2" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || fail "Unsafe output run name."
            output_run_name="$2"; shift 2 ;;
        *) fail "Unsupported argument: $1" ;;
    esac
done
[[ -n "${output_run_name}" ]] || output_run_name="v12_final_rescue_${mix}_14901_to15000"
[[ "${output_run_name}" != final_rescue_seed_* ]] \
    || fail "Output run name is reserved for the immutable seed."

for path in "${source_checkpoint}" "${parent_report}" "${failed_report}"; do
    [[ -f "${path}" && ! -L "${path}" ]] || fail "Not a regular file: ${path}"
done
source_checkpoint="$(realpath -e -- "${source_checkpoint}")"
parent_report="$(realpath -e -- "${parent_report}")"
failed_report="$(realpath -e -- "${failed_report}")"
[[ "${source_checkpoint##*/}" == "model_${SOURCE_ITERATION}.pt" ]] \
    || fail "Final rescue parent must be model_${SOURCE_ITERATION}.pt."
# The failed gate must belong to the parent's own run.
failed_checkpoint_path="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"]["path"])' "${failed_report}")"
[[ "$(dirname -- "${failed_checkpoint_path}")" == "$(dirname -- "${source_checkpoint}")" ]] \
    || fail "The failed final gate report is not from the parent's run."
source_sha="$(sha256sum -- "${source_checkpoint}" | awk '{print $1}')"
parent_report_sha="$(sha256sum -- "${parent_report}" | awk '{print $1}')"
failed_report_sha="$(sha256sum -- "${failed_report}" | awk '{print $1}')"

cd -- "${PROJECT_ROOT}"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] \
    || fail "Final rescue training requires a clean committed source tree."
uv run --locked python -m mjlab_microban.scripts.teleop_v12_final_rescue \
    validate-parent "${source_checkpoint}" "${parent_report}" "${failed_report}" \
    --mix "${mix}" >/dev/null

seed_run="final_rescue_seed_${source_sha:0:16}"
seed_dir="${LOG_ROOT}/${seed_run}"
mkdir -p -- "${seed_dir}"
for pair in "${source_checkpoint}|${seed_dir}/model_${SOURCE_ITERATION}.pt|${source_sha}" \
            "${parent_report}|${seed_dir}/${PARENT_REPORT_NAME}|${parent_report_sha}" \
            "${failed_report}|${seed_dir}/${FAILED_REPORT_NAME}|${failed_report_sha}"; do
    IFS='|' read -r src dst sha <<<"${pair}"
    if [[ -e "${dst}" ]]; then
        [[ -f "${dst}" && ! -L "${dst}" ]] \
            || fail "Existing rescue seed file is not a regular file: ${dst}"
        [[ "$(sha256sum -- "${dst}" | awk '{print $1}')" == "${sha}" ]] \
            || fail "Existing rescue seed file has the wrong SHA-256: ${dst}"
    else
        cp --reflink=auto --no-clobber -- "${src}" "${dst}"
        [[ "$(sha256sum -- "${dst}" | awk '{print $1}')" == "${sha}" ]] \
            || fail "Staged rescue seed changed during copy: ${dst}"
    fi
done

echo "[INFO] authenticated final rescue mix=${mix} parent=${source_sha}"
echo "[INFO] parent_report=${parent_report_sha} failed_gate_report=${failed_report_sha}"
echo "[INFO] completed=14901 target=15000 process_updates=${PROCESS_UPDATES}"
export MICROBAN_V12_FINAL_RESCUE_MIX="${mix}"
exec uv run --locked train Mjlab-Teleop-V12-Final-Rescue-Microban \
    --env.scene.num-envs 2048 --env.seed 42 --agent.seed 42 \
    --agent.num-steps-per-env 24 --agent.max-iterations "${PROCESS_UPDATES}" \
    --agent.save-interval "${PROCESS_UPDATES}" --agent.logger tensorboard \
    --agent.upload-model False --enable-nan-guard True \
    --agent.resume True \
    --agent.load-run "^${seed_run}$" \
    --agent.load-checkpoint "^model_${SOURCE_ITERATION}[.]pt$" \
    --agent.run-name "${output_run_name}"
