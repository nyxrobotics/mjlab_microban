#!/usr/bin/env bash
# 99-update failed-scenario replay (14901->15000) from the pose-release
# model_14900 of a run whose model_14999 failed the unchanged final gate.
# The parent and the failed gate report are recorded in the rescue marker and
# re-hashed on load; saves keep the pose-release recipe.
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly LOG_ROOT="${PROJECT_ROOT}/logs/rsl_rl/mjlab_microban_teleop_v12"
readonly SOURCE_ITERATION=14900
readonly PROCESS_UPDATES=99
readonly FAILED_REPORT_NAME="failed_final_gate_tracking.json"
readonly SEED_PREFIX="pr_final_rescue_seed_"
readonly TASK="Mjlab-Teleop-V12-HandPoseRelease-Final-Rescue-Microban"

usage() {
    cat <<'EOF_USAGE'
Usage:
  scripts/train_microban_teleop_v12_hand_pose_release_final_rescue.sh \
    MODEL_14900 FAILED_FINAL_GATE_TRACKING_REPORT --mix pr_vN \
    [--agent.run-name NAME]

MODEL_14900 must be the unmarked pose-release model_14900.pt (fresh chain,
optionally through the pose-release model9900 corner rescue; all 20 adapter
columns active, Adam step 298020) of the run whose model_14999 produced
FAILED_FINAL_GATE_TRACKING_REPORT: its stage-gate tracking report under the
unchanged final profile
(full_body_reachable_performance_perturbation_v2_completion_allowance_v1),
failing only hand/foot accuracy, actual_soft_limits and/or
twist_directional_response.  The selected mix must replay every scenario that
failed there:
  pr_v1  50 % ordinary, mixed_forward_left 30 %, bounded_both_feet 20 %
  pr_v2  70 % ordinary, mixed_forward_left 20 %, bounded_both_feet 10 %
  pr_v3  30 % ordinary, mixed_forward_left 45 %, bounded_both_feet 25 %
  pr_v4  50 % ordinary, mixed_forward_left 25 %, bounded_both_feet 15 %,
         mixed_backward_right 10 %
The launcher validates both on CPU, stages the immutable parent, the report
and the parent run's resume record (params/agent.yaml, so the packager's
resume-ancestry walk reaches the gated 10100/10000 boundaries) under
pr_final_rescue_seed_<sha16>/ and runs exactly 99 updates to model_14999.
Gate the result with scripts/evaluate_microban_teleop_v12_stage.sh RUN 14999.
EOF_USAGE
}

fail() { echo "$*" >&2; exit 2; }

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
(( $# >= 2 )) || { usage >&2; exit 2; }
source_checkpoint="$1"
failed_report="$2"
shift 2
mix=""
output_run_name=""
while (( $# > 0 )); do
    case "$1" in
        --mix)
            (( $# >= 2 )) || fail "--mix requires NAME"
            [[ "$2" =~ ^pr_v[0-9]+$ ]] || fail "Unsafe mix name."
            mix="$2"; shift 2 ;;
        --agent.run-name)
            (( $# >= 2 )) || fail "--agent.run-name requires NAME"
            [[ "$2" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || fail "Unsafe output run name."
            output_run_name="$2"; shift 2 ;;
        *) fail "Unsupported argument: $1" ;;
    esac
done
[[ -n "${mix}" ]] || fail "--mix pr_vN is required."
[[ -n "${output_run_name}" ]] || output_run_name="v12_pr_final_rescue_${mix}_14901_to15000"
[[ "${output_run_name}" != "${SEED_PREFIX}"* && "${output_run_name}" != final_rescue_seed_* ]] \
    || fail "Output run name is reserved for the immutable seed."

for path in "${source_checkpoint}" "${failed_report}"; do
    [[ -f "${path}" && ! -L "${path}" ]] || fail "Not a regular file: ${path}"
done
source_checkpoint="$(realpath -e -- "${source_checkpoint}")"
failed_report="$(realpath -e -- "${failed_report}")"
[[ "${source_checkpoint##*/}" == "model_${SOURCE_ITERATION}.pt" ]] \
    || fail "Final rescue parent must be model_${SOURCE_ITERATION}.pt."
parent_run_dir="$(dirname -- "${source_checkpoint}")"
[[ "$(dirname -- "${parent_run_dir}")" == "$(realpath -e -- "${LOG_ROOT}")" ]] \
    || fail "Final rescue parent must be a run under ${LOG_ROOT}."
parent_params="${parent_run_dir}/params/agent.yaml"
[[ -f "${parent_params}" && ! -L "${parent_params}" ]] \
    || fail "Parent run has no resume record: ${parent_params}"
grep -qx 'resume: true' "${parent_params}" \
    || fail "Parent run did not resume from a gated boundary: ${parent_params}"
source_sha="$(sha256sum -- "${source_checkpoint}" | awk '{print $1}')"
failed_report_sha="$(sha256sum -- "${failed_report}" | awk '{print $1}')"
params_sha="$(sha256sum -- "${parent_params}" | awk '{print $1}')"

cd -- "${PROJECT_ROOT}"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] \
    || fail "Final rescue training requires a clean committed source tree."
# Same run as the failed gate, rescuable failures, mix covers them.
uv run --locked python -m mjlab_microban.scripts.teleop_v12_hand_pose_release_final_rescue \
    validate-parent "${source_checkpoint}" "${failed_report}" --mix "${mix}" >/dev/null

seed_run="${SEED_PREFIX}${source_sha:0:16}"
seed_dir="${LOG_ROOT}/${seed_run}"
mkdir -p -- "${seed_dir}/params"
for pair in "${source_checkpoint}|${seed_dir}/model_${SOURCE_ITERATION}.pt|${source_sha}" \
            "${failed_report}|${seed_dir}/${FAILED_REPORT_NAME}|${failed_report_sha}" \
            "${parent_params}|${seed_dir}/params/agent.yaml|${params_sha}"; do
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
printf '%s\n' "${parent_run_dir##*/}" > "${seed_dir}/.parent_run.tmp"
if [[ -e "${seed_dir}/parent_run.txt" ]]; then
    cmp -s -- "${seed_dir}/.parent_run.tmp" "${seed_dir}/parent_run.txt" \
        || fail "Existing rescue seed names another parent run."
    rm -f -- "${seed_dir}/.parent_run.tmp"
else
    mv -- "${seed_dir}/.parent_run.tmp" "${seed_dir}/parent_run.txt"
fi

echo "[INFO] authenticated ${TASK} mix=${mix} parent=${source_sha} (${parent_run_dir##*/})"
echo "[INFO] failed_gate_report=${failed_report_sha}"
echo "[INFO] completed=14901 target=15000 process_updates=${PROCESS_UPDATES}"
export MICROBAN_V12_PR_FINAL_RESCUE_MIX="${mix}"
exec uv run --locked train "${TASK}" \
    --env.scene.num-envs 2048 --env.seed 42 --agent.seed 42 \
    --agent.num-steps-per-env 24 --agent.max-iterations "${PROCESS_UPDATES}" \
    --agent.save-interval "${PROCESS_UPDATES}" --agent.logger tensorboard \
    --agent.upload-model False --enable-nan-guard True \
    --agent.resume True \
    --agent.load-run "^${seed_run}$" \
    --agent.load-checkpoint "^model_${SOURCE_ITERATION}[.]pt$" \
    --agent.run-name "${output_run_name}"
