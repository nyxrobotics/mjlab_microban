#!/usr/bin/env bash
# 99-update corner replay (9901->10000) from a canonical model9900 whose strict
# HMD/hand report fails only hand RMS.  The parent and its report are recorded
# in the rescue marker and re-hashed on load (no literal pins).
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly LOG_ROOT="${PROJECT_ROOT}/logs/rsl_rl/mjlab_microban_teleop_v12"
readonly SOURCE_ITERATION=9900
readonly PROCESS_UPDATES=99
readonly REPORT_NAME="parent_strict_tracking.json"

usage() {
    cat <<'EOF_USAGE'
Usage:
  scripts/train_microban_teleop_v12_corner_rescue.sh MODEL_9900 PARENT_TRACKING_REPORT \
    [--hand-pose-release] [--agent.run-name NAME]

MODEL_9900 must be a canonical contract-v12 model_9900.pt (HMD+hand columns
active, foot columns exact zero, Adam step 198020). PARENT_TRACKING_REPORT
must be its strict HMD/hand-profile tracking report whose only failing check is
hand_tracking_rms. The launcher validates both on CPU, stages the immutable
bytes under corner_rescue_seed_<sha16>/ and runs exactly 99 updates to
model_9999 with the 5/90/5 uniform/LF+RB/LB+RF hand sampler. No training
override other than the output run name is accepted.

--hand-pose-release: MODEL_9900 is a fresh active-hand arm pose-release
chain's model_9900 whose strict HMD/hand report fails only hand accuracy
(RMS and/or P95); trains Mjlab-Teleop-V12-HandPoseRelease-Corner-Rescue-Microban
(pose-release env, 5/60/35 sampler).  Its model_9999 keeps the pose-release
recipe; gate it with scripts/evaluate_microban_teleop_v12_stage.sh RUN 9999 and
resume it with train_microban_teleop_v12.sh resume RUN --hand-pose-release.
EOF_USAGE
}

fail() { echo "$*" >&2; exit 2; }

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
(( $# >= 2 )) || { usage >&2; exit 2; }
source_checkpoint="$1"
parent_tracking_report="$2"
shift 2
output_run_name="v12_corner_rescue_v3_9901_to10000"
hand_pose_release=0
if (( $# > 0 )) && [[ "$1" == "--hand-pose-release" ]]; then
    hand_pose_release=1
    output_run_name="v12_pr_corner_rescue_v1_9901_to10000"
    shift
fi
if (( $# > 0 )); then
    [[ "$1" == "--agent.run-name" && $# == 2 ]] \
        || fail "Only one optional --agent.run-name NAME is supported."
    [[ "$2" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] \
        || fail "Unsafe output run name."
    output_run_name="$2"
fi
task=Mjlab-Teleop-V12-Corner-Rescue-Microban
validate_args=()
if (( hand_pose_release == 1 )); then
    task=Mjlab-Teleop-V12-HandPoseRelease-Corner-Rescue-Microban
    validate_args=(--hand-pose-release)
fi
[[ "${output_run_name}" != corner_rescue_seed_* ]] \
    || fail "Output run name is reserved for the immutable seed."

[[ -f "${source_checkpoint}" && ! -L "${source_checkpoint}" ]] \
    || fail "model9900 must be a regular file."
[[ -f "${parent_tracking_report}" && ! -L "${parent_tracking_report}" ]] \
    || fail "Parent tracking report must be a regular file."
source_checkpoint="$(realpath -e -- "${source_checkpoint}")"
parent_tracking_report="$(realpath -e -- "${parent_tracking_report}")"
[[ "${source_checkpoint##*/}" == "model_${SOURCE_ITERATION}.pt" ]] \
    || fail "Corner rescue parent must be model_${SOURCE_ITERATION}.pt."
source_sha="$(sha256sum -- "${source_checkpoint}" | awk '{print $1}')"
report_sha="$(sha256sum -- "${parent_tracking_report}" | awk '{print $1}')"

cd -- "${PROJECT_ROOT}"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] \
    || fail "Corner rescue training requires a clean committed source tree."
uv run --locked python -m mjlab_microban.scripts.teleop_v12_corner_rescue \
    validate-parent "${source_checkpoint}" "${parent_tracking_report}" \
    "${validate_args[@]}" >/dev/null

seed_run="corner_rescue_seed_${source_sha:0:16}"
seed_dir="${LOG_ROOT}/${seed_run}"
seed_checkpoint="${seed_dir}/model_${SOURCE_ITERATION}.pt"
seed_report="${seed_dir}/${REPORT_NAME}"
mkdir -p -- "${seed_dir}"
for pair in "${source_checkpoint}|${seed_checkpoint}|${source_sha}" \
            "${parent_tracking_report}|${seed_report}|${report_sha}"; do
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

echo "[INFO] authenticated ${task} parent=${source_sha} report=${report_sha}"
echo "[INFO] completed=9901 target=10000 process_updates=${PROCESS_UPDATES}"
exec uv run --locked train "${task}" \
    --env.scene.num-envs 2048 --env.seed 42 --agent.seed 42 \
    --agent.num-steps-per-env 24 --agent.max-iterations "${PROCESS_UPDATES}" \
    --agent.save-interval "${PROCESS_UPDATES}" --agent.logger tensorboard \
    --agent.upload-model False --enable-nan-guard True \
    --agent.resume True \
    --agent.load-run "^${seed_run}$" \
    --agent.load-checkpoint "^model_${SOURCE_ITERATION}[.]pt$" \
    --agent.run-name "${output_run_name}"
