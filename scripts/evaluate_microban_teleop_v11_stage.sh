#!/usr/bin/env bash
# Evaluate one exact contract-v11 stage boundary or interrupted canary.
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly LOG_ROOT="${PROJECT_ROOT}/logs/rsl_rl/mjlab_microban_teleop"
readonly GATE_ROOT="${PROJECT_ROOT}/artifacts/teleop_v11_gates"
readonly CANARY_ROOT="${PROJECT_ROOT}/artifacts/teleop_v11_canaries"

usage() {
    cat <<'EOF'
Usage:
  scripts/evaluate_microban_teleop_v11_stage.sh RUN_NAME
  scripts/evaluate_microban_teleop_v11_stage.sh --canary RUN_NAME

Boundary mode evaluates seeds 42/43/44. Boundaries 10,000 and 15,000 also
require moving-HMD reports. Only the final 15,000 boundary enforces deployment
performance. Canary mode publishes non-deployment hard-safety evidence for one
aligned 100-update interrupted checkpoint.
EOF
}

fail() { echo "$*" >&2; exit 2; }

canary=0
if [[ "${1:-}" == "--canary" ]]; then canary=1; shift; fi
(( $# == 1 )) || { usage >&2; exit 2; }
readonly RUN_NAME="$1"
[[ "${RUN_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] \
    || fail "RUN_NAME must be one literal run directory name."

cd -- "${PROJECT_ROOT}"
readonly RUN_DIR="${LOG_ROOT}/${RUN_NAME}"
[[ -d "${RUN_DIR}" ]] || fail "Run does not exist: ${RUN_DIR}"
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

readonly COMPLETED_ITERATIONS=$((checkpoint_iteration + 1))
readonly CHECKPOINT_PATH="${RUN_DIR}/model_${checkpoint_iteration}.pt"

if (( canary == 1 )); then
    scenarios="neutral,low_forward,low_yaw_left,mid_yaw_left,low_yaw_right,mid_yaw_right"
    seeds=(42)
    moving_hmd_required=0
    performance_enforced=0
    output_root="${CANARY_ROOT}"
    receipt_path="${CANARY_ROOT}/${RUN_NAME}_model_${checkpoint_iteration}_canary_gate.json"
else
    seeds=(42 43 44)
    performance_enforced=0
    moving_hmd_required=0
    output_root="${GATE_ROOT}"
    receipt_path="${GATE_ROOT}/${RUN_NAME}_boundary_${COMPLETED_ITERATIONS}_gate.json"
    case "${COMPLETED_ITERATIONS}" in
        3000)
            scenarios="neutral,low_forward,mid_forward,low_backward,mid_backward,low_lateral_left,mid_lateral_left,low_lateral_right,mid_lateral_right,low_yaw_left,mid_yaw_left,low_yaw_right,mid_yaw_right"
            ;;
        7000)
            scenarios="neutral,low_forward,mid_forward,low_backward,mid_backward,low_lateral_left,mid_lateral_left,low_lateral_right,mid_lateral_right,low_yaw_left,mid_yaw_left,low_yaw_right,mid_yaw_right,max_forward,max_backward,max_lateral_left,max_lateral_right,max_moving_yaw_left,max_moving_yaw_right,max_stationary_yaw_left,max_stationary_yaw_right,mixed_twist_forward_left,mixed_twist_backward_right"
            ;;
        10000)
            scenarios="neutral,low_forward,mid_forward,low_backward,mid_backward,low_lateral_left,mid_lateral_left,low_lateral_right,mid_lateral_right,low_yaw_left,mid_yaw_left,low_yaw_right,mid_yaw_right,max_forward,max_backward,max_lateral_left,max_lateral_right,max_moving_yaw_left,max_moving_yaw_right,max_stationary_yaw_left,max_stationary_yaw_right,mixed_twist_forward_left,mixed_twist_backward_right,max_hands_left,max_hands_right"
            moving_hmd_required=1
            ;;
        15000)
            scenarios=""
            moving_hmd_required=1
            performance_enforced=1
            ;;
        *) fail "Latest checkpoint has ${COMPLETED_ITERATIONS} completed updates, not a v11 stage boundary. Use --canary for an aligned interrupted checkpoint." ;;
    esac
fi

uv run --locked python - "${CHECKPOINT_PATH}" "${COMPLETED_ITERATIONS}" "${canary}" <<'PY'
import sys
from mjlab_microban.scripts.teleop_v11_stage import validate_checkpoint_for_evaluation

result = validate_checkpoint_for_evaluation(
    sys.argv[1], completed_iterations=int(sys.argv[2]), canary=sys.argv[3] == "1"
)
print(f"[PASS] canonical v11 checkpoint preflight: sha256={result['checkpoint_sha256']}")
PY

mkdir -p -- "${output_root}"
force_args=()
if [[ "${MICROBAN_TELEOP_GATE_FORCE:-0}" == "1" ]]; then
    force_args=(--force)
    if [[ -e "${receipt_path}" ]]; then
        invalidated_path="${receipt_path}.invalidated.$(date -u +%Y%m%dT%H%M%S).$$"
        mv -- "${receipt_path}" "${invalidated_path}"
        echo "[INFO] Invalidated prior receipt: ${invalidated_path}"
    fi
elif [[ -e "${receipt_path}" ]]; then
    fail "Receipt already exists: ${receipt_path}. Set MICROBAN_TELEOP_GATE_FORCE=1 only for intentional re-evaluation."
fi

report_paths=()
moving_hmd_report_paths=()
for seed in "${seeds[@]}"; do
    if (( canary == 1 )); then
        report_path="${CANARY_ROOT}/${RUN_NAME}_model_${checkpoint_iteration}_seed_${seed}_canary.json"
    else
        report_path="${GATE_ROOT}/${RUN_NAME}_model_${checkpoint_iteration}_boundary_${COMPLETED_ITERATIONS}_seed_${seed}.json"
    fi
    report_paths+=("${report_path}")
    evaluator_args=(
        --checkpoint "${CHECKPOINT_PATH}"
        --device "${MICROBAN_TELEOP_EVALUATION_DEVICE:-cuda:0}"
        --seed "${seed}"
        --steps 1000
        --settle-steps 50
        --minimum-checkpoint-age-s 0
        --output "${report_path}"
        "${force_args[@]}"
    )
    if (( canary == 1 )); then
        evaluator_args+=(--canary-hard-safety-only)
    elif (( performance_enforced == 0 )); then
        evaluator_args+=(--intermediate-hard-safety-only)
    fi
    [[ -z "${scenarios}" ]] || evaluator_args+=(--scenarios "${scenarios}")

    set +e
    uv run --locked python -m mjlab_microban.scripts.evaluate_teleop_checkpoint "${evaluator_args[@]}"
    evaluator_status=$?
    set -e
    if (( performance_enforced == 1 && canary == 0 )); then expected_status=0; else expected_status=3; fi
    (( evaluator_status == expected_status )) \
        || fail "Evaluation ${COMPLETED_ITERATIONS}, seed ${seed}: expected exit ${expected_status}, got ${evaluator_status}."

    if (( moving_hmd_required == 1 )); then
        moving_path="${GATE_ROOT}/${RUN_NAME}_model_${checkpoint_iteration}_boundary_${COMPLETED_ITERATIONS}_seed_${seed}_moving_hmd.json"
        moving_hmd_report_paths+=("${moving_path}")
        moving_args=(
            --checkpoint "${CHECKPOINT_PATH}"
            --device "${MICROBAN_TELEOP_EVALUATION_DEVICE:-cuda:0}"
            --seed "${seed}"
            --steps 1000
            --settle-steps 50
            --minimum-checkpoint-age-s 0
            --moving-hmd-neck
            --output "${moving_path}"
            "${force_args[@]}"
        )
        (( performance_enforced == 1 )) || moving_args+=(--intermediate-hard-safety-only)
        [[ -z "${scenarios}" ]] || moving_args+=(--scenarios "${scenarios}")
        set +e
        uv run --locked python -m mjlab_microban.scripts.evaluate_teleop_checkpoint "${moving_args[@]}"
        moving_status=$?
        set -e
        (( moving_status == 3 )) \
            || fail "Moving-HMD evaluation ${COMPLETED_ITERATIONS}, seed ${seed}: expected exit 3, got ${moving_status}."
    fi
done

if (( canary == 1 )); then
    uv run --locked python - "${receipt_path}" "${CHECKPOINT_PATH}" "${RUN_NAME}" "${COMPLETED_ITERATIONS}" "${report_paths[0]}" <<'PY'
import sys
from mjlab_microban.scripts.teleop_v11_stage import publish_canary_receipt

receipt = publish_canary_receipt(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])
print(f"[PASS] v11 interrupted canary (non-deployment): {sys.argv[1]} checkpoint={receipt['checkpoint_sha256']}")
PY
else
    uv run --locked python - "${receipt_path}" "${CHECKPOINT_PATH}" "${RUN_NAME}" "${COMPLETED_ITERATIONS}" "${#report_paths[@]}" "${report_paths[@]}" "${moving_hmd_report_paths[@]}" <<'PY'
import sys
from mjlab_microban.scripts.teleop_v11_stage import publish_stage_gate

nominal_count = int(sys.argv[5])
nominal = sys.argv[6 : 6 + nominal_count]
moving = sys.argv[6 + nominal_count :]
gate = publish_stage_gate(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), nominal, moving)
print(f"[PASS] v11 boundary {sys.argv[4]}, seeds 42/43/44: {sys.argv[1]} acceptance_profile={gate['acceptance_profile']}")
PY
fi
