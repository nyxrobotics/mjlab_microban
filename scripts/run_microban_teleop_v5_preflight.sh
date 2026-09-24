#!/usr/bin/env bash
# Train exactly two production-width updates and gate pristine/model_0/model_1.
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly LOG_ROOT="${PROJECT_ROOT}/logs/rsl_rl/mjlab_microban_teleop"
readonly ARTIFACT_ROOT="${PROJECT_ROOT}/artifacts"
readonly START_MARKER="$(mktemp)"
trap 'rm -f -- "${START_MARKER}"' EXIT

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MICROBAN_TELEOP_NUM_ENVS=4096
export MICROBAN_TELEOP_TARGET_ITERS=2
export MICROBAN_TELEOP_SAVE_INTERVAL=1

"${SCRIPT_DIR}/train_microban_teleop_v5_canary.sh" \
    --agent.run-name v5_bounded_preflight

mapfile -t pristine_paths < <(
    find "${LOG_ROOT}" -mindepth 2 -maxdepth 2 -type f \
        -name model_pristine.pt -newer "${START_MARKER}" -print
)
if (( ${#pristine_paths[@]} != 1 )); then
    echo "Expected exactly one new model_pristine.pt, found ${#pristine_paths[@]}." >&2
    exit 2
fi
readonly RUN_DIR="$(dirname -- "${pristine_paths[0]}")"
readonly RUN_NAME="$(basename -- "${RUN_DIR}")"
mkdir -p -- "${ARTIFACT_ROOT}"

for checkpoint_name in model_pristine.pt model_0.pt model_1.pt; do
    checkpoint_path="${RUN_DIR}/${checkpoint_name}"
    if [[ ! -f "${checkpoint_path}" ]]; then
        echo "Missing preflight checkpoint: ${checkpoint_path}" >&2
        exit 2
    fi
    report_name="${checkpoint_name%.pt}"
    report_path="${ARTIFACT_ROOT}/${RUN_NAME}_${report_name}_neutral_300.json"
    set +e
    uv run --locked python -m mjlab_microban.scripts.evaluate_teleop_checkpoint \
        --checkpoint "${checkpoint_path}" \
        --device cuda:0 \
        --steps 300 \
        --settle-steps 50 \
        --scenarios neutral \
        --minimum-checkpoint-age-s 0 \
        --output "${report_path}"
    evaluator_status=$?
    set -e
    if (( evaluator_status != 3 )); then
        echo "Expected a reduced-coverage diagnostic (exit 3), got ${evaluator_status}." >&2
        exit 2
    fi
    uv run --locked python - "${report_path}" <<'PY'
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
report = json.loads(report_path.read_text(encoding="utf-8"))
scenario = report["scenarios"][0]
failures = []
if not scenario["finite"]:
    failures.append("non-finite state")
if scenario["fell_over"]:
    failures.append("fall")
if not scenario["reached_time_limit"]:
    failures.append("did not reach time limit")
if scenario["self_collision"]["contact_count_total"] != 0:
    failures.append("self collision")
clip = scenario["action"]["target_clip_fraction"]
if clip is None or clip > 0.001:
    failures.append(f"target clip fraction {clip!r} > 0.001")
actual = scenario["joint_soft_limits"]["max_actual_violation_rad"]
if actual is None or actual > 1.0e-6:
    failures.append(f"actual soft-limit violation {actual!r} > 1e-6 rad")
if failures:
    raise SystemExit(f"{report_path.name}: " + "; ".join(failures))
print(
    f"[PASS] {report_path.name}: target_clip={clip:.6g}, "
    f"actual_limit_violation={actual:.6g} rad"
)
PY
done

echo "[PASS] v5 production-width preflight: ${RUN_DIR}"
