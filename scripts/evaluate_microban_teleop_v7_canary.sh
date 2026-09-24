#!/usr/bin/env bash
# Gate the four required checkpoints from one v7 production-width canary.
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly LOG_ROOT="${PROJECT_ROOT}/logs/rsl_rl/mjlab_microban_teleop"
readonly ARTIFACT_ROOT="${PROJECT_ROOT}/artifacts"

if (( $# < 1 || $# > 2 )); then
    echo "Usage: $0 RUN_NAME [EVALUATION_SEED]" >&2
    exit 2
fi
readonly RUN_NAME="$1"
readonly EVALUATION_SEED="${2:-42}"
if [[ ! "${RUN_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
    echo "RUN_NAME must be one literal run directory name." >&2
    exit 2
fi
if [[ ! "${EVALUATION_SEED}" =~ ^[0-9]+$ ]] || (( EVALUATION_SEED > 2147483647 )); then
    echo "EVALUATION_SEED must be an integer in [0, 2147483647]." >&2
    exit 2
fi
readonly RUN_DIR="${LOG_ROOT}/${RUN_NAME}"
if [[ ! -d "${RUN_DIR}" ]]; then
    echo "Canary run does not exist: ${RUN_DIR}" >&2
    exit 2
fi
mkdir -p -- "${ARTIFACT_ROOT}"

for checkpoint_iteration in 0 50 100 250; do
    checkpoint_path="${RUN_DIR}/model_${checkpoint_iteration}.pt"
    report_path="${ARTIFACT_ROOT}/${RUN_NAME}_model_${checkpoint_iteration}_neutral_300.json"
    if [[ ! -f "${checkpoint_path}" ]]; then
        echo "Missing canary checkpoint: ${checkpoint_path}" >&2
        exit 2
    fi
    set +e
    uv run --locked python -m mjlab_microban.scripts.evaluate_teleop_checkpoint \
        --checkpoint "${checkpoint_path}" \
        --device cuda:0 \
        --seed "${EVALUATION_SEED}" \
        --steps 300 \
        --settle-steps 50 \
        --scenarios neutral \
        --minimum-checkpoint-age-s 0 \
        --output "${report_path}"
    evaluator_status=$?
    set -e
    if (( evaluator_status != 3 )); then
        echo "Expected reduced-coverage diagnostic exit 3, got ${evaluator_status}." >&2
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
if report["training_contract"]["version"] != "7":
    failures.append("checkpoint is not contract v7")
if not scenario["finite"]:
    failures.append("non-finite state")
if scenario["fell_over"]:
    failures.append("fall")
if not scenario["reached_time_limit"]:
    failures.append("did not reach time limit")
if scenario["self_collision"]["contact_count_total"] != 0:
    failures.append("self collision")
for name in ("target_clip_fraction", "wrapper_clip_fraction"):
    value = scenario["action"][name]
    if value is None or value != 0.0:
        failures.append(f"{name} {value!r} is not zero")
actual = scenario["joint_soft_limits"]["max_actual_violation_rad"]
if actual is None or actual != 0.0:
    failures.append(f"actual soft-limit violation {actual!r} is not zero")
if failures:
    raise SystemExit(f"{report_path.name}: " + "; ".join(failures))
print(
    f"[PASS] {report_path.name}: finite, no fall/self-collision, "
    f"target_clip=0, wrapper_clip=0, actual_limit_violation=0"
)
PY
done

echo "[PASS] v7 canary checkpoints 0/50/100/250: ${RUN_DIR}"
