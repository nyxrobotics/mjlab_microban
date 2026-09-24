#!/usr/bin/env bash
# Train one production-width v7 update for each required seed and gate
# pristine/model_0 before spending time on the 251-update canaries.
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly LOG_ROOT="${PROJECT_ROOT}/logs/rsl_rl/mjlab_microban_teleop"
readonly ARTIFACT_ROOT="${PROJECT_ROOT}/artifacts"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MICROBAN_TELEOP_NUM_ENVS=4096
export MICROBAN_TELEOP_TARGET_ITERS=1
export MICROBAN_TELEOP_SAVE_INTERVAL=1
mkdir -p -- "${ARTIFACT_ROOT}"

for seed in 42 43 44; do
    start_marker="$(mktemp)"
    MICROBAN_TELEOP_SEED="${seed}" \
        "${SCRIPT_DIR}/train_microban_teleop_v7_canary.sh" \
        --agent.run-name "v7_guarded_shoulder_preflight_seed${seed}"

    mapfile -t pristine_paths < <(
        find "${LOG_ROOT}" -mindepth 2 -maxdepth 2 -type f \
            -name model_pristine.pt -newer "${start_marker}" -print
    )
    rm -f -- "${start_marker}"
    if (( ${#pristine_paths[@]} != 1 )); then
        echo "Seed ${seed}: expected one new pristine checkpoint, found ${#pristine_paths[@]}." >&2
        exit 2
    fi
    run_dir="$(dirname -- "${pristine_paths[0]}")"
    run_name="$(basename -- "${run_dir}")"

    for checkpoint_name in model_pristine.pt model_0.pt; do
        checkpoint_path="${run_dir}/${checkpoint_name}"
        if [[ ! -f "${checkpoint_path}" ]]; then
            echo "Missing preflight checkpoint: ${checkpoint_path}" >&2
            exit 2
        fi
        report_name="${checkpoint_name%.pt}"
        report_path="${ARTIFACT_ROOT}/${run_name}_${report_name}_neutral_300.json"
        set +e
        uv run --locked python -m mjlab_microban.scripts.evaluate_teleop_checkpoint \
            --checkpoint "${checkpoint_path}" \
            --device cuda:0 \
            --seed "${seed}" \
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
target_clip = scenario["action"]["target_clip_fraction"]
wrapper_clip = scenario["action"]["wrapper_clip_fraction"]
if target_clip is None or target_clip != 0.0:
    failures.append(f"target clip fraction {target_clip!r} is not zero")
if wrapper_clip is None or wrapper_clip != 0.0:
    failures.append(f"wrapper clip fraction {wrapper_clip!r} is not zero")
actual = scenario["joint_soft_limits"]["max_actual_violation_rad"]
if actual is None or actual != 0.0:
    failures.append(f"actual soft-limit violation {actual!r} is not zero")
if failures:
    raise SystemExit(f"{report_path.name}: " + "; ".join(failures))
print(
    f"[PASS] {report_path.name}: target_clip={target_clip:.6g}, "
    f"wrapper_clip={wrapper_clip:.6g}, "
    f"actual_limit_violation={actual:.6g} rad"
)
PY
    done
    echo "[PASS] v7 seed ${seed} production-width one-update preflight: ${run_dir}"
done

echo "[PASS] v7 seeds 42/43/44 production-width one-update preflight"
