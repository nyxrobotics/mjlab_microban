#!/usr/bin/env bash
# Fail-closed stage driver for the Microban PICO training contract v8.
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly LOG_ROOT="${PROJECT_ROOT}/logs/rsl_rl/mjlab_microban_teleop"
readonly GATE_ROOT="${PROJECT_ROOT}/artifacts/teleop_v8_gates"
readonly STAGE_BOUNDARIES=(1500 3000 4500 6000 8000 12000 14000 16000 18000 20000)

cd -- "${PROJECT_ROOT}"

usage() {
    cat <<'EOF'
Usage:
  scripts/train_microban_teleop_v8_stage.sh start [--agent.run-name NAME]
  scripts/train_microban_teleop_v8_stage.sh resume RUN_NAME [--agent.run-name NAME]

The start command stops after 1,500 completed PPO updates.  Evaluate that exact
checkpoint with evaluate_microban_teleop_v8_stage.sh, then pass the run directory
name to resume.  At a boundary, resume verifies the three-seed gate bound to the
checkpoint SHA-256 and advances exactly one boundary.  If a stage was interrupted
between boundaries, the same command verifies its pinned stage/parent provenance
and continues only to that stage's original target:

  1500, 3000, 4500, 6000, 8000, 12000, 14000, 16000, 18000, 20000

MjLab writes a resume into a new timestamp directory.  Always evaluate and use
that new directory for the next stage. The PPO rollout length is fixed at 24
steps per environment and cannot be overridden.
EOF
}

fail() {
    echo "$*" >&2
    exit 2
}

if [[ -n "${MICROBAN_TELEOP_TARGET_ITERS+x}" || -n "${MICROBAN_TELEOP_MAX_ITERS+x}" ]]; then
    fail "This stage driver owns the iteration target; unset MICROBAN_TELEOP_TARGET_ITERS and MICROBAN_TELEOP_MAX_ITERS."
fi
if [[ "${MICROBAN_TELEOP_NUM_ENVS:-4096}" != "4096" ]]; then
    fail "The canonical v8 stage driver requires MICROBAN_TELEOP_NUM_ENVS=4096."
fi
if [[ "${MICROBAN_TELEOP_SEED:-42}" != "42" ]]; then
    fail "The canonical v8 stage driver requires MICROBAN_TELEOP_SEED=42."
fi
if [[ "${MICROBAN_TELEOP_SAVE_INTERVAL:-500}" != "500" ]]; then
    fail "The canonical v8 stage driver requires MICROBAN_TELEOP_SAVE_INTERVAL=500."
fi

mode="${1:-}"
run_name=""
case "${mode}" in
    -h|--help|"")
        usage
        exit 0
        ;;
    start)
        shift
        ;;
    resume)
        (( $# >= 2 )) || { usage >&2; exit 2; }
        run_name="$2"
        [[ "${run_name}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] \
            || fail "RUN_NAME must be one literal run directory name."
        shift 2
        ;;
    *)
        fail "Unknown mode: ${mode}"
        ;;
esac

# The canonical recipe is deliberately narrow: arbitrary Tyro overrides could
# silently change rewards, PPO or the environment while retaining a v8 marker.
additional_args=("$@")
while (( $# > 0 )); do
    case "$1" in
        --agent.run-name)
            (( $# >= 2 )) || fail "--agent.run-name requires a value."
            [[ -n "$2" && "$2" != --* ]] || fail "--agent.run-name requires a value."
            shift 2
            ;;
        --agent.run-name=*)
            [[ -n "${1#*=}" ]] || fail "--agent.run-name requires a value."
            shift
            ;;
        --agent.num-steps-per-env|--agent.num-steps-per-env=*)
            fail "Contract v8 fixes --agent.num-steps-per-env at 24."
            ;;
        --agent.bootstrap-velocity-checkpoint|--agent.bootstrap-velocity-checkpoint=*|\
        --agent.bootstrap-velocity-checkpoint-sha256|--agent.bootstrap-velocity-checkpoint-sha256=*|\
        --agent.save-pristine-checkpoint|--agent.save-pristine-checkpoint=*)
            fail "Contract v8 requires a clean actor; velocity bootstrap/pristine options are forbidden."
            ;;
        *)
            fail "Unsupported canonical v8 override: $1 (only --agent.run-name is allowed)."
            ;;
    esac
done
set -- "${additional_args[@]}"

target_boundary="${STAGE_BOUNDARIES[0]}"
stage_start_boundary=0
parent_checkpoint_sha256=""
parent_gate_sha256=""
if [[ "${mode}" == "resume" ]]; then
    readonly RUN_DIR="${LOG_ROOT}/${run_name}"
    [[ -d "${RUN_DIR}" ]] || fail "Resume run does not exist: ${RUN_DIR}"

    checkpoint_iteration=-1
    shopt -s nullglob
    checkpoint_paths=("${RUN_DIR}"/model_*.pt)
    shopt -u nullglob
    for checkpoint_path in "${checkpoint_paths[@]}"; do
        checkpoint_name="${checkpoint_path##*/}"
        if [[ "${checkpoint_name}" =~ ^model_([0-9]+)\.pt$ ]]; then
            candidate_iteration=$((10#${BASH_REMATCH[1]}))
            if (( candidate_iteration > checkpoint_iteration )); then
                checkpoint_iteration="${candidate_iteration}"
            fi
        fi
    done
    (( checkpoint_iteration >= 0 )) \
        || fail "No numeric model_<iteration>.pt checkpoint found in ${RUN_DIR}."

    completed_iterations=$((checkpoint_iteration + 1))
    boundary_index=-1
    interrupted_stage=0
    previous_boundary=0
    for index in "${!STAGE_BOUNDARIES[@]}"; do
        if (( STAGE_BOUNDARIES[index] == completed_iterations )); then
            boundary_index="${index}"
            break
        fi
        if (( completed_iterations < STAGE_BOUNDARIES[index] )); then
            interrupted_stage=1
            stage_start_boundary="${previous_boundary}"
            target_boundary="${STAGE_BOUNDARIES[index]}"
            break
        fi
        previous_boundary="${STAGE_BOUNDARIES[index]}"
    done
    if (( completed_iterations >= STAGE_BOUNDARIES[${#STAGE_BOUNDARIES[@]} - 1] )); then
        fail "The final 20,000-update boundary is already reached or exceeded; evaluate/export it instead of resuming."
    fi

    readonly CHECKPOINT_PATH="${RUN_DIR}/model_${checkpoint_iteration}.pt"
    if (( interrupted_stage == 1 )); then
        set +e
        stage_metadata_output="$(
            uv run --locked python \
                -m mjlab_microban.scripts.teleop_v8_stage \
                "${CHECKPOINT_PATH}" \
                --completed-iterations "${completed_iterations}" \
                --stage-start-boundary "${stage_start_boundary}" \
                --stage-target-boundary "${target_boundary}" \
                --gate-root "${GATE_ROOT}"
        )"
        metadata_status=$?
        set -e
        (( metadata_status == 0 )) \
            || fail "Interrupted v8 stage provenance validation failed."
        mapfile -t stage_metadata <<<"${stage_metadata_output}"
        (( ${#stage_metadata[@]} == 4 )) \
            || fail "Interrupted v8 stage validator returned malformed metadata."
        [[ "${stage_metadata[0]}" == "${stage_start_boundary}" ]] \
            || fail "Interrupted v8 stage start changed during validation."
        [[ "${stage_metadata[1]}" == "${target_boundary}" ]] \
            || fail "Interrupted v8 stage target changed during validation."
        parent_checkpoint_sha256="${stage_metadata[2]}"
        parent_gate_sha256="${stage_metadata[3]}"
        echo "[PASS] verified interrupted v8 stage ${stage_start_boundary}->${target_boundary} provenance"
    else
        (( boundary_index >= 0 )) \
            || fail "Latest checkpoint does not belong to a v8 stage interval."
        (( boundary_index + 1 < ${#STAGE_BOUNDARIES[@]} )) \
            || fail "The final 20,000-update boundary is already reached; evaluate/export it instead of resuming."

        readonly GATE_PATH="${GATE_ROOT}/${run_name}_boundary_${completed_iterations}_gate.json"
        [[ -f "${GATE_PATH}" ]] \
            || fail "Missing gate receipt: ${GATE_PATH}. Run evaluate_microban_teleop_v8_stage.sh ${run_name} first."

        uv run --locked python - "${GATE_PATH}" "${CHECKPOINT_PATH}" \
            "${run_name}" "${completed_iterations}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import torch

import mjlab_microban.scripts.evaluate_teleop_checkpoint as evaluator_module
from mjlab_microban.scripts.evaluate_teleop_checkpoint import (
    TELEOP_ACCEPTANCE_REVISION,
    TELEOP_EVALUATOR_REVISION,
)
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_TELEOP_ACTOR_INITIALIZATION,
    MICROBAN_TELEOP_RECIPE_REVISION,
    MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION,
)
from mjlab_microban.tasks.microban_teleop_provenance import (
    MICROBAN_TELEOP_TRAINING_PROVENANCE_KEY,
    MICROBAN_TELEOP_TRAINING_PROVENANCE_SHA256_KEY,
    sha256_file,
    validate_canonical_stage_critical_config,
    validate_training_provenance,
)

gate_path = Path(sys.argv[1])
checkpoint_path = Path(sys.argv[2]).resolve()
expected_run = sys.argv[3]
expected_boundary = int(sys.argv[4])
gate = json.loads(gate_path.read_text(encoding="utf-8"))
failures = []
digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
infos = payload.get("infos") if isinstance(payload, dict) else None
if not isinstance(infos, dict):
    failures.append("checkpoint infos are missing")
    infos = {}
try:
    training_provenance = validate_training_provenance(
        infos.get(MICROBAN_TELEOP_TRAINING_PROVENANCE_KEY),
        infos.get(MICROBAN_TELEOP_TRAINING_PROVENANCE_SHA256_KEY),
        require_canonical_stage=True,
        expected_contract_version=MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION,
        expected_recipe_revision=MICROBAN_TELEOP_RECIPE_REVISION,
        expected_actor_initialization=MICROBAN_TELEOP_ACTOR_INITIALIZATION,
        require_current_source=True,
    )
    validate_canonical_stage_critical_config(training_provenance)
except (TypeError, ValueError) as exc:
    failures.append(f"invalid checkpoint training provenance: {exc}")
    training_provenance = {}
training_provenance_sha256 = infos.get(
    MICROBAN_TELEOP_TRAINING_PROVENANCE_SHA256_KEY
)
if gate.get("schema_version") != 3:
    failures.append("unknown gate schema")
if gate.get("status") != "pass":
    failures.append("gate status is not pass")
if gate.get("training_contract_version") != "8":
    failures.append("gate is not contract v8")
if gate.get("recipe_revision") != MICROBAN_TELEOP_RECIPE_REVISION:
    failures.append("gate recipe revision mismatch")
if gate.get("training_provenance_sha256") != training_provenance_sha256:
    failures.append("gate training provenance mismatch")
if gate.get("evaluator_revision") != TELEOP_EVALUATOR_REVISION:
    failures.append("gate evaluator revision mismatch")
if gate.get("acceptance_revision") != TELEOP_ACCEPTANCE_REVISION:
    failures.append("gate acceptance revision mismatch")
if gate.get("evaluator_source_sha256") != sha256_file(
    Path(evaluator_module.__file__)
):
    failures.append("gate evaluator source SHA-256 mismatch")
if gate.get("run_name") != expected_run:
    failures.append("run name mismatch")
if gate.get("completed_iterations") != expected_boundary:
    failures.append("stage boundary mismatch")
if Path(gate.get("checkpoint", "")).resolve() != checkpoint_path:
    failures.append("checkpoint path mismatch")
if gate.get("checkpoint_sha256") != digest:
    failures.append("checkpoint SHA-256 mismatch")
if gate.get("evaluation_seeds") != [42, 43, 44]:
    failures.append("gate did not cover fixed seeds 42/43/44")
reports = gate.get("reports")
if not isinstance(reports, list) or len(reports) != 3:
    failures.append("gate does not reference exactly three evaluator reports")
    reports = []
report_sha256 = gate.get("report_sha256")
if not isinstance(report_sha256, dict) or set(report_sha256) != set(reports):
    failures.append("gate nominal report SHA-256 coverage mismatch")
else:
    for report in reports:
        report_path = Path(report)
        if (
            not report_path.is_file()
            or sha256_file(report_path) != report_sha256[report]
        ):
            failures.append(f"nominal evaluator report changed: {report}")
expected_moving_hmd_reports = 3 if expected_boundary >= 12000 else 0
moving_hmd_reports = gate.get("moving_hmd_reports")
if not isinstance(moving_hmd_reports, list):
    failures.append("gate moving_hmd_reports is not a list")
    moving_hmd_reports = []
elif len(moving_hmd_reports) != expected_moving_hmd_reports:
    failures.append(
        "gate has the wrong number of moving-HMD reports "
        f"({len(moving_hmd_reports)} != "
        f"{expected_moving_hmd_reports})"
    )
    moving_hmd_reports = []
moving_hmd_report_sha256 = gate.get("moving_hmd_report_sha256")
if (
    not isinstance(moving_hmd_report_sha256, dict)
    or set(moving_hmd_report_sha256) != set(moving_hmd_reports)
):
    failures.append("gate moving-HMD report SHA-256 coverage mismatch")
else:
    for report in moving_hmd_reports:
        report_path = Path(report)
        if (
            not report_path.is_file()
            or sha256_file(report_path) != moving_hmd_report_sha256[report]
        ):
            failures.append(f"moving-HMD evaluator report changed: {report}")
invocation = training_provenance.get("invocation")
if not isinstance(invocation, dict):
    failures.append("checkpoint canonical invocation is missing")
elif invocation.get("stage_target_boundary") != expected_boundary:
    failures.append("checkpoint canonical stage target mismatch")
if failures:
    raise SystemExit("Invalid v8 gate receipt: " + "; ".join(failures))
print(f"[PASS] verified v8 boundary {expected_boundary} gate: {gate_path}")
PY

        stage_start_boundary="${completed_iterations}"
        target_boundary="${STAGE_BOUNDARIES[boundary_index + 1]}"
        parent_checkpoint_sha256="$(sha256sum -- "${CHECKPOINT_PATH}" | awk '{print $1}')"
        parent_gate_sha256="$(sha256sum -- "${GATE_PATH}" | awk '{print $1}')"
    fi
fi

export MICROBAN_TELEOP_TARGET_ITERS="${target_boundary}"
export MICROBAN_TELEOP_NUM_ENVS=4096
export MICROBAN_TELEOP_SEED=42
export MICROBAN_TELEOP_SAVE_INTERVAL=500
export MICROBAN_TELEOP_PROVENANCE_MODE=canonical_v8_stage
export MICROBAN_TELEOP_STAGE_START_BOUNDARY="${stage_start_boundary}"
export MICROBAN_TELEOP_STAGE_TARGET_BOUNDARY="${target_boundary}"
export MICROBAN_TELEOP_PARENT_CHECKPOINT_SHA256="${parent_checkpoint_sha256}"
export MICROBAN_TELEOP_PARENT_GATE_SHA256="${parent_gate_sha256}"
echo "[INFO] Contract v8 stage target: ${target_boundary} completed PPO updates"
if [[ "${mode}" == "start" ]]; then
    exec "${SCRIPT_DIR}/train_microban_teleop.sh" train "$@"
fi
exec "${SCRIPT_DIR}/train_microban_teleop.sh" resume "${run_name}" "$@"
