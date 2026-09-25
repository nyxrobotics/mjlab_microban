#!/usr/bin/env bash
# Fail-closed canonical contract-v9 stage driver.
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly LOG_ROOT="${PROJECT_ROOT}/logs/rsl_rl/mjlab_microban_teleop"
readonly GATE_ROOT="${PROJECT_ROOT}/artifacts/teleop_v9_gates"
readonly STAGE_BOUNDARIES=(1500 3000 4500 6000 8000 12000 14000 16000 18000 20000)

usage() {
    cat <<'EOF'
Usage:
  scripts/train_microban_teleop_v9.sh start SAFE_MODEL_PT SAFE_SHA256 PASS_RECEIPT_JSON [--agent.run-name NAME]
  scripts/train_microban_teleop_v9.sh resume RUN_NAME [--agent.run-name NAME]

start verifies the accepted bounded/raw safe-velocity actor and trains only the
canonical 0->1500 stage. Evaluate that boundary with
evaluate_microban_teleop_v9_stage.sh, then use resume. Each resume verifies the
exact checkpoint/gate (or an interrupted stage's pinned lineage) and advances
only to its next boundary, ending with 18000->20000.

Canonical settings are fixed at 2048 environments, seed 42, 24 rollout steps,
and save interval 500. 4096 environments are forbidden: nconmax=512/njmax=2048
caused a measured single-EPA allocation OOM on the 24 GiB RTX 4090.
EOF
}

fail() {
    echo "$*" >&2
    exit 2
}

command -v uv >/dev/null 2>&1 || fail "Required command not found: uv"
command -v sha256sum >/dev/null 2>&1 || fail "Required command not found: sha256sum"
if [[ -n "${MICROBAN_TELEOP_TARGET_ITERS+x}" || -n "${MICROBAN_TELEOP_MAX_ITERS+x}" ]]; then
    fail "The canonical v9 stage driver owns the target; unset MICROBAN_TELEOP_TARGET_ITERS and MICROBAN_TELEOP_MAX_ITERS."
fi
for owned_name in \
    MICROBAN_TELEOP_RESUME_SOURCE_CHECKPOINT_PATH \
    MICROBAN_TELEOP_RESUME_SOURCE_CHECKPOINT_SHA256 \
    MICROBAN_TELEOP_RESUME_SOURCE_CHECKPOINT_ITERATION; do
    [[ -z "${!owned_name+x}" ]] \
        || fail "The canonical v9 stage driver owns ${owned_name}; unset it."
done
[[ "${MICROBAN_TELEOP_NUM_ENVS:-2048}" == "2048" ]] \
    || fail "The canonical v9 stage driver requires MICROBAN_TELEOP_NUM_ENVS=2048; 4096 is forbidden."
[[ "${MICROBAN_TELEOP_SEED:-42}" == "42" ]] \
    || fail "The canonical v9 stage driver requires MICROBAN_TELEOP_SEED=42."
[[ "${MICROBAN_TELEOP_SAVE_INTERVAL:-500}" == "500" ]] \
    || fail "The canonical v9 stage driver requires MICROBAN_TELEOP_SAVE_INTERVAL=500."

mode="${1:-}"
safe_checkpoint=""
safe_sha256=""
safe_receipt=""
run_name=""
case "${mode}" in
    -h|--help|"") usage; exit 0 ;;
    start)
        (( $# >= 4 )) || { usage >&2; exit 2; }
        safe_checkpoint="$2"
        safe_sha256="$3"
        safe_receipt="$4"
        shift 4
        ;;
    resume)
        (( $# >= 2 )) || { usage >&2; exit 2; }
        run_name="$2"
        [[ "${run_name}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] \
            || fail "RUN_NAME must be one literal run directory name."
        shift 2
        ;;
    *) fail "Unknown mode: ${mode}" ;;
esac

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
        *) fail "Unsupported canonical v9 override: $1 (only --agent.run-name is allowed)." ;;
    esac
done
set -- "${additional_args[@]}"

target_boundary=1500
stage_start_boundary=0
completed_iterations=0
parent_checkpoint_sha256=""
parent_gate_sha256=""
runner_mode_args=()
resume_source_checkpoint_path=""
resume_source_checkpoint_sha256=""
resume_source_checkpoint_iteration=""

cd -- "${PROJECT_ROOT}"
if [[ "${mode}" == "start" ]]; then
    [[ "${safe_sha256}" =~ ^[0-9a-f]{64}$ ]] \
        || fail "SAFE_SHA256 must be 64 lowercase hexadecimal characters."
    [[ -f "${safe_checkpoint}" ]] || fail "Safe checkpoint not found: ${safe_checkpoint}"
    [[ -f "${safe_receipt}" ]] || fail "Safe acceptance receipt not found: ${safe_receipt}"
    safe_checkpoint="$(realpath --canonicalize-existing -- "${safe_checkpoint}")"
    safe_receipt="$(realpath --canonicalize-existing -- "${safe_receipt}")"
    actual_sha256="$(sha256sum -- "${safe_checkpoint}" | awk '{print $1}')"
    [[ "${actual_sha256}" == "${safe_sha256}" ]] \
        || fail "Safe checkpoint SHA-256 mismatch: ${actual_sha256}"
    uv run --locked python - "${safe_checkpoint}" "${safe_sha256}" "${safe_receipt}" <<'PY'
import sys
from mjlab_microban.tasks.microban_safe_velocity_checkpoint import inspect_safe_velocity_checkpoint
from mjlab_microban.tasks.microban_teleop_bootstrap import validate_safe_velocity_acceptance_receipt

identity = inspect_safe_velocity_checkpoint(sys.argv[1], expected_sha256=sys.argv[2])
receipt = validate_safe_velocity_acceptance_receipt(sys.argv[3], identity)
print(
    "[PASS] accepted bounded/raw safe source: "
    f"iteration={identity.iteration} recipe={identity.recipe_revision} "
    f"receipt_sha256={receipt.sha256}"
)
PY
    runner_mode_args=(
        --agent.safe-velocity-checkpoint "${safe_checkpoint}"
        --agent.safe-velocity-checkpoint-sha256 "${safe_sha256}"
        --agent.safe-velocity-acceptance-receipt "${safe_receipt}"
    )
else
    readonly RUN_DIR="${LOG_ROOT}/${run_name}"
    [[ -d "${RUN_DIR}" ]] || fail "Resume run does not exist: ${RUN_DIR}"
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
    completed_iterations=$((checkpoint_iteration + 1))
    (( completed_iterations < 20000 )) \
        || fail "The final 20,000-update boundary is already reached; evaluate/export it."
    readonly CHECKPOINT_PATH="${RUN_DIR}/model_${checkpoint_iteration}.pt"

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

    if (( interrupted_stage == 1 )); then
        set +e
        metadata_output="$(
            uv run --locked python -m mjlab_microban.scripts.teleop_v9_stage \
                "${CHECKPOINT_PATH}" \
                --completed-iterations "${completed_iterations}" \
                --stage-start-boundary "${stage_start_boundary}" \
                --stage-target-boundary "${target_boundary}" \
                --gate-root "${GATE_ROOT}"
        )"
        metadata_status=$?
        set -e
        (( metadata_status == 0 )) || fail "Interrupted v9 stage provenance validation failed."
        mapfile -t stage_metadata <<<"${metadata_output}"
        (( ${#stage_metadata[@]} == 4 )) \
            || fail "Interrupted v9 stage validator returned malformed metadata."
        [[ "${stage_metadata[0]}" == "${stage_start_boundary}" ]] \
            || fail "Interrupted v9 stage start changed during validation."
        [[ "${stage_metadata[1]}" == "${target_boundary}" ]] \
            || fail "Interrupted v9 stage target changed during validation."
        parent_checkpoint_sha256="${stage_metadata[2]}"
        parent_gate_sha256="${stage_metadata[3]}"
    else
        (( boundary_index >= 0 && boundary_index + 1 < ${#STAGE_BOUNDARIES[@]} )) \
            || fail "Latest checkpoint does not identify a resumable v9 boundary."
        readonly GATE_PATH="${GATE_ROOT}/${run_name}_boundary_${completed_iterations}_gate.json"
        [[ -f "${GATE_PATH}" ]] \
            || fail "Missing gate receipt: ${GATE_PATH}. Run evaluate_microban_teleop_v9_stage.sh ${run_name} first."
        uv run --locked python - "${GATE_PATH}" "${CHECKPOINT_PATH}" \
            "${run_name}" "${completed_iterations}" <<'PY'
import json
import sys
from pathlib import Path

import mjlab_microban.scripts.evaluate_teleop_checkpoint as evaluator
from mjlab_microban.scripts.evaluate_teleop_checkpoint import (
    DEPLOYMENT_PERFORMANCE_PROFILE,
    INTERMEDIATE_HARD_SAFETY_PROFILE,
    TELEOP_ACCEPTANCE_REVISION,
    TELEOP_EVALUATOR_REVISION,
)
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_TELEOP_RECIPE_REVISION,
    MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION,
    validate_teleop_checkpoint_contract,
)
from mjlab_microban.tasks.microban_teleop_provenance import sha256_file

gate_path = Path(sys.argv[1]).resolve()
checkpoint = Path(sys.argv[2]).resolve()
run_name = sys.argv[3]
boundary = int(sys.argv[4])
gate = json.loads(gate_path.read_text(encoding="utf-8"))
contract = validate_teleop_checkpoint_contract(checkpoint)
training = contract.training_provenance_identity
failures = []
checkpoint_sha256 = sha256_file(checkpoint)
if training is None or not training.canonical_stage:
    failures.append("checkpoint is not a canonical stage")
if gate.get("schema_version") != 3 or gate.get("status") != "pass":
    failures.append("gate schema/status mismatch")
if gate.get("training_contract_version") != MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION:
    failures.append("gate training contract mismatch")
if gate.get("recipe_revision") != MICROBAN_TELEOP_RECIPE_REVISION:
    failures.append("gate recipe mismatch")
if training is not None and gate.get("training_provenance_sha256") != training.sha256:
    failures.append("gate training provenance mismatch")
if gate.get("evaluator_revision") != TELEOP_EVALUATOR_REVISION:
    failures.append("gate evaluator revision mismatch")
if gate.get("acceptance_revision") != TELEOP_ACCEPTANCE_REVISION:
    failures.append("gate acceptance revision mismatch")
expected_acceptance_profile = (
    DEPLOYMENT_PERFORMANCE_PROFILE
    if boundary == 20_000
    else INTERMEDIATE_HARD_SAFETY_PROFILE
)
if gate.get("acceptance_profile") != expected_acceptance_profile:
    failures.append("gate acceptance profile mismatch")
if gate.get("evaluator_source_sha256") != sha256_file(Path(evaluator.__file__)):
    failures.append("gate evaluator source mismatch")
if gate.get("run_name") != run_name or gate.get("completed_iterations") != boundary:
    failures.append("gate run/boundary mismatch")
if Path(gate.get("checkpoint", "")).resolve() != checkpoint:
    failures.append("gate checkpoint path mismatch")
if gate.get("checkpoint_sha256") != checkpoint_sha256:
    failures.append("gate checkpoint SHA-256 mismatch")
if gate.get("evaluation_seeds") != [42, 43, 44]:
    failures.append("gate seed coverage mismatch")
for paths_key, hashes_key, expected_count in (
    ("reports", "report_sha256", 3),
    ("moving_hmd_reports", "moving_hmd_report_sha256", 3 if boundary >= 12000 else 0),
):
    paths = gate.get(paths_key)
    hashes = gate.get(hashes_key)
    if not isinstance(paths, list) or len(paths) != expected_count:
        failures.append(f"{paths_key} count mismatch")
        continue
    if not isinstance(hashes, dict) or set(hashes) != set(paths):
        failures.append(f"{hashes_key} coverage mismatch")
        continue
    for value in paths:
        path = Path(value)
        if not path.is_file() or sha256_file(path) != hashes[value]:
            failures.append(f"changed evaluator report: {value}")
if failures:
    raise SystemExit("Invalid v9 boundary gate: " + "; ".join(failures))
print(f"[PASS] verified v9 boundary {boundary}: {gate_path}")
PY
        stage_start_boundary="${completed_iterations}"
        target_boundary="${STAGE_BOUNDARIES[boundary_index + 1]}"
        parent_checkpoint_sha256="$(sha256sum -- "${CHECKPOINT_PATH}" | awk '{print $1}')"
        parent_gate_sha256="$(sha256sum -- "${GATE_PATH}" | awk '{print $1}')"
    fi
    runner_mode_args=(
        --agent.resume True
        --agent.load-run "^${run_name}$"
        --agent.load-checkpoint "^model_${checkpoint_iteration}[.]pt$"
    )
    resume_source_checkpoint_path="$(realpath --canonicalize-existing -- "${CHECKPOINT_PATH}")"
    resume_source_checkpoint_sha256="$(sha256sum -- "${resume_source_checkpoint_path}" | awk '{print $1}')"
    resume_source_checkpoint_iteration="${checkpoint_iteration}"
fi

iterations_to_run=$((target_boundary - completed_iterations))
(( iterations_to_run > 0 )) || fail "Stage has no remaining updates."
export MICROBAN_TELEOP_TARGET_ITERS="${target_boundary}"
export MICROBAN_TELEOP_NUM_ENVS=2048
export MICROBAN_TELEOP_SEED=42
export MICROBAN_TELEOP_SAVE_INTERVAL=500
export MICROBAN_TELEOP_PROVENANCE_MODE=canonical_v9_stage
export MICROBAN_TELEOP_STAGE_START_BOUNDARY="${stage_start_boundary}"
export MICROBAN_TELEOP_STAGE_TARGET_BOUNDARY="${target_boundary}"
export MICROBAN_TELEOP_PARENT_CHECKPOINT_SHA256="${parent_checkpoint_sha256}"
export MICROBAN_TELEOP_PARENT_GATE_SHA256="${parent_gate_sha256}"
export MICROBAN_TELEOP_RESUME_SOURCE_CHECKPOINT_PATH="${resume_source_checkpoint_path}"
export MICROBAN_TELEOP_RESUME_SOURCE_CHECKPOINT_SHA256="${resume_source_checkpoint_sha256}"
export MICROBAN_TELEOP_RESUME_SOURCE_CHECKPOINT_ITERATION="${resume_source_checkpoint_iteration}"
echo "[INFO] Contract-v9 canonical stage ${stage_start_boundary}->${target_boundary}; remaining=${iterations_to_run}"

exec uv run --locked train Mjlab-Teleop-Microban \
    --env.scene.num-envs 2048 \
    --env.seed 42 \
    --agent.seed 42 \
    --agent.max-iterations "${iterations_to_run}" \
    --agent.save-interval 500 \
    --agent.logger tensorboard \
    --agent.upload-model False \
    --enable-nan-guard True \
    "${runner_mode_args[@]}" \
    "$@"
