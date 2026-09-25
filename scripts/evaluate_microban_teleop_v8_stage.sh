#!/usr/bin/env bash
# Evaluate one exact current-v9 curriculum boundary under three fixed seeds.
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly LOG_ROOT="${PROJECT_ROOT}/logs/rsl_rl/mjlab_microban_teleop"
readonly GATE_ROOT="${PROJECT_ROOT}/artifacts/teleop_v9_gates"
readonly EVALUATION_SEEDS=(42 43 44)

cd -- "${PROJECT_ROOT}"

if (( $# != 1 )); then
    echo "Usage: $0 RUN_NAME" >&2
    exit 2
fi
readonly RUN_NAME="$1"
if [[ ! "${RUN_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
    echo "RUN_NAME must be one literal run directory name." >&2
    exit 2
fi
readonly RUN_DIR="${LOG_ROOT}/${RUN_NAME}"
if [[ ! -d "${RUN_DIR}" ]]; then
    echo "Run does not exist: ${RUN_DIR}" >&2
    exit 2
fi

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
if (( checkpoint_iteration < 0 )); then
    echo "No numeric model_<iteration>.pt checkpoint found in ${RUN_DIR}." >&2
    exit 2
fi

readonly COMPLETED_ITERATIONS=$((checkpoint_iteration + 1))
readonly CHECKPOINT_PATH="${RUN_DIR}/model_${checkpoint_iteration}.pt"
case "${COMPLETED_ITERATIONS}" in
    1500)
        scenarios="neutral,low_forward,low_backward,low_lateral_left,low_lateral_right,low_yaw_left,low_yaw_right"
        ;;
    3000)
        scenarios="neutral,low_forward,mid_forward,low_backward,mid_backward,low_lateral_left,mid_lateral_left,low_lateral_right,mid_lateral_right,low_yaw_left,mid_yaw_left,low_yaw_right,mid_yaw_right"
        ;;
    4500)
        scenarios="neutral,low_forward,mid_forward,low_backward,mid_backward,low_lateral_left,mid_lateral_left,low_lateral_right,mid_lateral_right,low_yaw_left,mid_yaw_left,low_yaw_right,mid_yaw_right,max_forward,max_backward,max_lateral_left,max_lateral_right,max_moving_yaw_left,max_moving_yaw_right"
        ;;
    6000)
        scenarios="neutral,low_forward,mid_forward,low_backward,mid_backward,low_lateral_left,mid_lateral_left,low_lateral_right,mid_lateral_right,low_yaw_left,mid_yaw_left,low_yaw_right,mid_yaw_right,max_forward,max_backward,max_lateral_left,max_lateral_right,max_moving_yaw_left,max_moving_yaw_right,max_stationary_yaw_left,max_stationary_yaw_right"
        ;;
    8000|12000)
        scenarios="neutral,low_forward,mid_forward,low_backward,mid_backward,low_lateral_left,mid_lateral_left,low_lateral_right,mid_lateral_right,low_yaw_left,mid_yaw_left,low_yaw_right,mid_yaw_right,max_forward,max_backward,max_lateral_left,max_lateral_right,max_moving_yaw_left,max_moving_yaw_right,max_stationary_yaw_left,max_stationary_yaw_right,mixed_twist_forward_left,mixed_twist_backward_right"
        ;;
    14000|16000)
        scenarios="neutral,low_forward,mid_forward,low_backward,mid_backward,low_lateral_left,mid_lateral_left,low_lateral_right,mid_lateral_right,low_yaw_left,mid_yaw_left,low_yaw_right,mid_yaw_right,max_forward,max_backward,max_lateral_left,max_lateral_right,max_moving_yaw_left,max_moving_yaw_right,max_stationary_yaw_left,max_stationary_yaw_right,mixed_twist_forward_left,mixed_twist_backward_right,max_hands_left,max_hands_right"
        ;;
    18000)
        scenarios="neutral,low_forward,mid_forward,low_backward,mid_backward,low_lateral_left,mid_lateral_left,low_lateral_right,mid_lateral_right,low_yaw_left,mid_yaw_left,low_yaw_right,mid_yaw_right,max_forward,max_backward,max_lateral_left,max_lateral_right,max_moving_yaw_left,max_moving_yaw_right,max_stationary_yaw_left,max_stationary_yaw_right,mixed_twist_forward_left,mixed_twist_backward_right,max_hands_left,max_hands_right,floor_band_edge_single,floor_band_edge_both,max_keypoints_left,max_keypoints_right"
        ;;
    20000)
        scenarios=""
        ;;
    *)
        echo "Latest checkpoint has ${COMPLETED_ITERATIONS} completed updates, not a v9 stage boundary." >&2
        exit 2
        ;;
esac

mkdir -p -- "${GATE_ROOT}"
readonly GATE_PATH="${GATE_ROOT}/${RUN_NAME}_boundary_${COMPLETED_ITERATIONS}_gate.json"

# Reject generic/debug runs before spending GPU time.  A human-readable recipe
# marker is not sufficient: the checkpoint must bind the actual resolved
# config and exact training-source bytes used by the canonical stage driver.
uv run --locked python - "${CHECKPOINT_PATH}" "${COMPLETED_ITERATIONS}" <<'PY'
import sys
from pathlib import Path

import torch

from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_TELEOP_ACTOR_INITIALIZATION,
    MICROBAN_TELEOP_RECIPE_REVISION,
    MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION,
)
from mjlab_microban.tasks.microban_teleop_provenance import (
    MICROBAN_TELEOP_TRAINING_PROVENANCE_KEY,
    MICROBAN_TELEOP_TRAINING_PROVENANCE_SHA256_KEY,
    validate_canonical_stage_critical_config,
    validate_training_provenance,
)

checkpoint_path = Path(sys.argv[1]).resolve()
completed_iterations = int(sys.argv[2])
payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
infos = payload.get("infos") if isinstance(payload, dict) else None
if not isinstance(infos, dict):
    raise SystemExit("Canonical stage checkpoint infos are missing")
manifest = validate_training_provenance(
    infos.get(MICROBAN_TELEOP_TRAINING_PROVENANCE_KEY),
    infos.get(MICROBAN_TELEOP_TRAINING_PROVENANCE_SHA256_KEY),
    require_canonical_stage=True,
    expected_contract_version=MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION,
    expected_recipe_revision=MICROBAN_TELEOP_RECIPE_REVISION,
    expected_actor_initialization=MICROBAN_TELEOP_ACTOR_INITIALIZATION,
    require_current_source=True,
)
validate_canonical_stage_critical_config(manifest)
invocation = manifest["invocation"]
if invocation.get("stage_target_boundary") != completed_iterations:
    raise SystemExit(
        "Checkpoint provenance target does not match this boundary "
        f"({invocation.get('stage_target_boundary')!r} != {completed_iterations})"
    )
print(
    "[PASS] canonical training provenance: "
    f"{infos[MICROBAN_TELEOP_TRAINING_PROVENANCE_SHA256_KEY]}"
)
PY

force_args=()
if [[ "${MICROBAN_TELEOP_GATE_FORCE:-0}" == "1" ]]; then
    force_args=(--force)
    if [[ -e "${GATE_PATH}" ]]; then
        readonly INVALIDATED_GATE_PATH="${GATE_PATH}.invalidated.$(date -u +%Y%m%dT%H%M%S).$$"
        mv -- "${GATE_PATH}" "${INVALIDATED_GATE_PATH}"
        echo "[INFO] Invalidated prior gate receipt: ${INVALIDATED_GATE_PATH}"
    fi
elif [[ -e "${GATE_PATH}" ]]; then
    echo "Gate receipt already exists: ${GATE_PATH}" >&2
    echo "Set MICROBAN_TELEOP_GATE_FORCE=1 only when intentionally re-evaluating it." >&2
    exit 2
fi

report_paths=()
moving_hmd_report_paths=()
moving_hmd_required=0
if (( COMPLETED_ITERATIONS >= 12000 )); then
    moving_hmd_required=1
fi
for seed in "${EVALUATION_SEEDS[@]}"; do
    report_path="${GATE_ROOT}/${RUN_NAME}_model_${checkpoint_iteration}_boundary_${COMPLETED_ITERATIONS}_seed_${seed}.json"
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
    if (( COMPLETED_ITERATIONS < 20000 )); then
        evaluator_args+=(--intermediate-hard-safety-only)
    fi
    if [[ -n "${scenarios}" ]]; then
        evaluator_args+=(--scenarios "${scenarios}")
    fi

    set +e
    uv run --locked python -m mjlab_microban.scripts.evaluate_teleop_checkpoint \
        "${evaluator_args[@]}"
    evaluator_status=$?
    set -e
    if (( COMPLETED_ITERATIONS == 20000 )); then
        expected_status=0
    else
        expected_status=3
    fi
    if (( evaluator_status != expected_status )); then
        echo "Boundary ${COMPLETED_ITERATIONS}, seed ${seed}: expected evaluator exit ${expected_status}, got ${evaluator_status}." >&2
        exit 2
    fi

    if (( moving_hmd_required == 1 )); then
        moving_hmd_report_path="${GATE_ROOT}/${RUN_NAME}_model_${checkpoint_iteration}_boundary_${COMPLETED_ITERATIONS}_seed_${seed}_moving_hmd.json"
        moving_hmd_report_paths+=("${moving_hmd_report_path}")
        moving_hmd_args=(
            --checkpoint "${CHECKPOINT_PATH}"
            --device "${MICROBAN_TELEOP_EVALUATION_DEVICE:-cuda:0}"
            --seed "${seed}"
            --steps 1000
            --settle-steps 50
            --minimum-checkpoint-age-s 0
            --moving-hmd-neck
            --output "${moving_hmd_report_path}"
            "${force_args[@]}"
        )
        if (( COMPLETED_ITERATIONS < 20000 )); then
            moving_hmd_args+=(--intermediate-hard-safety-only)
        fi
        if [[ -n "${scenarios}" ]]; then
            moving_hmd_args+=(--scenarios "${scenarios}")
        fi
        set +e
        uv run --locked python -m mjlab_microban.scripts.evaluate_teleop_checkpoint \
            "${moving_hmd_args[@]}"
        moving_hmd_status=$?
        set -e
        if (( moving_hmd_status != 3 )); then
            echo "Boundary ${COMPLETED_ITERATIONS}, seed ${seed}: moving-HMD evaluator must exit 3, got ${moving_hmd_status}." >&2
            exit 2
        fi
    fi
done

uv run --locked python - "${GATE_PATH}" "${CHECKPOINT_PATH}" "${RUN_NAME}" \
    "${COMPLETED_ITERATIONS}" "${scenarios}" "${moving_hmd_required}" \
    "${report_paths[@]}" "${moving_hmd_report_paths[@]}" <<'PY'
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import torch

from mjlab_microban.scripts.evaluate_teleop_checkpoint import (
    DEPLOYMENT_PERFORMANCE_PROFILE,
    INTERMEDIATE_HARD_SAFETY_PROFILE,
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
run_name = sys.argv[3]
completed_iterations = int(sys.argv[4])
scenario_csv = sys.argv[5]
moving_hmd_required = sys.argv[6] == "1"
report_paths = [Path(value).resolve() for value in sys.argv[7:10]]
moving_hmd_report_paths = [Path(value).resolve() for value in sys.argv[10:]]
expected_seeds = [42, 43, 44]
expected_scenarios = scenario_csv.split(",") if scenario_csv else None
expected_acceptance_profile = (
    DEPLOYMENT_PERFORMANCE_PROFILE
    if completed_iterations == 20_000
    else INTERMEDIATE_HARD_SAFETY_PROFILE
)
failures = []
reports = []
moving_hmd_reports = []
digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
checkpoint_payload = torch.load(
    checkpoint_path, map_location="cpu", weights_only=False
)
checkpoint_infos = (
    checkpoint_payload.get("infos") if isinstance(checkpoint_payload, dict) else None
)
if not isinstance(checkpoint_infos, dict):
    raise SystemExit("v9 stage gate failed: checkpoint infos are missing")
training_provenance = validate_training_provenance(
    checkpoint_infos.get(MICROBAN_TELEOP_TRAINING_PROVENANCE_KEY),
    checkpoint_infos.get(MICROBAN_TELEOP_TRAINING_PROVENANCE_SHA256_KEY),
    require_canonical_stage=True,
    expected_contract_version=MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION,
    expected_recipe_revision=MICROBAN_TELEOP_RECIPE_REVISION,
    expected_actor_initialization=MICROBAN_TELEOP_ACTOR_INITIALIZATION,
    require_current_source=True,
)
validate_canonical_stage_critical_config(training_provenance)
training_provenance_sha256 = checkpoint_infos[
    MICROBAN_TELEOP_TRAINING_PROVENANCE_SHA256_KEY
]


def validate_report(
    report_path: Path, expected_seed: int, *, moving_hmd: bool
) -> tuple[dict, list[str]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    actual_scenarios = [item["name"] for item in report.get("scenarios", [])]
    if report.get("seed") != expected_seed:
        failures.append(f"{report_path.name}: seed mismatch")
    if report.get("checkpoint_iteration") != completed_iterations - 1:
        failures.append(f"{report_path.name}: checkpoint iteration mismatch")
    if Path(report.get("checkpoint", "")).resolve() != checkpoint_path:
        failures.append(f"{report_path.name}: checkpoint path mismatch")
    if report.get("checkpoint_sha256") != digest:
        failures.append(f"{report_path.name}: checkpoint SHA-256 mismatch")
    if report.get("evaluator_revision") != TELEOP_EVALUATOR_REVISION:
        failures.append(f"{report_path.name}: evaluator revision mismatch")
    if report.get("acceptance_revision") != TELEOP_ACCEPTANCE_REVISION:
        failures.append(f"{report_path.name}: acceptance revision mismatch")
    if report.get("acceptance_profile") != expected_acceptance_profile:
        failures.append(f"{report_path.name}: acceptance profile mismatch")
    if report.get("steps_per_scenario") != 1000:
        failures.append(f"{report_path.name}: evaluation was not 1000 steps")
    if report.get("settle_steps") != 50:
        failures.append(f"{report_path.name}: settle window was not 50 steps")
    if (
        report.get("training_contract", {}).get("version")
        != MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION
    ):
        failures.append(f"{report_path.name}: not current training contract")
    if (
        report.get("training_contract", {}).get("training_provenance_sha256")
        != training_provenance_sha256
    ):
        failures.append(f"{report_path.name}: training provenance mismatch")
    if (
        report.get("training_contract", {}).get("canonical_training_stage")
        is not True
    ):
        failures.append(f"{report_path.name}: checkpoint was not a canonical stage")
    if not report.get("training_contract", {}).get("deployment_compatible"):
        failures.append(f"{report_path.name}: checkpoint is not deployment-compatible")
    summary = report.get("summary", {})
    expected_performance_enforcement = completed_iterations == 20_000
    if not summary.get("hard_safety_checks_passed"):
        failures.append(f"{report_path.name}: hard safety checks failed")
    if not summary.get("acceptance_checks_passed"):
        failures.append(f"{report_path.name}: acceptance checks failed")
    if (
        summary.get("performance_acceptance_checks_enforced")
        is not expected_performance_enforcement
    ):
        failures.append(f"{report_path.name}: performance enforcement mismatch")
    scenario_reports = report.get("scenarios", [])
    if any(
        item.get("acceptance", {}).get("performance_checks_enforced")
        is not expected_performance_enforcement
        for item in scenario_reports
        if isinstance(item, dict)
    ):
        failures.append(
            f"{report_path.name}: scenario performance enforcement mismatch"
        )
    expected_status = (
        "diagnostic"
        if moving_hmd or completed_iterations != 20000
        else "pass"
    )
    if report.get("status") != expected_status:
        failures.append(f"{report_path.name}: status is not {expected_status}")
    if expected_scenarios is not None and actual_scenarios != expected_scenarios:
        failures.append(f"{report_path.name}: scenario coverage mismatch")
    if moving_hmd and summary.get("canonical_coverage"):
        failures.append(f"{report_path.name}: moving-HMD report became canonical")
    if (
        not moving_hmd
        and completed_iterations == 20000
        and not summary.get("canonical_coverage")
    ):
        failures.append(f"{report_path.name}: final coverage is not canonical")
    hmd_motion = report.get("hmd_neck_motion", {})
    nominal_environment = report.get("nominal_environment", {})
    if hmd_motion.get("enabled") is not moving_hmd:
        failures.append(f"{report_path.name}: moving-HMD flag mismatch")
    if nominal_environment.get("hmd_neck_motion") is not moving_hmd:
        failures.append(f"{report_path.name}: nominal environment HMD flag mismatch")
    if moving_hmd:
        params = hmd_motion.get("params")
        if not isinstance(params, dict) or params.get("neutral_probability") != 0.0:
            failures.append(
                f"{report_path.name}: moving-HMD neutral probability is not zero"
            )
        evidence = hmd_motion.get("evidence")
        if not isinstance(evidence, dict) or evidence.get("passed") is not True:
            failures.append(
                f"{report_path.name}: moving-HMD motion evidence did not pass"
            )
        else:
            membership = evidence.get("active_event_membership")
            if (
                not isinstance(membership, dict)
                or membership.get("all_scenarios") is not True
                or membership.get("inactive_scenarios") != []
                or membership.get("malformed_scenarios") != []
            ):
                failures.append(
                    f"{report_path.name}: HMD event was not active in every scenario"
                )
            expected_target_excursion = 0.10
            expected_actual_excursion = 0.05
            if (
                evidence.get("minimum_required_target_peak_to_peak_rad")
                != expected_target_excursion
                or evidence.get("minimum_required_actual_peak_to_peak_rad")
                != expected_actual_excursion
            ):
                failures.append(
                    f"{report_path.name}: moving-HMD excursion threshold drift"
                )
            for key, minimum in (
                (
                    "minimum_observed_target_peak_to_peak_rad_by_axis",
                    expected_target_excursion,
                ),
                (
                    "minimum_observed_actual_peak_to_peak_rad_by_axis",
                    expected_actual_excursion,
                ),
            ):
                values = evidence.get(key)
                if not isinstance(values, dict) or set(values) != {
                    "head",
                    "neck_roll",
                    "neck_pitch",
                }:
                    failures.append(
                        f"{report_path.name}: incomplete moving-HMD {key}"
                    )
                elif any(
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or value < minimum
                    for value in values.values()
                ):
                    failures.append(
                        f"{report_path.name}: insufficient moving-HMD {key}"
                    )
    elif hmd_motion.get("params") is not None:
        failures.append(f"{report_path.name}: nominal HMD params must be null")
    return report, actual_scenarios


for expected_seed, report_path in zip(expected_seeds, report_paths, strict=True):
    report, _actual_scenarios = validate_report(
        report_path, expected_seed, moving_hmd=False
    )
    reports.append(report)
if len(report_paths) != len(expected_seeds):
    failures.append("did not receive exactly three reports")
if moving_hmd_required:
    if len(moving_hmd_report_paths) != len(expected_seeds):
        failures.append("did not receive exactly three moving-HMD reports")
    else:
        for index, (expected_seed, report_path) in enumerate(
            zip(expected_seeds, moving_hmd_report_paths, strict=True)
        ):
            report, actual_scenarios = validate_report(
                report_path, expected_seed, moving_hmd=True
            )
            moving_hmd_reports.append(report)
            nominal_scenarios = [
                item["name"] for item in reports[index].get("scenarios", [])
            ]
            if actual_scenarios != nominal_scenarios:
                failures.append(
                    f"{report_path.name}: moving-HMD scenarios differ from nominal"
                )
elif moving_hmd_report_paths:
    failures.append("moving-HMD reports are forbidden before boundary 12000")
if failures:
    raise SystemExit("v9 stage gate failed: " + "; ".join(failures))

gate = {
    "schema_version": 3,
    "status": "pass",
    "training_contract_version": MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION,
    "recipe_revision": MICROBAN_TELEOP_RECIPE_REVISION,
    "training_provenance_sha256": training_provenance_sha256,
    "evaluator_revision": TELEOP_EVALUATOR_REVISION,
    "acceptance_revision": TELEOP_ACCEPTANCE_REVISION,
    "acceptance_profile": expected_acceptance_profile,
    "evaluator_source_sha256": sha256_file(
        Path(__import__(
            "mjlab_microban.scripts.evaluate_teleop_checkpoint",
            fromlist=["__file__"],
        ).__file__)
    ),
    "run_name": run_name,
    "completed_iterations": completed_iterations,
    "checkpoint": str(checkpoint_path),
    "checkpoint_sha256": digest,
    "evaluation_seeds": expected_seeds,
    "scenarios": expected_scenarios if expected_scenarios is not None else "canonical",
    "reports": [str(path) for path in report_paths],
    "report_sha256": {
        str(path): sha256_file(path) for path in report_paths
    },
    "moving_hmd_reports": [str(path) for path in moving_hmd_report_paths],
    "moving_hmd_report_sha256": {
        str(path): sha256_file(path) for path in moving_hmd_report_paths
    },
}
gate_path.parent.mkdir(parents=True, exist_ok=True)
fd, temporary_name = tempfile.mkstemp(
    prefix=f".{gate_path.name}.", suffix=".tmp", dir=gate_path.parent
)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(gate, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_name, gate_path)
finally:
    if os.path.exists(temporary_name):
        os.unlink(temporary_name)
print(f"[PASS] v9 boundary {completed_iterations}, seeds 42/43/44: {gate_path}")
PY
