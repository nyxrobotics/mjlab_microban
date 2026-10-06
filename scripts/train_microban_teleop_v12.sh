#!/usr/bin/env bash
# Contract-v12 3000/7000/10000/15000 stage driver at the forward-lean HOME
# (trunk 10 deg forward, COM over the sole centre) with the target = HOME +
# raw action, saturated only at the servo's +-pi goal range (no software
# clip).  The frozen locomotion actor is a forward-lean-HOME
# Mjlab-Velocity-Microban checkpoint chosen with --source on a fresh start;
# its run's recorded params/env.yaml must show that HOME.
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly LOG_ROOT="${PROJECT_ROOT}/logs/rsl_rl/mjlab_microban_teleop_v12"
readonly GATE_ROOT="${PROJECT_ROOT}/artifacts/teleop_v12_gates"
readonly PROBE_ROOT="${PROJECT_ROOT}/artifacts/legacy_teleop_probe"
readonly BOOTSTRAP_ROOT="${PROJECT_ROOT}/artifacts/teleop_v12_bootstrap"

usage() {
    cat <<'EOF'
Usage:
  scripts/train_microban_teleop_v12.sh start --source VELOCITY_MODEL.pt [--source-sha256 SHA]
      [--canary] [--agent.run-name NAME] [--num-envs N] [--max-updates N]
      [--hand-pose-release]
  scripts/train_microban_teleop_v12.sh resume RUN_NAME [--canary] [--agent.run-name NAME]
      [--num-envs N] [--max-updates N] [--dry-run-skip-gate] [--hand-pose-release]
      [--lateral-fidelity [--lateral-fidelity-weight 8|16|s24|s40]] [--seed N]

Fresh start checks that the velocity checkpoint's run recorded the current
(forward-lean) HOME, hashes it, runs the 9x300 raw
recurrence probe of it in the teleop task (must pass), publishes the pristine
parity/ONNX bootstrap receipt, and writes model_pristine.pt.  The source path,
its SHA-256 and the probe receipt are recorded in every checkpoint and
re-hashed on every save/resume, so keep the source file in place.
--num-envs (default 2048) and --max-updates (cap on this process's updates)
exist for plumbing dry runs; --dry-run-skip-gate resumes without a stage gate
and must never be used for a release chain. Resume requires a schema-v2 hash-bound locomotion + tracking
+ ONNX gate produced by scripts/evaluate_microban_teleop_v12_stage.sh. Normal
runs stop at 3000/7000/10000/15000. A boundary that activates push, HMD/hand,
or feet is followed by a mandatory 100-update canary and a second gate before
the remaining stage may run. --canary also limits any other segment to 100.

--hand-pose-release trains Mjlab-Teleop-V12-HandPoseRelease-Microban (the
active-hand arm pose-release recipe, forward-lean HOME string) through the
same stage route and gates.  At the forward-lean HOME the intended release
route is a fresh pose-release chain: start --source ... --hand-pose-release.
No canonical model_7099 is pinned as a recipe-switch parent here, so resuming
a canonical v11 checkpoint with this option is refused.  A pose-release
checkpoint is resumed with this option (it is refused without it).

--lateral-fidelity (with --hand-pose-release) trains
Mjlab-Teleop-V12-HandPoseRelease-LateralFidelity-Microban: the pose-release
recipe plus one lateral penalty (microban_teleop_v12_lateral_fidelity): the
v1 mixed-command lateral deficit (labels 8, 16) or the v2 hand-active lateral
shortfall that does not read forward speed (labels s24, s40).  Resuming an unmarked fresh-chain
pose-release model_7099 starts the variant: its stage gate is passed to the
runner, which records parent and gate in every save.  A marked checkpoint is
resumed only with this option, at the weight its marker records.
--lateral-fidelity-weight picks the registered weight label for the start
(8 = -8, the default; 16 = -16; s24 = v2 at -24; s40 = v2 at -40).

--seed N (default 42) sets --env.seed and --agent.seed of this training
process only.  It is training randomness, not a gate: no checkpoint lineage
marker, stage gate or package records it, every gate still evaluates with its
own fixed seeds, and the run's params/ records the value used.  Use it to
retrain a segment from the same gated parent when repeated same-seed attempts
fail identically.
EOF
}
fail() { echo "$*" >&2; exit 2; }

mode="${1:-}"
run_name=""
case "${mode}" in
    -h|--help|"") usage; exit 0 ;;
    start) shift ;;
    resume)
        (( $# >= 2 )) || fail "resume requires RUN_NAME"
        run_name="$2"
        [[ "${run_name}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] \
            || fail "RUN_NAME must be one safe literal directory name."
        shift 2
        ;;
    *) fail "Unknown mode: ${mode}" ;;
esac

canary=0
extra_args=()
source_path=""
source_sha=""
num_envs=2048
max_updates=0
skip_gate=0
hand_pose_release=0
lateral_fidelity=0
lateral_fidelity_weight=""
train_seed=42
while (( $# > 0 )); do
    case "$1" in
        --hand-pose-release)
            (( hand_pose_release == 0 )) || fail "Duplicate --hand-pose-release"
            hand_pose_release=1; shift ;;
        --lateral-fidelity)
            (( lateral_fidelity == 0 )) || fail "Duplicate --lateral-fidelity"
            lateral_fidelity=1; shift ;;
        --lateral-fidelity-weight)
            (( $# >= 2 )) && [[ "$2" == 8 || "$2" == 16 || "$2" == s24 || "$2" == s40 ]] \
                || fail "--lateral-fidelity-weight must be 8, 16, s24 or s40"
            lateral_fidelity_weight="$2"; shift 2
            ;;
        --canary) (( canary == 0 )) || fail "Duplicate --canary"; canary=1; shift ;;
        --source)
            (( $# >= 2 )) || fail "--source requires a checkpoint path"
            source_path="$(realpath -e -- "$2")" || fail "Source not found: $2"
            shift 2
            ;;
        --source-sha256)
            (( $# >= 2 )) && [[ "$2" =~ ^[0-9a-f]{64}$ ]] \
                || fail "--source-sha256 requires a lowercase SHA-256"
            source_sha="$2"; shift 2
            ;;
        --num-envs)
            (( $# >= 2 )) && [[ "$2" =~ ^[1-9][0-9]*$ ]] \
                || fail "--num-envs requires a positive integer"
            num_envs="$2"; shift 2
            ;;
        --max-updates)
            (( $# >= 2 )) && [[ "$2" =~ ^[1-9][0-9]*$ ]] \
                || fail "--max-updates requires a positive integer"
            max_updates="$2"; shift 2
            ;;
        --dry-run-skip-gate) skip_gate=1; shift ;;
        --seed)
            (( $# >= 2 )) && [[ "$2" =~ ^(0|[1-9][0-9]{0,8})$ ]] \
                || fail "--seed requires a non-negative integer"
            train_seed="$2"; shift 2
            ;;
        --agent.run-name)
            (( $# >= 2 )) || fail "--agent.run-name requires NAME"
            [[ "$2" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] \
                || fail "Unsafe output run name."
            extra_args+=("$1" "$2"); shift 2
            ;;
        *) fail "Unsupported override: $1" ;;
    esac
done

if (( lateral_fidelity == 1 )); then
    (( hand_pose_release == 1 )) || fail "--lateral-fidelity requires --hand-pose-release"
    [[ "${mode}" == resume ]] \
        || fail "--lateral-fidelity restarts a gated pose-release model_7099 (resume only)"
elif [[ -n "${lateral_fidelity_weight}" ]]; then
    fail "--lateral-fidelity-weight requires --lateral-fidelity"
fi

cd -- "${PROJECT_ROOT}"
completed=0
target=3000
mandatory_activation_canary=0
runner_args=()
save_interval=100
if [[ "${mode}" == "start" ]]; then
    [[ -n "${source_path}" ]] || fail "start requires --source VELOCITY_MODEL.pt"
    (( skip_gate == 0 )) || fail "--dry-run-skip-gate is resume-only"
    actual_sha="$(sha256sum -- "${source_path}" | awk '{print $1}')"
    if [[ -n "${source_sha}" ]]; then
        [[ "${actual_sha}" == "${source_sha}" ]] \
            || fail "Velocity source SHA-256 mismatch: ${actual_sha}"
    fi
    source_sha="${actual_sha}"
    # The source must be a walking run trained at this HOME: its recorded
    # env and its checkpoint's HOME stamp must pass the walking exporter's
    # contract checks.
    uv run --locked python -c 'import sys; from mjlab_microban.scripts.export_walk_onnx import require_current_home_walk_checkpoint as check; check(sys.argv[1])' "${source_path}" \
        || fail "Velocity source was not trained at the current HOME: ${source_path}"
    probe="${PROBE_ROOT}/velocity_${source_sha:0:16}_teleop83_raw_9x300.json"
    mkdir -p -- "${PROBE_ROOT}" "${BOOTSTRAP_ROOT}"
    uv run --locked python -m mjlab_microban.scripts.probe_legacy_actor_in_teleop_env \
        --checkpoint "${source_path}" --expected-sha256 "${source_sha}" \
        --output "${probe}" --force
    probe_sha="$(sha256sum -- "${probe}" | awk '{print $1}')"
    # Fails unless the probe passed (9/9 completed, no fall, 8/8 directional).
    uv run --locked --with onnxruntime --with 'protobuf<7' python -m \
        mjlab_microban.scripts.teleop_v12_bootstrap_gate \
        --checkpoint "${source_path}" --checkpoint-sha256 "${source_sha}" \
        --probe-receipt "${probe}" --probe-receipt-sha256 "${probe_sha}" \
        --output-dir "${BOOTSTRAP_ROOT}/velocity_${source_sha:0:16}" \
        --force >/dev/null
    runner_args=(
        --agent.legacy-velocity-checkpoint "${source_path}"
        --agent.legacy-velocity-checkpoint-sha256 "${source_sha}"
        --agent.legacy-teleop-probe-receipt "${probe}"
        --agent.legacy-teleop-probe-receipt-sha256 "${probe_sha}"
        --agent.save-pristine-checkpoint True
    )
else
    [[ -z "${source_path}${source_sha}" ]] \
        || fail "resume reads its velocity source from the checkpoint"
    run_dir="${LOG_ROOT}/${run_name}"
    [[ -d "${run_dir}" ]] || fail "Resume run not found: ${run_dir}"
    iteration=-1
    shopt -s nullglob
    candidates=("${run_dir}"/model_*.pt)
    shopt -u nullglob
    for path in "${candidates[@]}"; do
        name="${path##*/}"
        if [[ "${name}" =~ ^model_([0-9]+)[.]pt$ ]]; then
            value=$((10#${BASH_REMATCH[1]}))
            (( value > iteration )) && iteration="${value}"
        fi
    done
    (( iteration >= 0 )) || fail "No numeric checkpoint found."
    completed=$((iteration + 1))
    (( completed < 15000 )) || fail "Final 15000-update boundary reached."
    checkpoint="${run_dir}/model_${iteration}.pt"
    gate="${GATE_ROOT}/${run_name}_model_${iteration}_gate.json"
    if (( skip_gate == 1 )); then
        echo "[WARN] --dry-run-skip-gate: resuming WITHOUT a stage gate" >&2
        resume_mode=canonical
    else
        [[ -f "${gate}" ]] \
            || fail "Missing gate; run scripts/evaluate_microban_teleop_v12_stage.sh ${run_name} ${iteration}"
        uv run --locked python -m mjlab_microban.scripts.teleop_v12_stage validate \
            "${gate}" "${checkpoint}" >/dev/null
        resume_mode="$(uv run --locked python -m \
            mjlab_microban.scripts.teleop_v12_stage resume-mode \
            "${gate}" "${checkpoint}" --shell)"
    fi
    case "${resume_mode}" in
        canonical) ;;
        deadline_fallback)
            gate_sha="$(sha256sum -- "${gate}" | awk '{print $1}')"
            runner_args+=(
                --agent.deadline-fallback-resume True
                --agent.deadline-fallback-resume-gate "${gate}"
                --agent.deadline-fallback-resume-gate-sha256 "${gate_sha}"
            )
            save_interval=15000
            ;;
        deadline_fallback_post_canary)
            gate_sha="$(sha256sum -- "${gate}" | awk '{print $1}')"
            runner_args+=(
                --agent.deadline-fallback-resume True
                --agent.deadline-fallback-resume-gate "${gate}"
                --agent.deadline-fallback-resume-gate-sha256 "${gate_sha}"
            )
            save_interval=15000
            ;;
        deadline_fallback_canary_complete)
            fail "Deadline fallback canary reached 10100; explicit post-canary promotion is required."
            ;;
        *) fail "Unknown validated resume mode: ${resume_mode}" ;;
    esac
    recipe_kind="$(uv run --locked python -m \
        mjlab_microban.scripts.teleop_v12_stage checkpoint-recipe \
        "${checkpoint}" --shell)"
    if (( hand_pose_release == 1 )); then
        [[ "${resume_mode}" == canonical ]] \
            || fail "--hand-pose-release has no deadline-fallback route"
        case "${recipe_kind}" in
            hand_pose_release) ;;
            canonical)
                # Forward-lean HOME: no pinned switch parent (see
                # microban_teleop_v12_hand_pose_release_lineage.py).
                fail "No release-eligible recipe switch at the forward-lean HOME; start a fresh chain with: start --source VELOCITY_MODEL.pt --hand-pose-release"
                ;;
            *) fail "--hand-pose-release cannot resume a ${recipe_kind} checkpoint" ;;
        esac
    elif [[ "${recipe_kind}" == hand_pose_release ]]; then
        fail "Checkpoint uses the hand pose-release recipe; pass --hand-pose-release"
    fi
    if [[ "${recipe_kind}" == hand_pose_release ]]; then
        recorded_lf="$(uv run --locked python -m \
            mjlab_microban.scripts.teleop_v12_stage checkpoint-lateral-fidelity \
            "${checkpoint}" | tail -n 1)"
    else
        recorded_lf=none
    fi
    if (( lateral_fidelity == 1 )); then
        if [[ "${recorded_lf}" == none ]]; then
            (( iteration == 7099 )) \
                || fail "The lateral-fidelity variant starts only at a gated pose-release model_7099"
            (( skip_gate == 0 )) || fail "The lateral-fidelity start needs the stage gate"
            lateral_fidelity_weight="${lateral_fidelity_weight:-8}"
            gate_sha="$(sha256sum -- "${gate}" | awk '{print $1}')"
            runner_args+=(
                --agent.lateral-fidelity-switch-gate "${gate}"
                --agent.lateral-fidelity-switch-gate-sha256 "${gate_sha}"
            )
        else
            [[ -z "${lateral_fidelity_weight}" || "${lateral_fidelity_weight}" == "${recorded_lf}" ]] \
                || fail "Checkpoint records lateral-fidelity weight ${recorded_lf}, not ${lateral_fidelity_weight}"
            lateral_fidelity_weight="${recorded_lf}"
        fi
        export MICROBAN_V12_LATERAL_FIDELITY_WEIGHT="${lateral_fidelity_weight}"
    elif [[ "${recorded_lf}" != none ]]; then
        fail "Checkpoint uses the lateral-fidelity variant; pass --lateral-fidelity"
    fi
    read -r target mandatory_activation_canary < <(
        uv run --locked python -m mjlab_microban.scripts.teleop_v12_stage \
            route "${completed}" --shell
    )
    runner_args+=(
        --agent.resume True
        --agent.load-run "^${run_name}$"
        --agent.load-checkpoint "^model_${iteration}[.]pt$"
    )
fi

iterations=$((target - completed))
if (( canary == 1 && mandatory_activation_canary == 0 )); then
    (( iterations > 100 )) || fail "At most 100 updates remain; run to boundary."
    iterations=100
fi
if (( max_updates > 0 && iterations > max_updates )); then
    iterations="${max_updates}"
fi
if (( mandatory_activation_canary == 1 )); then
    echo "[INFO] mandatory activation canary endpoint=${target}"
fi
echo "[INFO] v12 completed=${completed} target=${target} process_updates=${iterations}"

task=Mjlab-Teleop-V12-Microban
if (( hand_pose_release == 1 )); then
    task=Mjlab-Teleop-V12-HandPoseRelease-Microban
fi
if (( lateral_fidelity == 1 )); then
    task=Mjlab-Teleop-V12-HandPoseRelease-LateralFidelity-Microban
    echo "[INFO] lateral-fidelity weight label=${MICROBAN_V12_LATERAL_FIDELITY_WEIGHT}"
fi
echo "[INFO] task=${task} seed=${train_seed}"
exec uv run --locked train "${task}" \
    --env.scene.num-envs "${num_envs}" --env.seed "${train_seed}" --agent.seed "${train_seed}" \
    --agent.num-steps-per-env 24 --agent.max-iterations "${iterations}" \
    --agent.save-interval "${save_interval}" --agent.logger tensorboard \
    --agent.upload-model False --enable-nan-guard True \
    "${runner_args[@]}" "${extra_args[@]}"
