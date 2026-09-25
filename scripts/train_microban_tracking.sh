#!/usr/bin/env bash
# Reproducible, fail-closed launcher for the initial walk004 tracking stage.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly PROJECT_ROOT
readonly LOG_ROOT="${PROJECT_ROOT}/logs/rsl_rl/mjlab_microban_tracking"
readonly PROVENANCE_ROOT="${PROJECT_ROOT}/artifacts/microban_tracking_training/invocations"
readonly MOTION_PATH="${PROJECT_ROOT}/data/motions/microban_twist2_walk004_locomotion_prior.npz"
readonly MOTION_SHA256="100656a04438e5b7c09d80e63f79f3e758650a20d87326f3e3970616e6fb69e2"
readonly NUM_ENVS=2048
readonly SEED=42
readonly ROLLOUT_STEPS=24
readonly CLIP_STEPS=267
readonly CLIP_DURATION_S=5.34
readonly LEARNING_RATE=3e-5
readonly LEARNING_EPOCHS=3
readonly DEFAULT_TARGET_ITERATIONS=51
readonly DEFAULT_SAVE_INTERVAL=250

usage() {
    cat <<'EOF'
Usage:
  scripts/train_microban_tracking.sh start [OPTIONS]
  scripts/train_microban_tracking.sh resume RUN_NAME [OPTIONS]
  scripts/train_microban_tracking.sh evaluate RUN_NAME [EVALUATION OPTIONS]

Training options:
  --target-iterations N   Exact total PPO updates across the run lineage
                          (default: 51)
  --save-interval N       Checkpoint interval (default: 250)
  --run-name NAME         Label appended to the new timestamped output run
  --evaluate-after        Gate the final checkpoint after successful training
  --dry-run               Validate and print/record the exact command only

Evaluation options:
  --checkpoint-iteration N  Evaluate model_N.pt (default: latest numeric model)
  --device DEVICE           Evaluator device (default: cuda:0)
  --output PATH             Receipt path (default: under artifacts/)

The initial-stage contract is intentionally closed: unknown options and direct
Tyro overrides are rejected. In particular, motion, environment count, seed,
clip timing, actor inputs, PPO hyperparameters, corruption and randomization
cannot be changed through this wrapper.
EOF
}

fail() {
    echo "$*" >&2
    exit 2
}

require_positive_integer() {
    local name="$1"
    local value="$2"
    [[ "${value}" =~ ^[0-9]+$ ]] || fail "${name} must be a positive integer."
    (( 10#${value} > 0 && 10#${value} <= 2147483647 )) \
        || fail "${name} must be in 1..2147483647."
}

require_literal_name() {
    local name="$1"
    [[ "${name}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] \
        || fail "RUN_NAME must use only letters, digits, '_' and '-', and start with a letter or digit."
}

sha256_file() {
    sha256sum -- "$1" | awk '{print $1}'
}

latest_checkpoint() {
    local run_dir="$1"
    local candidate candidate_name candidate_iteration
    local latest_path=""
    local latest_iteration=-1
    local -a candidates=()

    shopt -s nullglob
    candidates=("${run_dir}"/model_*.pt)
    shopt -u nullglob
    for candidate in "${candidates[@]}"; do
        candidate_name="${candidate##*/}"
        if [[ "${candidate_name}" =~ ^model_([0-9]+)\.pt$ ]]; then
            candidate_iteration=$((10#${BASH_REMATCH[1]}))
            if (( candidate_iteration > latest_iteration )); then
                latest_iteration="${candidate_iteration}"
                latest_path="${candidate}"
            fi
        fi
    done
    [[ -n "${latest_path}" ]] \
        || fail "No numeric model_<iteration>.pt checkpoint found in ${run_dir}."
    printf '%s\n' "${latest_path}"
}

validate_motion() {
    [[ -f "${MOTION_PATH}" ]] || fail "Pinned walk004 motion is missing: ${MOTION_PATH}"
    local actual_sha256
    actual_sha256="$(sha256_file "${MOTION_PATH}")"
    [[ "${actual_sha256}" == "${MOTION_SHA256}" ]] \
        || fail "Pinned walk004 SHA-256 mismatch: ${actual_sha256}"
    if [[ -n "${MICROBAN_TRACKING_MOTION_FILE+x}" ]]; then
        fail "Unset MICROBAN_TRACKING_MOTION_FILE; this wrapper pins the walk004 path and SHA-256."
    fi
}

# Print three lines: internal iteration, completed PPO updates, common step count.
# The completed count deliberately comes from the persisted environment counter.
# RSL-RL resumes at the saved numeric iteration rather than iteration+1, so a
# filename suffix is not an exact lineage-wide update count after a resume.
inspect_checkpoint() {
    local checkpoint_path="$1"
    uv run --locked python - "${checkpoint_path}" "${ROLLOUT_STEPS}" <<'PY'
import sys
from pathlib import Path

import torch

path = Path(sys.argv[1]).resolve()
rollout_steps = int(sys.argv[2])
payload = torch.load(path, map_location="cpu", weights_only=False)
if not isinstance(payload, dict):
    raise SystemExit("checkpoint root is not a mapping")
actor_state = payload.get("actor_state_dict")
if not isinstance(actor_state, dict):
    raise SystemExit("checkpoint has no actor_state_dict")
input_weight = actor_state.get("mlp.0.weight")
output_weight = actor_state.get("mlp.6.weight")
if getattr(input_weight, "shape", None) != torch.Size((512, 99)):
    raise SystemExit(
        f"checkpoint is not the fixed 99-input actor: mlp.0.weight={getattr(input_weight, 'shape', None)}"
    )
if getattr(output_weight, "shape", None) != torch.Size((18, 128)):
    raise SystemExit(
        f"checkpoint is not the bounded 18-action actor: mlp.6.weight={getattr(output_weight, 'shape', None)}"
    )
for key in (
    "distribution.log_std_param",
    "distribution.lower_bound",
    "distribution.upper_bound",
    "distribution.operational_action_lower",
    "distribution.operational_action_upper",
):
    if getattr(actor_state.get(key), "shape", None) != torch.Size((18,)):
        raise SystemExit(f"checkpoint is missing bounded actor state {key}")
iteration = payload.get("iter")
if not isinstance(iteration, int) or iteration < 0:
    raise SystemExit("checkpoint has no non-negative integer 'iter'")
expected_name = f"model_{iteration}.pt"
if path.name != expected_name:
    raise SystemExit(
        f"checkpoint filename/internal iteration mismatch: {path.name} != {expected_name}"
    )
infos = payload.get("infos")
env_state = infos.get("env_state") if isinstance(infos, dict) else None
common_steps = env_state.get("common_step_counter") if isinstance(env_state, dict) else None
if not isinstance(common_steps, int) or common_steps <= 0:
    raise SystemExit("checkpoint has no positive integer environment common_step_counter")
if common_steps % rollout_steps:
    raise SystemExit(
        f"common_step_counter {common_steps} is not divisible by rollout length {rollout_steps}"
    )
print(iteration)
print(common_steps // rollout_steps)
print(common_steps)
PY
}

validate_resume_contract() {
    local run_dir="$1"
    local checkpoint_path="$2"
    uv run --locked python - \
        "${run_dir}/params/env.yaml" \
        "${run_dir}/params/agent.yaml" \
        "${MOTION_PATH}" \
        "${MOTION_SHA256}" \
        "${checkpoint_path}" <<'PY'
import hashlib
import math
import sys
from pathlib import Path

import yaml


class LenientConfigLoader(yaml.SafeLoader):
    """Load MjLab's Python-tagged YAML as inert values."""


def _unknown_tag(loader: LenientConfigLoader, node: yaml.Node):
    if isinstance(node, yaml.ScalarNode):
        value = loader.construct_scalar(node)
        return value if value else node.tag
    if isinstance(node, yaml.SequenceNode):
        return tuple(loader.construct_sequence(node, deep=True))
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    raise TypeError(f"unsupported YAML node: {type(node).__name__}")


LenientConfigLoader.add_constructor(None, _unknown_tag)


def load(path: Path):
    if not path.is_file():
        raise SystemExit(f"missing resolved config: {path}")
    return yaml.load(path.read_text(encoding="utf-8"), Loader=LenientConfigLoader)


def require_equal(label: str, actual, expected) -> None:
    if actual != expected:
        raise SystemExit(f"unsafe resume config: {label}={actual!r}, expected {expected!r}")


def require_close(label: str, actual, expected: float) -> None:
    if not isinstance(actual, (int, float)) or not math.isclose(
        float(actual), expected, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise SystemExit(f"unsafe resume config: {label}={actual!r}, expected {expected!r}")


env_path, agent_path, motion_path, motion_sha256, checkpoint_path = map(
    Path, sys.argv[1:]
)
env = load(env_path)
agent = load(agent_path)

motion_resolved = Path(env["commands"]["motion"]["motion_file"]).resolve()
require_equal("motion_file", motion_resolved, motion_path.resolve())
actual_motion_sha256 = hashlib.sha256(motion_path.read_bytes()).hexdigest()
require_equal("motion_sha256", actual_motion_sha256, str(motion_sha256))
require_equal("num_envs", env["scene"]["num_envs"], 2048)
require_equal("env_seed", env["seed"], 42)
require_equal("decimation", env["decimation"], 4)
require_close("mujoco_timestep", env["sim"]["mujoco"]["timestep"], 0.005)
require_close("episode_length_s", env["episode_length_s"], 5.34)
require_equal("is_finite_horizon", env["is_finite_horizon"], True)
require_equal(
    "actor_corruption", env["observations"]["actor"]["enable_corruption"], False
)

motion = env["commands"]["motion"]
require_equal("motion_sampling", motion["sampling_mode"], "start")
require_equal("motion_joint_range", tuple(motion["joint_position_range"]), (0, 0))
for group in ("pose_range", "velocity_range"):
    for axis in ("x", "y", "z", "roll", "pitch", "yaw"):
        require_equal(f"motion_{group}_{axis}", tuple(motion[group][axis]), (0, 0))

events = env["events"]
for axis in ("x", "y", "z", "roll", "pitch", "yaw"):
    require_equal(
        f"push_{axis}",
        tuple(events["push_robot"]["params"]["velocity_range"][axis]),
        (0, 0),
    )
for axis in (0, 1, 2):
    require_equal(
        f"base_com_{axis}",
        tuple(events["base_com"]["params"]["ranges"][axis]),
        (0, 0),
    )
require_equal(
    "encoder_bias", tuple(events["encoder_bias"]["params"]["bias_range"]), (0, 0)
)
require_equal(
    "foot_friction", tuple(events["foot_friction"]["params"]["ranges"]), (1, 1)
)
require_close("anchor_pos_threshold", env["terminations"]["anchor_pos"]["params"]["threshold"], 0.12)
require_close("ee_body_pos_threshold", env["terminations"]["ee_body_pos"]["params"]["threshold"], 1.0)

require_equal("agent_seed", agent["seed"], 42)
require_equal("rollout_steps", agent["num_steps_per_env"], 24)
require_equal("experiment_name", agent["experiment_name"], "mjlab_microban_tracking")
require_equal("actor_obs_normalization", agent["actor"]["obs_normalization"], False)
require_equal("critic_obs_normalization", agent["critic"]["obs_normalization"], True)
require_equal("actor_hidden_dims", tuple(agent["actor"]["hidden_dims"]), (512, 256, 128))
require_equal("actor_activation", agent["actor"]["activation"], "elu")
require_equal("critic_hidden_dims", tuple(agent["critic"]["hidden_dims"]), (512, 256, 128))
require_equal("critic_activation", agent["critic"]["activation"], "elu")
distribution = agent["actor"]["distribution_cfg"]
if not str(distribution["class_name"]).endswith(
    ":python/name:mjlab_microban.tasks.microban_tracking_mdp.MicrobanTrackingBoundedGaussianDistribution"
):
    raise SystemExit("unsafe resume config: actor distribution is not the bounded Microban distribution")
require_equal("actor_std_type", distribution["std_type"], "log")
require_equal("actor_std_width", len(distribution["init_std"]), 18)
require_equal("actor_lower_bound_width", len(distribution["lower_bound"]), 18)
require_equal("actor_upper_bound_width", len(distribution["upper_bound"]), 18)
algorithm = agent["algorithm"]
require_equal(
    "algorithm_class",
    algorithm["class_name"],
    "mjlab_microban.tasks.microban_tracking_mdp:MicrobanTrackingBoundedPPO",
)
require_close("learning_rate", algorithm["learning_rate"], 3.0e-5)
require_equal("learning_epochs", algorithm["num_learning_epochs"], 3)
require_equal("mini_batches", algorithm["num_mini_batches"], 4)
require_equal("schedule", algorithm["schedule"], "adaptive")
require_close("gamma", algorithm["gamma"], 0.99)
require_close("lam", algorithm["lam"], 0.95)
require_close("desired_kl", algorithm["desired_kl"], 0.01)
require_close("entropy_coef", algorithm["entropy_coef"], 0.0)
require_close("max_grad_norm", algorithm["max_grad_norm"], 1.0)
require_close("value_loss_coef", algorithm["value_loss_coef"], 1.0)
require_equal("use_clipped_value_loss", algorithm["use_clipped_value_loss"], True)
require_close("clip_param", algorithm["clip_param"], 0.2)
require_equal(
    "normalize_advantage_per_mini_batch",
    algorithm["normalize_advantage_per_mini_batch"],
    False,
)
require_equal("optimizer", algorithm["optimizer"], "adam")
require_equal("logger", agent["logger"], "tensorboard")
require_equal("upload_model", agent["upload_model"], False)

if not checkpoint_path.is_file():
    raise SystemExit(f"missing checkpoint: {checkpoint_path}")
print("[PASS] resume source matches the pinned start-clean contract")
PY
}

write_manifest() {
    local manifest_path="$1"
    local status="$2"
    local exit_status="$3"
    local mode="$4"
    local source_run="$5"
    local source_checkpoint="$6"
    local source_checkpoint_sha256="$7"
    local completed_iterations="$8"
    local target_iterations="$9"
    shift 9
    local remaining_iterations="$1"
    shift
    local save_interval="$1"
    shift
    local -a command=("$@")

    python3 - \
        "${manifest_path}" "${status}" "${exit_status}" "${mode}" \
        "${source_run}" "${source_checkpoint}" "${source_checkpoint_sha256}" \
        "${completed_iterations}" "${target_iterations}" \
        "${remaining_iterations}" "${save_interval}" \
        "${PROJECT_ROOT}" "${MOTION_PATH}" "${MOTION_SHA256}" \
        -- "${command[@]}" <<'PY'
import datetime
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

(
    manifest_arg,
    status,
    exit_status,
    mode,
    source_run,
    source_checkpoint,
    source_checkpoint_sha256,
    completed_iterations,
    target_iterations,
    remaining_iterations,
    save_interval,
    project_root_arg,
    motion_arg,
    motion_sha256,
    separator,
    *command,
) = sys.argv[1:]
if separator != "--":
    raise SystemExit("internal manifest argument error")

root = Path(project_root_arg).resolve()
manifest = Path(manifest_arg).resolve()


def output(*args: str) -> str:
    return subprocess.run(
        args,
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


source_files = {}
for relative in (
    "pyproject.toml",
    "uv.lock",
    "scripts/train_microban_tracking.sh",
    "src/mjlab_microban/robot/microban/robot.xml",
    "src/mjlab_microban/robot/microban_constants.py",
    "src/mjlab_microban/robot/xc330_actuator.py",
    "src/mjlab_microban/robot/xc330_params.json",
    "src/mjlab_microban/tasks/microban_tracking_env_cfg.py",
    "src/mjlab_microban/tasks/microban_tracking_mdp.py",
    "src/mjlab_microban/tasks/microban_tracking_policy_export.py",
):
    path = root / relative
    if path.is_file():
        source_files[relative] = digest_bytes(path.read_bytes())

payload = {
    "schema_version": 1,
    "recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "status": status,
    "exit_status": None if exit_status == "" else int(exit_status),
    "contract": {
        "clip_duration_s": 5.34,
        "clip_steps": 267,
        "initial_stage": "start_clean_no_observation_corruption_no_domain_randomization",
        "learning_epochs": 3,
        "learning_rate": 3.0e-5,
        "num_envs": 2048,
        "rollout_steps": 24,
        "seed": 42,
    },
    "invocation": {
        "mode": mode,
        "completed_iterations_before": int(completed_iterations),
        "remaining_iterations": int(remaining_iterations),
        "save_interval": int(save_interval),
        "target_iterations": int(target_iterations),
    },
    "motion": {
        "path": str(Path(motion_arg).resolve()),
        "sha256": motion_sha256,
    },
    "source_checkpoint": None
    if not source_checkpoint
    else {
        "path": source_checkpoint,
        "run_name": source_run,
        "sha256": source_checkpoint_sha256,
    },
    "source": {
        "files": source_files,
        "git_commit": output("git", "rev-parse", "HEAD").strip(),
        "git_diff_sha256": digest_bytes(output("git", "diff", "--binary").encode()),
        "git_status": output("git", "status", "--short").splitlines(),
    },
    "command": command,
    "command_shell": shlex.join(command),
}
encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
manifest.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile(dir=manifest.parent, prefix=f".{manifest.name}.", delete=False) as stream:
    temporary = Path(stream.name)
    stream.write(encoded)
    stream.flush()
    os.fsync(stream.fileno())
temporary.replace(manifest)
PY
}

finish_manifest() {
    local manifest_path="$1"
    local exit_status="$2"
    local status="$3"
    local output_run="${4:-}"
    local output_checkpoint="${5:-}"
    local output_checkpoint_sha256="${6:-}"
    local final_completed_iterations="${7:-}"
    python3 - \
        "${manifest_path}" "${exit_status}" "${status}" "${output_run}" \
        "${output_checkpoint}" "${output_checkpoint_sha256}" \
        "${final_completed_iterations}" <<'PY'
import datetime
import json
import os
import sys
import tempfile
from pathlib import Path

manifest = Path(sys.argv[1]).resolve()
payload = json.loads(manifest.read_text(encoding="utf-8"))
payload["status"] = sys.argv[3]
payload["exit_status"] = int(sys.argv[2])
payload["finished_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
if sys.argv[4]:
    payload["output"] = {
        "run_name": sys.argv[4],
        "checkpoint": sys.argv[5],
        "checkpoint_sha256": sys.argv[6],
        "completed_iterations": int(sys.argv[7]),
    }
encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
with tempfile.NamedTemporaryFile(dir=manifest.parent, prefix=f".{manifest.name}.", delete=False) as stream:
    temporary = Path(stream.name)
    stream.write(encoded)
    stream.flush()
    os.fsync(stream.fileno())
temporary.replace(manifest)
PY
}

run_evaluator() {
    local run_name="$1"
    local checkpoint_iteration="$2"
    local device="$3"
    local output_path="$4"
    local run_dir="${LOG_ROOT}/${run_name}"
    local checkpoint_path

    require_literal_name "${run_name}"
    [[ -d "${run_dir}" ]] || fail "Run does not exist: ${run_dir}"
    if [[ -n "${checkpoint_iteration}" ]]; then
        require_positive_integer "--checkpoint-iteration" "$((10#${checkpoint_iteration} + 1))"
        checkpoint_path="${run_dir}/model_$((10#${checkpoint_iteration})).pt"
        [[ -f "${checkpoint_path}" ]] || fail "Checkpoint does not exist: ${checkpoint_path}"
    else
        checkpoint_path="$(latest_checkpoint "${run_dir}")"
        checkpoint_iteration="${checkpoint_path##*/model_}"
        checkpoint_iteration="${checkpoint_iteration%.pt}"
    fi
    if [[ -z "${output_path}" ]]; then
        output_path="${PROJECT_ROOT}/artifacts/microban_tracking_training/${run_name}_model_${checkpoint_iteration}_gate.json"
    fi
    echo "[INFO] Evaluating: ${checkpoint_path}"
    uv run --locked python -m mjlab_microban.scripts.evaluate_tracking_checkpoint \
        --checkpoint "${checkpoint_path}" \
        --motion "${MOTION_PATH}" \
        --expected-motion-sha256 "${MOTION_SHA256}" \
        --device "${device}" \
        --output "${output_path}"
}

for command_name in uv sha256sum awk git python3; do
    command -v "${command_name}" >/dev/null 2>&1 \
        || fail "Required command not found: ${command_name}"
done

mode="${1:-}"
case "${mode}" in
    -h|--help|"")
        usage
        exit 0
        ;;
    start)
        shift
        ;;
    resume|evaluate)
        (( $# >= 2 )) || { usage >&2; exit 2; }
        source_run="$2"
        require_literal_name "${source_run}"
        shift 2
        ;;
    *)
        fail "Unknown mode: ${mode}"
        ;;
esac

cd -- "${PROJECT_ROOT}"
validate_motion
echo "[INFO] Fixed reference window: ${CLIP_STEPS} policy steps (${CLIP_DURATION_S} s)"

if [[ "${mode}" == "evaluate" ]]; then
    checkpoint_iteration=""
    evaluation_device="cuda:0"
    output_path=""
    while (( $# > 0 )); do
        case "$1" in
            --checkpoint-iteration)
                (( $# >= 2 )) || fail "--checkpoint-iteration requires a value."
                checkpoint_iteration="$2"
                [[ "${checkpoint_iteration}" =~ ^[0-9]+$ ]] \
                    || fail "--checkpoint-iteration must be a non-negative integer."
                shift 2
                ;;
            --checkpoint-iteration=*)
                checkpoint_iteration="${1#*=}"
                [[ "${checkpoint_iteration}" =~ ^[0-9]+$ ]] \
                    || fail "--checkpoint-iteration must be a non-negative integer."
                shift
                ;;
            --device)
                (( $# >= 2 )) || fail "--device requires a value."
                evaluation_device="$2"
                shift 2
                ;;
            --device=*)
                evaluation_device="${1#*=}"
                [[ -n "${evaluation_device}" ]] || fail "--device requires a value."
                shift
                ;;
            --output)
                (( $# >= 2 )) || fail "--output requires a value."
                output_path="$2"
                shift 2
                ;;
            --output=*)
                output_path="${1#*=}"
                [[ -n "${output_path}" ]] || fail "--output requires a value."
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                fail "Unsupported evaluation option: $1"
                ;;
        esac
    done
    run_evaluator "${source_run}" "${checkpoint_iteration}" "${evaluation_device}" "${output_path}"
    exit $?
fi

target_iterations="${DEFAULT_TARGET_ITERATIONS}"
save_interval="${DEFAULT_SAVE_INTERVAL}"
run_label=""
evaluate_after=0
dry_run=0
while (( $# > 0 )); do
    case "$1" in
        --target-iterations)
            (( $# >= 2 )) || fail "--target-iterations requires a value."
            target_iterations="$2"
            shift 2
            ;;
        --target-iterations=*)
            target_iterations="${1#*=}"
            shift
            ;;
        --save-interval)
            (( $# >= 2 )) || fail "--save-interval requires a value."
            save_interval="$2"
            shift 2
            ;;
        --save-interval=*)
            save_interval="${1#*=}"
            shift
            ;;
        --run-name)
            (( $# >= 2 )) || fail "--run-name requires a value."
            run_label="$2"
            shift 2
            ;;
        --run-name=*)
            run_label="${1#*=}"
            shift
            ;;
        --evaluate-after)
            evaluate_after=1
            shift
            ;;
        --dry-run)
            dry_run=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            fail "Arbitrary passthrough is disabled; add reviewed options to this wrapper explicitly."
            ;;
        *)
            fail "Unsupported training option: $1"
            ;;
    esac
done

require_positive_integer "--target-iterations" "${target_iterations}"
require_positive_integer "--save-interval" "${save_interval}"
if [[ -z "${run_label}" ]]; then
    if [[ "${mode}" == "start" ]]; then
        run_label="walk004_startclean_${target_iterations}updates"
    else
        run_label="walk004_resume_to_${target_iterations}updates"
    fi
fi
require_literal_name "${run_label}"

completed_iterations=0
remaining_iterations="${target_iterations}"
source_checkpoint=""
source_checkpoint_sha256=""
resume_args=()
if [[ "${mode}" == "resume" ]]; then
    readonly SOURCE_RUN_DIR="${LOG_ROOT}/${source_run}"
    [[ -d "${SOURCE_RUN_DIR}" ]] || fail "Resume run does not exist: ${SOURCE_RUN_DIR}"
    source_checkpoint="$(latest_checkpoint "${SOURCE_RUN_DIR}")"
    validate_resume_contract "${SOURCE_RUN_DIR}" "${source_checkpoint}"
    mapfile -t checkpoint_info < <(inspect_checkpoint "${source_checkpoint}")
    (( ${#checkpoint_info[@]} == 3 )) || fail "Checkpoint inspector returned malformed output."
    checkpoint_iteration="${checkpoint_info[0]}"
    completed_iterations="${checkpoint_info[1]}"
    common_step_counter="${checkpoint_info[2]}"
    if (( completed_iterations >= target_iterations )); then
        echo "Target already reached: ${completed_iterations} completed, target ${target_iterations}." >&2
        exit 3
    fi
    remaining_iterations=$((target_iterations - completed_iterations))
    source_checkpoint_sha256="$(sha256_file "${source_checkpoint}")"
    resume_args=(
        --agent.resume True
        --agent.load-run "^${source_run}$"
        --agent.load-checkpoint "^model_${checkpoint_iteration}[.]pt$"
    )
    echo "[INFO] Resume source: ${source_checkpoint}"
    echo "[INFO] Persisted environment steps: ${common_step_counter}"
    echo "[INFO] Completed PPO updates: ${completed_iterations}; target: ${target_iterations}; remaining: ${remaining_iterations}"
else
    source_run=""
    echo "[INFO] New run target: ${target_iterations} completed PPO updates"
fi

training_command=(
    uv run --locked train Mjlab-Tracking-Microban
    --env.scene.num-envs "${NUM_ENVS}"
    --env.seed "${SEED}"
    --env.decimation 4
    --env.sim.mujoco.timestep 0.005
    --env.episode-length-s "${CLIP_DURATION_S}"
    --env.is-finite-horizon True
    --env.observations.actor.enable-corruption False
    --env.commands.motion.motion-file "${MOTION_PATH}"
    --env.commands.motion.sampling-mode start
    --env.commands.motion.joint-position-range '(0.0,0.0)'
    --env.commands.motion.pose-range.x '(0.0,0.0)'
    --env.commands.motion.pose-range.y '(0.0,0.0)'
    --env.commands.motion.pose-range.z '(0.0,0.0)'
    --env.commands.motion.pose-range.roll '(0.0,0.0)'
    --env.commands.motion.pose-range.pitch '(0.0,0.0)'
    --env.commands.motion.pose-range.yaw '(0.0,0.0)'
    --env.commands.motion.velocity-range.x '(0.0,0.0)'
    --env.commands.motion.velocity-range.y '(0.0,0.0)'
    --env.commands.motion.velocity-range.z '(0.0,0.0)'
    --env.commands.motion.velocity-range.roll '(0.0,0.0)'
    --env.commands.motion.velocity-range.pitch '(0.0,0.0)'
    --env.commands.motion.velocity-range.yaw '(0.0,0.0)'
    --env.events.push-robot.params.velocity-range.x '(0.0,0.0)'
    --env.events.push-robot.params.velocity-range.y '(0.0,0.0)'
    --env.events.push-robot.params.velocity-range.z '(0.0,0.0)'
    --env.events.push-robot.params.velocity-range.roll '(0.0,0.0)'
    --env.events.push-robot.params.velocity-range.pitch '(0.0,0.0)'
    --env.events.push-robot.params.velocity-range.yaw '(0.0,0.0)'
    --env.events.base-com.params.ranges.0 '(0.0,0.0)'
    --env.events.base-com.params.ranges.1 '(0.0,0.0)'
    --env.events.base-com.params.ranges.2 '(0.0,0.0)'
    --env.events.encoder-bias.params.bias-range '(0.0,0.0)'
    --env.events.foot-friction.params.ranges '(1.0,1.0)'
    --env.terminations.anchor-pos.params.threshold 0.12
    --env.terminations.ee-body-pos.params.threshold 1.0
    --agent.seed "${SEED}"
    --agent.num-steps-per-env "${ROLLOUT_STEPS}"
    --agent.max-iterations "${remaining_iterations}"
    --agent.save-interval "${save_interval}"
    --agent.experiment-name mjlab_microban_tracking
    --agent.run-name "${run_label}"
    --agent.logger tensorboard
    --agent.upload-model False
    --agent.actor.obs-normalization False
    --agent.actor.hidden-dims '(512,256,128)'
    --agent.actor.activation elu
    --agent.critic.obs-normalization True
    --agent.critic.hidden-dims '(512,256,128)'
    --agent.critic.activation elu
    --agent.algorithm.learning-rate "${LEARNING_RATE}"
    --agent.algorithm.num-learning-epochs "${LEARNING_EPOCHS}"
    --agent.algorithm.num-mini-batches 4
    --agent.algorithm.schedule adaptive
    --agent.algorithm.gamma 0.99
    --agent.algorithm.lam 0.95
    --agent.algorithm.desired-kl 0.01
    --agent.algorithm.entropy-coef 0.0
    --agent.algorithm.max-grad-norm 1.0
    --agent.algorithm.value-loss-coef 1.0
    --agent.algorithm.use-clipped-value-loss True
    --agent.algorithm.clip-param 0.2
    --agent.algorithm.normalize-advantage-per-mini-batch False
    --agent.algorithm.optimizer adam
    --enable-nan-guard True
    --gpu-ids '[0]'
    "${resume_args[@]}"
)

mkdir -p -- "${PROVENANCE_ROOT}"
invocation_stamp="$(date -u +%Y%m%dT%H%M%SZ)_${mode}_${run_label}_$$"
manifest_path="${PROVENANCE_ROOT}/${invocation_stamp}.json"
write_manifest \
    "${manifest_path}" planned "" "${mode}" "${source_run}" \
    "${source_checkpoint}" "${source_checkpoint_sha256}" \
    "${completed_iterations}" "${target_iterations}" "${remaining_iterations}" \
    "${save_interval}" "${training_command[@]}"

printf '[INFO] Command:'
printf ' %q' "${training_command[@]}"
printf '\n[INFO] Invocation provenance: %s\n' "${manifest_path}"

if (( dry_run == 1 )); then
    finish_manifest "${manifest_path}" 0 dry_run
    echo "[INFO] Dry run complete; training was not started."
    exit 0
fi

marker_path="$(mktemp)"
trap 'rm -f -- "${marker_path}"' EXIT
touch -- "${marker_path}"
set +e
"${training_command[@]}"
training_status=$?
set -e
if (( training_status != 0 )); then
    finish_manifest "${manifest_path}" "${training_status}" failed
    exit "${training_status}"
fi

mapfile -t new_runs < <(
    find "${LOG_ROOT}" -mindepth 1 -maxdepth 1 -type d \
        -name "*_${run_label}" -newer "${marker_path}" -printf '%f\n' | sort
)
if (( ${#new_runs[@]} != 1 )); then
    finish_manifest "${manifest_path}" 2 failed
    fail "Could not identify exactly one new output run (found ${#new_runs[@]})."
fi
output_run="${new_runs[0]}"
output_checkpoint="$(latest_checkpoint "${LOG_ROOT}/${output_run}")"
mapfile -t output_checkpoint_info < <(inspect_checkpoint "${output_checkpoint}")
(( ${#output_checkpoint_info[@]} == 3 )) \
    || fail "Output checkpoint inspector returned malformed output."
output_completed_iterations="${output_checkpoint_info[1]}"
if (( output_completed_iterations != target_iterations )); then
    finish_manifest "${manifest_path}" 2 failed
    fail "Training returned success but persisted ${output_completed_iterations} updates; expected ${target_iterations}."
fi
output_checkpoint_sha256="$(sha256_file "${output_checkpoint}")"
finish_manifest \
    "${manifest_path}" 0 finished "${output_run}" "${output_checkpoint}" \
    "${output_checkpoint_sha256}" "${output_completed_iterations}"
echo "[PASS] Exact target persisted: ${output_completed_iterations} updates in ${output_run}"

if (( evaluate_after == 1 )); then
    run_evaluator "${output_run}" "" cuda:0 ""
fi
