"""The one contract between the trained policies and the robot runtime.

``microban-policy-1`` is defined by the robot (microban
``src/policy_contract.py``; docs/policies.md here).  This
module writes exactly what the robot reads, for all three policies:

* the ONNX metadata (``contract_metadata``): contract version, kind and
  recipe id, the full-precision HOME stamp, the joint and action layout, the
  observation schema, the checkpoint and its passed gate, and the startup
  self-test (observations recorded from rollouts of the exported checkpoint
  with the torch actor's deterministic output for each);
* ``src/agents/manifest.json`` (``manifest``), which binds the installed files
  of one release (file and checkpoint SHA-256, the HOME tag, the dry-run flag).

Compatibility is the contract version plus the recipe ids (raised by hand on
both sides when a meaning or a reward family changes) and the self-test: no
source hashes of either side are exchanged.  Before publishing, every exporter
runs the robot's self-test rule on its own file (``check_self_test``).

Nothing here imports mjlab: the pipeline and the exporters both read it.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from mjlab_microban.robot.home_pose import HOME

POLICY_CONTRACT = "microban-policy-1"
# The training recipe of each kind (the robot accepts exactly these).  A recipe
# id changes with the reward family or a target meaning; fixing a failed run
# inside the family (weights, switch updates) does not change it.
RECIPES: Mapping[str, str] = {
    "walk": "microban-walk-track-velocity-1",
    "getup": "microban-getup-single-run-1",
    "pico": "microban-pico-arm-overlay-track-velocity-1",
}
KINDS = tuple(RECIPES)
POLICY_FILES: Mapping[str, str] = {
    "walk": "walk.onnx",
    "getup": "getup.onnx",
    "pico": "pico_teleop.onnx",
}
DRY_RUN_METADATA_KEY = "dry_run_not_deployable"

# The robot's OBSERVATION_DOF_ORDER: action order of every policy and the
# joint order of the walk/get-up joint observations.
ACTION_JOINT_NAMES = (
    "right_shoulder_pitch", "right_shoulder_roll", "right_elbow",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee",
    "right_ankle_pitch", "right_ankle_roll",
    "left_shoulder_pitch", "left_shoulder_roll", "left_elbow",
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee",
    "left_ankle_pitch", "left_ankle_roll",
)
ACTION_WIDTH = len(ACTION_JOINT_NAMES)
HEAD_JOINTS = ("head", "neck_roll", "neck_pitch")
OBSERVATION_SCHEMAS: Mapping[str, tuple[tuple[str, int], ...]] = {
    "walk": (
        ("base_ang_vel", 3), ("projected_gravity", 3), ("joint_pos", ACTION_WIDTH),
        ("joint_vel", ACTION_WIDTH), ("actions", ACTION_WIDTH), ("command", 3),
    ),
    "getup": (
        ("base_ang_vel", 3), ("projected_gravity", 3), ("joint_pos", ACTION_WIDTH),
        ("joint_vel", ACTION_WIDTH), ("actions", ACTION_WIDTH),
    ),
    "pico": (
        ("base_ang_vel", 3), ("projected_gravity", 3), ("joint_pos", 21), ("joint_vel", 21),
        ("actions", ACTION_WIDTH), ("command", 3), ("foot_target", 6), ("arm_target", 6),
    ),
}
OBSERVATION_WIDTHS = {kind: sum(width for _, width in schema) for kind, schema in OBSERVATION_SCHEMAS.items()}
OBSERVATION_JOINT_NAMES: Mapping[str, tuple[str, ...]] = {
    "walk": ACTION_JOINT_NAMES,
    "getup": ACTION_JOINT_NAMES,
    "pico": (*HEAD_JOINTS, *ACTION_JOINT_NAMES),
}
PREVIOUS_ACTION_SEMANTICS = "raw_policy_output"
BASE_ANG_VEL_FRAME = "imu_sensor_xyz"
CONTROL_HZ = 50
# The servo's one-turn goal range (microban_constants.SERVO_TARGET_RANGE_RAD).
SERVO_TARGET_RANGE_RAD = math.pi

# Startup self-test (the robot applies the same row conditions and bound).
SELF_TEST_MIN_ROWS = 8
SELF_TEST_MAX_ROWS = 64
SELF_TEST_ATOL = 1.0e-4
SELF_TEST_RTOL = 1.0e-5
SELF_TEST_GRAVITY_NORM_TOLERANCE = 0.05
SELF_TEST_MAX_JOINT_SPEED_RAD_S = 12.1
SELF_TEST_JOINT_RANGE_MARGIN_RAD = math.radians(5.0)
SELF_TEST_ARM_TARGET_TOLERANCE_RAD = 1.0e-6

# PICO targets: the command support of the training task.
PICO_FOOT_TARGET_LOWER = [-0.03, -0.03, 0.0] * 2
PICO_FOOT_TARGET_UPPER = [0.03, 0.03, 0.05] * 2
PICO_BOTH_FEET_TARGET_LOWER = [-0.01, -0.01, 0.0] * 2
PICO_BOTH_FEET_TARGET_UPPER = [0.01, 0.01, 0.02] * 2
# PICO arm targets (columns 75-80): the angles last written to the six arm
# servos minus HOME, left then right, each (shoulder_pitch, shoulder_roll,
# elbow).  The box is the robot's pico_arm_contract (absolute angles) and the
# slew its pico_arms rate.
PICO_ARM_TARGET_CONTRACT = "microban_pico_arm_target_rel_home_v1"
PICO_ARM_JOINT_NAMES = ("left_shoulder_pitch", "left_shoulder_roll", "left_elbow",
                        "right_shoulder_pitch", "right_shoulder_roll", "right_elbow")
PICO_ARM_LOWER_RAD = [math.radians(value) for value in (-100.0, 10.0, -110.0, -100.0, -120.0, -110.0)]
PICO_ARM_UPPER_RAD = [math.radians(value) for value in (100.0, 120.0, 0.0, 100.0, -10.0, 0.0)]
PICO_ARM_SLEW_RATE_RAD_S = 4.0
PICO_ADAPTER_COLUMNS = [6, 7, 8, 27, 28, 29, *range(69, 81)]
PICO_CURRICULUM_KEYS = ("critic_warmup", "arm_start", "foot_start", "foot_tighten", "total")


class PolicyContractError(ValueError):
    """An export does not satisfy microban-policy-1."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _csv(values: Sequence[object]) -> str:
    """Full-precision CSV (``repr``), never mjlab's 3-decimal rounding."""

    return ",".join(repr(float(value)) if isinstance(value, (int, float)) else str(value) for value in values)


def home_pose_stamp() -> dict[str, Any]:
    """The HOME every policy is stamped with (full precision)."""

    return {
        "root_pos_m": [float(value) for value in HOME.root_pos],
        "root_quat_wxyz": [float(value) for value in HOME.root_quat_wxyz],
        "joint_pos_rad": {name: float(value) for name, value in sorted(HOME.joint_pos_rad.items())},
    }


@lru_cache(maxsize=1)
def joint_ranges() -> dict[str, tuple[float, float]]:
    """The MJCF joint ranges of the 21 hinge joints (radians)."""

    import mujoco

    path = Path(__file__).resolve().parent / "robot" / "microban" / "robot.xml"
    model = mujoco.MjModel.from_xml_path(str(path))
    ranges = {}
    for index in range(model.njnt):
        if model.jnt_type[index] == mujoco.mjtJoint.mjJNT_HINGE:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
            ranges[name] = (float(model.jnt_range[index][0]), float(model.jnt_range[index][1]))
    return ranges


def _term_slices(kind: str) -> dict[str, slice]:
    slices, start = {}, 0
    for name, width in OBSERVATION_SCHEMAS[kind]:
        slices[name] = slice(start, start + width)
        start += width
    return slices


def physical_row_problem(kind: str, row: Sequence[float]) -> str | None:
    """Why ``row`` is not a possible robot state (the robot's self-test rule), or None."""

    values = np.asarray(row, dtype=np.float64)
    if values.shape != (OBSERVATION_WIDTHS[kind],):
        return f"width {values.shape} is not {OBSERVATION_WIDTHS[kind]}"
    if not np.isfinite(values).all():
        return "non-finite value"
    terms = _term_slices(kind)
    if abs(float(np.linalg.norm(values[terms["projected_gravity"]])) - 1.0) > SELF_TEST_GRAVITY_NORM_TOLERANCE:
        return "projected_gravity is not a unit vector"
    if np.any(np.abs(values[terms["joint_vel"]]) > SELF_TEST_MAX_JOINT_SPEED_RAD_S):
        return "joint speed above the servo's no-load speed"
    ranges = joint_ranges()
    for name, residual in zip(OBSERVATION_JOINT_NAMES[kind], values[terms["joint_pos"]], strict=True):
        if name in HEAD_JOINTS:
            continue
        lower, upper = ranges[name]
        angle = residual + HOME.joint_pos_rad[name]
        if not lower - SELF_TEST_JOINT_RANGE_MARGIN_RAD <= angle <= upper + SELF_TEST_JOINT_RANGE_MARGIN_RAD:
            return f"{name} outside its range"
    if "arm_target" in terms:
        for name, value, lower, upper in zip(PICO_ARM_JOINT_NAMES, values[terms["arm_target"]],
                                             PICO_ARM_LOWER_RAD, PICO_ARM_UPPER_RAD, strict=True):
            angle = value + HOME.joint_pos_rad[name]
            if not lower - SELF_TEST_ARM_TARGET_TOLERANCE_RAD <= angle <= upper + SELF_TEST_ARM_TARGET_TOLERANCE_RAD:
                return f"arm_target {name} outside the arm box"
    return None


def select_self_test_rows(kind: str, rows: Sequence[Sequence[float]], count: int = 32) -> list[list[float]]:
    """At most ``count`` possible-state rows of ``rows``, evenly spread, in order."""

    usable = [[float(value) for value in row] for row in rows if physical_row_problem(kind, row) is None]
    if len(usable) < SELF_TEST_MIN_ROWS:
        raise PolicyContractError(
            f"{kind} self-test: only {len(usable)} of {len(rows)} recorded observations are possible "
            f"robot states (need {SELF_TEST_MIN_ROWS})"
        )
    if len(usable) <= count:
        return usable
    picks = np.linspace(0, len(usable) - 1, count).round().astype(int)
    return [usable[index] for index in picks]


# Self-test rows of walk and get-up: a seeded rollout of the exported
# checkpoint in its play env (one env, CPU), one observation every 10 control
# steps.
SELF_TEST_SEED = 0
SELF_TEST_ROLLOUT_STEPS = 400
SELF_TEST_RECORD_EVERY = 10


def rollout_self_test(kind: str, env: Any, policy: Any) -> tuple[list[list[float]], list[list[float]]]:
    """Self-test rows from a rollout and the policy's deterministic output for each.

    ``env`` is the RslRlVecEnvWrapper of a one-env play env created with
    ``seed=SELF_TEST_SEED``; ``policy`` the runner's CPU inference policy.
    """

    import torch
    from tensordict import TensorDict

    recorded = []
    observations = env.get_observations()
    with torch.inference_mode():
        for step in range(SELF_TEST_ROLLOUT_STEPS):
            if step % SELF_TEST_RECORD_EVERY == SELF_TEST_RECORD_EVERY - 1:
                recorded.append(observations["actor"][0].detach().cpu().double().tolist())
            observations, *_ = env.step(policy(observations))
        rows = select_self_test_rows(kind, recorded)
        batch = torch.tensor(rows, dtype=torch.float32)
        actions = policy(TensorDict({"actor": batch}, batch_size=[len(rows)]))
    return rows, actions.detach().cpu().double().tolist()


def contract_metadata(
    kind: str,
    *,
    joint_names: Sequence[str],
    checkpoint: Path,
    checkpoint_sha256: str,
    gate_report_sha256: str,
    self_test_observations: Sequence[Sequence[float]],
    self_test_actions: Sequence[Sequence[float]],
    dry_run: bool,
) -> dict[str, str]:
    """The metadata every policy of ``kind`` carries (all values are strings)."""

    if kind not in RECIPES:
        raise PolicyContractError(f"unknown policy kind {kind!r}")
    names = [str(name) for name in joint_names]
    if len(names) != len(HOME.joint_pos_rad) or set(names) != set(HOME.joint_pos_rad):
        raise PolicyContractError("joint_names must name the 21 HOME joints once")
    stem = Path(checkpoint).name
    if not (stem.startswith("model_") and stem.endswith(".pt") and stem[6:-3].isdigit()):
        raise PolicyContractError(f"checkpoint {stem!r} is not model_N.pt")
    iteration = int(stem[6:-3])
    for key, value in (("checkpoint_sha256", checkpoint_sha256), ("gate_report_sha256", gate_report_sha256)):
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise PolicyContractError(f"{key} must be a lowercase SHA-256")
    rows = [[float(value) for value in row] for row in self_test_observations]
    actions = [[float(value) for value in row] for row in self_test_actions]
    if not SELF_TEST_MIN_ROWS <= len(rows) <= SELF_TEST_MAX_ROWS or len(actions) != len(rows):
        raise PolicyContractError(f"self-test needs {SELF_TEST_MIN_ROWS}..{SELF_TEST_MAX_ROWS} rows and one "
                                  "action row each")
    for index, row in enumerate(rows):
        problem = physical_row_problem(kind, row)
        if problem is not None:
            raise PolicyContractError(f"self-test row {index}: {problem}")
    if any(len(row) != ACTION_WIDTH or not all(math.isfinite(v) for v in row) for row in actions):
        raise PolicyContractError(f"self-test actions must be finite rows of {ACTION_WIDTH}")
    if kind == "pico":
        terms = _term_slices(kind)
        for term in ("foot_target", "arm_target"):
            if not any(any(value != 0.0 for value in row[terms[term]]) for row in rows):
                raise PolicyContractError(f"the PICO self-test has no row with a non-zero {term}")
    metadata = {
        "microban_policy_contract": POLICY_CONTRACT,
        "microban_policy_kind": kind,
        "microban_recipe": RECIPES[kind],
        "home_pose": json.dumps(home_pose_stamp(), sort_keys=True, separators=(",", ":")),
        "joint_names": ",".join(names),
        "default_joint_pos": _csv([HOME.joint_pos_rad[name] for name in names]),
        "action_joint_names": ",".join(ACTION_JOINT_NAMES),
        "action_scale": "1.0",
        "action_clip_lower": _csv([-SERVO_TARGET_RANGE_RAD] * ACTION_WIDTH),
        "action_clip_upper": _csv([SERVO_TARGET_RANGE_RAD] * ACTION_WIDTH),
        "observation_schema_json": _json([list(term) for term in OBSERVATION_SCHEMAS[kind]]),
        "observation_joint_names": ",".join(OBSERVATION_JOINT_NAMES[kind]),
        "previous_action_semantics": PREVIOUS_ACTION_SEMANTICS,
        "base_ang_vel_frame": BASE_ANG_VEL_FRAME,
        "control_hz": str(CONTROL_HZ),
        "checkpoint_filename": stem,
        "checkpoint_iteration": str(iteration),
        "checkpoint_sha256": checkpoint_sha256,
        "gate_status": "pass",
        "gate_report_sha256": gate_report_sha256,
        "self_test_observations_json": _json(rows),
        "self_test_actions_json": _json(actions),
    }
    if dry_run:
        metadata[DRY_RUN_METADATA_KEY] = "true"
    return metadata


def pico_arm_target_record() -> dict[str, Any]:
    """The arm-target contract the robot checks (pico_arm_target_json)."""

    return {
        "contract": PICO_ARM_TARGET_CONTRACT,
        "joint_names": list(PICO_ARM_JOINT_NAMES),
        "lower_rad": list(PICO_ARM_LOWER_RAD),
        "upper_rad": list(PICO_ARM_UPPER_RAD),
        "slew_rad_s": PICO_ARM_SLEW_RATE_RAD_S,
    }


def pico_metadata(
    *,
    walk_checkpoint_sha256: str,
    target_frame: str,
    raw_action_guard: Sequence[float],
    curriculum: Mapping[str, int],
    active_adapter_columns: Sequence[int],
    checkpoint_iteration: int,
) -> dict[str, str]:
    """PICO's additional keys."""

    guard = [float(value) for value in raw_action_guard]
    with np.errstate(over="ignore"):
        guard32 = np.asarray(guard, dtype=np.float32)
    if len(guard) != ACTION_WIDTH or not np.isfinite(guard32).all() or min(guard) <= 0.0:
        raise PolicyContractError("the PICO raw-action guard must be 18 positive finite float32 values")
    record = {key: curriculum[key] for key in PICO_CURRICULUM_KEYS}
    if not 0 < record["arm_start"] <= record["foot_start"] <= record["foot_tighten"] <= record["total"]:
        raise PolicyContractError(f"PICO curriculum {record} is not ordered")
    if not record["foot_tighten"] <= checkpoint_iteration + 1 <= record["total"]:
        raise PolicyContractError("the PICO checkpoint is not from the final curriculum stage")
    if list(active_adapter_columns) != PICO_ADAPTER_COLUMNS:
        raise PolicyContractError("the PICO checkpoint did not train every adapter column")
    if len(walk_checkpoint_sha256) != 64:
        raise PolicyContractError("pico_walk_checkpoint_sha256 must be a SHA-256")
    return {
        "pico_walk_checkpoint_sha256": walk_checkpoint_sha256,
        "pico_target_frame": target_frame,
        "pico_foot_target_lower_json": _json(PICO_FOOT_TARGET_LOWER),
        "pico_foot_target_upper_json": _json(PICO_FOOT_TARGET_UPPER),
        "pico_both_feet_target_lower_json": _json(PICO_BOTH_FEET_TARGET_LOWER),
        "pico_both_feet_target_upper_json": _json(PICO_BOTH_FEET_TARGET_UPPER),
        "pico_arm_target_json": _json(pico_arm_target_record()),
        "pico_raw_action_guard_json": _json(guard),
        "pico_curriculum_json": _json(record),
        "pico_active_adapter_columns_json": _json(list(active_adapter_columns)),
    }


def check_self_test(onnx_path: Path, metadata: Mapping[str, str]) -> float:
    """Run the recorded rows through ONNX Runtime (CPU) as the robot does.

    Returns the largest error relative to its bound (<= 1 passes).
    """

    import onnxruntime as ort

    rows = np.asarray(json.loads(metadata["self_test_observations_json"]), dtype=np.float32)
    expected = np.asarray(json.loads(metadata["self_test_actions_json"]), dtype=np.float64)
    guard = (np.asarray(json.loads(metadata["pico_raw_action_guard_json"]), dtype=np.float64)
             if "pico_raw_action_guard_json" in metadata else None)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    worst = 0.0
    for index, (row, want) in enumerate(zip(rows, expected, strict=True)):
        output = np.asarray(session.run(None, {"obs": row[None, :]})[0], dtype=np.float64)
        if output.shape != (1, ACTION_WIDTH) or not np.isfinite(output).all():
            raise PolicyContractError(f"self-test row {index}: unsafe ONNX output")
        bound = SELF_TEST_ATOL + SELF_TEST_RTOL * float(np.max(np.abs(want)))
        worst = max(worst, float(np.max(np.abs(output[0] - want))) / bound)
        if guard is not None and np.any(np.abs(output[0]) > guard):
            raise PolicyContractError(f"self-test row {index} exceeds the PICO raw-action guard")
    if worst > 1.0:
        raise PolicyContractError(f"ONNX Runtime differs from the recorded torch outputs ({worst:.3g} x bound)")
    return worst


def gate_report(kind: str, path: Path, checkpoint_sha256: str, *, dry_run: bool) -> str:
    """Check a pipeline gate report (``write_gate_report``); return its SHA-256.

    Dry-run evidence (a forced pass) is accepted only with ``dry_run``.
    """

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {"contract": POLICY_CONTRACT, "kind": kind, "status": "pass", "checkpoint_sha256": checkpoint_sha256}
    wrong = [key for key, value in expected.items() if data.get(key) != value]
    if wrong:
        raise PolicyContractError(f"{path} is not a passed {kind} gate of this checkpoint ({wrong})")
    if data.get("dry_run") and not dry_run:
        raise PolicyContractError(f"{path} is dry-run evidence; only a dry run exports it")
    return sha256_file(path)


def write_gate_report(path: Path, kind: str, checkpoint_sha256: str, *, passed: bool, dry_run: bool,
                      evidence: Mapping[str, Any]) -> None:
    """The judgment of one policy as its exporter checks it.

    A dry run whose (few-update) policy failed records a forced pass marked
    ``dry_run`` with the real outcome in ``passed``.
    """

    if not passed and not dry_run:
        raise PolicyContractError(f"{kind} failed its judgment: no gate report")
    report = {"contract": POLICY_CONTRACT, "kind": kind, "status": "pass", "checkpoint_sha256": checkpoint_sha256,
              "dry_run": dry_run, "passed": passed, "evidence": evidence}
    Path(path).write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")


def manifest(*, files: Mapping[str, Path], checkpoint_sha256: Mapping[str, str], training_commit: str,
             dry_run: bool, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """``src/agents/manifest.json`` of one release (extra keys are not read by the robot)."""

    if set(files) != set(KINDS) or set(checkpoint_sha256) != set(KINDS):
        raise PolicyContractError(f"a release has exactly {list(KINDS)}")
    for kind, path in files.items():
        if Path(path).name != POLICY_FILES[kind]:
            raise PolicyContractError(f"{kind} must be installed as {POLICY_FILES[kind]}")
    return {
        "contract": POLICY_CONTRACT,
        "home_tag": HOME.tag,
        "training_commit": training_commit,
        "dry_run": bool(dry_run),
        "policies": {kind: {"file": POLICY_FILES[kind], "sha256": sha256_file(files[kind]),
                            "checkpoint_sha256": checkpoint_sha256[kind]} for kind in KINDS},
        **(dict(extra) if extra else {}),
    }
