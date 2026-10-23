"""``microban-wholebody-1``: what the whole-body ONNX carries for the robot.

A contract of its own (not ``microban-policy-1``: that release holds exactly
walk, get-up and IK tracking, each with one input).  It reuses that contract's
parts: the HOME stamp, the SHA-256 records and the startup self-test rule
(recorded rows with the torch actor's deterministic output, re-run in ONNX
Runtime within the same bounds).

Input: ``obs_history`` [1, 10, 118], the last 10 frames oldest first, the
current one last (one reading per control step); after a reset every row is
the first frame.  A frame is the 12 terms of OBSERVATION_SCHEMA in order.
Output: ``actions`` [1, 18], the raw action of ACTION_JOINT_NAMES: target =
HOME + action, saturated at +-pi.

Reference.  The ``ref_*`` terms of the frame read at a control tick are the
reference one control period later (the target for the state after this
step).  Its yaw is turned about z so that, when tracking starts, the
reference trunk's yaw is the robot IMU's yaw (Teleopit's
align_motion_qpos_yaw); the ``*_b`` terms use the IMU's trunk orientation,
yaw included.

Neck.  head, neck_roll and neck_pitch are not the policy's; the policy
observes them.  Their targets are the reference's angles of these joints,
clamped to NECK_LIMITS_RAD within the joint range and slewed at
NECK_SLEW_RATES_RAD_S, starting at the clamped reference: the same
computation as in training (wholebody.motion.ReferenceNeckTarget), not the
HMD head tracking of the IK tracking policy.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from mjlab_microban.policy_contract import (
    ACTION_JOINT_NAMES,
    BASE_ANG_VEL_FRAME,
    CONTROL_HZ,
    HEAD_JOINTS,
    PREVIOUS_ACTION_SEMANTICS,
    SELF_TEST_ATOL,
    SELF_TEST_GRAVITY_NORM_TOLERANCE,
    SELF_TEST_JOINT_RANGE_MARGIN_RAD,
    SELF_TEST_MAX_JOINT_SPEED_RAD_S,
    SELF_TEST_MAX_ROWS,
    SELF_TEST_MIN_ROWS,
    SELF_TEST_RTOL,
    SERVO_TARGET_RANGE_RAD,
    PolicyContractError,
    _csv,
    _json,
    home_pose_stamp,
    joint_ranges,
)
from mjlab_microban.robot.home_pose import HOME
from mjlab_microban.tasks.microban_ik_tracking_mdp import (
    MICROBAN_HMD_RUNTIME_LIMITS_RAD,
    MICROBAN_HMD_SLEW_RATES_RAD_S,
)
from mjlab_microban.wholebody.env_cfg import HISTORY_LENGTH

CONTRACT = "microban-wholebody-1"
KIND = "wholebody"
RECIPE = "microban-wholebody-teleopit-track-1"
POLICY_FILE = "wholebody.onnx"
INPUT_NAMES = ("obs_history",)
OUTPUT_NAME = "actions"
NECK_LIMITS_RAD = {name: MICROBAN_HMD_RUNTIME_LIMITS_RAD[name] for name in HEAD_JOINTS}
NECK_SLEW_RATES_RAD_S = {name: MICROBAN_HMD_SLEW_RATES_RAD_S[name] for name in HEAD_JOINTS}
OBSERVATION_JOINT_NAMES = (*HEAD_JOINTS, *ACTION_JOINT_NAMES)
REFERENCE_JOINT_NAMES = ACTION_JOINT_NAMES
OBSERVATION_SCHEMA: tuple[tuple[str, int], ...] = (
    ("ref_joint_pos", 18),
    ("ref_joint_vel", 18),
    ("ref_anchor_ori_b", 6),
    ("base_ang_vel", 3),
    ("joint_pos", 21),
    ("joint_vel", 21),
    ("actions", 18),
    ("projected_gravity", 3),
    ("ref_anchor_lin_vel_b", 3),
    ("ref_anchor_ang_vel_b", 3),
    ("ref_projected_gravity_b", 3),
    ("ref_anchor_height", 1),
)
OBSERVATION_WIDTH = sum(width for _, width in OBSERVATION_SCHEMA)
# What each term means (trunk = the IMU body; "reference" = the retargeted pose).
OBSERVATION_SEMANTICS = {
    "ref_joint_pos": "reference angles of REFERENCE_JOINT_NAMES, absolute, rad",
    "ref_joint_vel": "reference joint velocities of REFERENCE_JOINT_NAMES, rad/s",
    "ref_anchor_ori_b": "R = rotation of the reference trunk in the robot trunk frame; R[:, :2] row-major",
    "base_ang_vel": "gyro, rad/s, IMU frame",
    "joint_pos": "measured angle minus HOME of OBSERVATION_JOINT_NAMES, rad",
    "joint_vel": "measured joint velocities of OBSERVATION_JOINT_NAMES, rad/s",
    "actions": "the policy's previous raw output",
    "projected_gravity": "unit gravity in the robot trunk frame",
    "ref_anchor_lin_vel_b": "reference trunk linear velocity (world) rotated into the robot trunk frame, m/s",
    "ref_anchor_ang_vel_b": "reference trunk angular velocity (world) rotated into the robot trunk frame, rad/s",
    "ref_projected_gravity_b": "unit gravity in the reference trunk frame",
    "ref_anchor_height": "reference trunk height above the floor, m",
}


def _slices() -> dict[str, slice]:
    out, start = {}, 0
    for name, width in OBSERVATION_SCHEMA:
        out[name] = slice(start, start + width)
        start += width
    return out


def frame_problem(row: Sequence[float]) -> str | None:
    """Why one frame is not a possible robot state (the self-test row rule), or None."""
    values = np.asarray(row, dtype=np.float64)
    if values.shape != (OBSERVATION_WIDTH,):
        return f"width {values.shape} is not {OBSERVATION_WIDTH}"
    if not np.isfinite(values).all():
        return "non-finite value"
    terms = _slices()
    for name in ("projected_gravity", "ref_projected_gravity_b"):
        if abs(float(np.linalg.norm(values[terms[name]])) - 1.0) > SELF_TEST_GRAVITY_NORM_TOLERANCE:
            return f"{name} is not a unit vector"
    if np.any(np.abs(values[terms["joint_vel"]]) > SELF_TEST_MAX_JOINT_SPEED_RAD_S):
        return "joint speed above the servo's no-load speed"
    ranges = joint_ranges()
    for name, residual in zip(OBSERVATION_JOINT_NAMES, values[terms["joint_pos"]], strict=True):
        lower, upper = ranges[name]
        angle = residual + HOME.joint_pos_rad[name]
        if name not in HEAD_JOINTS and not (
            lower - SELF_TEST_JOINT_RANGE_MARGIN_RAD <= angle <= upper + SELF_TEST_JOINT_RANGE_MARGIN_RAD
        ):
            return f"{name} outside its range"
    return None


def select_self_test_rows(rows: Sequence[Sequence[Sequence[float]]], count: int = 32) -> list[list[list[float]]]:
    """At most ``count`` histories whose every frame is a possible state, evenly spread."""
    usable = []
    for history in rows:
        frames = [list(map(float, frame)) for frame in history]
        if len(frames) == HISTORY_LENGTH and all(frame_problem(f) is None for f in frames):
            usable.append(frames)
    if len(usable) < SELF_TEST_MIN_ROWS:
        raise PolicyContractError(
            f"self-test: only {len(usable)} of {len(rows)} recorded rows are possible robot states "
            f"(need {SELF_TEST_MIN_ROWS})"
        )
    if len(usable) <= count:
        return usable
    picks = np.linspace(0, len(usable) - 1, count).round().astype(int)
    return [usable[index] for index in picks]


def export_policy_to_onnx(policy, path: Path) -> None:
    """Teleopit's export (rl/runner.py export_policy_to_onnx): opset 18, the
    normalizers inside, the deterministic mean action."""
    import torch

    model = policy.as_onnx(verbose=False)
    model.to("cpu")
    model.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        model.get_dummy_inputs(),
        str(path),
        export_params=True,
        opset_version=18,
        verbose=False,
        input_names=model.input_names,
        output_names=model.output_names,
        dynamic_axes={},
        dynamo=False,
    )


def onnx_io(path: Path) -> dict[str, tuple[int, ...]]:
    """Input and output names with their static shapes."""
    import onnx

    model = onnx.load(str(path))
    shapes = {}
    for value in (*model.graph.input, *model.graph.output):
        shapes[value.name] = tuple(d.dim_value for d in value.type.tensor_type.shape.dim)
    return shapes


def check_onnx_io(path: Path) -> None:
    expected = {
        "obs_history": (1, HISTORY_LENGTH, OBSERVATION_WIDTH),
        OUTPUT_NAME: (1, len(ACTION_JOINT_NAMES)),
    }
    if onnx_io(path) != expected:
        raise PolicyContractError(f"{path} has inputs/outputs {onnx_io(path)}, expected {expected}")


def contract_metadata(
    *,
    joint_names: Sequence[str],
    checkpoint: Path,
    checkpoint_sha256: str,
    benchmark_report_sha256: str,
    eval_only_benchmark_report_sha256: str,
    self_test_rows: Sequence[Sequence[Sequence[float]]],
    self_test_actions: Sequence[Sequence[float]],
    dry_run: bool,
) -> dict[str, str]:
    """The metadata of ``wholebody.onnx`` (all values are strings)."""
    names = [str(name) for name in joint_names]
    if len(names) != len(HOME.joint_pos_rad) or set(names) != set(HOME.joint_pos_rad):
        raise PolicyContractError("joint_names must name the 21 HOME joints once")
    stem = Path(checkpoint).name
    if not (stem.startswith("model_") and stem.endswith(".pt") and stem[6:-3].isdigit()):
        raise PolicyContractError(f"checkpoint {stem!r} is not model_N.pt")
    for key, value in (
        ("checkpoint_sha256", checkpoint_sha256),
        ("benchmark_report_sha256", benchmark_report_sha256),
        ("eval_only_benchmark_report_sha256", eval_only_benchmark_report_sha256),
    ):
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise PolicyContractError(f"{key} must be a lowercase SHA-256")
    if not SELF_TEST_MIN_ROWS <= len(self_test_rows) <= SELF_TEST_MAX_ROWS or len(self_test_actions) != len(
        self_test_rows
    ):
        raise PolicyContractError(f"self-test needs {SELF_TEST_MIN_ROWS}..{SELF_TEST_MAX_ROWS} rows, one action each")
    for index, history in enumerate(self_test_rows):
        problem = None if len(history) == HISTORY_LENGTH else "history length"
        problem = problem or next((p for p in map(frame_problem, history) if p), None)
        if problem:
            raise PolicyContractError(f"self-test row {index}: {problem}")
    actions = [[float(v) for v in row] for row in self_test_actions]
    if any(len(row) != len(ACTION_JOINT_NAMES) or not all(math.isfinite(v) for v in row) for row in actions):
        raise PolicyContractError("self-test actions must be finite rows of 18")
    metadata = {
        "microban_policy_contract": CONTRACT,
        "microban_policy_kind": KIND,
        "microban_recipe": RECIPE,
        "home_pose": json.dumps(home_pose_stamp(), sort_keys=True, separators=(",", ":")),
        "joint_names": ",".join(names),
        "default_joint_pos": _csv([HOME.joint_pos_rad[name] for name in names]),
        "action_joint_names": ",".join(ACTION_JOINT_NAMES),
        "action_scale": "1.0",
        "action_clip_lower": _csv([-SERVO_TARGET_RANGE_RAD] * len(ACTION_JOINT_NAMES)),
        "action_clip_upper": _csv([SERVO_TARGET_RANGE_RAD] * len(ACTION_JOINT_NAMES)),
        "input_names": ",".join(INPUT_NAMES),
        "output_name": OUTPUT_NAME,
        "observation_schema_json": _json([list(term) for term in OBSERVATION_SCHEMA]),
        "observation_semantics_json": _json(OBSERVATION_SEMANTICS),
        "observation_joint_names": ",".join(OBSERVATION_JOINT_NAMES),
        "reference_joint_names": ",".join(REFERENCE_JOINT_NAMES),
        "history_length": str(HISTORY_LENGTH),
        "history_order": "oldest_first_current_last",
        "history_after_reset": "every_row_the_first_frame",
        "reference_time": "one_control_period_ahead",
        "reference_yaw": "aligned_to_imu_yaw_at_start",
        "neck_joint_names": ",".join(HEAD_JOINTS),
        "neck_control": "reference_clamped_slewed",
        "neck_limits_rad_json": _json({name: list(NECK_LIMITS_RAD[name]) for name in HEAD_JOINTS}),
        "neck_slew_rates_rad_s_json": _json(NECK_SLEW_RATES_RAD_S),
        "previous_action_semantics": PREVIOUS_ACTION_SEMANTICS,
        "base_ang_vel_frame": BASE_ANG_VEL_FRAME,
        "control_hz": str(CONTROL_HZ),
        "checkpoint_filename": stem,
        "checkpoint_iteration": str(int(stem[6:-3])),
        "checkpoint_sha256": checkpoint_sha256,
        "benchmark_report_sha256": benchmark_report_sha256,
        "eval_only_benchmark_report_sha256": eval_only_benchmark_report_sha256,
        "self_test_histories_json": _json([[list(f) for f in history] for history in self_test_rows]),
        "self_test_actions_json": _json(actions),
    }
    if dry_run:
        metadata["dry_run_not_deployable"] = "true"
    return metadata


def check_self_test(onnx_path: Path, metadata: dict[str, str]) -> float:
    """Run the recorded rows through ONNX Runtime (CPU); return the worst error / bound (<= 1 passes)."""
    import onnxruntime as ort

    history = np.asarray(json.loads(metadata["self_test_histories_json"]), dtype=np.float32)
    expected = np.asarray(json.loads(metadata["self_test_actions_json"]), dtype=np.float64)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    worst = 0.0
    for index in range(len(history)):
        output = session.run(None, {"obs_history": history[index : index + 1]})[0]
        output = np.asarray(output, dtype=np.float64)
        if output.shape != (1, len(ACTION_JOINT_NAMES)) or not np.isfinite(output).all():
            raise PolicyContractError(f"self-test row {index}: unsafe ONNX output")
        bound = SELF_TEST_ATOL + SELF_TEST_RTOL * float(np.max(np.abs(expected[index])))
        worst = max(worst, float(np.max(np.abs(output[0] - expected[index]))) / bound)
    if worst > 1.0:
        raise PolicyContractError(f"ONNX Runtime differs from the recorded torch outputs ({worst:.3g} x bound)")
    return worst
