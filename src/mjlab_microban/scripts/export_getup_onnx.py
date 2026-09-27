"""Export a newly trained Microban get-up checkpoint for the robot.

Both paths must be explicit.  The checkpoint's v3 training marker and the
exported ONNX normalizer are checked before an artifact is published.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np
import onnx
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from onnx import numpy_helper
from onnx.reference import ReferenceEvaluator

from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_TELEOP_ACTION_JOINT_NAMES,
)
from mjlab_microban.tasks.microban_tracking_env_cfg import MICROBAN_BODY_JOINT_SOFT_LIMITS
from mjlab_microban.tasks.microban_getup_runner import (
    GETUP_ANGULAR_VELOCITY_FRAME,
    getup_home_pose,
    require_current_getup_home_pose,
)

TASK = "Mjlab-Getup-Microban"
CONTRACT_VERSION = "v3"
OBSERVATION_TERMS = (
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
)
OBSERVATION_WIDTH = 60
ACTION_WIDTH = len(MICROBAN_TELEOP_ACTION_JOINT_NAMES)
PREVIOUS_ACTION_START = OBSERVATION_WIDTH - ACTION_WIDTH
_STATS_TOLERANCE_RAD = 0.05
_RUNTIME_MAX_RAW_ACTION = 120.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_new_checkpoint(path: Path) -> str:
    """Require a fresh v3 training checkpoint, not an old actor relabeled v3."""

    before = _sha256(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    infos = checkpoint.get("infos")
    if not isinstance(infos, dict) or (
        infos.get("microban_getup_contract") != CONTRACT_VERSION
        or infos.get("microban_getup_angular_velocity_frame") != GETUP_ANGULAR_VELOCITY_FRAME
    ):
        raise ValueError(
            "Checkpoint lacks the get-up v3 / IMU-frame training marker; "
            "retrain from scratch with the current Mjlab-Getup-Microban task"
        )
    require_current_getup_home_pose(infos)
    if not isinstance(checkpoint.get("actor_state_dict"), dict):
        raise ValueError("Checkpoint has no actor_state_dict")
    if _sha256(path) != before:
        raise ValueError("Checkpoint changed during inspection")
    return before


def _tensor_shape(value: onnx.ValueInfoProto) -> tuple[int | str, ...]:
    shape: list[int | str] = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            shape.append(dim.dim_param)
        else:
            shape.append("?")
    return tuple(shape)


def _normalizer_arrays(model: onnx.ModelProto) -> tuple[np.ndarray, np.ndarray]:
    """Read the actual Sub/Div constants on the ONNX observation input path."""

    initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    sub_nodes = [
        node
        for node in model.graph.node
        if node.op_type == "Sub" and len(node.input) == 2 and node.input[0] == "obs"
    ]
    if len(sub_nodes) != 1 or sub_nodes[0].input[1] not in initializers:
        raise ValueError("Cannot identify ONNX observation normalizer mean")
    mean = initializers[sub_nodes[0].input[1]]
    div_nodes = [
        node
        for node in model.graph.node
        if node.op_type == "Div"
        and len(node.input) == 2
        and node.input[0] == sub_nodes[0].output[0]
    ]
    if len(div_nodes) != 1 or div_nodes[0].input[1] not in initializers:
        raise ValueError("Cannot identify ONNX observation normalizer divisor")
    std = initializers[div_nodes[0].input[1]]
    if mean.shape != (1, OBSERVATION_WIDTH) or std.shape != (1, OBSERVATION_WIDTH):
        raise ValueError(f"Invalid ONNX normalizer shapes: {mean.shape}, {std.shape}")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("ONNX observation normalizer is nonfinite or nonpositive")
    return mean[0], std[0]


def _validate_onnx(
    path: Path,
    *,
    feedback_lower: np.ndarray,
    feedback_upper: np.ndarray,
    default: np.ndarray,
    scale: np.ndarray,
) -> None:
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    for initializer in model.graph.initializer:
        values = numpy_helper.to_array(initializer)
        if np.issubdtype(values.dtype, np.floating) and not np.isfinite(values).all():
            raise ValueError(f"Nonfinite ONNX initializer: {initializer.name}")
    if len(model.graph.input) != 1 or (
        model.graph.input[0].name,
        _tensor_shape(model.graph.input[0]),
    ) != ("obs", (1, OBSERVATION_WIDTH)):
        raise ValueError("Get-up ONNX must have one obs input of shape [1, 60]")
    if len(model.graph.output) != 1 or (
        model.graph.output[0].name,
        _tensor_shape(model.graph.output[0]),
    ) != ("actions", (1, ACTION_WIDTH)):
        raise ValueError("Get-up ONNX must have one actions output of shape [1, 18]")

    mean, std = _normalizer_arrays(model)
    action_mean = mean[PREVIOUS_ACTION_START:]
    action_std = std[PREVIOUS_ACTION_START:]
    # V3 observes the applied (clipped, unlimited-rate) target directly, which
    # is always within the action's own clip -- no slew transient to widen the
    # expected range for.
    minimum = (feedback_lower - default) / scale
    maximum = (feedback_upper - default) / scale
    if (
        np.any(action_mean < minimum - _STATS_TOLERANCE_RAD)
        or np.any(action_mean > maximum + _STATS_TOLERANCE_RAD)
        or np.any(action_std > (maximum - minimum) / 2 + _STATS_TOLERANCE_RAD)
    ):
        raise ValueError(
            "ONNX previous-action normalizer contradicts the reachable applied-target "
            f"range (max |mean|={np.max(np.abs(action_mean)):.3g}, "
            f"max std={np.max(action_std):.3g}, "
            f"max permitted |action|={np.max(np.maximum(np.abs(minimum), np.abs(maximum))):.3g})"
        )

    # Screen a small set of physically meaningful initial orientations.  This
    # catches a model that passes the statistics gate yet immediately triggers
    # the deployment runtime's raw-action fault threshold.
    evaluator = ReferenceEvaluator(model)
    for gravity in ((0.0, 0.0, -1.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)):
        observation = np.zeros((1, OBSERVATION_WIDTH), dtype=np.float32)
        observation[0, 3:6] = gravity
        outputs = evaluator.run(None, {"obs": observation})
        if len(outputs) != 1 or outputs[0].shape != (1, ACTION_WIDTH):
            raise ValueError("ONNX action inference returned the wrong shape")
        actions = outputs[0]
        if not np.isfinite(actions).all() or np.max(np.abs(actions)) > _RUNTIME_MAX_RAW_ACTION:
            raise ValueError(
                "Get-up actor produces invalid raw actions for a canonical "
                f"orientation: gravity={gravity}, max_abs={np.max(np.abs(actions)):.3g}"
            )


def _action_contract(
    env: ManagerBasedRlEnv,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    action = env.action_manager.get_term("joint_pos")
    if tuple(action.target_names) != MICROBAN_TELEOP_ACTION_JOINT_NAMES:
        raise ValueError("Get-up action joint order differs from the robot runtime")
    terms = tuple(env.observation_manager.active_terms["actor"])
    if terms != OBSERVATION_TERMS:
        raise ValueError(f"Get-up actor observation terms changed: {terms}")
    if env.observation_manager.group_obs_dim["actor"] != (OBSERVATION_WIDTH,):
        raise ValueError("Get-up actor observation width changed")

    def vector(value: float | torch.Tensor) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().numpy()
            if array.shape == (1, ACTION_WIDTH):
                return array[0].astype(np.float64)
            if array.shape == (ACTION_WIDTH,):
                return array.astype(np.float64)
            raise ValueError(f"Unexpected get-up action tensor shape {array.shape}")
        return np.full(ACTION_WIDTH, float(value), dtype=np.float64)

    scale = vector(action._scale)
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Invalid get-up action scale")
    default = env.scene["robot"].data.default_joint_pos[0, action.target_ids]
    default = default.detach().cpu().numpy().astype(np.float64)
    offset = vector(action._offset)
    if not np.allclose(offset, default, rtol=0, atol=1e-5):
        raise ValueError("Get-up action offset is not the default joint pose")
    clip = action._clip.detach().cpu().numpy()
    if clip.shape != (1, ACTION_WIDTH, 2):
        raise ValueError(f"Unexpected get-up action clip shape {clip.shape}")
    lower = clip[0, :, 0].astype(np.float64)
    upper = clip[0, :, 1].astype(np.float64)
    if not np.isfinite(clip).all() or np.any(lower >= upper):
        raise ValueError("Invalid get-up action target clip")
    expected_lower = np.array(
        [MICROBAN_BODY_JOINT_SOFT_LIMITS[name][0] for name in action.target_names]
    )
    expected_upper = np.array(
        [MICROBAN_BODY_JOINT_SOFT_LIMITS[name][1] for name in action.target_names]
    )
    if not np.allclose(lower, expected_lower, rtol=0, atol=1e-5) or not np.allclose(
        upper, expected_upper, rtol=0, atol=1e-5
    ):
        raise ValueError(
            "Get-up action target clip differs from each joint's soft limit "
            "(MICROBAN_BODY_JOINT_SOFT_LIMITS)"
        )
    if hasattr(action.cfg, "max_target_speed_rad_s"):
        raise ValueError(
            "Get-up training action must not rate-limit the active policy's "
            "own target (see microban_getup_action.py's module docstring)"
        )
    # No slew transient to widen the expected previous-action range for: the
    # observed target is always exactly the action's own clip.
    return lower, upper, lower, upper, default, scale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Fresh v3 model_N.pt")
    parser.add_argument("--output", required=True, type=Path, help="Destination ONNX artifact")
    parser.add_argument("--replace", action="store_true", help="Replace an existing output")
    parser.add_argument("--device", default="cpu", help="Model load device (default: cpu)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve(strict=True)
    output = args.output.resolve()
    if checkpoint.suffix != ".pt" or output.suffix != ".onnx":
        raise ValueError("Expected a .pt checkpoint and .onnx output")
    if output.exists() and not args.replace:
        raise FileExistsError(f"Output already exists: {output} (use --replace explicitly)")
    checkpoint_sha256 = _require_new_checkpoint(checkpoint)

    env_cfg = load_env_cfg(TASK, play=True)
    env_cfg.scene.num_envs = 1
    agent_cfg = load_rl_cfg(TASK)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    wrapped = RslRlVecEnvWrapper(env)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp.onnx")
    try:
        lower, upper, feedback_lower, feedback_upper, default, scale = _action_contract(env)
        runner = load_runner_cls(TASK)(wrapped, asdict(agent_cfg), device=args.device)
        runner.load(
            str(checkpoint), load_cfg={"actor": True}, strict=True,
            map_location=args.device,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        runner.export_policy_to_onnx(str(temporary.parent), temporary.name)
        _validate_onnx(
            temporary,
            feedback_lower=feedback_lower,
            feedback_upper=feedback_upper,
            default=default,
            scale=scale,
        )
        metadata = get_base_metadata(env, run_path=checkpoint.parent.name)
        metadata.update(
            {
                "action_joint_names": list(MICROBAN_TELEOP_ACTION_JOINT_NAMES),
                "action_clip_lower": lower.tolist(),
                "action_clip_upper": upper.tolist(),
                "microban_getup_previous_action_semantics": (
                    "applied_target_delta_from_default"
                ),
                "microban_getup_previous_action_lower": (
                    (feedback_lower - default) / scale
                ).tolist(),
                "microban_getup_previous_action_upper": (
                    (feedback_upper - default) / scale
                ).tolist(),
                "microban_getup_contract": CONTRACT_VERSION,
                "microban_getup_angular_velocity_frame": GETUP_ANGULAR_VELOCITY_FRAME,
                "microban_getup_home_pose": json.dumps(
                    getup_home_pose(), sort_keys=True, separators=(",", ":")
                ),
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_filename": checkpoint.name,
            }
        )
        attach_metadata_to_onnx(str(temporary), metadata)
        _validate_onnx(
            temporary,
            feedback_lower=feedback_lower,
            feedback_upper=feedback_upper,
            default=default,
            scale=scale,
        )
        if _sha256(checkpoint) != checkpoint_sha256:
            raise ValueError("Checkpoint changed during export")
        if args.replace:
            os.replace(temporary, output)
        else:
            # Both paths are in one directory.  Linking publishes the finished
            # file atomically and fails if another process created the name.
            os.link(temporary, output)
            temporary.unlink()
        print(f"Exported validated get-up ONNX: {output}")
    finally:
        temporary.unlink(missing_ok=True)
        env.close()


if __name__ == "__main__":
    main()
