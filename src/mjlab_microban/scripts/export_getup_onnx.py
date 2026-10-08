"""Export a newly trained Microban get-up checkpoint for the robot.

Both paths must be explicit.  The checkpoint's get-up contract (its stamp and
the run's recorded params/env.yaml: the current HOME, +-pi servo-range clip,
raw previous-action feedback) and the exported ONNX normalizer are checked
before an artifact is published.  The metadata is the robot's contract
microban-policy-1 (policy_contract.py): the gate report is the pipeline's
judgment of this checkpoint, and the startup self-test rows come from a seeded
rollout of the checkpoint in the play env.
"""

from __future__ import annotations

import argparse
import os
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np
import onnx
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from onnx import numpy_helper
from onnx.reference import ReferenceEvaluator

from mjlab_microban import policy_contract
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_TELEOP_ACTION_JOINT_NAMES,
)
from mjlab_microban.robot.microban_constants import (
    HOME_PROJECTED_GRAVITY,
    SERVO_TARGET_RANGE_RAD,
)
from mjlab_microban.tasks.microban_getup_env_cfg import GETUP_ACTION_CLIP_RAD
from mjlab_microban.tasks.microban_getup_runner import require_getup_checkpoint_contract

TASK = "Mjlab-Getup-Microban"
OBSERVATION_TERMS = (
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
)
OBSERVATION_WIDTH = 60
ACTION_WIDTH = len(MICROBAN_TELEOP_ACTION_JOINT_NAMES)


def _require_new_checkpoint(path: Path) -> tuple[str, str]:
    """Require a current-contract training checkpoint; return its SHA-256 and stamp.

    Validity comes from the run's recorded env (the current HOME, the +-pi
    clip, raw previous-action feedback).
    """

    before = policy_contract.sha256_file(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    stamp = require_getup_checkpoint_contract(
        path, checkpoint.get("infos"), require_recorded_env=True
    )
    if not isinstance(checkpoint.get("actor_state_dict"), dict):
        raise ValueError("Checkpoint has no actor_state_dict")
    if policy_contract.sha256_file(path) != before:
        raise ValueError("Checkpoint changed during inspection")
    return before, stamp


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


def _validate_onnx(path: Path) -> None:
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

    # Finite, positive normalizer. No range check on the previous-action
    # slots: the policy observes the raw output, which is unbounded by design (a
    # standing policy drives it to hundreds of radians to saturate the clip).
    _normalizer_arrays(model)

    # Screen a small set of physically meaningful initial orientations for
    # non-finite output, the runtime's only actor fault.
    evaluator = ReferenceEvaluator(model)
    # HOME's own gravity (a leaning HOME trunk reads (sin p, 0, -cos p)), then
    # vertical upright, upside down and on its side.
    gravities = [tuple(HOME_PROJECTED_GRAVITY)]
    for gravity in ((0.0, 0.0, -1.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)):
        if gravity not in gravities:
            gravities.append(gravity)
    for gravity in gravities:
        observation = np.zeros((1, OBSERVATION_WIDTH), dtype=np.float32)
        observation[0, 3:6] = gravity
        outputs = evaluator.run(None, {"obs": observation})
        if len(outputs) != 1 or outputs[0].shape != (1, ACTION_WIDTH):
            raise ValueError("ONNX action inference returned the wrong shape")
        if not np.isfinite(outputs[0]).all():
            raise ValueError(
                f"Get-up actor produces non-finite raw actions for gravity={gravity}"
            )


def _action_contract(env: ManagerBasedRlEnv) -> tuple[np.ndarray, np.ndarray]:
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
    if GETUP_ACTION_CLIP_RAD != SERVO_TARGET_RANGE_RAD or SERVO_TARGET_RANGE_RAD != float(np.pi):
        raise ValueError("Get-up action clip is not the servo goal range (+-pi)")
    if not np.allclose(lower, -SERVO_TARGET_RANGE_RAD, rtol=0, atol=1e-6) or not np.allclose(
        upper, SERVO_TARGET_RANGE_RAD, rtol=0, atol=1e-6
    ):
        raise ValueError("Get-up action target clip differs from the servo range (+-pi)")
    # The env holds the clip as float32 (3.1415927 > pi); publish the exact
    # float64 bound so the robot's "not wider than the servo range" check holds.
    lower = np.full(ACTION_WIDTH, -SERVO_TARGET_RANGE_RAD, dtype=np.float64)
    upper = np.full(ACTION_WIDTH, SERVO_TARGET_RANGE_RAD, dtype=np.float64)
    return lower, upper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="get-up model_N.pt (in its run directory with params/env.yaml)")
    parser.add_argument("--output", required=True, type=Path, help="Destination ONNX artifact")
    parser.add_argument("--gate-report", required=True, type=Path,
                        help="The pipeline's passed get-up gate report of this checkpoint")
    parser.add_argument("--dry-run", action="store_true",
                        help="Accept a dry run's forced gate and mark the ONNX not deployable")
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
    checkpoint_sha256, _stamp = _require_new_checkpoint(checkpoint)
    gate_report_sha256 = policy_contract.gate_report("getup", args.gate_report, checkpoint_sha256,
                                                     dry_run=args.dry_run)

    env_cfg = load_env_cfg(TASK, play=True)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = policy_contract.SELF_TEST_SEED
    agent_cfg = load_rl_cfg(TASK)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    wrapped = RslRlVecEnvWrapper(env)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp.onnx")
    try:
        _action_contract(env)
        robot = env.scene["robot"]
        default = robot.data.default_joint_pos[0].detach().cpu().numpy().astype(np.float64)
        home = np.array([policy_contract.HOME.joint_pos_rad[name] for name in robot.joint_names])
        if not np.allclose(default, home, rtol=0, atol=1e-6):
            raise ValueError("Get-up env default joint pose is not the current HOME")
        runner = load_runner_cls(TASK)(wrapped, asdict(agent_cfg), device=args.device)
        runner.load(
            str(checkpoint), load_cfg={"actor": True}, strict=True,
            map_location=args.device,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        runner.export_policy_to_onnx(str(temporary.parent), temporary.name)
        _validate_onnx(temporary)
        policy = runner.get_inference_policy(device="cpu")
        rows, actions = policy_contract.rollout_self_test("getup", wrapped, policy)
        metadata = policy_contract.contract_metadata(
            "getup",
            joint_names=robot.joint_names,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            gate_report_sha256=gate_report_sha256,
            self_test_observations=rows,
            self_test_actions=actions,
            dry_run=args.dry_run,
        )
        attach_metadata_to_onnx(str(temporary), metadata)
        _validate_onnx(temporary)
        self_test = policy_contract.check_self_test(temporary, metadata)
        if policy_contract.sha256_file(checkpoint) != checkpoint_sha256:
            raise ValueError("Checkpoint changed during export")
        if args.replace:
            os.replace(temporary, output)
        else:
            # Both paths are in one directory.  Linking publishes the finished
            # file atomically and fails if another process created the name.
            os.link(temporary, output)
            temporary.unlink()
        print(f"self-test {len(rows)} rows at {self_test:.3f} x bound")
        print(f"Exported validated get-up ONNX: {output}")
    finally:
        temporary.unlink(missing_ok=True)
        env.close()


if __name__ == "__main__":
    main()
