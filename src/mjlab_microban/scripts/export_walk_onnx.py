"""Export a Microban walking checkpoint for the robot's WalkMove.

Usage::

    uv run --locked python -m mjlab_microban.scripts.export_walk_onnx \\
        --checkpoint logs/rsl_rl/mjlab_microban_velocity/<run>/model_<N>.pt \\
        --output artifacts/<name>.onnx [--replace]

Both paths must be explicit.  The walking contract is the one shared by every
Microban policy: target = HOME + raw_action * 1.0 on the 18 body joints, with
no software clip, saturated only at the servo's one-turn goal range (+-pi rad), at
the HOME of config/home_pose.yaml (centered: trunk vertical; forward-lean: trunk
10 deg forward), and the previous-action observation is the
policy's own raw (unclipped) output.  The checkpoint's run directory must have
recorded that same HOME and clip, the live play env must match it, and the
exported ONNX is checked against the torch actor before an artifact is published.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np
import onnx
import torch
import yaml
# mjlab first: importing it loads the Microban task entry point, which must
# not start from a half-initialized microban_constants.
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.observations import last_action
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import get_base_metadata
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from onnx import numpy_helper
from onnx.reference import ReferenceEvaluator

from mjlab_microban.policy_contract import contract_metadata
from mjlab_microban.robot import home_contracts
from mjlab_microban.robot.microban_constants import (
    HOME_FRAME,
    HOME_PROJECTED_GRAVITY,
    HOME_TRUNK_PITCH_RAD,
    SERVO_TARGET_RANGE_RAD,
)
from mjlab_microban.tasks.microban_getup_runner import HOME_ROOT_RECORDED_ATOL, getup_home_pose
from mjlab_microban.tasks.microban_velocity_runner import require_walk_home_pose
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_TELEOP_ACTION_JOINT_NAMES,
)

TASK = "Mjlab-Velocity-Microban"
# HOME-bound (robot/home_contracts.py): "v3_centered_home_servo_range" at the
# centered HOME, "v4_forward_lean_home_servo_range" at the forward-lean HOME
# (v3's target rule there), "v3_/v4_<label>_<joint hash>_servo_range" at any
# other.  v2 was the centered HOME with a +-1.57 clip.
CONTRACT_VERSION = home_contracts.WALK_CONTRACT_VERSION
# WalkMove feeds back the ONNX model's own last raw output (mjlab last_action).
PREVIOUS_ACTION_SEMANTICS = "raw_policy_output"
ACTION_SCALE = 1.0
# Same layout WalkMove.build_observation assembles: gyro, gravity, joint
# position minus default, joint velocity, last raw action, (vx, vy, vyaw).
OBSERVATION_TERMS = (
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
    "command",
)
ACTION_JOINT_NAMES = MICROBAN_TELEOP_ACTION_JOINT_NAMES
ACTION_WIDTH = len(ACTION_JOINT_NAMES)
OBSERVATION_TERM_DIMS = (3, 3, ACTION_WIDTH, ACTION_WIDTH, ACTION_WIDTH, 3)
OBSERVATION_WIDTH = sum(OBSERVATION_TERM_DIMS)
COMMAND_NAMES = ("twist",)
PARITY_SAMPLES = 256
PARITY_TOLERANCE = 1e-4
# Base-metadata keys the robot (and mjlab's own auto-export) already use.
BASE_METADATA_KEYS = (
    "run_path",
    "joint_names",
    "joint_stiffness",
    "joint_damping",
    "default_joint_pos",
    "command_names",
    "observation_names",
    "action_scale",
)


def walk_home_pose() -> dict[str, object]:
    """The shared HOME, in the same JSON shape as get-up's metadata."""

    return getup_home_pose()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv(values: Sequence[object]) -> str:
    """CSV like mjlab's list_to_csv_str, but without its 3-decimal rounding.

    Readers parse each field with float(); repr keeps HOME exact (e.g. the
    0.0209158 rad hip pitch, which the 3-decimal form turns into 0.021).
    """

    return ",".join(
        repr(float(value)) if isinstance(value, (int, float)) else str(value)
        for value in values
    )


def _metadata_value(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return _csv(value)
    if isinstance(value, float):
        return repr(value)
    return str(value)


def build_walk_metadata(
    base: Mapping[str, object],
    *,
    action_clip_lower: Sequence[float],
    action_clip_upper: Sequence[float],
    checkpoint_sha256: str,
    checkpoint_filename: str,
    run_dir: str,
    iteration: int,
) -> dict[str, str]:
    """Build the string metadata map attached to a walking ONNX.

    ``base`` is mjlab's get_base_metadata(); its key names and CSV formats are
    kept so WalkMove (joint_names + default_joint_pos) keeps working.  The
    default pose is replaced by the exact float64 HOME after checking it is
    the env default the actor was trained against.
    """

    missing = [key for key in BASE_METADATA_KEYS if key not in base]
    if missing:
        raise ValueError(f"Base metadata lacks {missing}")
    joint_names = [str(name) for name in base["joint_names"]]  # type: ignore[union-attr]
    if sorted(joint_names) != sorted(HOME_FRAME.joint_pos):
        raise ValueError("Base metadata joints differ from HOME_FRAME's joints")
    default = np.asarray(base["default_joint_pos"], dtype=np.float64)
    home = np.array([HOME_FRAME.joint_pos[name] for name in joint_names], dtype=np.float64)
    if default.shape != home.shape or not np.allclose(default, home, rtol=0, atol=1e-6):
        raise ValueError("Env default joint pose is not the current HOME")
    if not set(ACTION_JOINT_NAMES) <= set(joint_names):
        raise ValueError("Action joints are missing from joint_names")
    if tuple(base["observation_names"]) != OBSERVATION_TERMS:  # type: ignore[arg-type]
        raise ValueError(f"Unexpected observation_names {base['observation_names']}")
    if tuple(base["command_names"]) != COMMAND_NAMES:  # type: ignore[arg-type]
        raise ValueError(f"Unexpected command_names {base['command_names']}")
    if float(base["action_scale"]) != ACTION_SCALE:  # type: ignore[arg-type]
        raise ValueError(f"Walking action scale must be {ACTION_SCALE}")
    lower = [float(value) for value in action_clip_lower]
    upper = [float(value) for value in action_clip_upper]
    if len(lower) != ACTION_WIDTH or len(upper) != ACTION_WIDTH:
        raise ValueError("Action clip must have one bound per action joint")
    if lower != [-SERVO_TARGET_RANGE_RAD] * ACTION_WIDTH or upper != [
        SERVO_TARGET_RANGE_RAD
    ] * ACTION_WIDTH:
        raise ValueError("Walking action clip differs from the servo goal range (+-pi)")
    if len(checkpoint_sha256) != 64 or any(c not in "0123456789abcdef" for c in checkpoint_sha256):
        raise ValueError("checkpoint_sha256 must be a lowercase SHA-256 hex digest")
    if not checkpoint_filename.endswith(".pt") or not run_dir or int(iteration) < 0:
        raise ValueError("Invalid checkpoint provenance")

    metadata: dict[str, object] = {key: base[key] for key in BASE_METADATA_KEYS}
    metadata["run_path"] = run_dir
    metadata["default_joint_pos"] = home.tolist()
    metadata["action_scale"] = ACTION_SCALE
    metadata.update(
        {
            "action_joint_names": list(ACTION_JOINT_NAMES),
            "action_clip_lower": lower,
            "action_clip_upper": upper,
            "previous_action_semantics": PREVIOUS_ACTION_SEMANTICS,
            "walk_contract_version": CONTRACT_VERSION,
            "home_pose": json.dumps(walk_home_pose(), sort_keys=True, separators=(",", ":")),
            "checkpoint_filename": checkpoint_filename,
            "checkpoint_sha256": checkpoint_sha256,
            "run_dir": run_dir,
            "iteration": int(iteration),
            **contract_metadata(),
        }
    )
    return {key: _metadata_value(value) for key, value in metadata.items()}


def _attach_metadata(path: Path, metadata: Mapping[str, str]) -> None:
    model = onnx.load(str(path))
    del model.metadata_props[:]
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.save(model, str(path))


class _RecordedConfigLoader(yaml.SafeLoader):
    """Reads mjlab's params/env.yaml without executing its python tags."""


def _construct_python_tag(loader: yaml.SafeLoader, suffix: str, node: yaml.Node) -> object:
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    return suffix


_RecordedConfigLoader.add_multi_constructor("tag:yaml.org,2002:python/", _construct_python_tag)


def require_recorded_walk_contract(env_yaml: Path) -> None:
    """Refuse a run whose recorded HOME, clip or feedback differs from CONTRACT_VERSION.

    The metadata is built from the current code, so exporting an older run
    (another HOME, hip -10 deg HOME, no clip, ...) would silently mislabel it.
    """

    if not env_yaml.is_file():
        raise ValueError(f"Run has no recorded env config: {env_yaml}")
    recorded = yaml.load(env_yaml.read_text(), Loader=_RecordedConfigLoader)
    try:
        init_state = recorded["scene"]["entities"]["robot"]["init_state"]
        action = recorded["actions"]["joint_pos"]
        previous_action = recorded["observations"]["actor"]["terms"]["actions"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Recorded env config lacks {error}") from error
    joints = init_state.get("joint_pos")
    if not isinstance(joints, dict) or set(joints) != set(HOME_FRAME.joint_pos) or any(
        abs(float(joints[name]) - value) > 1e-12 for name, value in HOME_FRAME.joint_pos.items()
    ):
        raise ValueError("Run was not trained from the current HOME joint pose")
    for key, expected in (("pos", HOME_FRAME.pos), ("rot", HOME_FRAME.rot)):
        if not np.allclose(
            np.asarray(init_state[key], dtype=np.float64), expected, rtol=0, atol=HOME_ROOT_RECORDED_ATOL
        ):
            raise ValueError(f"Run was not trained from the current HOME root {key}")
    clip = action.get("clip")
    if clip is None or list(clip) != [".*"] or [float(v) for v in clip[".*"]] != [
        -SERVO_TARGET_RANGE_RAD,
        SERVO_TARGET_RANGE_RAD,
    ]:
        raise ValueError(f"Run was not trained with the servo goal range (+-pi) as its target clip: {clip}")
    if (
        float(action.get("scale")) != ACTION_SCALE
        or float(action.get("offset")) != 0.0
        or action.get("use_default_offset") is not True
    ):
        raise ValueError("Run action is not target = HOME + raw_action * 1.0")
    if (
        previous_action.get("func") != "name:mjlab.envs.mdp.observations.last_action"
        or previous_action.get("clip") is not None
        or previous_action.get("scale") is not None
    ):
        raise ValueError("Run did not observe the raw previous action")


def require_current_home_walk_checkpoint(path: Path) -> None:
    """Refuse a walking checkpoint not trained at the current HOME.

    Checks both the run's recorded params/env.yaml and the checkpoint's own
    HOME stamp (consumers such as the teleop v12 bootstrap call this).
    """

    path = Path(path).resolve(strict=True)
    require_recorded_walk_contract(path.parent / "params" / "env.yaml")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    require_walk_home_pose(checkpoint.get("infos"))


def _inspect_checkpoint(path: Path) -> tuple[str, int]:
    before = _sha256(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    require_walk_home_pose(checkpoint.get("infos"))
    actor = checkpoint.get("actor_state_dict")
    if not isinstance(actor, dict):
        raise ValueError("Checkpoint has no actor_state_dict")
    first = actor.get("mlp.0.weight")
    if not isinstance(first, torch.Tensor) or first.shape[-1] != OBSERVATION_WIDTH:
        raise ValueError("Checkpoint actor does not take the 63-wide walking observation")
    iteration = checkpoint.get("iter")
    if not isinstance(iteration, int) or iteration < 0:
        raise ValueError("Checkpoint has no training iteration")
    if _sha256(path) != before:
        raise ValueError("Checkpoint changed during inspection")
    return before, iteration


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
        raise ValueError("Walking ONNX must have one obs input of shape [1, 63]")
    if len(model.graph.output) != 1 or (
        model.graph.output[0].name,
        _tensor_shape(model.graph.output[0]),
    ) != ("actions", (1, ACTION_WIDTH)):
        raise ValueError("Walking ONNX must have one actions output of shape [1, 18]")
    _normalizer_arrays(model)

    # Standing at HOME (HOME projected gravity) and a few commands, with zero
    # previous action.
    evaluator = ReferenceEvaluator(model)
    for command in ((0.0, 0.0, 0.0), (0.3, 0.0, 0.0), (0.0, 0.0, 1.0)):
        observation = np.zeros((1, OBSERVATION_WIDTH), dtype=np.float32)
        observation[0, 3:6] = HOME_PROJECTED_GRAVITY
        observation[0, -3:] = command
        outputs = evaluator.run(None, {"obs": observation})
        if len(outputs) != 1 or outputs[0].shape != (1, ACTION_WIDTH):
            raise ValueError("ONNX action inference returned the wrong shape")
        if not np.isfinite(outputs[0]).all():
            raise ValueError(f"Walking actor produces non-finite actions for command={command}")


def _action_contract(env: ManagerBasedRlEnv) -> tuple[np.ndarray, np.ndarray]:
    if tuple(env.action_manager.active_terms) != ("joint_pos",):
        raise ValueError("Walking must have exactly one joint_pos action term")
    action = env.action_manager.get_term("joint_pos")
    if tuple(action.target_names) != ACTION_JOINT_NAMES:
        raise ValueError("Walking action joint order differs from the robot runtime")
    terms = tuple(env.observation_manager.active_terms["actor"])
    if terms != OBSERVATION_TERMS:
        raise ValueError(f"Walking actor observation terms changed: {terms}")
    term_dims = tuple(
        int(np.prod(dim)) for dim in env.observation_manager.group_obs_term_dim["actor"]
    )
    if term_dims != OBSERVATION_TERM_DIMS:
        raise ValueError(f"Walking actor observation term widths changed: {term_dims}")
    if tuple(env.command_manager.active_terms) != COMMAND_NAMES:
        raise ValueError("Walking command terms changed")
    # WalkMove fills joint_pos/joint_vel in the action order; the env resolves
    # their regex in the robot's natural joint order, which must be the same.
    for term_name in ("joint_pos", "joint_vel"):
        asset_cfg = env.observation_manager.get_term_cfg("actor", term_name).params.get(
            "asset_cfg"
        )
        if asset_cfg is None or tuple(asset_cfg.joint_names or ()) != ACTION_JOINT_NAMES:
            raise ValueError(f"Walking {term_name} observation joint order differs from the actions")
    previous = env.cfg.observations["actor"].terms["actions"]
    if previous.func is not last_action or previous.params.get("action_name") not in (
        None,
        "joint_pos",
    ) or previous.clip is not None or previous.scale is not None:
        raise ValueError("Walking actor must observe its raw previous action")

    if isinstance(action._scale, torch.Tensor) or float(action._scale) != ACTION_SCALE:
        raise ValueError(f"Walking action scale must be the scalar {ACTION_SCALE}")
    robot = env.scene["robot"]
    default_all = robot.data.default_joint_pos[0].detach().cpu().numpy().astype(np.float64)
    home_all = np.array([HOME_FRAME.joint_pos[name] for name in robot.joint_names])
    if not np.allclose(default_all, home_all, rtol=0, atol=1e-6):
        raise ValueError("Walking env default joint pose is not the current HOME")
    offset = action._offset
    if not isinstance(offset, torch.Tensor) or offset.shape != (1, ACTION_WIDTH):
        raise ValueError("Walking action offset must be the per-joint default pose")
    if not np.allclose(
        offset[0].detach().cpu().numpy(), default_all[action.target_ids.cpu().numpy()], rtol=0, atol=1e-6
    ):
        raise ValueError("Walking action offset is not the default joint pose")
    clip = action._clip.detach().cpu().numpy()
    if action.cfg.clip is None or clip.shape != (1, ACTION_WIDTH, 2):
        raise ValueError(f"Unexpected walking action clip shape {clip.shape}")
    lower = clip[0, :, 0].astype(np.float64)
    upper = clip[0, :, 1].astype(np.float64)
    if not np.allclose(lower, -SERVO_TARGET_RANGE_RAD, rtol=0, atol=1e-6) or not np.allclose(
        upper, SERVO_TARGET_RANGE_RAD, rtol=0, atol=1e-6
    ):
        raise ValueError("Walking action target clip differs from the servo goal range (+-pi)")
    # Report the configured value, not its float32 copy.
    return (
        np.full(ACTION_WIDTH, -SERVO_TARGET_RANGE_RAD),
        np.full(ACTION_WIDTH, SERVO_TARGET_RANGE_RAD),
    )


def _parity(policy, path: Path, *, seed: int = 0) -> float:
    """Max |onnxruntime - torch actor| over PARITY_SAMPLES random observations."""

    import onnxruntime as ort
    from tensordict import TensorDict

    if list(policy.obs_groups) != ["actor"]:
        raise ValueError(f"Walking actor reads unexpected observation groups {policy.obs_groups}")
    generator = torch.Generator().manual_seed(seed)
    observations = torch.randn(PARITY_SAMPLES, OBSERVATION_WIDTH, generator=generator)
    observations[:, 3:6] = torch.nn.functional.normalize(observations[:, 3:6], dim=-1)
    with torch.inference_mode():
        expected = policy(
            TensorDict({"actor": observations}, batch_size=[PARITY_SAMPLES])
        ).numpy()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = np.concatenate(
        [session.run(None, {"obs": row[None, :]})[0] for row in observations.numpy()]
    )
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError("onnxruntime returned nonfinite or misshaped actions")
    return float(np.max(np.abs(actual - expected)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Walking model_N.pt")
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
    run_dir = checkpoint.parent
    require_recorded_walk_contract(run_dir / "params" / "env.yaml")
    checkpoint_sha256, iteration = _inspect_checkpoint(checkpoint)

    env_cfg = load_env_cfg(TASK, play=True)
    env_cfg.scene.num_envs = 1
    agent_cfg = load_rl_cfg(TASK)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    wrapped = RslRlVecEnvWrapper(env)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp.onnx")
    try:
        lower, upper = _action_contract(env)
        runner = load_runner_cls(TASK)(wrapped, asdict(agent_cfg), device=args.device)
        runner.load(
            str(checkpoint), load_cfg={"actor": True}, strict=True,
            map_location=args.device,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        runner.export_policy_to_onnx(str(temporary.parent), temporary.name)
        _validate_onnx(temporary)
        metadata = build_walk_metadata(
            get_base_metadata(env, run_path=run_dir.name),
            action_clip_lower=lower.tolist(),
            action_clip_upper=upper.tolist(),
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_filename=checkpoint.name,
            run_dir=run_dir.name,
            iteration=iteration,
        )
        _attach_metadata(temporary, metadata)
        _validate_onnx(temporary)
        policy = runner.get_inference_policy(device="cpu")
        max_abs_diff = _parity(policy, temporary)
        if max_abs_diff > PARITY_TOLERANCE:
            raise ValueError(f"ONNX/torch actor mismatch {max_abs_diff:.3g} > {PARITY_TOLERANCE}")
        if _sha256(checkpoint) != checkpoint_sha256:
            raise ValueError("Checkpoint changed during export")
        if args.replace:
            os.replace(temporary, output)
        else:
            # Both paths are in one directory.  Linking publishes the finished
            # file atomically and fails if another process created the name.
            os.link(temporary, output)
            temporary.unlink()
        print(json.dumps(metadata, indent=2))
        print(
            f"onnxruntime vs torch actor, {PARITY_SAMPLES} random observations: "
            f"max abs diff {max_abs_diff:.3e}"
        )
        print(f"Exported validated walking ONNX: {output}")
    finally:
        temporary.unlink(missing_ok=True)
        env.close()


if __name__ == "__main__":
    main()
