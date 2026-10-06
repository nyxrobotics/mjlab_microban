# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Contract tests for the walking ONNX export: microban-policy-1 at the current HOME.

The metadata itself is policy_contract.contract_metadata (tests/test_policy_contract.py);
here: the play env's layout, the builder's drift checks and the recorded run.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from mjlab.envs.mdp.observations import last_action
from onnx import TensorProto, helper

from mjlab_microban.robot.microban_constants import (
    HOME_FRAME,
    HOME_PROJECTED_GRAVITY,
    HOME_TRUNK_PITCH_RAD,
    SERVO_TARGET_RANGE_RAD,
)
from mjlab_microban.scripts.export_walk_onnx import (
    ACTION_JOINT_NAMES,
    OBSERVATION_TERMS,
    OBSERVATION_WIDTH,
    _attach_metadata,
    build_walk_metadata,
    require_recorded_walk_contract,
)
from mjlab_microban.tasks.microban_velocity_env_cfg import make_microban_velocity_env_cfg

# The robot runtime's microban/src/constants.py OBSERVATION_DOF_ORDER, which
# WalkMove uses for both the observation and the action order.
ROBOT_OBSERVATION_DOF_ORDER = (
    "right_shoulder_pitch", "right_shoulder_roll", "right_elbow",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee",
    "right_ankle_pitch", "right_ankle_roll",
    "left_shoulder_pitch", "left_shoulder_roll", "left_elbow",
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee",
    "left_ankle_pitch", "left_ankle_roll",
)
JOINT_NAMES = ("head", "neck_roll", "neck_pitch", *ROBOT_OBSERVATION_DOF_ORDER)
SHA = "ab" * 32


def _base(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "run_path": "local",
        "joint_names": list(JOINT_NAMES),
        # float32 copy of HOME, as the env's default_joint_pos tensor holds it.
        "default_joint_pos": [
            float(np.float32(HOME_FRAME.joint_pos[name])) for name in JOINT_NAMES
        ],
        "command_names": ["twist"],
        "observation_names": list(OBSERVATION_TERMS),
        "action_scale": 1.0,
    }
    base.update(overrides)
    return base


def _row(scale: float) -> list[float]:
    row = [0.0] * OBSERVATION_WIDTH
    row[3:6] = list(HOME_PROJECTED_GRAVITY)
    row[-1] = scale
    return row


def _build(base: dict[str, object] | None = None, **overrides: object) -> dict[str, str]:
    kwargs: dict[str, object] = {
        "action_clip_lower": [-SERVO_TARGET_RANGE_RAD] * 18,
        "action_clip_upper": [SERVO_TARGET_RANGE_RAD] * 18,
        "checkpoint": Path("/logs/2026-10-03_13-40-19_chome_servo_walk/model_500.pt"),
        "checkpoint_sha256": SHA,
        "gate_report_sha256": "cd" * 32,
        "self_test_observations": [_row(0.01 * i) for i in range(8)],
        "self_test_actions": [[0.0] * 18 for _ in range(8)],
        "dry_run": False,
    }
    kwargs.update(overrides)
    return build_walk_metadata(base or _base(), **kwargs)  # type: ignore[arg-type]


_ENV_YAML = """\
scene:
  entities:
    robot:
      init_state:
        pos: !!python/tuple
        - 0.0
        - 0.0
        - {z!r}
        rot: !!python/tuple {rot!r}
        joint_pos:
{joints}
actions:
  joint_pos:
    clip:
      .*: !!python/tuple
      - {lower!r}
      - {upper!r}
    scale: 1.0
    offset: 0.0
    use_default_offset: true
observations:
  actor:
    terms:
      actions:
        func: !!python/name:mjlab.envs.mdp.observations.last_action ''
        params: {{}}
        clip: null
        scale: null
"""


def _env_yaml(
    path: Path,
    *,
    hip_pitch: float | None = None,
    lower: float = -SERVO_TARGET_RANGE_RAD,
    rot: tuple[float, float, float, float] | None = None,
) -> Path:
    joints = dict(HOME_FRAME.joint_pos)
    if hip_pitch is not None:
        joints["left_hip_pitch"] = hip_pitch
    path.write_text(
        _ENV_YAML.format(
            z=HOME_FRAME.pos[2],
            rot=list(HOME_FRAME.rot if rot is None else rot),
            lower=lower,
            upper=SERVO_TARGET_RANGE_RAD,
            joints="\n".join(f"          {name}: {value!r}" for name, value in joints.items()),
        )
    )
    return path


class WalkExportMetadataContractTest(unittest.TestCase):
    def test_action_order_is_the_robot_runtime_order(self) -> None:
        self.assertEqual(ACTION_JOINT_NAMES, ROBOT_OBSERVATION_DOF_ORDER)
        self.assertEqual(OBSERVATION_WIDTH, 3 + 3 + 3 * 18 + 3)

    def test_velocity_task_uses_the_shared_contract(self) -> None:
        cfg = make_microban_velocity_env_cfg(play=True)
        action = cfg.actions["joint_pos"]
        self.assertEqual(action.scale, 1.0)
        self.assertTrue(action.use_default_offset)
        self.assertEqual(action.clip, {r".*": (-SERVO_TARGET_RANGE_RAD, SERVO_TARGET_RANGE_RAD)})
        self.assertEqual(cfg.scene.entities["robot"].init_state.joint_pos, HOME_FRAME.joint_pos)
        self.assertEqual(tuple(cfg.scene.entities["robot"].init_state.rot), HOME_FRAME.rot)
        # Upright reward and the twist-ratio velocity term are centred on
        # HOME's trunk lean (0 for a vertical trunk).
        self.assertEqual(cfg.rewards["upright"].params["pitch"], HOME_TRUNK_PITCH_RAD)
        self.assertEqual(cfg.rewards["twist_ratio_velocity"].params["trunk_pitch"], HOME_TRUNK_PITCH_RAD)
        self.assertEqual(tuple(cfg.observations["actor"].terms), OBSERVATION_TERMS)
        previous = cfg.observations["actor"].terms["actions"]
        self.assertIs(previous.func, last_action)
        self.assertIsNone(previous.clip)
        self.assertIsNone(previous.scale)

    def test_metadata_is_the_walk_contract(self) -> None:
        metadata = _build()
        self.assertEqual(metadata["microban_policy_kind"], "walk")
        self.assertEqual(metadata["microban_recipe"], "microban-walk-twist-ratio-1")
        names = metadata["joint_names"].split(",")
        self.assertEqual(names, list(JOINT_NAMES))
        default = dict(zip(names, (float(v) for v in metadata["default_joint_pos"].split(","))))
        # Exact float64 HOME, not the env's float32 copy or mjlab's 3 decimals.
        self.assertEqual(default, dict(HOME_FRAME.joint_pos))
        self.assertEqual(metadata["action_joint_names"].split(","), list(ROBOT_OBSERVATION_DOF_ORDER))
        self.assertEqual(metadata["observation_joint_names"].split(","), list(ROBOT_OBSERVATION_DOF_ORDER))
        self.assertEqual(metadata["observation_schema_json"],
                         '[["base_ang_vel",3],["projected_gravity",3],["joint_pos",18],["joint_vel",18],'
                         '["actions",18],["command",3]]')
        self.assertEqual(metadata["checkpoint_filename"], "model_500.pt")
        self.assertEqual(metadata["checkpoint_iteration"], "500")
        self.assertEqual(metadata["checkpoint_sha256"], SHA)

    def test_metadata_round_trips_through_onnx(self) -> None:
        obs = helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, OBSERVATION_WIDTH])
        actions = helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 18])
        weight = helper.make_tensor(
            "weight", TensorProto.FLOAT, [OBSERVATION_WIDTH, 18], [0.0] * (OBSERVATION_WIDTH * 18)
        )
        graph = helper.make_graph(
            [helper.make_node("MatMul", ("obs", "weight"), ("actions",))],
            "walk_export_contract_test", [obs], [actions], [weight],
        )
        metadata = _build()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "walk.onnx"
            onnx.save(helper.make_model(graph), path)
            _attach_metadata(path, {"stale": "x"})
            _attach_metadata(path, metadata)
            stored = {item.key: item.value for item in onnx.load(path).metadata_props}
        self.assertEqual(stored, metadata)

    def test_metadata_builder_rejects_contract_drift(self) -> None:
        old_home = _base()
        old_home["default_joint_pos"] = list(old_home["default_joint_pos"])  # type: ignore[arg-type]
        old_home["default_joint_pos"][JOINT_NAMES.index("left_hip_pitch")] = -0.1745  # type: ignore[index]
        cases = (
            lambda: _build(old_home),
            lambda: _build(_base(observation_names=list(OBSERVATION_TERMS[:-1]))),
            lambda: _build(_base(action_scale=0.5)),
            lambda: _build(_base(command_names=["base_velocity"])),
            lambda: _build(action_clip_lower=[-1.57] * 18),
            lambda: _build(action_clip_upper=[SERVO_TARGET_RANGE_RAD] * 17),
            lambda: _build(checkpoint_sha256="not-a-digest"),
            lambda: _build(checkpoint=Path("/logs/run/last.pt")),
        )
        for case in cases:
            with self.assertRaises(ValueError):
                case()

    def test_recorded_run_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            require_recorded_walk_contract(_env_yaml(root / "ok.yaml"))
            for bad in (
                _env_yaml(root / "old_home.yaml", hip_pitch=-0.17453292519943295),
                _env_yaml(root / "old_clip.yaml", lower=-1.57),
                # HOME joints with another root attitude (the forward-lean
                # joints with an upright root, or the reverse).
                _env_yaml(
                    root / "other_root.yaml",
                    rot=(1.0, 0.0, 0.0, 0.0) if HOME_TRUNK_PITCH_RAD != 0.0 else (0.996, 0.0, 0.0872, 0.0),
                ),
            ):
                with self.assertRaises(ValueError):
                    require_recorded_walk_contract(bad)
            with self.assertRaises(ValueError):
                require_recorded_walk_contract(root / "missing.yaml")


if __name__ == "__main__":
    unittest.main()
