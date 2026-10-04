# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Get-up v6 checkpoint/export contract: a v6 stamp plus the recorded env.

v6 is v5's target rule at the forward-lean HOME. Every earlier stamp (v3-v5)
was trained at another HOME and is refused, as is a run whose recorded env
shows the centered/old HOME, the +-1.57 clip or post-clip feedback.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from mjlab.envs.mdp.observations import last_action
from mjlab.utils.os import dump_yaml

from mjlab_microban.scripts.export_getup_onnx import (
    CONTRACT_VERSION,
    _full_precision_csv,
)
from mjlab_microban.tasks.microban_getup_env_cfg import make_microban_getup_env_cfg
from mjlab_microban.tasks.microban_getup_runner import (
    GETUP_ANGULAR_VELOCITY_FRAME,
    GETUP_CONTRACT_VERSION,
    getup_home_pose,
    require_getup_checkpoint_contract,
    require_recorded_getup_env,
)


def _infos(contract: str = "v6", **overrides: object) -> dict[str, object]:
    infos: dict[str, object] = {
        "microban_getup_contract": contract,
        "microban_getup_angular_velocity_frame": GETUP_ANGULAR_VELOCITY_FRAME,
        "microban_getup_home_pose": getup_home_pose(),
    }
    infos.update(overrides)
    return infos


def _run(root: Path, name: str, mutate=None) -> Path:
    """A run directory whose params/env.yaml is dumped the way mjlab's train does."""

    cfg = make_microban_getup_env_cfg()
    if mutate is not None:
        mutate(cfg)
    run = root / name
    dump_yaml(run / "params" / "env.yaml", cfg)
    return run / "model_0.pt"


def _old_clip(cfg) -> None:
    cfg.actions["joint_pos"].clip = {r".*": (-1.57, 1.57)}


def _old_home(cfg) -> None:
    robot = cfg.scene.entities["robot"]
    joints = dict(robot.init_state.joint_pos)
    joints["left_hip_pitch"] = -0.17453292519943295
    robot.init_state = dataclasses.replace(robot.init_state, joint_pos=joints)


def _centered_home(cfg) -> None:
    """The centered upright HOME of contract v5."""

    robot = cfg.scene.entities["robot"]
    joints = dict(robot.init_state.joint_pos)
    for side in ("left", "right"):
        joints[f"{side}_hip_pitch"] = math.radians(1.198384259489)
        joints[f"{side}_ankle_pitch"] = -math.radians(1.198384259489)
    robot.init_state = dataclasses.replace(
        robot.init_state,
        joint_pos=joints,
        pos=(0.0, 0.0, 0.170554885633559),
        rot=(1.0, 0.0, 0.0, 0.0),
    )


def _upright_root(cfg) -> None:
    """Forward-lean HOME joints with the trunk vertical (unflat soles)."""

    robot = cfg.scene.entities["robot"]
    robot.init_state = dataclasses.replace(robot.init_state, rot=(1.0, 0.0, 0.0, 0.0))


def _post_clip_feedback(cfg) -> None:
    term = copy.copy(cfg.observations["actor"].terms["actions"])
    term.func = last_action
    cfg.observations["actor"].terms["actions"] = term


class GetupExportContractTest(unittest.TestCase):
    def test_version(self) -> None:
        self.assertEqual(GETUP_CONTRACT_VERSION, "v6")
        self.assertEqual(CONTRACT_VERSION, "v6")

    def test_recorded_env(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ok = _run(root, "ok")
            require_recorded_getup_env(ok.parent / "params" / "env.yaml")
            for name, mutate in (
                ("old_clip", _old_clip),
                ("old_home", _old_home),
                ("centered_home", _centered_home),
                ("upright_root", _upright_root),
                ("post_clip_feedback", _post_clip_feedback),
            ):
                bad = _run(root, name, mutate)
                with self.subTest(name), self.assertRaises(ValueError):
                    require_recorded_getup_env(bad.parent / "params" / "env.yaml")
            with self.assertRaises(ValueError):
                require_recorded_getup_env(root / "missing" / "params" / "env.yaml")

    def test_checkpoint_stamps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ok = _run(root, "ok")
            old_clip = _run(root, "old_clip", _old_clip)
            centered = _run(root, "centered_home", _centered_home)
            bare = root / "copied" / "model_0.pt"  # e.g. reset_getup_action_std output

            # v6 runs export; a v6 stamp is trusted for resume without params.
            self.assertEqual(require_getup_checkpoint_contract(ok, _infos("v6"), require_recorded_env=True), "v6")
            self.assertEqual(require_getup_checkpoint_contract(bare, _infos("v6"), require_recorded_env=False), "v6")
            for path in (bare, old_clip, centered):
                with self.assertRaises(ValueError):
                    require_getup_checkpoint_contract(path, _infos("v6"), require_recorded_env=True)

            # Earlier stamps were trained at other HOMEs: never accepted.
            for stamp in ("v4", "v5"):
                for required in (True, False):
                    with self.subTest(stamp=stamp, required=required), self.assertRaises(ValueError):
                        require_getup_checkpoint_contract(ok, _infos(stamp), require_recorded_env=required)

            for infos in (
                None,
                _infos("v3"),
                _infos("v6", microban_getup_angular_velocity_frame="body"),
                _infos("v6", microban_getup_home_pose={"joint_pos_rad": {}}),
            ):
                with self.assertRaises(ValueError):
                    require_getup_checkpoint_contract(ok, infos, require_recorded_env=True)

    def test_clip_metadata_is_exactly_pi(self) -> None:
        text = _full_precision_csv(np.full(18, math.pi))
        values = [float(value) for value in text.split(",")]
        self.assertEqual(values, [math.pi] * 18)


if __name__ == "__main__":
    unittest.main()
