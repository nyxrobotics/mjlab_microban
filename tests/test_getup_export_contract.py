# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Get-up v5 checkpoint/export contract: validity comes from the recorded env.

Runs started on 2026-10-03 before the v5 bump trained with the +-pi clip at
the centered HOME but stamped "v4"; they must export (as v5), while a v4 run
with the +-1.57 clip or the old HOME must not.
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
    require_recorded_getup_v5_env,
)


def _infos(contract: str = "v5", **overrides: object) -> dict[str, object]:
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


def _post_clip_feedback(cfg) -> None:
    term = copy.copy(cfg.observations["actor"].terms["actions"])
    term.func = last_action
    cfg.observations["actor"].terms["actions"] = term


class GetupExportContractTest(unittest.TestCase):
    def test_version(self) -> None:
        self.assertEqual(GETUP_CONTRACT_VERSION, "v5")
        self.assertEqual(CONTRACT_VERSION, "v5")

    def test_recorded_env(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ok = _run(root, "ok")
            require_recorded_getup_v5_env(ok.parent / "params" / "env.yaml")
            for name, mutate in (
                ("old_clip", _old_clip),
                ("old_home", _old_home),
                ("post_clip_feedback", _post_clip_feedback),
            ):
                bad = _run(root, name, mutate)
                with self.subTest(name), self.assertRaises(ValueError):
                    require_recorded_getup_v5_env(bad.parent / "params" / "env.yaml")
            with self.assertRaises(ValueError):
                require_recorded_getup_v5_env(root / "missing" / "params" / "env.yaml")

    def test_checkpoint_stamps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ok = _run(root, "ok")
            old_clip = _run(root, "old_clip", _old_clip)
            bare = root / "copied" / "model_0.pt"  # e.g. reset_getup_action_std output

            # v5 runs export; a v5 stamp is trusted for resume without params.
            self.assertEqual(require_getup_checkpoint_contract(ok, _infos("v5"), require_recorded_env=True), "v5")
            self.assertEqual(require_getup_checkpoint_contract(bare, _infos("v5"), require_recorded_env=False), "v5")
            with self.assertRaises(ValueError):
                require_getup_checkpoint_contract(bare, _infos("v5"), require_recorded_env=True)
            with self.assertRaises(ValueError):
                require_getup_checkpoint_contract(old_clip, _infos("v5"), require_recorded_env=True)

            # "v4" stamp: valid only when the recorded env proves +-pi at HOME.
            for required in (True, False):
                self.assertEqual(
                    require_getup_checkpoint_contract(ok, _infos("v4"), require_recorded_env=required), "v4"
                )
                for path in (old_clip, bare):
                    with self.assertRaises(ValueError):
                        require_getup_checkpoint_contract(path, _infos("v4"), require_recorded_env=required)

            for infos in (
                None,
                _infos("v3"),
                _infos("v5", microban_getup_angular_velocity_frame="body"),
                _infos("v5", microban_getup_home_pose={"joint_pos_rad": {}}),
            ):
                with self.assertRaises(ValueError):
                    require_getup_checkpoint_contract(ok, infos, require_recorded_env=True)

    def test_clip_metadata_is_exactly_pi(self) -> None:
        text = _full_precision_csv(np.full(18, math.pi))
        values = [float(value) for value in text.split(",")]
        self.assertEqual(values, [math.pi] * 18)


if __name__ == "__main__":
    unittest.main()
