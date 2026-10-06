# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Get-up checkpoint/export contract: a HOME-bound stamp plus the recorded env.

The stamp is v5 at the centered upright HOME, v6 at the forward-lean HOME and
"<v>_<tag>" at any other HOME.  At the centered HOME, runs started on
2026-10-03 before the v5 bump trained with the +-pi clip but stamped "v4";
they must export (as v5), while a v4 run with the +-1.57 clip or the old HOME
must not.  Elsewhere every other stamp was trained at another HOME and is
refused, as is a run whose recorded env shows another HOME (joints or root),
the +-1.57 clip or post-clip feedback.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from mjlab.envs.mdp.observations import last_action
from mjlab.utils.os import dump_yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from home_cases import CENTERED_HOME_TAG, PUBLISHED_CONTRACT_STRINGS, home_tag  # noqa: E402

from mjlab_microban.robot.home_contracts import GETUP_LEGACY_STAMP, contract_strings
from mjlab_microban.robot.microban_constants import HOME_FRAME
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

V = GETUP_CONTRACT_VERSION


def _infos(contract: str = V, **overrides: object) -> dict[str, object]:
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


def _other_root(cfg) -> None:
    """HOME joints with another root attitude (upright at a pitched HOME, or pitched)."""

    robot = cfg.scene.entities["robot"]
    pitched = (math.cos(math.radians(5.0)), 0.0, math.sin(math.radians(5.0)), 0.0)
    rot = (1.0, 0.0, 0.0, 0.0) if tuple(HOME_FRAME.rot) != (1.0, 0.0, 0.0, 0.0) else pitched
    robot.init_state = dataclasses.replace(robot.init_state, rot=rot)


def _post_clip_feedback(cfg) -> None:
    term = copy.copy(cfg.observations["actor"].terms["actions"])
    term.func = last_action
    cfg.observations["actor"].terms["actions"] = term


class GetupExportContractTest(unittest.TestCase):
    def test_version(self) -> None:
        expected = PUBLISHED_CONTRACT_STRINGS.get(home_tag(), contract_strings())
        self.assertEqual(GETUP_CONTRACT_VERSION, expected["getup_contract_version"])
        self.assertEqual(CONTRACT_VERSION, GETUP_CONTRACT_VERSION)
        # Only the centered HOME accepts the v4-stamped v5 runs of 2026-10-03.
        self.assertEqual(GETUP_LEGACY_STAMP, "v4" if home_tag() == CENTERED_HOME_TAG else None)

    def test_recorded_env(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ok = _run(root, "ok")
            require_recorded_getup_env(ok.parent / "params" / "env.yaml")
            for name, mutate in (
                ("old_clip", _old_clip),
                ("old_home", _old_home),
                ("other_root", _other_root),
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
            bare = root / "copied" / "model_0.pt"  # e.g. reset_getup_action_std output

            other_root = _run(root, "other_root", _other_root)

            # Current runs export; a current stamp is trusted for resume without params.
            self.assertEqual(require_getup_checkpoint_contract(ok, _infos(V), require_recorded_env=True), V)
            self.assertEqual(require_getup_checkpoint_contract(bare, _infos(V), require_recorded_env=False), V)
            for path in (bare, old_clip, other_root):
                with self.assertRaises(ValueError):
                    require_getup_checkpoint_contract(path, _infos(V), require_recorded_env=True)

            if GETUP_LEGACY_STAMP is not None:
                # "v4" stamp: valid only when the recorded env proves +-pi at HOME.
                for required in (True, False):
                    self.assertEqual(
                        require_getup_checkpoint_contract(
                            ok, _infos(GETUP_LEGACY_STAMP), require_recorded_env=required
                        ),
                        GETUP_LEGACY_STAMP,
                    )
                    for path in (old_clip, bare):
                        with self.assertRaises(ValueError):
                            require_getup_checkpoint_contract(
                                path, _infos(GETUP_LEGACY_STAMP), require_recorded_env=required
                            )
            # Every other stamp was trained at another HOME: never accepted.
            for stamp in sorted({"v3", "v4", "v5", "v6"} - {V, GETUP_LEGACY_STAMP}):
                for required in (True, False):
                    with self.subTest(stamp=stamp, required=required), self.assertRaises(ValueError):
                        require_getup_checkpoint_contract(ok, _infos(stamp), require_recorded_env=required)

            for infos in (
                None,
                _infos("v3"),
                _infos(V, microban_getup_angular_velocity_frame="body"),
                _infos(V, microban_getup_home_pose={"joint_pos_rad": {}}),
            ):
                with self.assertRaises(ValueError):
                    require_getup_checkpoint_contract(ok, infos, require_recorded_env=True)

    def test_clip_metadata_is_exactly_pi(self) -> None:
        text = _full_precision_csv(np.full(18, math.pi))
        values = [float(value) for value in text.split(",")]
        self.assertEqual(values, [math.pi] * 18)


if __name__ == "__main__":
    unittest.main()
