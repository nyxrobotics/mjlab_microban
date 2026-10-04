# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""The shared HOME leans the trunk 10 deg forward with the COM over the soles.

Checked by MuJoCo forward kinematics on robot.xml: both soles flat on the
ground, trunk pitched exactly HOME_TRUNK_PITCH_RAD forward, mass-weighted COM
over the fore-aft centre of the sole contact patch.
"""

from __future__ import annotations

import math
import unittest

import mujoco
import numpy as np
import torch

from mjlab_microban.robot.microban_constants import (
    HOME_ANKLE_PITCH_RAD,
    HOME_FRAME,
    HOME_HIP_PITCH_RAD,
    HOME_TRUNK_PITCH_RAD,
    get_spec,
)
from mjlab_microban.tasks.mdp import _home_levelled_quat
from mjlab_microban.tasks.microban_getup_runner import getup_home_pose
from mjlab_microban.tasks.microban_velocity_runner import (
    WALK_HOME_POSE_INFO_KEY,
    require_walk_home_pose,
)

_SOLES = tuple(f"{side}_foot_collision_{i}" for side in ("left", "right") for i in range(1, 7))


def _home_state() -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = get_spec().compile()
    data = mujoco.MjData(model)
    data.qpos[:3] = HOME_FRAME.pos
    data.qpos[3:7] = HOME_FRAME.rot
    for name, value in HOME_FRAME.joint_pos.items():
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    return model, data


def _sole_corners(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    corners = []
    for name in _SOLES:
        geom = model.geom(name).id
        rotation = data.geom_xmat[geom].reshape(3, 3)
        for signs in np.array(np.meshgrid((-1, 1), (-1, 1), (-1, 1))).T.reshape(-1, 3):
            corners.append(data.geom_xpos[geom] + rotation @ (signs * model.geom_size[geom]))
    return np.array(corners)


class ForwardLeanHomeTest(unittest.TestCase):
    def test_named_constants_define_home(self) -> None:
        self.assertAlmostEqual(HOME_TRUNK_PITCH_RAD, math.radians(10.0), places=15)
        for side in ("left", "right"):
            self.assertEqual(HOME_FRAME.joint_pos[f"{side}_hip_pitch"], HOME_HIP_PITCH_RAD)
            self.assertEqual(HOME_FRAME.joint_pos[f"{side}_ankle_pitch"], HOME_ANKLE_PITCH_RAD)
            self.assertEqual(HOME_FRAME.joint_pos[f"{side}_knee"], 0.0)
        half = HOME_TRUNK_PITCH_RAD / 2.0
        np.testing.assert_allclose(HOME_FRAME.rot, (math.cos(half), 0.0, math.sin(half), 0.0), atol=1e-15)

    def test_trunk_leans_forward_with_flat_soles_on_the_ground(self) -> None:
        model, data = _home_state()
        trunk = model.body("trunk").id
        trunk_x = data.xmat[trunk].reshape(3, 3)[:, 0]
        # Forward lean: the trunk's x axis tips 10 deg below the horizon.
        self.assertAlmostEqual(math.asin(-trunk_x[2]), HOME_TRUNK_PITCH_RAD, places=12)
        for name in _SOLES:
            normal = data.geom_xmat[model.geom(name).id].reshape(3, 3)[:, 2]
            # Flat fore-aft; the +-5 deg hip/ankle rolls leave 0.076 deg sideways.
            self.assertLess(abs(normal[0]), 1e-9, name)
            self.assertLess(abs(normal[1]), 1.5e-3, name)
        corners = _sole_corners(model, data)
        self.assertAlmostEqual(corners[:, 2].min(), 0.0, places=9)

    def test_com_is_over_the_sole_centre(self) -> None:
        model, data = _home_state()
        corners = _sole_corners(model, data)
        contact = corners[corners[:, 2] < 1e-4]
        heel, toe = contact[:, 0].min(), contact[:, 0].max()
        com_x = data.subtree_com[model.body("trunk").id][0]
        self.assertLess(abs(com_x - 0.5 * (heel + toe)), 1e-7)
        self.assertGreater(com_x - heel, 0.030)
        self.assertGreater(toe - com_x, 0.030)

    def test_levelled_frame_removes_the_home_lean(self) -> None:
        home = torch.tensor([HOME_FRAME.rot], dtype=torch.float64)
        levelled = _home_levelled_quat(home, HOME_TRUNK_PITCH_RAD)
        torch.testing.assert_close(levelled, torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64))
        self.assertIs(_home_levelled_quat(home, 0.0), home)

    def test_walking_checkpoints_are_bound_to_home(self) -> None:
        require_walk_home_pose({WALK_HOME_POSE_INFO_KEY: getup_home_pose()})
        centered = getup_home_pose()
        centered["root_quat_wxyz"] = [1.0, 0.0, 0.0, 0.0]
        for infos in (None, {}, {WALK_HOME_POSE_INFO_KEY: centered}):
            with self.subTest(infos=infos), self.assertRaises(ValueError):
                require_walk_home_pose(infos)


if __name__ == "__main__":
    unittest.main()
