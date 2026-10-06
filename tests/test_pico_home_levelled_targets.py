"""PICO foot/hand targets live in the HOME-levelled trunk frame.

The frame is R_trunk * R_y(-HOME_TRUNK_PITCH_RAD): at the forward-lean HOME it
is level (x forward, y left, z up), so a world-vertical foot lift reads
(0, 0, dz) and a level headset holds neck_pitch at -HOME_TRUNK_PITCH_RAD.
(forward-lean-v2 cd0ea78; at a vertical HOME trunk the frame is the trunk
frame and the tasks keep their original terms.)
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch
from mjlab.utils.lab_api.math import (
    quat_apply,
    quat_from_euler_xyz,
    quat_mul,
    subtract_frame_transforms,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from home_cases import pitched_home_only  # noqa: E402

from mjlab_microban.robot.microban_constants import (
    HOME_ROOT_QUAT_WXYZ,
    HOME_TRUNK_PITCH_RAD,
)
from mjlab_microban.robot.microban_hand_fk import (
    MICROBAN_ARM_JOINT_LOWER_RAD,
    MICROBAN_ARM_JOINT_UPPER_RAD,
    microban_hand_offsets_from_arm_joints,
    rotate_trunk_offsets_to_home_levelled,
    sample_microban_reachable_hand_targets,
)
from mjlab_microban.tasks.mdp import (
    FootTargetCommandCfg,
    HandTargetCommandCfg,
    _home_levelled_quat,
)
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_TELEOP_TARGET_FRAME,
    deterministic_teleop_parity_inputs,
)
from mjlab_microban.tasks.microban_teleop_env_cfg import (
    make_microban_teleop_env_cfg,
)
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MICROBAN_TELEOP_V12_HMD_NEUTRAL_POSITION_RAD,
    MICROBAN_TELEOP_V12_RECIPE_REVISION,
    make_microban_teleop_v12_env_cfg,
)


def _home_quat(count: int = 1) -> torch.Tensor:
    return torch.tensor(HOME_ROOT_QUAT_WXYZ, dtype=torch.float64).expand(count, 4)


def _levelled_offset(
    trunk_quat: torch.Tensor, origin: torch.Tensor, point: torch.Tensor
) -> torch.Tensor:
    frame = _home_levelled_quat(trunk_quat, HOME_TRUNK_PITCH_RAD)
    offset, _ = subtract_frame_transforms(origin, frame, point)
    return offset


class HomeLevelledTargetFrameTest(unittest.TestCase):
    def test_command_cfg_default_keeps_the_trunk_frame(self) -> None:
        self.assertEqual(FootTargetCommandCfg(resampling_time_range=(1, 2)).trunk_pitch, 0.0)
        self.assertEqual(HandTargetCommandCfg(resampling_time_range=(1, 2)).trunk_pitch, 0.0)

    @pitched_home_only
    def test_teleop_tasks_define_targets_in_the_levelled_frame(self) -> None:
        for make in (make_microban_teleop_env_cfg, make_microban_teleop_v12_env_cfg):
            for play in (False, True):
                cfg = make(play=play)
                self.assertEqual(
                    cfg.commands["foot_target"].trunk_pitch, HOME_TRUNK_PITCH_RAD
                )
                self.assertEqual(
                    cfg.commands["hand_target"].trunk_pitch, HOME_TRUNK_PITCH_RAD
                )
        self.assertEqual(
            MICROBAN_TELEOP_TARGET_FRAME,
            "robot_home_levelled_trunk_xyz_forward_left_up",
        )
        self.assertTrue(MICROBAN_TELEOP_V12_RECIPE_REVISION.endswith("_v17"))

    def test_home_targets_are_zero_and_a_vertical_lift_reads_vertical(self) -> None:
        trunk = torch.tensor([[0.01, -0.02, 0.170430569776402]], dtype=torch.float64)
        foot = trunk + torch.tensor([[-0.004, 0.047, -0.150]], dtype=torch.float64)
        reference = _levelled_offset(_home_quat(), trunk, foot)
        # Zero offset at HOME: the target is relative to the reset reference.
        torch.testing.assert_close(
            _levelled_offset(_home_quat(), trunk, foot) - reference,
            torch.zeros(1, 3, dtype=torch.float64),
            rtol=0.0,
            atol=0.0,
        )
        for lift in (0.0025, 0.02, 0.05):
            lifted = foot + torch.tensor([[0.0, 0.0, lift]], dtype=torch.float64)
            offset = _levelled_offset(_home_quat(), trunk, lifted) - reference
            torch.testing.assert_close(
                offset,
                torch.tensor([[0.0, 0.0, lift]], dtype=torch.float64),
                rtol=0.0,
                atol=1.0e-15,
            )
            # The leaning trunk frame would have tipped the same lift by the
            # lean: 10 deg puts sin(10 deg) * lift of it along trunk -x.
            plain, _ = subtract_frame_transforms(trunk, _home_quat(), lifted)
            plain_reference, _ = subtract_frame_transforms(trunk, _home_quat(), foot)
            self.assertAlmostEqual(
                float((plain - plain_reference)[0, 0]),
                -math.sin(HOME_TRUNK_PITCH_RAD) * lift,
                places=15,
            )

    def test_levelled_frame_ignores_heading_and_keeps_yaw(self) -> None:
        # A yawed HOME: the levelled frame still has z up and x along the
        # heading, so a world-vertical lift is still (0, 0, dz).
        yaw = torch.tensor([0.7], dtype=torch.float64)
        zero = torch.zeros(1, dtype=torch.float64)
        yaw_quat = quat_from_euler_xyz(zero, zero, yaw)
        pitch_quat = quat_from_euler_xyz(
            zero, torch.tensor([HOME_TRUNK_PITCH_RAD], dtype=torch.float64), zero
        )
        trunk_quat = quat_mul(yaw_quat, pitch_quat)
        origin = torch.zeros(1, 3, dtype=torch.float64)
        lifted = torch.tensor([[0.0, 0.0, 0.03]], dtype=torch.float64)
        torch.testing.assert_close(
            _levelled_offset(trunk_quat, origin, lifted),
            lifted,
            rtol=0.0,
            atol=1.0e-15,
        )

    def test_rotated_fk_offsets_equal_the_levelled_reading_of_trunk_offsets(
        self,
    ) -> None:
        # The hand command rotates trunk-frame FK offsets by R_y(lean); this is
        # exactly how the levelled frame reads the same displacement, for any
        # trunk attitude.
        generator = torch.Generator().manual_seed(11)
        lower = torch.tensor(MICROBAN_ARM_JOINT_LOWER_RAD, dtype=torch.float64)
        upper = torch.tensor(MICROBAN_ARM_JOINT_UPPER_RAD, dtype=torch.float64)
        joints = lower + torch.rand(
            (32, 2, 3), generator=generator, dtype=torch.float64
        ) * (upper - lower)
        trunk_offsets = microban_hand_offsets_from_arm_joints(joints).reshape(-1, 3)
        angles = (torch.rand((64, 3), generator=generator, dtype=torch.float64) - 0.5)
        trunk_quat = quat_from_euler_xyz(angles[:, 0], angles[:, 1], angles[:, 2])
        world = quat_apply(trunk_quat, trunk_offsets)
        expected, _ = subtract_frame_transforms(
            torch.zeros_like(world),
            _home_levelled_quat(trunk_quat, HOME_TRUNK_PITCH_RAD),
            world,
        )
        torch.testing.assert_close(
            rotate_trunk_offsets_to_home_levelled(trunk_offsets, HOME_TRUNK_PITCH_RAD),
            expected,
            rtol=0.0,
            atol=1.0e-15,
        )
        # Inactive hands remain exact zero after the rotation.
        _, offsets = sample_microban_reachable_hand_targets(
            torch.zeros(4, 2, dtype=torch.bool), dtype=torch.float64
        )
        self.assertTrue(torch.equal(offsets, torch.zeros_like(offsets)))

    @pitched_home_only
    def test_v12_hmd_neutral_is_the_level_headset_pose(self) -> None:
        self.assertEqual(
            MICROBAN_TELEOP_V12_HMD_NEUTRAL_POSITION_RAD,
            {"head": 0.0, "neck_roll": 0.0, "neck_pitch": -HOME_TRUNK_PITCH_RAD},
        )
        cfg = make_microban_teleop_v12_env_cfg(play=False)
        self.assertEqual(
            cfg.events["hmd_neck_target_motion"].params["neutral_position_rad"],
            MICROBAN_TELEOP_V12_HMD_NEUTRAL_POSITION_RAD,
        )
        # The legacy teleop task keeps its HOME neutral.
        legacy = make_microban_teleop_env_cfg(play=False)
        self.assertNotIn(
            "neutral_position_rad",
            legacy.events["hmd_neck_target_motion"].params,
        )

    def test_parity_neutral_row_observes_home_gravity(self) -> None:
        neutral = deterministic_teleop_parity_inputs()[0, 0]
        self.assertAlmostEqual(
            float(neutral[3]), math.sin(HOME_TRUNK_PITCH_RAD), places=7
        )
        self.assertEqual(float(neutral[4]), 0.0)
        self.assertAlmostEqual(
            float(neutral[5]), -math.cos(HOME_TRUNK_PITCH_RAD), places=7
        )


if __name__ == "__main__":
    unittest.main()
