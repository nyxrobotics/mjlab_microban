"""Exact-FK and reachable hand-target contract tests at the forward-lean HOME.

forward-lean-v2 cd0ea78's tests/test_microban_hand_fk.py: the HOME-levelled target
frame and the +-64 mm receiver box of hand FK v4 (pinned for the +10 deg trunk).
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from home_cases import forward_lean_home_only  # noqa: E402

from mjlab_microban.robot.microban_constants import HOME_TRUNK_PITCH_RAD
from mjlab_microban.robot.microban_hand_fk import (
    MICROBAN_ARM_HOME_JOINT_RAD,
    MICROBAN_ARM_JOINT_LOWER_RAD,
    MICROBAN_ARM_JOINT_UPPER_RAD,
    MICROBAN_HAND_FK_BOUND_GRID_POINTS_PER_AXIS,
    MICROBAN_HAND_FK_OFFSET_AABB_MAX_M,
    MICROBAN_HAND_FK_OFFSET_AABB_MIN_M,
    MICROBAN_HAND_FK_TRUNK_OFFSET_AABB_MAX_M,
    MICROBAN_HAND_FK_TRUNK_OFFSET_AABB_MIN_M,
    MICROBAN_HAND_TARGET_FRAME,
    MICROBAN_HAND_TARGET_FRAME_PITCH_RAD,
    MICROBAN_HAND_TARGET_MAX_REJECTION_ROUNDS,
    MICROBAN_HAND_TARGET_NORMALIZER_ABS_BOUND_M,
    MICROBAN_HAND_TARGET_RUNTIME_VALIDATED_ABS_LIMIT_M,
    MICROBAN_HAND_TARGET_WIRE_ABS_BOUND_M,
    MICROBAN_REACHABLE_HAND_EVALUATION_JOINTS_DEG,
    microban_default_hand_positions,
    microban_hand_fk_metadata,
    microban_hand_offsets_from_arm_joints,
    microban_hand_positions_from_arm_joints,
    microban_hand_target_offsets_from_arm_joints,
    microban_hand_target_offsets_within_limit,
    microban_reachable_hand_evaluation_offsets,
    rotate_trunk_offsets_to_home_levelled,
    sample_microban_reachable_hand_targets,
)

ROOT = Path(__file__).resolve().parents[1]
ROBOT_XML = ROOT / "src/mjlab_microban/robot/microban/robot.xml"


@forward_lean_home_only
class MicrobanHandFkForwardLeanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))

    def _mujoco_hand_positions(self, joint_positions: np.ndarray) -> np.ndarray:
        data = mujoco.MjData(self.model)
        for side_index, side in enumerate(("left", "right")):
            for joint_index, suffix in enumerate(
                ("shoulder_pitch", "shoulder_roll", "elbow")
            ):
                joint_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    f"{side}_{suffix}",
                )
                data.qpos[self.model.jnt_qposadr[joint_id]] = joint_positions[
                    side_index, joint_index
                ]
        mujoco.mj_forward(self.model, data)
        return np.stack(
            [
                data.site_xpos[
                    mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_hand"
                    )
                ].copy()
                for side in ("left", "right")
            ]
        )

    def test_vectorized_fk_matches_mujoco_robot_xml(self) -> None:
        cases = torch.tensor(
            [
                MICROBAN_ARM_HOME_JOINT_RAD,
                MICROBAN_ARM_JOINT_LOWER_RAD,
                MICROBAN_ARM_JOINT_UPPER_RAD,
                (
                    (np.deg2rad(7.0), np.deg2rad(21.0), np.deg2rad(-33.0)),
                    (np.deg2rad(-13.0), np.deg2rad(-18.0), np.deg2rad(-11.0)),
                ),
            ],
            dtype=torch.float64,
        )
        actual = microban_hand_positions_from_arm_joints(cases).numpy()
        expected = np.stack(
            [self._mujoco_hand_positions(case.numpy()) for case in cases]
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-15)

    def test_home_offsets_are_exact_zero(self) -> None:
        home = torch.tensor(MICROBAN_ARM_HOME_JOINT_RAD, dtype=torch.float64)
        offsets = microban_hand_offsets_from_arm_joints(home)
        self.assertTrue(torch.equal(offsets, torch.zeros_like(offsets)))
        torch.testing.assert_close(
            microban_default_hand_positions(device="cpu", dtype=torch.float64),
            microban_hand_positions_from_arm_joints(home),
            rtol=0.0,
            atol=0.0,
        )

    def test_joint_box_samples_are_reachable_finite_and_bounded(self) -> None:
        count = 100_000
        active = torch.ones(count, 2, dtype=torch.bool)
        generator = torch.Generator().manual_seed(20260925)
        joints, offsets = sample_microban_reachable_hand_targets(
            active, generator=generator, dtype=torch.float64
        )
        lower = torch.tensor(MICROBAN_ARM_JOINT_LOWER_RAD, dtype=torch.float64)
        upper = torch.tensor(MICROBAN_ARM_JOINT_UPPER_RAD, dtype=torch.float64)
        self.assertTrue(bool(torch.isfinite(joints).all().item()))
        self.assertTrue(bool(torch.isfinite(offsets).all().item()))
        self.assertTrue(bool((joints >= lower).all().item()))
        self.assertTrue(bool((joints <= upper).all().item()))
        torch.testing.assert_close(
            offsets,
            microban_hand_target_offsets_from_arm_joints(joints),
            rtol=0.0,
            atol=0.0,
        )
        bounds = torch.tensor(
            MICROBAN_HAND_TARGET_NORMALIZER_ABS_BOUND_M, dtype=torch.float64
        )
        self.assertTrue(bool((offsets.abs() <= bounds).all().item()))
        self.assertTrue(
            bool(microban_hand_target_offsets_within_limit(offsets).all().item())
        )

        # The samples are the uniform joint box conditioned on the receiver
        # box: their mean matches an independent filter of unconditioned
        # uniform draws (about 1 % removed, all forward-up), and stays within
        # a few mrad of each interval's midpoint.  This also catches
        # accidental Cartesian-box sampling.
        reference = lower + torch.rand(
            (400_000, 2, 3),
            generator=torch.Generator().manual_seed(99),
            dtype=torch.float64,
        ) * (upper - lower)
        reference_offsets = microban_hand_target_offsets_from_arm_joints(reference)
        expected_mean = torch.stack(
            [
                reference[:, side][
                    microban_hand_target_offsets_within_limit(
                        reference_offsets[:, side]
                    )
                ].mean(dim=0)
                for side in range(2)
            ]
        )
        torch.testing.assert_close(
            joints.mean(dim=0), expected_mean, rtol=0.0, atol=2.0e-3
        )
        torch.testing.assert_close(
            joints.mean(dim=0), (lower + upper) * 0.5, rtol=0.0, atol=1.0e-2
        )

    def test_rejection_removes_only_targets_outside_the_receiver_box(self) -> None:
        count = 200_000
        lower = torch.tensor(MICROBAN_ARM_JOINT_LOWER_RAD, dtype=torch.float64)
        upper = torch.tensor(MICROBAN_ARM_JOINT_UPPER_RAD, dtype=torch.float64)
        generator = torch.Generator().manual_seed(11)
        joints = lower + torch.rand(
            (count, 2, 3), generator=generator, dtype=torch.float64
        ) * (upper - lower)
        inside = microban_hand_target_offsets_within_limit(
            microban_hand_target_offsets_from_arm_joints(joints)
        )
        # The unrestricted joint box loses about 1 % per hand (0.997 % of the
        # 401^3 grid), all forward: the box never leaves +-64 mm elsewhere.
        excluded = 1.0 - inside.double().mean(dim=0)
        self.assertTrue(bool(((excluded > 0.008) & (excluded < 0.012)).all()))
        outside_offsets = microban_hand_target_offsets_from_arm_joints(joints)[~inside]
        self.assertTrue(bool((outside_offsets[:, 0] > 0.064).all().item()))
        # The receiver keeps the endpoints and drops anything past them, in
        # float64 even for float32 targets.
        limit = MICROBAN_HAND_TARGET_RUNTIME_VALIDATED_ABS_LIMIT_M[0]
        self.assertEqual(limit, 0.8 * 0.08)
        edge = torch.tensor(
            [[limit, -limit, 0.0], [math.nextafter(limit, 1.0), 0.0, 0.0]],
            dtype=torch.float64,
        )
        self.assertEqual(
            microban_hand_target_offsets_within_limit(edge).tolist(), [True, False]
        )
        float32_limit = torch.tensor([[limit, 0.0, 0.0]], dtype=torch.float32)
        self.assertGreater(float(float32_limit[0, 0]), limit)
        self.assertFalse(
            bool(microban_hand_target_offsets_within_limit(float32_limit).item())
        )
        self.assertGreater(MICROBAN_HAND_TARGET_MAX_REJECTION_ROUNDS, 8)

    def test_exhausted_rejection_falls_back_to_home(self) -> None:
        # A box only HOME fits in: every draw is rejected, so after the
        # bounded number of rounds the active hands hold HOME (zero offset).
        joints, offsets = sample_microban_reachable_hand_targets(
            torch.tensor([[True, False], [True, True]]),
            generator=torch.Generator().manual_seed(1),
            dtype=torch.float64,
            abs_limit_m=(1.0e-9, 1.0e-9, 1.0e-9),
        )
        home = torch.tensor(MICROBAN_ARM_HOME_JOINT_RAD, dtype=torch.float64)
        self.assertTrue(torch.equal(joints, home.expand(2, -1, -1)))
        self.assertTrue(torch.equal(offsets, torch.zeros_like(offsets)))

    def test_float32_samples_stay_inside_the_receiver_box(self) -> None:
        active = torch.ones(200_000, 2, dtype=torch.bool)
        joints, offsets = sample_microban_reachable_hand_targets(
            active, generator=torch.Generator().manual_seed(3)
        )
        self.assertEqual(offsets.dtype, torch.float32)
        limit = torch.tensor(
            MICROBAN_HAND_TARGET_RUNTIME_VALIDATED_ABS_LIMIT_M, dtype=torch.float64
        )
        self.assertTrue(bool((offsets.double().abs() <= limit).all().item()))
        # Not a Cartesian clip: no target sits on the box face, and the
        # forward reach still gets close to it.
        self.assertFalse(bool((offsets.double().abs() == limit).any().item()))
        self.assertGreater(float(offsets[..., 0].max()), 0.0635)
        lower = torch.tensor(MICROBAN_ARM_JOINT_LOWER_RAD)
        upper = torch.tensor(MICROBAN_ARM_JOINT_UPPER_RAD)
        self.assertTrue(bool(((joints >= lower) & (joints <= upper)).all().item()))
        torch.testing.assert_close(
            offsets,
            microban_hand_target_offsets_from_arm_joints(joints),
            rtol=0.0,
            atol=0.0,
        )

    def test_inactive_hands_use_home_joints_and_exact_zero_offsets(self) -> None:
        active = torch.tensor(
            [[False, False], [True, False], [False, True], [True, True]]
        )
        joints, offsets = sample_microban_reachable_hand_targets(
            active,
            generator=torch.Generator().manual_seed(4),
            dtype=torch.float64,
        )
        home = torch.tensor(MICROBAN_ARM_HOME_JOINT_RAD, dtype=torch.float64)
        inactive = ~active
        self.assertTrue(torch.equal(joints[inactive], home.expand(4, -1, -1)[inactive]))
        self.assertTrue(
            torch.equal(offsets[inactive], torch.zeros_like(offsets[inactive]))
        )
        self.assertTrue(bool(torch.count_nonzero(offsets[active]).item() > 0))

    def test_metadata_exposes_joint_box_and_normalizer_bound(self) -> None:
        metadata = microban_hand_fk_metadata()
        self.assertEqual(
            metadata["normalizer_abs_bound_m"],
            list(MICROBAN_HAND_TARGET_NORMALIZER_ABS_BOUND_M),
        )
        self.assertEqual(
            metadata["home_joint_deg"], [[0.0, 10.0, -20.0], [0.0, -10.0, -20.0]]
        )
        self.assertEqual(
            metadata["joint_upper_deg"],
            [[25.0, 30.0, -10.0], [25.0, -10.0, -10.0]],
        )
        self.assertEqual(
            metadata["bound_grid_points_per_axis"],
            MICROBAN_HAND_FK_BOUND_GRID_POINTS_PER_AXIS,
        )
        self.assertEqual(
            metadata["wire_abs_bound_m"], list(MICROBAN_HAND_TARGET_WIRE_ABS_BOUND_M)
        )
        self.assertIn("uniform_independent_joint_box", metadata["sampling"])

    def test_target_frame_is_the_home_levelled_trunk_frame(self) -> None:
        self.assertEqual(MICROBAN_HAND_TARGET_FRAME_PITCH_RAD, HOME_TRUNK_PITCH_RAD)
        self.assertEqual(
            MICROBAN_HAND_TARGET_FRAME,
            "robot_home_levelled_trunk_xyz_forward_left_up",
        )
        metadata = microban_hand_fk_metadata()
        self.assertEqual(metadata["target_frame"], MICROBAN_HAND_TARGET_FRAME)
        self.assertEqual(
            metadata["target_frame_trunk_pitch_rad"], HOME_TRUNK_PITCH_RAD
        )
        # Trunk z (forward-up at the lean HOME) reads forward and up in the
        # levelled frame: R_y(+lean) @ (0, 0, 1) = (sin, 0, cos).
        rotated = rotate_trunk_offsets_to_home_levelled(
            torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=torch.float64),
            HOME_TRUNK_PITCH_RAD,
        )
        sine = np.sin(HOME_TRUNK_PITCH_RAD)
        cosine = np.cos(HOME_TRUNK_PITCH_RAD)
        np.testing.assert_allclose(
            rotated.numpy(),
            [[sine, 0.0, cosine], [cosine, 0.0, -sine]],
            rtol=0.0,
            atol=1.0e-15,
        )
        # Zero lean is the plain trunk frame, and the trunk-frame sampler
        # returns the raw FK offsets.
        joints, offsets = sample_microban_reachable_hand_targets(
            torch.ones(64, 2, dtype=torch.bool),
            generator=torch.Generator().manual_seed(7),
            dtype=torch.float64,
            trunk_pitch=0.0,
        )
        torch.testing.assert_close(
            offsets, microban_hand_offsets_from_arm_joints(joints), rtol=0.0, atol=0.0
        )

    def test_target_aabb_is_the_rotated_joint_box_fk(self) -> None:
        points = 61
        lower = torch.tensor(MICROBAN_ARM_JOINT_LOWER_RAD, dtype=torch.float64)
        upper = torch.tensor(MICROBAN_ARM_JOINT_UPPER_RAD, dtype=torch.float64)
        home = torch.tensor(MICROBAN_ARM_HOME_JOINT_RAD, dtype=torch.float64)
        for side in range(2):
            axes = [
                torch.linspace(lower[side, k], upper[side, k], points, dtype=torch.float64)
                for k in range(3)
            ]
            grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
            joints = home.expand(grid.shape[0], -1, -1).clone()
            joints[:, side] = grid
            trunk = microban_hand_offsets_from_arm_joints(joints)[:, side]
            target = microban_hand_target_offsets_from_arm_joints(joints)[:, side]
            # The target AABB covers only the receiver-box subset.
            target = target[microban_hand_target_offsets_within_limit(target)]
            for values, minimum, maximum in (
                (
                    trunk,
                    MICROBAN_HAND_FK_TRUNK_OFFSET_AABB_MIN_M[side],
                    MICROBAN_HAND_FK_TRUNK_OFFSET_AABB_MAX_M[side],
                ),
                (
                    target,
                    MICROBAN_HAND_FK_OFFSET_AABB_MIN_M[side],
                    MICROBAN_HAND_FK_OFFSET_AABB_MAX_M[side],
                ),
            ):
                minimum = torch.tensor(minimum, dtype=torch.float64)
                maximum = torch.tensor(maximum, dtype=torch.float64)
                # The recorded 401-point extrema bound this coarser grid and
                # lie within 0.5 mm of it (the receiver-box cut of the coarse
                # grid lands up to ~0.4 mm short of the 64 mm face).
                self.assertTrue(bool((values >= minimum - 1.0e-12).all().item()))
                self.assertTrue(bool((values <= maximum + 1.0e-12).all().item()))
                torch.testing.assert_close(
                    values.amin(dim=0), minimum, rtol=0.0, atol=5.0e-4
                )
                torch.testing.assert_close(
                    values.amax(dim=0), maximum, rtol=0.0, atol=5.0e-4
                )

    def test_grid_aabb_fits_normalizer_and_runtime_live_margin(self) -> None:
        minimum = torch.tensor(MICROBAN_HAND_FK_OFFSET_AABB_MIN_M)
        maximum = torch.tensor(MICROBAN_HAND_FK_OFFSET_AABB_MAX_M)
        observed_abs_max = torch.maximum(minimum.abs(), maximum.abs()).amax(dim=0)
        normalizer = torch.tensor(MICROBAN_HAND_TARGET_NORMALIZER_ABS_BOUND_M)
        live_limit = torch.tensor(MICROBAN_HAND_TARGET_RUNTIME_VALIDATED_ABS_LIMIT_M)
        self.assertTrue(bool((observed_abs_max <= normalizer).all().item()))
        # Outward 0.1 mm rounding: the normalizer is no looser than needed.
        self.assertTrue(bool((normalizer - observed_abs_max < 1.0e-4).all().item()))
        self.assertTrue(bool((normalizer <= live_limit).all().item()))
        self.assertEqual(
            MICROBAN_HAND_TARGET_RUNTIME_VALIDATED_ABS_LIMIT_M, (0.064, 0.064, 0.064)
        )
        # Every named evaluation pose is sendable through the live receiver,
        # with at least 1 mm to spare.
        for _name, offsets in microban_reachable_hand_evaluation_offsets():
            self.assertTrue(
                bool(
                    (torch.tensor(offsets).abs() < live_limit - 1.0e-3).all().item()
                )
            )

    def test_named_evaluation_offsets_are_exact_fk_and_bilateral(self) -> None:
        offsets_by_name = dict(microban_reachable_hand_evaluation_offsets())
        self.assertEqual(tuple(offsets_by_name), ("F", "B", "f", "b"))
        for name, left_degrees in MICROBAN_REACHABLE_HAND_EVALUATION_JOINTS_DEG:
            pitch, roll, elbow = left_degrees
            self.assertGreaterEqual(pitch, -25.0)
            self.assertLessEqual(pitch, 25.0)
            self.assertGreaterEqual(roll, 10.0)
            self.assertLessEqual(roll, 30.0)
            self.assertGreaterEqual(elbow, -50.0)
            self.assertLessEqual(elbow, -10.0)
            joints = torch.deg2rad(
                torch.tensor(
                    ((pitch, roll, elbow), (pitch, -roll, elbow)),
                    dtype=torch.float64,
                )
            )
            expected = microban_hand_target_offsets_from_arm_joints(joints)
            actual = torch.tensor(offsets_by_name[name], dtype=torch.float64)
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
            torch.testing.assert_close(actual[0, (0, 2)], actual[1, (0, 2)])
            self.assertAlmostEqual(actual[0, 1].item(), -actual[1, 1].item())


if __name__ == "__main__":
    unittest.main()
