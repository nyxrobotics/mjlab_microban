"""Exact-FK and reachable hand-target contract tests."""

from __future__ import annotations

import unittest
from pathlib import Path

import mujoco
import numpy as np
import torch

from mjlab_microban.robot.microban_hand_fk import (
    MICROBAN_ARM_HOME_JOINT_RAD,
    MICROBAN_ARM_JOINT_LOWER_RAD,
    MICROBAN_ARM_JOINT_UPPER_RAD,
    MICROBAN_HAND_FK_BOUND_GRID_POINTS_PER_AXIS,
    MICROBAN_HAND_FK_OFFSET_AABB_MAX_M,
    MICROBAN_HAND_FK_OFFSET_AABB_MIN_M,
    MICROBAN_HAND_TARGET_NORMALIZER_ABS_BOUND_M,
    MICROBAN_HAND_TARGET_RUNTIME_VALIDATED_ABS_LIMIT_M,
    MICROBAN_HAND_TARGET_WIRE_ABS_BOUND_M,
    MICROBAN_REACHABLE_HAND_EVALUATION_JOINTS_DEG,
    microban_default_hand_positions,
    microban_hand_fk_metadata,
    microban_hand_offsets_from_arm_joints,
    microban_hand_positions_from_arm_joints,
    microban_reachable_hand_evaluation_offsets,
    sample_microban_reachable_hand_targets,
)

ROOT = Path(__file__).resolve().parents[1]
ROBOT_XML = ROOT / "src/mjlab_microban/robot/microban/robot.xml"


class MicrobanHandFkTest(unittest.TestCase):
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
            microban_hand_offsets_from_arm_joints(joints),
            rtol=0.0,
            atol=0.0,
        )
        bounds = torch.tensor(
            MICROBAN_HAND_TARGET_NORMALIZER_ABS_BOUND_M, dtype=torch.float64
        )
        self.assertTrue(bool((offsets.abs() <= bounds).all().item()))

        # Independent uniform joint sampling has expectation at each interval's
        # midpoint.  This also catches accidental Cartesian-box sampling.
        expected_mean = (lower + upper) * 0.5
        torch.testing.assert_close(
            joints.mean(dim=0), expected_mean, rtol=0.0, atol=2.0e-3
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

    def test_grid_aabb_fits_normalizer_and_runtime_live_margin(self) -> None:
        minimum = torch.tensor(MICROBAN_HAND_FK_OFFSET_AABB_MIN_M)
        maximum = torch.tensor(MICROBAN_HAND_FK_OFFSET_AABB_MAX_M)
        observed_abs_max = torch.maximum(minimum.abs(), maximum.abs()).amax(dim=0)
        normalizer = torch.tensor(MICROBAN_HAND_TARGET_NORMALIZER_ABS_BOUND_M)
        live_limit = torch.tensor(MICROBAN_HAND_TARGET_RUNTIME_VALIDATED_ABS_LIMIT_M)
        self.assertTrue(bool((observed_abs_max <= normalizer).all().item()))
        self.assertTrue(bool((normalizer < live_limit).all().item()))

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
            expected = microban_hand_offsets_from_arm_joints(joints)
            actual = torch.tensor(offsets_by_name[name], dtype=torch.float64)
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
            torch.testing.assert_close(actual[0, (0, 2)], actual[1, (0, 2)])
            self.assertAlmostEqual(actual[0, 1].item(), -actual[1, 1].item())


if __name__ == "__main__":
    unittest.main()
