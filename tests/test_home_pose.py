"""config/home_pose.yaml is the single HOME source and reproduces today's HOME."""

from __future__ import annotations

import math
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from home_cases import CENTERED_HOME_YAML, centered_home_only, home_yaml_override  # noqa: E402

from mjlab_microban.robot import home_pose
from mjlab_microban.robot.home_pose import (
    HOME,
    HOME_JOINT_NAMES,
    analyze_pose,
    home_pose_from_values,
    load_home_pose,
    rewrite_home_pose_yaml,
    signed_degree_token,
)

# The centered fixture HOME (tests/fixtures/home_pose_centered.yaml).
CENTERED_DEG = {
    "head": 0.0,
    "neck_roll": 0.0,
    "neck_pitch": 0.0,
    "left_shoulder_roll": 10.0,
    "right_shoulder_roll": -10.0,
    "left_shoulder_pitch": 0.0,
    "right_shoulder_pitch": 0.0,
    "left_elbow": -20.0,
    "right_elbow": -20.0,
    "left_hip_roll": 5.0,
    "right_hip_roll": -5.0,
    "left_hip_pitch": 1.198384259489,
    "right_hip_pitch": 1.198384259489,
    "left_hip_yaw": 0.0,
    "right_hip_yaw": 0.0,
    "left_knee": 0.0,
    "right_knee": 0.0,
    "left_ankle_roll": -5.0,
    "right_ankle_roll": 5.0,
    "left_ankle_pitch": -1.198384259489,
    "right_ankle_pitch": -1.198384259489,
}
# FK root z 0.17055488563355944 rounded to 1e-12 m (home_pose.ROOT_Z_DECIMALS).
CENTERED_ROOT_Z = 0.170554885634
# The forward-lean HOME's unrounded hip/ankle pitch and its published root z.
LEAN_HIP_DEG = -14.166561199931119
LEAN_ANKLE_DEG = 4.127976841869204
LEAN_ROOT_Z = 0.170430569776402


def lean_joints_deg() -> dict[str, float]:
    joints = dict(CENTERED_DEG)
    for side in ("left", "right"):
        joints[f"{side}_hip_pitch"] = LEAN_HIP_DEG
        joints[f"{side}_ankle_pitch"] = LEAN_ANKLE_DEG
    return joints


def radians(joints_deg):
    return {name: float(np.deg2rad(value)) for name, value in joints_deg.items()}


def centered_home():
    """The centered HOME, loaded from its fixture (any checkout HOME)."""

    return load_home_pose(CENTERED_HOME_YAML)


@centered_home_only
class CenteredHomeIsReproducedTest(unittest.TestCase):
    """The centered fixture HOME gives its FK values and the vertical-trunk hand box."""

    def test_yaml_inputs(self):
        self.assertEqual(HOME.path, Path(home_yaml_override() or home_pose.HOME_POSE_YAML))
        self.assertEqual(HOME.label, "centered_home")
        self.assertEqual(HOME.trunk_pitch_deg, 0.0)
        self.assertEqual(dict(HOME.joint_pos_deg), CENTERED_DEG)
        self.assertEqual(tuple(HOME.joint_pos_deg), HOME_JOINT_NAMES)

    def test_home_frame_is_bit_identical(self):
        from mjlab_microban.robot.microban_constants import (
            HOME_FRAME,
            HOME_PROJECTED_GRAVITY,
            HOME_TRUNK_PITCH_RAD,
        )

        self.assertEqual(HOME_FRAME.pos, (0.0, 0.0, CENTERED_ROOT_Z))
        self.assertEqual(HOME_FRAME.rot, (1.0, 0.0, 0.0, 0.0))
        self.assertEqual(
            list(HOME_FRAME.joint_pos.items()),
            [(name, float(np.deg2rad(value))) for name, value in CENTERED_DEG.items()],
        )
        self.assertEqual(HOME_TRUNK_PITCH_RAD, 0.0)
        self.assertEqual(HOME_PROJECTED_GRAVITY, (0.0, 0.0, -1.0))

    def test_hand_fk_contract(self):
        from mjlab_microban.robot import microban_hand_fk as fk

        self.assertEqual(fk.MICROBAN_ARM_HOME_JOINT_DEG, ((0.0, 10.0, -20.0), (0.0, -10.0, -20.0)))
        self.assertEqual(
            fk.MICROBAN_HAND_FK_REVISION,
            "microban_robot_xml_arm_fk_reachable_box_elbow_upper_minus10_v2",
        )
        self.assertEqual(fk.MICROBAN_HAND_TARGET_NORMALIZER_ABS_BOUND_M, (0.0630, 0.0388, 0.0605))
        self.assertEqual(
            fk.MICROBAN_HAND_FK_OFFSET_AABB_MAX_M[0],
            (0.06289464331528255, 0.0387512193701912, 0.060477220857479266),
        )

    def test_fk_analysis_matches_the_centered_solve(self):
        analysis = HOME.analysis
        self.assertEqual(round(analysis.root_pos[2], 12), CENTERED_ROOT_Z)
        self.assertLess(abs(analysis.com_offset_x), 1.0e-12)
        self.assertAlmostEqual(analysis.heel_margin_m, 0.03083, delta=1.0e-9)
        self.assertAlmostEqual(analysis.toe_margin_m, 0.03083, delta=1.0e-9)
        self.assertLess(abs(analysis.flat_sole_trunk_pitch_rad), 1.0e-12)
        self.assertLess(max(abs(value) for value in analysis.sole_pitch_rad), 1.0e-12)
        self.assertEqual(analysis.sole_contact_corner_count, 48)
        self.assertAlmostEqual(analysis.head_height_m, 0.2965341, delta=1.0e-7)
        self.assertAlmostEqual(analysis.feet_lateral_m, 0.0934618, delta=1.0e-7)
        self.assertAlmostEqual(analysis.total_mass_kg, 1.13601, delta=1.0e-5)


class AnalyzePoseTest(unittest.TestCase):
    def test_forward_lean_reference_solution(self):
        lean = home_pose_from_values(
            joint_pos_deg=lean_joints_deg(), trunk_pitch_deg=10.0, label="forward_lean_home"
        )
        # Full-digit lean values: not the pinned (12-decimal) lean HOME, so the
        # published root z is the FK value rounded to 1e-12 m.
        self.assertAlmostEqual(lean.analysis.root_pos[2], LEAN_ROOT_Z, delta=1.0e-15)
        self.assertEqual(lean.root_pos[2], round(lean.analysis.root_pos[2], 12))
        self.assertTrue(lean.analysis.soles_on_floor)
        self.assertEqual(
            lean.root_quat_wxyz,
            (
                float(np.cos(np.deg2rad(10.0) / 2.0)),
                0.0,
                float(np.sin(np.deg2rad(10.0) / 2.0)),
                0.0,
            ),
        )
        self.assertLess(abs(lean.analysis.flat_sole_trunk_pitch_rad - np.deg2rad(10.0)), 1.0e-12)
        self.assertLess(abs(lean.analysis.com_offset_x), 1.0e-12)
        self.assertAlmostEqual(lean.analysis.heel_margin_m * 1e3, 30.96, delta=0.005)
        self.assertAlmostEqual(lean.analysis.toe_margin_m * 1e3, 30.96, delta=0.005)
        self.assertEqual(lean.head_standing_height_m, 0.2953)
        self.assertEqual(lean.feet_lateral_m, 0.0941)
        self.assertEqual(
            lean.projected_gravity,
            (math.sin(math.radians(10.0)), 0.0, -math.cos(math.radians(10.0))),
        )
        self.assertEqual(lean.tag, f"forward_lean_home_{lean.joint_hash}")

    def test_arbitrary_pose_and_level_sole_pitch(self):
        joints = radians(CENTERED_DEG)
        joints["left_knee"] = joints["right_knee"] = 0.3
        level = analyze_pose(joints)
        self.assertLess(max(abs(value) for value in level.sole_pitch_rad), 1.0e-12)
        self.assertNotAlmostEqual(level.trunk_pitch_rad, 0.0, places=3)
        tilted = analyze_pose(joints, 0.0)
        for actual in tilted.sole_pitch_rad:
            self.assertAlmostEqual(actual, -level.trunk_pitch_rad, places=12)
        # Root z puts the lowest sole corner on the ground at any pitch.
        self.assertGreater(tilted.root_pos[2], 0.1)

    def test_cheap_enough_for_a_solver_loop(self):
        joints = dict(HOME.joint_pos_rad)
        start = time.perf_counter()
        for _ in range(200):
            analyze_pose(joints, 0.0)
        self.assertLess((time.perf_counter() - start) / 200, 2.0e-3)

    def test_invalid_homes_are_refused(self):
        missing = dict(CENTERED_DEG)
        del missing["head"]
        asymmetric = dict(CENTERED_DEG, left_knee=1.0)
        asymmetric_roll = dict(CENTERED_DEG, left_hip_roll=6.0)
        out_of_range = dict(CENTERED_DEG, left_knee=400.0, right_knee=400.0)
        tilted = dict(CENTERED_DEG, left_hip_pitch=3.0, right_hip_pitch=3.0)
        head_turned = dict(CENTERED_DEG, head=30.0)
        neck_tilted = dict(CENTERED_DEG, neck_roll=5.0)
        for case, joints in (
            ("missing", missing),
            ("head_turned", head_turned),
            ("neck_tilted", neck_tilted),
            ("asymmetric", asymmetric),
            ("asymmetric_roll", asymmetric_roll),
            ("out_of_range", out_of_range),
            ("soles_not_flat", tilted),
        ):
            with self.subTest(case=case), self.assertRaises(ValueError):
                home_pose_from_values(joint_pos_deg=joints, trunk_pitch_deg=0.0)
        with self.assertRaises(ValueError):
            home_pose_from_values(joint_pos_deg=CENTERED_DEG, trunk_pitch_deg=0.0, label="Bad-Label")

    def test_signed_degree_token(self):
        self.assertEqual(signed_degree_token(1.198384259489), "plus1p198384259489")
        self.assertEqual(signed_degree_token(-1.198384259489), "minus1p198384259489")
        self.assertEqual(signed_degree_token(-14.166561199931119), "minus14p166561199931119")
        self.assertEqual(signed_degree_token(0.0), "plus0p0")


class YamlEditTest(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.path = self.directory / "home_pose.yaml"
        shutil.copyfile(CENTERED_HOME_YAML, self.path)

    def tearDown(self):
        shutil.rmtree(self.directory)

    def test_rewrite_keeps_comments_and_reloads(self):
        before = self.path.read_text().splitlines()
        rewrite_home_pose_yaml(
            self.path,
            joint_pos_deg={
                "left_hip_pitch": LEAN_HIP_DEG,
                "right_hip_pitch": LEAN_HIP_DEG,
                "left_ankle_pitch": LEAN_ANKLE_DEG,
                "right_ankle_pitch": LEAN_ANKLE_DEG,
            },
            trunk_pitch_deg=10.0,
        )
        after = self.path.read_text().splitlines()
        self.assertEqual(len(before), len(after))
        changed = [index for index, (old, new) in enumerate(zip(before, after)) if old != new]
        self.assertEqual(len(changed), 5)
        for index in changed:
            self.assertFalse(before[index].lstrip().startswith("#"))
        self.assertEqual(
            [line for line in before if line.lstrip().startswith("#")],
            [line for line in after if line.lstrip().startswith("#")],
        )
        edited = load_home_pose(self.path)
        self.assertEqual(dict(edited.joint_pos_deg), lean_joints_deg())
        self.assertEqual(edited.trunk_pitch_deg, 10.0)
        self.assertAlmostEqual(edited.root_pos[2], LEAN_ROOT_Z, delta=1.0e-12)
        # A changed HOME gets a derived tag.
        self.assertEqual(edited.tag, f"centered_home_{edited.joint_hash}")
        self.assertNotEqual(edited.joint_hash, centered_home().joint_hash)
        self.assertEqual(edited.feet_lateral_m, 0.0941)

    def test_unflat_edit_is_refused_on_load(self):
        rewrite_home_pose_yaml(self.path, joint_pos_deg={"left_hip_pitch": 3.0, "right_hip_pitch": 3.0})
        with self.assertRaisesRegex(ValueError, "not flat"):
            load_home_pose(self.path)

    def test_unknown_keys_and_joints_are_refused(self):
        with self.assertRaises(KeyError):
            rewrite_home_pose_yaml(self.path, joint_pos_deg={"tail": 1.0})
        document = yaml.safe_load(self.path.read_text())
        document["extra"] = 1
        self.path.write_text(yaml.safe_dump(document))
        with self.assertRaises(ValueError):
            load_home_pose(self.path)


class FloorContactAndRootZPinTest(unittest.TestCase):
    """Rolled soles, the published root z and the HOME stamp tolerance."""

    def centered_deg(self, **pairs) -> dict[str, float]:
        joints = dict(CENTERED_DEG)
        for joint, (left, right) in pairs.items():
            joints[f"left_{joint}"], joints[f"right_{joint}"] = left, right
        return joints

    def test_rolled_soles_are_refused(self):
        # Hip roll 10 alone rolls each sole 5 deg: only an edge (4 corners)
        # touches the floor, so the loader refuses it.
        with self.assertRaisesRegex(ValueError, "not flat on the floor"):
            home_pose_from_values(
                joint_pos_deg=self.centered_deg(hip_roll=(10.0, -10.0)), trunk_pitch_deg=0.0
            )
        analysis = analyze_pose(radians(self.centered_deg(hip_roll=(10.0, -10.0))), 0.0)
        self.assertFalse(analysis.soles_on_floor)
        self.assertEqual(analysis.ground_contact_corner_count, 4)
        self.assertAlmostEqual(math.degrees(analysis.sole_roll_rad[0]), 5.0, delta=1e-9)
        # Ankle roll compensating the hip roll keeps the soles flat.
        flat = home_pose_from_values(
            joint_pos_deg=self.centered_deg(hip_roll=(10.0, -10.0), ankle_roll=(-10.0, 10.0)),
            trunk_pitch_deg=0.0,
        )
        self.assertTrue(flat.analysis.soles_on_floor)
        self.assertEqual(flat.analysis.ground_contact_corner_count, 48)

    def test_current_and_lean_homes_are_on_the_floor(self):
        self.assertTrue(HOME.analysis.soles_on_floor)
        self.assertEqual(HOME.analysis.ground_contact_corner_count, 48)
        # A pitched HOME's 12-decimal hip/ankle values leave the soles level
        # to within FK rounding (5.4e-5 m of lift at the forward-lean HOME).
        self.assertLess(HOME.analysis.sole_contact_lift_m, 1.0e-4)
        self.assertEqual(centered_home().analysis.sole_contact_lift_m, 0.0)
        lean = home_pose_from_values(joint_pos_deg=lean_joints_deg(), trunk_pitch_deg=10.0)
        self.assertTrue(lean.analysis.soles_on_floor)
        self.assertLess(lean.analysis.sole_contact_lift_m, 1.0e-4)

    def test_pinned_root_z_must_match_fk(self):
        original = home_pose.PUBLISHED_HOME_OVERRIDES
        centered = centered_home()
        try:
            home_pose.PUBLISHED_HOME_OVERRIDES = {
                centered.joint_hash: {"tag": "centered_home", "root_z_m": 0.1706}
            }
            with self.assertRaisesRegex(ValueError, "differs from its published value"):
                home_pose_from_values(
                    joint_pos_deg=dict(centered.joint_pos_deg), trunk_pitch_deg=0.0,
                    label="centered_home",
                )
        finally:
            home_pose.PUBLISHED_HOME_OVERRIDES = original

    def test_mjcf_range_message_is_unambiguous(self):
        joints = self.centered_deg(knee=(135.0, 135.0))
        with self.assertRaisesRegex(ValueError, r"134\.9999999999\d* \] deg|above the upper limit"):
            home_pose_from_values(joint_pos_deg=joints, trunk_pitch_deg=0.0)


if __name__ == "__main__":
    unittest.main()
