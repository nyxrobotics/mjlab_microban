"""config/home_pose.yaml is the single HOME source and reproduces today's HOME."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import yaml

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

# The centered HOME of 2026-10-05 (HOME_FRAME before config/home_pose.yaml).
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
CENTERED_ROOT_Z = 0.170554885633559
# The forward-lean HOME solved on 2026-10-04 (branch forward-lean-v2).
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


class CenteredHomeIsReproducedTest(unittest.TestCase):
    def test_yaml_inputs(self):
        self.assertEqual(HOME.path, home_pose.HOME_POSE_YAML)
        self.assertEqual(HOME.label, "centered_home")
        self.assertEqual(HOME.trunk_pitch_deg, 0.0)
        self.assertEqual(dict(HOME.joint_pos_deg), CENTERED_DEG)
        self.assertEqual(tuple(HOME.joint_pos_deg), HOME_JOINT_NAMES)

    def test_home_frame_is_bit_identical(self):
        from mjlab_microban.robot.microban_constants import (
            HOME_FRAME,
            HOME_PITCH_RAD,
            HOME_PROJECTED_GRAVITY,
            HOME_TRUNK_PITCH_RAD,
        )

        self.assertEqual(HOME_FRAME.pos, (0.0, 0.0, CENTERED_ROOT_Z))
        self.assertEqual(HOME_FRAME.rot, (1.0, 0.0, 0.0, 0.0))
        self.assertEqual(
            list(HOME_FRAME.joint_pos.items()),
            [(name, float(np.deg2rad(value))) for name, value in CENTERED_DEG.items()],
        )
        self.assertEqual(HOME_PITCH_RAD, float(np.deg2rad(1.198384259489)))
        self.assertEqual(HOME_TRUNK_PITCH_RAD, 0.0)
        self.assertEqual(HOME_PROJECTED_GRAVITY, (0.0, 0.0, -1.0))

    def test_derived_training_values(self):
        from mjlab_microban.tasks.microban_getup_env_cfg import (
            HEAD_STANDING_HEIGHT,
            HOME_FEET_LATERAL_M,
            STANDING_GATE_HEIGHT,
            STANDING_HEIGHT,
        )

        self.assertEqual(HEAD_STANDING_HEIGHT, 0.2965)
        self.assertEqual(STANDING_GATE_HEIGHT, 0.9 * 0.2965)
        self.assertEqual(HOME_FEET_LATERAL_M, 0.094)
        self.assertEqual(STANDING_HEIGHT, CENTERED_ROOT_Z)

    def test_identity_strings(self):
        from mjlab_microban.robot.home_pose_robot import robot_contract_strings
        from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
            MICROBAN_TELEOP_V12_HOME_POSE_REVISION,
        )

        self.assertEqual(HOME.tag, "centered_home")
        self.assertEqual(
            MICROBAN_TELEOP_V12_HOME_POSE_REVISION,
            "centered_home_hip_plus1p198384259489_ankle_minus1p198384259489_shoulder_zero_v5",
        )
        self.assertEqual(
            robot_contract_strings(),
            {
                "walk_contract_version": "v3_centered_home_servo_range",
                "v12_home_pose_revision": (
                    "centered_home_hip_plus1p198384259489_ankle_minus1p198384259489_"
                    "shoulder_zero_v5"
                ),
                "v12_recipe_revision": (
                    "centered_home_velocity_source_staged_mask_reachable_fk_elbow_minus10_"
                    "raw_prev_action_servo_range_pi_v11"
                ),
                "v12_hand_pose_release_recipe_revision": (
                    "centered_home_velocity_source_staged_mask_reachable_fk_elbow_minus10_"
                    "raw_prev_action_servo_range_pi_active_hand_arm_pose_release_v12"
                ),
                "v12_packager_revision": (
                    "microban_teleop_v12_final_deployment_packager_v6_centered_home_servo_range"
                ),
            },
        )

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
        self.assertEqual(round(analysis.root_pos[2], 15), CENTERED_ROOT_Z)
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
        self.assertEqual(lean.root_pos[2], LEAN_ROOT_Z)
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
        shutil.copyfile(home_pose.HOME_POSE_YAML, self.path)

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
        self.assertEqual(edited.root_pos[2], LEAN_ROOT_Z)
        # A changed HOME drops the centered compatibility overrides.
        self.assertEqual(edited.tag, f"centered_home_{edited.joint_hash}")
        self.assertNotEqual(edited.joint_hash, HOME.joint_hash)
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


class RobotYamlTest(unittest.TestCase):
    def test_robot_document_round_trips_through_yaml(self):
        from mjlab_microban.robot.home_pose_robot import (
            render_robot_home_pose_yaml,
            robot_home_pose_document,
        )
        from mjlab_microban.robot.microban_hand_fk import microban_hand_fk_metadata

        document = robot_home_pose_document()
        parsed = yaml.safe_load(render_robot_home_pose_yaml(document))
        self.assertEqual(parsed, json.loads(json.dumps(document)))
        self.assertEqual(parsed["joint_pos_rad"], dict(HOME.joint_pos_rad))
        self.assertEqual(parsed["root_pos_m"], [0.0, 0.0, CENTERED_ROOT_Z])
        self.assertEqual(parsed["hand_target_fk"], json.loads(json.dumps(microban_hand_fk_metadata())))
        self.assertEqual(parsed["fk"]["com_m"][1], HOME.analysis.com[1])

    def test_exponent_floats_stay_yaml_floats(self):
        from mjlab_microban.robot.home_pose_robot import render_robot_home_pose_yaml

        text = render_robot_home_pose_yaml({"a": 1e-05, "b": [2.5e-17, -3e20]})
        self.assertIn("a: 1.0e-05", text)
        self.assertEqual(yaml.safe_load(text), {"a": 1e-05, "b": [2.5e-17, -3e20]})
        with self.assertRaises(ValueError):
            render_robot_home_pose_yaml({"a": float("nan")})

    def test_write_and_check(self):
        from mjlab_microban.robot.home_pose_robot import write_robot_home_pose

        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / "src").mkdir()
            (repo / "src" / "constants.py").write_text("")
            path, up_to_date = write_robot_home_pose(repo, check=True)
            self.assertFalse(up_to_date)
            self.assertFalse(path.exists())
            write_robot_home_pose(repo)
            self.assertTrue(write_robot_home_pose(repo, check=True)[1])
            path.write_text(path.read_text().replace("0.170554885633559", "0.17"))
            self.assertFalse(write_robot_home_pose(repo, check=True)[1])
            with self.assertRaises(FileNotFoundError):
                write_robot_home_pose(repo / "src")


class WalkHomeStampTest(unittest.TestCase):
    def test_stamp_rules(self):
        from mjlab_microban.tasks.microban_getup_runner import getup_home_pose
        from mjlab_microban.tasks.microban_velocity_runner import (
            WALK_HOME_POSE_INFO_KEY,
            require_walk_home_pose,
        )

        require_walk_home_pose({WALK_HOME_POSE_INFO_KEY: getup_home_pose()})
        # Unstamped walking checkpoints predate the stamp: accepted only at the
        # centered HOME they were trained at.
        require_walk_home_pose({})
        other = getup_home_pose()
        other["root_pos_m"] = [0.0, 0.0, 0.17]
        with self.assertRaises(ValueError):
            require_walk_home_pose({WALK_HOME_POSE_INFO_KEY: other})


if __name__ == "__main__":
    unittest.main()
