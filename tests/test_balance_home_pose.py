"""config/balance_home_pose.py: re-balance the HOME with hip/ankle pitch only."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from mjlab_microban.robot import home_pose
from mjlab_microban.robot.home_pose import (
    HOME_JOINT_NAMES,
    home_pose_from_values,
    load_home_pose,
    rewrite_home_pose_yaml,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "balance_home_pose", REPO_ROOT / "config" / "balance_home_pose.py"
)
balance = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = balance
_SPEC.loader.exec_module(balance)

CENTERED_PITCH_DEG = 1.198384259489
CENTERED_ROOT_Z = 0.170554885633559
# The forward-lean HOME solved on 2026-10-04 (lean/solve_pose.py, fsolve on
# right sole normal x and COM x - sole centre; branch forward-lean-v2).
LEAN_HIP_DEG = -14.166561199931119
LEAN_ANKLE_DEG = 4.127976841869204
LEAN_ROOT_Z = 0.170430569776402
# Requested agreement with the reference solves.
REFERENCE_TOLERANCE_DEG = 1.0e-6
PITCH_JOINTS = set(balance.BALANCED_JOINTS)


def yaml_joints() -> dict[str, float]:
    return dict(load_home_pose().joint_pos_deg)


def with_pairs(joints: dict[str, float], **pairs: float) -> dict[str, float]:
    joints = dict(joints)
    for joint, value in pairs.items():
        joints[f"left_{joint}"] = value
        joints[f"right_{joint}"] = value
    return joints


def run_cli(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = balance.main(list(argv))
    return code, out.getvalue(), err.getvalue()


class SolverTest(unittest.TestCase):
    def assert_balanced(self, result, trunk_pitch_deg: float) -> None:
        after = result.after
        self.assertLessEqual(
            abs(after.flat_sole_trunk_pitch_rad - math.radians(trunk_pitch_deg)),
            1.0e-13,
        )
        self.assertLessEqual(
            max(abs(v - math.radians(0.0)) for v in after.sole_pitch_rad), 1.0e-13
        )
        self.assertLessEqual(abs(after.com_offset_x), 1.0e-13)  # 1e-10 mm
        self.assertEqual(after.sole_contact_corner_count, 48)
        self.assertEqual(after.trunk_pitch_rad, math.radians(trunk_pitch_deg))
        for joint in ("hip_pitch", "ankle_pitch"):
            self.assertEqual(
                result.after_deg[f"left_{joint}"], result.after_deg[f"right_{joint}"]
            )
        # The loader accepts it as a HOME.
        home_pose_from_values(
            joint_pos_deg=result.after_deg, trunk_pitch_deg=trunk_pitch_deg
        )

    def test_centered_yaml_is_a_fixed_point(self):
        result = balance.balance_home_yaml(home_pose.HOME_POSE_YAML)
        self.assertFalse(result.changed)
        self.assertEqual(result.method, "already balanced")
        self.assertEqual(dict(result.after_deg), dict(result.before_deg))
        self.assertEqual(result.after_deg["left_hip_pitch"], CENTERED_PITCH_DEG)
        self.assertEqual(result.after_deg["left_ankle_pitch"], -CENTERED_PITCH_DEG)
        self.assertEqual(round(result.after.root_pos[2], 15), CENTERED_ROOT_Z)
        self.assertEqual(round(result.after.heel_margin_m * 1e3), 31)
        self.assertEqual(round(result.after.toe_margin_m * 1e3), 31)
        self.assertAlmostEqual(result.after.heel_margin_m * 1e3, 30.83, delta=1.0e-6)
        self.assertAlmostEqual(result.after.toe_margin_m * 1e3, 30.83, delta=1.0e-6)
        self.assert_balanced(result, 0.0)

    def test_perturbed_pitches_converge_back_to_the_centered_solve(self):
        start = with_pairs(yaml_joints(), hip_pitch=3.0, ankle_pitch=-3.0)
        result = balance.balance_joint_pos(start, 0.0)
        self.assertTrue(result.changed)
        self.assertAlmostEqual(
            result.after_deg["left_hip_pitch"], CENTERED_PITCH_DEG, delta=1.0e-10
        )
        self.assertAlmostEqual(
            result.after_deg["left_ankle_pitch"], -CENTERED_PITCH_DEG, delta=1.0e-10
        )
        self.assert_balanced(result, 0.0)

    def test_trunk_pitch_10_reproduces_the_forward_lean_solve(self):
        result = balance.balance_home_yaml(
            home_pose.HOME_POSE_YAML, trunk_pitch_deg=10.0
        )
        self.assertTrue(result.changed)
        self.assertAlmostEqual(
            result.after_deg["left_hip_pitch"],
            LEAN_HIP_DEG,
            delta=REFERENCE_TOLERANCE_DEG,
        )
        self.assertAlmostEqual(
            result.after_deg["left_ankle_pitch"],
            LEAN_ANKLE_DEG,
            delta=REFERENCE_TOLERANCE_DEG,
        )
        # In fact the two solves agree to floating-point noise.
        self.assertAlmostEqual(
            result.after_deg["left_hip_pitch"], LEAN_HIP_DEG, delta=1.0e-12
        )
        self.assertAlmostEqual(
            result.after_deg["left_ankle_pitch"], LEAN_ANKLE_DEG, delta=1.0e-12
        )
        self.assertEqual(round(result.after.root_pos[2], 15), LEAN_ROOT_Z)
        half = math.radians(10.0) / 2.0
        self.assertEqual(
            result.after.root_quat_wxyz,
            (float(np.cos(half)), 0.0, float(np.sin(half)), 0.0),
        )
        self.assertAlmostEqual(result.after.heel_margin_m * 1e3, 30.96, delta=0.005)
        self.assertAlmostEqual(result.after.toe_margin_m * 1e3, 30.96, delta=0.005)
        self.assert_balanced(result, 10.0)
        for name in HOME_JOINT_NAMES:
            if name not in PITCH_JOINTS:
                self.assertEqual(result.after_deg[name], result.before_deg[name])

    def test_perturbations_keep_every_other_joint(self):
        base = yaml_joints()
        cases = {
            "knee 15": (with_pairs(base, knee=15.0), 0.0),
            "shoulder pitch 30": (with_pairs(base, shoulder_pitch=30.0), 0.0),
            "elbow -60": (with_pairs(base, elbow=-60.0), 0.0),
            "trunk 5": (base, 5.0),
            "all of them": (
                with_pairs(base, knee=15.0, shoulder_pitch=30.0, elbow=-60.0),
                5.0,
            ),
            "deep crouch": (with_pairs(base, knee=100.0), 0.0),
        }
        for case, (joints, trunk) in cases.items():
            with self.subTest(case=case):
                result = balance.balance_joint_pos(joints, trunk)
                self.assertTrue(result.changed)
                self.assertEqual(result.trunk_pitch_deg, trunk)
                for name in HOME_JOINT_NAMES:
                    if name not in PITCH_JOINTS:
                        self.assertEqual(result.after_deg[name], joints[name], name)
                self.assert_balanced(result, trunk)
                # Balancing the result again is a no-op.
                again = balance.balance_joint_pos(result.after_deg, trunk)
                self.assertFalse(again.changed)
                self.assertEqual(dict(again.after_deg), dict(result.after_deg))

    def test_scan_and_brent_fallback_matches_newton(self):
        joints = with_pairs(yaml_joints(), knee=15.0)
        newton = balance.balance_joint_pos(joints, 0.0)
        self.assertEqual(newton.method, "newton")
        with mock.patch.object(balance, "MAX_NEWTON_ITERATIONS", 0):
            fallback = balance.balance_joint_pos(joints, 0.0)
        self.assertEqual(fallback.method, "scan + brent")
        for name in PITCH_JOINTS:
            self.assertAlmostEqual(
                fallback.after_deg[name], newton.after_deg[name], delta=1.0e-10
            )
        self.assert_balanced(fallback, 0.0)

    def test_infeasible_requests_are_refused(self):
        base = yaml_joints()
        with self.assertRaisesRegex(balance.BalanceError, "outside its MJCF range"):
            balance.balance_joint_pos(with_pairs(base, knee=-44.0), 40.0)
        with self.assertRaisesRegex(balance.BalanceError, "soft limits"):
            balance.balance_joint_pos(with_pairs(base, knee=-44.0), 20.0)
        with self.assertRaisesRegex(balance.BalanceError, "outside"):
            balance.balance_joint_pos(base, 60.0)
        # Restricted to a hip range that does not contain the solution, the
        # scan proves the COM cannot reach the sole centre.
        base_rad = {name: math.radians(value) for name, value in base.items()}
        ranges = dict(
            balance._pitch_ranges(), hip_pitch=(math.radians(10), math.radians(20))
        )
        with self.assertRaisesRegex(balance.BalanceError, "COM cannot reach"):
            balance._bracketed(
                base_rad, math.radians(15), math.radians(-15), 0.0, ranges
            )

    def test_non_mirror_and_invalid_inputs_are_refused(self):
        base = yaml_joints()
        for joint in ("hip_pitch", "ankle_pitch"):
            joints = dict(base, **{f"left_{joint}": 1.0, f"right_{joint}": 2.0})
            with (
                self.subTest(joint=joint),
                self.assertRaisesRegex(balance.BalanceError, "not mirror-consistent"),
            ):
                balance.balance_joint_pos(joints, 0.0)
        with self.assertRaisesRegex(balance.BalanceError, "mirror symmetric"):
            balance.balance_joint_pos(dict(base, left_knee=10.0), 0.0)
        missing = dict(base)
        del missing["left_knee"]
        with self.assertRaisesRegex(balance.BalanceError, "all 21 joints"):
            balance.balance_joint_pos(missing, 0.0)

    def test_soft_limit_factor_matches_the_robot_cfg(self):
        from mjlab_microban.robot.microban_constants import MICROBAN_ROBOT_CFG

        self.assertEqual(
            balance.SOFT_JOINT_POS_LIMIT_FACTOR,
            MICROBAN_ROBOT_CFG.articulation.soft_joint_pos_limit_factor,
        )


class CliTest(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.path = self.directory / "home_pose.yaml"
        shutil.copyfile(home_pose.HOME_POSE_YAML, self.path)

    def tearDown(self):
        shutil.rmtree(self.directory)

    def changed_lines(self, before: str, after: str) -> list[tuple[str, str]]:
        old, new = before.splitlines(), after.splitlines()
        self.assertEqual(len(old), len(new))
        return [(a, b) for a, b in zip(old, new) if a != b]

    def test_dry_run_and_check_on_the_centered_yaml(self):
        before = self.path.read_text()
        code, out, _ = run_cli("--yaml", str(self.path))
        self.assertEqual(code, 0)
        self.assertIn("Already balanced", out)
        self.assertIn("COM - sole centre", out)
        self.assertEqual(run_cli("--yaml", str(self.path), "--check")[0], 0)
        self.assertEqual(run_cli("--yaml", str(self.path), "--write")[0], 0)
        self.assertEqual(self.path.read_text(), before)

    def test_write_touches_only_the_four_pitch_values(self):
        rewrite_home_pose_yaml(
            self.path, joint_pos_deg={"left_knee": 15.0, "right_knee": 15.0}
        )
        with self.assertRaisesRegex(ValueError, "not flat"):
            load_home_pose(self.path)
        edited = self.path.read_text()

        code, out, _ = run_cli("--yaml", str(self.path))
        self.assertEqual(code, 0)
        self.assertIn("Dry run", out)
        self.assertEqual(self.path.read_text(), edited)
        self.assertEqual(run_cli("--yaml", str(self.path), "--check")[0], 1)

        code, out, _ = run_cli("--yaml", str(self.path), "--write")
        self.assertEqual(code, 0)
        self.assertIn("Next steps", out)
        written = self.path.read_text()
        changes = self.changed_lines(edited, written)
        self.assertEqual(
            sorted(old.split(":")[0].strip() for old, _ in changes),
            sorted(PITCH_JOINTS),
        )
        for old, new in changes:
            self.assertEqual(
                old.split(":")[0], new.split(":")[0]
            )  # same key and indent
            self.assertFalse(old.lstrip().startswith("#"))
        home = load_home_pose(self.path)
        self.assertEqual(home.joint_pos_deg["left_knee"], 15.0)
        self.assertEqual(home.trunk_pitch_deg, 0.0)
        self.assertLessEqual(abs(home.analysis.com_offset_x), 1.0e-13)
        # Full float precision: the YAML holds exactly the solved values.
        result = balance.balance_joint_pos(with_pairs(yaml_joints(), knee=15.0), 0.0)
        self.assertEqual(dict(home.joint_pos_deg), dict(result.after_deg))

        # Idempotent: a second run changes nothing.
        code, out, _ = run_cli("--yaml", str(self.path), "--write")
        self.assertEqual(code, 0)
        self.assertIn("Already balanced", out)
        self.assertEqual(self.path.read_text(), written)
        self.assertEqual(run_cli("--yaml", str(self.path), "--check")[0], 0)

    def test_write_with_a_new_trunk_pitch(self):
        before = self.path.read_text()
        code, out, _ = run_cli(
            "--yaml", str(self.path), "--trunk-pitch-deg", "10", "--write"
        )
        self.assertEqual(code, 0)
        self.assertIn("only trunk_pitch_deg 0.0", out)
        changes = self.changed_lines(before, self.path.read_text())
        self.assertEqual(
            sorted(old.split(":")[0].strip() for old, _ in changes),
            sorted(PITCH_JOINTS | {"trunk_pitch_deg"}),
        )
        home = load_home_pose(self.path)
        self.assertEqual(home.trunk_pitch_deg, 10.0)
        self.assertAlmostEqual(home.hip_pitch_deg, LEAN_HIP_DEG, delta=1.0e-12)
        self.assertAlmostEqual(home.ankle_pitch_deg, LEAN_ANKLE_DEG, delta=1.0e-12)
        self.assertEqual(home.root_pos[2], LEAN_ROOT_Z)
        # Keeping the (new) YAML trunk pitch is a no-op.
        self.assertEqual(run_cli("--yaml", str(self.path), "--check")[0], 0)

    def test_errors_exit_1_and_leave_the_yaml_alone(self):
        rewrite_home_pose_yaml(self.path, joint_pos_deg={"left_hip_pitch": 2.0})
        before = self.path.read_text()
        code, _, err = run_cli("--yaml", str(self.path), "--write")
        self.assertEqual(code, 1)
        self.assertIn("not mirror-consistent", err)
        self.assertEqual(self.path.read_text(), before)


class LazyHomeTest(unittest.TestCase):
    def test_module_import_does_not_load_the_yaml(self):
        code = (
            "import mjlab_microban.robot.home_pose as h\n"
            "assert 'HOME' not in vars(h)\n"
            "home = h.HOME\n"
            "assert vars(h)['HOME'] is home\n"
            "from mjlab_microban.robot.home_pose import HOME\n"
            "assert HOME is home\n"
        )
        subprocess.run([sys.executable, "-c", code], check=True, cwd=REPO_ROOT)


if __name__ == "__main__":
    unittest.main()
