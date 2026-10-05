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
        self.assertTrue(result.before_within_tolerance)
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
        with self.assertRaisesRegex(balance.BalanceError, "soft limits"):
            balance.balance_joint_pos(base, 60.0)
        with self.assertRaisesRegex(balance.BalanceError, "< 90 deg"):
            balance.balance_joint_pos(base, 95.0)
        # No arbitrary trunk cap: 50 deg is solvable inside the soft limits.
        steep = balance.balance_joint_pos(base, 50.0)
        self.assertAlmostEqual(steep.after_deg["left_hip_pitch"], -73.719, delta=1.0e-3)
        self.assertAlmostEqual(steep.after_deg["left_ankle_pitch"], 23.491, delta=1.0e-3)
        self.assert_balanced(steep, 50.0)
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
        # The solved joints are validated too, not stood in for.
        for name in PITCH_JOINTS:
            missing = dict(base)
            del missing[name]
            with self.subTest(missing=name), self.assertRaisesRegex(
                balance.BalanceError, f"has no {name}"
            ):
                balance.balance_joint_pos(missing, 0.0)
        for bad, message in (
            (True, "must be a number"),
            ("abc", "must be a number"),
            (None, "must be a number"),
            (math.inf, "must be finite"),
            (math.nan, "must be finite"),
        ):
            joints = with_pairs(base, hip_pitch=bad)
            with self.subTest(value=bad), self.assertRaisesRegex(
                balance.BalanceError, message
            ):
                balance.balance_joint_pos(joints, 0.0)

    def test_solution_does_not_depend_on_the_yaml_pitches(self):
        """One pose, one canonical answer (so one HOME hash), whatever the start."""

        cases = {
            "knee 15": (with_pairs(yaml_joints(), knee=15.0), 0.0),
            "combined": (
                with_pairs(yaml_joints(), knee=15.0, shoulder_pitch=30.0, elbow=-60.0),
                5.0,
            ),
            "lean": (yaml_joints(), 10.0),
        }
        starts = [(1.198384259489, -1.198384259489), (1.2, -1.2), (1.0, -1.0),
                  (60.0, -60.0), (-80.0, 30.0), (45.0, -85.0), (-30.0, 0.0)]
        for case, (joints, trunk) in cases.items():
            answers = set()
            for hip, ankle in starts:
                result = balance.balance_joint_pos(
                    with_pairs(joints, hip_pitch=hip, ankle_pitch=ankle), trunk
                )
                answers.add(
                    (result.after_deg["left_hip_pitch"], result.after_deg["left_ankle_pitch"])
                )
            with self.subTest(case=case):
                self.assertEqual(len(answers), 1, answers)
                for value in next(iter(answers)):
                    self.assertEqual(value, round(value, balance.CANONICAL_DECIMALS))

    def test_canonical_values_are_the_rounded_float_precision_root(self):
        # The centered root to float precision is 1.1983842594888645 deg; the
        # canonical (12-decimal) value is the one the installed policies carry.
        base = with_pairs(yaml_joints(), hip_pitch=1.1983842594888645,
                          ankle_pitch=-1.1983842594888645)
        result = balance.balance_joint_pos(base, 0.0)
        self.assertTrue(result.before_within_tolerance)
        self.assertTrue(result.changed)  # balanced, but not canonical
        self.assertEqual(result.after_deg["left_hip_pitch"], CENTERED_PITCH_DEG)
        self.assertEqual(result.after_deg["left_ankle_pitch"], -CENTERED_PITCH_DEG)

    def test_sole_roll_and_yaw_are_reported(self):
        (roll0, yaw0) = balance.sole_roll_yaw_deg(yaml_joints(), 0.0)
        for value in roll0 + yaw0:
            self.assertLess(abs(value), 1.0e-9)
        lean = balance.balance_joint_pos(yaml_joints(), 10.0)
        roll, yaw = balance.sole_roll_yaw_deg(lean.after_deg, 10.0)
        self.assertAlmostEqual(roll[0], -0.0762, delta=1.0e-3)
        self.assertAlmostEqual(roll[1], 0.0762, delta=1.0e-3)
        self.assertAlmostEqual(yaw[0], -0.8705, delta=1.0e-3)  # toe-in
        self.assertAlmostEqual(yaw[1], 0.8705, delta=1.0e-3)

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
        # The YAML holds exactly the (canonical) solved values.
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

    def test_reverting_an_edit_restores_the_home_identity(self):
        original = self.path.read_bytes()
        identity = load_home_pose(self.path)
        self.assertEqual(identity.tag, "centered_home")
        rewrite_home_pose_yaml(
            self.path, joint_pos_deg={"left_knee": 15.0, "right_knee": 15.0}
        )
        self.assertEqual(run_cli("--yaml", str(self.path), "--write")[0], 0)
        self.assertNotEqual(load_home_pose(self.path).joint_hash, identity.joint_hash)
        rewrite_home_pose_yaml(
            self.path, joint_pos_deg={"left_knee": 0.0, "right_knee": 0.0}
        )
        self.assertEqual(run_cli("--yaml", str(self.path), "--write")[0], 0)
        self.assertEqual(self.path.read_bytes(), original)
        # Trunk 10 and back.
        args = ("--yaml", str(self.path), "--write", "--trunk-pitch-deg")
        self.assertEqual(run_cli(*args, "10")[0], 0)
        self.assertEqual(run_cli(*args, "0")[0], 0)
        self.assertEqual(self.path.read_bytes(), original)
        home = load_home_pose(self.path)
        self.assertEqual((home.joint_hash, home.tag), (identity.joint_hash, "centered_home"))

    def test_crlf_and_quoted_keys_are_kept(self):
        text = self.path.read_text()
        for side in ("left", "right"):
            text = text.replace(f"  {side}_knee: 0.0", f"  {side}_knee: 15.0")
            for joint in ("hip_pitch", "ankle_pitch"):
                text = text.replace(f"  {side}_{joint}:", f'  "{side}_{joint}":')
        self.path.write_bytes(text.replace("\n", "\r\n").encode())
        code, _, err = run_cli("--yaml", str(self.path), "--write")
        self.assertEqual(code, 0, err)
        data = self.path.read_bytes()
        self.assertEqual(data.count(b"\r\n"), data.count(b"\n"))
        self.assertIn(b'  "left_hip_pitch": -6.513901179137\r\n', data)
        self.assertIn(b'  "right_ankle_pitch": -8.486098820863\r\n', data)
        self.assertEqual(load_home_pose(self.path).joint_pos_deg["left_knee"], 15.0)

    def test_tilted_before_column_is_not_reported_as_contact(self):
        rewrite_home_pose_yaml(
            self.path, joint_pos_deg={"left_knee": 15.0, "right_knee": 15.0}
        )
        _, out, _ = run_cli("--yaml", str(self.path))
        rows = {line.split("  ")[0]: line for line in out.splitlines()}
        for row in ("COM - sole centre (x) [mm]", "heel margin [mm]", "toe margin [mm]",
                    "sole contact corners"):
            self.assertIn("n/a (sole pitch +14.942 deg)", rows[row])
        self.assertIn("sole roll L / R [deg]", out)
        self.assertIn("sole yaw L / R [deg]", out)

    def test_bad_files_give_one_line_errors(self):
        good = self.path.read_text()
        cases = {
            "missing pitch": (good.replace("  left_hip_pitch: 1.198384259489\n", ""),
                              "has no left_hip_pitch"),
            "string": (good.replace("_hip_pitch: 1.198384259489", "_hip_pitch: abc"),
                       "must be a number of degrees, got 'abc'"),
            "null": (good.replace("_hip_pitch: 1.198384259489", "_hip_pitch:"),
                     "got None"),
            "bool": (good.replace("_hip_pitch: 1.198384259489", "_hip_pitch: true"),
                     "got True"),
            "inf": (good.replace("_hip_pitch: 1.198384259489", "_hip_pitch: .inf"),
                    "must be finite"),
            "syntax": (good.replace("left_knee: 0.0", "left_knee: [1,"), "invalid YAML"),
            "extra key": (good + "extra: 1\n", "keys must be exactly"),
        }
        for case, (text, message) in cases.items():
            with self.subTest(case=case):
                self.path.write_text(text)
                code, _, err = run_cli("--yaml", str(self.path), "--write")
                self.assertEqual(code, 1)
                self.assertEqual(err.count("\n"), 1, err)
                self.assertTrue(err.startswith(f"error: cannot balance {self.path}: "), err)
                self.assertIn(message, err)
                self.assertEqual(err.count(str(self.path)), 1, err)
                self.assertNotIn("Traceback", err)
                self.assertEqual(self.path.read_text(), text)
        for target, message in ((self.directory / "nope.yaml", "no such file"),
                                (self.directory, "is a directory")):
            code, _, err = run_cli("--yaml", str(target))
            self.assertEqual(code, 1)
            self.assertIn(message, err)
        # Flow style: the rewrite cannot find the value lines; clean error, file intact.
        import yaml

        document = yaml.safe_load(good)
        document["joint_pos_deg"]["left_knee"] = document["joint_pos_deg"]["right_knee"] = 15.0
        flow = yaml.safe_dump(document, default_flow_style=True)
        self.path.write_text(flow)
        code, _, err = run_cli("--yaml", str(self.path), "--write")
        self.assertEqual(code, 1)
        self.assertIn("could not find HOME lines", err)
        self.assertIn("file left unchanged", err)
        self.assertEqual(self.path.read_text(), flow)

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
