"""Trunk pitch is just a value of config/home_pose.yaml.

* the forward-lean HOME (tests/fixtures/home_pose_forward_lean.yaml: what
  ``config/balance_home_pose.py --trunk-pitch-deg 10 --write`` writes into a copy
  of the centered fixture, with name/label edited) gives the values and strings
  its policies were trained and published with;
* any other HOME gets "<label>_<hash>" strings of its trunk's mechanism;
* the tests pinned to one fixture HOME (tests/home_cases.py) pass at that HOME
  from any checkout.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures"
LEAN_YAML = FIXTURES / "home_pose_forward_lean.yaml"
CENTERED_YAML = FIXTURES / "home_pose_centered.yaml"
sys.path.insert(0, str(REPO_ROOT / "tests"))


def _environment(yaml_path: Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONHASHSEED": "0",
            "CUDA_VISIBLE_DEVICES": "",
            "MJLAB_MICROBAN_HOME_POSE_YAML": str(yaml_path),
            "PYTHONPATH": os.pathsep.join(
                [str(REPO_ROOT / "src"), *filter(None, [os.environ.get("PYTHONPATH")])]
            ),
        }
    )
    return environment


def _run_python(yaml_path: Path, code: str) -> str:
    completed = subprocess.run(
        [sys.executable, "-c", code],
        env=_environment(yaml_path),
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stderr[-3000:])
    return completed.stdout.strip().splitlines()[-1]


class ForwardLeanHomeValuesTest(unittest.TestCase):
    """The forward-lean YAML gives the values its policies were published with."""

    def test_home_values_and_contract_strings(self):
        result = json.loads(
            _run_python(
                LEAN_YAML,
                "import json\n"
                "from mjlab_microban.robot.home_pose import HOME\n"
                "from mjlab_microban.robot import home_contracts as c\n"
                "print(json.dumps({'tag': HOME.tag, 'hash': HOME.joint_hash,"
                " 'rad': dict(HOME.joint_pos_rad), 'deg_in': dict(HOME.input_joint_pos_deg),"
                " 'root': HOME.root_pos, 'quat': HOME.root_quat_wxyz, 'g': HOME.projected_gravity,"
                " 'head': HOME.head_standing_height_m, 'feet': HOME.feet_lateral_m,"
                " 'robot': c.contract_strings()}))",
            )
        )
        self.assertEqual(result["tag"], "forward_lean_home")
        self.assertEqual(result["hash"], "481503d292")
        # The YAML holds the canonical 12-decimal values; HOME publishes the
        # solver's unrounded ones bit for bit.
        self.assertEqual(result["deg_in"]["left_hip_pitch"], -14.166561199931)
        self.assertEqual(result["rad"]["left_hip_pitch"], float(np.deg2rad(-14.166561199931119)))
        self.assertEqual(result["rad"]["right_ankle_pitch"], float(np.deg2rad(4.127976841869204)))
        self.assertEqual(result["root"], [0.0, 0.0, 0.170430569776402])
        pitch = float(np.deg2rad(10.0))
        self.assertEqual(
            result["quat"],
            [float(np.cos(pitch / 2.0)), 0.0, float(np.sin(pitch / 2.0)), 0.0],
        )
        self.assertEqual(result["g"], [math.sin(pitch), 0.0, -math.cos(pitch)])
        self.assertEqual((result["head"], result["feet"]), (0.2953, 0.0941))
        self.assertEqual(
            result["robot"],
            {
                "getup_contract_version": "v6",
                "v12_home_pose_revision": (
                    "forward_lean10_hip_minus14p166561199931_ankle_plus4p127976841869_"
                    "shoulder_zero_v6"
                ),
                "v12_recipe_revision": (
                    "forward_lean_home_velocity_source_staged_mask_reachable_fk_elbow_minus10_"
                    "raw_prev_action_servo_range_pi_home_levelled_targets_level_hmd_"
                    "receiver_box_hands_v17"
                ),
                "v12_hand_pose_release_recipe_revision": (
                    "forward_lean_home_velocity_source_staged_mask_reachable_fk_elbow_minus10_"
                    "raw_prev_action_servo_range_pi_home_levelled_targets_level_hmd_"
                    "receiver_box_hands_active_hand_arm_pose_release_one_run_warmup1000_"
                    "total9000_v1"
                ),
                "v12_target_frame": "robot_home_levelled_trunk_xyz_forward_left_up",
            },
        )

    def test_fixture_is_the_balance_tool_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "home_pose.yaml"
            shutil.copyfile(CENTERED_YAML, path)
            completed = subprocess.run(
                [sys.executable, str(REPO_ROOT / "config" / "balance_home_pose.py"), "--yaml",
                 str(path), "--trunk-pitch-deg", "10", "--write", "--no-training-check"],
                cwd=REPO_ROOT, capture_output=True, text=True, timeout=600, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            written = path.read_text().splitlines()
        fixture = LEAN_YAML.read_text().splitlines()
        # Same file except the fixture's header comment and name/label.
        values = lambda lines: [  # noqa: E731
            line for line in lines
            if line and not line.startswith("#") and not line.startswith(("name:", "label:"))
        ]
        self.assertEqual(values(written), values(fixture))


class DerivedHomeStringsTest(unittest.TestCase):
    """Any other HOME gets "<label>_<hash>" strings of its trunk's mechanism."""

    def test_pitched_and_vertical_derived_strings(self):
        with tempfile.TemporaryDirectory() as directory:
            for pitch, knee in ((5.0, 0.0), (0.0, 15.0)):
                path = Path(directory) / f"home_{pitch}_{knee}.yaml"
                shutil.copyfile(CENTERED_YAML, path)
                from mjlab_microban.robot.home_pose import rewrite_home_pose_yaml

                rewrite_home_pose_yaml(path, joint_pos_deg={"left_knee": knee, "right_knee": knee})
                completed = subprocess.run(
                    [sys.executable, str(REPO_ROOT / "config" / "balance_home_pose.py"),
                     "--yaml", str(path), "--trunk-pitch-deg", str(pitch), "--write",
                     "--no-training-check"],
                    cwd=REPO_ROOT, capture_output=True, text=True, timeout=600, check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                result = json.loads(
                    _run_python(
                        path,
                        "import json\n"
                        "from mjlab_microban.robot.home_pose import HOME\n"
                        "from mjlab_microban.robot import home_contracts as c\n"
                        "print(json.dumps({'tag': HOME.tag,"
                        " 'getup': c.GETUP_CONTRACT_VERSION,"
                        " 'recipe': c.V12_RECIPE_REVISION}))",
                    )
                )
                tag = result["tag"]
                self.assertRegex(tag, r"^centered_home_[0-9a-f]{10}$")
                self.assertEqual(result["getup"], f"{'v5' if pitch == 0.0 else 'v6'}_{tag}")
                self.assertTrue(result["recipe"].startswith(f"{tag}_velocity_source_"))
                self.assertEqual(
                    result["recipe"].endswith("_v11"), pitch == 0.0, result["recipe"]
                )


class HomePinnedTestsTest(unittest.TestCase):
    """The tests pinned to one fixture HOME pass at that HOME, from any checkout.

    A HOME branch's own config/home_pose.yaml runs its pinned tests directly
    (tests/home_cases.py); the other fixture's pinned tests are run here in a
    subprocess at its YAML, so a vertical-trunk checkout still checks the
    pitched-trunk mechanisms and the reverse.  Any other HOME runs both.
    """

    def _run_marked(self, yaml_path: Path, marker: str) -> None:
        try:
            import pytest  # noqa: F401
        except ImportError:
            self.skipTest("pytest is not installed")
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rs", "-m", marker,
             "tests"],
            env=_environment(yaml_path), cwd=REPO_ROOT, capture_output=True, text=True,
            timeout=3000, check=False,
        )
        summary = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
        self.assertEqual(completed.returncode, 0, completed.stdout[-4000:] + completed.stderr[-2000:])
        self.assertIn(" passed", summary)
        self.assertNotIn("skipped", summary, completed.stdout[-4000:])

    def test_forward_lean_pinned_tests_pass_at_the_forward_lean_home(self):
        import home_cases

        if home_cases.AT_FORWARD_LEAN_HOME:
            self.skipTest("this checkout is the forward-lean HOME: its pinned tests ran directly")
        self._run_marked(LEAN_YAML, "forward_lean_home_pinned")

    def test_centered_pinned_tests_pass_at_the_centered_home(self):
        import home_cases

        if home_cases.AT_CENTERED_HOME:
            self.skipTest("this checkout is the centered HOME: its pinned tests ran directly")
        self._run_marked(CENTERED_YAML, "centered_home_pinned")


if __name__ == "__main__":
    unittest.main()
