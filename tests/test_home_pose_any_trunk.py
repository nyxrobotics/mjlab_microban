"""Trunk pitch is just a value of config/home_pose.yaml.

* the forward-lean HOME (tests/fixtures/home_pose_forward_lean.yaml) is what
  ``config/balance_home_pose.py --trunk-pitch-deg 10 --write`` writes into a
  copy of the centered fixture, with name/label edited;
* the tests pinned to one fixture HOME (tests/home_cases.py) pass at that HOME
  from any checkout.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


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


class ForwardLeanHomeValuesTest(unittest.TestCase):
    """The forward-lean fixture is the balance tool's solution at +10 deg."""

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
