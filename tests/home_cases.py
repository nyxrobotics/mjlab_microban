"""Which HOME the suite runs at, for the tests whose expectations are HOME-bound.

A HOME branch is this code plus its own config/home_pose.yaml (and
its own policies), and its suite must be green at that HOME.  So:

* tests of the tools (balance, YAML edits, robot YAML) read the fixtures below,
  never the checkout's config/home_pose.yaml;
* tests that pin the literal values of one fixture HOME (the vertical-trunk
  hand-FK box of the centered fixture; the published strings and the
  receiver-capped hand box of the forward-lean HOME) run at that HOME only
  (``centered_home_only`` / ``forward_lean_home_only``) and, through
  tests/test_home_pose_any_trunk.py::HomePinnedTestsTest, are also run in a
  subprocess at the other fixture (MJLAB_MICROBAN_HOME_POSE_YAML), so every
  branch checks both;
* everything else holds at any HOME (expected values derived from HOME).
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures"
CENTERED_HOME_YAML = FIXTURES / "home_pose_centered.yaml"
FORWARD_LEAN_HOME_YAML = FIXTURES / "home_pose_forward_lean.yaml"

FORWARD_LEAN_HOME_TAG = "forward_lean_home"

# The contract strings the forward-lean HOME was published with.
PUBLISHED_CONTRACT_STRINGS = {
    FORWARD_LEAN_HOME_TAG: {
        "v12_home_pose_revision": (
            "forward_lean10_hip_minus14p166561199931_ankle_plus4p127976841869_shoulder_zero_v6"
        ),
        "v12_recipe_revision": (
            "forward_lean_home_velocity_source_staged_mask_reachable_fk_elbow_minus10_"
            "raw_prev_action_servo_range_pi_home_levelled_targets_level_hmd_"
            "receiver_box_hands_v17"
        ),
    },
}


def home_tag() -> str:
    from mjlab_microban.robot.home_pose import HOME

    return HOME.tag


def home_yaml_override() -> str | None:
    """The YAML this process loads instead of config/home_pose.yaml, if any."""

    from mjlab_microban.robot.home_pose import HOME_POSE_YAML_ENV

    return os.environ.get(HOME_POSE_YAML_ENV) or None


def _home_joint_hash() -> str:
    from mjlab_microban.robot.home_pose import HOME

    return HOME.joint_hash


def _fixture_joint_hash(path: Path) -> str:
    from mjlab_microban.robot.home_pose import read_home_pose_yaml, home_joint_hash

    document = read_home_pose_yaml(path)
    return home_joint_hash(document["joint_pos_deg"], document["trunk_pitch_deg"])


AT_CENTERED_HOME = _home_joint_hash() == _fixture_joint_hash(CENTERED_HOME_YAML)
AT_FORWARD_LEAN_HOME = home_tag() == FORWARD_LEAN_HOME_TAG

try:  # pytest marks select the pinned tests for HomePinnedTestsTest; plain unittest works too
    import pytest

    _MARK = {"centered": pytest.mark.centered_home_pinned, "forward_lean": pytest.mark.forward_lean_home_pinned}
except ImportError:  # pragma: no cover - unittest without pytest
    _MARK = {}


def _pinned(kind: str, active: bool, reason: str):
    skip = unittest.skipUnless(active, reason)

    def decorate(test):
        test = skip(test)
        mark = _MARK.get(kind)
        return mark(test) if mark is not None else test

    return decorate


centered_home_only = _pinned(
    "centered",
    AT_CENTERED_HOME,
    "pins the centered fixture HOME's values; run at that HOME by "
    "test_home_pose_any_trunk.py::HomePinnedTestsTest",
)
forward_lean_home_only = _pinned(
    "forward_lean",
    AT_FORWARD_LEAN_HOME,
    "pins the forward-lean HOME's published values; run at the forward-lean HOME by "
    "test_home_pose_any_trunk.py::HomePinnedTestsTest",
)


def _home_trunk_pitch_deg() -> float:
    from mjlab_microban.robot.home_pose import HOME

    return float(HOME.trunk_pitch_deg)


# Mechanisms that exist only with a pitched trunk (a vertical trunk keeps the
# unlevelled terms); the subprocess run checks them at the forward-lean HOME.
pitched_home_only = _pinned(
    "forward_lean",
    _home_trunk_pitch_deg() != 0.0,
    "a vertical HOME trunk keeps the unlevelled terms; run at the forward-lean "
    "HOME by test_home_pose_any_trunk.py::HomePinnedTestsTest",
)
