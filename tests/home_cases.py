"""Which HOME the suite runs at, for the tests whose expectations are HOME-bound.

A HOME branch is the home-config code plus its own config/home_pose.yaml (and
its own policies), and its suite must be green at that HOME.  So:

* tests of the tools (balance, YAML edits, robot YAML) read the fixtures below,
  never the checkout's config/home_pose.yaml;
* tests that pin one published HOME's literal values (contract strings, the
  hand-FK box, the pose-release switch parent, recorded artifacts) run at that
  HOME only (``centered_home_only`` / ``forward_lean_home_only``) and, through
  tests/test_home_pose_any_trunk.py::HomePinnedTestsTest, are also run in a
  subprocess at the other published HOME (MJLAB_MICROBAN_HOME_POSE_YAML), so
  every branch checks both;
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

CENTERED_HOME_TAG = "centered_home"
FORWARD_LEAN_HOME_TAG = "forward_lean_home"

# The robot contract strings each published HOME's branches carry.
PUBLISHED_CONTRACT_STRINGS = {
    CENTERED_HOME_TAG: {
        "walk_contract_version": "v3_centered_home_servo_range",
        "getup_contract_version": "v5",
        "getup_checkpoint_stamp": "",
        "v12_home_pose_revision": (
            "centered_home_hip_plus1p198384259489_ankle_minus1p198384259489_shoulder_zero_v5"
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
        "v12_target_frame": "robot_trunk_xyz_forward_left_up",
    },
    FORWARD_LEAN_HOME_TAG: {
        "walk_contract_version": "v4_forward_lean_home_servo_range",
        "getup_contract_version": "v6",
        "getup_checkpoint_stamp": "v6",
        "v12_home_pose_revision": (
            "forward_lean10_hip_minus14p166561199931_ankle_plus4p127976841869_shoulder_zero_v6"
        ),
        "v12_recipe_revision": (
            "forward_lean_home_velocity_source_staged_mask_reachable_fk_elbow_minus10_"
            "raw_prev_action_servo_range_pi_home_levelled_targets_level_hmd_"
            "receiver_box_hands_v17"
        ),
        "v12_hand_pose_release_recipe_revision": (
            "forward_lean_home_velocity_source_staged_mask_reachable_fk_elbow_minus10_"
            "raw_prev_action_servo_range_pi_home_levelled_targets_level_hmd_"
            "receiver_box_hands_active_hand_arm_pose_release_v18"
        ),
        "v12_packager_revision": (
            "microban_teleop_v12_final_deployment_packager_v7_forward_lean_home_servo_range"
        ),
        "v12_target_frame": "robot_home_levelled_trunk_xyz_forward_left_up",
    },
}


def home_tag() -> str:
    from mjlab_microban.robot.home_pose import HOME

    return HOME.tag


def home_yaml_override() -> str | None:
    """The YAML this process loads instead of config/home_pose.yaml, if any."""

    from mjlab_microban.robot.home_pose import HOME_POSE_YAML_ENV

    return os.environ.get(HOME_POSE_YAML_ENV) or None


AT_CENTERED_HOME = home_tag() == CENTERED_HOME_TAG
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
    "pins the centered HOME's published values; run at the centered HOME by "
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
# original terms); the subprocess run checks them at the forward-lean HOME.
pitched_home_only = _pinned(
    "forward_lean",
    _home_trunk_pitch_deg() != 0.0,
    "a vertical HOME trunk keeps the original (unlevelled) terms; run at the forward-lean "
    "HOME by test_home_pose_any_trunk.py::HomePinnedTestsTest",
)
