"""Bind contract-v12 checkpoints to the HOME pose used by their training task."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from mjlab_microban.robot.microban_constants import HOME_FRAME
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_HMD_JOINT_NAMES,
    MICROBAN_TELEOP_ACTION_JOINT_NAMES,
)
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION,
    MICROBAN_TELEOP_V12_HOME_POSE_REVISION,
    MICROBAN_TELEOP_V12_RECIPE_REVISION,
)

TELEOP_V12_HOME_POSE_INFO_KEY = "microban_teleop_v12_home_pose"
TELEOP_V12_HOME_POSE_SCHEMA_VERSION = 2
TELEOP_V12_HOME_JOINT_NAMES = (
    *MICROBAN_HMD_JOINT_NAMES,
    *MICROBAN_TELEOP_ACTION_JOINT_NAMES,
)


def teleop_v12_home_pose_marker() -> dict[str, Any]:
    """Return the historical task-local HOME, with both shoulders at zero."""

    shared_defaults = HOME_FRAME.joint_pos
    if not isinstance(shared_defaults, Mapping):
        raise TypeError("Contract-v12 HOME_FRAME joint pose is malformed")
    defaults = dict(shared_defaults)
    defaults["left_shoulder_pitch"] = 0.0
    defaults["right_shoulder_pitch"] = 0.0
    if len(set(TELEOP_V12_HOME_JOINT_NAMES)) != 21 or set(defaults) != set(
        TELEOP_V12_HOME_JOINT_NAMES
    ):
        raise ValueError("Contract-v12 HOME_FRAME joint names drifted")
    positions = [float(defaults[name]) for name in TELEOP_V12_HOME_JOINT_NAMES]
    if not all(math.isfinite(value) for value in positions):
        raise ValueError("Contract-v12 HOME_FRAME contains a nonfinite joint default")
    root_pos = [float(value) for value in HOME_FRAME.pos]
    root_quat = [float(value) for value in HOME_FRAME.rot]
    if len(root_pos) != 3 or len(root_quat) != 4 or not all(
        math.isfinite(value) for value in (*root_pos, *root_quat)
    ):
        raise ValueError("Contract-v12 HOME_FRAME root pose is malformed")
    if not math.isclose(
        sum(value * value for value in root_quat), 1.0, abs_tol=1e-6
    ):
        raise ValueError("Contract-v12 HOME_FRAME root quaternion is not normalized")
    return {
        "schema_version": TELEOP_V12_HOME_POSE_SCHEMA_VERSION,
        "revision": MICROBAN_TELEOP_V12_HOME_POSE_REVISION,
        "root_pos_xyz_m": root_pos,
        "root_quat_wxyz": root_quat,
        "joint_names": list(TELEOP_V12_HOME_JOINT_NAMES),
        "joint_pos_rad": positions,
    }


def validate_teleop_v12_home_pose(
    infos: Mapping[str, Any], *, allow_hand_pose_release_recipe: bool = False
) -> dict[str, Any]:
    """Reject old or relabeled checkpoints before loading actor or optimizer state.

    ``allow_hand_pose_release_recipe`` additionally accepts the opt-in
    active-hand arm pose-release recipe.  Only its own runner and explicitly
    flagged evaluators pass it; stage gates and the exporter never do.
    """

    if not isinstance(infos, Mapping):
        raise TypeError("Contract-v12 checkpoint infos are malformed")
    # The corner-rescue recipe is the current recipe with only the hand
    # sampler changed; its marker must name the current recipe as its source.
    from mjlab_microban.tasks.microban_teleop_v12_corner_rescue import (
        MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY,
        MICROBAN_TELEOP_V12_CORNER_RESCUE_RECIPE_REVISION,
    )

    recipe = infos.get("microban_teleop_recipe_revision")
    rescue = infos.get(MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY)
    if allow_hand_pose_release_recipe and recipe == (
        MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION
    ):
        if rescue is not None:
            raise ValueError("Hand pose-release checkpoints cannot carry a rescue")
    elif recipe != MICROBAN_TELEOP_V12_RECIPE_REVISION and not (
        recipe == MICROBAN_TELEOP_V12_CORNER_RESCUE_RECIPE_REVISION
        and isinstance(rescue, Mapping)
        and rescue.get("source_recipe_revision") == MICROBAN_TELEOP_V12_RECIPE_REVISION
    ):
        raise ValueError("Checkpoint recipe does not match the current HOME pose")
    expected = teleop_v12_home_pose_marker()
    if infos.get(TELEOP_V12_HOME_POSE_INFO_KEY) != expected:
        raise ValueError("Checkpoint HOME pose does not match the training task")
    return expected
