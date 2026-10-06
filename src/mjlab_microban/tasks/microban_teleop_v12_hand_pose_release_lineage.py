"""Lineage of an active-hand arm pose-release checkpoint.

Every pose-release chain starts fresh from a walker trained at the current HOME
(``scripts/train_microban_teleop_v12.sh start``).  A checkpoint of the recipe
is one of:

* ``fresh_chain``: an ordinary checkpoint of that chain;
* ``fresh_chain_model9900_corner_rescue``: the model_9999 of its pose-release
  corner rescue (microban_teleop_v12_corner_rescue) or a descendant of it;
* ``fresh_chain_model14900_final_rescue`` /
  ``fresh_chain_model9900_corner_rescue_model14900_final_rescue``: the
  model_14999 of its pose-release final rescue
  (microban_teleop_v12_hand_pose_release_final_rescue).
"""

from __future__ import annotations

from collections.abc import Mapping

from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION,
)

HAND_POSE_RELEASE_LINEAGE_FRESH = "fresh_chain"
# A fresh chain whose model_9900 went through the pose-release corner rescue
# (microban_teleop_v12_corner_rescue, pose-release variant): its model_9999 and
# that model's ordinary pose-release descendants.
HAND_POSE_RELEASE_LINEAGE_FRESH_CORNER_RESCUE = "fresh_chain_model9900_corner_rescue"
# Its pose-release final rescue (microban_teleop_v12_hand_pose_release_final_
# rescue): only the rescue's model_14999, from a model_14900 of either lineage
# above.
HAND_POSE_RELEASE_LINEAGE_FRESH_FINAL_RESCUE = "fresh_chain_model14900_final_rescue"
HAND_POSE_RELEASE_LINEAGE_FRESH_CORNER_FINAL_RESCUE = (
    "fresh_chain_model9900_corner_rescue_model14900_final_rescue"
)


def hand_pose_release_lineage(
    infos: Mapping[str, object], *, iteration: int | None = None
) -> str:
    """Classify (and validate) the lineage of a pose-release-recipe checkpoint."""

    if not isinstance(infos, Mapping):
        raise TypeError("Contract-v12 checkpoint infos are malformed")
    if infos.get("microban_teleop_recipe_revision") != (
        MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION
    ):
        raise ValueError("Checkpoint is not the hand pose-release recipe")
    from mjlab_microban.tasks.microban_teleop_v12_corner_rescue import (
        MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY,
        MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_ITERATION,
        is_hand_pose_release_corner_rescue_marker,
        validate_corner_rescue_lineage_marker,
    )
    from mjlab_microban.tasks.microban_teleop_v12_final_rescue import (
        MICROBAN_TELEOP_V12_FINAL_RESCUE_INFO_KEY,
    )

    corner = infos.get(MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY)
    final = infos.get(MICROBAN_TELEOP_V12_FINAL_RESCUE_INFO_KEY)
    validated_corner = None
    if corner is not None:
        # Only the pose-release corner rescue is part of this lineage; its
        # intermediate saves (9901..9998) are not consumable.
        if not is_hand_pose_release_corner_rescue_marker(corner):
            raise ValueError("Hand pose-release checkpoints cannot carry a rescue")
        validated_corner = validate_corner_rescue_lineage_marker(corner)
        if iteration is not None and (
            isinstance(iteration, bool)
            or not isinstance(iteration, int)
            or iteration < MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_ITERATION
        ):
            raise ValueError(
                "Only model_9999 of a pose-release corner rescue (or a "
                "descendant) is consumable"
            )
    if final is not None:
        # Only the pose-release final rescue is part of this lineage, and only
        # its model_14999 is consumable; the marker names the inherited corner.
        from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release_final_rescue import (
            is_hand_pose_release_final_rescue_marker,
            validate_hand_pose_release_final_rescue_infos,
        )

        if not is_hand_pose_release_final_rescue_marker(final):
            raise ValueError("Hand pose-release checkpoints cannot carry a rescue")
        validate_hand_pose_release_final_rescue_infos(
            infos, iteration=iteration, corner_marker=validated_corner
        )
    if final is not None:
        return (
            HAND_POSE_RELEASE_LINEAGE_FRESH_FINAL_RESCUE
            if corner is None
            else HAND_POSE_RELEASE_LINEAGE_FRESH_CORNER_FINAL_RESCUE
        )
    if corner is not None:
        return HAND_POSE_RELEASE_LINEAGE_FRESH_CORNER_RESCUE
    return HAND_POSE_RELEASE_LINEAGE_FRESH
