"""Runner for the v13 arm-overlay recipe.

It trains and records the arm-overlay recipe revision (every run starts
from a walker and may resume from its own checkpoints).
"""

from __future__ import annotations

from typing import Any

from mjlab_microban.tasks.microban_ik_tracking_mdp import (
    IkTrackingArmOverlayJointPositionAction,
    IkTrackingArmTargetMotion,
)
from mjlab_microban.tasks.microban_ik_tracking_v12_env_cfg import (
    MICROBAN_IK_TRACKING_V13_ARM_OVERLAY_RECIPE_REVISION,
)
from mjlab_microban.tasks.microban_ik_tracking_v12_runner import (
    MicrobanIkTrackingV12OnPolicyRunner,
)
from mjlab_microban.tasks.microban_ik_tracking_v13_arm_overlay import (
    foot_target_leg_released_posture,
)


class MicrobanIkTrackingV13ArmOverlayOnPolicyRunner(MicrobanIkTrackingV12OnPolicyRunner):
    """V12 runner that trains and records the arm-overlay recipe."""

    def __init__(self, env, train_cfg: dict, *args: Any, **kwargs: Any) -> None:
        super().__init__(env, train_cfg, *args, **kwargs)
        self._assert_arm_overlay_environment()

    def _assert_arm_overlay_environment(self) -> None:
        raw = self.env.unwrapped
        pose = raw.reward_manager.get_term_cfg("pose")
        if not isinstance(pose.func, foot_target_leg_released_posture):
            raise RuntimeError("Leg pose-release reward term is not installed")
        if not isinstance(
            raw.action_manager.get_term("joint_pos"), IkTrackingArmOverlayJointPositionAction
        ):
            raise RuntimeError("The arm overlay action is not installed")
        if not isinstance(
            raw.event_manager.get_term_cfg("ik_tracking_arm_target_motion").func,
            IkTrackingArmTargetMotion,
        ):
            raise RuntimeError("The arm target motion is not installed")

    def _contract_infos(self, infos: dict | None = None) -> dict:
        self._assert_arm_overlay_environment()
        result = super()._contract_infos(infos)
        result["microban_teleop_recipe_revision"] = (
            MICROBAN_IK_TRACKING_V13_ARM_OVERLAY_RECIPE_REVISION
        )
        return result
