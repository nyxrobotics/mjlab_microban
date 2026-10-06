"""Runner for the active-hand arm pose-release recipe.

It trains and records the pose-release recipe revision; a resume accepts only a
checkpoint of that recipe (every chain starts fresh from a walker).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION,
)
from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release import (
    active_hand_arm_released_posture,
)
from mjlab_microban.tasks.microban_teleop_v12_runner import (
    MicrobanTeleopV12OnPolicyRunner,
)


class MicrobanTeleopV12HandPoseReleaseOnPolicyRunner(
    MicrobanTeleopV12OnPolicyRunner
):
    """V12 runner that trains and records the pose-release recipe."""

    accepts_hand_pose_release_recipe = True

    def __init__(self, env, train_cfg: dict, *args: Any, **kwargs: Any) -> None:
        super().__init__(env, train_cfg, *args, **kwargs)
        self._assert_hand_pose_release_environment()

    def _assert_hand_pose_release_environment(self) -> None:
        pose = self.env.unwrapped.reward_manager.get_term_cfg("pose")
        if not isinstance(pose.func, active_hand_arm_released_posture) or (
            pose.params.get("hand_command_name") != "hand_target"
        ):
            raise RuntimeError("Hand pose-release reward term is not installed")

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        resolved = Path(path).expanduser().resolve(strict=True)
        payload = torch.load(resolved, map_location="cpu", weights_only=False)
        infos = payload.get("infos") if isinstance(payload, dict) else None
        if not isinstance(infos, dict):
            raise TypeError("Hand pose-release parent payload is malformed")
        if infos.get("microban_teleop_recipe_revision") != (
            MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION
        ):
            raise ValueError(
                "Hand pose release resumes only a checkpoint of its own recipe"
            )
        loaded = super().load(
            str(resolved), load_cfg=load_cfg, strict=strict, map_location=map_location
        )
        self._assert_hand_pose_release_environment()
        return loaded

    def _contract_infos(self, infos: dict | None = None) -> dict:
        self._assert_hand_pose_release_environment()
        result = super()._contract_infos(infos)
        result["microban_teleop_recipe_revision"] = (
            MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION
        )
        return result
