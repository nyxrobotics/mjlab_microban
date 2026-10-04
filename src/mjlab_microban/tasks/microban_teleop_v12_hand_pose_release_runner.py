"""Runner for the opt-in active-hand arm pose-release recipe.

A fresh start of this task records the new recipe revision.  Resuming a v11
checkpoint is a recipe switch: it is refused unless
``experimental_recipe_switch`` is set, and every save of such a run carries a
``release_eligible: False`` marker.  Canonical stage gates and the exporter do
not accept this recipe at all.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from mjlab_microban.tasks.microban_teleop_v12_bootstrap import sha256_file
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION,
    MICROBAN_TELEOP_V12_RECIPE_REVISION,
)
from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release import (
    MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_SWITCH_INFO_KEY,
    active_hand_arm_released_posture,
)
from mjlab_microban.tasks.microban_teleop_v12_runner import (
    MicrobanTeleopV12OnPolicyRunner,
)

HAND_POSE_RELEASE_SWITCH_SCHEMA_VERSION = 1


def validate_hand_pose_release_switch_marker(value: Any) -> dict[str, Any]:
    """Validate the not-for-release marker of a v11 -> pose-release switch."""

    if not isinstance(value, dict):
        raise TypeError("Hand pose-release switch marker is malformed")
    expected_keys = {
        "schema_version",
        "release_eligible",
        "parent_recipe_revision",
        "recipe_revision",
        "parent_checkpoint_sha256",
        "parent_iteration",
    }
    if set(value) != expected_keys:
        raise ValueError("Hand pose-release switch marker keys drifted")
    sha = value["parent_checkpoint_sha256"]
    if (
        value["schema_version"] != HAND_POSE_RELEASE_SWITCH_SCHEMA_VERSION
        or value["release_eligible"] is not False
        or value["parent_recipe_revision"] != MICROBAN_TELEOP_V12_RECIPE_REVISION
        or value["recipe_revision"]
        != MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION
        or not isinstance(sha, str)
        or len(sha) != 64
        or any(c not in "0123456789abcdef" for c in sha)
        or not isinstance(value["parent_iteration"], int)
        or isinstance(value["parent_iteration"], bool)
    ):
        raise ValueError("Hand pose-release switch marker drifted")
    return deepcopy(value)


class MicrobanTeleopV12HandPoseReleaseOnPolicyRunner(
    MicrobanTeleopV12OnPolicyRunner
):
    """V12 runner that trains and records the pose-release recipe."""

    accepts_hand_pose_release_recipe = True

    def __init__(self, env, train_cfg: dict, *args: Any, **kwargs: Any) -> None:
        cfg = deepcopy(train_cfg)
        switch = cfg.pop("experimental_recipe_switch", False)
        if type(switch) is not bool:
            raise TypeError("experimental_recipe_switch must be a boolean")
        if switch and not cfg.get("resume", False):
            raise ValueError("experimental_recipe_switch applies only to a resume")
        if cfg.get("deadline_fallback_resume") or cfg.get("simulation_preview_mode"):
            raise ValueError("Hand pose release has no deadline or preview route")
        self.experimental_recipe_switch = switch
        self.teleop_v12_hand_pose_release_switch: dict[str, Any] | None = None
        super().__init__(env, cfg, *args, **kwargs)
        self._assert_hand_pose_release_environment()

    def _assert_hand_pose_release_environment(self) -> None:
        pose = self.env.unwrapped.reward_manager.get_term_cfg("pose")
        if not isinstance(pose.func, active_hand_arm_released_posture) or (
            pose.params.get("hand_command_name") != "hand_target"
        ):
            raise RuntimeError("Hand pose-release reward term is not installed")

    def load(
        self,
        path: str | bytes,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        if isinstance(path, bytes):
            raise TypeError("Hand pose-release training loads a filesystem checkpoint")
        resolved = Path(path).expanduser().resolve(strict=True)
        before = sha256_file(resolved)
        payload = torch.load(resolved, map_location="cpu", weights_only=False)
        infos = payload.get("infos") if isinstance(payload, dict) else None
        if not isinstance(infos, dict):
            raise TypeError("Hand pose-release parent payload is malformed")
        recipe = infos.get("microban_teleop_recipe_revision")
        existing = infos.get(MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_SWITCH_INFO_KEY)
        if recipe == MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION:
            marker = (
                None
                if existing is None
                else validate_hand_pose_release_switch_marker(existing)
            )
        elif recipe == MICROBAN_TELEOP_V12_RECIPE_REVISION:
            if not self.experimental_recipe_switch:
                raise ValueError(
                    "Resuming a v11 checkpoint into the hand pose-release recipe "
                    "mixes recipes; pass --agent.experimental-recipe-switch True "
                    "for an experiment, or start a fresh chain"
                )
            if existing is not None:
                raise ValueError("A v11 checkpoint cannot carry a switch marker")
            marker = {
                "schema_version": HAND_POSE_RELEASE_SWITCH_SCHEMA_VERSION,
                "release_eligible": False,
                "parent_recipe_revision": MICROBAN_TELEOP_V12_RECIPE_REVISION,
                "recipe_revision": (
                    MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION
                ),
                "parent_checkpoint_sha256": before,
                "parent_iteration": payload.get("iter"),
            }
            validate_hand_pose_release_switch_marker(marker)
        else:
            raise ValueError("Hand pose release resumes only v11 or its own recipe")
        loaded = super().load(
            str(resolved), load_cfg=load_cfg, strict=strict, map_location=map_location
        )
        if sha256_file(resolved) != before:
            raise ValueError("Hand pose-release parent changed while loading")
        self.teleop_v12_hand_pose_release_switch = marker
        self._assert_hand_pose_release_environment()
        return loaded

    def _contract_infos(self, infos: dict | None = None) -> dict:
        self._assert_hand_pose_release_environment()
        result = super()._contract_infos(infos)
        result["microban_teleop_recipe_revision"] = (
            MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION
        )
        if self.teleop_v12_hand_pose_release_switch is not None:
            result[MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_SWITCH_INFO_KEY] = deepcopy(
                self.teleop_v12_hand_pose_release_switch
            )
        return result
