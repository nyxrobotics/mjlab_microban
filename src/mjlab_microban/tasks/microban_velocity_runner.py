"""Walking runner that binds checkpoints to the HOME they were trained at."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from mjlab_microban.tasks.microban_getup_runner import getup_home_pose

# Checkpoint marker: the full training HOME (joints, root position and root
# quaternion), in the same JSON-safe form as get-up's microban_getup_home_pose.
WALK_HOME_POSE_INFO_KEY = "microban_walk_home_pose"


def require_walk_home_pose(infos: object) -> None:
    """Refuse a walking checkpoint not stamped with the current HOME.

    Walking checkpoints from before the forward-lean HOME (2026-10-04) carry
    no stamp and are refused too: their policy was trained around another
    reference pose and trunk orientation.
    """

    if not isinstance(infos, Mapping) or infos.get(WALK_HOME_POSE_INFO_KEY) != getup_home_pose():
        raise ValueError(
            "Walking checkpoint was not trained at the current HOME "
            f"({WALK_HOME_POSE_INFO_KEY} missing or different); train from scratch"
        )


class MicrobanVelocityOnPolicyRunner(VelocityOnPolicyRunner):
    """mjlab's velocity runner plus the HOME stamp on save and check on load."""

    def save(self, path: str, infos=None) -> None:
        super().save(path, {**(infos or {}), WALK_HOME_POSE_INFO_KEY: getup_home_pose()})

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        require_walk_home_pose(checkpoint.get("infos"))
        return super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)
