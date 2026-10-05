"""Walking runner that binds checkpoints to the HOME they were trained at."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from mjlab_microban.robot.home_pose import HOME
from mjlab_microban.tasks.microban_getup_runner import getup_home_pose

# Checkpoint marker: the full training HOME (joints, root position and root
# quaternion), in the same JSON-safe form as get-up's microban_getup_home_pose.
WALK_HOME_POSE_INFO_KEY = "microban_walk_home_pose"
# Walking checkpoints saved before the stamp existed (the centered-HOME
# checkpoints of 2026-10-03/04, e.g. checkpoints/centered_home_velocity_cont/
# model_20000.pt) carry no marker.  They are accepted only while the current
# HOME is that centered HOME; at any other HOME an unstamped checkpoint is
# refused.
LEGACY_UNSTAMPED_WALK_HOME_TAGS = frozenset({"centered_home"})


def require_walk_home_pose(infos: object) -> None:
    """Refuse a walking checkpoint not trained at the current HOME."""

    stamp = infos.get(WALK_HOME_POSE_INFO_KEY) if isinstance(infos, Mapping) else None
    if stamp is None and HOME.tag in LEGACY_UNSTAMPED_WALK_HOME_TAGS:
        return
    if stamp != getup_home_pose():
        raise ValueError(
            "Walking checkpoint was not trained at the current HOME "
            f"({WALK_HOME_POSE_INFO_KEY} missing or different from "
            f"{HOME.path}); train walking from scratch"
        )


def require_current_home_walk_checkpoint(path: str | Path) -> None:
    """Load a walking checkpoint (CPU) and refuse it unless trained at HOME."""

    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"{path} is not an rsl_rl checkpoint")
    require_walk_home_pose(checkpoint.get("infos"))


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
        require_current_home_walk_checkpoint(path)
        return super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)


if __name__ == "__main__":
    import sys

    for argument in sys.argv[1:]:
        require_current_home_walk_checkpoint(argument)
        print(f"{argument}: trained at the current HOME ({HOME.tag})")
