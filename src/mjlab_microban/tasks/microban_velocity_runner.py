"""Walking runner that binds checkpoints to the HOME they were trained at."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from mjlab_microban.robot import home_contracts
from mjlab_microban.robot.home_pose import HOME
from mjlab_microban.tasks.microban_getup_runner import getup_home_pose, home_pose_stamps_match

# Checkpoint marker: the full training HOME (joints, root position and root
# quaternion), in the same JSON-safe form as get-up's microban_getup_home_pose.
WALK_HOME_POSE_INFO_KEY = "microban_walk_home_pose"
# Walking checkpoints saved before the stamp existed (the centered-HOME
# checkpoints of 2026-10-03/04, e.g. checkpoints/centered_home_velocity_cont/
# model_20000.pt) carry no marker.  They are accepted only while the current
# HOME is that centered HOME (home_pose.LEGACY_HOME_OVERRIDES
# "accepts_unstamped_walk_checkpoints"); at any other HOME, the forward-lean one
# included (its walking runs were stamped from the start), an unstamped
# checkpoint is refused.
ACCEPTS_UNSTAMPED_WALK_CHECKPOINTS = home_contracts.ACCEPTS_UNSTAMPED_WALK_CHECKPOINTS


def require_walk_home_pose(infos: object) -> None:
    """Refuse a walking checkpoint not trained at the current HOME."""

    stamp = infos.get(WALK_HOME_POSE_INFO_KEY) if isinstance(infos, Mapping) else None
    if stamp is None and ACCEPTS_UNSTAMPED_WALK_CHECKPOINTS:
        return
    if not home_pose_stamps_match(stamp, getup_home_pose()):
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

    # The command-line check is the walking exporter's full one: the run's
    # recorded params/env.yaml (HOME joints/root, +-pi clip, raw previous
    # action) as well as the checkpoint's HOME stamp.
    from mjlab_microban.scripts.export_walk_onnx import (
        require_current_home_walk_checkpoint as require_walking_run,
    )

    for argument in sys.argv[1:]:
        require_walking_run(Path(argument))
        print(f"{argument}: trained at the current HOME ({HOME.tag}) under the walking contract")
