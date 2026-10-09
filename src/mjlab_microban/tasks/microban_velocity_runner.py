"""Walking runner that binds checkpoints to the HOME they were trained at."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from mjlab_microban.robot.home_pose import HOME
from mjlab_microban.tasks.curriculum import (
    CURRICULUM_STATE_INFO_KEY,
    bind_update_clock,
    curriculum_state,
    resume_run,
)
from mjlab_microban.tasks.microban_getup_runner import getup_home_pose, home_pose_stamps_match

# Checkpoint marker: the full training HOME (joints, root position and root
# quaternion), in the same JSON-safe form as get-up's microban_getup_home_pose.
WALK_HOME_POSE_INFO_KEY = "microban_walk_home_pose"


def require_walk_home_pose(infos: object) -> None:
    """Refuse a walking checkpoint not trained at the current HOME."""

    stamp = infos.get(WALK_HOME_POSE_INFO_KEY) if isinstance(infos, Mapping) else None
    if not home_pose_stamps_match(stamp, getup_home_pose()):
        raise ValueError(
            "Walking checkpoint was not trained at the current HOME "
            f"({WALK_HOME_POSE_INFO_KEY} missing or different from "
            f"{HOME.path}); train walking from scratch"
        )


def require_walk_checkpoint_home_stamp(path: str | Path) -> None:
    """Load a walking checkpoint (CPU) and refuse it unless stamped with HOME.

    The walking exporter's require_current_home_walk_checkpoint also checks the
    run's recorded params/env.yaml.
    """

    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"{path} is not an rsl_rl checkpoint")
    require_walk_home_pose(checkpoint.get("infos"))


class MicrobanVelocityOnPolicyRunner(VelocityOnPolicyRunner):
    """mjlab's velocity runner plus the update clock, the HOME stamp on save and its check on load."""

    def __init__(self, env, train_cfg: dict, *args, **kwargs) -> None:
        bind_update_clock(env.unwrapped, int(train_cfg["num_steps_per_env"]))
        super().__init__(env, train_cfg, *args, **kwargs)

    def save(self, path: str, infos=None) -> None:
        super().save(
            path,
            {
                **(infos or {}),
                WALK_HOME_POSE_INFO_KEY: getup_home_pose(),
                CURRICULUM_STATE_INFO_KEY: curriculum_state(self.env.unwrapped),
            },
        )

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        require_walk_checkpoint_home_stamp(path)
        infos = super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)
        if load_cfg is None:
            resume_run(self, infos)
        return infos
