# Adapted from Teleopit f926386 train_mimic/tasks/tracking/rl/runner.py
# (Apache-2.0, https://github.com/BotRunner64/Teleopit).  Changes: rsl_rl's
# own 0-based learning loop (the last checkpoint is model_<total-1>.pt, like
# every Microban task) instead of Teleopit's 1-based one; the HOME stamp of
# the other Microban runners.  Its export_policy_to_onnx (opset 18,
# normalizers inside, deterministic mean action) is mjlab's own, unchanged.

"""Whole-body runner: checkpoints carry the HOME they were trained at."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from mjlab.rl.runner import MjlabOnPolicyRunner

from mjlab_microban.robot.home_pose import HOME
from mjlab_microban.tasks.microban_getup_runner import getup_home_pose, home_pose_stamps_match

WHOLEBODY_HOME_POSE_INFO_KEY = "microban_wholebody_home_pose"


def require_wholebody_home_pose(infos: object) -> None:
    """Refuse a checkpoint not trained at the current HOME."""
    stamp = infos.get(WHOLEBODY_HOME_POSE_INFO_KEY) if isinstance(infos, Mapping) else None
    if not home_pose_stamps_match(stamp, getup_home_pose()):
        raise ValueError(
            f"Whole-body checkpoint was not trained at the current HOME ({WHOLEBODY_HOME_POSE_INFO_KEY} "
            f"missing or different from {HOME.path}); train from scratch"
        )


def require_wholebody_checkpoint(path: str | Path) -> None:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"{path} is not an rsl_rl checkpoint")
    require_wholebody_home_pose(checkpoint.get("infos"))


class WholebodyOnPolicyRunner(MjlabOnPolicyRunner):
    """mjlab's runner plus the HOME stamp on save and its check on load."""

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu", **_unused) -> None:
        super().__init__(env, train_cfg, log_dir, device)

    def save(self, path: str, infos=None) -> None:
        super().save(path, {**(infos or {}), WHOLEBODY_HOME_POSE_INFO_KEY: getup_home_pose()})

    def load(self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None):
        require_wholebody_checkpoint(path)
        return super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)
