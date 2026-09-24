"""Microban-specific observation helpers for motion tracking.

MjLab's generic tracking command carries the complete reference articulation.  The
three camera-neck joints are intentionally controlled by the PICO HMD outside the
body policy, so the actor and critic must only receive the 18 body-joint portion of
that command.  Keeping this projection here also makes the deployment observation
contract explicit and deterministic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.tracking.mdp import MotionCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def controlled_motion_command(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return reference position/velocity for policy-controlled joints only.

    ``MotionCommand`` still loads and resets the complete 21-joint Microban state;
    the NPZ therefore remains compatible with MjLab's standard motion format.  This
    observation removes ``head``, ``neck_roll`` and ``neck_pitch`` from the policy
    input in exactly the same resolved joint order used by the action term.
    """

    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    joint_ids = asset_cfg.joint_ids
    return torch.cat(
        (command.joint_pos[:, joint_ids], command.joint_vel[:, joint_ids]), dim=-1
    )
