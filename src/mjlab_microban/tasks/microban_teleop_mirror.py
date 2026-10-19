"""Left-right mirror of the PICO observations and actions, for the mirror loss.

The robot, the command and arm/foot target distributions, the rewards and
the evaluation are left-right symmetric, but the trained policy was not: a
mirrored policy (M pi(M o)) moved the bad single-foot side and the weak turn
to the other side.  rsl_rl's ``symmetry_cfg`` with ``use_mirror_loss`` adds
``coeff * MSE(pi(M o), M pi(o))`` to the PPO loss so one run learns one
symmetric solution (microban_teleop_v12_actor.LegacyAdapterPPO keeps the
loss to the twelve leg outputs).

The mirror (microban_velocity_mirror) is the walker's, with the arm and
foot targets added: they swap sides, and the foot targets (offsets in the
trunk frame) flip y.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from mjlab_microban.policy_contract import PICO_ARM_JOINT_NAMES
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_HMD_JOINT_NAMES,
    MICROBAN_TELEOP_ACTION_JOINT_NAMES,
)
from mjlab_microban.tasks.microban_teleop_v12_runner import (
    TELEOP_V12_OBSERVATION_TERM_LAYOUTS,
)
from mjlab_microban.tasks.microban_velocity_mirror import (
    TERM_MIRRORS,
    apply_mirror,
    augment,
    group_mirror,
    joint_mirror,
    mirror_groups,
    swap_feet,
)

_BODY_JOINTS = (*MICROBAN_HMD_JOINT_NAMES, *MICROBAN_TELEOP_ACTION_JOINT_NAMES)
_TERM_MIRRORS = {
    **TERM_MIRRORS,
    "joint_pos": joint_mirror(_BODY_JOINTS),
    "joint_vel": joint_mirror(_BODY_JOINTS),
    "actions": joint_mirror(MICROBAN_TELEOP_ACTION_JOINT_NAMES),
    "foot_target": swap_feet(3, (1.0, -1.0, 1.0)),  # (left xyz, right xyz)
    "arm_target": joint_mirror(PICO_ARM_JOINT_NAMES),
}

MIRROR_OBSERVATION = {
    group: group_mirror(layout, _TERM_MIRRORS)
    for group, layout in TELEOP_V12_OBSERVATION_TERM_LAYOUTS.items()
}
MIRROR_ACTION = group_mirror((("actions", len(MICROBAN_TELEOP_ACTION_JOINT_NAMES)),), _TERM_MIRRORS)


def mirror_observations(obs: TensorDict) -> TensorDict:
    return mirror_groups(MIRROR_OBSERVATION, obs)


def mirror_actions(actions: torch.Tensor) -> torch.Tensor:
    return apply_mirror(MIRROR_ACTION, actions)


def mirror_augmentation(
    env=None,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """rsl_rl's ``data_augmentation_func``: each input followed by its mirror."""

    del env
    return augment(mirror_observations, mirror_actions, obs, actions)
