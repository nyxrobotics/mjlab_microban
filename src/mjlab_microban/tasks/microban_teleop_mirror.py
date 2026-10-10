"""Left-right mirror of the PICO observations and actions, for the mirror loss.

The robot, the command and arm/foot target distributions, the rewards and
the evaluation are left-right symmetric, but the trained policy was not: a
mirrored policy (M pi(M o)) moved the bad single-foot side and the weak turn
to the other side.  rsl_rl's ``symmetry_cfg`` with ``use_mirror_loss`` adds
``coeff * MSE(pi(M o), M pi(o))`` to the PPO loss so one run learns one
symmetric solution (microban_teleop_v12_actor.LegacyAdapterPPO keeps the
loss to the twelve leg outputs).

The mirror is the reflection across the robot's sagittal (x-z) plane: left
and right joints, feet and arms swap, a joint's angle keeps its sign when
the mirrored axis is the partner's axis (pitch-like joints, knee, elbow)
and flips it otherwise (roll and yaw), polar vectors in the trunk frame flip
y, axial vectors (angular velocity) flip x and z, and the twist flips its
lateral velocity and its yaw rate.  The IMU frame is the trunk frame turned
(imu x = trunk y, imu y = -trunk z, imu z = -trunk x), so in the IMU frame
the angular velocity flips y and z and the linear velocity flips x.  The
foot contact forces are in the world frame: mirroring the whole scene
across the world x-z plane (the floor is flat) flips their y.  The IMU site
sits 2 mm right of the mid-plane, so the critic's IMU linear velocity is
mirrored up to omega x 4 mm; the actor does not observe it.
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

# Sign of a joint angle under the mirror, by joint (the partner's angle is
# this sign times the joint's).  tests/test_teleop_mirror.py derives it from
# the MJCF joint axes.
MIRROR_JOINT_SIGNS = {
    "shoulder_pitch": 1.0,
    "shoulder_roll": -1.0,
    "elbow": 1.0,
    "hip_yaw": -1.0,
    "hip_roll": -1.0,
    "hip_pitch": 1.0,
    "knee": 1.0,
    "ankle_pitch": 1.0,
    "ankle_roll": -1.0,
    "head": -1.0,
    "neck_roll": -1.0,
    "neck_pitch": 1.0,
}


def mirror_partner(name: str) -> str:
    """The joint's left/right partner (itself for a mid-plane joint)."""

    for side, other in (("left_", "right_"), ("right_", "left_")):
        if name.startswith(side):
            return other + name[len(side):]
    return name


def mirror_joint_sign(name: str) -> float:
    return MIRROR_JOINT_SIGNS[name.removeprefix("left_").removeprefix("right_")]


def _joints(names: tuple[str, ...]) -> tuple[list[int], list[float]]:
    return (
        [names.index(mirror_partner(name)) for name in names],
        [mirror_joint_sign(name) for name in names],
    )


def _swap_feet(per_foot: int, signs: tuple[float, ...]) -> tuple[list[int], list[float]]:
    """Two feet of ``per_foot`` values each, swapped, each value times ``signs``."""

    index = [*range(per_foot, 2 * per_foot), *range(per_foot)]
    return index, [*signs, *signs]


_BODY_JOINTS = (*MICROBAN_HMD_JOINT_NAMES, *MICROBAN_TELEOP_ACTION_JOINT_NAMES)
# Per term: the source column (within the term) and the sign of each column.
_TERM_MIRRORS: dict[str, tuple[list[int], list[float]]] = {
    "base_lin_vel": ([0, 1, 2], [-1.0, 1.0, 1.0]),  # IMU frame, polar
    "base_ang_vel": ([0, 1, 2], [1.0, -1.0, -1.0]),  # IMU frame, axial
    "projected_gravity": ([0, 1, 2], [1.0, -1.0, 1.0]),  # trunk frame
    "joint_pos": _joints(_BODY_JOINTS),
    "joint_vel": _joints(_BODY_JOINTS),
    "actions": _joints(MICROBAN_TELEOP_ACTION_JOINT_NAMES),
    "command": ([0, 1, 2], [1.0, -1.0, -1.0]),  # vx, vy, yaw rate
    "foot_height": _swap_feet(1, (1.0,)),
    "foot_air_time": _swap_feet(1, (1.0,)),
    "foot_contact": _swap_feet(1, (1.0,)),
    "foot_contact_forces": _swap_feet(3, (1.0, -1.0, 1.0)),  # world frame
    "foot_target": _swap_feet(3, (1.0, -1.0, 1.0)),  # (left xyz, right xyz)
    "arm_target": _joints(PICO_ARM_JOINT_NAMES),
}


def _group_mirror(layout: tuple[tuple[str, int], ...]) -> tuple[torch.Tensor, torch.Tensor]:
    index: list[int] = []
    sign: list[float] = []
    offset = 0
    for term, width in layout:
        source, term_sign = _TERM_MIRRORS[term]
        if len(source) != width:
            raise ValueError(f"Mirror of {term} has {len(source)} columns, the term {width}")
        index.extend(offset + column for column in source)
        sign.extend(term_sign)
        offset += width
    return torch.tensor(index), torch.tensor(sign)


MIRROR_OBSERVATION = {
    group: _group_mirror(layout) for group, layout in TELEOP_V12_OBSERVATION_TERM_LAYOUTS.items()
}
MIRROR_ACTION = _group_mirror((("actions", len(MICROBAN_TELEOP_ACTION_JOINT_NAMES)),))


def _apply(mirror: tuple[torch.Tensor, torch.Tensor], value: torch.Tensor) -> torch.Tensor:
    index, sign = mirror
    if value.shape[-1] != index.numel():
        raise ValueError(f"Mirror expects {index.numel()} columns, got {value.shape[-1]}")
    return value[..., index.to(value.device)] * sign.to(value.device, value.dtype)


def mirror_observations(obs: TensorDict) -> TensorDict:
    unknown = set(obs.keys()) - set(MIRROR_OBSERVATION)
    if unknown:
        raise KeyError(f"No mirror for observation groups {sorted(unknown)}")
    return TensorDict(
        {group: _apply(MIRROR_OBSERVATION[group], value) for group, value in obs.items()},
        batch_size=obs.batch_size,
        device=obs.device,
    )


def mirror_actions(actions: torch.Tensor) -> torch.Tensor:
    return _apply(MIRROR_ACTION, actions)


def mirror_augmentation(
    env=None,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """rsl_rl's ``data_augmentation_func``: each input followed by its mirror."""

    del env
    if obs is not None:
        obs = torch.cat([obs, mirror_observations(obs)], dim=0)
    if actions is not None:
        actions = torch.cat([actions, mirror_actions(actions)], dim=0)
    return obs, actions
