"""Left-right mirror of the walking observations and actions, for the mirror loss.

The robot, the command distribution, the rewards and the evaluation are
left-right symmetric, but the walkers trained on the refitted servo model are
not: one side's lateral walk is weak, and which side changes during training
(0.1 m/s left on the 0.1 s floor with friction 0.3: 0.009-0.020 m/s, right
0.032-0.035).  rsl_rl's ``symmetry_cfg`` with ``use_mirror_loss`` adds
``coeff * MSE(pi(M o), M pi(o))`` to the PPO loss so that one run learns one
gait for both sides.  PICO (microban_teleop_mirror) uses the same mirror.

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
mirrored up to omega x 4 mm; the actor does not observe it.  Joint angles are
observed and commanded relative to HOME, which is mirror symmetric
(robot/home_pose.py).
"""

from __future__ import annotations

import math

import torch
from tensordict import TensorDict

from mjlab_microban.policy_contract import ACTION_JOINT_NAMES, HEAD_JOINTS, OBSERVATION_SCHEMAS

# (source column within the term, sign of each column)
TermMirror = tuple[list[int], list[float]]
# (source index, sign) over a whole observation group or action vector.
Mirror = tuple[torch.Tensor, torch.Tensor]

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


def joint_mirror(names: tuple[str, ...]) -> TermMirror:
    return (
        [names.index(mirror_partner(name)) for name in names],
        [mirror_joint_sign(name) for name in names],
    )


def swap_feet(per_foot: int, signs: tuple[float, ...]) -> TermMirror:
    """Two feet of ``per_foot`` values each, swapped, each value times ``signs``."""

    index = [*range(per_foot, 2 * per_foot), *range(per_foot)]
    return index, [*signs, *signs]


# The terms whose mirror does not depend on which joints a policy observes.
TERM_MIRRORS: dict[str, TermMirror] = {
    "base_lin_vel": ([0, 1, 2], [-1.0, 1.0, 1.0]),  # IMU frame, polar
    "base_ang_vel": ([0, 1, 2], [1.0, -1.0, -1.0]),  # IMU frame, axial
    "projected_gravity": ([0, 1, 2], [1.0, -1.0, 1.0]),  # trunk frame
    "command": ([0, 1, 2], [1.0, -1.0, -1.0]),  # vx, vy, yaw rate
    "foot_height": swap_feet(1, (1.0,)),
    "foot_air_time": swap_feet(1, (1.0,)),
    "foot_contact": swap_feet(1, (1.0,)),
    "foot_contact_forces": swap_feet(3, (1.0, -1.0, 1.0)),  # world frame
}


def group_mirror(layout: tuple[tuple[str, int], ...], terms: dict[str, TermMirror]) -> Mirror:
    index: list[int] = []
    sign: list[float] = []
    offset = 0
    for term, width in layout:
        source, term_sign = terms[term]
        if len(source) != width:
            raise ValueError(f"Mirror of {term} has {len(source)} columns, the term {width}")
        index.extend(offset + column for column in source)
        sign.extend(term_sign)
        offset += width
    return torch.tensor(index), torch.tensor(sign)


def apply_mirror(mirror: Mirror, value: torch.Tensor) -> torch.Tensor:
    index, sign = mirror
    if value.shape[-1] != index.numel():
        raise ValueError(f"Mirror expects {index.numel()} columns, got {value.shape[-1]}")
    return value[..., index.to(value.device)] * sign.to(value.device, value.dtype)


def mirror_groups(mirrors: dict[str, Mirror], obs: TensorDict) -> TensorDict:
    unknown = set(obs.keys()) - set(mirrors)
    if unknown:
        raise KeyError(f"No mirror for observation groups {sorted(unknown)}")
    return TensorDict(
        {group: apply_mirror(mirrors[group], value) for group, value in obs.items()},
        batch_size=obs.batch_size,
        device=obs.device,
    )


def augment(
    mirror_obs, mirror_act, obs: TensorDict | None, actions: torch.Tensor | None
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """rsl_rl's ``data_augmentation_func`` contract: each input followed by its mirror."""

    if obs is not None:
        obs = torch.cat([obs, mirror_obs(obs)], dim=0)
    if actions is not None:
        actions = torch.cat([actions, mirror_act(actions)], dim=0)
    return obs, actions


# The walker's observation groups (mjlab's velocity task as
# microban_velocity_env_cfg trims it): the actor observes the 18 commanded
# joints (policy_contract's walk schema), the critic all 21 in the model's
# natural order.  MicrobanVelocityOnPolicyRunner checks the env against these.
WALK_OBSERVATION_LAYOUTS: dict[str, tuple[tuple[str, int], ...]] = {
    "actor": OBSERVATION_SCHEMAS["walk"],
    "critic": (
        ("base_lin_vel", 3),
        ("base_ang_vel", 3),
        ("projected_gravity", 3),
        ("joint_pos", 21),
        ("joint_vel", 21),
        ("actions", len(ACTION_JOINT_NAMES)),
        ("command", 3),
        ("foot_height", 2),
        ("foot_air_time", 2),
        ("foot_contact", 2),
        ("foot_contact_forces", 6),
    ),
}
WALK_OBSERVATION_JOINTS: dict[str, tuple[str, ...]] = {
    "actor": ACTION_JOINT_NAMES,
    "critic": (*HEAD_JOINTS, *ACTION_JOINT_NAMES),
}


def _walk_terms(joints: tuple[str, ...]) -> dict[str, TermMirror]:
    return {
        **TERM_MIRRORS,
        "joint_pos": joint_mirror(joints),
        "joint_vel": joint_mirror(joints),
        "actions": joint_mirror(ACTION_JOINT_NAMES),
    }


WALK_MIRROR_OBSERVATION = {
    group: group_mirror(layout, _walk_terms(WALK_OBSERVATION_JOINTS[group]))
    for group, layout in WALK_OBSERVATION_LAYOUTS.items()
}
WALK_MIRROR_ACTION = group_mirror(
    (("actions", len(ACTION_JOINT_NAMES)),), {"actions": joint_mirror(ACTION_JOINT_NAMES)}
)


def mirror_observations(obs: TensorDict) -> TensorDict:
    return mirror_groups(WALK_MIRROR_OBSERVATION, obs)


def mirror_actions(actions: torch.Tensor) -> torch.Tensor:
    return apply_mirror(WALK_MIRROR_ACTION, actions)


def mirror_augmentation(
    env=None,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """rsl_rl's ``data_augmentation_func`` for the walker."""

    del env
    return augment(mirror_observations, mirror_actions, obs, actions)


def check_walk_mirror_layout(env) -> None:
    """Refuse an env whose observation terms, joint order or action joints differ from the mirror's."""

    manager = env.observation_manager
    robot = env.scene["robot"]
    if tuple(robot.joint_names) != WALK_OBSERVATION_JOINTS["critic"]:
        raise ValueError(f"Walk mirror: robot joint order drifted: {tuple(robot.joint_names)}")
    for group, layout in WALK_OBSERVATION_LAYOUTS.items():
        widths = (math.prod(dims) for dims in manager.group_obs_term_dim[group])
        actual = tuple(zip(manager.active_terms[group], widths, strict=True))
        if actual != layout:
            raise ValueError(f"Walk mirror: {group} observation layout drifted: {actual}")
        for term in ("joint_pos", "joint_vel"):
            names = getattr(manager.get_term_cfg(group, term).params.get("asset_cfg"), "joint_names", None)
            names = tuple(robot.joint_names) if names is None else tuple(names)
            if names != WALK_OBSERVATION_JOINTS[group]:
                raise ValueError(f"Walk mirror: {group}/{term} joint order drifted: {names}")
    targets = tuple(env.action_manager.get_term("joint_pos").target_names)
    if targets != ACTION_JOINT_NAMES:
        raise ValueError(f"Walk mirror: action joint order drifted: {targets}")
