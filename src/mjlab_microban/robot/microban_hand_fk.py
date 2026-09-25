"""Exact Microban arm forward kinematics for reachable hand commands.

The constants below are the ``body`` and ``site`` transforms from
``robot/microban/robot.xml``.  Every arm joint is a hinge about its local +Z
axis.  Joint tensors and results use the fixed side order ``(left, right)`` and
joint order ``(shoulder_pitch, shoulder_roll, elbow)``.
"""

from __future__ import annotations

import math

import torch

MICROBAN_HAND_FK_REVISION = (
    "microban_robot_xml_arm_fk_reachable_box_elbow_upper_minus10_v2"
)
MICROBAN_HAND_SIDE_ORDER = ("left", "right")
MICROBAN_ARM_JOINT_ORDER = ("shoulder_pitch", "shoulder_roll", "elbow")

MICROBAN_ARM_JOINT_LOWER_DEG = (
    (-25.0, 10.0, -50.0),
    (-25.0, -30.0, -50.0),
)
MICROBAN_ARM_JOINT_UPPER_DEG = (
    (25.0, 30.0, -10.0),
    (25.0, -10.0, -10.0),
)
MICROBAN_ARM_HOME_JOINT_DEG = (
    (0.0, 10.0, -20.0),
    (0.0, -10.0, -20.0),
)
MICROBAN_ARM_JOINT_LOWER_RAD = tuple(
    tuple(math.radians(value) for value in side)
    for side in MICROBAN_ARM_JOINT_LOWER_DEG
)
MICROBAN_ARM_JOINT_UPPER_RAD = tuple(
    tuple(math.radians(value) for value in side)
    for side in MICROBAN_ARM_JOINT_UPPER_DEG
)
MICROBAN_ARM_HOME_JOINT_RAD = tuple(
    tuple(math.radians(value) for value in side) for side in MICROBAN_ARM_HOME_JOINT_DEG
)

# A 401^3 grid per side over the complete joint box produced these exact-FK
# extrema in trunk-frame metres.  The bilateral chains mirror only Y.
MICROBAN_HAND_FK_BOUND_GRID_POINTS_PER_AXIS = 401
MICROBAN_HAND_FK_OFFSET_AABB_MIN_M = (
    (-0.06120170602356862, -0.0034550417440758485, -0.0035619649312883805),
    (-0.06120170602356864, -0.0387512193701912, -0.0035619649312883944),
)
MICROBAN_HAND_FK_OFFSET_AABB_MAX_M = (
    (0.06289464331528255, 0.0387512193701912, 0.060477220857479266),
    (0.06289464331528258, 0.0034550417440758485, 0.060477220857479225),
)

# Conservative outward rounding of the per-axis maximum absolute grid values.
# These are actor-normalizer denominators, not the wire protocol's ±0.08 m
# command envelope.  The reachable joint/FK subset also remains strictly
# inside the receiver's independently validated ±0.064 m live margin.
MICROBAN_HAND_TARGET_NORMALIZER_ABS_BOUND_M = (0.0630, 0.0388, 0.0605)
MICROBAN_HAND_TARGET_WIRE_ABS_BOUND_M = (0.08, 0.08, 0.08)
MICROBAN_HAND_TARGET_RUNTIME_VALIDATED_ABS_LIMIT_M = (0.064, 0.064, 0.064)

# Named points shared by the v12 evaluator and training contract.  Each tuple is
# one left-arm (pitch, roll, elbow) pose in degrees; the right pose mirrors only
# shoulder roll because the MJCF arm chains are bilateral mirrors.
MICROBAN_REACHABLE_HAND_EVALUATION_JOINTS_DEG = (
    ("F", (-25.0, 25.0, -50.0)),
    ("B", (25.0, 20.0, -10.0)),
    ("f", (-12.0, 18.0, -32.0)),
    ("b", (12.0, 18.0, -32.0)),
)

# MJCF body/site transforms in (left, right) order.  Quaternions are wxyz and
# normalized in the same way as MuJoCo before conversion to rotation matrices.
_SHOULDER_BODY_POS_M = ((-0.0, 0.0505, 0.081), (0.0, -0.0505, 0.081))
_SHOULDER_BODY_QUAT_WXYZ = (
    (0.0, 0.0, -0.707107, -0.707107),
    (0.707107, -0.707107, -0.0, 0.0),
)
_HUMERUS_BODY_POS_M = ((-0.0145, 0.0, 0.0195), (0.0145, 0.0, -0.0195))
_HUMERUS_BODY_QUAT_WXYZ = (
    (0.0, -0.707107, -0.0, 0.707107),
    (0.0, 0.707107, 0.0, 0.707107),
)
_RADIUS_BODY_POS_M = ((-0.0145, 0.0515, -0.0145), (-0.0145, -0.0515, -0.0145))
_RADIUS_BODY_QUAT_WXYZ = (
    (0.707107, 0.0, -0.707107, 0.0),
    (0.707107, 0.0, 0.707107, -0.0),
)
_HAND_SITE_POS_M = ((0.0, 0.067014, -0.0135), (0.0, -0.067014, 0.0135))


def _constant(
    values: tuple[tuple[float, ...], tuple[float, ...]], like: torch.Tensor
) -> torch.Tensor:
    return torch.as_tensor(values, dtype=like.dtype, device=like.device)


def _quaternion_wxyz_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.stack(
        (
            1.0 - 2.0 * (y.square() + z.square()),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x.square() + z.square()),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x.square() + y.square()),
        ),
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)


def _rotation_z(angle: torch.Tensor) -> torch.Tensor:
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    zero = torch.zeros_like(angle)
    one = torch.ones_like(angle)
    return torch.stack(
        (cosine, -sine, zero, sine, cosine, zero, zero, zero, one), dim=-1
    ).reshape(*angle.shape, 3, 3)


def _transform_vector(rotation: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return torch.matmul(rotation, vector.unsqueeze(-1)).squeeze(-1)


def microban_hand_positions_from_arm_joints(
    joint_positions_rad: torch.Tensor,
) -> torch.Tensor:
    """Return exact hand-site XYZ in the trunk frame for ``[..., 2, 3]`` joints."""

    if not isinstance(joint_positions_rad, torch.Tensor):
        raise TypeError("Microban arm joints must be a torch.Tensor")
    if joint_positions_rad.shape[-2:] != (2, 3):
        raise ValueError("Microban arm joints must end in shape (left/right=2, dof=3)")
    if not joint_positions_rad.is_floating_point():
        raise TypeError("Microban arm joints must use a floating dtype")
    if not bool(torch.isfinite(joint_positions_rad).all().item()):
        raise ValueError("Microban arm joints must be finite")

    shoulder_pos = _constant(_SHOULDER_BODY_POS_M, joint_positions_rad)
    shoulder_rotation = _quaternion_wxyz_to_matrix(
        _constant(_SHOULDER_BODY_QUAT_WXYZ, joint_positions_rad)
    )
    humerus_pos = _constant(_HUMERUS_BODY_POS_M, joint_positions_rad)
    humerus_rotation = _quaternion_wxyz_to_matrix(
        _constant(_HUMERUS_BODY_QUAT_WXYZ, joint_positions_rad)
    )
    radius_pos = _constant(_RADIUS_BODY_POS_M, joint_positions_rad)
    radius_rotation = _quaternion_wxyz_to_matrix(
        _constant(_RADIUS_BODY_QUAT_WXYZ, joint_positions_rad)
    )
    hand_site_pos = _constant(_HAND_SITE_POS_M, joint_positions_rad)

    rotation = shoulder_rotation @ _rotation_z(joint_positions_rad[..., 0])
    position = shoulder_pos + _transform_vector(rotation, humerus_pos)
    rotation = rotation @ humerus_rotation @ _rotation_z(joint_positions_rad[..., 1])
    position = position + _transform_vector(rotation, radius_pos)
    rotation = rotation @ radius_rotation @ _rotation_z(joint_positions_rad[..., 2])
    return position + _transform_vector(rotation, hand_site_pos)


def microban_default_hand_positions(
    *, device: torch.device | str, dtype: torch.dtype
) -> torch.Tensor:
    """Return the two exact hand positions at Microban's software HOME pose."""

    home = torch.tensor(
        MICROBAN_ARM_HOME_JOINT_RAD, dtype=dtype, device=torch.device(device)
    )
    return microban_hand_positions_from_arm_joints(home)


def microban_hand_offsets_from_arm_joints(
    joint_positions_rad: torch.Tensor,
) -> torch.Tensor:
    """Return hand XYZ offsets from the exact software-HOME FK positions."""

    default = microban_default_hand_positions(
        device=joint_positions_rad.device, dtype=joint_positions_rad.dtype
    )
    return microban_hand_positions_from_arm_joints(joint_positions_rad) - default


def sample_microban_reachable_hand_targets(
    is_active: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniformly sample the arm joint box and return joints plus reachable offsets.

    Inactive hands are set to the exact HOME joint tuple before FK, producing an
    exact-zero offset instead of sampling a Cartesian point that will be ignored.
    """

    if not isinstance(is_active, torch.Tensor) or is_active.dtype != torch.bool:
        raise TypeError("Microban hand activation mask must be a bool tensor")
    if is_active.ndim != 2 or is_active.shape[1] != 2:
        raise ValueError("Microban hand activation mask must have shape (N, 2)")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError("Microban hand target samples require a floating dtype")
    lower = torch.tensor(
        MICROBAN_ARM_JOINT_LOWER_RAD,
        dtype=dtype,
        device=is_active.device,
    )
    upper = torch.tensor(
        MICROBAN_ARM_JOINT_UPPER_RAD, dtype=lower.dtype, device=is_active.device
    )
    home = torch.tensor(
        MICROBAN_ARM_HOME_JOINT_RAD, dtype=lower.dtype, device=is_active.device
    )
    sampled = home.expand(is_active.shape[0], -1, -1).clone()
    active_env, active_side = is_active.nonzero(as_tuple=True)
    if len(active_env) > 0:
        random = torch.rand(
            (len(active_env), 3),
            dtype=lower.dtype,
            device=is_active.device,
            generator=generator,
        )
        sampled[active_env, active_side] = lower[active_side] + random * (
            upper[active_side] - lower[active_side]
        )
    offsets = microban_hand_offsets_from_arm_joints(sampled)
    offsets = torch.where(is_active.unsqueeze(-1), offsets, torch.zeros_like(offsets))
    return sampled, offsets


def microban_reachable_hand_evaluation_offsets() -> tuple[
    tuple[str, tuple[tuple[float, float, float], tuple[float, float, float]]], ...
]:
    """Return named left/right reachable offsets as JSON-safe Python tuples."""

    result = []
    for name, left_degrees in MICROBAN_REACHABLE_HAND_EVALUATION_JOINTS_DEG:
        pitch, roll, elbow = left_degrees
        joints = torch.tensor(
            (
                (pitch, roll, elbow),
                (pitch, -roll, elbow),
            ),
            dtype=torch.float64,
        )
        offsets = microban_hand_offsets_from_arm_joints(torch.deg2rad(joints))
        left = tuple(float(value) for value in offsets[0].tolist())
        right = tuple(float(value) for value in offsets[1].tolist())
        result.append((name, (left, right)))
    return tuple(result)


def microban_hand_fk_metadata() -> dict[str, object]:
    """Return JSON-safe FK sampling and normalizer provenance."""

    return {
        "revision": MICROBAN_HAND_FK_REVISION,
        "side_order": list(MICROBAN_HAND_SIDE_ORDER),
        "joint_order": list(MICROBAN_ARM_JOINT_ORDER),
        "joint_lower_deg": [list(side) for side in MICROBAN_ARM_JOINT_LOWER_DEG],
        "joint_upper_deg": [list(side) for side in MICROBAN_ARM_JOINT_UPPER_DEG],
        "home_joint_deg": [list(side) for side in MICROBAN_ARM_HOME_JOINT_DEG],
        "bound_grid_points_per_axis": MICROBAN_HAND_FK_BOUND_GRID_POINTS_PER_AXIS,
        "offset_aabb_min_m": [
            list(side) for side in MICROBAN_HAND_FK_OFFSET_AABB_MIN_M
        ],
        "offset_aabb_max_m": [
            list(side) for side in MICROBAN_HAND_FK_OFFSET_AABB_MAX_M
        ],
        "normalizer_abs_bound_m": list(MICROBAN_HAND_TARGET_NORMALIZER_ABS_BOUND_M),
        "wire_abs_bound_m": list(MICROBAN_HAND_TARGET_WIRE_ABS_BOUND_M),
        "runtime_validated_abs_limit_m": list(
            MICROBAN_HAND_TARGET_RUNTIME_VALIDATED_ABS_LIMIT_M
        ),
        "evaluation_joint_degrees": [
            [name, list(values)]
            for name, values in MICROBAN_REACHABLE_HAND_EVALUATION_JOINTS_DEG
        ],
        "evaluation_offsets_m": [
            [name, [list(side) for side in offsets]]
            for name, offsets in microban_reachable_hand_evaluation_offsets()
        ],
        "source": "src/mjlab_microban/robot/microban/robot.xml",
        "sampling": "uniform_independent_joint_box_then_exact_fk_offset_from_home",
    }
