"""Exact Microban arm forward kinematics for reachable hand commands.

The PICO policy no longer tracks hand targets (its arms are driven from
outside, microban_teleop_mdp); the FK draws its arm targets clear of the
trunk, and the hand-target record below is still written to the robot's HOME
file (``hand_target_fk``), whose target frame the teleop reads.

The constants below are the ``body`` and ``site`` transforms from
``robot/microban/robot.xml``.  Every arm joint is a hinge about its local +Z
axis.  Joint tensors and results use the fixed side order ``(left, right)`` and
joint order ``(shoulder_pitch, shoulder_roll, elbow)``.

The FK itself is in the trunk frame.  PICO hand targets are offsets in the
HOME-levelled trunk frame ``R_trunk * R_y(-HOME_TRUNK_PITCH_RAD)`` (gravity
level at HOME, x forward, y left, z up), so a trunk-frame FK offset ``v`` is the
target ``R_y(HOME_TRUNK_PITCH_RAD) @ v`` (``microban_hand_target_offsets_from_
arm_joints``).  With a vertical trunk at HOME that frame is the trunk frame.
The AABB, normalizer and evaluation offsets describe targets.

Sampled targets are restricted to the robot receiver's hand box (microban
``network_input``: each component within +-0.8 * 0.08 m = +-64 mm, endpoints
accepted) by rejecting joint samples whose target offset leaves it, so every
training target is both FK-reachable and deliverable on hardware.  The
vertical-trunk HOME's whole joint box is inside that box, so nothing is ever
rejected there.

Two contracts follow from the HOME (config/home_pose.yaml):

* trunk vertical: the trunk-frame contract (revision ``..._v2``; another arm
  HOME shifts the box and appends its hash);
* trunk pitched: the HOME-levelled receiver-box contract (revision
  ``..._receiver_box64mm_v4``; the F evaluation pose at (-20, 25, -50) deg).  The forward-lean HOME's values are the recorded ones;
  any other pitched HOME computes its box on the same 401^3 grid.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import torch

from mjlab_microban.robot.home_pose import HOME

HOME_TRUNK_PITCH_RAD = HOME.trunk_pitch_rad
# Revision of the FK box computed at the legacy arm HOME below; another arm
# HOME appends its own hash (see _derive_hand_fk_bounds).
_LEGACY_HAND_FK_REVISION = (
    "microban_robot_xml_arm_fk_reachable_box_elbow_upper_minus10_v2"
)
_LEVELLED_HAND_FK_REVISION_PREFIX = (
    "microban_robot_xml_arm_fk_reachable_box_elbow_upper_minus10_home_levelled_"
)
_LEVELLED_HAND_FK_REVISION_SUFFIX = "_receiver_box64mm_v4"
_TRUNK_TARGET_FRAME = "robot_trunk_xyz_forward_left_up"
_LEVELLED_TARGET_FRAME = "robot_home_levelled_trunk_xyz_forward_left_up"
# Hand/foot target frame: the trunk frame with the HOME lean rotated out (the
# trunk frame itself when the trunk is vertical at HOME).
MICROBAN_HAND_TARGET_FRAME = (
    _TRUNK_TARGET_FRAME if HOME_TRUNK_PITCH_RAD == 0.0 else _LEVELLED_TARGET_FRAME
)
MICROBAN_HAND_TARGET_FRAME_PITCH_RAD = HOME_TRUNK_PITCH_RAD
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
# The arm part of HOME (config/home_pose.yaml), (left, right) x (pitch, roll, elbow).
MICROBAN_ARM_HOME_JOINT_DEG = HOME.arm_joint_deg()
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
# extrema in trunk-frame metres, as offsets from the hand FK at the legacy arm
# HOME (0, +-10, -20) deg.  The bilateral chains mirror only Y.
MICROBAN_HAND_FK_BOUND_GRID_POINTS_PER_AXIS = 401
_LEGACY_ARM_HOME_JOINT_DEG = (
    (0.0, 10.0, -20.0),
    (0.0, -10.0, -20.0),
)
_LEGACY_HAND_FK_OFFSET_AABB_MIN_M = (
    (-0.06120170602356862, -0.0034550417440758485, -0.0035619649312883805),
    (-0.06120170602356864, -0.0387512193701912, -0.0035619649312883944),
)
_LEGACY_HAND_FK_OFFSET_AABB_MAX_M = (
    (0.06289464331528255, 0.0387512193701912, 0.060477220857479266),
    (0.06289464331528258, 0.0034550417440758485, 0.060477220857479225),
)

# Conservative outward rounding of the per-axis maximum absolute grid values.
# These are actor-normalizer denominators, not the wire protocol's ±0.08 m
# command envelope.  The reachable joint/FK subset also remains strictly
# inside the receiver's independently validated ±0.064 m live margin.
_LEGACY_HAND_TARGET_NORMALIZER_ABS_BOUND_M = (0.0630, 0.0388, 0.0605)
MICROBAN_HAND_TARGET_WIRE_ABS_BOUND_M = (0.08, 0.08, 0.08)
# The receiver's live hand box: each component within +-0.8 * 0.08 m (microban
# network_input _PICO_HAND_TARGET_LOWER/UPPER; the sender declares the 0.8
# margin).  0.08 * 0.8 is exactly the float 0.064.
MICROBAN_HAND_TARGET_RECEIVER_SAFETY_MARGIN = 0.8
MICROBAN_HAND_TARGET_RUNTIME_VALIDATED_ABS_LIMIT_M = tuple(
    value * MICROBAN_HAND_TARGET_RECEIVER_SAFETY_MARGIN
    for value in MICROBAN_HAND_TARGET_WIRE_ABS_BOUND_M
)
# Rejected joint samples are redrawn; after this many rounds (each loses only
# ~1 % of the draws at the forward-lean HOME) any still-rejected hand falls
# back to its HOME joints.
MICROBAN_HAND_TARGET_MAX_REJECTION_ROUNDS = 64

# Named points shared by the v12 evaluator and training contract.  Each tuple is
# one left-arm (pitch, roll, elbow) pose in degrees; the right pose mirrors only
# shoulder roll because the MJCF arm chains are bilateral mirrors.  With a
# pitched HOME trunk, F (-25, 25, -50) reads 68.6 mm forward in the levelled
# frame of the 10 deg lean, outside the receiver box, so the levelled contract
# raises its shoulder pitch to -20 deg: (62.3, 20.0, 39.7) mm.
_TRUNK_FRAME_EVALUATION_JOINTS_DEG = (
    ("F", (-25.0, 25.0, -50.0)),
    ("B", (25.0, 20.0, -10.0)),
    ("f", (-12.0, 18.0, -32.0)),
    ("b", (12.0, 18.0, -32.0)),
)
_LEVELLED_EVALUATION_JOINTS_DEG = (
    ("F", (-20.0, 25.0, -50.0)),
    ("B", (25.0, 20.0, -10.0)),
    ("f", (-12.0, 18.0, -32.0)),
    ("b", (12.0, 18.0, -32.0)),
)
MICROBAN_REACHABLE_HAND_EVALUATION_JOINTS_DEG = (
    _TRUNK_FRAME_EVALUATION_JOINTS_DEG
    if HOME_TRUNK_PITCH_RAD == 0.0
    else _LEVELLED_EVALUATION_JOINTS_DEG
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


def _derive_trunk_frame_bounds() -> tuple[
    str,
    tuple[tuple[float, float, float], tuple[float, float, float]],
    tuple[tuple[float, float, float], tuple[float, float, float]],
    tuple[float, float, float],
]:
    """Return (revision, trunk-frame offset AABB min, max, normalizer) at the arm HOME.

    The joint box's absolute hand-position AABB does not depend on HOME, so an
    offset AABB at another arm HOME is the legacy one shifted by the FK
    difference of the two HOMEs.  The legacy arm HOME keeps its recorded values
    bit for bit.  Another arm HOME rounds the normalizer outward to 0.1 mm and
    must stay inside the receiver's validated runtime box.
    """

    if MICROBAN_ARM_HOME_JOINT_DEG == _LEGACY_ARM_HOME_JOINT_DEG:
        return (
            _LEGACY_HAND_FK_REVISION,
            _LEGACY_HAND_FK_OFFSET_AABB_MIN_M,
            _LEGACY_HAND_FK_OFFSET_AABB_MAX_M,
            _LEGACY_HAND_TARGET_NORMALIZER_ABS_BOUND_M,
        )
    to_rad = lambda table: torch.tensor(  # noqa: E731
        [[math.radians(value) for value in side] for side in table], dtype=torch.float64
    )
    shift = microban_hand_positions_from_arm_joints(
        to_rad(_LEGACY_ARM_HOME_JOINT_DEG)
    ) - microban_hand_positions_from_arm_joints(to_rad(MICROBAN_ARM_HOME_JOINT_DEG))
    low = torch.tensor(_LEGACY_HAND_FK_OFFSET_AABB_MIN_M, dtype=torch.float64) + shift
    high = torch.tensor(_LEGACY_HAND_FK_OFFSET_AABB_MAX_M, dtype=torch.float64) + shift
    extreme = torch.maximum(low.abs(), high.abs()).amax(dim=0)
    normalizer = _outward_normalizer(extreme.tolist())
    if HOME_TRUNK_PITCH_RAD == 0.0:
        for value, limit in zip(normalizer, MICROBAN_HAND_TARGET_RUNTIME_VALIDATED_ABS_LIMIT_M):
            if value >= limit:
                raise ValueError(
                    f"Arm HOME {MICROBAN_ARM_HOME_JOINT_DEG} moves the reachable hand box "
                    f"outside the runtime-validated +-{limit} m receiver box"
                )
    digest = hashlib.sha256(repr(MICROBAN_ARM_HOME_JOINT_DEG).encode("ascii")).hexdigest()
    return (
        f"{_LEGACY_HAND_FK_REVISION}_arm_home_{digest[:10]}",
        tuple(tuple(float(v) for v in side) for side in low.tolist()),  # type: ignore[return-value]
        tuple(tuple(float(v) for v in side) for side in high.tolist()),  # type: ignore[return-value]
        normalizer,  # type: ignore[return-value]
    )


def _outward_normalizer(extreme: list[float]) -> tuple[float, float, float]:
    """Outward 0.1 mm rounding of per-axis maximum absolute values."""

    return tuple(math.ceil(float(value) * 1.0e4 - 1.0e-9) / 1.0e4 for value in extreme)  # type: ignore[return-value]


# The HOME-levelled receiver-box contract recorded for the forward-lean HOME
# (trunk +10 deg, legacy arm HOME): a 401^3 grid per side over the joint box,
# rotated into the levelled frame (x' = cos*x + sin*z, z' = -sin*x + cos*z) and
# restricted to the grid points inside the receiver box.  Unrestricted, that
# grid reaches x = 70.6 mm and z = 49.5 mm; the restriction removes 0.997 % of
# each side's points (all forward-up: x > 64 mm), which also lowers the z
# maximum to 45.7 mm.  Keyed by (arm HOME deg, trunk pitch deg).
_RECORDED_LEVELLED_HAND_FK = {
    (_LEGACY_ARM_HOME_JOINT_DEG, 10.0): (
        f"{_LEVELLED_HAND_FK_REVISION_PREFIX}lean10{_LEVELLED_HAND_FK_REVISION_SUFFIX}",
        (
            (-0.05976319909948816, -0.0034550417440758485, -0.0012919613069190107),
            (-0.05976319909948818, -0.0387512193701912, -0.0012919613069190233),
        ),
        (
            (0.06399997388471845, 0.0387512193701912, 0.04573068026176937),
            (0.06399997388471848, 0.0034550417440758485, 0.04573068026176935),
        ),
        # x equals the receiver limit: the restricted box touches it.
        (0.0640, 0.0388, 0.0458),
    ),
}


def levelled_receiver_box_grid_bounds(
    arm_home_joint_deg: tuple[tuple[float, float, float], tuple[float, float, float]],
    trunk_pitch: float,
    *,
    points_per_axis: int = MICROBAN_HAND_FK_BOUND_GRID_POINTS_PER_AXIS,
) -> tuple[
    tuple[tuple[float, float, float], tuple[float, float, float]],
    tuple[tuple[float, float, float], tuple[float, float, float]],
]:
    """AABB of the HOME-levelled target offsets inside the receiver box (grid).

    Each side's joint box is sampled on a ``points_per_axis``^3 grid; offsets
    are from the FK at ``arm_home_joint_deg``, rotated by ``R_y(trunk_pitch)``,
    and only points with every component within the receiver's live limit
    count.  The chain is evaluated separably (elbow, then roll, then pitch), a
    few seconds on CPU; it agrees with the per-point FK to ~1e-17 m.
    """

    dtype = torch.float64
    limit = torch.tensor(MICROBAN_HAND_TARGET_RUNTIME_VALIDATED_ABS_LIMIT_M, dtype=dtype)
    home = torch.tensor(
        [[math.radians(value) for value in side] for side in arm_home_joint_deg], dtype=dtype
    )
    default = microban_hand_positions_from_arm_joints(home)
    lower = torch.tensor(MICROBAN_ARM_JOINT_LOWER_RAD, dtype=dtype)
    upper = torch.tensor(MICROBAN_ARM_JOINT_UPPER_RAD, dtype=dtype)
    like = home
    shoulder_pos = _constant(_SHOULDER_BODY_POS_M, like)
    shoulder_rotation = _quaternion_wxyz_to_matrix(_constant(_SHOULDER_BODY_QUAT_WXYZ, like))
    humerus_pos = _constant(_HUMERUS_BODY_POS_M, like)
    humerus_rotation = _quaternion_wxyz_to_matrix(_constant(_HUMERUS_BODY_QUAT_WXYZ, like))
    radius_pos = _constant(_RADIUS_BODY_POS_M, like)
    radius_rotation = _quaternion_wxyz_to_matrix(_constant(_RADIUS_BODY_QUAT_WXYZ, like))
    hand_site_pos = _constant(_HAND_SITE_POS_M, like)
    cosine, sine = math.cos(trunk_pitch), math.sin(trunk_pitch)
    lows, highs = [], []
    for side in range(2):
        axes = [
            torch.linspace(lower[side, k], upper[side, k], points_per_axis, dtype=dtype)
            for k in range(3)
        ]
        # r + Rd Rz(c) p, then H Rz(b) (...): shape (b, c, 3).
        inner = radius_pos[side] + torch.einsum(
            "cij,j->ci", radius_rotation[side] @ _rotation_z(axes[2]), hand_site_pos[side]
        )
        middle = humerus_pos[side] + torch.einsum(
            "bij,cj->bci", humerus_rotation[side] @ _rotation_z(axes[1]), inner
        )
        outer = shoulder_rotation[side] @ _rotation_z(axes[0])  # (a, 3, 3)
        low = torch.full((3,), math.inf, dtype=dtype)
        high = torch.full((3,), -math.inf, dtype=dtype)
        for index in range(points_per_axis):
            offsets = (
                shoulder_pos[side]
                + torch.einsum("ij,bcj->bci", outer[index], middle).reshape(-1, 3)
                - default[side]
            )
            x, y, z = offsets.unbind(dim=-1)
            target = torch.stack((cosine * x + sine * z, y, -sine * x + cosine * z), dim=-1)
            inside = ((target >= -limit) & (target <= limit)).all(dim=-1, keepdim=True)
            low = torch.minimum(low, torch.where(inside, target, math.inf).amin(dim=0))
            high = torch.maximum(high, torch.where(inside, target, -math.inf).amax(dim=0))
        lows.append(tuple(float(v) for v in low.tolist()))
        highs.append(tuple(float(v) for v in high.tolist()))
    return tuple(lows), tuple(highs)  # type: ignore[return-value]


# The grid takes tens of seconds; its result is cached per (arm HOME, trunk
# pitch, this file, robot.xml) so every process of a new pitched HOME does not
# recompute it.
LEVELLED_HAND_FK_CACHE_DIR = Path(
    os.environ.get("MJLAB_MICROBAN_CACHE_DIR", Path.home() / ".cache" / "mjlab_microban")
) / "levelled_hand_fk"


def _cached_levelled_grid_bounds(arm_home_joint_deg, trunk_pitch):  # noqa: ANN001, ANN202
    source = Path(__file__).read_bytes() + (Path(__file__).parent / "microban" / "robot.xml").read_bytes()
    digest = hashlib.sha256(
        repr((arm_home_joint_deg, float(trunk_pitch))).encode("ascii") + source
    ).hexdigest()
    path = LEVELLED_HAND_FK_CACHE_DIR / f"{digest}.json"
    try:
        cached = json.loads(path.read_text())
        return (
            tuple(tuple(float(v) for v in side) for side in cached["low"]),
            tuple(tuple(float(v) for v in side) for side in cached["high"]),
        )
    except (OSError, ValueError, KeyError, TypeError):
        pass
    low, high = levelled_receiver_box_grid_bounds(arm_home_joint_deg, trunk_pitch)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps({"low": low, "high": high}))
        temporary.replace(path)
    except OSError:
        pass
    return low, high


def _derive_levelled_bounds() -> tuple[
    str,
    tuple[tuple[float, float, float], tuple[float, float, float]],
    tuple[tuple[float, float, float], tuple[float, float, float]],
    tuple[float, float, float],
]:
    """(revision, levelled target AABB min, max, normalizer) at a pitched HOME."""

    key = (MICROBAN_ARM_HOME_JOINT_DEG, HOME.trunk_pitch_deg)
    if key in _RECORDED_LEVELLED_HAND_FK:
        return _RECORDED_LEVELLED_HAND_FK[key]
    low, high = _cached_levelled_grid_bounds(MICROBAN_ARM_HOME_JOINT_DEG, HOME_TRUNK_PITCH_RAD)
    extreme = [
        max(abs(low[side][axis]), abs(high[side][axis])) for axis in range(3) for side in range(2)
    ]
    normalizer = _outward_normalizer(
        [max(extreme[2 * axis], extreme[2 * axis + 1]) for axis in range(3)]
    )
    digest = hashlib.sha256(repr(key).encode("ascii")).hexdigest()
    return (
        f"{_LEVELLED_HAND_FK_REVISION_PREFIX}{digest[:10]}{_LEVELLED_HAND_FK_REVISION_SUFFIX}",
        low,
        high,
        normalizer,
    )


(
    _TRUNK_FRAME_REVISION,
    MICROBAN_HAND_FK_TRUNK_OFFSET_AABB_MIN_M,
    MICROBAN_HAND_FK_TRUNK_OFFSET_AABB_MAX_M,
    _TRUNK_FRAME_NORMALIZER,
) = _derive_trunk_frame_bounds()
if HOME_TRUNK_PITCH_RAD == 0.0:
    # Vertical trunk: targets are trunk-frame offsets.
    MICROBAN_HAND_FK_REVISION = _TRUNK_FRAME_REVISION
    MICROBAN_HAND_FK_OFFSET_AABB_MIN_M = MICROBAN_HAND_FK_TRUNK_OFFSET_AABB_MIN_M
    MICROBAN_HAND_FK_OFFSET_AABB_MAX_M = MICROBAN_HAND_FK_TRUNK_OFFSET_AABB_MAX_M
    MICROBAN_HAND_TARGET_NORMALIZER_ABS_BOUND_M = _TRUNK_FRAME_NORMALIZER
else:
    (
        MICROBAN_HAND_FK_REVISION,
        MICROBAN_HAND_FK_OFFSET_AABB_MIN_M,
        MICROBAN_HAND_FK_OFFSET_AABB_MAX_M,
        MICROBAN_HAND_TARGET_NORMALIZER_ABS_BOUND_M,
    ) = _derive_levelled_bounds()


def microban_hand_offsets_from_arm_joints(
    joint_positions_rad: torch.Tensor,
) -> torch.Tensor:
    """Return hand XYZ offsets from the exact software-HOME FK positions."""

    default = microban_default_hand_positions(
        device=joint_positions_rad.device, dtype=joint_positions_rad.dtype
    )
    return microban_hand_positions_from_arm_joints(joint_positions_rad) - default


def rotate_trunk_offsets_to_home_levelled(
    offsets: torch.Tensor, trunk_pitch: float
) -> torch.Tensor:
    """Express trunk-frame ``[..., 3]`` vectors in the HOME-levelled frame.

    The levelled frame is ``R_trunk * R_y(-trunk_pitch)``, so a trunk-frame
    vector ``v`` reads ``R_y(trunk_pitch) @ v`` there.  ``trunk_pitch = 0``
    returns the input unchanged.
    """

    if trunk_pitch == 0.0:
        return offsets
    cosine = math.cos(trunk_pitch)
    sine = math.sin(trunk_pitch)
    x, y, z = offsets.unbind(dim=-1)
    return torch.stack((cosine * x + sine * z, y, -sine * x + cosine * z), dim=-1)


def microban_hand_target_offsets_from_arm_joints(
    joint_positions_rad: torch.Tensor,
    *,
    trunk_pitch: float = MICROBAN_HAND_TARGET_FRAME_PITCH_RAD,
) -> torch.Tensor:
    """Return HOME-levelled hand target offsets for ``[..., 2, 3]`` joints."""

    return rotate_trunk_offsets_to_home_levelled(
        microban_hand_offsets_from_arm_joints(joint_positions_rad), trunk_pitch
    )


def microban_hand_target_offsets_within_limit(
    offsets: torch.Tensor,
    abs_limit_m: tuple[float, float, float] = (
        MICROBAN_HAND_TARGET_RUNTIME_VALIDATED_ABS_LIMIT_M
    ),
) -> torch.Tensor:
    """Return ``[...]`` bools: every component of ``[..., 3]`` lies in the box.

    Mirrors the receiver's rule (a component is dropped only when it is below
    ``-limit`` or above ``+limit``); the comparison runs in float64 so a
    float32 target never passes by rounding up to the float32 limit.
    """

    limit = torch.tensor(abs_limit_m, dtype=torch.float64, device=offsets.device)
    values = offsets.to(torch.float64)
    return ((values >= -limit) & (values <= limit)).all(dim=-1)


def microban_reachable_hand_evaluation_offsets() -> tuple[
    tuple[str, tuple[tuple[float, float, float], tuple[float, float, float]]], ...
]:
    """Return named left/right reachable target offsets (HOME-levelled frame)."""

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
        offsets = microban_hand_target_offsets_from_arm_joints(torch.deg2rad(joints))
        left = tuple(float(value) for value in offsets[0].tolist())
        right = tuple(float(value) for value in offsets[1].tolist())
        result.append((name, (left, right)))
    return tuple(result)


if HOME_TRUNK_PITCH_RAD != 0.0:
    for _name, _sides in microban_reachable_hand_evaluation_offsets():
        for _offset in _sides:
            if any(
                abs(value) > limit
                for value, limit in zip(_offset, MICROBAN_HAND_TARGET_RUNTIME_VALIDATED_ABS_LIMIT_M)
            ):
                raise ValueError(
                    f"Hand evaluation pose {_name} reads {_offset} m in the HOME-levelled "
                    "frame, outside the receiver's +-64 mm hand box"
                )


def microban_hand_fk_metadata() -> dict[str, object]:
    """Return JSON-safe FK sampling and normalizer provenance."""

    if HOME_TRUNK_PITCH_RAD == 0.0:
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
    return {
        "revision": MICROBAN_HAND_FK_REVISION,
        "target_frame": MICROBAN_HAND_TARGET_FRAME,
        "target_frame_trunk_pitch_rad": MICROBAN_HAND_TARGET_FRAME_PITCH_RAD,
        "side_order": list(MICROBAN_HAND_SIDE_ORDER),
        "joint_order": list(MICROBAN_ARM_JOINT_ORDER),
        "joint_lower_deg": [list(side) for side in MICROBAN_ARM_JOINT_LOWER_DEG],
        "joint_upper_deg": [list(side) for side in MICROBAN_ARM_JOINT_UPPER_DEG],
        "home_joint_deg": [list(side) for side in MICROBAN_ARM_HOME_JOINT_DEG],
        "bound_grid_points_per_axis": MICROBAN_HAND_FK_BOUND_GRID_POINTS_PER_AXIS,
        "trunk_offset_aabb_min_m": [
            list(side) for side in MICROBAN_HAND_FK_TRUNK_OFFSET_AABB_MIN_M
        ],
        "trunk_offset_aabb_max_m": [
            list(side) for side in MICROBAN_HAND_FK_TRUNK_OFFSET_AABB_MAX_M
        ],
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
        "max_rejection_rounds": MICROBAN_HAND_TARGET_MAX_REJECTION_ROUNDS,
        "evaluation_joint_degrees": [
            [name, list(values)]
            for name, values in MICROBAN_REACHABLE_HAND_EVALUATION_JOINTS_DEG
        ],
        "evaluation_offsets_m": [
            [name, [list(side) for side in offsets]]
            for name, offsets in microban_reachable_hand_evaluation_offsets()
        ],
        "source": "src/mjlab_microban/robot/microban/robot.xml",
        "sampling": (
            "uniform_independent_joint_box_then_exact_fk_offset_from_home_"
            "rotated_into_home_levelled_frame_rejecting_joint_samples_outside_"
            "runtime_validated_abs_limit"
        ),
    }
