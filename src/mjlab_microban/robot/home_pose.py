"""Microban HOME pose: one YAML file, everything else derived by MuJoCo FK.

The only HOME inputs are in ``config/home_pose.yaml`` at the repository root:
the 21 joint angles (degrees), the trunk pitch the soles are flat at, a
human-readable name and an identifier label.  This module loads that file and
computes every value that follows from it on ``robot/microban/robot.xml``:

* root pose: quaternion from the trunk pitch, z such that the lowest sole
  collision corner touches the ground (z = 0);
* projected gravity in the trunk frame at HOME;
* whole-body COM, the fore-aft extent of the sole contact area, the COM
  margins to heel and toe;
* the virtual head height (trunk COM + 0.07324 m along the trunk's inertial up
  axis, exactly as ``mdp._head_height``) and the lateral foot distance;
* the HOME identity: a short hash of the canonical joint values and trunk
  pitch, and the ``tag`` that every HOME-bound contract string embeds, so a
  checkpoint or ONNX trained at another HOME is refused automatically.

``analyze_pose`` is the pure FK function behind all of it.  It takes an
arbitrary joint-angle mapping (radians) and an optional trunk pitch and is
cheap (one ``mj_kinematics`` + ``mj_comPos`` on a cached model, tens of
microseconds), so solvers such as ``config/balance_home_pose.py`` can call it
in their inner loop.  ``rewrite_home_pose_yaml`` edits joint values in the
YAML in place, keeping every comment.

Nothing here depends on mjlab; only numpy, MuJoCo and PyYAML.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

import mujoco
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
HOME_POSE_YAML = REPO_ROOT / "config" / "home_pose.yaml"
MICROBAN_XML = Path(__file__).resolve().parent / "microban" / "robot.xml"

HOME_POSE_SCHEMA_VERSION = 1

# Canonical joint order (the historical HOME_FRAME order).  The YAML must name
# exactly these joints; every derived mapping uses this order.
HOME_JOINT_NAMES = (
    "head",
    "neck_roll",
    "neck_pitch",
    "left_shoulder_roll",
    "right_shoulder_roll",
    "left_shoulder_pitch",
    "right_shoulder_pitch",
    "left_elbow",
    "right_elbow",
    "left_hip_roll",
    "right_hip_roll",
    "left_hip_pitch",
    "right_hip_pitch",
    "left_hip_yaw",
    "right_hip_yaw",
    "left_knee",
    "right_knee",
    "left_ankle_roll",
    "right_ankle_roll",
    "left_ankle_pitch",
    "right_ankle_pitch",
)
# Left/right pairs that must be equal (pitch-like) or opposite (roll/yaw-like)
# so the HOME stays mirror symmetric (walking and get-up train with mirror
# symmetry augmentation).
MIRROR_EQUAL_JOINTS = ("shoulder_pitch", "elbow", "hip_pitch", "knee", "ankle_pitch")
MIRROR_OPPOSITE_JOINTS = ("shoulder_roll", "hip_roll", "hip_yaw", "ankle_roll")
# Single (midline) joints whose mirror image is their negative: head yaw and
# neck roll must be 0 for a mirror-symmetric HOME (neck pitch is free).
MIRROR_ZERO_JOINTS = ("head", "neck_roll")

TRUNK_BODY = "trunk"
# Virtual head point used by the get-up rewards (mdp._TRUNK_TO_HEAD_OFFSET).
TRUNK_TO_HEAD_OFFSET_M = 0.07324
SOLE_GEOMS = {
    side: tuple(f"{side}_foot_collision_{index}" for index in range(1, 7))
    for side in ("left", "right")
}
# The get-up feet_stance reward measures the foot BODY origins.
FOOT_BODIES = {"left": "foot_2", "right": "foot"}
# A sole corner belongs to the contact area when it lies within this distance
# of the lowest corner along the sole normal (box 1's local z axis).
SOLE_CONTACT_TOLERANCE_M = 1.0e-4
# The declared trunk pitch must make both soles flat to this tolerance.
FLAT_SOLE_TOLERANCE_RAD = 1.0e-9

# A HOME's soles must lie flat on the floor: every sole-face corner (the
# contact area measured along the sole normal) must be within
# SOLE_ON_FLOOR_TOLERANCE_M of the floor in WORLD height.  Otherwise a sole is
# rolled (or pitched) and stands on an edge, so the contact area the COM is
# centred over is not the one the robot stands on.  0.5 mm over the ~40 mm
# sole width is about 0.7 deg of roll: hip roll +-10 deg alone (sole roll 5 deg,
# 3.5 mm) is refused, the forward-lean HOME (0.08 deg) is accepted.
SOLE_ON_FLOOR_TOLERANCE_M = 5.0e-4

# Published (rounded) values.  Root z is rounded to 1e-12 m (1 pm): FK noise
# from another MuJoCo build or CPU is ~1e-16 m, so a rounding flip needs the
# FK value to sit within that of a 1e-12 boundary.  HOMEs published before
# this rule keep their exact historical value through LEGACY_HOME_OVERRIDES
# ("root_z_m", checked against FK to ROOT_Z_PIN_TOLERANCE_M).  The head height
# is the 0.1 mm value the get-up gates have always used.
ROOT_Z_DECIMALS = 12
ROOT_Z_PIN_TOLERANCE_M = 1.0e-12
HEAD_STANDING_HEIGHT_DECIMALS = 4
FEET_LATERAL_DECIMALS = 4

# Compatibility table: values published for one exact HOME before this file
# existed, keyed by its joint hash.  Two HOMEs have trained and deployed
# artifacts:
#
# * the centered HOME (trunk vertical; mjlab_microban track-centered-home-clip,
#   robot feature/neck-roll-pitch-camera a62a793), and
# * the forward-lean HOME (trunk +10 deg; mjlab_microban forward-lean-centered-
#   home / forward-lean-v2, robot forward-lean-home).
#
# Each keeps every value and string it was trained and deployed with, so its
# checkpoints, gates and ONNX stay valid and a checkout of this code with that
# HOME's YAML reproduces the branch it came from bit for bit:
#
# * ``tag``: the identifier used where a string embeds the HOME;
# * ``contracts``: every HOME-bound contract/recipe/revision string by key
#   (robot/home_contracts.py derives "<label>_<hash>" strings for any other
#   HOME);
# * ``root_z_m``: the published root z (checked against FK to
#   ROOT_Z_PIN_TOLERANCE_M);
# * ``feet_lateral_m``: the centered HOME's hand-rounded get-up feet target
#   (0.094 m; FK says 0.0935);
# * ``joint_pos_deg``: published full-precision joint values that differ from
#   the YAML's canonical 12-decimal ones in the last digits (the forward-lean
#   branch wrote the solver's unrounded hip/ankle pitch); checked against the
#   YAML to JOINT_PIN_TOLERANCE_DEG;
# * ``getup_near_home_reset``: the default near-HOME reset of the get-up tasks
#   on that branch (the centered line kept the wide (0.2, 0.6) default; the
#   forward-lean line, and any new HOME, use NEAR_HOME_RESET (0.1, 0.09));
# * ``accepts_unstamped_walk_checkpoints``: walking checkpoints saved before
#   the HOME stamp existed (centered line only);
# * ``v12_pose_release_switch_parent_sha256``: the pinned canonical model_7099
#   of the release-eligible pose-release recipe switch (centered line only;
#   every other HOME trains the pose-release recipe as a fresh chain).
#
# Any edit of the HOME changes the hash and drops all of this: strings then
# carry "<label>_<hash>", targets are the FK values, no unstamped or switched
# checkpoint is accepted.
JOINT_PIN_TOLERANCE_DEG = 1.0e-9

CENTERED_HOME_HASH = "bbef07cab8"
FORWARD_LEAN_HOME_HASH = "481503d292"

_CENTERED_HOME_CONTRACTS = {
    "walk_contract_version": "v3_centered_home_servo_range",
    "getup_contract_version": "v5",
    # Runs started on 2026-10-03 before the v5 bump stamped "v4"; accepted as
    # v5 only when their recorded env proves v5 (microban_getup_runner).
    "getup_legacy_stamp": "v4",
    "v12_home_pose_revision": (
        "centered_home_hip_plus1p198384259489_ankle_minus1p198384259489_shoulder_zero_v5"
    ),
    "v12_recipe_revision": (
        "centered_home_velocity_source_staged_mask_reachable_fk_elbow_minus10_"
        "raw_prev_action_servo_range_pi_v11"
    ),
    "v12_hand_pose_release_recipe_revision": (
        "centered_home_velocity_source_staged_mask_reachable_fk_elbow_minus10_"
        "raw_prev_action_servo_range_pi_active_hand_arm_pose_release_v12"
    ),
    "v12_packager_revision": (
        "microban_teleop_v12_final_deployment_packager_v6_centered_home_servo_range"
    ),
    "v12_final_rescue_recipe_revision": "model14900_targeted_final_scenario_replay_to15000_v1",
    "v12_final_rescue_marker_revision": (
        "recorded_model14900_ordinary10_final_scenarios90_99_updates_v1"
    ),
    "v12_final_rescue_sampler_revision": (
        "episode_shared_twist_foot_hand_evaluator_scenario_replay_v1"
    ),
    "v12_corner_rescue_recipe_revision": (
        "model9900_targeted_bilateral_corner_pair_replay_to10000_v3"
    ),
    "v12_corner_rescue_marker_revision": (
        "recorded_model9900_uniform5_lf_rb90_lb_rf5_99_updates_v3"
    ),
    "v12_corner_rescue_sampler_revision": "uniform_joint_box5pct_lf_rb90pct_lb_rf5pct_v2",
    "upright_fullbody_recipe_revision": "physical_neutral_full_actor_from_scratch_raw83x18_v4",
    "upright_fullbody_home_revision": "physical_neutral_shoulder_zero_hip_neg10_v5",
}

_FORWARD_LEAN_HOME_CONTRACTS = {
    "walk_contract_version": "v4_forward_lean_home_servo_range",
    "getup_contract_version": "v6",
    "v12_home_pose_revision": (
        "forward_lean10_hip_minus14p166561199931_ankle_plus4p127976841869_shoulder_zero_v6"
    ),
    "v12_recipe_revision": (
        "forward_lean_home_velocity_source_staged_mask_reachable_fk_elbow_minus10_"
        "raw_prev_action_servo_range_pi_home_levelled_targets_level_hmd_"
        "receiver_box_hands_v17"
    ),
    "v12_hand_pose_release_recipe_revision": (
        "forward_lean_home_velocity_source_staged_mask_reachable_fk_elbow_minus10_"
        "raw_prev_action_servo_range_pi_home_levelled_targets_level_hmd_"
        "receiver_box_hands_active_hand_arm_pose_release_v18"
    ),
    "v12_packager_revision": (
        "microban_teleop_v12_final_deployment_packager_v7_forward_lean_home_servo_range"
    ),
    "v12_final_rescue_recipe_revision": (
        "model14900_targeted_final_scenario_replay_to15000_forward_lean_"
        "home_levelled_receiver_box_v2"
    ),
    "v12_final_rescue_marker_revision": (
        "recorded_model14900_ordinary10_final_scenarios90_99_updates_forward_lean_v2"
    ),
    "v12_final_rescue_sampler_revision": (
        "episode_shared_twist_foot_hand_evaluator_scenario_replay_home_levelled_v2"
    ),
    "v12_corner_rescue_recipe_revision": (
        "model9900_targeted_bilateral_corner_pair_replay_to10000_receiver_box_f_v4"
    ),
    "v12_corner_rescue_marker_revision": (
        "recorded_model9900_uniform5_lf_rb90_lb_rf5_99_updates_receiver_box_f_v4"
    ),
    "v12_corner_rescue_sampler_revision": (
        "uniform_joint_box5pct_lf_rb90pct_lb_rf5pct_receiver_box_f_v3"
    ),
    "v12_pr_corner_rescue_lf60_marker_revision": (
        "recorded_pose_release_model9900_uniform5_lf_rb60_lb_rf35_99_updates_"
        "receiver_box_f_v1"
    ),
    "v12_pr_corner_rescue_lf60_sampler_revision": (
        "uniform_joint_box5pct_lf_rb60pct_lb_rf35pct_receiver_box_f_v1"
    ),
    "v12_pr_corner_rescue_lf65_marker_revision": (
        "recorded_pose_release_model9900_uniform5_lf_rb65_lb_rf30_99_updates_"
        "receiver_box_f_v1"
    ),
    "v12_pr_corner_rescue_lf65_sampler_revision": (
        "uniform_joint_box5pct_lf_rb65pct_lb_rf30pct_receiver_box_f_pose_release_v1"
    ),
    "v12_pr_corner_rescue_lf72_marker_revision": (
        "recorded_pose_release_model9900_uniform5_lf_rb72_lb_rf23_99_updates_"
        "receiver_box_f_v1"
    ),
    "v12_pr_corner_rescue_lf72_sampler_revision": (
        "uniform_joint_box5pct_lf_rb72pct_lb_rf23pct_receiver_box_f_pose_release_v1"
    ),
    "v12_pr_corner_rescue_lf90_marker_revision": (
        "recorded_pose_release_model9900_uniform5_lf_rb90_lb_rf5_99_updates_"
        "receiver_box_f_v1"
    ),
    "v12_pr_corner_rescue_lf90_sampler_revision": (
        "uniform_joint_box5pct_lf_rb90pct_lb_rf5pct_receiver_box_f_pose_release_v1"
    ),
    "v12_pr_final_rescue_marker_revision": (
        "recorded_pose_release_model14900_failed_final_scenarios_replay_"
        "99_updates_forward_lean_v1"
    ),
    "v12_pr_final_rescue_sampler_revision": (
        "episode_shared_twist_foot_hand_failed_scenario_replay_home_levelled_"
        "pose_release_v1"
    ),
    "upright_fullbody_recipe_revision": (
        "physical_neutral_full_actor_from_scratch_raw83x18_home_levelled_targets_"
        "receiver_box_hands_v6"
    ),
    "upright_fullbody_home_revision": "forward_lean10_com_centered_shoulder_zero_v6",
}

LEGACY_HOME_OVERRIDES: Mapping[str, Mapping[str, object]] = MappingProxyType(
    {
        # The centered HOME's FK root z is 0.17055488563355944, two ulps below
        # the 15-decimal rounding boundary, so it is pinned (not re-rounded).
        CENTERED_HOME_HASH: MappingProxyType(
            {
                "tag": "centered_home",
                "feet_lateral_m": 0.094,
                "root_z_m": 0.170554885633559,
                "contracts": MappingProxyType(_CENTERED_HOME_CONTRACTS),
                "getup_near_home_reset": (0.2, 0.6),
                "accepts_unstamped_walk_checkpoints": True,
                # The centered line judged its 10000 boundary / 10100 canary
                # with hand RMS <= 0.035 m (track-centered-home-clip 5b5a9d0);
                # the 0.040 m allowance of forward-lean-v2 e3271de / ec67f1e
                # does not exist at this HOME.
                "v12_hand_rms_40mm_boundary_profiles": False,
                "v12_pose_release_switch_parent_sha256": (
                    # The gated canonical centered-HOME v12 model_7099 (run
                    # 2026-10-04_10-18-35_c20k_v12_7000_to7100, source walking
                    # model_20000).
                    "366763f233f30947554c2f66c5619a8b8c1230ac8529f17740fb1d27cfe0d224"
                ),
            }
        ),
        # The forward-lean HOME (trunk +10 deg) as published on the
        # forward-lean branches: root z 0.170430569776402 and the solver's
        # unrounded hip/ankle pitch (the YAML holds the canonical 12-decimal
        # values -14.166561199931 / 4.127976841869, 1.2e-13 deg away).
        FORWARD_LEAN_HOME_HASH: MappingProxyType(
            {
                "tag": "forward_lean_home",
                "root_z_m": 0.170430569776402,
                "joint_pos_deg": MappingProxyType(
                    {
                        "left_hip_pitch": -14.166561199931119,
                        "right_hip_pitch": -14.166561199931119,
                        "left_ankle_pitch": 4.127976841869204,
                        "right_ankle_pitch": 4.127976841869204,
                    }
                ),
                "contracts": MappingProxyType(_FORWARD_LEAN_HOME_CONTRACTS),
            }
        ),
    }
)

# Explicit HOME YAML for this process (tests, side-by-side comparisons, the
# training-line check): when set, ``HOME`` is loaded from this path instead of
# config/home_pose.yaml.
HOME_POSE_YAML_ENV = "MJLAB_MICROBAN_HOME_POSE_YAML"


# --------------------------------------------------------------------------
# Pure FK analysis


@dataclass(frozen=True)
class PoseAnalysis:
    """Everything FK says about one standing pose (all lengths in metres).

    Positions are world-frame with the root at x = y = 0 and the lowest sole
    collision corner at z = 0.  ``x`` is forward.
    """

    trunk_pitch_rad: float
    """Root pitch the pose was evaluated at (positive leans forward)."""
    flat_sole_trunk_pitch_rad: float
    """Root pitch that makes the soles level (mean of both feet)."""
    sole_pitch_rad: tuple[float, float]
    """(left, right) sole pitch in the world at ``trunk_pitch_rad``; 0 = flat."""
    root_pos: tuple[float, float, float]
    root_quat_wxyz: tuple[float, float, float, float]
    projected_gravity: tuple[float, float, float]
    total_mass_kg: float
    com: tuple[float, float, float]
    sole_x_min: float
    """Heel: rearmost x of the sole contact corners (both feet)."""
    sole_x_max: float
    """Toe: foremost x of the sole contact corners (both feet)."""
    sole_contact_corner_count: int
    head_height_m: float
    feet_lateral_m: float
    """|y(left foot body) - y(right foot body)| (``foot_2`` / ``foot``)."""
    sole_roll_rad: tuple[float, float] = (0.0, 0.0)
    """(left, right) sole roll in the world (side tilt; 0 = level)."""
    ground_contact_corner_count: int = 0
    """Sole corners within SOLE_CONTACT_TOLERANCE_M of the lowest corner in
    WORLD height (the corners that actually touch a flat floor)."""
    ground_x_min: float = 0.0
    ground_x_max: float = 0.0
    sole_contact_lift_m: float = 0.0
    """Highest sole-face corner above the floor (0 when the soles are flat)."""

    @property
    def soles_on_floor(self) -> bool:
        """True when the whole sole contact area lies on the floor.

        The contact area (``sole_x_min``/``sole_x_max``) is measured along each
        sole's own normal; it is the real ground contact only when no sole is
        rolled or pitched, i.e. when every one of its corners is within
        SOLE_ON_FLOOR_TOLERANCE_M of the floor.
        """

        return self.sole_contact_lift_m <= SOLE_ON_FLOOR_TOLERANCE_M

    @property
    def sole_center_x(self) -> float:
        return 0.5 * (self.sole_x_min + self.sole_x_max)

    @property
    def com_offset_x(self) -> float:
        """COM x minus the fore-aft centre of the sole contact area."""

        return self.com[0] - self.sole_center_x

    @property
    def heel_margin_m(self) -> float:
        return self.com[0] - self.sole_x_min

    @property
    def toe_margin_m(self) -> float:
        return self.sole_x_max - self.com[0]

    def summary(self) -> dict[str, object]:
        return {
            "trunk_pitch_deg": math.degrees(self.trunk_pitch_rad),
            "flat_sole_trunk_pitch_deg": math.degrees(self.flat_sole_trunk_pitch_rad),
            "sole_pitch_deg": [math.degrees(value) for value in self.sole_pitch_rad],
            "root_pos_m": list(self.root_pos),
            "root_quat_wxyz": list(self.root_quat_wxyz),
            "projected_gravity": list(self.projected_gravity),
            "total_mass_kg": self.total_mass_kg,
            "com_m": list(self.com),
            "sole_x_range_m": [self.sole_x_min, self.sole_x_max],
            "sole_center_x_m": self.sole_center_x,
            "com_offset_x_m": self.com_offset_x,
            "heel_margin_m": self.heel_margin_m,
            "toe_margin_m": self.toe_margin_m,
            "sole_contact_corner_count": self.sole_contact_corner_count,
            "sole_roll_deg": [math.degrees(value) for value in self.sole_roll_rad],
            "ground_contact_corner_count": self.ground_contact_corner_count,
            "ground_x_range_m": [self.ground_x_min, self.ground_x_max],
            "sole_contact_lift_m": self.sole_contact_lift_m,
            "soles_on_floor": self.soles_on_floor,
            "head_height_m": self.head_height_m,
            "feet_lateral_m": self.feet_lateral_m,
        }


def root_quat_from_trunk_pitch(trunk_pitch_rad: float) -> tuple[float, float, float, float]:
    """Root quaternion (w, x, y, z) of a trunk pitched forward about +y."""

    half = 0.5 * float(trunk_pitch_rad)
    return (float(np.cos(half)), 0.0, float(np.sin(half)), 0.0)


def projected_gravity_from_trunk_pitch(trunk_pitch_rad: float) -> tuple[float, float, float]:
    """Unit gravity in the trunk frame when the trunk is pitched forward."""

    pitch = float(trunk_pitch_rad)
    return (float(np.sin(pitch)), 0.0, float(-np.cos(pitch)))


class _PoseModel:
    """Cached MuJoCo model + data with precomputed ids (not thread safe)."""

    def __init__(self, xml_path: Path) -> None:
        self.xml_path = Path(xml_path)
        self.model = mujoco.MjSpec.from_file(str(self.xml_path)).compile()
        self.data = mujoco.MjData(self.model)
        model = self.model
        self.free_qposadr = None
        for joint_id in range(model.njnt):
            if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                self.free_qposadr = int(model.jnt_qposadr[joint_id])
                break
        if self.free_qposadr is None:
            raise ValueError(f"{xml_path} has no free joint")
        self.joint_qposadr = {
            model.joint(joint_id).name: int(model.jnt_qposadr[joint_id])
            for joint_id in range(model.njnt)
            if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE
        }
        missing = set(HOME_JOINT_NAMES) - set(self.joint_qposadr)
        if missing:
            raise ValueError(f"{xml_path} lacks HOME joints {sorted(missing)}")
        self.joint_range = {
            name: tuple(float(v) for v in model.jnt_range[model.joint(name).id])
            for name in HOME_JOINT_NAMES
        }
        self.joint_limited = {
            name: bool(model.jnt_limited[model.joint(name).id]) for name in HOME_JOINT_NAMES
        }
        self.trunk_id = model.body(TRUNK_BODY).id
        if model.body_parentid[self.trunk_id] != 0:
            raise ValueError("The trunk must be the floating root body")
        self.sole_geom_ids = {
            side: np.array([model.geom(name).id for name in names])
            for side, names in SOLE_GEOMS.items()
        }
        self.foot_body_ids = {side: model.body(name).id for side, name in FOOT_BODIES.items()}
        signs = np.array(
            [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
            dtype=np.float64,
        )
        self.corner_signs = signs
        self.total_mass = float(model.body_subtreemass[self.trunk_id])

    def set_pose(self, joint_pos_rad: Mapping[str, float], trunk_pitch_rad: float) -> None:
        data = self.data
        data.qpos[:] = 0.0
        adr = self.free_qposadr
        data.qpos[adr + 3 : adr + 7] = root_quat_from_trunk_pitch(trunk_pitch_rad)
        for name, value in joint_pos_rad.items():
            data.qpos[self.joint_qposadr[name]] = value
        mujoco.mj_kinematics(self.model, data)
        mujoco.mj_comPos(self.model, data)

    def sole_corners(self, side: str) -> np.ndarray:
        """All 8 corners of the side's six sole boxes, shape (48, 3)."""

        ids = self.sole_geom_ids[side]
        centres = self.data.geom_xpos[ids]  # (6, 3)
        rotations = self.data.geom_xmat[ids].reshape(-1, 3, 3)  # (6, 3, 3)
        local = self.corner_signs[None, :, :] * self.model.geom_size[ids][:, None, :]
        return (centres[:, None, :] + np.einsum("gij,gcj->gci", rotations, local)).reshape(-1, 3)

    def sole_normal(self, side: str) -> np.ndarray:
        return self.data.geom_xmat[self.sole_geom_ids[side][0]].reshape(3, 3)[:, 2].copy()


@lru_cache(maxsize=4)
def _pose_model(xml_path: str) -> _PoseModel:
    return _PoseModel(Path(xml_path))


def _sole_pitch(normal: np.ndarray) -> float:
    """World pitch of a sole whose normal is ``normal`` (0 when level)."""

    return float(math.atan2(normal[0], normal[2]))


def _sole_roll(normal: np.ndarray) -> float:
    """World roll (side tilt) of a sole whose normal is ``normal`` (0 when level)."""

    return float(math.atan2(-normal[1], normal[2]))


def analyze_pose(
    joint_pos_rad: Mapping[str, float],
    trunk_pitch_rad: float | None = None,
    *,
    xml_path: Path | str = MICROBAN_XML,
) -> PoseAnalysis:
    """Return the FK standing analysis of a joint pose.

    ``joint_pos_rad`` maps joint names to radians; unnamed joints are 0.
    ``trunk_pitch_rad`` is the root pitch to evaluate at; ``None`` uses the
    pitch that levels the soles.  The root is put at x = y = 0 and at the
    height where the lowest sole collision corner touches z = 0.
    """

    pose_model = _pose_model(str(Path(xml_path).resolve()))
    unknown = set(joint_pos_rad) - set(pose_model.joint_qposadr)
    if unknown:
        raise KeyError(f"Unknown Microban joints {sorted(unknown)}")
    # Rotating the root about world y adds the same angle to every sole's world
    # pitch, so the level-sole root pitch follows from the pose at pitch 0.
    pose_model.set_pose(joint_pos_rad, 0.0)
    pitch_at_zero = [
        _sole_pitch(pose_model.sole_normal(side)) for side in ("left", "right")
    ]
    flat_pitch = float(-0.5 * (pitch_at_zero[0] + pitch_at_zero[1]))
    pitch = flat_pitch if trunk_pitch_rad is None else float(trunk_pitch_rad)
    if pitch != 0.0:
        pose_model.set_pose(joint_pos_rad, pitch)
    data = pose_model.data

    normals = {side: pose_model.sole_normal(side) for side in ("left", "right")}
    corners = {side: pose_model.sole_corners(side) for side in ("left", "right")}
    all_corners = np.concatenate([corners["left"], corners["right"]])
    root_z = float(-all_corners[:, 2].min())

    contact = []
    for side in ("left", "right"):
        heights = corners[side] @ normals[side]
        contact.append(corners[side][heights <= heights.min() + SOLE_CONTACT_TOLERANCE_M])
    contact_corners = np.concatenate(contact)
    # The reference definition (what a flat floor touches): world height.
    ground_corners = all_corners[all_corners[:, 2] <= all_corners[:, 2].min() + SOLE_CONTACT_TOLERANCE_M]

    com = data.subtree_com[pose_model.trunk_id].copy()
    com[2] += root_z
    trunk_ipos = data.xipos[pose_model.trunk_id]
    trunk_iup_z = data.ximat[pose_model.trunk_id][8]
    head = float(trunk_ipos[2] + root_z + TRUNK_TO_HEAD_OFFSET_M * trunk_iup_z)
    left_foot = data.xpos[pose_model.foot_body_ids["left"]]
    right_foot = data.xpos[pose_model.foot_body_ids["right"]]

    return PoseAnalysis(
        trunk_pitch_rad=pitch,
        flat_sole_trunk_pitch_rad=flat_pitch,
        sole_pitch_rad=(_sole_pitch(normals["left"]), _sole_pitch(normals["right"])),
        root_pos=(0.0, 0.0, root_z),
        root_quat_wxyz=root_quat_from_trunk_pitch(pitch),
        projected_gravity=projected_gravity_from_trunk_pitch(pitch),
        total_mass_kg=pose_model.total_mass,
        com=(float(com[0]), float(com[1]), float(com[2])),
        sole_x_min=float(contact_corners[:, 0].min()),
        sole_x_max=float(contact_corners[:, 0].max()),
        sole_contact_corner_count=int(len(contact_corners)),
        head_height_m=head,
        feet_lateral_m=float(abs(left_foot[1] - right_foot[1])),
        sole_roll_rad=(_sole_roll(normals["left"]), _sole_roll(normals["right"])),
        ground_contact_corner_count=int(len(ground_corners)),
        ground_x_min=float(ground_corners[:, 0].min()),
        ground_x_max=float(ground_corners[:, 0].max()),
        sole_contact_lift_m=float(contact_corners[:, 2].max() - all_corners[:, 2].min()),
    )


def joint_limits_rad(xml_path: Path | str = MICROBAN_XML) -> dict[str, tuple[float, float]]:
    """Return the MJCF range of every limited HOME joint."""

    pose_model = _pose_model(str(Path(xml_path).resolve()))
    return {
        name: pose_model.joint_range[name]
        for name in HOME_JOINT_NAMES
        if pose_model.joint_limited[name]
    }


# --------------------------------------------------------------------------
# HOME identity


def normalized_float(value: float) -> float:
    """``float(value)`` with -0.0 mapped to 0.0 (one pose, one identity)."""

    return float(value) + 0.0


def canonical_home_string(joint_pos_deg: Mapping[str, float], trunk_pitch_deg: float) -> str:
    """Canonical text of the HOME inputs (what the identity hash covers).

    -0.0 and 0.0 are the same angle and give the same text.
    """

    parts = [
        f"{name}={normalized_float(joint_pos_deg[name])!r}" for name in sorted(HOME_JOINT_NAMES)
    ]
    parts.append(f"trunk_pitch_deg={normalized_float(trunk_pitch_deg)!r}")
    return ";".join(parts)


def yaml_float_text(value: float) -> str:
    """Text for a finite float that YAML 1.1 (PyYAML) reads back as that float.

    ``repr`` round-trips, but PyYAML only takes ``1e-05`` as a float with a
    dot in the mantissa (``1.0e-05``); -0.0 is written as 0.0.
    """

    value = normalized_float(value)
    if not math.isfinite(value):
        raise ValueError(f"HOME values must be finite, got {value!r}")
    mantissa, marker, exponent = repr(value).partition("e")
    if "." not in mantissa:
        mantissa += ".0"
    text = mantissa + marker + exponent
    if yaml.safe_load(text) != value:  # pragma: no cover - guards the rule above
        raise AssertionError(f"{text!r} does not read back as {value!r}")
    return text


def home_joint_hash(joint_pos_deg: Mapping[str, float], trunk_pitch_deg: float) -> str:
    """10 hex digits of SHA-256 over the canonical HOME inputs."""

    text = canonical_home_string(joint_pos_deg, trunk_pitch_deg)
    return hashlib.sha256(text.encode("ascii")).hexdigest()[:10]


def signed_degree_token(value_deg: float) -> str:
    """``+1.198384259489`` -> ``plus1p198384259489`` (identifier-safe)."""

    value = float(value_deg)
    sign = "minus" if math.copysign(1.0, value) < 0 else "plus"
    return sign + repr(abs(value)).replace(".", "p").replace("-", "m").replace("+", "")


_LABEL_RE = re.compile(r"^[a-z][a-z0-9_]*$")


# --------------------------------------------------------------------------
# The HOME


@dataclass(frozen=True)
class HomePose:
    """The loaded HOME: the YAML inputs plus every FK-derived value."""

    path: Path
    name: str
    label: str
    trunk_pitch_deg: float
    joint_pos_deg: Mapping[str, float]
    joint_pos_rad: Mapping[str, float]
    analysis: PoseAnalysis
    joint_hash: str
    overrides: Mapping[str, object] = field(default_factory=dict)
    input_joint_pos_deg: Mapping[str, float] | None = None
    """The YAML's joint values (canonical; what ``joint_hash`` covers).  They
    equal ``joint_pos_deg`` except where a legacy HOME pins published
    full-precision values (LEGACY_HOME_OVERRIDES ``joint_pos_deg``)."""

    # -- inputs in radians ---------------------------------------------
    @property
    def trunk_pitch_rad(self) -> float:
        return float(np.deg2rad(self.trunk_pitch_deg))

    # -- root pose -------------------------------------------------------
    @property
    def root_pos(self) -> tuple[float, float, float]:
        if "root_z_m" in self.overrides:
            return (0.0, 0.0, float(self.overrides["root_z_m"]))  # type: ignore[arg-type]
        return (0.0, 0.0, round(self.analysis.root_pos[2], ROOT_Z_DECIMALS))

    @property
    def root_quat_wxyz(self) -> tuple[float, float, float, float]:
        return root_quat_from_trunk_pitch(self.trunk_pitch_rad)

    @property
    def projected_gravity(self) -> tuple[float, float, float]:
        """Unit gravity in the trunk frame while standing at HOME."""

        return projected_gravity_from_trunk_pitch(self.trunk_pitch_rad)

    # -- get-up targets --------------------------------------------------
    @property
    def head_standing_height_m(self) -> float:
        return round(self.analysis.head_height_m, HEAD_STANDING_HEIGHT_DECIMALS)

    @property
    def feet_lateral_m(self) -> float:
        if "feet_lateral_m" in self.overrides:
            return float(self.overrides["feet_lateral_m"])  # type: ignore[arg-type]
        return round(self.analysis.feet_lateral_m, FEET_LATERAL_DECIMALS)

    # -- identity --------------------------------------------------------
    @property
    def tag(self) -> str:
        """Identifier embedded in every HOME-bound contract string."""

        if "tag" in self.overrides:
            return str(self.overrides["tag"])
        return f"{self.label}_{self.joint_hash}"

    @property
    def is_legacy(self) -> bool:
        """True for a HOME with published artifacts (LEGACY_HOME_OVERRIDES)."""

        return "contracts" in self.overrides

    @property
    def trunk_is_vertical(self) -> bool:
        return self.trunk_pitch_deg == 0.0

    def contract(self, key: str, derived: str) -> str:
        """A HOME-bound string: the published one of a legacy HOME, else ``derived``."""

        contracts = self.overrides.get("contracts")
        if isinstance(contracts, Mapping) and key in contracts:
            return str(contracts[key])
        return derived

    def override(self, key: str, default: object = None) -> object:
        return self.overrides.get(key, default)

    @property
    def hip_pitch_deg(self) -> float:
        return float(self.joint_pos_deg["left_hip_pitch"])

    @property
    def ankle_pitch_deg(self) -> float:
        return float(self.joint_pos_deg["left_ankle_pitch"])

    def arm_joint_deg(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """((left), (right)) arm HOME as (shoulder_pitch, shoulder_roll, elbow) degrees."""

        return tuple(  # type: ignore[return-value]
            tuple(
                float(self.joint_pos_deg[f"{side}_{joint}"])
                for joint in ("shoulder_pitch", "shoulder_roll", "elbow")
            )
            for side in ("left", "right")
        )

    def summary(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "name": self.name,
            "label": self.label,
            "joint_hash": self.joint_hash,
            "tag": self.tag,
            "trunk_pitch_deg": self.trunk_pitch_deg,
            "joint_pos_deg": dict(self.joint_pos_deg),
            "root_pos_m": list(self.root_pos),
            "root_quat_wxyz": list(self.root_quat_wxyz),
            "projected_gravity": list(self.projected_gravity),
            "head_standing_height_m": self.head_standing_height_m,
            "feet_lateral_m": self.feet_lateral_m,
            "analysis": self.analysis.summary(),
        }


def validate_home_inputs(joint_pos_deg: Mapping[str, float], trunk_pitch_deg: float) -> None:
    """Raise ValueError unless the inputs are a complete, symmetric, in-range HOME."""

    names = set(joint_pos_deg)
    if names != set(HOME_JOINT_NAMES):
        missing = sorted(set(HOME_JOINT_NAMES) - names)
        extra = sorted(names - set(HOME_JOINT_NAMES))
        raise ValueError(f"HOME must name all 21 joints (missing {missing}, unknown {extra})")
    for name, value in joint_pos_deg.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"HOME joint {name} must be a number of degrees, got {value!r}")
        if not math.isfinite(float(value)):
            raise ValueError(f"HOME joint {name} must be finite")
    if isinstance(trunk_pitch_deg, bool) or not isinstance(trunk_pitch_deg, (int, float)):
        raise ValueError("trunk_pitch_deg must be a number")
    if not math.isfinite(float(trunk_pitch_deg)):
        raise ValueError("trunk_pitch_deg must be finite")
    for joint in MIRROR_EQUAL_JOINTS:
        left, right = joint_pos_deg[f"left_{joint}"], joint_pos_deg[f"right_{joint}"]
        if float(left) != float(right):
            raise ValueError(f"HOME is not mirror symmetric: left/right_{joint} {left} != {right}")
    for joint in MIRROR_OPPOSITE_JOINTS:
        left, right = joint_pos_deg[f"left_{joint}"], joint_pos_deg[f"right_{joint}"]
        if float(left) != -float(right):
            raise ValueError(
                f"HOME is not mirror symmetric: left_{joint} {left} != -right_{joint} {right}"
            )
    for joint in MIRROR_ZERO_JOINTS:
        if float(joint_pos_deg[joint]) != 0.0:
            raise ValueError(
                f"HOME is not mirror symmetric: {joint} = {joint_pos_deg[joint]} deg turns or "
                "tilts the head to one side (it must be 0)"
            )
    for name, (lower, upper) in joint_limits_rad().items():
        value = float(np.deg2rad(float(joint_pos_deg[name])))
        if not lower <= value <= upper:
            raise ValueError(
                f"HOME joint {name} = {joint_pos_deg[name]} deg is outside its MJCF range "
                f"[{math.degrees(lower)!r}, {math.degrees(upper)!r}] deg "
                f"({'below the lower' if value < lower else 'above the upper'} limit by "
                f"{abs(math.degrees(value - (lower if value < lower else upper))):.3e} deg)"
            )


def home_pose_from_values(
    *,
    joint_pos_deg: Mapping[str, float],
    trunk_pitch_deg: float,
    name: str = "",
    label: str = "home",
    path: Path | str = "<memory>",
    require_flat_soles: bool = True,
) -> HomePose:
    """Build a HomePose from in-memory inputs (same rules as the YAML)."""

    if not isinstance(label, str) or not _LABEL_RE.match(label):
        raise ValueError(f"HOME label must match {_LABEL_RE.pattern}, got {label!r}")
    validate_home_inputs(joint_pos_deg, trunk_pitch_deg)
    degrees = {name_: normalized_float(joint_pos_deg[name_]) for name_ in HOME_JOINT_NAMES}
    radians = {name_: float(np.deg2rad(value)) for name_, value in degrees.items()}
    pitch_deg = normalized_float(trunk_pitch_deg)
    pitch_rad = float(np.deg2rad(pitch_deg))
    analysis = analyze_pose(radians, pitch_rad)
    if require_flat_soles:
        error = abs(analysis.flat_sole_trunk_pitch_rad - pitch_rad)
        if error > FLAT_SOLE_TOLERANCE_RAD:
            raise ValueError(
                "HOME soles are not flat at the declared trunk pitch "
                f"{pitch_deg!r} deg: the soles are level at "
                f"{math.degrees(analysis.flat_sole_trunk_pitch_rad)!r} deg. Edit "
                "trunk_pitch_deg or rebalance the hip/ankle pitches with "
                "config/balance_home_pose.py."
            )
        if not analysis.soles_on_floor:
            roll = " / ".join(f"{math.degrees(v):+.3f}" for v in analysis.sole_roll_rad)
            raise ValueError(
                "HOME soles are not flat on the floor: sole roll L / R "
                f"{roll} deg lifts a sole edge {analysis.sole_contact_lift_m * 1e3:.2f} mm off the "
                f"floor (limit {SOLE_ON_FLOOR_TOLERANCE_M * 1e3:.1f} mm; "
                f"{analysis.ground_contact_corner_count} of {analysis.sole_contact_corner_count} "
                "sole corners touch it), so the robot would stand on an edge, not on the sole "
                "the COM is centred over. Level the soles in roll -- usually "
                "ankle_roll = -hip_roll and hip_yaw = 0 -- before balancing; "
                "config/balance_home_pose.py moves only hip/ankle pitch."
            )
    joint_hash = home_joint_hash(degrees, pitch_deg)
    overrides = LEGACY_HOME_OVERRIDES.get(joint_hash, MappingProxyType({}))
    if "root_z_m" in overrides:
        drift = abs(float(overrides["root_z_m"]) - analysis.root_pos[2])  # type: ignore[arg-type]
        if drift > ROOT_Z_PIN_TOLERANCE_M:
            raise ValueError(
                f"HOME {joint_hash}: FK root z {analysis.root_pos[2]!r} differs from its published "
                f"value {overrides['root_z_m']!r} by {drift:.3e} m (robot.xml changed?)"
            )
    published = dict(degrees)
    pins = overrides.get("joint_pos_deg")
    if isinstance(pins, Mapping):
        for joint, value in pins.items():
            if abs(float(value) - degrees[joint]) > JOINT_PIN_TOLERANCE_DEG:
                raise ValueError(  # pragma: no cover - a hash collision
                    f"HOME {joint_hash}: {joint} {degrees[joint]!r} deg is not its published "
                    f"value {value!r}"
                )
            published[joint] = float(value)
    published_rad = {name_: float(np.deg2rad(value)) for name_, value in published.items()}
    return HomePose(
        path=Path(path),
        name=str(name),
        label=label,
        trunk_pitch_deg=pitch_deg,
        joint_pos_deg=MappingProxyType(published),
        joint_pos_rad=MappingProxyType(published_rad),
        analysis=analysis,
        joint_hash=joint_hash,
        overrides=overrides,
        input_joint_pos_deg=MappingProxyType(degrees),
    )


def read_home_pose_yaml(path: Path | str = HOME_POSE_YAML) -> dict[str, object]:
    """Parse and schema-check the HOME YAML (no FK)."""

    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise ValueError(f"{path}: HOME YAML must be a mapping")
    expected = {"schema_version", "name", "label", "trunk_pitch_deg", "joint_pos_deg"}
    if set(document) != expected:
        raise ValueError(f"{path}: HOME YAML keys must be exactly {sorted(expected)}")
    if document["schema_version"] != HOME_POSE_SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported schema_version {document['schema_version']!r}")
    if not isinstance(document["joint_pos_deg"], dict):
        raise ValueError(f"{path}: joint_pos_deg must be a mapping")
    return document


def describe_home_yaml_error(error: BaseException, path: Path | str) -> str:
    """One-line reason for a HOME YAML that cannot be read or loaded."""

    path = Path(path)
    if isinstance(error, FileNotFoundError):
        return "no such file"
    if isinstance(error, IsADirectoryError):
        return "is a directory, not a YAML file"
    if isinstance(error, UnicodeDecodeError):
        return f"not UTF-8 text ({error.reason})"
    if isinstance(error, OSError):
        return f"cannot read: {error.strerror or error}"
    if isinstance(error, yaml.YAMLError):
        mark = getattr(error, "problem_mark", None)
        problem = getattr(error, "problem", None) or " ".join(str(error).split())
        where = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        return f"invalid YAML{where}: {problem}"
    return " ".join(str(error).split()).removeprefix(f"{path}: ")


def load_home_pose(path: Path | str = HOME_POSE_YAML) -> HomePose:
    """Load the HOME YAML and compute every derived value."""

    document = read_home_pose_yaml(path)
    return home_pose_from_values(
        joint_pos_deg=document["joint_pos_deg"],  # type: ignore[arg-type]
        trunk_pitch_deg=document["trunk_pitch_deg"],  # type: ignore[arg-type]
        name=document["name"],  # type: ignore[arg-type]
        label=document["label"],  # type: ignore[arg-type]
        path=Path(path),
    )


_YAML_VALUE_LINE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<quote>[\"']?)(?P<key>[A-Za-z_][A-Za-z0-9_]*)(?P=quote):"
    r"(?P<gap>[ \t]+)(?P<value>[^#\s][^#]*?)(?P<tail>[ \t]*(?:#.*)?)$"
)


def rewrite_home_pose_yaml(
    path: Path | str,
    *,
    joint_pos_deg: Mapping[str, float] | None = None,
    trunk_pitch_deg: float | None = None,
) -> str:
    """Rewrite HOME values in place, keeping every comment and line.

    Only the value text of the named keys changes (``yaml_float_text``: the
    shortest repr with a guaranteed dot, so every finite value round-trips;
    -0.0 is written as 0.0); quoted keys and line endings are kept.  Returns
    the new file text.  The result is re-parsed and must contain exactly the
    requested values.
    """

    path = Path(path)
    updates = {name: normalized_float(value) for name, value in (joint_pos_deg or {}).items()}
    if trunk_pitch_deg is not None:
        trunk_pitch_deg = normalized_float(trunk_pitch_deg)
    unknown = set(updates) - set(HOME_JOINT_NAMES)
    if unknown:
        raise KeyError(f"Unknown HOME joints {sorted(unknown)}")
    # Bytes in, bytes out: keeps CRLF (or any) line endings exactly.
    lines = path.read_bytes().decode("utf-8").splitlines(keepends=True)
    seen: set[str] = set()
    in_joints = False
    out = []
    for line in lines:
        body = line.rstrip("\r\n")
        newline = line[len(body) :]
        match = _YAML_VALUE_LINE.match(body)
        if body and not body[0].isspace() and not body.startswith("#"):
            in_joints = body.split(":", 1)[0].strip("\"'") == "joint_pos_deg"
        if match is not None:
            key = match["key"]
            indented = bool(match["indent"])
            quoted = f"{match['quote']}{key}{match['quote']}"
            if in_joints and indented and key in updates:
                value_text = yaml_float_text(updates[key])
                body = f"{match['indent']}{quoted}:{match['gap']}{value_text}{match['tail']}"
                seen.add(key)
            elif not indented and key == "trunk_pitch_deg" and trunk_pitch_deg is not None:
                body = f"{quoted}:{match['gap']}{yaml_float_text(trunk_pitch_deg)}{match['tail']}"
                seen.add(key)
        out.append(body + newline)
    wanted = set(updates) | ({"trunk_pitch_deg"} if trunk_pitch_deg is not None else set())
    if seen != wanted:
        raise ValueError(f"{path}: could not find HOME lines for {sorted(wanted - seen)}")
    text = "".join(out)
    document = yaml.safe_load(text)
    def read_back(value: object) -> float | None:
        return float(value) if isinstance(value, float) else None

    for name, value in updates.items():
        if read_back(document["joint_pos_deg"][name]) != value:
            raise AssertionError(f"rewrite of {name} did not round-trip")
    if trunk_pitch_deg is not None and read_back(document["trunk_pitch_deg"]) != trunk_pitch_deg:
        raise AssertionError("rewrite of trunk_pitch_deg did not round-trip")
    path.write_bytes(text.encode("utf-8"))
    return text


def __getattr__(name: str) -> object:
    """Load ``HOME`` (the HOME of this checkout) on first access.

    ``from mjlab_microban.robot.home_pose import HOME`` loads and checks
    ``config/home_pose.yaml`` (or the file named by ``$MJLAB_MICROBAN_HOME_POSE_YAML``)
    once and caches the result.  Loading lazily
    keeps this module importable while the YAML is being edited (for example
    a knee change that ``config/balance_home_pose.py`` has not re-levelled
    yet); only code that actually needs the HOME is refused.
    """

    if name == "HOME":
        import os

        explicit = os.environ.get(HOME_POSE_YAML_ENV)
        home = load_home_pose(explicit) if explicit else load_home_pose()
        globals()["HOME"] = home
        return home
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
