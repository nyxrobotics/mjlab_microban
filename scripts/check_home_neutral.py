"""Report the MuJoCo center of mass and sole geometry at the configured HOME.

Run with ``.venv/bin/python scripts/check_home_neutral.py`` to inspect HOME_FRAME.
Pass ``--solve`` to reproduce its hip/ankle angle and root height from MuJoCo
geometry. The script reports values; it never edits the configured pose.
"""

from __future__ import annotations

import argparse

import mujoco
import numpy as np

import mjlab_microban.tasks  # register tasks before importing the robot config
from mjlab_microban.robot.microban_constants import HOME_FRAME, MICROBAN_XML
from mjlab_microban.tasks.mdp import _TRUNK_TO_HEAD_OFFSET


FOOT_BOX_NAMES = tuple(
    f"{side}_foot_collision_{index}"
    for side in ("left", "right")
    for index in range(1, 7)
)
SUPPORT_GEOM_NAMES = ("left_foot_collision_1", "right_foot_collision_1")


def _sole_geom_ids(model: mujoco.MjModel) -> tuple[int, ...]:
    foot_box_ids = tuple(model.geom(name).id for name in FOOT_BOX_NAMES)
    if any(
        model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_BOX
        for geom_id in foot_box_ids
    ):
        raise ValueError("Expected every foot_collision_[1-6] geometry to be a box")
    return foot_box_ids


def _measure_geometry(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[float, float, float]:
    foot_box_ids = _sole_geom_ids(model)
    support_ids = tuple(model.geom(name).id for name in SUPPORT_GEOM_NAMES)

    total_mass = float(model.body_mass.sum())
    if total_mass <= 0.0:
        raise ValueError("MuJoCo model has no positive body mass")
    com_x = float(np.dot(data.xipos[:, 0], model.body_mass) / total_mass)
    support_x = float(np.mean(data.geom_xpos[list(support_ids), 0]))

    # collision_1 is the widest of the six nested sole boxes on each foot.
    # Confirm its midpoint also describes the full twelve-box footprint.
    min_x = min(
        data.geom_xpos[geom_id, 0]
        - np.dot(
            np.abs(data.geom_xmat[geom_id].reshape(3, 3)[0]),
            model.geom_size[geom_id],
        )
        for geom_id in foot_box_ids
    )
    max_x = max(
        data.geom_xpos[geom_id, 0]
        + np.dot(
            np.abs(data.geom_xmat[geom_id].reshape(3, 3)[0]),
            model.geom_size[geom_id],
        )
        for geom_id in foot_box_ids
    )
    if not np.isclose(support_x, (min_x + max_x) / 2.0, rtol=0.0, atol=1.0e-9):
        raise ValueError("Support geom centers no longer match sole footprint midpoint")

    # A rotated box reaches z = center_z - sum(abs(R_zj) * half_extent_j).
    lowest_sole_z = float(min(
        data.geom_xpos[geom_id, 2]
        - np.dot(
            np.abs(data.geom_xmat[geom_id].reshape(3, 3)[2]),
            model.geom_size[geom_id],
        )
        for geom_id in foot_box_ids
    ))
    return com_x, support_x, lowest_sole_z


def _set_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    root_z: float,
    root_rot: tuple[float, float, float, float],
    hip_pitch_deg: float | None = None,
) -> None:
    data.qpos[:] = model.qpos0
    data.qpos[:3] = (0.0, 0.0, root_z)
    data.qpos[3:7] = root_rot
    for name, angle in HOME_FRAME.joint_pos.items():
        data.qpos[model.joint(name).qposadr] = angle
    if hip_pitch_deg is not None:
        hip_rad = np.deg2rad(hip_pitch_deg)
        for side in ("left", "right"):
            data.qpos[model.joint(f"{side}_hip_pitch").qposadr] = hip_rad
            data.qpos[model.joint(f"{side}_ankle_pitch").qposadr] = -hip_rad
    mujoco.mj_forward(model, data)


def _solve(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    if any(
        HOME_FRAME.joint_pos[name] != 0.0
        for name in (
            "left_shoulder_pitch", "right_shoulder_pitch", "left_knee", "right_knee"
        )
    ):
        raise ValueError("This solve requires zero shoulder pitch and knee angles")

    def residual(hip_deg: float) -> float:
        _set_pose(model, data, root_z=0.0, root_rot=(1.0, 0.0, 0.0, 0.0), hip_pitch_deg=hip_deg)
        com_x, support_x, _ = _measure_geometry(model, data)
        return com_x - support_x

    lo, hi = -10.0, 10.0
    if not residual(lo) < 0.0 < residual(hi):
        raise ValueError("The centered COM is not bracketed by +/-10 degrees hip pitch")
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if residual(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    hip_deg = (lo + hi) / 2.0
    _set_pose(model, data, root_z=0.0, root_rot=(1.0, 0.0, 0.0, 0.0), hip_pitch_deg=hip_deg)
    com_x, support_x, lowest_sole_z = _measure_geometry(model, data)
    root_z = -lowest_sole_z
    trunk_id = model.body("trunk").id
    head_height = (
        root_z
        + data.xipos[trunk_id, 2]
        + _TRUNK_TO_HEAD_OFFSET * data.ximat[trunk_id].reshape(3, 3)[2, 2]
    )
    _set_pose(model, data, root_z=0.0, root_rot=(1.0, 0.0, 0.0, 0.0), hip_pitch_deg=0.0)
    _, _, zero_lowest_sole_z = _measure_geometry(model, data)
    zero_root_z = -zero_lowest_sole_z

    print(f"robot_xml: {MICROBAN_XML}")
    print("root_pitch_deg: 0.000000000000")
    print("shoulder_pitch_deg: 0.000000000000 0.000000000000")
    print("knee_deg: 0.000000000000 0.000000000000")
    print(f"hip_pitch_deg: {hip_deg:.12f} {hip_deg:.12f}")
    print(f"ankle_pitch_deg: {-hip_deg:.12f} {-hip_deg:.12f}")
    print(f"root_z_m: {root_z:.15f}")
    print(f"all_zero_hip_ankle_root_z_m: {zero_root_z:.15f}")
    print(f"root_z_change_mm: {1000.0 * (root_z - zero_root_z):.9f}")
    print(f"com_minus_support_x_mm: {1000.0 * (com_x - support_x):.9f}")
    print(f"getup_virtual_head_height_m: {head_height:.15f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solve", action="store_true", help="Solve vertical-trunk HOME from robot.xml")
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(MICROBAN_XML))
    data = mujoco.MjData(model)
    if args.solve:
        _solve(model, data)
        return
    _set_pose(model, data, root_z=float(HOME_FRAME.pos[2]), root_rot=tuple(HOME_FRAME.rot))
    com_x, support_x, lowest_sole_z = _measure_geometry(model, data)

    print(f"robot_xml: {MICROBAN_XML}")
    print(
        "hip_pitch_deg: "
        f"{np.rad2deg(HOME_FRAME.joint_pos['left_hip_pitch']):.12f} "
        f"{np.rad2deg(HOME_FRAME.joint_pos['right_hip_pitch']):.12f}"
    )
    print(
        "shoulder_pitch_deg: "
        f"{np.rad2deg(HOME_FRAME.joint_pos['left_shoulder_pitch']):.12f} "
        f"{np.rad2deg(HOME_FRAME.joint_pos['right_shoulder_pitch']):.12f}"
    )
    print(
        "ankle_pitch_deg: "
        f"{np.rad2deg(HOME_FRAME.joint_pos['left_ankle_pitch']):.12f} "
        f"{np.rad2deg(HOME_FRAME.joint_pos['right_ankle_pitch']):.12f}"
    )
    print(f"root_z_m: {HOME_FRAME.pos[2]:.12f}")
    print(f"root_quat_wxyz: {HOME_FRAME.rot}")
    print(f"com_x_m: {com_x:.15f}")
    print(f"support_center_x_m: {support_x:.15f}")
    print(f"com_minus_support_x_mm: {1000.0 * (com_x - support_x):.9f}")
    print(f"lowest_sole_corner_z_m: {lowest_sole_z:.12e}")


if __name__ == "__main__":
    main()
