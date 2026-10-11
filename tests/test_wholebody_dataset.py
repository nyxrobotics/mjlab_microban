"""The whole-body dataset path: a GMR PKL is cut, written and read back on the CPU."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation as R

from mjlab_microban.robot.home_pose import HOME
from mjlab_microban.wholebody.data import build
from mjlab_microban.wholebody.data.build import process_file, split_files
from mjlab_microban.wholebody.data.fk import quat_wxyz_to_xyzw, quat_xyzw_to_wxyz
from mjlab_microban.wholebody.data.hdf5 import write_motion_shard
from mjlab_microban.wholebody.data.preprocess import (
    ClipKinematics,
    ExclusionSpec,
    convex_hull_2d,
    distance_outside,
    frame_flags,
    joint_jumps,
)
from mjlab_microban.wholebody.motion import MotionLib

FPS = 30
JOINTS = (
    "head", "neck_roll", "neck_pitch",
    "right_shoulder_pitch", "right_shoulder_roll", "right_elbow",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "left_shoulder_pitch", "left_shoulder_roll", "left_elbow",
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle_pitch", "left_ankle_roll",
)


def test_quaternion_order_round_trip():
    wxyz = np.array([0.9, 0.1, -0.2, 0.3])
    assert np.array_equal(quat_xyzw_to_wxyz(np.array([0.1, -0.2, 0.3, 0.9])), wxyz)
    assert np.array_equal(quat_wxyz_to_xyzw(wxyz), [0.1, -0.2, 0.3, 0.9])


def test_home_clip_with_sunk_feet_splits_into_named_sections(tmp_path):
    # 3 s standing at HOME; frames 40-48 (0.3 s) are 2 cm lower: the soles
    # are 2 cm into the floor, beyond the 1 cm limit.
    frames = 3 * FPS
    root_pos = np.tile(np.asarray(HOME.root_pos), (frames, 1))
    root_pos[40:49, 2] -= 0.02
    clip = {
        "fps": float(FPS),
        "root_pos": root_pos,
        "root_rot": np.tile(quat_wxyz_to_xyzw(np.asarray(HOME.root_quat_wxyz)), (frames, 1)),
        "dof_pos": np.tile([HOME.joint_pos_rad[name] for name in JOINTS], (frames, 1)),
        "joint_names": list(JOINTS),
    }
    path = tmp_path / "stand1_subject1.pkl"
    path.write_bytes(pickle.dumps(clip))

    spec = ExclusionSpec()
    result = process_file(path, "train", spec)
    (part,) = result["parts"]
    assert part["flagged_frames"] == {
        "floor_penetration": 9, "self_penetration": 0, "joint_range": 0, "joint_jump": 0, "lifted": 0,
        "off_ground": 0, "unsupported": 0,
    }
    pad = round(spec.pad_s * FPS)
    names = [section["name"] for section in part["sections"]]
    assert names == [f"stand1_subject1__f0-{40 - pad}", f"stand1_subject1__f{49 + pad}-{frames}"]

    shard = tmp_path / "train" / "motion.h5"
    assert result["joint_names"] == JOINTS
    write_motion_shard(part["sections"], shard, body_names=result["body_names"], joint_names=JOINTS)
    library = MotionLib(str(tmp_path / "train"), ("trunk", "foot"), JOINTS, device="cpu")
    assert library.window_clip_names == tuple(names)
    assert library.window_categories == ("stand", "stand")

    ref = library.get_frames(torch.tensor([0, 1]), torch.tensor([0.0, 0.5]))
    # The trunk's orientation is the PKL's xyzw quaternion read as wxyz.
    assert torch.allclose(ref["body_quat_w"][:, 0], torch.tensor(HOME.root_quat_wxyz, dtype=torch.float32), atol=1e-5)
    assert torch.allclose(ref["joint_vel"], torch.zeros_like(ref["joint_vel"]), atol=1e-5)


def test_split_keeps_subject5_out_of_training_and_tails_kinds_without_it():
    split = split_files(["walk1_subject1", "walk1_subject5", "walk3_subject5", "sprint1_subject2",
                         "sprint1_subject4", "push1_subject2"])
    assert split == {
        "walk1_subject1": "train", "walk1_subject5": "eval", "walk3_subject5": "eval",
        # No subject5: every file gives its last 20 % to the evaluation set.
        "sprint1_subject2": "tail", "sprint1_subject4": "tail", "push1_subject2": "tail",
    }


def test_joint_jump_marks_a_one_frame_step_but_not_a_fast_smooth_motion():
    spec = ExclusionSpec()
    # A knee swinging at up to 0.46 rad per frame (walk, 30 fps) and an elbow
    # stepping 0.8 rad between two frames (an IK branch switch).
    swing = np.abs(np.diff([0.0, 0.01, 0.06, 0.52, 0.87, 1.17, 1.34]))[:, None]
    switch = np.abs(np.diff([1.62, 1.59, 1.55, 1.45, 0.57, 0.37, 0.40]))[:, None]
    assert not joint_jumps(swing, FPS, spec).any()
    assert joint_jumps(switch, FPS, spec).tolist() == [False, False, False, True, False, False]


def _home_frames(frames: int):
    root_pos = np.tile(np.asarray(HOME.root_pos), (frames, 1))
    root_quat = np.tile(np.asarray(HOME.root_quat_wxyz), (frames, 1))
    return root_pos, root_quat, np.tile([HOME.joint_pos_rad[name] for name in JOINTS], (frames, 1))


def test_support_hull_and_the_distance_outside_it():
    square = convex_hull_2d(np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5], [1, 0]]))
    assert len(square) == 4
    assert distance_outside(np.array([0.5, 0.5]), square) == 0.0
    assert distance_outside(np.array([2.0, 0.5]), square) == 1.0
    assert distance_outside(np.array([0.5, 0.0]), convex_hull_2d(np.array([[0, 1], [1, 1]]))) == 1.0
    assert distance_outside(np.array([0.0, 0.0]), convex_hull_2d(np.zeros((0, 2)))) == np.inf


def test_leaning_back_upright_for_over_a_second_is_unsupported():
    # HOME turned 30 degrees backwards about the back edge of the soles: the
    # toes rise, the centre of mass is behind the heels.  1.5 s of it is cut,
    # 0.5 s (a step, a fall) is not, and HOME stands inside its support.
    checker = build._checker()
    spec = ExclusionSpec()
    root_pos, root_quat, joints = _home_frames(4 * FPS)
    home = checker.run(root_pos[:1], root_quat[:1], joints[:1])
    assert home.com_outside_support_m[0] == 0.0
    corners = checker.data.geom_xpos[checker.corner_geom] + np.einsum(
        "nij,nj->ni", checker.data.geom_xmat[checker.corner_geom].reshape(-1, 3, 3), checker.corner_local)
    pivot = np.array([corners[corners[:, 2] < 0.005, 0].min(), 0.0, 0.0])
    back = R.from_euler("y", -30.0, degrees=True)
    tilted = (pivot + back.apply(root_pos[0] - pivot), (back * R.from_quat(root_quat[0], scalar_first=True)).as_quat(scalar_first=True))
    for first, last in ((15, 15 + 45), (90, 90 + 15)):  # 1.5 s and 0.5 s
        root_pos[first:last], root_quat[first:last] = tilted
    kin = checker.run(root_pos, root_quat, joints)
    assert kin.com_outside_support_m[20] > spec.support_margin_m
    assert np.degrees(kin.trunk_tilt_rad[20]) == pytest.approx(20.0, abs=0.1)  # HOME leans 10 degrees forward
    flags = frame_flags(kin, FPS, spec)
    assert np.flatnonzero(flags["unsupported"]).tolist() == list(range(15, 60))


def test_unsupported_needs_contact_an_upright_trunk_and_time():
    spec = ExclusionSpec()
    frames = 3 * FPS
    outside = np.full(frames, 0.05)
    tilt = np.zeros(frames)

    def flagged(outside, tilt):
        kin = ClipKinematics(
            body_pos_w=np.zeros((frames, 1, 3)), body_quat_w=np.zeros((frames, 1, 4)), lowest_point_z=np.zeros(frames),
            self_penetration_m=np.zeros(frames), joint_excess_rad=np.zeros(frames), joint_step_rad=np.zeros((frames - 1, 1)),
            com_outside_support_m=outside, trunk_tilt_rad=tilt,
        )
        return int(frame_flags(kin, FPS, spec)["unsupported"].sum())

    assert flagged(outside, tilt) == frames
    assert flagged(np.full(frames, 0.015), tilt) == 0  # within the margin
    assert flagged(outside, np.full(frames, np.radians(60.0))) == 0  # lying or crawling
    assert flagged(np.full(frames, np.inf), tilt) == 0  # airborne: the off-ground rule's
    short = np.zeros(frames)
    short[10:10 + FPS] = 0.05  # exactly the duration: kept
    assert flagged(short, tilt) == 0
