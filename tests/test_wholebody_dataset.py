"""The whole-body dataset path: a GMR PKL is cut, written and read back on the CPU."""

from __future__ import annotations

import pickle

import numpy as np
import torch

from mjlab_microban.robot.home_pose import HOME
from mjlab_microban.wholebody.data.build import process_file, split_files
from mjlab_microban.wholebody.data.fk import quat_wxyz_to_xyzw, quat_xyzw_to_wxyz
from mjlab_microban.wholebody.data.hdf5 import write_motion_shard
from mjlab_microban.wholebody.data.preprocess import ExclusionSpec, joint_jumps
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
        "off_ground": 0,
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
