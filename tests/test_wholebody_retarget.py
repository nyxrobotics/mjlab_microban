"""The human-to-Microban retargeting on a synthetic CMU-style skeleton (CPU, no downloaded data).

The standing reference lands on HOME, a mirrored motion gives the mirrored
robot, the human's left limbs move the robot's left limbs, the heading and
position of the motion do not change the joints, and an AMASS SMPL-X file
reads as the body model's forward kinematics.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from mjlab_microban.wholebody.retarget import core, kinds, model, skeleton

FPS = 30.0

# CMU-style skeleton in the BVH frame (Y up, facing +Z, the subject's left at +X): (name, parent, offset).
JOINTS = [
    ("Hips", None, (0.0, 0.0, 0.0)),
    ("LowerBack", "Hips", (0.0, 4.0, 0.0)),
    ("Spine1", "LowerBack", (0.0, 8.0, 0.0)),
    ("Neck", "Spine1", (0.0, 8.0, 0.0)),
    ("Head", "Neck", (0.0, 4.0, 0.0)),
    *[j for side, x in (("Left", 1.0), ("Right", -1.0)) for j in (
        (f"{side}Shoulder", "Spine1", (x * 1.0, 7.0, 0.0)),
        (f"{side}Arm", f"{side}Shoulder", (x * 6.0, 0.0, 0.0)),
        (f"{side}ForeArm", f"{side}Arm", (0.0, -10.5, 0.0)),
        (f"{side}Hand", f"{side}ForeArm", (0.0, -9.5, 0.0)),
        (f"{side}UpLeg", "Hips", (x * 3.5, -2.0, 0.0)),
        (f"{side}Leg", f"{side}UpLeg", (0.0, -16.5, 0.0)),
        (f"{side}Foot", f"{side}Leg", (0.0, -16.5, 0.0)),
        (f"{side}ToeBase", f"{side}Foot", (0.0, -1.5, 5.0)),
    )],
]
NAMES = [name for name, _, _ in JOINTS]
STANDING_HIPS_Y = 35.5  # the feet's toe bases on the floor


def _bvh(motion: dict[str, np.ndarray], frames: int, root_pos: np.ndarray | None = None) -> skeleton.Bvh:
    """A Bvh whose joints rotate by ``motion[name]`` ((frames, 3) ZYX Euler degrees); others stay at rest."""
    parents = np.array([NAMES.index(p) if p else -1 for _, p, _ in JOINTS])
    offsets = np.array([o for _, _, o in JOINTS], dtype=float)
    channels = [["Xposition", "Yposition", "Zposition", "Zrotation", "Yrotation", "Xrotation"]]
    channels += [["Zrotation", "Yrotation", "Xrotation"]] * (len(NAMES) - 1)
    if root_pos is None:
        root_pos = np.tile([0.0, STANDING_HIPS_Y, 0.0], (frames, 1))
    cols = [root_pos] + [motion.get(name, np.zeros((frames, 3))) for name in NAMES]
    return skeleton.Bvh(NAMES, parents, offsets, channels, np.concatenate(cols, axis=1), 1.0 / FPS)


def _wave(frames: int, amplitude_deg: float, phase: float = 0.0) -> np.ndarray:
    return amplitude_deg * np.sin(np.linspace(0.0, 2.0 * np.pi, frames) + phase)


def _motion(frames: int = 46) -> skeleton.Bvh:
    """An asymmetric motion: the left arm swings forward and out, the right knee bends, the trunk turns and leans."""
    zeros = np.zeros(frames)
    return _bvh({
        "LowerBack": np.stack([_wave(frames, 5), _wave(frames, 15), zeros], 1),
        "LeftArm": np.stack([_wave(frames, 25), zeros, _wave(frames, 50, 0.5)], 1),
        "LeftForeArm": np.stack([zeros, zeros, -np.abs(_wave(frames, 60))], 1),
        "RightUpLeg": np.stack([zeros, zeros, -np.abs(_wave(frames, 30))], 1),
        "RightLeg": np.stack([zeros, zeros, np.abs(_wave(frames, 50))], 1),
    }, frames)


def _retarget(tmp_path, stem, frames, ref):
    report = core.retarget_frames(stem, "bvh_cmu", frames, FPS, ref, tmp_path)
    return report, np.load(tmp_path / f"{stem}.npz")["qpos"]


def _site_positions(qpos: np.ndarray) -> dict[str, np.ndarray]:
    m = mujoco.MjModel.from_xml_path(str(model.gmr_model_xml()))
    d = mujoco.MjData(m)
    names = [f"{s}_{p}" for s in ("left", "right") for p in ("hip_c", "knee_c", "ankle_c", "shoulder_c", "elbow_c", "hand")]
    out = {n: [] for n in names}
    for q in qpos:
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        for n in names:
            out[n].append(d.site_xpos[m.site(n).id].copy())
    return {n: np.asarray(v) for n, v in out.items()}


def _mirror(frame: dict) -> dict:
    """Mirror a frame across the sagittal plane (y -> -y) and swap left and right."""
    swap = lambda n: n.replace("Left", "@").replace("Right", "Left").replace("@", "Right")
    out = {}
    for name, (pos, quat) in frame.items():
        w, x, y, z = quat
        out[swap(name)] = (np.array([pos[0], -pos[1], pos[2]]), np.array([w, -x, y, -z]))
    return out


def test_standing_reference_retargets_to_home(tmp_path):
    b = _bvh({}, 2)
    ref = skeleton.standing_reference(b, "bvh_cmu")
    report, qpos = _retarget(tmp_path, "stand", [ref] * 5, ref)
    assert report["ref_to_home_max_joint_err_deg"] < 1e-3
    assert report["ref_to_home_root_err_mm"] < 1e-3
    m = mujoco.MjModel.from_xml_path(str(model.gmr_model_xml()))
    home = model.home_qpos(m)
    # Standing still from a reset: within 0.05 degrees of HOME from the first frame.
    assert np.degrees(np.abs(qpos[:, 7:] - home[7:]).max()) < 0.05


def test_mirrored_motion_gives_the_mirrored_robot(tmp_path):
    b = _motion()
    ref = skeleton.standing_reference(b, "bvh_cmu")
    frames, _, _ = skeleton.motion_frames(b, "bvh_cmu", FPS)
    _, q = _retarget(tmp_path, "motion", frames, ref)
    _, q_m = _retarget(tmp_path, "mirrored", [_mirror(f) for f in frames], _mirror(ref))
    sites, sites_m = _site_positions(q), _site_positions(q_m)
    flip = np.array([1.0, -1.0, 1.0])
    for name, pos in sites.items():
        other = name.replace("left", "@").replace("right", "left").replace("@", "right")
        assert np.abs(pos - sites_m[other] * flip).max() < 2e-3, name
    # The motion is asymmetric, so the check is not trivially met.
    assert np.abs(sites["left_hand"] - sites["right_hand"] * flip).max() > 0.02


def test_the_human_left_drives_the_robot_left(tmp_path):
    # The human raises the left arm sideways and lifts the left foot (hip forward, knee back); the right side stays.
    frames = 31
    ramp = np.linspace(0.0, 1.0, frames)
    zeros = np.zeros(frames)
    b = _bvh({
        "LeftArm": np.stack([80.0 * ramp, zeros, zeros], 1),  # -Y (hanging) towards +X (the subject's left)
        "LeftUpLeg": np.stack([zeros, zeros, -45.0 * ramp], 1),  # the thigh forward (+Z)
        "LeftLeg": np.stack([zeros, zeros, 70.0 * ramp], 1),  # the shank back
    }, frames)
    ref = skeleton.standing_reference(b, "bvh_cmu")
    frames_, _, _ = skeleton.motion_frames(b, "bvh_cmu", FPS)
    _, q = _retarget(tmp_path, "left", frames_, ref)
    sites = _site_positions(q)
    rise = {name: sites[name][-1, 2] - sites[name][0, 2] for name in ("left_hand", "right_hand", "left_ankle_c", "right_ankle_c")}
    assert rise["left_hand"] > 0.05 and abs(rise["right_hand"]) < 0.005, rise
    assert rise["left_ankle_c"] > 0.02 and abs(rise["right_ankle_c"]) < 0.005, rise
    # The robot faces +x, so its left is +y: the raised hand goes out to the left.
    assert sites["left_hand"][-1, 1] - sites["left_hand"][0, 1] > 0.03


def test_heading_and_position_do_not_change_the_joints(tmp_path):
    b = _motion()
    ref = skeleton.standing_reference(b, "bvh_cmu")
    frames, _, _ = skeleton.motion_frames(b, "bvh_cmu", FPS)
    turn = R.from_euler("z", 73.0, degrees=True)
    shift = np.array([30.0, -20.0, 0.0])
    moved = [{n: (turn.apply(p) + shift, (turn * R.from_quat(q, scalar_first=True)).as_quat(scalar_first=True))
              for n, (p, q) in f.items()} for f in frames]
    _, q = _retarget(tmp_path, "motion", frames, ref)
    _, q_moved = _retarget(tmp_path, "moved", moved, ref)
    assert np.abs(q_moved[:, 7:] - q[:, 7:]).max() < 1e-6
    yaw = R.from_quat(q_moved[:, 3:7], scalar_first=True) * R.from_quat(q[:, 3:7], scalar_first=True).inv()
    assert np.allclose(np.degrees(yaw.magnitude()), 73.0, atol=1e-4)


def test_motion_frames_face_x_with_the_feet_on_the_floor():
    frames, fps, tilt = skeleton.motion_frames(_motion(), "bvh_cmu", FPS)
    assert fps == FPS and tilt == 0.0
    first = frames[0]
    left = first["LeftUpLeg"][0] - first["RightUpLeg"][0]
    assert left[1] > 0 and abs(left[0]) < 1e-6  # facing +x: the left hip is at +y
    legs = [n for n in NAMES if "Leg" in n or "Foot" in n or "Toe" in n]
    assert min(f[n][0][2] for f in frames for n in legs) == pytest.approx(0.0, abs=1e-9)
    assert first["Hips"][0][:2] == pytest.approx([0.0, 0.0], abs=1e-9)


def test_read_bvh_takes_tabs_and_skips_end_sites(tmp_path):
    text = (
        "HIERARCHY\nROOT Hips\n{\n\tOFFSET 0 0 0\n\tCHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation\n"
        "\tJOINT LeftUpLeg\n\t{\n\t\tOFFSET 3.5\t-2 0\n\t\tCHANNELS 3 Zrotation Yrotation Xrotation\n"
        "\t\tEnd Site\n\t\t{\n\t\t\tOFFSET 0 -16.5 0\n\t\t}\n\t}\n}\n"
        "MOTION\nFrames: 2\nFrame Time: 0.0083333\n0 35 0 0 0 0 0 0 90\n1\t35\t0\t0\t0\t0\t0\t0\t0\n"
    )
    path = tmp_path / "tiny.bvh"
    path.write_text(text)
    b = skeleton.read_bvh(str(path))
    assert b.names == ["Hips", "LeftUpLeg"] and b.parents.tolist() == [-1, 0]
    assert b.offsets[1].tolist() == [3.5, -2.0, 0.0]
    assert b.motion.shape == (2, 9) and b.frame_time == pytest.approx(1 / 120, rel=1e-4)
    lrot, lpos = skeleton.local_transforms(b)
    grot, gpos = skeleton.fk(b, lrot, lpos)
    assert gpos[1, 1] == pytest.approx([4.5, 33.0, 0.0])
    # 90 degrees about x turns the leg's down-pointing child from -y to -z.
    assert grot[1][0].apply([0.0, -16.5, 0.0]) == pytest.approx([0.0, 0.0, -16.5])


def test_cmu_kinds_follow_the_index_words():
    assert kinds.kind("WalkStraight") == "walk"
    assert kinds.kind("Getting up from chair") == "sit"
    assert kinds.kind("sit on high stool, stand up") == "climb"  # stools come first: the stool holds the subject
    assert kinds.kind("jumping jacks") == "jump"
    assert kinds.kind("crawl on the floor") == "ground"
    assert kinds.kind("") == "misc"


def _smplx_model(folder, rng):
    """A synthetic SMPL-X body model: 55 joints near SMPL-X's layout (Y up, facing +Z), vertices on the joints."""
    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19] + [20] * 33
    offsets = {1: [.08, -.08, 0], 2: [-.08, -.08, 0], 3: [0, .1, 0], 4: [0, -.4, 0], 5: [0, -.4, 0], 6: [0, .12, 0],
               7: [0, -.4, 0], 8: [0, -.4, 0], 9: [0, .05, 0], 10: [0, -.05, .12], 11: [0, -.05, .12], 12: [0, .2, 0],
               13: [.08, .12, 0], 14: [-.08, .12, 0], 15: [0, .1, 0], 16: [.1, 0, 0], 17: [-.1, 0, 0], 18: [.28, 0, 0],
               19: [-.28, 0, 0], 20: [.25, 0, 0], 21: [-.25, 0, 0]}
    joints = np.zeros((55, 3))
    joints[0] = [0, .95, 0]
    for i in range(1, 55):
        joints[i] = joints[parents[i]] + np.array(offsets.get(i, [.01, 0, 0]))
    n_verts = 300
    regressor, verts = np.zeros((55, n_verts)), np.zeros((n_verts, 3))
    for i in range(55):
        regressor[i, rng.choice(n_verts, 4, replace=False)] = .25
    for i in range(55):
        verts[np.flatnonzero(regressor[i])] = joints[i]  # later joints win shared vertices; J = regressor @ v either way
    kintree = np.stack([np.array(parents), np.arange(55)])
    kintree[0, 0] = 4294967295
    np.savez(folder / "SMPLX_MALE.npz", v_template=verts, shapedirs=rng.normal(0, .01, (n_verts, 3, 400)),
             J_regressor=regressor, kintree_table=kintree)


def test_amass_smplx_motion_matches_the_body_model_and_retargets(tmp_path):
    from mjlab_microban.wholebody.retarget import smplx

    rng = np.random.default_rng(0)
    _smplx_model(tmp_path, rng)
    frames, rate = 50, 100.0  # 100 fps resampled to 30
    betas = rng.normal(0, 1, 16)
    root_orient = (R.from_euler("x", 90, degrees=True) * R.from_euler("y", np.linspace(0, 90, frames)[:, None], degrees=True)).as_rotvec()
    pose_body = rng.normal(0, .3, (frames, 63)) * np.linspace(0, 1, frames)[:, None]
    trans = np.c_[np.linspace(0, 1, frames), np.zeros(frames), np.full(frames, .1)]
    np.savez(tmp_path / "seq_stageii.npz", gender=np.array("male"), betas=betas, root_orient=root_orient,
             pose_body=pose_body, trans=trans, mocap_frame_rate=np.array(rate))
    b = smplx.load_amass(tmp_path / "seq_stageii.npz", tmp_path, FPS)
    assert 1.0 / b.frame_time == pytest.approx(FPS)
    lrot, lpos = skeleton.local_transforms(b)
    _, gpos = skeleton.fk(b, lrot, lpos)
    pos = skeleton.YUP_TO_ZUP.apply(gpos.reshape(-1, 3)).reshape(gpos.shape)
    # SMPL-X's forward kinematics in AMASS's Z-up world, at the source frames on the 30 fps grid.
    joints, parents = smplx.rest_joints(tmp_path, "male", betas)
    for k in range(0, len(pos), 3):
        f = k * 10 // 3
        rv = np.r_[root_orient[f], pose_body[f]].reshape(22, 3)
        glob, expect = [None] * 22, np.zeros((22, 3))
        for j in range(22):
            rj = R.from_rotvec(rv[j])
            if j == 0:
                glob[0], expect[0] = rj, joints[0] + trans[f]
            else:
                glob[j] = glob[parents[j]] * rj
                expect[j] = expect[parents[j]] + glob[parents[j]].apply(joints[j] - joints[parents[j]])
        assert np.abs(expect - pos[k]).max() < 1e-9
    ref = skeleton.standing_reference(b, "smplx")
    report = core.retarget_frames("seq", "smplx", [ref] * 3, FPS, ref, tmp_path)
    assert report["ref_to_home_max_joint_err_deg"] < 1e-3 and report["ref_to_home_root_err_mm"] < 1e-3


def test_support_shift_puts_the_zmp_on_the_standing_foot(tmp_path):
    # The human lifts the right foot (hip forward, knee back) and stands on the left with the hips where they were:
    # scaled to Microban's wide stance, the zero-moment point stays near the middle, 3-4 cm outside the left sole's
    # support.  With the support shift the trunk moves over the left foot and the zero-moment point comes to the
    # support (but in the last half window of the derivatives' fit, fitted from one side and not constrained), the
    # standing foot stays put and stays on the floor, and the arms keep their pose relative to the trunk.
    from mjlab_microban.wholebody.retarget import support

    frames = 60
    lift = np.clip(np.linspace(-1.0, 2.0, frames), 0.0, 1.0)
    zeros = np.zeros(frames)
    b = _bvh({"RightUpLeg": np.stack([zeros, zeros, -40.0 * lift], 1), "RightLeg": np.stack([zeros, zeros, 60.0 * lift], 1)},
             frames)
    ref = skeleton.standing_reference(b, "bvh_cmu")
    frames_, _, _ = skeleton.motion_frames(b, "bvh_cmu", FPS)
    out = {}
    for name, spec in (("plain", None), ("shifted", support.SupportSpec())):
        core.retarget_frames(name, "bvh_cmu", frames_, FPS, ref, tmp_path, support_spec=spec)
        out[name] = np.load(tmp_path / f"{name}.npz")
    plain, shifted = out["plain"], out["shifted"]
    left, right = (support.PARTS.index(p) for p in ("left_foot", "right_foot"))
    one = plain["contact"][:, left] & ~plain["contact"][:, right] & np.isfinite(plain["zmp_outside"])
    one[-support.windows(frames, FPS, support.SupportSpec.smooth_s)[0].shape[1] // 2:] = False
    assert one.sum() >= 20 and (shifted["contact"] == plain["contact"]).all()
    assert np.median(plain["zmp_outside"][one]) > 0.03
    assert np.percentile(shifted["zmp_outside"][one], 95) < 0.008
    assert np.median(shifted["support_shift"][one, 1]) > 0.02  # the trunk went left
    sites = {name: _site_positions(out[name]["qpos"]) for name in out}
    assert np.abs(sites["shifted"]["left_ankle_c"] - sites["plain"]["left_ankle_c"]).max() < 3e-3
    hand = {name: s["left_hand"] - s["left_shoulder_c"] for name, s in sites.items()}
    assert np.abs(hand["shifted"] - hand["plain"]).max() < 3e-3
