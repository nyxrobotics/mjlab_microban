"""The whole-body data tools: clip names, the arm-flip cut, the evaluation-only build, the mix links, PICO parts."""

from __future__ import annotations

import json
import pickle

import numpy as np

from mjlab_microban.robot.home_pose import HOME
from mjlab_microban.wholebody.data.fk import quat_wxyz_to_xyzw
from mjlab_microban.wholebody.data.preprocess import ExclusionSpec
from mjlab_microban.wholebody.retarget import pico, pipeline

FPS = 30


def test_cmu_clips_are_named_by_kind_subject_and_take(tmp_path):
    cmu = tmp_path / "raw" / "cmu"
    (cmu / "data" / "007").mkdir(parents=True)
    (cmu / "data" / "113").mkdir(parents=True)
    (cmu / "data" / "007" / "07_01.bvh").write_text("")
    (cmu / "data" / "113" / "113_25.bvh").write_text("")
    (cmu / "cmu-mocap-index-text.txt").write_text("07_01\twalk\n")
    jobs = pipeline._jobs("cmu", tmp_path / "raw")
    assert [(name, desc) for _, fmt, name, desc in jobs] == [("cmuwalk_07_01", "walk"), ("cmumisc_113_25", "")]
    assert {fmt for _, fmt, _, _ in jobs} == {"bvh_cmu"}


def _write_clip(folder, name, frames, flip=None):
    folder.mkdir(parents=True, exist_ok=True)
    clip = {"fps": float(FPS), "root_pos": np.zeros((frames, 3)), "root_rot": np.tile([0, 0, 0, 1.0], (frames, 1)),
            "dof_pos": np.zeros((frames, 21)), "joint_names": [f"j{i}" for i in range(21)]}
    (folder / f"{name}.pkl").write_bytes(pickle.dumps(clip))
    np.savez(folder / f"{name}.npz", flip=np.zeros(frames, bool) if flip is None else flip)


def test_finalize_cuts_only_flips_of_three_frames_or_more(tmp_path):
    gmr = tmp_path / "gmr" / "cmu"
    long_flip = np.zeros(200, bool)
    long_flip[100:103] = True
    short_flip = np.zeros(200, bool)
    short_flip[50:52] = True
    _write_clip(gmr, "cmuwalk_01_01", 200, long_flip)
    _write_clip(gmr, "cmuwalk_01_02", 200, short_flip)
    _write_clip(gmr, "cmuwalk_01_03", 200)
    summary = pipeline.finalize("cmu", tmp_path)
    pad = round(ExclusionSpec().pad_s * FPS)
    out = tmp_path / "cmu" / "retarget"
    assert sorted(p.name for p in out.glob("*.pkl")) == [
        "cmuwalk_01_01_p0.pkl", "cmuwalk_01_01_p1.pkl", "cmuwalk_01_02.pkl", "cmuwalk_01_03.pkl"]
    first = pickle.loads((out / "cmuwalk_01_01_p0.pkl").read_bytes())
    second = pickle.loads((out / "cmuwalk_01_01_p1.pkl").read_bytes())
    assert len(first["root_pos"]) == 100 - pad and len(second["dof_pos"]) == 200 - (103 + pad)
    assert first["joint_names"] == [f"j{i}" for i in range(21)]
    assert summary["cut_for_arm_flip"] == 3 + 2 * pad and summary["clips_cut"] == 1
    assert json.loads((tmp_path / "cmu" / "finalize.json").read_text())["summary"] == summary


def test_evaluation_only_set_writes_every_file_to_the_evaluation_shard(tmp_path):
    joints = ("head", "neck_roll", "neck_pitch",
              "right_shoulder_pitch", "right_shoulder_roll", "right_elbow",
              "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle_pitch", "right_ankle_roll",
              "left_shoulder_pitch", "left_shoulder_roll", "left_elbow",
              "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle_pitch", "left_ankle_roll")
    folder = tmp_path / "pico_eval" / "retarget"
    folder.mkdir(parents=True)
    for name in ("picoteleopit_rec00", "picotwist_0807_walk_001"):
        clip = {"fps": float(FPS), "root_pos": np.tile(np.asarray(HOME.root_pos), (2 * FPS, 1)),
                "root_rot": np.tile(quat_wxyz_to_xyzw(np.asarray(HOME.root_quat_wxyz)), (2 * FPS, 1)),
                "dof_pos": np.tile([HOME.joint_pos_rad[j] for j in joints], (2 * FPS, 1)), "joint_names": list(joints)}
        (folder / f"{name}.pkl").write_bytes(pickle.dumps(clip))
    report = pipeline.build("pico_eval", tmp_path, workers=1)
    assert list(report["shards"]) == ["eval"]
    assert (tmp_path / "pico_eval" / "eval" / "motion.h5").exists()
    assert not (tmp_path / "pico_eval" / "train").exists()
    assert report["kinds"]["picoteleopit"]["files"] == {"train": [], "eval": ["picoteleopit_rec00"]}


def test_mix_links_the_training_sets(tmp_path):
    for name in ("lafan1", "cmu"):
        for part in ("train", "eval"):
            (tmp_path / name / part).mkdir(parents=True)
            (tmp_path / name / part / "motion.h5").write_text(name)
    pipeline.mix(["lafan1", "cmu"], tmp_path)
    for part in ("train", "eval"):
        links = sorted((tmp_path / "mix" / part).iterdir())
        assert [p.name for p in links] == ["cmu.h5", "lafan1.h5"]
        assert all(p.is_symlink() and not p.readlink().is_absolute() for p in links)
        assert (tmp_path / "mix" / part / "cmu.h5").read_text() == "cmu"


def test_pico_recordings_split_at_root_teleports_and_resample_to_30_fps():
    root = np.zeros((100, 3))
    root[:, 0] = np.arange(100) * 0.01
    root[40:, 0] += 0.5  # a teleport between frames 39 and 40
    root[95:, 0] += 0.5  # and a last part shorter than a second
    assert pico.pieces(root, 30.0) == [(0, 40), (40, 95)]
    frames = 60
    clip = {"root_pos": np.zeros((frames, 3)), "root_rot": np.tile([0, 0, 0, 1.0], (frames, 1)), "dof_pos": np.zeros((frames, 29))}
    clip["root_pos"][:, 0] = np.arange(frames) / 29.8  # 1 m/s at 29.8 fps
    source, name, fps, root_pos, root_quat, joints = pico.resample(clip, 29.8, "picotwist_x")
    assert joints.shape == (len(root_pos), 29)
    assert (source, name, fps) == ("twist2", "picotwist_x", 30.0)
    assert np.allclose(np.diff(root_pos[:, 0]), 1 / 30)
    assert root_quat.shape == (len(root_pos), 4) and np.allclose(root_quat, [1, 0, 0, 0])


def test_retargeting_leaves_no_stale_output(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    from mjlab_microban.wholebody.retarget import core

    data = tmp_path / "raw" / "cmu" / "data" / "007"
    data.mkdir(parents=True)
    for take in ("07_01", "07_02", "07_03"):
        (data / f"{take}.bvh").write_text("")
    (tmp_path / "raw" / "cmu" / "cmu-mocap-index-text.txt").write_text("07_01\twalk\n07_02\twalk\n07_03\twalk\n")
    failing: set[str] = set()

    def fake_run(path, fmt, fps, out_dir, stem, smplx_model_dir=None, support_spec=None):
        if stem in failing:
            raise RuntimeError("IK failed")
        for suffix in pipeline.CLIP_OUTPUTS:
            (out_dir / f"{stem}{suffix}").write_text("")
        return {"frames": 1}

    monkeypatch.setattr(core, "run", fake_run)
    monkeypatch.setattr(pipeline, "ProcessPoolExecutor", ThreadPoolExecutor)
    out = tmp_path / "gmr" / "cmu"
    out.mkdir(parents=True)
    (out / "cmuold_99_99.pkl").write_text("")  # a clip the current rules no longer make
    (out / "cmuwalk_07_03.pkl").write_text("")  # made last time, fails this time

    failing.add("cmuwalk_07_03")
    errors = pipeline.retarget("cmu", tmp_path, workers=2)
    assert sorted(p.name for p in out.glob("*.pkl")) == ["cmuwalk_07_01.pkl", "cmuwalk_07_02.pkl"]
    assert list(errors) == ["cmuwalk_07_03"]

    # Redoing some clips: their old output goes, the other clips' output and errors stay.
    failing.clear()
    failing.add("cmuwalk_07_02")
    pipeline.retarget("cmu", tmp_path, workers=2, only={"cmuwalk_07_02", "cmuwalk_07_03", "lafan_only_name"})
    assert sorted(p.name for p in out.glob("*.pkl")) == ["cmuwalk_07_01.pkl", "cmuwalk_07_03.pkl"]
    assert list(json.loads((out / "errors.json").read_text())) == ["cmuwalk_07_02"]
    # Names of another set leave this one alone.
    assert pipeline.retarget("cmu", tmp_path, workers=2, only={"walk1_subject1"}) == {}
    assert list(json.loads((out / "errors.json").read_text())) == ["cmuwalk_07_02"]
