"""The whole-body data, from the downloads to the HDF5 shards (docs/wholebody_data.md).

Layout (everything under data/wholebody/, which git ignores):

* ``raw/``: the sources.  ``fetch`` downloads LAFAN1, CMU and G1's model at
  pinned versions; the PICO recordings (``raw/pico/``) and AMASS
  (``raw/amass/``, ``raw/body_models/smplx/``) are placed by hand.
* ``gmr/<set>/``: the retargeting's output per clip (``<name>.pkl``, ``.npz``,
  ``_report.json``, ``_ik.json``), ``manifest.json`` (name, source file,
  description) and ``errors.json`` (clips that failed).  A run of the whole
  set starts from an empty folder, and a run of some clips (``only``) first
  removes their old output, so a clip that fails or is no longer made leaves
  no PKL for ``finalize`` to pick up.
* ``<set>/retarget/``: the PKLs with the long arm-branch flips cut
  (``finalize``), ``<set>/finalize.json`` (what was cut), and the training
  side's ``<set>/{train,eval}/motion.h5`` and ``report.json``.  The PICO set
  (``pico_eval``) is evaluation only.
* ``mix/{train,eval}/<set>.h5``: links that put the sets in one training.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import subprocess
import time
import traceback
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from mjlab_microban.wholebody.retarget import kinds

ROOT = Path("data/wholebody")
FPS = 30.0
TRAINING_SETS = ("lafan1", "cmu", "amass")

LAFAN1_REPO = "https://github.com/ubisoft/ubisoft-laforge-animation-dataset"
LAFAN1_ZIP = "https://media.githubusercontent.com/media/ubisoft/ubisoft-laforge-animation-dataset/master/lafan1/lafan1.zip"
LAFAN1_ZIP_SHA256 = "ea918082b500a5d158e9d3aa39039df04cd42e25f5c02fe8f7e88e8e9365a977"
LAFAN1_TEXTS = {"LICENSE_lafan1.txt": "license.txt", "README_lafan1.md": "README.md"}
CMU_REPO = "https://github.com/una-dinosauria/cmu-mocap"
CMU_COMMIT = "09a07f54f3bbb58797325f009282d0b2048a2871"
GMR_RAW = "https://raw.githubusercontent.com/YanjieZe/GMR/bb1bbe40774794fceb2a7c579a3464a28e68c844"
G1_FILES = {"g1_mocap_29dof.xml": "assets/unitree_g1/g1_mocap_29dof.xml", "LICENSE": "assets/unitree_g1/LICENSE"}

# Arm-branch flips (core.arm_flips) the training rules do not cut: runs of at
# least MIN_FLIP_RUN frames are cut with the training side's padding and
# shortest piece (ExclusionSpec's pad_s and min_frames).  Shorter runs are
# mostly passes through the branch switch or an elbow too straight to tell its
# bend (1666 of LAFAN1's runs are 1-2 frames); their joint jumps are cut by
# the training rules.
MIN_FLIP_RUN = 3


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as stream:
        shutil.copyfileobj(response, stream)
    tmp.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(root: Path = ROOT) -> None:
    """Download LAFAN1, CMU (cgspeed's BVH) and G1's model (for the PICO recordings) into ``root/raw``."""
    raw = root / "raw"
    lafan = raw / "lafan1"
    if not (lafan / "lafan1.zip").exists():
        _download(LAFAN1_ZIP, lafan / "lafan1.zip")
    if _sha256(lafan / "lafan1.zip") != LAFAN1_ZIP_SHA256:
        raise ValueError(f"{lafan / 'lafan1.zip'} is not the LAFAN1 zip this pipeline was built on")
    with zipfile.ZipFile(lafan / "lafan1.zip") as archive:
        archive.extractall(lafan)
    for name, source in LAFAN1_TEXTS.items():
        if not (lafan / name).exists():
            _download(f"{LAFAN1_REPO}/raw/master/{source}", lafan / name)
    cmu = raw / "cmu"
    if not (cmu / ".git").exists():
        cmu.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "-q"], cwd=cmu, check=True)
        subprocess.run(["git", "remote", "add", "origin", CMU_REPO], cwd=cmu, check=True)
    head = subprocess.run(["git", "rev-parse", "-q", "--verify", "HEAD"], cwd=cmu, capture_output=True, text=True, check=False).stdout.strip()
    if head != CMU_COMMIT:
        subprocess.run(["git", "fetch", "-q", "--depth", "1", "origin", CMU_COMMIT], cwd=cmu, check=True)
        subprocess.run(["git", "checkout", "-q", "FETCH_HEAD"], cwd=cmu, check=True)
    for name, source in G1_FILES.items():
        if not (raw / "g1" / name).exists():
            _download(f"{GMR_RAW}/{source}", raw / "g1" / name)


def _jobs(name: str, raw: Path) -> list[tuple[str, str, str, str]]:
    """[(source file, format, clip name, description)] of one training set."""
    if name == "lafan1":
        return [(str(f), "bvh_lafan1", f.stem, "") for f in sorted((raw / "lafan1").glob("*.bvh"))]
    if name == "cmu":
        index = kinds.load_index(str(raw / "cmu" / "cmu-mocap-index-text.txt"))
        jobs = []
        for f in sorted((raw / "cmu" / "data").glob("*/*.bvh")):
            subject, take = f.stem.split("_")
            key = f"{int(subject):02d}_{int(take):02d}"
            desc = index.get(key, "")
            jobs.append((str(f), "bvh_cmu", f"cmu{kinds.kind(desc)}_{key}", desc))
        return jobs
    if name == "amass":
        # AMASS's CMU is the CMU capture again (fitted anew): kept out of training while the BVH set is in.
        files = sorted(p for p in (raw / "amass" / "smplx_g").glob("**/*_stageii.npz")
                       if p.relative_to(raw / "amass" / "smplx_g").parts[0] != "CMU")
        jobs = []
        for number, f in enumerate(files):
            desc = str(f.relative_to(raw / "amass" / "smplx_g")).removesuffix("_stageii.npz")
            source = desc.split("/")[0]
            kind = kinds.kind(desc.replace("_", " ").replace("-", " ").replace("/", " "))
            jobs.append((str(f), "smplx", f"amass{kind}_{''.join(c for c in source if c.isalnum())}_{number:05d}", desc))
        return jobs
    raise ValueError(f"unknown training set {name!r}")


CLIP_OUTPUTS = (".pkl", ".npz", "_report.json", "_ik.json")


def _fresh_output(out: Path, only: set[str] | None) -> None:
    """Empty ``out`` for a whole run, or remove the outputs of the clips in ``only``."""
    if only is None:
        if out.exists():
            shutil.rmtree(out)
    else:
        for stem in only:
            for suffix in CLIP_OUTPUTS:
                (out / f"{stem}{suffix}").unlink(missing_ok=True)
    out.mkdir(parents=True, exist_ok=True)


def clip_names(name: str, root: Path = ROOT) -> set[str]:
    """The clip names a training set's retargeting makes."""
    return {stem for _, _, stem, _ in _jobs(name, root / "raw")}


def _retarget_job(args):
    path, fmt, out_dir, stem, smplx_dir = args
    from mjlab_microban.wholebody.retarget import core

    t0 = time.time()
    try:
        report = core.run(path, fmt, FPS, Path(out_dir), stem=stem, smplx_model_dir=smplx_dir)
        return stem, None, time.time() - t0, report["frames"]
    except Exception:  # noqa: BLE001  (a clip that fails is listed in errors.json, the rest go on)
        return stem, traceback.format_exc(), time.time() - t0, 0


def retarget(name: str, root: Path = ROOT, workers: int = 12, only: set[str] | None = None) -> dict:
    """Retarget one training set into ``root/gmr/<name>/`` (``only``: just these clip names)."""
    raw, out = root / "raw", root / "gmr" / name
    items = _jobs(name, raw)
    if not items:
        raise FileNotFoundError(f"no source files for {name} under {raw}")
    if only is not None:
        only = only & {stem for _, _, stem, _ in items}
        if not only:
            return {}
    previous = {}
    if only is not None and (out / "errors.json").exists():
        previous = {stem: error for stem, error in json.loads((out / "errors.json").read_text()).items()
                    if stem not in only}
    _fresh_output(out, only)
    if only is None:
        (out / "manifest.json").write_text(json.dumps(
            [{"name": s, "source": os.path.relpath(f, raw), "description": d} for f, _, s, d in items],
            ensure_ascii=False, indent=0))
    smplx_dir = str(raw / "body_models" / "smplx")
    jobs = [(f, fm, str(out), s, smplx_dir) for f, fm, s, _ in sorted(items, key=lambda x: -os.path.getsize(x[0]))
            if only is None or s in only]  # the longest first, so the workers end together
    errors, t0 = dict(previous), time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_retarget_job, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            stem, error, _, _ = future.result()
            if error:
                errors[stem] = error
            if done % 100 == 0 or done == len(jobs):
                print(f"{name}: {done}/{len(jobs)} {time.time() - t0:.0f}s errors={len(errors)}", flush=True)
    (out / "errors.json").write_text(json.dumps(dict(sorted(errors.items())), indent=1))
    return errors


def _pico_job(item):
    _, name, fps, rp, rq, jp, ref, out_dir, g1_xml = item
    from mjlab_microban.wholebody.retarget import core, pico

    frames = pico.normalize(pico.g1_frames(pico.g1_model(g1_xml), rp, rq, jp))
    report = core.retarget_frames(name, "g1", frames, fps, ref, Path(out_dir), 0.0, (pico.NAMES, pico.PARENTS))
    return name, report["frames"]


def retarget_pico(root: Path = ROOT, workers: int = 8) -> dict:
    """Retarget the PICO recordings (``root/raw/pico``) into ``root/gmr/pico_eval/``."""
    from mjlab_microban.wholebody.retarget import pico

    raw, out = root / "raw", root / "gmr" / "pico_eval"
    g1_xml = str(raw / "g1" / "g1_mocap_29dof.xml")
    model = pico.g1_model(g1_xml)
    sources = pico.load_sources(raw / "pico", pico.g1_joint_names(model))
    if not sources:
        raise FileNotFoundError(f"no PICO recordings under {raw / 'pico'}")
    refs, chosen = pico.references(sources, model)
    _fresh_output(out, None)
    items = [(s, n, f, rp, rq, jp, refs[s], str(out), g1_xml) for s, n, f, rp, rq, jp in sources]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        frames = dict(pool.map(_pico_job, items))
    (out / "manifest.json").write_text(json.dumps(
        {"reference_stance": chosen,
         "clips": [{"name": n, "source": s, "frames": frames[n]} for s, n, *_ in sources]}, indent=0))
    return frames


def long_runs(mask: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(len(mask), bool)
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a >= n:
            out[a:b] = True
    return out


def kept(mask: np.ndarray, fps: float, pad_s: float, min_frames: int) -> list[tuple[int, int]]:
    pad = int(round(pad_s * fps))
    drop = np.zeros(len(mask), bool)
    for t in np.flatnonzero(mask):
        drop[max(0, t - pad):t + pad + 1] = True
    edges = np.flatnonzero(np.diff(np.r_[0, (~drop).astype(int), 0]))
    return [(int(s), int(e)) for s, e in zip(edges[::2], edges[1::2]) if e - s >= min_frames]


def finalize(name: str, root: Path = ROOT) -> dict:
    """Copy ``root/gmr/<name>/*.pkl`` to ``root/<name>/retarget/``, cutting arm-branch flips of MIN_FLIP_RUN+
    frames (a cut clip becomes ``<name>_p<k>.pkl``; the kind and the evaluation split do not change)."""
    from mjlab_microban.wholebody.data.preprocess import ExclusionSpec

    spec = ExclusionSpec()
    src, dst = root / "gmr" / name, root / name / "retarget"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    rows, total, cut = [], 0, 0
    for pk in sorted(src.glob("*.pkl")):
        flip = long_runs(np.load(pk.with_suffix(".npz"))["flip"], MIN_FLIP_RUN)
        with open(pk, "rb") as stream:
            data = pickle.load(stream)
        T = len(flip)
        total += T
        if not flip.any():
            shutil.copy(pk, dst / pk.name)
            continue
        segments = kept(flip, float(data["fps"]), spec.pad_s, spec.min_frames)
        n_kept = sum(e - s for s, e in segments)
        cut += T - n_kept
        rows.append({"clip": pk.stem, "frames": T, "flip_frames": int(flip.sum()), "kept_frames": n_kept, "pieces": segments})
        for k, (s, e) in enumerate(segments):
            piece = {key: (v[s:e] if isinstance(v, np.ndarray) and len(v) == T else v) for key, v in data.items()}
            with open(dst / f"{pk.stem}_p{k}.pkl", "wb") as stream:
                pickle.dump(piece, stream)
    summary = {"dataset": name, "frames": total, "cut_for_arm_flip": cut, "cut_share": round(cut / max(total, 1), 4),
               "clips_cut": len(rows), "files_written": len(list(dst.glob("*.pkl")))}
    (root / name / "finalize.json").write_text(json.dumps({"summary": summary, "clips": rows}, indent=1))
    return summary


def build(name: str, root: Path = ROOT, workers: int = 12) -> dict:
    """The training side's shards of one set (``pico_eval``: evaluation only)."""
    from mjlab_microban.wholebody.data.build import build_dataset

    return build_dataset(root / name / "retarget", root / name, workers=workers, eval_only=name == "pico_eval")


def mix(names, root: Path = ROOT) -> None:
    """``root/mix/{train,eval}/<set>.h5``: links to the training sets' shards."""
    for part in ("train", "eval"):
        folder = root / "mix" / part
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        for name in names:
            (folder / f"{name}.h5").symlink_to(Path("..") / ".." / name / part / "motion.h5")


def _quality_one(pkl: str):
    from mjlab_microban.wholebody.data import build as data_build
    from mjlab_microban.wholebody.data.preprocess import (
        REASONS,
        ExclusionSpec,
        frame_flags,
        kept_segments,
    )

    spec = ExclusionSpec()
    p = Path(pkl)
    checker = data_build._checker()
    clip = data_build.load_retarget_pkl(p, checker.joint_names)
    kin = checker.run(clip["root_pos"], clip["root_quat_w"], clip["joint_pos"])
    flags = frame_flags(kin, clip["fps"], spec)
    keep = np.zeros(len(next(iter(flags.values()))), bool)
    for s, e in kept_segments(flags, clip["fps"], spec):
        keep[s:e] = True
    drop = ~keep
    rep = json.loads(p.with_name(p.stem + "_report.json").read_text())
    z = np.load(p.with_suffix(".npz"))
    flip = z["flip"]
    tj = np.r_[z["target_jump"], False] | np.r_[False, z["target_jump"]]
    herr = z["hand_err"]
    q = clip["joint_pos"]
    edge = (np.minimum(q - checker.joint_lower, checker.joint_upper - q) < np.radians(1.0)).any(1)
    return {
        "name": p.stem, "frames": len(drop), "fps": clip["fps"],
        "train_flagged": {r: int(flags[r].sum()) for r in REASONS},
        "train_dropped": int(drop.sum()),
        "self_penetration_mm_max": round(1000 * float(kin.self_penetration_m.max()), 1),
        "lowest_point_mm_min_max": [round(1000 * float(kin.lowest_point_z.min()), 1), round(1000 * float(kin.lowest_point_z.max()), 1)],
        "joint_excess_deg_max": round(float(np.degrees(kin.joint_excess_rad.max())), 2),
        "joint_edge_frames": int(edge.sum()),
        "joint_speed_rad_s_p99_max": rep["joint_speed_rad_s_p99_max"],
        "slide_mm_s_med_p90": rep["contact_ankle_slide_vs_human_mm_s_med_p90"],
        "arm_flip_frames": int(flip.sum()), "arm_flip_longest": rep["arm_flip_frames_longest"][1],
        "arm_flip_frames_kept": int((flip & ~drop).sum()),
        "target_jump_frames": int(tj.sum()), "target_jump_frames_kept": int((tj & ~drop).sum()),
        "hand_err_over_30mm_frames": int((herr > 0.03).sum()), "hand_err_over_30mm_kept": int(((herr > 0.03) & ~drop).sum()),
        "retarget_joint_jump_frames": rep["joint_jump_frames"],
        "pos_err_mm_mean": {k: v[0] for k, v in rep["pos_err_mm_mean_p95"].items()},
    }


def quality(src: Path, workers: int = 8) -> dict:
    """Per clip of a retargeting output folder: the retargeting's own marks (flips, target jumps, hand error)
    beside the training rules (preprocess.py), and how many of the marks the rules leave in."""
    from mjlab_microban.wholebody.data.preprocess import REASONS

    pkls = sorted(str(p) for p in Path(src).glob("*.pkl"))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_quality_one, pkls, chunksize=4))
    tot = sum(r["frames"] for r in rows)
    share = lambda key: round(sum(r[key] for r in rows) / tot, 4)
    summary = {
        "clips": len(rows), "frames": tot, "hours": round(sum(r["frames"] / r["fps"] for r in rows) / 3600, 2),
        "train_flagged_share": {k: round(sum(r["train_flagged"][k] for r in rows) / tot, 4) for k in REASONS},
        "train_dropped_share": share("train_dropped"),
        "arm_flip_share": share("arm_flip_frames"), "arm_flip_kept_share": share("arm_flip_frames_kept"),
        "target_jump_share": share("target_jump_frames"), "target_jump_kept_share": share("target_jump_frames_kept"),
        "hand_err_over_30mm_share": share("hand_err_over_30mm_frames"),
        "hand_err_over_30mm_kept_share": share("hand_err_over_30mm_kept"),
        "joint_edge_share": share("joint_edge_frames"),
        "self_penetration_mm_max": max(r["self_penetration_mm_max"] for r in rows),
        "joint_excess_deg_max": max(r["joint_excess_deg_max"] for r in rows),
        "lowest_point_mm_min": min(r["lowest_point_mm_min_max"][0] for r in rows),
    }
    return {"summary": summary, "clips": rows}
