"""Draw a retargeted clip with MuJoCo on the CPU (Mesa llvmpipe through EGL; no GPU).

The human skeleton scaled by the root scale (blue) stands 0.3 m to the robot's
left, and the IK targets (red dots) are drawn over the robot.  Writes
<name>.mp4, an 8-frame contact sheet <name>_sheet.png and the joint angles
<name>_joints.png next to the npz.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/50_mesa.json")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from mjlab_microban.wholebody.retarget.model import gmr_model_xml

HUMAN_OFFSET = np.array([0.0, 0.30, 0.0])


def add_geom(scn, gtype, size, pos, mat, rgba):
    if scn.ngeom >= scn.maxgeom:
        return None
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.asarray(size, float), np.asarray(pos, float), np.asarray(mat, float).ravel(), np.asarray(rgba, np.float32))
    scn.ngeom += 1
    return g


def add_line(scn, a, b, width, rgba):
    g = add_geom(scn, mujoco.mjtGeom.mjGEOM_CAPSULE, [0, 0, 0], [0, 0, 0], np.eye(3), rgba)
    if g is not None:
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(a, float), np.asarray(b, float))


def draw_extras(scn, t, z, human_root_xy):
    # floor
    add_geom(scn, mujoco.mjtGeom.mjGEOM_PLANE, [3, 3, 0.01], [0, 0, 0], np.eye(3), [0.95, 0.95, 0.92, 1])
    # the human skeleton, beside the robot (its floor at z = 0)
    hp = z["human_pos"][t] + HUMAN_OFFSET
    for j, p in enumerate(z["human_parents"]):
        if p >= 0:
            add_line(scn, hp[p], hp[j], 0.003, [0.2, 0.4, 0.9, 1])
    # the IK targets
    for p in z["targets"][t]:
        add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.006, 0, 0], p, np.eye(3), [0.9, 0.1, 0.1, 0.8])


def render(npz_path: Path, width: int, height: int, fps_out: float | None):
    z = dict(np.load(npz_path, allow_pickle=True))
    qpos, fps = z["qpos"], float(z["fps"])
    model = mujoco.MjModel.from_xml_path(str(gmr_model_xml()))
    model.vis.global_.offwidth = max(width, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(height, model.vis.global_.offheight)
    model.vis.headlight.ambient[:] = [0.45, 0.45, 0.45]
    model.vis.headlight.diffuse[:] = [0.6, 0.6, 0.6]
    model.vis.headlight.specular[:] = [0.1, 0.1, 0.1]
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height, width, max_geom=2000)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = 0.75, 145.0, -12.0
    opt = mujoco.MjvOption()
    step = 1 if fps_out is None else max(1, int(round(fps / fps_out)))
    # the camera follows the root's xy, smoothed
    look = qpos[:, :2].copy()
    k = max(1, int(fps))
    kern = np.ones(k) / k
    look = np.stack([np.convolve(np.pad(look[:, i], (k // 2, k - 1 - k // 2), mode="edge"), kern, mode="valid") for i in range(2)], 1)
    frames = []
    for t in range(0, len(qpos), step):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        cam.lookat[:] = [look[t, 0], look[t, 1] + HUMAN_OFFSET[1] / 2, 0.13]
        renderer.update_scene(data, cam, opt)
        draw_extras(renderer.scene, t, z, None)
        frames.append(renderer.render().copy())
    stem = npz_path.with_suffix("")
    imageio.mimwrite(f"{stem}.mp4", frames, fps=fps / step, quality=7, macro_block_size=1)
    picks = np.linspace(0, len(frames) - 1, 8).astype(int)
    rows = [np.concatenate([frames[i] for i in picks[r * 4:(r + 1) * 4]], axis=1) for r in range(2)]
    imageio.imwrite(f"{stem}_sheet.png", np.concatenate(rows, axis=0)[::2, ::2])
    plot_joints(z, f"{stem}_joints.png")
    renderer.close()
    return f"{stem}.mp4"


def plot_joints(z, path):
    q, names, fps = z["qpos"], [str(n) for n in z["joint_names"]], float(z["fps"])
    t = np.arange(len(q)) / fps
    groups = [("hip_yaw", "hip_roll", "hip_pitch"), ("knee", "ankle_pitch", "ankle_roll"), ("shoulder_pitch", "shoulder_roll", "elbow")]
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    for ax, grp in zip(axes[:3], groups):
        for side, ls in (("left", "-"), ("right", "--")):
            for gname in grp:
                n = f"{side}_{gname}"
                ax.plot(t, np.degrees(q[:, 7 + names.index(n)]), ls, label=n, lw=1)
        ax.set_ylabel("deg")
        ax.legend(ncol=3, fontsize=7, loc="upper right")
        ax.grid(alpha=0.3)
    axes[3].plot(t, q[:, 2], label="root z [m]")
    axes[3].set_ylabel("m")
    axes[3].set_xlabel("s")
    axes[3].legend(fontsize=7)
    axes[3].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)
