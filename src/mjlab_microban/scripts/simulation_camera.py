# Copyright 2026 Marc Duclusaud
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Loopback-only SBS MJPEG camera for live Microban simulation.

The camera geometry is synthetic and exact: two MuJoCo pinhole views are built
from the CAD-derived left/right camera sites.  Their declared dimensions, FOV
and symmetric ray-tangent bounds are the values consumed by the PICO WebXR
renderer; no physical-camera calibration claim is made.

The MJLab entity assembler intentionally keeps sites but does not keep MJCF
``camera`` elements.  Rendering from the sites avoids a second, subtly
different camera transform and works with the model actually used by the task.
"""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from typing import Any

import mujoco
import numpy as np
from PIL import Image

LEFT_CAMERA_SITE = "camera_left"
RIGHT_CAMERA_SITE = "camera_right"
EYE_WIDTH_PX = 640
EYE_HEIGHT_PX = 480
EYE_ASPECT = EYE_WIDTH_PX / EYE_HEIGHT_PX
# PICO 4 Ultra's published nominal display FoV is 105 degrees.  Synthetic
# rendering has no physical-lens constraint, so keep both source axes wider;
# the browser's live projection-matrix coverage gate remains authoritative.
PICO_NOMINAL_FOV_DEG = 105.0
# Render a conservative source frustum that covers the PICO per-eye projection
# observed by the WebXR path.  The browser still checks every live projection
# matrix and fails closed if a future device/view exceeds these ray bounds.
HORIZONTAL_FOV_DEG = 122.0
HORIZONTAL_TAN = math.tan(math.radians(HORIZONTAL_FOV_DEG) / 2.0)
VERTICAL_TAN = HORIZONTAL_TAN / EYE_ASPECT
VERTICAL_FOV_DEG = math.degrees(2.0 * math.atan(VERTICAL_TAN))
TAN_BOUNDS = (HORIZONTAL_TAN, HORIZONTAL_TAN, VERTICAL_TAN, VERTICAL_TAN)
EXPECTED_BASELINE_M = math.sqrt(0.059016**2 + 0.000051**2)
MJPEG_BOUNDARY = "microban-sim-frame"


@dataclass(frozen=True)
class SimulationCameraGeometry:
    """Exact browser-facing geometry for the generated SBS image."""

    eye_width_px: int = EYE_WIDTH_PX
    eye_height_px: int = EYE_HEIGHT_PX
    sbs_width_px: int = EYE_WIDTH_PX * 2
    sbs_height_px: int = EYE_HEIGHT_PX
    horizontal_fov_deg: float = HORIZONTAL_FOV_DEG
    vertical_fov_deg: float = VERTICAL_FOV_DEG
    eye_aspect: float = EYE_ASPECT
    left_eye_first: bool = True
    left_tan_bounds: tuple[float, float, float, float] = TAN_BOUNDS
    right_tan_bounds: tuple[float, float, float, float] = TAN_BOUNDS
    calibrated: bool = True
    calibration_kind: str = "exact_synthetic_pinhole_geometry"
    baseline_m: float = EXPECTED_BASELINE_M


def camera_geometry_dict() -> dict[str, Any]:
    return asdict(SimulationCameraGeometry())


def _site_id(model: mujoco.MjModel, name: str) -> int:
    """Resolve a site in raw MJCF or in MJLab's namespaced entity model."""

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if site_id >= 0:
        return site_id
    suffix = f"/{name}"
    matches = [
        index
        for index in range(model.nsite)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, index) or "").endswith(
            suffix
        )
    ]
    return matches[0] if len(matches) == 1 else -1


def validate_simulation_camera_model(model: mujoco.MjModel) -> None:
    """Reject model drift from the geometry advertised to PICO Browser."""

    site_ids: list[int] = []
    for name in (LEFT_CAMERA_SITE, RIGHT_CAMERA_SITE):
        site_id = _site_id(model, name)
        if site_id < 0:
            raise ValueError(f"Simulation camera site is missing from MJCF: {name}")
        site_ids.append(site_id)

    left_id, right_id = site_ids
    if model.site_bodyid[left_id] != model.site_bodyid[right_id]:
        raise ValueError("Simulation stereo camera sites must share one rigid body")
    baseline = float(np.linalg.norm(model.site_pos[left_id] - model.site_pos[right_id]))
    if not math.isclose(baseline, EXPECTED_BASELINE_M, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(
            f"Simulation camera baseline {baseline:.9f} m does not match "
            f"{EXPECTED_BASELINE_M:.9f} m"
        )
    if not np.allclose(
        model.site_quat[left_id],
        model.site_quat[right_id],
        atol=1.0e-12,
        rtol=0.0,
    ):
        raise ValueError("Simulation stereo camera site orientations differ")


def _point_scene_camera_at_site(
    scene: mujoco.MjvScene,
    data: mujoco.MjData,
    site_id: int,
) -> None:
    """Set the offscreen camera pair to an exact pinhole at one site.

    Microban's CAD sites use +X as optical forward and +Z as optical up.  The
    two GL cameras are made identical because the SBS stereo separation is
    supplied explicitly by rendering once from each physical site.
    """

    rotation = data.site_xmat[site_id].reshape(3, 3)
    position = data.site_xpos[site_id]
    forward = rotation[:, 0]
    up = rotation[:, 2]
    for camera in scene.camera:
        near = float(camera.frustum_near)
        camera.pos[:] = position
        camera.forward[:] = forward
        camera.up[:] = up
        camera.frustum_center = 0.0
        camera.frustum_width = 0.0
        camera.frustum_bottom = -near * VERTICAL_TAN
        camera.frustum_top = near * VERTICAL_TAN
        camera.orthographic = 0


class _FrameStore:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.jpeg: bytes | None = None
        self.sequence = 0
        self.captured_at_s = 0.0
        self.closed = False

    def publish(self, jpeg: bytes) -> None:
        if not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
            raise ValueError("Published camera frame is not a complete JPEG")
        with self.condition:
            self.jpeg = jpeg
            self.sequence += 1
            self.captured_at_s = time.monotonic()
            self.condition.notify_all()

    def wait_after(
        self, sequence: int, timeout_s: float
    ) -> tuple[int, bytes, float] | None:
        with self.condition:
            self.condition.wait_for(
                lambda: self.closed or self.sequence > sequence,
                timeout=timeout_s,
            )
            if self.closed or self.jpeg is None or self.sequence <= sequence:
                return None
            return self.sequence, self.jpeg, self.captured_at_s

    def close(self) -> None:
        with self.condition:
            self.closed = True
            self.condition.notify_all()


class _LoopbackHttpServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, port: int, frame_store: _FrameStore) -> None:
        super().__init__(("127.0.0.1", port), _MjpegHandler)
        self.frame_store = frame_store


class _MjpegHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @property
    def frame_store(self) -> _FrameStore:
        return self.server.frame_store  # type: ignore[attr-defined, no-any-return]

    def do_GET(self) -> None:
        if self.path == "/stream":
            self._stream()
        elif self.path == "/healthz":
            age_s = (
                None
                if self.frame_store.captured_at_s == 0.0
                else max(0.0, time.monotonic() - self.frame_store.captured_at_s)
            )
            self._json(
                {
                    "ok": self.frame_store.jpeg is not None
                    and not self.frame_store.closed,
                    "frame_sequence": self.frame_store.sequence,
                    "frame_age_s": age_s,
                    "geometry": camera_geometry_dict(),
                }
            )
        elif self.path == "/calibration.json":
            self._json(camera_geometry_dict())
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def _json(self, value: Any) -> None:
        payload = json.dumps(value, sort_keys=True).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _stream(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header(
            "Content-Type", f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}"
        )
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Connection", "close")
        self.end_headers()
        sequence = 0
        try:
            while True:
                frame = self.frame_store.wait_after(sequence, timeout_s=1.0)
                if frame is None:
                    if self.frame_store.closed:
                        return
                    continue
                sequence, jpeg, _captured_at_s = frame
                self.wfile.write(f"--{MJPEG_BOUNDARY}\r\n".encode())
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return

    def log_message(self, _format: str, *_args: Any) -> None:
        return


class StereoMjpegPublisher:
    """Render one simulation environment and publish loopback SBS MJPEG."""

    def __init__(
        self,
        env: Any,
        *,
        port: int = 8081,
        fps: float = 20.0,
        jpeg_quality: int = 82,
        clock: Any = time.monotonic,
    ) -> None:
        if not 1 <= port <= 65535:
            raise ValueError("camera port must be between 1 and 65535")
        if not math.isfinite(fps) or not 1.0 <= fps <= 30.0:
            raise ValueError("camera fps must be in [1, 30]")
        if not 1 <= jpeg_quality <= 95:
            raise ValueError("JPEG quality must be in [1, 95]")
        self.env = env
        self.clock = clock
        self.period_s = 1.0 / fps
        self.jpeg_quality = jpeg_quality
        self._next_capture_s = 0.0
        self._closed = False

        self.model = env.sim.mj_model
        validate_simulation_camera_model(self.model)
        self.camera_site_ids = tuple(
            _site_id(self.model, name) for name in (LEFT_CAMERA_SITE, RIGHT_CAMERA_SITE)
        )
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(
            self.model, height=EYE_HEIGHT_PX, width=EYE_WIDTH_PX
        )
        self.frame_store = _FrameStore()
        self.server = _LoopbackHttpServer(port, self.frame_store)
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            name="microban-sim-camera-http",
            daemon=True,
        )
        self.thread.start()

    @property
    def port(self) -> int:
        return int(self.server.server_address[1])

    def _sync_state(self) -> None:
        source = self.env.sim.data
        self.data.qpos[:] = source.qpos[0].detach().cpu().numpy()
        self.data.qvel[:] = source.qvel[0].detach().cpu().numpy()
        if self.model.na > 0:
            self.data.act[:] = source.act[0].detach().cpu().numpy()
        if self.model.nmocap > 0:
            self.data.mocap_pos[:] = source.mocap_pos[0].detach().cpu().numpy()
            self.data.mocap_quat[:] = source.mocap_quat[0].detach().cpu().numpy()
        mujoco.mj_forward(self.model, self.data)

    def capture_if_due(self) -> bool:
        if self._closed:
            return False
        now = self.clock()
        if now < self._next_capture_s:
            return False
        self._next_capture_s = now + self.period_s
        self._sync_state()
        eyes: list[np.ndarray] = []
        for site_id in self.camera_site_ids:
            self.renderer.update_scene(self.data)
            _point_scene_camera_at_site(self.renderer.scene, self.data, site_id)
            eyes.append(self.renderer.render().copy())
        sbs = np.concatenate(eyes, axis=1)
        output = BytesIO()
        Image.fromarray(sbs, mode="RGB").save(
            output,
            format="JPEG",
            quality=self.jpeg_quality,
            optimize=False,
        )
        self.frame_store.publish(output.getvalue())
        return True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.frame_store.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)
        self.renderer.close()


def webxr_camera_toml(port: int) -> str:
    """Return the exact config block for microban_teleop's camera proxy."""

    geometry = SimulationCameraGeometry()
    bounds = ", ".join(f"{value:.12g}" for value in TAN_BOUNDS)
    return "\n".join(
        (
            "[camera]",
            f'url = "http://127.0.0.1:{port}/stream"',
            f"horizontal_fov_deg = {geometry.horizontal_fov_deg:.12g}",
            f"vertical_fov_deg = {geometry.vertical_fov_deg:.12g}",
            f"eye_aspect = {geometry.eye_aspect:.12g}",
            f"eye_width_px = {geometry.eye_width_px}",
            f"eye_height_px = {geometry.eye_height_px}",
            "left_eye_first = true",
            f"left_tan_bounds = [{bounds}]",
            f"right_tan_bounds = [{bounds}]",
            # Valid only because this is exact pinhole render geometry, not an
            # unmeasured physical-lens claim.
            "calibrated = true",
        )
    )
