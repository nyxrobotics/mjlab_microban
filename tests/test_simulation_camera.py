# Copyright 2026 Marc Duclusaud
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Contract tests for the simulation SBS camera companion."""

from __future__ import annotations

import http.client
import json
import math
import threading
import unittest
from pathlib import Path

import mujoco
import numpy as np

from mjlab_microban.scripts.simulation_camera import (
    CAMERA_SCHEMA,
    EXPECTED_BASELINE_M,
    EYE_ASPECT,
    EYE_HEIGHT_PX,
    EYE_WIDTH_PX,
    HORIZONTAL_FOV_DEG,
    MAX_FRAME_AGE_MS,
    MJPEG_BOUNDARY,
    PICO_NOMINAL_FOV_DEG,
    TAN_BOUNDS,
    VERTICAL_FOV_DEG,
    WARNING_OVERLAY_HEIGHT_PX,
    _apply_warning_overlay,
    _FrameStore,
    _LoopbackHttpServer,
    _point_scene_camera_at_site,
    camera_geometry_dict,
    camera_geometry_sha256,
    validate_simulation_camera_model,
    webxr_camera_toml,
)

ROBOT_XML = (
    Path(__file__).parents[1]
    / "src"
    / "mjlab_microban"
    / "robot"
    / "microban"
    / "robot.xml"
)


class CameraGeometryTests(unittest.TestCase):
    def test_advertised_intrinsics_are_exact_for_render_dimensions(self) -> None:
        geometry = camera_geometry_dict()
        self.assertEqual(geometry["schema"], CAMERA_SCHEMA)
        self.assertEqual(geometry["eye_width_px"], EYE_WIDTH_PX)
        self.assertEqual(geometry["eye_height_px"], EYE_HEIGHT_PX)
        self.assertEqual(geometry["sbs_width_px"], 2 * EYE_WIDTH_PX)
        self.assertEqual(geometry["eye_aspect"], EYE_ASPECT)
        self.assertEqual(geometry["horizontal_fov_deg"], HORIZONTAL_FOV_DEG)
        self.assertAlmostEqual(
            math.tan(math.radians(VERTICAL_FOV_DEG) / 2.0),
            math.tan(math.radians(HORIZONTAL_FOV_DEG) / 2.0) / EYE_ASPECT,
            places=12,
        )
        self.assertGreaterEqual(HORIZONTAL_FOV_DEG, PICO_NOMINAL_FOV_DEG)
        self.assertGreaterEqual(VERTICAL_FOV_DEG, PICO_NOMINAL_FOV_DEG)
        self.assertEqual(tuple(geometry["left_tan_bounds"]), TAN_BOUNDS)
        self.assertEqual(tuple(geometry["right_tan_bounds"]), TAN_BOUNDS)
        self.assertTrue(geometry["calibrated"])
        self.assertEqual(
            geometry["calibration_kind"], "exact_synthetic_pinhole_geometry"
        )

    def test_mjcf_camera_sites_match_advertised_geometry(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
        validate_simulation_camera_model(model)
        left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "camera_left")
        right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "camera_right")
        baseline = math.dist(model.site_pos[left], model.site_pos[right])
        self.assertAlmostEqual(baseline, EXPECTED_BASELINE_M, places=12)

    def test_scene_camera_uses_site_optical_frame_and_exact_fov(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "camera_left")
        renderer = mujoco.Renderer(model, height=EYE_HEIGHT_PX, width=EYE_WIDTH_PX)
        try:
            renderer.update_scene(data)
            _point_scene_camera_at_site(renderer.scene, data, site_id)
            rotation = data.site_xmat[site_id].reshape(3, 3)
            for camera in renderer.scene.camera:
                np.testing.assert_allclose(camera.pos, data.site_xpos[site_id])
                np.testing.assert_allclose(camera.forward, rotation[:, 0])
                np.testing.assert_allclose(camera.up, rotation[:, 2])
                self.assertAlmostEqual(
                    camera.frustum_top / camera.frustum_near,
                    TAN_BOUNDS[2],
                    places=7,
                )
                self.assertAlmostEqual(
                    camera.frustum_bottom / camera.frustum_near,
                    -TAN_BOUNDS[3],
                    places=7,
                )
        finally:
            renderer.close()

    def test_webxr_config_uses_loopback_and_exact_geometry(self) -> None:
        config = webxr_camera_toml(8081)
        self.assertIn('url = "http://127.0.0.1:8081/stream"', config)
        self.assertIn(f"eye_width_px = {EYE_WIDTH_PX}", config)
        self.assertIn(f"eye_height_px = {EYE_HEIGHT_PX}", config)
        self.assertIn("left_eye_first = true", config)
        self.assertIn("calibrated = true", config)

    def test_unaccepted_warning_is_burned_into_both_eye_images(self) -> None:
        frame = np.zeros(
            (EYE_HEIGHT_PX, EYE_WIDTH_PX * 2, 3), dtype=np.uint8
        )
        rendered = _apply_warning_overlay(
            frame, "UNACCEPTED | SIM ONLY | NO PHYSICAL OUTPUT"
        )
        self.assertEqual(rendered.shape, frame.shape)
        self.assertEqual(rendered.dtype, np.uint8)
        self.assertGreater(int(rendered[:WARNING_OVERLAY_HEIGHT_PX, :640].max()), 0)
        self.assertGreater(int(rendered[:WARNING_OVERLAY_HEIGHT_PX, 640:].max()), 0)
        self.assertTrue(
            np.array_equal(rendered[WARNING_OVERLAY_HEIGHT_PX:], frame[WARNING_OVERLAY_HEIGHT_PX:])
        )


class LoopbackMjpegServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = _FrameStore()
        self.server = _LoopbackHttpServer(0, self.store)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = int(self.server.server_address[1])

    def tearDown(self) -> None:
        self.store.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)

    def test_server_is_hard_bound_to_loopback(self) -> None:
        self.assertEqual(self.server.server_address[0], "127.0.0.1")

    def test_health_and_calibration_endpoints_report_exact_contract(self) -> None:
        self.store.publish(b"\xff\xd8test\xff\xd9")
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=2.0)
        connection.request("GET", "/healthz")
        response = connection.getresponse()
        payload = json.loads(response.read())
        connection.close()
        self.assertEqual(response.status, 200)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["frame_sequence"], 1)
        self.assertEqual(payload["geometry"]["eye_width_px"], EYE_WIDTH_PX)
        self.assertEqual(payload["geometry_sha256"], camera_geometry_sha256())

        connection = http.client.HTTPConnection(
            "127.0.0.1", self.port, timeout=2.0
        )
        connection.request("GET", "/calibration.json")
        response = connection.getresponse()
        calibration = json.loads(response.read())
        connection.close()
        self.assertEqual(response.status, 200)
        self.assertEqual(calibration["geometry"]["schema"], CAMERA_SCHEMA)
        self.assertEqual(
            calibration["geometry_sha256"], camera_geometry_sha256()
        )
        self.assertEqual(calibration["latest_frame_path"], "/frame.jpg")
        self.assertEqual(MAX_FRAME_AGE_MS, 30_000)
        self.assertEqual(calibration["max_frame_age_ms"], MAX_FRAME_AGE_MS)

    def test_latest_frame_endpoint_is_one_slot_and_freshness_annotated(self) -> None:
        first = b"\xff\xd8first\xff\xd9"
        second = b"\xff\xd8second\xff\xd9"
        self.store.publish(first)
        self.store.publish(second)

        connection = http.client.HTTPConnection(
            "127.0.0.1", self.port, timeout=2.0
        )
        connection.request("GET", "/frame.jpg?after=0&wait_ms=0")
        response = connection.getresponse()
        payload = response.read()
        headers = dict(response.getheaders())
        connection.close()

        self.assertEqual(response.status, 200)
        self.assertEqual(payload, second)
        self.assertEqual(headers["X-Microban-Camera-Schema"], CAMERA_SCHEMA)
        self.assertEqual(headers["X-Microban-Frame-Sequence"], "2")
        self.assertGreaterEqual(int(headers["X-Microban-Frame-Age-Ns"]), 0)
        self.assertEqual(
            headers["X-Microban-Geometry-SHA256"], camera_geometry_sha256()
        )
        self.assertEqual(headers["Cache-Control"], "no-store")

        connection = http.client.HTTPConnection(
            "127.0.0.1", self.port, timeout=2.0
        )
        connection.request("GET", "/frame.jpg?after=2&wait_ms=0")
        response = connection.getresponse()
        self.assertEqual(response.read(), b"")
        connection.close()
        self.assertEqual(response.status, 204)

    def test_latest_frame_endpoint_rejects_ambiguous_or_unbounded_query(self) -> None:
        for path in (
            "/frame.jpg?after=1&after=2",
            "/frame.jpg?after=-1",
            "/frame.jpg?wait_ms=1001",
            "/frame.jpg?unknown=1",
            "/frame.jpg?after=not-a-number",
        ):
            with self.subTest(path=path):
                connection = http.client.HTTPConnection(
                    "127.0.0.1", self.port, timeout=2.0
                )
                connection.request("GET", path)
                response = connection.getresponse()
                response.read()
                connection.close()
                self.assertEqual(response.status, 400)

    def test_stream_is_sbs_mjpeg_compatible(self) -> None:
        jpeg = b"\xff\xd8test\xff\xd9"
        self.store.publish(jpeg)
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=2.0)
        connection.request("GET", "/stream")
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        self.assertEqual(
            response.getheader("Content-Type"),
            f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}",
        )
        expected = (
            f"--{MJPEG_BOUNDARY}\r\n".encode()
            + b"Content-Type: image/jpeg\r\n"
            + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
            + jpeg
            + b"\r\n"
        )
        payload = response.read(len(expected))
        connection.close()
        self.assertIn(f"--{MJPEG_BOUNDARY}\r\n".encode(), payload)
        self.assertIn(b"Content-Type: image/jpeg\r\n", payload)
        self.assertIn(jpeg, payload)


if __name__ == "__main__":
    unittest.main()
