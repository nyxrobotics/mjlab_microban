# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""CPU-only tests for the locomotion-prior geometric suitability gate."""

from __future__ import annotations

import copy
import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import mujoco
import numpy as np

from mjlab_microban.locomotion_prior_suitability import (
    DEFAULT_LOCOMOTION_PRIOR_PATH,
    DEFAULT_ROBOT_XML_PATH,
    DEFAULT_ROBOT_XML_SHA256,
    FOOT_COLLISION_GEOM_NAMES,
    _foot_geom_ids,
    _sole_min_world_z,
    audit_locomotion_prior_suitability,
    publish_suitability_receipt,
    serialize_suitability_receipt,
    sha256_file,
    validate_suitability_receipt_digest,
)
from mjlab_microban.scripts.audit_locomotion_prior_suitability import main


def _write_provenance_candidate(
    output: Path,
    *,
    omit: str | None = None,
    overrides: dict[str, object] | None = None,
    root_z_offset_m: float = 0.0,
    retarget_model_sha256: str = DEFAULT_ROBOT_XML_SHA256,
) -> str:
    with np.load(DEFAULT_LOCOMOTION_PRIOR_PATH, allow_pickle=False) as archive:
        payload = {name: np.asarray(archive[name]).copy() for name in archive.files}
    payload.update(
        {
            "source_capture_sha256": np.asarray("1" * 64),
            "retarget_model_sha256": np.asarray(retarget_model_sha256),
        }
    )
    if overrides is not None:
        payload.update({name: np.asarray(value) for name, value in overrides.items()})
    if root_z_offset_m:
        payload["body_pos_w"][:, 0, 2] += root_z_offset_m
    if omit is not None:
        del payload[omit]
    np.savez(output, **payload)
    return sha256_file(output)


class ExactBoxCornerFkTest(unittest.TestCase):
    def test_rotated_box_uses_all_eight_corners_in_world_coordinates(self) -> None:
        model = mujoco.MjModel.from_xml_string(
            """
            <mujoco model="corner-test">
              <worldbody>
                <body name="box_body">
                  <geom name="box" type="box" pos="0 0 0.1"
                        size="0.04 0.01 0.005"
                        quat="0.9238795325112867 0 0.3826834323650898 0"/>
                </body>
              </worldbody>
            </mujoco>
            """
        )
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "box")

        observed = _sole_min_world_z(model, data, (geom_id,))
        expected = 0.1 - (0.04 + 0.005) / math.sqrt(2.0)
        self.assertAlmostEqual(observed, expected, places=14)
        self.assertNotAlmostEqual(observed, float(data.geom_xpos[geom_id, 2]))

    def test_real_model_resolves_exactly_six_numbered_boxes_per_foot(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(DEFAULT_ROBOT_XML_PATH))
        resolved = _foot_geom_ids(model)

        self.assertEqual(set(resolved), {"left", "right"})
        self.assertEqual(sum(len(ids) for ids in resolved.values()), 12)
        for side, ids in resolved.items():
            self.assertEqual(len(ids), 6)
            self.assertEqual(
                tuple(
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                    for geom_id in ids
                ),
                FOOT_COLLISION_GEOM_NAMES[side],
            )
            self.assertTrue(
                all(
                    model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX
                    for geom_id in ids
                )
            )


class TrackedLocomotionPriorSuitabilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary_directory = tempfile.TemporaryDirectory()
        cls.candidate = Path(cls.temporary_directory.name) / "candidate.npz"
        cls.candidate_sha256 = _write_provenance_candidate(cls.candidate)
        cls.report = audit_locomotion_prior_suitability(
            cls.candidate,
            expected_prior_sha256=cls.candidate_sha256,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary_directory.cleanup()

    def test_receipt_binds_exact_tracked_inputs_and_frame_range(self) -> None:
        report = self.report
        self.assertEqual(report["schema_version"], 3)
        self.assertEqual(report["configuration"]["start_frame_inclusive"], 109)
        self.assertEqual(report["configuration"]["end_frame_inclusive"], 267)
        self.assertEqual(report["aggregate"]["frame_count"], 159)
        self.assertEqual(report["frames"][0]["frame"], 109)
        self.assertEqual(report["frames"][-1]["frame"], 267)
        self.assertEqual(
            report["inputs"]["locomotion_prior"]["sha256"],
            self.candidate_sha256,
        )
        self.assertEqual(
            report["inputs"]["robot_xml"]["sha256"],
            DEFAULT_ROBOT_XML_SHA256,
        )

    def test_all_static_checks_match_their_declared_thresholds(self) -> None:
        report = self.report
        checks = report["aggregate"]["checks"]

        self.assertEqual(
            checks["grounding_error_m_max"]["passed"],
            checks["grounding_error_m_max"]["value"]
            <= checks["grounding_error_m_max"]["maximum"],
        )
        for name in (
            "relative_swing_clearance_max_m",
            "relative_swing_clearance_p95_m",
        ):
            self.assertEqual(
                checks[name]["passed"], checks[name]["value"] >= checks[name]["minimum"]
            )
        passed = all(check["passed"] for check in checks.values())
        self.assertEqual(report["summary"]["passed"], passed)
        self.assertEqual(report["status"], "pass" if passed else "fail")

    def test_producer_provenance_is_required_and_bound_to_robot_xml(self) -> None:
        provenance = self.report["inputs"]["locomotion_prior"]["provenance"]
        self.assertEqual(provenance["source_capture_sha256"], "1" * 64)
        self.assertEqual(provenance["retarget_model_sha256"], DEFAULT_ROBOT_XML_SHA256)
        self.assertEqual(
            provenance["retarget_recipe"], "microban-bilateral-sagittal-sole-v2"
        )
        self.assertEqual(provenance["swing_target_boost_m"], 0.0)
        self.assertGreater(provenance["retarget_root_translation_scale"], 0.0)
        self.assertGreater(provenance["source_swing"]["left_frame_count"], 0)
        self.assertGreater(provenance["source_swing"]["right_frame_count"], 0)
        self.assertGreaterEqual(provenance["source_swing"]["transition_count"], 2)

        missing = Path(self.temporary_directory.name) / "missing-provenance.npz"
        missing_sha256 = _write_provenance_candidate(
            missing, omit="source_capture_sha256"
        )
        with self.assertRaisesRegex(ValueError, "missing fields"):
            audit_locomotion_prior_suitability(
                missing, expected_prior_sha256=missing_sha256
            )

        mismatched = Path(self.temporary_directory.name) / "wrong-model.npz"
        mismatched_sha256 = _write_provenance_candidate(
            mismatched, retarget_model_sha256="0" * 64
        )
        with self.assertRaisesRegex(ValueError, "does not match robot XML"):
            audit_locomotion_prior_suitability(
                mismatched, expected_prior_sha256=mismatched_sha256
            )

    def test_sampling_profile_recipe_and_joint_order_fail_closed(self) -> None:
        cases = (
            ("wrong-fps", {"fps": np.asarray([49.0], dtype=np.float32)}, "50 Hz"),
            (
                "wrong-profile",
                {"retarget_profile": "tracking"},
                "retarget_profile mismatch",
            ),
            (
                "wrong-recipe",
                {"locomotion_prior_retarget_recipe": "unreviewed-recipe"},
                "retarget recipe mismatch",
            ),
        )
        for stem, overrides, message in cases:
            with self.subTest(stem=stem):
                candidate = Path(self.temporary_directory.name) / f"{stem}.npz"
                digest = _write_provenance_candidate(candidate, overrides=overrides)
                with self.assertRaisesRegex(ValueError, message):
                    audit_locomotion_prior_suitability(
                        candidate, expected_prior_sha256=digest
                    )

        with np.load(DEFAULT_LOCOMOTION_PRIOR_PATH, allow_pickle=False) as archive:
            swapped_names = np.asarray(archive["joint_names"]).copy()
        swapped_names[[0, 1]] = swapped_names[[1, 0]]
        candidate = Path(self.temporary_directory.name) / "wrong-joint-order.npz"
        digest = _write_provenance_candidate(
            candidate, overrides={"joint_names": swapped_names}
        )
        with self.assertRaisesRegex(ValueError, "joint order"):
            audit_locomotion_prior_suitability(candidate, expected_prior_sha256=digest)

    def test_per_frame_grounding_is_explicit_and_clearance_is_root_invariant(
        self,
    ) -> None:
        frame = self.report["frames"][0]

        self.assertAlmostEqual(
            frame["grounding_translation_m"],
            -frame["lower_sole_min_world_z_m"],
            places=14,
        )
        grounded_heights = (
            frame["grounded_left_sole_height_m"],
            frame["grounded_right_sole_height_m"],
        )
        self.assertAlmostEqual(min(grounded_heights), 0.0, places=14)
        self.assertAlmostEqual(
            max(grounded_heights),
            frame["relative_swing_clearance_m"],
            places=14,
        )

    def test_receipt_and_serialization_are_deterministic(self) -> None:
        repeated = audit_locomotion_prior_suitability(
            self.candidate,
            expected_prior_sha256=self.candidate_sha256,
        )

        self.assertEqual(repeated, self.report)
        self.assertEqual(
            serialize_suitability_receipt(repeated),
            serialize_suitability_receipt(self.report),
        )
        validate_suitability_receipt_digest(repeated)

    def test_payload_digest_detects_tampering(self) -> None:
        tampered = copy.deepcopy(self.report)
        tampered["frames"][0]["relative_swing_clearance_m"] += 0.001

        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            validate_suitability_receipt_digest(tampered)

    def test_wrong_expected_input_digest_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "locomotion prior SHA-256 mismatch"):
            audit_locomotion_prior_suitability(
                self.candidate,
                expected_prior_sha256="0" * 64,
            )


class ReceiptPublicationTest(unittest.TestCase):
    def test_receipt_publication_is_atomic_and_requires_force_to_replace(self) -> None:
        report = {"receipt_payload_sha256": "0" * 64, "status": "fail"}
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "receipt.json"
            publish_suitability_receipt(output, report, force=False)
            expected = serialize_suitability_receipt(report)
            self.assertEqual(output.read_text(encoding="utf-8"), expected)

            with self.assertRaisesRegex(FileExistsError, "refusing to replace"):
                publish_suitability_receipt(output, report, force=False)
            self.assertEqual(output.read_text(encoding="utf-8"), expected)

            changed = copy.deepcopy(report)
            changed["status"] = "changed-for-publication-test"
            publish_suitability_receipt(output, changed, force=True)
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                serialize_suitability_receipt(changed),
            )


class SuitabilityCliTest(unittest.TestCase):
    def test_cli_writes_failing_static_receipt_and_returns_two(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            candidate = Path(temporary_directory) / "candidate.npz"
            candidate_sha256 = _write_provenance_candidate(
                candidate, root_z_offset_m=0.02
            )
            output = Path(temporary_directory) / "receipt.json"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = main(
                    [
                        "--prior",
                        str(candidate),
                        "--expected-prior-sha256",
                        candidate_sha256,
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(exit_code, 2)
            self.assertEqual(stderr.getvalue(), "")
            summary = json.loads(stdout.getvalue())
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "fail")
            self.assertEqual(report["status"], "fail")
            validate_suitability_receipt_digest(report)


if __name__ == "__main__":
    unittest.main()
