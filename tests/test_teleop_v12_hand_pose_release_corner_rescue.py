"""CPU-only tests for the pose-release variant of the model9900 corner rescue."""

from __future__ import annotations

import unittest

import torch

from mjlab_microban.tasks.microban_teleop_v12_corner_rescue import (
    MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY,
    MICROBAN_TELEOP_V12_CORNER_RESCUE_MARKER_REVISION,
    MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_CORNER_RESCUE_MARKER_REVISION,
    MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_LF_RB_PROBABILITY,
    corner_pair_selection,
    corner_rescue_marker,
    is_hand_pose_release_corner_rescue_marker,
    validate_corner_rescue_canonical_lineage,
    validate_corner_rescue_lineage_marker,
)
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION,
    MICROBAN_TELEOP_V12_RECIPE_REVISION,
)
from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release_lineage import (
    HAND_POSE_RELEASE_LINEAGE_FRESH,
    HAND_POSE_RELEASE_LINEAGE_FRESH_CORNER_RESCUE,
    hand_pose_release_lineage,
)

PARENT_SHA = "a" * 64
REPORT_SHA = "b" * 64


def _pr_marker(failed=("hand_tracking_rms", "hand_tracking_p95")) -> dict:
    return corner_rescue_marker(
        parent_checkpoint_sha256=PARENT_SHA,
        parent_strict_tracking_report_sha256=REPORT_SHA,
        hand_pose_release=True,
        parent_strict_failed_checks=failed,
    )


def _pr_infos(marker: dict | None) -> dict:
    infos = {
        "microban_teleop_recipe_revision": (
            MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION
        )
    }
    if marker is not None:
        infos[MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY] = marker
    return infos


class HandPoseReleaseCornerRescueTest(unittest.TestCase):
    def test_marker_variant_round_trips_and_records_failed_checks(self) -> None:
        marker = _pr_marker()
        self.assertTrue(is_hand_pose_release_corner_rescue_marker(marker))
        self.assertEqual(
            marker["revision"],
            MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_CORNER_RESCUE_MARKER_REVISION,
        )
        self.assertEqual(
            marker["parent_strict_failed_checks"],
            ["hand_tracking_p95", "hand_tracking_rms"],
        )
        self.assertEqual(
            marker["source_recipe_revision"],
            MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION,
        )
        self.assertEqual(validate_corner_rescue_lineage_marker(marker), marker)
        tampered = dict(marker)
        tampered["sampler_probabilities"] = {"left_forward_right_backward": 0.9}
        with self.assertRaises(ValueError):
            validate_corner_rescue_lineage_marker(tampered)

    def test_canonical_marker_is_unchanged(self) -> None:
        canonical = corner_rescue_marker(
            parent_checkpoint_sha256=PARENT_SHA,
            parent_strict_tracking_report_sha256=REPORT_SHA,
        )
        self.assertEqual(
            canonical["revision"], MICROBAN_TELEOP_V12_CORNER_RESCUE_MARKER_REVISION
        )
        self.assertEqual(canonical["parent_strict_failed_checks"], ["hand_tracking_rms"])
        self.assertEqual(
            canonical["source_recipe_revision"], MICROBAN_TELEOP_V12_RECIPE_REVISION
        )
        self.assertFalse(is_hand_pose_release_corner_rescue_marker(canonical))

    def test_parent_failed_checks_must_be_hand_accuracy_only(self) -> None:
        for failed in ((), ("falls",), ("hand_tracking_rms", "foot_tracking_rms")):
            with self.subTest(failed=failed), self.assertRaises(ValueError):
                _pr_marker(failed)

    def test_lineage_accepts_only_model9999_and_descendants(self) -> None:
        marker = _pr_marker()
        self.assertEqual(
            hand_pose_release_lineage(_pr_infos(None), iteration=9900),
            HAND_POSE_RELEASE_LINEAGE_FRESH,
        )
        for iteration in (9999, 10099, 14999):
            with self.subTest(iteration=iteration):
                self.assertEqual(
                    hand_pose_release_lineage(_pr_infos(marker), iteration=iteration),
                    HAND_POSE_RELEASE_LINEAGE_FRESH_CORNER_RESCUE,
                )
                self.assertEqual(
                    validate_corner_rescue_canonical_lineage(
                        _pr_infos(marker), iteration=iteration
                    ),
                    marker,
                )
        for iteration in (9901, 9950, 9998):
            with self.subTest(iteration=iteration), self.assertRaises(ValueError):
                hand_pose_release_lineage(_pr_infos(marker), iteration=iteration)

    def test_markers_cannot_cross_recipes(self) -> None:
        canonical = corner_rescue_marker(
            parent_checkpoint_sha256=PARENT_SHA,
            parent_strict_tracking_report_sha256=REPORT_SHA,
        )
        with self.assertRaises(ValueError):
            hand_pose_release_lineage(_pr_infos(canonical), iteration=10099)
        v11_infos = {
            "microban_teleop_recipe_revision": MICROBAN_TELEOP_V12_RECIPE_REVISION,
            MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY: _pr_marker(),
        }
        with self.assertRaises(ValueError):
            validate_corner_rescue_canonical_lineage(v11_infos, iteration=10099)

    def test_sampler_mix(self) -> None:
        selector = torch.linspace(0.0, 0.999999, 200_001, dtype=torch.float64)
        choice = corner_pair_selection(
            selector,
            lf_rb_probability=MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_LF_RB_PROBABILITY,
        )
        fractions = [float((choice == k).double().mean()) for k in (0, 1, 2)]
        self.assertAlmostEqual(fractions[0], 0.05, places=3)
        self.assertAlmostEqual(fractions[1], 0.60, places=3)
        self.assertAlmostEqual(fractions[2], 0.35, places=3)
        with self.assertRaises(ValueError):
            corner_pair_selection(selector, lf_rb_probability=0.5)


if __name__ == "__main__":
    unittest.main()
