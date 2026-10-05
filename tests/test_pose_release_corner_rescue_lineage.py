"""Lineages the automatic 9999 escalation of scripts/retrain_all_for_home.py produces.

The pipeline continues a fresh pose-release chain from either a passing
pose-release model_9900 corner rescue (any registered mix: lf60, lf65, lf72, lf90)
or a retrained 7100->10000 attempt from the gated model_7099.  Every lineage
validator must accept both, for model_9999 and its ordinary descendants (the
10100 canary, the 15000 final and the deployment package), and refuse the
non-consumable or tampered variants.
"""

from __future__ import annotations

import unittest

from mjlab_microban.scripts.export_teleop_v12_deployment import _deployment_recipe_revision
from mjlab_microban.tasks.microban_teleop_v12_corner_rescue import (
    MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY,
    MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_CORNER_RESCUE_MIXES,
    corner_rescue_marker,
    validate_corner_rescue_canonical_lineage,
)
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION as POSE_RELEASE,
    MICROBAN_TELEOP_V12_RECIPE_REVISION as CANONICAL,
)
from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release_lineage import (
    HAND_POSE_RELEASE_LINEAGE_FRESH,
    HAND_POSE_RELEASE_LINEAGE_FRESH_CORNER_RESCUE,
    hand_pose_release_lineage,
)

MIXES = ("lf60", "lf65", "lf72", "lf90")


def rescue_infos(mix: str, failed=("hand_tracking_rms",)) -> dict:
    return {
        "microban_teleop_recipe_revision": POSE_RELEASE,
        MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY: corner_rescue_marker(
            parent_checkpoint_sha256="1" * 64,
            parent_strict_tracking_report_sha256="2" * 64,
            hand_pose_release=True,
            parent_strict_failed_checks=failed,
            pose_release_mix=mix,
        ),
    }


class PoseReleaseCornerRescueLineageTest(unittest.TestCase):
    def test_every_registered_mix_is_part_of_the_escalation(self):
        self.assertEqual(set(MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_CORNER_RESCUE_MIXES), set(MIXES))

    def test_rescue_model_9999_and_descendants_are_consumable(self):
        for mix in MIXES:
            for failed in (("hand_tracking_rms",), ("hand_tracking_p95", "hand_tracking_rms")):
                infos = rescue_infos(mix, failed)
                for iteration in (9999, 10099, 14999):
                    with self.subTest(mix=mix, failed=failed, iteration=iteration):
                        self.assertEqual(hand_pose_release_lineage(infos, iteration=iteration),
                                         HAND_POSE_RELEASE_LINEAGE_FRESH_CORNER_RESCUE)
                        marker = validate_corner_rescue_canonical_lineage(infos, iteration=iteration)
                        self.assertEqual(marker["sampler_mix"], mix)
                        self.assertEqual(marker["parent_strict_failed_checks"], sorted(failed))
                self.assertEqual(_deployment_recipe_revision(infos), POSE_RELEASE)

    def test_intermediate_rescue_saves_are_not_consumable(self):
        for mix in MIXES:
            with self.assertRaisesRegex(ValueError, "model_9999"):
                hand_pose_release_lineage(rescue_infos(mix), iteration=9950)

    def test_tampered_or_misplaced_markers_are_refused(self):
        infos = rescue_infos("lf72")
        infos[MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY]["sampler_mix"] = "lf90"
        with self.assertRaisesRegex(ValueError, "drifted"):
            hand_pose_release_lineage(infos, iteration=9999)
        infos = rescue_infos("lf90")
        infos[MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY]["parent_strict_failed_checks"] = [
            "foot_tracking_rms"]
        with self.assertRaises(ValueError):
            hand_pose_release_lineage(infos, iteration=9999)
        canonical = {**rescue_infos("lf60"), "microban_teleop_recipe_revision": CANONICAL}
        with self.assertRaisesRegex(ValueError, "Pose-release corner marker"):
            validate_corner_rescue_canonical_lineage(canonical, iteration=10099)

    def test_a_retrained_7100_to_10000_attempt_is_an_ordinary_fresh_chain(self):
        infos = {"microban_teleop_recipe_revision": POSE_RELEASE}
        for iteration in (9999, 10099, 14999):
            self.assertEqual(hand_pose_release_lineage(infos, iteration=iteration),
                             HAND_POSE_RELEASE_LINEAGE_FRESH)
            self.assertIsNone(validate_corner_rescue_canonical_lineage(infos, iteration=iteration))
        self.assertEqual(_deployment_recipe_revision(infos), POSE_RELEASE)


if __name__ == "__main__":
    unittest.main()
