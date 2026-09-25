"""CPU-only contract tests for the isolated simulation preview path."""

from __future__ import annotations

import unittest

from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MicrobanTeleopV12PreviewRlCfg,
    MicrobanTeleopV12RlCfg,
)
from mjlab_microban.tasks.microban_teleop_v12_preview import (
    MICROBAN_TELEOP_V12_PREVIEW_TASK_ID,
    TELEOP_V12_PREVIEW_LIFTED_ITERATION,
    TELEOP_V12_PREVIEW_SOURCE_SHA256,
    canonical_preview_info,
    reject_preview_checkpoint,
    validate_preview_marker,
)
from mjlab_microban.tasks.microban_teleop_v12_runner import preview_actor_load_cfg


class TeleopV12PreviewTest(unittest.TestCase):
    def test_marker_is_exact_non_deployable_and_standard_path_rejects_it(self) -> None:
        infos = {
            "preview_non_deployable": True,
            "teleop_v12_preview": canonical_preview_info(),
        }
        marker = validate_preview_marker(
            infos, iteration=TELEOP_V12_PREVIEW_LIFTED_ITERATION
        )
        self.assertEqual(
            marker["source_checkpoint_sha256"], TELEOP_V12_PREVIEW_SOURCE_SHA256
        )
        self.assertTrue(marker["simulation_only"])
        with self.assertRaisesRegex(ValueError, "forbidden"):
            reject_preview_checkpoint(infos)

    def test_partial_or_changed_markers_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            validate_preview_marker(
                {"teleop_v12_preview": canonical_preview_info()},
                iteration=TELEOP_V12_PREVIEW_LIFTED_ITERATION,
            )
        changed = canonical_preview_info()
        changed["simulation_only"] = False
        with self.assertRaises(ValueError):
            validate_preview_marker(
                {
                    "preview_non_deployable": True,
                    "teleop_v12_preview": changed,
                },
                iteration=TELEOP_V12_PREVIEW_LIFTED_ITERATION,
            )
        with self.assertRaises(ValueError):
            validate_preview_marker(
                {
                    "preview_non_deployable": True,
                    "teleop_v12_preview": canonical_preview_info(),
                },
                iteration=TELEOP_V12_PREVIEW_LIFTED_ITERATION - 1,
            )

    def test_preview_has_a_distinct_task_and_explicit_consumer_load_mask(self) -> None:
        self.assertEqual(
            MICROBAN_TELEOP_V12_PREVIEW_TASK_ID,
            "Mjlab-Teleop-V12-Preview-Microban",
        )
        self.assertFalse(MicrobanTeleopV12RlCfg.simulation_preview_mode)
        self.assertTrue(MicrobanTeleopV12PreviewRlCfg.simulation_preview_mode)
        self.assertNotEqual(
            MicrobanTeleopV12RlCfg.experiment_name,
            MicrobanTeleopV12PreviewRlCfg.experiment_name,
        )
        self.assertEqual(
            preview_actor_load_cfg(),
            {
                "actor": True,
                "critic": False,
                "optimizer": False,
                "iteration": False,
                "rnd": False,
            },
        )


if __name__ == "__main__":
    unittest.main()
