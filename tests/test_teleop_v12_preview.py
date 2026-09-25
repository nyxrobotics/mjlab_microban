"""CPU-only contract tests for the isolated simulation preview path."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch

from mjlab_microban.scripts.evaluate_teleop_v12_preview import (
    run_preview_evaluation,
)
from mjlab_microban.scripts.evaluate_teleop_v12_tracking import FINAL_PROFILE
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_TELEOP_NUM_STEPS_PER_ENV,
)
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MICROBAN_TELEOP_V12_PREVIEW_HAND_ACQUISITION,
    MICROBAN_TELEOP_V12_PREVIEW_HAND_ACQUISITION_UPDATE,
    MICROBAN_TELEOP_V12_PREVIEW_HAND_FOCUS,
    MICROBAN_TELEOP_V12_PREVIEW_HAND_FOCUS_UPDATE,
    MICROBAN_TELEOP_V12_PREVIEW_HAND_STANDARD,
    MicrobanTeleopV12PreviewRlCfg,
    MicrobanTeleopV12RlCfg,
    make_microban_teleop_v12_env_cfg,
    make_microban_teleop_v12_preview_env_cfg,
    preview_hand_tracking_settings,
)
from mjlab_microban.tasks.microban_teleop_v12_preview import (
    MICROBAN_TELEOP_V12_PREVIEW_TASK_ID,
    TELEOP_V12_PREVIEW_PHASE1_ACCEPTANCE_INFO_KEY,
    TELEOP_V12_PREVIEW_PHASE1_TRAINED_ITERATION,
    TELEOP_V12_PREVIEW_PHASE1_VISUAL_QUALITY,
    TELEOP_V12_PREVIEW_PHASE2_MINIMUM_LIVE_ITERATION,
    TELEOP_V12_PREVIEW_PHASE_FULL_BODY,
    TELEOP_V12_PREVIEW_PHASE_HMD_HAND,
    TELEOP_V12_PREVIEW_SOURCE_SHA256,
    canonical_preview_info,
    reject_preview_checkpoint,
    staged_preview_info,
    validate_preview_marker,
)
from mjlab_microban.tasks.microban_teleop_v12_runner import (
    MicrobanTeleopV12ControllerOnlyPreviewOnPolicyRunner,
    MicrobanTeleopV12PreviewOnPolicyRunner,
    MicrobanTeleopV12UnacceptedSimulationPreviewOnPolicyRunner,
    preview_actor_load_cfg,
)


class TeleopV12PreviewTest(unittest.TestCase):
    def test_controller_preview_consumer_is_phase1_immutable_and_read_only(
        self,
    ) -> None:
        runner = MicrobanTeleopV12ControllerOnlyPreviewOnPolicyRunner
        self.assertEqual(
            runner.consumer_required_preview_phase,
            TELEOP_V12_PREVIEW_PHASE_HMD_HAND,
        )
        self.assertFalse(runner.consumer_requires_live_candidate)
        self.assertTrue(runner.require_immutable_checkpoint_bytes)

    def test_preview_hand_curriculum_is_progressive_and_canonical_is_unchanged(
        self,
    ) -> None:
        canonical = make_microban_teleop_v12_env_cfg()
        preview = make_microban_teleop_v12_preview_env_cfg()
        canonical_stages = canonical.curriculum["staged_curriculum"].params["stages"]
        preview_stages = preview.curriculum["staged_curriculum"].params["stages"]
        canonical_names = {stage["name"] for stage in canonical_stages}
        preview_by_name = {stage["name"]: stage for stage in preview_stages}
        acquisition_name = "preview reachable-hand acquisition"
        focus_name = "preview reachable-hand focus"

        self.assertNotIn(acquisition_name, canonical_names)
        self.assertNotIn(focus_name, canonical_names)
        self.assertEqual(canonical.rewards["hand_target_tracking"].weight, 0.0)
        self.assertEqual(canonical.rewards["joint_soft_limit_guard"].weight, -5.0)
        self.assertEqual(preview.rewards["foot_target_tracking"].weight, 0.0)
        self.assertEqual(
            preview_by_name[acquisition_name]["step"],
            MICROBAN_TELEOP_V12_PREVIEW_HAND_ACQUISITION_UPDATE
            * MICROBAN_TELEOP_NUM_STEPS_PER_ENV,
        )
        self.assertEqual(
            preview_by_name[focus_name]["step"],
            MICROBAN_TELEOP_V12_PREVIEW_HAND_FOCUS_UPDATE
            * MICROBAN_TELEOP_NUM_STEPS_PER_ENV,
        )
        self.assertEqual(
            [stage["step"] for stage in preview_stages],
            sorted(stage["step"] for stage in preview_stages),
        )

        hand_reward = SimpleNamespace(weight=0.0, params={"std": 0.08})
        guard = SimpleNamespace(weight=-5.0)
        rewards = {
            "hand_target_tracking": hand_reward,
            "joint_soft_limit_guard": guard,
        }
        env = SimpleNamespace(
            reward_manager=SimpleNamespace(get_term_cfg=rewards.__getitem__),
        )
        for name, expected in (
            (acquisition_name, MICROBAN_TELEOP_V12_PREVIEW_HAND_ACQUISITION),
            (focus_name, MICROBAN_TELEOP_V12_PREVIEW_HAND_FOCUS),
        ):
            preview_by_name[name]["apply"](env)
            self.assertEqual(hand_reward.weight, expected.reward_weight)
            self.assertEqual(hand_reward.params["std"], expected.reward_std_m)
            self.assertEqual(guard.weight, expected.joint_soft_limit_guard_weight)

    def test_preview_hand_settings_reconstruct_resume_boundaries(self) -> None:
        with self.assertRaisesRegex(ValueError, "7001"):
            preview_hand_tracking_settings(7_000)
        self.assertEqual(
            preview_hand_tracking_settings(7_001),
            MICROBAN_TELEOP_V12_PREVIEW_HAND_ACQUISITION,
        )
        self.assertEqual(
            preview_hand_tracking_settings(7_049),
            MICROBAN_TELEOP_V12_PREVIEW_HAND_ACQUISITION,
        )
        self.assertEqual(
            preview_hand_tracking_settings(7_050),
            MICROBAN_TELEOP_V12_PREVIEW_HAND_FOCUS,
        )
        self.assertEqual(
            preview_hand_tracking_settings(8_499),
            MICROBAN_TELEOP_V12_PREVIEW_HAND_FOCUS,
        )
        self.assertEqual(
            preview_hand_tracking_settings(8_500),
            MICROBAN_TELEOP_V12_PREVIEW_HAND_STANDARD,
        )

    def test_preview_runner_rejects_hand_curriculum_drift(self) -> None:
        settings = preview_hand_tracking_settings(7_101)
        hand = SimpleNamespace(rel_active=0.7)
        foot = SimpleNamespace(
            rel_single_support_envs=0.0,
            rel_both_feet_envs=0.0,
        )
        hand_reward = SimpleNamespace(
            weight=settings.reward_weight,
            params={"std": settings.reward_std_m},
        )
        rewards = {
            "hand_target_tracking": hand_reward,
            "foot_target_tracking": SimpleNamespace(weight=0.0),
            "joint_soft_limit_guard": SimpleNamespace(
                weight=settings.joint_soft_limit_guard_weight
            ),
        }
        commands = {"hand_target": hand, "foot_target": foot}
        events = {
            "hmd_neck_target_motion": SimpleNamespace(
                func=SimpleNamespace(neutral_probability=0.2)
            )
        }
        raw_env = SimpleNamespace(
            common_step_counter=7_101 * MICROBAN_TELEOP_NUM_STEPS_PER_ENV,
            command_manager=SimpleNamespace(get_term_cfg=commands.__getitem__),
            reward_manager=SimpleNamespace(get_term_cfg=rewards.__getitem__),
            event_manager=SimpleNamespace(get_term_cfg=events.__getitem__),
        )
        runner = object.__new__(MicrobanTeleopV12PreviewOnPolicyRunner)
        runner.env = SimpleNamespace(unwrapped=raw_env)
        runner.teleop_v12_preview = {"phase": TELEOP_V12_PREVIEW_PHASE_HMD_HAND}
        runner._assert_preview_curriculum_active()

        hand_reward.params["std"] = 0.12
        with self.assertRaisesRegex(RuntimeError, "curriculum drifted"):
            runner._assert_preview_curriculum_active()

    def test_marker_is_exact_non_deployable_and_standard_path_rejects_it(self) -> None:
        expected = staged_preview_info(
            phase=TELEOP_V12_PREVIEW_PHASE_HMD_HAND,
            phase_source_checkpoint_sha256=TELEOP_V12_PREVIEW_SOURCE_SHA256,
        )
        infos = {
            "preview_non_deployable": True,
            "teleop_v12_preview": expected,
        }
        marker = validate_preview_marker(
            infos, iteration=TELEOP_V12_PREVIEW_PHASE1_TRAINED_ITERATION
        )
        self.assertEqual(
            marker["root_source_checkpoint_sha256"],
            TELEOP_V12_PREVIEW_SOURCE_SHA256,
        )
        self.assertEqual(marker["phase"], TELEOP_V12_PREVIEW_PHASE_HMD_HAND)
        self.assertTrue(marker["simulation_only"])
        with self.assertRaisesRegex(ValueError, "forbidden"):
            reject_preview_checkpoint(infos)

    def test_partial_or_changed_markers_fail_closed(self) -> None:
        marker = staged_preview_info(
            phase=TELEOP_V12_PREVIEW_PHASE_HMD_HAND,
            phase_source_checkpoint_sha256=TELEOP_V12_PREVIEW_SOURCE_SHA256,
        )
        with self.assertRaises(ValueError):
            validate_preview_marker(
                {"teleop_v12_preview": marker},
                iteration=TELEOP_V12_PREVIEW_PHASE1_TRAINED_ITERATION,
            )
        changed = dict(marker)
        changed["simulation_only"] = False
        with self.assertRaises(ValueError):
            validate_preview_marker(
                {
                    "preview_non_deployable": True,
                    "teleop_v12_preview": changed,
                },
                iteration=TELEOP_V12_PREVIEW_PHASE1_TRAINED_ITERATION,
            )
        with self.assertRaises(ValueError):
            validate_preview_marker(
                {
                    "preview_non_deployable": True,
                    "teleop_v12_preview": marker,
                },
                iteration=6_999,
            )
        with self.assertRaisesRegex(ValueError, "explicit read-only"):
            validate_preview_marker(
                {
                    "preview_non_deployable": True,
                    "teleop_v12_preview": canonical_preview_info(),
                },
                iteration=10_100,
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

    def test_dedicated_evaluator_rejects_shallow_child_reports(self) -> None:
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model_10100.pt"
            marker = staged_preview_info(
                phase=TELEOP_V12_PREVIEW_PHASE_FULL_BODY,
                phase_source_checkpoint_sha256="a" * 64,
                phase1_acceptance_receipt_sha256="b" * 64,
                phase1_quality_class=TELEOP_V12_PREVIEW_PHASE1_VISUAL_QUALITY,
            )
            torch.save(
                {
                    "iter": 10_100,
                    "infos": {
                        "preview_non_deployable": True,
                        "teleop_v12_preview": marker,
                    },
                },
                checkpoint,
            )
            locomotion = {"status": "pass", "checks": {"finite": True}}
            tracking = {
                "status": "pass",
                "profile": FINAL_PROFILE,
                "checks": {"finite": True},
            }
            with (
                patch(
                    "mjlab_microban.scripts.evaluate_teleop_v12_preview."
                    "validate_embedded_phase1_acceptance",
                    return_value={},
                ),
                patch(
                    "mjlab_microban.scripts.evaluate_teleop_v12_preview."
                    "run_locomotion_evaluation",
                    return_value=locomotion,
                ),
                patch(
                    "mjlab_microban.scripts.evaluate_teleop_v12_preview."
                    "run_tracking_evaluation",
                    return_value=tracking,
                ),
                self.assertRaisesRegex(ValueError, "check set"),
            ):
                run_preview_evaluation(
                    checkpoint=checkpoint,
                    expected_sha256=None,
                    device="cpu",
                )

    def test_live_candidate_requires_trained_fullbody_phase(self) -> None:
        marker = staged_preview_info(
            phase=TELEOP_V12_PREVIEW_PHASE_FULL_BODY,
            phase_source_checkpoint_sha256="a" * 64,
            phase1_acceptance_receipt_sha256="b" * 64,
            phase1_quality_class=TELEOP_V12_PREVIEW_PHASE1_VISUAL_QUALITY,
        )
        infos = {"preview_non_deployable": True, "teleop_v12_preview": marker}
        with self.assertRaisesRegex(ValueError, "trained"):
            validate_preview_marker(
                infos,
                iteration=10_000,
                require_live_candidate=True,
            )
        validate_preview_marker(
            infos,
            iteration=TELEOP_V12_PREVIEW_PHASE2_MINIMUM_LIVE_ITERATION,
            require_live_candidate=True,
        )

    def test_unaccepted_consumer_skips_only_absent_embedded_phase1(self) -> None:
        marker = staged_preview_info(
            phase=TELEOP_V12_PREVIEW_PHASE_FULL_BODY,
            phase_source_checkpoint_sha256="a" * 64,
            phase1_acceptance_receipt_sha256="b" * 64,
            phase1_quality_class=TELEOP_V12_PREVIEW_PHASE1_VISUAL_QUALITY,
        )
        unaccepted = object.__new__(
            MicrobanTeleopV12UnacceptedSimulationPreviewOnPolicyRunner
        )
        self.assertIsNone(unaccepted._validate_preview_phase1_acceptance({}, marker))
        with self.assertRaises((TypeError, ValueError)):
            unaccepted._validate_preview_phase1_acceptance(
                {TELEOP_V12_PREVIEW_PHASE1_ACCEPTANCE_INFO_KEY: {}}, marker
            )

        accepted = object.__new__(MicrobanTeleopV12PreviewOnPolicyRunner)
        with self.assertRaisesRegex(TypeError, "lacks embedded phase-1"):
            accepted._validate_preview_phase1_acceptance({}, marker)

    def test_unaccepted_consumer_requires_bytes_and_is_permanently_read_only(
        self,
    ) -> None:
        runner = object.__new__(
            MicrobanTeleopV12UnacceptedSimulationPreviewOnPolicyRunner
        )
        runner.checkpoint_consumer_mode = True
        runner.teleop_v12_training_resume = False
        runner.simulation_preview_mode = True
        with self.assertRaisesRegex(ValueError, "immutable checkpoint bytes"):
            runner.load("mutable-path.pt", load_cfg=preview_actor_load_cfg())
        with self.assertRaisesRegex(RuntimeError, "cannot train"):
            runner.learn(1)
        with self.assertRaisesRegex(RuntimeError, "cannot save"):
            runner.save("forbidden.pt")
        with self.assertRaisesRegex(RuntimeError, "cannot export"):
            runner.export_policy_to_onnx("forbidden")


if __name__ == "__main__":
    unittest.main()
