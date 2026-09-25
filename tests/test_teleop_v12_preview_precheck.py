"""CPU-only tests for the fail-first v12 preview safety precheck."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from mjlab_microban.scripts.evaluate_teleop_v12_preview_precheck import (
    TARGETED_SCENARIO_NAMES,
    _directional_response_passed,
    _preview_identity,
    run_preview_precheck,
)
from mjlab_microban.scripts.evaluate_teleop_v12_tracking import (
    TARGET_COLUMN_ABLATION_ACTION_DELTA_MIN,
    _target_column_ablation_evidence,
    target_column_ablation_observation_columns,
)
from mjlab_microban.tasks.microban_teleop_v12_bootstrap import sha256_file
from mjlab_microban.tasks.microban_teleop_v12_preview import (
    TELEOP_V12_PREVIEW_PHASE1_VISUAL_QUALITY,
    TELEOP_V12_PREVIEW_PHASE2_LIFTED_COMPLETED_UPDATES,
    TELEOP_V12_PREVIEW_PHASE2_LIFTED_ITERATION,
    TELEOP_V12_PREVIEW_PHASE_FULL_BODY,
    canonical_preview_info,
    staged_preview_info,
)


def _full_body_marker() -> dict:
    return staged_preview_info(
        phase=TELEOP_V12_PREVIEW_PHASE_FULL_BODY,
        phase_source_checkpoint_sha256="b" * 64,
        phase1_acceptance_receipt_sha256="c" * 64,
        phase1_quality_class=TELEOP_V12_PREVIEW_PHASE1_VISUAL_QUALITY,
    )


def _checkpoint_payload(
    *, iteration: int = 10_100, common_step_counter: int | None = None
) -> dict:
    if common_step_counter is None:
        common_step_counter = (iteration + 1) * 24
    return {
        "iter": iteration,
        "infos": {
            "preview_non_deployable": True,
            "teleop_v12_preview": _full_body_marker(),
            "env_state": {"common_step_counter": common_step_counter},
        },
    }


def _scenario_result(
    name: str,
    *,
    actual_soft_limits: bool = True,
    target_ablation_response: bool = True,
) -> dict:
    checks = {
        "completed": True,
        "no_fall": True,
        "finite": True,
        "actual_soft_limits": actual_soft_limits,
        "raw_action_recurrence": True,
        "directional_response": True,
        "forced_hmd_motion": True,
        "nonzero_observation_coverage": True,
        "target_column_ablation_response": target_ablation_response,
    }
    return {"name": name, "status": "pass" if all(checks.values()) else "fail", "checks": checks}


class TeleopV12PreviewPrecheckTest(unittest.TestCase):
    def test_target_column_ablation_is_target_specific_and_strictly_above_floor(
        self,
    ) -> None:
        evidence = _target_column_ablation_evidence(
            {
                "hand": TARGET_COLUMN_ABLATION_ACTION_DELTA_MIN * 2.0,
                "foot": TARGET_COLUMN_ABLATION_ACTION_DELTA_MIN,
            },
            {"hand": True, "foot": True},
        )
        self.assertTrue(evidence["hand"]["passed"])
        self.assertFalse(evidence["foot"]["passed"])
        hand_zeroed, hand_preserved = target_column_ablation_observation_columns("hand")
        self.assertEqual(
            evidence["hand"]["ablated_observation_columns"], list(hand_zeroed)
        )
        self.assertEqual(
            evidence["hand"]["preserved_observation_columns"],
            list(hand_preserved),
        )
        inactive = _target_column_ablation_evidence(
            {"hand": None, "foot": None},
            {"hand": False, "foot": False},
        )
        self.assertTrue(inactive["hand"]["passed"])
        self.assertTrue(inactive["foot"]["passed"])

    def test_stationary_scenario_needs_no_directional_axis(self) -> None:
        self.assertTrue(_directional_response_passed((0.0, 0.0, 0.0), {}))
        response = {
            "vx_m_s": {"passed": True},
            "vy_m_s": {"passed": True},
            "yaw_rad_s": {"passed": True},
        }
        self.assertTrue(_directional_response_passed((0.7, 0.3, 1.5), response))
        del response["vy_m_s"]
        self.assertFalse(_directional_response_passed((0.7, 0.3, 1.5), response))

    def test_identity_requires_exact_hash_filename_marker_and_clock(self) -> None:
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model_10100.pt"
            torch.save(_checkpoint_payload(), checkpoint)
            digest = sha256_file(checkpoint)
            marker, iteration, actual = _preview_identity(checkpoint, digest)
            self.assertEqual(iteration, 10_100)
            self.assertEqual(actual, digest)
            self.assertTrue(marker["simulation_only"])

            legacy = _checkpoint_payload()
            legacy["infos"]["teleop_v12_preview"] = canonical_preview_info()
            torch.save(legacy, checkpoint)
            with self.assertRaisesRegex(ValueError, "Legacy v1"):
                _preview_identity(checkpoint, sha256_file(checkpoint))

            torch.save(_checkpoint_payload(common_step_counter=0), checkpoint)
            with self.assertRaisesRegex(ValueError, "clock"):
                _preview_identity(checkpoint, sha256_file(checkpoint))
            with self.assertRaisesRegex(ValueError, "lowercase"):
                _preview_identity(checkpoint, "not-a-sha")

    def test_phase2_seed_diagnostic_is_exact_and_never_passes(self) -> None:
        checkpoint = Path("/tmp/model_10000.pt")
        digest = "d" * 64
        marker = _full_body_marker()
        scenarios = tuple(SimpleNamespace(name=name) for name in TARGETED_SCENARIO_NAMES)
        wrapper = MagicMock()
        with (
            patch(
                "mjlab_microban.scripts.evaluate_teleop_v12_preview_precheck."
                "_preview_identity",
                return_value=(marker, TELEOP_V12_PREVIEW_PHASE2_LIFTED_ITERATION, digest),
            ) as identity,
            patch(
                "mjlab_microban.scripts.evaluate_teleop_v12_preview_precheck._load_actor",
                return_value=(
                    object(),
                    TELEOP_V12_PREVIEW_PHASE2_LIFTED_ITERATION,
                    {},
                ),
            ),
            patch(
                "mjlab_microban.scripts.evaluate_teleop_v12_preview_precheck._tracking_cfg",
                return_value=object(),
            ),
            patch(
                "mjlab_microban.scripts.evaluate_teleop_v12_preview_precheck.ManagerBasedRlEnv",
                return_value=object(),
            ),
            patch(
                "mjlab_microban.scripts.evaluate_teleop_v12_preview_precheck.RslRlVecEnvWrapper",
                return_value=wrapper,
            ),
            patch(
                "mjlab_microban.scripts.evaluate_teleop_v12_preview_precheck."
                "validate_teleop_v12_environment_contract"
            ),
            patch(
                "mjlab_microban.scripts.evaluate_teleop_v12_preview_precheck._scenarios",
                return_value=scenarios,
            ),
            patch(
                "mjlab_microban.scripts.evaluate_teleop_v12_preview_precheck."
                "_evaluate_targeted_scenario",
                side_effect=[_scenario_result(name) for name in TARGETED_SCENARIO_NAMES],
            ),
            patch(
                "mjlab_microban.scripts.evaluate_teleop_v12_preview_precheck.sha256_file",
                return_value=digest,
            ),
            patch(
                "mjlab_microban.scripts.evaluate_teleop_v12_preview_precheck."
                "configure_torch_backends"
            ),
            patch("torch.use_deterministic_algorithms"),
        ):
            report = run_preview_precheck(
                checkpoint=checkpoint,
                expected_sha256=digest,
                device="cpu",
                diagnose_phase2_seed=True,
            )
        identity.assert_called_once_with(
            checkpoint.resolve(), digest, diagnose_phase2_seed=True
        )
        self.assertEqual(report["status"], "diagnostic")
        self.assertTrue(report["diagnostic_checks_all_passed"])
        self.assertFalse(report["pico_live_accepted"])
        self.assertFalse(report["precheck_pass_is_go"])
        self.assertEqual(
            report["checkpoint"]["completed_updates"],
            TELEOP_V12_PREVIEW_PHASE2_LIFTED_COMPLETED_UPDATES,
        )

    def test_phase2_seed_identity_rejects_any_other_iteration(self) -> None:
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model_10001.pt"
            torch.save(_checkpoint_payload(iteration=10_001), checkpoint)
            with self.assertRaisesRegex(ValueError, "exactly the lifted model_10000"):
                _preview_identity(
                    checkpoint,
                    sha256_file(checkpoint),
                    diagnose_phase2_seed=True,
                )

    def test_cpu_precheck_requires_both_exact_mixed_scenarios(self) -> None:
        checkpoint = Path("/tmp/model_10100.pt")
        digest = "a" * 64
        marker = _full_body_marker()
        scenarios = tuple(SimpleNamespace(name=name) for name in TARGETED_SCENARIO_NAMES)
        wrapper = MagicMock()
        with (
            patch(
                "mjlab_microban.scripts.evaluate_teleop_v12_preview_precheck."
                "_preview_identity",
                return_value=(marker, 10_100, digest),
            ),
            patch(
                "mjlab_microban.scripts.evaluate_teleop_v12_preview_precheck."
                "_load_actor",
                return_value=(object(), 10_100, {}),
            ) as load_actor,
            patch(
                "mjlab_microban.scripts.evaluate_teleop_v12_preview_precheck."
                "_tracking_cfg",
                return_value=object(),
            ),
            patch(
                "mjlab_microban.scripts.evaluate_teleop_v12_preview_precheck."
                "ManagerBasedRlEnv",
                return_value=object(),
            ),
            patch(
                "mjlab_microban.scripts.evaluate_teleop_v12_preview_precheck."
                "RslRlVecEnvWrapper",
                return_value=wrapper,
            ),
            patch(
                "mjlab_microban.scripts.evaluate_teleop_v12_preview_precheck."
                "validate_teleop_v12_environment_contract"
            ),
            patch(
                "mjlab_microban.scripts.evaluate_teleop_v12_preview_precheck."
                "_scenarios",
                return_value=scenarios,
            ),
            patch(
                "mjlab_microban.scripts.evaluate_teleop_v12_preview_precheck."
                "_evaluate_targeted_scenario",
                side_effect=[_scenario_result(name) for name in TARGETED_SCENARIO_NAMES],
            ) as evaluate,
            patch(
                "mjlab_microban.scripts.evaluate_teleop_v12_preview_precheck."
                "sha256_file",
                return_value=digest,
            ),
            patch(
                "mjlab_microban.scripts.evaluate_teleop_v12_preview_precheck."
                "configure_torch_backends"
            ),
            patch("torch.use_deterministic_algorithms"),
        ):
            report = run_preview_precheck(
                checkpoint=checkpoint,
                expected_sha256=digest,
                device="cpu",
            )
        self.assertEqual(report["status"], "pass")
        self.assertTrue(report["simulation_only"])
        self.assertTrue(report["preview_non_deployable"])
        self.assertFalse(report["canonical_deployment_accepted"])
        self.assertFalse(report["pico_live_accepted"])
        self.assertFalse(report["precheck_pass_is_go"])
        self.assertTrue(report["full_preview_acceptance_required"])
        self.assertEqual(
            report["settings"]["scenario_names"], list(TARGETED_SCENARIO_NAMES)
        )
        self.assertEqual(evaluate.call_count, 3)
        load_actor.assert_called_once_with(
            checkpoint.resolve(),
            device="cpu",
            allow_nondeployable_preview=True,
        )
        wrapper.close.assert_called_once_with()

    def test_any_joint_limit_or_target_response_failure_fails_precheck(self) -> None:
        checkpoint = Path("/tmp/model_10100.pt")
        digest = "b" * 64
        marker = _full_body_marker()
        scenarios = tuple(SimpleNamespace(name=name) for name in TARGETED_SCENARIO_NAMES)
        wrapper = MagicMock()
        with (
            patch(
                "mjlab_microban.scripts.evaluate_teleop_v12_preview_precheck."
                "_preview_identity",
                return_value=(marker, 10_100, digest),
            ),
            patch(
                "mjlab_microban.scripts.evaluate_teleop_v12_preview_precheck."
                "_load_actor",
                return_value=(object(), 10_100, {}),
            ),
            patch(
                "mjlab_microban.scripts.evaluate_teleop_v12_preview_precheck."
                "_tracking_cfg",
                return_value=object(),
            ),
            patch(
                "mjlab_microban.scripts.evaluate_teleop_v12_preview_precheck."
                "ManagerBasedRlEnv",
                return_value=object(),
            ),
            patch(
                "mjlab_microban.scripts.evaluate_teleop_v12_preview_precheck."
                "RslRlVecEnvWrapper",
                return_value=wrapper,
            ),
            patch(
                "mjlab_microban.scripts.evaluate_teleop_v12_preview_precheck."
                "validate_teleop_v12_environment_contract"
            ),
            patch(
                "mjlab_microban.scripts.evaluate_teleop_v12_preview_precheck."
                "_scenarios",
                return_value=scenarios,
            ),
            patch(
                "mjlab_microban.scripts.evaluate_teleop_v12_preview_precheck."
                "_evaluate_targeted_scenario",
                side_effect=[
                    _scenario_result(
                        TARGETED_SCENARIO_NAMES[0],
                        target_ablation_response=False,
                    ),
                    _scenario_result(
                        TARGETED_SCENARIO_NAMES[1], actual_soft_limits=False
                    ),
                    _scenario_result(TARGETED_SCENARIO_NAMES[2]),
                ],
            ),
            patch(
                "mjlab_microban.scripts.evaluate_teleop_v12_preview_precheck."
                "sha256_file",
                return_value=digest,
            ),
            patch(
                "mjlab_microban.scripts.evaluate_teleop_v12_preview_precheck."
                "configure_torch_backends"
            ),
            patch("torch.use_deterministic_algorithms"),
        ):
            report = run_preview_precheck(
                checkpoint=checkpoint,
                expected_sha256=digest,
                device="cpu",
            )
        self.assertEqual(report["status"], "fail")
        self.assertFalse(report["checks"]["actual_soft_limits"])
        self.assertFalse(report["checks"]["target_column_ablation_response"])


if __name__ == "__main__":
    unittest.main()
