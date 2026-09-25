# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""CPU-only tests for the original velocity-policy diagnostic receipt."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from mjlab_microban.legacy_velocity_diagnostics import (
    DEFAULT_LEGACY_VELOCITY_RAW_OUTPUT,
    DEFAULT_LEGACY_VELOCITY_SOFTCLIP_OUTPUT,
    TwistScenario,
    build_report,
    default_scenarios,
    directional_response,
    publish_json_atomic,
    resolve_output_path,
    select_scenarios,
    summarize_samples,
    validate_scenarios,
)


def _result(
    scenario: TwistScenario,
    *,
    measured: tuple[float, float, float] | None = None,
    completed: bool = True,
    fell: bool = False,
    projected_values: int = 0,
    actual_violation: float = 0.0,
    both_feet_airborne: bool = True,
    execution_applied_projection: bool = False,
) -> dict[str, object]:
    if measured is None:
        measured = scenario.twist
    return {
        "command": dict(
            zip(("vx_m_s", "vy_m_s", "yaw_rad_s"), scenario.twist, strict=True)
        ),
        "completed": completed,
        "fell": fell,
        "foot_evidence": {
            "left": {"ever_airborne": both_feet_airborne},
            "right": {"ever_airborne": both_feet_airborne},
            "both_feet_airborne_step_count": int(both_feet_airborne),
        },
        "measured_velocity_body": {
            name: {"mean": value}
            for name, value in zip(
                ("vx_m_s", "vy_m_s", "yaw_rad_s"), measured, strict=True
            )
        },
        "name": scenario.name,
        "nonfinite_detected": False,
        "target_soft_limits": {
            "execution_applied_projection": execution_applied_projection,
            "maximum_actual_violation_rad": actual_violation,
            "projected_target_value_count": projected_values,
        },
    }


class ScenarioTest(unittest.TestCase):
    def test_default_suite_has_exact_required_signed_axes(self) -> None:
        scenarios = default_scenarios()
        validate_scenarios(scenarios)
        self.assertEqual(len(scenarios), 9)
        self.assertEqual(
            {scenario.twist for scenario in scenarios},
            {
                (0.0, 0.0, 0.0),
                (0.1, 0.0, 0.0),
                (0.2, 0.0, 0.0),
                (-0.1, 0.0, 0.0),
                (-0.2, 0.0, 0.0),
                (0.0, 0.1, 0.0),
                (0.0, -0.1, 0.0),
                (0.0, 0.0, 0.5),
                (0.0, 0.0, -0.5),
            },
        )

    def test_selection_preserves_requested_order_and_rejects_unknown(self) -> None:
        selected = select_scenarios(
            default_scenarios(), "yaw_right_0p5,neutral,forward_0p1"
        )
        self.assertEqual(
            [scenario.name for scenario in selected],
            ["yaw_right_0p5", "neutral", "forward_0p1"],
        )
        with self.assertRaisesRegex(ValueError, "Unknown"):
            select_scenarios(default_scenarios(), "missing")

    def test_validation_rejects_mixed_nonfinite_and_duplicate_commands(self) -> None:
        with self.assertRaisesRegex(ValueError, "single-axis"):
            validate_scenarios((TwistScenario("mixed", (0.1, 0.1, 0.0)),))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_scenarios((TwistScenario("bad", (math.inf, 0.0, 0.0)),))
        with self.assertRaisesRegex(ValueError, "Duplicate scenario twist"):
            validate_scenarios(
                (
                    TwistScenario("a", (0.1, 0.0, 0.0)),
                    TwistScenario("b", (0.1, 0.0, 0.0)),
                )
            )

    def test_projected_mode_has_separate_default_and_cannot_replace_raw(self) -> None:
        self.assertEqual(
            resolve_output_path(None, execute_soft_limit_projection=False),
            DEFAULT_LEGACY_VELOCITY_RAW_OUTPUT,
        )
        self.assertEqual(
            resolve_output_path(None, execute_soft_limit_projection=True),
            DEFAULT_LEGACY_VELOCITY_SOFTCLIP_OUTPUT,
        )
        with self.assertRaisesRegex(ValueError, "may not overwrite"):
            resolve_output_path(
                DEFAULT_LEGACY_VELOCITY_RAW_OUTPUT,
                execute_soft_limit_projection=True,
            )


class MetricTest(unittest.TestCase):
    def test_sample_summary_is_linear_and_json_safe(self) -> None:
        summary = summarize_samples((0.0, 10.0, float("nan")))
        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["mean"], 5.0)
        self.assertEqual(summary["p05"], 0.5)
        self.assertEqual(summary["p95"], 9.5)
        self.assertEqual(summarize_samples((math.nan,))["mean"], None)
        json.dumps(summary, allow_nan=False)

    def test_directional_response_handles_negative_commands(self) -> None:
        backward = TwistScenario("backward", (-0.2, 0.0, 0.0))
        correct = directional_response(_result(backward, measured=(-0.12, 0.0, 0.0)))
        self.assertIsNotNone(correct)
        assert correct is not None
        self.assertAlmostEqual(correct["signed_response"], 0.12)
        self.assertTrue(correct["sign_matches"])

        wrong = directional_response(_result(backward, measured=(0.04, 0.0, 0.0)))
        assert wrong is not None
        self.assertAlmostEqual(wrong["signed_response"], -0.04)
        self.assertFalse(wrong["sign_matches"])


class ReportTest(unittest.TestCase):
    def test_report_aggregates_safety_motion_and_foot_evidence(self) -> None:
        neutral, forward, backward = default_scenarios()[:3:1]
        # Use a named backward scenario from the canonical suite.
        backward = default_scenarios()[3]
        results = [
            _result(neutral),
            _result(forward, projected_values=1),
            _result(
                backward,
                measured=(0.03, 0.0, 0.0),
                completed=False,
                fell=True,
                actual_violation=0.002,
                both_feet_airborne=False,
            ),
        ]
        report = build_report(
            checkpoint="model_14999.pt",
            checkpoint_sha256="a" * 64,
            device="cpu",
            seed=42,
            steps=300,
            settle_steps=50,
            step_dt_s=0.02,
            results=results,
            generated_at_utc="2026-09-25T00:00:00+00:00",
        )
        summary = report["summary"]
        self.assertTrue(report["diagnostic_only"])
        self.assertFalse(
            report["environment"]["policy_output_safety_projection_applied"]
        )
        self.assertEqual(summary["scenario_count"], 3)
        self.assertEqual(summary["completed_scenario_count"], 2)
        self.assertEqual(summary["fall_scenario_count"], 1)
        self.assertEqual(summary["target_projection_scenario_count"], 1)
        self.assertEqual(summary["actual_soft_limit_violation_scenario_count"], 1)
        self.assertEqual(summary["bilateral_air_evidence_scenario_count"], 2)
        self.assertEqual(summary["simultaneous_flight_scenario_count"], 2)
        self.assertEqual(summary["directionally_correct_scenario_count"], 1)
        json.dumps(report, allow_nan=False)

    def test_report_records_closed_loop_soft_limit_projection_mode(self) -> None:
        scenario = default_scenarios()[1]
        result = _result(scenario, execution_applied_projection=True)
        report = build_report(
            checkpoint="model_14999.pt",
            checkpoint_sha256="b" * 64,
            device="cuda:0",
            seed=42,
            steps=300,
            settle_steps=50,
            step_dt_s=0.02,
            results=[result],
            execute_soft_limit_projection=True,
            generated_at_utc="2026-09-25T00:00:00+00:00",
        )
        self.assertTrue(
            report["environment"]["policy_output_safety_projection_applied"]
        )
        self.assertEqual(
            report["environment"]["previous_action_observation"],
            "executed_raw_action_after_soft_limit_projection",
        )
        self.assertEqual(
            report["environment"]["action_execution_mode"],
            "project_actor_absolute_targets_to_current_soft_limits",
        )
        self.assertTrue(report["settings"]["execute_soft_limit_projection"])

    def test_report_rejects_mixed_execution_modes(self) -> None:
        result = _result(
            default_scenarios()[1], execution_applied_projection=True
        )
        with self.assertRaisesRegex(ValueError, "execution mode"):
            build_report(
                checkpoint="model_14999.pt",
                checkpoint_sha256="c" * 64,
                device="cpu",
                seed=42,
                steps=10,
                settle_steps=0,
                step_dt_s=0.02,
                results=[result],
                execute_soft_limit_projection=False,
            )

    def test_atomic_writer_replaces_complete_json_without_temp_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "report.json"
            resolved = publish_json_atomic(output, {"revision": 1})
            self.assertEqual(json.loads(resolved.read_text()), {"revision": 1})
            publish_json_atomic(output, {"revision": 2})
            self.assertEqual(json.loads(resolved.read_text()), {"revision": 2})
            self.assertEqual(list(resolved.parent.glob(".report.json.*.tmp")), [])

    def test_atomic_writer_rejects_nonfinite_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            with self.assertRaises(ValueError):
                publish_json_atomic(output, {"bad": math.nan})
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
