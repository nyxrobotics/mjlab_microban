# Copyright 2026 nyxrobotics

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
    TwistScenario,
    default_scenarios,
    directional_response,
    publish_json_atomic,
    summarize_samples,
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


class AtomicWriterTest(unittest.TestCase):
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
