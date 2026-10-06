"""Unit tests for the fixed teleop evaluation helpers of the v12 tracking gate."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from tensordict import TensorDict

from mjlab_microban.scripts.teleop_v12_scenarios import (
    HmdMotionStats,
    _patch_initial_command_observation,
    _percentile,
)
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_HMD_JOINT_NAMES,
    MICROBAN_TELEOP_OBSERVATION_SCHEMA,
    MICROBAN_TELEOP_OBSERVATION_WIDTH,
)


class MovingHmdEvidenceTest(unittest.TestCase):
    @staticmethod
    def _scenario_motion(
        *, target_peak_to_peak: float, actual_peak_to_peak: float
    ) -> dict[str, object]:
        return {
            "active_event_member": True,
            "sample_count": 1001,
            "joint_names": list(MICROBAN_HMD_JOINT_NAMES),
            "per_axis": {
                name: {
                    "target_peak_to_peak_rad": target_peak_to_peak,
                    "actual_peak_to_peak_rad": actual_peak_to_peak,
                }
                for name in MICROBAN_HMD_JOINT_NAMES
            },
        }

    def test_streaming_stats_report_per_axis_target_and_actual_excursion(self) -> None:
        stats = HmdMotionStats.start(
            joint_names=MICROBAN_HMD_JOINT_NAMES,
            target=torch.tensor([0.0, -0.1, 0.2]),
            actual=torch.tensor([0.0, -0.05, 0.1]),
        )
        stats.add(
            target=torch.tensor([0.2, 0.1, -0.1]),
            actual=torch.tensor([0.1, 0.02, -0.05]),
        )
        report = stats.report(step_dt=0.02)
        self.assertTrue(report["active_event_member"])
        self.assertEqual(report["sample_count"], 2)
        self.assertAlmostEqual(
            report["per_axis"]["head"]["target_peak_to_peak_rad"], 0.2
        )
        self.assertAlmostEqual(
            report["per_axis"]["neck_pitch"]["actual_peak_to_peak_rad"],
            0.15,
            places=5,
        )
        self.assertAlmostEqual(
            report["per_axis"]["head"]["maximum_target_slew_rad_s"],
            10.0,
            places=5,
        )


class ObservationPatchTest(unittest.TestCase):
    def test_only_command_slices_are_replaced(self) -> None:
        original_actor = torch.arange(
            MICROBAN_TELEOP_OBSERVATION_WIDTH, dtype=torch.float32
        ).unsqueeze(0)
        observations = TensorDict(
            {"actor": original_actor.clone()},
            batch_size=[1],
        )
        values = {
            "twist": torch.tensor([[0.1, 0.2, 0.3]]),
            "foot_target": torch.arange(6, dtype=torch.float32).unsqueeze(0) + 100.0,
            "hand_target": torch.arange(8, dtype=torch.float32).unsqueeze(0) + 200.0,
        }
        terms = {name: SimpleNamespace(command=value) for name, value in values.items()}
        manager = SimpleNamespace(get_term=lambda name: terms[name])
        env = SimpleNamespace(num_envs=1, command_manager=manager)

        patched = _patch_initial_command_observation(observations, env)
        self.assertTrue(torch.equal(observations["actor"], original_actor))

        offset = 0
        for name, width in MICROBAN_TELEOP_OBSERVATION_SCHEMA:
            actual = patched["actor"][:, offset : offset + width]
            source_name = "twist" if name == "command" else name
            if source_name in values:
                self.assertTrue(torch.equal(actual, values[source_name]))
            else:
                self.assertTrue(
                    torch.equal(actual, original_actor[:, offset : offset + width])
                )
            offset += width


class MetricHelperTest(unittest.TestCase):
    def test_percentile_interpolates_and_handles_empty_input(self) -> None:
        self.assertIsNone(_percentile([], 0.95))
        self.assertAlmostEqual(_percentile([0.0, 10.0], 0.95), 9.5)


if __name__ == "__main__":
    unittest.main()
