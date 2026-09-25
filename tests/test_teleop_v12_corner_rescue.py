"""CPU-only regression tests for the pinned model9900 corner rescue."""

from __future__ import annotations

import copy
import subprocess
import unittest
from pathlib import Path

import torch

from mjlab_microban.tasks.microban_teleop_v12_actor import (
    TELEOP_V12_FOOT_OBSERVATION_COLUMNS,
    TELEOP_V12_TARGET_POSITION_NORMALIZER_STORED_STD,
)
from mjlab_microban.tasks.microban_teleop_v12_corner_rescue import (
    MICROBAN_TELEOP_V12_CORNER_RESCUE_ACTIVE_COLUMNS,
    MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY,
    MICROBAN_TELEOP_V12_CORNER_RESCUE_PARENT_COMMON_STEP,
    MICROBAN_TELEOP_V12_CORNER_RESCUE_PARENT_ITERATION,
    MICROBAN_TELEOP_V12_CORNER_RESCUE_PARENT_OPTIMIZER_STEP,
    MICROBAN_TELEOP_V12_CORNER_RESCUE_PARENT_SHA256,
    MICROBAN_TELEOP_V12_CORNER_RESCUE_RECIPE_REVISION,
    MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_COMMON_STEP,
    MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_ITERATION,
    MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_OPTIMIZER_STEP,
    corner_pair_joint_targets,
    corner_pair_selection,
    corner_rescue_marker,
    validate_corner_rescue_canonical_lineage,
    validate_corner_rescue_marker,
)
from mjlab_microban.tasks.microban_teleop_v12_corner_rescue_runner import (
    assert_corner_rescue_foot_adapter_zero,
    assert_corner_rescue_optimizer_step,
    validate_corner_rescue_parent_payload,
)
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MICROBAN_TELEOP_V12_RECIPE_REVISION,
)

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/train_microban_teleop_v12_corner_rescue.sh"


def _optimizer(step: int) -> dict:
    first = torch.ones(512, 83)
    second = torch.ones(512, 83)
    first[:, TELEOP_V12_FOOT_OBSERVATION_COLUMNS] = 0.0
    second[:, TELEOP_V12_FOOT_OBSERVATION_COLUMNS] = 0.0
    return {
        "state": {
            1: {
                "step": torch.tensor(float(step)),
                "exp_avg": first,
                "exp_avg_sq": second,
            },
            2: {
                "step": torch.tensor(float(step)),
                "exp_avg": torch.ones(18),
                "exp_avg_sq": torch.ones(18),
            },
        },
        "param_groups": [{"params": [1, 2]}],
    }


def _actor_state() -> dict[str, torch.Tensor]:
    stored_std = torch.tensor(TELEOP_V12_TARGET_POSITION_NORMALIZER_STORED_STD)
    mean = torch.zeros(1, 83)
    var = torch.ones(1, 83)
    std = torch.ones(1, 83)
    target_columns = list(range(69, 81))
    var[:, target_columns] = stored_std.square()
    std[:, target_columns] = stored_std
    weight = torch.ones(512, 83)
    weight[:, TELEOP_V12_FOOT_OBSERVATION_COLUMNS] = 0.0
    return {
        "obs_normalizer._mean": mean,
        "obs_normalizer._var": var,
        "obs_normalizer._std": std,
        "mlp.0.weight": weight,
    }


def _parent_payload() -> dict:
    return {
        "actor_state_dict": _actor_state(),
        "optimizer_state_dict": _optimizer(
            MICROBAN_TELEOP_V12_CORNER_RESCUE_PARENT_OPTIMIZER_STEP
        ),
        "iter": MICROBAN_TELEOP_V12_CORNER_RESCUE_PARENT_ITERATION,
        "infos": {
            "microban_teleop_recipe_revision": MICROBAN_TELEOP_V12_RECIPE_REVISION,
            "active_actor_columns_at_save": list(
                MICROBAN_TELEOP_V12_CORNER_RESCUE_ACTIVE_COLUMNS
            ),
            "env_state": {
                "common_step_counter": (
                    MICROBAN_TELEOP_V12_CORNER_RESCUE_PARENT_COMMON_STEP
                )
            },
        },
    }


def _final_infos() -> dict:
    return {
        "microban_teleop_training_contract_version": "12",
        "microban_teleop_recipe_revision": (
            MICROBAN_TELEOP_V12_CORNER_RESCUE_RECIPE_REVISION
        ),
        "adapter_gradient_schedule_revision": (
            "freeze_extra_to7000_then_hmd_hand_to10000_then_all_v1"
        ),
        "active_actor_columns_at_save": list(
            MICROBAN_TELEOP_V12_CORNER_RESCUE_ACTIVE_COLUMNS
        ),
        "env_state": {
            "common_step_counter": MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_COMMON_STEP
        },
        MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY: corner_rescue_marker(),
    }


class TeleopV12CornerRescueTest(unittest.TestCase):
    def test_sampler_boundaries_are_exact_5_90_5(self) -> None:
        selector = torch.tensor(
            [0.0, 0.899999, 0.9, 0.949999, 0.95, 0.999999],
            dtype=torch.float64,
        )
        self.assertEqual(
            corner_pair_selection(selector).tolist(), [1, 1, 2, 2, 0, 0]
        )
        marker = corner_rescue_marker()
        self.assertEqual(
            marker["sampler_probabilities"],
            {
                "ordinary_uniform_independent": 0.05,
                "left_forward_right_backward": 0.9,
                "left_backward_right_forward": 0.05,
            },
        )

    def test_corner_targets_preserve_left_right_and_forward_backward(self) -> None:
        degrees = torch.rad2deg(
            corner_pair_joint_targets(device="cpu", dtype=torch.float64)
        )
        expected = torch.tensor(
            [
                [[-25.0, 25.0, -50.0], [25.0, -20.0, -10.0]],
                [[25.0, 20.0, -10.0], [-25.0, -25.0, -50.0]],
            ],
            dtype=torch.float64,
        )
        self.assertTrue(torch.allclose(degrees, expected, atol=1.0e-12, rtol=0.0))

    def test_parent_requires_exact_optimizer_clock_and_foot_state(self) -> None:
        payload = _parent_payload()
        validate_corner_rescue_parent_payload(
            payload, checkpoint_sha256=MICROBAN_TELEOP_V12_CORNER_RESCUE_PARENT_SHA256
        )
        drifted = copy.deepcopy(payload)
        drifted["optimizer_state_dict"]["state"][2]["step"] += 1
        with self.assertRaisesRegex(ValueError, "optimizer clock drifted"):
            validate_corner_rescue_parent_payload(
                drifted,
                checkpoint_sha256=MICROBAN_TELEOP_V12_CORNER_RESCUE_PARENT_SHA256,
            )
        drifted = copy.deepcopy(payload)
        drifted["actor_state_dict"]["obs_normalizer._std"][0, 69] += 1.0e-4
        with self.assertRaisesRegex(ValueError, "foot obs_normalizer._std drifted"):
            assert_corner_rescue_foot_adapter_zero(drifted)

    def test_final_optimizer_clock_is_exact_for_every_state(self) -> None:
        payload = {"optimizer_state_dict": _optimizer(200_000)}
        assert_corner_rescue_optimizer_step(
            payload,
            expected_step=MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_OPTIMIZER_STEP,
        )
        payload["optimizer_state_dict"]["state"][1]["step"] -= 1
        with self.assertRaisesRegex(ValueError, "optimizer clock drifted"):
            assert_corner_rescue_optimizer_step(
                payload,
                expected_step=(
                    MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_OPTIMIZER_STEP
                ),
            )

    def test_final_marker_is_exact_and_tamper_evident(self) -> None:
        infos = _final_infos()
        marker = validate_corner_rescue_marker(
            infos, iteration=MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_ITERATION
        )
        self.assertEqual(marker["optimizer_step"], {"parent": 198_020, "target": 200_000})
        self.assertEqual(
            marker["foot_command_state_at_save"]["target_offset_exact_zero"], True
        )
        tampered = copy.deepcopy(infos)
        tampered[MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY][
            "sampler_probabilities"
        ]["left_forward_right_backward"] = 0.39
        with self.assertRaisesRegex(ValueError, "lineage marker drifted"):
            validate_corner_rescue_marker(
                tampered,
                iteration=MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_ITERATION,
            )

    def test_canonical_consumers_accept_only_final_rescue_and_descendants(self) -> None:
        final = _final_infos()
        self.assertEqual(
            validate_corner_rescue_canonical_lineage(final, iteration=9_999),
            corner_rescue_marker(),
        )
        with self.assertRaisesRegex(ValueError, "Only final model9999"):
            validate_corner_rescue_canonical_lineage(final, iteration=9_998)
        descendant = copy.deepcopy(final)
        descendant["microban_teleop_recipe_revision"] = (
            MICROBAN_TELEOP_V12_RECIPE_REVISION
        )
        descendant["env_state"]["common_step_counter"] = 10_001 * 24
        self.assertEqual(
            validate_corner_rescue_canonical_lineage(descendant, iteration=10_000),
            corner_rescue_marker(),
        )
        with self.assertRaisesRegex(ValueError, "descendant clock"):
            validate_corner_rescue_canonical_lineage(descendant, iteration=9_999)

    def test_launcher_is_syntax_checked_and_fully_pinned(self) -> None:
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
        help_text = subprocess.run(
            ["bash", str(LAUNCHER), "--help"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        self.assertIn("exactly 99", help_text)
        self.assertIn("PARENT_TRACKING_REPORT", help_text)
        self.assertIn("SUPERSEDED_V1_TRACKING_REPORT", help_text)
        launcher = LAUNCHER.read_text(encoding="utf-8")
        for fixed in (
            "mjlab_microban_teleop_v12\"",
            "PARENT_TRACKING_SHA=",
            "SUPERSEDED_V1_TRACKING_SHA=",
            "--env.scene.num-envs 2048",
            "--env.seed 42",
            "--agent.seed 42",
            "--agent.num-steps-per-env 24",
            'load-checkpoint "^model_${SOURCE_ITERATION}[.]pt$"',
        ):
            self.assertIn(fixed, launcher)


if __name__ == "__main__":
    unittest.main()
