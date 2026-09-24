# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for the opt-in exclusive signed-axis velocity sampler."""

from __future__ import annotations

import math
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import torch
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand

from mjlab_microban.tasks.mdp import (
    UniformVelocityCommandWithRotation,
    _validate_signed_axis_sampler_cfg,
)

MODES = (
    "standing",
    "forward",
    "backward",
    "lateral_left",
    "lateral_right",
    "yaw_left",
    "yaw_right",
    "mixed",
)


def _probabilities(**overrides: float) -> dict[str, float]:
    probabilities = dict.fromkeys(MODES, 0.0)
    probabilities.update(overrides)
    return probabilities


def _ranges() -> dict[str, tuple[float, float]]:
    return {
        "forward": (0.10, 0.20),
        "backward": (-0.20, -0.10),
        "lateral_left": (0.08, 0.12),
        "lateral_right": (-0.12, -0.08),
        "yaw_left": (0.40, 0.60),
        "yaw_right": (-0.60, -0.40),
    }


def _cfg(
    *,
    probabilities: dict[str, float] | None = None,
    ranges: dict[str, tuple[float, float]] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        signed_axis_probabilities=probabilities,
        signed_axis_ranges=ranges,
        ranges=SimpleNamespace(
            lin_vel_x=(-0.5, 0.7),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-1.5, 1.5),
        ),
        rotation_env_ang_vel_range=(-3.0, 3.0),
        rotation_min_ang_vel=0.5,
        rel_standing_envs=0.0,
        rel_forward_envs=0.0,
        rel_rotation_envs=0.0,
        heading_command=True,
        rel_heading_envs=0.0,
        rel_world_envs=0.0,
        init_velocity_prob=0.0,
    )


def _command(cfg: SimpleNamespace, num_envs: int) -> UniformVelocityCommandWithRotation:
    command = object.__new__(UniformVelocityCommandWithRotation)
    command.cfg = cfg
    command._env = SimpleNamespace(num_envs=num_envs, device="cpu")
    command.vel_command_b = torch.full((num_envs, 3), math.nan)
    command.vel_command_w = torch.full((num_envs, 3), math.nan)
    command.is_standing_env = torch.ones(num_envs, dtype=torch.bool)
    command.is_forward_env = torch.ones(num_envs, dtype=torch.bool)
    command.is_heading_env = torch.ones(num_envs, dtype=torch.bool)
    command.is_world_env = torch.ones(num_envs, dtype=torch.bool)
    command.is_rotation_env = torch.ones(num_envs, dtype=torch.bool)
    return command


class SignedAxisSamplerValidationTest(unittest.TestCase):
    def test_none_preserves_legacy_opt_out(self) -> None:
        self.assertIsNone(_validate_signed_axis_sampler_cfg(_cfg()))

    def test_requires_both_opt_in_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires both"):
            _validate_signed_axis_sampler_cfg(
                _cfg(probabilities=_probabilities(standing=1.0))
            )
        with self.assertRaisesRegex(ValueError, "requires both"):
            _validate_signed_axis_sampler_cfg(_cfg(ranges=_ranges()))

    def test_probability_keys_values_and_sum_fail_closed(self) -> None:
        valid = _probabilities(standing=1.0)
        cases: list[tuple[str, dict[str, float], str]] = []
        missing = deepcopy(valid)
        del missing["mixed"]
        cases.append(("missing", missing, "missing"))
        unknown = deepcopy(valid)
        unknown["typo"] = 0.0
        cases.append(("unknown", unknown, "unknown"))
        negative = deepcopy(valid)
        negative["standing"] = -1.0
        cases.append(("negative", negative, "non-negative"))
        nonfinite = deepcopy(valid)
        nonfinite["standing"] = math.nan
        cases.append(("nonfinite", nonfinite, "finite"))
        wrong_sum = deepcopy(valid)
        wrong_sum["standing"] = 0.9
        cases.append(("sum", wrong_sum, "sum to 1"))

        for name, probabilities, message in cases:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                _validate_signed_axis_sampler_cfg(
                    _cfg(probabilities=probabilities, ranges=_ranges())
                )

    def test_range_keys_sign_finiteness_and_envelope_fail_closed(self) -> None:
        valid_probabilities = _probabilities(standing=1.0)
        cases: list[tuple[str, dict[str, tuple[float, float]], str]] = []
        missing = _ranges()
        del missing["yaw_right"]
        cases.append(("missing", missing, "missing"))
        unknown = _ranges()
        unknown["typo"] = (0.1, 0.2)
        cases.append(("unknown", unknown, "unknown"))
        wrong_sign = _ranges()
        wrong_sign["backward"] = (0.1, 0.2)
        cases.append(("sign", wrong_sign, "strictly negative"))
        nonfinite = _ranges()
        nonfinite["yaw_left"] = (0.4, math.inf)
        cases.append(("nonfinite", nonfinite, "finite"))
        reversed_range = _ranges()
        reversed_range["forward"] = (0.2, 0.1)
        cases.append(("order", reversed_range, "lower <= upper"))
        outside = _ranges()
        outside["lateral_left"] = (0.1, 0.4)
        cases.append(("envelope", outside, "escapes its envelope"))

        for name, ranges, message in cases:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                _validate_signed_axis_sampler_cfg(
                    _cfg(probabilities=valid_probabilities, ranges=ranges)
                )

    def test_rejects_legacy_world_heading_and_init_velocity_fractions(self) -> None:
        for field in (
            "rel_standing_envs",
            "rel_forward_envs",
            "rel_rotation_envs",
            "rel_heading_envs",
            "rel_world_envs",
            "init_velocity_prob",
        ):
            cfg = _cfg(probabilities=_probabilities(standing=1.0), ranges=_ranges())
            setattr(cfg, field, 0.1)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                _validate_signed_axis_sampler_cfg(cfg)

    def test_malformed_parent_envelope_fails_closed(self) -> None:
        cfg = _cfg(probabilities=_probabilities(standing=1.0), ranges=_ranges())
        cfg.ranges.lin_vel_x = (math.nan, 0.7)
        with self.assertRaisesRegex(ValueError, "linear-x envelope"):
            _validate_signed_axis_sampler_cfg(cfg)

    def test_heading_cfg_is_allowed_when_heading_fraction_is_zero(self) -> None:
        cfg = _cfg(probabilities=_probabilities(standing=1.0), ranges=_ranges())
        cfg.heading_command = True
        probabilities, _ = _validate_signed_axis_sampler_cfg(cfg)  # type: ignore[misc]
        self.assertEqual(probabilities[0], 1.0)

    def test_mixed_ranges_keep_dead_bands_and_use_moving_yaw_envelope(self) -> None:
        ranges = _ranges()
        ranges["yaw_left"] = (0.4, 2.5)
        ranges["yaw_right"] = (-2.5, -0.4)
        _, validated = _validate_signed_axis_sampler_cfg(
            _cfg(probabilities=_probabilities(mixed=1.0), ranges=ranges)
        )  # type: ignore[misc]
        self.assertEqual(validated["mixed_yaw_left"], (0.4, 1.5))
        self.assertEqual(validated["mixed_yaw_right"], (-1.5, -0.4))

        cfg = _cfg(probabilities=_probabilities(mixed=1.0), ranges=ranges)
        cfg.ranges.ang_vel_z = (-0.3, 0.3)
        with self.assertRaisesRegex(ValueError, "no intersection"):
            _validate_signed_axis_sampler_cfg(cfg)


class SignedAxisSamplerBehaviorTest(unittest.TestCase):
    def test_legacy_opt_out_delegates_to_parent_sampler(self) -> None:
        cfg = _cfg()
        cfg.rel_rotation_envs = 0.0
        command = _command(cfg, 4)
        env_ids = torch.arange(4)
        with patch.object(
            UniformVelocityCommand, "_resample_command"
        ) as parent_resample:
            command._resample_command(env_ids)
        parent_resample.assert_called_once_with(env_ids)

    def test_each_mode_is_exclusive_and_sets_compatibility_flags(self) -> None:
        torch.manual_seed(7)
        num_envs = 128
        env_ids = torch.arange(num_envs)
        for mode in MODES:
            cfg = _cfg(probabilities=_probabilities(**{mode: 1.0}), ranges=_ranges())
            command = _command(cfg, num_envs)
            command._resample_command(env_ids)
            body = command.vel_command_b
            nonzero = body.ne(0.0)

            with self.subTest(mode=mode):
                self.assertTrue(torch.equal(command.vel_command_w, body))
                self.assertFalse(command.is_heading_env.any())
                self.assertFalse(command.is_world_env.any())
                if mode == "standing":
                    self.assertFalse(nonzero.any())
                    self.assertTrue(command.is_standing_env.all())
                elif mode == "forward":
                    self.assertTrue((body[:, 0] > 0.0).all())
                    self.assertTrue((nonzero.sum(dim=1) == 1).all())
                    self.assertTrue(command.is_forward_env.all())
                elif mode == "backward":
                    self.assertTrue((body[:, 0] < 0.0).all())
                    self.assertTrue((nonzero.sum(dim=1) == 1).all())
                elif mode == "lateral_left":
                    self.assertTrue((body[:, 1] > 0.0).all())
                    self.assertTrue((nonzero.sum(dim=1) == 1).all())
                elif mode == "lateral_right":
                    self.assertTrue((body[:, 1] < 0.0).all())
                    self.assertTrue((nonzero.sum(dim=1) == 1).all())
                elif mode == "yaw_left":
                    self.assertTrue((body[:, 2] > 0.0).all())
                    self.assertTrue((nonzero.sum(dim=1) == 1).all())
                    self.assertTrue(command.is_rotation_env.all())
                elif mode == "yaw_right":
                    self.assertTrue((body[:, 2] < 0.0).all())
                    self.assertTrue((nonzero.sum(dim=1) == 1).all())
                    self.assertTrue(command.is_rotation_env.all())
                else:
                    self.assertTrue(nonzero.all())

                expected_standing = mode == "standing"
                expected_forward = mode == "forward"
                expected_rotation = mode in ("yaw_left", "yaw_right")
                self.assertEqual(
                    command.is_standing_env.all().item(), expected_standing
                )
                self.assertEqual(command.is_forward_env.all().item(), expected_forward)
                self.assertEqual(
                    command.is_rotation_env.all().item(), expected_rotation
                )

    def test_categorical_probabilities_are_exclusive_and_observed(self) -> None:
        probabilities = {
            "standing": 0.10,
            "forward": 0.15,
            "backward": 0.15,
            "lateral_left": 0.15,
            "lateral_right": 0.15,
            "yaw_left": 0.15,
            "yaw_right": 0.15,
            "mixed": 0.0,
        }
        num_envs = 50_000
        command = _command(
            _cfg(probabilities=probabilities, ranges=_ranges()), num_envs
        )
        torch.manual_seed(1234)
        command._resample_command(torch.arange(num_envs))
        body = command.vel_command_b
        observed = {
            "standing": body.eq(0.0).all(dim=1),
            "forward": (body[:, 0] > 0.0) & body[:, 1:].eq(0.0).all(dim=1),
            "backward": (body[:, 0] < 0.0) & body[:, 1:].eq(0.0).all(dim=1),
            "lateral_left": (body[:, 1] > 0.0) & body[:, (0, 2)].eq(0.0).all(dim=1),
            "lateral_right": (body[:, 1] < 0.0) & body[:, (0, 2)].eq(0.0).all(dim=1),
            "yaw_left": (body[:, 2] > 0.0) & body[:, :2].eq(0.0).all(dim=1),
            "yaw_right": (body[:, 2] < 0.0) & body[:, :2].eq(0.0).all(dim=1),
        }
        membership_count = torch.stack(tuple(observed.values())).sum(dim=0)
        self.assertTrue((membership_count == 1).all())
        for name, mask in observed.items():
            self.assertAlmostEqual(
                float(mask.float().mean()), probabilities[name], delta=0.01
            )

    def test_mixed_samples_all_axes_outside_dead_bands(self) -> None:
        ranges = _ranges()
        ranges["yaw_left"] = (0.4, 2.5)
        ranges["yaw_right"] = (-2.5, -0.4)
        num_envs = 10_000
        command = _command(
            _cfg(probabilities=_probabilities(mixed=1.0), ranges=ranges), num_envs
        )
        torch.manual_seed(4321)
        command._resample_command(torch.arange(num_envs))
        absolute = command.vel_command_b.abs()
        self.assertTrue((absolute[:, 0] >= 0.10).all())
        self.assertTrue((absolute[:, 1] >= 0.08).all())
        self.assertTrue((absolute[:, 2] >= 0.40).all())
        self.assertTrue((absolute[:, 2] <= 1.50).all())
        for axis in range(3):
            positive_fraction = float(
                (command.vel_command_b[:, axis] > 0.0).float().mean()
            )
            self.assertAlmostEqual(positive_fraction, 0.5, delta=0.02)

    def test_runtime_probability_and_range_updates_apply_on_next_resample(self) -> None:
        cfg = _cfg(probabilities=_probabilities(forward=1.0), ranges=_ranges())
        command = _command(cfg, 64)
        env_ids = torch.arange(64)
        command._resample_command(env_ids)
        self.assertTrue((command.vel_command_b[:, 0] >= 0.10).all())
        self.assertTrue((command.vel_command_b[:, 0] <= 0.20).all())

        cfg.signed_axis_probabilities = _probabilities(backward=1.0)
        cfg.signed_axis_ranges["backward"] = (-0.40, -0.30)
        command._resample_command(env_ids)
        self.assertTrue((command.vel_command_b[:, 0] >= -0.40).all())
        self.assertTrue((command.vel_command_b[:, 0] <= -0.30).all())
        self.assertTrue(command.vel_command_b[:, 1:].eq(0.0).all())


if __name__ == "__main__":
    unittest.main()
