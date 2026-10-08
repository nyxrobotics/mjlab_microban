"""Single-foot targets with a standing twist (microban_teleop_foot_command.py)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from mjlab_microban.tasks import microban_teleop_foot_command as fc
from mjlab_microban.tasks.microban_teleop_mdp import ResetFixedFootTargetCommand


def _command(stationary_probability: float, n: int = 3):
    command = object.__new__(fc.StationaryFootTargetCommand)
    twist = torch.tensor([[0.3, 0.1, 0.5], [0.2, 0.0, 0.0], [-0.1, 0.0, 1.0]])[:n]
    velocity = SimpleNamespace(command_counter=torch.zeros(n), vel_command_b=twist.clone(),
                               vel_command_w=twist.clone(), is_rotation_env=torch.zeros(n, dtype=torch.bool))
    command._env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: velocity), device="cpu")
    command.cfg = SimpleNamespace(velocity_command_name="twist",
                                  single_support_stationary_probability=stationary_probability)
    command._reference_pending = torch.zeros(n, dtype=torch.bool)
    command.is_single_support_env = torch.zeros(n, dtype=torch.bool)
    command.is_both_feet_env = torch.zeros(n, dtype=torch.bool)
    command._previous_both_feet_env = torch.zeros(n, dtype=torch.bool)
    command._velocity_cache_valid = torch.zeros(n, dtype=torch.bool)
    command._velocity_command_counter = None
    command._saved_vel_command_b = command._saved_vel_command_w = command._saved_is_rotation_env = None
    command.is_stationary_single_support_env = torch.zeros(n, dtype=torch.bool)
    return command, velocity, twist


class StationaryFootTargetTest(unittest.TestCase):
    def test_a_stationary_single_foot_target_holds_the_twist_at_zero(self) -> None:
        command, velocity, twist = _command(0.5)
        command.is_single_support_env[:2] = True
        command.is_stationary_single_support_env[0] = True
        command._update_command()
        self.assertTrue(torch.equal(velocity.vel_command_b[0], torch.zeros(3)))
        self.assertTrue(torch.equal(velocity.vel_command_b[1:], twist[1:]))  # moving single foot, no target
        self.assertFalse(bool(command.is_both_feet_env.any()))  # restored
        # The target ends: the twist comes back at once.
        command.is_stationary_single_support_env[0] = False
        command.is_single_support_env[0] = False
        command._update_command()
        self.assertTrue(torch.equal(velocity.vel_command_b, twist))

    def test_two_foot_targets_still_stand(self) -> None:
        command, velocity, twist = _command(0.0)
        command.is_both_feet_env[2] = True
        command._update_command()
        self.assertTrue(torch.equal(velocity.vel_command_b[2], torch.zeros(3)))
        self.assertTrue(torch.equal(velocity.vel_command_b[:2], twist[:2]))
        self.assertTrue(bool(command.is_both_feet_env[2]))

    def test_the_share_of_stationary_single_foot_targets(self) -> None:
        def parent(self, env_ids):
            self.is_single_support_env[env_ids] = True

        for probability, expected in ((0.0, 0), (1.0, 4000)):
            command, _velocity, _twist = _command(probability)
            n = 4000
            command.is_single_support_env = torch.zeros(n, dtype=torch.bool)
            command.is_stationary_single_support_env = torch.zeros(n, dtype=torch.bool)
            with mock.patch.object(ResetFixedFootTargetCommand, "_resample_command", parent):
                command._resample_command(torch.arange(n))
            self.assertEqual(int(command.is_stationary_single_support_env.sum()), expected)
        command, _velocity, _twist = _command(0.5)
        n = 4000
        command.is_single_support_env = torch.zeros(n, dtype=torch.bool)
        command.is_stationary_single_support_env = torch.zeros(n, dtype=torch.bool)
        torch.manual_seed(0)
        with mock.patch.object(ResetFixedFootTargetCommand, "_resample_command", parent):
            command._resample_command(torch.arange(n))
        self.assertAlmostEqual(float(command.is_stationary_single_support_env.float().mean()), 0.5, delta=0.03)

    def test_the_foot_stage_sets_half(self) -> None:
        from mjlab_microban.tasks.microban_teleop_env_cfg import TELEOP_STAGES

        values = [s.value for stage in TELEOP_STAGES for s in stage.settings
                  if s.path == "single_support_stationary_probability"]
        self.assertEqual(values, [0.5])


if __name__ == "__main__":
    unittest.main()
