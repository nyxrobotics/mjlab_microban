"""Regression tests for the Microban motion-tracking training contract."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from mjlab_microban.tasks.microban_tracking_env_cfg import (
    DEFAULT_MICROBAN_TRACKING_MOTION_FILE,
    MICROBAN_TRACKING_ACTION_JOINT_NAMES,
    MICROBAN_TRACKING_CLIP_DURATION_S,
    MICROBAN_TRACKING_CLIP_STEP_COUNT,
    MicrobanTrackingRlCfg,
    make_microban_tracking_env_cfg,
    microban_tracking_action_delta_bounds,
    microban_tracking_initial_action_std,
)
from mjlab_microban.tasks.microban_tracking_mdp import (
    MICROBAN_TRACKING_ARM_ACTION_IDS,
    MicrobanTrackingBoundedGaussianDistribution,
    motion_anchor_planar_position_error_l1,
    motion_anchor_planar_velocity_error_l1,
)
from mjlab_microban.tasks.microban_tracking_policy_export import (
    MicrobanTrackingOnPolicyRunner,
)


class _CommandManager:
    def __init__(self, command: object) -> None:
        self.command = command

    def get_term(self, name: str) -> object:
        if name != "motion":
            raise KeyError(name)
        return self.command


class TrackingRewardTest(unittest.TestCase):
    def test_planar_errors_are_l1_and_ignore_height(self) -> None:
        command = SimpleNamespace(
            anchor_pos_w=torch.tensor([[1.0, -2.0, 100.0], [-1.0, 4.0, -100.0]]),
            robot_anchor_pos_w=torch.tensor([[0.5, 1.0, -100.0], [2.0, 1.0, 100.0]]),
            anchor_lin_vel_w=torch.tensor([[0.1, -0.2, 50.0], [-0.3, 0.4, -50.0]]),
            robot_anchor_lin_vel_w=torch.tensor([[-0.1, 0.1, -50.0], [0.2, 0.0, 50.0]]),
        )
        env = SimpleNamespace(command_manager=_CommandManager(command))

        self.assertTrue(
            torch.equal(
                motion_anchor_planar_position_error_l1(env, "motion"),
                torch.tensor([3.5, 6.0]),
            )
        )
        self.assertTrue(
            torch.allclose(
                motion_anchor_planar_velocity_error_l1(env, "motion"),
                torch.tensor([0.5, 0.9]),
            )
        )


class TrackingConfigTest(unittest.TestCase):
    def test_actor_is_bounded_with_safe_per_joint_std(self) -> None:
        lower, upper = microban_tracking_action_delta_bounds()
        std = microban_tracking_initial_action_std()

        self.assertEqual(len(MICROBAN_TRACKING_ACTION_JOINT_NAMES), 18)
        self.assertEqual(len(lower), 18)
        self.assertEqual(len(upper), 18)
        self.assertEqual(len(std), 18)
        self.assertTrue(all(lo < 0.0 < hi for lo, hi in zip(lower, upper, strict=True)))
        self.assertAlmostEqual(std[1], std[10])
        self.assertLess(std[1], 0.006)
        self.assertTrue(
            all(std[index] == 0.15 for index in range(18) if index not in (1, 10))
        )
        self.assertIs(
            MicrobanTrackingRlCfg.actor.distribution_cfg["class_name"],
            MicrobanTrackingBoundedGaussianDistribution,
        )
        self.assertFalse(MicrobanTrackingRlCfg.actor.obs_normalization)
        self.assertTrue(MicrobanTrackingRlCfg.critic.obs_normalization)
        self.assertEqual(MicrobanTrackingRlCfg.algorithm.entropy_coef, 0.0)
        self.assertEqual(MicrobanTrackingRlCfg.algorithm.num_learning_epochs, 3)
        self.assertEqual(MicrobanTrackingRlCfg.algorithm.learning_rate, 3.0e-5)

    def test_tracking_distribution_initializes_all_arms_at_home(self) -> None:
        lower, upper = microban_tracking_action_delta_bounds()
        distribution = MicrobanTrackingBoundedGaussianDistribution(
            18,
            microban_tracking_initial_action_std(),
            lower,
            upper,
            std_type="log",
        )
        mlp = torch.nn.Sequential(torch.nn.Linear(99, 32), torch.nn.Linear(32, 18))

        distribution.init_mlp_weights(mlp)

        final = mlp[-1]
        ids = list(MICROBAN_TRACKING_ARM_ACTION_IDS)
        self.assertTrue(
            torch.equal(final.weight[ids], torch.zeros_like(final.weight[ids]))
        )
        self.assertTrue(torch.equal(final.bias[ids], torch.zeros_like(final.bias[ids])))

    def test_training_config_rejects_stationary_survival_solution(self) -> None:
        cfg = make_microban_tracking_env_cfg()

        self.assertEqual(
            cfg.commands["motion"].motion_file,
            str(DEFAULT_MICROBAN_TRACKING_MOTION_FILE.resolve()),
        )
        self.assertAlmostEqual(cfg.episode_length_s, MICROBAN_TRACKING_CLIP_DURATION_S)
        self.assertEqual(MICROBAN_TRACKING_CLIP_STEP_COUNT, 267)
        self.assertTrue(cfg.is_finite_horizon)
        self.assertAlmostEqual(
            cfg.rewards["motion_global_root_pos"].params["std"], 0.20
        )
        self.assertAlmostEqual(cfg.rewards["motion_global_root_pos"].weight, 2.0)
        self.assertAlmostEqual(
            cfg.rewards["motion_anchor_planar_position_l1"].weight, -2.0
        )
        self.assertAlmostEqual(
            cfg.rewards["motion_anchor_planar_velocity_l1"].weight, -4.0
        )
        self.assertGreaterEqual(cfg.sim.nconmax, 512)
        self.assertGreaterEqual(cfg.sim.njmax, 2048)

    def test_play_config_does_not_time_out(self) -> None:
        cfg = make_microban_tracking_env_cfg(play=True)

        self.assertEqual(cfg.episode_length_s, int(1e9))
        self.assertEqual(cfg.commands["motion"].sampling_mode, "start")

    def test_runner_disables_generic_random_episode_length(self) -> None:
        runner = object.__new__(MicrobanTrackingOnPolicyRunner)
        parent = MicrobanTrackingOnPolicyRunner.__mro__[1]

        with patch.object(parent, "learn", autospec=True) as learn:
            runner.learn(7, init_at_random_ep_len=True)

        learn.assert_called_once_with(
            runner,
            num_learning_iterations=7,
            init_at_random_ep_len=False,
        )


if __name__ == "__main__":
    unittest.main()
