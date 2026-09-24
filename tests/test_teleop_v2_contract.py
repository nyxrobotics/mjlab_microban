# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Focused tests for the Microban teleop-v5 training/deployment contract."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from mjlab.envs.mdp.actions import JointPositionAction
from mjlab.rl.runner import MjlabOnPolicyRunner
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from mjlab_microban.tasks.mdp import no_stepping_penalty
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_TELEOP_ACTION_JOINT_NAMES,
    MICROBAN_TELEOP_PREVIOUS_ACTION_SEMANTICS,
    MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION,
    MicrobanTeleopOnPolicyRunner,
    TeleopCheckpointContract,
    _command_target_bounds,
    validate_bounded_actor_checkpoint_buffers,
    validate_teleop_checkpoint_contract,
)
from mjlab_microban.tasks.microban_teleop_env_cfg import (
    MICROBAN_TELEOP_ANGULAR_TRACKING_STD_RAD_S,
    MICROBAN_TELEOP_FINAL_VELOCITY_ENVELOPE,
    MICROBAN_TELEOP_FOOT_TRACKING_FINAL_STD_M,
    MICROBAN_TELEOP_HAND_TRACKING_FINAL_STD_M,
    MICROBAN_TELEOP_HAND_TRACKING_STD_M,
    MICROBAN_TELEOP_INITIAL_VELOCITY_ENVELOPE,
    MICROBAN_TELEOP_LINEAR_TRACKING_STD_M_S,
    MicrobanTeleopRlCfg,
    make_microban_teleop_env_cfg,
    microban_teleop_action_delta_bounds,
    microban_teleop_initial_action_std,
)
from mjlab_microban.tasks.microban_teleop_mdp import (
    MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M,
    AsymmetricBoundedGaussianDistribution,
    PerJointGaussianDistribution,
    ResetFixedFootTargetCommand,
    effective_action_after_target_clip,
    linear_velocity_tracking_error_l1,
    normalized_target_clip_excess_l1_sum,
    normalized_target_near_limit_l1_sum,
    raw_action_l2,
    yaw_velocity_tracking_error_l1,
)


def _action_env(
    raw: torch.Tensor,
    *,
    offset: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    scale: torch.Tensor | float = 1.0,
) -> SimpleNamespace:
    """Construct only the initialized JointPositionAction fields terms use."""

    action = object.__new__(JointPositionAction)
    action._raw_actions = raw
    action._scale = scale
    action._offset = offset
    action._clip = torch.stack((lower, upper), dim=-1)
    action.cfg = SimpleNamespace(clip={".*": (-1.0, 1.0)})
    manager = SimpleNamespace(get_term=lambda name: action)
    return SimpleNamespace(action_manager=manager)


class EffectiveActionTest(unittest.TestCase):
    def test_effective_action_and_normalized_l1_sum_penalty(self) -> None:
        raw = torch.tensor([[3.0, -2.0, 0.25], [0.0, 0.0, 0.0]])
        offset = torch.tensor([[0.5, -0.5, 0.0], [0.5, -0.5, 0.0]])
        lower = torch.tensor([[-1.0, -1.0, -2.0], [-1.0, -1.0, -2.0]])
        upper = torch.tensor([[1.0, 1.0, 2.0], [1.0, 1.0, 2.0]])
        env = _action_env(
            raw,
            offset=offset,
            lower=lower,
            upper=upper,
            scale=torch.tensor(2.0),
        )

        effective = effective_action_after_target_clip(env)
        expected = torch.tensor([[0.25, -0.25, 0.25], [0.0, 0.0, 0.0]])
        torch.testing.assert_close(effective, expected)

        target = raw * 2.0 + offset
        clipped = torch.clamp(target, min=lower, max=upper)
        normalized = (target - clipped) / (0.5 * (upper - lower))
        expected_penalty = normalized.abs().sum(dim=-1)
        torch.testing.assert_close(
            normalized_target_clip_excess_l1_sum(env), expected_penalty
        )
        torch.testing.assert_close(raw_action_l2(env), raw.square().mean(dim=-1))

    def test_penalty_is_zero_at_and_inside_target_limits(self) -> None:
        raw = torch.tensor([[0.5, -0.5]])
        env = _action_env(
            raw,
            offset=torch.zeros_like(raw),
            lower=-torch.ones_like(raw),
            upper=torch.ones_like(raw),
        )
        torch.testing.assert_close(
            normalized_target_clip_excess_l1_sum(env), torch.zeros(1)
        )

    def test_sparse_clip_penalty_is_not_joint_averaged_or_quadratic(self) -> None:
        raw = torch.tensor([[1.02, 0.0, 0.0, 0.0]])
        env = _action_env(
            raw,
            offset=torch.zeros_like(raw),
            lower=-torch.ones_like(raw),
            upper=torch.ones_like(raw),
        )
        torch.testing.assert_close(
            normalized_target_clip_excess_l1_sum(env), torch.tensor([0.02])
        )

    def test_near_limit_penalty_is_asymmetric_and_default_is_always_free(self) -> None:
        raw = torch.tensor([[0.03, -0.15, 0.15], [0.0, 0.0, 0.0]])
        offset = torch.tensor([[0.95, 0.0, 0.0], [0.95, 0.0, 0.0]])
        env = _action_env(
            raw,
            offset=offset,
            lower=-torch.ones_like(raw),
            upper=torch.ones_like(raw),
        )
        penalty = normalized_target_near_limit_l1_sum(env, margin_ratio=0.1)
        # Joint 0's default is inside the nominal upper margin. It must remain a
        # valid zero-cost target, while excursions beyond that default are costly.
        self.assertEqual(penalty[1].item(), 0.0)
        self.assertGreater(penalty[0].item(), 0.0)


class AcceptanceAlignedRewardTest(unittest.TestCase):
    def test_velocity_l1_terms_match_body_frame_acceptance_errors(self) -> None:
        entity = SimpleNamespace(
            data=SimpleNamespace(
                root_link_lin_vel_b=torch.tensor([[0.1, -0.2, 9.0], [-0.4, 0.3, -9.0]]),
                root_link_ang_vel_b=torch.tensor([[8.0, 7.0, -0.25], [6.0, 5.0, 0.75]]),
            )
        )
        command = torch.tensor([[0.4, 0.2, 0.75], [-0.4, -0.1, -0.25]])
        env = SimpleNamespace(
            scene={"robot": entity},
            command_manager=SimpleNamespace(get_command=lambda name: command),
        )
        torch.testing.assert_close(
            linear_velocity_tracking_error_l1(env),
            torch.tensor([0.7, 0.4]),
        )
        torch.testing.assert_close(
            yaw_velocity_tracking_error_l1(env),
            torch.tensor([1.0, 1.0]),
        )

    def test_no_stepping_exempts_only_active_foot_target_rows(self) -> None:
        command = torch.zeros((4, 3))
        foot_target = SimpleNamespace(
            is_single_support_env=torch.tensor([False, True, False, False]),
            is_both_feet_env=torch.tensor([False, False, True, False]),
        )
        command_manager = SimpleNamespace(
            get_command=lambda name: command,
            get_term=lambda name: foot_target,
        )
        sensor = SimpleNamespace(
            data=SimpleNamespace(
                found=torch.tensor(
                    [[False, True], [False, True], [False, False], [True, True]]
                )
            )
        )
        env = SimpleNamespace(
            num_envs=4,
            command_manager=command_manager,
            scene=SimpleNamespace(sensors={"feet": sensor}),
        )
        torch.testing.assert_close(
            no_stepping_penalty(
                env,
                sensor_name="feet",
                foot_target_command_name="foot_target",
            ),
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
        )


class PerJointGaussianTest(unittest.TestCase):
    def test_exact_vector_initialization_in_scalar_and_log_space(self) -> None:
        expected = torch.tensor([0.01, 0.2, 0.7])
        for std_type in ("scalar", "log"):
            with self.subTest(std_type=std_type):
                distribution = PerJointGaussianDistribution(
                    3, expected.tolist(), std_type=std_type
                )
                distribution.update(torch.zeros((2, 3)))
                torch.testing.assert_close(
                    distribution.std,
                    expected.expand(2, -1),
                    rtol=1e-6,
                    atol=1e-7,
                )

    def test_rejects_wrong_width_and_nonpositive_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "contain 3"):
            PerJointGaussianDistribution(3, [0.1, 0.2])
        with self.assertRaisesRegex(ValueError, "positive"):
            PerJointGaussianDistribution(2, [0.1, 0.0])


class AsymmetricBoundedGaussianTest(unittest.TestCase):
    LOWER = (-0.3, -0.4, -0.8)
    UPPER = (0.5, 0.6, 0.2)
    STD = (0.1, 0.2, 0.05)

    def _distribution(self) -> AsymmetricBoundedGaussianDistribution:
        return AsymmetricBoundedGaussianDistribution(
            3,
            self.STD,
            self.LOWER,
            self.UPPER,
            std_type="log",
        )

    def test_zero_anchor_and_finite_extremes_are_strictly_inside_bounds(self) -> None:
        distribution = self._distribution()
        latent = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0e30, -1.0e30, 1.0e30]],
            dtype=torch.float32,
        )
        action = distribution.deterministic_output(latent)
        torch.testing.assert_close(action[0], torch.zeros(3))
        lower = torch.tensor(self.LOWER)
        upper = torch.tensor(self.UPPER)
        self.assertTrue(torch.all(action > lower).item())
        self.assertTrue(torch.all(action < upper).item())
        self.assertTrue(
            torch.isfinite(distribution._log_abs_det_jacobian(latent)).all()
        )

    def test_sample_storage_inverse_log_prob_is_consistent(self) -> None:
        torch.manual_seed(1234)
        mean = torch.tensor([[0.02, -0.03, 0.04], [-0.1, 0.2, -0.05]])
        distribution = self._distribution()
        distribution.update(mean)
        action = distribution.sample().detach()
        rollout_log_prob = distribution.log_prob(action).detach()

        # A PPO minibatch reconstructs the distribution later and evaluates the
        # stored bounded action through the inverse transform.
        distribution.update(mean.clone())
        replay_log_prob = distribution.log_prob(action)
        torch.testing.assert_close(replay_log_prob, rollout_log_prob)
        torch.testing.assert_close(
            torch.exp(replay_log_prob - rollout_log_prob), torch.ones(2)
        )

        latent = distribution._inverse(action)
        latent_dist = torch.distributions.Normal(
            mean, torch.tensor(self.STD).expand_as(mean)
        )
        expected = latent_dist.log_prob(latent).sum(dim=-1)
        expected -= distribution._log_abs_det_jacobian(latent).sum(dim=-1)
        torch.testing.assert_close(replay_log_prob, expected)

    def test_round_trip_handles_tiny_asymmetric_shoulder_headroom(self) -> None:
        lower, upper = microban_teleop_action_delta_bounds()
        shoulder = MICROBAN_TELEOP_ACTION_JOINT_NAMES.index("right_shoulder_roll")
        self.assertLess(min(-lower[shoulder], upper[shoulder]), math.radians(1.01))
        distribution = AsymmetricBoundedGaussianDistribution(
            1,
            [0.002],
            [lower[shoulder]],
            [upper[shoulder]],
        )
        negative_scale = -lower[shoulder]
        positive_scale = upper[shoulder]
        latent = torch.tensor(
            [[-0.75 * negative_scale], [0.0], [0.75 * positive_scale]]
        )
        action = distribution.deterministic_output(latent)
        round_trip = distribution._inverse(action)
        torch.testing.assert_close(round_trip, latent, rtol=2e-6, atol=1e-8)

    def test_exact_or_outside_bounds_fail_instead_of_clamping_inverse(self) -> None:
        distribution = self._distribution()
        distribution.update(torch.zeros((1, 3)))
        for invalid in (
            torch.tensor([[self.UPPER[0], 0.0, 0.0]]),
            torch.tensor([[self.LOWER[0], 0.0, 0.0]]),
            torch.tensor([[float("nan"), 0.0, 0.0]]),
        ):
            with (
                self.subTest(invalid=invalid),
                self.assertRaisesRegex(ValueError, "strictly inside"),
            ):
                distribution.log_prob(invalid)

    def test_fixed_bijection_kl_equals_latent_gaussian_kl(self) -> None:
        distribution = self._distribution()
        old_mean = torch.tensor([[0.1, -0.2, 0.0], [0.0, 0.1, -0.1]])
        old_std = torch.tensor(self.STD).expand_as(old_mean)
        new_mean = old_mean + torch.tensor([0.03, -0.01, 0.02])
        new_std = old_std * torch.tensor([1.1, 0.9, 1.2])
        actual = distribution.kl_divergence((old_mean, old_std), (new_mean, new_std))
        expected = torch.distributions.kl_divergence(
            torch.distributions.Normal(old_mean, old_std),
            torch.distributions.Normal(new_mean, new_std),
        ).sum(dim=-1)
        torch.testing.assert_close(actual, expected)

    def test_entropy_and_export_module_are_finite_and_action_shaped(self) -> None:
        torch.manual_seed(7)
        mean = torch.tensor([[0.0, 0.0, 0.0], [0.15, -0.25, 0.1]], requires_grad=True)
        distribution = self._distribution()
        distribution.update(mean)
        distribution.sample()
        entropy = distribution.entropy
        self.assertEqual(entropy.shape, (2,))
        self.assertTrue(torch.isfinite(entropy).all().item())
        entropy.sum().backward()
        self.assertIsNotNone(mean.grad)
        self.assertTrue(torch.isfinite(mean.grad).all().item())

        deterministic = distribution.deterministic_output(mean.detach())
        export_output = distribution.as_deterministic_output_module()(mean.detach())
        torch.testing.assert_close(export_output, deterministic)

    def test_rsl_actor_replay_of_stored_action_has_unit_ppo_ratio(self) -> None:
        torch.manual_seed(19)
        obs = TensorDict({"policy": torch.randn(8, 5)}, batch_size=[8])
        actor = MLPModel(
            obs=obs,
            obs_groups={"actor": ["policy"]},
            obs_set="actor",
            output_dim=3,
            hidden_dims=(16,),
            distribution_cfg={
                "class_name": AsymmetricBoundedGaussianDistribution,
                "init_std": self.STD,
                "lower_bound": self.LOWER,
                "upper_bound": self.UPPER,
                "std_type": "log",
            },
        )
        stored_action = actor(obs, stochastic_output=True).detach()
        old_log_prob = actor.get_output_log_prob(stored_action).detach()
        old_params = tuple(
            value.detach().clone() for value in actor.output_distribution_params
        )

        # This is the PPO update path: a fresh stochastic forward updates the
        # distribution, then log_prob is evaluated for the stored rollout action.
        actor(obs, stochastic_output=True)
        replay_log_prob = actor.get_output_log_prob(stored_action)
        new_params = actor.output_distribution_params
        ratio = torch.exp(replay_log_prob - old_log_prob)
        torch.testing.assert_close(ratio, torch.ones_like(ratio))
        torch.testing.assert_close(
            actor.get_kl_divergence(old_params, new_params), torch.zeros(8)
        )

    def test_rejects_invalid_widths_bounds_and_std(self) -> None:
        with self.assertRaisesRegex(ValueError, "lower_bound must contain 3"):
            AsymmetricBoundedGaussianDistribution(
                3, self.STD, self.LOWER[:2], self.UPPER
            )
        with self.assertRaisesRegex(ValueError, "strictly negative"):
            AsymmetricBoundedGaussianDistribution(
                3, self.STD, (-0.3, 0.0, -0.8), self.UPPER
            )
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            AsymmetricBoundedGaussianDistribution(
                3, self.STD, self.LOWER, (0.5, 0.0, 0.2)
            )


class TeleopConfigurationTest(unittest.TestCase):
    def test_export_separates_single_and_simultaneous_foot_support(self) -> None:
        foot_cfg = SimpleNamespace(
            reach_xy_range=(-0.03, 0.03),
            lift_height_range=(MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M, 0.05),
            both_feet_reach_xy_range=(-0.01, 0.01),
            both_feet_lift_height_range=(
                MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M,
                0.02,
            ),
        )
        hand_cfg = SimpleNamespace(
            reach_xy_range=(-0.08, 0.08),
            reach_z_range=(-0.08, 0.08),
        )
        manager = SimpleNamespace(
            get_term_cfg=lambda name: {
                "foot_target": foot_cfg,
                "hand_target": hand_cfg,
            }[name]
        )
        bounds = _command_target_bounds(SimpleNamespace(command_manager=manager))
        self.assertEqual(bounds[0], [-0.03, -0.03, 0.0] * 2)
        self.assertEqual(bounds[1], [0.03, 0.03, 0.05] * 2)
        self.assertEqual(bounds[2], [-0.01, -0.01, 0.0] * 2)
        self.assertEqual(bounds[3], [0.01, 0.01, 0.02] * 2)

    def test_ordered_std_vector_and_reward_contract(self) -> None:
        std = microban_teleop_initial_action_std()
        self.assertEqual(len(std), len(MICROBAN_TELEOP_ACTION_JOINT_NAMES))
        expected_shoulder_roll = math.radians(1.0) / 3.0
        for name in ("right_shoulder_roll", "left_shoulder_roll"):
            self.assertAlmostEqual(
                std[MICROBAN_TELEOP_ACTION_JOINT_NAMES.index(name)],
                expected_shoulder_roll,
                places=7,
            )
        self.assertTrue(all(0.0 < value <= 0.15 for value in std))

        cfg = make_microban_teleop_env_cfg()
        self.assertIs(
            cfg.observations["actor"].terms["actions"].func,
            effective_action_after_target_clip,
        )
        self.assertIs(
            cfg.observations["critic"].terms["actions"].func,
            effective_action_after_target_clip,
        )
        self.assertEqual(cfg.rewards["target_clip_excess"].weight, -2.0)
        self.assertEqual(cfg.rewards["target_near_limit"].weight, -1.0)
        self.assertEqual(cfg.rewards["raw_action_l2"].weight, -0.01)
        self.assertEqual(cfg.rewards["action_rate_l2"].weight, -0.02)
        self.assertEqual(cfg.rewards["linear_velocity_error_l1"].weight, -2.0)
        self.assertEqual(cfg.rewards["yaw_velocity_error_l1"].weight, -0.5)
        self.assertEqual(cfg.rewards["dof_pos_limits"].weight, -10.0)
        self.assertEqual(cfg.rewards["feet_distance"].weight, -100.0)
        self.assertEqual(cfg.rewards["feet_distance"].params["min_dist"], 0.07)
        self.assertEqual(
            cfg.rewards["no_stepping"].params["foot_target_command_name"],
            "foot_target",
        )
        self.assertEqual(
            cfg.rewards["track_linear_velocity"].params["std"],
            MICROBAN_TELEOP_LINEAR_TRACKING_STD_M_S,
        )
        self.assertEqual(
            cfg.rewards["track_angular_velocity"].params["std"],
            MICROBAN_TELEOP_ANGULAR_TRACKING_STD_RAD_S,
        )
        self.assertEqual(
            cfg.rewards["hand_target_tracking"].params["std"],
            MICROBAN_TELEOP_HAND_TRACKING_STD_M,
        )
        self.assertEqual(MicrobanTeleopRlCfg.algorithm.entropy_coef, 0.0)
        self.assertEqual(MicrobanTeleopRlCfg.algorithm.learning_rate, 1.0e-4)
        self.assertEqual(MicrobanTeleopRlCfg.algorithm.num_learning_epochs, 3)
        self.assertEqual(MicrobanTeleopRlCfg.algorithm.schedule, "fixed")
        self.assertIs(
            MicrobanTeleopRlCfg.actor.distribution_cfg["class_name"],
            AsymmetricBoundedGaussianDistribution,
        )
        self.assertEqual(
            tuple(MicrobanTeleopRlCfg.actor.distribution_cfg["init_std"]), std
        )
        lower, upper = microban_teleop_action_delta_bounds()
        self.assertEqual(
            tuple(MicrobanTeleopRlCfg.actor.distribution_cfg["lower_bound"]), lower
        )
        self.assertEqual(
            tuple(MicrobanTeleopRlCfg.actor.distribution_cfg["upper_bound"]), upper
        )
        self.assertTrue(all(value < 0.0 for value in lower))
        self.assertTrue(all(value > 0.0 for value in upper))
        defaults = cfg.scene.entities["robot"].init_state.joint_pos
        assert defaults is not None
        clips = cfg.actions["joint_pos"].clip
        assert clips is not None
        for index, name in enumerate(MICROBAN_TELEOP_ACTION_JOINT_NAMES):
            physical_lower = clips[name][0] - defaults[name]
            physical_upper = clips[name][1] - defaults[name]
            self.assertGreater(lower[index], physical_lower)
            self.assertLess(upper[index], physical_upper)
        for name in ("right_shoulder_roll", "left_shoulder_roll"):
            index = MICROBAN_TELEOP_ACTION_JOINT_NAMES.index(name)
            self.assertAlmostEqual(min(-lower[index], upper[index]), 1.0e-4)

    def test_curriculum_finishes_at_runtime_envelope_and_both_feet(self) -> None:
        self.assertEqual(MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M, 0.0025)
        cfg = make_microban_teleop_env_cfg()
        stages = cfg.curriculum["staged_curriculum"].params["stages"]
        self.assertEqual(
            cfg.commands["twist"].ranges.lin_vel_x,
            MICROBAN_TELEOP_INITIAL_VELOCITY_ENVELOPE["lin_vel_x"],
        )
        self.assertEqual(
            cfg.commands["twist"].ranges.lin_vel_y,
            MICROBAN_TELEOP_INITIAL_VELOCITY_ENVELOPE["lin_vel_y"],
        )
        self.assertEqual(
            cfg.commands["twist"].ranges.ang_vel_z,
            MICROBAN_TELEOP_INITIAL_VELOCITY_ENVELOPE["ang_vel_z"],
        )
        self.assertEqual(
            cfg.commands["twist"].rotation_env_ang_vel_range,
            MICROBAN_TELEOP_INITIAL_VELOCITY_ENVELOPE["rotation_ang_vel_z"],
        )
        self.assertEqual(
            [stage["step"] for stage in stages],
            [
                1000 * 24,
                2500 * 24,
                4500 * 24,
                6500 * 24,
                7500 * 24,
                9500 * 24,
            ],
        )
        self.assertEqual(cfg.commands["foot_target"].rel_both_feet_envs, 0.0)
        self.assertEqual(
            cfg.commands["foot_target"].lift_height_range,
            (MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M, 0.05),
        )
        self.assertEqual(
            cfg.commands["foot_target"].both_feet_lift_height_range,
            (MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M, 0.012),
        )

        twist = SimpleNamespace(
            cfg=SimpleNamespace(
                ranges=SimpleNamespace(
                    lin_vel_x=(0.0, 0.0),
                    lin_vel_y=(0.0, 0.0),
                    ang_vel_z=(0.0, 0.0),
                ),
                rotation_env_ang_vel_range=(0.0, 0.0),
                rel_standing_envs=0.0,
                rel_rotation_envs=0.0,
            )
        )
        foot = SimpleNamespace(cfg=cfg.commands["foot_target"])
        hand = SimpleNamespace(cfg=cfg.commands["hand_target"])
        reward_cfgs = {
            name: SimpleNamespace(weight=term.weight, params=dict(term.params))
            for name, term in cfg.rewards.items()
        }
        command_manager = SimpleNamespace(
            get_term_cfg=lambda name: {
                "twist": twist.cfg,
                "foot_target": foot.cfg,
                "hand_target": hand.cfg,
            }[name]
        )
        reward_manager = SimpleNamespace(get_term_cfg=lambda name: reward_cfgs[name])
        env = SimpleNamespace(
            command_manager=command_manager,
            reward_manager=reward_manager,
        )
        expected_both_lift_ranges = (
            (MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M, 0.012),
            (MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M, 0.012),
            (MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M, 0.012),
            (MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M, 0.012),
            (MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M, 0.012),
            (MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M, 0.02),
        )
        for stage, expected_both_lift_range in zip(
            stages, expected_both_lift_ranges, strict=True
        ):
            stage["apply"](env)
            self.assertEqual(
                foot.cfg.lift_height_range,
                (MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M, 0.05),
            )
            self.assertEqual(
                foot.cfg.both_feet_lift_height_range,
                expected_both_lift_range,
            )
        self.assertEqual(
            twist.cfg.ranges.lin_vel_x,
            MICROBAN_TELEOP_FINAL_VELOCITY_ENVELOPE["lin_vel_x"],
        )
        self.assertEqual(
            twist.cfg.ranges.lin_vel_y,
            MICROBAN_TELEOP_FINAL_VELOCITY_ENVELOPE["lin_vel_y"],
        )
        self.assertEqual(
            twist.cfg.ranges.ang_vel_z,
            MICROBAN_TELEOP_FINAL_VELOCITY_ENVELOPE["ang_vel_z"],
        )
        self.assertEqual(
            twist.cfg.rotation_env_ang_vel_range,
            MICROBAN_TELEOP_FINAL_VELOCITY_ENVELOPE["rotation_ang_vel_z"],
        )
        self.assertEqual(twist.cfg.rel_rotation_envs, 0.25)
        self.assertEqual(foot.cfg.rel_both_feet_envs, 0.1)
        self.assertEqual(
            foot.cfg.both_feet_lift_height_range,
            (MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M, 0.02),
        )
        self.assertEqual(reward_cfgs["hand_target_tracking"].weight, 2.0)
        self.assertEqual(
            reward_cfgs["hand_target_tracking"].params["std"],
            MICROBAN_TELEOP_HAND_TRACKING_FINAL_STD_M,
        )
        self.assertEqual(reward_cfgs["foot_target_tracking"].weight, 3.0)
        self.assertEqual(
            reward_cfgs["foot_target_tracking"].params["std"],
            MICROBAN_TELEOP_FOOT_TRACKING_FINAL_STD_M,
        )

    def test_both_foot_targets_force_only_their_rows_stationary(self) -> None:
        velocity = SimpleNamespace(
            vel_command_b=torch.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
            ),
            vel_command_w=torch.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
            ),
            is_rotation_env=torch.tensor([True, True, True]),
            command_counter=torch.zeros(3, dtype=torch.long),
        )
        term = object.__new__(ResetFixedFootTargetCommand)
        term.is_both_feet_env = torch.tensor([True, False, True])
        term._previous_both_feet_env = torch.zeros(3, dtype=torch.bool)
        term._velocity_cache_valid = torch.zeros(3, dtype=torch.bool)
        term._velocity_command_counter = None
        term._saved_vel_command_b = None
        term._saved_vel_command_w = None
        term._saved_is_rotation_env = None
        term._reference_pending = torch.zeros(3, dtype=torch.bool)
        term.cfg = SimpleNamespace(velocity_command_name="twist")
        term._env = SimpleNamespace(
            command_manager=SimpleNamespace(get_term=lambda name: velocity)
        )

        term._update_command()
        torch.testing.assert_close(
            velocity.vel_command_b,
            torch.tensor([[0.0, 0.0, 0.0], [4.0, 5.0, 6.0], [0.0, 0.0, 0.0]]),
        )
        torch.testing.assert_close(
            velocity.vel_command_w,
            torch.tensor([[0.0, 0.0, 0.0], [4.0, 5.0, 6.0], [0.0, 0.0, 0.0]]),
        )
        self.assertEqual(velocity.is_rotation_env.tolist(), [False, True, False])

        # Leaving the both-feet regime must immediately restore the command
        # masked on entry, without waiting for the independent twist timer.
        term.is_both_feet_env[:] = False
        term._update_command()
        torch.testing.assert_close(
            velocity.vel_command_b,
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        )
        torch.testing.assert_close(
            velocity.vel_command_w,
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        )
        self.assertEqual(velocity.is_rotation_env.tolist(), [True, True, True])

    def test_episode_reset_invalidates_equal_counter_velocity_cache(self) -> None:
        fresh_twist = torch.tensor([[10.0, 20.0, 30.0], [4.0, 5.0, 6.0]])
        velocity = SimpleNamespace(
            vel_command_b=fresh_twist.clone(),
            vel_command_w=fresh_twist.clone(),
            is_rotation_env=torch.tensor([True, False]),
            # CommandTerm.reset() commonly returns this counter to the same
            # post-resample value used by the preceding episode.
            command_counter=torch.tensor([1, 5], dtype=torch.long),
        )
        term = object.__new__(ResetFixedFootTargetCommand)
        term.is_both_feet_env = torch.tensor([False, False])
        term._previous_both_feet_env = torch.tensor([True, False])
        term._velocity_cache_valid = torch.tensor([True, True])
        term._velocity_command_counter = torch.tensor([1, 5], dtype=torch.long)
        term._saved_vel_command_b = torch.tensor([[-1.0, -2.0, -3.0], [4.0, 5.0, 6.0]])
        term._saved_vel_command_w = term._saved_vel_command_b.clone()
        term._saved_is_rotation_env = torch.tensor([False, False])
        term._reference_pending = torch.zeros(2, dtype=torch.bool)
        term.cfg = SimpleNamespace(velocity_command_name="twist")
        term._env = SimpleNamespace(
            command_manager=SimpleNamespace(get_term=lambda name: velocity)
        )

        # Supply only the CommandTerm fields needed by its reset implementation;
        # foot sampling itself is not under test here.
        term.metrics = {}
        term.command_counter = torch.ones(2, dtype=torch.long)
        term._resample = lambda env_ids: None
        term._capture_pending_reference = lambda: None

        reset_ids = torch.tensor([0])
        term.reset(reset_ids)
        self.assertFalse(term._previous_both_feet_env[0].item())
        self.assertFalse(term._velocity_cache_valid[0].item())
        self.assertTrue(term._velocity_cache_valid[1].item())

        # Even though the velocity counter is unchanged, the first post-reset
        # update must snapshot the fresh command instead of retaining episode N.
        term._update_command()
        torch.testing.assert_close(term._saved_vel_command_b[0], fresh_twist[0])
        torch.testing.assert_close(term._saved_vel_command_w[0], fresh_twist[0])
        self.assertTrue(term._saved_is_rotation_env[0].item())
        torch.testing.assert_close(velocity.vel_command_b, fresh_twist)
        torch.testing.assert_close(velocity.vel_command_w, fresh_twist)


class CheckpointContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _save(
        self,
        iteration: int,
        *,
        filename_iteration: int | None = None,
        contract: bool,
    ) -> Path:
        filename_iteration = (
            iteration if filename_iteration is None else filename_iteration
        )
        infos: dict[str, object] = {
            "env_state": {"common_step_counter": (iteration + 1) * 24}
        }
        if contract:
            infos.update(
                {
                    "microban_teleop_training_contract_version": (
                        MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION
                    ),
                    "previous_action_semantics": (
                        MICROBAN_TELEOP_PREVIOUS_ACTION_SEMANTICS
                    ),
                }
            )
        path = self.root / f"model_{filename_iteration}.pt"
        torch.save({"iter": iteration, "infos": infos}, path)
        return path

    @staticmethod
    def _bounded_actor_state() -> dict[str, torch.Tensor]:
        lower = torch.tensor([-1.25, -1.0e-4], dtype=torch.float32)
        upper = torch.tensor([0.75, 0.4], dtype=torch.float32)
        zero = torch.zeros_like(lower)
        return {
            "distribution.lower_bound": lower,
            "distribution.upper_bound": upper,
            "distribution.inward_lower_bound": torch.nextafter(lower, zero),
            "distribution.inward_upper_bound": torch.nextafter(upper, zero),
        }

    @staticmethod
    def _v5_infos(*, pristine: bool = False) -> dict[str, object]:
        infos: dict[str, object] = {
            "env_state": {"common_step_counter": 0 if pristine else 24},
            "microban_teleop_training_contract_version": (
                MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION
            ),
            "previous_action_semantics": MICROBAN_TELEOP_PREVIOUS_ACTION_SEMANTICS,
        }
        if pristine:
            infos["pristine_pre_update"] = True
        return infos

    def test_v5_marker_is_required_and_iteration_must_match_filename(self) -> None:
        valid = self._save(12, contract=True)
        parsed = validate_teleop_checkpoint_contract(valid)
        self.assertEqual(parsed.version, "5")
        self.assertFalse(parsed.diagnostic_legacy)
        self.assertEqual(parsed.common_step_counter, 13 * 24)

        legacy = self._save(13, contract=False)
        with self.assertRaisesRegex(ValueError, "clean retrain"):
            validate_teleop_checkpoint_contract(legacy)
        diagnostic = validate_teleop_checkpoint_contract(
            legacy, allow_legacy_diagnostic=True
        )
        self.assertTrue(diagnostic.diagnostic_legacy)
        self.assertEqual(
            diagnostic.previous_action_semantics,
            "raw_policy_output_before_target_clip",
        )

        mismatched = self._save(14, filename_iteration=15, contract=True)
        with self.assertRaisesRegex(ValueError, "must match"):
            validate_teleop_checkpoint_contract(mismatched)

        # A well-formed v4 checkpoint is still a non-resumable training contract,
        # even though tensor shapes and previous-action semantics happen to match.
        v4 = self._save(16, contract=True)
        payload = torch.load(v4, map_location="cpu", weights_only=False)
        payload["infos"]["microban_teleop_training_contract_version"] = "4"
        torch.save(payload, v4)
        with self.assertRaisesRegex(ValueError, "v1/v2/v3/v4"):
            validate_teleop_checkpoint_contract(v4)

    def test_pristine_checkpoint_requires_explicit_pre_update_contract(self) -> None:
        path = self.root / "model_pristine.pt"
        infos = {
            "env_state": {"common_step_counter": 0},
            "microban_teleop_training_contract_version": (
                MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION
            ),
            "previous_action_semantics": MICROBAN_TELEOP_PREVIOUS_ACTION_SEMANTICS,
            "pristine_pre_update": True,
        }
        torch.save({"iter": -1, "infos": infos}, path)
        contract = validate_teleop_checkpoint_contract(path)
        self.assertEqual(contract.iteration, -1)
        self.assertTrue(contract.pristine_pre_update)

        infos["pristine_pre_update"] = False
        torch.save({"iter": -1, "infos": infos}, path)
        with self.assertRaisesRegex(ValueError, "pristine_pre_update=true"):
            validate_teleop_checkpoint_contract(path)

    def test_bounded_actor_checkpoint_buffers_are_exactly_pinned(self) -> None:
        path = self.root / "bounds.pt"
        expected = self._bounded_actor_state()

        def save(candidate: dict[str, torch.Tensor]) -> None:
            torch.save({"actor_state_dict": candidate}, path)

        save({key: value.clone() for key, value in expected.items()})
        validate_bounded_actor_checkpoint_buffers(path, expected)

        invalid_cases = (
            (
                "missing",
                {
                    key: value.clone()
                    for key, value in expected.items()
                    if key != "distribution.lower_bound"
                },
                TypeError,
                "missing Tensor",
            ),
            (
                "shape",
                {
                    **{key: value.clone() for key, value in expected.items()},
                    "distribution.lower_bound": expected["distribution.lower_bound"][
                        :1
                    ],
                },
                ValueError,
                "shape differs",
            ),
            (
                "dtype",
                {
                    **{key: value.clone() for key, value in expected.items()},
                    "distribution.upper_bound": expected[
                        "distribution.upper_bound"
                    ].double(),
                },
                ValueError,
                "dtype differs",
            ),
            (
                "non-finite",
                {
                    **{key: value.clone() for key, value in expected.items()},
                    "distribution.inward_upper_bound": torch.tensor(
                        [float("nan"), 0.1], dtype=torch.float32
                    ),
                },
                ValueError,
                "non-finite",
            ),
            (
                "different",
                {
                    **{key: value.clone() for key, value in expected.items()},
                    "distribution.inward_lower_bound": expected[
                        "distribution.inward_lower_bound"
                    ]
                    + torch.tensor([0.0, 1.0e-7]),
                },
                ValueError,
                "differs from the current",
            ),
        )
        for name, candidate, error_type, message in invalid_cases:
            with self.subTest(name=name):
                save(candidate)
                with self.assertRaisesRegex(error_type, message):
                    validate_bounded_actor_checkpoint_buffers(path, expected)

        nonfinite_expected = {
            **expected,
            "distribution.lower_bound": torch.tensor(
                [float("inf"), -1.0e-4], dtype=torch.float32
            ),
        }
        save({key: value.clone() for key, value in nonfinite_expected.items()})
        with self.assertRaisesRegex(ValueError, "Current bounded actor.*non-finite"):
            validate_bounded_actor_checkpoint_buffers(path, nonfinite_expected)

    def test_v5_load_pins_bounds_before_base_load_including_pristine(self) -> None:
        expected = self._bounded_actor_state()
        policy = SimpleNamespace(state_dict=lambda: expected)

        for pristine in (False, True):
            with self.subTest(pristine=pristine):
                path = self.root / ("model_pristine.pt" if pristine else "model_0.pt")
                torch.save(
                    {
                        "actor_state_dict": {
                            key: value.clone() for key, value in expected.items()
                        },
                        "iter": -1 if pristine else 0,
                        "infos": self._v5_infos(pristine=pristine),
                    },
                    path,
                )
                runner = object.__new__(MicrobanTeleopOnPolicyRunner)
                runner.alg = SimpleNamespace(get_policy=lambda: policy)
                with patch.object(
                    MjlabOnPolicyRunner, "load", return_value={}
                ) as base_load:
                    runner.load(str(path), load_cfg={"actor": True})
                base_load.assert_called_once()
                self.assertEqual(
                    runner.loaded_checkpoint_contract.pristine_pre_update,
                    pristine,
                )

        mismatched_path = self.root / "model_1.pt"
        mismatched = {key: value.clone() for key, value in expected.items()}
        mismatched["distribution.upper_bound"][0] += 1.0e-4
        torch.save(
            {
                "actor_state_dict": mismatched,
                "iter": 1,
                "infos": {
                    **self._v5_infos(),
                    "env_state": {"common_step_counter": 48},
                },
            },
            mismatched_path,
        )
        runner = object.__new__(MicrobanTeleopOnPolicyRunner)
        runner.alg = SimpleNamespace(get_policy=lambda: policy)
        with (
            patch.object(MjlabOnPolicyRunner, "load") as base_load,
            self.assertRaisesRegex(ValueError, "guarded action-bound contract"),
        ):
            runner.load(str(mismatched_path), load_cfg={"actor": True})
        base_load.assert_not_called()

    def test_save_adds_v5_marker_and_legacy_runner_cannot_write(self) -> None:
        fresh = object.__new__(MicrobanTeleopOnPolicyRunner)
        fresh.loaded_checkpoint_contract = None
        fresh.velocity_actor_bootstrap_info = {
            "mapping_version": (
                "xc330_velocity_63_to_teleop_83_v3_bounded_actor_"
                "preserve_normalizer_count"
            ),
            "source_checkpoint_sha256": "a" * 64,
        }
        with patch.object(MjlabOnPolicyRunner, "save") as base_save:
            fresh.save(
                str(self.root / "model_0.pt"),
                {
                    "source": "test",
                    "velocity_actor_bootstrap": {"forged": True},
                },
            )
        saved_infos = base_save.call_args.args[-1]
        self.assertEqual(saved_infos["microban_teleop_training_contract_version"], "5")
        self.assertEqual(
            saved_infos["previous_action_semantics"],
            MICROBAN_TELEOP_PREVIOUS_ACTION_SEMANTICS,
        )
        self.assertEqual(
            saved_infos["velocity_actor_bootstrap"],
            fresh.velocity_actor_bootstrap_info,
        )

        legacy = object.__new__(MicrobanTeleopOnPolicyRunner)
        legacy.loaded_checkpoint_contract = TeleopCheckpointContract(
            version="legacy_unversioned_v1",
            previous_action_semantics="raw_policy_output_before_target_clip",
            iteration=14999,
            common_step_counter=360000,
            diagnostic_legacy=True,
        )
        with patch.object(MjlabOnPolicyRunner, "save") as base_save:
            with self.assertRaisesRegex(ValueError, "diagnostics-only"):
                legacy.save(str(self.root / "model_14999.pt"))
            base_save.assert_not_called()
        with patch.object(MjlabOnPolicyRunner, "export_policy_to_onnx") as export:
            with self.assertRaisesRegex(ValueError, "diagnostics-only"):
                legacy.export_policy_to_onnx(str(self.root))
            export.assert_not_called()

    def test_legacy_load_is_actor_only_and_never_optimizer_resume(self) -> None:
        checkpoint = self._save(14999, contract=False)
        runner = object.__new__(MicrobanTeleopOnPolicyRunner)
        with (
            patch.object(MjlabOnPolicyRunner, "load") as base_load,
            self.assertRaisesRegex(ValueError, "cannot be resumed/exported"),
        ):
            runner.load(str(checkpoint), load_cfg={"actor": True})
        base_load.assert_not_called()

        with patch.object(MjlabOnPolicyRunner, "load", return_value={}) as base_load:
            runner.load(
                str(checkpoint),
                load_cfg={"actor": True},
                allow_legacy_teleop_contract=True,
            )
            base_load.assert_called_once()
        self.assertTrue(runner.loaded_checkpoint_contract.diagnostic_legacy)

        for load_cfg in (
            None,
            {"actor": True, "iteration": True},
            {"actor": True, "optimizer": True},
            {"critic": True},
        ):
            with (
                self.subTest(load_cfg=load_cfg),
                patch.object(MjlabOnPolicyRunner, "load") as base_load,
                self.assertRaisesRegex(ValueError, "cannot resume training"),
            ):
                runner.load(
                    str(checkpoint),
                    load_cfg=load_cfg,
                    allow_legacy_teleop_contract=True,
                )
            base_load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
