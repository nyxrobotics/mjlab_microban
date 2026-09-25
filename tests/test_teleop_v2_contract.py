# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Focused tests for the Microban teleop training/deployment contract."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from mjlab.envs.mdp.actions import JointPositionAction
from mjlab.rl.runner import MjlabOnPolicyRunner
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from mjlab_microban.tasks.mdp import no_stepping_penalty
from mjlab_microban.tasks.microban_locomotion_prior import (
    MICROBAN_LOCOMOTION_PRIOR_COMMAND_WIDTH,
    LocomotionPriorCommand,
    LocomotionPriorCommandCfg,
)
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_TELEOP_ACTION_JOINT_NAMES,
    MICROBAN_TELEOP_ACTOR_INITIALIZATION,
    MICROBAN_TELEOP_ACTOR_LATENT_ABS_MAX,
    MICROBAN_TELEOP_ACTOR_LATENT_MEAN_FRACTION,
    MICROBAN_TELEOP_ACTOR_LATENT_SCALE_MULTIPLIER,
    MICROBAN_TELEOP_ACTOR_STD_ABS_MAX,
    MICROBAN_TELEOP_ACTOR_STD_ENVELOPE_DIVISOR,
    MICROBAN_TELEOP_ACTOR_STD_MIN_ABS_MAX,
    MICROBAN_TELEOP_ACTOR_STD_MIN_ENVELOPE_DIVISOR,
    MICROBAN_TELEOP_FINAL_BOTH_FEET_LIFT_UPPER_M,
    MICROBAN_TELEOP_NUM_STEPS_PER_ENV,
    MICROBAN_TELEOP_PREVIOUS_ACTION_SEMANTICS,
    MICROBAN_TELEOP_RECIPE_REVISION,
    MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION,
    MicrobanTeleopOnPolicyRunner,
    TeleopCheckpointContract,
    _command_target_bounds,
    validate_bounded_actor_checkpoint_buffers,
    validate_finite_actor_state,
    validate_finite_checkpoint_actor_state,
    validate_safe_velocity_actor_bootstrap_info,
    validate_teleop_checkpoint_contract,
)
from mjlab_microban.tasks.microban_safe_velocity_checkpoint import (
    inspect_safe_velocity_checkpoint,
)
from mjlab_microban.tasks.microban_safe_velocity_env_cfg import (
    MICROBAN_SAFE_VELOCITY_SAGITTAL_LEG_JOINT_NAMES,
    microban_safe_velocity_action_delta_bounds,
    microban_safe_velocity_initial_action_std,
)
from mjlab_microban.tasks.microban_safe_velocity_mdp import (
    MICROBAN_SAFE_VELOCITY_RECIPE_INFO_KEY,
    MICROBAN_SAFE_VELOCITY_RECIPE_REVISION,
    MicrobanSafeVelocityBoundedGaussianDistribution,
)
from mjlab_microban.tasks.microban_teleop_bootstrap import (
    SAFE_VELOCITY_ACTOR_BOOTSTRAP_MAPPING_VERSION,
    SAFE_VELOCITY_COPIED_DISTRIBUTION_KEYS,
    SAFE_VELOCITY_NEW_TELEOP_OBSERVATION_COLUMNS,
    SAFE_VELOCITY_TARGET_ACTOR_TOPOLOGY,
    TELEOP_SHOULDER_ROLL_ACTION_INDICES,
    TELEOP_SHOULDER_ROLL_INITIAL_LATENT_BIASES,
    SafeVelocityActorBootstrapProvenance,
    serialize_safe_velocity_actor_bootstrap_provenance,
    validate_safe_velocity_acceptance_receipt,
)
from mjlab_microban.tasks.microban_teleop_env_cfg import (
    MICROBAN_TELEOP_FINAL_VELOCITY_ENVELOPE,
    MICROBAN_TELEOP_FOOT_TRACKING_FINAL_STD_M,
    MICROBAN_TELEOP_HAND_TRACKING_FINAL_STD_M,
    MICROBAN_TELEOP_HAND_TRACKING_STD_M,
    MICROBAN_TELEOP_INITIAL_ANGULAR_TRACKING_STD_RAD_S,
    MICROBAN_TELEOP_INITIAL_HMD_NEUTRAL_PROBABILITY,
    MICROBAN_TELEOP_INITIAL_LINEAR_TRACKING_STD_M_S,
    MICROBAN_TELEOP_INITIAL_SIGNED_AXIS_RANGES,
    MICROBAN_TELEOP_INITIAL_VELOCITY_ENVELOPE,
    MICROBAN_TELEOP_ISOLATED_AXIS_PROBABILITIES,
    MICROBAN_TELEOP_JOINT_LIMIT_GUARD_LOOKAHEAD_S,
    MICROBAN_TELEOP_JOINT_LIMIT_GUARD_MARGIN_RATIO,
    MICROBAN_TELEOP_MIXED_AXIS_PROBABILITIES,
    MICROBAN_TELEOP_MOVING_HMD_NEUTRAL_PROBABILITY,
    MICROBAN_TELEOP_NEUTRAL_FOOT_TRACKING_WEIGHT,
    MICROBAN_TELEOP_PRIOR_ACTION_REWARD_WEIGHT,
    MICROBAN_TELEOP_PRIOR_INITIAL_AXIS_PROBABILITIES,
    MICROBAN_TELEOP_PRIOR_JOINT_REWARD_WEIGHT,
    MICROBAN_TELEOP_PRIOR_REWARD_STD_RAD,
    MICROBAN_TELEOP_PRIOR_SIGNED_AXIS_RANGES,
    MicrobanTeleopRlCfg,
    make_microban_teleop_env_cfg,
    microban_teleop_action_delta_bounds,
    microban_teleop_initial_action_std,
)
from mjlab_microban.tasks.microban_teleop_mdp import (
    MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M,
    AsymmetricBoundedGaussianDistribution,
    HmdNeckTargetMotion,
    LatentActionPPO,
    PerJointGaussianDistribution,
    ResetFixedFootTargetCommand,
    effective_action_after_target_clip,
    linear_velocity_tracking_error_l1,
    normalized_joint_soft_limit_guard_l1_sum,
    normalized_target_clip_excess_l1_sum,
    normalized_target_near_limit_l1_sum,
    raw_action_l2,
    yaw_velocity_tracking_error_l1,
)
from mjlab_microban.tasks.microban_teleop_provenance import (
    MICROBAN_TELEOP_TRAINING_PROVENANCE_KEY,
    MICROBAN_TELEOP_TRAINING_PROVENANCE_SCHEMA_VERSION,
    MICROBAN_TELEOP_TRAINING_PROVENANCE_SHA256_KEY,
    canonical_json_sha256,
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

    def test_joint_soft_limit_guard_uses_current_and_projected_position(self) -> None:
        action = object.__new__(JointPositionAction)
        action._target_ids = torch.tensor([0, 2])
        action._offset = torch.tensor([[0.5, 0.5]])
        action._entity = SimpleNamespace(
            data=SimpleNamespace(
                joint_pos=torch.tensor([[0.11, 99.0, 0.50], [0.50, 99.0, 0.89]]),
                joint_vel=torch.tensor([[-0.20, 0.0, 0.00], [0.00, 0.0, 0.20]]),
                soft_joint_pos_limits=torch.tensor(
                    [[[0.0, 1.0], [-100.0, 100.0], [0.0, 1.0]]]
                ).expand(2, -1, -1),
            )
        )
        env = SimpleNamespace(
            action_manager=SimpleNamespace(get_term=lambda name: action)
        )

        penalty = normalized_joint_soft_limit_guard_l1_sum(
            env, margin_ratio=0.1, lookahead_s=0.1
        )

        # First row projects from 0.11 to 0.09, 0.01 inside the lower guard;
        # second row projects from 0.89 to 0.91, 0.01 past the upper guard.
        torch.testing.assert_close(penalty, torch.tensor([0.02, 0.02]))

    def test_joint_soft_limit_guard_does_not_move_near_limit_default(self) -> None:
        action = object.__new__(JointPositionAction)
        action._target_ids = torch.tensor([0])
        action._offset = torch.tensor([[0.05]])
        action._entity = SimpleNamespace(
            data=SimpleNamespace(
                joint_pos=torch.tensor([[0.05], [0.04]]),
                joint_vel=torch.zeros(2, 1),
                soft_joint_pos_limits=torch.tensor([[[0.0, 1.0]]]).expand(2, -1, -1),
            )
        )
        env = SimpleNamespace(
            action_manager=SimpleNamespace(get_term=lambda name: action)
        )

        penalty = normalized_joint_soft_limit_guard_l1_sum(
            env, margin_ratio=0.1, lookahead_s=0.1
        )

        torch.testing.assert_close(penalty, torch.tensor([0.0, 0.02]))

    def test_joint_soft_limit_guard_rejects_invalid_parameters(self) -> None:
        env = SimpleNamespace(action_manager=SimpleNamespace())
        for margin in (0.0, 0.5, float("nan")):
            with self.assertRaisesRegex(ValueError, "margin_ratio"):
                normalized_joint_soft_limit_guard_l1_sum(env, margin_ratio=margin)
        for lookahead in (-0.1, float("inf")):
            with self.assertRaisesRegex(ValueError, "lookahead_s"):
                normalized_joint_soft_limit_guard_l1_sum(env, lookahead_s=lookahead)

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
    STD = (0.1, 0.1, 0.05)

    def _distribution(self) -> AsymmetricBoundedGaussianDistribution:
        return AsymmetricBoundedGaussianDistribution(
            3,
            self.STD,
            self.LOWER,
            self.UPPER,
            std_type="log",
        )

    def test_zero_anchor_and_guarded_mean_are_strictly_inside_bounds(self) -> None:
        distribution = self._distribution()
        mlp_output = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0e30, -1.0e30, 1.0e30]],
            dtype=torch.float32,
        )
        action = distribution.deterministic_output(mlp_output)
        torch.testing.assert_close(action[0], torch.zeros(3))
        lower = torch.tensor(self.LOWER)
        upper = torch.tensor(self.UPPER)
        self.assertTrue(torch.all(action > lower).item())
        self.assertTrue(torch.all(action < upper).item())
        self.assertTrue(torch.isfinite(action).all().item())

        with self.assertRaisesRegex(FloatingPointError, "MLP output"):
            distribution.update(torch.tensor([[float("inf"), 0.0, 0.0]]))

    def test_sample_storage_uses_exact_latent_log_prob(self) -> None:
        torch.manual_seed(1234)
        mean = torch.tensor([[0.02, -0.03, 0.04], [-0.1, 0.2, -0.05]])
        distribution = self._distribution()
        distribution.update(mean)
        latent = distribution.sample().detach()
        rollout_log_prob = distribution.log_prob(latent).detach()
        environment_action = distribution.to_environment_action(latent)

        # A PPO minibatch reconstructs the distribution later and evaluates the
        # exact stored latent.  Only the tensor returned to the environment is
        # transformed, and it is a distinct bounded tensor.
        distribution.update(mean.clone())
        replay_log_prob = distribution.log_prob(latent)
        torch.testing.assert_close(replay_log_prob, rollout_log_prob)
        torch.testing.assert_close(
            torch.exp(replay_log_prob - rollout_log_prob), torch.ones(2)
        )
        latent_dist = torch.distributions.Normal(distribution.mean, distribution.std)
        expected = latent_dist.log_prob(latent).sum(dim=-1)
        torch.testing.assert_close(replay_log_prob, expected)
        self.assertFalse(torch.equal(environment_action, latent))
        self.assertTrue(torch.all(environment_action > torch.tensor(self.LOWER)))
        self.assertTrue(torch.all(environment_action < torch.tensor(self.UPPER)))

    def test_operational_envelope_round_trip_budget_covers_all_joints(self) -> None:
        lower, upper = microban_teleop_action_delta_bounds()
        initial_std = microban_teleop_initial_action_std()
        shoulder = MICROBAN_TELEOP_ACTION_JOINT_NAMES.index("right_shoulder_roll")
        self.assertLess(min(-lower[shoulder], upper[shoulder]), 1.01e-4)

        for index, name in enumerate(MICROBAN_TELEOP_ACTION_JOINT_NAMES):
            with self.subTest(joint=name):
                distribution = AsymmetricBoundedGaussianDistribution(
                    1,
                    [initial_std[index]],
                    [lower[index]],
                    [upper[index]],
                )
                max_std = float(distribution.max_std.item())
                reachable_lower = max(
                    float(distribution.operational_lower_bound.item()),
                    float(distribution.mean_lower_bound.item()) - 10.0 * max_std,
                )
                reachable_upper = min(
                    float(distribution.operational_upper_bound.item()),
                    float(distribution.mean_upper_bound.item()) + 10.0 * max_std,
                )
                latent = torch.linspace(
                    reachable_lower,
                    reachable_upper,
                    100_001,
                    dtype=torch.float32,
                ).unsqueeze(-1)
                action = distribution.to_environment_action(latent)
                round_trip = distribution._inverse(action)
                max_error = torch.max(torch.abs(round_trip - latent))
                self.assertLessEqual(
                    float(max_error), 0.025 * float(distribution.min_std.item())
                )
                self.assertFalse(
                    bool(
                        (
                            (action == distribution.inward_lower_bound)
                            | (action == distribution.inward_upper_bound)
                        ).any()
                    )
                )

    def test_outside_latent_and_non_bijective_action_endpoints_fail(self) -> None:
        distribution = self._distribution()
        distribution.update(torch.zeros((1, 3)))
        for invalid_latent in (
            distribution.operational_upper_bound.unsqueeze(0) + 1.0,
            distribution.operational_lower_bound.unsqueeze(0) - 1.0,
            torch.tensor([[float("nan"), 0.0, 0.0]]),
        ):
            with (
                self.subTest(invalid=invalid_latent),
                self.assertRaisesRegex(FloatingPointError, "operational envelope"),
            ):
                distribution.log_prob(invalid_latent)

        for endpoint in (
            distribution.inward_lower_bound.unsqueeze(0),
            distribution.inward_upper_bound.unsqueeze(0),
        ):
            with self.assertRaisesRegex(ValueError, "operational action envelope"):
                distribution._inverse(endpoint)

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
        entropy.sum().backward(retain_graph=True)
        self.assertIsNotNone(distribution.log_std_param.grad)
        self.assertTrue(torch.isfinite(distribution.log_std_param.grad).all().item())

        deterministic = distribution.deterministic_output(mean)
        deterministic.sum().backward()
        self.assertIsNotNone(mean.grad)
        self.assertTrue(torch.isfinite(mean.grad).all().item())
        export_output = distribution.as_deterministic_output_module()(mean.detach())
        torch.testing.assert_close(export_output, deterministic.detach())
        with self.assertRaisesRegex(FloatingPointError, "Exportable.*non-finite"):
            distribution.as_deterministic_output_module()(
                torch.tensor([[float("inf"), 0.0, 0.0]])
            )

    def test_nonfinite_std_parameter_is_rejected_before_clamp(self) -> None:
        for std_type, parameter_name in (
            ("scalar", "std_param"),
            ("log", "log_std_param"),
        ):
            with self.subTest(std_type=std_type):
                distribution = AsymmetricBoundedGaussianDistribution(
                    3,
                    self.STD,
                    self.LOWER,
                    self.UPPER,
                    std_type=std_type,
                )
                parameter = getattr(distribution, parameter_name)
                with torch.no_grad():
                    parameter[0] = float("inf")
                with self.assertRaisesRegex(
                    FloatingPointError, "parameter.*before clamp"
                ):
                    distribution.update(torch.zeros((1, 3)))

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
        stored_latent = actor(obs, stochastic_output=True).detach()
        old_log_prob = actor.get_output_log_prob(stored_latent).detach()
        distribution = actor.distribution
        assert isinstance(distribution, AsymmetricBoundedGaussianDistribution)
        environment_action = distribution.to_environment_action(stored_latent)
        old_params = tuple(
            value.detach().clone() for value in actor.output_distribution_params
        )

        # This is the PPO update path: a fresh stochastic forward updates the
        # distribution, then log_prob is evaluated for the stored rollout action.
        actor(obs, stochastic_output=True)
        replay_log_prob = actor.get_output_log_prob(stored_latent)
        new_params = actor.output_distribution_params
        ratio = torch.exp(replay_log_prob - old_log_prob)
        torch.testing.assert_close(ratio, torch.ones_like(ratio))
        torch.testing.assert_close(
            actor.get_kl_divergence(old_params, new_params), torch.zeros(8)
        )
        self.assertFalse(torch.equal(environment_action, stored_latent))
        torch.testing.assert_close(
            environment_action, distribution.to_environment_action(stored_latent)
        )

    def test_real_mlp_and_latent_ppo_store_latent_but_step_bounded_action(
        self,
    ) -> None:
        """Exercise the real RSL-RL constructor and rollout transition path."""

        torch.manual_seed(29)
        num_envs = 4
        obs = TensorDict({"policy": torch.randn(num_envs, 83)}, batch_size=[num_envs])
        obs_groups = {"actor": ["policy"], "critic": ["policy"]}
        lower, upper = microban_teleop_action_delta_bounds()
        actor = MLPModel(
            obs=obs,
            obs_groups=obs_groups,
            obs_set="actor",
            output_dim=18,
            hidden_dims=(16,),
            distribution_cfg={
                "class_name": AsymmetricBoundedGaussianDistribution,
                "init_std": microban_teleop_initial_action_std(),
                "lower_bound": lower,
                "upper_bound": upper,
                "std_type": "log",
            },
        )
        final_layer = actor.mlp[-1]
        assert isinstance(final_layer, torch.nn.Linear)
        shoulder_indices = list(TELEOP_SHOULDER_ROLL_ACTION_INDICES)
        unguarded_indices = sorted(set(range(18)) - set(shoulder_indices))
        torch.testing.assert_close(
            final_layer.weight[shoulder_indices],
            torch.zeros_like(final_layer.weight[shoulder_indices]),
        )
        torch.testing.assert_close(
            final_layer.bias[shoulder_indices],
            final_layer.bias.new_tensor(TELEOP_SHOULDER_ROLL_INITIAL_LATENT_BIASES),
        )
        self.assertGreater(
            float(final_layer.weight[unguarded_indices].abs().sum().item()), 0.0
        )
        critic = MLPModel(
            obs=obs,
            obs_groups=obs_groups,
            obs_set="critic",
            output_dim=1,
            hidden_dims=(16,),
        )
        storage = RolloutStorage("rl", num_envs, 2, obs, [18], "cpu")
        algorithm = LatentActionPPO(
            actor,
            critic,
            storage,
            device="cpu",
            rnd_cfg=None,
            symmetry_cfg=None,
        )

        environment_action = algorithm.act(obs)
        stored_latent = algorithm.transition.actions
        assert stored_latent is not None
        old_log_prob = algorithm.transition.actions_log_prob
        assert old_log_prob is not None
        self.assertEqual(environment_action.shape, (num_envs, 18))
        self.assertEqual(stored_latent.shape, (num_envs, 18))
        self.assertFalse(torch.equal(environment_action, stored_latent))
        distribution = actor.distribution
        assert isinstance(distribution, AsymmetricBoundedGaussianDistribution)
        torch.testing.assert_close(
            environment_action, distribution.to_environment_action(stored_latent)
        )

        # The first PPO replay before an optimizer step must have ratio exactly
        # one in the same latent sample space stored by RolloutStorage.
        actor(obs, stochastic_output=True)
        replay_log_prob = actor.get_output_log_prob(stored_latent)
        ratio = torch.exp(replay_log_prob - old_log_prob)
        torch.testing.assert_close(ratio, torch.ones_like(ratio))

    def test_latent_ppo_returns_bounded_copy_without_overwriting_transition(
        self,
    ) -> None:
        distribution = self._distribution()
        latent = torch.tensor([[0.2, -0.1, 0.05]])
        algorithm = object.__new__(LatentActionPPO)
        algorithm.actor = SimpleNamespace(distribution=distribution)
        algorithm.transition = SimpleNamespace(actions=latent.clone())
        obs = TensorDict({"policy": torch.zeros(1, 1)}, batch_size=[1])

        with patch.object(PPO, "act", return_value=algorithm.transition.actions):
            environment_action = algorithm.act(obs)

        torch.testing.assert_close(algorithm.transition.actions, latent)
        torch.testing.assert_close(
            environment_action, distribution.to_environment_action(latent)
        )
        self.assertNotEqual(
            environment_action.untyped_storage().data_ptr(),
            algorithm.transition.actions.untyped_storage().data_ptr(),
        )

    def test_latent_ppo_construct_fails_closed_on_wrapper_and_extensions(self) -> None:
        obs = TensorDict({"policy": torch.zeros(1, 1)}, batch_size=[1])
        valid_env = SimpleNamespace(clip_actions=None, num_actions=18)
        with self.assertRaisesRegex(ValueError, "BC fields are missing"):
            LatentActionPPO.construct_algorithm(
                obs, valid_env, {"algorithm": {}}, "cpu"
            )

        invalid_cases = (
            (SimpleNamespace(clip_actions=1.0, num_actions=18), {}, "clip_actions"),
            (SimpleNamespace(clip_actions=None, num_actions=17), {}, "18-wide"),
            (valid_env, {"rnd_cfg": {}}, "RND"),
            (valid_env, {"symmetry_cfg": {}}, "symmetry"),
        )
        for env, algorithm_cfg, message in invalid_cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                LatentActionPPO.construct_algorithm(
                    obs, env, {"algorithm": algorithm_cfg}, "cpu"
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
        # Export constructs this exact play configuration. Its curriculum is
        # disabled, so the ordinary metadata path must retain the early/play
        # 12 mm simultaneous-foot ceiling.
        play_cfg = make_microban_teleop_env_cfg(play=True)
        foot_cfg = play_cfg.commands["foot_target"]
        manager = SimpleNamespace(get_term_cfg=lambda name: play_cfg.commands[name])
        diagnostic_bounds = _command_target_bounds(
            SimpleNamespace(command_manager=manager)
        )
        self.assertEqual(diagnostic_bounds[0], [-0.03, -0.03, 0.0] * 2)
        self.assertEqual(diagnostic_bounds[1], [0.03, 0.03, 0.05] * 2)
        self.assertEqual(diagnostic_bounds[2], [-0.01, -0.01, 0.0] * 2)
        self.assertEqual(diagnostic_bounds[3], [0.01, 0.01, 0.012] * 2)

        final_bounds = _command_target_bounds(
            SimpleNamespace(command_manager=manager),
            canonical_final_stage=True,
        )
        self.assertEqual(
            final_bounds[3],
            [0.01, 0.01, MICROBAN_TELEOP_FINAL_BOTH_FEET_LIFT_UPPER_M] * 2,
        )
        # Rendering final deployment metadata must not mutate the play config
        # subsequently used by diagnostic/non-accepted exports.
        self.assertEqual(
            foot_cfg.both_feet_lift_height_range,
            (MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M, 0.012),
        )

    def test_ordered_std_vector_and_reward_contract(self) -> None:
        std = microban_teleop_initial_action_std()
        self.assertEqual(len(std), len(MICROBAN_TELEOP_ACTION_JOINT_NAMES))
        safe_std = microban_safe_velocity_initial_action_std()
        differing = {
            name
            for name, teleop_value, safe_value in zip(
                MICROBAN_TELEOP_ACTION_JOINT_NAMES, std, safe_std, strict=True
            )
            if teleop_value != safe_value
        }
        self.assertEqual(
            differing,
            set(MICROBAN_SAFE_VELOCITY_SAGITTAL_LEG_JOINT_NAMES),
        )
        self.assertTrue(all(0.0 < value <= 1.0 for value in std))
        for name, value in zip(MICROBAN_TELEOP_ACTION_JOINT_NAMES, std, strict=True):
            if "shoulder_roll" in name:
                self.assertLess(value, 0.01)
            elif "shoulder" in name or "elbow" in name:
                self.assertEqual(value, 0.05)
            else:
                self.assertEqual(value, 0.08)

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
        self.assertEqual(cfg.rewards["joint_soft_limit_guard"].weight, -5.0)
        self.assertIs(
            cfg.rewards["joint_soft_limit_guard"].func,
            normalized_joint_soft_limit_guard_l1_sum,
        )
        self.assertEqual(
            cfg.rewards["joint_soft_limit_guard"].params,
            {
                "action_name": "joint_pos",
                "margin_ratio": MICROBAN_TELEOP_JOINT_LIMIT_GUARD_MARGIN_RATIO,
                "lookahead_s": MICROBAN_TELEOP_JOINT_LIMIT_GUARD_LOOKAHEAD_S,
            },
        )
        self.assertEqual(cfg.rewards["raw_action_l2"].weight, -0.01)
        self.assertEqual(cfg.rewards["action_rate_l2"].weight, -0.02)
        self.assertEqual(cfg.rewards["linear_velocity_error_l1"].weight, -4.0)
        self.assertEqual(cfg.rewards["yaw_velocity_error_l1"].weight, -1.0)
        self.assertEqual(
            cfg.rewards["locomotion_prior_action_target"].weight,
            MICROBAN_TELEOP_PRIOR_ACTION_REWARD_WEIGHT,
        )
        self.assertEqual(
            cfg.rewards["locomotion_prior_joint_position"].weight,
            MICROBAN_TELEOP_PRIOR_JOINT_REWARD_WEIGHT,
        )
        self.assertEqual(
            cfg.rewards["locomotion_prior_action_target"].params["std"],
            MICROBAN_TELEOP_PRIOR_REWARD_STD_RAD,
        )
        self.assertNotIn("locomotion_prior", cfg.observations["actor"].terms)
        self.assertIn("locomotion_prior", cfg.observations["critic"].terms)
        self.assertIsInstance(
            cfg.commands["locomotion_prior"], LocomotionPriorCommandCfg
        )
        self.assertFalse(cfg.commands["locomotion_prior"].enabled)
        self.assertEqual(cfg.rewards["locomotion_prior_action_target"].weight, 0.0)
        self.assertEqual(cfg.rewards["locomotion_prior_joint_position"].weight, 0.0)
        self.assertNotIn("locomotion_prior_clip_finished", cfg.terminations)
        self.assertEqual(MICROBAN_LOCOMOTION_PRIOR_COMMAND_WIDTH, 39)
        self.assertEqual(cfg.rewards["dof_pos_limits"].weight, -10.0)
        self.assertEqual(cfg.rewards["feet_distance"].weight, -100.0)
        self.assertEqual(cfg.rewards["feet_distance"].params["min_dist"], 0.07)
        self.assertEqual(
            cfg.rewards["no_stepping"].params["foot_target_command_name"],
            "foot_target",
        )
        self.assertEqual(
            cfg.rewards["track_linear_velocity"].params["std"],
            MICROBAN_TELEOP_INITIAL_LINEAR_TRACKING_STD_M_S,
        )
        self.assertEqual(
            cfg.rewards["track_angular_velocity"].params["std"],
            MICROBAN_TELEOP_INITIAL_ANGULAR_TRACKING_STD_RAD_S,
        )
        self.assertEqual(
            cfg.rewards["hand_target_tracking"].params["std"],
            MICROBAN_TELEOP_HAND_TRACKING_STD_M,
        )
        self.assertEqual(MicrobanTeleopRlCfg.algorithm.entropy_coef, 0.0)
        self.assertEqual(MicrobanTeleopRlCfg.algorithm.learning_rate, 1.0e-4)
        self.assertEqual(MicrobanTeleopRlCfg.algorithm.num_learning_epochs, 3)
        self.assertEqual(MicrobanTeleopRlCfg.algorithm.schedule, "adaptive")
        self.assertEqual(
            MicrobanTeleopRlCfg.algorithm.locomotion_prior_bc_forward_coefficient,
            0.0,
        )
        self.assertEqual(
            MicrobanTeleopRlCfg.algorithm.locomotion_prior_bc_neutral_leg_coefficient,
            0.0,
        )
        self.assertEqual(
            MicrobanTeleopRlCfg.algorithm.locomotion_prior_bc_arm_home_coefficient,
            0.0,
        )
        self.assertEqual(
            MicrobanTeleopRlCfg.algorithm.locomotion_prior_bc_error_scale_rad,
            0.15,
        )
        self.assertEqual(MicrobanTeleopRlCfg.algorithm.locomotion_prior_bc_chunks, 4)
        self.assertEqual(
            MicrobanTeleopRlCfg.algorithm.locomotion_prior_bc_max_target_projection_rad,
            1.0e-3,
        )
        self.assertEqual(
            MicrobanTeleopRlCfg.algorithm.class_name,
            "mjlab_microban.tasks.microban_teleop_mdp:LatentActionPPO",
        )
        self.assertIs(
            MicrobanTeleopRlCfg.actor.distribution_cfg["class_name"],
            MicrobanSafeVelocityBoundedGaussianDistribution,
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
        self.assertEqual(
            MicrobanTeleopRlCfg.actor.distribution_cfg["latent_scale_multiplier"],
            MICROBAN_TELEOP_ACTOR_LATENT_SCALE_MULTIPLIER,
        )
        self.assertEqual(
            MicrobanTeleopRlCfg.actor.distribution_cfg["latent_abs_max"],
            MICROBAN_TELEOP_ACTOR_LATENT_ABS_MAX,
        )
        self.assertEqual(
            MicrobanTeleopRlCfg.actor.distribution_cfg["latent_mean_fraction"],
            MICROBAN_TELEOP_ACTOR_LATENT_MEAN_FRACTION,
        )
        self.assertEqual(
            MicrobanTeleopRlCfg.actor.distribution_cfg["std_min_abs_max"],
            MICROBAN_TELEOP_ACTOR_STD_MIN_ABS_MAX,
        )
        self.assertEqual(
            MicrobanTeleopRlCfg.actor.distribution_cfg["std_min_envelope_divisor"],
            MICROBAN_TELEOP_ACTOR_STD_MIN_ENVELOPE_DIVISOR,
        )
        self.assertEqual(
            MicrobanTeleopRlCfg.actor.distribution_cfg["std_abs_max"],
            MICROBAN_TELEOP_ACTOR_STD_ABS_MAX,
        )
        self.assertEqual(
            MicrobanTeleopRlCfg.actor.distribution_cfg["std_envelope_divisor"],
            MICROBAN_TELEOP_ACTOR_STD_ENVELOPE_DIVISOR,
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
                500 * 24,
                1500 * 24,
                3000 * 24,
                4500 * 24,
                6000 * 24,
                8000 * 24,
                12000 * 24,
                14000 * 24,
                16000 * 24,
                18000 * 24,
            ],
        )
        self.assertEqual(
            cfg.commands["twist"].signed_axis_ranges,
            MICROBAN_TELEOP_PRIOR_SIGNED_AXIS_RANGES,
        )
        self.assertEqual(
            cfg.commands["twist"].signed_axis_probabilities,
            MICROBAN_TELEOP_PRIOR_INITIAL_AXIS_PROBABILITIES,
        )
        self.assertEqual(cfg.commands["foot_target"].rel_both_feet_envs, 0.0)
        self.assertEqual(
            cfg.rewards["foot_target_tracking"].weight,
            MICROBAN_TELEOP_NEUTRAL_FOOT_TRACKING_WEIGHT,
        )
        self.assertEqual(
            cfg.rewards["foot_target_tracking"].params["velocity_fade_range"],
            (0.0, 0.01),
        )
        self.assertEqual(
            cfg.events["hmd_neck_target_motion"].params["neutral_probability"],
            MICROBAN_TELEOP_INITIAL_HMD_NEUTRAL_PROBABILITY,
        )
        self.assertEqual(
            stages[5]["name"],
            "enable moving-HMD and stationary no-step guard",
        )
        self.assertEqual(
            cfg.commands["foot_target"].lift_height_range,
            (MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M, 0.05),
        )
        self.assertEqual(
            cfg.commands["foot_target"].both_feet_lift_height_range,
            (MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M, 0.012),
        )

        twist = SimpleNamespace(cfg=deepcopy(cfg.commands["twist"]))
        foot = SimpleNamespace(cfg=cfg.commands["foot_target"])
        hand = SimpleNamespace(cfg=cfg.commands["hand_target"])
        prior_cfg = cfg.commands["locomotion_prior"]
        prior = object.__new__(LocomotionPriorCommand)
        prior.cfg = prior_cfg
        prior.eligible = torch.zeros(1, dtype=torch.bool)
        prior.finished = torch.zeros(1, dtype=torch.bool)
        prior.teleported = torch.zeros(1, dtype=torch.bool)
        prior.launching = torch.zeros(1, dtype=torch.bool)
        prior.launch_step = torch.zeros(1, dtype=torch.long)
        prior.launch_start_joint_pos = torch.zeros(1, 12)
        prior.phase_rate = torch.zeros(1)
        prior._skip_next_advance = torch.zeros(1, dtype=torch.bool)
        reward_cfgs = {
            name: SimpleNamespace(weight=term.weight, params=dict(term.params))
            for name, term in cfg.rewards.items()
        }
        command_manager = SimpleNamespace(
            get_term_cfg=lambda name: {
                "twist": twist.cfg,
                "foot_target": foot.cfg,
                "hand_target": hand.cfg,
                "locomotion_prior": prior_cfg,
            }[name]
        )
        command_manager.get_term = lambda name: {
            "locomotion_prior": prior,
        }[name]
        reward_manager = SimpleNamespace(get_term_cfg=lambda name: reward_cfgs[name])
        hmd_motion = object.__new__(HmdNeckTargetMotion)
        hmd_motion.neutral_probability = MICROBAN_TELEOP_INITIAL_HMD_NEUTRAL_PROBABILITY
        hmd_event_cfg = SimpleNamespace(
            func=hmd_motion,
            params={
                "neutral_probability": MICROBAN_TELEOP_INITIAL_HMD_NEUTRAL_PROBABILITY
            },
        )
        event_manager = SimpleNamespace(
            get_term_cfg=lambda name: (
                hmd_event_cfg if name == "hmd_neck_target_motion" else None
            )
        )
        env = SimpleNamespace(
            command_manager=command_manager,
            reward_manager=reward_manager,
            event_manager=event_manager,
        )
        expected_both_lift_ranges = (
            (MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M, 0.012),
            (MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M, 0.012),
            (MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M, 0.012),
            (MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M, 0.012),
            (MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M, 0.012),
            (MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M, 0.012),
            (MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M, 0.012),
            (MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M, 0.012),
            (MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M, 0.012),
            (MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M, 0.02),
        )
        expected_foot_weights = (
            MICROBAN_TELEOP_NEUTRAL_FOOT_TRACKING_WEIGHT,
            MICROBAN_TELEOP_NEUTRAL_FOOT_TRACKING_WEIGHT,
            MICROBAN_TELEOP_NEUTRAL_FOOT_TRACKING_WEIGHT,
            MICROBAN_TELEOP_NEUTRAL_FOOT_TRACKING_WEIGHT,
            MICROBAN_TELEOP_NEUTRAL_FOOT_TRACKING_WEIGHT,
            MICROBAN_TELEOP_NEUTRAL_FOOT_TRACKING_WEIGHT,
            MICROBAN_TELEOP_NEUTRAL_FOOT_TRACKING_WEIGHT,
            MICROBAN_TELEOP_NEUTRAL_FOOT_TRACKING_WEIGHT,
            2.0,
            3.0,
        )
        expected_hmd_neutral_probabilities = (
            MICROBAN_TELEOP_INITIAL_HMD_NEUTRAL_PROBABILITY,
            MICROBAN_TELEOP_INITIAL_HMD_NEUTRAL_PROBABILITY,
            MICROBAN_TELEOP_INITIAL_HMD_NEUTRAL_PROBABILITY,
            MICROBAN_TELEOP_INITIAL_HMD_NEUTRAL_PROBABILITY,
            MICROBAN_TELEOP_INITIAL_HMD_NEUTRAL_PROBABILITY,
            MICROBAN_TELEOP_MOVING_HMD_NEUTRAL_PROBABILITY,
            MICROBAN_TELEOP_MOVING_HMD_NEUTRAL_PROBABILITY,
            MICROBAN_TELEOP_MOVING_HMD_NEUTRAL_PROBABILITY,
            MICROBAN_TELEOP_MOVING_HMD_NEUTRAL_PROBABILITY,
            MICROBAN_TELEOP_MOVING_HMD_NEUTRAL_PROBABILITY,
        )
        for stage_index, (
            stage,
            expected_both_lift_range,
            expected_foot_weight,
            expected_hmd_neutral_probability,
        ) in enumerate(
            zip(
                stages,
                expected_both_lift_ranges,
                expected_foot_weights,
                expected_hmd_neutral_probabilities,
                strict=True,
            )
        ):
            stage["apply"](env)
            if stage_index == 0:
                self.assertEqual(
                    twist.cfg.signed_axis_ranges,
                    MICROBAN_TELEOP_INITIAL_SIGNED_AXIS_RANGES,
                )
                self.assertEqual(
                    twist.cfg.signed_axis_probabilities,
                    MICROBAN_TELEOP_ISOLATED_AXIS_PROBABILITIES,
                )
                self.assertFalse(prior_cfg.enabled)
                self.assertFalse(prior.eligible.any())
            self.assertEqual(
                reward_cfgs["foot_target_tracking"].weight,
                expected_foot_weight,
            )
            self.assertEqual(
                foot.cfg.lift_height_range,
                (MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M, 0.05),
            )
            self.assertEqual(
                foot.cfg.both_feet_lift_height_range,
                expected_both_lift_range,
            )
            self.assertEqual(
                hmd_motion.neutral_probability,
                expected_hmd_neutral_probability,
            )
            self.assertEqual(
                hmd_event_cfg.params["neutral_probability"],
                expected_hmd_neutral_probability,
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
        self.assertEqual(twist.cfg.rel_rotation_envs, 0.0)
        self.assertEqual(
            twist.cfg.signed_axis_probabilities,
            MICROBAN_TELEOP_MIXED_AXIS_PROBABILITIES,
        )
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
        self.assertEqual(MicrobanTeleopRlCfg.max_iterations, 20_000)
        self.assertEqual(
            MicrobanTeleopRlCfg.num_steps_per_env,
            MICROBAN_TELEOP_NUM_STEPS_PER_ENV,
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
        safe_root = self.root / "safe"
        safe_root.mkdir()
        self.safe_source_path = safe_root / "model_7.pt"
        lower, upper = microban_safe_velocity_action_delta_bounds()
        observation = TensorDict(
            {"actor": torch.zeros((1, 63), dtype=torch.float32)},
            batch_size=[1],
        )
        safe_actor = MLPModel(
            obs=observation,
            obs_groups={"actor": ["actor"]},
            obs_set="actor",
            output_dim=18,
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=False,
            distribution_cfg={
                "class_name": MicrobanSafeVelocityBoundedGaussianDistribution,
                "init_std": microban_safe_velocity_initial_action_std(),
                "lower_bound": lower,
                "upper_bound": upper,
                "std_type": "log",
            },
        )
        torch.save(
            {
                "actor_state_dict": safe_actor.state_dict(),
                "iter": 7,
                "infos": {
                    MICROBAN_SAFE_VELOCITY_RECIPE_INFO_KEY: (
                        MICROBAN_SAFE_VELOCITY_RECIPE_REVISION
                    ),
                    "env_state": {"common_step_counter": 8 * 24},
                },
            },
            self.safe_source_path,
        )
        self.safe_source_sha256 = hashlib.sha256(
            self.safe_source_path.read_bytes()
        ).hexdigest()
        source_identity = inspect_safe_velocity_checkpoint(
            self.safe_source_path,
            expected_sha256=self.safe_source_sha256,
        )
        self.safe_receipt_path = safe_root / "acceptance.json"
        self.safe_receipt_path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "gate": "microban_safe_velocity_fixed_forward_v3",
                    "checkpoint": {
                        "path": str(source_identity.path),
                        "sha256": source_identity.sha256,
                        "iteration": source_identity.iteration,
                        "checkpoint_schema_version": source_identity.schema_version,
                        "recipe_revision": source_identity.recipe_revision,
                        "actor_topology": list(source_identity.actor_topology),
                        "actor_obs_normalization": (
                            source_identity.actor_obs_normalization
                        ),
                        "observation_schema": [
                            list(item) for item in source_identity.observation_schema
                        ],
                        "action_joint_names": list(source_identity.action_joint_names),
                    },
                    "configuration": {
                        "device": "cuda:0",
                        "num_envs": 64,
                        "steps_requested": 200,
                        "steps_executed": 200,
                        "seed": 42,
                        "command_vx_m_s": 0.08,
                        "actual_command_exactly_verified_each_step": True,
                        "step_dt_s": 0.02,
                        "guard_lookahead_s": 0.12,
                        "guard_margin_ratio": 0.05,
                        "absolute_clip_max_tensor_error_rad": 0.0,
                    },
                    "thresholds": {
                        "completion_fraction_min": 0.95,
                        "fall_fraction_max": 0.0,
                        "nonfinite_fraction_max": 0.0,
                        "forward_velocity_p05_m_s_min": 0.01,
                        "forward_displacement_p05_m_min": 0.02,
                        "actual_soft_limit_violation_rad_max": 1.0e-6,
                        "actual_lookahead_soft_limit_violation_rad_max": 1.0e-6,
                        "target_clip_rad_max": 1.0e-7,
                    },
                    "metrics": {
                        "completion_fraction": 1.0,
                        "fall_fraction": 0.0,
                        "nonfinite_fraction": 0.0,
                        "forward_velocity_p05_m_s": 0.03,
                        "forward_velocity_median_m_s": 0.03,
                        "forward_displacement_p05_m": 0.03,
                        "forward_displacement_median_m": 0.03,
                        "maximum_actual_soft_limit_violation_rad": 0.0,
                        "maximum_actual_lookahead_soft_limit_violation_rad": 0.0,
                        "maximum_preferred_margin_lookahead_excess_rad": 0.0,
                        "maximum_target_clip_rad": 0.0,
                        "minimum_root_height_m": 0.2,
                    },
                    "checks": {
                        name: True
                        for name in (
                            "completion",
                            "no_falls",
                            "finite",
                            "forward_velocity",
                            "forward_displacement",
                            "actual_soft_limits",
                            "actual_lookahead_soft_limits",
                            "absolute_target_clip",
                        )
                    },
                    "status": "pass",
                    "summary": {"passed": True, "failed_checks": []},
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        self.safe_receipt_identity = validate_safe_velocity_acceptance_receipt(
            self.safe_receipt_path,
            source_identity,
        )

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
            training_provenance, training_provenance_sha256 = (
                self._training_provenance()
            )
            infos.update(
                {
                    "microban_teleop_training_contract_version": (
                        MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION
                    ),
                    "previous_action_semantics": (
                        MICROBAN_TELEOP_PREVIOUS_ACTION_SEMANTICS
                    ),
                    "microban_teleop_actor_initialization": (
                        MICROBAN_TELEOP_ACTOR_INITIALIZATION
                    ),
                    "microban_teleop_recipe_revision": (
                        MICROBAN_TELEOP_RECIPE_REVISION
                    ),
                    MICROBAN_TELEOP_TRAINING_PROVENANCE_KEY: (training_provenance),
                    MICROBAN_TELEOP_TRAINING_PROVENANCE_SHA256_KEY: (
                        training_provenance_sha256
                    ),
                    "safe_velocity_actor_bootstrap": self._valid_bootstrap_info(),
                }
            )
        path = self.root / f"model_{filename_iteration}.pt"
        torch.save(
            {
                "actor_state_dict": {"mlp.weight": torch.zeros(1)},
                "iter": iteration,
                "infos": infos,
            },
            path,
        )
        return path

    @staticmethod
    def _bounded_actor_state() -> dict[str, torch.Tensor]:
        lower = torch.tensor([-1.25, -1.0e-4], dtype=torch.float32)
        upper = torch.tensor([0.75, 0.4], dtype=torch.float32)
        distribution = AsymmetricBoundedGaussianDistribution(
            2,
            [0.05, 0.005],
            lower.tolist(),
            upper.tolist(),
            std_type="log",
        )
        return {
            f"distribution.{key}": value.detach().clone()
            for key, value in distribution.state_dict().items()
        }

    def _current_infos(self, *, pristine: bool = False) -> dict[str, object]:
        training_provenance, training_provenance_sha256 = self._training_provenance()
        infos: dict[str, object] = {
            "env_state": {"common_step_counter": 0 if pristine else 24},
            "microban_teleop_training_contract_version": (
                MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION
            ),
            "previous_action_semantics": MICROBAN_TELEOP_PREVIOUS_ACTION_SEMANTICS,
            "microban_teleop_actor_initialization": (
                MICROBAN_TELEOP_ACTOR_INITIALIZATION
            ),
            "microban_teleop_recipe_revision": MICROBAN_TELEOP_RECIPE_REVISION,
            MICROBAN_TELEOP_TRAINING_PROVENANCE_KEY: training_provenance,
            MICROBAN_TELEOP_TRAINING_PROVENANCE_SHA256_KEY: (
                training_provenance_sha256
            ),
            "safe_velocity_actor_bootstrap": self._valid_bootstrap_info(),
        }
        if pristine:
            infos["pristine_pre_update"] = True
        return infos

    @staticmethod
    def _training_provenance() -> tuple[dict[str, object], str]:
        files = {"src/mjlab_microban/test_recipe.py": "a" * 64}
        manifest: dict[str, object] = {
            "schema_version": (MICROBAN_TELEOP_TRAINING_PROVENANCE_SCHEMA_VERSION),
            "canonical_stage": False,
            "training_contract_version": (MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION),
            "recipe_revision": MICROBAN_TELEOP_RECIPE_REVISION,
            "actor_initialization": MICROBAN_TELEOP_ACTOR_INITIALIZATION,
            "resolved_config": {"critical": {}, "environment": {}, "runner": {}},
            "source": {
                "algorithm": "sha256(canonical_json_path_to_sha256_v1)",
                "tree_sha256": canonical_json_sha256(files),
                "files": files,
            },
            "invocation": {"mode": "generic"},
        }
        return manifest, canonical_json_sha256(manifest)

    def _canonical_initial_training_provenance(
        self,
        *,
        resume: bool,
        include_safe_identity: bool,
        resume_source: Path | None = None,
    ) -> tuple[dict[str, object], str]:
        manifest, _digest = self._training_provenance()
        safe_values = {
            "safe_velocity_checkpoint": (
                str(self.safe_source_path.resolve()) if include_safe_identity else None
            ),
            "safe_velocity_checkpoint_sha256": (
                self.safe_source_sha256 if include_safe_identity else None
            ),
            "safe_velocity_acceptance_receipt": (
                str(self.safe_receipt_path.resolve()) if include_safe_identity else None
            ),
        }
        manifest["canonical_stage"] = True
        manifest["resolved_config"] = {
            "critical": {
                "num_envs": 2048,
                "environment_seed": 42,
                "runner_seed": 42,
                "num_steps_per_env": 24,
                "save_interval": 500,
                "max_iterations_for_process": 1000,
                "resume": resume,
                "logger": "tensorboard",
                "upload_model": False,
                "wrapper_clip_actions": None,
                "checkpoint_consumer_mode": False,
                "save_pristine_checkpoint": False,
                **safe_values,
            },
            "environment": {},
            "runner": {"resume": resume, **safe_values},
        }
        manifest["invocation"] = {
            "mode": "canonical_v9_stage",
            "stage_start_boundary": 0,
            "stage_target_boundary": 1500,
            "parent_checkpoint_sha256": None,
            "parent_gate_sha256": None,
            "resume_source_checkpoint_path": (
                str(resume_source.resolve()) if resume_source is not None else None
            ),
            "resume_source_checkpoint_sha256": (
                hashlib.sha256(resume_source.read_bytes()).hexdigest()
                if resume_source is not None
                else None
            ),
            "resume_source_checkpoint_iteration": (
                int(resume_source.stem.removeprefix("model_"))
                if resume_source is not None
                else None
            ),
        }
        return manifest, canonical_json_sha256(manifest)

    def _valid_bootstrap_info(self) -> dict[str, object]:
        identity = inspect_safe_velocity_checkpoint(
            self.safe_source_path,
            expected_sha256=self.safe_source_sha256,
        )
        return serialize_safe_velocity_actor_bootstrap_provenance(
            SafeVelocityActorBootstrapProvenance(
                source=identity,
                acceptance_receipt=self.safe_receipt_identity,
                mapping_version=SAFE_VELOCITY_ACTOR_BOOTSTRAP_MAPPING_VERSION,
                target_actor_topology=SAFE_VELOCITY_TARGET_ACTOR_TOPOLOGY,
                new_teleop_observation_columns=(
                    SAFE_VELOCITY_NEW_TELEOP_OBSERVATION_COLUMNS
                ),
                copied_distribution_keys=SAFE_VELOCITY_COPIED_DISTRIBUTION_KEYS,
            )
        )

    def test_current_marker_is_required_and_iteration_must_match_filename(self) -> None:
        valid = self._save(12, contract=True)
        parsed = validate_teleop_checkpoint_contract(valid)
        self.assertEqual(parsed.version, "9")
        self.assertFalse(parsed.diagnostic_legacy)
        self.assertEqual(parsed.common_step_counter, 13 * 24)

        legacy = self._save(13, contract=False)
        with self.assertRaisesRegex(ValueError, "v1-v8"):
            validate_teleop_checkpoint_contract(legacy)
        with self.assertRaisesRegex(ValueError, "v1-v8"):
            validate_teleop_checkpoint_contract(legacy, allow_legacy_diagnostic=True)

        mismatched = self._save(14, filename_iteration=15, contract=True)
        with self.assertRaisesRegex(ValueError, "must match"):
            validate_teleop_checkpoint_contract(mismatched)

        missing_initialization = self._save(15, contract=True)
        payload = torch.load(
            missing_initialization, map_location="cpu", weights_only=False
        )
        del payload["infos"]["microban_teleop_actor_initialization"]
        torch.save(payload, missing_initialization)
        with self.assertRaisesRegex(ValueError, "actor initialization"):
            validate_teleop_checkpoint_contract(missing_initialization)

        missing_recipe = self._save(16, contract=True)
        payload = torch.load(missing_recipe, map_location="cpu", weights_only=False)
        del payload["infos"]["microban_teleop_recipe_revision"]
        torch.save(payload, missing_recipe)
        with self.assertRaisesRegex(ValueError, "recipe revision"):
            validate_teleop_checkpoint_contract(missing_recipe)

        v8i = self._save(17, contract=True)
        payload = torch.load(v8i, map_location="cpu", weights_only=False)
        payload["infos"]["microban_teleop_actor_initialization"] = (
            "clean_random_except_inward_shoulder_roll_v1_nonshoulder_std_1_v1"
        )
        payload["infos"]["microban_teleop_recipe_revision"] = (
            "v8i_clean_shoulder_std1_twist2_direct_bc_launch_v2"
        )
        torch.save(payload, v8i)
        with self.assertRaisesRegex(ValueError, "actor initialization"):
            validate_teleop_checkpoint_contract(v8i)

        # Every superseded version remains non-resumable even when tensor widths
        # and previous-action semantics happen to match the current contract.
        for index, version in enumerate(("2", "3", "4", "5", "6", "7", "8"), start=16):
            with self.subTest(version=version):
                old = self._save(index, contract=True)
                payload = torch.load(old, map_location="cpu", weights_only=False)
                payload["infos"]["microban_teleop_training_contract_version"] = version
                torch.save(payload, old)
                with self.assertRaisesRegex(ValueError, "v1-v8"):
                    validate_teleop_checkpoint_contract(old)

    def test_safe_velocity_bootstrap_provenance_is_fail_closed(self) -> None:
        valid = self._valid_bootstrap_info()
        validate_safe_velocity_actor_bootstrap_info(valid)

        invalid_cases = {
            "mapping_version": "superseded_v3",
            "source_recipe_revision": "invalid-old-canary",
            "source_actor_observation_normalization": True,
            "target_actor_topology": [83, 1, 18],
            "new_teleop_observation_columns": [0],
            "copied_state": "actor_mlp_only",
            "distribution_contract_equal": False,
        }
        for key, value in invalid_cases.items():
            with self.subTest(key=key):
                candidate = deepcopy(valid)
                candidate[key] = value
                with self.assertRaisesRegex(ValueError, "contract v9"):
                    validate_safe_velocity_actor_bootstrap_info(
                        candidate, verify_source_checkpoint=False
                    )

        path = self._save(30, contract=True)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["infos"]["velocity_actor_bootstrap"] = {
            **valid,
            "mapping_version": "superseded_v3",
        }
        torch.save(payload, path)
        with self.assertRaisesRegex(ValueError, "retired legacy"):
            validate_teleop_checkpoint_contract(path)

    def test_pristine_checkpoint_requires_explicit_pre_update_contract(self) -> None:
        path = self.root / "model_pristine.pt"
        infos = self._current_infos(pristine=True)
        torch.save({"iter": -1, "infos": infos}, path)
        contract = validate_teleop_checkpoint_contract(path)
        self.assertEqual(contract.iteration, -1)
        self.assertTrue(contract.pristine_pre_update)

        infos["pristine_pre_update"] = False
        torch.save({"iter": -1, "infos": infos}, path)
        with self.assertRaisesRegex(ValueError, "pristine_pre_update=true"):
            validate_teleop_checkpoint_contract(path)

    def test_current_checkpoint_step_counter_matches_completed_updates(self) -> None:
        path = self._save(12, contract=True)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["infos"]["env_state"]["common_step_counter"] = 12 * 24
        torch.save(payload, path)
        with self.assertRaisesRegex(
            ValueError,
            r"common_step_counter must equal \(iteration \+ 1\) \* 24",
        ):
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
                FloatingPointError,
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
        with self.assertRaisesRegex(
            FloatingPointError, "Checkpoint actor_state_dict.*non-finite"
        ):
            validate_bounded_actor_checkpoint_buffers(path, nonfinite_expected)

    def test_every_floating_actor_state_is_finite_before_load_or_export(self) -> None:
        valid_state = {
            "mlp.0.weight": torch.zeros(2, 2),
            "obs_normalizer.count": torch.tensor(3, dtype=torch.long),
        }
        validate_finite_actor_state(valid_state, context="test actor")

        for bad_value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=bad_value):
                invalid_state = {
                    **valid_state,
                    "mlp.0.weight": torch.tensor([[bad_value, 0.0]]),
                }
                with self.assertRaisesRegex(
                    FloatingPointError, "mlp.0.weight.*non-finite"
                ):
                    validate_finite_actor_state(invalid_state, context="test actor")

        checkpoint = self.root / "actor_finite.pt"
        torch.save({"actor_state_dict": valid_state}, checkpoint)
        validate_finite_checkpoint_actor_state(checkpoint)
        valid_state["mlp.0.weight"][0, 0] = float("nan")
        torch.save({"actor_state_dict": valid_state}, checkpoint)
        with self.assertRaisesRegex(FloatingPointError, "non-finite"):
            validate_finite_checkpoint_actor_state(checkpoint)

    def test_current_load_pins_bounds_before_base_load_including_pristine(self) -> None:
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
                        "infos": self._current_infos(pristine=pristine),
                    },
                    path,
                )
                runner = object.__new__(MicrobanTeleopOnPolicyRunner)
                runner.alg = SimpleNamespace(get_policy=lambda: policy)
                with patch.object(
                    MjlabOnPolicyRunner,
                    "load",
                    return_value=self._current_infos(pristine=pristine),
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
                    **self._current_infos(),
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

    def test_iteration_resume_resets_once_under_restored_stage_config(self) -> None:
        expected = self._bounded_actor_state()
        policy = SimpleNamespace(state_dict=lambda: expected)
        checkpoint = self.root / "model_999.pt"
        infos = {
            **self._current_infos(),
            "env_state": {"common_step_counter": 1000 * 24},
        }
        torch.save(
            {
                "actor_state_dict": {
                    key: value.clone() for key, value in expected.items()
                },
                "iter": 999,
                "infos": infos,
            },
            checkpoint,
        )

        cfg = make_microban_teleop_env_cfg(play=False)
        twist_cfg = deepcopy(cfg.commands["twist"])
        prior_cfg = deepcopy(cfg.commands["locomotion_prior"])
        prior = object.__new__(LocomotionPriorCommand)
        prior.cfg = prior_cfg
        prior.eligible = torch.zeros(2, dtype=torch.bool)
        prior.finished = torch.zeros(2, dtype=torch.bool)
        prior.teleported = torch.zeros(2, dtype=torch.bool)
        prior.launching = torch.zeros(2, dtype=torch.bool)
        prior.launch_step = torch.zeros(2, dtype=torch.long)
        prior.launch_start_joint_pos = torch.zeros(2, 12)
        prior.phase_rate = torch.zeros(2)
        prior._skip_next_advance = torch.zeros(2, dtype=torch.bool)
        reward_cfgs = {
            name: SimpleNamespace(weight=term.weight, params=dict(term.params))
            for name, term in cfg.rewards.items()
        }
        command_manager = SimpleNamespace(
            get_term_cfg=lambda name: {
                "twist": twist_cfg,
                "locomotion_prior": prior_cfg,
            }[name],
            get_term=lambda name: {"locomotion_prior": prior}[name],
        )
        reward_manager = SimpleNamespace(get_term_cfg=lambda name: reward_cfgs[name])
        env = SimpleNamespace(
            common_step_counter=0,
            command_manager=command_manager,
            reward_manager=reward_manager,
        )
        stages = cfg.curriculum["staged_curriculum"].params["stages"]

        def restore_curriculum() -> None:
            self.assertEqual(env.common_step_counter, 1000 * 24)
            stages[0]["apply"](env)

        env.curriculum_manager = SimpleNamespace(
            compute=Mock(side_effect=restore_curriculum)
        )
        reset_snapshots: list[tuple[int, dict[str, tuple[float, float]], bool]] = []

        def reset_under_restored_config():
            reset_snapshots.append(
                (
                    env.common_step_counter,
                    deepcopy(twist_cfg.signed_axis_ranges),
                    prior_cfg.enabled,
                )
            )
            return {}, {}

        env.reset = Mock(side_effect=reset_under_restored_config)
        runner = object.__new__(MicrobanTeleopOnPolicyRunner)
        runner.alg = SimpleNamespace(get_policy=lambda: policy)
        runner.env = SimpleNamespace(unwrapped=env)
        runner.current_learning_iteration = 0

        def restore_checkpoint(*args, **kwargs):
            del args, kwargs
            runner.current_learning_iteration = 999
            env.common_step_counter = infos["env_state"]["common_step_counter"]
            return infos

        with patch.object(
            MjlabOnPolicyRunner, "load", side_effect=restore_checkpoint
        ) as base_load:
            runner.load(str(checkpoint))

        base_load.assert_called_once()
        env.curriculum_manager.compute.assert_called_once_with()
        env.reset.assert_called_once_with()
        self.assertEqual(env.common_step_counter, 1000 * 24)
        self.assertEqual(
            twist_cfg.signed_axis_ranges,
            MICROBAN_TELEOP_INITIAL_SIGNED_AXIS_RANGES,
        )
        self.assertEqual(
            twist_cfg.signed_axis_probabilities,
            MICROBAN_TELEOP_ISOLATED_AXIS_PROBABILITIES,
        )
        self.assertFalse(prior_cfg.enabled)
        self.assertFalse(prior.eligible.any())
        self.assertEqual(
            reset_snapshots,
            [
                (
                    1000 * 24,
                    MICROBAN_TELEOP_INITIAL_SIGNED_AXIS_RANGES,
                    False,
                )
            ],
        )

    def test_interrupted_initial_resume_preserves_safe_identity_on_save(self) -> None:
        expected = self._bounded_actor_state()
        policy = SimpleNamespace(state_dict=lambda: expected)
        loaded_manifest, loaded_digest = self._canonical_initial_training_provenance(
            resume=False,
            include_safe_identity=True,
        )
        infos = {
            "env_state": {"common_step_counter": 500 * 24},
            "microban_teleop_training_contract_version": (
                MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION
            ),
            "previous_action_semantics": MICROBAN_TELEOP_PREVIOUS_ACTION_SEMANTICS,
            "microban_teleop_actor_initialization": (
                MICROBAN_TELEOP_ACTOR_INITIALIZATION
            ),
            "microban_teleop_recipe_revision": MICROBAN_TELEOP_RECIPE_REVISION,
            MICROBAN_TELEOP_TRAINING_PROVENANCE_KEY: loaded_manifest,
            MICROBAN_TELEOP_TRAINING_PROVENANCE_SHA256_KEY: loaded_digest,
            "safe_velocity_actor_bootstrap": self._valid_bootstrap_info(),
        }
        checkpoint = self.root / "model_499.pt"
        torch.save(
            {
                "actor_state_dict": {
                    key: value.clone() for key, value in expected.items()
                },
                "iter": 499,
                "infos": infos,
            },
            checkpoint,
        )

        current_manifest, current_digest = self._canonical_initial_training_provenance(
            resume=True,
            include_safe_identity=False,
            resume_source=checkpoint,
        )
        runner = object.__new__(MicrobanTeleopOnPolicyRunner)
        runner.alg = SimpleNamespace(get_policy=lambda: policy)
        runner.teleop_training_resume = True
        runner.checkpoint_consumer_mode = False
        runner.safe_velocity_bootstrap_provenance = None
        runner.teleop_training_provenance = current_manifest
        runner.teleop_training_provenance_sha256 = current_digest
        runner.loaded_checkpoint_contract = None

        with patch.object(MjlabOnPolicyRunner, "load", return_value=infos) as base_load:
            runner.load(str(checkpoint), load_cfg={"actor": True})
        base_load.assert_called_once()

        inherited = runner.teleop_training_provenance
        critical = inherited["resolved_config"]["critical"]
        self.assertEqual(
            critical["safe_velocity_checkpoint"],
            str(self.safe_source_path.resolve()),
        )
        self.assertEqual(
            critical["safe_velocity_checkpoint_sha256"],
            self.safe_source_sha256,
        )
        self.assertEqual(
            critical["safe_velocity_acceptance_receipt"],
            str(self.safe_receipt_path.resolve()),
        )
        self.assertEqual(
            runner.teleop_training_provenance_sha256,
            canonical_json_sha256(inherited),
        )
        self.assertEqual(
            inherited["invocation"]["resume_source_checkpoint_path"],
            str(checkpoint.resolve()),
        )
        self.assertEqual(
            inherited["invocation"]["resume_source_checkpoint_sha256"],
            hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            inherited["invocation"]["resume_source_checkpoint_iteration"], 499
        )

        output = self.root / "model_500.pt"
        with patch.object(MjlabOnPolicyRunner, "save") as base_save:
            runner.save(str(output))
        saved_infos = base_save.call_args.args[-1]
        self.assertEqual(
            saved_infos[MICROBAN_TELEOP_TRAINING_PROVENANCE_KEY], inherited
        )
        self.assertEqual(
            saved_infos[MICROBAN_TELEOP_TRAINING_PROVENANCE_SHA256_KEY],
            runner.teleop_training_provenance_sha256,
        )
        checkpoint.write_bytes(b"replaced after runner load")
        with (
            patch.object(MjlabOnPolicyRunner, "save") as replaced_save,
            self.assertRaisesRegex(ValueError, "source checkpoint SHA-256 mismatch"),
        ):
            runner.save(str(self.root / "model_501.pt"))
        replaced_save.assert_not_called()

    def test_later_interrupted_resume_pins_exact_intermediate_checkpoint(self) -> None:
        expected = self._bounded_actor_state()
        policy = SimpleNamespace(state_dict=lambda: expected)
        boundary_source = self.root / "model_1499.pt"
        boundary_source.write_bytes(b"accepted boundary checkpoint")
        parent_checkpoint_sha256 = hashlib.sha256(
            boundary_source.read_bytes()
        ).hexdigest()
        parent_gate_sha256 = "d" * 64

        loaded_manifest, _loaded_digest = self._canonical_initial_training_provenance(
            resume=True,
            include_safe_identity=False,
            resume_source=boundary_source,
        )
        loaded_manifest["invocation"].update(
            {
                "stage_start_boundary": 1500,
                "stage_target_boundary": 3000,
                "parent_checkpoint_sha256": parent_checkpoint_sha256,
                "parent_gate_sha256": parent_gate_sha256,
            }
        )
        loaded_digest = canonical_json_sha256(loaded_manifest)
        infos = {
            "env_state": {"common_step_counter": 2000 * 24},
            "microban_teleop_training_contract_version": (
                MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION
            ),
            "previous_action_semantics": MICROBAN_TELEOP_PREVIOUS_ACTION_SEMANTICS,
            "microban_teleop_actor_initialization": (
                MICROBAN_TELEOP_ACTOR_INITIALIZATION
            ),
            "microban_teleop_recipe_revision": MICROBAN_TELEOP_RECIPE_REVISION,
            MICROBAN_TELEOP_TRAINING_PROVENANCE_KEY: loaded_manifest,
            MICROBAN_TELEOP_TRAINING_PROVENANCE_SHA256_KEY: loaded_digest,
            "safe_velocity_actor_bootstrap": self._valid_bootstrap_info(),
        }
        checkpoint = self.root / "model_1999.pt"
        torch.save(
            {
                "actor_state_dict": {
                    key: value.clone() for key, value in expected.items()
                },
                "iter": 1999,
                "infos": infos,
            },
            checkpoint,
        )

        current_manifest, _current_digest = self._canonical_initial_training_provenance(
            resume=True,
            include_safe_identity=False,
            resume_source=checkpoint,
        )
        current_manifest["invocation"].update(
            {
                "stage_start_boundary": 1500,
                "stage_target_boundary": 3000,
                "parent_checkpoint_sha256": parent_checkpoint_sha256,
                "parent_gate_sha256": parent_gate_sha256,
            }
        )

        def make_runner(manifest: dict[str, object]) -> MicrobanTeleopOnPolicyRunner:
            runner = object.__new__(MicrobanTeleopOnPolicyRunner)
            runner.alg = SimpleNamespace(get_policy=lambda: policy)
            runner.teleop_training_resume = True
            runner.checkpoint_consumer_mode = False
            runner.safe_velocity_bootstrap_provenance = None
            runner.teleop_training_provenance = manifest
            runner.teleop_training_provenance_sha256 = canonical_json_sha256(manifest)
            runner.loaded_checkpoint_contract = None
            return runner

        runner = make_runner(current_manifest)
        with (
            patch.object(MjlabOnPolicyRunner, "load", return_value=infos),
            patch.object(
                MicrobanTeleopOnPolicyRunner,
                "_validate_canonical_parent_gate",
            ),
        ):
            runner.load(str(checkpoint), load_cfg={"actor": True})
        self.assertEqual(
            runner.teleop_training_provenance["invocation"][
                "resume_source_checkpoint_sha256"
            ],
            hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        )

        changed_source = deepcopy(current_manifest)
        changed_source["invocation"]["resume_source_checkpoint_sha256"] = "0" * 64
        changed_runner = make_runner(changed_source)
        with (
            patch.object(MjlabOnPolicyRunner, "load") as base_load,
            patch.object(
                MicrobanTeleopOnPolicyRunner,
                "_validate_canonical_parent_gate",
            ),
            self.assertRaisesRegex(ValueError, "resume source.*SHA-256"),
        ):
            changed_runner.load(str(checkpoint), load_cfg={"actor": True})
        base_load.assert_not_called()

    def test_boundary_resume_hashes_resolved_path_when_loader_passes_string(
        self,
    ) -> None:
        expected = self._bounded_actor_state()
        policy = SimpleNamespace(state_dict=lambda: expected)
        loaded_manifest, loaded_digest = self._canonical_initial_training_provenance(
            resume=False,
            include_safe_identity=True,
        )
        infos = {
            "env_state": {"common_step_counter": 1500 * 24},
            "microban_teleop_training_contract_version": (
                MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION
            ),
            "previous_action_semantics": MICROBAN_TELEOP_PREVIOUS_ACTION_SEMANTICS,
            "microban_teleop_actor_initialization": (
                MICROBAN_TELEOP_ACTOR_INITIALIZATION
            ),
            "microban_teleop_recipe_revision": MICROBAN_TELEOP_RECIPE_REVISION,
            MICROBAN_TELEOP_TRAINING_PROVENANCE_KEY: loaded_manifest,
            MICROBAN_TELEOP_TRAINING_PROVENANCE_SHA256_KEY: loaded_digest,
            "safe_velocity_actor_bootstrap": self._valid_bootstrap_info(),
        }
        checkpoint = self.root / "model_1499.pt"
        torch.save(
            {
                "actor_state_dict": {
                    key: value.clone() for key, value in expected.items()
                },
                "iter": 1499,
                "infos": infos,
            },
            checkpoint,
        )
        checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

        current_manifest, _current_digest = (
            self._canonical_initial_training_provenance(
                resume=True,
                include_safe_identity=False,
                resume_source=checkpoint,
            )
        )
        current_manifest["invocation"].update(
            {
                "stage_start_boundary": 1500,
                "stage_target_boundary": 3000,
                "parent_checkpoint_sha256": checkpoint_sha256,
                "parent_gate_sha256": "d" * 64,
            }
        )
        runner = object.__new__(MicrobanTeleopOnPolicyRunner)
        runner.alg = SimpleNamespace(get_policy=lambda: policy)
        runner.teleop_training_resume = True
        runner.checkpoint_consumer_mode = False
        runner.safe_velocity_bootstrap_provenance = None
        runner.teleop_training_provenance = current_manifest
        runner.teleop_training_provenance_sha256 = canonical_json_sha256(
            current_manifest
        )
        runner.loaded_checkpoint_contract = None

        with (
            patch.object(MjlabOnPolicyRunner, "load", return_value=infos) as base_load,
            patch.object(
                MicrobanTeleopOnPolicyRunner,
                "_validate_canonical_parent_gate",
            ) as parent_gate,
        ):
            runner.load(str(checkpoint), load_cfg={"actor": True})

        base_load.assert_called_once()
        parent_gate.assert_called_once_with(
            boundary=1500,
            parent_checkpoint_sha256=checkpoint_sha256,
            parent_gate_sha256="d" * 64,
        )

    def test_save_adds_current_marker_and_legacy_runner_cannot_write(self) -> None:
        fresh = object.__new__(MicrobanTeleopOnPolicyRunner)
        fresh.loaded_checkpoint_contract = None
        fresh.alg = SimpleNamespace(
            get_policy=lambda: SimpleNamespace(
                state_dict=lambda: {"mlp.weight": torch.zeros(1)}
            )
        )
        fresh.safe_velocity_actor_bootstrap_info = self._valid_bootstrap_info()
        (
            fresh.teleop_training_provenance,
            fresh.teleop_training_provenance_sha256,
        ) = self._training_provenance()
        with patch.object(MjlabOnPolicyRunner, "save") as base_save:
            fresh.save(
                str(self.root / "model_0.pt"),
                {
                    "source": "test",
                    "velocity_actor_bootstrap": {"forged": True},
                },
            )
        saved_infos = base_save.call_args.args[-1]
        self.assertEqual(saved_infos["microban_teleop_training_contract_version"], "9")
        self.assertEqual(
            saved_infos["previous_action_semantics"],
            MICROBAN_TELEOP_PREVIOUS_ACTION_SEMANTICS,
        )
        self.assertEqual(
            saved_infos["microban_teleop_actor_initialization"],
            MICROBAN_TELEOP_ACTOR_INITIALIZATION,
        )
        self.assertEqual(
            saved_infos["microban_teleop_recipe_revision"],
            MICROBAN_TELEOP_RECIPE_REVISION,
        )
        self.assertEqual(
            saved_infos["microban_teleop_training_provenance_sha256"],
            fresh.teleop_training_provenance_sha256,
        )
        self.assertNotIn("velocity_actor_bootstrap", saved_infos)
        self.assertEqual(
            saved_infos["safe_velocity_actor_bootstrap"],
            fresh.safe_velocity_actor_bootstrap_info,
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
        runner.alg = SimpleNamespace(
            get_policy=lambda: SimpleNamespace(
                state_dict=lambda: {"mlp.weight": torch.zeros(1)}
            )
        )
        with (
            patch.object(MjlabOnPolicyRunner, "load") as base_load,
            self.assertRaisesRegex(ValueError, "v1-v8"),
        ):
            runner.load(str(checkpoint), load_cfg={"actor": True})
        base_load.assert_not_called()

        for load_cfg in (
            None,
            {"actor": True, "iteration": True},
            {"actor": True, "optimizer": True},
            {"critic": True},
        ):
            with (
                self.subTest(load_cfg=load_cfg),
                patch.object(MjlabOnPolicyRunner, "load") as base_load,
                self.assertRaisesRegex(ValueError, "v1-v8"),
            ):
                runner.load(
                    str(checkpoint),
                    load_cfg=load_cfg,
                    allow_legacy_teleop_contract=True,
                )
            base_load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
