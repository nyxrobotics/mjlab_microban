"""Contract tests for the isolated bounded Microban velocity task."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
from mjlab.envs.mdp.actions import JointPositionAction
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from mjlab_microban.scripts.evaluate_safe_velocity_checkpoint import (
    _make_evaluation_env_cfg,
    _TerminalStateRecorder,
)
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_TELEOP_ACTION_JOINT_NAMES,
)
from mjlab_microban.tasks.microban_safe_velocity_checkpoint import (
    MICROBAN_SAFE_VELOCITY_ACTOR_TOPOLOGY,
    inspect_safe_velocity_checkpoint,
    load_frozen_safe_velocity_actor,
)
from mjlab_microban.tasks.microban_safe_velocity_env_cfg import (
    MICROBAN_SAFE_VELOCITY_CURRICULUM_STAGES,
    MICROBAN_SAFE_VELOCITY_INITIAL_COMMAND,
    MICROBAN_SAFE_VELOCITY_JOINT_NAMES,
    MICROBAN_SAFE_VELOCITY_OBSERVATION_SCHEMA,
    MICROBAN_SAFE_VELOCITY_OBSERVATION_WIDTH,
    MICROBAN_SAFE_VELOCITY_SAGITTAL_LEG_JOINT_NAMES,
    MicrobanSafeVelocityRlCfg,
    make_microban_safe_velocity_env_cfg,
    microban_safe_velocity_action_delta_bounds,
    microban_safe_velocity_initial_action_std,
)
from mjlab_microban.tasks.microban_safe_velocity_mdp import (
    MICROBAN_SAFE_VELOCITY_GUARD_LOOKAHEAD_S,
    MICROBAN_SAFE_VELOCITY_GUARD_MARGIN_RATIO,
    MICROBAN_SAFE_VELOCITY_RECIPE_INFO_KEY,
    MICROBAN_SAFE_VELOCITY_RECIPE_REVISION,
    MicrobanSafeVelocityBoundedGaussianDistribution,
    MicrobanSafeVelocityBoundedPPO,
    SafeVelocityStagedCurriculum,
    commanded_planar_velocity_progress,
    measured_joint_margin_lookahead_l1_sum,
    planar_velocity_tracking_exp,
    preferred_joint_position_bounds,
    preferred_target_margin_l1_sum,
    set_safe_velocity_reward_overrides,
)
from mjlab_microban.tasks.microban_teleop_bootstrap import (
    TELEOP_SHOULDER_ROLL_ACTION_INDICES,
    TELEOP_SHOULDER_ROLL_INITIAL_LATENT_BIASES,
)
from mjlab_microban.tasks.microban_tracking_env_cfg import (
    MICROBAN_BODY_JOINT_SOFT_LIMITS,
    MICROBAN_TRACKING_ACTION_JOINT_NAMES,
    microban_tracking_action_delta_bounds,
)
from mjlab_microban.tasks.microban_velocity_env_cfg import (
    MicrobanVelocityRlCfg,
    make_microban_velocity_env_cfg,
)


def _action(
    *,
    raw: torch.Tensor,
    offset: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    position: torch.Tensor | None = None,
    velocity: torch.Tensor | None = None,
) -> JointPositionAction:
    term = object.__new__(JointPositionAction)
    term._raw_actions = raw
    term._scale = 1.0
    term._offset = offset
    term._clip = torch.stack((lower, upper), dim=-1)
    term._target_ids = torch.arange(raw.shape[-1])
    term.cfg = SimpleNamespace(clip={".*": (-1.0, 1.0)})
    if position is not None and velocity is not None:
        term._entity = SimpleNamespace(
            data=SimpleNamespace(
                joint_pos=position,
                joint_vel=velocity,
                soft_joint_pos_limits=torch.stack((lower, upper), dim=-1),
            )
        )
    return term


def _env(action: JointPositionAction) -> SimpleNamespace:
    return SimpleNamespace(
        action_manager=SimpleNamespace(get_term=lambda _name: action)
    )


class SafeVelocityConfigurationTest(unittest.TestCase):
    def test_exact_wire_contract_and_absolute_clips(self) -> None:
        self.assertEqual(
            MICROBAN_SAFE_VELOCITY_JOINT_NAMES,
            MICROBAN_TELEOP_ACTION_JOINT_NAMES,
        )
        self.assertEqual(
            MICROBAN_SAFE_VELOCITY_JOINT_NAMES,
            MICROBAN_TRACKING_ACTION_JOINT_NAMES,
        )
        self.assertEqual(
            microban_safe_velocity_action_delta_bounds(),
            microban_tracking_action_delta_bounds(),
        )
        self.assertEqual(MICROBAN_SAFE_VELOCITY_OBSERVATION_WIDTH, 63)
        self.assertEqual(
            tuple(name for name, _width in MICROBAN_SAFE_VELOCITY_OBSERVATION_SCHEMA),
            (
                "base_ang_vel",
                "projected_gravity",
                "joint_pos",
                "joint_vel",
                "actions",
                "command",
            ),
        )
        cfg = make_microban_safe_velocity_env_cfg()
        action = cfg.actions["joint_pos"]
        self.assertEqual(
            tuple(action.actuator_names), MICROBAN_SAFE_VELOCITY_JOINT_NAMES
        )
        self.assertEqual(action.scale, 1.0)
        self.assertTrue(action.use_default_offset)
        self.assertEqual(action.clip, MICROBAN_BODY_JOINT_SOFT_LIMITS)

    def test_new_task_does_not_mutate_legacy_velocity_contract(self) -> None:
        legacy_before = make_microban_velocity_env_cfg()
        safe = make_microban_safe_velocity_env_cfg()
        legacy_after = make_microban_velocity_env_cfg()

        self.assertIsNone(legacy_before.actions["joint_pos"].clip)
        self.assertIsNone(legacy_after.actions["joint_pos"].clip)
        self.assertEqual(legacy_before.actions["joint_pos"].scale, 1.0)
        self.assertEqual(legacy_after.actions["joint_pos"].scale, 1.0)
        self.assertIsNotNone(safe.actions["joint_pos"].clip)
        self.assertEqual(
            MicrobanVelocityRlCfg.actor.distribution_cfg["class_name"],
            "GaussianDistribution",
        )

    def test_initial_forward_curriculum_and_safety_rewards_are_pinned(self) -> None:
        cfg = make_microban_safe_velocity_env_cfg()
        command = cfg.commands["twist"]
        self.assertEqual(
            command.ranges.lin_vel_x,
            MICROBAN_SAFE_VELOCITY_INITIAL_COMMAND["lin_vel_x"],
        )
        self.assertGreaterEqual(command.ranges.lin_vel_x[0], 0.0)
        self.assertEqual(command.ranges.lin_vel_y, (0.0, 0.0))
        self.assertEqual(command.ranges.ang_vel_z, (0.0, 0.0))
        self.assertEqual(command.rel_forward_envs, 0.0)
        steps = [stage[1] for stage in MICROBAN_SAFE_VELOCITY_CURRICULUM_STAGES]
        self.assertEqual(steps, [100 * 24, 300 * 24, 600 * 24, 1200 * 24, 2500 * 24])
        self.assertEqual(cfg.rewards["track_linear_velocity"].weight, 3.0)
        self.assertEqual(cfg.rewards["track_linear_velocity"].params["std"], 0.2)
        self.assertIs(
            cfg.rewards["track_linear_velocity"].func,
            planar_velocity_tracking_exp,
        )
        self.assertEqual(cfg.rewards["linear_velocity_error_l1"].weight, -8.0)
        stage_zero = MICROBAN_SAFE_VELOCITY_CURRICULUM_STAGES[0]
        self.assertEqual(stage_zero[2], MICROBAN_SAFE_VELOCITY_INITIAL_COMMAND)
        self.assertEqual(
            stage_zero[3],
            {
                "track_linear_velocity": {"weight": 5.0, "std": 0.10},
                "linear_velocity_error_l1": {"weight": -16.0},
            },
        )
        self.assertEqual(cfg.rewards["target_near_limit"].weight, -2.0)
        self.assertEqual(cfg.rewards["joint_limit_lookahead"].weight, -10.0)
        self.assertEqual(
            cfg.rewards["joint_limit_lookahead"].params["lookahead_s"],
            MICROBAN_SAFE_VELOCITY_GUARD_LOOKAHEAD_S,
        )
        self.assertEqual(cfg.rewards["raw_action_l2"].weight, -0.01)
        self.assertEqual(cfg.rewards["air_time"].weight, 2.0)
        self.assertIs(
            cfg.rewards["air_time"].func,
            commanded_planar_velocity_progress,
        )
        self.assertEqual(
            cfg.rewards["air_time"].params,
            {"command_name": "twist", "command_threshold": 0.01},
        )
        self.assertEqual(MicrobanSafeVelocityRlCfg.algorithm.entropy_coef, 0.0)
        self.assertGreaterEqual(cfg.sim.nconmax, 512)
        self.assertGreaterEqual(cfg.sim.njmax, 2048)

    def test_reward_and_command_curriculum_reconstructs_overdue_v4_stages(
        self,
    ) -> None:
        cfg = make_microban_safe_velocity_env_cfg()
        term_cfg = cfg.curriculum["safe_velocity_stages"]
        command_cfg = cfg.commands["twist"]
        reward_cfgs = cfg.rewards
        fake_env = SimpleNamespace(
            common_step_counter=600 * 24,
            device="cpu",
            command_manager=SimpleNamespace(
                get_term_cfg=lambda name: command_cfg if name == "twist" else None
            ),
            reward_manager=SimpleNamespace(get_term_cfg=lambda name: reward_cfgs[name]),
        )
        curriculum = SafeVelocityStagedCurriculum(term_cfg, fake_env)
        state = curriculum(
            fake_env,
            torch.tensor([0]),
            stages=term_cfg.params["stages"],
        )
        self.assertEqual(float(state["stage"].item()), 3.0)
        self.assertEqual(command_cfg.ranges.lin_vel_x, (0.05, 0.14))
        self.assertEqual(command_cfg.ranges.lin_vel_y, (0.0, 0.0))
        self.assertEqual(command_cfg.ranges.ang_vel_z, (0.0, 0.0))
        self.assertEqual(reward_cfgs["track_linear_velocity"].weight, 5.0)
        self.assertEqual(reward_cfgs["track_linear_velocity"].params["std"], 0.10)
        self.assertEqual(reward_cfgs["linear_velocity_error_l1"].weight, -16.0)
        self.assertEqual(reward_cfgs["air_time"].weight, 2.0)

    def test_reward_override_validation_is_fail_closed_and_atomic(self) -> None:
        reward_cfgs = make_microban_safe_velocity_env_cfg().rewards
        fake_env = SimpleNamespace(
            reward_manager=SimpleNamespace(
                get_term_cfg=lambda name: reward_cfgs.get(name)
            )
        )
        initial_tracking_weight = reward_cfgs["track_linear_velocity"].weight
        with self.assertRaisesRegex(ValueError, "Unsupported 'air_time'"):
            set_safe_velocity_reward_overrides(
                fake_env,
                {
                    "track_linear_velocity": {"weight": 9.0},
                    "air_time": {"std": 0.1},
                },
            )
        self.assertEqual(
            reward_cfgs["track_linear_velocity"].weight, initial_tracking_weight
        )
        with self.assertRaisesRegex(ValueError, "Unsupported safe velocity"):
            set_safe_velocity_reward_overrides(
                fake_env, {"unreviewed_term": {"weight": 1.0}}
            )
        with self.assertRaisesRegex(ValueError, "must be finite"):
            set_safe_velocity_reward_overrides(
                fake_env, {"air_time": {"weight": float("inf")}}
            )
        set_safe_velocity_reward_overrides(fake_env, {"air_time": {"weight": 6.0}})
        self.assertEqual(reward_cfgs["air_time"].weight, 6.0)

    def test_planar_tracking_reward_is_invariant_to_vertical_velocity(self) -> None:
        command = torch.tensor([[0.08, 0.0, 0.0], [0.08, 0.0, 0.0]])
        velocity = torch.tensor([[0.03, -0.02, 0.0], [0.03, -0.02, 1.25]])
        fake_env = SimpleNamespace(
            scene={
                "robot": SimpleNamespace(
                    data=SimpleNamespace(root_link_lin_vel_b=velocity)
                )
            },
            command_manager=SimpleNamespace(
                get_command=lambda name: command if name == "twist" else None
            ),
        )
        reward = planar_velocity_tracking_exp(fake_env, std=0.10)
        torch.testing.assert_close(reward[0], reward[1])
        expected = torch.exp(torch.tensor(-((0.08 - 0.03) ** 2 + 0.02**2) / 0.10**2))
        torch.testing.assert_close(reward[0], expected)

    def test_planar_progress_is_bounded_and_command_aligned(self) -> None:
        commands = torch.tensor(
            [
                [0.10, 0.0, 0.0],
                [0.10, 0.0, 0.0],
                [0.10, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.10, 0.0],
                [-0.10, 0.0, 0.0],
                [0.10, 0.0, 0.0],
                [0.10, 0.0, 0.0],
            ]
        )
        body_velocity = torch.tensor(
            [
                [0.05, 0.0, 0.0],  # forward half-speed
                [0.20, 0.0, 0.0],  # forward cap
                [-0.10, 0.0, 0.0],  # reverse of command
                [0.20, 0.0, 0.0],  # zero command
                [0.0, 0.05, 0.0],  # lateral half-speed
                [-0.10, 0.0, 0.0],  # reverse command matched
                [0.0, 0.10, 0.0],  # orthogonal slip
                [0.0, 0.0, 0.0],  # stationary
            ]
        )
        fake_env = SimpleNamespace(
            num_envs=8,
            scene={
                "robot": SimpleNamespace(
                    data=SimpleNamespace(root_link_lin_vel_b=body_velocity)
                ),
            },
            command_manager=SimpleNamespace(
                get_command=lambda name: commands if name == "twist" else None
            ),
            extras={"log": {}},
        )
        reward = commanded_planar_velocity_progress(
            fake_env, command_name="twist", command_threshold=0.01
        )
        torch.testing.assert_close(
            reward,
            torch.tensor([0.5, 1.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0]),
        )
        self.assertTrue(torch.all((0.0 <= reward) & (reward <= 1.0)).item())
        torch.testing.assert_close(
            fake_env.extras["log"]["Metrics/commanded_planar_velocity_progress"],
            reward.mean(),
        )

        fake_env.scene["robot"].data.root_link_lin_vel_b = body_velocity[:, :2]
        with self.assertRaisesRegex(ValueError, "body velocity"):
            commanded_planar_velocity_progress(fake_env)
        fake_env.scene["robot"].data.root_link_lin_vel_b = body_velocity.clone()
        fake_env.scene["robot"].data.root_link_lin_vel_b[0, 0] = float("inf")
        with self.assertRaisesRegex(ValueError, "finite"):
            commanded_planar_velocity_progress(fake_env)
        fake_env.scene["robot"].data.root_link_lin_vel_b = body_velocity.clone()
        bad_commands = commands[:, :2]
        fake_env.command_manager.get_command = lambda _name: bad_commands
        with self.assertRaisesRegex(ValueError, "command"):
            commanded_planar_velocity_progress(fake_env)
        bad_commands = commands.clone()
        bad_commands[0, 0] = float("nan")
        fake_env.command_manager.get_command = lambda _name: bad_commands
        with self.assertRaisesRegex(ValueError, "finite"):
            commanded_planar_velocity_progress(fake_env)

    def test_gate_uses_pre_reset_terminal_recorder_without_partial_resets(
        self,
    ) -> None:
        cfg = _make_evaluation_env_cfg(
            num_envs=4,
            steps=200,
            command_vx_m_s=0.08,
            seed=42,
        )
        self.assertTrue(cfg.auto_reset)
        self.assertEqual(set(cfg.recorders), {"safe_velocity_terminal_state"})
        self.assertIs(
            cfg.recorders["safe_velocity_terminal_state"].func,
            _TerminalStateRecorder,
        )
        command = cfg.commands["twist"]
        self.assertEqual(command.ranges.lin_vel_x, (0.08, 0.08))
        self.assertEqual(command.rel_forward_envs, 0.0)

    def test_actor_is_raw_input_bounded_and_critic_remains_normalized(self) -> None:
        self.assertFalse(MicrobanSafeVelocityRlCfg.actor.obs_normalization)
        self.assertTrue(MicrobanSafeVelocityRlCfg.critic.obs_normalization)
        self.assertIs(
            MicrobanSafeVelocityRlCfg.actor.distribution_cfg["class_name"],
            MicrobanSafeVelocityBoundedGaussianDistribution,
        )
        self.assertEqual(
            MicrobanSafeVelocityRlCfg.algorithm.class_name,
            "mjlab_microban.tasks.microban_safe_velocity_mdp:"
            "MicrobanSafeVelocityBoundedPPO",
        )
        lower, upper = microban_safe_velocity_action_delta_bounds()
        self.assertEqual(len(lower), 18)
        self.assertTrue(all(lo < 0.0 < hi for lo, hi in zip(lower, upper, strict=True)))
        distribution = MicrobanSafeVelocityBoundedGaussianDistribution(
            18,
            microban_safe_velocity_initial_action_std(),
            lower,
            upper,
        )
        distribution.update(torch.zeros((1, 18)))
        self.assertTrue(torch.all(distribution.min_std <= distribution.std).item())
        self.assertTrue(torch.all(distribution.std <= distribution.max_std).item())

    def test_only_sagittal_leg_exploration_is_widened(self) -> None:
        std_by_joint = dict(
            zip(
                MICROBAN_SAFE_VELOCITY_JOINT_NAMES,
                microban_safe_velocity_initial_action_std(),
                strict=True,
            )
        )
        self.assertEqual(
            {name for name, std in std_by_joint.items() if std == 0.15},
            set(MICROBAN_SAFE_VELOCITY_SAGITTAL_LEG_JOINT_NAMES),
        )
        for name in MICROBAN_SAFE_VELOCITY_JOINT_NAMES:
            if name in MICROBAN_SAFE_VELOCITY_SAGITTAL_LEG_JOINT_NAMES:
                self.assertEqual(std_by_joint[name], 0.15)
            elif "shoulder_roll" in name:
                self.assertLess(std_by_joint[name], 0.08)
            elif "shoulder" in name or "elbow" in name:
                self.assertEqual(std_by_joint[name], 0.05)
            else:
                self.assertEqual(std_by_joint[name], 0.08)

        lower, upper = microban_safe_velocity_action_delta_bounds()
        distribution = MicrobanSafeVelocityBoundedGaussianDistribution(
            18,
            tuple(std_by_joint[name] for name in MICROBAN_SAFE_VELOCITY_JOINT_NAMES),
            lower,
            upper,
        )
        distribution.update(torch.zeros((1, 18)))
        self.assertTrue(torch.all(distribution.std <= distribution.max_std).item())


class SafeVelocityGuardTest(unittest.TestCase):
    def test_preferred_margin_includes_near_limit_default(self) -> None:
        default = torch.tensor([[0.95, 0.50]])
        lower = torch.zeros_like(default)
        upper = torch.ones_like(default)
        preferred_lower, preferred_upper = preferred_joint_position_bounds(
            default,
            lower,
            upper,
            margin_ratio=MICROBAN_SAFE_VELOCITY_GUARD_MARGIN_RATIO,
        )
        torch.testing.assert_close(preferred_lower, torch.tensor([[0.05, 0.05]]))
        torch.testing.assert_close(preferred_upper, torch.tensor([[0.95, 0.95]]))

    def test_target_penalty_and_q_plus_point12_qdot_guard(self) -> None:
        raw = torch.tensor([[0.01, 0.0], [0.0, 0.0]])
        offset = torch.tensor([[0.95, 0.50], [0.95, 0.50]])
        lower = torch.zeros_like(raw)
        upper = torch.ones_like(raw)
        position = torch.tensor([[0.95, 0.50], [0.95, 0.50]])
        velocity = torch.tensor([[0.10, 0.00], [0.00, -4.00]])
        action = _action(
            raw=raw,
            offset=offset,
            lower=lower,
            upper=upper,
            position=position,
            velocity=velocity,
        )
        env = _env(action)

        target_penalty = preferred_target_margin_l1_sum(env)
        # Row zero asks 0.01 rad outward from the default-preserving boundary.
        torch.testing.assert_close(target_penalty, torch.tensor([0.02, 0.0]))

        guard = measured_joint_margin_lookahead_l1_sum(env, lookahead_s=0.12)
        # Row zero projects 0.012 above its default boundary; row one projects
        # joint 1 from 0.50 to 0.02, 0.03 below the preferred 0.05 margin.
        torch.testing.assert_close(guard, torch.tensor([0.024, 0.06]))

    def test_clean_actor_head_is_zero_and_ppo_steps_bounded_copy(self) -> None:
        torch.manual_seed(20260925)
        batch = 4
        observations = TensorDict(
            {"policy": torch.randn(batch, 63)}, batch_size=[batch]
        )
        groups = {"actor": ["policy"], "critic": ["policy"]}
        lower, upper = microban_safe_velocity_action_delta_bounds()
        actor = MLPModel(
            obs=observations,
            obs_groups=groups,
            obs_set="actor",
            output_dim=18,
            hidden_dims=(16,),
            distribution_cfg={
                "class_name": MicrobanSafeVelocityBoundedGaussianDistribution,
                "init_std": microban_safe_velocity_initial_action_std(),
                "lower_bound": lower,
                "upper_bound": upper,
                "std_type": "log",
            },
        )
        final = actor.mlp[-1]
        assert isinstance(final, torch.nn.Linear)
        shoulder_ids = list(TELEOP_SHOULDER_ROLL_ACTION_INDICES)
        neutral_ids = sorted(set(range(18)) - set(shoulder_ids))
        torch.testing.assert_close(
            final.weight[neutral_ids], torch.zeros_like(final.weight[neutral_ids])
        )
        torch.testing.assert_close(
            final.bias[neutral_ids], torch.zeros_like(final.bias[neutral_ids])
        )
        torch.testing.assert_close(
            final.weight[shoulder_ids], torch.zeros_like(final.weight[shoulder_ids])
        )
        torch.testing.assert_close(
            final.bias[shoulder_ids],
            final.bias.new_tensor(TELEOP_SHOULDER_ROLL_INITIAL_LATENT_BIASES),
        )
        deterministic = actor.distribution.deterministic_output(
            actor.mlp(observations["policy"])
        )
        torch.testing.assert_close(
            deterministic[:, neutral_ids],
            torch.zeros_like(deterministic[:, neutral_ids]),
        )
        self.assertTrue(torch.all(deterministic[:, 1] < 0.0).item())
        self.assertTrue(torch.all(deterministic[:, 10] > 0.0).item())

        critic = MLPModel(
            obs=observations,
            obs_groups=groups,
            obs_set="critic",
            output_dim=1,
            hidden_dims=(16,),
        )
        storage = RolloutStorage("rl", batch, 2, observations, [18], "cpu")
        algorithm = MicrobanSafeVelocityBoundedPPO(
            actor,
            critic,
            storage,
            device="cpu",
            rnd_cfg=None,
            symmetry_cfg=None,
        )
        environment_action = algorithm.act(observations)
        latent = algorithm.transition.actions
        assert latent is not None
        self.assertFalse(torch.equal(environment_action, latent))
        torch.testing.assert_close(
            environment_action, actor.distribution.to_environment_action(latent)
        )
        self.assertTrue(torch.all(environment_action > torch.tensor(lower)).item())
        self.assertTrue(torch.all(environment_action < torch.tensor(upper)).item())


class SafeVelocityCheckpointTest(unittest.TestCase):
    @staticmethod
    def _checkpoint_payload(actor: MLPModel, iteration: int) -> dict:
        return {
            "iter": iteration,
            "actor_state_dict": actor.state_dict(),
            "infos": {
                MICROBAN_SAFE_VELOCITY_RECIPE_INFO_KEY: (
                    MICROBAN_SAFE_VELOCITY_RECIPE_REVISION
                ),
                "env_state": {"common_step_counter": (iteration + 1) * 24},
            },
        }

    @staticmethod
    def _actor() -> MLPModel:
        observation = TensorDict({"actor": torch.zeros((1, 63))}, batch_size=[1])
        lower, upper = microban_safe_velocity_action_delta_bounds()
        return MLPModel(
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

    def test_loader_validates_topology_bounds_and_returns_frozen_actor(self) -> None:
        actor = self._actor()
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model_7.pt"
            torch.save(self._checkpoint_payload(actor, 7), checkpoint)
            identity = inspect_safe_velocity_checkpoint(checkpoint)
            self.assertEqual(identity.iteration, 7)
            self.assertEqual(
                identity.actor_topology, MICROBAN_SAFE_VELOCITY_ACTOR_TOPOLOGY
            )
            loaded, loaded_identity = load_frozen_safe_velocity_actor(checkpoint)
            self.assertEqual(loaded_identity.sha256, identity.sha256)
            output = loaded(TensorDict({"actor": torch.zeros((2, 63))}, batch_size=[2]))
            self.assertEqual(tuple(output.shape), (2, 18))
            self.assertTrue(torch.isfinite(output).all().item())
            self.assertTrue(
                all(not value.requires_grad for value in loaded.parameters())
            )

    def test_loader_rejects_legacy_or_wrong_bounded_state(self) -> None:
        actor = self._actor()
        state = actor.state_dict()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            missing = dict(state)
            del missing["distribution.lower_bound"]
            missing_path = root / "model_1.pt"
            missing_payload = self._checkpoint_payload(actor, 1)
            missing_payload["actor_state_dict"] = missing
            torch.save(missing_payload, missing_path)
            with self.assertRaisesRegex(ValueError, "lower_bound"):
                inspect_safe_velocity_checkpoint(missing_path)

            normalized = dict(state)
            normalized["obs_normalizer._mean"] = torch.zeros(63)
            normalized_path = root / "model_2.pt"
            normalized_payload = self._checkpoint_payload(actor, 2)
            normalized_payload["actor_state_dict"] = normalized
            torch.save(normalized_payload, normalized_path)
            with self.assertRaisesRegex(ValueError, "observation normalizer"):
                inspect_safe_velocity_checkpoint(normalized_path)

    def test_loader_rejects_wrong_recipe_step_count_and_derived_buffers(self) -> None:
        actor = self._actor()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            wrong_recipe = self._checkpoint_payload(actor, 3)
            wrong_recipe["infos"][MICROBAN_SAFE_VELOCITY_RECIPE_INFO_KEY] = "v1"
            wrong_recipe_path = root / "model_3.pt"
            torch.save(wrong_recipe, wrong_recipe_path)
            with self.assertRaisesRegex(ValueError, "recipe mismatch"):
                inspect_safe_velocity_checkpoint(wrong_recipe_path)

            wrong_step = self._checkpoint_payload(actor, 4)
            wrong_step["infos"]["env_state"]["common_step_counter"] = 0
            wrong_step_path = root / "model_4.pt"
            torch.save(wrong_step, wrong_step_path)
            with self.assertRaisesRegex(ValueError, "iteration/step mismatch"):
                inspect_safe_velocity_checkpoint(wrong_step_path)

            corrupt_buffer = self._checkpoint_payload(actor, 5)
            corrupt_state = dict(corrupt_buffer["actor_state_dict"])
            corrupt_state["distribution.inward_lower_bound"] = torch.full((18,), 100.0)
            corrupt_buffer["actor_state_dict"] = corrupt_state
            corrupt_buffer_path = root / "model_5.pt"
            torch.save(corrupt_buffer, corrupt_buffer_path)
            with self.assertRaisesRegex(ValueError, "inward_lower_bound"):
                inspect_safe_velocity_checkpoint(corrupt_buffer_path)


if __name__ == "__main__":
    unittest.main()
