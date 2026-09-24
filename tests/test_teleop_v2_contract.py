# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Focused tests for the Microban teleop-v2 training/deployment contract."""

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

from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_TELEOP_ACTION_JOINT_NAMES,
    MICROBAN_TELEOP_PREVIOUS_ACTION_SEMANTICS,
    MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION,
    MicrobanTeleopOnPolicyRunner,
    TeleopCheckpointContract,
    _command_target_bounds,
    validate_teleop_checkpoint_contract,
)
from mjlab_microban.tasks.microban_teleop_env_cfg import (
    MICROBAN_TELEOP_ANGULAR_TRACKING_STD_RAD_S,
    MICROBAN_TELEOP_FINAL_VELOCITY_ENVELOPE,
    MICROBAN_TELEOP_HAND_TRACKING_STD_M,
    MICROBAN_TELEOP_LINEAR_TRACKING_STD_M_S,
    MicrobanTeleopRlCfg,
    make_microban_teleop_env_cfg,
    microban_teleop_initial_action_std,
)
from mjlab_microban.tasks.microban_teleop_mdp import (
    MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M,
    PerJointGaussianDistribution,
    ResetFixedFootTargetCommand,
    effective_action_after_target_clip,
    normalized_target_clip_excess_huber,
    raw_action_l2,
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
    def test_effective_action_and_normalized_huber_penalty(self) -> None:
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
        expected_penalty = torch.nn.functional.smooth_l1_loss(
            normalized,
            torch.zeros_like(normalized),
            beta=0.1,
            reduction="none",
        ).mean(dim=-1)
        torch.testing.assert_close(
            normalized_target_clip_excess_huber(env), expected_penalty
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
            normalized_target_clip_excess_huber(env), torch.zeros(1)
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
        self.assertEqual(cfg.rewards["target_clip_excess"].weight, -0.5)
        self.assertEqual(cfg.rewards["raw_action_l2"].weight, -0.002)
        self.assertEqual(cfg.rewards["action_rate_l2"].weight, -0.02)
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
        self.assertIs(
            MicrobanTeleopRlCfg.actor.distribution_cfg["class_name"],
            PerJointGaussianDistribution,
        )
        self.assertEqual(
            tuple(MicrobanTeleopRlCfg.actor.distribution_cfg["init_std"]), std
        )

    def test_curriculum_finishes_at_runtime_envelope_and_both_feet(self) -> None:
        self.assertEqual(MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M, 0.0025)
        cfg = make_microban_teleop_env_cfg()
        stages = cfg.curriculum["staged_curriculum"].params["stages"]
        self.assertEqual(
            [stage["step"] for stage in stages],
            [1000 * 24, 2000 * 24, 3000 * 24, 4500 * 24, 6000 * 24],
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
            name: SimpleNamespace(weight=term.weight)
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
            (MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M, 0.016),
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

    def test_v2_marker_is_required_and_iteration_must_match_filename(self) -> None:
        valid = self._save(12, contract=True)
        parsed = validate_teleop_checkpoint_contract(valid)
        self.assertEqual(parsed.version, "2")
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

    def test_save_adds_v2_marker_and_legacy_runner_cannot_write(self) -> None:
        fresh = object.__new__(MicrobanTeleopOnPolicyRunner)
        fresh.loaded_checkpoint_contract = None
        with patch.object(MjlabOnPolicyRunner, "save") as base_save:
            fresh.save(str(self.root / "model_0.pt"), {"source": "test"})
        saved_infos = base_save.call_args.args[-1]
        self.assertEqual(saved_infos["microban_teleop_training_contract_version"], "2")
        self.assertEqual(
            saved_infos["previous_action_semantics"],
            MICROBAN_TELEOP_PREVIOUS_ACTION_SEMANTICS,
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
