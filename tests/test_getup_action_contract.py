# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Contract tests for the get-up v4 action and previous-action feedback."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from mjlab_microban.tasks.microban_getup_action import (
    GetupJointPositionAction,
    GetupJointPositionActionCfg,
    raw_getup_action,
)
from mjlab_microban.tasks.microban_getup_env_cfg import (
    GETUP_ACTION_CLIP,
    GETUP_REWARD_SETS,
    make_microban_getup_env_cfg,
)
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_TELEOP_ACTION_JOINT_NAMES,
)


class GetupActionObservationContractTest(unittest.TestCase):
    def test_actor_and_critic_observe_raw_previous_action(self) -> None:
        for reward_set in GETUP_REWARD_SETS:
            cfg = make_microban_getup_env_cfg(reward_set=reward_set)
            for group_name in ("actor", "critic"):
                term = cfg.observations[group_name].terms["actions"]
                self.assertIs(term.func, raw_getup_action)
                self.assertEqual(term.params, {"action_name": "joint_pos"})
                self.assertEqual(term.delay_max_lag, 0)

    def test_no_clip_excess_penalty_pushes_or_imu_delay(self) -> None:
        # Every standing policy was trained without these (see the env
        # module docstring); the v4 runner rejects the penalty outright.
        for reward_set in GETUP_REWARD_SETS:
            cfg = make_microban_getup_env_cfg(reward_set=reward_set)
            self.assertNotIn("raw_target_clip_excess", cfg.rewards)
            self.assertNotIn("push_robot", cfg.events)
            for name in ("base_ang_vel", "projected_gravity"):
                self.assertEqual(cfg.observations["actor"].terms[name].delay_max_lag, 0)

    def test_unknown_reward_set_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            make_microban_getup_env_cfg(reward_set="nope")

    def test_current_clip_and_default_offset_are_runtime_reconstructible(self) -> None:
        cfg = make_microban_getup_env_cfg()
        action_cfg = cfg.actions["joint_pos"]

        # Pin the get-up action contract used by deployment: the network emits
        # deltas from the robot's default pose, while the configured limits clip
        # the resulting absolute targets rather than the raw deltas. No rate
        # limit on that target -- see microban_getup_action.py's module
        # docstring for why an earlier version of this had one and why that
        # was a bug (it modeled an unrelated part of the robot runtime).
        self.assertEqual(action_cfg.scale, 1.0)
        self.assertEqual(action_cfg.offset, 0.0)
        self.assertTrue(action_cfg.use_default_offset)
        self.assertEqual(action_cfg.clip, dict(GETUP_ACTION_CLIP))
        self.assertIsInstance(action_cfg, GetupJointPositionActionCfg)
        self.assertFalse(hasattr(action_cfg, "max_target_speed_rad_s"))

        default_joint_pos = cfg.scene.entities["robot"].init_state.joint_pos
        default_pose = torch.tensor(
            [[default_joint_pos[name] for name in MICROBAN_TELEOP_ACTION_JOINT_NAMES]],
            dtype=torch.float32,
        )
        raw = torch.linspace(-4.0, 4.0, default_pose.shape[-1]).unsqueeze(0)
        lower = torch.full_like(default_pose, -1.57)
        upper = torch.full_like(default_pose, 1.57)

        measured = torch.full_like(default_pose, 2.0)
        measured_all = torch.cat(
            (measured, torch.tensor([[0.1, 0.2, 0.3]], dtype=measured.dtype)), dim=-1
        )
        target_all = torch.zeros_like(measured_all)

        def set_joint_position_target(position, joint_ids, env_ids):
            target_all[env_ids, joint_ids] = position

        action = object.__new__(GetupJointPositionAction)
        action.cfg = action_cfg
        action._env = SimpleNamespace(num_envs=1, device="cpu", step_dt=0.02)
        action._entity = SimpleNamespace(
            data=SimpleNamespace(joint_pos=measured_all),
            set_joint_position_target=set_joint_position_target,
        )
        action._target_ids = torch.arange(default_pose.shape[-1])
        action._raw_actions = torch.zeros_like(raw)
        action._processed_actions = torch.zeros_like(raw)
        action._neck_target_ids = torch.arange(default_pose.shape[-1], measured_all.shape[-1])
        action._scale = action_cfg.scale
        # JointPositionAction replaces cfg.offset with default_joint_pos when
        # use_default_offset is true; mirror that initialized state here.
        action._offset = default_pose
        action._clip = torch.stack((lower, upper), dim=-1)
        env = SimpleNamespace(
            action_manager=SimpleNamespace(get_term=lambda name: action)
        )

        # The first actor observation must have zero previous action, even
        # though process_actions hasn't run yet.
        action.reset()
        torch.testing.assert_close(raw_getup_action(env), torch.zeros_like(raw))
        torch.testing.assert_close(target_all[:, -3:], measured_all[:, -3:])
        measured_all[:, -3:] += 0.05
        action.process_actions(raw)
        # Neck joints are held at whatever's currently measured, every tick.
        torch.testing.assert_close(target_all[:, -3:], measured_all[:, -3:])

        # No slew: the target is the clipped absolute target in the very same
        # tick, and the observation is the raw output itself, not the target.
        expected_target = torch.clamp(
            raw * action_cfg.scale + default_pose,
            min=lower,
            max=upper,
        )
        torch.testing.assert_close(action._processed_actions, expected_target)
        torch.testing.assert_close(raw_getup_action(env), raw)
        # raw spans [-4, 4], well past the +-1.57 clip, so this is a real check.
        self.assertTrue(bool(torch.any(raw.abs() > upper.abs()).item()))

        action.reset()
        torch.testing.assert_close(raw_getup_action(env), torch.zeros_like(raw))


if __name__ == "__main__":
    unittest.main()
