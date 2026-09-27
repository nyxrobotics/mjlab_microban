# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Contract tests for get-up previous-action feedback."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from mjlab_microban.tasks.microban_getup_action import (
    GetupJointPositionAction,
    GetupJointPositionActionCfg,
    effective_getup_action_after_target_clip,
)
from mjlab_microban.tasks.microban_getup_env_cfg import (
    make_microban_getup_env_cfg,
)
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_TELEOP_ACTION_JOINT_NAMES,
)
from mjlab_microban.tasks.microban_tracking_env_cfg import MICROBAN_BODY_JOINT_SOFT_LIMITS


class GetupActionObservationContractTest(unittest.TestCase):
    def test_actor_and_critic_observe_effective_post_clip_action(self) -> None:
        cfg = make_microban_getup_env_cfg()

        for group_name in ("actor", "critic"):
            term = cfg.observations[group_name].terms["actions"]
            self.assertIs(term.func, effective_getup_action_after_target_clip)
            self.assertEqual(term.params, {"action_name": "joint_pos"})

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
        self.assertEqual(action_cfg.clip, dict(MICROBAN_BODY_JOINT_SOFT_LIMITS))
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
        action._has_processed_action = torch.zeros(1, 1, dtype=torch.bool)
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
        torch.testing.assert_close(
            effective_getup_action_after_target_clip(env), torch.zeros_like(raw)
        )
        torch.testing.assert_close(target_all[:, -3:], measured_all[:, -3:])
        measured_all[:, -3:] += 0.05
        action.process_actions(raw)
        # Neck joints are held at whatever's currently measured, every tick.
        torch.testing.assert_close(target_all[:, -3:], measured_all[:, -3:])

        # No slew: the observed target is exactly the clipped target, in the
        # very same tick the raw action arrives.
        expected_target = torch.clamp(
            raw * action_cfg.scale + default_pose,
            min=lower,
            max=upper,
        )
        expected_effective = (
            expected_target - default_pose
        ) / action_cfg.scale
        effective = effective_getup_action_after_target_clip(env)
        torch.testing.assert_close(effective, expected_effective)
        runtime_target = default_pose + effective * action_cfg.scale
        torch.testing.assert_close(runtime_target, expected_target)
        # raw spans [-4, 4], well past the +-1.57 clip, so this is a real check.
        self.assertTrue(bool(torch.any(raw.abs() > upper.abs()).item()))

        action.reset()
        torch.testing.assert_close(
            effective_getup_action_after_target_clip(env), torch.zeros_like(raw)
        )


if __name__ == "__main__":
    unittest.main()
