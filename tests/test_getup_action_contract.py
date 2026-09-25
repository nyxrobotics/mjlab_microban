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
from mjlab.envs.mdp.actions import JointPositionAction

from mjlab_microban.tasks.microban_getup_env_cfg import (
    make_microban_getup_env_cfg,
)
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_TELEOP_ACTION_JOINT_NAMES,
)
from mjlab_microban.tasks.microban_teleop_mdp import (
    effective_action_after_target_clip,
)


class GetupActionObservationContractTest(unittest.TestCase):
    def test_actor_and_critic_observe_effective_post_clip_action(self) -> None:
        cfg = make_microban_getup_env_cfg()

        for group_name in ("actor", "critic"):
            term = cfg.observations[group_name].terms["actions"]
            self.assertIs(term.func, effective_action_after_target_clip)
            self.assertEqual(term.params, {"action_name": "joint_pos"})

    def test_current_clip_and_default_offset_are_runtime_reconstructible(self) -> None:
        cfg = make_microban_getup_env_cfg()
        action_cfg = cfg.actions["joint_pos"]

        # Pin the get-up action contract used by deployment: the network emits
        # deltas from the robot's default pose, while the configured limits clip
        # the resulting absolute targets rather than the raw deltas.
        self.assertEqual(action_cfg.scale, 1.0)
        self.assertEqual(action_cfg.offset, 0.0)
        self.assertTrue(action_cfg.use_default_offset)
        self.assertEqual(action_cfg.clip, {r".*": (-1.57, 1.57)})

        default_joint_pos = cfg.scene.entities["robot"].init_state.joint_pos
        default_pose = torch.tensor(
            [[default_joint_pos[name] for name in MICROBAN_TELEOP_ACTION_JOINT_NAMES]],
            dtype=torch.float32,
        )
        raw = torch.linspace(-4.0, 4.0, default_pose.shape[-1]).unsqueeze(0)
        lower = torch.full_like(default_pose, -1.57)
        upper = torch.full_like(default_pose, 1.57)

        action = object.__new__(JointPositionAction)
        action._raw_actions = raw
        action._scale = action_cfg.scale
        # JointPositionAction replaces cfg.offset with default_joint_pos when
        # use_default_offset is true; mirror that initialized state here.
        action._offset = default_pose
        action._clip = torch.stack((lower, upper), dim=-1)
        action.cfg = action_cfg
        env = SimpleNamespace(
            action_manager=SimpleNamespace(get_term=lambda name: action)
        )

        effective = effective_action_after_target_clip(env)
        expected_target = torch.clamp(
            raw * action_cfg.scale + default_pose,
            min=lower,
            max=upper,
        )
        expected_effective = (
            expected_target - default_pose
        ) / action_cfg.scale
        torch.testing.assert_close(effective, expected_effective)

        # This is the exact arithmetic available to the Python robot runtime:
        # store the effective delta as previous-action feedback, then add the
        # ONNX default_joint_pos metadata to reproduce the commanded target.
        runtime_target = default_pose + effective * action_cfg.scale
        torch.testing.assert_close(runtime_target, expected_target)
        self.assertTrue(bool(torch.any(effective != raw).item()))


if __name__ == "__main__":
    unittest.main()
