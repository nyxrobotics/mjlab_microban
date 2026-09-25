# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Contract tests for the audited v8j locomotion-prior teacher."""

from __future__ import annotations

import hashlib
import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from mjlab.envs.mdp.actions import JointPositionAction

from mjlab_microban.tasks.microban_locomotion_prior import (
    MICROBAN_LOCOMOTION_PRIOR_COMMAND_WIDTH,
    MICROBAN_LOCOMOTION_PRIOR_END_FRAME,
    MICROBAN_LOCOMOTION_PRIOR_FADE_END_STEP,
    MICROBAN_LOCOMOTION_PRIOR_FORWARD_VELOCITY_RANGE_M_S,
    MICROBAN_LOCOMOTION_PRIOR_FULL_BLEND_END_STEP,
    MICROBAN_LOCOMOTION_PRIOR_FULL_TELEPORT_END_STEP,
    MICROBAN_LOCOMOTION_PRIOR_LAUNCH_STEPS,
    MICROBAN_LOCOMOTION_PRIOR_LEAD_FRAMES,
    MICROBAN_LOCOMOTION_PRIOR_LEG_JOINT_NAMES,
    MICROBAN_LOCOMOTION_PRIOR_MAX_SOURCE_JOINT_SPEED_RAD_S,
    MICROBAN_LOCOMOTION_PRIOR_NOMINAL_FORWARD_VELOCITY_M_S,
    MICROBAN_LOCOMOTION_PRIOR_PATH,
    MICROBAN_LOCOMOTION_PRIOR_SHA256,
    MICROBAN_LOCOMOTION_PRIOR_START_FRAME,
    MICROBAN_LOCOMOTION_PRIOR_TELEPORT_FADE_END_STEP,
    LocomotionPriorCommand,
    _load_locomotion_prior,
    locomotion_prior_action_target_error_exp,
    locomotion_prior_blend,
    locomotion_prior_clip_finished,
    locomotion_prior_joint_position_error_exp,
    locomotion_prior_teleport_probability,
)
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_TELEOP_ACTION_JOINT_NAMES,
    MICROBAN_TELEOP_OBSERVATION_SCHEMA,
    MICROBAN_TELEOP_OBSERVATION_WIDTH,
)
from mjlab_microban.tasks.microban_teleop_env_cfg import (
    MICROBAN_TELEOP_FORWARD_ONLY_WIDE_PROBABILITIES,
    MICROBAN_TELEOP_FORWARD_ONLY_WIDE_RANGES,
    MICROBAN_TELEOP_INITIAL_SIGNED_AXIS_RANGES,
    MICROBAN_TELEOP_ISOLATED_AXIS_PROBABILITIES,
    MICROBAN_TELEOP_LOW_SIGNED_AXIS_RANGES,
    MICROBAN_TELEOP_PLANAR_AXIS_PROBABILITIES,
    MICROBAN_TELEOP_PLANAR_AXIS_RANGES,
    MICROBAN_TELEOP_PRIOR_INITIAL_AXIS_PROBABILITIES,
    MICROBAN_TELEOP_PRIOR_SIGNED_AXIS_RANGES,
    MICROBAN_TELEOP_SAGITTAL_AXIS_PROBABILITIES,
    MICROBAN_TELEOP_SAGITTAL_AXIS_RANGES,
    make_microban_teleop_env_cfg,
)
from mjlab_microban.tasks.microban_teleop_provenance import (
    collect_training_source_manifest,
)


def _prior_term(num_envs: int = 2) -> LocomotionPriorCommand:
    term = object.__new__(LocomotionPriorCommand)
    term._env = SimpleNamespace(
        num_envs=num_envs, common_step_counter=0, device="cpu", step_dt=0.02
    )
    term.cfg = SimpleNamespace(enabled=True)
    term.arrays = _load_locomotion_prior(
        MICROBAN_LOCOMOTION_PRIOR_PATH,
        expected_sha256=MICROBAN_LOCOMOTION_PRIOR_SHA256,
        device="cpu",
    )
    term.phase = torch.full((num_envs,), float(MICROBAN_LOCOMOTION_PRIOR_START_FRAME))
    term.phase_rate = torch.ones(num_envs)
    term.eligible = torch.ones(num_envs, dtype=torch.bool)
    term.finished = torch.zeros(num_envs, dtype=torch.bool)
    term.teleported = torch.ones(num_envs, dtype=torch.bool)
    term.launching = torch.zeros(num_envs, dtype=torch.bool)
    term.launch_step = torch.zeros(num_envs, dtype=torch.long)
    term.launch_start_joint_pos = torch.zeros(num_envs, 12)
    term.launch_dt = 0.02
    term.leg_joint_ids = torch.arange(12, dtype=torch.long)
    term.leg_action_ids = torch.tensor(
        [
            MICROBAN_TELEOP_ACTION_JOINT_NAMES.index(name)
            for name in MICROBAN_LOCOMOTION_PRIOR_LEG_JOINT_NAMES
        ],
        dtype=torch.long,
    )
    term._skip_next_advance = torch.zeros(num_envs, dtype=torch.bool)
    return term


class LocomotionPriorArtifactTest(unittest.TestCase):
    def test_vendored_artifact_is_exact_and_has_audited_shapes(self) -> None:
        digest = hashlib.sha256(MICROBAN_LOCOMOTION_PRIOR_PATH.read_bytes()).hexdigest()
        self.assertEqual(digest, MICROBAN_LOCOMOTION_PRIOR_SHA256)
        arrays = _load_locomotion_prior(
            MICROBAN_LOCOMOTION_PRIOR_PATH,
            expected_sha256=MICROBAN_LOCOMOTION_PRIOR_SHA256,
            device="cpu",
        )
        self.assertEqual(arrays.joint_pos.shape, (268, 12))
        self.assertEqual(arrays.joint_vel.shape, (268, 12))
        self.assertEqual(arrays.root_pos.shape, (268, 3))
        # The runtime phase is deliberately the absolute source frame number.
        # Guard against accidentally slicing the returned tensors to 159 frames
        # while leaving phases in the absolute 109..267 range.
        with np.load(MICROBAN_LOCOMOTION_PRIOR_PATH, allow_pickle=False) as archive:
            source_joint_names = tuple(archive["joint_names"].tolist())
            source_leg_ids = [
                source_joint_names.index(name)
                for name in MICROBAN_LOCOMOTION_PRIOR_LEG_JOINT_NAMES
            ]
            source_max_qdot = float(
                np.max(np.abs(archive["joint_vel"][:, source_leg_ids]))
            )
            consumed_source_max_qdot = float(
                np.max(
                    np.abs(
                        archive["joint_vel"][
                            MICROBAN_LOCOMOTION_PRIOR_START_FRAME : (
                                MICROBAN_LOCOMOTION_PRIOR_END_FRAME + 1
                            ),
                            source_leg_ids,
                        ]
                    )
                )
            )
            for frame in (
                MICROBAN_LOCOMOTION_PRIOR_START_FRAME,
                MICROBAN_LOCOMOTION_PRIOR_END_FRAME,
            ):
                torch.testing.assert_close(
                    arrays.joint_pos[frame],
                    torch.from_numpy(archive["joint_pos"][frame, source_leg_ids]),
                )
                torch.testing.assert_close(
                    arrays.root_pos[frame],
                    torch.from_numpy(archive["body_pos_w"][frame, 0]),
                )
        rate = (
            MICROBAN_LOCOMOTION_PRIOR_FORWARD_VELOCITY_RANGE_M_S[1]
            / MICROBAN_LOCOMOTION_PRIOR_NOMINAL_FORWARD_VELOCITY_M_S
        )
        loaded_consumed_max_qdot = (
            arrays.joint_vel[
                MICROBAN_LOCOMOTION_PRIOR_START_FRAME : (
                    MICROBAN_LOCOMOTION_PRIOR_END_FRAME + 1
                )
            ]
            .abs()
            .max()
            .item()
        )
        self.assertAlmostEqual(
            loaded_consumed_max_qdot, consumed_source_max_qdot, places=7
        )
        self.assertLessEqual(
            source_max_qdot,
            MICROBAN_LOCOMOTION_PRIOR_MAX_SOURCE_JOINT_SPEED_RAD_S,
        )
        self.assertLessEqual(
            loaded_consumed_max_qdot * rate,
            MICROBAN_LOCOMOTION_PRIOR_MAX_SOURCE_JOINT_SPEED_RAD_S * rate,
        )

    def test_training_source_manifest_hashes_the_binary_prior(self) -> None:
        manifest = collect_training_source_manifest()
        relative = "data/motions/microban_twist2_walk004_locomotion_prior.npz"
        self.assertEqual(manifest["files"][relative], MICROBAN_LOCOMOTION_PRIOR_SHA256)
        lock_path = MICROBAN_LOCOMOTION_PRIOR_PATH.parents[2] / "uv.lock"
        self.assertEqual(
            manifest["files"]["uv.lock"],
            hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        )


class LocomotionPriorScheduleTest(unittest.TestCase):
    def test_blend_is_full_then_linear_then_exact_zero(self) -> None:
        self.assertEqual(locomotion_prior_blend(0), 1.0)
        self.assertEqual(
            locomotion_prior_blend(MICROBAN_LOCOMOTION_PRIOR_FULL_BLEND_END_STEP),
            1.0,
        )
        midpoint = (
            MICROBAN_LOCOMOTION_PRIOR_FULL_BLEND_END_STEP
            + MICROBAN_LOCOMOTION_PRIOR_FADE_END_STEP
        ) // 2
        self.assertEqual(locomotion_prior_blend(midpoint), 0.5)
        self.assertEqual(
            locomotion_prior_blend(MICROBAN_LOCOMOTION_PRIOR_FADE_END_STEP),
            0.0,
        )
        self.assertEqual(
            locomotion_prior_blend(MICROBAN_LOCOMOTION_PRIOR_FADE_END_STEP + 1),
            0.0,
        )

    def test_teleport_fades_before_imitation_and_reaches_exact_zero(self) -> None:
        self.assertEqual(locomotion_prior_teleport_probability(0), 1.0)
        self.assertEqual(
            locomotion_prior_teleport_probability(
                MICROBAN_LOCOMOTION_PRIOR_FULL_TELEPORT_END_STEP
            ),
            1.0,
        )
        midpoint = (
            MICROBAN_LOCOMOTION_PRIOR_FULL_TELEPORT_END_STEP
            + MICROBAN_LOCOMOTION_PRIOR_TELEPORT_FADE_END_STEP
        ) // 2
        self.assertEqual(locomotion_prior_teleport_probability(midpoint), 0.5)
        self.assertEqual(
            locomotion_prior_teleport_probability(
                MICROBAN_LOCOMOTION_PRIOR_TELEPORT_FADE_END_STEP
            ),
            0.0,
        )
        self.assertEqual(
            locomotion_prior_blend(MICROBAN_LOCOMOTION_PRIOR_TELEPORT_FADE_END_STEP),
            0.0,
        )

    def test_command_is_39_wide_non_looping_and_play_can_be_exact_zero(self) -> None:
        term = _prior_term()
        term.eligible[1] = False
        command = term.command
        self.assertEqual(command.shape, (2, MICROBAN_LOCOMOTION_PRIOR_COMMAND_WIDTH))
        self.assertEqual(command[0, 0].item(), 1.0)
        torch.testing.assert_close(command[1], torch.zeros(39))

        term.phase[0] = MICROBAN_LOCOMOTION_PRIOR_END_FRAME - 0.25
        term.phase_rate[0] = 1.0
        term.compute(0.02)
        self.assertEqual(term.phase[0].item(), MICROBAN_LOCOMOTION_PRIOR_END_FRAME)
        self.assertTrue(term.finished[0].item())
        self.assertTrue(
            locomotion_prior_clip_finished(
                SimpleNamespace(
                    command_manager=SimpleNamespace(get_term=lambda name: term)
                ),
                "locomotion_prior",
            )[0].item()
        )
        term.compute(0.02)
        self.assertEqual(term.phase[0].item(), MICROBAN_LOCOMOTION_PRIOR_END_FRAME)

        term.cfg.enabled = False
        torch.testing.assert_close(term.command, torch.zeros(2, 39))
        self.assertFalse(
            locomotion_prior_clip_finished(
                SimpleNamespace(
                    command_manager=SimpleNamespace(get_term=lambda name: term)
                ),
                "locomotion_prior",
            ).any()
        )

    def test_phase_advance_aligns_with_explicit_and_automatic_reset(self) -> None:
        explicit = _prior_term(num_envs=1)
        explicit._skip_next_advance.fill_(True)
        explicit.compute(0.0)
        self.assertEqual(explicit.phase.item(), MICROBAN_LOCOMOTION_PRIOR_START_FRAME)
        self.assertFalse(explicit._skip_next_advance.item())
        explicit.compute(0.02)
        self.assertEqual(
            explicit.phase.item(), MICROBAN_LOCOMOTION_PRIOR_START_FRAME + 1
        )

        automatic = _prior_term(num_envs=2)
        automatic._skip_next_advance[0] = True
        automatic.compute(0.02)
        torch.testing.assert_close(
            automatic.phase,
            torch.tensor(
                [
                    MICROBAN_LOCOMOTION_PRIOR_START_FRAME,
                    MICROBAN_LOCOMOTION_PRIOR_START_FRAME + 1,
                ],
                dtype=torch.float32,
            ),
        )
        self.assertFalse(automatic._skip_next_advance.any())
        automatic.compute(0.02)
        torch.testing.assert_close(
            automatic.phase,
            torch.tensor(
                [
                    MICROBAN_LOCOMOTION_PRIOR_START_FRAME + 1,
                    MICROBAN_LOCOMOTION_PRIOR_START_FRAME + 2,
                ],
                dtype=torch.float32,
            ),
        )


class LocomotionPriorResetTest(unittest.TestCase):
    def test_reset_latches_only_pure_forward_and_writes_only_leg_joints(self) -> None:
        term = _prior_term(num_envs=3)
        term.cfg = SimpleNamespace(
            enabled=True,
            velocity_command_name="twist",
            forward_velocity_range_m_s=(0.06, 0.11),
        )
        term.leg_joint_ids = torch.tensor([6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20])
        term.robot = SimpleNamespace(
            write_joint_state_to_sim=Mock(),
            set_joint_position_target=Mock(),
            write_root_state_to_sim=Mock(),
        )
        twist = torch.tensor(
            [
                [0.06, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.08, 0.0, 0.2],
            ]
        )
        term._env = SimpleNamespace(
            num_envs=3,
            common_step_counter=0,
            device="cpu",
            command_manager=SimpleNamespace(get_command=lambda name: twist),
            scene=SimpleNamespace(
                env_origins=torch.tensor(
                    [[1.0, 2.0, 0.0], [3.0, 4.0, 0.0], [5.0, 6.0, 0.0]]
                )
            ),
        )
        term.command_counter = torch.ones(3, dtype=torch.long)
        term.time_left = torch.zeros(3)

        term.reset(torch.arange(3))

        self.assertEqual(term.eligible.tolist(), [True, False, False])
        self.assertAlmostEqual(
            term.phase_rate[0].item(),
            0.06 / MICROBAN_LOCOMOTION_PRIOR_NOMINAL_FORWARD_VELOCITY_M_S,
            places=6,
        )
        joint_call = term.robot.write_joint_state_to_sim.call_args
        self.assertEqual(joint_call.kwargs["env_ids"].tolist(), [0])
        self.assertEqual(
            joint_call.kwargs["joint_ids"].tolist(), term.leg_joint_ids.tolist()
        )
        self.assertEqual(joint_call.args[0].shape, (1, 12))
        self.assertEqual(joint_call.args[1].shape, (1, 12))
        # The subset contains all and only the twelve hips/knees/ankles.  Arms
        # (indices 3..5/12..14) and HMD joints (0..2) are never overwritten.
        self.assertEqual(len(MICROBAN_LOCOMOTION_PRIOR_LEG_JOINT_NAMES), 12)
        self.assertTrue(
            set(term.leg_joint_ids.tolist()).isdisjoint({0, 1, 2, 3, 4, 5, 12, 13, 14})
        )
        root_state = term.robot.write_root_state_to_sim.call_args.args[0]
        torch.testing.assert_close(root_state[0, :2], torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(
            root_state[0, 3:7], torch.tensor([1.0, 0.0, 0.0, 0.0])
        )

    def test_nonteleported_reset_uses_smooth_launch_then_enters_clip(self) -> None:
        term = _prior_term(num_envs=1)
        term.cfg = SimpleNamespace(
            enabled=True,
            velocity_command_name="twist",
            forward_velocity_range_m_s=(0.06, 0.11),
        )
        start = torch.linspace(-0.12, 0.10, 12).unsqueeze(0)
        term.robot = SimpleNamespace(
            data=SimpleNamespace(joint_pos=start.clone()),
            write_joint_state_to_sim=Mock(),
            set_joint_position_target=Mock(),
            write_root_state_to_sim=Mock(),
        )
        twist = torch.tensor(
            [[MICROBAN_LOCOMOTION_PRIOR_NOMINAL_FORWARD_VELOCITY_M_S, 0.0, 0.0]]
        )
        term._env = SimpleNamespace(
            num_envs=1,
            common_step_counter=(MICROBAN_LOCOMOTION_PRIOR_TELEPORT_FADE_END_STEP - 1),
            device="cpu",
            step_dt=0.02,
            command_manager=SimpleNamespace(get_command=lambda name: twist),
            scene=SimpleNamespace(env_origins=torch.zeros(1, 3)),
        )
        term.command_counter = torch.ones(1, dtype=torch.long)
        term.time_left = torch.zeros(1)

        with patch(
            "mjlab_microban.tasks.microban_locomotion_prior.torch.rand",
            return_value=torch.ones(1),
        ):
            term.reset(torch.tensor([0]))

        self.assertFalse(term.teleported.item())
        self.assertTrue(term.launching.item())
        term.robot.write_joint_state_to_sim.assert_not_called()
        torch.testing.assert_close(term.reference_joint_pos, start)
        frame_109 = term.arrays.joint_pos[MICROBAN_LOCOMOTION_PRIOR_START_FRAME]
        expected_lead_progress = (
            MICROBAN_LOCOMOTION_PRIOR_LEAD_FRAMES
            / MICROBAN_LOCOMOTION_PRIOR_LAUNCH_STEPS
        )
        expected_smoothstep = expected_lead_progress**2 * (
            3.0 - 2.0 * expected_lead_progress
        )
        torch.testing.assert_close(
            term.lead_joint_pos,
            start + expected_smoothstep * (frame_109 - start),
        )

        # An automatic reset is followed by a positive-dt command update in
        # the same env.step.  That update consumes the marker without using up
        # the first of the twenty launch steps.
        term.compute(0.02)
        self.assertTrue(term.launching.item())
        self.assertEqual(term.launch_step.item(), 0)
        for _ in range(MICROBAN_LOCOMOTION_PRIOR_LAUNCH_STEPS):
            term.compute(0.02)
        self.assertFalse(term.launching.item())
        self.assertEqual(
            term.launch_step.item(), MICROBAN_LOCOMOTION_PRIOR_LAUNCH_STEPS
        )
        self.assertEqual(term.phase.item(), MICROBAN_LOCOMOTION_PRIOR_START_FRAME)
        torch.testing.assert_close(term.reference_joint_pos, frame_109.unsqueeze(0))
        term.compute(0.02)
        self.assertEqual(term.phase.item(), MICROBAN_LOCOMOTION_PRIOR_START_FRAME + 1)

    def test_launch_lead_joins_clip_by_one_policy_step_at_all_rates(self) -> None:
        minimum, maximum = MICROBAN_LOCOMOTION_PRIOR_FORWARD_VELOCITY_RANGE_M_S
        rates = (
            minimum / MICROBAN_LOCOMOTION_PRIOR_NOMINAL_FORWARD_VELOCITY_M_S,
            1.0,
            maximum / MICROBAN_LOCOMOTION_PRIOR_NOMINAL_FORWARD_VELOCITY_M_S,
        )
        for rate in rates:
            with self.subTest(rate=rate):
                term = _prior_term(num_envs=1)
                term.phase_rate.fill_(rate)
                term.teleported.zero_()
                term.launching.fill_(True)
                term.launch_step.fill_(MICROBAN_LOCOMOTION_PRIOR_LAUNCH_STEPS - 1)
                term.launch_start_joint_pos.copy_(
                    torch.linspace(-0.12, 0.10, 12).unsqueeze(0)
                )

                before_join = term.lead_joint_pos.clone()
                expected_before = term._interpolate(
                    term.arrays.joint_pos,
                    torch.tensor(
                        [
                            MICROBAN_LOCOMOTION_PRIOR_START_FRAME
                            + MICROBAN_LOCOMOTION_PRIOR_LEAD_FRAMES
                            - rate
                        ]
                    ),
                )
                torch.testing.assert_close(before_join, expected_before)

                term.compute(0.02)
                self.assertFalse(term.launching.item())
                after_join = term.lead_joint_pos
                expected_after = term._interpolate(
                    term.arrays.joint_pos,
                    torch.tensor(
                        [
                            MICROBAN_LOCOMOTION_PRIOR_START_FRAME
                            + MICROBAN_LOCOMOTION_PRIOR_LEAD_FRAMES
                        ]
                    ),
                )
                torch.testing.assert_close(after_join, expected_after)

    def test_partial_teleport_selects_only_eligible_forward_rows(self) -> None:
        term = _prior_term(num_envs=5)
        term.cfg = SimpleNamespace(
            enabled=True,
            velocity_command_name="twist",
            forward_velocity_range_m_s=(0.06, 0.11),
        )
        term.robot = SimpleNamespace(
            data=SimpleNamespace(joint_pos=torch.zeros(5, 21)),
            write_joint_state_to_sim=Mock(),
            set_joint_position_target=Mock(),
            write_root_state_to_sim=Mock(),
        )
        twist = torch.tensor(
            [
                [0.06, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.08, 0.0, 0.0],
                [0.11, 0.0, 0.0],
                [0.08, 0.1, 0.0],
            ]
        )
        midpoint = (
            MICROBAN_LOCOMOTION_PRIOR_FULL_TELEPORT_END_STEP
            + MICROBAN_LOCOMOTION_PRIOR_TELEPORT_FADE_END_STEP
        ) // 2
        term._env = SimpleNamespace(
            num_envs=5,
            common_step_counter=midpoint,
            device="cpu",
            step_dt=0.02,
            command_manager=SimpleNamespace(get_command=lambda name: twist),
            scene=SimpleNamespace(env_origins=torch.zeros(5, 3)),
        )
        term.command_counter = torch.ones(5, dtype=torch.long)
        term.time_left = torch.zeros(5)

        # p=0.5 at the midpoint.  Draws correspond only to eligible rows
        # [0, 2, 3], proving that non-forward rows cannot leak into either set.
        with patch(
            "mjlab_microban.tasks.microban_locomotion_prior.torch.rand",
            return_value=torch.tensor([0.1, 0.9, 0.2]),
        ):
            term.reset(torch.arange(5))

        self.assertEqual(term.eligible.tolist(), [True, False, True, True, False])
        self.assertEqual(term.teleported.tolist(), [True, False, False, True, False])
        self.assertEqual(term.launching.tolist(), [False, False, True, False, False])
        written_ids = term.robot.write_joint_state_to_sim.call_args.kwargs["env_ids"]
        self.assertEqual(written_ids.tolist(), [0, 3])


class LocomotionPriorRewardTest(unittest.TestCase):
    def _reward_env(self, reference_error: float) -> SimpleNamespace:
        term = _prior_term(num_envs=1)
        term.arrays.joint_pos[MICROBAN_LOCOMOTION_PRIOR_START_FRAME:] = reference_error
        term.robot = SimpleNamespace(data=SimpleNamespace(joint_pos=torch.zeros(1, 12)))

        action = object.__new__(JointPositionAction)
        action._raw_actions = torch.zeros(1, 18)
        action._scale = 1.0
        action._offset = torch.zeros(1, 18)
        action._clip = torch.stack((-torch.ones(1, 18), torch.ones(1, 18)), dim=-1)
        action._target_names = list(MICROBAN_TELEOP_ACTION_JOINT_NAMES)
        action.cfg = SimpleNamespace(clip={".*": (-1.0, 1.0)})
        env = SimpleNamespace(
            command_manager=SimpleNamespace(get_term=lambda name: term),
            action_manager=SimpleNamespace(get_term=lambda name: action),
        )
        return env

    def test_rewards_use_per_joint_mean_square_not_sum(self) -> None:
        env = self._reward_env(0.15)
        expected = torch.tensor([math.exp(-1.0)])
        torch.testing.assert_close(
            locomotion_prior_action_target_error_exp(
                env, "locomotion_prior", "joint_pos", 0.15
            ),
            expected,
        )
        torch.testing.assert_close(
            locomotion_prior_joint_position_error_exp(env, "locomotion_prior", 0.15),
            expected,
        )
        self.assertGreater(expected.item(), math.exp(-12.0) * 1000.0)

    def test_rewards_are_masked_by_eligibility_and_blend(self) -> None:
        env = self._reward_env(0.0)
        term = env.command_manager.get_term("locomotion_prior")
        term.eligible.zero_()
        self.assertEqual(
            locomotion_prior_joint_position_error_exp(
                env, "locomotion_prior", 0.15
            ).item(),
            0.0,
        )
        term.eligible.fill_(True)
        term._env.common_step_counter = MICROBAN_LOCOMOTION_PRIOR_FADE_END_STEP
        self.assertEqual(
            locomotion_prior_action_target_error_exp(
                env, "locomotion_prior", "joint_pos", 0.15
            ).item(),
            0.0,
        )


class LocomotionPriorConfigurationTest(unittest.TestCase):
    def test_actor_contract_stays_byte_compatible_and_prior_is_critic_only(
        self,
    ) -> None:
        train = make_microban_teleop_env_cfg(play=False)
        play = make_microban_teleop_env_cfg(play=True)
        self.assertEqual(
            sum(width for _, width in MICROBAN_TELEOP_OBSERVATION_SCHEMA), 83
        )
        self.assertEqual(MICROBAN_TELEOP_OBSERVATION_WIDTH, 83)
        self.assertNotIn("locomotion_prior", train.observations["actor"].terms)
        self.assertIn("locomotion_prior", train.observations["critic"].terms)
        self.assertNotIn("locomotion_prior", play.observations["actor"].terms)
        self.assertIn("locomotion_prior", play.observations["critic"].terms)
        self.assertFalse(train.commands["locomotion_prior"].enabled)
        self.assertFalse(play.commands["locomotion_prior"].enabled)
        self.assertEqual(train.rewards["locomotion_prior_action_target"].weight, 0.0)
        self.assertEqual(
            train.rewards["locomotion_prior_joint_position"].weight, 0.0
        )
        self.assertNotIn("locomotion_prior_clip_finished", train.terminations)
        self.assertEqual(set(train.rewards), set(play.rewards))
        self.assertEqual(set(train.terminations), set(play.terminations))

    def test_v11_acquisition_curriculum_stages_match_audit(self) -> None:
        cfg = make_microban_teleop_env_cfg(play=False)
        twist = cfg.commands["twist"]
        self.assertEqual(
            twist.signed_axis_probabilities,
            MICROBAN_TELEOP_PRIOR_INITIAL_AXIS_PROBABILITIES,
        )
        self.assertEqual(
            twist.signed_axis_ranges, MICROBAN_TELEOP_PRIOR_SIGNED_AXIS_RANGES
        )
        self.assertEqual(sum(twist.signed_axis_probabilities.values()), 1.0)
        stages = cfg.curriculum["staged_curriculum"].params["stages"]
        self.assertEqual(
            [stage["step"] // 24 for stage in stages[:5]],
            [400, 900, 1500, 2200, 3000],
        )
        self.assertEqual(
            MICROBAN_TELEOP_PRIOR_SIGNED_AXIS_RANGES["forward"], (0.06, 0.11)
        )
        self.assertEqual(
            MICROBAN_TELEOP_INITIAL_SIGNED_AXIS_RANGES["forward"], (0.25, 0.40)
        )
        self.assertEqual(
            MICROBAN_TELEOP_FORWARD_ONLY_WIDE_RANGES["forward"], (0.08, 0.16)
        )
        self.assertEqual(
            MICROBAN_TELEOP_FORWARD_ONLY_WIDE_PROBABILITIES["forward"], 0.90
        )
        self.assertEqual(
            MICROBAN_TELEOP_SAGITTAL_AXIS_RANGES["backward"], (-0.15, -0.04)
        )
        self.assertEqual(MICROBAN_TELEOP_SAGITTAL_AXIS_PROBABILITIES["backward"], 0.30)
        self.assertEqual(
            MICROBAN_TELEOP_PLANAR_AXIS_RANGES["lateral_left"], (0.06, 0.15)
        )
        self.assertEqual(
            MICROBAN_TELEOP_PLANAR_AXIS_PROBABILITIES["lateral_left"], 0.15
        )
        self.assertEqual(MICROBAN_TELEOP_LOW_SIGNED_AXIS_RANGES["yaw_left"], (0.4, 1.2))
        self.assertEqual(MICROBAN_TELEOP_ISOLATED_AXIS_PROBABILITIES["forward"], 0.15)


if __name__ == "__main__":
    unittest.main()
