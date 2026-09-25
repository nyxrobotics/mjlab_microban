# Copyright 2026 Marc Duclusaud
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the simulation-only live PICO bridge."""

from __future__ import annotations

import asyncio
import math
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from tensordict import TensorDict

from mjlab_microban.scripts.live_pico_teleop_sim import (
    AUDITED_LEGACY_WALK_SHA256,
    FOOT_INACTIVE_Z_MAX_M,
    FOOT_UPPER_M,
    HAND_UPPER_M,
    LivePicoSimulationPolicy,
    SimulationCommand,
    WebXrSimulationMapper,
    WebXrSimulationSource,
    _default_native_config,
    _default_teleop_root,
    _default_walk_checkpoint,
    _legacy_body_joint_indices,
    _legacy_walk_adapters,
    _load_native_classes,
    _NativeServerThread,
    _patch_command_observation,
    _patch_native_walk_observation,
    _runtime_task,
    _selected_checkpoint,
    _sha256,
    _walk_actor_observation,
    build_parser,
    command_for_simulation,
    scale_normalized_velocity,
    solve_hmd_neck_target,
)
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_HMD_JOINT_NAMES,
    MICROBAN_TELEOP_OBSERVATION_SCHEMA,
)
from mjlab_microban.tasks.microban_teleop_mdp import (
    MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M,
)


def _valid_command() -> dict[str, object]:
    return {
        "velocity": {"vx": 1.0, "vy": -0.5, "vtheta": 0.5},
        "active_moves": ["walk", "hmd_head"],
        "locomotion_policy": "pico_teleop",
        "head_orientation": {"roll": 0.1, "pitch": -0.2, "yaw": 0.3},
        "head_yaw_front": False,
        "foot_target": {
            "left": FOOT_UPPER_M,
            "right": (0.0, 0.0, 0.0),
        },
        "hand_target": {
            "left": HAND_UPPER_M,
            "right": (0.0, 0.0, 0.0),
        },
        "body_target_calibrated": True,
        "body_target_fresh": True,
    }


class SimulationCommandTests(unittest.TestCase):
    def test_floor_band_matches_current_training_contract(self) -> None:
        self.assertEqual(FOOT_INACTIVE_Z_MAX_M, 0.0025)
        self.assertEqual(
            FOOT_INACTIVE_Z_MAX_M,
            MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M,
        )

    def test_physical_velocity_scaling_matches_robot_runtime(self) -> None:
        self.assertEqual(
            scale_normalized_velocity({"vx": 1.0, "vy": -1.0, "vtheta": 1.0}),
            (0.7, -0.3, 1.5),
        )
        self.assertEqual(
            scale_normalized_velocity({"vx": -1.0, "vy": 0.0, "vtheta": -1.0}),
            (-0.5, 0.0, -1.5),
        )
        self.assertEqual(
            scale_normalized_velocity({"vx": 0.0, "vy": 0.0, "vtheta": 1.0}),
            (0.0, 0.0, 3.0),
        )

    def test_valid_hybrid_deadman_command_is_enabled(self) -> None:
        command = command_for_simulation(_valid_command())
        self.assertTrue(command.enabled)
        self.assertEqual(command.twist, (0.7, -0.15, 0.75))
        self.assertEqual(command.foot_target[0], FOOT_UPPER_M)
        self.assertEqual(command.hand_target[0], HAND_UPPER_M)
        self.assertEqual(command.hand_active, (True, True))

    def test_walk_policy_cannot_silently_run_hybrid_checkpoint(self) -> None:
        value = _valid_command()
        value["locomotion_policy"] = "walk"
        command = command_for_simulation(value)
        self.assertFalse(command.enabled)
        self.assertEqual(command.twist, (0.0, 0.0, 0.0))
        # HMD motion remains independent of body-policy selection.
        self.assertEqual(command.head_orientation, (0.1, -0.2, 0.3))

    def test_walk_policy_uses_legacy_actor_when_available(self) -> None:
        value = _valid_command()
        value["locomotion_policy"] = "walk"
        command = command_for_simulation(value, legacy_walk_available=True)
        self.assertTrue(command.enabled)
        self.assertEqual(command.twist, (0.7, -0.15, 0.75))
        self.assertEqual(command.hand_active, (False, False))

    def test_missing_or_out_of_range_body_target_fails_closed(self) -> None:
        for mutation in ("stale", "missing", "range"):
            with self.subTest(mutation=mutation):
                value = _valid_command()
                if mutation == "stale":
                    value["body_target_fresh"] = False
                elif mutation == "missing":
                    value["foot_target"] = None
                else:
                    value["hand_target"] = {
                        "left": (HAND_UPPER_M[0] + 1.0e-5, 0.0, 0.0),
                        "right": (0.0, 0.0, 0.0),
                    }
                command = command_for_simulation(value)
                self.assertFalse(command.enabled)
                self.assertEqual(command.twist, (0.0, 0.0, 0.0))
                self.assertIsNotNone(command.fault)

    def test_boolean_numeric_target_is_rejected(self) -> None:
        value = _valid_command()
        value["hand_target"] = {
            "left": (True, 0.0, 0.0),
            "right": (0.0, 0.0, 0.0),
        }
        self.assertFalse(command_for_simulation(value).enabled)

    def test_webxr_uses_zero_offsets_with_hands_inactive(self) -> None:
        source = WebXrSimulationSource()
        mapper = WebXrSimulationMapper(source)
        webxr_command = _valid_command()
        webxr_command["foot_target"] = None
        webxr_command["hand_target"] = None
        source.publish(webxr_command, 1_000_000_000)
        command = command_for_simulation(mapper.map_sample(source.read()))
        self.assertTrue(command.enabled)
        self.assertEqual(command.foot_target, ((0.0, 0.0, 0.0),) * 2)
        self.assertEqual(command.hand_target, ((0.0, 0.0, 0.0),) * 2)
        self.assertEqual(command.hand_active, (False, False))

    def test_malformed_hand_activity_fails_closed(self) -> None:
        value = _valid_command()
        value["hand_active"] = {"left": False, "right": 0}
        command = command_for_simulation(value)
        self.assertFalse(command.enabled)
        self.assertEqual(command.fault, "hand_active is malformed")

    def test_foot_floor_jitter_projects_to_exact_inactive_zero(self) -> None:
        value = _valid_command()
        value["foot_target"] = {
            "left": (0.02, -0.02, 0.0025),
            "right": (0.0, 0.0, 0.0),
        }
        command = command_for_simulation(value)
        self.assertTrue(command.enabled)
        self.assertEqual(command.foot_target[0], (0.0, 0.0, 0.0))

    def test_both_feet_require_subset_and_stationary_twist(self) -> None:
        for feet, velocity, enabled in (
            (
                {
                    "left": (0.008, 0.0, 0.016),
                    "right": (-0.008, 0.0, 0.016),
                },
                {"vx": 0.0, "vy": 0.0, "vtheta": 0.0},
                True,
            ),
            (
                {
                    "left": (0.009, 0.0, 0.01),
                    "right": (-0.004, 0.0, 0.01),
                },
                {"vx": 0.0, "vy": 0.0, "vtheta": 0.0},
                False,
            ),
            (
                {
                    "left": (0.004, 0.0, 0.01),
                    "right": (-0.004, 0.0, 0.01),
                },
                {"vx": 0.1, "vy": 0.0, "vtheta": 0.0},
                False,
            ),
            (
                {
                    "left": (0.004, 0.0, 0.01),
                    "right": (-0.004, 0.0, 0.01),
                },
                {"vx": 1.0e-7, "vy": 0.0, "vtheta": 0.0},
                False,
            ),
        ):
            with self.subTest(feet=feet, velocity=velocity):
                value = _valid_command()
                value["foot_target"] = feet
                value["velocity"] = velocity
                command = command_for_simulation(value)
                self.assertEqual(command.enabled, enabled)


class HmdSolverTests(unittest.TestCase):
    def test_identity_trunk_recovers_zxy_hmd_request(self) -> None:
        target = solve_hmd_neck_target(
            (0.1, -0.2, 0.3), (1.0, 0.0, 0.0, 0.0), yaw_front=False
        )
        self.assertIsNotNone(target)
        assert target is not None
        for actual, expected in zip(target, (0.3, 0.1, -0.2), strict=True):
            self.assertAlmostEqual(actual, expected, places=7)

    def test_right_trigger_forces_head_yaw_to_zero(self) -> None:
        target = solve_hmd_neck_target(
            (0.1, -0.2, 1.1), (1.0, 0.0, 0.0, 0.0), yaw_front=True
        )
        self.assertIsNotNone(target)
        assert target is not None
        self.assertEqual(target[0], 0.0)
        self.assertAlmostEqual(target[1], 0.1, places=7)
        self.assertAlmostEqual(target[2], -0.2, places=7)

    def test_trunk_roll_is_gravity_compensated(self) -> None:
        half = 0.05
        target = solve_hmd_neck_target(
            (0.0, 0.0, 0.0),
            (math.cos(half), math.sin(half), 0.0, 0.0),
            yaw_front=False,
        )
        self.assertIsNotNone(target)
        assert target is not None
        self.assertAlmostEqual(target[1], -0.1, places=7)

    def test_invalid_body_quaternion_holds_target(self) -> None:
        self.assertIsNone(
            solve_hmd_neck_target(
                (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), yaw_front=False
            )
        )


class WatchdogTests(unittest.TestCase):
    class _Source:
        strict_control_presence = True

        def __init__(self, sampled_at_ns: int) -> None:
            self.sampled_at_ns = sampled_at_ns

        def read(self) -> SimpleNamespace:
            return SimpleNamespace(sampled_at_ns=self.sampled_at_ns)

    class _Mapper:
        def __init__(self) -> None:
            self.reset_count = 0

        def map_sample(self, _frame: object) -> dict[str, object]:
            return _valid_command()

        def reset(self) -> None:
            self.reset_count += 1

    @staticmethod
    def _policy(sampled_at_ns: int, now_ns: int) -> LivePicoSimulationPolicy:
        policy = object.__new__(LivePicoSimulationPolicy)
        policy.source = WatchdogTests._Source(sampled_at_ns)
        policy.mapper = WatchdogTests._Mapper()
        policy.clock_ns = lambda: now_ns
        policy._previous_sampled_at_ns = None
        return policy

    def test_fresh_frame_reaches_mapper(self) -> None:
        policy = self._policy(1_000_000_000, 1_010_000_000)
        self.assertTrue(policy._read_command().enabled)
        self.assertEqual(policy.mapper.reset_count, 0)

    def test_aged_frame_disarms_mapper(self) -> None:
        policy = self._policy(1_000_000_000, 1_200_000_000)
        command = policy._read_command()
        self.assertFalse(command.enabled)
        self.assertEqual(policy.mapper.reset_count, 1)

    def test_host_gap_disarms_even_if_latest_frame_is_fresh(self) -> None:
        policy = self._policy(1_000_000_000, 1_000_000_000)
        self.assertTrue(policy._read_command().enabled)
        policy.source.sampled_at_ns = 1_200_000_000
        policy.clock_ns = lambda: 1_200_000_000
        command = policy._read_command()
        self.assertFalse(command.enabled)
        self.assertEqual(policy.mapper.reset_count, 1)

    def test_same_fresh_snapshot_may_be_read_twice(self) -> None:
        policy = self._policy(1_000_000_000, 1_010_000_000)
        self.assertTrue(policy._read_command().enabled)
        self.assertTrue(policy._read_command().enabled)
        self.assertEqual(policy.mapper.reset_count, 0)

    def test_legacy_only_forces_hybrid_selector_to_audited_walk_path(self) -> None:
        policy = self._policy(1_000_000_000, 1_010_000_000)
        policy.legacy_only = True
        command = policy._read_command()
        self.assertTrue(command.enabled)
        self.assertEqual(command.locomotion_policy, "walk")
        self.assertEqual(command.twist, (0.7, -0.15, 0.75))

    def test_legacy_only_ignores_x_before_native_mapper_body_gate(self) -> None:
        _source_class, mapper_class = _load_native_classes(_default_teleop_root())
        mapper = mapper_class()

        def frame(
            sampled_at_ns: int, trigger: float, head_orientation: tuple[float, ...]
        ) -> SimpleNamespace:
            controller_health = SimpleNamespace(fresh=True, valid=True)
            stale_body_health = SimpleNamespace(fresh=False, valid=False)
            return SimpleNamespace(
                sampled_at_ns=sampled_at_ns,
                head=SimpleNamespace(
                    pose=SimpleNamespace(orientation=head_orientation)
                ),
                left_controller=SimpleNamespace(
                    axis=(0.0, 1.0),
                    trigger=trigger,
                    primary_button=True,
                ),
                right_controller=SimpleNamespace(
                    axis=(0.5, 0.0),
                    trigger=0.0,
                ),
                controller_health=controller_health,
                body_health=stale_body_health,
                body=None,
                body_jumps=(),
            )

        frames = iter(
            (
                frame(1_000_000_000, 0.0, (0.0, 0.0, 0.0, 1.0)),
                frame(
                    1_020_000_000,
                    1.0,
                    (0.0, 0.0, math.sin(0.1), math.cos(0.1)),
                ),
            )
        )
        policy = object.__new__(LivePicoSimulationPolicy)
        policy.source = SimpleNamespace(read=lambda: next(frames))
        policy.mapper = mapper
        policy.clock_ns = lambda: 1_020_000_000
        policy._previous_sampled_at_ns = None
        policy.legacy_only = True

        self.assertFalse(policy._read_command().enabled)  # release-to-rearm
        command = policy._read_command()
        self.assertTrue(command.enabled)
        self.assertEqual(command.locomotion_policy, "walk")
        self.assertGreater(command.twist[0], 0.0)
        self.assertLess(command.twist[2], 0.0)
        self.assertTrue(any(abs(value) > 0.0 for value in command.head_orientation))
        self.assertEqual(mapper.locomotion_policy, "walk")


class ObservationPatchTests(unittest.TestCase):
    def test_only_live_command_slices_are_replaced(self) -> None:
        widths = dict(MICROBAN_TELEOP_OBSERVATION_SCHEMA)
        total = sum(widths.values())
        original = torch.arange(total, dtype=torch.float32).unsqueeze(0)
        values = {
            "twist": torch.full((1, widths["command"]), 101.0),
            "foot_target": torch.full((1, widths["foot_target"]), 102.0),
            "hand_target": torch.full((1, widths["hand_target"]), 103.0),
        }
        manager = SimpleNamespace(
            get_term=lambda name: SimpleNamespace(command=values[name])
        )
        env = SimpleNamespace(num_envs=1, command_manager=manager)
        observations = TensorDict({"actor": original}, batch_size=(1,))
        patched = _patch_command_observation(observations, env)["actor"]

        offset = 0
        for name, width in MICROBAN_TELEOP_OBSERVATION_SCHEMA:
            actual = patched[:, offset : offset + width]
            source_name = "twist" if name == "command" else name
            if source_name in values:
                self.assertTrue(torch.equal(actual, values[source_name]))
            else:
                self.assertTrue(
                    torch.equal(actual, original[:, offset : offset + width])
                )
            offset += width

    def test_native_walk_patch_preserves_original_63_wide_recurrence(self) -> None:
        original = torch.arange(63, dtype=torch.float32).unsqueeze(0)
        command = torch.tensor([[0.2, -0.1, 0.5]])
        env = SimpleNamespace(
            num_envs=1,
            command_manager=SimpleNamespace(
                get_term=lambda name: SimpleNamespace(command=command)
            ),
        )
        observations = TensorDict({"actor": original}, batch_size=(1,))
        patched = _patch_native_walk_observation(observations, env)["actor"]
        self.assertTrue(torch.equal(patched[:, :-3], original[:, :-3]))
        self.assertTrue(torch.equal(patched[:, -3:], command))
        self.assertTrue(
            torch.equal(
                original, torch.arange(63, dtype=torch.float32).unsqueeze(0)
            )
        )

    def test_legacy_walk_projection_excludes_three_hmd_joints(self) -> None:
        original = torch.arange(83, dtype=torch.float32).unsqueeze(0)
        observations = TensorDict({"actor": original}, batch_size=(1,))
        indices = torch.arange(3, 21)
        position_offset = torch.arange(18, dtype=torch.float32).unsqueeze(0)
        last_action = torch.full((1, 18), -7.0)
        projected = _walk_actor_observation(
            observations, indices, position_offset, last_action
        )["actor"]
        self.assertEqual(tuple(projected.shape), (1, 63))
        expected = torch.cat(
            (
                original[:, :6],
                original[:, 9:27] + position_offset,
                original[:, 30:48],
                last_action,
                original[:, 66:69],
            ),
            dim=1,
        )
        self.assertTrue(torch.equal(projected, expected))

    def test_legacy_walk_adapter_preserves_absolute_joint_targets(self) -> None:
        main_scale = torch.ones((1, 18))
        walk_scale = torch.ones((1, 18))
        main_offset = torch.zeros((1, 18))
        walk_offset = torch.zeros((1, 18))
        shoulder_pitch = math.radians(0.0)
        main_offset[:, [0, 9]] = shoulder_pitch
        position_offset, output_scale, output_offset = _legacy_walk_adapters(
            main_scale, main_offset, walk_scale, walk_offset
        )
        self.assertAlmostEqual(position_offset[0, 0].item(), shoulder_pitch, places=7)
        walk_raw = torch.zeros((1, 18))
        main_raw = walk_raw * output_scale + output_offset
        desired_from_main = main_raw * main_scale + main_offset
        desired_from_walk = walk_raw * walk_scale + walk_offset
        self.assertTrue(torch.allclose(desired_from_main, desired_from_walk))

    def test_legacy_walk_joint_projection_follows_exact_action_order(self) -> None:
        action_names = tuple(f"body_{index}" for index in range(18))
        robot_names = (
            action_names[3],
            MICROBAN_HMD_JOINT_NAMES[0],
            *action_names[:3],
            MICROBAN_HMD_JOINT_NAMES[1],
            *action_names[4:],
            MICROBAN_HMD_JOINT_NAMES[2],
        )
        indices = _legacy_body_joint_indices(robot_names, action_names)
        self.assertEqual(
            indices,
            tuple(robot_names.index(name) for name in action_names),
        )
        with self.assertRaisesRegex(ValueError, "joint set mismatch"):
            _legacy_body_joint_indices(robot_names, (*action_names[:-1], "missing"))


class DualActorDispatchTests(unittest.TestCase):
    @staticmethod
    def _policy(command: SimulationCommand) -> LivePicoSimulationPolicy:
        widths = dict(MICROBAN_TELEOP_OBSERVATION_SCHEMA)
        terms = {
            "twist": SimpleNamespace(command=torch.zeros((1, widths["command"]))),
            "foot_target": SimpleNamespace(
                command=torch.zeros((1, widths["foot_target"]))
            ),
            "hand_target": SimpleNamespace(
                command=torch.zeros((1, widths["hand_target"]))
            ),
        }
        policy = object.__new__(LivePicoSimulationPolicy)
        policy.env = SimpleNamespace(
            num_envs=1,
            command_manager=SimpleNamespace(get_term=lambda name: terms[name]),
        )
        policy._read_command = lambda: command
        policy._inject_command = lambda _command: None
        policy._write_hmd_target = lambda _command: None
        policy._print_status = lambda _command: None
        policy.camera_publisher = None
        policy.body_joint_observation_indices = torch.arange(3, 21)
        policy.walk_position_offset = torch.zeros((1, 18))
        policy.walk_output_scale = torch.full((1, 18), 2.0)
        policy.walk_output_offset = torch.full((1, 18), 3.0)
        policy.walk_last_action = torch.zeros((1, 18))
        policy.zero_action = torch.zeros((1, 18))
        policy.actor = lambda _observation: (_ for _ in ()).throw(
            AssertionError("teleop actor must not run for released X")
        )
        return policy

    def test_x_released_dispatches_legacy_walk_actor(self) -> None:
        command = SimulationCommand(
            enabled=True,
            twist=(0.1, 0.0, 0.0),
            foot_target=((0.0, 0.0, 0.0),) * 2,
            hand_target=((0.0, 0.0, 0.0),) * 2,
            hand_active=(False, False),
            head_orientation=(0.0, 0.0, 0.0),
            head_yaw_front=False,
            locomotion_policy="walk",
        )
        policy = self._policy(command)
        seen_widths: list[int] = []

        def walk_actor(observation: TensorDict) -> torch.Tensor:
            seen_widths.append(observation["actor"].shape[1])
            return torch.ones((1, 18))

        policy.walk_actor = walk_actor
        observations = TensorDict({"actor": torch.zeros((1, 83))}, batch_size=(1,))
        action = policy(observations)
        self.assertEqual(seen_widths, [63])
        self.assertTrue(torch.equal(action, torch.full((1, 18), 5.0)))
        self.assertTrue(torch.equal(policy.walk_last_action, torch.ones((1, 18))))

    def test_trigger_release_returns_shared_teleop_initial_pose(self) -> None:
        command = SimulationCommand(
            enabled=False,
            twist=(0.0, 0.0, 0.0),
            foot_target=((0.0, 0.0, 0.0),) * 2,
            hand_target=((0.0, 0.0, 0.0),) * 2,
            hand_active=(False, False),
            head_orientation=(0.0, 0.0, 0.0),
            head_yaw_front=False,
            locomotion_policy="walk",
        )
        policy = self._policy(command)
        policy.walk_actor = lambda _observation: (_ for _ in ()).throw(
            AssertionError("walk actor must not run while trigger is released")
        )
        policy.walk_last_action.fill_(9.0)
        observations = TensorDict({"actor": torch.zeros((1, 83))}, batch_size=(1,))
        action = policy(observations)
        self.assertTrue(torch.equal(action, torch.zeros((1, 18))))
        self.assertTrue(torch.equal(policy.walk_last_action, torch.zeros((1, 18))))

    def test_native_legacy_dispatch_returns_unmodified_actor_action(self) -> None:
        command = SimulationCommand(
            enabled=True,
            twist=(0.2, -0.1, 0.5),
            foot_target=((0.0, 0.0, 0.0),) * 2,
            hand_target=((0.0, 0.0, 0.0),) * 2,
            hand_active=(False, False),
            head_orientation=(0.0, 0.0, 0.0),
            head_yaw_front=False,
            locomotion_policy="walk",
        )
        policy = self._policy(command)
        policy.native_legacy_action_semantics = True
        policy.env.command_manager = SimpleNamespace(
            get_term=lambda name: SimpleNamespace(
                command=torch.tensor([[0.2, -0.1, 0.5]])
            )
        )
        seen: list[torch.Tensor] = []

        def walk_actor(observation: TensorDict) -> torch.Tensor:
            seen.append(observation["actor"].clone())
            return torch.full((1, 18), 0.25)

        policy.walk_actor = walk_actor
        observations = TensorDict({"actor": torch.zeros((1, 63))}, batch_size=(1,))
        action = policy(observations)
        self.assertEqual(len(seen), 1)
        self.assertTrue(
            torch.equal(seen[0][:, -3:], torch.tensor([[0.2, -0.1, 0.5]]))
        )
        # The old teleop adapter would return 3.5 here. Native velocity semantics
        # must execute the actor's original raw action unchanged.
        self.assertTrue(torch.equal(action, torch.full((1, 18), 0.25)))

    def test_episode_reset_transition_disarms_before_held_command_is_used(self) -> None:
        held = SimulationCommand(
            enabled=True,
            twist=(0.2, 0.0, 0.0),
            foot_target=((0.0, 0.0, 0.0),) * 2,
            hand_target=((0.0, 0.0, 0.0),) * 2,
            hand_active=(False, False),
            head_orientation=(0.0, 0.0, 0.0),
            head_yaw_front=False,
            locomotion_policy="walk",
        )
        policy = self._policy(held)
        policy.native_legacy_action_semantics = True
        policy.env.episode_length_buf = torch.tensor([0])
        policy.env.reset_buf = torch.tensor([False])
        policy._previous_episode_length = torch.tensor([17])
        reset_count = 0

        def disarm() -> None:
            nonlocal reset_count
            reset_count += 1
            policy._read_command = lambda: SimulationCommand(
                enabled=False,
                twist=(0.0, 0.0, 0.0),
                foot_target=((0.0, 0.0, 0.0),) * 2,
                hand_target=((0.0, 0.0, 0.0),) * 2,
                hand_active=(False, False),
                head_orientation=(0.0, 0.0, 0.0),
                head_yaw_front=False,
                locomotion_policy="walk",
                fault="reset requires trigger release",
            )
            policy._previous_episode_length = torch.tensor([0])

        policy.reset = disarm
        policy.walk_actor = lambda _observation: (_ for _ in ()).throw(
            AssertionError("held trigger must not run actor immediately after reset")
        )
        observations = TensorDict({"actor": torch.zeros((1, 63))}, batch_size=(1,))
        action = policy(observations)
        self.assertEqual(reset_count, 1)
        self.assertTrue(torch.equal(action, torch.zeros((1, 18))))

    def test_camera_failure_degrades_video_without_disarming_control(self) -> None:
        command = SimulationCommand(
            enabled=True,
            twist=(0.2, 0.0, 0.0),
            foot_target=((0.0, 0.0, 0.0),) * 2,
            hand_target=((0.0, 0.0, 0.0),) * 2,
            hand_active=(False, False),
            head_orientation=(0.0, 0.0, 0.0),
            head_yaw_front=False,
            locomotion_policy="walk",
        )
        policy = self._policy(command)
        policy._camera_fault = None

        class BrokenCamera:
            @staticmethod
            def capture_if_due() -> bool:
                raise RuntimeError("synthetic camera loss")

        policy.camera_publisher = BrokenCamera()
        policy.walk_actor = lambda _observation: torch.full((1, 18), 0.25)
        observations = TensorDict({"actor": torch.zeros((1, 83))}, batch_size=(1,))
        action = policy(observations)
        self.assertTrue(torch.equal(action, torch.full((1, 18), 3.5)))
        self.assertIsNone(policy.camera_publisher)
        self.assertIn("synthetic camera loss", policy._camera_fault or "")

    def test_authority_change_between_map_and_injection_forces_neutral(self) -> None:
        command = SimulationCommand(
            enabled=True,
            twist=(0.1, 0.0, 0.0),
            foot_target=((0.0, 0.0, 0.0),) * 2,
            hand_target=((0.0, 0.0, 0.0),) * 2,
            hand_active=(False, False),
            head_orientation=(0.0, 0.0, 0.0),
            head_yaw_front=False,
            locomotion_policy="walk",
        )
        policy = self._policy(command)
        injected: list[SimulationCommand] = []
        reset_count = 0

        class ChangedAuthoritySource:
            @staticmethod
            def run_if_current(_token, _callback):
                return False, None

        def reset_mapper() -> None:
            nonlocal reset_count
            reset_count += 1

        policy.source = ChangedAuthoritySource()
        policy.mapper = SimpleNamespace(reset=reset_mapper)
        policy._pending_authority_token = object()
        policy._inject_command = injected.append
        policy._write_hmd_target = lambda _command: None
        policy.walk_actor = lambda _observation: (_ for _ in ()).throw(
            AssertionError("actor must not run after authority loss")
        )
        observations = TensorDict({"actor": torch.zeros((1, 83))}, batch_size=(1,))
        action = policy(observations)

        self.assertEqual(reset_count, 1)
        self.assertEqual(len(injected), 1)
        self.assertFalse(injected[0].enabled)
        self.assertIn("authority changed", injected[0].fault or "")
        self.assertTrue(torch.equal(action, torch.zeros((1, 18))))


class CliSafetyTests(unittest.TestCase):
    def test_cli_has_no_robot_destination_or_send_switch(self) -> None:
        parser = build_parser()
        destinations = {action.dest for action in parser._actions}
        option_strings = {
            option for action in parser._actions for option in action.option_strings
        }
        self.assertNotIn("robot", destinations)
        self.assertNotIn("robot_port", destinations)
        self.assertNotIn("send", destinations)
        self.assertNotIn("--robot", option_strings)
        self.assertNotIn("--send", option_strings)

    def test_default_teleop_root_points_to_sibling_repository(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual(args.teleop_root.name, "microban_teleop")
        self.assertEqual(args.input, "webxr")
        self.assertIsNone(args.checkpoint)
        self.assertEqual(_runtime_task(args.checkpoint), "Mjlab-Velocity-Microban")

    def test_hybrid_checkpoint_keeps_the_teleop_task(self) -> None:
        self.assertEqual(
            _runtime_task(Path("hybrid.pt")),
            "Mjlab-Teleop-Microban",
        )

    def test_v12_preview_requires_dedicated_cli_option_and_task(self) -> None:
        checkpoint = Path("preview.pt")
        args = build_parser().parse_args(
            ["--v12-preview-checkpoint", str(checkpoint), "--input", "pico-app"]
        )
        self.assertEqual(_selected_checkpoint(args), checkpoint)
        self.assertEqual(
            _runtime_task(args.checkpoint, args.v12_preview_checkpoint),
            "Mjlab-Teleop-V12-Preview-Microban",
        )

        with self.assertRaises(SystemExit):
            build_parser().parse_args(
                [
                    "--checkpoint",
                    "ordinary.pt",
                    "--v12-preview-checkpoint",
                    str(checkpoint),
                ]
            )

    def test_default_legacy_checkpoint_is_the_audited_model_14999(self) -> None:
        checkpoint = _default_walk_checkpoint()
        self.assertEqual(checkpoint.name, "model_14999.pt")
        self.assertEqual(_sha256(checkpoint), AUDITED_LEGACY_WALK_SHA256)

    def test_authenticated_pico_app_is_a_distinct_input_backend(self) -> None:
        args = build_parser().parse_args(
            ["--checkpoint", "model_14999.pt", "--input", "pico-app"]
        )
        self.assertEqual(args.input, "pico-app")
        self.assertEqual(args.native_config, _default_native_config())

    def test_native_server_thread_starts_and_closes_async_server(self) -> None:
        class FakeServer:
            def __init__(self) -> None:
                self.started = threading.Event()
                self.closed = threading.Event()

            async def start(self) -> None:
                await asyncio.sleep(0)
                self.started.set()

            async def close(self) -> None:
                await asyncio.sleep(0)
                self.closed.set()

        server = FakeServer()
        thread = _NativeServerThread(server)
        thread.start()
        self.assertTrue(server.started.is_set())
        thread.close()
        self.assertTrue(server.closed.is_set())


if __name__ == "__main__":
    unittest.main()
