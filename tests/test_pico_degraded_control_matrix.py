# Copyright 2026 Marc Duclusaud
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Cross-repository regression tests for the deadline PICO fallback.

These tests never open a robot socket.  They exercise the authenticated native
mailbox, the controller mapper, and the simulation policy facade in memory.
"""

from __future__ import annotations

import importlib
import unittest
from functools import partial
from types import SimpleNamespace

import torch
from tensordict import TensorDict

from mjlab_microban.scripts.live_pico_teleop_sim import (
    LivePicoSimulationPolicy,
    SimulationCommand,
    _default_teleop_root,
    _load_pico_app_classes,
)
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_TELEOP_OBSERVATION_SCHEMA,
)


class _ManualClock:
    def __init__(self, value: int = 1_000_000_000) -> None:
        self.value = value

    def __call__(self) -> int:
        return self.value

    def advance_frame(self) -> None:
        self.value += 20_000_000


def _raw_snapshot(*, capture_ns: int, trigger: float, primary: bool) -> dict:
    pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    def controller(*, left: bool) -> dict:
        return {
            "pose": list(pose),
            "trigger": trigger if left else 0.0,
            "grip": 0.0,
            "stick": [0.0, 1.0] if left else [0.0, 0.0],
            "stick_click": False,
            "primary": primary if left else False,
            "secondary": False,
            "menu": False,
        }

    return {
        "schema": "microban_pico_raw_v1",
        "coordinate_frame": "pico_native_rh_x_right_y_up_z_back_m_xyzw",
        "app": {"focused": True, "paused": False},
        "head": {
            "pose": list(pose),
            "status": 1,
            "sensor_frame": capture_ns,
        },
        "controllers": {
            "left": controller(left=True),
            "right": controller(left=False),
        },
        # This is the real zero-tracker wire state.  Controller validity must
        # remain independent of the unavailable body stream.
        "body": None,
    }


def _command(
    *, enabled: bool = True, policy: str = "walk"
) -> SimulationCommand:
    return SimulationCommand(
        enabled=enabled,
        twist=(0.2, 0.0, 0.0) if enabled else (0.0, 0.0, 0.0),
        foot_target=((0.0, 0.0, 0.0),) * 2,
        hand_target=((0.0, 0.0, 0.0),) * 2,
        hand_active=(False, False),
        head_orientation=(0.0, 0.0, 0.0),
        head_yaw_front=False,
        locomotion_policy=policy,
    )


def _mapped_command(*, trigger: float, policy: str) -> dict[str, object]:
    walking = trigger >= 0.65
    zero_pair = {"left": (0.0, 0.0, 0.0), "right": (0.0, 0.0, 0.0)}
    return {
        "velocity": {
            "vx": 0.5 if walking else 0.0,
            "vy": 0.0,
            "vtheta": 0.0,
        },
        "active_moves": ["walk", "hmd_head"] if walking else ["hmd_head"],
        "locomotion_policy": policy,
        "head_orientation": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        "head_yaw_front": False,
        "foot_target": zero_pair,
        "hand_target": zero_pair,
        "hand_active": {"left": False, "right": False},
        "body_target_calibrated": True,
        "body_target_fresh": True,
    }


def _record_command(
    output: list[SimulationCommand], command: SimulationCommand
) -> SimulationCommand:
    output.append(command)
    return command


class PicoDegradedControlMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        (
            _load_pairing_store,
            _server_class,
            cls.source_class,
            cls.mapper_class,
        ) = _load_pico_app_classes(_default_teleop_root())
        cls.parser_class = importlib.import_module(
            "microban_teleop.native_tracking"
        ).NativeTrackingParser

    @staticmethod
    def _read_policy(
        source,
        mapper,
        clock: _ManualClock,
        *,
        legacy_only: bool = True,
        legacy_fallback_mapper=None,
    ):
        policy = object.__new__(LivePicoSimulationPolicy)
        policy.source = source
        policy.mapper = mapper
        policy.legacy_fallback_mapper = legacy_fallback_mapper
        policy.clock_ns = clock
        policy._previous_sampled_at_ns = None
        policy._pending_authority_token = None
        policy._pending_legacy_command = None
        policy._legacy_fallback_latched = False
        policy._legacy_fallback_reason = None
        policy.legacy_only = legacy_only
        return policy

    @staticmethod
    def _publish(
        source,
        authority,
        parser,
        clock: _ManualClock,
        *,
        capture_ns: int,
        trigger: float,
    ) -> None:
        frame = parser.ingest(
            _raw_snapshot(
                capture_ns=capture_ns,
                trigger=trigger,
                primary=True,
            ),
            capture_mono_ns=capture_ns,
            received_at_ns=clock(),
        )
        assert frame.controller_health.fresh
        assert not frame.body_health.fresh
        source.publish(authority, frame)

    def test_zero_trackers_do_not_disable_legacy_joystick(self) -> None:
        clock = _ManualClock()
        source = self.source_class(clock_ns=clock)
        mapper = self.mapper_class(deadzone=0.0)
        source.set_consumer_reset_callback(mapper.reset)
        policy = self._read_policy(source, mapper, clock)
        parser = self.parser_class()
        authority = source.begin_authority()

        self._publish(
            source,
            authority,
            parser,
            clock,
            capture_ns=1,
            trigger=0.0,
        )
        self.assertFalse(policy._read_command().enabled)
        clock.advance_frame()
        self._publish(
            source,
            authority,
            parser,
            clock,
            capture_ns=2,
            trigger=1.0,
        )
        command = policy._read_command()
        self.assertTrue(command.enabled)
        self.assertEqual(command.locomotion_policy, "walk")
        self.assertEqual(command.twist, (0.7, 0.0, 0.0))
        source.close()

    def test_explicit_checkpoint_zero_trackers_uses_controller_policy(self) -> None:
        clock = _ManualClock()
        source = self.source_class(clock_ns=clock)
        mapper = self.mapper_class(deadzone=0.0)
        fallback_mapper = self.mapper_class(deadzone=0.0)

        def reset_mappers() -> None:
            mapper.reset()
            fallback_mapper.reset()

        source.set_consumer_reset_callback(reset_mappers)
        policy = self._read_policy(
            source,
            mapper,
            clock,
            legacy_only=False,
            legacy_fallback_mapper=fallback_mapper,
        )
        parser = self.parser_class()
        authority = source.begin_authority()

        self._publish(
            source,
            authority,
            parser,
            clock,
            capture_ns=1,
            trigger=0.0,
        )
        self.assertFalse(policy._read_command().enabled)
        clock.advance_frame()
        self._publish(
            source,
            authority,
            parser,
            clock,
            capture_ns=2,
            trigger=1.0,
        )
        command = policy._read_command()
        self.assertTrue(command.enabled)
        self.assertEqual(command.locomotion_policy, "pico_teleop")
        self.assertEqual(command.twist, (0.7, 0.0, 0.0))
        self.assertFalse(policy._legacy_fallback_latched)
        self.assertIsNone(command.fault)

        clock.advance_frame()
        self._publish(
            source,
            authority,
            parser,
            clock,
            capture_ns=3,
            trigger=1.0,
        )
        self.assertEqual(policy._read_command().locomotion_policy, "pico_teleop")
        self.assertFalse(policy._legacy_fallback_latched)

        clock.advance_frame()
        self._publish(
            source,
            authority,
            parser,
            clock,
            capture_ns=4,
            trigger=0.0,
        )
        released = policy._read_command()
        self.assertFalse(released.enabled)
        self.assertFalse(policy._legacy_fallback_latched)
        source.close()

    def test_transient_reconnect_recovers_after_release_then_press(self) -> None:
        clock = _ManualClock()
        source = self.source_class(clock_ns=clock)
        mapper = self.mapper_class(deadzone=0.0)
        source.set_consumer_reset_callback(mapper.reset)
        policy = self._read_policy(source, mapper, clock)

        first = source.begin_authority()
        first_parser = self.parser_class()
        self._publish(
            source,
            first,
            first_parser,
            clock,
            capture_ns=1,
            trigger=0.0,
        )
        self.assertFalse(policy._read_command().enabled)
        clock.advance_frame()
        self._publish(
            source,
            first,
            first_parser,
            clock,
            capture_ns=2,
            trigger=1.0,
        )
        self.assertTrue(policy._read_command().enabled)

        self.assertTrue(source.invalidate_authority(first, "synthetic disconnect"))
        clock.advance_frame()
        self.assertFalse(policy._read_command().enabled)

        second = source.begin_authority()
        second_parser = self.parser_class()
        clock.advance_frame()
        self._publish(
            source,
            second,
            second_parser,
            clock,
            capture_ns=1,
            trigger=1.0,
        )
        self.assertFalse(
            policy._read_command().enabled,
            "a held deadman must not restart motion across reconnect",
        )
        clock.advance_frame()
        self._publish(
            source,
            second,
            second_parser,
            clock,
            capture_ns=2,
            trigger=0.0,
        )
        self.assertFalse(policy._read_command().enabled)
        clock.advance_frame()
        self._publish(
            source,
            second,
            second_parser,
            clock,
            capture_ns=3,
            trigger=1.0,
        )
        self.assertTrue(policy._read_command().enabled)
        source.close()

    def test_deadline_fallback_never_calls_optional_learned_actor(self) -> None:
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
            reset_buf=None,
            episode_length_buf=None,
            command_manager=SimpleNamespace(get_term=lambda name: terms[name]),
        )
        policy._previous_episode_length = None
        policy._read_command = lambda: _command()
        policy._inject_with_authority = lambda command: command
        policy._print_status = lambda _command_value: None
        policy.camera_publisher = None
        policy._camera_fault = None
        policy.native_legacy_action_semantics = False
        policy.body_joint_observation_indices = torch.arange(3, 21)
        policy.walk_position_offset = torch.zeros((1, 18))
        policy.walk_output_scale = torch.ones((1, 18))
        policy.walk_output_offset = torch.zeros((1, 18))
        policy.walk_last_action = torch.zeros((1, 18))
        policy.zero_action = torch.zeros((1, 18))
        policy.walk_actor = lambda _observation: torch.full((1, 18), 0.25)

        observations = TensorDict(
            {"actor": torch.zeros((1, 83))}, batch_size=(1,)
        )
        for learned_actor in (
            None,
            lambda _observation: (_ for _ in ()).throw(
                RuntimeError("synthetic learned-policy failure")
            ),
        ):
            with self.subTest(learned_actor=learned_actor):
                policy.actor = learned_actor
                action = policy(observations)
                self.assertTrue(
                    torch.equal(action, torch.full((1, 18), 0.25))
                )

    def test_learned_actor_faults_fallback_in_the_same_policy_call(self) -> None:
        widths = dict(MICROBAN_TELEOP_OBSERVATION_SCHEMA)
        observations = TensorDict(
            {"actor": torch.zeros((1, 83))}, batch_size=(1,)
        )

        actor_cases = {
            "absent": None,
            "exception": lambda _observation: (_ for _ in ()).throw(
                RuntimeError("synthetic inference failure")
            ),
            "nonfinite": lambda _observation: torch.full((1, 18), float("nan")),
        }
        for name, actor in actor_cases.items():
            with self.subTest(name=name):
                terms = {
                    "twist": SimpleNamespace(
                        command=torch.zeros((1, widths["command"]))
                    ),
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
                    reset_buf=None,
                    episode_length_buf=None,
                    command_manager=SimpleNamespace(
                        get_term=terms.__getitem__
                    ),
                )
                policy._previous_episode_length = None
                policy._read_command = lambda: _command(policy="pico_teleop")
                policy._pending_legacy_command = _command(policy="walk")
                injected: list[SimulationCommand] = []
                policy._inject_with_authority = partial(
                    _record_command, injected
                )
                policy._print_status = lambda _command_value: None
                policy.camera_publisher = None
                policy._camera_fault = None
                policy._legacy_fallback_latched = False
                policy._legacy_fallback_reason = None
                policy.native_legacy_action_semantics = False
                policy.body_joint_observation_indices = torch.arange(3, 21)
                policy.walk_position_offset = torch.zeros((1, 18))
                policy.walk_output_scale = torch.ones((1, 18))
                policy.walk_output_offset = torch.zeros((1, 18))
                policy.walk_last_action = torch.zeros((1, 18))
                policy.zero_action = torch.zeros((1, 18))
                policy.walk_actor = lambda _observation: torch.full((1, 18), 0.25)
                policy.actor = actor

                action = policy(observations)
                self.assertTrue(
                    torch.equal(action, torch.full((1, 18), 0.25))
                )
                self.assertEqual(
                    [command.locomotion_policy for command in injected],
                    ["pico_teleop"],
                )
                self.assertTrue(policy._legacy_fallback_latched)

    def test_learned_fault_latch_clears_only_on_release_then_retries(self) -> None:
        frames = iter(
            (
                SimpleNamespace(
                    sampled_at_ns=1_000_000_000,
                    left_controller=SimpleNamespace(trigger=1.0),
                ),
                SimpleNamespace(
                    sampled_at_ns=1_020_000_000,
                    left_controller=SimpleNamespace(trigger=0.0),
                ),
                SimpleNamespace(
                    sampled_at_ns=1_040_000_000,
                    left_controller=SimpleNamespace(trigger=1.0),
                ),
            )
        )

        class Mapper:
            def __init__(self, policy_name: str) -> None:
                self.policy_name = policy_name

            def map_sample(self, frame) -> dict[str, object]:
                return _mapped_command(
                    trigger=frame.left_controller.trigger,
                    policy=self.policy_name,
                )

            def reset(self) -> None:
                return None

        clock = _ManualClock(1_000_000_000)

        def read_frame():
            frame = next(frames)
            clock.value = frame.sampled_at_ns
            return frame

        policy = self._read_policy(
            SimpleNamespace(read=read_frame),
            Mapper("pico_teleop"),
            clock,
            legacy_only=False,
            legacy_fallback_mapper=Mapper("walk"),
        )
        policy._latch_legacy_fallback("synthetic learned actor failure")

        held = policy._read_command()
        self.assertEqual(held.locomotion_policy, "walk")
        self.assertTrue(policy._legacy_fallback_latched)

        released = policy._read_command()
        self.assertFalse(released.enabled)
        self.assertFalse(policy._legacy_fallback_latched)

        retried = policy._read_command()
        self.assertTrue(retried.enabled)
        self.assertEqual(retried.locomotion_policy, "pico_teleop")

    def test_latch_never_hot_swaps_when_fallback_is_temporarily_unarmed(self) -> None:
        frame = SimpleNamespace(
            sampled_at_ns=1_000_000_000,
            left_controller=SimpleNamespace(trigger=1.0),
        )

        class FixedMapper:
            def __init__(self, mapped: dict[str, object]) -> None:
                self.mapped = mapped

            def map_sample(self, _frame) -> dict[str, object]:
                return self.mapped

            def reset(self) -> None:
                return None

        policy = self._read_policy(
            SimpleNamespace(read=lambda: frame),
            FixedMapper(_mapped_command(trigger=1.0, policy="pico_teleop")),
            _ManualClock(1_000_000_000),
            legacy_only=False,
            legacy_fallback_mapper=FixedMapper(
                _mapped_command(trigger=0.0, policy="walk")
            ),
        )
        policy._latch_legacy_fallback("synthetic learned actor failure")

        command = policy._read_command()
        self.assertFalse(command.enabled)
        self.assertEqual(command.locomotion_policy, "walk")
        self.assertTrue(policy._legacy_fallback_latched)

    def test_stale_trigger_release_does_not_clear_fallback_latch(self) -> None:
        frame = SimpleNamespace(
            sampled_at_ns=1_000_000_000,
            left_controller=SimpleNamespace(trigger=0.0),
            controller_health=SimpleNamespace(fresh=False, valid=False),
        )

        class FixedMapper:
            def map_sample(self, _frame) -> dict[str, object]:
                return _mapped_command(trigger=0.0, policy="walk")

            def reset(self) -> None:
                return None

        policy = self._read_policy(
            SimpleNamespace(read=lambda: frame),
            FixedMapper(),
            _ManualClock(1_000_000_000),
            legacy_only=False,
            legacy_fallback_mapper=FixedMapper(),
        )
        policy._latch_legacy_fallback("synthetic learned actor failure")

        self.assertFalse(policy._read_command().enabled)
        self.assertTrue(policy._legacy_fallback_latched)


if __name__ == "__main__":
    unittest.main()
