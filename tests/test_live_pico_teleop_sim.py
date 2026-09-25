# Copyright 2026 Marc Duclusaud
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the simulation-only live PICO bridge."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import json
import math
import runpy
import threading
import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
from tensordict import TensorDict

from mjlab_microban.robot.microban_hand_fk import (
    MICROBAN_ARM_HOME_JOINT_RAD,
    MICROBAN_ARM_JOINT_UPPER_RAD,
    microban_hand_offsets_from_arm_joints,
)
from mjlab_microban.scripts.live_pico_teleop_sim import (
    ARM_TRACKING_HOLD_MAX_S,
    AUDITED_LEGACY_WALK_SHA256,
    FOOT_INACTIVE_Z_MAX_M,
    FOOT_UPPER_M,
    HAND_UPPER_M,
    UNACCEPTED_SIMULATION_WARNING,
    ControllerOnlyPreviewMapper,
    LivePicoSimulationPolicy,
    MICROBAN_DIRECT_ARM_JOINT_UPPER_RAD,
    MICROBAN_DIRECT_ARM_TARGET_CONTRACT_REVISION,
    SimulationCommand,
    WebXrSimulationMapper,
    WebXrSimulationSource,
    _configure_live_environment,
    _construct_checkpoint_consumer_runner,
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
    _validate_controller_preview_live_authority,
    _validate_preview_live_authority,
    _validate_unaccepted_simulation_preview,
    _walk_actor_observation,
    build_parser,
    command_for_simulation,
    scale_normalized_velocity,
    solve_hmd_neck_target,
)
from mjlab_microban.scripts.live_pico_teleop_sim import (
    run as run_live_simulation,
)
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_HMD_JOINT_NAMES,
    MICROBAN_TELEOP_OBSERVATION_SCHEMA,
)
from mjlab_microban.tasks.microban_teleop_mdp import (
    MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M,
    ResetFixedHandTargetCommand,
)
from mjlab_microban.tasks.microban_teleop_v12_preview import (
    TELEOP_V12_PREVIEW_FULLBODY_STRICT_QUALITY,
    TELEOP_V12_PREVIEW_FULLBODY_VISUAL_QUALITY,
    TELEOP_V12_PREVIEW_PHASE1_STRICT_QUALITY,
    TELEOP_V12_PREVIEW_PHASE1_VISUAL_QUALITY,
    TELEOP_V12_PREVIEW_PHASE_FULL_BODY,
    TELEOP_V12_PREVIEW_PHASE_HMD_HAND,
    staged_preview_info,
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


def _arm_joint_target(fraction: float) -> tuple[tuple[float, ...], ...]:
    """Return one non-HOME, contract-bounded controller IK target."""

    return tuple(
        tuple(
            home + fraction * (upper - home)
            for home, upper in zip(home_side, upper_side, strict=True)
        )
        for home_side, upper_side in zip(
            MICROBAN_ARM_HOME_JOINT_RAD,
            MICROBAN_ARM_JOINT_UPPER_RAD,
            strict=True,
        )
    )


def _controller_arm_mapper_command(
    *,
    walking: bool,
    arm_tracking_enabled: bool,
    arm_joint_target: tuple[tuple[float, ...], ...] | None = None,
) -> dict[str, object]:
    """Build the local mapper contract consumed by the live simulator."""

    zero_pair = {"left": (0.0, 0.0, 0.0), "right": (0.0, 0.0, 0.0)}
    if arm_tracking_enabled:
        assert arm_joint_target is not None
        hand_values = microban_hand_offsets_from_arm_joints(
            torch.tensor(arm_joint_target, dtype=torch.float64)
        ).tolist()
        hand_target = {"left": hand_values[0], "right": hand_values[1]}
        serialized_arm_target = {
            "left": list(arm_joint_target[0]),
            "right": list(arm_joint_target[1]),
        }
    else:
        hand_target = zero_pair
        serialized_arm_target = None

    active_moves = ["walk", "hmd_head"] if walking else []
    if arm_tracking_enabled:
        active_moves.append("pico_arms")
    return {
        "velocity": {
            "vx": 0.5 if walking else 0.0,
            "vy": 0.0,
            "vtheta": 0.0,
        },
        "active_moves": active_moves,
        "locomotion_policy": "pico_teleop",
        "head_orientation": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        # Right-grip neck-yaw mapping belongs to the external mapper contract;
        # these tests isolate the independent controller-arm fields.
        "head_yaw_front": False,
        "foot_target": zero_pair,
        "hand_target": hand_target,
        "hand_active": {
            "left": arm_tracking_enabled,
            "right": arm_tracking_enabled,
        },
        "arm_tracking_enabled": arm_tracking_enabled,
        "arm_joint_target": serialized_arm_target,
        "body_target_calibrated": True,
        "body_target_fresh": True,
        "twist2_body_target_preview": {
            "target_source": "controllers",
            "controller_only_hand_fallback": True,
            "absolute_arm_target_contract_revision": (
                MICROBAN_DIRECT_ARM_TARGET_CONTRACT_REVISION
            ),
            "commanded_ik_q_rad": serialized_arm_target,
        },
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

    def test_controller_arm_joint_target_is_independently_bounded(self) -> None:
        value = _valid_command()
        value["foot_target"] = {
            "left": (0.0, 0.0, 0.0),
            "right": (0.0, 0.0, 0.0),
        }
        value["controller_arm_joint_target"] = {
            side: list(MICROBAN_DIRECT_ARM_JOINT_UPPER_RAD[index])
            for index, side in enumerate(("left", "right"))
        }
        command = command_for_simulation(value)
        self.assertTrue(command.enabled)
        self.assertEqual(
            command.arm_joint_target, MICROBAN_DIRECT_ARM_JOINT_UPPER_RAD
        )

        value["controller_arm_joint_target"]["left"][0] = (
            MICROBAN_DIRECT_ARM_JOINT_UPPER_RAD[0][0] + 1.0e-6
        )
        rejected = command_for_simulation(value)
        self.assertFalse(rejected.enabled)
        self.assertEqual(rejected.fault, "controller arm joints exceed contract")

    def test_arm_tracking_gate_is_independent_of_locomotion_deadman(self) -> None:
        desired = _arm_joint_target(0.5)

        stationary = command_for_simulation(
            _controller_arm_mapper_command(
                walking=False,
                arm_tracking_enabled=True,
                arm_joint_target=desired,
            )
        )
        self.assertFalse(stationary.enabled)
        self.assertEqual(stationary.twist, (0.0, 0.0, 0.0))
        self.assertTrue(stationary.arm_tracking_enabled)
        self.assertEqual(stationary.arm_joint_target, desired)
        self.assertEqual(stationary.hand_active, (True, True))

        walking = command_for_simulation(
            _controller_arm_mapper_command(
                walking=True,
                arm_tracking_enabled=True,
                arm_joint_target=desired,
            )
        )
        self.assertTrue(walking.enabled)
        self.assertEqual(walking.twist, (0.35, 0.0, 0.0))
        self.assertTrue(walking.arm_tracking_enabled)
        self.assertEqual(walking.arm_joint_target, desired)
        self.assertEqual(stationary.hand_target, walking.hand_target)

        released_while_walking = command_for_simulation(
            _controller_arm_mapper_command(
                walking=True,
                arm_tracking_enabled=False,
            )
        )
        self.assertTrue(released_while_walking.enabled)
        self.assertEqual(released_while_walking.twist, (0.35, 0.0, 0.0))
        self.assertFalse(released_while_walking.arm_tracking_enabled)
        self.assertIsNone(released_while_walking.arm_joint_target)

    def test_arm_tracking_rejects_mismatched_absolute_contract_revision(self) -> None:
        desired = _arm_joint_target(0.5)
        value = _controller_arm_mapper_command(
            walking=False,
            arm_tracking_enabled=True,
            arm_joint_target=desired,
        )
        preview = value["twist2_body_target_preview"]
        assert isinstance(preview, dict)
        preview["absolute_arm_target_contract_revision"] = "stale-contract"

        rejected = command_for_simulation(value)

        self.assertFalse(rejected.arm_tracking_enabled)
        self.assertEqual(
            rejected.fault, "absolute arm target contract revision mismatch"
        )


class ControllerOnlyPreviewTests(unittest.TestCase):
    def test_deterministic_overlay_evaluator_actually_enables_arm_tracking(self) -> None:
        namespace = runpy.run_path(
            str(
                Path(__file__).resolve().parents[1]
                / "scripts"
                / "evaluate_pico_controller_overlay.py"
            )
        )
        scenario_type = namespace["Scenario"]
        make_command = namespace["_command"]
        target = _arm_joint_target(0.5)
        command = make_command(
            scenario_type("arm_motion", (0.0, 0.0, 0.0), target),
            step=0,
            ramp_steps=1,
        )
        self.assertTrue(command.arm_tracking_enabled)
        self.assertEqual(command.arm_joint_target, target)

        scenario_values = namespace["scenarios"]()
        self.assertIn("narrow_policy", {value.target_domain for value in scenario_values})
        self.assertIn(
            "absolute_direct_overlay",
            {value.target_domain for value in scenario_values},
        )
        self.assertEqual(
            namespace["MICROBAN_DIRECT_ARM_TARGET_CONTRACT_REVISION"],
            MICROBAN_DIRECT_ARM_TARGET_CONTRACT_REVISION,
        )

    def test_mapper_forces_controller_source_and_exact_zero_feet(self) -> None:
        class Mapper:
            def map_sample(self, frame):
                self.frame = frame
                joints = {
                    side: list(MICROBAN_ARM_HOME_JOINT_RAD[index])
                    for index, side in enumerate(("left", "right"))
                }
                hands = microban_hand_offsets_from_arm_joints(
                    torch.tensor(MICROBAN_ARM_HOME_JOINT_RAD, dtype=torch.float64)
                ).tolist()
                return {
                    "locomotion_policy": "pico_teleop",
                    "arm_tracking_enabled": True,
                    "arm_joint_target": joints,
                    "body_target_calibrated": True,
                    "body_target_fresh": True,
                    "hand_target": {
                        "left": hands[0],
                        "right": hands[1],
                    },
                    "foot_target": {
                        "left": [0.01, 0.0, 0.02],
                        "right": [0.0, 0.0, 0.0],
                    },
                    "twist2_body_target_preview": {
                        "target_source": "controllers",
                        "controller_only_hand_fallback": True,
                        "commanded_ik_q_rad": joints,
                    },
                }

            def reset(self):
                self.reset_called = True

            @staticmethod
            def neutral():
                return {"locomotion_policy": "pico_teleop"}

        inner = Mapper()
        mapper = ControllerOnlyPreviewMapper(inner)
        result = mapper.map_sample(
            SimpleNamespace(body=object(), body_jumps=("jump",))
        )
        self.assertIsNone(inner.frame.body)
        self.assertEqual(inner.frame.body_jumps, ())
        self.assertEqual(
            result["foot_target"],
            {"left": [0.0, 0.0, 0.0], "right": [0.0, 0.0, 0.0]},
        )
        self.assertEqual(
            result["arm_joint_target"],
            result["twist2_body_target_preview"]["commanded_ik_q_rad"],
        )

    def test_direct_overlay_tracks_stationary_and_walking_then_homes_on_release(
        self,
    ) -> None:
        policy = object.__new__(LivePicoSimulationPolicy)
        policy.zero_action = torch.zeros((1, 18))
        policy.main_action_scale = torch.ones((1, 18))
        policy.main_action_offset = torch.zeros((1, 18))
        policy.arm_action_indices = {
            "left": torch.tensor([1, 4, 7]),
            "right": torch.tensor([2, 5, 8]),
        }
        policy.arm_joint_ids = {
            "left": torch.tensor([3, 6, 9]),
            "right": torch.tensor([4, 7, 10]),
        }
        limits = torch.empty((1, 21, 2))
        limits[..., 0] = -3.0
        limits[..., 1] = 3.0
        policy.robot = SimpleNamespace(
            data=SimpleNamespace(soft_joint_pos_limits=limits)
        )
        base = torch.arange(18, dtype=torch.float32).unsqueeze(0) / 100.0
        desired = (
            MICROBAN_DIRECT_ARM_JOINT_UPPER_RAD[0],
            MICROBAN_DIRECT_ARM_JOINT_UPPER_RAD[1],
        )
        arm_columns = {1, 2, 4, 5, 7, 8}
        for locomotion_enabled in (False, True):
            with self.subTest(locomotion_enabled=locomotion_enabled):
                active = SimulationCommand(
                    enabled=locomotion_enabled,
                    twist=(0.1, 0.0, 0.0) if locomotion_enabled else (0.0, 0.0, 0.0),
                    foot_target=((0.0, 0.0, 0.0),) * 2,
                    hand_target=((0.0, 0.0, 0.0),) * 2,
                    hand_active=(True, True),
                    head_orientation=(0.0, 0.0, 0.0),
                    head_yaw_front=False,
                    locomotion_policy="pico_teleop",
                    arm_tracking_enabled=True,
                    arm_joint_target=desired,
                )
                overlaid = policy._apply_controller_arm_overlay(base, active)
                for column in range(18):
                    if column not in arm_columns:
                        self.assertEqual(overlaid[0, column], base[0, column])
                self.assertTrue(
                    torch.allclose(
                        overlaid[0, policy.arm_action_indices["left"]],
                        torch.tensor(desired[0]),
                    )
                )
                self.assertTrue(
                    torch.allclose(
                        overlaid[0, policy.arm_action_indices["right"]],
                        torch.tensor(desired[1]),
                    )
                )

        released_while_walking = SimulationCommand(
            enabled=True,
            twist=(0.1, 0.0, 0.0),
            foot_target=((0.0, 0.0, 0.0),) * 2,
            hand_target=((0.0, 0.0, 0.0),) * 2,
            hand_active=(False, False),
            head_orientation=(0.0, 0.0, 0.0),
            head_yaw_front=False,
            locomotion_policy="pico_teleop",
            arm_tracking_enabled=False,
            # A stale value must never override the explicit right-trigger release.
            arm_joint_target=desired,
        )
        released = policy._apply_controller_arm_overlay(base, released_while_walking)
        self.assertTrue(
            torch.allclose(
                released[0, policy.arm_action_indices["left"]],
                torch.tensor(MICROBAN_ARM_HOME_JOINT_RAD[0]),
            )
        )
        self.assertTrue(
            torch.allclose(
                released[0, policy.arm_action_indices["right"]],
                torch.tensor(MICROBAN_ARM_HOME_JOINT_RAD[1]),
            )
        )

    def test_deadman_release_neutralizes_hmd_and_every_body_target(self) -> None:
        value = _valid_command()
        value["active_moves"] = []
        value["head_yaw_front"] = True

        command = command_for_simulation(value)

        self.assertFalse(command.enabled)
        self.assertEqual(command.twist, (0.0, 0.0, 0.0))
        self.assertEqual(command.head_orientation, (0.0, 0.0, 0.0))
        self.assertFalse(command.head_yaw_front)
        self.assertEqual(command.foot_target, ((0.0, 0.0, 0.0),) * 2)
        self.assertEqual(command.hand_target, ((0.0, 0.0, 0.0),) * 2)
        self.assertEqual(command.hand_active, (False, False))

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


class ArmTrackingContinuityTests(unittest.TestCase):
    class _Source:
        def __init__(self, values: list[object]) -> None:
            self.values = iter(values)
            self.now_ns = 0

        def read(self) -> SimpleNamespace:
            value = next(self.values)
            if isinstance(value, Exception):
                raise value
            assert isinstance(value, SimpleNamespace)
            self.now_ns = value.sampled_at_ns
            return value

    class _Mapper:
        def __init__(self, values: list[dict[str, object]]) -> None:
            self.values = iter(values)
            self.reset_count = 0

        def map_sample(self, _frame: object) -> dict[str, object]:
            return next(self.values)

        def reset(self) -> None:
            self.reset_count += 1

    @staticmethod
    def _frame(
        sampled_at_ns: int,
        *,
        left_trigger: float | None,
        right_trigger: float | None,
        valid: bool = True,
        fresh: bool = True,
        error: str | None = None,
    ) -> SimpleNamespace:
        def controller(trigger: float | None) -> SimpleNamespace | None:
            return None if trigger is None else SimpleNamespace(trigger=trigger)

        return SimpleNamespace(
            sampled_at_ns=sampled_at_ns,
            left_controller=controller(left_trigger),
            right_controller=controller(right_trigger),
            controller_health=SimpleNamespace(
                valid=valid,
                fresh=fresh,
                error=error,
            ),
        )

    @staticmethod
    def _policy(
        source: _Source, mapper: _Mapper
    ) -> LivePicoSimulationPolicy:
        policy = object.__new__(LivePicoSimulationPolicy)
        policy.source = source
        policy.mapper = mapper
        policy.legacy_fallback_mapper = None
        policy.clock_ns = lambda: source.now_ns
        policy._previous_sampled_at_ns = None
        policy._pending_authority_token = None
        policy._pending_legacy_command = None
        policy._legacy_fallback_latched = False
        policy._legacy_fallback_reason = None
        policy.legacy_only = False
        policy.controller_only_preview = True
        policy.controller_arm_overlay = True
        return policy

    def test_brief_invalid_tracking_holds_then_recovers_without_repress(self) -> None:
        first_target = _arm_joint_target(0.25)
        recovered_target = _arm_joint_target(0.5)
        invalid = _controller_arm_mapper_command(
            walking=False,
            arm_tracking_enabled=False,
        )
        source = self._Source(
            [
                self._frame(
                    1_000_000_000,
                    left_trigger=1.0,
                    right_trigger=1.0,
                ),
                self._frame(
                    1_020_000_000,
                    left_trigger=None,
                    right_trigger=None,
                    valid=False,
                    fresh=False,
                    error="synthetic transient pose loss",
                ),
                self._frame(
                    1_040_000_000,
                    left_trigger=1.0,
                    right_trigger=1.0,
                ),
            ]
        )
        mapper = self._Mapper(
            [
                _controller_arm_mapper_command(
                    walking=True,
                    arm_tracking_enabled=True,
                    arm_joint_target=first_target,
                ),
                invalid,
                _controller_arm_mapper_command(
                    walking=True,
                    arm_tracking_enabled=True,
                    arm_joint_target=recovered_target,
                ),
            ]
        )
        policy = self._policy(source, mapper)

        active = policy._read_command()
        self.assertTrue(active.enabled)
        self.assertTrue(active.arm_tracking_enabled)
        self.assertEqual(active.arm_joint_target, first_target)

        held = policy._read_command()
        self.assertFalse(held.enabled)
        self.assertEqual(held.twist, (0.0, 0.0, 0.0))
        self.assertTrue(held.arm_tracking_enabled)
        self.assertEqual(held.arm_joint_target, first_target)
        self.assertEqual(held.hand_target, active.hand_target)
        self.assertFalse(policy._legacy_fallback_latched)

        recovered = policy._read_command()
        self.assertTrue(recovered.enabled)
        self.assertTrue(recovered.arm_tracking_enabled)
        self.assertEqual(recovered.arm_joint_target, recovered_target)
        self.assertNotEqual(recovered.arm_joint_target, first_target)
        self.assertEqual(mapper.reset_count, 0)

    def test_controller_preview_reports_absolute_pico_hands_in_hmd_frame(self) -> None:
        desired = _arm_joint_target(0.25)
        mapped = _controller_arm_mapper_command(
            walking=False,
            arm_tracking_enabled=True,
            arm_joint_target=desired,
        )
        preview = mapped["twist2_body_target_preview"]
        assert isinstance(preview, dict)
        preview.update(
            {
                "absolute_arm_target_source": "current_hmd_to_controller_absolute",
                "absolute_arm_rejection_reason": None,
                "controller_hand_hmd_m": {
                    "left": [0.1, -0.2, 0.3],
                    "right": [-0.4, 0.5, -0.6],
                },
            }
        )
        source = self._Source(
            [
                self._frame(
                    1_000_000_000,
                    left_trigger=0.0,
                    right_trigger=1.0,
                )
            ]
        )
        policy = self._policy(source, self._Mapper([mapped]))

        policy._read_command()

        assert policy._last_controller_hand_hmd_m is not None
        for actual, expected in zip(
            policy._last_controller_hand_hmd_m,
            ((0.1, -0.2, 0.3), (-0.4, 0.5, -0.6)),
            strict=True,
        ):
            self.assertSequenceEqual(
                tuple(round(value, 12) for value in actual), expected
            )

    def test_valid_right_release_homes_arms_without_stopping_locomotion(self) -> None:
        desired = _arm_joint_target(0.5)
        source = self._Source(
            [
                self._frame(
                    1_000_000_000,
                    left_trigger=1.0,
                    right_trigger=1.0,
                ),
                self._frame(
                    1_020_000_000,
                    left_trigger=1.0,
                    right_trigger=0.0,
                ),
                self._frame(
                    1_040_000_000,
                    left_trigger=None,
                    right_trigger=None,
                    valid=False,
                    fresh=False,
                ),
            ]
        )
        mapper = self._Mapper(
            [
                _controller_arm_mapper_command(
                    walking=True,
                    arm_tracking_enabled=True,
                    arm_joint_target=desired,
                ),
                _controller_arm_mapper_command(
                    walking=True,
                    arm_tracking_enabled=False,
                ),
                _controller_arm_mapper_command(
                    walking=False,
                    arm_tracking_enabled=False,
                ),
            ]
        )
        policy = self._policy(source, mapper)

        self.assertTrue(policy._read_command().arm_tracking_enabled)
        released = policy._read_command()
        self.assertTrue(released.enabled)
        self.assertEqual(released.twist, (0.35, 0.0, 0.0))
        self.assertFalse(released.arm_tracking_enabled)
        self.assertIsNone(released.arm_joint_target)

        invalid_after_release = policy._read_command()
        self.assertFalse(invalid_after_release.arm_tracking_enabled)
        self.assertIsNone(invalid_after_release.arm_joint_target)

    def test_invalid_tracking_hold_expires_at_the_bounded_deadline(self) -> None:
        desired = _arm_joint_target(0.5)
        start_ns = 1_000_000_000
        step_ns = 20_000_000
        hold_ns = int(ARM_TRACKING_HOLD_MAX_S * 1.0e9)
        offsets = list(range(step_ns, hold_ns + 1, step_ns))
        offsets.append(offsets[-1] + step_ns)
        invalid_frames = [
            self._frame(
                start_ns + offset,
                left_trigger=None,
                right_trigger=None,
                valid=False,
                fresh=False,
            )
            for offset in offsets
        ]
        invalid_mapping = _controller_arm_mapper_command(
            walking=False,
            arm_tracking_enabled=False,
        )
        source = self._Source(
            [
                self._frame(
                    start_ns,
                    left_trigger=0.0,
                    right_trigger=1.0,
                ),
                *invalid_frames,
            ]
        )
        mapper = self._Mapper(
            [
                _controller_arm_mapper_command(
                    walking=False,
                    arm_tracking_enabled=True,
                    arm_joint_target=desired,
                ),
                *[invalid_mapping for _offset in offsets],
            ]
        )
        policy = self._policy(source, mapper)

        self.assertTrue(policy._read_command().arm_tracking_enabled)
        for _offset in offsets[:-1]:
            held = policy._read_command()
            self.assertTrue(held.arm_tracking_enabled)
            self.assertEqual(held.arm_joint_target, desired)

        expired = policy._read_command()
        self.assertFalse(expired.arm_tracking_enabled)
        self.assertIsNone(expired.arm_joint_target)

    def test_transport_disconnect_clears_held_arm_target(self) -> None:
        desired = _arm_joint_target(0.5)
        source = self._Source(
            [
                self._frame(
                    1_000_000_000,
                    left_trigger=0.0,
                    right_trigger=1.0,
                ),
                RuntimeError("synthetic transport disconnect"),
                self._frame(
                    1_020_000_000,
                    left_trigger=None,
                    right_trigger=None,
                    valid=False,
                    fresh=False,
                    error="disconnected",
                ),
            ]
        )
        mapper = self._Mapper(
            [
                _controller_arm_mapper_command(
                    walking=False,
                    arm_tracking_enabled=True,
                    arm_joint_target=desired,
                ),
                _controller_arm_mapper_command(
                    walking=False,
                    arm_tracking_enabled=False,
                ),
            ]
        )
        policy = self._policy(source, mapper)

        stationary = policy._read_command()
        self.assertFalse(stationary.enabled)
        self.assertTrue(stationary.arm_tracking_enabled)
        self.assertEqual(stationary.arm_joint_target, desired)

        disconnected = policy._read_command()
        self.assertFalse(disconnected.enabled)
        self.assertFalse(disconnected.arm_tracking_enabled)
        self.assertIsNone(disconnected.arm_joint_target)
        self.assertIn("source read failed", disconnected.fault or "")
        self.assertEqual(mapper.reset_count, 1)

        still_invalid = policy._read_command()
        self.assertFalse(still_invalid.arm_tracking_enabled)
        self.assertIsNone(still_invalid.arm_joint_target)


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

    def test_left_trigger_alone_enables_after_native_mapper_rearm(self) -> None:
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
                    pose=SimpleNamespace(
                        position=(0.0, 0.0, 1.6),
                        orientation=head_orientation,
                    )
                ),
                left_controller=SimpleNamespace(
                    pose=SimpleNamespace(
                        position=(0.35, 0.25, 1.25),
                        orientation=(0.0, 0.0, 0.0, 1.0),
                    ),
                    axis=(0.0, 1.0),
                    trigger=trigger,
                    grip=0.0,
                    axis_click=False,
                    primary_button=False,
                    secondary_button=False,
                ),
                right_controller=SimpleNamespace(
                    pose=SimpleNamespace(
                        position=(0.35, -0.25, 1.25),
                        orientation=(0.0, 0.0, 0.0, 1.0),
                    ),
                    axis=(0.5, 0.0),
                    trigger=0.0,
                    grip=0.0,
                    axis_click=False,
                    primary_button=False,
                    secondary_button=False,
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
        # The native mapper now has one trigger-only hybrid mode; legacy-only
        # simulation adapts its output to the audited walk actor without
        # mutating the mapper's policy state.
        self.assertEqual(mapper.locomotion_policy, "pico_teleop")


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

    def test_trigger_release_uses_zero_twist_audited_standing_actor(self) -> None:
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
        seen: list[torch.Tensor] = []

        def walk_actor(observation: TensorDict) -> torch.Tensor:
            seen.append(observation["actor"].clone())
            return torch.full((1, 18), 0.25)

        policy.walk_actor = walk_actor
        observations = TensorDict({"actor": torch.zeros((1, 83))}, batch_size=(1,))
        action = policy(observations)
        self.assertEqual(len(seen), 1)
        self.assertTrue(torch.equal(seen[0][:, -3:], torch.zeros((1, 3))))
        self.assertTrue(torch.equal(action, torch.full((1, 18), 3.5)))
        self.assertTrue(torch.equal(policy.walk_last_action, torch.full((1, 18), 0.25)))

        policy(observations)
        self.assertTrue(
            torch.equal(seen[1][:, 42:60], torch.full((1, 18), 0.25))
        )

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

    def test_episode_reset_transition_preserves_held_trigger_command(self) -> None:
        held = SimulationCommand(
            enabled=True,
            twist=(0.2, 0.0, 0.0),
            foot_target=((0.0, 0.0, 0.0),) * 2,
            hand_target=((0.0, 0.0, 0.0),) * 2,
            hand_active=(False, False),
            head_orientation=(0.1, -0.2, 0.3),
            head_yaw_front=False,
            locomotion_policy="pico_teleop",
        )
        policy = self._policy(held)
        policy.native_legacy_action_semantics = False
        policy.env.episode_length_buf = torch.tensor([0])
        policy.env.reset_buf = torch.tensor([False])
        policy._previous_episode_length = torch.tensor([17])
        policy._pending_authority_token = object()
        policy._pending_legacy_command = held
        policy._legacy_fallback_latched = False
        policy._legacy_fallback_reason = None
        policy.mapper = SimpleNamespace(reset=lambda: self.fail("mapper was reset"))
        policy.legacy_fallback_mapper = SimpleNamespace(
            reset=lambda: self.fail("fallback mapper was reset")
        )
        policy.hmd_joint_ids = torch.tensor([0, 1, 2])
        policy.robot = SimpleNamespace(
            data=SimpleNamespace(joint_pos=torch.tensor([[0.1, 0.2, 0.3]]))
        )
        policy.hmd_current_target = torch.full((1, 3), 9.0)
        policy.walk_last_action.fill_(0.75)
        injected: list[SimulationCommand] = []
        policy._inject_command = injected.append
        policy.walk_actor = lambda _observation: torch.full((1, 18), 0.25)
        policy.actor = lambda _observation: torch.full((1, 18), 0.625)
        observations = TensorDict({"actor": torch.zeros((1, 83))}, batch_size=(1,))
        action = policy(observations)

        self.assertTrue(torch.equal(action, torch.full((1, 18), 0.625)))
        self.assertEqual(injected, [held])
        self.assertEqual(policy._previous_episode_length.tolist(), [0])
        self.assertTrue(
            torch.equal(policy.hmd_current_target, torch.tensor([[0.1, 0.2, 0.3]]))
        )
        self.assertTrue(torch.equal(policy.walk_last_action, torch.zeros((1, 18))))

    def test_trigger_release_remains_neutral_after_episode_reset(self) -> None:
        released = SimulationCommand(
            enabled=False,
            twist=(0.0, 0.0, 0.0),
            foot_target=((0.0, 0.0, 0.0),) * 2,
            hand_target=((0.0, 0.0, 0.0),) * 2,
            hand_active=(False, False),
            head_orientation=(0.0, 0.0, 0.0),
            head_yaw_front=False,
            locomotion_policy="walk",
        )
        policy = self._policy(released)
        policy.native_legacy_action_semantics = True
        policy.env.episode_length_buf = torch.tensor([0])
        policy.env.reset_buf = torch.tensor([True])
        policy._previous_episode_length = torch.tensor([12])
        policy._pending_authority_token = None
        policy._pending_legacy_command = None
        policy._legacy_fallback_latched = False
        policy._legacy_fallback_reason = None
        policy.mapper = SimpleNamespace(reset=lambda: self.fail("mapper was reset"))
        policy.legacy_fallback_mapper = None
        policy.hmd_joint_ids = torch.tensor([0, 1, 2])
        policy.robot = SimpleNamespace(
            data=SimpleNamespace(joint_pos=torch.zeros((1, 3)))
        )
        policy.hmd_current_target = torch.ones((1, 3))
        policy.walk_actor = lambda _observation: torch.full((1, 18), 0.125)
        observations = TensorDict({"actor": torch.zeros((1, 63))}, batch_size=(1,))

        action = policy(observations)

        self.assertTrue(torch.equal(action, torch.full((1, 18), 0.125)))
        self.assertTrue(torch.equal(policy.hmd_current_target, torch.zeros((1, 3))))

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
        policy.walk_actor = lambda _observation: torch.full((1, 18), 0.25)
        observations = TensorDict({"actor": torch.zeros((1, 83))}, batch_size=(1,))
        action = policy(observations)

        self.assertEqual(reset_count, 1)
        self.assertEqual(len(injected), 1)
        self.assertFalse(injected[0].enabled)
        self.assertIn("authority changed", injected[0].fault or "")
        self.assertTrue(torch.equal(action, torch.full((1, 18), 3.5)))


class CliSafetyTests(unittest.TestCase):
    def test_live_environment_auto_resets_falls_and_keeps_fall_termination(self) -> None:
        reset_base = SimpleNamespace(
            mode="reset",
            params={"pose_range": {"x": (-1.0, 1.0)}, "velocity_range": {"x": (-1.0, 1.0)}},
        )
        cfg = SimpleNamespace(
            scene=SimpleNamespace(num_envs=99),
            viewer=SimpleNamespace(
                body_name=None,
                distance=3.0,
                fovy=60.0,
                elevation=-15.0,
                azimuth=90.0,
            ),
            auto_reset=False,
            observations={"actor": SimpleNamespace(enable_corruption=True)},
            curriculum={"difficulty": object()},
            events={
                "reset_base": reset_base,
                "push_robot": SimpleNamespace(mode="interval"),
            },
            terminations={"time_out": object(), "fell_over": object()},
        )
        _configure_live_environment(cfg)
        self.assertTrue(cfg.auto_reset)
        self.assertIn("fell_over", cfg.terminations)
        self.assertNotIn("time_out", cfg.terminations)
        self.assertEqual(set(cfg.events), {"reset_base"})
        self.assertEqual(cfg.scene.num_envs, 1)
        self.assertEqual(cfg.viewer.body_name, "trunk")
        self.assertEqual(cfg.viewer.distance, 0.8)
        self.assertEqual(cfg.viewer.fovy, 45.0)
        self.assertEqual(cfg.viewer.elevation, -12.0)
        self.assertEqual(cfg.viewer.azimuth, 135.0)

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
        receipt = Path("acceptance.json")
        args = build_parser().parse_args(
            [
                "--v12-preview-checkpoint",
                str(checkpoint),
                "--v12-preview-acceptance-receipt",
                str(receipt),
                "--v12-preview-acceptance-receipt-sha256",
                "a" * 64,
                "--input",
                "pico-app",
            ]
        )
        self.assertEqual(_selected_checkpoint(args), checkpoint)
        self.assertEqual(args.v12_preview_acceptance_receipt, receipt)
        self.assertEqual(args.v12_preview_acceptance_receipt_sha256, "a" * 64)
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

    def test_controller_preview_has_distinct_checkpoint_and_receipt_options(
        self,
    ) -> None:
        checkpoint = Path("phase1.pt")
        receipt = Path("phase1_visual.json")
        args = build_parser().parse_args(
            [
                "--v12-controller-preview-checkpoint",
                str(checkpoint),
                "--v12-controller-preview-acceptance-receipt",
                str(receipt),
                "--v12-controller-preview-acceptance-receipt-sha256",
                "a" * 64,
                "--input",
                "pico-app",
            ]
        )
        self.assertEqual(_selected_checkpoint(args), checkpoint)
        self.assertEqual(
            args.v12_controller_preview_acceptance_receipt, receipt
        )
        self.assertEqual(
            _runtime_task(
                args.checkpoint,
                args.v12_preview_checkpoint,
                args.unaccepted_sim_preview_checkpoint,
                args.v12_controller_preview_checkpoint,
            ),
            "Mjlab-Teleop-V12-Preview-Microban",
        )
        with self.assertRaises(SystemExit):
            build_parser().parse_args(
                [
                    "--v12-controller-preview-checkpoint",
                    str(checkpoint),
                    "--v12-preview-checkpoint",
                    "fullbody.pt",
                ]
            )

    def test_unaccepted_preview_is_a_separate_explicit_hash_bound_cli(self) -> None:
        checkpoint = Path("unaccepted.pt")
        option = "--unaccepted-simulation-observation-only-v12-preview-checkpoint"
        sha_option = "--unaccepted-simulation-observation-only-v12-preview-sha256"
        args = build_parser().parse_args(
            [option, str(checkpoint), sha_option, "d" * 64, "--input", "pico-app"]
        )
        self.assertEqual(_selected_checkpoint(args), checkpoint)
        self.assertEqual(args.unaccepted_sim_preview_sha256, "d" * 64)
        self.assertEqual(
            _runtime_task(
                args.checkpoint,
                args.v12_preview_checkpoint,
                args.unaccepted_sim_preview_checkpoint,
            ),
            "Mjlab-Teleop-V12-Preview-Microban",
        )

        with self.assertRaises(SystemExit):
            build_parser().parse_args(
                [
                    "--v12-preview-checkpoint",
                    "accepted.pt",
                    option,
                    str(checkpoint),
                ]
            )

    def test_only_explicit_unaccepted_mode_selects_legacy_evidence_loader(
        self,
    ) -> None:
        module = "mjlab_microban.scripts.live_pico_teleop_sim"
        with (
            patch(f"{module}.make_teleop_v12_preview_consumer") as accepted,
            patch(
                f"{module}.make_teleop_v12_unaccepted_simulation_preview_consumer"
            ) as unaccepted,
        ):
            accepted.return_value = "accepted"
            unaccepted.return_value = "unaccepted"
            self.assertEqual(
                _construct_checkpoint_consumer_runner(
                    "env",
                    "cfg",
                    "cuda:0",
                    runtime_task="Mjlab-Teleop-V12-Preview-Microban",
                ),
                "accepted",
            )
            self.assertEqual(
                _construct_checkpoint_consumer_runner(
                    "env",
                    "cfg",
                    "cuda:0",
                    runtime_task="Mjlab-Teleop-V12-Preview-Microban",
                    unaccepted_preview_mode=True,
                ),
                "unaccepted",
            )
            accepted.assert_called_once_with("env", "cfg", "cuda:0")
            unaccepted.assert_called_once_with("env", "cfg", "cuda:0")

        with self.assertRaisesRegex(ValueError, "dedicated preview task"):
            _construct_checkpoint_consumer_runner(
                "env",
                "cfg",
                "cuda:0",
                runtime_task="Mjlab-Teleop-Microban",
                unaccepted_preview_mode=True,
            )

    def test_unaccepted_preview_requires_exact_hash_marker_and_clock(self) -> None:
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model_10100.pt"
            marker = staged_preview_info(
                phase=TELEOP_V12_PREVIEW_PHASE_FULL_BODY,
                phase_source_checkpoint_sha256="b" * 64,
                phase1_acceptance_receipt_sha256="c" * 64,
                phase1_quality_class=TELEOP_V12_PREVIEW_PHASE1_STRICT_QUALITY,
            )
            torch.save(
                {
                    "iter": 10_100,
                    "infos": {
                        "preview_non_deployable": True,
                        "teleop_v12_preview": marker,
                        "env_state": {"common_step_counter": 10_101 * 24},
                    },
                },
                checkpoint,
            )
            digest = _sha256(checkpoint)
            authority = _validate_unaccepted_simulation_preview(
                checkpoint=checkpoint,
                expected_checkpoint_sha256=digest,
            )
            self.assertEqual(authority.checkpoint_sha256, digest)
            self.assertEqual(authority.iteration, 10_100)
            checkpoint.write_bytes(b"replacement path bytes")
            immutable_payload = torch.load(
                BytesIO(authority.checkpoint_bytes),
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(hashlib.sha256(authority.checkpoint_bytes).hexdigest(), digest)
            self.assertEqual(immutable_payload["iter"], 10_100)

            torch.save(
                {
                    "iter": 10_100,
                    "infos": {
                        "preview_non_deployable": True,
                        "teleop_v12_preview": marker,
                        "env_state": {"common_step_counter": 10_101 * 24},
                    },
                },
                checkpoint,
            )

            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                _validate_unaccepted_simulation_preview(
                    checkpoint=checkpoint,
                    expected_checkpoint_sha256="0" * 64,
                )
            with self.assertRaisesRegex(ValueError, "exact checkpoint SHA-256"):
                _validate_unaccepted_simulation_preview(
                    checkpoint=checkpoint,
                    expected_checkpoint_sha256=None,
                )

    def test_unaccepted_preview_rejects_acceptance_receipt_options(self) -> None:
        option = "--unaccepted-simulation-observation-only-v12-preview-checkpoint"
        sha_option = "--unaccepted-simulation-observation-only-v12-preview-sha256"
        args = build_parser().parse_args(
            [
                option,
                "unaccepted.pt",
                sha_option,
                "d" * 64,
                "--v12-preview-acceptance-receipt",
                "must-not-be-used.json",
                "--v12-preview-acceptance-receipt-sha256",
                "e" * 64,
            ]
        )
        with (
            patch(
                "mjlab_microban.scripts.live_pico_teleop_sim."
                "configure_torch_backends"
            ),
            self.assertRaisesRegex(ValueError, "accepted.*path"),
        ):
            run_live_simulation(args)

    def test_unaccepted_preview_status_is_visually_loud(self) -> None:
        policy = object.__new__(LivePicoSimulationPolicy)
        policy.clock_ns = lambda: 2_000_000_000
        policy.status_period_ns = 1
        policy._last_status_ns = 0
        policy._camera_fault = None
        policy.unaccepted_observation_only = True
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
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            policy._print_status(command)
        self.assertIn(UNACCEPTED_SIMULATION_WARNING, output.getvalue())
        self.assertIn("SIMULATION ONLY", output.getvalue())

    def test_status_distinguishes_hand_input_from_measured_arm_response(self) -> None:
        policy = object.__new__(LivePicoSimulationPolicy)
        policy.clock_ns = lambda: 2_000_000_000
        policy.status_period_ns = 1
        policy._last_status_ns = 0
        policy._camera_fault = None
        policy.unaccepted_observation_only = False
        policy.robot = SimpleNamespace(
            data=SimpleNamespace(
                joint_pos=torch.deg2rad(
                    torch.tensor([[10.0, -20.0, 30.0, -5.0, 0.0, 5.0]])
                ),
                default_joint_pos=torch.zeros((1, 6)),
            )
        )
        policy.arm_joint_ids = {
            "left": torch.tensor([0, 1, 2]),
            "right": torch.tensor([3, 4, 5]),
        }
        policy._last_controller_hand_hmd_m = (
            (0.1, -0.2, 0.3),
            (-0.4, 0.5, -0.6),
        )
        policy._last_absolute_arm_target_source = (
            "current_hmd_to_controller_absolute"
        )
        policy._last_absolute_arm_rejection = None
        command = SimulationCommand(
            enabled=True,
            twist=(0.0, 0.0, 0.0),
            foot_target=((0.0, 0.0, 0.0),) * 2,
            hand_target=((0.12345, -0.2, 0.0), (-0.01111, 0.02222, 0.03333)),
            hand_active=(True, False),
            head_orientation=(0.0, 0.0, 0.0),
            head_yaw_front=False,
            locomotion_policy="pico_teleop",
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            policy._print_status(command)
        status = output.getvalue()
        self.assertIn("hand_cmd_m=L(0.123, -0.2, 0.0)/R(-0.011, 0.022, 0.033)", status)
        self.assertIn("hand_active=LTrue/RFalse", status)
        self.assertIn(
            "arm_delta_deg=L(10.0, -20.0, 30.0)/R(-5.0, 0.0, 5.0)",
            status,
        )
        self.assertIn(
            "controller_hmd_abs_m=L(0.1, -0.2, 0.3)/R(-0.4, 0.5, -0.6)",
            status,
        )
        self.assertIn(
            "arm_target_abs_deg=L(0.0, 10.0, -20.0)/R(0.0, -10.0, -20.0)",
            status,
        )
        self.assertIn(
            "arm_actual_abs_deg=L(10.0, -20.0, 30.0)/R(-5.0, 0.0, 5.0)",
            status,
        )
        self.assertIn("hand_actual_m=unavailable", status)
        self.assertIn("hand_error_m=unavailable", status)

    def test_status_reports_cartesian_hand_tracking_not_only_joint_motion(self) -> None:
        policy = object.__new__(LivePicoSimulationPolicy)
        policy.clock_ns = lambda: 2_000_000_000
        policy.status_period_ns = 1
        policy._last_status_ns = 0
        policy._camera_fault = None
        policy.unaccepted_observation_only = False
        policy._measured_arm_joint_deviation_deg = lambda: None

        hand = object.__new__(ResetFixedHandTargetCommand)
        hand._default_hand_pos_b = torch.tensor(
            [[[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]]]
        )
        hand.current_hand_pos_b = lambda: torch.tensor(
            [[[0.13, 0.18, 0.31], [-0.11, -0.16, -0.27]]]
        )
        policy.env = SimpleNamespace(
            command_manager=SimpleNamespace(get_term=lambda name: hand)
        )
        command = SimulationCommand(
            enabled=True,
            twist=(0.0, 0.0, 0.0),
            foot_target=((0.0, 0.0, 0.0),) * 2,
            hand_target=((0.02, -0.01, 0.0), (-0.02, 0.05, 0.01)),
            hand_active=(True, True),
            head_orientation=(0.0, 0.0, 0.0),
            head_yaw_front=False,
            locomotion_policy="pico_teleop",
        )

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            policy._print_status(command)

        status = output.getvalue()
        self.assertIn(
            "hand_actual_m=L(0.03, -0.02, 0.01)/R(-0.01, 0.04, 0.03)",
            status,
        )
        self.assertIn("hand_error_m=L0.017/R0.024", status)

    def test_preview_live_authority_binds_receipt_checkpoint_marker_and_clock(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model_10100.pt"
            marker = staged_preview_info(
                phase=TELEOP_V12_PREVIEW_PHASE_FULL_BODY,
                phase_source_checkpoint_sha256="b" * 64,
                phase1_acceptance_receipt_sha256="c" * 64,
                phase1_quality_class=TELEOP_V12_PREVIEW_PHASE1_STRICT_QUALITY,
            )
            torch.save(
                {
                    "iter": 10_100,
                    "infos": {
                        "preview_non_deployable": True,
                        "teleop_v12_preview": marker,
                        "env_state": {"common_step_counter": 10_101 * 24},
                    },
                },
                checkpoint,
            )
            checkpoint_sha256 = _sha256(checkpoint)
            receipt = root / "acceptance.json"
            receipt_payload = {
                "gate": "microban_teleop_v12_nondeployable_preview_acceptance",
                "status": "pass",
                "live_simulation_candidate": True,
                "quality_class": TELEOP_V12_PREVIEW_FULLBODY_STRICT_QUALITY,
                "checkpoint": {
                    "sha256": checkpoint_sha256,
                    "iteration": 10_100,
                },
                "teleop_v12_preview": marker,
            }
            receipt.write_text(json.dumps(receipt_payload))
            receipt_sha256 = _sha256(receipt)
            with (
                patch(
                    "mjlab_microban.scripts.live_pico_teleop_sim."
                    "validate_embedded_phase1_acceptance",
                    return_value={},
                ),
                patch(
                    "mjlab_microban.scripts.live_pico_teleop_sim."
                    "validate_preview_evaluation_report",
                    return_value=receipt_payload,
                ) as validate,
            ):
                authority = _validate_preview_live_authority(
                    checkpoint=checkpoint,
                    acceptance_receipt=receipt,
                    expected_receipt_sha256=receipt_sha256,
                )
            self.assertEqual(authority.checkpoint_sha256, checkpoint_sha256)
            self.assertEqual(
                authority.quality_class,
                TELEOP_V12_PREVIEW_FULLBODY_STRICT_QUALITY,
            )
            validate.assert_called_once_with(
                receipt_payload,
                checkpoint_sha256=checkpoint_sha256,
                iteration=10_100,
                marker=marker,
            )

            receipt_payload["gate"] = (
                "microban_teleop_v12_preview_fullbody_visual_acceptance"
            )
            receipt_payload["quality_class"] = (
                TELEOP_V12_PREVIEW_FULLBODY_VISUAL_QUALITY
            )
            receipt.write_text(json.dumps(receipt_payload))
            visual_receipt_sha256 = _sha256(receipt)
            with (
                patch(
                    "mjlab_microban.scripts.live_pico_teleop_sim."
                    "validate_embedded_phase1_acceptance",
                    return_value={},
                ),
                patch(
                    "mjlab_microban.scripts.live_pico_teleop_sim."
                    "validate_visual_promotion_report",
                    return_value=receipt_payload,
                ) as validate_visual,
            ):
                visual_authority = _validate_preview_live_authority(
                    checkpoint=checkpoint,
                    acceptance_receipt=receipt,
                    expected_receipt_sha256=visual_receipt_sha256,
                )
            self.assertEqual(
                visual_authority.quality_class,
                TELEOP_V12_PREVIEW_FULLBODY_VISUAL_QUALITY,
            )
            validate_visual.assert_called_once_with(
                receipt_payload,
                checkpoint_sha256=checkpoint_sha256,
                iteration=10_100,
                marker=marker,
            )

            receipt_payload["checkpoint"]["sha256"] = "d" * 64
            receipt.write_text(json.dumps(receipt_payload))
            with self.assertRaisesRegex(ValueError, "does not match"):
                _validate_preview_live_authority(
                    checkpoint=checkpoint,
                    acceptance_receipt=receipt,
                    expected_receipt_sha256=_sha256(receipt),
                )

    def test_preview_live_authority_requires_receipt_and_sha(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires its final PASS"):
            _validate_preview_live_authority(
                checkpoint=Path("model_10100.pt"),
                acceptance_receipt=None,
                expected_receipt_sha256=None,
            )

    def test_controller_preview_authority_binds_phase1_visual_receipt(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model_7100.pt"
            marker = staged_preview_info(
                phase=TELEOP_V12_PREVIEW_PHASE_HMD_HAND,
                phase_source_checkpoint_sha256=(
                    "ab0dbe0db9cadd6bb937e5eebf3d2b8f6bb3fadcc5e025aa15d28fb87e85d3a2"
                ),
            )
            torch.save(
                {
                    "iter": 7_100,
                    "infos": {
                        "preview_non_deployable": True,
                        "teleop_v12_preview": marker,
                        "env_state": {"common_step_counter": 7_101 * 24},
                    },
                },
                checkpoint,
            )
            checkpoint_sha256 = _sha256(checkpoint)
            receipt_payload = {
                "gate": "microban_teleop_v12_preview_phase1_visual_promotion",
                "status": "pass",
                "simulation_only": True,
                "canonical_deployment_accepted": False,
                "quality_class": TELEOP_V12_PREVIEW_PHASE1_VISUAL_QUALITY,
                "checkpoint": {
                    "sha256": checkpoint_sha256,
                    "iteration": 7_100,
                },
                "teleop_v12_preview": marker,
            }
            receipt = root / "phase1_visual.json"
            receipt.write_text(json.dumps(receipt_payload))
            with patch(
                "mjlab_microban.scripts.live_pico_teleop_sim."
                "validate_visual_promotion_report",
                return_value=receipt_payload,
            ) as validate:
                authority = _validate_controller_preview_live_authority(
                    checkpoint=checkpoint,
                    acceptance_receipt=receipt,
                    expected_receipt_sha256=_sha256(receipt),
                )
            self.assertEqual(authority.checkpoint_sha256, checkpoint_sha256)
            self.assertEqual(
                authority.quality_class,
                TELEOP_V12_PREVIEW_PHASE1_VISUAL_QUALITY,
            )
            validate.assert_called_once_with(
                receipt_payload,
                checkpoint_sha256=checkpoint_sha256,
                iteration=7_100,
                marker=marker,
            )

            receipt_payload["teleop_v12_preview"] = staged_preview_info(
                phase=TELEOP_V12_PREVIEW_PHASE_FULL_BODY,
                phase_source_checkpoint_sha256="b" * 64,
                phase1_acceptance_receipt_sha256="c" * 64,
                phase1_quality_class=TELEOP_V12_PREVIEW_PHASE1_STRICT_QUALITY,
            )
            receipt.write_text(json.dumps(receipt_payload))
            with self.assertRaisesRegex(ValueError, "marker does not match"):
                _validate_controller_preview_live_authority(
                    checkpoint=checkpoint,
                    acceptance_receipt=receipt,
                    expected_receipt_sha256=_sha256(receipt),
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
