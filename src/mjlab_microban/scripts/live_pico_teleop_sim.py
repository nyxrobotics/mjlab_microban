# Copyright 2026 Marc Duclusaud
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Drive the Microban MJLab simulation from a live PICO 4 Ultra stream.

This entry point is deliberately simulation-only.  It has no robot address,
UDP sender, motor controller, or deployment option.  Legacy XRoboToolkit or
authenticated Microban Unity frames pass through ``microban_teleop``'s
validated native mapper. Without a hybrid checkpoint, the audited legacy actor
runs in its original 63-observation ``Mjlab-Velocity-Microban`` environment and
its raw 18 actions are executed unchanged. Supplying a hybrid checkpoint keeps
the separate 83-observation teleoperation environment.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import importlib
import math
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from mjlab_microban.scripts.simulation_camera import (
    EYE_ASPECT,
    EYE_HEIGHT_PX,
    EYE_WIDTH_PX,
    HORIZONTAL_FOV_DEG,
    TAN_BOUNDS,
    VERTICAL_FOV_DEG,
    StereoMjpegPublisher,
    webxr_camera_toml,
)
from mjlab_microban.tasks.mdp import UniformVelocityCommandWithRotation
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_HMD_JOINT_NAMES,
    MICROBAN_TELEOP_ACTION_WIDTH,
    MICROBAN_TELEOP_OBSERVATION_SCHEMA,
    validate_microban_teleop_observation_contract,
)
from mjlab_microban.tasks.microban_teleop_mdp import (
    MICROBAN_HMD_RUNTIME_LIMITS_RAD,
    MICROBAN_HMD_SLEW_RATES_RAD_S,
    MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M,
    ResetFixedFootTargetCommand,
    ResetFixedHandTargetCommand,
)

TASK = "Mjlab-Teleop-Microban"
WALK_TASK = "Mjlab-Velocity-Microban"
AUDITED_LEGACY_WALK_SHA256 = (
    "b0bcdadac39716be784207dd6b2b93157162a3e80650e23c05f490c400b9e141"
)
MAX_FRAME_GAP_S = 0.1
MAX_FRAME_AGE_S = 0.1
LEFT_TRIGGER_RELEASE_THRESHOLD = 0.45

FORWARD_MAX_M_S = 0.7
BACKWARD_MAX_M_S = 0.5
LATERAL_MAX_M_S = 0.3
MOVING_YAW_MAX_RAD_S = 1.5
STATIONARY_YAW_MAX_RAD_S = 3.0

TARGET_SAFETY_MARGIN = 0.8
FOOT_LOWER_M = tuple(value * TARGET_SAFETY_MARGIN for value in (-0.03, -0.03, 0.0))
FOOT_UPPER_M = tuple(value * TARGET_SAFETY_MARGIN for value in (0.03, 0.03, 0.05))
SIMULTANEOUS_FEET_LOWER_M = tuple(
    value * TARGET_SAFETY_MARGIN for value in (-0.01, -0.01, 0.0)
)
SIMULTANEOUS_FEET_UPPER_M = tuple(
    value * TARGET_SAFETY_MARGIN for value in (0.01, 0.01, 0.02)
)
FOOT_INACTIVE_Z_MAX_M = MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M
HAND_LOWER_M = tuple(value * TARGET_SAFETY_MARGIN for value in (-0.08, -0.08, -0.08))
HAND_UPPER_M = tuple(value * TARGET_SAFETY_MARGIN for value in (0.08, 0.08, 0.08))


class _Source(Protocol):
    strict_control_presence: bool

    def read(self) -> Any: ...

    def close(self) -> None: ...


class _Mapper(Protocol):
    def map_sample(self, frame: Any) -> dict[str, Any]: ...

    def reset(self) -> None: ...

    @staticmethod
    def neutral() -> dict[str, Any]: ...


class _FieldOverrideView:
    """Read-only field view used to sanitize one mapper input without mutation."""

    def __init__(self, value: Any, /, **overrides: Any) -> None:
        self._value = value
        self._overrides = overrides

    def __getattr__(self, name: str) -> Any:
        if name in self._overrides:
            return self._overrides[name]
        if isinstance(self._value, Mapping):
            try:
                return self._value[name]
            except KeyError as exc:
                raise AttributeError(name) from exc
        return getattr(self._value, name)


def _legacy_walk_input_frame(frame: Any) -> Any:
    """Hide the hybrid selector before the native mapper's state machine runs."""

    if isinstance(frame, Mapping):
        left = frame.get("left_controller")
    else:
        left = getattr(frame, "left_controller", None)
    if left is None:
        return frame
    legacy_left = _FieldOverrideView(left, primary_button=False)
    return _FieldOverrideView(frame, left_controller=legacy_left)


def _legacy_walk_mapping(mapped: Any) -> Any:
    """Return a legacy-only view of an already mapped controller snapshot."""

    if not isinstance(mapped, Mapping):
        return mapped
    return {
        **mapped,
        "locomotion_policy": "walk",
        "foot_target": None,
        "hand_target": None,
        "hand_active": {"left": False, "right": False},
        "body_target_calibrated": False,
        "body_target_fresh": False,
    }


def _left_trigger_explicitly_released(frame: Any, mapped: Any) -> bool:
    """Recognize a real deadman release without treating data loss as one."""

    health = (
        frame.get("controller_health")
        if isinstance(frame, Mapping)
        else getattr(frame, "controller_health", None)
    )
    if health is not None:
        fresh = (
            health.get("fresh")
            if isinstance(health, Mapping)
            else getattr(health, "fresh", False)
        )
        valid = (
            health.get("valid")
            if isinstance(health, Mapping)
            else getattr(health, "valid", False)
        )
        if not bool(fresh) or not bool(valid):
            return False

    if isinstance(frame, Mapping):
        left = frame.get("left_controller")
    else:
        left = getattr(frame, "left_controller", None)
    if left is not None:
        trigger = (
            left.get("trigger")
            if isinstance(left, Mapping)
            else getattr(left, "trigger", None)
        )
        value = _finite_number(trigger)
        return value is not None and value <= LEFT_TRIGGER_RELEASE_THRESHOLD

    # WebXR hands off an already mapped command rather than a native controller
    # object. A fresh mapped snapshot with no walk move is its explicit release.
    if isinstance(frame, _WebXrFrame) and isinstance(mapped, Mapping):
        moves = mapped.get("active_moves")
        return (
            isinstance(moves, Sequence)
            and not isinstance(moves, (str, bytes, bytearray))
            and "walk" not in moves
        )
    return False


@dataclass(frozen=True)
class _WebXrFrame:
    sampled_at_ns: int
    command: dict[str, Any]


class WebXrSimulationSource:
    """Thread-safe in-process handoff from WebXR to the simulation loop."""

    strict_control_presence = True

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: _WebXrFrame | None = None
        self._reset_callback: Any = None

    def set_reset_callback(self, callback: Any) -> None:
        self._reset_callback = callback

    def publish(self, command: dict[str, Any], sampled_at_ns: int) -> None:
        if not isinstance(sampled_at_ns, int) or sampled_at_ns < 0:
            return
        frame = _WebXrFrame(sampled_at_ns, copy.deepcopy(command))
        with self._lock:
            self._latest = frame

    def read(self) -> _WebXrFrame:
        with self._lock:
            frame = self._latest
        if frame is None:
            raise RuntimeError("no WebXR control snapshot has arrived")
        return frame

    def request_reset(self) -> None:
        with self._lock:
            self._latest = None
        if self._reset_callback is not None:
            self._reset_callback()

    def close(self) -> None:
        with self._lock:
            self._latest = None


class WebXrSimulationMapper:
    """Adapt controller-only WebXR snapshots to neutral body targets."""

    def __init__(self, source: WebXrSimulationSource) -> None:
        self.source = source

    def map_sample(self, frame: _WebXrFrame) -> dict[str, Any]:
        command = copy.deepcopy(frame.command)
        # WebXR has no Motion Tracker body skeleton.  The dedicated hybrid policy
        # still receives its full observation contract, with zero relative limb
        # targets (the calibrated/initial pose), while sticks and HMD remain live.
        if command.get("locomotion_policy") == "pico_teleop":
            zero_pair = {"left": [0.0, 0.0, 0.0], "right": [0.0, 0.0, 0.0]}
            command["foot_target"] = copy.deepcopy(zero_pair)
            command["hand_target"] = copy.deepcopy(zero_pair)
            command["hand_active"] = {"left": False, "right": False}
            command["body_target_calibrated"] = True
            command["body_target_fresh"] = True
        return command

    def reset(self) -> None:
        self.source.request_reset()

    @staticmethod
    def neutral() -> dict[str, Any]:
        return {
            "velocity": {"vx": 0.0, "vy": 0.0, "vtheta": 0.0},
            "active_moves": [],
            "locomotion_policy": "walk",
            "head_orientation": None,
            "head_yaw_front": False,
            "foot_target": None,
            "hand_target": None,
        }


class _WebServerThread:
    """Run the loopback-only PICO WebXR app beside the blocking viewer."""

    def __init__(self, application: Any, *, port: int) -> None:
        self.application = application
        self.port = port
        self._ready = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._runner: Any = None
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="microban-sim-webxr",
            daemon=True,
        )

    def _run(self) -> None:
        from aiohttp import web

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        runner = web.AppRunner(self.application)
        self._runner = runner
        try:
            loop.run_until_complete(runner.setup())
            site = web.TCPSite(runner, host="127.0.0.1", port=self.port)
            loop.run_until_complete(site.start())
        except BaseException as exc:  # noqa: BLE001 - propagate thread startup
            self._error = exc
            self._ready.set()
            loop.run_until_complete(runner.cleanup())
            loop.close()
            return
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            loop.run_until_complete(runner.cleanup())
            loop.close()

    def start(self) -> None:
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            raise RuntimeError("WebXR simulation server did not start")
        if self._error is not None:
            raise RuntimeError("WebXR simulation server failed") from self._error

    def close(self) -> None:
        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)


class _NativeServerThread:
    """Run the authenticated PICO listener beside the blocking viewer."""

    def __init__(self, server: Any) -> None:
        self.server = server
        self._ready = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="microban-sim-pico-native",
            daemon=True,
        )

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self.server.start())
        except BaseException as exc:  # noqa: BLE001 - propagate thread startup
            self._error = exc
            self._ready.set()
            loop.run_until_complete(self.server.close())
            loop.close()
            return
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            loop.run_until_complete(self.server.close())
            loop.close()

    def start(self) -> None:
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            raise RuntimeError("authenticated PICO input server did not start")
        if self._error is not None:
            raise RuntimeError("authenticated PICO input server failed") from self._error

    def close(self) -> None:
        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            raise RuntimeError("authenticated PICO input server did not stop")


@dataclass(frozen=True)
class SimulationCommand:
    """Validated physical-unit command consumed by the live policy wrapper."""

    enabled: bool
    twist: tuple[float, float, float]
    foot_target: tuple[tuple[float, float, float], tuple[float, float, float]]
    hand_target: tuple[tuple[float, float, float], tuple[float, float, float]]
    hand_active: tuple[bool, bool]
    head_orientation: tuple[float, float, float]
    head_yaw_front: bool
    locomotion_policy: str
    fault: str | None = None


@dataclass(frozen=True)
class LegacyWalkActor:
    policy: Any
    joint_names: tuple[str, ...]
    scale: torch.Tensor
    offset: torch.Tensor


def _zero_targets() -> tuple[
    tuple[tuple[float, float, float], tuple[float, float, float]],
    tuple[tuple[float, float, float], tuple[float, float, float]],
]:
    zero = (0.0, 0.0, 0.0)
    return (zero, zero), (zero, zero)


def neutral_simulation_command(*, fault: str | None = None) -> SimulationCommand:
    feet, hands = _zero_targets()
    return SimulationCommand(
        enabled=False,
        twist=(0.0, 0.0, 0.0),
        foot_target=feet,
        hand_target=hands,
        hand_active=(False, False),
        head_orientation=(0.0, 0.0, 0.0),
        head_yaw_front=False,
        locomotion_policy="walk",
        fault=fault,
    )


def _command_with_fault(
    command: SimulationCommand, fault: str | None
) -> SimulationCommand:
    return SimulationCommand(**{**asdict(command), "fault": fault})


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _bounded_pair(
    value: Any,
    lower: tuple[float, float, float],
    upper: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    if not isinstance(value, Mapping) or set(value) != {"left", "right"}:
        return None
    result: list[tuple[float, float, float]] = []
    for side in ("left", "right"):
        vector = value[side]
        if (
            not isinstance(vector, Sequence)
            or isinstance(vector, (str, bytes, bytearray))
            or len(vector) != 3
        ):
            return None
        parsed: list[float] = []
        for index, item in enumerate(vector):
            number = _finite_number(item)
            if number is None or not lower[index] <= number <= upper[index]:
                return None
            parsed.append(number)
        result.append((parsed[0], parsed[1], parsed[2]))
    return result[0], result[1]


def _nonzero_target(value: Sequence[float]) -> bool:
    return any(abs(float(component)) > 1.0e-6 for component in value)


def _twist_is_exactly_zero(value: Sequence[float]) -> bool:
    """Match the physical receiver's stationary both-feet contract exactly."""

    return len(value) == 3 and all(float(component) == 0.0 for component in value)


def scale_normalized_velocity(value: Any) -> tuple[float, float, float] | None:
    """Apply the physical runtime's asymmetric stick limits exactly once."""

    if not isinstance(value, Mapping):
        return None
    normalized: list[float] = []
    for name in ("vx", "vy", "vtheta"):
        number = _finite_number(value.get(name, 0.0))
        if number is None or not -1.0 <= number <= 1.0:
            return None
        normalized.append(number)
    vx, vy, yaw = normalized
    moving = abs(vx) > 1.0e-6 or abs(vy) > 1.0e-6
    vx_limit = FORWARD_MAX_M_S if vx >= 0.0 else BACKWARD_MAX_M_S
    yaw_limit = MOVING_YAW_MAX_RAD_S if moving else STATIONARY_YAW_MAX_RAD_S
    return vx * vx_limit, vy * LATERAL_MAX_M_S, yaw * yaw_limit


def command_for_simulation(
    command: Any, *, legacy_walk_available: bool = False
) -> SimulationCommand:
    """Convert a native mapper snapshot and fail closed on any contract fault."""

    if not isinstance(command, Mapping):
        return neutral_simulation_command(fault="mapper output is not a mapping")

    policy = command.get("locomotion_policy", "walk")
    if policy not in {"walk", "pico_teleop"}:
        return neutral_simulation_command(fault="unknown locomotion policy")

    moves = command.get("active_moves", ())
    if not isinstance(moves, Sequence) or isinstance(moves, (str, bytes, bytearray)):
        return neutral_simulation_command(fault="active_moves is malformed")

    orientation_value = command.get("head_orientation")
    orientation = (0.0, 0.0, 0.0)
    if orientation_value is not None:
        if not isinstance(orientation_value, Mapping):
            return neutral_simulation_command(fault="head_orientation is malformed")
        parsed_orientation = tuple(
            _finite_number(orientation_value.get(axis, 0.0))
            for axis in ("roll", "pitch", "yaw")
        )
        if any(value is None for value in parsed_orientation):
            return neutral_simulation_command(fault="head_orientation is non-finite")
        orientation = (
            float(parsed_orientation[0]),
            float(parsed_orientation[1]),
            float(parsed_orientation[2]),
        )

    base = neutral_simulation_command()
    base = SimulationCommand(
        **{
            **asdict(base),
            "head_orientation": orientation,
            "head_yaw_front": bool(command.get("head_yaw_front", False)),
            "locomotion_policy": str(policy),
        }
    )

    walking = "walk" in moves
    if not walking:
        return base

    velocity = scale_normalized_velocity(command.get("velocity"))
    if velocity is None:
        return SimulationCommand(
            **{**asdict(base), "fault": "live velocity exceeds the simulation contract"}
        )
    if policy == "walk":
        if not legacy_walk_available:
            return SimulationCommand(
                **{**asdict(base), "fault": "legacy walk actor is unavailable"}
            )
        return SimulationCommand(
            enabled=True,
            twist=velocity,
            foot_target=base.foot_target,
            hand_target=base.hand_target,
            hand_active=(False, False),
            head_orientation=orientation,
            head_yaw_front=bool(command.get("head_yaw_front", False)),
            locomotion_policy="walk",
        )

    if command.get("body_target_calibrated") is not True:
        return SimulationCommand(
            **{**asdict(base), "fault": "body targets are not calibrated"}
        )
    if command.get("body_target_fresh") is not True:
        return SimulationCommand(**{**asdict(base), "fault": "body targets are stale"})

    feet = _bounded_pair(command.get("foot_target"), FOOT_LOWER_M, FOOT_UPPER_M)
    hands = _bounded_pair(command.get("hand_target"), HAND_LOWER_M, HAND_UPPER_M)
    if velocity is None or feet is None or hands is None:
        return SimulationCommand(
            **{**asdict(base), "fault": "live command exceeds the simulation contract"}
        )

    projected_feet = tuple(
        (0.0, 0.0, 0.0) if foot[2] <= FOOT_INACTIVE_Z_MAX_M else foot for foot in feet
    )
    both_feet_active = all(_nonzero_target(foot) for foot in projected_feet)
    if both_feet_active:
        both_feet_bounded = _bounded_pair(
            {"left": projected_feet[0], "right": projected_feet[1]},
            SIMULTANEOUS_FEET_LOWER_M,
            SIMULTANEOUS_FEET_UPPER_M,
        )
        if both_feet_bounded is None or not _twist_is_exactly_zero(velocity):
            return SimulationCommand(
                **{
                    **asdict(base),
                    "fault": "simultaneous both-feet targets require conservative "
                    "bounds and zero twist",
                }
            )

    hand_active_value = command.get("hand_active")
    hand_active = (True, True)
    if hand_active_value is not None:
        if (
            not isinstance(hand_active_value, Mapping)
            or set(hand_active_value) != {"left", "right"}
            or not all(
                isinstance(hand_active_value[side], bool) for side in ("left", "right")
            )
        ):
            return SimulationCommand(
                **{**asdict(base), "fault": "hand_active is malformed"}
            )
        hand_active = (
            hand_active_value["left"],
            hand_active_value["right"],
        )

    return SimulationCommand(
        enabled=True,
        twist=velocity,
        foot_target=(projected_feet[0], projected_feet[1]),
        hand_target=hands,
        hand_active=hand_active,
        head_orientation=orientation,
        head_yaw_front=bool(command.get("head_yaw_front", False)),
        locomotion_policy=str(policy),
    )


def _rx(angle: float) -> tuple[tuple[float, float, float], ...]:
    cosine, sine = math.cos(angle), math.sin(angle)
    return ((1.0, 0.0, 0.0), (0.0, cosine, -sine), (0.0, sine, cosine))


def _ry(angle: float) -> tuple[tuple[float, float, float], ...]:
    cosine, sine = math.cos(angle), math.sin(angle)
    return ((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine))


def _rz(angle: float) -> tuple[tuple[float, float, float], ...]:
    cosine, sine = math.cos(angle), math.sin(angle)
    return ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))


def _matmul(
    a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]
) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        tuple(sum(a[row][k] * b[k][column] for k in range(3)) for column in range(3))
        for row in range(3)
    )


def _transpose(
    value: Sequence[Sequence[float]],
) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(value[column][row] for column in range(3)) for row in range(3))


def solve_hmd_neck_target(
    head_orientation: tuple[float, float, float],
    body_quat_wxyz: Sequence[float],
    *,
    yaw_front: bool,
) -> tuple[float, float, float] | None:
    """Match the physical robot's gravity-compensated Z-X-Y neck solver."""

    if len(body_quat_wxyz) != 4:
        return None
    quaternion = tuple(_finite_number(value) for value in body_quat_wxyz)
    if any(value is None for value in quaternion):
        return None
    w, x, y, z = (float(value) for value in quaternion)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1.0e-6:
        return None
    w, x, y, z = (value / norm for value in (w, x, y, z))
    trunk_roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    trunk_pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))

    hmd_roll, hmd_pitch, hmd_yaw = head_orientation
    camera_yaw = 0.0 if yaw_front else hmd_yaw
    desired_camera = _matmul(_matmul(_rz(camera_yaw), _rx(hmd_roll)), _ry(hmd_pitch))
    trunk_tilt = _matmul(_ry(trunk_pitch), _rx(trunk_roll))
    neck = _matmul(_transpose(trunk_tilt), desired_camera)
    solved_roll = math.asin(max(-1.0, min(1.0, neck[2][1])))
    solved_pitch = math.atan2(-neck[2][0], neck[2][2])
    solved_yaw = math.atan2(-neck[0][1], neck[1][1])
    return (0.0 if yaw_front else solved_yaw), solved_roll, solved_pitch


def _patch_command_observation(observations: Any, env: ManagerBasedRlEnv) -> Any:
    command_values = {
        "command": env.command_manager.get_term("twist").command,
        "foot_target": env.command_manager.get_term("foot_target").command,
        "hand_target": env.command_manager.get_term("hand_target").command,
    }
    patched = observations.clone()
    actor = patched["actor"].clone()
    offset = 0
    for name, width in MICROBAN_TELEOP_OBSERVATION_SCHEMA:
        if name in command_values:
            value = command_values[name]
            if value.shape != (env.num_envs, width):
                raise ValueError(
                    f"Unexpected {name} observation shape: {tuple(value.shape)}"
                )
            actor[:, offset : offset + width] = value
        offset += width
    patched["actor"] = actor
    return patched


def _patch_native_walk_observation(
    observations: Any, env: ManagerBasedRlEnv
) -> Any:
    """Patch only the live twist in the original 63-wide velocity observation."""

    patched = observations.clone()
    actor = patched["actor"].clone()
    if tuple(actor.shape) != (env.num_envs, 63):
        raise ValueError(
            f"Unexpected native legacy observation shape: {tuple(actor.shape)}"
        )
    command = env.command_manager.get_term("twist").command
    if tuple(command.shape) != (env.num_envs, 3):
        raise ValueError(f"Unexpected native legacy command shape: {tuple(command.shape)}")
    actor[:, -3:] = command
    patched["actor"] = actor
    return patched


def _action_parameter_tensor(value: Any, width: int, device: Any) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.numel() == 1:
        tensor = tensor.reshape(1, 1).expand(1, width).clone()
    elif tensor.numel() == width:
        tensor = tensor.reshape(1, width).clone()
    else:
        raise ValueError(
            f"Action parameter has {tensor.numel()} values, expected 1 or {width}"
        )
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError("Action parameter contains a non-finite value")
    return tensor


def _legacy_walk_adapters(
    main_scale: torch.Tensor,
    main_offset: torch.Tensor,
    walk_scale: torch.Tensor,
    walk_offset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return observation and raw-action transforms between policy defaults."""

    if not (
        main_scale.shape == main_offset.shape == walk_scale.shape == walk_offset.shape
    ):
        raise ValueError("Legacy and main action parameter shapes differ")
    if bool((main_scale.abs() < 1.0e-9).any().item()):
        raise ValueError("Main action scale contains zero")
    position_offset = main_offset - walk_offset
    output_scale = walk_scale / main_scale
    output_offset = (walk_offset - main_offset) / main_scale
    return position_offset, output_scale, output_offset


def _walk_actor_observation(
    observations: Any,
    body_joint_indices: torch.Tensor,
    position_offset: torch.Tensor,
    last_action: torch.Tensor,
) -> Any:
    """Project the 83-wide hybrid observation onto the legacy 63-wide schema."""

    actor = observations["actor"]
    widths = dict(MICROBAN_TELEOP_OBSERVATION_SCHEMA)
    joint_width = widths["joint_pos"]
    if joint_width != 21 or body_joint_indices.numel() != 18:
        raise ValueError("Unexpected Microban legacy-walk joint projection")
    joint_pos_start = widths["base_ang_vel"] + widths["projected_gravity"]
    joint_vel_start = joint_pos_start + joint_width
    actions_start = joint_vel_start + widths["joint_vel"]
    command_start = actions_start + widths["actions"]
    projected_actor = torch.cat(
        (
            actor[:, :joint_pos_start],
            actor[:, joint_pos_start : joint_pos_start + joint_width].index_select(
                1, body_joint_indices
            )
            + position_offset,
            actor[:, joint_vel_start : joint_vel_start + joint_width].index_select(
                1, body_joint_indices
            ),
            last_action,
            actor[:, command_start : command_start + widths["command"]],
        ),
        dim=1,
    )
    if projected_actor.shape[1] != 63:
        raise ValueError(
            f"Unexpected legacy walk observation width: {projected_actor.shape[1]}"
        )
    projected = observations.clone()
    projected["actor"] = projected_actor
    return projected


def _legacy_body_joint_indices(
    robot_joint_names: Sequence[str], action_joint_names: Sequence[str]
) -> tuple[int, ...]:
    """Resolve the legacy actor's exact 18-joint order in 21-joint observations."""

    expected_width = MICROBAN_TELEOP_ACTION_WIDTH
    action_names = tuple(action_joint_names)
    robot_names = tuple(robot_joint_names)
    if len(action_names) != expected_width or len(set(action_names)) != expected_width:
        raise ValueError("Legacy walk action joint names are not 18 unique joints")
    if len(robot_names) != expected_width + len(MICROBAN_HMD_JOINT_NAMES):
        raise ValueError(f"Unexpected robot joint count: {len(robot_names)}")
    if len(set(robot_names)) != len(robot_names):
        raise ValueError("Robot joint names are not unique")

    robot_indices = {name: index for index, name in enumerate(robot_names)}
    body_names = tuple(
        name for name in robot_names if name not in MICROBAN_HMD_JOINT_NAMES
    )
    if set(body_names) != set(action_names):
        raise ValueError(
            "Legacy walk/main observation joint set mismatch: "
            f"{action_names} != {body_names}"
        )
    return tuple(robot_indices[name] for name in action_names)


class LivePicoSimulationPolicy:
    """Policy facade that injects one validated PICO frame per 50 Hz step."""

    def __init__(
        self,
        *,
        env: ManagerBasedRlEnv,
        actor: Any,
        walk_actor: LegacyWalkActor,
        source: _Source,
        mapper: _Mapper,
        legacy_fallback_mapper: _Mapper | None = None,
        camera_publisher: StereoMjpegPublisher | None = None,
        legacy_only: bool = False,
        native_legacy_action_semantics: bool = False,
        clock_ns: Any = time.monotonic_ns,
        status_period_s: float = 1.0,
    ) -> None:
        self.env = env
        self.actor = actor
        self.walk_actor = walk_actor.policy
        self.source = source
        self.mapper = mapper
        self.legacy_fallback_mapper = legacy_fallback_mapper
        self.camera_publisher = camera_publisher
        self._camera_fault: str | None = None
        self._legacy_fallback_latched = False
        self._legacy_fallback_reason: str | None = None
        self._pending_legacy_command: SimulationCommand | None = None
        self.legacy_only = legacy_only
        self.native_legacy_action_semantics = native_legacy_action_semantics
        self.clock_ns = clock_ns
        self.status_period_ns = int(status_period_s * 1.0e9)
        self._last_status_ns = 0
        self._previous_sampled_at_ns: int | None = None
        self._previous_episode_length = env.episode_length_buf.clone()
        self._last_fault: str | None = None
        self._pending_authority_token: Any = None

        self.robot = env.scene["robot"]
        joint_ids, names = self.robot.find_joints(
            MICROBAN_HMD_JOINT_NAMES, preserve_order=True
        )
        if tuple(names) != MICROBAN_HMD_JOINT_NAMES:
            raise ValueError(f"Unexpected HMD joint order: {names}")
        self.hmd_joint_ids = torch.tensor(
            joint_ids, dtype=torch.long, device=env.device
        )
        main_action = env.action_manager.get_term("joint_pos")
        main_joint_names = tuple(main_action.target_names)
        if main_joint_names != walk_actor.joint_names:
            raise ValueError(
                "Legacy walk/main action joint order mismatch: "
                f"{walk_actor.joint_names} != {main_joint_names}"
            )
        main_scale = _action_parameter_tensor(
            main_action.scale, MICROBAN_TELEOP_ACTION_WIDTH, env.device
        )
        main_offset = _action_parameter_tensor(
            main_action.offset, MICROBAN_TELEOP_ACTION_WIDTH, env.device
        )
        walk_scale = walk_actor.scale.to(device=env.device, dtype=torch.float32)
        walk_offset = walk_actor.offset.to(device=env.device, dtype=torch.float32)
        if native_legacy_action_semantics:
            if main_action.cfg.clip is not None:
                raise ValueError(
                    "Original velocity runtime must not clip absolute action targets"
                )
            if not bool(getattr(main_action.cfg, "use_default_offset", False)):
                raise ValueError(
                    "Original velocity runtime must use the HOME/default action offset"
                )
            if not torch.equal(main_scale, walk_scale) or not torch.equal(
                main_offset, walk_offset
            ):
                raise ValueError(
                    "Native legacy actor/environment action parameters disagree"
                )
            default = self.robot.data.default_joint_pos[:, main_action.target_ids]
            if not torch.equal(main_offset, default):
                raise ValueError(
                    "Raw zero does not map exactly to the original velocity HOME pose"
                )
            self.body_joint_observation_indices = torch.empty(
                0, dtype=torch.long, device=env.device
            )
            self.walk_position_offset = torch.zeros_like(main_scale)
            self.walk_output_scale = torch.ones_like(main_scale)
            self.walk_output_offset = torch.zeros_like(main_offset)
        else:
            body_joint_ids = _legacy_body_joint_indices(
                self.robot.joint_names, main_joint_names
            )
            self.body_joint_observation_indices = torch.tensor(
                body_joint_ids, dtype=torch.long, device=env.device
            )
            (
                self.walk_position_offset,
                self.walk_output_scale,
                self.walk_output_offset,
            ) = _legacy_walk_adapters(
                main_scale, main_offset, walk_scale, walk_offset
            )
        self.walk_last_action = torch.zeros(
            (env.num_envs, MICROBAN_TELEOP_ACTION_WIDTH), device=env.device
        )
        self.hmd_current_target = self.robot.data.joint_pos[
            :, self.hmd_joint_ids
        ].clone()
        runtime_bounds = torch.tensor(
            [MICROBAN_HMD_RUNTIME_LIMITS_RAD[name] for name in names],
            dtype=torch.float32,
            device=env.device,
        ).unsqueeze(0)
        soft_bounds = self.robot.data.soft_joint_pos_limits[:, self.hmd_joint_ids]
        self.hmd_lower = torch.maximum(runtime_bounds[..., 0], soft_bounds[..., 0])
        self.hmd_upper = torch.minimum(runtime_bounds[..., 1], soft_bounds[..., 1])
        self.hmd_slew = torch.tensor(
            [MICROBAN_HMD_SLEW_RATES_RAD_S[name] for name in names],
            dtype=torch.float32,
            device=env.device,
        ).unsqueeze(0)
        self.hmd_default = self.robot.data.default_joint_pos[
            :, self.hmd_joint_ids
        ].clone()
        self.zero_action = torch.zeros(
            (env.num_envs, MICROBAN_TELEOP_ACTION_WIDTH), device=env.device
        )
        self._validate_command_terms()

    def _validate_command_terms(self) -> None:
        twist = self.env.command_manager.get_term("twist")
        if not isinstance(twist, UniformVelocityCommandWithRotation):
            raise TypeError(f"Unexpected twist command type: {type(twist).__name__}")
        if self.native_legacy_action_semantics:
            if set(self.env.command_manager.active_terms) != {"twist"}:
                raise ValueError(
                    "Original velocity runtime must contain only the twist command"
                )
            return
        foot = self.env.command_manager.get_term("foot_target")
        hand = self.env.command_manager.get_term("hand_target")
        if not isinstance(foot, ResetFixedFootTargetCommand):
            raise TypeError(f"Unexpected foot command type: {type(foot).__name__}")
        if not isinstance(hand, ResetFixedHandTargetCommand):
            raise TypeError(f"Unexpected hand command type: {type(hand).__name__}")

    def reset(self) -> None:
        self._reset_mappers()
        self._previous_sampled_at_ns = None
        self._pending_authority_token = None
        self._pending_legacy_command = None
        self._clear_legacy_fallback_latch()
        self.hmd_current_target.copy_(self.robot.data.joint_pos[:, self.hmd_joint_ids])
        self.walk_last_action.zero_()
        self._previous_episode_length = self.env.episode_length_buf.clone()

    def _reset_mappers(self) -> None:
        self.mapper.reset()
        fallback_mapper = getattr(self, "legacy_fallback_mapper", None)
        if fallback_mapper is not None and fallback_mapper is not self.mapper:
            fallback_mapper.reset()

    def _latch_legacy_fallback(self, reason: str) -> None:
        self._legacy_fallback_latched = True
        self._legacy_fallback_reason = reason

    def _clear_legacy_fallback_latch(self) -> None:
        self._legacy_fallback_latched = False
        self._legacy_fallback_reason = None

    def _read_command(self) -> SimulationCommand:
        try:
            read_with_token = getattr(self.source, "read_with_token", None)
            if callable(read_with_token):
                frame, self._pending_authority_token = read_with_token()
            else:
                frame = self.source.read()
                self._pending_authority_token = None
        except Exception as exc:  # noqa: BLE001 - SDK faults must fail closed
            self._pending_authority_token = None
            self._reset_mappers()
            return neutral_simulation_command(
                fault=f"PICO source read failed: {type(exc).__name__}: {exc}"
            )

        sampled_at_ns = getattr(frame, "sampled_at_ns", None)
        if (
            not isinstance(sampled_at_ns, int)
            or isinstance(sampled_at_ns, bool)
            or sampled_at_ns < 0
        ):
            self._reset_mappers()
            return neutral_simulation_command(fault="invalid host sample timestamp")

        previous = self._previous_sampled_at_ns
        self._previous_sampled_at_ns = sampled_at_ns
        if previous is not None:
            delta_ns = sampled_at_ns - previous
            # A fast simulation step may observe the same immutable 50 Hz WebXR
            # snapshot twice. Its original age still expires at 100 ms; only a
            # backwards timestamp or a genuine host-side gap disarms immediately.
            if delta_ns < 0 or delta_ns > int(MAX_FRAME_GAP_S * 1.0e9):
                self._reset_mappers()
                return neutral_simulation_command(
                    fault="host sampling gap; trigger rearm required"
                )

        age_ns = self.clock_ns() - sampled_at_ns
        if age_ns < 0 or age_ns > int(MAX_FRAME_AGE_S * 1.0e9):
            self._reset_mappers()
            return neutral_simulation_command(
                fault="aged PICO frame; trigger rearm required"
            )

        try:
            mapper_frame = (
                _legacy_walk_input_frame(frame)
                if getattr(self, "legacy_only", False)
                else frame
            )
            mapped = self.mapper.map_sample(mapper_frame)
            if getattr(self, "legacy_only", False):
                if not isinstance(mapped, Mapping):
                    return neutral_simulation_command(
                        fault="mapper output is not a mapping"
                    )
                # Deadline fallback: the audited velocity actor owns locomotion
                # regardless of the optional X-button hybrid-policy selector.  The
                # mapper still owns the trigger deadman and all of its rearm rules.
                mapped = {**mapped, "locomotion_policy": "walk"}
                command = command_for_simulation(
                    mapped, legacy_walk_available=True
                )
                self._pending_legacy_command = command
                return command

            fallback_mapper = getattr(self, "legacy_fallback_mapper", None)
            if fallback_mapper is None:
                fallback_mapped = _legacy_walk_mapping(mapped)
            else:
                fallback_mapped = fallback_mapper.map_sample(
                    _legacy_walk_input_frame(frame)
                )
                fallback_mapped = _legacy_walk_mapping(fallback_mapped)
            fallback_command = command_for_simulation(
                fallback_mapped, legacy_walk_available=True
            )
            self._pending_legacy_command = fallback_command
            primary_command = command_for_simulation(
                mapped, legacy_walk_available=True
            )

            explicit_release = _left_trigger_explicitly_released(
                frame, fallback_mapped
            )
            if explicit_release:
                self._clear_legacy_fallback_latch()
            elif getattr(self, "_legacy_fallback_latched", False):
                return _command_with_fault(
                    fallback_command,
                    self._legacy_fallback_reason
                    or "legacy fallback latched until trigger release",
                )

            if (
                primary_command.locomotion_policy == "pico_teleop"
                and fallback_command.enabled
                and (
                    not primary_command.enabled
                    or primary_command.fault is not None
                )
            ):
                reason = (
                    "PICO body policy unavailable; using legacy joystick "
                    "until trigger release"
                )
                self._latch_legacy_fallback(reason)
                return _command_with_fault(fallback_command, reason)
            return primary_command
        except Exception as exc:  # noqa: BLE001 - mapper faults must fail closed
            self._reset_mappers()
            return neutral_simulation_command(
                fault=f"PICO mapper failed: {type(exc).__name__}: {exc}"
            )

    def _inject_command(self, command: SimulationCommand) -> None:
        twist = self.env.command_manager.get_term("twist")

        twist_value = torch.tensor(
            command.twist, dtype=torch.float32, device=self.env.device
        ).unsqueeze(0)
        twist.vel_command_b.copy_(twist_value)
        twist.vel_command_w.copy_(twist_value)
        for flag_name in (
            "is_heading_env",
            "is_standing_env",
            "is_world_env",
            "is_forward_env",
            "is_rotation_env",
        ):
            getattr(twist, flag_name).fill_(False)
        twist.time_left.fill_(float("inf"))

        if getattr(self, "native_legacy_action_semantics", False):
            return

        foot = self.env.command_manager.get_term("foot_target")
        hand = self.env.command_manager.get_term("hand_target")

        foot_value = torch.tensor(
            command.foot_target, dtype=torch.float32, device=self.env.device
        ).unsqueeze(0)
        foot.foot_target_offset_b.copy_(foot_value)
        foot.is_single_support_env.copy_(foot_value.norm(dim=-1).gt(0.0).any(dim=-1))
        foot.lifted_foot_idx.copy_(foot_value.norm(dim=-1).argmax(dim=-1))
        foot.time_left.fill_(float("inf"))

        hand_value = torch.tensor(
            command.hand_target, dtype=torch.float32, device=self.env.device
        ).unsqueeze(0)
        hand_active = torch.tensor(
            command.hand_active, dtype=torch.bool, device=self.env.device
        ).unsqueeze(0)
        hand.hand_target_offset_b.copy_(hand_value)
        hand.is_active.copy_(hand_active)
        hand.time_left.fill_(float("inf"))

    def _write_hmd_target(self, command: SimulationCommand) -> None:
        body_quaternion = self.robot.data.root_link_quat_w[0].detach().cpu().tolist()
        desired = solve_hmd_neck_target(
            command.head_orientation,
            body_quaternion,
            yaw_front=command.head_yaw_front,
        )
        if desired is not None:
            desired_tensor = torch.tensor(
                desired, dtype=torch.float32, device=self.env.device
            ).unsqueeze(0)
            desired_tensor = torch.clamp(
                desired_tensor + self.hmd_default,
                min=self.hmd_lower,
                max=self.hmd_upper,
            )
            max_step = self.hmd_slew * float(self.env.step_dt)
            delta = torch.clamp(
                desired_tensor - self.hmd_current_target,
                min=-max_step,
                max=max_step,
            )
            self.hmd_current_target.add_(delta)
        self.robot.set_joint_position_target(
            self.hmd_current_target, joint_ids=self.hmd_joint_ids
        )

    def _inject_with_authority(
        self, command: SimulationCommand
    ) -> SimulationCommand:
        """Serialize final simulation injection with native owner invalidation."""

        def inject() -> None:
            # This callback is deliberately bounded and never performs network
            # I/O. NativeNetworkSource holds its authority lock across it so a
            # disconnect cannot occur between validation and command injection.
            self._inject_command(command)
            self._write_hmd_target(command)

        authority = getattr(self, "_pending_authority_token", None)
        guarded_inject = getattr(
            getattr(self, "source", None), "run_if_current", None
        )
        if authority is None or not callable(guarded_inject):
            inject()
            return command

        injected, _result = guarded_inject(authority, inject)
        if injected:
            return command

        self._reset_mappers()
        self._pending_authority_token = None
        neutral = neutral_simulation_command(
            fault="native authority changed before simulation injection"
        )
        self._inject_command(neutral)
        self._write_hmd_target(neutral)
        return neutral

    def _print_status(self, command: SimulationCommand) -> None:
        now_ns = self.clock_ns()
        if now_ns - self._last_status_ns < self.status_period_ns:
            return
        self._last_status_ns = now_ns
        fault = command.fault or "none"
        print(
            "SIMULATION ONLY | "
            f"policy={command.locomotion_policy} enabled={command.enabled} "
            f"twist={tuple(round(value, 3) for value in command.twist)} "
            f"fault={fault} camera_fault={self._camera_fault or 'none'}",
            flush=True,
        )

    def _legacy_actor_action(self, observations: Any) -> torch.Tensor:
        native_legacy = bool(
            getattr(self, "native_legacy_action_semantics", False)
        )
        if native_legacy:
            actor_observation = _patch_native_walk_observation(
                observations, self.env
            )
            action = self.walk_actor(actor_observation)
            self._require_safe_action(action, "legacy walk")
            return action

        patched = _patch_command_observation(observations, self.env)
        actor_observation = _walk_actor_observation(
            patched,
            self.body_joint_observation_indices,
            self.walk_position_offset,
            self.walk_last_action,
        )
        walk_action = self.walk_actor(actor_observation)
        self._require_safe_action(walk_action, "legacy walk")
        self.walk_last_action.copy_(walk_action)
        action = walk_action * self.walk_output_scale + self.walk_output_offset
        self._require_safe_action(action, "adapted legacy walk")
        return action

    def _require_safe_action(self, action: Any, label: str) -> None:
        if not isinstance(action, torch.Tensor):
            raise TypeError(f"{label} returned {type(action).__name__}, not Tensor")
        finite = bool(torch.isfinite(action).all().item())
        if action.shape != self.zero_action.shape or not finite:
            raise RuntimeError(
                f"unsafe {label} output: shape={tuple(action.shape)}, "
                f"finite={finite}"
            )

    def _fallback_after_learned_fault(
        self, observations: Any, reason: str
    ) -> torch.Tensor:
        self._latch_legacy_fallback(reason)
        fallback = getattr(self, "_pending_legacy_command", None)
        if fallback is None or not fallback.enabled:
            self.walk_last_action.zero_()
            return self.zero_action.clone()

        fallback = _command_with_fault(fallback, reason)
        authority = getattr(self, "_pending_authority_token", None)
        guarded_inject = getattr(
            getattr(self, "source", None), "run_if_current", None
        )
        if authority is not None and callable(guarded_inject):
            still_current, _result = guarded_inject(authority, lambda: None)
            if not still_current:
                self._reset_mappers()
                self.walk_last_action.zero_()
                return self.zero_action.clone()
        # The primary command already injected the same joystick twist and HMD
        # orientation for this policy step. Do not inject a second time: doing
        # so would advance the neck slew limiter twice. The legacy observation
        # projection ignores hybrid hand/foot target columns.
        self._print_status(fallback)
        try:
            return self._legacy_actor_action(observations)
        except Exception as exc:  # noqa: BLE001 - keep the viewer alive at HOME
            self._legacy_fallback_reason = (
                f"{reason}; legacy fallback failed: {type(exc).__name__}: {exc}"
            )
            self._reset_mappers()
            self.walk_last_action.zero_()
            neutral = neutral_simulation_command(
                fault=self._legacy_fallback_reason
            )
            self._inject_with_authority(neutral)
            return self.zero_action.clone()

    def __call__(self, observations: Any) -> torch.Tensor:
        # Auto-reset after a simulated fall is visible for one policy call.
        # Disarm before consuming another frame so a held trigger cannot restart.
        reset_buf = getattr(self.env, "reset_buf", None)
        episode_length = getattr(self.env, "episode_length_buf", None)
        previous_episode_length = getattr(self, "_previous_episode_length", None)
        manual_reset = (
            episode_length is not None
            and previous_episode_length is not None
            and bool((episode_length < previous_episode_length).any().item())
        )
        if (reset_buf is not None and bool(reset_buf.any().item())) or manual_reset:
            self.reset()
        elif episode_length is not None:
            self._previous_episode_length = episode_length.clone()

        command = self._inject_with_authority(self._read_command())
        if self.camera_publisher is not None:
            try:
                self.camera_publisher.capture_if_due()
            except Exception as exc:  # noqa: BLE001 - video faults must not stop control.
                self._camera_fault = f"{type(exc).__name__}: {exc}"
                self.camera_publisher = None
                print(
                    "Simulation camera degraded; control remains active and the "
                    f"PICO should use passthrough/last-frame: {self._camera_fault}",
                    flush=True,
                )
        self._print_status(command)
        if not command.enabled:
            self.walk_last_action.zero_()
            # Deadman release always returns the shared Microban initial pose:
            # raw zero in the teleop environment's 0 degree shoulder frame.
            # The legacy walk actor's own 0 degree HOME is used only while its
            # policy is actively selected and the trigger is held.
            return self.zero_action.clone()

        if (
            bool(getattr(self, "native_legacy_action_semantics", False))
            or command.locomotion_policy == "walk"
        ):
            return self._legacy_actor_action(observations)

        self.walk_last_action.zero_()
        patched = _patch_command_observation(observations, self.env)
        try:
            if self.actor is None:
                raise RuntimeError("learned PICO actor is unavailable")
            action = self.actor(patched)
            self._require_safe_action(action, "learned PICO actor")
            return action
        except Exception as exc:  # noqa: BLE001 - same-cycle legacy degradation
            reason = (
                "learned PICO actor failed; using legacy joystick until trigger "
                f"release: {type(exc).__name__}: {exc}"
            )
            return self._fallback_after_learned_fault(observations, reason)


def _positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def _optional_scale(value: str) -> float:
    result = _positive_float(value)
    if not 0.01 <= result <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0.01 and 1.0")
    return result


def _port(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if not 1 <= result <= 65535:
        raise argparse.ArgumentTypeError("must be between 1 and 65535")
    return result


def _default_teleop_root() -> Path:
    return Path(__file__).resolve().parents[4] / "microban_teleop"


def _default_native_config() -> Path:
    return Path.home() / ".config" / "microban-teleop" / "native_transport.json"


def _default_walk_checkpoint() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "checkpoints"
        / "xc330_velocity"
        / "model_14999.pt"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Simulation-only PICO 4 Ultra controller for Microban; without "
            "--checkpoint it uses the audited legacy walking actor exclusively and "
            "has no robot UDP or motor path"
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "optional hybrid-policy checkpoint; load/inference/body faults "
            "degrade to the audited legacy walking actor"
        ),
    )
    parser.add_argument(
        "--walk-checkpoint",
        type=Path,
        default=_default_walk_checkpoint(),
        help=(
            "audited legacy walk checkpoint (the default is the exact 15,000-update "
            "model_14999.pt used by the deadline fallback)"
        ),
    )
    parser.add_argument(
        "--teleop-root",
        type=Path,
        default=_default_teleop_root(),
        help="microban_teleop repository root",
    )
    parser.add_argument("--device", default=None, help="default: cuda:0 when available")
    parser.add_argument("--viewer", choices=("native", "viser"), default="native")
    parser.add_argument(
        "--input",
        choices=("webxr", "native", "pico-app"),
        default="webxr",
        help=(
            "webxr: one PICO Browser app for controls and camera (default); "
            "native: legacy XRoboToolkit full-body service; "
            "pico-app: authenticated Microban Unity client"
        ),
    )
    parser.add_argument(
        "--native-config",
        type=Path,
        default=_default_native_config(),
        help=(
            "owner-only pairing/listen configuration used by --input pico-app "
            "(default: ~/.config/microban-teleop/native_transport.json)"
        ),
    )
    parser.add_argument(
        "--web-port",
        type=_port,
        default=8443,
        help="loopback-only PICO Browser port for --input webxr (default: 8443)",
    )
    parser.add_argument("--stale-s", type=_positive_float, default=0.25)
    parser.add_argument("--body-scale", type=_optional_scale)
    parser.add_argument("--hand-scale", type=_optional_scale)
    parser.add_argument("--foot-scale", type=_optional_scale)
    parser.add_argument(
        "--camera-port",
        type=_port,
        default=8081,
        help="loopback-only SBS MJPEG port (default: 8081)",
    )
    parser.add_argument("--camera-fps", type=_positive_float, default=20.0)
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="disable the simulation stereo MJPEG companion",
    )
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not args.walk_checkpoint.is_file():
        parser.error(f"walk checkpoint not found: {args.walk_checkpoint}")
    walk_sha256 = _sha256(args.walk_checkpoint)
    if walk_sha256 != AUDITED_LEGACY_WALK_SHA256:
        parser.error(
            "walk checkpoint is not the audited model_14999.pt: "
            f"expected {AUDITED_LEGACY_WALK_SHA256}, got {walk_sha256}"
        )
    if not 0.05 <= args.stale_s < 0.3:
        parser.error("--stale-s must be at least 0.05 and below 0.3 seconds")
    if not 1.0 <= args.camera_fps <= 30.0:
        parser.error("--camera-fps must be in [1, 30]")
    if args.body_scale is not None and (
        args.hand_scale is not None or args.foot_scale is not None
    ):
        parser.error("--body-scale cannot be combined with per-limb scale overrides")
    if args.input == "webxr" and any(
        value is not None
        for value in (args.body_scale, args.hand_scale, args.foot_scale)
    ):
        parser.error("body scale overrides require --input native or pico-app")
    if args.input == "pico-app" and not args.native_config.expanduser().is_file():
        parser.error(f"native pairing config not found: {args.native_config}")
    package = args.teleop_root.resolve() / "src" / "microban_teleop"
    if not package.is_dir():
        parser.error(f"microban_teleop package not found below: {args.teleop_root}")


def _load_native_classes(teleop_root: Path) -> tuple[Any, Any]:
    source_root = str(teleop_root.resolve() / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    mapping = importlib.import_module("microban_teleop.twist2.mapping")
    tracking = importlib.import_module("microban_teleop.twist2.tracking")
    return tracking.XRobotSource, mapping.NativeControlMapper


def _load_pico_app_classes(teleop_root: Path) -> tuple[Any, Any, Any, Any]:
    source_root = str(teleop_root.resolve() / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    mapping = importlib.import_module("microban_teleop.twist2.mapping")
    native_server = importlib.import_module("microban_teleop.native_server")
    return (
        native_server.load_pairing_store,
        native_server.NativeInputServer,
        native_server.NativeNetworkSource,
        mapping.NativeControlMapper,
    )


def _load_webxr_classes(teleop_root: Path) -> tuple[Any, Any]:
    source_root = str(teleop_root.resolve() / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    server = importlib.import_module("microban_teleop.server")
    return server.Settings, server.TeleopApplication


def _configure_live_environment(cfg: Any) -> None:
    cfg.scene.num_envs = 1
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    # Nominal simulation: keep only deterministic reset events and do not apply
    # training-time physical randomization or interval pushes.
    cfg.events = {
        name: term for name, term in cfg.events.items() if term.mode == "reset"
    }
    reset_base = cfg.events["reset_base"]
    reset_base.params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    reset_base.params["velocity_range"] = {}
    # The viewer should not reset a healthy session every 20 seconds.  Physical
    # failure/out-of-bounds terminations remain enabled and disarm the mapper.
    cfg.terminations.pop("time_out", None)


def _load_walk_actor_in_environment(
    *,
    raw_env: ManagerBasedRlEnv,
    wrapped_env: RslRlVecEnvWrapper,
    agent_cfg: Any,
    checkpoint: Path,
    device: str,
) -> LegacyWalkActor:
    """Strict-load the legacy actor against the environment that will execute it."""

    runner_class = load_runner_cls(WALK_TASK)
    if runner_class is None:
        raise RuntimeError(f"No runner is registered for {WALK_TASK}")
    runner = runner_class(wrapped_env, asdict(agent_cfg), device=device)
    runner.load(
        str(checkpoint.resolve()),
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
    )
    action = raw_env.action_manager.get_term("joint_pos")
    if action.action_dim != MICROBAN_TELEOP_ACTION_WIDTH:
        raise ValueError(
            f"Legacy walk action width is {action.action_dim}, expected "
            f"{MICROBAN_TELEOP_ACTION_WIDTH}"
        )
    return LegacyWalkActor(
        policy=runner.get_inference_policy(device=device),
        joint_names=tuple(action.target_names),
        scale=_action_parameter_tensor(
            action.scale, MICROBAN_TELEOP_ACTION_WIDTH, device
        ).clone(),
        offset=_action_parameter_tensor(
            action.offset, MICROBAN_TELEOP_ACTION_WIDTH, device
        ).clone(),
    )


def _load_legacy_walk_actor(checkpoint: Path, device: str) -> LegacyWalkActor:
    """Load the separately trained legacy walk actor, then release its env."""

    env_cfg = load_env_cfg(WALK_TASK, play=True)
    env_cfg.scene.num_envs = 1
    agent_cfg = load_rl_cfg(WALK_TASK)
    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    wrapped_env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    try:
        return _load_walk_actor_in_environment(
            raw_env=raw_env,
            wrapped_env=wrapped_env,
            agent_cfg=agent_cfg,
            checkpoint=checkpoint,
            device=device,
        )
    finally:
        wrapped_env.close()


def _construct_checkpoint_consumer_runner(env, agent_cfg, device: str):
    """Construct the explicit actor-load-only runner used by live simulation."""

    agent_cfg.checkpoint_consumer_mode = True
    runner_class = load_runner_cls(TASK)
    if runner_class is None:
        raise RuntimeError(f"No runner is registered for {TASK}")
    return runner_class(env, asdict(agent_cfg), device=device)


def _runtime_task(checkpoint: Path | None) -> str:
    """Select the original velocity task unless a hybrid actor was requested."""

    return WALK_TASK if checkpoint is None else TASK


def run(args: argparse.Namespace) -> int:
    configure_torch_backends()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    legacy_only = args.checkpoint is None
    runtime_task = _runtime_task(args.checkpoint)
    env_cfg = load_env_cfg(runtime_task, play=True)
    # Nominalize resets and remove timeout/DR/push events for an operator-owned
    # live session. This deliberately does not change the selected task's action
    # or observation contract: legacy-only still keeps the original velocity
    # task's unclipped raw actions and 63-wide actor observation.
    _configure_live_environment(env_cfg)
    agent_cfg = load_rl_cfg(runtime_task)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    source: _Source | None = None
    legacy_fallback_mapper: _Mapper | None = None
    camera_publisher: StereoMjpegPublisher | None = None
    web_server: _WebServerThread | None = None
    native_server_thread: _NativeServerThread | None = None
    try:
        if legacy_only:
            if wrapped_env.clip_actions is not None:
                raise ValueError(
                    "Original velocity runtime unexpectedly clips actor outputs"
                )
            walk_actor = _load_walk_actor_in_environment(
                raw_env=env,
                wrapped_env=wrapped_env,
                agent_cfg=agent_cfg,
                checkpoint=args.walk_checkpoint,
                device=device,
            )
        else:
            validate_microban_teleop_observation_contract(env)
        actor = None
        if not legacy_only:
            walk_actor = _load_legacy_walk_actor(args.walk_checkpoint, device)
            try:
                runner = _construct_checkpoint_consumer_runner(
                    wrapped_env, agent_cfg, device
                )
                runner.load(
                    str(args.checkpoint.resolve()),
                    load_cfg={"actor": True},
                    strict=True,
                    map_location=device,
                )
                actor = runner.get_inference_policy(device=device)
            except Exception as exc:  # noqa: BLE001 - legacy actor remains usable
                print(
                    "Learned PICO checkpoint unavailable; legacy joystick "
                    "fallback remains active: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

        if not args.no_camera:
            camera_publisher = StereoMjpegPublisher(
                env,
                port=args.camera_port,
                fps=args.camera_fps,
            )

        if args.input == "native":
            source_class, mapper_class = _load_native_classes(args.teleop_root)
            source = source_class(stale_after_ns=int(args.stale_s * 1.0e9))
            if not source.strict_control_presence:
                raise RuntimeError(
                    "The presence-aware pinned XRoboToolkit binding is required for "
                    "live simulation input"
                )
            mapper = mapper_class(
                body_scale=args.body_scale,
                hand_scale_override=args.hand_scale,
                foot_scale_override=args.foot_scale,
            )
            if not legacy_only:
                legacy_fallback_mapper = mapper_class(
                    body_scale=args.body_scale,
                    hand_scale_override=args.hand_scale,
                    foot_scale_override=args.foot_scale,
                )
        elif args.input == "pico-app":
            (
                load_pairing_store,
                server_class,
                source_class,
                mapper_class,
            ) = _load_pico_app_classes(args.teleop_root)
            pairing_store = load_pairing_store(args.native_config.expanduser())
            source = source_class(stale_after_ns=int(args.stale_s * 1.0e9))
            mapper = mapper_class(
                body_scale=args.body_scale,
                hand_scale_override=args.hand_scale,
                foot_scale_override=args.foot_scale,
            )
            if not legacy_only:
                legacy_fallback_mapper = mapper_class(
                    body_scale=args.body_scale,
                    hand_scale_override=args.hand_scale,
                    foot_scale_override=args.foot_scale,
                )

            def reset_native_mappers() -> None:
                mapper.reset()
                if legacy_fallback_mapper is not None:
                    legacy_fallback_mapper.reset()

            native_server = server_class(
                pairing_store,
                source=source,
                on_reset=reset_native_mappers,
            )
            native_server_thread = _NativeServerThread(native_server)
            native_server_thread.start()
        else:
            settings_class, application_class = _load_webxr_classes(args.teleop_root)
            webxr_source = WebXrSimulationSource()
            source = webxr_source
            mapper = WebXrSimulationMapper(webxr_source)
            camera_url = (
                ""
                if camera_publisher is None
                else f"http://127.0.0.1:{camera_publisher.port}/stream"
            )
            settings = settings_class(
                bind="127.0.0.1",
                web_port=args.web_port,
                camera_url=camera_url,
                camera_horizontal_fov_deg=HORIZONTAL_FOV_DEG,
                camera_vertical_fov_deg=VERTICAL_FOV_DEG,
                camera_eye_aspect=EYE_ASPECT,
                camera_eye_width_px=EYE_WIDTH_PX,
                camera_eye_height_px=EYE_HEIGHT_PX,
                camera_left_eye_first=True,
                camera_calibrated=camera_publisher is not None,
                camera_left_tan_bounds=TAN_BOUNDS,
                camera_right_tan_bounds=TAN_BOUNDS,
                simulation_only=True,
            )
            teleop = application_class(
                settings,
                args.teleop_root.resolve() / "static",
                simulation_sink=webxr_source.publish,
            )
            webxr_source.set_reset_callback(teleop.request_control_reset)
            web_server = _WebServerThread(teleop.app, port=args.web_port)
            web_server.start()

        policy = LivePicoSimulationPolicy(
            env=env,
            actor=actor,
            walk_actor=walk_actor,
            source=source,
            mapper=mapper,
            legacy_fallback_mapper=legacy_fallback_mapper,
            camera_publisher=camera_publisher,
            legacy_only=legacy_only,
            native_legacy_action_semantics=legacy_only,
        )

        print("SIMULATION ONLY: no robot UDP socket or motor interface is opened.")
        if args.checkpoint is None:
            print(
                "DEADLINE FALLBACK: audited model_14999.pt owns locomotion; "
                "the X-button hybrid selector is forced to legacy walk."
            )
        else:
            print(
                "RESILIENT HYBRID: body/checkpoint/inference faults use the "
                "audited legacy actor until left-trigger release; the next "
                "activation retries the learned actor."
            )
        if args.input == "webxr":
            print(
                f"PICO Browser: http://localhost:{args.web_port}/ "
                f"(run: adb reverse tcp:{args.web_port} tcp:{args.web_port})"
            )
            if legacy_only:
                print(
                    "Legacy-only mode ignores left X; release the left trigger "
                    "once, then hold it to move."
                )
            else:
                print(
                    "Hold left X for pico_teleop; release the left trigger once, "
                    "then hold it to move."
                )
            print(
                "WebXR supplies neutral hand/foot targets; Motion Tracker full-body "
                "targets require --input native."
            )
        elif args.input == "native":
            if legacy_only:
                print(
                    "Legacy-only mode ignores left X and Motion Tracker body targets."
                )
            else:
                print(
                    "Hold left X with the left trigger released to calibrate body "
                    "targets, then keep X held."
                )
            print(
                "Native XRoboToolkit and PICO Browser are separate foreground apps; "
                "the MJPEG endpoint below is diagnostic only in native mode."
            )
        else:
            address = native_server.address
            print(
                "Authenticated Microban PICO app input: "
                f"{address[0]}:{address[1] if address is not None else 'not-started'}"
                if address is not None
                else "Authenticated Microban PICO app input did not expose an address."
            )
            if legacy_only:
                print(
                    "Legacy-only mode ignores left X and Motion Tracker body targets."
                )
            else:
                print(
                    "Hold left X with the left trigger released to calibrate body "
                    "targets, then keep X held."
                )
            print(
                "Unity receives the synthetic stereo calibration and latest-frame "
                "endpoints on the camera port."
            )
        print("Hold left trigger to move; hold right trigger to center head yaw.")
        if camera_publisher is not None:
            print(
                f"Simulation SBS MJPEG: http://127.0.0.1:{camera_publisher.port}/stream"
            )
            print(
                "Exact WebXR camera config:\n"
                + webxr_camera_toml(camera_publisher.port)
            )
        if args.viewer == "native":
            NativeMujocoViewer(wrapped_env, policy).run()
        else:
            ViserPlayViewer(wrapped_env, policy).run()
        return 0
    finally:
        if web_server is not None:
            web_server.close()
        if native_server_thread is not None:
            native_server_thread.close()
        if camera_publisher is not None:
            camera_publisher.close()
        if source is not None:
            source.close()
        wrapped_env.close()


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(args, parser)
    try:
        status = run(args)
    except KeyboardInterrupt:
        status = 130
    except Exception as exc:  # noqa: BLE001 - CLI boundary reports cleanly
        parser.exit(1, f"live PICO simulation failed: {exc}\n")
    raise SystemExit(status)


if __name__ == "__main__":
    main()
