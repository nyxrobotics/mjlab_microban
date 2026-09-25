"""Run the 99-observation Microban tracking actor from live PICO body data.

This executable is simulation-only: it contains no robot address, UDP socket,
servo bus, or deployment adapter.  It intentionally supports only PICO inputs
which include the 24-joint Motion Tracker body stream (``native`` and
``pico-app``); controller-only WebXR cannot satisfy the tracking contract.
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MethodType
from typing import Any

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from mjlab_microban.live_pico_tracking_bridge import (
    LIVE_TRACKING_ACTOR_SCHEMA,
    LIVE_TRACKING_ACTOR_WIDTH,
    LivePicoTrackingReferenceBuilder,
    LiveTrackingBridgeError,
    clip_tracking_action_to_soft_limits,
    patch_tracking_actor_observation,
)
from mjlab_microban.robot.microban_constants import HOME_FRAME
from mjlab_microban.scripts.live_pico_teleop_sim import (
    MAX_FRAME_AGE_S,
    MAX_FRAME_GAP_S,
    SimulationCommand,
    _action_parameter_tensor,
    _default_native_config,
    _default_teleop_root,
    _load_native_classes,
    _load_pico_app_classes,
    _NativeServerThread,
    _optional_scale,
    _port,
    _positive_float,
    command_for_simulation,
    neutral_simulation_command,
    solve_hmd_neck_target,
)
from mjlab_microban.scripts.simulation_camera import StereoMjpegPublisher
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_HMD_JOINT_NAMES,
    MICROBAN_TELEOP_ACTION_JOINT_NAMES,
    MICROBAN_TELEOP_ACTION_WIDTH,
)
from mjlab_microban.tasks.microban_teleop_mdp import (
    MICROBAN_HMD_RUNTIME_LIMITS_RAD,
    MICROBAN_HMD_SLEW_RATES_RAD_S,
)

TASK = "Mjlab-Tracking-Microban"

_FALL_MIN_ROOT_HEIGHT_M = 0.10
_FALL_MIN_UP_Z = 0.50
_FALL_MAX_ROOT_LINEAR_SPEED_M_S = 2.0
_FALL_MAX_ROOT_ANGULAR_SPEED_RAD_S = 10.0
_CALIBRATION_MAX_ROOT_HEIGHT_ERROR_M = 0.025
_CALIBRATION_MIN_UP_Z = 0.94
_CALIBRATION_MAX_ROOT_LINEAR_SPEED_M_S = 0.05
_CALIBRATION_MAX_ROOT_ANGULAR_SPEED_RAD_S = 0.35
_CALIBRATION_MAX_HOME_JOINT_ERROR_RAD = 0.12
_CALIBRATION_MAX_JOINT_SPEED_RAD_S = 0.75
_DEFAULT_TRIGGER_RELEASE_THRESHOLD = 0.15


@dataclass(frozen=True, slots=True)
class LiveRobotSafetyAssessment:
    """Fail-closed robot health plus stricter released-trigger readiness."""

    fall_reason: str | None
    calibration_block_reason: str | None

    @property
    def calibration_ready(self) -> bool:
        return self.fall_reason is None and self.calibration_block_reason is None


@dataclass(slots=True)
class LiveFallLatch:
    """A fall remains disarmed until a safe frame has a released left trigger."""

    reason: str | None = None

    @property
    def latched(self) -> bool:
        return self.reason is not None

    def update(
        self, *, fall_reason: str | None, left_trigger_released: bool
    ) -> tuple[bool, bool, bool]:
        """Return ``(latched, entered, cleared)`` after one observation."""

        was_latched = self.latched
        if fall_reason is not None:
            self.reason = fall_reason
            return True, not was_latched, False
        if was_latched and left_trigger_released:
            self.reason = None
            return False, False, True
        return was_latched, False, False


def _assess_live_robot_state(
    *,
    root_pos_w: torch.Tensor,
    root_quat_wxyz: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    body_joint_pos: torch.Tensor,
    body_joint_vel: torch.Tensor,
    home_body_joint_pos: torch.Tensor,
    environment_origin_z: float,
) -> LiveRobotSafetyAssessment:
    """Validate the single simulated robot without trusting policy observations."""

    expected_shapes = (
        (root_pos_w, (1, 3)),
        (root_quat_wxyz, (1, 4)),
        (root_lin_vel, (1, 3)),
        (root_ang_vel, (1, 3)),
        (body_joint_pos, (1, MICROBAN_TELEOP_ACTION_WIDTH)),
        (body_joint_vel, (1, MICROBAN_TELEOP_ACTION_WIDTH)),
        (home_body_joint_pos, (1, MICROBAN_TELEOP_ACTION_WIDTH)),
    )
    if any(tuple(value.shape) != shape for value, shape in expected_shapes):
        reason = "robot state shape is invalid"
        return LiveRobotSafetyAssessment(reason, reason)
    if not math.isfinite(environment_origin_z) or any(
        not bool(torch.isfinite(value).all().item())
        for value, _shape in expected_shapes
    ):
        reason = "robot state is non-finite"
        return LiveRobotSafetyAssessment(reason, reason)

    root_height = float(root_pos_w[0, 2].item()) - environment_origin_z
    quaternion = root_quat_wxyz[0]
    quaternion_norm = float(torch.linalg.vector_norm(quaternion).item())
    if quaternion_norm < 1.0e-6:
        reason = "robot root quaternion has zero length"
        return LiveRobotSafetyAssessment(reason, reason)
    _w, x, y, _z = (float(value.item()) / quaternion_norm for value in quaternion)
    up_z = 1.0 - 2.0 * (x * x + y * y)
    root_linear_speed = float(torch.linalg.vector_norm(root_lin_vel[0]).item())
    root_angular_speed = float(torch.linalg.vector_norm(root_ang_vel[0]).item())

    fall_reason: str | None = None
    if root_height < _FALL_MIN_ROOT_HEIGHT_M:
        fall_reason = f"robot root height {root_height:.4f} m is below the fall limit"
    elif up_z < _FALL_MIN_UP_Z:
        fall_reason = f"robot up-vector z {up_z:.4f} is below the fall limit"
    elif root_linear_speed > _FALL_MAX_ROOT_LINEAR_SPEED_M_S:
        fall_reason = (
            f"robot root speed {root_linear_speed:.4f} m/s exceeds the fall limit"
        )
    elif root_angular_speed > _FALL_MAX_ROOT_ANGULAR_SPEED_RAD_S:
        fall_reason = (
            "robot root angular speed "
            f"{root_angular_speed:.4f} rad/s exceeds the fall limit"
        )
    if fall_reason is not None:
        return LiveRobotSafetyAssessment(fall_reason, fall_reason)

    home_error = float(
        torch.max(torch.abs(body_joint_pos - home_body_joint_pos)).item()
    )
    maximum_joint_speed = float(torch.max(torch.abs(body_joint_vel)).item())
    calibration_reason: str | None = None
    if (
        abs(root_height - float(HOME_FRAME.pos[2]))
        > _CALIBRATION_MAX_ROOT_HEIGHT_ERROR_M
    ):
        calibration_reason = "robot root is not near HOME height"
    elif up_z < _CALIBRATION_MIN_UP_Z:
        calibration_reason = "robot trunk is not upright enough for calibration"
    elif root_linear_speed > _CALIBRATION_MAX_ROOT_LINEAR_SPEED_M_S:
        calibration_reason = "robot root is still moving"
    elif root_angular_speed > _CALIBRATION_MAX_ROOT_ANGULAR_SPEED_RAD_S:
        calibration_reason = "robot trunk is still rotating"
    elif home_error > _CALIBRATION_MAX_HOME_JOINT_ERROR_RAD:
        calibration_reason = "robot body joints are not near HOME"
    elif maximum_joint_speed > _CALIBRATION_MAX_JOINT_SPEED_RAD_S:
        calibration_reason = "robot body joints are still moving"
    return LiveRobotSafetyAssessment(None, calibration_reason)


def _left_trigger_released(frame: Any, *, threshold: float) -> bool:
    """Read the physical left trigger, independently of X/policy selection."""

    controller = (
        frame.get("left_controller")
        if isinstance(frame, Mapping)
        else getattr(frame, "left_controller", None)
    )
    value = (
        controller.get("trigger")
        if isinstance(controller, Mapping)
        else getattr(controller, "trigger", None)
    )
    if isinstance(value, bool):
        return False
    try:
        trigger = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(trigger) and 0.0 <= trigger <= threshold


def _write_live_home_state(command: Any, env_ids: torch.Tensor) -> None:
    """Write explicit HOME root and all joints, including the HMD-owned neck."""

    env = command._env
    robot = command.robot
    command.time_steps[env_ids] = 0
    root_state = robot.data.default_root_state[env_ids].clone()
    root_state[:, :3] += env.scene.env_origins[env_ids]
    robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos[env_ids].clone(),
        robot.data.default_joint_vel[env_ids].clone(),
        env_ids=env_ids,
    )
    robot.reset(env_ids=env_ids)
    command.update_relative_body_poses()


def _install_live_motion_freeze(env: ManagerBasedRlEnv) -> Any:
    """Make the offline MotionCommand a non-advancing HOME-only reset hook."""

    command = env.command_manager.get_term("motion")
    required = (
        "_update_metrics",
        "update_relative_body_poses",
        "time_steps",
        "robot",
    )
    if any(not hasattr(command, name) for name in required):
        raise RuntimeError("tracking motion command lacks the live freeze contract")

    def resample_home(self: Any, env_ids: torch.Tensor) -> None:
        _write_live_home_state(self, env_ids)

    def compute_frozen(self: Any, dt: float) -> None:
        del dt
        self.time_steps.zero_()
        self._update_metrics()
        self.update_relative_body_poses()

    def disable_gui_reset(self: Any, env_ids: torch.Tensor) -> bool:
        del self, env_ids
        return False

    def disable_command_gui(self: Any, *args: Any, **kwargs: Any) -> None:
        del self, args, kwargs

    command._resample_command = MethodType(resample_home, command)
    command.compute = MethodType(compute_frozen, command)
    command.apply_gui_reset = MethodType(disable_gui_reset, command)
    command.create_gui = MethodType(disable_command_gui, command)
    all_env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    _write_live_home_state(command, all_env_ids)
    return command


def _twist_is_zero(command: SimulationCommand) -> bool:
    return all(abs(value) <= 1.0e-6 for value in command.twist)


def _tracking_actor_terms(env: ManagerBasedRlEnv) -> tuple[tuple[str, int], ...]:
    """Resolve the runtime actor term order and dimensions."""

    group = env.observation_manager._group_obs_term_names.get("actor")
    dimensions = env.observation_manager._group_obs_term_dim.get("actor")
    if group is None or dimensions is None or len(group) != len(dimensions):
        raise RuntimeError("tracking actor observation metadata is unavailable")
    resolved: list[tuple[str, int]] = []
    for name, dimension in zip(group, dimensions, strict=True):
        if isinstance(dimension, int):
            width = dimension
        else:
            width = math.prod(dimension)
        resolved.append((str(name), int(width)))
    return tuple(resolved)


def _configure_tracking_environment(cfg: Any) -> None:
    cfg.scene.num_envs = 1
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    # Startup domain randomization makes a live visual test needlessly
    # nondeterministic. The offline MotionCommand remains available so the task
    # can construct its observation managers; ``_install_live_motion_freeze``
    # later turns it into a HOME-only reset hook with no time advancement.
    cfg.events = {}
    for name in ("anchor_pos", "anchor_ori", "ee_body_pos"):
        cfg.terminations.pop(name, None)
    cfg.terminations.pop("time_out", None)
    cfg.commands["motion"].debug_vis = False


class LivePicoTrackingSimulationPolicy:
    """Fail-closed facade from one authenticated PICO frame to one sim action."""

    def __init__(
        self,
        *,
        env: ManagerBasedRlEnv,
        actor: Any,
        source: Any,
        mapper: Any,
        reference_builder: LivePicoTrackingReferenceBuilder | None = None,
        camera_publisher: StereoMjpegPublisher | None = None,
        clock_ns: Any = time.monotonic_ns,
        status_period_s: float = 1.0,
    ) -> None:
        if env.num_envs != 1:
            raise ValueError("live PICO tracking requires exactly one environment")
        runtime_schema = _tracking_actor_terms(env)
        if runtime_schema != LIVE_TRACKING_ACTOR_SCHEMA:
            raise ValueError(
                f"tracking actor schema is {runtime_schema}, expected "
                f"{LIVE_TRACKING_ACTOR_SCHEMA}"
            )
        self.env = env
        self.actor = actor
        self.source = source
        self.mapper = mapper
        self.reference_builder = reference_builder or LivePicoTrackingReferenceBuilder()
        self.camera_publisher = camera_publisher
        self.clock_ns = clock_ns
        self.status_period_ns = int(status_period_s * 1.0e9)
        self.robot = env.scene["robot"]
        self.zero_action = torch.zeros(
            (1, MICROBAN_TELEOP_ACTION_WIDTH),
            dtype=torch.float32,
            device=env.device,
        )
        self._previous_sampled_at_ns: int | None = None
        self._pending_authority_token: Any = None
        self._last_fault: str | None = None
        self._last_ik_error_m: float | None = None
        self._last_status_ns = 0
        self.fall_latch = LiveFallLatch()
        trigger_release = getattr(mapper, "_trigger_release", None)
        if (
            isinstance(trigger_release, bool)
            or not isinstance(trigger_release, (int, float))
            or not math.isfinite(float(trigger_release))
            or not 0.0 <= float(trigger_release) <= 1.0
        ):
            trigger_release = _DEFAULT_TRIGGER_RELEASE_THRESHOLD
        self.trigger_release_threshold = float(trigger_release)

        action = env.action_manager.get_term("joint_pos")
        if tuple(action.target_names) != MICROBAN_TELEOP_ACTION_JOINT_NAMES:
            raise ValueError(
                f"tracking action order is {tuple(action.target_names)}, expected "
                f"{MICROBAN_TELEOP_ACTION_JOINT_NAMES}"
            )
        self.action_scale = _action_parameter_tensor(
            action.scale, MICROBAN_TELEOP_ACTION_WIDTH, env.device
        )
        self.action_offset = _action_parameter_tensor(
            action.offset, MICROBAN_TELEOP_ACTION_WIDTH, env.device
        )
        body_ids, body_names = self.robot.find_joints(
            MICROBAN_TELEOP_ACTION_JOINT_NAMES, preserve_order=True
        )
        if tuple(body_names) != MICROBAN_TELEOP_ACTION_JOINT_NAMES:
            raise ValueError(f"tracking body joint lookup changed order: {body_names}")
        self.body_joint_ids = torch.tensor(
            body_ids, dtype=torch.long, device=env.device
        )
        limits = self.robot.data.soft_joint_pos_limits[:, self.body_joint_ids]
        self.action_lower = limits[..., 0]
        self.action_upper = limits[..., 1]
        if not bool((self.action_lower < self.action_upper).all().item()):
            raise ValueError("tracking action soft limits are invalid")

        hmd_ids, hmd_names = self.robot.find_joints(
            MICROBAN_HMD_JOINT_NAMES, preserve_order=True
        )
        if tuple(hmd_names) != MICROBAN_HMD_JOINT_NAMES:
            raise ValueError(f"unexpected HMD joint order: {hmd_names}")
        self.hmd_joint_ids = torch.tensor(hmd_ids, dtype=torch.long, device=env.device)
        runtime_bounds = torch.tensor(
            [MICROBAN_HMD_RUNTIME_LIMITS_RAD[name] for name in hmd_names],
            dtype=torch.float32,
            device=env.device,
        ).unsqueeze(0)
        hmd_soft = self.robot.data.soft_joint_pos_limits[:, self.hmd_joint_ids]
        self.hmd_lower = torch.maximum(runtime_bounds[..., 0], hmd_soft[..., 0])
        self.hmd_upper = torch.minimum(runtime_bounds[..., 1], hmd_soft[..., 1])
        self.hmd_slew = torch.tensor(
            [MICROBAN_HMD_SLEW_RATES_RAD_S[name] for name in hmd_names],
            dtype=torch.float32,
            device=env.device,
        ).unsqueeze(0)
        self.hmd_default = self.robot.data.default_joint_pos[
            :, self.hmd_joint_ids
        ].clone()
        self.hmd_current_target = self.robot.data.joint_pos[
            :, self.hmd_joint_ids
        ].clone()
        self.home_body_joint_pos = torch.as_tensor(
            self.reference_builder.home,
            dtype=torch.float32,
            device=env.device,
        ).reshape(1, MICROBAN_TELEOP_ACTION_WIDTH)

    def reset(self) -> None:
        self.mapper.reset()
        self.reference_builder.reset()
        self._previous_sampled_at_ns = None
        self._pending_authority_token = None
        self._last_ik_error_m = None
        self.hmd_current_target.copy_(self.robot.data.joint_pos[:, self.hmd_joint_ids])

    def _fault(self, message: str, *, reset_mapper: bool = True) -> SimulationCommand:
        if reset_mapper:
            self.mapper.reset()
        self.reference_builder.reset()
        self._last_ik_error_m = None
        return neutral_simulation_command(fault=message)

    def _read(self) -> tuple[Any | None, Mapping[str, Any] | None, SimulationCommand]:
        try:
            read_with_token = getattr(self.source, "read_with_token", None)
            if callable(read_with_token):
                frame, self._pending_authority_token = read_with_token()
            else:
                frame = self.source.read()
                self._pending_authority_token = None
        except Exception as exc:  # noqa: BLE001 - SDK/network faults fail closed
            self._pending_authority_token = None
            return (
                None,
                None,
                self._fault(f"PICO source read failed: {type(exc).__name__}: {exc}"),
            )

        sampled_at_ns = getattr(frame, "sampled_at_ns", None)
        if (
            not isinstance(sampled_at_ns, int)
            or isinstance(sampled_at_ns, bool)
            or sampled_at_ns <= 0
        ):
            return None, None, self._fault("invalid host sample timestamp")
        previous = self._previous_sampled_at_ns
        self._previous_sampled_at_ns = sampled_at_ns
        if previous is not None:
            delta_ns = sampled_at_ns - previous
            if delta_ns < 0 or delta_ns > int(MAX_FRAME_GAP_S * 1.0e9):
                return (
                    None,
                    None,
                    self._fault(
                        "host sampling gap; trigger release/recalibration required"
                    ),
                )
        age_ns = self.clock_ns() - sampled_at_ns
        if age_ns < 0 or age_ns > int(MAX_FRAME_AGE_S * 1.0e9):
            return (
                None,
                None,
                self._fault("aged PICO frame; trigger release/recalibration required"),
            )

        try:
            mapped = self.mapper.map_sample(frame)
            if not isinstance(mapped, Mapping):
                raise TypeError("mapper output is not a mapping")
            command = command_for_simulation(mapped, legacy_walk_available=False)
        except Exception as exc:  # noqa: BLE001 - mapper faults fail closed
            return (
                None,
                None,
                self._fault(f"PICO mapper failed: {type(exc).__name__}: {exc}"),
            )

        if command.enabled and command.locomotion_policy != "pico_teleop":
            return None, None, self._fault("99-input actor requires pico_teleop mode")
        if command.enabled and not _twist_is_zero(command):
            # The 99-value actor has no velocity-command observation.  Silently
            # accepting the sticks would claim a control path that does not exist.
            return (
                None,
                None,
                self._fault(
                    "99-input tracking has no stick-velocity channel; center both sticks"
                ),
            )
        if command.fault is not None:
            return None, None, self._fault(command.fault)
        return frame, mapped, command

    def _write_hmd_target(self, command: SimulationCommand) -> None:
        force_home = self.fall_latch.latched or command.fault is not None
        if force_home:
            desired_tensor = self.hmd_default.clone()
        else:
            body_quaternion = (
                self.robot.data.root_link_quat_w[0].detach().cpu().tolist()
            )
            desired = solve_hmd_neck_target(
                command.head_orientation,
                body_quaternion,
                yaw_front=command.head_yaw_front,
            )
            if desired is None:
                desired_tensor = self.hmd_default.clone()
            else:
                desired_tensor = torch.tensor(
                    desired, dtype=torch.float32, device=self.env.device
                ).unsqueeze(0)
                desired_tensor = torch.clamp(
                    desired_tensor + self.hmd_default,
                    min=self.hmd_lower,
                    max=self.hmd_upper,
                )
        if not bool(torch.isfinite(self.hmd_current_target).all().item()):
            self.hmd_current_target.copy_(self.hmd_default)
        desired_tensor = torch.clamp(
            desired_tensor,
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

    def _robot_safety(self) -> LiveRobotSafetyAssessment:
        return _assess_live_robot_state(
            root_pos_w=self.robot.data.root_link_pos_w,
            root_quat_wxyz=self.robot.data.root_link_quat_w,
            root_lin_vel=self.robot.data.root_link_lin_vel_b,
            root_ang_vel=self.robot.data.root_link_ang_vel_b,
            body_joint_pos=self.robot.data.joint_pos[:, self.body_joint_ids],
            body_joint_vel=self.robot.data.joint_vel[:, self.body_joint_ids],
            home_body_joint_pos=self.home_body_joint_pos,
            environment_origin_z=float(self.env.scene.env_origins[0, 2].item()),
        )

    def _authority_current_and_write_hmd(self, command: SimulationCommand) -> bool:
        authority = self._pending_authority_token
        guarded = getattr(self.source, "run_if_current", None)
        if authority is None or not callable(guarded):
            self._write_hmd_target(command)
            return True

        injected, _result = guarded(authority, lambda: self._write_hmd_target(command))
        if injected:
            return True
        self.mapper.reset()
        self.reference_builder.reset()
        self._pending_authority_token = None
        self._write_hmd_target(
            neutral_simulation_command(
                fault="native authority changed before simulation action"
            )
        )
        self._last_fault = "native authority changed before simulation action"
        return False

    def _print_status(self, command: SimulationCommand) -> None:
        now_ns = self.clock_ns()
        if now_ns - self._last_status_ns < self.status_period_ns:
            return
        self._last_status_ns = now_ns
        fault = self._last_fault or command.fault or "none"
        ik = (
            "none" if self._last_ik_error_m is None else f"{self._last_ik_error_m:.4f}m"
        )
        print(
            "SIMULATION ONLY | actor=tracking99 "
            f"enabled={command.enabled} calibrated={self.reference_builder.calibrated} "
            f"fall_latched={self.fall_latch.latched} ik={ik} fault={fault}",
            flush=True,
        )

    def __call__(self, observations: Any) -> torch.Tensor:
        reset_buf = getattr(self.env, "reset_buf", None)
        if reset_buf is not None and bool(reset_buf.any().item()):
            self.reset()

        safety = self._robot_safety()
        frame, mapped, command = self._read()
        released = (
            frame is not None
            and mapped is not None
            and command.fault is None
            and _left_trigger_released(frame, threshold=self.trigger_release_threshold)
        )
        latched, entered_latch, cleared_latch = self.fall_latch.update(
            fall_reason=safety.fall_reason,
            left_trigger_released=released,
        )
        if entered_latch:
            self.mapper.reset()
            self.reference_builder.reset()
            self._last_ik_error_m = None

        action = self.zero_action.clone()
        self._last_fault = command.fault
        if latched:
            reason = self.fall_latch.reason or "robot fall latch is active"
            command = neutral_simulation_command(
                fault=f"{reason}; reset upright and release the left trigger"
            )
            self._last_fault = command.fault
        elif cleared_latch:
            # Do not reuse the frame which cleared the latch.  Start a new mapper
            # epoch and require its normal released-trigger body calibration.
            self.mapper.reset()
            self.reference_builder.reset()
            self._previous_sampled_at_ns = None
            command = neutral_simulation_command(
                fault="fall latch cleared; released-trigger recalibration required"
            )
            self._last_fault = command.fault
        elif frame is not None and mapped is not None:
            try:
                body_quaternion = (
                    self.robot.data.root_link_quat_w[0].detach().cpu().tolist()
                )
                reference = self.reference_builder.update(
                    frame,
                    calibration_ready=(
                        mapped.get("body_target_calibrated") is True
                        and mapped.get("body_target_fresh") is True
                    ),
                    robot_calibration_ready=safety.calibration_ready,
                    enabled=command.enabled,
                    robot_trunk_quat_wxyz=body_quaternion,
                )
                if (
                    not command.enabled
                    and reference is None
                    and safety.calibration_block_reason is not None
                ):
                    self._last_fault = safety.calibration_block_reason
                if command.enabled:
                    if reference is None:
                        raise LiveTrackingBridgeError(
                            "left trigger was held before released-trigger calibration"
                        )
                    patched = patch_tracking_actor_observation(
                        observations,
                        reference,
                        current_trunk_quat_wxyz=body_quaternion,
                    )
                    raw_action = self.actor(patched)
                    if raw_action.shape != (1, MICROBAN_TELEOP_ACTION_WIDTH):
                        raise LiveTrackingBridgeError(
                            f"tracking actor output shape is {tuple(raw_action.shape)}"
                        )
                    action = clip_tracking_action_to_soft_limits(
                        raw_action,
                        scale=self.action_scale,
                        offset=self.action_offset,
                        lower=self.action_lower,
                        upper=self.action_upper,
                    )
                    self._last_ik_error_m = reference.ik_error_m
            except Exception as exc:  # noqa: BLE001 - IK/actor faults fail closed
                command = self._fault(
                    f"tracking bridge failed: {type(exc).__name__}: {exc}"
                )
                self._last_fault = command.fault
                action.zero_()

        if not self._authority_current_and_write_hmd(command):
            action.zero_()
        if self.camera_publisher is not None:
            self.camera_publisher.capture_if_due()
        self._print_status(command)
        return action


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Simulation-only PICO Motion Tracker bridge for the Microban "
            "99-observation tracking actor"
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teleop-root", type=Path, default=_default_teleop_root())
    parser.add_argument("--device", default=None)
    parser.add_argument("--viewer", choices=("native", "viser"), default="native")
    parser.add_argument(
        "--input",
        choices=("native", "pico-app"),
        default="pico-app",
        help="both choices require a validated 24-joint body stream",
    )
    parser.add_argument("--native-config", type=Path, default=_default_native_config())
    parser.add_argument("--stale-s", type=_positive_float, default=0.25)
    parser.add_argument("--body-scale", type=_optional_scale)
    parser.add_argument("--hand-scale", type=_optional_scale)
    parser.add_argument("--foot-scale", type=_optional_scale)
    parser.add_argument("--max-ik-error-m", type=_positive_float, default=0.02)
    parser.add_argument("--max-joint-speed", type=_positive_float, default=5.0)
    parser.add_argument("--camera-port", type=_port, default=8081)
    parser.add_argument("--camera-fps", type=_positive_float, default=20.0)
    parser.add_argument("--no-camera", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    if not 0.05 <= args.stale_s < 0.3:
        parser.error("--stale-s must be at least 0.05 and below 0.3 seconds")
    if not 0.005 <= args.max_ik_error_m <= 0.05:
        parser.error("--max-ik-error-m must be in [0.005, 0.05]")
    if not 0.25 <= args.max_joint_speed <= 8.0:
        parser.error("--max-joint-speed must be in [0.25, 8.0]")
    if not 1.0 <= args.camera_fps <= 30.0:
        parser.error("--camera-fps must be in [1, 30]")
    if args.body_scale is not None and (
        args.hand_scale is not None or args.foot_scale is not None
    ):
        parser.error("--body-scale cannot be combined with per-limb overrides")
    if args.input == "pico-app" and not args.native_config.expanduser().is_file():
        parser.error(f"native pairing config not found: {args.native_config}")
    package = args.teleop_root.resolve() / "src" / "microban_teleop"
    if not package.is_dir():
        parser.error(f"microban_teleop package not found below: {args.teleop_root}")


def run(args: argparse.Namespace) -> int:
    configure_torch_backends()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = load_env_cfg(TASK, play=True)
    _configure_tracking_environment(env_cfg)
    agent_cfg = load_rl_cfg(TASK)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    _install_live_motion_freeze(env)
    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    source: Any = None
    native_server_thread: _NativeServerThread | None = None
    camera_publisher: StereoMjpegPublisher | None = None
    try:
        runner_class = load_runner_cls(TASK)
        if runner_class is None:
            raise RuntimeError(f"no runner is registered for {TASK}")
        runner = runner_class(wrapped_env, asdict(agent_cfg), device=device)
        runner.load(
            str(args.checkpoint.resolve()),
            load_cfg={"actor": True},
            strict=True,
            map_location=device,
        )
        actor = runner.get_inference_policy(device=device)

        if not args.no_camera:
            camera_publisher = StereoMjpegPublisher(
                env, port=args.camera_port, fps=args.camera_fps
            )
        mapper_kwargs = {
            "body_scale": args.body_scale,
            "hand_scale_override": args.hand_scale,
            "foot_scale_override": args.foot_scale,
        }
        if args.input == "native":
            source_class, mapper_class = _load_native_classes(args.teleop_root)
            source = source_class(stale_after_ns=int(args.stale_s * 1.0e9))
            if not source.strict_control_presence:
                raise RuntimeError(
                    "the presence-aware pinned XRoboToolkit binding is required"
                )
            mapper = mapper_class(**mapper_kwargs)
        else:
            (
                load_pairing_store,
                server_class,
                source_class,
                mapper_class,
            ) = _load_pico_app_classes(args.teleop_root)
            pairing_store = load_pairing_store(args.native_config.expanduser())
            source = source_class(stale_after_ns=int(args.stale_s * 1.0e9))
            mapper = mapper_class(**mapper_kwargs)
            server = server_class(pairing_store, source=source, on_reset=mapper.reset)
            native_server_thread = _NativeServerThread(server)
            native_server_thread.start()

        reference_builder = LivePicoTrackingReferenceBuilder(
            max_ik_error_m=args.max_ik_error_m,
            max_joint_speed_rad_s=args.max_joint_speed,
        )
        policy = LivePicoTrackingSimulationPolicy(
            env=env,
            actor=actor,
            source=source,
            mapper=mapper,
            reference_builder=reference_builder,
            camera_publisher=camera_publisher,
        )
        print("SIMULATION ONLY: no robot UDP socket or motor interface is opened.")
        print(
            f"Tracking actor contract: {LIVE_TRACKING_ACTOR_WIDTH} values "
            f"{LIVE_TRACKING_ACTOR_SCHEMA}"
        )
        print(
            "Hold left X, keep the left trigger released until calibrated=true, "
            "then hold the left trigger to follow Motion Tracker poses."
        )
        print(
            "Release the left trigger for the exact initial pose. Hold the right "
            "trigger to center head yaw. Keep both sticks centered in tracking99 mode."
        )
        if args.input == "pico-app":
            address = server.address
            print(
                "Authenticated Microban PICO input: "
                + (
                    f"{address[0]}:{address[1]}"
                    if address is not None
                    else "not-started"
                )
            )
        if camera_publisher is not None:
            print(
                f"Simulation SBS MJPEG: http://127.0.0.1:{camera_publisher.port}/stream"
            )
            print(
                "PICO USB camera route: adb reverse "
                f"tcp:{camera_publisher.port} tcp:{camera_publisher.port}"
            )
            if camera_publisher.port != 8081:
                print(
                    "Use the same non-default port in the PICO app's "
                    "camera_transport.json base_url."
                )
        if args.viewer == "native":
            NativeMujocoViewer(wrapped_env, policy).run()
        else:
            ViserPlayViewer(wrapped_env, policy).run()
        return 0
    finally:
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
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
