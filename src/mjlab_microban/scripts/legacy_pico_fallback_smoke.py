# Copyright 2026 Marc Duclusaud
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Hardware-free integration smoke for the legacy PICO deadline fallback."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import math
import socket
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from tensordict import TensorDict

from mjlab_microban.legacy_velocity_diagnostics import (
    build_report,
    default_scenarios,
)
from mjlab_microban.scripts.evaluate_legacy_velocity_checkpoint import (
    _evaluate_scenario,
)
from mjlab_microban.scripts.live_pico_teleop_sim import (
    AUDITED_LEGACY_WALK_SHA256,
    WALK_TASK,
    _configure_live_environment,
    _default_teleop_root,
    _default_walk_checkpoint,
    _load_legacy_walk_actor,
    _load_walk_actor_in_environment,
    _sha256,
    command_for_simulation,
    solve_hmd_neck_target,
)
from mjlab_microban.scripts.simulation_camera import (
    CAMERA_SCHEMA,
    StereoMjpegPublisher,
    camera_geometry_sha256,
)
from mjlab_microban.tasks.mdp import UniformVelocityCommandWithRotation


def _control_frame(
    *,
    orientation: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
    left_trigger: float = 0.0,
    left_stick: Sequence[float] = (0.0, 0.0),
    right_trigger: float = 0.0,
    right_stick: Sequence[float] = (0.0, 0.0),
) -> dict[str, Any]:
    return {
        "type": "control",
        "hmd": {"orientation": list(orientation)},
        "left": {
            "trigger": left_trigger,
            "squeeze": 0.0,
            "stick": list(left_stick),
            "primary_button": False,
        },
        "right": {
            "trigger": right_trigger,
            "squeeze": 0.0,
            "stick": list(right_stick),
            "primary_button": False,
        },
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _control_mapper_class(teleop_root: Path) -> Any:
    source_root = str(teleop_root.resolve() / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    return importlib.import_module("microban_teleop.control").ControlMapper


def _exercise_synthetic_controls(teleop_root: Path) -> dict[str, Any]:
    mapper = _control_mapper_class(teleop_root)(deadzone=0.0)

    released = command_for_simulation(
        mapper.map_frame(_control_frame()), legacy_walk_available=True
    )
    _require(not released.enabled, "released left trigger did not hold neutral")
    _require(released.twist == (0.0, 0.0, 0.0), "released twist was not zero")

    cases = {
        "forward": (
            _control_frame(left_trigger=1.0, left_stick=(0.0, -0.5)),
            (0.35, 0.0, 0.0),
        ),
        "lateral": (
            _control_frame(left_trigger=1.0, left_stick=(-0.5, 0.0)),
            (0.0, 0.15, 0.0),
        ),
        "yaw": (
            _control_frame(left_trigger=1.0, right_stick=(-0.5, 0.0)),
            (0.0, 0.0, 1.5),
        ),
    }
    mapped_twists: dict[str, tuple[float, float, float]] = {}
    for name, (frame, expected) in cases.items():
        command = command_for_simulation(
            mapper.map_frame(frame), legacy_walk_available=True
        )
        _require(command.enabled, f"{name} did not arm while trigger was held")
        _require(command.locomotion_policy == "walk", f"{name} left legacy policy")
        _require(
            all(
                math.isclose(actual, target, rel_tol=0.0, abs_tol=1.0e-7)
                for actual, target in zip(command.twist, expected, strict=True)
            ),
            f"{name} twist mismatch: {command.twist} != {expected}",
        )
        mapped_twists[name] = command.twist
        released = command_for_simulation(
            mapper.map_frame(_control_frame()), legacy_walk_available=True
        )
        _require(not released.enabled, f"{name} remained armed after trigger release")

    # Establish an identity HMD reference, then apply a finite mixed quaternion.
    mapper.map_frame(_control_frame())
    quaternion = (0.10, -0.15, 0.05, math.sqrt(1.0 - 0.035))
    mapped_head = mapper.map_frame(
        _control_frame(orientation=quaternion, right_trigger=1.0)
    )
    head = mapped_head["head_orientation"]
    _require(isinstance(head, dict), "HMD mapper did not produce an orientation")
    head_tuple = tuple(float(head[name]) for name in ("roll", "pitch", "yaw"))
    _require(any(abs(value) > 1.0e-4 for value in head_tuple), "HMD path stayed zero")
    neck = solve_hmd_neck_target(
        head_tuple,
        (1.0, 0.0, 0.0, 0.0),
        yaw_front=bool(mapped_head["head_yaw_front"]),
    )
    _require(neck is not None, "HMD neck solver rejected a finite synthetic pose")
    assert neck is not None
    _require(neck[0] == 0.0, "held right trigger did not center neck yaw")
    _require(all(math.isfinite(value) for value in neck), "neck target was non-finite")

    return {
        "release_neutral": True,
        "hold_to_run": True,
        "twists_m_s_rad_s": mapped_twists,
        "hmd_orientation_rad": head_tuple,
        "right_trigger_neck_target_rad": neck,
    }


def _exercise_actor(checkpoint: Path) -> dict[str, Any]:
    with contextlib.redirect_stdout(io.StringIO()):
        walk_actor = _load_legacy_walk_actor(checkpoint, "cpu")

    outputs: dict[str, torch.Tensor] = {}
    commands = {
        "neutral": (0.0, 0.0, 0.0),
        "forward": (0.35, 0.0, 0.0),
        "lateral": (0.0, 0.15, 0.0),
        "yaw": (0.0, 0.0, 1.5),
    }
    for name, command in commands.items():
        actor_tensor = torch.zeros((1, 63), dtype=torch.float32)
        actor_tensor[:, -3:] = torch.tensor(command, dtype=torch.float32)
        observation = TensorDict({"actor": actor_tensor}, batch_size=(1,))
        action = walk_actor.policy(observation)
        _require(tuple(action.shape) == (1, 18), f"{name} actor shape is {action.shape}")
        _require(bool(torch.isfinite(action).all().item()), f"{name} action is non-finite")
        outputs[name] = action.detach().cpu()

    neutral = outputs["neutral"]
    for name in ("forward", "lateral", "yaw"):
        _require(
            not bool(torch.allclose(outputs[name], neutral, atol=1.0e-7, rtol=0.0)),
            f"legacy actor ignored the {name} command input",
        )
    return {
        "observation_width": 63,
        "action_width": 18,
        "finite": True,
        "command_sensitive": True,
        "joint_names": list(walk_actor.joint_names),
    }


def _unused_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _read_json(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=3.0) as response:
        return json.loads(response.read())


def _exercise_camera() -> dict[str, Any]:
    env_cfg = load_env_cfg(WALK_TASK, play=True)
    _configure_live_environment(env_cfg)
    port = _unused_loopback_port()
    with contextlib.redirect_stdout(io.StringIO()):
        env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
    publisher: StereoMjpegPublisher | None = None
    try:
        publisher = StereoMjpegPublisher(env, port=port, fps=20.0)
        _require(publisher.capture_if_due(), "camera did not capture its first frame")
        health = _read_json(f"http://127.0.0.1:{port}/healthz")
        calibration = _read_json(f"http://127.0.0.1:{port}/calibration.json")
        _require(health.get("ok") is True, "camera health endpoint is not ready")
        _require(
            health.get("geometry_sha256") == camera_geometry_sha256(),
            "camera health geometry digest mismatch",
        )
        _require(
            calibration.get("geometry", {}).get("schema") == CAMERA_SCHEMA,
            "camera calibration schema mismatch",
        )
        with urlopen(
            f"http://127.0.0.1:{port}/frame.jpg?after=0&wait_ms=0",
            timeout=3.0,
        ) as response:
            jpeg = response.read()
        _require(
            jpeg.startswith(b"\xff\xd8") and jpeg.endswith(b"\xff\xd9"),
            "camera latest-frame endpoint did not return a complete JPEG",
        )
        geometry = health["geometry"]
        return {
            "loopback_http_started": True,
            "first_frame_jpeg": True,
            "schema": geometry["schema"],
            "horizontal_fov_deg": geometry["horizontal_fov_deg"],
            "vertical_fov_deg": geometry["vertical_fov_deg"],
            "eye_width_px": geometry["eye_width_px"],
            "eye_height_px": geometry["eye_height_px"],
            "geometry_sha256": health["geometry_sha256"],
        }
    finally:
        if publisher is not None:
            publisher.close()
        env.close()


def _exercise_closed_loop(
    checkpoint: Path,
    *,
    device: str,
    steps: int,
    settle_steps: int,
) -> dict[str, Any]:
    """Run all nine signed-axis scenarios in the exact legacy live runtime."""

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    cfg = load_env_cfg(WALK_TASK, play=True)
    _configure_live_environment(cfg)
    cfg.seed = 42
    # The live policy overwrites the twist term on every 50 Hz call. Freeze the
    # task sampler here so _evaluate_scenario's one installed synthetic command
    # is behaviorally identical and cannot be replaced after 3--8 seconds.
    command_cfg = cfg.commands["twist"]
    command_cfg.heading_command = False
    command_cfg.ranges.heading = None
    command_cfg.rel_standing_envs = 0.0
    command_cfg.rel_heading_envs = 0.0
    command_cfg.rel_world_envs = 0.0
    command_cfg.rel_forward_envs = 0.0
    command_cfg.rel_rotation_envs = 0.0
    command_cfg.init_velocity_prob = 0.0
    command_cfg.ranges.lin_vel_x = (0.0, 0.0)
    command_cfg.ranges.lin_vel_y = (0.0, 0.0)
    command_cfg.ranges.ang_vel_z = (0.0, 0.0)
    command_cfg.resampling_time_range = (1.0e6, 1.0e6)
    # The registered task carries a closure-created command builder. Rebind it
    # after these changes so the term receives this exact fixed-command config.
    command_cfg.build = (
        lambda env, _cmd=command_cfg: UniformVelocityCommandWithRotation(_cmd, env)
    )
    env: ManagerBasedRlEnv | None = None
    wrapped = None
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            env = ManagerBasedRlEnv(cfg=cfg, device=device)
            agent_cfg = load_rl_cfg(WALK_TASK)
            wrapped = RslRlVecEnvWrapper(
                env, clip_actions=agent_cfg.clip_actions
            )
            _require(
                wrapped.clip_actions is None,
                "native legacy wrapper unexpectedly clips raw actions",
            )
            actor = _load_walk_actor_in_environment(
                raw_env=env,
                wrapped_env=wrapped,
                agent_cfg=agent_cfg,
                checkpoint=checkpoint,
                device=device,
            )
            results = [
                _evaluate_scenario(
                    env=env,
                    wrapped=wrapped,
                    policy=actor.policy,
                    scenario=scenario,
                    steps=steps,
                    settle_steps=settle_steps,
                    seed=42,
                    execute_soft_limit_projection=False,
                )
                for scenario in default_scenarios()
            ]
        report = build_report(
            checkpoint=checkpoint,
            checkpoint_sha256=_sha256(checkpoint),
            device=device,
            seed=42,
            steps=steps,
            settle_steps=settle_steps,
            step_dt_s=env.step_dt,
            results=results,
            execute_soft_limit_projection=False,
        )
        summary = report["summary"]
        _require(summary["scenario_count"] == 9, "nine-scenario parity was lost")
        _require(
            summary["completed_scenario_count"] == 9,
            "not every native-runtime scenario completed",
        )
        _require(summary["fall_scenario_count"] == 0, "native runtime fell")
        _require(
            summary["nonfinite_scenario_count"] == 0,
            "native runtime produced non-finite state",
        )
        _require(
            summary["directionally_correct_scenario_count"] == 8,
            "one or more signed commands moved in the wrong direction",
        )
        _require(
            summary["actual_soft_limit_violation_scenario_count"] == 0,
            "native runtime exceeded a simulated joint soft limit",
        )
        env.reset(seed=42)
        action_term = env.action_manager.get_term("joint_pos")
        probe = torch.full_like(action_term.raw_action, 0.25)
        observations, _reward, _done, _extras = wrapped.step(probe)
        _require(
            torch.equal(observations["actor"][:, 42:60], probe),
            "native previous-action observation did not retain an executed action",
        )
        zero = torch.zeros_like(probe)
        observations, _reward, _done, _extras = wrapped.step(zero)
        _require(
            torch.equal(action_term.raw_action, zero)
            and torch.equal(observations["actor"][:, 42:60], zero),
            "trigger-release raw zero did not clear native action recurrence",
        )
        offset = torch.broadcast_to(
            torch.as_tensor(
                action_term.offset,
                dtype=zero.dtype,
                device=zero.device,
            ),
            zero.shape,
        )
        default = env.scene["robot"].data.default_joint_pos[
            :, action_term.target_ids
        ]
        _require(
            torch.equal(offset, default),
            "trigger-release raw zero does not map exactly to HOME",
        )
        return {
            "task": WALK_TASK,
            "actor_observation_width": 63,
            "action_width": 18,
            "action_target_clip": None,
            "wrapper_action_clip": None,
            "policy_output_projection_applied": False,
            "release_zero_clears_action_history": True,
            "release_zero_maps_to_home": True,
            "steps_per_scenario": steps,
            "settle_steps": settle_steps,
            "summary": summary,
        }
    finally:
        if wrapped is not None:
            wrapped.close()
        elif env is not None:
            env.close()
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()


def run(
    checkpoint: Path,
    teleop_root: Path,
    *,
    closed_loop_device: str,
    closed_loop_steps: int,
    closed_loop_settle_steps: int,
) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    teleop_root = teleop_root.resolve()
    _require(checkpoint.is_file(), f"checkpoint not found: {checkpoint}")
    digest = _sha256(checkpoint)
    _require(
        digest == AUDITED_LEGACY_WALK_SHA256,
        f"legacy checkpoint digest mismatch: {digest}",
    )
    _require(
        (teleop_root / "src" / "microban_teleop").is_dir(),
        f"microban_teleop package not found below: {teleop_root}",
    )

    configure_torch_backends()
    torch.manual_seed(0)
    return {
        "status": "pass",
        "hardware_required": False,
        "checkpoint": {"path": str(checkpoint), "sha256": digest},
        "controls": _exercise_synthetic_controls(teleop_root),
        "legacy_actor": _exercise_actor(checkpoint),
        "closed_loop": _exercise_closed_loop(
            checkpoint,
            device=closed_loop_device,
            steps=closed_loop_steps,
            settle_steps=closed_loop_settle_steps,
        ),
        "camera": _exercise_camera(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--walk-checkpoint", type=Path, default=_default_walk_checkpoint()
    )
    parser.add_argument("--teleop-root", type=Path, default=_default_teleop_root())
    parser.add_argument("--closed-loop-device", default="cuda:0")
    parser.add_argument("--closed-loop-steps", type=int, default=300)
    parser.add_argument("--closed-loop-settle-steps", type=int, default=50)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if not 0 <= args.closed_loop_settle_steps < args.closed_loop_steps:
            raise ValueError("closed-loop steps must exceed settle steps >= 0")
        report = run(
            args.walk_checkpoint,
            args.teleop_root,
            closed_loop_device=args.closed_loop_device,
            closed_loop_steps=args.closed_loop_steps,
            closed_loop_settle_steps=args.closed_loop_settle_steps,
        )
    except Exception as exc:
        report = {
            "status": "fail",
            "error": f"{type(exc).__name__}: {exc}",
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(1) from exc
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
