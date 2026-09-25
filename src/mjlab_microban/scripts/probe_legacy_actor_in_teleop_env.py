"""Run the original 63-input walk actor inside the nominal 83-input teleop task.

This is a bounded feasibility probe, not a deployment gate.  It intentionally
changes only two teleop execution details needed to preserve the legacy actor's
closed-loop contract:

* the joint-position action term has no target clip; and
* the previous-action observation is the raw 18-value actor output.

The 21 joint positions and velocities are mapped by resolved joint name into
the legacy actor's 18-joint order.  The requested twist is injected explicitly,
and the actor's raw 18 actions are passed to the environment unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.utils.nan_guard import NanGuard
from mjlab.utils.torch import configure_torch_backends
from tensordict import TensorDict

from mjlab_microban.legacy_velocity_diagnostics import (
    DEFAULT_LEGACY_VELOCITY_CHECKPOINT,
    DEFAULT_LEGACY_VELOCITY_SHA256,
    TwistScenario,
    checkpoint_sha256,
    default_scenarios,
    publish_json_atomic,
    summarize_samples,
)
from mjlab_microban.tasks.microban_policy_export import MICROBAN_HMD_JOINT_NAMES
from mjlab_microban.tasks.microban_teleop_env_cfg import (
    make_microban_teleop_env_cfg,
)
from mjlab_microban.tasks.microban_velocity_env_cfg import (
    MicrobanVelocityRlCfg,
    make_microban_velocity_env_cfg,
)


@dataclass(frozen=True)
class ActorLayout:
    """Resolved concatenated actor-observation layout."""

    terms: dict[str, slice]
    width: int
    joint_pos_names: tuple[str, ...]
    joint_vel_names: tuple[str, ...]
    action_names: tuple[str, ...]


def _term_slices(env: ManagerBasedRlEnv) -> tuple[dict[str, slice], int]:
    names = env.observation_manager.active_terms["actor"]
    dims = env.observation_manager.group_obs_term_dim["actor"]
    result: dict[str, slice] = {}
    start = 0
    for name, shape in zip(names, dims, strict=True):
        width = math.prod(shape)
        result[name] = slice(start, start + width)
        start += width
    return result, start


def _resolved_joint_names(
    env: ManagerBasedRlEnv, term_name: str
) -> tuple[str, ...]:
    cfg = env.observation_manager.get_term_cfg("actor", term_name)
    asset_cfg = cfg.params.get("asset_cfg")
    names = getattr(asset_cfg, "joint_names", None)
    if not isinstance(names, list) or not names:
        raise ValueError(f"Actor term {term_name!r} has no resolved joint names")
    return tuple(str(name) for name in names)


def _actor_layout(env: ManagerBasedRlEnv) -> ActorLayout:
    terms, width = _term_slices(env)
    required = {
        "base_ang_vel",
        "projected_gravity",
        "joint_pos",
        "joint_vel",
        "actions",
        "command",
    }
    missing = required - terms.keys()
    if missing:
        raise ValueError(f"Actor observation is missing terms: {sorted(missing)}")
    action = env.action_manager.get_term("joint_pos")
    return ActorLayout(
        terms=terms,
        width=width,
        joint_pos_names=_resolved_joint_names(env, "joint_pos"),
        joint_vel_names=_resolved_joint_names(env, "joint_vel"),
        action_names=tuple(action.target_names),
    )


def _indices_by_name(
    source_names: tuple[str, ...], target_names: tuple[str, ...], *, label: str
) -> tuple[int, ...]:
    if len(set(source_names)) != len(source_names):
        raise ValueError(f"Duplicate source {label} names")
    if len(set(target_names)) != len(target_names):
        raise ValueError(f"Duplicate target {label} names")
    target_by_name = {name: index for index, name in enumerate(target_names)}
    missing = [name for name in source_names if name not in target_by_name]
    if missing:
        raise ValueError(f"Teleop observation is missing legacy {label}: {missing}")
    return tuple(target_by_name[name] for name in source_names)


def _assemble_legacy_observation(
    teleop_actor: torch.Tensor,
    *,
    teleop: ActorLayout,
    legacy: ActorLayout,
    joint_pos_indices: tuple[int, ...],
    joint_vel_indices: tuple[int, ...],
    action_indices: tuple[int, ...],
    twist: torch.Tensor,
) -> torch.Tensor:
    """Map one nominal teleop observation into the exact legacy 63 columns."""

    if teleop_actor.ndim != 2 or teleop_actor.shape[1] != teleop.width:
        raise ValueError(
            f"Expected teleop actor shape (*, {teleop.width}), got "
            f"{tuple(teleop_actor.shape)}"
        )
    result = teleop_actor.new_zeros((teleop_actor.shape[0], legacy.width))
    for name in ("base_ang_vel", "projected_gravity"):
        result[:, legacy.terms[name]] = teleop_actor[:, teleop.terms[name]]
    result[:, legacy.terms["joint_pos"]] = teleop_actor[
        :, teleop.terms["joint_pos"]
    ][:, joint_pos_indices]
    result[:, legacy.terms["joint_vel"]] = teleop_actor[
        :, teleop.terms["joint_vel"]
    ][:, joint_vel_indices]
    result[:, legacy.terms["actions"]] = teleop_actor[:, teleop.terms["actions"]][
        :, action_indices
    ]
    if tuple(twist.shape) != (teleop_actor.shape[0], 3):
        raise ValueError(f"Expected twist shape {(teleop_actor.shape[0], 3)}")
    result[:, legacy.terms["command"]] = twist
    return result


def _configure_fixed_twist(command: Any) -> None:
    command.debug_vis = False
    command.heading_command = False
    command.ranges.heading = None
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.rel_world_envs = 0.0
    command.rel_forward_envs = 0.0
    command.rel_rotation_envs = 0.0
    command.init_velocity_prob = 0.0
    command.ranges.lin_vel_x = (0.0, 0.0)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.resampling_time_range = (1.0e6, 1.0e6)


def _source_cfg(*, seed: int, steps: int) -> Any:
    cfg = make_microban_velocity_env_cfg(play=True)
    cfg.scene.num_envs = 1
    cfg.seed = seed
    cfg.auto_reset = False
    cfg.episode_length_s = (steps + 2) * cfg.decimation * cfg.sim.mujoco.timestep
    _configure_fixed_twist(cfg.commands["twist"])
    return cfg


def _teleop_cfg(*, seed: int, steps: int) -> Any:
    cfg = make_microban_teleop_env_cfg(play=True)
    cfg.scene.num_envs = 1
    cfg.seed = seed
    cfg.auto_reset = False
    cfg.episode_length_s = (steps + 2) * cfg.decimation * cfg.sim.mujoco.timestep
    _configure_fixed_twist(cfg.commands["twist"])
    cfg.actions["joint_pos"].clip = None
    raw_action_term = ObservationTermCfg(
        func=velocity_mdp.last_action,
        params={"action_name": "joint_pos"},
    )
    cfg.observations["actor"].terms["actions"] = raw_action_term
    cfg.observations["critic"].terms["actions"] = raw_action_term
    # These three diagnostics intentionally require the bounded teleop action
    # clip.  Rewards cannot affect a fixed inference rollout, and removing them
    # is necessary to exercise the requested raw legacy action semantics.
    for reward_name in ("target_clip_excess", "target_near_limit", "raw_action_l2"):
        cfg.rewards.pop(reward_name, None)
    return cfg


def _tensor_parameter(value: torch.Tensor | float, reference: torch.Tensor) -> torch.Tensor:
    return torch.broadcast_to(
        torch.as_tensor(value, dtype=reference.dtype, device=reference.device),
        reference.shape,
    )


def _termination_names(env: ManagerBasedRlEnv) -> list[str]:
    return [
        name
        for name in env.termination_manager.active_terms
        if bool(env.termination_manager.get_term(name)[0].item())
    ]


def _zero_neutral_targets(env: ManagerBasedRlEnv) -> None:
    foot = env.command_manager.get_term("foot_target")
    hand = env.command_manager.get_term("hand_target")
    foot.foot_target_offset_b.zero_()
    foot.is_single_support_env.fill_(False)
    foot.is_both_feet_env.fill_(False)
    hand.hand_target_offset_b.zero_()
    hand.is_active.fill_(False)


def _targets_are_neutral(env: ManagerBasedRlEnv) -> bool:
    foot = env.command_manager.get_term("foot_target")
    hand = env.command_manager.get_term("hand_target")
    return bool(
        torch.count_nonzero(foot.foot_target_offset_b).item() == 0
        and not foot.is_single_support_env.any().item()
        and not foot.is_both_feet_env.any().item()
        and torch.count_nonzero(hand.hand_target_offset_b).item() == 0
        and not hand.is_active.any().item()
    )


def _evaluate_scenario(
    *,
    env: ManagerBasedRlEnv,
    wrapped: RslRlVecEnvWrapper,
    policy: Any,
    teleop_layout: ActorLayout,
    legacy_layout: ActorLayout,
    joint_pos_indices: tuple[int, ...],
    joint_vel_indices: tuple[int, ...],
    action_indices: tuple[int, ...],
    scenario: TwistScenario,
    seed: int,
    steps: int,
    settle_steps: int,
    policy_observation_mode: str = "legacy63",
) -> dict[str, Any]:
    env.reset(seed=seed)
    command = env.command_manager.get_term("twist")
    expected_twist = torch.tensor(
        scenario.twist, dtype=command.vel_command_b.dtype, device=env.device
    ).unsqueeze(0)
    command.vel_command_b[:] = expected_twist
    _zero_neutral_targets(env)
    observations = wrapped.get_observations()

    robot = env.scene["robot"]
    action = env.action_manager.get_term("joint_pos")
    if action.cfg.clip is not None or wrapped.clip_actions is not None:
        raise ValueError("Probe must pass legacy raw actions without clipping")
    if tuple(action.target_names) != teleop_layout.action_names:
        raise ValueError("Teleop action order drifted after layout capture")

    scale = _tensor_parameter(action.scale, action.raw_action)
    offset = _tensor_parameter(action.offset, action.raw_action)
    target_soft_limits = robot.data.soft_joint_pos_limits[:, action.target_ids]
    all_soft_limits = robot.data.soft_joint_pos_limits
    neck_ids = tuple(robot.joint_names.index(name) for name in MICROBAN_HMD_JOINT_NAMES)
    fall_height = float(
        env.termination_manager.get_term_cfg("fell_over").params["minimum_height"]
    )

    velocity_samples = {name: [] for name in ("vx_m_s", "vy_m_s", "yaw_rad_s")}
    executed_steps = 0
    fell = False
    fall_step: int | None = None
    nonfinite: dict[str, Any] | None = None
    termination_names: list[str] = []
    maximum_actual_soft_limit_violation = 0.0
    maximum_raw_target_soft_limit_excess = 0.0
    maximum_neck_home_position_error = 0.0
    maximum_neck_velocity = 0.0
    minimum_root_height = float(robot.data.root_link_pos_w[0, 2].item())
    neutral_target_verified_steps = 0
    raw_action_recurrence_verified_steps = 0

    for step in range(steps):
        actor_obs = observations["actor"]
        if not bool(torch.isfinite(actor_obs).all().item()):
            nonfinite = {"phase": "teleop_observation", "step": step}
            break
        if policy_observation_mode == "legacy63":
            policy_obs = _assemble_legacy_observation(
                actor_obs,
                teleop=teleop_layout,
                legacy=legacy_layout,
                joint_pos_indices=joint_pos_indices,
                joint_vel_indices=joint_vel_indices,
                action_indices=action_indices,
                twist=expected_twist,
            )
            if tuple(policy_obs.shape) != (1, 63):
                raise ValueError(
                    f"Legacy mapped observation drifted: {policy_obs.shape}"
                )
        elif policy_observation_mode == "teleop83":
            policy_obs = actor_obs
            if tuple(policy_obs.shape) != (1, 83):
                raise ValueError(f"Teleop observation drifted: {policy_obs.shape}")
        else:
            raise ValueError(
                f"Unknown policy observation mode: {policy_observation_mode!r}"
            )
        with torch.inference_mode():
            actions = policy(TensorDict({"actor": policy_obs}, batch_size=[1]))
        if not bool(torch.isfinite(actions).all().item()):
            nonfinite = {"phase": f"{policy_observation_mode}_policy", "step": step}
            break
        if tuple(actions.shape) != (1, 18):
            raise ValueError(f"Legacy action shape drifted: {actions.shape}")

        raw_target = actions * scale + offset
        target_excess = torch.maximum(
            torch.clamp(target_soft_limits[..., 0] - raw_target, min=0.0),
            torch.clamp(raw_target - target_soft_limits[..., 1], min=0.0),
        )
        maximum_raw_target_soft_limit_excess = max(
            maximum_raw_target_soft_limit_excess, float(target_excess.max().item())
        )

        observations, rewards, dones, _extras = wrapped.step(actions)
        executed_steps += 1
        if not bool(torch.equal(action.raw_action, actions)):
            raise RuntimeError("Teleop action term modified the legacy raw action")
        next_raw_action = observations["actor"][:, teleop_layout.terms["actions"]]
        if not bool(torch.equal(next_raw_action, actions)):
            raise RuntimeError("Teleop previous-action observation is not raw action")
        raw_action_recurrence_verified_steps += 1
        if not bool(torch.equal(command.vel_command_b, expected_twist)):
            raise RuntimeError(f"Twist command drifted in {scenario.name}")
        if not _targets_are_neutral(env):
            raise RuntimeError(f"Neutral foot/hand targets drifted in {scenario.name}")
        neutral_target_verified_steps += 1

        finite = all(
            bool(torch.isfinite(value).all().item())
            for value in (
                observations["actor"],
                rewards,
                robot.data.joint_pos,
                robot.data.joint_vel,
                robot.data.root_link_pose_w,
                robot.data.root_link_vel_w,
            )
        )
        if bool(NanGuard.detect_nans(env.sim.data)[0].item()) or not finite:
            nonfinite = {"phase": "physics", "step": step}
            break

        root_height = float(robot.data.root_link_pos_w[0, 2].item())
        minimum_root_height = min(minimum_root_height, root_height)
        current_fell = bool(
            env.termination_manager.get_term("fell_over")[0].item()
        ) or root_height < fall_height
        if current_fell and not fell:
            fell = True
            fall_step = executed_steps

        actual_violation = torch.maximum(
            torch.clamp(all_soft_limits[..., 0] - robot.data.joint_pos, min=0.0),
            torch.clamp(robot.data.joint_pos - all_soft_limits[..., 1], min=0.0),
        )
        maximum_actual_soft_limit_violation = max(
            maximum_actual_soft_limit_violation,
            float(actual_violation.max().item()),
        )
        neck_pos_error = torch.abs(
            robot.data.joint_pos[:, neck_ids]
            - robot.data.default_joint_pos[:, neck_ids]
        )
        maximum_neck_home_position_error = max(
            maximum_neck_home_position_error, float(neck_pos_error.max().item())
        )
        maximum_neck_velocity = max(
            maximum_neck_velocity,
            float(torch.abs(robot.data.joint_vel[:, neck_ids]).max().item()),
        )

        if step >= settle_steps:
            measured = (
                float(robot.data.root_link_lin_vel_b[0, 0].item()),
                float(robot.data.root_link_lin_vel_b[0, 1].item()),
                float(robot.data.root_link_ang_vel_b[0, 2].item()),
            )
            for name, value in zip(velocity_samples, measured, strict=True):
                velocity_samples[name].append(value)
        if bool(dones[0].item()):
            termination_names = _termination_names(env)
            break

    velocity = {
        name: summarize_samples(values) for name, values in velocity_samples.items()
    }
    nonzero = [
        index for index, value in enumerate(scenario.twist) if not math.isclose(value, 0.0)
    ]
    directional_response: dict[str, Any] | None = None
    if len(nonzero) == 1:
        index = nonzero[0]
        axis = ("vx_m_s", "vy_m_s", "yaw_rad_s")[index]
        measured_mean = velocity[axis]["mean"]
        signed = (
            None
            if measured_mean is None
            else float(measured_mean)
            * (1.0 if scenario.twist[index] > 0.0 else -1.0)
        )
        directional_response = {
            "axis": axis,
            "command": scenario.twist[index],
            "measured_mean": measured_mean,
            "signed_response": signed,
            "sign_matches": signed is not None and signed > 0.0,
        }
    completed = (
        executed_steps == steps
        and not fell
        and nonfinite is None
        and not termination_names
    )
    return {
        "name": scenario.name,
        "command": {
            "vx_m_s": scenario.twist[0],
            "vy_m_s": scenario.twist[1],
            "yaw_rad_s": scenario.twist[2],
        },
        "completed": completed,
        "executed_steps": executed_steps,
        "fell": fell,
        "fall_step": fall_step,
        "nonfinite": nonfinite,
        "termination_names": termination_names,
        "minimum_root_height_m": minimum_root_height,
        "measured_velocity_body": velocity,
        "directional_response": directional_response,
        "maximum_actual_soft_limit_violation_rad": (
            maximum_actual_soft_limit_violation
        ),
        "maximum_hypothetical_raw_target_soft_limit_excess_rad": (
            maximum_raw_target_soft_limit_excess
        ),
        "neutral_foot_hand_target_verified_steps": neutral_target_verified_steps,
        "raw_action_recurrence_verified_steps": (
            raw_action_recurrence_verified_steps
        ),
        "neck_home": {
            "joint_names": list(MICROBAN_HMD_JOINT_NAMES),
            "maximum_position_error_rad": maximum_neck_home_position_error,
            "maximum_velocity_rad_s": maximum_neck_velocity,
        },
    }


def run_probe(
    *,
    checkpoint: Path,
    expected_sha256: str,
    device: str,
    seed: int,
    steps: int,
    settle_steps: int,
) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve()
    digest = checkpoint_sha256(checkpoint)
    if digest != expected_sha256:
        raise ValueError(f"Legacy checkpoint SHA-256 mismatch: {digest}")
    if steps < 1 or not 0 <= settle_steps < steps:
        raise ValueError("Require steps > settle_steps >= 0")

    configure_torch_backends(allow_tf32=False, deterministic=True)
    torch.use_deterministic_algorithms(True, warn_only=True)

    source_env = ManagerBasedRlEnv(cfg=_source_cfg(seed=seed, steps=steps), device=device)
    source_wrapped = RslRlVecEnvWrapper(
        source_env, clip_actions=MicrobanVelocityRlCfg.clip_actions
    )
    try:
        legacy_layout = _actor_layout(source_env)
        source_action = source_env.action_manager.get_term("joint_pos")
        source_scale = _tensor_parameter(source_action.scale, source_action.raw_action).clone()
        source_offset = _tensor_parameter(
            source_action.offset, source_action.raw_action
        ).clone()
        runner = VelocityOnPolicyRunner(
            source_wrapped, asdict(MicrobanVelocityRlCfg), device=device
        )
        runner.load(
            str(checkpoint),
            load_cfg={"actor": True},
            strict=True,
            map_location=device,
        )
        policy = runner.get_inference_policy(device=device)
    finally:
        source_wrapped.close()

    teleop_env = ManagerBasedRlEnv(cfg=_teleop_cfg(seed=seed, steps=steps), device=device)
    teleop_wrapped = RslRlVecEnvWrapper(teleop_env, clip_actions=None)
    try:
        teleop_layout = _actor_layout(teleop_env)
        if legacy_layout.width != 63 or teleop_layout.width != 83:
            raise ValueError(
                f"Unexpected actor widths legacy={legacy_layout.width}, "
                f"teleop={teleop_layout.width}"
            )
        joint_pos_indices = _indices_by_name(
            legacy_layout.joint_pos_names,
            teleop_layout.joint_pos_names,
            label="joint-position",
        )
        joint_vel_indices = _indices_by_name(
            legacy_layout.joint_vel_names,
            teleop_layout.joint_vel_names,
            label="joint-velocity",
        )
        action_indices = _indices_by_name(
            legacy_layout.action_names,
            teleop_layout.action_names,
            label="previous-action",
        )
        teleop_action = teleop_env.action_manager.get_term("joint_pos")
        teleop_scale = _tensor_parameter(
            teleop_action.scale, teleop_action.raw_action
        )
        teleop_offset = _tensor_parameter(
            teleop_action.offset, teleop_action.raw_action
        )
        if not bool(torch.equal(source_scale, teleop_scale)):
            raise ValueError("Legacy and teleop raw-action scales differ")
        if not bool(torch.equal(source_offset, teleop_offset)):
            raise ValueError("Legacy and teleop raw-action offsets differ")

        results = []
        for scenario in default_scenarios():
            print(f"[INFO] probing teleop mapping {scenario.name}", flush=True)
            results.append(
                _evaluate_scenario(
                    env=teleop_env,
                    wrapped=teleop_wrapped,
                    policy=policy,
                    teleop_layout=teleop_layout,
                    legacy_layout=legacy_layout,
                    joint_pos_indices=joint_pos_indices,
                    joint_vel_indices=joint_vel_indices,
                    action_indices=action_indices,
                    scenario=scenario,
                    seed=seed,
                    steps=steps,
                    settle_steps=settle_steps,
                )
            )
    finally:
        teleop_wrapped.close()
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    directional = [
        result["directional_response"]
        for result in results
        if result["directional_response"] is not None
    ]
    shared_target_columns: list[int] = []
    for term_name in ("base_ang_vel", "projected_gravity"):
        shared_target_columns.extend(
            range(
                teleop_layout.terms[term_name].start,
                teleop_layout.terms[term_name].stop,
            )
        )
    shared_target_columns.extend(
        teleop_layout.terms["joint_pos"].start + index
        for index in joint_pos_indices
    )
    shared_target_columns.extend(
        teleop_layout.terms["joint_vel"].start + index
        for index in joint_vel_indices
    )
    shared_target_columns.extend(
        teleop_layout.terms["actions"].start + index for index in action_indices
    )
    shared_target_columns.extend(
        range(
            teleop_layout.terms["command"].start,
            teleop_layout.terms["command"].stop,
        )
    )
    return {
        "probe": "legacy_velocity_actor_in_nominal_teleop_env_v1",
        "checkpoint": {"path": str(checkpoint), "sha256": digest},
        "settings": {
            "device": device,
            "seed": seed,
            "steps": steps,
            "settle_steps": settle_steps,
            "step_dt_s": teleop_env.step_dt,
            "action_clip": None,
            "previous_action": "raw_actor_output",
            "foot_target": "exact_zero_inactive",
            "hand_target": "exact_zero_inactive",
        },
        "mapping": {
            "legacy_observation_width": legacy_layout.width,
            "teleop_observation_width": teleop_layout.width,
            "legacy_joint_names": list(legacy_layout.joint_pos_names),
            "teleop_joint_names": list(teleop_layout.joint_pos_names),
            "teleop_joint_indices_for_legacy": list(joint_pos_indices),
            "legacy_action_names": list(legacy_layout.action_names),
            "teleop_action_indices_for_legacy": list(action_indices),
            "legacy_source_columns_to_teleop_target_columns": [
                [source, target]
                for source, target in enumerate(shared_target_columns)
            ],
            "new_teleop_columns": sorted(
                set(range(teleop_layout.width)) - set(shared_target_columns)
            ),
        },
        "results": results,
        "summary": {
            "scenario_count": len(results),
            "completed_scenario_count": sum(
                bool(result["completed"]) for result in results
            ),
            "fall_scenario_count": sum(bool(result["fell"]) for result in results),
            "nonfinite_scenario_count": sum(
                result["nonfinite"] is not None for result in results
            ),
            "actual_soft_limit_violation_scenario_count": sum(
                float(result["maximum_actual_soft_limit_violation_rad"]) > 1.0e-7
                for result in results
            ),
            "maximum_actual_soft_limit_violation_rad": max(
                float(result["maximum_actual_soft_limit_violation_rad"])
                for result in results
            ),
            "directionally_correct_scenario_count": sum(
                bool(response["sign_matches"]) for response in directional
            ),
            "directional_scenario_count": len(directional),
            "maximum_neck_home_position_error_rad": max(
                float(result["neck_home"]["maximum_position_error_rad"])
                for result in results
            ),
            "maximum_neck_velocity_rad_s": max(
                float(result["neck_home"]["maximum_velocity_rad_s"])
                for result in results
            ),
            "neutral_target_contract_all_steps": all(
                result["neutral_foot_hand_target_verified_steps"]
                == result["executed_steps"]
                for result in results
            ),
            "raw_action_recurrence_all_steps": all(
                result["raw_action_recurrence_verified_steps"]
                == result["executed_steps"]
                for result in results
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_LEGACY_VELOCITY_CHECKPOINT)
    parser.add_argument("--expected-sha256", default=DEFAULT_LEGACY_VELOCITY_SHA256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--settle-steps", type=int, default=50)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_probe(
        checkpoint=args.checkpoint,
        expected_sha256=args.expected_sha256,
        device=args.device,
        seed=args.seed,
        steps=args.steps,
        settle_steps=args.settle_steps,
    )
    if args.output is not None:
        if args.output.expanduser().exists() and not args.force:
            raise FileExistsError(f"Output already exists (pass --force): {args.output}")
        output = publish_json_atomic(args.output, report)
        report = {"output": str(output), "summary": report["summary"]}
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
