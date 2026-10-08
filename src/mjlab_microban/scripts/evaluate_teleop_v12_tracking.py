"""The PICO judgment of the checkpoint a PICO run ends with (J1-J4).

Every part runs 64 environments per scenario on two seeds with the HMD neck
moving as in training (scenarios: teleop_v12_scenarios.py).  The arms are
driven from outside as on the robot (pico_arms): the evaluator writes the arm
goal of the arm-overlay action, the policy's arm outputs are not used.

* J1 feet: a policy that does not lift the foot fails ``foot_lift`` and
  ``foot_error``; the foot is measured above the floor and from the support
  foot, so a moving trunk does not count as a tracked foot.
* J2 pushes: falls are compared with the walker the adapter was built on.
* J3 standing still: standing with the HMD and the arms moving.
* J4 walking with the arms raised or moving: the speed of arms at HOME.

The raw action contract has no software target clip (the target saturates
only at the servo's +-pi goal range), so non-finite values, broken raw-action
recurrence and joint soft-limit violations are rejected too.  The per-joint
action envelopes (policy, source walker, difference) set the robot's runtime
action guard, and the recorded actor observations its startup self-test.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.nan_guard import NanGuard
from mjlab.utils.torch import configure_torch_backends
from tensordict import TensorDict

from mjlab_microban.legacy_velocity_diagnostics import publish_json_atomic
from mjlab_microban.pipeline.walk_probe import _home_levelled_twist
from mjlab_microban.robot.microban_constants import HOME_TRUNK_PITCH_RAD
from mjlab_microban.schedules import PICO_SCHEDULE, PICO_TOTAL_UPDATES
from mjlab_microban.scripts.evaluate_teleop_v12_checkpoint import _load_actor
from mjlab_microban.scripts.teleop_v12_bootstrap_gate import _legacy_model
from mjlab_microban.scripts.teleop_v12_scenarios import (
    ARM_JOINT_NAMES,
    ARM_MODES,
    ARM_STEPS,
    ARMS_RAISED_FORWARD_RAD,
    ENVS_PER_SCENARIO,
    FOOT_SCORE_FROM_STEP,
    FOOT_STEPS,
    FOOT_TARGET_STEP,
    HMD_ACTUAL_PEAK_TO_PEAK_MIN_RAD,
    HMD_TARGET_PEAK_TO_PEAK_MIN_RAD,
    MOVING_ARM_EVENT,
    PUSH_DIRECTIONS,
    PUSH_EVERY_STEPS,
    PUSH_SPEED_M_S,
    PUSH_STEPS,
    SCENARIOS_PER_RUN,
    SETTLE_STEPS,
    FootTargetRamp,
    Scenario,
    actor_term_slice,
    arm_scenarios,
    configure_nominal_evaluation,
    copy_forced_moving_hmd_neck_event,
    copy_moving_arm_event,
    foot_scenarios,
    mirrored_foot_pairs,
    patch_observation,
    per_env,
    push_scenarios,
    push_velocity,
    scenario_index,
    write_commands,
)
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_HMD_JOINT_NAMES,
    MICROBAN_TELEOP_ACTION_JOINT_NAMES,
)
from mjlab_microban.tasks.microban_teleop_mdp import HmdNeckTargetMotion
from mjlab_microban.tasks.microban_teleop_v12_actor import (
    LEGACY_TO_TELEOP_OBSERVATION_INDEX,
)
from mjlab_microban.tasks.microban_teleop_v12_bootstrap import (
    load_bootstrap_source_state,
    sha256_file,
    validate_bootstrap_provenance,
)
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MICROBAN_TELEOP_V12_ACTION_CLIP,
    make_microban_teleop_v12_env_cfg,
)
from mjlab_microban.tasks.microban_teleop_v12_runner import (
    TELEOP_V12_BOOTSTRAP_INFO_KEY,
    validate_teleop_v12_environment_contract,
)
from mjlab_microban.teleop_v12_safety import ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD
from mjlab_microban.twist_pass_line import twist_pass_line_record, twist_passes

FINAL_PROFILE = "pico_feet_push_still_arms_v1"
TRACKING_PROFILES = (FINAL_PROFILE,)
SEED_COUNT = 2

# J1.  The foot reward is exp(-mean_feet |e|^2 / 0.03^2) at the end of
# training: a foot 0.3 x 52 mm off scores 0.87, a foot that stays down 0.22.
FOOT_LIFT_MIN_SHARE = 0.7  # 40 mm target: 12 mm short alone still scores 0.92
FOOT_ERROR_SHARE = 0.3
FOOT_ERROR_FLOOR_M = 0.008  # the floor for small targets: still scores 0.965
# Each foot is rewarded against its own reset position: a support foot 10 mm
# off costs 5 % of the reward; the mirrored halves have the same reward.
SUPPORT_FOOT_MOVE_MAX_M = 0.010
FOOT_LEFT_RIGHT_MAX_M = 0.005
# J2.  The adapter keeps the walker's velocity and survival rewards and its
# pushes (+-0.5 m/s): it may fall at most 5 points more often than the walker.
PUSH_FALL_RATE_MARGIN = 0.05
# J3.  The no-stepping penalty counts airborne feet on a standing command; the
# walking judgment allows 0.5 touchdowns per second (config/pipeline.yaml).
STILL_TOUCHDOWNS_PER_S_MAX = 0.5
# J4.  No reward depends on the arm pose, and the arms are driven from outside:
# the walking speed is the speed with the arms at HOME, within 10 %.
ARM_SPEED_RATIO_RANGE = (0.9, 1.1)

CHECK_NAMES = frozenset(
    (
        "finite",
        "actual_soft_limits",
        "raw_action_recurrence",
        "forced_hmd_motion",
        "arm_observation",
        "no_falls_without_push",
        "foot_lift",
        "foot_error",
        "support_foot_still",
        "foot_left_right",
        "push_falls",
        "standing_still",
        "walking_with_arms",
    )
)


def thresholds() -> dict[str, Any]:
    return {
        "actual_soft_limit_violation_rad_max": ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD,
        "hmd_target_peak_to_peak_rad_min": HMD_TARGET_PEAK_TO_PEAK_MIN_RAD,
        "hmd_actual_peak_to_peak_rad_min": HMD_ACTUAL_PEAK_TO_PEAK_MIN_RAD,
        "foot_lift_min_share": FOOT_LIFT_MIN_SHARE,
        "foot_error_share": FOOT_ERROR_SHARE,
        "foot_error_floor_m": FOOT_ERROR_FLOOR_M,
        "support_foot_move_max_m": SUPPORT_FOOT_MOVE_MAX_M,
        "foot_left_right_max_m": FOOT_LEFT_RIGHT_MAX_M,
        "push_fall_rate_margin": PUSH_FALL_RATE_MARGIN,
        "still_touchdowns_per_s_max": STILL_TOUCHDOWNS_PER_S_MAX,
        "arm_speed_ratio_range": list(ARM_SPEED_RATIO_RANGE),
        "twist_pass_line": twist_pass_line_record(),
    }


def required_tracking_check_names(profile: str) -> frozenset[str]:
    if profile != FINAL_PROFILE:
        raise ValueError(f"Unknown PICO judgment profile: {profile}")
    return CHECK_NAMES


def required_tracking_profile(completed_updates: int) -> str:
    """The profile of a checkpoint that may end a PICO run (every target active)."""

    if isinstance(completed_updates, bool) or not isinstance(completed_updates, int):
        raise ValueError("completed_updates must be an integer")
    if not PICO_SCHEDULE["foot_tighten"] < completed_updates <= PICO_TOTAL_UPDATES:
        raise ValueError(
            f"A PICO checkpoint is judged after update {PICO_SCHEDULE['foot_tighten']} "
            f"(every target active and tightened) and at most at {PICO_TOTAL_UPDATES}; "
            f"got {completed_updates}"
        )
    return FINAL_PROFILE


def foot_error_limit_m(relative_target_m: float) -> float:
    return max(FOOT_ERROR_SHARE * relative_target_m, FOOT_ERROR_FLOOR_M)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _at_most(value: Any, limit: float) -> bool:
    number = _number(value)
    return number is not None and number <= limit


def _feet_checks(feet: list[dict[str, Any]]) -> dict[str, bool]:
    lift = error = support = bool(feet)
    by_name = {item["name"]: item for item in feet}
    for item in feet:
        for lifted, goal, median in zip(item["lifted"], item["foot_goal"], item["lift_median_m"], strict=True):
            if lifted:
                value = _number(median)
                lift = lift and value is not None and value >= FOOT_LIFT_MIN_SHARE * float(goal[2])
        error = error and _at_most(
            item["relative_error_rms_m"], foot_error_limit_m(float(item["relative_target_m"]))
        )
        if sum(item["lifted"]) == 1:
            support = support and _at_most(item["support_move_median_m"], SUPPORT_FOOT_MOVE_MAX_M)
    left_right = bool(feet)
    for left, right in mirrored_foot_pairs():
        if left not in by_name or right not in by_name:
            left_right = False
            continue
        a = _number(by_name[left]["relative_error_rms_m"])
        b = _number(by_name[right]["relative_error_rms_m"])
        left_right = left_right and a is not None and b is not None and abs(a - b) <= FOOT_LEFT_RIGHT_MAX_M
    return {"foot_lift": lift, "foot_error": error, "support_foot_still": support, "foot_left_right": left_right}


def push_fall_rates(push: list[dict[str, Any]]) -> dict[str, float]:
    rollouts = sum(int(item["rollouts"]) for item in push)
    return {
        actor: sum(int(item["falls"][actor]) for item in push) / max(rollouts, 1)
        for actor in ("pico", "walker")
    }


def _arm_checks(arms: list[dict[str, Any]]) -> dict[str, bool]:
    by_key = {(item["base"], item["arms"]): item for item in arms}
    bases = sorted({item["base"] for item in arms})
    still = walking = bool(arms)
    for base in bases:
        for mode in ARM_MODES:
            item = by_key.get((base, mode))
            if item is None:
                return {"standing_still": False, "walking_with_arms": False}
            if all(value == 0.0 for value in item["twist"]):
                twist = [_number(value) for value in item["mean_twist"]]
                still = (
                    still
                    and _at_most(item["touchdowns_per_s"], STILL_TOUCHDOWNS_PER_S_MAX)
                    and None not in twist
                    and twist_passes((0.0, 0.0, 0.0), twist)
                )
        if all(value == 0.0 for value in by_key[(base, "home")]["twist"]):
            continue
        home = _number(by_key[(base, "home")]["signed_speed"])
        for mode in ("raised", "moving"):
            speed = _number(by_key[(base, mode)]["signed_speed"])
            walking = (
                walking
                and home is not None
                and home > 0.0
                and speed is not None
                and ARM_SPEED_RATIO_RANGE[0] <= speed / home <= ARM_SPEED_RATIO_RANGE[1]
            )
    return {"standing_still": still, "walking_with_arms": walking}


def _acceptance(results: dict[str, Any]) -> tuple[dict[str, bool], str]:
    safety = results["safety"]
    feet, push, arms = results["feet"], results["push"], results["arms"]
    soft = _number(safety["maximum_actual_soft_limit_violation_rad"])
    rates = push_fall_rates(push)
    checks = {
        "finite": safety["finite"] is True,
        "actual_soft_limits": soft is not None and 0.0 <= soft <= ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD,
        "raw_action_recurrence": safety["raw_action_recurrence"] is True,
        "forced_hmd_motion": all(
            _number(value) is not None and value >= HMD_TARGET_PEAK_TO_PEAK_MIN_RAD
            for value in safety["hmd_median_target_peak_to_peak_rad"]
        )
        and all(
            _number(value) is not None and value >= HMD_ACTUAL_PEAK_TO_PEAK_MIN_RAD
            for value in safety["hmd_median_actual_peak_to_peak_rad"]
        ),
        "arm_observation": safety["arm_observation"]["home_nonzero_steps"] == 0
        and safety["arm_observation"]["raised_zero_steps"] == 0,
        "no_falls_without_push": bool(feet) and bool(arms)
        and all(int(item["falls"]) == 0 for item in (*feet, *arms)),
        **_feet_checks(feet),
        "push_falls": bool(push) and rates["pico"] <= rates["walker"] + PUSH_FALL_RATE_MARGIN,
        **_arm_checks(arms),
    }
    if set(checks) != CHECK_NAMES:
        raise AssertionError("PICO judgment check set drifted")
    return checks, "pass" if all(checks.values()) else "fail"


def _summary(minimum: torch.Tensor, maximum: torch.Tensor) -> dict[str, list[float]]:
    return {
        "minimum": minimum.tolist(),
        "maximum": maximum.tolist(),
        "absolute_maximum": torch.maximum(minimum.abs(), maximum.abs()).tolist(),
    }


@dataclass
class _Context:
    env: ManagerBasedRlEnv
    wrapped: RslRlVecEnvWrapper
    policy: Any
    source_policy: Any
    fall_height: float
    arm_term: Any
    arm_home: torch.Tensor
    arm_raised: torch.Tensor
    arm_slice: slice
    action_slice: slice
    command_slice: slice
    foot_slice: slice
    legacy_index: list[int]
    # Accumulated over every PICO rollout.
    action_min: torch.Tensor
    action_max: torch.Tensor
    finite: bool = True
    recurrence: bool = True
    soft_limit: float = 0.0
    soft_limit_joint: str | None = None
    hmd_target_p2p: torch.Tensor | None = None
    hmd_actual_p2p: torch.Tensor | None = None
    home_nonzero_steps: int = 0
    raised_zero_steps: int = 0
    pico_steps: int = 0


def arm_overlay_term(env: ManagerBasedRlEnv) -> Any:
    """The joint_pos action term, which drives the arms to ``arm_goal_rad``."""

    term = env.action_manager.get_term("joint_pos")
    goal = getattr(term, "arm_goal_rad", None)
    if not isinstance(goal, torch.Tensor) or goal.shape != (env.num_envs, len(ARM_JOINT_NAMES)):
        raise TypeError("The PICO judgment needs the arm-overlay action (arm_goal_rad, N x 6)")
    return term


def arm_observation_slice(env: ManagerBasedRlEnv) -> slice:
    return actor_term_slice(env, "arm_target")


def _arm_poses(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
    robot = env.scene["robot"]
    ids = [robot.joint_names.index(name) for name in ARM_JOINT_NAMES]
    home = robot.data.default_joint_pos[:, ids].clone()
    raised = home.clone()
    for column, name in enumerate(ARM_JOINT_NAMES):
        joint = name.split("_", 1)[1]
        if joint in ARMS_RAISED_FORWARD_RAD:
            raised[:, column] = ARMS_RAISED_FORWARD_RAD[joint]
    return home, raised


def _cfg(seed: int, steps: int) -> tuple[Any, float]:
    cfg = make_microban_teleop_v12_env_cfg(play=True)
    training = make_microban_teleop_v12_env_cfg(play=False)
    fall_height = float(cfg.terminations["fell_over"].params["minimum_height"])
    configure_nominal_evaluation(
        cfg,
        steps=steps,
        step_events={
            "hmd_neck_target_motion": copy_forced_moving_hmd_neck_event(training),
            MOVING_ARM_EVENT: copy_moving_arm_event(training),
        },
    )
    cfg.scene.num_envs = ENVS_PER_SCENARIO * SCENARIOS_PER_RUN
    cfg.seed = seed
    return cfg, fall_height


def _rollout(
    ctx: _Context,
    scenarios: tuple[Scenario, ...],
    *,
    seed: int,
    steps: int,
    score_from: int,
    actor: str,
    smoke: list[list[float]] | None = None,
) -> dict[str, torch.Tensor]:
    """Run ``scenarios`` side by side for ``steps``; per-environment measurements."""

    env, robot = ctx.env, ctx.env.scene["robot"]
    n, device = env.num_envs, env.device
    index = scenario_index(len(scenarios), n, device)
    spec = per_env(scenarios, index)
    foot_term = env.command_manager.get_term("foot_target")
    sites = foot_term._foot_asset_cfg.site_ids
    contact = env.scene.sensors["feet_ground_contact"]
    hmd = env.event_manager.get_term_cfg("hmd_neck_target_motion").func
    if not isinstance(hmd, HmdNeckTargetMotion) or tuple(hmd.joint_names) != MICROBAN_HMD_JOINT_NAMES:
        raise TypeError("The PICO judgment needs the HmdNeckTargetMotion step event")

    env.reset(seed=seed)
    ramp = FootTargetRamp(spec["foot_goal"], env.step_dt)
    write_commands(env, spec["twist"], ramp.observed)
    observations = patch_observation(
        ctx.wrapped.get_observations(),
        {ctx.command_slice: spec["twist"], ctx.foot_slice: ramp.observed},
    )
    fixed_arms = spec["arms"] != ARM_MODES.index("moving")
    raised = spec["arms"] == ARM_MODES.index("raised")
    arm_goal = torch.where(raised[:, None], ctx.arm_raised, ctx.arm_home)
    kick = push_velocity(index)
    lifted = torch.linalg.vector_norm(spec["foot_goal"], dim=-1).gt(0.0)
    support = (lifted.sum(dim=-1) == 1)
    support_foot = (~lifted).long().argmax(dim=-1)
    rows = torch.arange(n, device=device)

    fell = torch.zeros(n, dtype=torch.bool, device=device)
    count = torch.zeros(n, device=device)
    lift_sum = torch.zeros(n, 2, device=device)
    error_sq_sum = torch.zeros(n, device=device)
    support_move = torch.zeros(n, device=device)
    twist_sum = torch.zeros(n, 3, device=device)
    touchdowns = torch.zeros(n, device=device)
    down = contact.data.found.reshape(n, -1)[:, :2] > 0
    hmd_min = torch.full((n, 3, 2), math.inf, device=device)
    hmd_max = torch.full((n, 3, 2), -math.inf, device=device)
    reference: dict[str, torch.Tensor] = {}
    first_env = [k * ENVS_PER_SCENARIO for k in range(len(scenarios))]
    smoke_steps = (score_from, steps - 1)

    for step in range(steps):
        actor_obs = observations["actor"]
        if not bool(torch.isfinite(actor_obs).all()):
            ctx.finite = False
        if smoke is not None and step in smoke_steps:
            smoke.extend(actor_obs[first_env].double().tolist())
        with torch.inference_mode():
            legacy = TensorDict({"actor": actor_obs[:, ctx.legacy_index]}, batch_size=[n])
            source = ctx.source_policy(legacy)
            actions = ctx.policy(TensorDict({"actor": actor_obs}, batch_size=[n])) if actor == "pico" else source
        if not bool(torch.isfinite(actions).all()):
            ctx.finite = False
        if actor == "pico":
            for row, value in enumerate((actions, source, actions - source)):
                ctx.action_min[row] = torch.minimum(ctx.action_min[row], value.amin(dim=0))
                ctx.action_max[row] = torch.maximum(ctx.action_max[row], value.amax(dim=0))
        ctx.arm_term.arm_goal_rad[fixed_arms] = arm_goal[fixed_arms]
        write_commands(env, spec["twist"], ramp.observed)
        observations, _rewards, _dones, _extras = ctx.wrapped.step(actions)
        if not torch.equal(observations["actor"][:, ctx.action_slice], actions):
            ctx.recurrence = False
        # The foot target the robot receives next (teleop's ramp, then the floor band).
        ramp.advance(step + 1 >= FOOT_TARGET_STEP)
        observations = patch_observation(observations, {ctx.foot_slice: ramp.observed})
        if bool(NanGuard.detect_nans(env.sim.data)[0].item()) or not bool(
            torch.isfinite(robot.data.joint_pos).all() and torch.isfinite(robot.data.root_link_pose_w).all()
        ):
            ctx.finite = False
        if (step + 1) % PUSH_EVERY_STEPS == 0 and bool(spec["push"].any()):
            ids = spec["push"].nonzero().flatten()
            velocity = robot.data.root_link_vel_w[ids].clone()
            velocity[:, :2] += kick[ids]
            robot.write_root_link_velocity_to_sim(velocity, env_ids=ids)

        fell |= robot.data.root_link_pos_w[:, 2] < ctx.fall_height
        up = (~fell).float()
        hmd_pair = torch.stack((hmd.current_target, robot.data.joint_pos[:, hmd.joint_ids]), dim=-1)
        hmd_min = torch.minimum(hmd_min, hmd_pair)
        hmd_max = torch.maximum(hmd_max, hmd_pair)
        if actor == "pico":
            # While up: a fallen robot is already judged by the falls.
            limits = robot.data.soft_joint_pos_limits
            violation = torch.maximum(limits[..., 0] - robot.data.joint_pos, robot.data.joint_pos - limits[..., 1])
            worst = (violation.clamp(min=0.0) * up[:, None]).amax(dim=0)
            if float(worst.max()) > ctx.soft_limit:
                ctx.soft_limit = float(worst.max())
                ctx.soft_limit_joint = robot.joint_names[int(worst.argmax())]
        feet_w = robot.data.site_pos_w[:, sites, :]
        feet_b = foot_term.current_foot_pos_b()
        now_down = contact.data.found.reshape(n, -1)[:, :2] > 0
        if step + 1 == FOOT_TARGET_STEP:
            reference = {"z": feet_w[..., 2].clone(), "xy": feet_w[..., :2].clone(),
                         "relative": (feet_b[:, 0] - feet_b[:, 1]).clone()}
        if step >= score_from:
            count += up
            twist_sum += _home_levelled_twist(robot.data, HOME_TRUNK_PITCH_RAD) * up[:, None]
            touchdowns += (now_down & ~down).sum(dim=-1).float() * up
            arm_nonzero = observations["actor"][:, ctx.arm_slice].ne(0.0).any(dim=-1)
            if actor == "pico":
                ctx.home_nonzero_steps += int((arm_nonzero & (spec["arms"] == 0) & (index >= 0)).sum())
                ctx.raised_zero_steps += int((~arm_nonzero & raised & (index >= 0)).sum())
            if reference:
                lift_sum += (feet_w[..., 2] - reference["z"]) * up[:, None]
                target = ramp.observed[:, 0] - ramp.observed[:, 1]
                error = feet_b[:, 0] - feet_b[:, 1] - reference["relative"] - target
                error_sq_sum += torch.square(error).sum(dim=-1) * up
                moved = torch.linalg.vector_norm(
                    feet_w[rows, support_foot, :2] - reference["xy"][rows, support_foot], dim=-1
                )
                support_move = torch.maximum(support_move, moved * up * support.float())
        down = now_down

    if actor == "pico":
        ctx.pico_steps += steps
        # The median environment's excursion of each axis (the smallest over the runs).
        p2p = (hmd_max - hmd_min)[index >= 0].median(dim=0).values
        ctx.hmd_target_p2p = p2p[:, 0] if ctx.hmd_target_p2p is None else torch.minimum(ctx.hmd_target_p2p, p2p[:, 0])
        ctx.hmd_actual_p2p = p2p[:, 1] if ctx.hmd_actual_p2p is None else torch.minimum(ctx.hmd_actual_p2p, p2p[:, 1])
    counted = count.clamp(min=1.0)
    return {
        "index": index,
        "fell": fell,
        "count": count,
        "lift": lift_sum / counted[:, None],
        "error_sq_sum": error_sq_sum,
        "support_move": support_move,
        "twist_sum": twist_sum,
        "touchdowns_per_s": touchdowns / (counted * env.step_dt),
        "push_slot": torch.arange(n, device=device) % PUSH_DIRECTIONS,
    }


def _runs(
    ctx: _Context,
    scenarios: tuple[Scenario, ...],
    seeds: tuple[int, ...],
    **kwargs: Any,
) -> list[tuple[Scenario, dict[str, torch.Tensor]]]:
    """Each scenario with its environments' measurements, over every seed."""

    measured: dict[str, list[dict[str, torch.Tensor]]] = {s.name: [] for s in scenarios}
    for start in range(0, len(scenarios), SCENARIOS_PER_RUN):
        chunk = scenarios[start : start + SCENARIOS_PER_RUN]
        for seed in seeds:
            print(f"[INFO] PICO judgment: {[s.name for s in chunk]} seed {seed} ({kwargs['actor']})", flush=True)
            run = _rollout(ctx, chunk, seed=seed, **kwargs)
            for k, scenario in enumerate(chunk):
                mask = run["index"] == k
                measured[scenario.name].append({key: value[mask] for key, value in run.items()})
    return [
        (s, {key: torch.cat([part[key] for part in measured[s.name]]) for key in measured[s.name][0]})
        for s in scenarios
    ]


def _median(values: torch.Tensor) -> float | None:
    return float(values.median()) if values.numel() else None


def _feet(ctx: _Context, seeds: tuple[int, ...], smoke: list[list[float]]) -> list[dict[str, Any]]:
    records = []
    for scenario, m in _runs(
        ctx, foot_scenarios(), seeds, steps=FOOT_STEPS, score_from=FOOT_SCORE_FROM_STEP, actor="pico", smoke=smoke
    ):
        up = ~m["fell"]
        goal = torch.tensor(scenario.foot_goal)
        lifted = list(scenario.lifted)
        samples = m["count"][up].sum()
        records.append({
            "name": scenario.name,
            "foot_goal": [list(value) for value in scenario.foot_goal],
            "lifted": lifted,
            "rollouts": int(m["fell"].numel()),
            "falls": int(m["fell"].sum()),
            # Median over the environments of the mean height above the floor.
            "lift_median_m": [_median(m["lift"][up, foot]) if lifted[foot] else None for foot in (0, 1)],
            # The lifted foot seen from the other foot (HOME-levelled trunk frame).
            "relative_target_m": float(torch.linalg.vector_norm(goal[0] - goal[1])),
            "relative_error_rms_m": float(torch.sqrt(m["error_sq_sum"][up].sum() / samples)) if samples > 0 else None,
            "support_move_median_m": _median(m["support_move"][up]) if sum(lifted) == 1 else None,
        })
    return records


def _push(ctx: _Context, seeds: tuple[int, ...], smoke: list[list[float]]) -> list[dict[str, Any]]:
    by_actor = {
        actor: _runs(ctx, push_scenarios(), seeds, steps=PUSH_STEPS, score_from=SETTLE_STEPS, actor=actor,
                     smoke=smoke if actor == "pico" else None)
        for actor in ("pico", "walker")
    }
    records = []
    for (scenario, pico), (_same, walker) in zip(by_actor["pico"], by_actor["walker"], strict=True):
        records.append({
            "name": scenario.name,
            "twist": list(scenario.twist),
            "rollouts": int(pico["fell"].numel()),
            "falls": {"pico": int(pico["fell"].sum()), "walker": int(walker["fell"].sum())},
            "falls_by_direction": {
                actor: [int(m["fell"][m["push_slot"] == slot].sum()) for slot in range(PUSH_DIRECTIONS)]
                for actor, m in (("pico", pico), ("walker", walker))
            },
        })
    return records


def _arms(ctx: _Context, seeds: tuple[int, ...], smoke: list[list[float]]) -> list[dict[str, Any]]:
    records = []
    for mode in ARM_MODES:
        for scenario, m in _runs(
            ctx, arm_scenarios(mode), seeds, steps=ARM_STEPS, score_from=SETTLE_STEPS, actor="pico", smoke=smoke
        ):
            up = ~m["fell"]
            samples = m["count"][up].sum()
            mean = (m["twist_sum"][up].sum(dim=0) / samples).tolist() if samples > 0 else [None] * 3
            axis = next((i for i, value in enumerate(scenario.twist) if value != 0.0), None)
            signed = (
                None if axis is None or mean[axis] is None
                else mean[axis] * (1.0 if scenario.twist[axis] > 0.0 else -1.0)
            )
            records.append({
                "name": scenario.name,
                "base": scenario.name.split("/")[0],
                "arms": mode,
                "twist": list(scenario.twist),
                "rollouts": int(m["fell"].numel()),
                "falls": int(m["fell"].sum()),
                "mean_twist": mean,  # HOME-levelled (v_x, v_y, w_z) after the 1 s settle
                "signed_speed": signed,
                "touchdowns_per_s": float(m["touchdowns_per_s"][up].mean()) if bool(up.any()) else None,
            })
    return records


def run_evaluation(
    *,
    checkpoint: Path,
    expected_sha256: str | None,
    device: str,
    seed: int,
) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve()
    digest = sha256_file(checkpoint)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"Checkpoint SHA-256 mismatch: {digest}")
    configure_torch_backends(allow_tf32=False, deterministic=True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    policy, iteration, infos = _load_actor(checkpoint, device=device)
    profile = required_tracking_profile(iteration + 1)
    # The walker the adapter was built on, re-verified by its recorded SHA-256.
    source_state = load_bootstrap_source_state(
        validate_bootstrap_provenance(infos.get(TELEOP_V12_BOOTSTRAP_INFO_KEY), verify_files=True)
    )
    source_policy = _legacy_model().to(device)
    source_policy.load_state_dict(source_state, strict=True)
    source_policy.eval()

    seeds = tuple(seed + offset for offset in range(SEED_COUNT))
    cfg, fall_height = _cfg(seed, max(FOOT_STEPS, PUSH_STEPS, ARM_STEPS))
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
    try:
        validate_teleop_v12_environment_contract(wrapped)
        home, raised = _arm_poses(env)
        ctx = _Context(
            env=env,
            wrapped=wrapped,
            policy=policy,
            source_policy=source_policy,
            fall_height=fall_height,
            arm_term=arm_overlay_term(env),
            arm_home=home,
            arm_raised=raised,
            arm_slice=arm_observation_slice(env),
            action_slice=actor_term_slice(env, "actions"),
            command_slice=actor_term_slice(env, "command"),
            foot_slice=actor_term_slice(env, "foot_target"),
            legacy_index=[target for _source, target in LEGACY_TO_TELEOP_OBSERVATION_INDEX],
            action_min=torch.full((3, 18), math.inf, device=env.device),
            action_max=torch.full((3, 18), -math.inf, device=env.device),
        )
        smoke: list[list[float]] = []
        results: dict[str, Any] = {
            "feet": _feet(ctx, seeds, smoke),
            "push": _push(ctx, seeds, smoke),
            "arms": _arms(ctx, seeds, smoke),
        }
    finally:
        wrapped.close()
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    results["safety"] = {
        "finite": ctx.finite,
        "raw_action_recurrence": ctx.recurrence,
        "maximum_actual_soft_limit_violation_rad": ctx.soft_limit,
        "maximum_actual_soft_limit_violation_joint": ctx.soft_limit_joint,
        "hmd_median_target_peak_to_peak_rad": ctx.hmd_target_p2p.tolist(),
        "hmd_median_actual_peak_to_peak_rad": ctx.hmd_actual_p2p.tolist(),
        "arm_observation": {
            "home_nonzero_steps": ctx.home_nonzero_steps,
            "raised_zero_steps": ctx.raised_zero_steps,
        },
    }
    checks, status = _acceptance(results)
    envelope = {
        "joint_names": list(MICROBAN_TELEOP_ACTION_JOINT_NAMES),
        "scenario_count": sum(len(results[part]) for part in ("feet", "push", "arms")),
        "step_count": ctx.pico_steps,
        **{
            name: _summary(ctx.action_min[row].cpu(), ctx.action_max[row].cpu())
            for row, name in enumerate(("v12", "legacy_source", "learned_minus_source"))
        },
    }
    return {
        "schema_version": 2,
        "gate": "microban_teleop_v12_tracking",
        "profile": profile,
        "status": status,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": digest,
            "iteration": iteration,
            "completed_updates": iteration + 1,
        },
        "settings": canonical_settings(device=device, seed=seed),
        "thresholds": thresholds(),
        "checks": checks,
        "push_fall_rates": push_fall_rates(results["push"]),
        "raw_action_envelope": envelope,
        # The robot's startup ONNX self-test corpus: the actor observations of
        # the first environment of every PICO scenario, after the settle and
        # at the end (feet, walking and arm targets among them).
        "runtime_smoke_observations": smoke,
        "results": results,
    }


def canonical_settings(*, device: str, seed: int) -> dict[str, Any]:
    return {
        "device": device,
        "seeds": [seed + offset for offset in range(SEED_COUNT)],
        "envs_per_scenario": ENVS_PER_SCENARIO,
        "steps": {"feet": FOOT_STEPS, "push": PUSH_STEPS, "arms": ARM_STEPS},
        "foot_target_step": FOOT_TARGET_STEP,
        "foot_score_from_step": FOOT_SCORE_FROM_STEP,
        "settle_steps": SETTLE_STEPS,
        "push": {"speed_m_s": PUSH_SPEED_M_S, "directions": PUSH_DIRECTIONS, "every_steps": PUSH_EVERY_STEPS},
        "moving_hmd": "forced_non_neutral",
        "arms": "external_overlay",
        "action_clip": list(MICROBAN_TELEOP_V12_ACTION_CLIP),
        "previous_action": "raw_actor_output",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42, help="the first of the two seeds")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_evaluation(
        checkpoint=args.checkpoint,
        expected_sha256=args.expected_sha256,
        device=args.device,
        seed=args.seed,
    )
    if args.output is not None:
        if args.output.expanduser().exists() and not args.force:
            raise FileExistsError(f"Output exists (pass --force): {args.output}")
        publish_json_atomic(args.output, report)
    print(json.dumps({key: report[key] for key in ("status", "checks", "push_fall_rates")}, sort_keys=True), flush=True)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
