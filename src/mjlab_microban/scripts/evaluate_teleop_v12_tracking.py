"""The PICO judgment of the checkpoint a PICO run ends with (J1-J4).

Every part runs 64 environments per scenario on two seeds with the HMD neck
moving as in training (scenarios: teleop_v12_scenarios.py).  The arms are
driven from outside as on the robot (pico_arms): the evaluator writes the arm
goal of the arm-overlay action, the policy's arm outputs are not used.

* J1 feet: a policy that does not lift the foot fails ``foot_lift`` and
  ``foot_error``; a single lifted foot is measured above and off the floor
  and from the support foot against their HOME places, as the foot reward
  measures it, so a moving trunk does not count as a tracked foot.  Both
  feet raised in the trunk frame is a crouch: the trunk must come down.
* J2 pushes: falls are compared with the walker the adapter was built on.
* J3 standing still: standing with the HMD and the arms moving or held out.
* J4 walking with the arms raised or moving: no further from the command
  than with the arms at HOME.

The raw action contract has no software target clip (the target saturates
only at the servo's +-pi goal range), so non-finite values, broken raw-action
recurrence and soft-limit violations of the twelve leg joints the policy
drives are rejected too.  The per-joint
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
from mjlab_microban.scripts.evaluate_teleop_v12_checkpoint import _load_actor, recipe_report_settings
from mjlab_microban.scripts.teleop_v12_bootstrap_gate import _legacy_model
from mjlab_microban.scripts.teleop_v12_scenarios import (
    ARM_JOINT_NAMES,
    ARM_MODES,
    ARM_STEPS,
    FIXED_ARM_POSES,
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

# J1, measured as the foot reward measures it at the end of training
# (mdp.foot_target_tracking_error_exp, exp(-|e|^2 / 0.03^2)).  One foot up:
# e is the lifted foot from the support foot in the trunk's heading frame
# levelled by gravity, so a trunk moving over the support foot or tilting does
# not count: a foot 0.3 x 52 mm off scores 0.76, a 40 mm target with the foot
# left down 0.17.  The lifted foot reaches the target height above the floor;
# both feet raised by dz in the trunk frame reach it by lowering the trunk by
# dz (the feet stay on the floor).  Either way 0.7 x dz:
FOOT_LIFT_MIN_SHARE = 0.7  # 40 mm target: 12 mm short alone still scores 0.85
# ... and off the floor (no pushes): a foot held up by its toes reaches the
# height on the floor.  upper_foot_unload does not ask this: it reads only the
# floor's push, so above its 10 mm threshold (every J1 single-foot target is
# 20 mm or more) a foot touching the floor with no weight on it scores in full
# there.  This check catches that.  The foot may touch down for a tenth of the
# scored time.
FOOT_AIR_MIN_SHARE = 0.9
FOOT_ERROR_SHARE = 0.3
FOOT_ERROR_FLOOR_M = 0.008  # the floor for small targets: still scores 0.93
# The support foot is not in the foot reward (it is the lifted foot's
# reference): it stays down (lifted_support_feet) and where it stood (the
# standing command).  It may move (3-D, on the floor) the lifted foot's error
# floor, rounded up.
SUPPORT_FOOT_MOVE_MAX_M = 0.010
# J2.  The adapter keeps the walker's velocity and survival rewards and its
# pushes (+-0.5 m/s): it may fall at most 5 points more often than the walker.
PUSH_FALL_RATE_MARGIN = 0.05
# J3.  The lifted-support-feet penalty counts the feet up on a standing command; the
# walking judgment allows 0.5 touchdowns per second (config/pipeline.yaml).
STILL_TOUCHDOWNS_PER_S_MAX = 0.5
# J4.  Walking slower with the arms raised or moving is accepted; it still
# walks along the command: at least 20 % of it (the walking judgment's
# smallest single-axis share), or the speed with the arms at HOME if lower.
ARM_WALK_MIN_SHARE = 0.2

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
        "foot_air_min_share": FOOT_AIR_MIN_SHARE,
        "foot_error_share": FOOT_ERROR_SHARE,
        "foot_error_floor_m": FOOT_ERROR_FLOOR_M,
        "support_foot_move_max_m": SUPPORT_FOOT_MOVE_MAX_M,
        "push_fall_rate_margin": PUSH_FALL_RATE_MARGIN,
        "still_touchdowns_per_s_max": STILL_TOUCHDOWNS_PER_S_MAX,
        "arm_walk_min_share": ARM_WALK_MIN_SHARE,
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


def _at_least(value: Any, limit: float) -> bool:
    number = _number(value)
    return number is not None and number >= limit


def _feet_checks(feet: list[dict[str, Any]]) -> dict[str, bool]:
    lift = error = support = bool(feet)
    for item in feet:
        if all(item["lifted"]):
            # Both feet: the trunk comes down by dz (the same dz for both feet).
            dz = float(item["foot_goal"][0][2])
            lift = lift and _at_least(item["trunk_drop_median_m"], FOOT_LIFT_MIN_SHARE * dz)
        else:
            for lifted, goal, median, air in zip(
                item["lifted"], item["foot_goal"], item["lift_median_m"], item["air_share_median"], strict=True
            ):
                if lifted:
                    lift = lift and _at_least(median, FOOT_LIFT_MIN_SHARE * float(goal[2]))
                    lift = lift and _at_least(air, FOOT_AIR_MIN_SHARE)
            support = support and _at_most(item["support_move_median_m"], SUPPORT_FOOT_MOVE_MAX_M)
        error = error and _at_most(
            item["relative_error_rms_m"], foot_error_limit_m(float(item["relative_target_m"]))
        )
    return {"foot_lift": lift, "foot_error": error, "support_foot_still": support}


def push_fall_rates(push: list[dict[str, Any]]) -> dict[str, float]:
    rollouts = sum(int(item["rollouts"]) for item in push)
    return {
        actor: sum(int(item["falls"][actor]) for item in push) / max(rollouts, 1)
        for actor in ("pico", "walker")
    }


def arm_walk_min_speed(home_speed: float, command: float) -> float:
    """The smallest speed along the command allowed with the arms moved (J4)."""

    return min(ARM_WALK_MIN_SHARE * abs(command), home_speed)


def _arm_checks(arms: list[dict[str, Any]]) -> dict[str, bool]:
    home = {item["base"]: item for item in arms if item["arms"] == "home"}
    still = walking = bool(arms)
    for item in arms:
        if all(value == 0.0 for value in item["twist"]):
            twist = [_number(value) for value in item["mean_twist"]]
            still = (
                still
                and _at_most(item["touchdowns_per_s"], STILL_TOUCHDOWNS_PER_S_MAX)
                and None not in twist
                and twist_passes((0.0, 0.0, 0.0), twist)
            )
        elif item["arms"] != "home":
            reference = _number(home[item["base"]]["signed_speed"]) if item["base"] in home else None
            speed = _number(item["signed_speed"])
            command = max(abs(float(value)) for value in item["twist"])
            walking = (
                walking
                and reference is not None
                and speed is not None
                and speed >= arm_walk_min_speed(reference, command)
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
        and safety["arm_observation"]["posed_zero_steps"] == 0,
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
    arm_poses: torch.Tensor  # absolute goal of each fixed arm mode (ARM_MODES order)
    leg_ids: list[int]
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
    posed_zero_steps: int = 0
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


def _arm_poses(term: Any) -> torch.Tensor:
    """Absolute arm goals of the fixed modes (the moving row is HOME, unused)."""

    if tuple(term.target_names[int(i)] for i in term.arm_columns) != ARM_JOINT_NAMES:
        raise ValueError("Arm-overlay joint order drifted")
    home = term.arm_home_rad[0]
    poses = torch.stack(
        [home + torch.tensor(FIXED_ARM_POSES.get(mode, (0.0,) * 6), device=home.device) for mode in ARM_MODES]
    )
    if not bool(((poses >= term.arm_lower_rad) & (poses <= term.arm_upper_rad)).all()):
        raise ValueError("A judged arm pose lies outside the PICO arm box")
    return poses


def _leg_joint_ids(term: Any) -> list[int]:
    """Robot joint ids of the policy-driven joints (the soft-limit guard's)."""

    return [int(i) for i in term.target_ids[term.policy_columns]]


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
    fixed_arms = (spec["arms"] != ARM_MODES.index("moving")).nonzero().flatten()
    posed = (spec["arms"] != ARM_MODES.index("moving")) & (spec["arms"] != ARM_MODES.index("home"))
    arm_goal = ctx.arm_poses[spec["arms"]][fixed_arms]
    kick = push_velocity(index)
    lifted = torch.linalg.vector_norm(spec["foot_goal"], dim=-1).gt(0.0)
    support = (lifted.sum(dim=-1) == 1)
    both = lifted.all(dim=-1)
    support_foot = (~lifted).long().argmax(dim=-1)
    rows = torch.arange(n, device=device)

    fell = torch.zeros(n, dtype=torch.bool, device=device)
    count = torch.zeros(n, device=device)
    lift_sum = torch.zeros(n, 2, device=device)
    air_sum = torch.zeros(n, 2, device=device)
    drop_sum = torch.zeros(n, device=device)
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
        ctx.arm_term.set_arm_target(fixed_arms, arm_goal, immediate=False)
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
            # The twelve leg joints the policy drives (the arms follow pico_arms).
            limits = robot.data.soft_joint_pos_limits[:, ctx.leg_ids]
            joint_pos = robot.data.joint_pos[:, ctx.leg_ids]
            violation = torch.maximum(limits[..., 0] - joint_pos, joint_pos - limits[..., 1])
            worst = (violation.clamp(min=0.0) * up[:, None]).amax(dim=0)
            if float(worst.max()) > ctx.soft_limit:
                ctx.soft_limit = float(worst.max())
                ctx.soft_limit_joint = robot.joint_names[ctx.leg_ids[int(worst.argmax())]]
        feet_w = robot.data.site_pos_w[:, sites, :]
        relative = foot_term.left_from_right_level()
        now_down = contact.data.found.reshape(n, -1)[:, :2] > 0
        if step + 1 == FOOT_TARGET_STEP:
            # The lifted foot's reference is the one the reward uses: the feet at reset (HOME).
            default = foot_term._default_foot_pos_b
            reference = {"feet": feet_w.clone(), "trunk_z": robot.data.root_link_pos_w[:, 2].clone(),
                         "relative": (default[:, 0] - default[:, 1]).clone()}
        if step >= score_from:
            count += up
            twist_sum += _home_levelled_twist(robot.data, HOME_TRUNK_PITCH_RAD) * up[:, None]
            touchdowns += (now_down & ~down).sum(dim=-1).float() * up
            arm_nonzero = observations["actor"][:, ctx.arm_slice].ne(0.0).any(dim=-1)
            if actor == "pico":
                ctx.home_nonzero_steps += int((arm_nonzero & (spec["arms"] == 0) & (index >= 0)).sum())
                ctx.posed_zero_steps += int((~arm_nonzero & posed & (index >= 0)).sum())
            if reference:
                lift_sum += (feet_w[..., 2] - reference["feet"][..., 2]) * up[:, None]
                air_sum += (~now_down).float() * up[:, None]
                drop_sum += (reference["trunk_z"] - robot.data.root_link_pos_w[:, 2]) * up
                target = ramp.observed[:, 0] - ramp.observed[:, 1]
                error = relative - reference["relative"] - target
                error_sq_sum += torch.square(error).sum(dim=-1) * up
                moved = torch.linalg.vector_norm(
                    feet_w[rows, support_foot] - reference["feet"][rows, support_foot], dim=-1
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
        "air": air_sum / counted[:, None],
        "trunk_drop": drop_sum / counted,
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
            # Median over the environments of the mean height above the floor
            # (one lifted foot), or of the mean trunk drop (both feet).
            "lift_median_m": [
                _median(m["lift"][up, foot]) if lifted[foot] and not all(lifted) else None for foot in (0, 1)
            ],
            # ... and of the share of the time that foot is off the floor.
            "air_share_median": [
                _median(m["air"][up, foot]) if lifted[foot] and not all(lifted) else None for foot in (0, 1)
            ],
            "trunk_drop_median_m": _median(m["trunk_drop"][up]) if all(lifted) else None,
            # The lifted foot seen from the other foot (heading frame levelled by gravity).
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
    for scenario, m in _runs(
        ctx, arm_scenarios(), seeds, steps=ARM_STEPS, score_from=SETTLE_STEPS, actor="pico", smoke=smoke
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
            "arms": scenario.arms,
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
        arm_term = arm_overlay_term(env)
        ctx = _Context(
            env=env,
            wrapped=wrapped,
            policy=policy,
            source_policy=source_policy,
            fall_height=fall_height,
            arm_term=arm_term,
            arm_poses=_arm_poses(arm_term),
            leg_ids=_leg_joint_ids(arm_term),
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
            "posed_zero_steps": ctx.posed_zero_steps,
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
        "settings": {**canonical_settings(device=device, seed=seed), **recipe_report_settings(infos)},
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
