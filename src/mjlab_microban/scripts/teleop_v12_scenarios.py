"""Scenarios and command helpers of the PICO judgment (evaluate_teleop_v12_tracking).

Four parts, each scenario run on ``ENVS_PER_SCENARIO`` environments for each
of two seeds, with the HMD neck moving as in training:

* feet (J1): standing, no push, arms at HOME (and two corners with one arm
  reaching out); one foot lifted (left and right mirrored), or both feet
  moved by the same offset, to a target that teleop's 0.12 m/s ramp reaches;
* pushes (J2): standing and the nine-scenario walking set, pushed every
  second from eight directions, against the walker the adapter was built on;
* arms (J3, J4): the same nine commands with the arms at HOME, raised forward
  70 degrees, or moved as in training (pico_arm_target_motion), and standing
  with one arm reaching out.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
from mjlab.envs import ManagerBasedRlEnv

from mjlab_microban.legacy_velocity_diagnostics import (
    default_scenarios as walking_scenarios,
)
from mjlab_microban.tasks.mdp import UniformVelocityCommandWithRotation
from mjlab_microban.tasks.microban_teleop_foot_command import StationaryFootTargetCommand
from mjlab_microban.tasks.microban_teleop_mdp import (
    MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M,
    HmdNeckTargetMotion,
)

Vec3 = tuple[float, float, float]
ZERO: Vec3 = (0.0, 0.0, 0.0)

ENVS_PER_SCENARIO = 64
SCENARIOS_PER_RUN = 9
SETTLE_STEPS = 50
FOOT_STEPS = 300
FOOT_TARGET_STEP = 50
# Scored from one second after the foot target changed (teleop's 0.12 m/s ramp
# reaches the largest target, 52 mm, in 0.43 s).
FOOT_SCORE_FROM_STEP = 100
PUSH_STEPS = 300
ARM_STEPS = 500
PUSH_EVERY_STEPS = 50
PUSH_SPEED_M_S = 0.40
PUSH_DIRECTIONS = 8
# teleop moves each foot target along a straight line at 0.12 m/s, then the
# wire sends a target whose z is within the floor band as exact zero.
TELEOP_FOOT_TARGET_SPEED_M_S = 0.12
FOOT_FLOOR_BAND_M = MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M

# Live PICO packets are limited to 80% of the training target envelope.
TARGET_SAFETY_MARGIN = 0.8
FOOT_TARGET_LIMIT_M = (0.03, 0.03, 0.05)
SIMULTANEOUS_BOTH_FEET_TARGET_LIMIT_M = (0.01, 0.01, 0.02)

# The six externally driven arm joints, in the order of the arm_target
# observation (left, then right; pitch, roll, elbow).
ARM_JOINT_NAMES = (
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_elbow",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_elbow",
)
# Fixed arm poses, minus HOME in ARM_JOINT_NAMES order (shoulder pitch < 0
# raises an arm forward), all inside the PICO arm box with the hands clear of
# the trunk: both arms 70 deg forward, and one arm reaching forward-out while
# the other is behind.
ARMS_FORWARD_70 = (-math.radians(70.0), 0.0, 0.0, -math.radians(70.0), 0.0, 0.0)
ARMS_REACH_LEFT = (-1.4, 0.8, -0.8, 0.6, 0.0, -0.4)
ARMS_REACH_RIGHT = (0.6, 0.0, -0.4, -1.4, -0.8, -0.8)
FIXED_ARM_POSES = {
    "home": (0.0,) * 6,
    "raised": ARMS_FORWARD_70,
    "reach_left": ARMS_REACH_LEFT,
    "reach_right": ARMS_REACH_RIGHT,
}
ARM_MODES = (*FIXED_ARM_POSES, "moving")
# J4 walks the nine commands with these arm modes (J3 stands with them).
WALKING_ARM_MODES = ("home", "raised", "moving")
MOVING_ARM_EVENT = "pico_arm_target_motion"

# The moving HMD must be seen on every axis of every environment.
HMD_TARGET_PEAK_TO_PEAK_MIN_RAD = 0.10
HMD_ACTUAL_PEAK_TO_PEAK_MIN_RAD = 0.05


@dataclass(frozen=True)
class Scenario:
    """One command held for a whole rollout of ENVS_PER_SCENARIO environments."""

    name: str
    twist: Vec3 = ZERO
    foot_goal: tuple[Vec3, Vec3] = (ZERO, ZERO)  # left, right (HOME-levelled trunk frame, m)
    arms: str = "home"
    push: bool = False

    def __post_init__(self) -> None:
        if self.arms not in ARM_MODES:
            raise ValueError(f"Unknown arm mode {self.arms!r}")

    @property
    def lifted(self) -> tuple[bool, bool]:
        return tuple(any(value != 0.0 for value in goal) for goal in self.foot_goal)  # type: ignore[return-value]


def _mirror(goal: Vec3) -> Vec3:
    return (goal[0], -goal[1], goal[2])


def foot_scenarios() -> tuple[Scenario, ...]:
    """J1: single-foot lifts and the corners of the sent box, mirrored; both feet.

    Both feet always get the same offset (training draws one (dx, dy, dz) for
    the two feet): in the trunk frame the feet rise by dz, so the trunk
    crouches by dz with the feet on the floor and shifts by (-dx, -dy).
    """

    x, y, z = (value * TARGET_SAFETY_MARGIN for value in FOOT_TARGET_LIMIT_M)
    bx, by, bz = (value * TARGET_SAFETY_MARGIN for value in SIMULTANEOUS_BOTH_FEET_TARGET_LIMIT_M)
    single = {
        "up20": (0.0, 0.0, 0.02),
        "up40": (0.0, 0.0, z),
        # +y is outward for the left foot (its mirror is outward for the right).
        "front_out": (x, y, z),
        "front_in": (x, -y, z),
        "back_out": (-x, y, z),
        "back_in": (-x, -y, z),
    }
    scenarios = []
    for name, goal in single.items():
        scenarios.append(Scenario(f"left_{name}", foot_goal=(goal, ZERO)))
        scenarios.append(Scenario(f"right_{name}", foot_goal=(ZERO, _mirror(goal))))
    # A corner with the arm on the lifted side reaching out (the CoM moves too).
    scenarios.append(
        Scenario("left_front_in/arms_reach_left", foot_goal=(single["front_in"], ZERO), arms="reach_left")
    )
    scenarios.append(
        Scenario("right_back_in/arms_reach_right", foot_goal=(ZERO, _mirror(single["back_in"])), arms="reach_right")
    )
    for name, goal in {"front_left": (bx, by, bz), "back_right": (-bx, -by, bz)}.items():
        scenarios.append(Scenario(f"both_{name}", foot_goal=(goal, goal)))
    return tuple(scenarios)


def push_scenarios() -> tuple[Scenario, ...]:
    """J2: standing and the eight walking commands, pushed, arms at HOME."""

    return tuple(
        Scenario(item.name, twist=tuple(item.twist), push=True)  # type: ignore[arg-type]
        for item in walking_scenarios()
    )


def arm_scenarios() -> tuple[Scenario, ...]:
    """J3/J4: standing and the eight walking commands with each walking arm
    mode, then standing with one arm reaching out (left, right)."""

    walking = walking_scenarios()
    standing = next(item for item in walking if not any(item.twist))
    return (
        *(
            Scenario(f"{item.name}/arms_{arms}", twist=tuple(item.twist), arms=arms)  # type: ignore[arg-type]
            for arms in WALKING_ARM_MODES
            for item in walking
        ),
        *(Scenario(f"{standing.name}/arms_{arms}", arms=arms) for arms in ("reach_left", "reach_right")),
    )


def scenario_index(count: int, num_envs: int, device: str) -> torch.Tensor:
    """The scenario of each environment (-1: an idle one), ENVS_PER_SCENARIO each."""

    if count * ENVS_PER_SCENARIO > num_envs:
        raise ValueError(f"{count} scenarios need {count * ENVS_PER_SCENARIO} environments")
    index = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    index[: count * ENVS_PER_SCENARIO] = torch.arange(count, device=device).repeat_interleave(
        ENVS_PER_SCENARIO
    )
    return index


def per_env(scenarios: tuple[Scenario, ...], index: torch.Tensor) -> dict[str, torch.Tensor]:
    """Twist, foot goal, arm mode and push flag of every environment."""

    device = index.device
    padded = (*scenarios, Scenario("idle"))
    rows = index.clone()
    rows[rows < 0] = len(scenarios)
    twist = torch.tensor([s.twist for s in padded], device=device)[rows]
    foot = torch.tensor([s.foot_goal for s in padded], device=device)[rows]
    arms = torch.tensor([ARM_MODES.index(s.arms) for s in padded], device=device)[rows]
    push = torch.tensor([s.push for s in padded], device=device)[rows]
    return {"twist": twist, "foot_goal": foot, "arms": arms, "push": push}


def push_velocity(index: torch.Tensor) -> torch.Tensor:
    """World-frame (vx, vy) kick of each environment: eight directions per scenario."""

    slot = torch.arange(index.numel(), device=index.device) % PUSH_DIRECTIONS
    angle = slot.float() * (2.0 * math.pi / PUSH_DIRECTIONS)
    return PUSH_SPEED_M_S * torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)


class FootTargetRamp:
    """The foot target as the robot receives it from teleop.

    Each foot's target moves to its goal along a straight line at
    TELEOP_FOOT_TARGET_SPEED_M_S; a target within the floor band is sent as
    exact zero (no target).  Shape (N, 2, 3), left then right.
    """

    def __init__(self, goal: torch.Tensor, step_dt: float) -> None:
        self.goal = goal
        self.internal = torch.zeros_like(goal)
        self.step_m = TELEOP_FOOT_TARGET_SPEED_M_S * step_dt

    def advance(self, active: bool) -> None:
        delta = (self.goal if active else torch.zeros_like(self.goal)) - self.internal
        distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        scale = torch.clamp(self.step_m / distance.clamp(min=1.0e-12), max=1.0)
        self.internal = self.internal + delta * scale

    @property
    def observed(self) -> torch.Tensor:
        floor = self.internal[..., 2:3] <= FOOT_FLOOR_BAND_M
        return torch.where(floor, torch.zeros_like(self.internal), self.internal)


def copy_forced_moving_hmd_neck_event(training_cfg: Any) -> Any:
    """Copy only the training HMD step event and force every waypoint non-neutral."""

    source = training_cfg.events.get("hmd_neck_target_motion")
    if source is None:
        raise ValueError("Teleop training config is missing hmd_neck_target_motion")
    if source.mode != "step" or source.func is not HmdNeckTargetMotion:
        raise TypeError("Training HMD event must be the HmdNeckTargetMotion step term")
    copied = deepcopy(source)
    copied.params["neutral_probability"] = 0.0
    return copied


def final_stage_value(manager: str, term: str, path: str) -> Any:
    """The value the PICO stage table (TELEOP_STAGES) last sets for one setting."""

    from mjlab_microban.tasks.microban_teleop_env_cfg import TELEOP_STAGES

    value = None
    for stage in TELEOP_STAGES:
        for setting in stage.settings:
            if (setting.manager, setting.term, setting.path) == (manager, term, path):
                value = setting.value
    if value is None:
        raise ValueError(f"TELEOP_STAGES sets no {manager}/{term}/{path}")
    return value


def copy_moving_arm_event(training_cfg: Any) -> Any:
    """Copy the training arm-target step event at its final HOME probability."""

    source = training_cfg.events.get(MOVING_ARM_EVENT)
    if source is None or source.mode != "step":
        raise ValueError(f"Teleop training config is missing the {MOVING_ARM_EVENT} step event")
    copied = deepcopy(source)
    copied.params["home_probability"] = float(
        final_stage_value("event", MOVING_ARM_EVENT, "params.home_probability")
    )
    return copied


def configure_nominal_evaluation(cfg: Any, *, steps: int, step_events: dict[str, Any]) -> None:
    """Reset events at the nominal pose, the given step events, no curriculum or noise.

    Only ``time_out`` terminates (the judgment detects falls itself, so a
    fallen environment needs no reset and the batch runs to the end).
    """

    cfg.auto_reset = False
    cfg.episode_length_s = (steps + 2) * cfg.decimation * cfg.sim.mujoco.timestep
    events = {name: term for name, term in cfg.events.items() if term.mode == "reset"}
    events.update(step_events)
    cfg.events = events
    reset_base = cfg.events.get("reset_base")
    if reset_base is None:
        raise ValueError("Teleop task is missing its reset_base event")
    reset_base.params["pose_range"] = {
        axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    reset_base.params["velocity_range"] = {}
    cfg.terminations = {
        name: term for name, term in cfg.terminations.items() if name == "time_out"
    }
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}


def write_commands(env: ManagerBasedRlEnv, twist: torch.Tensor, foot: torch.Tensor) -> None:
    """Hold each environment's twist and write its (ramped) foot target.

    The foot target goes through ``hold_targets`` (drawn = slewed = published),
    so the command's own slew in ``_update_command`` leaves it unchanged.
    """

    twist_term = env.command_manager.get_term("twist")
    foot_term = env.command_manager.get_term("foot_target")
    if not isinstance(twist_term, UniformVelocityCommandWithRotation):
        raise TypeError(f"Unexpected twist command type: {type(twist_term).__name__}")
    if not isinstance(foot_term, StationaryFootTargetCommand):
        raise TypeError(f"Unexpected foot command type: {type(foot_term).__name__}")
    twist_term.vel_command_b.copy_(twist)
    twist_term.vel_command_w.copy_(twist)
    for flag in ("is_heading_env", "is_standing_env", "is_world_env", "is_forward_env", "is_rotation_env"):
        getattr(twist_term, flag).fill_(False)
    twist_term.time_left.fill_(float("inf"))

    lifted = torch.linalg.vector_norm(foot, dim=-1).gt(0.0)
    foot_term.hold_targets(torch.arange(env.num_envs, device=env.device), foot)
    foot_term.is_both_feet_env.copy_(lifted.all(dim=-1))
    foot_term.is_single_support_env.copy_(lifted.any(dim=-1) & ~lifted.all(dim=-1))
    foot_term.is_stationary_single_support_env.fill_(False)
    foot_term.lifted_foot_idx.copy_(torch.linalg.vector_norm(foot, dim=-1).argmax(dim=-1))
    foot_term.time_left.fill_(float("inf"))


def actor_term_slice(env: ManagerBasedRlEnv, name: str) -> slice:
    """Columns of one actor observation term."""

    offset = 0
    for term, shape in zip(
        env.observation_manager.active_terms["actor"],
        env.observation_manager.group_obs_term_dim["actor"],
        strict=True,
    ):
        width = math.prod(shape)
        if term == name:
            return slice(offset, offset + width)
        offset += width
    raise ValueError(f"Actor observation lacks {name!r}")


def patch_observation(observations: Any, columns: dict[slice, torch.Tensor]) -> Any:
    """Replace actor observation columns (command values written after the step)."""

    patched = observations.clone()
    actor = patched["actor"].clone()
    for where, value in columns.items():
        actor[:, where] = value.reshape(actor.shape[0], -1)
    patched["actor"] = actor
    return patched
