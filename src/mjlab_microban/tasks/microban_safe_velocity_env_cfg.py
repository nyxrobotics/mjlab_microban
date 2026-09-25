"""Separate, bounded velocity task used to bootstrap safe Microban locomotion.

``Mjlab-Velocity-Microban`` is a historical task whose unbounded actor remains
loadable for reproducibility.  This configuration deliberately does not mutate
that task.  It trains a clean 63-input, 18-output actor with absolute soft-limit
target clipping and a default-preserving measured-state guard.
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microban.robot.microban_constants import MICROBAN_ROBOT_CFG
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_TELEOP_ACTION_JOINT_NAMES,
    guarded_teleop_actor_raw_bounds,
)
from mjlab_microban.tasks.microban_safe_velocity_mdp import (
    MICROBAN_SAFE_VELOCITY_GUARD_LOOKAHEAD_S,
    MICROBAN_SAFE_VELOCITY_GUARD_MARGIN_RATIO,
    MicrobanSafeVelocityBoundedGaussianDistribution,
    SafeVelocityStagedCurriculum,
    commanded_planar_velocity_progress,
    effective_action_after_absolute_clip,
    measured_joint_margin_lookahead_l1_sum,
    planar_velocity_error_l1,
    planar_velocity_tracking_exp,
    preferred_target_margin_l1_sum,
    raw_action_l2,
    yaw_velocity_error_l1,
)
from mjlab_microban.tasks.microban_tracking_env_cfg import (
    MICROBAN_BODY_JOINT_SOFT_LIMITS,
)
from mjlab_microban.tasks.microban_velocity_env_cfg import (
    make_microban_velocity_env_cfg,
)

MICROBAN_SAFE_VELOCITY_TASK_ID = "Mjlab-SafeVelocity-Microban"
MICROBAN_SAFE_VELOCITY_JOINT_NAMES: tuple[str, ...] = MICROBAN_TELEOP_ACTION_JOINT_NAMES
MICROBAN_SAFE_VELOCITY_OBSERVATION_SCHEMA: tuple[tuple[str, int], ...] = (
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("joint_pos", 18),
    ("joint_vel", 18),
    ("actions", 18),
    ("command", 3),
)
MICROBAN_SAFE_VELOCITY_OBSERVATION_WIDTH = sum(
    width for _name, width in MICROBAN_SAFE_VELOCITY_OBSERVATION_SCHEMA
)
MICROBAN_SAFE_VELOCITY_SAGITTAL_LEG_JOINT_NAMES = frozenset(
    {
        "right_hip_pitch",
        "right_knee",
        "right_ankle_pitch",
        "left_hip_pitch",
        "left_knee",
        "left_ankle_pitch",
    }
)
MICROBAN_SAFE_VELOCITY_POLICY_HZ = 50.0
MICROBAN_SAFE_VELOCITY_ROLLOUT_STEPS = 24

MICROBAN_SAFE_VELOCITY_INITIAL_COMMAND = {
    "lin_vel_x": (0.03, 0.07),
    "lin_vel_y": (0.0, 0.0),
    "ang_vel_z": (0.0, 0.0),
}
MICROBAN_SAFE_VELOCITY_CURRICULUM_STAGES: tuple[
    tuple[
        str,
        int,
        dict[str, tuple[float, float]],
        dict[str, dict[str, float]],
    ],
    ...,
] = (
    (
        "sharpen forward velocity objective after stability warmup",
        100 * MICROBAN_SAFE_VELOCITY_ROLLOUT_STEPS,
        dict(MICROBAN_SAFE_VELOCITY_INITIAL_COMMAND),
        {
            "track_linear_velocity": {"weight": 5.0, "std": 0.10},
            "linear_velocity_error_l1": {"weight": -16.0},
        },
    ),
    (
        "slightly widen forward speed",
        300 * MICROBAN_SAFE_VELOCITY_ROLLOUT_STEPS,
        {
            "lin_vel_x": (0.04, 0.10),
            "lin_vel_y": (0.0, 0.0),
            "ang_vel_z": (0.0, 0.0),
        },
        {},
    ),
    (
        "reach the initial PICO forward envelope",
        600 * MICROBAN_SAFE_VELOCITY_ROLLOUT_STEPS,
        {
            "lin_vel_x": (0.05, 0.14),
            "lin_vel_y": (0.0, 0.0),
            "ang_vel_z": (0.0, 0.0),
        },
        {},
    ),
    (
        "introduce small planar and yaw commands",
        1200 * MICROBAN_SAFE_VELOCITY_ROLLOUT_STEPS,
        {
            "lin_vel_x": (0.02, 0.18),
            "lin_vel_y": (-0.03, 0.03),
            "ang_vel_z": (-0.15, 0.15),
        },
        {},
    ),
    (
        "introduce slow reverse locomotion",
        2500 * MICROBAN_SAFE_VELOCITY_ROLLOUT_STEPS,
        {
            "lin_vel_x": (-0.08, 0.22),
            "lin_vel_y": (-0.06, 0.06),
            "ang_vel_z": (-0.30, 0.30),
        },
        {},
    ),
)


def microban_safe_velocity_action_delta_bounds() -> tuple[
    tuple[float, ...], tuple[float, ...]
]:
    """Return guarded deltas in the exact deployment action order."""

    defaults = dict(MICROBAN_ROBOT_CFG.init_state.joint_pos or {})
    ordered_defaults = tuple(
        float(defaults[name]) for name in MICROBAN_SAFE_VELOCITY_JOINT_NAMES
    )
    ordered_lower = tuple(
        MICROBAN_BODY_JOINT_SOFT_LIMITS[name][0]
        for name in MICROBAN_SAFE_VELOCITY_JOINT_NAMES
    )
    ordered_upper = tuple(
        MICROBAN_BODY_JOINT_SOFT_LIMITS[name][1]
        for name in MICROBAN_SAFE_VELOCITY_JOINT_NAMES
    )
    return guarded_teleop_actor_raw_bounds(
        ordered_defaults,
        ordered_lower,
        ordered_upper,
        (1.0,) * len(MICROBAN_SAFE_VELOCITY_JOINT_NAMES),
    )


def microban_safe_velocity_initial_action_std() -> tuple[float, ...]:
    """Conservative scratch exploration, including narrow shoulder headroom."""

    lower, upper = microban_safe_velocity_action_delta_bounds()
    values: list[float] = []
    for name, lower_value, upper_value in zip(
        MICROBAN_SAFE_VELOCITY_JOINT_NAMES, lower, upper, strict=True
    ):
        if "shoulder_roll" in name:
            # The bounded distribution's operational interval is 1024 times
            # wider than this raw side and reserves ten standard deviations.
            values.append(min(-lower_value, upper_value) * 1024.0 / 18.0)
        elif "shoulder" in name or "elbow" in name:
            values.append(0.05)
        elif name in MICROBAN_SAFE_VELOCITY_SAGITTAL_LEG_JOINT_NAMES:
            # V4/V5 converged to a contact-static shuffle with zero measured
            # foot air time.  Explore only the sagittal leg chain more widely;
            # bounded actions and absolute target clips remain unchanged.
            values.append(0.15)
        else:
            values.append(0.08)
    return tuple(values)


def _zero_startup_randomization(cfg: ManagerBasedRlEnvCfg) -> None:
    """Keep the scratch canary nominal without deleting event schemas."""

    cfg.events.pop("push_robot", None)
    cfg.events.pop("encoder_bias", None)
    cfg.events["foot_friction"].params["ranges"] = (1.0, 1.0)
    cfg.events["base_com"].params["ranges"] = {
        0: (0.0, 0.0),
        1: (0.0, 0.0),
        2: (0.0, 0.0),
    }
    for name in ("dof_armature_randomization", "dof_friction_randomization"):
        cfg.events[name].params["ranges"] = (1.0, 1.0)


def make_microban_safe_velocity_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create the isolated bounded velocity environment."""

    cfg = make_microban_velocity_env_cfg(play=play)
    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    action.actuator_names = MICROBAN_SAFE_VELOCITY_JOINT_NAMES
    action.scale = 1.0
    action.use_default_offset = True
    action.clip = MICROBAN_BODY_JOINT_SOFT_LIMITS

    controlled_joints = SceneEntityCfg(
        "robot", joint_names=MICROBAN_SAFE_VELOCITY_JOINT_NAMES
    )
    for group_name in ("actor", "critic"):
        terms = cfg.observations[group_name].terms
        terms["joint_pos"] = ObservationTermCfg(
            func=velocity_mdp.joint_pos_rel,
            params={"asset_cfg": controlled_joints},
            noise=(
                Unoise(n_min=-0.001, n_max=0.001) if group_name == "actor" else None
            ),
            delay_min_lag=0,
            delay_max_lag=0,
        )
        terms["joint_vel"] = ObservationTermCfg(
            func=velocity_mdp.joint_vel_rel,
            params={"asset_cfg": controlled_joints},
            noise=(Unoise(n_min=-0.15, n_max=0.15) if group_name == "actor" else None),
            delay_min_lag=0,
            delay_max_lag=1 if group_name == "actor" else 0,
        )
        terms["actions"] = ObservationTermCfg(
            func=effective_action_after_absolute_clip,
            params={"action_name": "joint_pos"},
        )

    # Actor inference uses only the 63 values available on the physical robot.
    for forbidden in ("base_lin_vel", "height_scan", "root_pos", "root_position"):
        cfg.observations["actor"].terms.pop(forbidden, None)

    command = cfg.commands["twist"]
    command.resampling_time_range = (2.0, 4.0)
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    # UniformVelocityCommand reserves rel_forward_envs for a special
    # |vx| >= 0.3 m/s subset.  This task's deliberately small command ranges
    # must pass through unchanged.
    command.rel_forward_envs = 0.0
    command.rel_rotation_envs = 0.0
    # UniformVelocityCommand requires heading_command=True while its inherited
    # heading range is configured.  rel_heading_envs=0 keeps this path inactive.
    command.heading_command = True
    command.ranges.lin_vel_x = MICROBAN_SAFE_VELOCITY_INITIAL_COMMAND["lin_vel_x"]
    command.ranges.lin_vel_y = MICROBAN_SAFE_VELOCITY_INITIAL_COMMAND["lin_vel_y"]
    command.ranges.ang_vel_z = MICROBAN_SAFE_VELOCITY_INITIAL_COMMAND["ang_vel_z"]
    command.rotation_env_ang_vel_range = (0.0, 0.0)
    command.rotation_min_ang_vel = 0.0

    cfg.rewards["track_linear_velocity"].func = planar_velocity_tracking_exp
    cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(0.04)
    cfg.rewards["track_linear_velocity"].weight = 3.0
    cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.1)
    cfg.rewards["track_angular_velocity"].weight = 1.0
    cfg.rewards["linear_velocity_error_l1"] = RewardTermCfg(
        func=planar_velocity_error_l1,
        weight=-8.0,
        params={"command_name": "twist"},
    )
    cfg.rewards["yaw_velocity_error_l1"] = RewardTermCfg(
        func=yaw_velocity_error_l1,
        weight=-1.0,
        params={"command_name": "twist"},
    )
    cfg.rewards["target_near_limit"] = RewardTermCfg(
        func=preferred_target_margin_l1_sum,
        weight=-2.0,
        params={
            "action_name": "joint_pos",
            "margin_ratio": MICROBAN_SAFE_VELOCITY_GUARD_MARGIN_RATIO,
        },
    )
    cfg.rewards["joint_limit_lookahead"] = RewardTermCfg(
        func=measured_joint_margin_lookahead_l1_sum,
        weight=-10.0,
        params={
            "action_name": "joint_pos",
            "margin_ratio": MICROBAN_SAFE_VELOCITY_GUARD_MARGIN_RATIO,
            "lookahead_s": MICROBAN_SAFE_VELOCITY_GUARD_LOOKAHEAD_S,
        },
    )
    cfg.rewards["raw_action_l2"] = RewardTermCfg(
        func=raw_action_l2,
        weight=-0.01,
        params={"action_name": "joint_pos"},
    )
    cfg.rewards["dof_pos_limits"].weight = -20.0
    cfg.rewards["action_rate_l2"].weight = -0.02
    cfg.rewards["pose"].weight = 0.5
    cfg.rewards["air_time"] = RewardTermCfg(
        func=commanded_planar_velocity_progress,
        weight=2.0,
        params={
            "command_name": "twist",
            "command_threshold": 0.01,
        },
    )
    cfg.rewards["feet_distance"].weight = -100.0
    cfg.rewards["feet_distance"].params["min_dist"] = 0.07

    cfg.events["reset_base"].params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.002),
        "yaw": (-0.03, 0.03),
    }
    cfg.events["reset_base"].params["velocity_range"] = {}
    _zero_startup_randomization(cfg)

    cfg.curriculum = {
        "safe_velocity_stages": CurriculumTermCfg(
            func=SafeVelocityStagedCurriculum,
            params={
                "stages": [
                    {
                        "name": name,
                        "step": step,
                        "ranges": dict(ranges),
                        "reward_overrides": {
                            term_name: dict(values)
                            for term_name, values in reward_overrides.items()
                        },
                    }
                    for name, step, ranges, reward_overrides in (
                        MICROBAN_SAFE_VELOCITY_CURRICULUM_STAGES
                    )
                ]
            },
        )
    }
    cfg.episode_length_s = 8.0
    cfg.is_finite_horizon = False
    cfg.viewer.distance = 1.2
    cfg.viewer.fovy = 55.0
    # Measured fall cases reached 248 contacts / 513 constraints.  Keep enough
    # headroom that a failed canary is observed as a fall rather than silently
    # truncating the contact/constraint model.
    cfg.sim.nconmax = 512
    cfg.sim.njmax = 2048

    if play:
        cfg.curriculum = {}
        cfg.observations["actor"].enable_corruption = False
        cfg.events = {
            name: event for name, event in cfg.events.items() if event.mode == "reset"
        }
        set_command = MICROBAN_SAFE_VELOCITY_INITIAL_COMMAND
        command.ranges.lin_vel_x = (0.08, 0.08)
        command.ranges.lin_vel_y = set_command["lin_vel_y"]
        command.ranges.ang_vel_z = set_command["ang_vel_z"]

    return cfg


MicrobanSafeVelocityRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        # RSL-RL updates observation normalizers during rollout collection.  A
        # changing actor normalizer makes stored old log-probabilities use a
        # different coordinate system at PPO update time, so this actor uses
        # physically scaled raw inputs.  The critic remains normalized.
        obs_normalization=False,
        distribution_cfg={
            "class_name": MicrobanSafeVelocityBoundedGaussianDistribution,
            "init_std": microban_safe_velocity_initial_action_std(),
            "lower_bound": microban_safe_velocity_action_delta_bounds()[0],
            "upper_bound": microban_safe_velocity_action_delta_bounds()[1],
            "std_type": "log",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
        class_name=(
            "mjlab_microban.tasks.microban_safe_velocity_mdp:"
            "MicrobanSafeVelocityBoundedPPO"
        ),
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=3,
        num_mini_batches=4,
        learning_rate=3.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    ),
    wandb_project="mjlab_microban_safe_velocity",
    experiment_name="mjlab_microban_safe_velocity",
    save_interval=25,
    num_steps_per_env=MICROBAN_SAFE_VELOCITY_ROLLOUT_STEPS,
    max_iterations=51,
)
