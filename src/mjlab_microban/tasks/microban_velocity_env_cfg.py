# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Microban velocity environment"""

import numpy as np
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.terminations import root_height_below_minimum
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef, TerrainHeightSensorCfg, RingPatternCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from mjlab_microban.robot.microban_constants import (
    HOME_TRUNK_PITCH_RAD,
    MICROBAN_ROBOT_CFG,
    SERVO_TARGET_RANGE_RAD,
)
from mjlab_microban.tasks.curriculum import Setting, Stage, StagedCurriculum
from mjlab_microban.schedules import WALK_MAX_UPDATES, WALK_WIDEN_UPDATE
from mjlab_microban.tasks.mdp import (
    no_stepping_penalty,
    feet_distance_penalty,
    UniformVelocityCommandWithRotation,
    reset_root_state_uniform_world_yaw,
    upright as local_upright,
)
from mjlab_microban.tasks.microban_teleop_mdp import normalized_target_clip_excess_l1_sum
from mjlab_microban.tasks.microban_twist_ratio_mdp import twist_ratio_velocity

SCENE_CFG = SceneCfg(
    terrain=TerrainEntityCfg(
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=0,
    ),
    num_envs=1,
    extent=2.0,
    entities={"robot": MICROBAN_ROBOT_CFG},
)

VIEWER_CONFIG = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY,
    entity_name="robot",
    body_name="trunk",
    distance=3.0,
    elevation=-15.0,
    azimuth=90.0,
)

SIM_CFG = SimulationCfg(
    mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
        ccd_iterations=100,
    ),
    # nconmax=256,
    # njmax=1024,
)

# The command envelope after the update-3000 stage (forward, lateral, yaw).
WALK_COMMAND_RANGES_FINAL = {
    "lin_vel_x": (-0.7, 0.7),
    "lin_vel_y": (-0.3, 0.3),
    "ang_vel_z": (-1.5, 1.5),
}
# The twist reward's axis scale: each axis's largest commanded magnitude
# (0.7 m/s, 0.3 m/s, 1.5 rad/s), shared with PICO.
TWIST_AXIS_SCALE = tuple(
    max(abs(value) for value in WALK_COMMAND_RANGES_FINAL[axis])
    for axis in ("lin_vel_x", "lin_vel_y", "ang_vel_z")
)
# The twist-ratio term is in [0, 1] (1/2 standing still on a moving command,
# 1 at exact tracking): weight 8 spans 4..8, the range of the two exp terms it
# replaces (exp/twist-ratio-validation AB_result.md, recommendation of
# 2026-10-07 01:15: the bounded time-filtered form, direction penalty 1).
WALK_TWIST_RATIO_WEIGHT = 8.0
# Per half-range (pi) of target excess beyond the servo's +-pi goal range,
# summed over the joints (the get-up task's raw_target_clip_excess term).
WALK_RAW_TARGET_CLIP_EXCESS_WEIGHT = -1.0
WALK_TWIST_RATIO_DIRECTION_PENALTY = 1.0

# One stage at update 3000: widen the forward and yaw command ranges and
# penalize standing still on a moving command.  (The command's rotation-env
# extensions below are instance attributes that the train CLI's config
# reconstruction drops, so training samples mjlab's UniformVelocityCommand and
# the stage writes only its fields.)
WALK_STAGES = (
    Stage(
        "penalize stepping + increase velocity",
        WALK_WIDEN_UPDATE,
        (
            Setting("command", "twist", "ranges.lin_vel_x", WALK_COMMAND_RANGES_FINAL["lin_vel_x"]),
            Setting("command", "twist", "ranges.ang_vel_z", WALK_COMMAND_RANGES_FINAL["ang_vel_z"]),
            Setting("reward", "no_stepping", "weight", -1.0),
        ),
    ),
)


def make_microban_velocity_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_velocity_env_cfg()

    cfg.viewer = deepcopy(VIEWER_CONFIG)
    cfg.sim = deepcopy(SIM_CFG)
    cfg.scene = deepcopy(SCENE_CFG)

    foot_site_names = ["left_foot", "right_foot"]

    #---------------------------- Sensors ---------------------------
    feet_ground_sensor_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(foot|foot_2)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    foot_height_scan_cfg = TerrainHeightSensorCfg(
        name="foot_height_scan",
        frame=tuple(ObjRef(type="site", name=s, entity="robot") for s in foot_site_names),
        pattern=RingPatternCfg.single_ring(radius=0.04, num_samples=2),
        ray_alignment="yaw",
        max_distance=1.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),
        debug_vis=False,
    )

    self_collision_sensor_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    
    cfg.scene.sensors = (
        feet_ground_sensor_cfg,
        foot_height_scan_cfg,
        self_collision_sensor_cfg,
    )

    #---------------------------- Terrain ---------------------------
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    #---------------------------- Actions ---------------------------
    # Excludes head AND the new neck_roll/neck_pitch (2026-09): the neck should hold its
    # default pose independently of the walking policy, not be used for balance.
    dofs_filter = r".*(?<!head)(?<!neck_roll)(?<!neck_pitch)$"

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0
    cfg.actions["joint_pos"].actuator_names = (dofs_filter,)
    # No software clip: the target saturates only at the servo's +-pi goal
    # range, as on the robot. The "actions" observation and action_rate_l2
    # see the raw policy output.
    joint_pos_action.clip = {r".*": (-SERVO_TARGET_RANGE_RAD, SERVO_TARGET_RANGE_RAD)}

    #---------------------------- Observations ----------------------
    del cfg.observations["actor"].terms["base_lin_vel"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    cfg.observations["actor"].terms["joint_pos"] = ObservationTermCfg(
        func=mdp.joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(dofs_filter,))},
        noise=Unoise(n_min=-0.001, n_max=0.001),    
        delay_min_lag=0,
        delay_max_lag=0,
    )

    cfg.observations["actor"].terms["joint_vel"] = ObservationTermCfg(
        func=mdp.joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(dofs_filter,))},
        noise=Unoise(n_min=-0.25, n_max=0.25),
        delay_min_lag=0,
        delay_max_lag=1,
    )

    # Observation delays/noises to simulate IMU sensor readings (only for actor, not critic)
    cfg.observations["actor"].terms["projected_gravity"] = deepcopy(cfg.observations["actor"].terms["projected_gravity"])
    cfg.observations["actor"].terms["projected_gravity"].noise = Unoise(n_min=-0.01, n_max=0.01)

    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(cfg.observations["actor"].terms["base_ang_vel"])
    cfg.observations["actor"].terms["base_ang_vel"].noise = Unoise(n_min=-0.03, n_max=0.03)

    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 3
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["actor"].terms["projected_gravity"].delay_min_lag = 0
    cfg.observations["actor"].terms["projected_gravity"].delay_max_lag = 3
    cfg.observations["actor"].terms["projected_gravity"].delay_update_period = 64

    #---------------------------- Rewards ---------------------------
    # Velocity: one twist-ratio term (microban_twist_ratio_mdp) in the
    # HOME-levelled trunk frame, the axes scaled by the final command
    # envelope.  It replaces mjlab's two exp tracking terms (weight 2 each).
    del cfg.rewards["track_linear_velocity"]
    del cfg.rewards["track_angular_velocity"]
    cfg.rewards["twist_ratio_velocity"] = RewardTermCfg(
        func=twist_ratio_velocity,
        weight=WALK_TWIST_RATIO_WEIGHT,
        params={
            "command_name": "twist",
            "trunk_pitch": HOME_TRUNK_PITCH_RAD,
            "axis_scale": TWIST_AXIS_SCALE,
            "direction_penalty": WALK_TWIST_RATIO_DIRECTION_PENALTY,
        },
    )

    std_standing = {
        r".*head.*": 0.3,
        r".*neck_roll.*": 0.3,
        r".*neck_pitch.*": 0.3,
        r".*shoulder_pitch.*": 0.1,
        r".*shoulder_roll.*": 0.1,
        r".*elbow.*": 0.1,
        r".*hip_roll.*": 0.1,
        r".*hip_pitch.*": 0.15,
        r".*hip_yaw.*": 0.1,
        r".*knee.*": 0.15,
        r".*ankle_pitch.*": 0.1,
        r".*ankle_roll.*": 0.1,
    }

    std_walking = {
        r".*head.*": 0.3,
        r".*neck_roll.*": 0.3,
        r".*neck_pitch.*": 0.3,
        r".*shoulder_pitch.*": 0.4,
        r".*shoulder_roll.*": 0.2,
        r".*elbow.*": 0.2,
        r".*hip_roll.*": 0.2,
        r".*hip_pitch.*": 0.4,
        r".*hip_yaw.*": 0.2,
        r".*knee.*": 0.4,
        r".*ankle_pitch.*": 0.3,
        r".*ankle_roll.*": 0.2,
    }

    walking_threshold = 0.01

    cfg.rewards["pose"].params["std_standing"] = std_standing
    cfg.rewards["pose"].params["std_walking"] = std_walking
    cfg.rewards["pose"].params["std_running"] = std_walking
    cfg.rewards["pose"].params["walking_threshold"] = walking_threshold
    cfg.rewards["pose"].weight = 1.0

    cfg.rewards["upright"].func = local_upright
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk",)
    # Peak at HOME's trunk pitch (config/home_pose.yaml; 0 = vertical).
    cfg.rewards["upright"].params["pitch"] = HOME_TRUNK_PITCH_RAD
    cfg.rewards["upright"].params["std"] = np.sqrt(0.1)
    cfg.rewards["upright"].weight = 1.0
    
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk",)
    cfg.rewards["body_ang_vel"].weight = -0.05

    cfg.rewards["angular_momentum"].weight = -0.02

    for reward_name in ["foot_clearance", "foot_slip"]:
        cfg.rewards[reward_name].params["asset_cfg"].site_names = foot_site_names

    cfg.rewards["foot_clearance"].params["command_threshold"] = walking_threshold
    cfg.rewards["foot_clearance"].params["target_height"] = 0.02

    cfg.rewards["foot_swing_height"].params["command_threshold"] = walking_threshold
    cfg.rewards["foot_swing_height"].params["target_height"] = 0.02

    cfg.rewards["air_time"].params["command_threshold"] = walking_threshold
    cfg.rewards["air_time"].params["threshold_min"] = 0.125
    cfg.rewards["air_time"].params["threshold_max"] = 0.300
    cfg.rewards["air_time"].weight = 3.0

    cfg.rewards["no_stepping"] = RewardTermCfg(
        func=no_stepping_penalty,
        weight=0.0,
        params={
            "sensor_name": feet_ground_sensor_cfg.name,
            "command_name": "twist",
            "command_threshold": walking_threshold,
        },
    )

    del cfg.rewards["soft_landing"]

    cfg.rewards["foot_slip"].params["command_threshold"] = walking_threshold
    cfg.rewards["foot_slip"].weight = -1.0

    cfg.rewards["action_rate_l2"].weight = -0.1

    # Raw output beyond the servo's +-pi goal range moves no joint, so no
    # position-based term has a gradient there: the 2026-10-07 release walker
    # (twist ratio B3) drifted its standing arm outputs to about 20 rad from
    # update 5000 and parked elbows and shoulder pitches at their stops
    # (9x300 soft-limit overshoot 0.16-0.23 rad; pose reward lost, dof_pos_limits
    # paid every step).  This linear barrier on the excess (zero inside +-pi,
    # so exploration noise around HOME never pays it) keeps a pull back toward
    # the range, where the action noise reaches positions off the stops again.
    # A barrier at the soft joint limits instead was tried first: it charged
    # the exploration noise on the small-range joints, the action std fell to
    # 0.74 (1.09 without it) and the walker stood still at update 1000.
    cfg.rewards["raw_target_clip_excess"] = RewardTermCfg(
        func=normalized_target_clip_excess_l1_sum,
        weight=WALK_RAW_TARGET_CLIP_EXCESS_WEIGHT,
        params={"action_name": "joint_pos"},
    )

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_sensor_cfg.name},
    )

    # Foot-site separation is about 0.0935 m at the centered HOME; min_dist
    # must stay below the HOME separation.
    cfg.rewards["feet_distance"] = RewardTermCfg(
        func=feet_distance_penalty,
        weight=-1000.0,
        params={
            "min_dist": 0.08,
            "asset_cfg": SceneEntityCfg("robot", site_names=foot_site_names),
        },
    )

    #---------------------------- Commands --------------------------
    command = cfg.commands["twist"]
    command.build = lambda env, _cmd=command: UniformVelocityCommandWithRotation(_cmd, env)
    command.viz.z_offset = 0.5

    command.rel_standing_envs = 0.1
    command.rel_heading_envs = 0.0
    command.rel_rotation_envs = 0.1

    command.ranges.lin_vel_x = (-0.5, 0.5)
    command.ranges.lin_vel_y = (-0.3, 0.3)
    command.ranges.ang_vel_z = (-0.75, 0.75)

    command.rotation_env_ang_vel_range = (-1.5, 1.5)
    command.rotation_min_ang_vel = 0.5

    #---------------------------- Events ----------------------------
    # A leaning HOME turns the random reset yaw about world z: mjlab's term
    # turns it about the HOME trunk's own axis, which would tip the soles and
    # the lean.  With a vertical trunk the two axes coincide (mjlab's term).
    if HOME_TRUNK_PITCH_RAD != 0.0:
        cfg.events["reset_base"].func = reset_root_state_uniform_world_yaw
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.0, 0.01)

    cfg.events["push_robot"].params["velocity_range"] = {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
    }

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
        r".*left_foot_collision.*",
        r".*right_foot_collision.*",
    )

    cfg.events["base_com"].params["ranges"] = {
        0: (-0.005, 0.005),
        1: (-0.005, 0.005),
        2: (-0.005, 0.005),
    }
    cfg.events["base_com"].params["asset_cfg"].body_names = ("trunk",)

    cfg.events["dof_armature_randomization"] = EventTermCfg(
        mode="startup",
        func=dr.joint_armature,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
            "operation": "scale",
            "ranges": (0.9, 1.1),
        },
    )

    cfg.events["dof_friction_randomization"] = EventTermCfg(
        mode="startup",
        func=dr.joint_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
            "operation": "scale",
            "ranges": (0.9, 1.1),
        },
    )

    #---------------------------- Curriculum ------------------------
    cfg.curriculum = {
        "staged_curriculum": CurriculumTermCfg(
            func=StagedCurriculum, params={"stages": WALK_STAGES}
        )
    }

    #---------------------------- Terminations ----------------------
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=root_height_below_minimum,
        params={"minimum_height": 0.10},
    )

    #---------------------------- Play mode -------------------------
    # No curriculum, standing or rotation-only commands, pushes or observation
    # noise.  The command ranges stay the initial ones: the PICO play env (its
    # 9x300 probe and stage gates, tests/fixtures/teleop_play_env_snapshot.json)
    # inherits them, and the walk evaluators write their commands directly.
    if play:
        cfg.curriculum = {}
        cfg.commands["twist"].rel_standing_envs = 0.0
        cfg.commands["twist"].rel_rotation_envs = 0.0
        cfg.events["push_robot"].params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
        }
        cfg.observations["actor"].enable_corruption = False

    return cfg


MicrobanVelocityRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    ),
    wandb_project="mjlab_microban_velocity",
    experiment_name="mjlab_microban_velocity",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=WALK_MAX_UPDATES,
)
