# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Microban velocity environment"""

import numpy as np
import torch
from copy import deepcopy
from dataclasses import dataclass, field, fields

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_microban.robot.microban_constants import (
    HOME_TRUNK_PITCH_RAD,
    MICROBAN_ROBOT_CFG,
    SERVO_TARGET_RANGE_RAD,
)
from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab.managers.event_manager import EventTermCfg, requires_model_fields
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.envs.mdp import dr

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from mjlab.envs.mdp import dr
from mjlab.envs.mdp.terminations import root_height_below_minimum

from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.scene import SceneCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef, TerrainHeightSensorCfg, RingPatternCfg

from mjlab_microban.schedules import WALK_TOTAL_UPDATES, WALK_WIDEN_UPDATE
from mjlab_microban.tasks.curriculum import Setting, Stage, StagedCurriculum
from mjlab_microban.tasks.mdp import (
    no_stepping_penalty,
    feet_distance_penalty,
    UniformVelocityCommandWithRotation,
    UniformVelocityCommandWithRotationCfg,
    reset_root_state_uniform_world_yaw,
    training_floor_and_delay,
    upright as local_upright,
)
from mjlab_microban.tasks.microban_velocity_tracking import (
    track_angular_velocity_home_frame,
    track_linear_velocity_home_frame,
)

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

# One stage at update 3000: widen the forward and yaw command ranges and
# penalize standing still on a moving command.
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

# Walking (and PICO, which inherits its config) trains on training's floors
# (robot/microban_constants.py) with no friction below the play env's lowest
# sole friction (0.3), and a third of its environments on the play env's
# floor (MuJoCo's 0.02 s, soles 0.3-1.2): on 0.15-1.5 on every environment
# the walker failed the rigid floor's judgment.  Kept here, not with the
# get-up's floors in tasks/mdp.py, so that a change of the walker's floors
# retrains the walker and PICO only.
WALK_FLOOR_FRICTION = (0.3, 1.5)
WALK_ROBOT_FLOOR_FRACTION = 1 / 3
# Half of the drawn floors (a third of all environments) are hard and
# slippery: robot.xml's contact (MuJoCo's 0.02 s) with friction 0.3-0.45, a
# polished floor.  Trained on 7.6% of such floors, the walker fell 6-18 of 90
# pushed on the judgment's friction 0.3 (2-3 on 0.1 s with 0.3), its loaded
# soles sliding with the push; these floors alone left 9-14.
WALK_SLIPPERY_FRICTION = (0.3, 0.45)
WALK_SLIPPERY_FRACTION = 0.5


@requires_model_fields("geom_friction", "geom_solref")
def slippery_floor_envs(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    fraction: float,
    friction: tuple[float, float],
    asset_cfg: SceneEntityCfg,
) -> None:
    """Put ``fraction`` of the environments, chosen at random, on a hard slippery floor: robot.xml's solref
    (MuJoCo's 0.02 s) and one friction from ``friction`` on every geom of ``asset_cfg``."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.to(env.device, torch.long)
    chosen = env_ids[torch.randperm(len(env_ids), device=env.device)[: round(fraction * len(env_ids))]]
    if len(chosen) == 0:
        return
    geoms = env.scene[asset_cfg.name].indexing.geom_ids[asset_cfg.geom_ids].to(torch.long)
    rows, cols = torch.meshgrid(chosen, geoms, indexing="ij")
    model = env.sim.model
    model.geom_solref[rows, cols] = env.sim.get_default_field("geom_solref")[geoms].to(model.geom_solref.dtype)
    dr.geom_friction(env, chosen, ranges=friction, asset_cfg=asset_cfg, operation="abs", shared_random=True)


@requires_model_fields("geom_friction", "geom_solref")
def robot_floor_envs(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    fraction: float,
    asset_cfg: SceneEntityCfg,
    sole_cfg: SceneEntityCfg,
    sole_friction: tuple[float, float],
) -> None:
    """Put ``fraction`` of the environments, chosen at random, back on the play env's floor.

    Their collision geoms (``asset_cfg``) get robot.xml's friction and solref again (MuJoCo's 0.02 s), and their
    soles (``sole_cfg``) one friction from ``sole_friction``, as the play env's foot_friction draws it.  With
    priority over the floor, the training robot's contacts are then the play robot's.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.to(env.device, torch.long)
    chosen = env_ids[torch.randperm(len(env_ids), device=env.device)[: round(fraction * len(env_ids))]]
    if len(chosen) == 0:
        return
    geoms = env.scene[asset_cfg.name].indexing.geom_ids[asset_cfg.geom_ids].to(torch.long)
    rows, cols = torch.meshgrid(chosen, geoms, indexing="ij")
    model = env.sim.model
    model.geom_solref[rows, cols] = env.sim.get_default_field("geom_solref")[geoms].to(model.geom_solref.dtype)
    model.geom_friction[rows, cols] = env.sim.get_default_field("geom_friction")[geoms].to(model.geom_friction.dtype)
    dr.geom_friction(env, chosen, ranges=sole_friction, asset_cfg=sole_cfg, operation="abs", shared_random=True)


def walk_training_floor_and_delay(cfg: ManagerBasedRlEnvCfg) -> None:
    """Training's floors and per-episode goal delay (tasks/mdp.py training_floor_and_delay), with friction from
    WALK_FLOOR_FRICTION, WALK_SLIPPERY_FRACTION of the environments on hard slippery floors and
    WALK_ROBOT_FLOOR_FRACTION on the play env's floor (each chosen from all environments)."""
    foot = cfg.events["foot_friction"]
    training_floor_and_delay(cfg)
    cfg.events["geom_friction"].params["ranges"] = WALK_FLOOR_FRICTION
    # After the two draws (startup events run in this order).
    cfg.events["slippery_floor"] = EventTermCfg(
        func=slippery_floor_envs,
        mode="startup",
        params={"fraction": WALK_SLIPPERY_FRACTION, "friction": WALK_SLIPPERY_FRICTION,
                "asset_cfg": deepcopy(cfg.events["geom_friction"].params["asset_cfg"])},
    )
    cfg.events["robot_floor"] = EventTermCfg(
        func=robot_floor_envs,
        mode="startup",
        params={"fraction": WALK_ROBOT_FLOOR_FRACTION,
                "asset_cfg": deepcopy(cfg.events["geom_friction"].params["asset_cfg"]),
                "sole_cfg": deepcopy(foot.params["asset_cfg"]), "sole_friction": foot.params["ranges"]},
    )


# A fifth of training's moving commands are single-axis and small: forward
# or backward, left or right, or a turn, the other two axes zero.  Drawn from
# the box (+-0.7 m/s, +-0.3 m/s, +-1.5 rad/s) a pure lateral command near the
# judgment's 0.1 m/s (forward below 0.05 m/s, turn below 0.1 rad/s) came about
# once in 600 commands, and the walker answered 0.1 m/s left or right with
# 0.01-0.045 m/s (and drifted forward 0.04-0.05 m/s and turned).  Half of the
# single-axis commands are lateral (the axis that failed; forward, backward
# and turning answered 2-4 times their lines), so 9% of the commands are pure
# lateral.  The speeds bracket the judgment's (0.1 m/s lateral, 0.1-0.2 m/s
# forward and backward, 0.5 rad/s turn) and stay above 0.05 m/s, where
# standing still already earns 97.5% of the tracking reward.  The other 80%
# keep the box's diagonal commands that the pushed falls are judged on.
WALK_SINGLE_AXIS_FRACTION = 0.2
# Shares of the forward/backward, lateral and turning axes (x, y, yaw).
WALK_SINGLE_AXIS_SHARES = (0.25, 0.5, 0.25)
# |speed| ranges of the three axes (m/s, m/s, rad/s); the sign is drawn.
WALK_SINGLE_AXIS_SPEEDS = ((0.05, 0.3), (0.05, 0.2), (0.2, 0.8))


class SingleAxisVelocityCommand(UniformVelocityCommandWithRotation):
    """The rotation command's sampling, then ``cfg.rel_single_axis_envs`` of the moving (non-standing)
    resampled environments get one small single-axis command: an axis drawn with ``cfg.single_axis_shares``
    (x, y, yaw), a speed from that axis's ``cfg.single_axis_speeds`` range, either sign, the other axes zero."""

    cfg: "SingleAxisVelocityCommandCfg"

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        moving = env_ids[~self.is_standing_env[env_ids]]
        chosen = moving[torch.rand(len(moving), device=self.device) < self.cfg.rel_single_axis_envs]
        if len(chosen) == 0:
            return
        shares = torch.tensor(self.cfg.single_axis_shares, dtype=torch.float32, device=self.device)
        axis = torch.multinomial(shares, len(chosen), replacement=True)
        low, high = torch.tensor(self.cfg.single_axis_speeds, dtype=torch.float32, device=self.device)[axis].unbind(-1)
        speed = low + (high - low) * torch.rand(len(chosen), device=self.device)
        sign = torch.where(torch.rand(len(chosen), device=self.device) < 0.5, -1.0, 1.0)
        self.vel_command_b[chosen] = 0.0
        self.vel_command_b[chosen, axis] = sign * speed
        self.vel_command_w[chosen] = self.vel_command_b[chosen]
        for flag in (self.is_forward_env, self.is_heading_env, self.is_world_env, self.is_rotation_env):
            flag[chosen] = False


@dataclass(kw_only=True)
class SingleAxisVelocityCommandCfg(UniformVelocityCommandWithRotationCfg):
    """Configuration for SingleAxisVelocityCommand."""

    rel_single_axis_envs: float = 0.0
    single_axis_shares: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    single_axis_speeds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (0.05, 0.3), (0.05, 0.2), (0.2, 0.8))

    def build(self, env: ManagerBasedRlEnv) -> SingleAxisVelocityCommand:
        return SingleAxisVelocityCommand(self, env)


def walk_single_axis_command(command: object) -> SingleAxisVelocityCommandCfg:
    """Training's twist: ``command``'s fields with no rotation-only environments (the train CLI dropped the
    velocity task's instance-only ``rel_rotation_envs``, so the walker never trained with them) and
    WALK_SINGLE_AXIS_FRACTION of the moving commands single-axis and small."""
    values = {f.name: deepcopy(getattr(command, f.name)) for f in fields(UniformVelocityCommandWithRotationCfg)
              if f.init and hasattr(command, f.name)}
    values["rel_rotation_envs"] = 0.0
    return SingleAxisVelocityCommandCfg(
        **values,
        rel_single_axis_envs=WALK_SINGLE_AXIS_FRACTION,
        single_axis_shares=WALK_SINGLE_AXIS_SHARES,
        single_axis_speeds=WALK_SINGLE_AXIS_SPEEDS,
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
    # Excludes head, neck_roll and neck_pitch: the neck holds its default pose
    # independently of the walking policy and is not used for balance.
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
    # Velocity: mjlab's two exp tracking terms, weight 2 each, std sqrt(0.045)
    # and sqrt(0.5), in the HOME-levelled trunk frame
    # (microban_velocity_tracking.py).  A vertical-trunk HOME keeps mjlab's own
    # terms.
    if HOME_TRUNK_PITCH_RAD != 0.0:
        for name, func in (
            ("track_linear_velocity", track_linear_velocity_home_frame),
            ("track_angular_velocity", track_angular_velocity_home_frame),
        ):
            cfg.rewards[name].func = func
            cfg.rewards[name].params["trunk_pitch"] = HOME_TRUNK_PITCH_RAD
    # The lateral range (0.3 m/s) is small: at std sqrt(0.1) ignoring a 0.1
    # m/s command kept 0.905 of the reward and ignoring the full 0.3 kept
    # 0.41, and the walkers of the refitted servos walked 0.03-0.045 m/s of a
    # 0.1 m/s sideways command (0.06 for a7c28c8a, which walked on the robot).
    # std^2 = 0.045 makes ignoring 0.1 m/s cost what half of 0.3 m/s cost at
    # sqrt(0.1) (reward 0.80); half of 0.3 m/s keeps exp(-0.5), none exp(-2).
    cfg.rewards["track_linear_velocity"].params["std"] = np.sqrt(0.045)
    cfg.rewards["track_linear_velocity"].weight = 2.0

    cfg.rewards["track_angular_velocity"].params["std"] = np.sqrt(0.5)
    cfg.rewards["track_angular_velocity"].weight = 2.0

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

    # A sole's horizontal speed while in contact, squared.  At -1.0 it was
    # 1% of the tracking reward (a 30 cm robot's soles move slowly): the
    # walker of the refitted servos kept its loaded soles moving (p90 0.11
    # m/s on high friction; 0.03 for the walker that walked on the robot,
    # a7c28c8a on its servo model) and, pushed on a hard floor of friction
    # 0.3, slid with the push and fell (6-18 of 90).
    cfg.rewards["foot_slip"].params["command_threshold"] = walking_threshold
    cfg.rewards["foot_slip"].weight = -10.0

    cfg.rewards["action_rate_l2"].weight = -0.1

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_sensor_cfg.name},
    )

    # Foot-site separation is about 0.094 m at HOME; min_dist must stay below
    # the HOME separation.
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

    # Training adds small single-axis commands (walk_single_axis_command);
    # play and the judgments write their commands themselves.
    if not play:
        cfg.commands["twist"] = walk_single_axis_command(command)

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

    # Training's soft and slippery floors and per-episode goal delay, with
    # friction from 0.3, a third of the environments on hard slippery floors
    # and a third on the play env's floor (walk_training_floor_and_delay);
    # play and the judgments keep the robot's.  PICO inherits this config.
    if not play:
        walk_training_floor_and_delay(cfg)

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

        # cfg.commands["twist"].ranges.lin_vel_x = (0.5, 0.5)
        # cfg.commands["twist"].ranges.lin_vel_y = (0.0, 0.0)
        # cfg.commands["twist"].ranges.ang_vel_z = (0.0, 0.0)
        # cfg.commands["twist"].rotation_env_ang_vel_range = (1.0, 1.0)

        # cfg.commands["twist"].ranges.lin_vel_x = (-0.7, 0.7)
        # cfg.commands["twist"].ranges.lin_vel_y = (-0.3, 0.3)
        # cfg.commands["twist"].ranges.ang_vel_z = (-1.5, 1.5)
        # cfg.commands["twist"].rotation_env_ang_vel_range = (-3.0, 3.0)

        # Can be used to edit neutral pose with a zero agent
        # cfg.events["reset_base"].params["pose_range"]["x"] = (0.0, 0.0)
        # cfg.events["reset_base"].params["pose_range"]["y"] = (0.0, 0.0)
        # cfg.events["reset_base"].params["pose_range"]["z"] = (0.3, 0.3)
        # cfg.events["reset_base"].params["pose_range"]["yaw"] = (0.0, 0.0)
        # cfg.events["reset_base"].interval_range_s = (0.0, 0.0)
        # cfg.events["reset_base"].mode = "interval"

        # del cfg.rewards["track_linear_velocity"]
        # del cfg.rewards["track_angular_velocity"]
        # del cfg.rewards["pose"]
        # del cfg.rewards["upright"]
        # cfg.terminations = {}

        cfg.observations["actor"].enable_corruption = False

        # Can be used to print something every step
        def debug(env: ManagerBasedRlEnv, _):
            env.observation_manager.compute_group("actor", update_history=True)

            terms = dict(env.observation_manager.get_active_iterable_terms(env_idx=0))

            base_ang_vel = terms["actor-base_ang_vel"]
            projected_gravity = terms["actor-projected_gravity"]

            print("\n")
            print(f"base_ang_vel: ")
            print(f"x: {base_ang_vel[0]:.3f}")
            print(f"y: {base_ang_vel[1]:.3f}")
            print(f"z: {base_ang_vel[2]:.3f}")
            print("\n")
            print(f"projected_gravity: ")
            print(f"x: {projected_gravity[0]:.3f}")
            print(f"y: {projected_gravity[1]:.3f}")
            print(f"z: {projected_gravity[2]:.3f}")

        # cfg.events["debug"] = EventTermCfg(
        #     func=debug, mode="interval", interval_range_s=(0.0, 0.0)
        # )

    return cfg


# Weight of the left-right mirror loss (microban_velocity_mirror).
WALK_MIRROR_LOSS_COEFF = 1.0


@dataclass
class WalkMirrorLossCfg:
    """rsl_rl's ``symmetry_cfg``: the mirror loss over the 18 outputs, no mirrored samples."""

    use_data_augmentation: bool = False
    use_mirror_loss: bool = True
    mirror_loss_coeff: float = WALK_MIRROR_LOSS_COEFF
    data_augmentation_func: str = "mjlab_microban.tasks.microban_velocity_mirror:mirror_augmentation"


@dataclass
class WalkPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    symmetry_cfg: WalkMirrorLossCfg = field(default_factory=WalkMirrorLossCfg)


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
    algorithm=WalkPpoAlgorithmCfg(
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
    max_iterations=WALK_TOTAL_UPDATES,
)
