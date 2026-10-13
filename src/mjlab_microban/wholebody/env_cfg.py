# Adapted from Teleopit f926386 train_mimic/tasks/tracking/tracking_env_cfg.py
# and config/env.py (Apache-2.0, https://github.com/BotRunner64/Teleopit),
# merged into one builder.  Changes: the Microban robot, its HOME + action
# servo targets and its reference-driven neck; lengths x0.21 and forces
# x0.034 (Teleopit's start and push velocities included); the noise, delays,
# physics randomization and action-rate weight of Microban's walking task;
# one observation history per network whose last row is the current frame;
# the hand points as the arms' end points; the joint-limit penalty at the
# MJCF range; the end of a window as a time-out.

"""The whole-body tracking task ``Mjlab-Wholebody-Microban``.

Scale.  GMR shrinks human motion by length and keeps its timing, so lengths
and linear velocities are Teleopit's G1 values x 0.21 (HOME trunk height
0.170 m / G1 torso 0.804 m), forces x 0.034 (the mass ratio, 1.136 kg / G1),
and angles, angular velocities and times are unchanged.  Robot-specific
values (sensor noise and delay, physics randomization, action smoothness,
actuators) are those of the walking policy that runs on the robot.

Actor: the last 10 frames, the current one last, of 118 values: reference
joint angles and velocities of the 18 policy joints, the reference trunk
orientation in the robot trunk frame, the gyro, the 21 joint angles (with the
encoder bias) and velocities, the previous raw action, gravity, and the
reference trunk's velocities, tilt and height.  No reference position and no
base linear velocity, as in Teleopit.  Teleopit gives the network the current
frame and the history as two groups with their own noise; here the current
frame is the history's last row, as it is on the robot (one sensor reading
per step, with one noise and one delay).

Joint limits.  The reference stays within the MJCF ranges and often sits on
them (an arm down is shoulder_roll 0), so the limit penalty and the reset
clip start at the MJCF range (soft limit factor 1.0, not the 0.9 of the
other tasks).

Collision.  This task's robot (not robot.xml, so walking, get-up and PICO are
unchanged) names the collision geoms ``<body>_collision_<n>`` (the friction
randomization covers all of them) and adds one box per head, shoulder,
upper-arm and hip-yaw link, the bounding box of its visual meshes, that
touches only the floor: robot geoms are contype 3 / conaffinity 2, the boxes
1 / 0 and the floor 1 / 1, so a box never overlaps the robot's own geoms.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
from functools import lru_cache

import mujoco
import numpy as np
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.utils.spec_config import CollisionCfg
from mjlab.viewer import ViewerConfig

from mjlab_microban.policy_contract import ACTION_JOINT_NAMES, HEAD_JOINTS
from mjlab_microban.robot.microban_constants import (
    MICROBAN_ROBOT_CFG,
    MICROBAN_XML,
    SERVO_TARGET_RANGE_RAD,
    get_spec,
)
from mjlab_microban.tasks.microban_teleop_mdp import (
    MICROBAN_HMD_RUNTIME_LIMITS_RAD,
    MICROBAN_HMD_SLEW_RATES_RAD_S,
)
from mjlab_microban.wholebody import observations as obs
from mjlab_microban.wholebody import rewards as rew
from mjlab_microban.wholebody import terminations as term
from mjlab_microban.wholebody.motion import MotionCommandCfg, ReferenceNeckTarget

TASK_ID = "Mjlab-Wholebody-Microban"
EXPERIMENT_NAME = "microban_wholebody"
TRAIN_MOTION_DIR = "data/wholebody/lafan1/train"
EVAL_MOTION_DIR = "data/wholebody/lafan1/eval"

# Teleopit's G1 values x 0.21 (x 0.034 for the force), rounded.
# The evaluation's error lines are the same stds (benchmark.py).
BODY_POS_STD_M = 0.06  # 0.3 m: trunk and body positions
LIN_VEL_STD_M_S = 0.2  # 1.0 m/s: trunk and body linear velocities
HAND_POS_STD_M = 0.025  # 0.12 m: the hand points
SELF_CONTACT_FORCE_N = 0.03  # 1 N
HEIGHT_TERMINATION_M = 0.05  # 0.25 m: trunk, feet and hands
ORI_STD_RAD = 0.4
JOINT_POS_STD_RAD = 0.5
# Not Teleopit's: a training episode ends when the trunk lags this far across the floor.  After 10 s the policy
# without it lags 0.13-0.14 m walking forward, 0.81 m standing on a backward walk (medians, PICO and mix eval sets).
ANCHOR_XY_TERMINATION_M = 0.4

ANCHOR_BODY_NAME = "trunk"
# Teleopit's 14 G1 bodies on Microban (no waist, no wrists): pelvis and torso
# are the trunk, hip_roll/knee/ankle_roll the hip block, tibia and foot,
# shoulder_roll/elbow the upper arm and forearm.
TRACKED_BODY_NAMES = (
    "trunk",
    "hip_block__configuration_right",
    "tibia__configuration_right",
    "foot",
    "hip_block__configuration_left",
    "tibia__configuration_left",
    "foot_2",
    "humerus__configuration_right",
    "radius",
    "humerus__configuration_left",
    "radius_2",
)
# The wrist points of Teleopit: the right_hand / left_hand sites on the
# forearms (the forearm bodies' origins are at the elbows).
HAND_BODY_NAMES = ("radius", "radius_2")
HAND_OFFSETS = ((0.0, -0.067014, 0.0135), (0.0, 0.067014, -0.0135))
# Teleopit's end points (ankle and wrist links): the feet and the hands.
END_BODY_NAMES = ("foot", "foot_2", *HAND_BODY_NAMES)
END_BODY_OFFSETS = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), *HAND_OFFSETS)
# Teleopit's reset and push velocities, linear ones x 0.21 (rounded).
VELOCITY_RANGE = {
    "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.04, 0.04),
    "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78),
}
# Frames of the actor's (and critic's) observation history, the current one included.
HISTORY_LENGTH = 10
ACTOR_GROUP = "actor_history"
CRITIC_GROUP = "critic_history"
FLOOR_ONLY_BOX_BODIES = (
    "head",
    "roll_pitch_link",
    "camera_cover",
    "shoulder",
    "shoulder_2",
    "humerus__configuration_right",
    "humerus__configuration_left",
    "hip",
    "hip_2",
)
TRAIN_ONLY_EVENTS = (
    "push_robot",
    "base_com",
    "encoder_bias",
    "geom_friction",
    "dof_armature",
    "dof_friction",
)


@lru_cache(maxsize=1)
def floor_only_boxes() -> dict[str, tuple[tuple[float, ...], tuple[float, ...]]]:
    """Centre and half size, in each body's frame, of its visual meshes' bounding box."""
    model = mujoco.MjModel.from_xml_path(str(MICROBAN_XML))
    boxes = {}
    for body in FLOOR_ONLY_BOX_BODIES:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        points = []
        for geom in range(model.ngeom):
            if model.geom_bodyid[geom] != body_id or int(model.geom_type[geom]) != int(mujoco.mjtGeom.mjGEOM_MESH):
                continue
            mesh = model.geom_dataid[geom]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            rotation = np.zeros(9)
            mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
            vertices = model.mesh_vert[start : start + count].astype(np.float64)
            points.append(vertices @ rotation.reshape(3, 3).T + model.geom_pos[geom])
        if not points:
            raise ValueError(f"{body} has no visual mesh")
        points = np.concatenate(points)
        low, high = points.min(axis=0), points.max(axis=0)
        boxes[body] = (tuple((0.5 * (low + high)).tolist()), tuple((0.5 * (high - low)).tolist()))
    return boxes


def wholebody_spec() -> mujoco.MjSpec:
    """robot.xml with named collision geoms and the floor-only boxes."""
    spec = get_spec()
    counts: dict[str, int] = defaultdict(int)
    for geom in spec.geoms:
        if geom.classname.name == "collision" and not geom.name:
            counts[geom.parent.name] += 1
            geom.name = f"{geom.parent.name}_collision_{counts[geom.parent.name]}"
    for body, (centre, half) in floor_only_boxes().items():
        spec.body(body).add_geom(
            name=f"{body}_floor_collision_1",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=half,
            pos=centre,
            group=3,
        )
    return spec


WHOLEBODY_COLLISION = CollisionCfg(
    geom_names_expr=(r".*_collision_\d+$",),
    contype={r".*_floor_collision_\d+$": 1, r".*": 3},
    conaffinity={r".*_floor_collision_\d+$": 0, r".*": 2},
    condim=3,
    # Priority over the floor: each geom's own (randomized) friction applies.
    priority=1,
    friction={r"^(left|right)_foot_collision_[1-6]$": (1.0,)},
)
WHOLEBODY_ROBOT_CFG = replace(
    MICROBAN_ROBOT_CFG,
    spec_fn=wholebody_spec,
    collisions=(WHOLEBODY_COLLISION,),
    articulation=replace(MICROBAN_ROBOT_CFG.articulation, soft_joint_pos_limit_factor=1.0),
)


def training_collision_model() -> mujoco.MjModel:
    """This task's robot with its collision settings over a floor plane at z = 0 (CPU)."""
    spec = wholebody_spec()
    WHOLEBODY_COLLISION.edit_spec(spec)
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(0.0, 0.0, 0.05), contype=1,
                            conaffinity=1)
    return spec.compile()


def _ref(func, **kwargs) -> ObservationTermCfg:
    return ObservationTermCfg(func=func, params={"command_name": "motion"}, **kwargs)


def _imu(func, noise: float, **params) -> ObservationTermCfg:
    # The walking task's gyro/gravity noise and its 0-3 tick IMU latency.
    return ObservationTermCfg(func=func, params=params, noise=Unoise(n_min=-noise, n_max=noise), delay_min_lag=0,
                              delay_max_lag=3, delay_update_period=64)


ALL_JOINTS = SceneEntityCfg("robot", joint_names=(r".*",))


def actor_terms() -> dict[str, ObservationTermCfg]:
    """The 12 actor terms in their exported order (118 values)."""
    return {
        "ref_joint_pos": _ref(obs.ref_joint_pos),
        "ref_joint_vel": _ref(obs.ref_joint_vel),
        "ref_anchor_ori_b": _ref(obs.ref_anchor_ori_b, noise=Unoise(n_min=-0.05, n_max=0.05)),
        "base_ang_vel": _imu(mdp.builtin_sensor, 0.03, sensor_name="robot/imu_ang_vel"),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel, params={"biased": True, "asset_cfg": ALL_JOINTS},
            noise=Unoise(n_min=-0.001, n_max=0.001),
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel, params={"asset_cfg": ALL_JOINTS}, noise=Unoise(n_min=-0.25, n_max=0.25),
            delay_min_lag=0, delay_max_lag=1,
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
        "projected_gravity": _imu(mdp.projected_gravity, 0.01),
        "ref_anchor_lin_vel_b": _ref(obs.ref_anchor_lin_vel_b),
        "ref_anchor_ang_vel_b": _ref(obs.ref_anchor_ang_vel_b),
        "ref_projected_gravity_b": _ref(obs.ref_projected_gravity_b),
        "ref_anchor_height": _ref(obs.ref_anchor_height),
    }


def critic_terms() -> dict[str, ObservationTermCfg]:
    """Teleopit's privileged critic (223 values)."""
    return {
        "ref_joint_pos": _ref(obs.ref_joint_pos),
        "ref_joint_vel": _ref(obs.ref_joint_vel),
        "ref_anchor_pos_b": _ref(obs.ref_anchor_pos_b),
        "ref_anchor_ori_b": _ref(obs.ref_anchor_ori_b),
        "body_pos_b": _ref(obs.robot_tracking_body_pos_b),
        "body_ori_b": _ref(obs.robot_tracking_body_ori_b),
        "base_lin_vel": ObservationTermCfg(func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"}),
        "base_ang_vel": ObservationTermCfg(func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}),
        "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel, params={"asset_cfg": ALL_JOINTS}),
        "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel, params={"asset_cfg": ALL_JOINTS}),
        "actions": ObservationTermCfg(func=mdp.last_action),
        "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
        "ref_anchor_lin_vel_b": _ref(obs.ref_anchor_lin_vel_b),
        "ref_anchor_ang_vel_b": _ref(obs.ref_anchor_ang_vel_b),
        "ref_projected_gravity_b": _ref(obs.ref_projected_gravity_b),
        "ref_anchor_height": _ref(obs.ref_anchor_height),
    }


def _tracking(func, weight: float, std: float, **params) -> RewardTermCfg:
    return RewardTermCfg(func=func, weight=weight, params={"command_name": "motion", "std": std, **params})


def make_wholebody_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    # (N, 10, D) each; the networks take the last row as the current frame.
    observations = {
        ACTOR_GROUP: ObservationGroupCfg(
            terms=actor_terms(), concatenate_terms=True, enable_corruption=not play,
            history_length=HISTORY_LENGTH, flatten_history_dim=False,
        ),
        CRITIC_GROUP: ObservationGroupCfg(
            terms=critic_terms(), concatenate_terms=True, enable_corruption=False,
            history_length=HISTORY_LENGTH, flatten_history_dim=False,
        ),
    }

    actions = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=ACTION_JOINT_NAMES,
            preserve_order=True,
            scale=1.0,
            use_default_offset=True,
            # target = HOME + raw action, saturated only at the servo's goal range.
            clip={r".*": (-SERVO_TARGET_RANGE_RAD, SERVO_TARGET_RANGE_RAD)},
        )
    }

    commands = {
        "motion": MotionCommandCfg(
            entity_name="robot",
            resampling_time_range=(1.0e9, 1.0e9),
            debug_vis=True,
            motion_file=EVAL_MOTION_DIR if play else TRAIN_MOTION_DIR,
            anchor_body_name=ANCHOR_BODY_NAME,
            body_names=TRACKED_BODY_NAMES,
            joint_names=ACTION_JOINT_NAMES,
            # Teleopit's start offsets, lengths x 0.21 (rounded).
            pose_range={} if play else {
                "x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.002, 0.002),
                "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.2, 0.2),
            },
            velocity_range={} if play else dict(VELOCITY_RANGE),
            joint_position_range=(0.0, 0.0) if play else (-0.1, 0.1),
            sampling_mode="start" if play else "rewind",
        )
    }

    collision_geoms = SceneEntityCfg("robot", geom_names=(r".*_collision_\d+$",))
    events = {
        # Not a randomization: the robot moves the neck the same way, kept in play.
        "neck_target": EventTermCfg(
            func=ReferenceNeckTarget,
            mode="step",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=HEAD_JOINTS, preserve_order=True),
                "command_name": "motion",
                "position_ranges_rad": MICROBAN_HMD_RUNTIME_LIMITS_RAD,
                "slew_rates_rad_s": MICROBAN_HMD_SLEW_RATES_RAD_S,
            },
        ),
        # Teleopit's push (every 4-6 s, its reset velocities).
        "push_robot": EventTermCfg(
            func=mdp.push_by_setting_velocity, mode="interval", interval_range_s=(4.0, 6.0),
            params={"velocity_range": dict(VELOCITY_RANGE)},
        ),
        # The walking task's randomization.
        "base_com": EventTermCfg(
            func=dr.body_com_offset, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk",)), "operation": "add",
                    "ranges": {0: (-0.005, 0.005), 1: (-0.005, 0.005), 2: (-0.005, 0.005)}},
        ),
        "encoder_bias": EventTermCfg(
            func=dr.encoder_bias, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.015, 0.015)},
        ),
        # Every collision geom, not only the soles: the motions lie on the floor.
        "geom_friction": EventTermCfg(
            func=dr.geom_friction, mode="startup",
            params={"asset_cfg": collision_geoms, "operation": "abs", "ranges": (0.3, 1.2), "shared_random": True},
        ),
        "dof_armature": EventTermCfg(
            func=dr.joint_armature, mode="startup",
            params={"asset_cfg": ALL_JOINTS, "operation": "scale", "ranges": (0.9, 1.1)},
        ),
        "dof_friction": EventTermCfg(
            func=dr.joint_friction, mode="startup",
            params={"asset_cfg": ALL_JOINTS, "operation": "scale", "ranges": (0.9, 1.1)},
        ),
    }
    if play:
        for name in TRAIN_ONLY_EVENTS:
            events.pop(name)

    policy_joints = SceneEntityCfg("robot", joint_names=ACTION_JOINT_NAMES, preserve_order=True)
    rewards = {
        "motion_global_root_pos": _tracking(rew.motion_global_anchor_position_error_exp, 0.5, BODY_POS_STD_M),
        "motion_global_root_ori": _tracking(rew.motion_global_anchor_orientation_error_exp, 0.5, ORI_STD_RAD),
        "motion_global_root_lin_vel": _tracking(
            rew.motion_global_anchor_linear_velocity_error_exp, 1.0, LIN_VEL_STD_M_S
        ),
        "motion_global_root_ang_vel": _tracking(rew.motion_global_anchor_angular_velocity_error_exp, 1.0, 3.0),
        "motion_body_pos": _tracking(rew.motion_relative_body_position_error_exp, 1.0, BODY_POS_STD_M),
        "motion_body_ori": _tracking(rew.motion_relative_body_orientation_error_exp, 1.0, ORI_STD_RAD),
        "motion_body_lin_vel": _tracking(rew.motion_global_body_linear_velocity_error_exp, 1.0, LIN_VEL_STD_M_S),
        "motion_body_ang_vel": _tracking(rew.motion_global_body_angular_velocity_error_exp, 1.0, 3.14),
        "motion_joint_pos": _tracking(rew.motion_joint_position_error_exp, 1.0, JOINT_POS_STD_RAD),
        "motion_joint_vel": _tracking(rew.motion_joint_velocity_error_exp, 0.5, 3.0),
        "hand_pos": _tracking(rew.motion_relative_body_point_position_error_exp, 1.0, HAND_POS_STD_M,
                              body_names=HAND_BODY_NAMES, body_offsets=HAND_OFFSETS),
        "survival": RewardTermCfg(func=rew.survival, weight=3.0),
        # The walking policy's weight on the robot (rad actions).  Not
        # Teleopit's: its -0.5 is per (rad / G1_ACTION_SCALE)^2, about -2 to
        # -100 per rad^2, far stronger.
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
        "joint_limit": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0, params={"asset_cfg": policy_joints}),
        "self_collisions": RewardTermCfg(
            func=rew.self_collision_cost, weight=-0.1,
            params={"sensor_name": "self_collision", "force_threshold": SELF_CONTACT_FORCE_N},
        ),
        "feet_acc": RewardTermCfg(
            func=mdp.joint_acc_l2, weight=-2.5e-6, params={"asset_cfg": SceneEntityCfg("robot", joint_names=r".*ankle.*")}
        ),
    }

    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "anchor_pos": TerminationTermCfg(
            func=term.bad_anchor_pos_z_only, params={"command_name": "motion", "threshold": HEIGHT_TERMINATION_M}
        ),
        "anchor_ori": TerminationTermCfg(func=term.bad_anchor_ori, params={"command_name": "motion", "threshold": 1.0}),
        "ee_body_pos": TerminationTermCfg(
            func=term.bad_motion_body_point_z_only,
            params={"command_name": "motion", "threshold": HEIGHT_TERMINATION_M, "body_names": END_BODY_NAMES,
                    "body_offsets": END_BODY_OFFSETS},
        ),
        # The window ran out (Teleopit moved the robot to a new window instead).
        "clip_end": TerminationTermCfg(func=term.clip_end, params={"command_name": "motion"}, time_out=True),
    }
    if not play:
        # Training only: over a whole piece the lag adds up without a bound (the actor sees no position), and the
        # benchmark's progress line judges it.
        terminations["anchor_xy"] = TerminationTermCfg(
            func=term.bad_anchor_pos_xy, params={"command_name": "motion", "threshold": ANCHOR_XY_TERMINATION_M}
        )

    # Get-up's self-contact sensor (every trunk-subtree pair), with Teleopit's
    # per-substep force history.
    self_collision = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
        fields=("found", "force"),
        reduce="maxforce",
        num_slots=1,
        history_length=4,
    )

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane", terrain_generator=None, max_init_terrain_level=0),
            num_envs=1,
            extent=2.0,
            entities={"robot": deepcopy(WHOLEBODY_ROBOT_CFG)},
            sensors=(self_collision,),
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot", body_name="trunk",
            distance=1.0, elevation=-10.0, azimuth=120.0,
        ),
        sim=SimulationCfg(
            # PICO's contact capacity; get-up's sensor match capacity for the
            # trunk-subtree self-contact sensor.
            nconmax=512,
            njmax=2048,
            contact_sensor_maxmatch=500,
            mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20, ccd_iterations=100),
        ),
        decimation=4,
        episode_length_s=1.0e9 if play else 10.0,
    )


_CNN_CFG = {"output_channels": (256, 128, 64), "kernel_size": 3, "activation": "elu", "global_pool": "avg"}
_MODEL_CLASS = "mjlab_microban.wholebody.model:TemporalCNNModel"
# Teleopit's encoder; its MLP head (2048, 1024, 512, 256, 128) cut to fit a
# 50 Hz step on the Raspberry Pi Zero 2 W (about 0.73 M weights).
_HIDDEN_DIMS = (512, 512, 256, 128)

WholebodyRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        class_name=_MODEL_CLASS,
        hidden_dims=_HIDDEN_DIMS,
        activation="elu",
        obs_normalization=True,
        cnn_cfg=dict(_CNN_CFG),
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(
        class_name=_MODEL_CLASS,
        hidden_dims=_HIDDEN_DIMS,
        activation="elu",
        obs_normalization=True,
        cnn_cfg=dict(_CNN_CFG),
    ),
    algorithm=RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    ),
    obs_groups={"actor": (ACTOR_GROUP,), "critic": (CRITIC_GROUP,)},
    experiment_name=EXPERIMENT_NAME,
    wandb_project=EXPERIMENT_NAME,
    logger="tensorboard",
    upload_model=False,
    # The small run saves every 500; the full run passes --agent.save-interval 2000.
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30_000,
)
