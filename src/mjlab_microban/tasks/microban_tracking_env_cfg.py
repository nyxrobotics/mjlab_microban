"""Microban whole-body motion-tracking policy configuration.

This specializes MjLab's BeyondMimic-style tracking task for retargeted PICO body
motion.  The learned policy controls the 18 arm/leg joints.  Camera yaw/roll/pitch
are deliberately excluded because those three joints are driven independently by
the live HMD controller.

Motion input follows :class:`mjlab.tasks.tracking.mdp.MotionLoader`: ``joint_pos``
and ``joint_vel`` contain all 21 robot joints, while each body array contains all 22
robot bodies (MuJoCo ``world`` excluded), in the orders exported below.  Training
selects the 19 body links relevant to locomotion and manipulation for its tracking
loss.  Set ``MICROBAN_TRACKING_MOTION_FILE`` or pass ``motion_file=...`` to use a
different retargeted clip.
"""

from __future__ import annotations

import os
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microban.robot.microban_constants import MICROBAN_ROBOT_CFG
from mjlab_microban.tasks.microban_tracking_mdp import controlled_motion_command


# Stable model/export contracts.  Retargeting must emit these exact orders.
MICROBAN_JOINT_NAMES: tuple[str, ...] = (
    "head",
    "neck_roll",
    "neck_pitch",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_elbow",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_elbow",
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
)

MICROBAN_BODY_NAMES: tuple[str, ...] = (
    "trunk",
    "head",
    "roll_pitch_link",
    "camera_cover",
    "shoulder",
    "humerus__configuration_right",
    "radius",
    "hip",
    "hip_block__configuration_right",
    "femur__configuration_right",
    "tibia__configuration_right",
    "ankle_block__configuration_left",
    "foot",
    "shoulder_2",
    "humerus__configuration_left",
    "radius_2",
    "hip_2",
    "hip_block__configuration_left",
    "femur__configuration_left",
    "tibia__configuration_left",
    "ankle_block__configuration_right",
    "foot_2",
)

# HMD-driven neck joints never enter the body policy action or observation vectors.
MICROBAN_BODY_JOINT_PATTERN = r"^(?!head$|neck_roll$|neck_pitch$).*$"

# Absolute position-target limits for the 18 policy-controlled joints.  MjLab
# shrinks each MJCF interval by the articulation's 0.9 soft-limit factor about
# that interval's midpoint (it does not multiply asymmetric endpoints by 0.9).
# The action term applies these after adding the default-pose offset, so the
# policy cannot ask the simulator (or a matching deployment adapter) for an
# out-of-range position target.
MICROBAN_BODY_JOINT_SOFT_LIMITS: dict[str, tuple[float, float]] = {
    "right_shoulder_pitch": (-2.827433388, 2.827433388),
    "right_shoulder_roll": (-2.984513021, -0.157079633),
    "right_elbow": (-1.963495408, 1.963495408),
    "right_hip_yaw": (-3.926990817, 0.785398163),
    "right_hip_roll": (-0.392698800, 0.392698800),
    "right_hip_pitch": (-1.413716694, 1.413716694),
    "right_knee": (-0.628318531, 2.199114858),
    "right_ankle_pitch": (-1.461713249, 0.501782161),
    "right_ankle_roll": (-0.549778714, 0.549778714),
    "left_shoulder_pitch": (-2.827433388, 2.827433388),
    "left_shoulder_roll": (0.157079633, 2.984513021),
    "left_elbow": (-1.963495408, 1.963495408),
    "left_hip_yaw": (-0.785398163, 3.926990817),
    "left_hip_roll": (-0.392698800, 0.392698800),
    "left_hip_pitch": (-1.413716694, 1.413716694),
    "left_knee": (-0.628318531, 2.199114858),
    "left_ankle_pitch": (-1.461713249, 0.501782161),
    "left_ankle_roll": (-0.549778714, 0.549778714),
}

# Track every link downstream of the 18 controlled joints, plus the trunk anchor.
MICROBAN_TRACKED_BODY_NAMES: tuple[str, ...] = (
    "trunk",
    "shoulder",
    "humerus__configuration_right",
    "radius",
    "hip",
    "hip_block__configuration_right",
    "femur__configuration_right",
    "tibia__configuration_right",
    "ankle_block__configuration_left",
    "foot",
    "shoulder_2",
    "humerus__configuration_left",
    "radius_2",
    "hip_2",
    "hip_block__configuration_left",
    "femur__configuration_left",
    "tibia__configuration_left",
    "ankle_block__configuration_right",
    "foot_2",
)

MICROBAN_END_EFFECTOR_BODY_NAMES: tuple[str, ...] = (
    "radius",
    "foot",
    "radius_2",
    "foot_2",
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MICROBAN_TRACKING_MOTION_FILE = (
    _REPOSITORY_ROOT / "data" / "motions" / "microban_twist2.npz"
)


def _motion_path(motion_file: str | Path | None) -> str:
    if motion_file is not None:
        return str(Path(motion_file).expanduser().resolve())
    configured = os.environ.get("MICROBAN_TRACKING_MOTION_FILE")
    if configured:
        return str(Path(configured).expanduser().resolve())
    return str(DEFAULT_MICROBAN_TRACKING_MOTION_FILE)


def make_microban_tracking_env_cfg(
    play: bool = False,
    motion_file: str | Path | None = None,
) -> ManagerBasedRlEnvCfg:
    """Create the Microban retargeted-motion tracking environment."""

    cfg = make_tracking_env_cfg()
    cfg.scene.entities = {"robot": MICROBAN_ROBOT_CFG}
    cfg.scene.extent = 1.0

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    cfg.scene.sensors = (self_collision_cfg,)

    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    action.actuator_names = (MICROBAN_BODY_JOINT_PATTERN,)
    action.scale = 1.0
    action.clip = MICROBAN_BODY_JOINT_SOFT_LIMITS

    motion = cfg.commands["motion"]
    assert isinstance(motion, MotionCommandCfg)
    motion.motion_file = _motion_path(motion_file)
    motion.anchor_body_name = "trunk"
    motion.body_names = MICROBAN_TRACKED_BODY_NAMES
    motion.joint_position_range = (-0.08, 0.08)
    motion.pose_range = {
        "x": (-0.02, 0.02),
        "y": (-0.02, 0.02),
        "z": (-0.005, 0.005),
        "roll": (-0.08, 0.08),
        "pitch": (-0.08, 0.08),
        "yaw": (-0.15, 0.15),
    }
    motion.velocity_range = {
        "x": (-0.25, 0.25),
        "y": (-0.25, 0.25),
        "z": (-0.10, 0.10),
        "roll": (-0.30, 0.30),
        "pitch": (-0.30, 0.30),
        "yaw": (-0.50, 0.50),
    }

    controlled_joints = SceneEntityCfg(
        "robot", joint_names=(MICROBAN_BODY_JOINT_PATTERN,)
    )
    for group_name in ("actor", "critic"):
        terms = cfg.observations[group_name].terms
        terms["command"] = ObservationTermCfg(
            func=controlled_motion_command,
            params={"command_name": "motion", "asset_cfg": controlled_joints},
        )
        terms["joint_pos"].params["asset_cfg"] = controlled_joints
        terms["joint_vel"].params["asset_cfg"] = controlled_joints

    # Microban currently has no base-position or linear-velocity estimator.
    # Keep these terms in the privileged critic, but never train the actor to
    # depend on measurements the physical robot cannot reproduce.  This mirrors
    # MjLab's official no-state-estimation tracking variant.
    del cfg.observations["actor"].terms["motion_anchor_pos_b"]
    del cfg.observations["actor"].terms["base_lin_vel"]

    # Microban's sensors/servos are noisier relative to its small link dimensions,
    # while the generic task's 25 cm anchor noise is larger than the whole robot.
    actor_terms = cfg.observations["actor"].terms
    actor_terms["motion_anchor_ori_b"].noise = Unoise(n_min=-0.03, n_max=0.03)
    actor_terms["base_ang_vel"].noise = Unoise(n_min=-0.10, n_max=0.10)
    actor_terms["joint_pos"].noise = Unoise(n_min=-0.005, n_max=0.005)
    actor_terms["joint_vel"].noise = Unoise(n_min=-0.25, n_max=0.25)

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
        r"^(left|right)_foot_collision_[1-6]$",
    )
    cfg.events["base_com"].params["asset_cfg"].body_names = ("trunk",)
    cfg.events["base_com"].params["ranges"] = {
        0: (-0.005, 0.005),
        1: (-0.005, 0.005),
        2: (-0.005, 0.005),
    }
    cfg.events["push_robot"].params["velocity_range"] = {
        "x": (-0.25, 0.25),
        "y": (-0.25, 0.25),
        "z": (-0.10, 0.10),
        "roll": (-0.30, 0.30),
        "pitch": (-0.30, 0.30),
        "yaw": (-0.50, 0.50),
    }

    # Position tolerances are scaled for a roughly 30 cm robot, rather than the
    # generic adult-size humanoid defaults.
    cfg.rewards["motion_global_root_pos"].params["std"] = 0.05
    cfg.rewards["motion_body_pos"].params["std"] = 0.04
    cfg.rewards["motion_body_lin_vel"].params["std"] = 0.5
    cfg.rewards["motion_body_ang_vel"].params["std"] = 2.0
    cfg.rewards["self_collisions"].params["force_threshold"] = 1.0

    cfg.terminations["anchor_pos"].params["threshold"] = 0.08
    cfg.terminations["ee_body_pos"].params.update(
        {
            "threshold": 0.08,
            "body_names": MICROBAN_END_EFFECTOR_BODY_NAMES,
        }
    )

    cfg.viewer.body_name = "trunk"
    cfg.viewer.distance = 1.2
    cfg.viewer.fovy = 55.0
    cfg.sim.nconmax = 128
    cfg.sim.njmax = 512

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        motion.pose_range = {}
        motion.velocity_range = {}
        motion.joint_position_range = (0.0, 0.0)
        motion.sampling_mode = "start"

    return cfg


MicrobanTrackingRlCfg = RslRlOnPolicyRunnerCfg(
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
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    ),
    wandb_project="mjlab_microban_tracking",
    experiment_name="mjlab_microban_tracking",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=30_000,
)
