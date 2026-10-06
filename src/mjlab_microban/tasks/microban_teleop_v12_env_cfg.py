"""Contract-v12 raw-action teleoperation environment and PPO configuration."""

from __future__ import annotations

from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from mjlab_microban.robot import home_contracts
from mjlab_microban.robot.microban_constants import (
    HOME_TRUNK_PITCH_RAD,
    SERVO_TARGET_RANGE_RAD,
)
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_TELEOP_NUM_STEPS_PER_ENV,
)
from mjlab_microban.tasks.microban_teleop_env_cfg import (
    make_microban_teleop_env_cfg,
)

MICROBAN_TELEOP_V12_TASK_ID = "Mjlab-Teleop-V12-Microban"
MICROBAN_TELEOP_V12_TRAINING_CONTRACT_VERSION = "12"
# HOME-bound identities (robot/home_contracts.py, from config/home_pose.yaml):
# the centered HOME keeps
# "centered_home_hip_plus1p198384259489_ankle_minus1p198384259489_shoulder_zero_v5"
# and its v11 / v12 (pose-release) recipes; the forward-lean HOME (trunk 10 deg
# forward; the root quaternion is part of the marker) keeps
# "forward_lean10_hip_minus14p166561199931_ankle_plus4p127976841869_shoulder_zero_v6"
# and its v17 / v18 recipes; any other HOME embeds "<label>_<joint hash>", so
# checkpoints, gates and packages of another HOME are refused.
MICROBAN_TELEOP_V12_HOME_POSE_REVISION = home_contracts.V12_HOME_POSE_REVISION
# Canonical recipe.  With a pitched HOME trunk (forward-lean v15/v17) the
# foot/hand targets are offsets in the HOME-levelled trunk frame
# R_trunk * R_y(-HOME_TRUNK_PITCH_RAD) (reachable-FK hand samples rotated into
# it and capped to the robot receiver's +-64 mm box, hand FK v4: joint-sample
# rejection, F evaluation/corner pose (-20, 25, -50) deg) and the neutral HMD
# neck pose is the level headset's neck_pitch = -HOME_TRUNK_PITCH_RAD.  With a
# vertical trunk all of that is the trunk frame and neck_pitch 0 (v11).
MICROBAN_TELEOP_V12_RECIPE_REVISION = home_contracts.V12_RECIPE_REVISION
# Opt-in successor recipe (v12 centered / v18 forward-lean): identical to the
# canonical recipe except that the inherited HOME pose reward drops the
# shoulder-pitch/shoulder-roll/elbow joints of every hand whose target is
# active (an inactive hand's arm and every other joint keep the canonical
# term).  It changes nothing before hand targets activate at update 7000.  Only
# its own task (``Mjlab-Teleop-V12-HandPoseRelease-Microban``) trains or records
# it.  Stage gates and the exporter accept its release-eligible lineages
# (microban_teleop_v12_hand_pose_release_lineage): a fresh pose-release chain,
# or (centered HOME only) the pinned recipe switch at the gated canonical
# model_7099.
MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION = (
    home_contracts.V12_HAND_POSE_RELEASE_RECIPE_REVISION
)
# The robot's hmd_head move keeps the camera at the headset's world attitude,
# so on a HOME whose trunk leans forward a level headset holds neck_pitch at
# -HOME_TRUNK_PITCH_RAD (and that is also where an active hmd_head sits with
# no head command).  V12 resets and neutral waypoints use this pose; random
# waypoints keep covering the full runtime range.  Absolute joint angles.  A
# vertical-trunk HOME keeps HOME itself (default_joint_pos) as the neutral.
MICROBAN_TELEOP_V12_HMD_NEUTRAL_POSITION_RAD = {
    "head": 0.0,
    "neck_roll": 0.0,
    "neck_pitch": -HOME_TRUNK_PITCH_RAD,
}
# Shared target rule of every Microban policy: target = HOME + raw_action on all
# 18 body joints with no software clip.  The only bound is the servo's one-turn
# goal range, modelled as an absolute target saturation at +-pi (the robot
# saturates its goal writes at the same range).  The previous-action
# observation stays the raw actor output.  The JSON-safe marker is what
# checkpoints, gate reports, provenance and deployment metadata record.
MICROBAN_TELEOP_V12_ACTION_CLIP = [-SERVO_TARGET_RANGE_RAD, SERVO_TARGET_RANGE_RAD]


def teleop_v12_action_clip_cfg() -> dict[str, tuple[float, float]]:
    """Return the JointPositionAction clip dict for the servo goal range."""

    return {r".*": (-SERVO_TARGET_RANGE_RAD, SERVO_TARGET_RANGE_RAD)}

MICROBAN_TELEOP_V12_FIXED_LEARNING_RATE = 1.0e-4
MICROBAN_TELEOP_V12_STAGE_BOUNDARIES = (3_000, 7_000, 10_000, 15_000)


def make_microban_teleop_v12_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Build teleop with the source actor's raw recurrence and servo-range bound."""

    cfg = make_microban_teleop_env_cfg(play=play)
    cfg.actions["joint_pos"].clip = teleop_v12_action_clip_cfg()

    hmd_event = cfg.events.get("hmd_neck_target_motion")
    if hmd_event is not None and HOME_TRUNK_PITCH_RAD != 0.0:
        hmd_event.params["neutral_position_rad"] = dict(
            MICROBAN_TELEOP_V12_HMD_NEUTRAL_POSITION_RAD
        )
    return cfg


@dataclass
class MicrobanTeleopV12RunnerCfg(RslRlOnPolicyRunnerCfg):
    """Pinned legacy-source bootstrap and strict full-state resume controls."""

    legacy_velocity_checkpoint: str | None = None
    legacy_velocity_checkpoint_sha256: str | None = None
    legacy_teleop_probe_receipt: str | None = None
    legacy_teleop_probe_receipt_sha256: str | None = None
    save_pristine_checkpoint: bool = False


MicrobanTeleopV12RlCfg = MicrobanTeleopV12RunnerCfg(
    actor=RslRlModelCfg(
        class_name=(
            "mjlab_microban.tasks.microban_teleop_v12_actor:LegacyAdapterTeleopActor"
        ),
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
        class_name=("mjlab_microban.tasks.microban_teleop_v12_actor:LegacyAdapterPPO"),
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # The source std is immutable and Gaussian entropy is independent of
        # the mean, so a non-zero entropy coefficient cannot train the adapter.
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=MICROBAN_TELEOP_V12_FIXED_LEARNING_RATE,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    ),
    wandb_project="mjlab_microban_teleop_v12",
    experiment_name="mjlab_microban_teleop_v12",
    save_interval=100,
    num_steps_per_env=MICROBAN_TELEOP_NUM_STEPS_PER_ENV,
    max_iterations=MICROBAN_TELEOP_V12_STAGE_BOUNDARIES[-1],
)
