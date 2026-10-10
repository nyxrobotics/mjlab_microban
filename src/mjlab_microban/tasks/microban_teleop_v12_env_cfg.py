"""Contract-v12 raw-action teleoperation environment and PPO configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

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
from mjlab_microban.schedules import PICO_TOTAL_UPDATES
from mjlab_microban.tasks.microban_teleop_env_cfg import (
    make_microban_teleop_env_cfg,
)

MICROBAN_TELEOP_V12_TRAINING_CONTRACT_VERSION = "13"
# Weight of the left-right mirror loss (MicrobanMirrorLossCfg).  Measured on
# the trainable actor weights over training rollouts, the loss's gradient is
# 0.6 times PPO's surrogate gradient at the pristine walker (loss 0.36) and
# 1.8 times at the asymmetric v13b policy of update 14999 (loss 1.43), both at
# weight 1.  At 0.1 it is a tenth to a fifth of the surrogate's: in 300-update
# trials (schedule x0.02, 512 envs) it kept the loss at 0.30-0.34 against
# 0.44-0.49 without it, at the same reward and episode length through the foot
# stage.  At 1.0 it slowed learning from the arm stage on and the raw outputs
# diverged at update 107.
MICROBAN_TELEOP_MIRROR_LOSS_COEFF = 0.1
# HOME-bound identities (robot/home_contracts.py, from config/home_pose.yaml):
# the forward-lean HOME (trunk 10 deg forward; the root quaternion is part of
# the marker) keeps
# "forward_lean10_hip_minus14p166561199931_ankle_plus4p127976841869_shoulder_zero_v6";
# any other HOME embeds "<label>_<joint hash>", so checkpoints, gates and
# packages of another HOME are refused.
MICROBAN_TELEOP_V12_HOME_POSE_REVISION = home_contracts.V12_HOME_POSE_REVISION
# Base recipe (the published HOME-bound string; the runner of the trained
# recipe records the one below instead).  With a pitched HOME trunk the foot
# targets are offsets in the HOME-levelled trunk frame
# R_trunk * R_y(-HOME_TRUNK_PITCH_RAD) and the neutral HMD neck pose is the
# level headset's neck_pitch = -HOME_TRUNK_PITCH_RAD.  With a vertical trunk
# that is the trunk frame and neck_pitch 0.
MICROBAN_TELEOP_V12_RECIPE_REVISION = home_contracts.V12_RECIPE_REVISION
# The recipe that is trained and packaged (task
# ``Mjlab-Teleop-V13-ArmOverlay-Microban``, one run from scratch): the arms
# are driven from outside as the robot's pico_arms drives them and observed as
# arm targets, and the inherited HOME pose reward covers the leg joints only,
# released on rows tracking a foot target, with the one-run schedule.
MICROBAN_TELEOP_V13_ARM_OVERLAY_RECIPE_REVISION = (
    home_contracts.V13_ARM_OVERLAY_RECIPE_REVISION
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

# 3e-4 (was 1e-4): with the foot stage's 48 % standing single-foot samples,
# the unload reward grew 33 % in 200 updates against 8 % at 1e-4.  Measured
# only in the foot stage (resumed at update 4000 of a 1e-4 run); the warm-up
# and arm stages before it ran at 1e-4 there and run at 3e-4 here.
MICROBAN_TELEOP_V12_FIXED_LEARNING_RATE = 3.0e-4


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
class MicrobanMirrorLossCfg:
    """rsl_rl's ``symmetry_cfg``: the mirror loss only, no mirrored samples.

    microban_teleop_mirror has the mirror; LegacyAdapterPPO keeps the loss
    to the leg outputs.
    """

    use_data_augmentation: bool = False
    use_mirror_loss: bool = True
    mirror_loss_coeff: float = MICROBAN_TELEOP_MIRROR_LOSS_COEFF
    data_augmentation_func: str = (
        "mjlab_microban.tasks.microban_teleop_mirror:mirror_augmentation"
    )


@dataclass
class MicrobanTeleopPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    symmetry_cfg: MicrobanMirrorLossCfg = field(default_factory=MicrobanMirrorLossCfg)


@dataclass
class MicrobanTeleopV12RunnerCfg(RslRlOnPolicyRunnerCfg):
    """The pinned legacy-source bootstrap of a fresh run."""

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
    algorithm=MicrobanTeleopPpoAlgorithmCfg(
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
    save_interval=500,
    num_steps_per_env=MICROBAN_TELEOP_NUM_STEPS_PER_ENV,
    max_iterations=PICO_TOTAL_UPDATES,
)
