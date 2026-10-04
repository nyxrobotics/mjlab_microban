"""Contract-v12 raw-action teleoperation environment and PPO configuration."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Any

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.tasks.velocity import mdp as velocity_mdp

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
# v6 (2026-10-04): HOME leans the trunk 10 deg forward with the COM over the
# sole centre (microban_constants.HOME_TRUNK_PITCH_RAD); the root quaternion
# is part of the marker, so no upright-HOME checkpoint matches it.
MICROBAN_TELEOP_V12_HOME_POSE_REVISION = (
    "forward_lean10_hip_minus14p166561199931_ankle_plus4p127976841869_shoulder_zero_v6"
)
# v15 (2026-10-04): foot/hand targets are offsets in the HOME-levelled trunk
# frame R_trunk * R_y(-HOME_TRUNK_PITCH_RAD) (reachable-FK hand samples rotated
# into it, normalizer re-derived) and the neutral HMD neck pose is the level
# headset's neck_pitch = -HOME_TRUNK_PITCH_RAD.  v13 used the leaning trunk
# frame and neck_pitch 0.
# v17 (2026-10-04): hand targets are capped to the robot receiver's +-64 mm
# box (hand FK v4: joint-sample rejection, normalizer (64.0, 38.8, 45.8) mm,
# F evaluation/corner pose (-20, 25, -50) deg).  v16 is the hand-pose-release
# successor of v15.
MICROBAN_TELEOP_V12_RECIPE_REVISION = (
    "forward_lean_home_velocity_source_staged_mask_reachable_fk_elbow_minus10_"
    "raw_prev_action_servo_range_pi_home_levelled_targets_level_hmd_"
    "receiver_box_hands_v17"
)
# Opt-in successor recipe: identical to v11 except that the inherited HOME
# pose reward drops the shoulder-pitch/shoulder-roll/elbow joints of every hand
# whose target is active (an inactive hand's arm and every other joint keep the
# v11 term).  It changes nothing before hand targets activate at update 7000.
# Only its own task (``Mjlab-Teleop-V12-HandPoseRelease-Microban``) trains or
# records it; canonical gates, stage files and the exporter still accept only
# the v11 recipe (and the corner rescue), so it needs its own fresh chain.
MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION = (
    "forward_lean_home_velocity_source_staged_mask_reachable_fk_elbow_minus10_"
    "raw_prev_action_servo_range_pi_home_levelled_targets_level_hmd_"
    "receiver_box_hands_active_hand_arm_pose_release_v18"
)
# The robot's hmd_head move keeps the camera at the headset's world attitude,
# so on the forward-lean HOME a level headset holds neck_pitch at
# -HOME_TRUNK_PITCH_RAD (and that is also where an active hmd_head sits with
# no head command).  V12 resets and neutral waypoints use this pose; random
# waypoints keep covering the full runtime range.  Absolute joint angles.
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


@dataclass(frozen=True)
class PreviewHandTrackingSettings:
    """One auditable hand-acquisition stage used only by the preview task."""

    reward_weight: float
    reward_std_m: float
    joint_soft_limit_guard_weight: float


MICROBAN_TELEOP_V12_PREVIEW_HAND_ACQUISITION_UPDATE = 7_001
MICROBAN_TELEOP_V12_PREVIEW_HAND_FOCUS_UPDATE = 7_050
MICROBAN_TELEOP_V12_PREVIEW_HAND_STANDARD_UPDATE = 8_500
MICROBAN_TELEOP_V12_PREVIEW_HAND_ACQUISITION = PreviewHandTrackingSettings(
    reward_weight=4.0,
    reward_std_m=0.12,
    joint_soft_limit_guard_weight=-10.0,
)
MICROBAN_TELEOP_V12_PREVIEW_HAND_FOCUS = PreviewHandTrackingSettings(
    reward_weight=4.0,
    reward_std_m=0.08,
    joint_soft_limit_guard_weight=-10.0,
)
MICROBAN_TELEOP_V12_PREVIEW_HAND_STANDARD = PreviewHandTrackingSettings(
    reward_weight=2.0,
    reward_std_m=0.05,
    joint_soft_limit_guard_weight=-10.0,
)


def preview_hand_tracking_settings(
    completed_updates: int,
) -> PreviewHandTrackingSettings:
    """Return the exact reconstructed preview hand stage for a saved clock."""

    if isinstance(completed_updates, bool) or completed_updates < (
        MICROBAN_TELEOP_V12_PREVIEW_HAND_ACQUISITION_UPDATE
    ):
        raise ValueError("Preview hand tracking starts at completed update 7001")
    if completed_updates < MICROBAN_TELEOP_V12_PREVIEW_HAND_FOCUS_UPDATE:
        return MICROBAN_TELEOP_V12_PREVIEW_HAND_ACQUISITION
    if completed_updates < MICROBAN_TELEOP_V12_PREVIEW_HAND_STANDARD_UPDATE:
        return MICROBAN_TELEOP_V12_PREVIEW_HAND_FOCUS
    return MICROBAN_TELEOP_V12_PREVIEW_HAND_STANDARD


def _apply_preview_hand_tracking_stage(
    env: Any, *, settings: PreviewHandTrackingSettings
) -> None:
    """Apply a preview-only hand stage without altering the canonical task."""

    reward = env.reward_manager.get_term_cfg("hand_target_tracking")
    guard = env.reward_manager.get_term_cfg("joint_soft_limit_guard")
    if not isinstance(reward.params, dict):
        raise TypeError("Preview hand reward params must be a dictionary")
    reward.weight = settings.reward_weight
    reward.params["std"] = settings.reward_std_m
    guard.weight = settings.joint_soft_limit_guard_weight


def make_microban_teleop_v12_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Build teleop with the source actor's raw recurrence and servo-range bound."""

    cfg = make_microban_teleop_env_cfg(play=play)
    cfg.actions["joint_pos"].clip = teleop_v12_action_clip_cfg()
    raw_previous_action = ObservationTermCfg(
        func=velocity_mdp.last_action,
        params={"action_name": "joint_pos"},
    )
    cfg.observations["actor"].terms["actions"] = raw_previous_action
    cfg.observations["critic"].terms["actions"] = raw_previous_action

    # These terms call the soft-limit target-clip helper of the bounded-action
    # contracts.  They do not describe the +-pi servo goal range used here.  The
    # measured joint-state soft-limit guard remains enabled as a reward only; it
    # does not filter or stop an action.
    for reward_name in ("target_clip_excess", "target_near_limit", "raw_action_l2"):
        cfg.rewards.pop(reward_name, None)

    hmd_event = cfg.events.get("hmd_neck_target_motion")
    if hmd_event is not None:
        hmd_event.params["neutral_position_rad"] = dict(
            MICROBAN_TELEOP_V12_HMD_NEUTRAL_POSITION_RAD
        )
    return cfg


def make_microban_teleop_v12_preview_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Build the isolated staged preview with progressive hand acquisition."""

    cfg = make_microban_teleop_v12_env_cfg(play=play)
    if not play:
        # sanitized601 is lifted just past the 7000 stage first.  Foot columns,
        # commands, and reward must remain inert until the later 10000 lift.
        cfg.rewards["foot_target_tracking"].weight = 0.0
        curriculum = cfg.curriculum.get("staged_curriculum")
        stages = None if curriculum is None else curriculum.params.get("stages")
        if not isinstance(stages, list):
            raise TypeError("V12 preview requires the resume-safe stage list")
        preview_stages = [
            {
                "name": "preview reachable-hand acquisition",
                "step": (
                    MICROBAN_TELEOP_V12_PREVIEW_HAND_ACQUISITION_UPDATE
                    * MICROBAN_TELEOP_NUM_STEPS_PER_ENV
                ),
                "apply": partial(
                    _apply_preview_hand_tracking_stage,
                    settings=MICROBAN_TELEOP_V12_PREVIEW_HAND_ACQUISITION,
                ),
            },
            {
                "name": "preview reachable-hand focus",
                "step": (
                    MICROBAN_TELEOP_V12_PREVIEW_HAND_FOCUS_UPDATE
                    * MICROBAN_TELEOP_NUM_STEPS_PER_ENV
                ),
                "apply": partial(
                    _apply_preview_hand_tracking_stage,
                    settings=MICROBAN_TELEOP_V12_PREVIEW_HAND_FOCUS,
                ),
            },
        ]
        existing_steps = {stage.get("step") for stage in stages}
        preview_steps = {stage["step"] for stage in preview_stages}
        if existing_steps & preview_steps:
            raise ValueError("V12 preview hand curriculum step collides")
        stages.extend(preview_stages)
        stages.sort(key=lambda stage: stage["step"])
    return cfg


@dataclass
class MicrobanTeleopV12RunnerCfg(RslRlOnPolicyRunnerCfg):
    """Pinned legacy-source bootstrap and strict full-state resume controls."""

    checkpoint_consumer_mode: bool = False
    legacy_velocity_checkpoint: str | None = None
    legacy_velocity_checkpoint_sha256: str | None = None
    legacy_teleop_probe_receipt: str | None = None
    legacy_teleop_probe_receipt_sha256: str | None = None
    save_pristine_checkpoint: bool = False
    simulation_preview_mode: bool = False
    deadline_fallback_resume: bool = False
    deadline_fallback_resume_gate: str | None = None
    deadline_fallback_resume_gate_sha256: str | None = None


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

# A separate registry entry makes preview loading an explicit operator action;
# canonical training and deployment never accept its permanent marker.
MicrobanTeleopV12PreviewRlCfg = deepcopy(MicrobanTeleopV12RlCfg)
MicrobanTeleopV12PreviewRlCfg.simulation_preview_mode = True
MicrobanTeleopV12PreviewRlCfg.experiment_name = "mjlab_microban_teleop_v12_preview"
MicrobanTeleopV12PreviewRlCfg.wandb_project = "mjlab_microban_teleop_v12_preview"
