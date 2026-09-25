"""Contract-v12 raw-action teleoperation environment and PPO configuration."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.tasks.velocity import mdp as velocity_mdp

from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_TELEOP_NUM_STEPS_PER_ENV,
)
from mjlab_microban.tasks.microban_teleop_env_cfg import (
    make_microban_teleop_env_cfg,
)

MICROBAN_TELEOP_V12_TASK_ID = "Mjlab-Teleop-V12-Microban"
MICROBAN_TELEOP_V12_TRAINING_CONTRACT_VERSION = "12"
MICROBAN_TELEOP_V12_RECIPE_REVISION = (
    "legacy_velocity_model14999_staged_mask_raw_actions_v2"
)
MICROBAN_TELEOP_V12_FIXED_LEARNING_RATE = 1.0e-4
MICROBAN_TELEOP_V12_STAGE_BOUNDARIES = (3_000, 7_000, 10_000, 15_000)


def make_microban_teleop_v12_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Build teleop with the proven legacy actor's raw recurrence semantics."""

    cfg = make_microban_teleop_env_cfg(play=play)
    cfg.actions["joint_pos"].clip = None
    raw_previous_action = ObservationTermCfg(
        func=velocity_mdp.last_action,
        params={"action_name": "joint_pos"},
    )
    cfg.observations["actor"].terms["actions"] = raw_previous_action
    cfg.observations["critic"].terms["actions"] = raw_previous_action

    # These terms call the bounded-action target-clip helper.  They are invalid
    # when the legacy actor's raw output/recurrence contract is active.  The
    # measured joint-state soft-limit guard remains enabled as a reward only; it
    # does not filter or stop an action.
    for reward_name in ("target_clip_excess", "target_near_limit", "raw_action_l2"):
        cfg.rewards.pop(reward_name, None)
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
