"""Independent, from-scratch full-body teleop training at the shared HOME."""

from __future__ import annotations

from math import isclose, radians

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from mjlab_microban.robot.microban_constants import HOME_FRAME
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_TELEOP_NUM_STEPS_PER_ENV,
)
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    make_microban_teleop_v12_env_cfg,
)

MICROBAN_TELEOP_UPRIGHT_FULLBODY_TASK_ID = (
    "Mjlab-Teleop-Upright-Fullbody-Microban"
)
MICROBAN_TELEOP_UPRIGHT_FULLBODY_EXPERIMENT = (
    "mjlab_microban_teleop_upright_fullbody"
)
MICROBAN_TELEOP_UPRIGHT_FULLBODY_RECIPE_REVISION = (
    "physical_neutral_full_actor_from_scratch_raw83x18_v4"
)
# v5: reverted alongside microban_constants.py's own HOME_FRAME revert (see its
# comment) -- the "com_centered" experiment (hip_pitch +1.198deg) this task was
# built for measured worse for get-up and is reverted; this task now trains at
# the original physical-neutral HOME (hip_pitch -10deg) instead. Bumped so a
# checkpoint from either pose is never silently accepted under the other.
MICROBAN_TELEOP_UPRIGHT_FULLBODY_HOME_REVISION = (
    "physical_neutral_shoulder_zero_hip_neg10_v5"
)
MICROBAN_TELEOP_UPRIGHT_FULLBODY_HOME_HIP_PITCH_RAD = radians(-10.0)
MICROBAN_TELEOP_UPRIGHT_FULLBODY_HOME_ANKLE_PITCH_RAD = 0.0
MICROBAN_TELEOP_UPRIGHT_FULLBODY_HOME_SHOULDER_PITCH_RAD = 0.0
MICROBAN_TELEOP_UPRIGHT_FULLBODY_HOME_ROOT_Z_M = 0.168


def make_microban_teleop_upright_fullbody_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Use the 83-observation/18-raw-action curriculum at the shared HOME."""

    cfg = make_microban_teleop_v12_env_cfg(play=play)
    joint_pos = cfg.scene.entities["robot"].init_state.joint_pos
    if not isinstance(joint_pos, dict):
        raise TypeError("Upright full-body training requires explicit HOME joints")
    # The historical v12 builder sets shoulder pitch to zero; the independent
    # policy now requires the same value in the shared physical neutral HOME.
    if joint_pos != HOME_FRAME.joint_pos:
        raise ValueError("Upright full-body training HOME differs from shared HOME")
    init_state = cfg.scene.entities["robot"].init_state
    if (
        tuple(init_state.pos) != tuple(HOME_FRAME.pos)
        or tuple(init_state.rot) != tuple(HOME_FRAME.rot)
    ):
        raise ValueError("Upright full-body root pose differs from shared HOME")
    if tuple(init_state.rot) != (1.0, 0.0, 0.0, 0.0) or not isclose(
        float(init_state.pos[2]),
        MICROBAN_TELEOP_UPRIGHT_FULLBODY_HOME_ROOT_Z_M,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("Upright full-body HOME trunk is not vertical on the ground")
    for name in ("left_hip_pitch", "right_hip_pitch"):
        if not isclose(
            float(joint_pos[name]),
            MICROBAN_TELEOP_UPRIGHT_FULLBODY_HOME_HIP_PITCH_RAD,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError(f"Upright full-body HOME drifted: {name}")
    for name in ("left_shoulder_pitch", "right_shoulder_pitch"):
        if not isclose(
            float(joint_pos[name]),
            MICROBAN_TELEOP_UPRIGHT_FULLBODY_HOME_SHOULDER_PITCH_RAD,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError(f"Upright full-body HOME drifted: {name}")
    for name in ("left_ankle_pitch", "right_ankle_pitch"):
        if not isclose(
            float(joint_pos[name]),
            MICROBAN_TELEOP_UPRIGHT_FULLBODY_HOME_ANKLE_PITCH_RAD,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError(f"Upright full-body HOME drifted: {name}")
    for name in ("left_knee", "right_knee"):
        if float(joint_pos[name]) != 0.0:
            raise ValueError(f"Upright full-body HOME drifted: {name}")
    # The disabled v12 prior still loads its old-HOME walk004 motion during
    # command construction. A new policy must not read or learn from it.
    cfg.commands.pop("locomotion_prior", None)
    cfg.observations["critic"].terms.pop("locomotion_prior", None)
    cfg.rewards.pop("locomotion_prior_action_target", None)
    cfg.rewards.pop("locomotion_prior_joint_position", None)
    return cfg


# A standard MLP/PPO learns every actor weight, bias, and normalizer from zero.
# The legacy v12 adapter actor, frozen 63-column weights, pinned checkpoint,
# optimizer schedule, gate and exporter are intentionally absent from this task.
MicrobanTeleopUprightFullbodyRlCfg = RslRlOnPolicyRunnerCfg(
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
    wandb_project=MICROBAN_TELEOP_UPRIGHT_FULLBODY_EXPERIMENT,
    experiment_name=MICROBAN_TELEOP_UPRIGHT_FULLBODY_EXPERIMENT,
    save_interval=100,
    num_steps_per_env=MICROBAN_TELEOP_NUM_STEPS_PER_ENV,
    max_iterations=15_000,
)
