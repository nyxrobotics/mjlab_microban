"""Independent, from-scratch full-body teleop training at the shared HOME."""

from __future__ import annotations

from math import isclose, radians

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from mjlab_microban.robot.microban_constants import HOME_FRAME
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_TELEOP_NUM_STEPS_PER_ENV,
)
from mjlab_microban.tasks.microban_teleop_env_cfg import (
    MICROBAN_TELEOP_ANGULAR_TRACKING_STD_RAD_S,
    MICROBAN_TELEOP_FINAL_TRANSLATION_SIGNED_AXIS_RANGES,
    MICROBAN_TELEOP_FINAL_TRANSLATION_VELOCITY_ENVELOPE,
    MICROBAN_TELEOP_HAND_TRACKING_STD_M,
    MICROBAN_TELEOP_ISOLATED_AXIS_PROBABILITIES,
    MICROBAN_TELEOP_LINEAR_TRACKING_STD_M_S,
    MICROBAN_TELEOP_MOVING_HMD_NEUTRAL_PROBABILITY,
    _set_hmd_neck_neutral_probability,
    _set_push_velocity_range,
    _set_teleop_locomotion_stage,
    set_stepping_parameters,
)
from mjlab_microban.tasks.microban_teleop_mdp import (
    normalized_target_clip_excess_l1_sum,
    normalized_target_near_limit_l1_sum,
    raw_action_l2,
)
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    make_microban_teleop_v12_env_cfg,
)
from mjlab_microban.tasks.microban_tracking_env_cfg import (
    MICROBAN_BODY_JOINT_SOFT_LIMITS,
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

    # push_robot's interval is inherited unchanged from the upstream velocity
    # task default (1.0-3.0 s). A completed 15,000-iteration crouched-HOME run
    # still showed a durable fall-rate regression once pushes reach full
    # strength even with a 3-step magnitude ramp (see the curriculum split
    # below). Comparing against two more mature humanoid whole-body-control
    # research recipes (TWIST2's G1 legged_gym config, and Teleopit's
    # near-stock mjlab tracking config) found neither ramps push MAGNITUDE at
    # all -- both apply a fixed full-strength push from iteration 0 -- but
    # Teleopit/stock mjlab uses a push interval of 4.0-6.0 s, roughly 2-4x
    # longer than this task's inherited 1.0-3.0 s, giving the policy
    # substantially more time to recover balance between pushes. Widened to
    # match, independent of (and in addition to) the existing magnitude ramp.
    if "push_robot" in cfg.events:
        cfg.events["push_robot"].interval_range_s = (4.0, 6.0)

    # v12 sets clip=None and drops target_clip_excess/target_near_limit/
    # raw_action_l2 because they assert on an absolute target clip, which the
    # legacy pretrained v12 actor's raw/unclipped recurrence contract requires
    # unmet. This task has no such actor (a fresh MLP, trained from scratch),
    # so nothing requires clip=None here. Restoring the base task's original
    # clip and reward trio (unchanged weights) because a from-scratch run
    # without them exhibited the exact runaway get-up hit and fixed the same
    # way: action std climbed slowly under the step-3000 envelope/tracking
    # escalation alone (1.23->1.66 across iterations 2800->3800, fell_over
    # ~0.5, stable), then broke into runaway growth the moment step-4000
    # restored external pushes alone (std 1.9->2.47 and fell_over 10->14,
    # still climbing, across iterations 4000->4600) -- get-up's own docstring
    # (microban_getup_env_cfg.py) describes the identical mechanism: an
    # unclipped/unpenalized raw action lets the actor grow std for free once
    # large corrective actions are frequently needed, since excess-beyond-clip
    # magnitude is invisible to reward but still entropy-rewarding.
    action = cfg.actions["joint_pos"]
    action.clip = MICROBAN_BODY_JOINT_SOFT_LIMITS
    cfg.rewards["target_clip_excess"] = RewardTermCfg(
        func=normalized_target_clip_excess_l1_sum,
        weight=-2.0,
        params={"action_name": "joint_pos"},
    )
    cfg.rewards["target_near_limit"] = RewardTermCfg(
        func=normalized_target_near_limit_l1_sum,
        weight=-1.0,
        params={
            "action_name": "joint_pos",
            "margin_ratio": 0.05,
        },
    )
    cfg.rewards["raw_action_l2"] = RewardTermCfg(
        func=raw_action_l2,
        weight=-0.01,
        params={"action_name": "joint_pos"},
    )

    # The shared base curriculum's "restore pushes and expand final
    # translation at formal locomotion gate" step (3000) bundles three
    # escalations at once (wider velocity envelope, tighter tracking_std,
    # and external pushes back on), unlike every earlier stage, which each
    # change exactly one thing. A from-scratch run at the new centered HOME
    # plateaued hard right at this step: fell_over stayed ~11-15 and
    # velocity-tracking progress stayed ~0.39-0.45 with no trend at all
    # across the entire 3000->15000 remainder of a 15,000-iteration run
    # (checked directly against that run's own per-1000-iteration binned
    # log averages). Overridden here, task-local only (the shared base and
    # its own contract test, and the historical v8/v12 lineages that still
    # use it, are untouched): split into expanding the envelope/std alone
    # at 3000, then restoring pushes separately at 4000, giving the policy
    # a dedicated window to consolidate the harder tracking task before
    # also having to reject external disturbances.
    if "staged_curriculum" in cfg.curriculum:
        stages = cfg.curriculum["staged_curriculum"].params["stages"]
        split_index = next(
            (
                index
                for index, stage in enumerate(stages)
                if stage["step"] == 3000 * 24
            ),
            None,
        )
        if split_index is None:
            raise ValueError(
                "Upright full-body curriculum split target (step 3000) not found"
            )
        stages[split_index] = {
            "name": (
                "expand final translation at formal locomotion gate "
                "(pushes still off)"
            ),
            "step": 3000 * 24,
            "apply": lambda env: _set_teleop_locomotion_stage(
                env,
                envelope=MICROBAN_TELEOP_FINAL_TRANSLATION_VELOCITY_ENVELOPE,
                signed_axis_ranges=(
                    MICROBAN_TELEOP_FINAL_TRANSLATION_SIGNED_AXIS_RANGES
                ),
                signed_axis_probabilities=(
                    MICROBAN_TELEOP_ISOLATED_AXIS_PROBABILITIES
                ),
                linear_tracking_std=MICROBAN_TELEOP_LINEAR_TRACKING_STD_M_S,
                angular_tracking_std=MICROBAN_TELEOP_ANGULAR_TRACKING_STD_RAD_S,
            ),
        }
        # Splitting the bundled step-3000 stage stopped the earlier unbounded
        # action-std runaway (see the raw target-clip-excess fix), but a
        # rendered play video of the completed crouched-HOME run showed the
        # robot still falling repeatedly, and log forensics found the
        # instant-full-strength push at step 4000 alone -- even split off
        # from the envelope change -- causes a durable ~3-4x regression that
        # the rest of training never recovers from: fell_over jumps from
        # 2.37/iter (the best point in the whole run, right before 4000) to
        # 16-21/iter within 100 iterations and stays in the 11-15 range for
        # the remaining ~10,000 iterations, with no further trend. Ramp the
        # push magnitude in three steps instead of one jump, reaching the
        # same final +-0.5 m/s by step 5400 (still well before the next
        # base-curriculum stage at 6000).
        def _insert_stage_sorted(stage: dict) -> None:
            """Insert a stage keyed by its own step, keeping steps sorted.

            The base list already has its own stages interleaved between the
            step values this task overrides (e.g. "final pure-yaw isolated
            axes" at 4500 sits between this task's pushes-ramp steps at 4000
            and 4700/5400) -- inserting at a fixed offset from an earlier
            index silently breaks the "steps strictly increasing" contract
            ResumeSafeStepBasedStagedCurriculum enforces. Insert by value
            instead.
            """

            index = next(
                (i for i, s in enumerate(stages) if s["step"] > stage["step"]),
                len(stages),
            )
            stages.insert(index, stage)

        _insert_stage_sorted(
            {
                "name": "restore pushes at the formal locomotion gate (1 of 3)",
                "step": 4000 * 24,
                "apply": lambda env: _set_push_velocity_range(
                    env, x=(-0.15, 0.15), y=(-0.15, 0.15)
                ),
            }
        )
        _insert_stage_sorted(
            {
                "name": "restore pushes at the formal locomotion gate (2 of 3)",
                "step": 4700 * 24,
                "apply": lambda env: _set_push_velocity_range(
                    env, x=(-0.3, 0.3), y=(-0.3, 0.3)
                ),
            }
        )
        _insert_stage_sorted(
            {
                "name": "restore pushes at the formal locomotion gate (3 of 3)",
                "step": 5400 * 24,
                "apply": lambda env: _set_push_velocity_range(
                    env, x=(-0.5, 0.5), y=(-0.5, 0.5)
                ),
            }
        )

        # The shared base curriculum's step-7000 stage ("enable moving-HMD,
        # stationary no-step guard, and broad hand tracking") bundles three
        # more escalations in one atomic apply(): HMD neck motion turning on
        # (neutral_probability 1.0->0.2), hand-target tracking activating
        # (reward weight 0->1.0, command rel_active 0->0.7), and the
        # no-stepping penalty activating (weight 0->-1.0). Log forensics on
        # the completed crouched-HOME run found this produced the single
        # worst fall-rate spike in the entire 15,000-iteration run (fell_over
        # 12.9->28.6/iter within 100 iterations, worse than the push shock
        # above), the same bundling anti-pattern already fixed once for the
        # step-3000/4000 stage. Replace it with a task-local, three-step
        # split: hand tracking alone, then HMD motion ramped in two steps,
        # then the no-stepping penalty last, all before the next base stage
        # at 8500.
        hmd_split_index = next(
            (
                index
                for index, stage in enumerate(stages)
                if stage["step"] == 7000 * 24
            ),
            None,
        )
        if hmd_split_index is None:
            raise ValueError(
                "Upright full-body curriculum split target (step 7000) not found"
            )
        stages[hmd_split_index] = {
            "name": "enable broad hand tracking (1 of 3)",
            "step": 7000 * 24,
            "apply": lambda env: (
                env.reward_manager.get_term_cfg("hand_target_tracking").__setattr__(
                    "weight", 1.0
                ),
                env.reward_manager.get_term_cfg("hand_target_tracking").params.__setitem__(
                    "std", MICROBAN_TELEOP_HAND_TRACKING_STD_M
                ),
                env.command_manager.get_term_cfg("hand_target").__setattr__(
                    "rel_active", 0.7
                ),
            ),
        }
        _insert_stage_sorted(
            {
                "name": "ramp moving-HMD toward final probability (2 of 3)",
                "step": 7500 * 24,
                "apply": lambda env: _set_hmd_neck_neutral_probability(
                    env, neutral_probability=0.6
                ),
            }
        )
        _insert_stage_sorted(
            {
                "name": (
                    "reach final moving-HMD probability and enable stationary "
                    "no-step guard (3 of 3)"
                ),
                "step": 8000 * 24,
                "apply": lambda env: (
                    _set_hmd_neck_neutral_probability(
                        env,
                        neutral_probability=(
                            MICROBAN_TELEOP_MOVING_HMD_NEUTRAL_PROBABILITY
                        ),
                    ),
                    set_stepping_parameters(
                        env,
                        air_time_weight=3.0,
                        no_stepping_penalty_weight=-1.0,
                    ),
                ),
            }
        )
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
