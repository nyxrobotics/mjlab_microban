# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Microban get-up (fall recovery) environment.

Separate task and policy from microban_velocity_env_cfg: getting up from an arbitrary
fallen pose (front/back/side, or any limb configuration from a cold power-on) has a
different contact/trajectory structure than steady-state walking, so training it
alongside locomotion risks interference. See microban_teleop/docs/getup_research.md
for the survey this follows (HumanUP, HoST, FRASA, ANYmal) and the switch heuristic
(height/orientation threshold, handled outside this env, in the real control loop).

All 21 joints are actuated here (including neck/head/arms) — getting up needs every
available DOF, and the "keep the neck predictable" concern that excludes it from the
walking policy doesn't apply during a recovery maneuver.
"""

import numpy as np
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.rewards import joint_torques_l2
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_microban.robot.microban_constants import MICROBAN_ROBOT_CFG
from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg

from mjlab.envs.mdp import dr
from mjlab.scene import SceneCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig

from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from mjlab_microban.tasks.mdp import (
    step_based_staged_curriculum,
    head_height_reward,
    standing_pose_reward,
    foot_flat_reward,
    hold_airborne,
    standing_bonus,
    on_feet_reward,
    standing_torque_penalty,
    extreme_joint_velocity,
)

STANDING_HEIGHT = 0.168  # trunk height when standing (HOME_FRAME.pos z, microban_constants.py)

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
    # Get-up's extreme fallen/tangled poses generate far more simultaneous contacts
    # than steady-state walking; the auto-sized default overflowed ("nefc overflow"
    # warnings, ~700-800 needed) within the first few dozen training iterations.
    # nconmax is left at its auto-sized default: it wasn't the one overflowing, and
    # setting it explicitly (even at a seemingly modest value) blew GPU memory — it's
    # apparently a much larger per-env allocation than njmax at the same number.
    njmax=1200,
    # Same root cause, different counter: self_collision's contact sensor (trunk
    # subtree vs. itself) overflowed its match-count buffer (default 64) once njmax
    # stopped silently capping how many contacts got that far — observed climbing
    # past 90 within seconds of training. mjlab's own G1/Go1 configs already bump this
    # to 500 for similar reasons.
    contact_sensor_maxmatch=500,
)


def make_microban_getup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_velocity_env_cfg()

    cfg.viewer = deepcopy(VIEWER_CONFIG)
    cfg.sim = deepcopy(SIM_CFG)
    cfg.scene = deepcopy(SCENE_CFG)

    #---------------------------- Sensors ---------------------------
    self_collision_sensor_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    # Needed so the policy can tell "both feet off the ground" from "on the ground" —
    # the airborne-compliance reward (see below) is gated on this, and the actor
    # observation of it (also below) is what lets that distinction survive to
    # inference, where there's no ground-truth "being held" flag to read instead.
    feet_ground_sensor_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="subtree", pattern=r"^(foot|foot_2)$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    cfg.scene.sensors = (self_collision_sensor_cfg, feet_ground_sensor_cfg)

    #---------------------------- Terrain ---------------------------
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    #---------------------------- Actions ---------------------------
    # Excludes head/neck_roll/neck_pitch, matching the walking policy's own exclusion
    # (microban_velocity_env_cfg.py) — get-up shouldn't move the neck either. Their
    # default pose is 0.0 for all three, so they don't need a hold_at_default_pose
    # event: EntityData.joint_pos_target resetting excluded joints to 0.0 every episode
    # (the bug the walking policy's own comment documents) is a no-op when the default
    # already IS 0.0, unlike e.g. the elbows/shoulders whose defaults are far from zero.
    dofs_filter = r".*(?<!head)(?<!neck_roll)(?<!neck_pitch)$"
    cfg.actions["joint_pos"].actuator_names = (dofs_filter,)
    cfg.actions["joint_pos"].scale = 1.0
    # Unclipped, early PPO exploration (init_std=1.0, no offset limit) occasionally
    # commands multi-radian targets on some joint, which — combined with 21 joints
    # moving at once from an already-extreme randomized fallen pose — produced a
    # genuine MuJoCo solver blowup (NaN qpos/qvel) reproduced at 4096 envs within ~70
    # steps of real training, well before hold_airborne's own >=2s cooldown could even
    # fire once. A flat +-90 deg clip is generic (not per-joint-tuned) but removes the
    # wild-target tail while still leaving room to discover a get-up motion; matches
    # HoST's "start with a restrictive action bound" curriculum idea, as a static
    # bound rather than a scheduled one for now.
    cfg.actions["joint_pos"].clip = {r".*": (-1.57, 1.57)}

    #---------------------------- Commands ---------------------------
    # No velocity command: this task is "get up", not "walk somewhere".
    del cfg.commands["twist"]

    #---------------------------- Observations ----------------------
    del cfg.observations["actor"].terms["command"]
    del cfg.observations["actor"].terms["base_lin_vel"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["command"]
    del cfg.observations["critic"].terms["height_scan"]
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["critic"].terms["foot_air_time"]
    del cfg.observations["critic"].terms["foot_contact"]
    del cfg.observations["critic"].terms["foot_contact_forces"]
    # projected_gravity/base_ang_vel keep the base config's defaults. joint_pos/
    # joint_vel/actions are filtered to dofs_filter (matching the action space, same
    # as the walking policy) rather than left at all 21 joints — "actions" naturally
    # shrinks to 18 with the action space above, so filtering joint_pos/joint_vel too
    # keeps everything consistent rather than mixing 21- and 18-wide state.
    cfg.observations["actor"].terms["joint_pos"] = ObservationTermCfg(
        func=velocity_mdp.joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(dofs_filter,))},
    )
    cfg.observations["actor"].terms["joint_vel"] = ObservationTermCfg(
        func=velocity_mdp.joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(dofs_filter,))},
    )
    cfg.observations["critic"].terms["joint_pos"] = deepcopy(cfg.observations["actor"].terms["joint_pos"])
    cfg.observations["critic"].terms["joint_vel"] = deepcopy(cfg.observations["actor"].terms["joint_vel"])

    cfg.observations["actor"].terms["foot_contact"] = ObservationTermCfg(
        func=velocity_mdp.foot_contact,
        params={"sensor_name": feet_ground_sensor_cfg.name},
    )

    #---------------------------- Rewards ---------------------------
    del cfg.rewards["track_linear_velocity"]
    del cfg.rewards["track_angular_velocity"]
    del cfg.rewards["pose"]  # tied to the (now-removed) twist command's standing/walking/running regimes
    del cfg.rewards["air_time"]
    del cfg.rewards["foot_clearance"]
    del cfg.rewards["foot_swing_height"]
    del cfg.rewards["foot_slip"]
    del cfg.rewards["soft_landing"]

    # Dropped (was on by default from the base config): it ran ungated for the whole
    # episode, including during the actual recovery motion — where rolling/tumbling
    # through non-upright orientations is exactly what's needed, so an always-on "be
    # upright" pressure fights it. The orientation check that actually matters (once
    # standing) already lives inside standing_bonus, gated to only apply there.
    del cfg.rewards["upright"]

    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk",)
    cfg.rewards["body_ang_vel"].weight = -0.05

    cfg.rewards["angular_momentum"].weight = -0.01

    cfg.rewards["action_rate_l2"].weight = -0.02  # lighter than walking: getting up needs large motions

    # Head height instead of trunk height: trunk-height-alone can't tell "right-side
    # up" from "upside down" (both have the trunk high), and once head height is also
    # rewarded, a separate trunk-height term is redundant — head height already only
    # goes up when the trunk is BOTH elevated and correctly oriented (an inverted pose
    # has the head low even with the trunk high), and it's just as orientation-blind
    # while still flat as the trunk version was (same world-Z-height shape). Keeping
    # both would let a high-but-inverted trunk still collect partial credit from the
    # trunk term, fighting the very thing head height is meant to fix.
    # HEAD_STANDING_HEIGHT measured directly (make_microban_velocity_env_cfg, standing
    # pose, "head" body world Z).
    # Lowered from the theoretical standing-pose value (0.298, computed from
    # make_microban_velocity_env_cfg's own standing keyframe) to what this policy
    # actually converges to in practice (measured: 256-env rollout of the trained
    # checkpoint settles at head height 0.260 mean, ~94.5% of envs above 0.25) — no
    # point rewarding height the motion doesn't actually reach.
    HEAD_STANDING_HEIGHT = 0.260
    HEAD_ASSET_CFG = SceneEntityCfg("robot", body_names=("head",))
    HEAD_STANDING_THRESHOLD = HEAD_STANDING_HEIGHT * 0.85
    cfg.rewards["head_height"] = RewardTermCfg(
        func=head_height_reward,
        weight=6.0,
        params={
            "target_height": HEAD_STANDING_HEIGHT,
            "asset_cfg": HEAD_ASSET_CFG,
            "sensor_name": feet_ground_sensor_cfg.name,
        },
    )

    # head_drop_penalty (explicit penalty for downward head velocity, to discourage
    # raise-then-drop bobbing) is dropped — head_height's own weight was raised
    # instead, which turned out to matter more than the drop penalty did.

    # HoST's "post-task" mechanism (arXiv:2502.08378): a reward that only pays out once
    # standing is (nearly) reached, so completing it earlier and holding it accumulates
    # strictly more over the episode. height+upright alone measurably plateau around
    # 80% of target height without ever finishing the motion (see handling_research.md)
    # — this term exists specifically to close that gap, not to duplicate height/upright.
    # "is standing" is gated on head height (HEAD_ASSET_CFG) everywhere below;
    # orientation itself still reads the trunk body specifically (asset_cfg).
    cfg.rewards["standing_bonus"] = RewardTermCfg(
        func=standing_bonus,
        weight=5.0,
        params={
            "height_threshold": HEAD_STANDING_THRESHOLD,
            "upright_std": np.sqrt(0.1),
            "head_asset_cfg": HEAD_ASSET_CFG,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk",)),
        },
    )

    # Distinguishes genuinely standing-on-feet from other stable-but-not-standing
    # configurations (sitting/kneeling) that height+upright alone can't tell apart.
    cfg.rewards["on_feet"] = RewardTermCfg(
        func=on_feet_reward,
        weight=2.0,
        params={
            "sensor_name": feet_ground_sensor_cfg.name,
            "target_height": HEAD_STANDING_HEIGHT,
            "head_asset_cfg": HEAD_ASSET_CFG,
        },
    )

    # on_feet only checks contact, not angle — a foot resting on its edge/toe still
    # scores well there. This rewards the sole actually being level once standing.
    cfg.rewards["foot_flat"] = RewardTermCfg(
        func=foot_flat_reward,
        weight=1.5,
        params={
            "std": np.sqrt(0.3),
            "height_threshold": HEAD_STANDING_THRESHOLD,
            "head_asset_cfg": HEAD_ASSET_CFG,
            "asset_cfg": SceneEntityCfg("robot", body_names=(r"^(foot|foot_2)$",)),
        },
    )

    # Hand-support shaping (bracing/releasing a hand/forearm during recovery) is
    # dropped for now: after several iterations (flat reward -> velocity-scaled ->
    # thresholded +/0/-1 -> thresholded +/0) it was still producing worse overall
    # behavior than leaving it out, more complexity than it was earning back. Head
    # height (+ head_drop) is the priority signal; hands can come back later if the
    # legs-and-torso motion alone isn't enough.

    # Once standing, push toward whichever stance costs the least holding torque —
    # observed behavior without this was a wide leg-split with heavily tilted ankles
    # (satisfies height/upright/on-feet, but is not how this robot would naturally
    # balance). A torque-cost signal is a more principled way to discourage that than
    # penalizing deviation from a specific hand-picked default pose.
    cfg.rewards["standing_torque"] = RewardTermCfg(
        func=standing_torque_penalty,
        weight=-0.5,
        params={
            "height_threshold": HEAD_STANDING_THRESHOLD,
            "head_asset_cfg": HEAD_ASSET_CFG,
            "asset_cfg": SceneEntityCfg("robot", actuator_names=(dofs_filter,)),
        },
    )

    # Light secondary nudge toward the natural default pose once standing — kept
    # small (well below head_height=6.0/standing_bonus=5.0) so it only polishes the
    # motion the other terms already produce, rather than dictating it.
    # gate_center=0.230 / gate_sharpness=0.005: near-zero below ~0.22, ramped to full
    # strength by ~0.24 — a steep transition around the user-specified 0.230m rather
    # than a hard cutoff or a linear ramp across the whole approach (which would stay
    # weak right up to standing instead of ever clearly taking over). Weight (4.0) is
    # a real increase from the flat version's 1.0 but deliberately still below
    # head_height (6.0) / standing_bonus (5.0) — not falling over matters more than
    # matching the default pose exactly.
    cfg.rewards["standing_pose"] = RewardTermCfg(
        func=standing_pose_reward,
        weight=4.0,
        params={
            "gate_center": 0.230,
            "gate_sharpness": 0.005,
            "std": 1.0,
            "head_asset_cfg": HEAD_ASSET_CFG,
            "asset_cfg": SceneEntityCfg("robot", joint_names=(dofs_filter,)),
        },
    )

    # "Don't flail when picked up" (airborne_compliance_reward) is dropped along with
    # hold_airborne — with the triggering event disabled, this was only ever paying
    # out for the rare accidental airborne moment, not exercising the behavior it was
    # meant to train. See microban_teleop/docs/handling_research.md for the shelved
    # design if this gets revisited later.

    # Torque penalty starts at 0 (stage 1: find any way up) and ramps in via curriculum
    # once standing succeeds often (stage 2: keep it within what the real servos can
    # deliver — see microban_teleop/docs/getup_research.md on why this matters for the
    # XC330-T288-T's torque/current limits).
    cfg.rewards["joint_torques_l2"] = RewardTermCfg(func=joint_torques_l2, weight=0.0)

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=velocity_mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_sensor_cfg.name},
    )

    #---------------------------- Terminations ----------------------
    # No early "fell over" exit — starting fallen is the whole point. Keep time_out and
    # out_of_terrain_bounds only.
    del cfg.terminations["fell_over"]

    # Safety net for hold_airborne (see events below): an external force applied to
    # the trunk on top of an already-extreme randomized fallen pose occasionally pushes
    # the MuJoCo solver into a genuine NaN/Inf blowup (reproduced directly: ~1 env per
    # few hundred steps out of 4096, even after staggering hold_airborne's first-trigger
    # timing and reducing its force range). A single NaN env otherwise aborts the whole
    # training run (rsl_rl's check_nan checks the full batch). Resetting just the
    # affected env here, before it corrupts anything else, is cheap and keeps training
    # running instead.
    cfg.terminations["nan_detection"] = TerminationTermCfg(func=envs_mdp.nan_detection)
    # See extreme_joint_velocity's docstring: catches unphysical-but-still-finite
    # blowups that precede (and, left unchecked, cause) the NaN nan_detection catches.
    cfg.terminations["extreme_joint_velocity"] = TerminationTermCfg(
        func=extreme_joint_velocity,
        params={"max_joint_vel": 50.0},
    )

    #---------------------------- Events ------------------------------
    # Full-range orientation (any lying pose) and joint angles (any limb configuration,
    # covering "powered on already on the ground"). position_range is wider than any
    # joint's own range so the post-clamp result saturates at that joint's actual limits
    # (reset_joints_by_offset clamps to soft_joint_pos_limits).
    cfg.events["reset_base"].params["pose_range"] = {
        "x": (-0.1, 0.1),
        "y": (-0.1, 0.1),
        "z": (0.05, 0.25),
        "roll": (-3.14159, 3.14159),
        "pitch": (-3.14159, 3.14159),
        "yaw": (-3.14159, 3.14159),
    }
    cfg.events["reset_robot_joints"].params["position_range"] = (-3.14159, 3.14159)

    del cfg.events["push_robot"]  # no external pushes needed while learning to stand up

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
        r".*left_foot_collision.*",
        r".*right_foot_collision.*",
    )
    cfg.events["base_com"].params["asset_cfg"].body_names = ("trunk",)
    cfg.events["base_com"].params["ranges"] = {
        0: (-0.005, 0.005),
        1: (-0.005, 0.005),
        2: (-0.005, 0.005),
    }

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

    # "Don't flail when picked up" (hold_airborne, simulating a hand lifting the
    # trunk) is shelved for now — parked here as a deliberate scope cut, not deleted:
    # combining it with get-up made the training signal harder to read (mid-episode
    # near-weightless trunk confusing to judge get-up quality against) and diluted
    # the reward the get-up motion itself needed (airborne_compliance_reward has since
    # been removed too, see below). The airborne gate on head_height_reward is left in
    # place (a harmless no-op without hold_airborne ever firing) so re-enabling this
    # later is still a small, contained change.
    #
    # cfg.events["hold_airborne"] = EventTermCfg(
    #     mode="step",
    #     func=hold_airborne,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=("trunk",)),
    #         "force_z_range": (8.0, 11.5),
    #         "force_lateral_range": (-0.5, 0.5),
    #         "torque_range": (0.0, 0.0),
    #         "duration_s": (1.0, 4.0),
    #         "cooldown_s": (3.0, 8.0),
    #     },
    # )

    #---------------------------- Curriculum ------------------------
    cfg.curriculum = {}
    cfg.curriculum["staged_curriculum"] = CurriculumTermCfg(
        func=step_based_staged_curriculum,
        params={
            "stages": [
                {
                    # -1e-4 (the original weight) proved too weak: the trained motion
                    # still looked violently forceful throughout the recovery, not just
                    # once standing (where standing_torque separately discourages it).
                    # 10x stronger to actually discourage that during the whole episode.
                    "name": "add torque regularization",
                    "step": 5000 * 24,
                    "apply": lambda env: env.reward_manager.get_term_cfg("joint_torques_l2").__setattr__(
                        "weight", -1e-3
                    ),
                },
            ],
        },
    )

    #---------------------------- Play mode -------------------------
    if play:
        cfg.curriculum = {}
        cfg.events["reset_base"].interval_range_s = (0.0, 0.0)
        cfg.observations["actor"].enable_corruption = False

    return cfg


MicrobanGetupRlCfg = RslRlOnPolicyRunnerCfg(
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
    wandb_project="mjlab_microban_getup",
    experiment_name="mjlab_microban_getup",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=15_000,
)
