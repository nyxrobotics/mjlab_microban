# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Microban get-up (fall recovery) environment, action contract v4.

Separate task and policy from microban_velocity_env_cfg: getting up from an
arbitrary fallen pose has a different contact/trajectory structure than
walking. The policy controls 18 body joints; head and both neck joints are
held at their measured angles (see microban_getup_action.py).

Why v4 (2026-09-29 investigation; scripts and numbers in the session's
redesign/ reports):

* Every policy that ever stood up in this task -- five independent
  from-scratch runs on 09-24/25, and the lineage the robot ran on 09-26 --
  was trained with a flat +-1.57 rad absolute target clip, the policy's RAW
  previous output as its previous-action observation, no clip-excess
  penalty, and no simulated IMU delay. All 17+ runs after 7e0b548 (09-27)
  changed exactly those and none stood, while reward tuning on 09-28/29
  moved the standing rate by < 3 %.
* Physics: at the robot's RL gain (P=125) one XC330 gives only ~0.42 Nm/rad,
  while gravity tips the standing robot at ~1.65 Nm/rad about the ankles.
  No pose holds statically (HOME falls in ~1-2 s). Standing is active
  balance that needs targets far past the joint angle to get useful torque
  (hip_roll at HOME: 0.13-0.20 Nm inside the per-joint soft-limit clip vs
  0.6-0.7 Nm with +-1.57). The per-joint clip and the clip-excess penalty
  removed exactly that authority.
* The model_46000 policy the robot ran stands in 0/64 envs of the v3 task
  and in 63/64 once the v4 clip, raw feedback and no-delay are restored,
  so the current robot model, actuator model and HOME still support it.

Two reward sets share this contract:

* "v42": the exact reward recipe of 2026-09-25_18-47-14, the from-scratch
  run that stood by iteration ~1200 and became the robot's policy.
* "redesign": the same core, minus the terms measured to hurt, plus HoST's
  post-standing terms -- every standing term gated above kneeling/squatting
  height (see _add_redesign_rewards).
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

from mjlab_microban.robot.microban_constants import HOME_FRAME
from mjlab_microban.tasks.mdp import (
    step_based_staged_curriculum,
    reward_based_staged_curriculum,
    head_height_reward,
    home_pose_reward,
    standing_bonus,
    feet_stance_reward,
    standing_stability_reward,
    upright_balance_reward,
    on_feet_reward,
    standing_torque_penalty,
    home_stillness_reward,
    extreme_joint_velocity,
    reset_near_home_fraction,
    hands_released_reward,
)
from mjlab_microban.tasks.microban_getup_action import (
    GetupJointPositionActionCfg,
    raw_getup_action,
)
from mjlab_microban.tasks.microban_getup_actuator import (
    make_getup_robot_cfg,
)

STANDING_HEIGHT = float(HOME_FRAME.pos[2])
GETUP_EPISODE_LENGTH_S = 20.0  # Match the robot's automatic get-up timeout.
# Flat absolute target clip on all 18 body joints (see module docstring).
GETUP_ACTION_CLIP_RAD = 1.57
GETUP_ACTION_CLIP = {r".*": (-GETUP_ACTION_CLIP_RAD, GETUP_ACTION_CLIP_RAD)}
GETUP_REWARD_SETS = ("v42", "redesign", "posture", "posture_strong")
# Final (post-curriculum) standing_pose / hip_pose weights per reward set,
# and the feet_stance weight (0 = term absent).
_POSE_FINAL_WEIGHTS = {
    "v42": (30.0, 15.0),
    "redesign": (30.0, 15.0),
    "posture": (120.0, 60.0),
    "posture_strong": (240.0, 120.0),
}
_FEET_STANCE_WEIGHTS = {"v42": 0.0, "redesign": 0.0, "posture": 10.0, "posture_strong": 20.0}
# Lateral distance between the two foot bodies at HOME, by forward kinematics.
HOME_FEET_LATERAL_M = 0.094

# Virtual head height (trunk COM + 0.07324 m along the trunk's up axis, see
# mdp._head_height) at HOME: 0.29398 m by MuJoCo forward kinematics. Kneeling
# upright reaches 0.226-0.239 and the deepest flat-foot squat 0.222-0.227,
# so the 0.9x standing gate (0.2646) is above both. 0.260 was tried on 09-28
# and made a forearm-propped tripod (0.205) earn 91 % of the height reward.
HEAD_STANDING_HEIGHT = 0.294
STANDING_GATE_HEIGHT = 0.9 * HEAD_STANDING_HEIGHT
# Despite the name, _head_height only uses .name to resolve the robot entity.
HEAD_ASSET_CFG = SceneEntityCfg("robot", body_names=("head",))
DOFS_FILTER = r".*(?<!head)(?<!neck_roll)(?<!neck_pitch)$"

SCENE_CFG = SceneCfg(
    terrain=TerrainEntityCfg(
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=0,
    ),
    num_envs=1,
    extent=2.0,
    entities={"robot": make_getup_robot_cfg()},
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
    # Tangled fallen poses generate far more contacts than walking; the
    # auto-sized njmax overflowed (~700-800 needed). nconmax stays auto:
    # setting it explicitly blew GPU memory.
    njmax=1200,
    # The self_collision sensor (trunk subtree vs itself) exceeded the
    # default 64 matches; mjlab's own G1/Go1 configs use 500 too.
    contact_sensor_maxmatch=500,
)


def make_microban_getup_env_cfg(
    play: bool = False,
    reward_set: str = "v42",
    imu_delay_max_lag: int = 0,
) -> ManagerBasedRlEnvCfg:
    if reward_set not in GETUP_REWARD_SETS:
        raise ValueError(f"Unknown get-up reward set {reward_set!r}; expected one of {GETUP_REWARD_SETS}")
    cfg = make_velocity_env_cfg()
    cfg.episode_length_s = GETUP_EPISODE_LENGTH_S

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
    # Reward-only (the robot has no foot or hand contact sensors, so neither
    # is an actor observation).
    feet_ground_sensor_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="subtree", pattern=r"^(foot|foot_2)$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    # "radius"/"radius_2" are the forearm bodies (see robot.xml).
    hands_ground_sensor_cfg = ContactSensorCfg(
        name="hands_ground_contact",
        primary=ContactMatch(mode="body", pattern=r"^(radius|radius_2)$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="netforce",
        num_slots=1,
    )
    cfg.scene.sensors = (self_collision_sensor_cfg, feet_ground_sensor_cfg, hands_ground_sensor_cfg)

    #---------------------------- Terrain ---------------------------
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    #---------------------------- Actions ---------------------------
    # 18 body joints; the robot holds head/neck at their measured angles with
    # P gain 400 during get-up, which the get-up action/actuator reproduce.
    # Absolute target = default pose + raw action, clipped at +-1.57 rad. The
    # clip exists to remove early exploration's multi-radian tail (which
    # blew up the solver when unclipped); it is deliberately NOT each joint's
    # soft limit -- see the module docstring for the measured reason.
    cfg.actions["joint_pos"].actuator_names = (DOFS_FILTER,)
    cfg.actions["joint_pos"].scale = 1.0
    base_action = cfg.actions["joint_pos"]
    assert isinstance(base_action, JointPositionActionCfg)
    cfg.actions["joint_pos"] = GetupJointPositionActionCfg(
        entity_name=base_action.entity_name,
        actuator_names=base_action.actuator_names,
        scale=base_action.scale,
        offset=base_action.offset,
        preserve_order=base_action.preserve_order,
        use_default_offset=base_action.use_default_offset,
        clip=dict(GETUP_ACTION_CLIP),
    )

    #---------------------------- Commands ---------------------------
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
    # IMU terms keep the base config's noise (gyro +-0.2, gravity +-0.05).
    # imu_delay_max_lag > 0 adds the walking task's simulated IMU latency
    # (0..N policy ticks, resampled every 64 steps). The first standing v4
    # policy, trained without it, dropped from 53/54 to 34/62 fallen-start
    # stands under 0-3 ticks, so the robot-bound policy is fine-tuned with it.
    if imu_delay_max_lag:
        for name in ("base_ang_vel", "projected_gravity"):
            term = deepcopy(cfg.observations["actor"].terms[name])
            term.delay_min_lag = 0
            term.delay_max_lag = imu_delay_max_lag
            term.delay_update_period = 64
            cfg.observations["actor"].terms[name] = term
    cfg.observations["actor"].terms["joint_pos"] = ObservationTermCfg(
        func=velocity_mdp.joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(DOFS_FILTER,))},
    )
    cfg.observations["actor"].terms["joint_vel"] = ObservationTermCfg(
        func=velocity_mdp.joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(DOFS_FILTER,))},
    )
    cfg.observations["critic"].terms["joint_pos"] = deepcopy(cfg.observations["actor"].terms["joint_pos"])
    cfg.observations["critic"].terms["joint_vel"] = deepcopy(cfg.observations["actor"].terms["joint_vel"])
    # Previous RAW policy output (zero right after a reset); the robot feeds
    # back its ONNX model's own last output the same way.
    for group_name in ("actor", "critic"):
        cfg.observations[group_name].terms["actions"] = ObservationTermCfg(
            func=raw_getup_action,
            params={"action_name": "joint_pos"},
        )

    #---------------------------- Rewards ---------------------------
    for name in (
        "track_linear_velocity",
        "track_angular_velocity",
        "pose",  # tied to the removed twist command
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "soft_landing",
        # Always-on "be upright" fights the tumbling the recovery needs.
        "upright",
    ):
        del cfg.rewards[name]

    sensors = {
        "feet": feet_ground_sensor_cfg.name,
        "hands": hands_ground_sensor_cfg.name,
        "self_collision": self_collision_sensor_cfg.name,
    }
    if reward_set == "v42":
        _add_v42_rewards(cfg, sensors)
    else:
        _add_redesign_rewards(cfg, sensors)
    if reward_set in ("posture", "posture_strong"):
        _add_posture_rewards(cfg, _FEET_STANCE_WEIGHTS[reward_set])
    standing_pose_final, hip_pose_final = _POSE_FINAL_WEIGHTS[reward_set]

    #---------------------------- Terminations ----------------------
    # No "fell over" exit: starting fallen is the whole point.
    del cfg.terminations["fell_over"]
    # A single NaN env would abort the whole run (rsl_rl checks the batch).
    cfg.terminations["nan_detection"] = TerminationTermCfg(func=envs_mdp.nan_detection)
    # Catches unphysical-but-finite blowups that precede NaNs.
    cfg.terminations["extreme_joint_velocity"] = TerminationTermCfg(
        func=extreme_joint_velocity,
        params={"max_joint_vel": 50.0},
    )

    #---------------------------- Events ------------------------------
    # Full-range fallen poses from step 0 (a widening fallen-pose curriculum
    # was measured to make things worse). position_range exceeds every
    # joint's range so the post-clamp sample saturates at real limits.
    cfg.events["reset_base"].params["pose_range"] = {
        "x": (-0.1, 0.1),
        "y": (-0.1, 0.1),
        "z": (0.05, 0.25),
        "roll": (-3.14159, 3.14159),
        "pitch": (-3.14159, 3.14159),
        "yaw": (-3.14159, 3.14159),
    }
    cfg.events["reset_robot_joints"].params["position_range"] = (-3.14159, 3.14159)
    # 10 % of resets start near HOME instead (FRASA's reset_final_p,
    # HumanUP's standing_init_prob). Added after reset_base/reset_robot_joints
    # so it overrides their sample for the selected envs.
    cfg.events["reset_near_home"] = EventTermCfg(
        mode="reset",
        func=reset_near_home_fraction,
        params={
            "rel_near_home_envs": 0.1,
            "joint_noise_range": (-0.05, 0.05),
            "orientation_noise_range": (-0.09, 0.09),  # ~+-5 deg roll/pitch
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # No pushes. Ankle-only balance absorbs about 0.2 m/s; the walking
    # task's +-0.5 m/s pushes (tried in 0e33eb3) knocked every stand over
    # into a worse-scoring posture, so standing attempts stopped paying.
    del cfg.events["push_robot"]

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

    #---------------------------- Curriculum ------------------------
    cfg.curriculum = {}
    cfg.curriculum["staged_curriculum"] = CurriculumTermCfg(
        func=step_based_staged_curriculum,
        params={
            "stages": [
                {
                    # Stage 1 finds any way up; then keep effort within what
                    # the XC330s deliver.
                    "name": "add torque regularization",
                    "step": 5000 * 24,
                    "apply": lambda env: env.reward_manager.get_term_cfg("joint_torques_l2").__setattr__(
                        "weight", -1e-3
                    ),
                },
            ],
        },
    )
    # Pose shaping starts low and ramps only once standing is reliable
    # (standing_bonus mean episode reward >= 2.0 at weight 5.0, i.e. raw
    # >= 0.4). A from-scratch run with every pose weight at its final value
    # from step 0 plateaued (standing_bonus ~0.09 for 20k iterations).
    cfg.curriculum["pose_curriculum"] = CurriculumTermCfg(
        func=reward_based_staged_curriculum,
        params={
            "stages": [
                {
                    "name": "ramp up pose matching",
                    "reward_term_name": "standing_bonus",
                    "threshold": 2.0,
                    "apply": lambda env: (
                        env.reward_manager.get_term_cfg("standing_pose").__setattr__("weight", standing_pose_final),
                        env.reward_manager.get_term_cfg("hip_pose").__setattr__("weight", hip_pose_final),
                    ),
                },
                {
                    "name": "enable home stillness",
                    "reward_term_name": "standing_pose",
                    "threshold": 15.0,
                    "apply": lambda env: env.reward_manager.get_term_cfg("home_stillness").__setattr__(
                        "weight", 50.0
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


def _add_shared_rewards(
    cfg: ManagerBasedRlEnvCfg,
    sensors: dict[str, str],
    *,
    standing_gate: float,
    pose_gate: float,
    effort_gate: float,
) -> None:
    """Reward terms common to both reward sets (only their gates differ)."""

    # Plain raw-action rate from the base config, lighter than walking's:
    # getting up needs large motions.
    cfg.rewards["action_rate_l2"].weight = -0.02
    # dof_pos_limits stays at the base config's -1.0.

    # Dense ascent signal on the virtual head height (orientation-aware: an
    # inverted trunk puts the virtual head low). Peaked at the target
    # (clamp(1-|h/H-1|, 0, 1)**power); linear plus squared terms keep a
    # never-vanishing early gradient and still pay the last 10 % of height
    # disproportionately. power 3 alone, or 4x weights, both failed.
    cfg.rewards["head_height"] = RewardTermCfg(
        func=head_height_reward,
        weight=36.0,
        params={
            "target_height": HEAD_STANDING_HEIGHT,
            "asset_cfg": HEAD_ASSET_CFG,
            "sensor_name": sensors["feet"],
            "power": 1.0,
        },
    )
    cfg.rewards["head_height_sq"] = RewardTermCfg(
        func=head_height_reward,
        weight=20.0,
        params={
            "target_height": HEAD_STANDING_HEIGHT,
            "asset_cfg": HEAD_ASSET_CFG,
            "sensor_name": sensors["feet"],
            "power": 2.0,
        },
    )
    # Pays only once standing, scaling up to the true target above the gate,
    # so standing earlier and longer strictly accumulates more (HoST's
    # post-task idea). Weight 5.0: 4x (20.0) reproduced a std runaway.
    cfg.rewards["standing_bonus"] = RewardTermCfg(
        func=standing_bonus,
        weight=5.0,
        params={
            "height_threshold": standing_gate,
            "target_height": HEAD_STANDING_HEIGHT,
            "head_asset_cfg": HEAD_ASSET_CFG,
        },
    )
    cfg.rewards["on_feet"] = RewardTermCfg(
        func=on_feet_reward,
        weight=2.0,
        params={
            "sensor_name": sensors["feet"],
            "target_height": HEAD_STANDING_HEIGHT,
            "head_asset_cfg": HEAD_ASSET_CFG,
        },
    )
    # Let go of the ground with the forearms once about halfway up; this is
    # what ended the "propped on hands" mode on 09-25.
    cfg.rewards["hands_released"] = RewardTermCfg(
        func=hands_released_reward,
        weight=1.5,
        params={
            "height_threshold": 0.5 * HEAD_STANDING_HEIGHT,
            "sensor_name": sensors["hands"],
            "head_asset_cfg": HEAD_ASSET_CFG,
        },
    )
    # Once up, prefer the stance that costs the least holding torque (fixed
    # a wide leg-split with tilted ankles).
    cfg.rewards["standing_torque"] = RewardTermCfg(
        func=standing_torque_penalty,
        weight=-0.5,
        params={
            "height_threshold": effort_gate,
            "head_asset_cfg": HEAD_ASSET_CFG,
            "asset_cfg": SceneEntityCfg("robot", actuator_names=(DOFS_FILTER,)),
        },
    )
    # Hold the commanded target still near HOME once standing; off (0.0)
    # until pose_curriculum's stage 2.
    cfg.rewards["home_stillness"] = RewardTermCfg(
        func=home_stillness_reward,
        weight=0.0,
        params={
            "height_threshold": effort_gate,
            "target_pose_worst": 1.7,
            "target_rate_worst": 60.0,
            "head_asset_cfg": HEAD_ASSET_CFG,
            "asset_cfg": SceneEntityCfg("robot", joint_names=(DOFS_FILTER,)),
        },
    )
    # Per-joint pull toward HOME on MEASURED angles, loose std (0.6 rad): a
    # tighter std underflowed to zero reward at the ~1 rad errors a fresh
    # stander has. Starts low; pose_curriculum ramps it.
    cfg.rewards["standing_pose"] = RewardTermCfg(
        func=home_pose_reward,
        weight=6.0,
        params={
            "height_threshold": pose_gate,
            "std": {r".*": 0.6},
            "head_asset_cfg": HEAD_ASSET_CFG,
            "asset_cfg": SceneEntityCfg("robot", joint_names=(DOFS_FILTER,)),
        },
    )
    # Separate hip term so a hip still far off can't crush the whole-body
    # term's exp() for the other 16 joints.
    cfg.rewards["hip_pose"] = RewardTermCfg(
        func=home_pose_reward,
        weight=3.0,
        params={
            "height_threshold": pose_gate,
            "std": {r".*": 0.6},
            "head_asset_cfg": HEAD_ASSET_CFG,
            "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*hip_roll.*", r".*hip_pitch.*")),
        },
    )
    # Off until staged_curriculum turns it on at step 120000.
    cfg.rewards["joint_torques_l2"] = RewardTermCfg(func=joint_torques_l2, weight=0.0)
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=velocity_mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": sensors["self_collision"]},
    )


def _add_v42_rewards(cfg: ManagerBasedRlEnvCfg, sensors: dict[str, str]) -> None:
    """Exact rewards of 2026-09-25_18-47-14 (stood from scratch by ~1200 it)."""

    _add_shared_rewards(
        cfg,
        sensors,
        standing_gate=STANDING_GATE_HEIGHT,
        pose_gate=0.8 * HEAD_STANDING_HEIGHT,
        effort_gate=0.85 * HEAD_STANDING_HEIGHT,
    )
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk",)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.01


def _add_redesign_rewards(cfg: ManagerBasedRlEnvCfg, sensors: dict[str, str]) -> None:
    """v42's core, with each change tied to a measurement.

    * Pose, stillness and torque terms gate at 0.9x H (0.2646), not 0.8x
      (0.2352): kneeling upright reaches 0.226-0.239, so the 0.8x gate paid
      pose reward to a kneel.
    * No always-on body_ang_vel/angular_momentum: the first penalizes the
      trunk rotation the recovery needs, the second measured exactly 0.
    * HoST's post-standing terms (arXiv:2502.08378, its code's weights and
      shapes): exp(-5|g_xy|^2) uprightness and a calm-trunk term, hard-gated
      at the same 0.9x H so no crouch or prop can earn them.
    """

    _add_shared_rewards(
        cfg,
        sensors,
        standing_gate=STANDING_GATE_HEIGHT,
        pose_gate=STANDING_GATE_HEIGHT,
        effort_gate=STANDING_GATE_HEIGHT,
    )
    del cfg.rewards["body_ang_vel"]
    del cfg.rewards["angular_momentum"]
    cfg.rewards["upright_standing"] = RewardTermCfg(
        func=upright_balance_reward,
        weight=5.0,
        params={
            "height_threshold": STANDING_GATE_HEIGHT,
            "head_asset_cfg": HEAD_ASSET_CFG,
            "tilt_std": float(1.0 / np.sqrt(5.0)),
        },
    )
    cfg.rewards["calm_standing"] = RewardTermCfg(
        func=standing_stability_reward,
        weight=5.0,
        params={
            "height_threshold": STANDING_GATE_HEIGHT,
            "head_asset_cfg": HEAD_ASSET_CFG,
            "lin_vel_std": float(1.0 / np.sqrt(5.0)),
            "ang_vel_std": float(1.0 / np.sqrt(2.0)),
        },
    )


def _add_posture_rewards(cfg: ManagerBasedRlEnvCfg, feet_stance_weight: float) -> None:
    """Pull a standing policy's stance toward HOME: feet together, legs straight.

    For fine-tuning a policy that already stands. The first robot-bound v4
    policy stood with feet 20 cm apart and 7.5 cm staggered (HOME: 9.4 cm, 0),
    right hip yawed +56 deg, a hip_roll pinned at its 25 deg stop and ~30 deg
    of knee bend: a wide base that eases balance under IMU delay. At the
    redesign set's final pose weights (30/15, std 0.6 rad) nothing outweighed
    that. So: a direct stance-geometry term, hip_yaw added to hip_pose, and
    larger final pose weights (see _POSE_FINAL_WEIGHTS; the 09-26 lineage
    reached ~6 deg RMS from HOME at 120-480).
    """
    cfg.rewards["hip_pose"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=(r".*hip_yaw.*", r".*hip_roll.*", r".*hip_pitch.*")
    )
    cfg.rewards["feet_stance"] = RewardTermCfg(
        func=feet_stance_reward,
        weight=feet_stance_weight,
        params={
            "height_threshold": STANDING_GATE_HEIGHT,
            "head_asset_cfg": HEAD_ASSET_CFG,
            "target_lateral": HOME_FEET_LATERAL_M,
            "scale": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=("foot", "foot_2")),
        },
    )


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
        # Single critic, as every standing run used. The multi-critic trial
        # (09-29) measured its per-group normalization weighting the standing
        # rewards at 0.02-0.26x the height reward and action-rate/torque at
        # 10-17x -- favouring stillness.
        class_name="PPO",
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
