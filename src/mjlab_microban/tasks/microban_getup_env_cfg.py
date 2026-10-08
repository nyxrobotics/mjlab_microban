# Copyright 2026 nyxrobotics

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Microban get-up (fall recovery) environment.

Separate task and policy from microban_velocity_env_cfg: getting up from an
arbitrary fallen pose has a different contact/trajectory structure than
walking. The policy controls 18 body joints; head and both neck joints are
held at their measured angles (see microban_getup_action.py).

Action contract: target = HOME + raw action on the 18 body joints, bounded
only by the servo's one-turn goal range (+-pi), with the policy's RAW previous
output as its previous-action observation.  At the robot's RL gain (P=125) one
XC330 gives only ~0.42 Nm/rad, while gravity tips the standing robot at
~1.65 Nm/rad about the ankles.  No pose holds statically (HOME falls in
~1-2 s): standing is active balance that needs targets far past the joint
angle to get useful torque (hip_roll at HOME: 0.13-0.20 Nm inside a per-joint
soft-limit clip vs 0.6-0.7 Nm with +-1.57), so a per-joint clip or a
clip-excess penalty would remove exactly that authority.

The HOME may lean the trunk forward (config/home_pose.yaml trunk_pitch_deg).
Every term that assumes a vertical trunk is centred on the HOME lean instead
(upright_standing's target gravity, the head height, the near-HOME reset's
yaw axis); at a vertical-trunk HOME they are the vertical terms.

One training run (GETUP_SCHEDULE).  The HOME-stance reward set (every standing
term gated on standing height) is active from the first update; the
step-scheduled curriculum (tasks/curriculum.py) then switches on, in one
process:

* update 2500: the walking task's 0-3 tick simulated IMU latency (the actor's
  gyro/gravity delay buffers are allocated for 3 ticks from the start and
  held at 0 until here);
* update 4000 ("refine"): the calm terms (measured joint velocity while
  standing, roll joints near HOME, wider-stance penalty x3, a light
  raw-target clip barrier); the runner resets the action std to 0.5, the
  Adam moments, the learning rate and sets entropy 0.001 at the same update;
* update 10000 ("effort"): shoulder roll joins the roll pose term, the
  target-vs-measured effort penalty and a 10x clip barrier;
* update 15000 ("push"): +-0.3 m/s pushes every 3-6 s, to the end (18000).

The pushes come after the effort stage: with pushes from the effort switch
on, one arm stays pressed into its shoulder_roll stop (target on the +-pi
clip almost all the standing time), while a push-free effort stage lets both
arms go 3000-4500 updates in.

The reward-based pose curriculum (standing_bonus >= 2 raises the pose
weights to 240/120, standing_pose >= 15 enables the HOME stillness term) runs
alongside.  The play/evaluation config keeps the full IMU latency and no
schedule (docs/getup_training_export.md).
"""

import numpy as np
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
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

from mjlab_microban.robot.home_pose import HOME
from mjlab_microban.robot.microban_constants import (
    MICROBAN_ROBOT_CFG,
    HOME_TRUNK_PITCH_RAD,
    SERVO_TARGET_RANGE_RAD,
)
from mjlab_microban.tasks.curriculum import Setting, Stage, StagedCurriculum
from mjlab_microban.schedules import GETUP_SCHEDULE, GETUP_TOTAL_UPDATES
from mjlab_microban.tasks.mdp import (
    reward_based_staged_curriculum,
    head_height_reward,
    home_pose_reward,
    standing_bonus,
    feet_stance_reward,
    standing_joint_vel_l2,
    standing_target_error_l1,
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
from mjlab_microban.tasks.microban_teleop_mdp import normalized_target_clip_excess_l1_sum

GETUP_EPISODE_LENGTH_S = 20.0  # Match the robot's automatic get-up timeout.
# Servo goal range on all 18 body joints (see module docstring).
GETUP_ACTION_CLIP_RAD = SERVO_TARGET_RANGE_RAD
GETUP_ACTION_CLIP = {r".*": (-GETUP_ACTION_CLIP_RAD, GETUP_ACTION_CLIP_RAD)}
# Update at which each scheduled switch happens (mjlab_microban/schedules.py; see the
# module docstring), and the length of the run.  The runner reads "refine" too.
GETUP_MAX_ITERATIONS = GETUP_TOTAL_UPDATES
GETUP_IMU_DELAY_MAX_LAG = 3
GETUP_REFINE_ACTION_STD = 0.5
GETUP_REFINE_ENTROPY_COEF = 0.001
# Lateral distance between the two foot bodies at HOME, by forward kinematics
# of config/home_pose.yaml (robot/home_pose.py; 0.1 mm rounding).
HOME_FEET_LATERAL_M = HOME.feet_lateral_m

# Virtual head height (trunk COM + 0.07324 m along the trunk's up axis, see
# mdp._head_height) at HOME by MuJoCo forward kinematics, rounded to 0.1 mm
# (robot/home_pose.py): 0.2965 m with a vertical trunk (FK 0.29653), 0.2953 m
# at the forward-lean HOME (FK 0.29527; the 10 deg lean lowers it). Kneeling
# upright reaches 0.226-0.239 and the deepest flat-foot squat 0.222-0.227,
# so the 0.9x standing gate (0.2669 / 0.2657) is above both.  A 0.260 target
# lets a forearm-propped tripod (0.205) earn 91 % of the height reward.
HEAD_STANDING_HEIGHT = HOME.head_standing_height_m
STANDING_GATE_HEIGHT = 0.9 * HEAD_STANDING_HEIGHT
# (fraction of resets near HOME, max roll/pitch noise in rad): 10 % of resets
# within +-5 deg of HOME.  A wide (0.2, 0.6) "tipping" reset trains a braced,
# staggered stance.
NEAR_HOME_RESET = (0.1, 0.09)
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
    # Tangled fallen poses generate far more contacts than walking; the
    # auto-sized njmax overflowed (~700-800 needed). nconmax stays auto:
    # setting it explicitly blew GPU memory.
    njmax=1200,
    # The self_collision sensor (trunk subtree vs itself) exceeded the
    # default 64 matches; mjlab's own G1/Go1 configs use 500 too.
    contact_sensor_maxmatch=500,
)


_IMU_TERMS = ("actor/base_ang_vel", "actor/projected_gravity")

# The step-scheduled switches of the one training run (module docstring).
GETUP_STAGES = (
    Stage(
        "start without IMU latency",
        0,
        tuple(Setting("observation", term, "delay_max_lag", 0) for term in _IMU_TERMS),
    ),
    Stage(
        "imu_delay",
        GETUP_SCHEDULE["imu_delay"],
        tuple(
            Setting("observation", term, "delay_max_lag", GETUP_IMU_DELAY_MAX_LAG)
            for term in _IMU_TERMS
        ),
    ),
    Stage(
        "refine",
        GETUP_SCHEDULE["refine"],
        (
            Setting("reward", "standing_joint_vel", "weight", -4.0),
            Setting("reward", "raw_target_clip_excess", "weight", -0.2),
            Setting("reward", "feet_lateral", "weight", 30.0),
            Setting("reward", "roll_pose", "weight", 60.0),
        ),
    ),
    Stage(
        "effort",
        GETUP_SCHEDULE["effort"],
        (
            Setting("reward", "roll_pose", "weight", 0.0),
            Setting("reward", "roll_pose_shoulder", "weight", 60.0),
            Setting("reward", "standing_target_error", "weight", -2.0),
            Setting("reward", "raw_target_clip_excess", "weight", -2.0),
        ),
    ),
    Stage(
        "push",
        GETUP_SCHEDULE["push"],
        (Setting("event", "push_robot", "params.velocity_range", {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}),),
    ),
)


def make_microban_getup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
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
    # 18 body joints; head/neck are held at their measured angles (the
    # get-up action), every servo at the shared policy gain.
    # Absolute target = default pose + raw action, saturated at the servo's
    # +-pi goal range. That bound also removes early exploration's
    # multi-radian tail (which blew up the solver when unbounded); it is
    # deliberately NOT each joint's soft limit -- see the module docstring.
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
    # IMU terms keep the base config's noise (gyro +-0.2, gravity +-0.05) and
    # carry the walking task's simulated IMU latency (0..3 policy ticks,
    # resampled every 64 steps).  Training holds it at 0 until the imu_delay
    # switch (a policy trained without it stands far less often under 0-3
    # ticks; one that learns with it from the first update stands far later);
    # evaluation and the robot always have it.
    for name in ("base_ang_vel", "projected_gravity"):
        term = deepcopy(cfg.observations["actor"].terms[name])
        term.delay_min_lag = 0
        term.delay_max_lag = GETUP_IMU_DELAY_MAX_LAG
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
    _add_redesign_rewards(cfg, sensors)
    _add_posture_rewards(cfg)
    _add_calm_rewards(cfg)
    standing_pose_final, hip_pose_final = 240.0, 120.0

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
    # A fraction of resets start near HOME instead (FRASA's reset_final_p,
    # HumanUP's standing_init_prob). Added after reset_base/reset_robot_joints
    # so it overrides their sample for the selected envs (NEAR_HOME_RESET:
    # 10 %, +-5 deg).
    cfg.events["reset_near_home"] = EventTermCfg(
        mode="reset",
        func=reset_near_home_fraction,
        params={
            "rel_near_home_envs": NEAR_HOME_RESET[0],
            "joint_noise_range": (-0.05, 0.05),
            "orientation_noise_range": (-NEAR_HOME_RESET[1], NEAR_HOME_RESET[1]),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # Modest horizontal kicks every 3-6 s, off (zero range) until the
    # push switch.  Ankle-only balance absorbs about 0.2 m/s; the
    # walking task's +-0.5 m/s pushes knock every stand over and break
    # learning from scratch.
    cfg.events["push_robot"] = EventTermCfg(
        mode="interval",
        func=envs_mdp.push_by_setting_velocity,
        interval_range_s=(3.0, 6.0),
        params={"velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
    )

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
    cfg.curriculum = {
        "staged_curriculum": CurriculumTermCfg(
            func=StagedCurriculum, params={"stages": GETUP_STAGES}
        )
    }
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
    # Let go of the ground with the forearms once about halfway up (otherwise
    # the policy stays propped on its hands).
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
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=velocity_mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": sensors["self_collision"]},
    )


def _add_redesign_rewards(cfg: ManagerBasedRlEnvCfg, sensors: dict[str, str]) -> None:
    """The from-scratch core reward set, with each choice tied to a measurement.

    * Pose, stillness and torque terms gate at 0.9x H (0.2646), not 0.8x
      (0.2352): kneeling upright reaches 0.226-0.239, so the 0.8x gate paid
      pose reward to a kneel.
    * No always-on body_ang_vel/angular_momentum: the first penalizes the
      trunk rotation the recovery needs, the second measured exactly 0.
    * HoST's post-standing terms (arXiv:2502.08378, its code's weights and
      shapes): exp(-5|g_xy - g_xy(HOME)|^2) uprightness -- centred on the
      projected gravity of the HOME trunk lean p, (sin p, 0), which is
      HoST's exp(-5|g_xy|^2) for a vertical-trunk HOME -- and a calm-trunk
      term, hard-gated at the same 0.9x H so no crouch or prop can earn them.
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
            # A pitched-trunk HOME: peak at its lean, not at a vertical trunk.
            **({"pitch": HOME_TRUNK_PITCH_RAD} if HOME_TRUNK_PITCH_RAD != 0.0 else {}),
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


def _add_posture_rewards(cfg: ManagerBasedRlEnvCfg) -> None:
    """HOME stance: feet side by side at HOME's width, legs straight.

    On top of _add_redesign_rewards. Active from step 0 (every term is
    gated on standing height): the stance forms when standing first
    emerges and fine-tuning does not change it afterwards (without these
    terms the policy keeps a braced stance: feet 12-20 cm apart and 7-13 cm
    staggered, an ankle and a hip_roll pressed on their stops).  Trained from
    scratch with these terms it stands by ~2000 iterations with feet 10.1 cm
    apart, 0.2 cm stagger, every leg joint within 3.5 deg of HOME, and takes
    fore-aft kicks of 0.2 m/s without falling (0/63; the braced stance: 40/61
    fell).

    * feet_lateral / feet_fore_aft: foot separation in the trunk heading
      frame, separate terms because a single summed error was satisfied by
      narrowing the stance while staggering the feet.
    * hip_pose also covers hip_yaw (the braced stance yawed a hip +56 deg).
    * Final pose weights 240/120 (see _POSE_FINAL_WEIGHTS).
    """
    feet_cfg = SceneEntityCfg("robot", body_names=("foot", "foot_2"))
    cfg.rewards["hip_pose"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=(r".*hip_yaw.*", r".*hip_roll.*", r".*hip_pitch.*")
    )
    for name, axis, target, weight in (
        ("feet_lateral", "lateral", HOME_FEET_LATERAL_M, 10.0),
        ("feet_fore_aft", "fore_aft", 0.0, 20.0),
    ):
        cfg.rewards[name] = RewardTermCfg(
            func=feet_stance_reward,
            weight=weight,
            params={
                "height_threshold": STANDING_GATE_HEIGHT,
                "head_asset_cfg": HEAD_ASSET_CFG,
                "axis": axis,
                "target": target,
                "scale": 0.05,
                "asset_cfg": feet_cfg,
            },
        )


def _add_calm_rewards(cfg: ManagerBasedRlEnvCfg) -> None:
    """The calm (refine) and effort terms, registered at weight 0.

    GETUP_STAGES sets their weights: refine at update 4000, effort at
    10000 (mjlab skips a zero-weight term, so they cost nothing before).

    refine. Without it the standing policy holds its stance with bang-bang
    targets on the clip under a large action std and trembles, although a
    linear ankle PD holds the same stance under the same IMU delay. A
    commanded-target-rate penalty does not calm it, so the MEASURED joint
    velocity is penalized while standing. On its own that calms the stance
    by splaying it, so feet_lateral is tripled and a tight (8.6 deg)
    roll_pose holds the roll joints near HOME. A light barrier penalizes raw
    output beyond the flat clip.

    effort. The calm policy presses an arm into the 0-deg shoulder_roll stop
    while standing. Its target sits on the clip and its measured angle cannot
    move, so a pose term has no gradient. So: shoulder_roll joins the roll
    pose (roll_pose_shoulder replaces roll_pose: one exp-mean term over the
    larger joint set), the target-vs-measured error (the P-term effort) is
    penalized, and the clip barrier is raised 10x so the raw output comes
    back to where the target responds. The effort penalty costs push
    tolerance; the pushes start later, at the push switch (with pushes from
    this switch on, an arm stays pressed into its stop).
    """
    gated = {"height_threshold": STANDING_GATE_HEIGHT, "head_asset_cfg": HEAD_ASSET_CFG}
    cfg.rewards["standing_joint_vel"] = RewardTermCfg(
        func=standing_joint_vel_l2,
        weight=0.0,
        params={**gated, "asset_cfg": SceneEntityCfg("robot", joint_names=(DOFS_FILTER,))},
    )
    # Brings raw output back to the clip edge, where a small change moves
    # the target (it may still sit at the edge for full torque).
    cfg.rewards["raw_target_clip_excess"] = RewardTermCfg(
        func=normalized_target_clip_excess_l1_sum,
        weight=0.0,
        params={"action_name": "joint_pos"},
    )
    roll_joints = (r".*hip_roll.*", r".*ankle_roll.*")
    for name, joints in (
        ("roll_pose", roll_joints),
        ("roll_pose_shoulder", roll_joints + (r".*shoulder_roll.*",)),
    ):
        cfg.rewards[name] = RewardTermCfg(
            func=home_pose_reward,
            weight=0.0,
            params={**gated, "std": {r".*": 0.15}, "asset_cfg": SceneEntityCfg("robot", joint_names=joints)},
        )
    cfg.rewards["standing_target_error"] = RewardTermCfg(
        func=standing_target_error_l1,
        weight=0.0,
        params={**gated, "asset_cfg": SceneEntityCfg("robot", joint_names=(DOFS_FILTER,))},
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
        # Single critic: a multi-critic's per-group normalization weights the
        # standing rewards at 0.02-0.26x the height reward and
        # action-rate/torque at 10-17x -- favouring stillness.
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
    max_iterations=GETUP_MAX_ITERATIONS,
)
