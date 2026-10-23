# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

import os
import numpy as np
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

MICROBAN_XML: Path = Path(os.path.dirname(__file__)) / "microban" / "robot.xml"
assert MICROBAN_XML.exists(), f"XML not found: {MICROBAN_XML}"

def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICROBAN_XML))

from mjlab_microban.robot.home_pose import HOME

# Shared reference pose of every policy (walking, IK tracking, get-up), loaded
# from config/home_pose.yaml (the single source of truth; see config/README.md).
# Everything below is derived from that file by MuJoCo FK
# (robot/home_pose.py): the soles are flat at the HOME trunk pitch and the
# mass-weighted COM is over the fore-aft centre of the sole contact patches.
# Every trunk-pitch-dependent term (upright reward pitch, HOME-levelled
# velocity and target frames, reset yaw axis, HOME gravity of the exporters,
# HMD neutral) reads HOME_TRUNK_PITCH_RAD and is exactly the vertical-trunk
# term at 0.
HOME_TRUNK_PITCH_RAD = HOME.trunk_pitch_rad
HOME_ROOT_POS = HOME.root_pos
# Root orientation at HOME: HOME_TRUNK_PITCH_RAD about the trunk's y axis
# (positive tips the trunk's x axis toward -z, i.e. leans forward), w-x-y-z.
HOME_ROOT_QUAT_WXYZ = HOME.root_quat_wxyz
# Unit gravity in the trunk frame while standing at HOME: (sin p, 0, -cos p).
HOME_PROJECTED_GRAVITY = HOME.projected_gravity
# Every policy commands target = HOME + action on all body joints, with no
# software clip, and observes its own raw previous output. The only bound is
# the servo's own goal-position range: one turn, [-pi, pi) rad. The robot
# saturates goals there when it writes them, and training models the same
# saturation as an absolute target clip. (A narrower clip, such as +-1.57
# rad, caps the XC330's torque: its current limit saturates only at a 2.0-2.8
# rad target error.)
SERVO_TARGET_RANGE_RAD = float(np.pi)

HOME_FRAME = EntityCfg.InitialStateCfg(
    # The lowest sole collision corner is on the ground at this z.
    pos=HOME_ROOT_POS,
    rot=HOME_ROOT_QUAT_WXYZ,
    joint_pos=dict(HOME.joint_pos_rad),
    joint_vel={r".*": 0.0},
)

FULL_COLLISION = CollisionCfg(
    geom_names_expr=(r".*_collision",),
    condim={r"^(left|right)_foot_collision_[1-6]$": 3, r".*_collision": 1},
    priority={r"^(left|right)_foot_collision_[1-6]$": 1},
    friction={r"^(left|right)_foot_collision_[1-6]$": (1.0,)},
)

import bam.actuators
from bam.testbench import Pendulum

from mjlab_microban.robot.xc330_actuator import XC330Actuator, XC330ActuatorCfg

# This bam branch has no built-in XC330-T288-T definition, so register one before
# XC330ActuatorCfg's json_path (below) needs to look it up by the "actuator" key the
# JSON was fit with. See xc330_actuator.py for why the class exists at all.
bam.actuators.actuators["xc330"] = lambda: XC330Actuator(Pendulum)

# The robot's servos are XC330-T288-T on a 3S pack.  json_path points at a
# real bam identification (tools/actuator_id/ in the microban repo, which
# carries the same file; 30 recordings, m6 model) and XC330ActuatorCfg is
# bam.mjlab's actuator with the static friction of bam's simulator (see
# xc330_actuator.py).  vin_range/vin_min follow a 3S LiPo and max_current is
# the XC330-T288-T firmware current limit.  vin_drop_gain_range is an empirical
# pack-level V/Nm coefficient (battery + wiring resistance across all 21
# servos), not derivable from a single motor's R/kt.
# Firmware position P gain of all 21 servos while a learned policy (walking,
# get-up, IK tracking) runs, in training and on the robot (the robot's KP_RL).  The
# robot holds static poses without a policy at its own, stiffer gain.
SERVO_KP_POLICY = 125

actuators = XC330ActuatorCfg(
    json_path=str(Path(os.path.dirname(__file__)) / "xc330_params.json"),
    target_names_expr=(r".*",),
    kp_fw=SERVO_KP_POLICY,
    vin_range=(9.0, 12.6),
    vin_drop_gain_range=(0.0, 0.2),
    vin_min=9.0,
    max_current=0.91,
    delay_min_lag=3,
    delay_max_lag=6,
)

# -- Old actuator (XML position, MuJoCo default) --
# actuators = XmlActuatorCfg(
#     target_names_expr=(r".*",),
#     delay_min_lag=0,
#     delay_max_lag=3,
# )

MICROBAN_ROBOT_CFG = EntityCfg(
    spec_fn=get_spec,
    init_state=HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

# Training's floors and servo-goal delay, for every learned policy (walking,
# IK tracking, get-up, whole-body).  Evaluation and play keep MICROBAN_ROBOT_CFG:
# MuJoCo's 0.02 s contact time constant (a hard floor), the sole friction of
# the foot_friction event and the 3-6-step goal delay redrawn every step.
# Each environment draws one floor: one contact time constant (solref[0])
# log-uniformly from 0.02 s to 0.1 s (standing at HOME, the most loaded sole
# corner sinks 0.06 mm at 0.02 s and 0.77 mm at 0.1 s, a hard floor to a
# carpet or an exercise mat) and one friction from 0.15 (hard plastic on a
# polished floor) to 1.5 (a rubber mat), both on every collision geom
# (tasks/mdp.py training_floor_and_delay; the whole-body task's module
# docstring has the measurements).  The servo goal follows the policy by one
# delay per episode of 3-8 physics steps of 5 ms: the robot's goal reaches its
# servos about 15 ms (at most 20 ms) after the observation is read, and the
# refitted servo model leaves out the servo's own delay of about 7 ms, so the
# goal acts about 22-28 ms after the observation.  (When any environment has
# reset in the previous step, nearly every step in training, mjlab's post-reset
# write adds one goal to every environment's delay buffer, so the longer draws
# delay the goal by one step less.)
FLOOR_TIME_CONSTANT_S = (0.02, 0.1)
FLOOR_FRICTION = (0.15, 1.5)
COMMAND_DELAY_STEPS = (3, 8)  # physics steps of 5 ms, as drawn


def collision_named_spec() -> mujoco.MjSpec:
    """robot.xml with every collision geom named ``<body>_collision_<n>`` (the soles keep their names)."""
    spec = get_spec()
    counts: dict[str, int] = defaultdict(int)
    for geom in spec.geoms:
        if geom.classname.name == "collision" and not geom.name:
            counts[geom.parent.name] += 1
            geom.name = f"{geom.parent.name}_collision_{counts[geom.parent.name]}"
    return spec


# Training's collision geoms: robot.xml's (condim 3, friction 1, MuJoCo's
# solref, as in MICROBAN_ROBOT_CFG), each with priority over the floor, so a
# floor contact takes the geom's friction and solref, which training draws.
# (In MICROBAN_ROBOT_CFG only the soles have priority; the other geoms mix
# with the floor's equal values, so its contacts are the same.)
TRAINING_COLLISION = CollisionCfg(geom_names_expr=(r".*_collision_\d+$",), priority=1)


def with_training_goal_delay(robot: EntityCfg) -> EntityCfg:
    """``robot`` whose servo goal is delayed by COMMAND_DELAY_STEPS, drawn when an episode starts (a reset clears
    the delay buffer) and kept to its end (no redraw period is reached)."""
    return replace(robot, articulation=replace(robot.articulation, actuators=tuple(
        replace(actuator, delay_min_lag=COMMAND_DELAY_STEPS[0], delay_max_lag=COMMAND_DELAY_STEPS[1],
                delay_hold_prob=0.0, delay_update_period=1 << 40, delay_per_env_phase=False)
        for actuator in robot.articulation.actuators
    )))


# Training's robot (walking, IK tracking, get-up): MICROBAN_ROBOT_CFG with
# TRAINING_COLLISION and the training goal delay.
MICROBAN_TRAIN_ROBOT_CFG = with_training_goal_delay(
    replace(MICROBAN_ROBOT_CFG, spec_fn=collision_named_spec, collisions=(TRAINING_COLLISION,))
)

if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.scene import Scene, SceneCfg
    from mjlab.terrains import TerrainEntityCfg

    SCENE_CFG = SceneCfg(
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities={"robot": MICROBAN_ROBOT_CFG},
    )

    scene = Scene(SCENE_CFG, device="cuda:0")
    model = scene.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
    viewer.launch(model, data=data)