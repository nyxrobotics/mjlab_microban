# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

import os
from pathlib import Path

import mujoco
import numpy as np
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

MICROBAN_XML: Path = Path(os.path.dirname(__file__)) / "microban" / "robot.xml"
assert MICROBAN_XML.exists(), f"XML not found: {MICROBAN_XML}"

def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICROBAN_XML))

# Shared reference pose of every policy (walking, tracking, get-up): the trunk
# leans 10 deg forward (the lean of the original hip -10 deg HOME), knees are
# straight, and the hip/ankle pitches keep both soles flat with the
# mass-weighted COM over the fore-aft centre of the sole contact patches.
# Solved with MuJoCo forward kinematics on robot.xml (2026-10-04): with the
# root pitched +10 deg (nose down) the soles are flat (normal x < 1e-15), the
# COM is 30.96 mm from the heel edge and 30.96 mm from the toe edge (|offset|
# < 1e-14 m), the lowest sole corner touches z = 0 at root z 0.170430569776,
# the virtual head (trunk COM + 0.07324 m along the trunk axis) is at 0.29562
# m and the foot bodies are 0.09278 m apart laterally. Hip/ankle roll (+-5 deg)
# leaves each sole rolled 0.076 deg, as in every earlier HOME. For reference:
# the earlier hip -10 deg pose had the same lean but 23.8/38.1 mm toe/heel
# margins; the centered upright HOME (hip +1.198, ankle -1.198 deg) kept the
# trunk vertical with 30.8/30.8 mm.
HOME_TRUNK_PITCH_RAD = float(np.deg2rad(10.0))
HOME_HIP_PITCH_RAD = float(np.deg2rad(-14.166561199931119))
HOME_ANKLE_PITCH_RAD = float(np.deg2rad(4.127976841869204))
HOME_ROOT_HEIGHT_M = 0.170430569776402
# Root orientation at HOME: +10 deg about the trunk's y axis (positive pitch
# tips the trunk's x axis toward -z, i.e. leans forward), w-x-y-z.
HOME_ROOT_QUAT_WXYZ = (
    float(np.cos(HOME_TRUNK_PITCH_RAD / 2.0)),
    0.0,
    float(np.sin(HOME_TRUNK_PITCH_RAD / 2.0)),
    0.0,
)
# Every policy commands target = HOME + action on all body joints, with no
# software clip, and observes its own raw previous output. The only bound is
# the servo's own goal-position range: one turn, [-pi, pi) rad. The robot
# saturates goals there when it writes them, and training models the same
# saturation as an absolute target clip. (A +-1.57 rad clip was tried on
# 2026-10-03: it caps the XC330's torque at ~70 % of its current limit, which
# saturates at a 2.0-2.8 rad target error, and walking stopped improving at
# 16 % timeouts while the unclipped run reached 83 % in half the iterations.)
SERVO_TARGET_RANGE_RAD = float(np.pi)

HOME_FRAME = EntityCfg.InitialStateCfg(
    # The lowest sole collision corner is on the ground at this z.
    pos=(0.0, 0.0, HOME_ROOT_HEIGHT_M),
    rot=HOME_ROOT_QUAT_WXYZ,
    joint_pos={
        "head": float(np.deg2rad(0.0)),
        "neck_roll": float(np.deg2rad(0.0)),
        "neck_pitch": float(np.deg2rad(0.0)),
        "left_shoulder_roll": float(np.deg2rad(10.0)),
        "right_shoulder_roll": float(np.deg2rad(-10.0)),
        "left_shoulder_pitch": float(np.deg2rad(0.0)),
        "right_shoulder_pitch": float(np.deg2rad(0.0)),
        "left_elbow": float(np.deg2rad(-20.0)),
        "right_elbow": float(np.deg2rad(-20.0)),
        "left_hip_roll": float(np.deg2rad(5.0)),
        "right_hip_roll": float(np.deg2rad(-5.0)),
        "left_hip_pitch": HOME_HIP_PITCH_RAD,
        "right_hip_pitch": HOME_HIP_PITCH_RAD,
        "left_hip_yaw": float(np.deg2rad(0.0)),
        "right_hip_yaw": float(np.deg2rad(0.0)),
        "left_knee": float(np.deg2rad(0.0)),
        "right_knee": float(np.deg2rad(0.0)),
        "left_ankle_roll": float(np.deg2rad(-5.0)),
        "right_ankle_roll": float(np.deg2rad(5.0)),
        "left_ankle_pitch": HOME_ANKLE_PITCH_RAD,
        "right_ankle_pitch": HOME_ANKLE_PITCH_RAD,
    },
    joint_vel={r".*": 0.0},
)

FULL_COLLISION = CollisionCfg(
    geom_names_expr=(r".*_collision",),
    condim={r"^(left|right)_foot_collision_[1-6]$": 3, r".*_collision": 1},
    priority={r"^(left|right)_foot_collision_[1-6]$": 1},
    friction={r"^(left|right)_foot_collision_[1-6]$": (1.0,)},
)

import bam.actuators
from bam.mjlab import BamActuatorCfg
from bam.testbench import Pendulum

from mjlab_microban.robot.xc330_actuator import XC330Actuator

# This bam branch has no built-in XC330-T288-T definition, so register one before
# BamActuatorCfg's json_path (below) needs to look it up by the "actuator" key the
# JSON was fit with. See xc330_actuator.py for why the class exists at all.
bam.actuators.actuators["xc330"] = lambda: XC330Actuator(Pendulum)

# Updated 2026-09-22: the robot is now XC330-T288-T + 3S (was XL330-M288-T + 2S).
# json_path points at a real bam identification (tools/actuator_id/ in the main
# microban repo; 30 recordings, m6 model) instead of the bundled xl330/m6 preset.
# vin_range/vin_min follow a 3S LiPo (was 2S: 7.0-8.0V range, 6.0V min).
# max_current is XC330-T288-T's firmware current limit (was XL330's 1.75A).
# vin_drop_gain_range is UNCHANGED (still XL330-tuned): it's an empirical pack-level
# V/Nm coefficient (battery + wiring resistance across all 21 servos), not derivable
# from a single motor's R/kt, so it needs its own re-tuning/measurement pass.
actuators = BamActuatorCfg(
    json_path=str(Path(os.path.dirname(__file__)) / "xc330_params.json"),
    target_names_expr=(r".*",),
    kp_fw=125,
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

if __name__ == "__main__":
    from mjlab.scene import Scene, SceneCfg
    from mjlab.terrains import TerrainEntityCfg
    from mujoco import viewer

    SCENE_CFG = SceneCfg(
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities={"robot": MICROBAN_ROBOT_CFG},
    )

    scene = Scene(SCENE_CFG, device="cuda:0")
    model = scene.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
    viewer.launch(model, data=data)
