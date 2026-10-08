# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

import os
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

# Shared reference pose of every policy, loaded
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
from bam.mjlab import BamActuatorCfg
from bam.testbench import Pendulum

from mjlab_microban.robot.xc330_actuator import XC330Actuator

# This bam branch has no built-in XC330-T288-T definition, so register one before
# BamActuatorCfg's json_path (below) needs to look it up by the "actuator" key the
# JSON was fit with. See xc330_actuator.py for why the class exists at all.
bam.actuators.actuators["xc330"] = lambda: XC330Actuator(Pendulum)

# The robot's servos are XC330-T288-T on a 3S pack.  json_path points at a
# real bam identification (tools/actuator_id/ in the microban repo; 30
# recordings, m6 model).  vin_range/vin_min follow a 3S LiPo and max_current is
# the XC330-T288-T firmware current limit.  vin_drop_gain_range is an empirical
# pack-level V/Nm coefficient (battery + wiring resistance across all 21
# servos), not derivable from a single motor's R/kt.
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