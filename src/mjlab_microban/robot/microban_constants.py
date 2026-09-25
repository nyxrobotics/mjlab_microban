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

HOME_FRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.168), #0.1676),
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
        "left_hip_pitch": float(np.deg2rad(-10.0)),
        "right_hip_pitch": float(np.deg2rad(-10.0)),
        "left_hip_yaw": float(np.deg2rad(0.0)),
        "right_hip_yaw": float(np.deg2rad(0.0)),
        "left_knee": float(np.deg2rad(0.0)),
        "right_knee": float(np.deg2rad(0.0)),
        "left_ankle_roll": float(np.deg2rad(-5.0)),
        "right_ankle_roll": float(np.deg2rad(5.0)),
        "left_ankle_pitch": float(np.deg2rad(0.0)),
        "right_ankle_pitch": float(np.deg2rad(0.0)),
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
