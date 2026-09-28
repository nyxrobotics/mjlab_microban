"""Shared geometry and MjLab setup for HOME experiments."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass

import mujoco
import numpy as np
import torch

import mjlab_microban.tasks  # noqa: F401  # task registration
from mjlab.tasks.registry import load_env_cfg
from mjlab_microban.robot.microban_constants import HOME_FRAME


TASK = "Mjlab-Velocity-Microban"
FOOT_BOX_NAMES = tuple(
    f"{side}_foot_collision_{index}"
    for side in ("left", "right")
    for index in range(1, 7)
)


@dataclass(frozen=True)
class Candidate:
    name: str
    hip_deg: float
    ankle_deg: float
    shoulder_deg: float
    root_z_m: float = math.nan


def parse_floats(value: str) -> list[float]:
    numbers = [float(part) for part in value.split(",")]
    if not numbers or not all(math.isfinite(number) for number in numbers):
        raise ValueError("Expected finite comma-separated numbers")
    return numbers


def parse_ints(value: str) -> list[int]:
    numbers = [int(part) for part in value.split(",")]
    if not numbers:
        raise ValueError("Expected at least one integer")
    return numbers


def contact_root_height_m(
    model: mujoco.MjModel, data: mujoco.MjData, pose: Candidate
) -> float:
    """Place the lowest corner of the twelve sole boxes at ground z=0."""
    data.qpos[:] = model.qpos0
    data.qpos[:3] = (0.0, 0.0, 0.0)
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    for name, angle in HOME_FRAME.joint_pos.items():
        data.qpos[model.joint(name).qposadr] = angle
    for side in ("left", "right"):
        for axis, degrees in (
            ("hip_pitch", pose.hip_deg),
            ("ankle_pitch", pose.ankle_deg),
            ("shoulder_pitch", pose.shoulder_deg),
        ):
            data.qpos[model.joint(f"{side}_{axis}").qposadr] = math.radians(degrees)
    mujoco.mj_forward(model, data)
    lowest = min(
        float(
            data.geom_xpos[model.geom(name).id, 2]
            - np.dot(
                np.abs(data.geom_xmat[model.geom(name).id].reshape(3, 3)[2]),
                model.geom_size[model.geom(name).id],
            )
        )
        for name in FOOT_BOX_NAMES
    )
    return -lowest


def configure_env(num_envs: int, duration_s: float):
    cfg = load_env_cfg(TASK, play=True)
    cfg.scene.num_envs = num_envs
    cfg.episode_length_s = max(20.0, duration_s + 1.0)
    cfg.curriculum = {}
    cfg.events = {name: term for name, term in cfg.events.items() if term.mode == "reset"}
    cfg.observations["actor"].enable_corruption = False
    cfg.events["reset_base"].params["pose_range"] = {
        name: (0.0, 0.0)
        for name in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    cfg.events["reset_base"].params["velocity_range"] = {}
    cfg.events["reset_robot_joints"].params["position_range"] = (0.0, 0.0)
    cfg.events["reset_robot_joints"].params["velocity_range"] = (0.0, 0.0)
    return cfg


def install_candidate_homes(env, candidates: list[Candidate]) -> None:
    """Use each candidate as both reset/observation HOME and action offset."""
    robot = env.scene["robot"]
    action = env.action_manager.get_term("joint_pos")
    joint_index = {name: index for index, name in enumerate(robot.joint_names)}
    defaults = robot.data.default_joint_pos
    for row, pose in enumerate(candidates):
        for side in ("left", "right"):
            for axis, degrees in (
                ("hip_pitch", pose.hip_deg),
                ("ankle_pitch", pose.ankle_deg),
                ("shoulder_pitch", pose.shoulder_deg),
            ):
                defaults[row, joint_index[f"{side}_{axis}"]] = math.radians(degrees)
        robot.data.default_root_state[row, 2] = pose.root_z_m
    if not isinstance(action.offset, torch.Tensor):
        raise TypeError("Walking action needs a per-environment default offset")
    action.offset.copy_(defaults[:, action._target_ids])


def write_csv(path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
