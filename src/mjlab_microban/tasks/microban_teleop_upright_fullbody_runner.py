"""Checkpoint identity for the independent upright full-body training task."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from mjlab.rl.runner import MjlabOnPolicyRunner

from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_HMD_JOINT_NAMES,
    MICROBAN_TELEOP_ACTION_JOINT_NAMES,
)
from mjlab_microban.tasks.microban_teleop_upright_fullbody_env_cfg import (
    MICROBAN_TELEOP_UPRIGHT_FULLBODY_HOME_REVISION,
    MICROBAN_TELEOP_UPRIGHT_FULLBODY_RECIPE_REVISION,
    MICROBAN_TELEOP_UPRIGHT_FULLBODY_TASK_ID,
    make_microban_teleop_upright_fullbody_env_cfg,
)

UPRIGHT_FULLBODY_INFO_KEY = "microban_teleop_upright_fullbody_identity"
UPRIGHT_FULLBODY_INFO_SCHEMA_VERSION = 1


def upright_fullbody_checkpoint_identity() -> dict[str, Any]:
    """Bind the run to its recipe and actual 21-joint training init state."""

    joint_names = (*MICROBAN_HMD_JOINT_NAMES, *MICROBAN_TELEOP_ACTION_JOINT_NAMES)
    init_state = make_microban_teleop_upright_fullbody_env_cfg().scene.entities[
        "robot"
    ].init_state
    defaults = init_state.joint_pos
    if (
        len(joint_names) != 21
        or len(set(joint_names)) != 21
        or not isinstance(defaults, Mapping)
        or set(defaults) != set(joint_names)
    ):
        raise ValueError("Upright full-body HOME joint inventory drifted")
    joint_pos = [float(defaults[name]) for name in joint_names]
    root_pos = [float(value) for value in init_state.pos]
    root_quat = [float(value) for value in init_state.rot]
    if (
        len(root_pos) != 3
        or len(root_quat) != 4
        or not all(math.isfinite(value) for value in (*joint_pos, *root_pos, *root_quat))
        or not math.isclose(
            sum(value * value for value in root_quat), 1.0, abs_tol=1.0e-6
        )
    ):
        raise ValueError("Upright full-body HOME contains invalid values")
    return {
        "schema_version": UPRIGHT_FULLBODY_INFO_SCHEMA_VERSION,
        "task": MICROBAN_TELEOP_UPRIGHT_FULLBODY_TASK_ID,
        "recipe_revision": MICROBAN_TELEOP_UPRIGHT_FULLBODY_RECIPE_REVISION,
        "home_revision": MICROBAN_TELEOP_UPRIGHT_FULLBODY_HOME_REVISION,
        "root_pos_xyz_m": root_pos,
        "root_quat_wxyz": root_quat,
        "joint_names": list(joint_names),
        "joint_pos_rad": joint_pos,
    }


class MicrobanTeleopUprightFullbodyOnPolicyRunner(MjlabOnPolicyRunner):
    """Reject another recipe or HOME before loading its actor and optimizer."""

    def save(self, path: str, infos: dict | None = None) -> None:
        if infos is not None and not isinstance(infos, dict):
            raise TypeError("Upright full-body checkpoint infos must be a dict")
        expected = upright_fullbody_checkpoint_identity()
        if infos is not None and (
            UPRIGHT_FULLBODY_INFO_KEY in infos
            and infos[UPRIGHT_FULLBODY_INFO_KEY] != expected
        ):
            raise ValueError("Upright full-body checkpoint identity was overridden")
        super().save(path, {**(infos or {}), UPRIGHT_FULLBODY_INFO_KEY: expected})

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError("Upright full-body checkpoint payload is malformed")
        infos = payload.get("infos")
        if not isinstance(infos, dict) or infos.get(
            UPRIGHT_FULLBODY_INFO_KEY
        ) != upright_fullbody_checkpoint_identity():
            raise ValueError("Checkpoint recipe or HOME differs from upright full-body")
        del payload
        return super().load(
            path, load_cfg=load_cfg, strict=strict, map_location=map_location
        )
