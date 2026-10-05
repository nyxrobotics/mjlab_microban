"""Checkpoint provenance for the get-up policy's physical action contract."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
import yaml

from mjlab.envs.mdp.observations import builtin_sensor
from mjlab.rl.runner import MjlabOnPolicyRunner

from mjlab_microban.robot import home_contracts
from mjlab_microban.robot.microban_constants import HOME_FRAME, SERVO_TARGET_RANGE_RAD
from mjlab_microban.tasks.microban_getup_action import (
    GetupJointPositionAction,
    raw_getup_action,
)
from mjlab_microban.tasks.microban_getup_actuator import (
    GETUP_BODY_KP_FW,
    GetupBamActuatorCfg,
)
from mjlab_microban.tasks.microban_getup_env_cfg import (
    GETUP_ACTION_CLIP,
    GETUP_EPISODE_LENGTH_S,
)


# The contract string is HOME-bound (robot/home_contracts.py):
# v5 (2026-10-03): target = centered HOME + raw action on the 18 body joints,
# bounded only by the servo's one-turn goal range (an absolute +-pi clip),
# the policy's RAW previous output as previous-action feedback, neck held.
# v6 (2026-10-04): the same rule at the forward-lean HOME (trunk 10 deg
# forward). Any other HOME: "v5_<tag>" (vertical trunk) or "v6_<tag>".
# v4 was the same rule with a flat +-1.57 rad clip -- the contract every
# standing get-up policy up to 2026-10-02 was trained under (old HOME, see
# microban_getup_env_cfg.py's module docstring). v3 (per-joint soft-limit
# clip, post-clip feedback) never produced a stand; its previous-action
# input meant the applied target, so v3 checkpoints/ONNX are incompatible.
GETUP_CONTRACT_VERSION = home_contracts.GETUP_CONTRACT_VERSION
# Centered HOME only: runs started on 2026-10-03 before the v5 bump (e.g.
# 2026-10-03_13-40-39_chome_servo_s1) already trained under v5 but stamped
# "v4". A "v4" stamp is therefore accepted as v5 only when the run's recorded
# params/env.yaml proves the +-pi clip at the centered HOME (see
# require_getup_checkpoint_contract); every other v4 checkpoint is rejected.
# None at every other HOME: only the current contract's stamp is accepted.
GETUP_LEGACY_STAMP = home_contracts.GETUP_LEGACY_STAMP
GETUP_ANGULAR_VELOCITY_FRAME = "imu_sensor_xyz"


def getup_home_pose() -> dict[str, object]:
    """Return the complete training HOME in a checkpoint-safe form."""

    joints = {name: float(value) for name, value in sorted(HOME_FRAME.joint_pos.items())}
    if len(joints) != 21:
        raise ValueError("Get-up HOME must define all 21 Microban joints")
    return {
        "root_pos_m": [float(value) for value in HOME_FRAME.pos],
        "root_quat_wxyz": [float(value) for value in HOME_FRAME.rot],
        "joint_pos_rad": joints,
    }


# Stamped HOME values are FK-derived (root z) or degree->radian conversions;
# 1e-9 (rad / m) absorbs last-digit FK noise of another MuJoCo build while any
# real HOME edit (>= 1e-12 deg in the YAML changes the hash, and a different
# pose moves these values by far more) still mismatches through the joint names
# or the HOME tag strings checked elsewhere.
HOME_STAMP_TOLERANCE = 1.0e-9


def home_pose_stamps_match(stamp: object, expected: object, tol: float = HOME_STAMP_TOLERANCE) -> bool:
    """Structural equality of two HOME stamps with a float tolerance."""

    if isinstance(expected, Mapping):
        return (
            isinstance(stamp, Mapping)
            and set(stamp) == set(expected)
            and all(home_pose_stamps_match(stamp[key], expected[key], tol) for key in expected)
        )
    if isinstance(expected, (list, tuple)):
        return (
            isinstance(stamp, (list, tuple))
            and len(stamp) == len(expected)
            and all(home_pose_stamps_match(a, b, tol) for a, b in zip(stamp, expected))
        )
    if isinstance(expected, bool) or isinstance(stamp, bool):
        return stamp is expected
    if isinstance(expected, (int, float)):
        return (
            isinstance(stamp, (int, float))
            and math.isfinite(float(stamp))
            and abs(float(stamp) - float(expected)) <= tol
        )
    return stamp == expected


def require_current_getup_home_pose(infos: dict) -> None:
    if not home_pose_stamps_match(infos.get("microban_getup_home_pose"), getup_home_pose()):
        raise ValueError(
            "Checkpoint has a different or unknown get-up HOME pose; "
            "train from scratch with the current task"
        )


class _RecordedConfigLoader(yaml.SafeLoader):
    """Reads mjlab's params/env.yaml without executing its python tags."""


def _construct_python_tag(loader: yaml.SafeLoader, suffix: str, node: yaml.Node) -> object:
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    return suffix


_RecordedConfigLoader.add_multi_constructor("tag:yaml.org,2002:python/", _construct_python_tag)

_RAW_GETUP_ACTION_FUNC = "name:mjlab_microban.tasks.microban_getup_action.raw_getup_action"


def recorded_env_path(checkpoint: Path) -> Path:
    """The params/env.yaml mjlab recorded next to a run's checkpoints."""

    return Path(checkpoint).resolve().parent / "params" / "env.yaml"


def require_recorded_getup_env(env_yaml: Path) -> None:
    """Refuse a run whose recorded HOME, target rule or feedback is not current.

    Decides from what the run actually trained with, not from the contract
    string its runner stamped.
    """

    if not env_yaml.is_file():
        raise ValueError(f"Run has no recorded env config: {env_yaml}")
    recorded = yaml.load(env_yaml.read_text(), Loader=_RecordedConfigLoader)
    try:
        init_state = recorded["scene"]["entities"]["robot"]["init_state"]
        action = recorded["actions"]["joint_pos"]
        feedback = {
            group: recorded["observations"][group]["terms"]["actions"]
            for group in ("actor", "critic")
        }
    except (KeyError, TypeError) as error:
        raise ValueError(f"Recorded env config lacks {error}") from error
    joints = init_state.get("joint_pos")
    if not isinstance(joints, dict) or set(joints) != set(HOME_FRAME.joint_pos) or any(
        abs(float(joints[name]) - value) > 1e-12 for name, value in HOME_FRAME.joint_pos.items()
    ):
        raise ValueError("Run was not trained from the current HOME joint pose")
    for key, expected in (("pos", HOME_FRAME.pos), ("rot", HOME_FRAME.rot)):
        if not np.allclose(np.asarray(init_state[key], dtype=np.float64), expected, rtol=0, atol=1e-9):
            raise ValueError(f"Run was not trained from the current HOME root {key}")
    clip = action.get("clip")
    if not isinstance(clip, dict) or list(clip) != [".*"] or [float(v) for v in clip[".*"]] != [
        -SERVO_TARGET_RANGE_RAD,
        SERVO_TARGET_RANGE_RAD,
    ]:
        raise ValueError(
            f"Run was not trained with the servo goal range (+-pi) as its target clip: {clip}"
        )
    if (
        float(action.get("scale")) != 1.0
        or float(action.get("offset")) != 0.0
        or action.get("use_default_offset") is not True
    ):
        raise ValueError("Run action is not target = HOME + raw_action * 1.0")
    for group, term in feedback.items():
        if (
            term.get("func") != _RAW_GETUP_ACTION_FUNC
            or term.get("clip") is not None
            or term.get("scale") is not None
            or term.get("delay_max_lag") != 0
        ):
            raise ValueError(f"Run did not observe the undelayed raw previous action ({group})")


# Historical name (contract v5 at the centered HOME).
require_recorded_getup_v5_env = require_recorded_getup_env


def require_getup_checkpoint_contract(
    checkpoint: Path, infos: object, *, require_recorded_env: bool
) -> str:
    """Validate a checkpoint's get-up markers; return the contract it stamped.

    A GETUP_CONTRACT_VERSION stamp is trusted (the runner refused anything
    else at start-up), and its recorded env is checked too when
    ``require_recorded_env``.  At the centered HOME a "v4" stamp
    (GETUP_LEGACY_STAMP) is accepted only when the run's recorded env proves v5.
    """

    if not isinstance(infos, dict):
        raise ValueError("Checkpoint lacks get-up training metadata; start a fresh run")
    if infos.get("microban_getup_angular_velocity_frame") != GETUP_ANGULAR_VELOCITY_FRAME:
        raise ValueError("Checkpoint lacks the get-up IMU-frame marker; start a fresh run")
    require_current_getup_home_pose(infos)
    stamp = infos.get("microban_getup_contract")
    if stamp == GETUP_CONTRACT_VERSION:
        if require_recorded_env:
            require_recorded_getup_env(recorded_env_path(checkpoint))
    elif GETUP_LEGACY_STAMP is not None and stamp == GETUP_LEGACY_STAMP:
        try:
            require_recorded_getup_env(recorded_env_path(checkpoint))
        except ValueError as error:
            raise ValueError(
                f"Checkpoint is a get-up v4 (+-1.57 clip) checkpoint, not {GETUP_CONTRACT_VERSION}: {error}"
            ) from error
    else:
        raise ValueError(
            f"Checkpoint has get-up contract {stamp!r}, not {GETUP_CONTRACT_VERSION}; start a fresh run"
        )
    return stamp


class MicrobanGetupOnPolicyRunner(MjlabOnPolicyRunner):
    """Stamp compatible checkpoints and reject old get-up resumes."""

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        unwrapped = env.unwrapped
        action = unwrapped.action_manager.get_term("joint_pos")
        if not isinstance(action, GetupJointPositionAction):
            raise ValueError(f"Get-up {GETUP_CONTRACT_VERSION} requires the get-up joint-position action")
        if env.clip_actions is not None or env.num_actions != 18:
            raise ValueError(f"Get-up {GETUP_CONTRACT_VERSION} requires 18 actions without wrapper clipping")
        if not math.isclose(unwrapped.step_dt, 0.02, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError(f"Get-up {GETUP_CONTRACT_VERSION} requires a 20 ms policy step")
        if unwrapped.cfg.episode_length_s != GETUP_EPISODE_LENGTH_S:
            raise ValueError(f"Get-up {GETUP_CONTRACT_VERSION} requires a 20-second training episode")
        if action.cfg.clip != dict(GETUP_ACTION_CLIP):
            raise ValueError(f"Get-up {GETUP_CONTRACT_VERSION} requires the servo-range (+-pi) absolute target clip")
        if action.cfg.scale != 1.0 or action.cfg.offset != 0.0 or not action.cfg.use_default_offset:
            raise ValueError(f"Get-up {GETUP_CONTRACT_VERSION} requires unit-scale default-relative actions")
        gyro_term = unwrapped.cfg.observations["actor"].terms["base_ang_vel"]
        if gyro_term.func is not builtin_sensor or gyro_term.params != {"sensor_name": "robot/imu_ang_vel"}:
            raise ValueError(f"Get-up {GETUP_CONTRACT_VERSION} requires raw IMU-sensor-frame angular velocity")
        for group in ("actor", "critic"):
            term = unwrapped.cfg.observations[group].terms["actions"]
            if term.func is not raw_getup_action:
                raise ValueError(f"Get-up {GETUP_CONTRACT_VERSION} requires raw previous-action {group} feedback")
            if term.delay_max_lag != 0:
                raise ValueError(f"Get-up {GETUP_CONTRACT_VERSION} requires undelayed {group} action feedback")
        robot_cfg = unwrapped.cfg.scene.entities["robot"]
        actuator_cfgs = robot_cfg.articulation.actuators if robot_cfg.articulation else ()
        if len(actuator_cfgs) != 1 or not isinstance(actuator_cfgs[0], GetupBamActuatorCfg):
            raise ValueError(f"Get-up {GETUP_CONTRACT_VERSION} requires its body/neck XC330 actuator model")
        if actuator_cfgs[0].kp_fw != GETUP_BODY_KP_FW or actuator_cfgs[0].max_current != 0.91:
            raise ValueError(f"Get-up {GETUP_CONTRACT_VERSION} requires body P125 and XC330 0.91 A current limit")
        super().__init__(env, train_cfg, log_dir, device)

    def save(self, path: str, infos=None) -> None:
        infos = {
            **(infos or {}),
            "microban_getup_contract": GETUP_CONTRACT_VERSION,
            "microban_getup_angular_velocity_frame": GETUP_ANGULAR_VELOCITY_FRAME,
            "microban_getup_home_pose": getup_home_pose(),
        }
        super().save(path, infos)

    def load(self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None) -> dict:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        # Resumed v4-stamped checkpoints are re-stamped v5 on the next save.
        require_getup_checkpoint_contract(
            Path(path), checkpoint.get("infos"), require_recorded_env=False
        )
        return super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)
