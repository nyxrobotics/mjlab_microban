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
from mjlab_microban.robot.home_pose import HOME
from bam.mjlab import BamActuatorCfg

from mjlab_microban.robot.microban_constants import (
    HOME_FRAME,
    SERVO_KP_POLICY,
    SERVO_TARGET_RANGE_RAD,
)
from mjlab_microban.tasks.curriculum import (
    STEPS_PER_UPDATE_ATTR,
    bind_update_clock,
    stage_log_line,
)
from mjlab_microban.tasks.microban_getup_action import (
    GetupJointPositionAction,
    raw_getup_action,
)
from mjlab_microban.tasks.microban_getup_env_cfg import (
    GETUP_ACTION_CLIP,
    GETUP_EPISODE_LENGTH_S,
    GETUP_REFINE_ACTION_STD,
    GETUP_REFINE_ENTROPY_COEF,
    GETUP_SCHEDULE,
)

# Log name of the refine switch's exploration reset (the pipeline monitor
# expects it at GETUP_SCHEDULE["refine"]).
GETUP_REFINE_EXPLORATION_STAGE = "refine exploration (std, Adam, learning rate, entropy)"
# Checkpoint marker: the refine switch's exploration reset has been applied.
GETUP_EXPLORATION_REFINED_INFO_KEY = "microban_getup_exploration_refined"


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
# The two HOMEs with published artifacts pin every stamped value
# (home_pose.LEGACY_HOME_OVERRIDES), so they keep their branches' exact
# comparison (stamp == HOME, recorded root atol 1e-12).
HOME_STAMP_TOLERANCE = 0.0 if HOME.is_legacy else 1.0e-9
HOME_ROOT_RECORDED_ATOL = 1.0e-12 if HOME.is_legacy else 1.0e-9


def home_pose_stamps_match(stamp: object, expected: object, tol: float = HOME_STAMP_TOLERANCE) -> bool:
    """Structural equality of two HOME stamps with a float tolerance (0: ``==``)."""

    if tol == 0.0:
        return stamp == expected
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
        if not np.allclose(
            np.asarray(init_state[key], dtype=np.float64), expected, rtol=0, atol=HOME_ROOT_RECORDED_ATOL
        ):
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
    """Stamp compatible checkpoints, reject old get-up resumes, and run the refine switch.

    At the update GETUP_SCHEDULE["refine"] (right after that update, so the
    next rollout already explores with it) the runner resets the action std to
    GETUP_REFINE_ACTION_STD, clears the Adam moments, puts the adaptive
    learning rate back to its configured start and lowers the entropy
    coefficient to GETUP_REFINE_ENTROPY_COEF -- once: the checkpoint records
    it, and a resumed run past the switch keeps its std and optimizer state
    and only gets the entropy coefficient back.  The switch prints a
    curriculum-format line for the pipeline monitor.
    """

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
        if len(actuator_cfgs) != 1 or type(actuator_cfgs[0]) is not BamActuatorCfg:
            raise ValueError(f"Get-up {GETUP_CONTRACT_VERSION} requires the shared XC330 actuator model")
        if actuator_cfgs[0].kp_fw != SERVO_KP_POLICY or actuator_cfgs[0].max_current != 0.91:
            raise ValueError(
                f"Get-up {GETUP_CONTRACT_VERSION} requires P{SERVO_KP_POLICY} on every servo and the "
                "XC330 0.91 A current limit"
            )
        bind_update_clock(unwrapped, int(train_cfg["num_steps_per_env"]))
        super().__init__(env, train_cfg, log_dir, device)
        self.exploration_refined = False
        self.initial_learning_rate = float(self.alg.learning_rate)
        update = self.alg.update

        def update_then_refine(*args, **kwargs):
            result = update(*args, **kwargs)
            self.maybe_refine_exploration()
            return result

        self.alg.update = update_then_refine

    def maybe_refine_exploration(self) -> bool:
        """Apply the refine switch once its update is reached (see class doc)."""

        if self.exploration_refined:
            return False
        env = self.env.unwrapped
        steps = getattr(env, STEPS_PER_UPDATE_ATTR)
        counter = int(env.common_step_counter)
        if counter < GETUP_SCHEDULE["refine"] * steps:
            return False
        policy = self.alg.get_policy()
        with torch.no_grad():
            policy.distribution.std_param.fill_(GETUP_REFINE_ACTION_STD)
        self.alg.optimizer.state.clear()
        self.alg.learning_rate = self.initial_learning_rate
        for group in self.alg.optimizer.param_groups:
            group["lr"] = self.initial_learning_rate
        self.alg.entropy_coef = GETUP_REFINE_ENTROPY_COEF
        self.exploration_refined = True
        print(stage_log_line(0, GETUP_REFINE_EXPLORATION_STAGE, counter, steps), flush=True)
        return True

    def save(self, path: str, infos=None) -> None:
        infos = {
            **(infos or {}),
            "microban_getup_contract": GETUP_CONTRACT_VERSION,
            "microban_getup_angular_velocity_frame": GETUP_ANGULAR_VELOCITY_FRAME,
            "microban_getup_home_pose": getup_home_pose(),
            GETUP_EXPLORATION_REFINED_INFO_KEY: self.exploration_refined,
        }
        super().save(path, infos)

    def load(self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None) -> dict:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        # Resumed v4-stamped checkpoints are re-stamped v5 on the next save.
        require_getup_checkpoint_contract(
            Path(path), checkpoint.get("infos"), require_recorded_env=False
        )
        infos = super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)
        if load_cfg is None or load_cfg.get("iteration", False):
            # Continue after the saved update (rsl_rl would repeat it).
            self.current_learning_iteration += 1
        if (infos or {}).get(GETUP_EXPLORATION_REFINED_INFO_KEY):
            self.exploration_refined = True
            self.alg.entropy_coef = GETUP_REFINE_ENTROPY_COEF
        return infos
