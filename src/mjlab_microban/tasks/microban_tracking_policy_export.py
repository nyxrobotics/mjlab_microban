# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Safe export contract for Microban's fixed-motion tracking policy.

The tracking ONNX deliberately bundles a fixed reference clip alongside the PPO
actor.  Its ``actions`` output controls 18 arm/leg joints, while the bundled
``joint_pos`` and ``joint_vel`` reference outputs retain all 21 robot joints.
This distinction is important: the three HMD joints are reference data only and
must never be sent through the body-policy action path.

The resulting artifact is useful for offline fixed-clip tracking evaluation.  It
is not the live PICO teleoperation policy; that is exported separately by
``MicrobanTeleopOnPolicyRunner``.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import onnx
import torch
import wandb
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.tracking.mdp import MotionCommand
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_HMD_JOINT_NAMES,
    MICROBAN_TELEOP_ACTION_JOINT_NAMES,
)
from mjlab_microban.tasks.microban_tracking_env_cfg import MICROBAN_JOINT_NAMES


TRACKING_ONNX_OUTPUT_NAMES: tuple[str, ...] = (
    "actions",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


def _as_action_vector(value: float | torch.Tensor, count: int) -> list[float]:
    if isinstance(value, torch.Tensor):
        values = (
            value[0].detach().cpu().tolist()
            if value.ndim == 2
            else value.detach().cpu().tolist()
        )
        if len(values) != count:
            raise ValueError(f"Expected {count} action values, got {len(values)}")
        return [float(item) for item in values]
    return [float(value)] * count


def _tensor_shape(value: onnx.ValueInfoProto) -> tuple[int | str, ...]:
    """Return a compact ONNX tensor shape for validation/error messages."""

    shape: list[int | str] = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            shape.append(dim.dim_param)
        else:
            shape.append("?")
    return tuple(shape)


def _require_trailing_shape(
    output: onnx.ValueInfoProto,
    expected: tuple[int, ...],
) -> None:
    actual = _tensor_shape(output)
    if len(actual) < len(expected) or tuple(actual[-len(expected) :]) != expected:
        raise ValueError(
            f"ONNX output {output.name!r} must end in {expected}, got {actual}"
        )


def validate_microban_tracking_onnx(
    onnx_path: str | Path,
    *,
    expected_action_count: int = 18,
    expected_reference_joint_count: int = 21,
    expected_reference_body_count: int = 19,
) -> None:
    """Validate all fixed-motion ONNX outputs before attaching metadata.

    This rejects the unsafe failure mode where a policy exports 18 actions but
    receives ambiguous 21-joint action metadata.
    """

    model = onnx.load(str(onnx_path))
    outputs = {output.name: output for output in model.graph.output}
    if tuple(outputs) != TRACKING_ONNX_OUTPUT_NAMES:
        raise ValueError(
            "Unexpected tracking ONNX outputs: "
            f"got {tuple(outputs)}, expected {TRACKING_ONNX_OUTPUT_NAMES}"
        )

    _require_trailing_shape(outputs["actions"], (expected_action_count,))
    _require_trailing_shape(
        outputs["joint_pos"], (expected_reference_joint_count,)
    )
    _require_trailing_shape(
        outputs["joint_vel"], (expected_reference_joint_count,)
    )
    _require_trailing_shape(
        outputs["body_pos_w"], (expected_reference_body_count, 3)
    )
    _require_trailing_shape(
        outputs["body_quat_w"], (expected_reference_body_count, 4)
    )
    _require_trailing_shape(
        outputs["body_lin_vel_w"], (expected_reference_body_count, 3)
    )
    _require_trailing_shape(
        outputs["body_ang_vel_w"], (expected_reference_body_count, 3)
    )


def get_microban_tracking_metadata(
    env: ManagerBasedRlEnv,
    run_path: str,
) -> dict[str, list | str | float]:
    """Build unambiguous metadata for the 18-action/21-reference artifact."""

    robot: Entity = env.scene["robot"]
    action = env.action_manager.get_term("joint_pos")
    if not isinstance(action, JointPositionAction):
        raise TypeError("Microban tracking requires JointPositionAction 'joint_pos'")

    action_joint_names = list(action.target_names)
    expected_action_names = list(MICROBAN_TELEOP_ACTION_JOINT_NAMES)
    if action_joint_names != expected_action_names:
        raise ValueError(
            "Unsafe Microban tracking action order: "
            f"resolved {action_joint_names}, expected {expected_action_names}"
        )

    reference_joint_names = list(MICROBAN_JOINT_NAMES)
    if list(robot.joint_names) != reference_joint_names:
        raise ValueError(
            "Reference joint order does not match the loaded robot: "
            f"resolved {list(robot.joint_names)}, expected {reference_joint_names}"
        )

    motion = cast(MotionCommand, env.command_manager.get_term("motion"))
    reference_body_names = list(motion.cfg.body_names)
    if motion.motion.joint_pos.shape[-1] != len(reference_joint_names):
        raise ValueError(
            "Motion reference joint width does not match reference_joint_names: "
            f"{motion.motion.joint_pos.shape[-1]} != {len(reference_joint_names)}"
        )
    if motion.motion.body_pos_w.shape[-2] != len(reference_body_names):
        raise ValueError(
            "Motion reference body width does not match reference_body_names: "
            f"{motion.motion.body_pos_w.shape[-2]} != {len(reference_body_names)}"
        )

    action_joint_ids = action.target_ids.detach().cpu().tolist()
    joint_name_to_ctrl_id = {
        actuator.target.split("/")[-1]: actuator.id
        for actuator in robot.spec.actuators
    }
    ctrl_ids = [joint_name_to_ctrl_id[name] for name in action_joint_names]
    stiffness = env.sim.mj_model.actuator_gainprm[ctrl_ids, 0].tolist()
    damping = (-env.sim.mj_model.actuator_biasprm[ctrl_ids, 2]).tolist()
    default_joint_pos = (
        robot.data.default_joint_pos[0, action_joint_ids].detach().cpu().tolist()
    )
    soft_limits = (
        robot.data.soft_joint_pos_limits[0, action_joint_ids].detach().cpu().tolist()
    )

    return {
        "run_path": run_path,
        "policy_type": "microban_fixed_motion_tracking_not_live_teleop",
        "deployment_scope": "offline_fixed_reference_only",
        # Backwards-compatible alias; unlike the generic exporter this is
        # deliberately the action order, never the 21-joint reference order.
        "joint_names": action_joint_names,
        "action_joint_names": action_joint_names,
        "hmd_joint_names": list(MICROBAN_HMD_JOINT_NAMES),
        "reference_joint_names": reference_joint_names,
        "anchor_body_name": motion.cfg.anchor_body_name,
        "body_names": reference_body_names,
        "reference_body_names": reference_body_names,
        "joint_stiffness": [float(value) for value in stiffness],
        "joint_damping": [float(value) for value in damping],
        "default_joint_pos": [float(value) for value in default_joint_pos],
        "soft_joint_pos_lower": [float(bounds[0]) for bounds in soft_limits],
        "soft_joint_pos_upper": [float(bounds[1]) for bounds in soft_limits],
        "command_names": list(env.command_manager.active_terms),
        "observation_names": list(env.observation_manager.active_terms["actor"]),
        "action_scale": _as_action_vector(action.scale, len(action_joint_names)),
    }


class MicrobanTrackingOnPolicyRunner(MotionTrackingOnPolicyRunner):
    """Fixed-motion runner with Microban-safe export validation and metadata."""

    env: RslRlVecEnvWrapper

    def save(self, path: str, infos=None) -> None:
        # Bypass MotionTrackingOnPolicyRunner.save(), whose generic metadata maps
        # every robot joint onto an 18-wide action tensor.  The direct parent
        # persists the checkpoint; this class then performs the validated fixed-
        # reference export exactly once.
        MjlabOnPolicyRunner.save(self, path, infos)
        policy_dir, filename, onnx_path = self._get_export_paths(path)
        temporary_filename = f".{filename}.tmp"
        temporary_path = policy_dir / temporary_filename
        try:
            # Publish atomically only after shape and metadata checks pass.  A
            # failed newer export therefore cannot replace the last known-good
            # artifact or expose a half-written model to deployment tooling.
            self.export_policy_to_onnx(str(policy_dir), temporary_filename)
            motion = cast(
                MotionCommand,
                self.env.unwrapped.command_manager.get_term("motion"),
            )
            validate_microban_tracking_onnx(
                temporary_path,
                expected_action_count=len(MICROBAN_TELEOP_ACTION_JOINT_NAMES),
                expected_reference_joint_count=len(MICROBAN_JOINT_NAMES),
                expected_reference_body_count=len(motion.cfg.body_names),
            )
            run_name: str = (
                wandb.run.name
                if self.logger.logger_type == "wandb" and wandb.run
                else "local"
            )
            metadata = get_microban_tracking_metadata(self.env.unwrapped, run_name)
            attach_metadata_to_onnx(str(temporary_path), metadata)
            temporary_path.replace(onnx_path)
            if self.logger.logger_type == "wandb" and self.cfg["upload_model"]:
                wandb.save(str(onnx_path), base_path=str(policy_dir))
                if self.registry_name is not None:
                    wandb.run.use_artifact(self.registry_name)
                    self.registry_name = None
        except Exception as exc:
            # Never leave a partially exported or ambiguously described model
            # where deployment tooling could pick it up.
            temporary_path.unlink(missing_ok=True)
            print(
                "[WARN] Microban tracking ONNX export failed "
                f"(training continues): {exc}"
            )
