# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Fail-closed velocity-teacher helpers for tracking-policy bootstrapping.

The legacy XC330 velocity policy is useful only as a *proposal generator*.  Its
Gaussian actor is unbounded and is therefore never copied into, or deployed as,
the 99-input tracking actor.  A proposal becomes a behavior-cloning label only
after it is converted to an absolute joint target and checked against all of the
current tracking-policy contracts:

* the action term's absolute soft limits;
* the bounded student's deterministic action closure; and
* the measured joint state plus the 120 ms maximum actuator-delay lookahead.

A proposal requiring more than 1 mrad of target projection is marked as no
longer faithful to the legacy actor.  It may still be a safe *projected teacher*
if its complete projected rollout passes the dynamic gate.  The 5% measured-
state margin produces one zero/one BC weight per joint so a narrow shoulder
margin cannot discard otherwise useful leg labels.  The hard soft-limit
lookahead remains an unconditional whole-sample admission gate.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse
from rsl_rl.models import MLPModel
from tensordict import TensorDict
from torch import nn

MICROBAN_TRACKING_TEACHER_OBSERVATION_SCHEMA: tuple[tuple[str, int], ...] = (
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("joint_pos", 18),
    ("joint_vel", 18),
    ("actions", 18),
    ("command", 3),
)
MICROBAN_TRACKING_TEACHER_OBSERVATION_WIDTH = sum(
    width for _, width in MICROBAN_TRACKING_TEACHER_OBSERVATION_SCHEMA
)
MICROBAN_TRACKING_TEACHER_ACTION_WIDTH = 18

# This is the last checkpoint of the short, measured-XC330 velocity canary.  It
# is deliberately pinned by content rather than selected with a model_*.pt glob.
MICROBAN_TRACKING_TEACHER_CHECKPOINT_SHA256 = (
    "90dc3a3e8dd7a79b73b09ed60e27dc707ee7656213c8084b1e047e65d1da3305"
)
MICROBAN_TRACKING_TEACHER_CHECKPOINT_ITERATION = 999
MICROBAN_TRACKING_TEACHER_ID = "xc330_velocity_4096_canary_model_999"

MICROBAN_TRACKING_TEACHER_LOOKAHEAD_S = 0.12
MICROBAN_TRACKING_TEACHER_MARGIN_RATIO = 0.05
MICROBAN_TRACKING_TEACHER_MAX_TARGET_PROJECTION_RAD = 1.0e-3


@dataclass(frozen=True)
class FrozenVelocityTeacherProvenance:
    """Exact identity of a locally verified frozen teacher actor."""

    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_iteration: int
    normalizer_count: int


@dataclass(frozen=True)
class TrackingTeacherProjection:
    """Projected labels and fail-closed per-sample admission evidence."""

    label_action: torch.Tensor
    label_target: torch.Tensor
    accepted: torch.Tensor
    faithful_to_legacy: torch.Tensor
    bc_weight: torch.Tensor
    finite: torch.Tensor
    measured_soft_limit_safe: torch.Tensor
    measured_preferred_margin_safe: torch.Tensor
    target_projection_rad: torch.Tensor
    maximum_target_projection_rad: torch.Tensor


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_last_width(
    value: torch.Tensor,
    *,
    width: int,
    name: str,
) -> None:
    if value.ndim < 1 or value.shape[-1] != width:
        raise ValueError(f"{name} must end in width {width}, got {tuple(value.shape)}")


def assemble_velocity_teacher_observation(
    *,
    base_ang_vel: torch.Tensor,
    projected_gravity: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    previous_action: torch.Tensor,
    command: torch.Tensor,
) -> torch.Tensor:
    """Assemble the legacy actor's exact 63-value observation wire format."""

    terms = (
        ("base_ang_vel", base_ang_vel, 3),
        ("projected_gravity", projected_gravity, 3),
        ("joint_pos", joint_pos, 18),
        ("joint_vel", joint_vel, 18),
        ("actions", previous_action, 18),
        ("command", command, 3),
    )
    prefix = base_ang_vel.shape[:-1]
    for name, value, width in terms:
        _require_last_width(value, width=width, name=name)
        if value.shape[:-1] != prefix:
            raise ValueError(
                f"{name} batch shape {value.shape[:-1]} does not match {prefix}"
            )
        if not bool(torch.isfinite(value).all().item()):
            raise FloatingPointError(f"{name} contains non-finite teacher input")
    observation = torch.cat([value for _name, value, _width in terms], dim=-1)
    _require_last_width(
        observation,
        width=MICROBAN_TRACKING_TEACHER_OBSERVATION_WIDTH,
        name="velocity teacher observation",
    )
    return observation


def reference_velocity_command_b(
    *,
    reference_quat_w: torch.Tensor,
    reference_lin_vel_w: torch.Tensor,
    reference_ang_vel_w: torch.Tensor,
) -> torch.Tensor:
    """Convert one fixed clip's reference root velocity to teacher commands.

    This command is admissible only for fixed-clip distillation: each reference
    ``q_ref/qdot_ref`` pair in the student's 99 inputs identifies the same clip
    phase that produced this value.  It must not be used as a hidden command for
    a dataset that contains identical student observations with different
    desired velocities.
    """

    _require_last_width(reference_quat_w, width=4, name="reference_quat_w")
    _require_last_width(reference_lin_vel_w, width=3, name="reference_lin_vel_w")
    _require_last_width(reference_ang_vel_w, width=3, name="reference_ang_vel_w")
    prefix = reference_quat_w.shape[:-1]
    if (
        reference_lin_vel_w.shape[:-1] != prefix
        or reference_ang_vel_w.shape[:-1] != prefix
    ):
        raise ValueError("Reference velocity tensors must share one batch shape")
    if not bool(
        torch.isfinite(reference_quat_w).all()
        and torch.isfinite(reference_lin_vel_w).all()
        and torch.isfinite(reference_ang_vel_w).all()
    ):
        raise FloatingPointError("Reference velocity command input is non-finite")
    lin_vel_b = quat_apply_inverse(reference_quat_w, reference_lin_vel_w)
    ang_vel_b = quat_apply_inverse(reference_quat_w, reference_ang_vel_w)
    return torch.stack((lin_vel_b[..., 0], lin_vel_b[..., 1], ang_vel_b[..., 2]), dim=-1)


def bounded_student_deterministic_action_closure(
    distribution: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return finite, strictly interior bounds of the student's mean output.

    The bounded distribution first limits its Gaussian mean and then applies the
    asymmetric action transform.  Therefore the deterministic actor's closure is
    narrower than the action term's hard interval.  Labels are moved one ULP
    toward zero so every accepted target is represented by a finite actor logit.
    """

    required = ("mean_lower_bound", "mean_upper_bound", "_transform")
    if any(not hasattr(distribution, name) for name in required):
        raise TypeError("Student distribution does not expose bounded mean closure")
    mean_lower = distribution.mean_lower_bound
    mean_upper = distribution.mean_upper_bound
    transform = distribution._transform
    if not isinstance(mean_lower, torch.Tensor) or not isinstance(
        mean_upper, torch.Tensor
    ):
        raise TypeError("Student distribution closure bounds must be tensors")
    with torch.no_grad():
        lower = transform(mean_lower)
        upper = transform(mean_upper)
        zero = torch.zeros((), dtype=lower.dtype, device=lower.device)
        lower = torch.nextafter(lower, zero)
        upper = torch.nextafter(upper, zero)
    if lower.shape != (MICROBAN_TRACKING_TEACHER_ACTION_WIDTH,) or upper.shape != (
        MICROBAN_TRACKING_TEACHER_ACTION_WIDTH,
    ):
        raise ValueError("Student deterministic closure must be exactly 18-wide")
    if not bool(torch.isfinite(lower).all() and torch.isfinite(upper).all()):
        raise FloatingPointError("Student deterministic closure is non-finite")
    if not bool(torch.all(lower < 0.0).item() and torch.all(upper > 0.0).item()):
        raise ValueError("Student deterministic closure must contain raw zero")
    return lower, upper


def _broadcast_joint_vector(
    value: torch.Tensor | Sequence[float] | float,
    *,
    reference: torch.Tensor,
    name: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    try:
        result = torch.broadcast_to(tensor, reference.shape)
    except RuntimeError as error:
        raise ValueError(
            f"{name} with shape {tuple(tensor.shape)} cannot broadcast to "
            f"{tuple(reference.shape)}"
        ) from error
    return result


def project_velocity_teacher_labels(
    teacher_action: torch.Tensor,
    *,
    teacher_scale: torch.Tensor | Sequence[float] | float,
    teacher_offset: torch.Tensor | Sequence[float] | float,
    student_scale: torch.Tensor | Sequence[float] | float,
    student_offset: torch.Tensor | Sequence[float] | float,
    soft_lower: torch.Tensor | Sequence[float],
    soft_upper: torch.Tensor | Sequence[float],
    student_action_lower: torch.Tensor | Sequence[float],
    student_action_upper: torch.Tensor | Sequence[float],
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    default_joint_pos: torch.Tensor | Sequence[float],
    lookahead_s: float = MICROBAN_TRACKING_TEACHER_LOOKAHEAD_S,
    margin_ratio: float = MICROBAN_TRACKING_TEACHER_MARGIN_RATIO,
    max_target_projection_rad: float = (
        MICROBAN_TRACKING_TEACHER_MAX_TARGET_PROJECTION_RAD
    ),
) -> TrackingTeacherProjection:
    """Project legacy proposals and compute fail-closed label admission masks.

    ``accepted`` enforces finite inputs and the hard measured-state lookahead
    gate.  ``faithful_to_legacy`` additionally requires at most
    ``max_target_projection_rad`` of absolute-target projection.  A non-faithful
    action is not silently described as the old policy's behavior, but can still
    be used as a new projected teacher after that projected controller passes its
    own dynamic gate.  ``bc_weight`` is per joint and drops only axes outside the
    neutral-preserving 5% preferred state margin.
    """

    _require_last_width(
        teacher_action,
        width=MICROBAN_TRACKING_TEACHER_ACTION_WIDTH,
        name="teacher_action",
    )
    if joint_pos.shape != teacher_action.shape or joint_vel.shape != teacher_action.shape:
        raise ValueError("joint_pos/joint_vel must exactly match teacher_action shape")
    if not math.isfinite(lookahead_s) or lookahead_s < 0.0:
        raise ValueError("lookahead_s must be finite and non-negative")
    if not math.isfinite(margin_ratio) or not 0.0 < margin_ratio < 0.5:
        raise ValueError("margin_ratio must be finite and in (0, 0.5)")
    if not math.isfinite(max_target_projection_rad) or max_target_projection_rad < 0.0:
        raise ValueError("max_target_projection_rad must be finite and non-negative")

    teacher_scale_t = _broadcast_joint_vector(
        teacher_scale, reference=teacher_action, name="teacher_scale"
    )
    teacher_offset_t = _broadcast_joint_vector(
        teacher_offset, reference=teacher_action, name="teacher_offset"
    )
    student_scale_t = _broadcast_joint_vector(
        student_scale, reference=teacher_action, name="student_scale"
    )
    student_offset_t = _broadcast_joint_vector(
        student_offset, reference=teacher_action, name="student_offset"
    )
    lower_t = _broadcast_joint_vector(
        soft_lower, reference=teacher_action, name="soft_lower"
    )
    upper_t = _broadcast_joint_vector(
        soft_upper, reference=teacher_action, name="soft_upper"
    )
    action_lower_t = _broadcast_joint_vector(
        student_action_lower,
        reference=teacher_action,
        name="student_action_lower",
    )
    action_upper_t = _broadcast_joint_vector(
        student_action_upper,
        reference=teacher_action,
        name="student_action_upper",
    )
    default_t = _broadcast_joint_vector(
        default_joint_pos, reference=teacher_action, name="default_joint_pos"
    )

    named = {
        "teacher_action": teacher_action,
        "teacher_scale": teacher_scale_t,
        "teacher_offset": teacher_offset_t,
        "student_scale": student_scale_t,
        "student_offset": student_offset_t,
        "soft_lower": lower_t,
        "soft_upper": upper_t,
        "student_action_lower": action_lower_t,
        "student_action_upper": action_upper_t,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "default_joint_pos": default_t,
    }
    finite_per_joint = torch.ones_like(teacher_action, dtype=torch.bool)
    for value in named.values():
        finite_per_joint &= torch.isfinite(value)
    finite = finite_per_joint.all(dim=-1)

    if not bool(
        torch.all(teacher_scale_t > 0.0).item()
        and torch.all(student_scale_t > 0.0).item()
    ):
        raise ValueError("Teacher/student action scales must be strictly positive")
    if not bool(torch.all(lower_t < upper_t).item()):
        raise ValueError("Every soft lower bound must be below its upper bound")
    if not bool(torch.all(action_lower_t < action_upper_t).item()):
        raise ValueError("Every student action lower bound must be below its upper")
    if not bool(
        torch.all((default_t >= lower_t) & (default_t <= upper_t)).item()
    ):
        raise ValueError("Every default joint position must lie in the soft limits")

    # Avoid propagating NaNs through clamp into the returned diagnostic label.
    safe_teacher_action = torch.where(
        finite_per_joint, teacher_action, torch.zeros_like(teacher_action)
    )
    proposal_target = teacher_offset_t + teacher_scale_t * safe_teacher_action
    soft_target = torch.clamp(proposal_target, min=lower_t, max=upper_t)
    student_raw = (soft_target - student_offset_t) / student_scale_t
    label_action = torch.clamp(
        student_raw, min=action_lower_t, max=action_upper_t
    )
    label_target = student_offset_t + student_scale_t * label_action
    # Retain the independent hard action clip after the actor-closure projection.
    label_target = torch.clamp(label_target, min=lower_t, max=upper_t)
    label_action = (label_target - student_offset_t) / student_scale_t

    projection = torch.abs(label_target - proposal_target)
    maximum_projection = projection.max(dim=-1).values

    predicted_pos = joint_pos + lookahead_s * joint_vel
    dangerous_lower = torch.minimum(joint_pos, predicted_pos)
    dangerous_upper = torch.maximum(joint_pos, predicted_pos)
    measured_soft_safe = (
        (dangerous_lower >= lower_t) & (dangerous_upper <= upper_t)
    ).all(dim=-1)

    span = upper_t - lower_t
    preferred_lower = torch.minimum(default_t, lower_t + margin_ratio * span)
    preferred_upper = torch.maximum(default_t, upper_t - margin_ratio * span)
    measured_preferred_safe = (
        (dangerous_lower >= preferred_lower)
        & (dangerous_upper <= preferred_upper)
    )

    accepted = finite & measured_soft_safe
    faithful = accepted & (maximum_projection <= max_target_projection_rad)
    bc_weight = (
        accepted.unsqueeze(-1) & measured_preferred_safe
    ).to(dtype=teacher_action.dtype)
    return TrackingTeacherProjection(
        label_action=label_action,
        label_target=label_target,
        accepted=accepted,
        faithful_to_legacy=faithful,
        bc_weight=bc_weight,
        finite=finite,
        measured_soft_limit_safe=measured_soft_safe,
        measured_preferred_margin_safe=measured_preferred_safe,
        target_projection_rad=projection,
        maximum_target_projection_rad=maximum_projection,
    )


class FrozenVelocityTeacher(nn.Module):
    """The exact 63->18 legacy actor, loaded actor-only and frozen."""

    def __init__(self, actor: MLPModel, provenance: FrozenVelocityTeacherProvenance):
        super().__init__()
        self.actor = actor
        self.provenance = provenance
        self.eval()
        self.actor.requires_grad_(False)

    @torch.inference_mode()
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        _require_last_width(
            observation,
            width=MICROBAN_TRACKING_TEACHER_OBSERVATION_WIDTH,
            name="velocity teacher observation",
        )
        if observation.ndim != 2:
            raise ValueError("Velocity teacher observation must be rank two")
        if not bool(torch.isfinite(observation).all().item()):
            raise FloatingPointError("Velocity teacher observation is non-finite")
        tensordict = TensorDict(
            {"teacher": observation}, batch_size=[observation.shape[0]]
        )
        action = self.actor(tensordict)
        if action.shape != (
            observation.shape[0],
            MICROBAN_TRACKING_TEACHER_ACTION_WIDTH,
        ):
            raise ValueError("Velocity teacher returned an unexpected action shape")
        if not bool(torch.isfinite(action).all().item()):
            raise FloatingPointError("Velocity teacher returned non-finite action")
        return action


def load_frozen_velocity_teacher(
    checkpoint_path: str | Path,
    *,
    checkpoint_sha256: str = MICROBAN_TRACKING_TEACHER_CHECKPOINT_SHA256,
    checkpoint_iteration: int = MICROBAN_TRACKING_TEACHER_CHECKPOINT_ITERATION,
    device: str | torch.device = "cpu",
) -> FrozenVelocityTeacher:
    """Verify and load only the pinned legacy actor; ignore critic/optimizer."""

    if len(checkpoint_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in checkpoint_sha256
    ):
        raise ValueError("Teacher checkpoint SHA-256 must be 64 lowercase hex digits")
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Velocity teacher checkpoint not found: {checkpoint}")
    actual_sha256 = sha256_file(checkpoint)
    if actual_sha256 != checkpoint_sha256:
        raise ValueError(
            "Velocity teacher checkpoint SHA-256 mismatch: "
            f"expected {checkpoint_sha256}, got {actual_sha256}"
        )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("Velocity teacher checkpoint root must be a mapping")
    if (
        isinstance(checkpoint_iteration, bool)
        or not isinstance(checkpoint_iteration, int)
        or checkpoint_iteration < 0
    ):
        raise ValueError("Teacher checkpoint iteration must be non-negative")
    iteration = payload.get("iter")
    if iteration != checkpoint_iteration:
        raise ValueError(
            "Velocity teacher checkpoint iteration mismatch: "
            f"expected {checkpoint_iteration}, got {iteration!r}"
        )
    actor_state = payload.get("actor_state_dict")
    if not isinstance(actor_state, Mapping):
        raise TypeError("Velocity teacher checkpoint has no actor_state_dict")
    for key, value in actor_state.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Velocity teacher actor state {key!r} is not a Tensor")
        if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
            raise FloatingPointError(
                f"Velocity teacher actor state {key!r} is non-finite"
            )

    placeholder = TensorDict(
        {
            "teacher": torch.zeros(
                1, MICROBAN_TRACKING_TEACHER_OBSERVATION_WIDTH, dtype=torch.float32
            )
        },
        batch_size=[1],
    )
    actor = MLPModel(
        obs=placeholder,
        obs_groups={"actor": ["teacher"]},
        obs_set="actor",
        output_dim=MICROBAN_TRACKING_TEACHER_ACTION_WIDTH,
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    )
    actor.load_state_dict(actor_state, strict=True)
    actor.to(device)
    count = actor_state.get("obs_normalizer.count")
    if not isinstance(count, torch.Tensor) or count.ndim != 0:
        raise ValueError("Velocity teacher normalizer count must be a scalar Tensor")
    normalizer_count = int(count.item())
    if normalizer_count < 0:
        raise ValueError("Velocity teacher normalizer count must be non-negative")
    provenance = FrozenVelocityTeacherProvenance(
        checkpoint_path=checkpoint,
        checkpoint_sha256=actual_sha256,
        checkpoint_iteration=iteration,
        normalizer_count=normalizer_count,
    )
    return FrozenVelocityTeacher(actor=actor, provenance=provenance)
