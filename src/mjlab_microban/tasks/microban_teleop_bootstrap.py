# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Explicit actor-only bootstrap from Microban's XC330 velocity policy.

The velocity actor observes 63 values while the teleoperation actor observes 83.
The shared fields have identical meanings but different joint-vector offsets
because the teleoperation observation also contains the three HMD-owned joints.
This module performs that mapping without ever copying a critic, optimizer, PPO
iteration, action distribution, or teleoperation checkpoint contract.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

VELOCITY_ACTOR_OBSERVATION_WIDTH = 63
TELEOP_ACTOR_OBSERVATION_WIDTH = 83
VELOCITY_ACTOR_BOOTSTRAP_MAPPING_VERSION = (
    "xc330_velocity_63_to_teleop_83_v4_guarded_shoulder_roll_head"
)
# The common ±10-degree shoulder-roll home pose starts only one degree inside
# its soft limit.  Copying the velocity actor's state-dependent shoulder rows
# verbatim makes the first PPO update seed-sensitive: one observed
# production-width seed held the target at the neutral edge long enough for
# gravity to pull the measured joint beyond the soft limit.  Initialize only
# these two rows to a deterministic inward target.
# The rest of the velocity actor remains byte-for-byte mapped, and the rows are
# ordinary trainable parameters after initialization.
TELEOP_SHOULDER_ROLL_ACTION_INDICES: tuple[int, int] = (1, 10)
TELEOP_SHOULDER_ROLL_INITIAL_LATENT_BIASES: tuple[float, float] = (-0.25, 0.25)
TELEOP_SHOULDER_ROLL_INITIALIZATION = (
    "zero_final_weights_constant_inward_latent_bias_v1"
)

# velocity: base_ang_vel(3), gravity(3), joint_pos(18), joint_vel(18),
# actions(18), command(3)
# teleop: base_ang_vel(3), gravity(3), joint_pos(21; HMD first),
# joint_vel(21; HMD first), actions(18), command(3), feet(6), hands(8)
VELOCITY_TO_TELEOP_OBSERVATION_INDEX: tuple[tuple[int, int], ...] = (
    *((index, index) for index in range(6)),
    *((index, index + 3) for index in range(6, 24)),
    *((index, index + 6) for index in range(24, 42)),
    *((index, index + 6) for index in range(42, 60)),
    *((index, index + 6) for index in range(60, 63)),
)
if tuple(source for source, _target in VELOCITY_TO_TELEOP_OBSERVATION_INDEX) != (
    tuple(range(VELOCITY_ACTOR_OBSERVATION_WIDTH))
):
    raise RuntimeError("Velocity-to-teleop mapping must cover every source column once")
if len({target for _source, target in VELOCITY_TO_TELEOP_OBSERVATION_INDEX}) != (
    VELOCITY_ACTOR_OBSERVATION_WIDTH
):
    raise RuntimeError("Velocity-to-teleop mapping target columns must be unique")

_NORMALIZER_KEYS = (
    "obs_normalizer._mean",
    "obs_normalizer._var",
    "obs_normalizer._std",
)
_COPIED_MLP_KEYS = (
    "mlp.0.bias",
    "mlp.2.weight",
    "mlp.2.bias",
    "mlp.4.weight",
    "mlp.4.bias",
    "mlp.6.weight",
    "mlp.6.bias",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class VelocityActorBootstrapProvenance:
    """Identity of the pinned velocity actor used to initialize teleoperation."""

    checkpoint_path: Path
    checkpoint_sha256: str
    source_normalizer_count: float
    installed_normalizer_count: float
    shoulder_roll_initial_latent_biases: tuple[float, float]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_tensor(
    state: Mapping[str, object], key: str, expected_shape: tuple[int, ...]
) -> torch.Tensor:
    value = state.get(key)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Actor state is missing Tensor {key!r}")
    if tuple(value.shape) != expected_shape:
        raise ValueError(
            f"Actor state {key!r} must have shape {expected_shape}, "
            f"got {tuple(value.shape)}"
        )
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"Actor state {key!r} contains non-finite values")
    return value


def expand_velocity_observation_to_teleop(
    velocity_observation: torch.Tensor,
) -> torch.Tensor:
    """Embed velocity observations in the teleop schema with new fields zeroed."""

    if velocity_observation.ndim < 1 or velocity_observation.shape[-1] != (
        VELOCITY_ACTOR_OBSERVATION_WIDTH
    ):
        raise ValueError(
            f"Velocity observation must end in width {VELOCITY_ACTOR_OBSERVATION_WIDTH}"
        )
    teleop = velocity_observation.new_zeros(
        (*velocity_observation.shape[:-1], TELEOP_ACTOR_OBSERVATION_WIDTH)
    )
    source_indices, target_indices = zip(
        *VELOCITY_TO_TELEOP_OBSERVATION_INDEX, strict=True
    )
    teleop[..., list(target_indices)] = velocity_observation[..., list(source_indices)]
    return teleop


def bootstrap_teleop_actor_state(
    target_actor_state: Mapping[str, object],
    velocity_actor_state: Mapping[str, object],
) -> dict[str, torch.Tensor]:
    """Return a teleop actor state initialized only from shared velocity inputs.

    New HMD/keypoint input columns are exactly zero in the first layer.  Their
    normalizer mean is zero and variance/std are one, which is the neutral
    identity initialization (zero variance would cause division by zero).  The
    target action-distribution parameters remain untouched.  The two final
    shoulder-roll rows are deliberately initialized to constant inward latent
    means because the common shoulder-roll home pose has only one degree of
    soft-limit headroom and the copied state-dependent rows did not reliably
    preserve the inward holding target through the first PPO update.
    """

    target_first = target_actor_state.get("mlp.0.weight")
    source_first = velocity_actor_state.get("mlp.0.weight")
    if not isinstance(target_first, torch.Tensor) or target_first.ndim != 2:
        raise ValueError("Target actor must contain a rank-2 'mlp.0.weight'")
    if not isinstance(source_first, torch.Tensor) or source_first.ndim != 2:
        raise ValueError("Velocity actor must contain a rank-2 'mlp.0.weight'")
    if target_first.shape[1] != TELEOP_ACTOR_OBSERVATION_WIDTH:
        raise ValueError(
            "Target actor first layer must consume 83 teleop observations, got "
            f"{target_first.shape[1]}"
        )
    if source_first.shape[1] != VELOCITY_ACTOR_OBSERVATION_WIDTH:
        raise ValueError(
            "Velocity actor first layer must consume 63 observations, got "
            f"{source_first.shape[1]}"
        )
    if target_first.shape[0] != source_first.shape[0]:
        raise ValueError("Source and target actor first-layer widths do not match")

    result: dict[str, torch.Tensor] = {}
    for key, value in target_actor_state.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Target actor state {key!r} is not a Tensor")
        result[key] = value.detach().clone()

    source_indices, target_indices = zip(
        *VELOCITY_TO_TELEOP_OBSERVATION_INDEX, strict=True
    )
    first = torch.zeros_like(target_first)
    first[:, list(target_indices)] = source_first[:, list(source_indices)].to(
        device=first.device, dtype=first.dtype
    )
    result["mlp.0.weight"] = first

    for key in _COPIED_MLP_KEYS:
        target_value = result.get(key)
        if target_value is None:
            raise ValueError(f"Target actor state is missing Tensor {key!r}")
        source_value = _require_tensor(
            velocity_actor_state, key, tuple(target_value.shape)
        )
        result[key] = (
            source_value.to(device=target_value.device, dtype=target_value.dtype)
            .detach()
            .clone()
        )

    final_weight = result["mlp.6.weight"]
    final_bias = result["mlp.6.bias"]
    if final_weight.ndim != 2 or final_weight.shape[0] != 18:
        raise ValueError("Target actor final weight must contain 18 action rows")
    if final_bias.shape != (18,):
        raise ValueError("Target actor final bias must contain 18 actions")
    shoulder_indices = list(TELEOP_SHOULDER_ROLL_ACTION_INDICES)
    final_weight[shoulder_indices] = 0.0
    final_bias[shoulder_indices] = final_bias.new_tensor(
        TELEOP_SHOULDER_ROLL_INITIAL_LATENT_BIASES
    )

    for key in _NORMALIZER_KEYS:
        target_value = _require_tensor(
            target_actor_state, key, (1, TELEOP_ACTOR_OBSERVATION_WIDTH)
        )
        source_value = _require_tensor(
            velocity_actor_state, key, (1, VELOCITY_ACTOR_OBSERVATION_WIDTH)
        )
        if not key.endswith("_mean") and not bool((source_value > 0.0).all().item()):
            raise ValueError(f"Velocity actor state {key!r} must be strictly positive")
        neutral = 0.0 if key.endswith("_mean") else 1.0
        mapped = torch.full_like(target_value, neutral)
        mapped[:, list(target_indices)] = source_value[:, list(source_indices)].to(
            device=mapped.device, dtype=mapped.dtype
        )
        result[key] = mapped

    target_count = result.get("obs_normalizer.count")
    source_count = velocity_actor_state.get("obs_normalizer.count")
    if not isinstance(target_count, torch.Tensor) or target_count.ndim != 0:
        raise ValueError("Target actor normalizer count must be a scalar Tensor")
    if not isinstance(source_count, torch.Tensor) or source_count.ndim != 0:
        raise ValueError("Velocity actor normalizer count must be a scalar Tensor")
    source_count_value = float(source_count.item())
    if not 0.0 <= source_count_value < float("inf"):
        raise ValueError(
            "Velocity actor normalizer count must be finite and non-negative"
        )
    # Preserve the source sample count exactly.  Capping the v3 bootstrap at
    # one million let a single 4096-env rollout contribute almost ten percent
    # of the running statistics; after 500 updates, the shared 63 input columns
    # had effectively lost the normalization under which the copied velocity
    # actor was trained.  The 20 new columns remain identity-normalized and can
    # still learn through their initially-zero first-layer weights.
    result["obs_normalizer.count"] = target_count.new_tensor(source_count_value)
    return result


def load_velocity_actor_bootstrap(
    target_actor: torch.nn.Module,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
) -> VelocityActorBootstrapProvenance:
    """Verify a pinned checkpoint and install only its mapped actor weights."""

    if _SHA256_RE.fullmatch(checkpoint_sha256) is None:
        raise ValueError("Velocity bootstrap SHA-256 must be 64 lowercase hex digits")
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Velocity bootstrap checkpoint not found: {checkpoint}"
        )
    actual_sha256 = _sha256_file(checkpoint)
    if actual_sha256 != checkpoint_sha256:
        raise ValueError(
            "Velocity bootstrap checkpoint SHA-256 mismatch: "
            f"expected {checkpoint_sha256}, got {actual_sha256}"
        )

    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(loaded, dict):
        raise TypeError("Velocity bootstrap checkpoint must contain a dictionary")
    velocity_actor_state = loaded.get("actor_state_dict")
    if not isinstance(velocity_actor_state, Mapping):
        raise TypeError("Velocity bootstrap checkpoint has no actor_state_dict")

    source_count = velocity_actor_state.get("obs_normalizer.count")
    if not isinstance(source_count, torch.Tensor) or source_count.ndim != 0:
        raise ValueError("Velocity actor normalizer count must be a scalar Tensor")
    mapped = bootstrap_teleop_actor_state(
        target_actor.state_dict(),
        velocity_actor_state,
    )
    target_actor.load_state_dict(mapped, strict=True)
    return VelocityActorBootstrapProvenance(
        checkpoint_path=checkpoint,
        checkpoint_sha256=actual_sha256,
        source_normalizer_count=float(source_count.item()),
        installed_normalizer_count=float(source_count.item()),
        shoulder_roll_initial_latent_biases=(
            TELEOP_SHOULDER_ROLL_INITIAL_LATENT_BIASES
        ),
    )
