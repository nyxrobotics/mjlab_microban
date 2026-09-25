"""Fail-closed loader for bounded Microban safe-velocity checkpoints."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from mjlab_microban.tasks.microban_safe_velocity_env_cfg import (
    MICROBAN_SAFE_VELOCITY_JOINT_NAMES,
    MICROBAN_SAFE_VELOCITY_OBSERVATION_SCHEMA,
    MICROBAN_SAFE_VELOCITY_OBSERVATION_WIDTH,
    microban_safe_velocity_action_delta_bounds,
    microban_safe_velocity_initial_action_std,
)
from mjlab_microban.tasks.microban_safe_velocity_mdp import (
    MICROBAN_SAFE_VELOCITY_RECIPE_INFO_KEY,
    MICROBAN_SAFE_VELOCITY_RECIPE_REVISION,
    MICROBAN_SAFE_VELOCITY_RESUME_PARENT_INFO_KEY,
    MicrobanSafeVelocityBoundedGaussianDistribution,
)

MICROBAN_SAFE_VELOCITY_CHECKPOINT_SCHEMA_VERSION = 2
MICROBAN_SAFE_VELOCITY_ACTOR_TOPOLOGY = (63, 512, 256, 128, 18)
_CHECKPOINT_RE = re.compile(r"model_(\d+)[.]pt")


@dataclass(frozen=True)
class SafeVelocityCheckpointIdentity:
    """Auditable identity and wire contract of one verified checkpoint."""

    path: Path
    sha256: str
    iteration: int
    schema_version: int
    recipe_revision: str
    actor_topology: tuple[int, ...]
    observation_schema: tuple[tuple[str, int], ...]
    action_joint_names: tuple[str, ...]
    actor_obs_normalization: bool
    common_step_counter: int
    resume_parent_path: str | None
    resume_parent_sha256: str | None
    resume_parent_iteration: int | None


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_safe_velocity_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> SafeVelocityCheckpointIdentity:
    """Reject legacy/unbounded or shape-compatible-but-incompatible actors."""

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Safe velocity checkpoint not found: {path}")
    match = _CHECKPOINT_RE.fullmatch(path.name)
    if match is None:
        raise ValueError("Safe velocity checkpoint must be named model_<iteration>.pt")
    filename_iteration = int(match.group(1))
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(
            f"Safe velocity checkpoint SHA-256 mismatch: {digest} != {expected_sha256}"
        )

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("Safe velocity checkpoint root must be a mapping")
    infos = payload.get("infos")
    recipe_revision = (
        infos.get(MICROBAN_SAFE_VELOCITY_RECIPE_INFO_KEY)
        if isinstance(infos, Mapping)
        else None
    )
    if recipe_revision != MICROBAN_SAFE_VELOCITY_RECIPE_REVISION:
        raise ValueError(
            "Safe velocity checkpoint recipe mismatch: "
            f"{recipe_revision!r} != {MICROBAN_SAFE_VELOCITY_RECIPE_REVISION!r}"
        )
    iteration = payload.get("iter")
    if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < 0:
        raise ValueError("Safe velocity checkpoint must contain integer iter >= 0")
    if iteration != filename_iteration:
        raise ValueError(
            "Safe velocity checkpoint filename/internal iteration mismatch: "
            f"{filename_iteration} != {iteration}"
        )
    env_state = infos.get("env_state") if isinstance(infos, Mapping) else None
    common_step_counter = (
        env_state.get("common_step_counter") if isinstance(env_state, Mapping) else None
    )
    if (
        not isinstance(common_step_counter, int)
        or isinstance(common_step_counter, bool)
        or common_step_counter != (iteration + 1) * 24
    ):
        raise ValueError(
            "Safe velocity checkpoint iteration/step mismatch: "
            f"{common_step_counter!r} != {(iteration + 1) * 24}"
        )
    resume_parent = infos.get(MICROBAN_SAFE_VELOCITY_RESUME_PARENT_INFO_KEY)
    resume_parent_path: str | None = None
    resume_parent_sha256: str | None = None
    resume_parent_iteration: int | None = None
    if resume_parent is not None:
        if not isinstance(resume_parent, Mapping) or set(resume_parent) != {
            "path",
            "sha256",
            "iteration",
        }:
            raise ValueError("Safe velocity resume parent provenance is malformed")
        resume_parent_path = resume_parent["path"]
        resume_parent_sha256 = resume_parent["sha256"]
        resume_parent_iteration = resume_parent["iteration"]
        if not isinstance(resume_parent_path, str) or not resume_parent_path:
            raise ValueError("Safe velocity resume parent path is invalid")
        if (
            not isinstance(resume_parent_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", resume_parent_sha256) is None
        ):
            raise ValueError("Safe velocity resume parent SHA-256 is invalid")
        if (
            not isinstance(resume_parent_iteration, int)
            or isinstance(resume_parent_iteration, bool)
            or resume_parent_iteration < 0
            or resume_parent_iteration >= iteration
        ):
            raise ValueError("Safe velocity resume parent iteration is invalid")
    actor_state = payload.get("actor_state_dict")
    if not isinstance(actor_state, Mapping):
        raise TypeError("Safe velocity checkpoint has no actor_state_dict mapping")

    expected_shapes = {
        "mlp.0.weight": (512, 63),
        "mlp.0.bias": (512,),
        "mlp.2.weight": (256, 512),
        "mlp.2.bias": (256,),
        "mlp.4.weight": (128, 256),
        "mlp.4.bias": (128,),
        "mlp.6.weight": (18, 128),
        "mlp.6.bias": (18,),
        "distribution.log_std_param": (18,),
        "distribution.lower_bound": (18,),
        "distribution.upper_bound": (18,),
        "distribution.inward_lower_bound": (18,),
        "distribution.inward_upper_bound": (18,),
        "distribution.operational_lower_bound": (18,),
        "distribution.operational_upper_bound": (18,),
        "distribution.mean_lower_bound": (18,),
        "distribution.mean_upper_bound": (18,),
        "distribution.min_std": (18,),
        "distribution.max_std": (18,),
        "distribution.operational_action_lower": (18,),
        "distribution.operational_action_upper": (18,),
    }
    if any(key.startswith("obs_normalizer.") for key in actor_state):
        raise ValueError(
            "Safe velocity actor must not contain an observation normalizer"
        )
    actual_keys = set(actor_state)
    expected_keys = set(expected_shapes)
    if actual_keys != expected_keys:
        raise ValueError(
            "Safe velocity actor state key set differs from the exact contract: "
            f"missing={sorted(expected_keys - actual_keys)}, "
            f"unexpected={sorted(actual_keys - expected_keys)}"
        )
    for key, shape in expected_shapes.items():
        value = actor_state.get(key)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(
                f"Safe velocity actor state {key!r} must have shape {shape}, "
                f"got {getattr(value, 'shape', None)}"
            )
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"Safe velocity actor state {key!r} is non-finite")
    expected_lower, expected_upper = microban_safe_velocity_action_delta_bounds()
    expected_distribution = MicrobanSafeVelocityBoundedGaussianDistribution(
        18,
        microban_safe_velocity_initial_action_std(),
        expected_lower,
        expected_upper,
    )
    for suffix, expected_value in expected_distribution.state_dict().items():
        if suffix == "log_std_param":
            continue
        key = f"distribution.{suffix}"
        value = actor_state[key]
        assert isinstance(value, torch.Tensor)
        expected_tensor = expected_value.to(dtype=value.dtype, device="cpu")
        if not bool(torch.equal(value.cpu(), expected_tensor)):
            max_error = float(torch.abs(value.cpu() - expected_tensor).max().item())
            raise ValueError(
                f"Safe velocity actor {key} differs from the exact distribution "
                "contract "
                f"(max error {max_error:.9g})"
            )
    log_std = actor_state["distribution.log_std_param"]
    min_std = actor_state["distribution.min_std"]
    max_std = actor_state["distribution.max_std"]
    assert isinstance(log_std, torch.Tensor)
    assert isinstance(min_std, torch.Tensor)
    assert isinstance(max_std, torch.Tensor)
    std = torch.exp(log_std)
    if not bool(torch.all((std >= min_std) & (std <= max_std)).item()):
        raise ValueError("Safe velocity checkpoint exploration std is out of bounds")

    if sha256_file(path) != digest:
        raise RuntimeError("Safe velocity checkpoint changed during inspection")
    return SafeVelocityCheckpointIdentity(
        path=path,
        sha256=digest,
        iteration=iteration,
        schema_version=MICROBAN_SAFE_VELOCITY_CHECKPOINT_SCHEMA_VERSION,
        recipe_revision=MICROBAN_SAFE_VELOCITY_RECIPE_REVISION,
        actor_topology=MICROBAN_SAFE_VELOCITY_ACTOR_TOPOLOGY,
        observation_schema=MICROBAN_SAFE_VELOCITY_OBSERVATION_SCHEMA,
        action_joint_names=MICROBAN_SAFE_VELOCITY_JOINT_NAMES,
        actor_obs_normalization=False,
        common_step_counter=common_step_counter,
        resume_parent_path=resume_parent_path,
        resume_parent_sha256=resume_parent_sha256,
        resume_parent_iteration=resume_parent_iteration,
    )


def load_frozen_safe_velocity_actor(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    expected_sha256: str | None = None,
) -> tuple[MLPModel, SafeVelocityCheckpointIdentity]:
    """Load a deterministic, bounded 63->18 actor after contract validation."""

    identity = inspect_safe_velocity_checkpoint(
        checkpoint_path, expected_sha256=expected_sha256
    )
    torch_device = torch.device(device)
    observation = TensorDict(
        {
            "actor": torch.zeros(
                (1, MICROBAN_SAFE_VELOCITY_OBSERVATION_WIDTH),
                dtype=torch.float32,
                device=torch_device,
            )
        },
        batch_size=[1],
        device=torch_device,
    )
    lower, upper = microban_safe_velocity_action_delta_bounds()
    actor = MLPModel(
        obs=observation,
        obs_groups={"actor": ["actor"]},
        obs_set="actor",
        output_dim=18,
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=False,
        distribution_cfg={
            "class_name": MicrobanSafeVelocityBoundedGaussianDistribution,
            "init_std": microban_safe_velocity_initial_action_std(),
            "lower_bound": lower,
            "upper_bound": upper,
            "std_type": "log",
        },
    ).to(torch_device)
    payload = torch.load(identity.path, map_location=torch_device, weights_only=True)
    actor_state = payload["actor_state_dict"]
    actor.load_state_dict(actor_state, strict=True)
    actor.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    if sha256_file(identity.path) != identity.sha256:
        raise RuntimeError("Safe velocity checkpoint changed while loading")
    return actor, identity
