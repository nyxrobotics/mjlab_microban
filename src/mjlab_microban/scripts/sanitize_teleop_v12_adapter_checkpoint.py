"""Clone a pre-schedule v12 checkpoint with the inactive adapter state zeroed.

This is a one-way migration for checkpoints produced before the staged adapter
gradient mask existed.  It never overwrites its source.  The legacy/shared
actor, critic, PPO counters, and critic optimizer state remain bit-identical;
only the 20 teleop-only W0 columns and their Adam moments are reset to zero.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from mjlab_microban.legacy_velocity_diagnostics import publish_json_atomic
from mjlab_microban.tasks.microban_teleop_v12_actor import (
    TELEOP_V12_ADAPTER_GRADIENT_SCHEDULE_REVISION,
    TELEOP_V12_ADAPTER_SANITIZATION_REVISION,
    TELEOP_V12_ADAPTER_SANITIZATION_SCHEMA_VERSION,
    TELEOP_V12_BOOTSTRAP_MAPPING_VERSION,
    TELEOP_V12_EXTRA_OBSERVATION_COLUMNS,
    TELEOP_V12_SHARED_OBSERVATION_COLUMNS,
    TELEOP_V12_TARGET_POSITION_OBSERVATION_COLUMNS,
    teleop_v12_target_normalizer_metadata,
    transplant_legacy_actor_state_to_teleop83,
)
from mjlab_microban.tasks.microban_teleop_v12_bootstrap import (
    inspect_legacy_velocity_checkpoint,
    serialize_bootstrap_provenance,
    sha256_file,
    validate_identity_normalizer_v1_bootstrap_provenance,
)
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MICROBAN_TELEOP_V12_RECIPE_REVISION,
)
from mjlab_microban.tasks.microban_teleop_v12_runner import (
    TELEOP_V12_BOOTSTRAP_INFO_KEY,
    _atomic_torch_save,
)

SANITIZATION_REVISION = TELEOP_V12_ADAPTER_SANITIZATION_REVISION
UNSAFE_PRE_SCHEDULE_RECIPE_REVISION = (
    "legacy_velocity_model14999_masked_extra20_raw_actions_v1"
)


def _zero_extra_columns_in_place(tensor: torch.Tensor) -> None:
    """Reset adapter columns on the original tensor, never an indexed copy."""

    if tuple(tensor.shape) != (512, 83):
        raise ValueError("Adapter reset requires a 512x83 tensor")
    columns = torch.tensor(
        TELEOP_V12_EXTRA_OBSERVATION_COLUMNS,
        dtype=torch.long,
        device=tensor.device,
    )
    tensor.index_fill_(1, columns, 0.0)


def _require_tensor_state(
    payload: dict[str, Any], name: str
) -> dict[str, torch.Tensor]:
    value = payload.get(name)
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(tensor, torch.Tensor)
        for key, tensor in value.items()
    ):
        raise TypeError(f"Checkpoint {name} is malformed")
    return value


def _authenticate_identity_normalizer_v1_actor(
    actor_state: dict[str, torch.Tensor], provenance: Any
) -> dict[str, torch.Tensor]:
    """Authenticate old v1 state and return the exact current scaled state."""

    source, source_state = inspect_legacy_velocity_checkpoint(
        provenance.source.path, provenance.source.sha256
    )
    if source != provenance.source:
        raise ValueError("Legacy source identity no longer matches v1 provenance")
    scaled_expected = transplant_legacy_actor_state_to_teleop83(
        source_state, actor_state
    )
    identity_expected = {
        name: value.detach().clone() for name, value in scaled_expected.items()
    }
    extra = torch.tensor(TELEOP_V12_EXTRA_OBSERVATION_COLUMNS, dtype=torch.long)
    for name, fill in (
        ("obs_normalizer._mean", 0.0),
        ("obs_normalizer._var", 1.0),
        ("obs_normalizer._std", 1.0),
    ):
        tensor = identity_expected[name]
        tensor[:, extra.to(tensor.device)] = fill

    for name, expected in identity_expected.items():
        candidate = actor_state[name].detach().to(device=expected.device)
        if name == "mlp.0.weight":
            candidate = candidate[:, TELEOP_V12_SHARED_OBSERVATION_COLUMNS]
            expected = expected[:, TELEOP_V12_SHARED_OBSERVATION_COLUMNS]
        if not torch.equal(candidate, expected):
            if name.startswith("obs_normalizer."):
                raise ValueError(
                    "Unsafe v1 source does not have the authenticated identity "
                    f"normalizer: {name}"
                )
            raise ValueError(
                f"Unsafe v1 actor differs from pinned legacy state: {name}"
            )
    return scaled_expected


def sanitize_checkpoint(source: Path, destination: Path) -> dict[str, Any]:
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if source == destination:
        raise ValueError("Sanitizer refuses to overwrite its source checkpoint")
    if destination.exists():
        raise FileExistsError(f"Sanitized destination already exists: {destination}")
    source_sha256 = sha256_file(source)
    original = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(original, dict) or not isinstance(original.get("infos"), dict):
        raise TypeError("Source checkpoint payload is malformed")
    infos = original["infos"]
    if infos.get("microban_teleop_training_contract_version") != "12":
        raise ValueError("Source checkpoint is not contract-v12")
    if (
        infos.get("microban_teleop_recipe_revision")
        != (UNSAFE_PRE_SCHEDULE_RECIPE_REVISION)
        or "adapter_gradient_schedule_revision" in infos
    ):
        raise ValueError("Source is not an authenticated pre-schedule v12 checkpoint")
    if infos.get("previous_action_semantics") != "raw_actor_output" or (
        infos.get("action_clip", object()) is not None
    ):
        raise ValueError("Source checkpoint is not raw-action contract-v12")
    provenance = validate_identity_normalizer_v1_bootstrap_provenance(
        infos.get(TELEOP_V12_BOOTSTRAP_INFO_KEY), verify_files=True
    )
    iteration = original.get("iter")
    if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < 0:
        raise ValueError("Source checkpoint iteration is invalid")
    completed = iteration + 1
    if completed >= 7_000:
        raise ValueError("Sanitizer is only valid before adapter activation at 7000")
    env_state = infos.get("env_state")
    if (
        not isinstance(env_state, dict)
        or env_state.get("common_step_counter") != completed * 24
    ):
        raise ValueError("Source checkpoint iteration/common-step relation drifted")

    actor_state = _require_tensor_state(original, "actor_state_dict")
    critic_state = _require_tensor_state(original, "critic_state_dict")
    if any(not bool(value.isfinite().all().item()) for value in critic_state.values()):
        raise ValueError("Source critic state is non-finite")
    scaled_expected_actor = _authenticate_identity_normalizer_v1_actor(
        actor_state, provenance
    )
    first = actor_state.get("mlp.0.weight")
    if not isinstance(first, torch.Tensor) or tuple(first.shape) != (512, 83):
        raise ValueError("Source actor W0 shape drifted")
    if not bool(torch.isfinite(first).all().item()):
        raise ValueError("Source actor W0 is non-finite")

    migrated = copy.deepcopy(original)
    migrated_actor = _require_tensor_state(migrated, "actor_state_dict")
    migrated_first = migrated_actor["mlp.0.weight"]
    maximum_before = float(
        migrated_first[:, TELEOP_V12_EXTRA_OBSERVATION_COLUMNS].abs().max().item()
    )
    _zero_extra_columns_in_place(migrated_first)
    target_position_columns = torch.tensor(
        TELEOP_V12_TARGET_POSITION_OBSERVATION_COLUMNS, dtype=torch.long
    )
    for name in (
        "obs_normalizer._mean",
        "obs_normalizer._var",
        "obs_normalizer._std",
    ):
        destination_tensor = migrated_actor[name]
        source_tensor = scaled_expected_actor[name]
        indices = target_position_columns.to(destination_tensor.device)
        destination_tensor[:, indices] = source_tensor.to(
            device=destination_tensor.device, dtype=destination_tensor.dtype
        )[:, indices]

    optimizer = migrated.get("optimizer_state_dict")
    if not isinstance(optimizer, dict) or not isinstance(optimizer.get("state"), dict):
        raise TypeError("Source optimizer state is malformed")
    candidates: list[tuple[object, dict[str, Any]]] = []
    for parameter_id, state in optimizer["state"].items():
        if not isinstance(state, dict):
            raise TypeError("Source optimizer parameter state is malformed")
        if any(
            isinstance(value, torch.Tensor) and tuple(value.shape) == (512, 83)
            for value in state.values()
        ):
            candidates.append((parameter_id, state))
    if len(candidates) != 1:
        raise ValueError(
            "Could not uniquely identify the 83-wide actor W0 optimizer state"
        )
    actor_parameter_id, actor_optimizer_state = candidates[0]
    zeroed_moments: list[str] = []
    for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        moment = actor_optimizer_state.get(name)
        if moment is None:
            continue
        if not isinstance(moment, torch.Tensor) or tuple(moment.shape) != (512, 83):
            raise ValueError(f"Actor optimizer {name} shape drifted")
        if not bool(torch.isfinite(moment).all().item()):
            raise ValueError(f"Actor optimizer {name} is non-finite")
        shared = moment[:, TELEOP_V12_SHARED_OBSERVATION_COLUMNS]
        if not torch.equal(shared, torch.zeros_like(shared)):
            raise ValueError(f"Actor optimizer {name} already mutated a shared column")
        _zero_extra_columns_in_place(moment)
        zeroed_moments.append(name)
    if set(zeroed_moments) != {"exp_avg", "exp_avg_sq"}:
        raise ValueError("Adam first and second moments were not both present")

    original_optimizer = original["optimizer_state_dict"]
    if optimizer.get("param_groups") != original_optimizer.get("param_groups"):
        raise RuntimeError("Sanitizer changed optimizer parameter groups")
    if optimizer["state"].keys() != original_optimizer["state"].keys():
        raise RuntimeError("Sanitizer changed optimizer parameter-state keys")
    for parameter_id, original_state in original_optimizer["state"].items():
        migrated_state = optimizer["state"][parameter_id]
        if original_state.keys() != migrated_state.keys():
            raise RuntimeError("Sanitizer changed optimizer state fields")
        for name, original_value in original_state.items():
            migrated_value = migrated_state[name]
            if isinstance(original_value, torch.Tensor):
                if not bool(torch.isfinite(original_value).all().item()):
                    raise ValueError("Source optimizer contains a non-finite tensor")
                if parameter_id == actor_parameter_id and name in zeroed_moments:
                    original_value = original_value[
                        :, TELEOP_V12_SHARED_OBSERVATION_COLUMNS
                    ]
                    migrated_value = migrated_value[
                        :, TELEOP_V12_SHARED_OBSERVATION_COLUMNS
                    ]
                if not torch.equal(original_value, migrated_value):
                    raise RuntimeError(
                        "Sanitizer changed optimizer state outside extra W0 moments"
                    )
            elif original_value != migrated_value:
                raise RuntimeError("Sanitizer changed scalar optimizer state")

    migrated_infos = migrated["infos"]
    migrated_infos["adapter_gradient_schedule_revision"] = (
        TELEOP_V12_ADAPTER_GRADIENT_SCHEDULE_REVISION
    )
    migrated_infos["microban_teleop_recipe_revision"] = (
        MICROBAN_TELEOP_V12_RECIPE_REVISION
    )
    migrated_infos[TELEOP_V12_BOOTSTRAP_INFO_KEY] = serialize_bootstrap_provenance(
        replace(provenance, mapping_version=TELEOP_V12_BOOTSTRAP_MAPPING_VERSION)
    )
    migrated_infos["active_actor_columns_at_save"] = []
    migrated_infos["adapter_sanitization"] = {
        "schema_version": TELEOP_V12_ADAPTER_SANITIZATION_SCHEMA_VERSION,
        "revision": SANITIZATION_REVISION,
        "parent_checkpoint_filename": source.name,
        "parent_checkpoint_sha256": source_sha256,
        "parent_iteration": iteration,
        "completed_updates": completed,
        "zeroed_actor_columns": list(TELEOP_V12_EXTRA_OBSERVATION_COLUMNS),
        "zeroed_optimizer_parameter_id": actor_parameter_id,
        "zeroed_optimizer_moments": zeroed_moments,
        "maximum_absolute_extra_w0_before": maximum_before,
        **teleop_v12_target_normalizer_metadata(),
    }

    # Explicitly prove all other model tensors were preserved by the clone.
    for name, value in actor_state.items():
        candidate = migrated_actor[name]
        if name == "mlp.0.weight":
            value = value[:, TELEOP_V12_SHARED_OBSERVATION_COLUMNS]
            candidate = candidate[:, TELEOP_V12_SHARED_OBSERVATION_COLUMNS]
        elif name in {
            "obs_normalizer._mean",
            "obs_normalizer._var",
            "obs_normalizer._std",
        }:
            value = scaled_expected_actor[name]
        if not torch.equal(value, candidate):
            raise RuntimeError(f"Sanitizer changed protected actor tensor {name}")
    migrated_critic = _require_tensor_state(migrated, "critic_state_dict")
    if any(
        not torch.equal(value, migrated_critic[name])
        for name, value in critic_state.items()
    ):
        raise RuntimeError("Sanitizer changed critic state")

    _atomic_torch_save(migrated, destination)
    if sha256_file(source) != source_sha256:
        destination.unlink(missing_ok=True)
        raise RuntimeError("Source checkpoint changed during sanitization")
    verified = torch.load(destination, map_location="cpu", weights_only=False)
    verified_first = verified["actor_state_dict"]["mlp.0.weight"]
    if not torch.equal(
        verified_first[:, TELEOP_V12_EXTRA_OBSERVATION_COLUMNS],
        torch.zeros_like(verified_first[:, TELEOP_V12_EXTRA_OBSERVATION_COLUMNS]),
    ):
        destination.unlink(missing_ok=True)
        raise RuntimeError("Sanitized checkpoint did not retain zero adapter weights")
    verified_actor = _require_tensor_state(verified, "actor_state_dict")
    for name in (
        "obs_normalizer._mean",
        "obs_normalizer._var",
        "obs_normalizer._std",
    ):
        if not torch.equal(verified_actor[name], scaled_expected_actor[name]):
            destination.unlink(missing_ok=True)
            raise RuntimeError("Sanitized target-position normalizer did not persist")
    return {
        "schema_version": TELEOP_V12_ADAPTER_SANITIZATION_SCHEMA_VERSION,
        "sanitizer": SANITIZATION_REVISION,
        "status": "pass",
        "source": {
            "path": str(source),
            "sha256": source_sha256,
            "iteration": iteration,
            "completed_updates": completed,
        },
        "output": {
            "path": str(destination),
            "sha256": sha256_file(destination),
            "iteration": iteration,
            "completed_updates": completed,
        },
        "checks": {
            "legacy_shared_actor_unchanged": True,
            "critic_unchanged": True,
            "extra_w0_zero": True,
            "extra_adam_moments_zero": True,
            "old_identity_normalizer_authenticated": True,
            "target_position_normalizer_scaled": True,
            "source_not_overwritten": True,
        },
        "normalizer_migration": teleop_v12_target_normalizer_metadata(),
        "maximum_absolute_extra_w0_before": maximum_before,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = sanitize_checkpoint(args.source, args.destination)
    if args.output is not None:
        if args.output.expanduser().exists():
            raise FileExistsError(f"Receipt already exists: {args.output}")
        publish_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
