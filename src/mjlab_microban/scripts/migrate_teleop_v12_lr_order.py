"""Migrate a contract-v12 checkpoint across the bilateral site-order fix.

Before the fix, Microban's right-before-left XML site order crossed the reward's
measured hand/foot offsets against left-before-right command tensors.  This
offline, CPU-only migration composes the learned networks with the exact L/R
input permutation.  It also permutes Adam moments and observation-normalizer
state so the checkpoint can continue training without resetting its clocks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from mjlab_microban.tasks.mdp import MICROBAN_BILATERAL_SITE_ORDER_REVISION
from mjlab_microban.tasks.microban_teleop_v12_actor import (
    TELEOP_V12_FOOT_OBSERVATION_COLUMNS,
    TELEOP_V12_HAND_OBSERVATION_COLUMNS,
)
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MICROBAN_TELEOP_V12_TRAINING_CONTRACT_VERSION,
)
from mjlab_microban.tasks.microban_teleop_v12_lr_order import (
    ACTOR_PERMUTATION,
    ACTOR_SWAP_BLOCKS,
    BILATERAL_SITE_ORDER_INFO_KEY,
    CRITIC_HAND_ACTIVE_OBSERVATION_COLUMNS,
    CRITIC_HAND_POSITION_OBSERVATION_COLUMNS,
    CRITIC_PERMUTATION,
    CRITIC_SWAP_BLOCKS,
    MIGRATION_INFO_KEY,
    MIGRATION_REVISION,
    MIGRATION_STRATEGIES,
    validate_lr_order_migration_marker,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _permute_last_dimension(tensor: torch.Tensor, permutation: tuple[int, ...]) -> None:
    if tensor.ndim == 0 or tensor.shape[-1] != len(permutation):
        raise ValueError(
            f"Cannot apply {len(permutation)}-column permutation to {tuple(tensor.shape)}"
        )
    index = torch.tensor(permutation, dtype=torch.long, device=tensor.device)
    tensor.copy_(tensor.index_select(-1, index))


def _normalizer_tensors(state: dict[str, Any], width: int) -> tuple[torch.Tensor, ...]:
    names = (
        "obs_normalizer._mean",
        "obs_normalizer._var",
        "obs_normalizer._std",
    )
    tensors: list[torch.Tensor] = []
    for name in names:
        value = state.get(name)
        if not isinstance(value, torch.Tensor) or value.shape[-1:] != (width,):
            raise ValueError(f"Checkpoint {name} is not a [..., {width}] tensor")
        tensors.append(value)
    return tuple(tensors)


def _optimizer_state_for_shape(
    optimizer: dict[str, Any], shape: torch.Size
) -> tuple[int, dict[str, Any]]:
    states = optimizer.get("state")
    if not isinstance(states, dict):
        raise TypeError("Checkpoint optimizer state is missing")
    candidates: list[tuple[int, dict[str, Any]]] = []
    for raw_parameter_id, raw_state in states.items():
        if not isinstance(raw_parameter_id, int) or not isinstance(raw_state, dict):
            continue
        exp_avg = raw_state.get("exp_avg")
        exp_avg_sq = raw_state.get("exp_avg_sq")
        if (
            isinstance(exp_avg, torch.Tensor)
            and isinstance(exp_avg_sq, torch.Tensor)
            and exp_avg.shape == shape
            and exp_avg_sq.shape == shape
        ):
            candidates.append((raw_parameter_id, raw_state))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one Adam state with shape {tuple(shape)}, got {len(candidates)}"
        )
    return candidates[0]


def _maximum_absolute(tensor: torch.Tensor, columns: tuple[int, ...]) -> float:
    return float(tensor[..., columns].abs().max().item())


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _tensor_leaves(value: Any, path: str = "") -> dict[str, torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return {path: value}
    if isinstance(value, dict):
        result: dict[str, torch.Tensor] = {}
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            result.update(_tensor_leaves(item, child))
        return result
    if isinstance(value, (list, tuple)):
        result = {}
        for index, item in enumerate(value):
            child = f"{path}[{index}]"
            result.update(_tensor_leaves(item, child))
        return result
    return {}


def _tensor_integrity_evidence(
    source: dict[str, Any],
    migrated: dict[str, Any],
    *,
    actor_optimizer_id: int,
    critic_optimizer_id: int,
) -> dict[str, Any]:
    """Prove every non-target tensor and every untouched column is bit-identical."""

    source_tensors = _tensor_leaves(source)
    migrated_tensors = _tensor_leaves(migrated)
    if set(source_tensors) != set(migrated_tensors):
        raise RuntimeError("Migration changed the checkpoint tensor-key set")
    actor_hand = tuple(TELEOP_V12_HAND_OBSERVATION_COLUMNS)
    critic_hand = (
        *CRITIC_HAND_POSITION_OBSERVATION_COLUMNS,
        *CRITIC_HAND_ACTIVE_OBSERVATION_COLUMNS,
    )
    partial_columns: dict[str, tuple[int, ...]] = {
        "actor_state_dict.mlp.0.weight": actor_hand,
        "actor_state_dict.obs_normalizer._mean": actor_hand,
        "actor_state_dict.obs_normalizer._var": actor_hand,
        "actor_state_dict.obs_normalizer._std": actor_hand,
        "critic_state_dict.mlp.0.weight": critic_hand,
        "critic_state_dict.obs_normalizer._mean": critic_hand,
        "critic_state_dict.obs_normalizer._var": critic_hand,
        "critic_state_dict.obs_normalizer._std": critic_hand,
        (f"optimizer_state_dict.state.{actor_optimizer_id}.exp_avg"): actor_hand,
        (f"optimizer_state_dict.state.{actor_optimizer_id}.exp_avg_sq"): actor_hand,
        (f"optimizer_state_dict.state.{critic_optimizer_id}.exp_avg"): critic_hand,
        (f"optimizer_state_dict.state.{critic_optimizer_id}.exp_avg_sq"): critic_hand,
    }
    unchanged: dict[str, str] = {}
    partially_transformed: dict[str, Any] = {}
    for path, before in source_tensors.items():
        after = migrated_tensors[path]
        if before.shape != after.shape or before.dtype != after.dtype:
            raise RuntimeError(f"Migration changed tensor shape/dtype: {path}")
        changed_columns = partial_columns.get(path)
        if changed_columns is None:
            if not torch.equal(before, after):
                raise RuntimeError(f"Migration changed an out-of-scope tensor: {path}")
            unchanged[path] = _tensor_sha256(before)
            continue
        changed = set(changed_columns)
        untouched_columns = tuple(
            index for index in range(before.shape[-1]) if index not in changed
        )
        before_untouched = before[..., untouched_columns]
        after_untouched = after[..., untouched_columns]
        if not torch.equal(before_untouched, after_untouched):
            raise RuntimeError(f"Migration changed out-of-scope columns: {path}")
        untouched_sha256 = _tensor_sha256(before_untouched)
        partially_transformed[path] = {
            "changed_columns": list(changed_columns),
            "untouched_column_count": len(untouched_columns),
            "untouched_source_sha256": untouched_sha256,
            "untouched_output_sha256": _tensor_sha256(after_untouched),
            "source_full_sha256": _tensor_sha256(before),
            "output_full_sha256": _tensor_sha256(after),
        }
    if set(partially_transformed) != set(partial_columns):
        missing = sorted(set(partial_columns) - set(partially_transformed))
        raise RuntimeError(f"Migration tensor contract is missing tensors: {missing}")
    return {
        "digest_definition": "sha256(dtype_nul_shape_json_nul_contiguous_raw_bytes)",
        "unchanged_tensor_count": len(unchanged),
        "unchanged_tensors": unchanged,
        "partially_transformed_tensors": partially_transformed,
        "passed": True,
    }


def migrate_checkpoint_payload(
    source_payload: dict[str, Any],
    *,
    source_path: Path,
    source_sha256: str,
    strategy: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a migrated deep copy and JSON-safe transform evidence."""

    if strategy not in MIGRATION_STRATEGIES:
        raise ValueError(f"Unsupported migration strategy: {strategy!r}")
    payload = deepcopy(source_payload)
    if not isinstance(payload.get("iter"), int) or isinstance(payload["iter"], bool):
        raise TypeError("Contract-v12 checkpoint iter must be an integer")
    infos = payload.get("infos")
    if not isinstance(infos, dict):
        raise TypeError("Contract-v12 checkpoint infos are missing")
    if infos.get("microban_teleop_training_contract_version") != (
        MICROBAN_TELEOP_V12_TRAINING_CONTRACT_VERSION
    ):
        raise ValueError("Checkpoint is not contract-v12")
    if MIGRATION_INFO_KEY in infos:
        raise ValueError("Checkpoint already carries an L/R-order migration marker")
    env_state = infos.get("env_state")
    if not isinstance(env_state, dict) or not isinstance(
        env_state.get("common_step_counter"), int
    ):
        raise TypeError("Checkpoint environment clock is missing")

    actor = payload.get("actor_state_dict")
    critic = payload.get("critic_state_dict")
    optimizer = payload.get("optimizer_state_dict")
    if not isinstance(actor, dict) or not isinstance(critic, dict):
        raise TypeError("Checkpoint actor/critic state is missing")
    if not isinstance(optimizer, dict):
        raise TypeError("Checkpoint optimizer state is missing")
    actor_w0 = actor.get("mlp.0.weight")
    critic_w0 = critic.get("mlp.0.weight")
    if not isinstance(actor_w0, torch.Tensor) or actor_w0.shape != (512, 83):
        raise ValueError("Contract-v12 actor W0 must have shape [512, 83]")
    if not isinstance(critic_w0, torch.Tensor) or critic_w0.shape != (512, 137):
        raise ValueError("Contract-v12 critic W0 must have shape [512, 137]")

    actor_optimizer_id, actor_optimizer = _optimizer_state_for_shape(
        optimizer, actor_w0.shape
    )
    critic_optimizer_id, critic_optimizer = _optimizer_state_for_shape(
        optimizer, critic_w0.shape
    )
    actor_normalizers = _normalizer_tensors(actor, 83)
    critic_normalizers = _normalizer_tensors(critic, 137)

    source_foot_max = _maximum_absolute(actor_w0, TELEOP_V12_FOOT_OBSERVATION_COLUMNS)
    source_foot_adam_max = max(
        _maximum_absolute(actor_optimizer[name], TELEOP_V12_FOOT_OBSERVATION_COLUMNS)
        for name in ("exp_avg", "exp_avg_sq")
    )
    active_columns = infos.get("active_actor_columns_at_save")
    foot_was_inactive = isinstance(active_columns, list) and not (
        set(TELEOP_V12_FOOT_OBSERVATION_COLUMNS) & set(active_columns)
    )
    if not foot_was_inactive:
        raise ValueError(
            "This migration is restricted to the pre-foot stage; active foot "
            "columns require a separately evaluated transform"
        )
    if source_foot_max != 0.0 or source_foot_adam_max != 0.0:
        raise ValueError("Inactive foot adapter columns are not exactly zero")

    _permute_last_dimension(actor_w0, ACTOR_PERMUTATION)
    _permute_last_dimension(critic_w0, CRITIC_PERMUTATION)
    for tensor in actor_normalizers:
        _permute_last_dimension(tensor, ACTOR_PERMUTATION)
    for tensor in critic_normalizers:
        _permute_last_dimension(tensor, CRITIC_PERMUTATION)
    for name in ("exp_avg", "exp_avg_sq"):
        _permute_last_dimension(actor_optimizer[name], ACTOR_PERMUTATION)
        _permute_last_dimension(critic_optimizer[name], CRITIC_PERMUTATION)

    zeroed_columns: tuple[int, ...] = ()
    if strategy == "zero_hand":
        zeroed_columns = tuple(TELEOP_V12_HAND_OBSERVATION_COLUMNS)
        actor_w0[:, zeroed_columns] = 0.0
        for name in ("exp_avg", "exp_avg_sq"):
            actor_optimizer[name][:, zeroed_columns] = 0.0

    tensor_integrity = _tensor_integrity_evidence(
        source_payload,
        payload,
        actor_optimizer_id=actor_optimizer_id,
        critic_optimizer_id=critic_optimizer_id,
    )

    clock = {
        "iteration": payload["iter"],
        "completed_updates": payload["iter"] + 1,
        "common_step_counter": env_state["common_step_counter"],
    }
    marker = {
        "schema_version": 1,
        "revision": MIGRATION_REVISION,
        "site_order_revision": MICROBAN_BILATERAL_SITE_ORDER_REVISION,
        "strategy": strategy,
        "source_checkpoint_path": str(source_path),
        "source_checkpoint_sha256": source_sha256,
        "source_clock": clock,
        "actor_w0_optimizer_parameter_id": actor_optimizer_id,
        "critic_w0_optimizer_parameter_id": critic_optimizer_id,
        "actor_swap_blocks": [list(block) for block in ACTOR_SWAP_BLOCKS],
        "critic_swap_blocks": [list(block) for block in CRITIC_SWAP_BLOCKS],
        "actor_permutation": list(ACTOR_PERMUTATION),
        "critic_permutation": list(CRITIC_PERMUTATION),
        "zeroed_actor_columns": list(zeroed_columns),
        "foot_adapter_at_source": {
            "active": not foot_was_inactive,
            "maximum_absolute_w0": source_foot_max,
            "maximum_absolute_adam_moment": source_foot_adam_max,
            "handling": "unlearned_exact_zero_left_untouched",
        },
        "tensor_integrity": tensor_integrity,
    }
    validate_lr_order_migration_marker(marker)
    infos[BILATERAL_SITE_ORDER_INFO_KEY] = MICROBAN_BILATERAL_SITE_ORDER_REVISION
    infos[MIGRATION_INFO_KEY] = marker
    evidence = deepcopy(marker)
    evidence["clock_preserved"] = (
        payload["iter"] == source_payload["iter"]
        and env_state["common_step_counter"]
        == source_payload["infos"]["env_state"]["common_step_counter"]
    )
    return payload, evidence


def _atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        temporary.chmod(0o600)
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_json_save(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            payload, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n"
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def migrate_checkpoint(
    *,
    source: Path,
    expected_sha256: str,
    output: Path,
    receipt: Path,
    strategy: str,
    force: bool,
) -> dict[str, Any]:
    source = source.expanduser().resolve(strict=True)
    output = output.expanduser().resolve()
    receipt = receipt.expanduser().resolve()
    if source in (output, receipt) or output == receipt:
        raise ValueError("Source, output checkpoint, and receipt must be distinct")
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise ValueError("Expected SHA-256 must be 64 lowercase hexadecimal digits")
    source_sha256 = sha256_file(source)
    if source_sha256 != expected_sha256:
        raise ValueError(f"Source checkpoint SHA-256 mismatch: {source_sha256}")
    for destination in (output, receipt):
        if destination.exists() and not force:
            raise FileExistsError(f"Output exists (pass --force): {destination}")
        if destination.is_symlink():
            raise ValueError(f"Output must not be a symlink: {destination}")

    source_payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(source_payload, dict):
        raise TypeError("Checkpoint root must be a dictionary")
    migrated, transform = migrate_checkpoint_payload(
        source_payload,
        source_path=source,
        source_sha256=source_sha256,
        strategy=strategy,
    )
    _atomic_torch_save(migrated, output)
    output_payload = torch.load(output, map_location="cpu", weights_only=False)
    if not isinstance(output_payload, dict):
        raise TypeError("Written checkpoint could not be reloaded")
    output_sha256 = sha256_file(output)
    source_clock = {
        "iteration": source_payload["iter"],
        "common_step_counter": source_payload["infos"]["env_state"][
            "common_step_counter"
        ],
    }
    output_clock = {
        "iteration": output_payload["iter"],
        "common_step_counter": output_payload["infos"]["env_state"][
            "common_step_counter"
        ],
    }
    receipt_payload = {
        "schema_version": 1,
        "gate": "microban_teleop_v12_lr_order_migration",
        "status": "pass" if source_clock == output_clock else "fail",
        "migration_revision": MIGRATION_REVISION,
        "site_order_revision": MICROBAN_BILATERAL_SITE_ORDER_REVISION,
        "strategy": strategy,
        "source_checkpoint": {
            "path": str(source),
            "sha256": source_sha256,
            "size_bytes": source.stat().st_size,
            **source_clock,
        },
        "output_checkpoint": {
            "path": str(output),
            "sha256": output_sha256,
            "size_bytes": output.stat().st_size,
            **output_clock,
        },
        "clock_preservation": {
            "source": source_clock,
            "output": output_clock,
            "passed": source_clock == output_clock,
        },
        "transform": transform,
    }
    if receipt_payload["status"] != "pass":
        output.unlink(missing_ok=True)
        raise RuntimeError("Migration changed the iteration/environment clock")
    _atomic_json_save(receipt_payload, receipt)
    return receipt_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--strategy", choices=MIGRATION_STRATEGIES, default="swap")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = migrate_checkpoint(
        source=args.source,
        expected_sha256=args.expected_sha256,
        output=args.output,
        receipt=args.receipt,
        strategy=args.strategy,
        force=args.force,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
