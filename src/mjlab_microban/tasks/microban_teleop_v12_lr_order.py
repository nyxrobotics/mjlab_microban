"""Authenticated lineage for the contract-v12 bilateral site-order repair."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mjlab_microban.tasks.mdp import MICROBAN_BILATERAL_SITE_ORDER_REVISION
from mjlab_microban.tasks.microban_teleop_v12_actor import (
    TELEOP_V12_HAND_ACTIVE_OBSERVATION_COLUMNS,
    TELEOP_V12_HAND_OBSERVATION_COLUMNS,
    TELEOP_V12_HAND_POSITION_OBSERVATION_COLUMNS,
)

MIGRATION_INFO_KEY = "microban_teleop_v12_lr_order_migration"
MIGRATION_REVISION = "bilateral_site_preserve_order_checkpoint_permutation_v1"
MIGRATION_STRATEGIES = ("swap", "zero_hand")
BILATERAL_SITE_ORDER_INFO_KEY = "bilateral_site_order_revision"
CRITIC_OBSERVATION_WIDTH = 137
CRITIC_HAND_POSITION_OBSERVATION_COLUMNS = tuple(range(90, 96))
CRITIC_HAND_ACTIVE_OBSERVATION_COLUMNS = tuple(range(96, 98))


def _pair_blocks(columns: tuple[int, ...], width: int) -> tuple[tuple[int, ...], ...]:
    if len(columns) != 2 * width:
        raise ValueError("Bilateral column block has the wrong width")
    return (tuple(columns[:width]), tuple(columns[width:]))


ACTOR_SWAP_BLOCKS = (
    *_pair_blocks(TELEOP_V12_HAND_POSITION_OBSERVATION_COLUMNS, 3),
    *_pair_blocks(TELEOP_V12_HAND_ACTIVE_OBSERVATION_COLUMNS, 1),
)
CRITIC_SWAP_BLOCKS = (
    *_pair_blocks(CRITIC_HAND_POSITION_OBSERVATION_COLUMNS, 3),
    *_pair_blocks(CRITIC_HAND_ACTIVE_OBSERVATION_COLUMNS, 1),
)


def bilateral_permutation(
    width: int, blocks: tuple[tuple[int, ...], ...]
) -> tuple[int, ...]:
    """Return a self-inverse permutation that swaps adjacent L/R blocks."""

    if len(blocks) % 2:
        raise ValueError("Swap blocks must be supplied in left/right pairs")
    permutation = list(range(width))
    seen: set[int] = set()
    for index in range(0, len(blocks), 2):
        left, right = blocks[index : index + 2]
        if len(left) != len(right):
            raise ValueError("Left/right swap blocks must have equal widths")
        for left_column, right_column in zip(left, right, strict=True):
            if (
                left_column in seen
                or right_column in seen
                or left_column == right_column
                or not 0 <= left_column < width
                or not 0 <= right_column < width
            ):
                raise ValueError("Swap blocks overlap or exceed the observation width")
            permutation[left_column] = right_column
            permutation[right_column] = left_column
            seen.update((left_column, right_column))
    if tuple(permutation[index] for index in permutation) != tuple(range(width)):
        raise RuntimeError("Bilateral permutation must be self-inverse")
    return tuple(permutation)


ACTOR_PERMUTATION = bilateral_permutation(83, ACTOR_SWAP_BLOCKS)
CRITIC_PERMUTATION = bilateral_permutation(CRITIC_OBSERVATION_WIDTH, CRITIC_SWAP_BLOCKS)


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def validate_lr_order_migration_marker(marker: Any) -> dict[str, Any]:
    """Validate immutable source lineage carried by migrated/future checkpoints."""

    if not isinstance(marker, Mapping):
        raise TypeError("V12 L/R migration marker must be a mapping")
    value = dict(marker)
    expected_keys = {
        "schema_version",
        "revision",
        "site_order_revision",
        "strategy",
        "source_checkpoint_path",
        "source_checkpoint_sha256",
        "source_clock",
        "actor_w0_optimizer_parameter_id",
        "critic_w0_optimizer_parameter_id",
        "actor_swap_blocks",
        "critic_swap_blocks",
        "actor_permutation",
        "critic_permutation",
        "zeroed_actor_columns",
        "foot_adapter_at_source",
        "tensor_integrity",
    }
    if set(value) != expected_keys:
        raise ValueError("V12 L/R migration marker keys drifted")
    if value["schema_version"] != 1 or value["revision"] != MIGRATION_REVISION:
        raise ValueError("V12 L/R migration revision drifted")
    if value["site_order_revision"] != MICROBAN_BILATERAL_SITE_ORDER_REVISION:
        raise ValueError("V12 L/R migration site-order revision drifted")
    strategy = value["strategy"]
    if strategy not in MIGRATION_STRATEGIES:
        raise ValueError("V12 L/R migration strategy is invalid")
    if (
        not isinstance(value["source_checkpoint_path"], str)
        or not value["source_checkpoint_path"]
    ):
        raise ValueError("V12 L/R migration source path is missing")
    _sha256(value["source_checkpoint_sha256"], "migration source checkpoint hash")
    clock = value["source_clock"]
    if not isinstance(clock, Mapping):
        raise TypeError("V12 L/R migration source clock is malformed")
    if set(clock) != {"iteration", "completed_updates", "common_step_counter"}:
        raise ValueError("V12 L/R migration source clock keys drifted")
    iteration = clock["iteration"]
    completed = clock["completed_updates"]
    common_step = clock["common_step_counter"]
    if (
        not isinstance(iteration, int)
        or isinstance(iteration, bool)
        or completed != iteration + 1
        or common_step != completed * 24
    ):
        raise ValueError("V12 L/R migration source clock is inconsistent")
    if (
        value["actor_w0_optimizer_parameter_id"] != 1
        or value["critic_w0_optimizer_parameter_id"] != 9
    ):
        raise ValueError("V12 L/R migration Adam parameter IDs drifted")
    if value["actor_swap_blocks"] != [list(block) for block in ACTOR_SWAP_BLOCKS]:
        raise ValueError("V12 L/R actor swap blocks drifted")
    if value["critic_swap_blocks"] != [list(block) for block in CRITIC_SWAP_BLOCKS]:
        raise ValueError("V12 L/R critic swap blocks drifted")
    if value["actor_permutation"] != list(ACTOR_PERMUTATION) or value[
        "critic_permutation"
    ] != list(CRITIC_PERMUTATION):
        raise ValueError("V12 L/R exact permutation drifted")
    expected_zero = (
        [] if strategy == "swap" else list(TELEOP_V12_HAND_OBSERVATION_COLUMNS)
    )
    if value["zeroed_actor_columns"] != expected_zero:
        raise ValueError("V12 L/R zeroed hand-column set drifted")
    if value["foot_adapter_at_source"] != {
        "active": False,
        "maximum_absolute_w0": 0.0,
        "maximum_absolute_adam_moment": 0.0,
        "handling": "unlearned_exact_zero_left_untouched",
    }:
        raise ValueError("V12 L/R pre-foot evidence drifted")
    integrity = value["tensor_integrity"]
    if not isinstance(integrity, Mapping) or integrity.get("passed") is not True:
        raise ValueError("V12 L/R tensor-integrity evidence is missing")
    unchanged = integrity.get("unchanged_tensors")
    partial = integrity.get("partially_transformed_tensors")
    if (
        not isinstance(unchanged, Mapping)
        or integrity.get("unchanged_tensor_count") != len(unchanged)
        or not isinstance(partial, Mapping)
    ):
        raise ValueError("V12 L/R tensor-integrity inventory is malformed")
    for digest in unchanged.values():
        _sha256(digest, "unchanged tensor hash")
    expected_partial = {
        "actor_state_dict.mlp.0.weight",
        "actor_state_dict.obs_normalizer._mean",
        "actor_state_dict.obs_normalizer._var",
        "actor_state_dict.obs_normalizer._std",
        "critic_state_dict.mlp.0.weight",
        "critic_state_dict.obs_normalizer._mean",
        "critic_state_dict.obs_normalizer._var",
        "critic_state_dict.obs_normalizer._std",
        "optimizer_state_dict.state.1.exp_avg",
        "optimizer_state_dict.state.1.exp_avg_sq",
        "optimizer_state_dict.state.9.exp_avg",
        "optimizer_state_dict.state.9.exp_avg_sq",
    }
    if set(partial) != expected_partial:
        raise ValueError("V12 L/R partially transformed tensor inventory drifted")
    for evidence in partial.values():
        if not isinstance(evidence, Mapping):
            raise TypeError("V12 L/R partial tensor evidence is malformed")
        for name in (
            "untouched_source_sha256",
            "untouched_output_sha256",
            "source_full_sha256",
            "output_full_sha256",
        ):
            _sha256(evidence.get(name), f"partial tensor {name}")
        if evidence["untouched_source_sha256"] != evidence["untouched_output_sha256"]:
            raise ValueError("V12 L/R untouched tensor columns changed")
    return value


def validate_lr_order_checkpoint_lineage(infos: Any) -> dict[str, Any] | None:
    """Validate a marker when present; legacy, unmigrated checkpoints return None."""

    if not isinstance(infos, Mapping):
        raise TypeError("Contract-v12 checkpoint infos must be a mapping")
    marker = infos.get(MIGRATION_INFO_KEY)
    if marker is None:
        return None
    return validate_lr_order_migration_marker(marker)


def validate_bilateral_site_order_checkpoint(
    infos: Any,
) -> dict[str, Any] | None:
    """Reject pre-fix raw checkpoints while accepting fresh or migrated lineage."""

    marker = validate_lr_order_checkpoint_lineage(infos)
    revision = infos.get(BILATERAL_SITE_ORDER_INFO_KEY)
    if revision == MICROBAN_BILATERAL_SITE_ORDER_REVISION:
        return marker
    # A valid migration marker itself authenticates an older migration utility;
    # the next runner save always materializes the top-level revision as well.
    if marker is not None and revision is None:
        return marker
    raise ValueError("Checkpoint predates the authenticated bilateral site-order fix")
