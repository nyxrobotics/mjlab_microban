"""Authenticated legacy-model bootstrap for contract-v12 teleoperation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_HMD_JOINT_NAMES,
    MICROBAN_TELEOP_ACTION_JOINT_NAMES,
)
from mjlab_microban.tasks.microban_teleop_v12_actor import (
    LEGACY_TO_TELEOP_OBSERVATION_INDEX,
    LEGACY_VELOCITY_ACTOR_STATE_KEYS,
    LEGACY_VELOCITY_CHECKPOINT_ITERATION,
    LEGACY_VELOCITY_CHECKPOINT_SHA256,
    LEGACY_VELOCITY_NORMALIZER_EPS,
    TELEOP_V12_ACTOR_TOPOLOGY,
    TELEOP_V12_BOOTSTRAP_MAPPING_VERSION,
    TELEOP_V12_EXTRA_OBSERVATION_COLUMNS,
    TELEOP_V12_SHARED_OBSERVATION_COLUMNS,
    LegacyAdapterTeleopActor,
    transplant_legacy_actor_state_to_teleop83,
)

TELEOP_V12_BOOTSTRAP_PROVENANCE_SCHEMA_VERSION = 1
PINNED_LEGACY_TELEOP_PROBE_SHA256 = (
    "f51378d59ff4d68fb1185a91eb2a863749e5c7be6ec4cd0ab4a0b08f1565e69d"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_REPO_PATH_PREFIX = "repo://"


@dataclass(frozen=True)
class LegacyVelocitySourceIdentity:
    path: str
    sha256: str
    iteration: int
    normalizer_count: int


@dataclass(frozen=True)
class LegacyTeleopProbeIdentity:
    path: str
    sha256: str
    scenario_count: int
    steps: int
    settle_steps: int
    seed: int


@dataclass(frozen=True)
class TeleopV12BootstrapProvenance:
    schema_version: int
    source: LegacyVelocitySourceIdentity
    probe: LegacyTeleopProbeIdentity
    mapping_version: str
    source_to_target_columns: tuple[tuple[int, int], ...]
    new_trainable_columns: tuple[int, ...]
    target_actor_topology: tuple[int, ...]
    normalizer_eps: float
    previous_action_semantics: str
    action_clip: None
    frozen_tensors: tuple[str, ...]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_bootstrap_artifact_path(path: str | Path) -> Path:
    """Resolve a portable repo-relative provenance path or a supplied path."""

    value = str(path)
    if value.startswith(_REPO_PATH_PREFIX):
        relative = Path(value.removeprefix(_REPO_PATH_PREFIX))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid repo-relative bootstrap artifact path")
        return (_PROJECT_ROOT / relative).resolve()
    return Path(path).expanduser().resolve()


def portable_bootstrap_artifact_path(path: str | Path) -> str:
    """Store in-repository evidence without pinning this computer's root path."""

    resolved = resolve_bootstrap_artifact_path(path)
    try:
        relative = resolved.relative_to(_PROJECT_ROOT)
    except ValueError:
        return str(resolved)
    return f"{_REPO_PATH_PREFIX}{relative.as_posix()}"


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON duplicates key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"JSON contains non-finite value {value!r}")


def inspect_legacy_velocity_checkpoint(
    checkpoint_path: str | Path,
    expected_sha256: str = LEGACY_VELOCITY_CHECKPOINT_SHA256,
) -> tuple[LegacyVelocitySourceIdentity, dict[str, torch.Tensor]]:
    """Load only the authenticated actor state from the proven legacy run."""

    path = resolve_bootstrap_artifact_path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Legacy velocity checkpoint not found: {path}")
    if _SHA256_RE.fullmatch(expected_sha256) is None:
        raise ValueError("Expected legacy checkpoint SHA-256 must be lowercase hex")
    digest = sha256_file(path)
    if digest != expected_sha256 or digest != LEGACY_VELOCITY_CHECKPOINT_SHA256:
        raise ValueError(
            f"Legacy velocity checkpoint SHA-256 mismatch: {digest}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("iter") != (
        LEGACY_VELOCITY_CHECKPOINT_ITERATION
    ):
        raise ValueError("Legacy velocity checkpoint iteration drifted")
    actor = payload.get("actor_state_dict")
    if not isinstance(actor, Mapping) or set(actor) != LEGACY_VELOCITY_ACTOR_STATE_KEYS:
        raise ValueError("Legacy velocity checkpoint actor keys drifted")
    state: dict[str, torch.Tensor] = {}
    for name in sorted(LEGACY_VELOCITY_ACTOR_STATE_KEYS):
        value = actor[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Legacy actor value {name!r} is not a tensor")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"Legacy actor tensor {name!r} is non-finite")
        state[name] = value.detach().clone()
    count_tensor = state["obs_normalizer.count"]
    if count_tensor.dtype != torch.int64 or count_tensor.ndim != 0:
        raise ValueError("Legacy normalizer count tensor drifted")
    count = int(count_tensor.item())
    if count != 1_474_560_000:
        raise ValueError(f"Legacy normalizer count drifted: {count}")
    identity = LegacyVelocitySourceIdentity(
        path=portable_bootstrap_artifact_path(path),
        sha256=digest,
        iteration=LEGACY_VELOCITY_CHECKPOINT_ITERATION,
        normalizer_count=count,
    )
    return identity, state


def validate_legacy_teleop_probe_receipt(
    receipt_path: str | Path,
    source: LegacyVelocitySourceIdentity,
    expected_sha256: str = PINNED_LEGACY_TELEOP_PROBE_SHA256,
) -> LegacyTeleopProbeIdentity:
    """Verify the hash-bound 9x300 raw-action closed-loop source receipt."""

    path = resolve_bootstrap_artifact_path(receipt_path)
    if not path.is_file():
        raise FileNotFoundError(f"Legacy teleop probe receipt not found: {path}")
    if _SHA256_RE.fullmatch(expected_sha256) is None:
        raise ValueError("Expected probe receipt SHA-256 must be lowercase hex")
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise ValueError(f"Legacy teleop probe receipt SHA-256 mismatch: {digest}")
    try:
        report = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite,
        )
    except UnicodeDecodeError as exc:
        raise ValueError("Probe receipt must be UTF-8 JSON") from exc
    if not isinstance(report, dict):
        raise TypeError("Probe receipt root must be an object")
    if report.get("probe") != "legacy_velocity_actor_in_nominal_teleop_env_v1":
        raise ValueError("Probe receipt revision drifted")
    checkpoint = report.get("checkpoint")
    if not isinstance(checkpoint, dict) or checkpoint.get("sha256") != source.sha256:
        raise ValueError("Probe receipt is not bound to the legacy source")
    settings = report.get("settings")
    expected_settings = {
        "seed": 42,
        "steps": 300,
        "settle_steps": 50,
        "step_dt_s": 0.02,
        "action_clip": None,
        "previous_action": "raw_actor_output",
        "foot_target": "exact_zero_inactive",
        "hand_target": "exact_zero_inactive",
    }
    if not isinstance(settings, dict) or any(
        settings.get(name) != value for name, value in expected_settings.items()
    ):
        raise ValueError("Probe receipt settings drifted")
    mapping = report.get("mapping")
    if not isinstance(mapping, dict):
        raise TypeError("Probe receipt mapping is malformed")
    exact_mapping = {
        "legacy_observation_width": 63,
        "teleop_observation_width": 83,
        "legacy_joint_names": list(MICROBAN_TELEOP_ACTION_JOINT_NAMES),
        "teleop_joint_names": [
            *MICROBAN_HMD_JOINT_NAMES,
            *MICROBAN_TELEOP_ACTION_JOINT_NAMES,
        ],
        "teleop_joint_indices_for_legacy": list(range(3, 21)),
        "legacy_action_names": list(MICROBAN_TELEOP_ACTION_JOINT_NAMES),
        "teleop_action_indices_for_legacy": list(range(18)),
        "legacy_source_columns_to_teleop_target_columns": [
            list(pair) for pair in LEGACY_TO_TELEOP_OBSERVATION_INDEX
        ],
        "new_teleop_columns": list(TELEOP_V12_EXTRA_OBSERVATION_COLUMNS),
    }
    if any(mapping.get(name) != value for name, value in exact_mapping.items()):
        raise ValueError("Probe receipt semantic mapping drifted")
    summary = report.get("summary")
    required_summary = {
        "scenario_count": 9,
        "completed_scenario_count": 9,
        "fall_scenario_count": 0,
        "nonfinite_scenario_count": 0,
        "actual_soft_limit_violation_scenario_count": 0,
        "directionally_correct_scenario_count": 8,
        "directional_scenario_count": 8,
        "neutral_target_contract_all_steps": True,
        "raw_action_recurrence_all_steps": True,
    }
    if not isinstance(summary, dict) or any(
        type(summary.get(name)) is not type(value) or summary.get(name) != value
        for name, value in required_summary.items()
    ):
        raise ValueError("Probe receipt did not pass the 9x300 source gate")
    results = report.get("results")
    if not isinstance(results, list) or len(results) != 9:
        raise ValueError("Probe receipt must contain exactly nine scenarios")
    for result in results:
        if (
            not isinstance(result, dict)
            or result.get("completed") is not True
            or result.get("fell") is not False
            or result.get("nonfinite") is not None
            or result.get("executed_steps") != 300
            or result.get("raw_action_recurrence_verified_steps") != 300
            or result.get("neutral_foot_hand_target_verified_steps") != 300
            or not math.isclose(
                float(result.get("maximum_actual_soft_limit_violation_rad", math.inf)),
                0.0,
                abs_tol=1.0e-7,
            )
        ):
            raise ValueError("Probe receipt contains a failing scenario")
    return LegacyTeleopProbeIdentity(
        path=portable_bootstrap_artifact_path(path),
        sha256=digest,
        scenario_count=9,
        steps=300,
        settle_steps=50,
        seed=42,
    )


def bootstrap_legacy_actor(
    actor: LegacyAdapterTeleopActor,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    probe_receipt_path: str | Path,
    probe_receipt_sha256: str,
) -> TeleopV12BootstrapProvenance:
    """Authenticate, transplant, freeze, and bind one fresh v12 actor."""

    if not isinstance(actor, LegacyAdapterTeleopActor):
        raise TypeError("Contract-v12 bootstrap requires LegacyAdapterTeleopActor")
    source, source_state = inspect_legacy_velocity_checkpoint(
        checkpoint_path, checkpoint_sha256
    )
    probe = validate_legacy_teleop_probe_receipt(
        probe_receipt_path, source, probe_receipt_sha256
    )
    mapped = transplant_legacy_actor_state_to_teleop83(
        source_state, actor.state_dict()
    )
    actor.load_state_dict(mapped, strict=True)
    actor.bind_frozen_legacy_reference()
    return TeleopV12BootstrapProvenance(
        schema_version=TELEOP_V12_BOOTSTRAP_PROVENANCE_SCHEMA_VERSION,
        source=source,
        probe=probe,
        mapping_version=TELEOP_V12_BOOTSTRAP_MAPPING_VERSION,
        source_to_target_columns=LEGACY_TO_TELEOP_OBSERVATION_INDEX,
        new_trainable_columns=TELEOP_V12_EXTRA_OBSERVATION_COLUMNS,
        target_actor_topology=TELEOP_V12_ACTOR_TOPOLOGY,
        normalizer_eps=LEGACY_VELOCITY_NORMALIZER_EPS,
        previous_action_semantics="raw_actor_output",
        action_clip=None,
        frozen_tensors=tuple(
            sorted(LEGACY_VELOCITY_ACTOR_STATE_KEYS - {"mlp.0.weight"})
        ),
    )


def serialize_bootstrap_provenance(
    provenance: TeleopV12BootstrapProvenance,
) -> dict[str, Any]:
    return asdict(provenance)


def validate_bootstrap_provenance(
    value: object,
    *,
    verify_files: bool = True,
) -> TeleopV12BootstrapProvenance:
    """Validate serialized checkpoint provenance and optionally rehash inputs."""

    if not isinstance(value, dict):
        raise TypeError("Contract-v12 checkpoint lacks bootstrap provenance")
    try:
        source_value = value["source"]
        probe_value = value["probe"]
        if not isinstance(source_value, dict) or not isinstance(probe_value, dict):
            raise TypeError
        source = LegacyVelocitySourceIdentity(**source_value)
        probe = LegacyTeleopProbeIdentity(**probe_value)
        result = TeleopV12BootstrapProvenance(
            schema_version=value["schema_version"],
            source=source,
            probe=probe,
            mapping_version=value["mapping_version"],
            source_to_target_columns=tuple(
                tuple(pair) for pair in value["source_to_target_columns"]
            ),
            new_trainable_columns=tuple(value["new_trainable_columns"]),
            target_actor_topology=tuple(value["target_actor_topology"]),
            normalizer_eps=value["normalizer_eps"],
            previous_action_semantics=value["previous_action_semantics"],
            action_clip=value["action_clip"],
            frozen_tensors=tuple(value["frozen_tensors"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Contract-v12 bootstrap provenance is malformed") from exc
    expected_static = {
        "schema_version": TELEOP_V12_BOOTSTRAP_PROVENANCE_SCHEMA_VERSION,
        "mapping_version": TELEOP_V12_BOOTSTRAP_MAPPING_VERSION,
        "source_to_target_columns": LEGACY_TO_TELEOP_OBSERVATION_INDEX,
        "new_trainable_columns": TELEOP_V12_EXTRA_OBSERVATION_COLUMNS,
        "target_actor_topology": TELEOP_V12_ACTOR_TOPOLOGY,
        "normalizer_eps": LEGACY_VELOCITY_NORMALIZER_EPS,
        "previous_action_semantics": "raw_actor_output",
        "action_clip": None,
        "frozen_tensors": tuple(
            sorted(LEGACY_VELOCITY_ACTOR_STATE_KEYS - {"mlp.0.weight"})
        ),
    }
    actual = asdict(result)
    if any(actual[name] != expected for name, expected in expected_static.items()):
        raise ValueError("Contract-v12 bootstrap provenance contract drifted")
    if (
        source.sha256 != LEGACY_VELOCITY_CHECKPOINT_SHA256
        or source.iteration != LEGACY_VELOCITY_CHECKPOINT_ITERATION
        or source.normalizer_count != 1_474_560_000
    ):
        raise ValueError("Contract-v12 legacy source identity drifted")
    if (
        probe.sha256 != PINNED_LEGACY_TELEOP_PROBE_SHA256
        or probe.scenario_count != 9
        or probe.steps != 300
        or probe.settle_steps != 50
        or probe.seed != 42
    ):
        raise ValueError("Contract-v12 probe identity drifted")
    if verify_files:
        verified_source, _state = inspect_legacy_velocity_checkpoint(
            source.path, source.sha256
        )
        verified_probe = validate_legacy_teleop_probe_receipt(
            probe.path, verified_source, probe.sha256
        )
        if verified_source != source or verified_probe != probe:
            raise ValueError("Contract-v12 bootstrap source files changed")
    return result


def assert_actor_frozen_against_source(
    actor: LegacyAdapterTeleopActor,
    provenance: TeleopV12BootstrapProvenance,
) -> None:
    """Reconstruct pinned expected tensors and allow only extra W0 to differ."""

    source, source_state = inspect_legacy_velocity_checkpoint(
        provenance.source.path, provenance.source.sha256
    )
    if source != provenance.source:
        raise ValueError("Legacy source identity no longer matches provenance")
    expected = transplant_legacy_actor_state_to_teleop83(
        source_state, actor.state_dict()
    )
    actual = actor.state_dict()
    for name, target in expected.items():
        candidate = actual[name].detach().cpu()
        target = target.detach().cpu()
        if name == "mlp.0.weight":
            candidate = candidate[:, TELEOP_V12_SHARED_OBSERVATION_COLUMNS]
            target = target[:, TELEOP_V12_SHARED_OBSERVATION_COLUMNS]
        if not torch.equal(candidate, target):
            raise RuntimeError(f"Actor no longer matches pinned legacy tensor: {name}")
