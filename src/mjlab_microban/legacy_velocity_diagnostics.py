"""Pure report helpers for the original Microban velocity-policy diagnostic.

This module deliberately has no MuJoCo or Torch dependency.  The headless
evaluator can therefore keep scenario validation, report aggregation and the
atomic receipt writer covered by ordinary CPU unit tests.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LEGACY_VELOCITY_DIAGNOSTIC_REVISION = "microban_legacy_velocity_diagnostic_v1"
DEFAULT_LEGACY_VELOCITY_CHECKPOINT = Path("checkpoints/xc330_velocity/model_14999.pt")
DEFAULT_LEGACY_VELOCITY_SHA256 = (
    "b0bcdadac39716be784207dd6b2b93157162a3e80650e23c05f490c400b9e141"
)
DEFAULT_LEGACY_VELOCITY_RAW_OUTPUT = Path(
    "artifacts/legacy_velocity_model_14999_diagnostic.json"
)
DEFAULT_LEGACY_VELOCITY_SOFTCLIP_OUTPUT = Path(
    "artifacts/legacy_velocity_model_14999_softclip_diagnostic.json"
)


@dataclass(frozen=True)
class TwistScenario:
    """One body-frame twist held throughout an original-policy rollout."""

    name: str
    twist: tuple[float, float, float]


def default_scenarios() -> tuple[TwistScenario, ...]:
    """Return the small, signed-axis suite used for teacher triage."""

    return (
        TwistScenario("neutral", (0.0, 0.0, 0.0)),
        TwistScenario("forward_0p1", (0.1, 0.0, 0.0)),
        TwistScenario("forward_0p2", (0.2, 0.0, 0.0)),
        TwistScenario("backward_0p1", (-0.1, 0.0, 0.0)),
        TwistScenario("backward_0p2", (-0.2, 0.0, 0.0)),
        TwistScenario("lateral_left_0p1", (0.0, 0.1, 0.0)),
        TwistScenario("lateral_right_0p1", (0.0, -0.1, 0.0)),
        TwistScenario("yaw_left_0p5", (0.0, 0.0, 0.5)),
        TwistScenario("yaw_right_0p5", (0.0, 0.0, -0.5)),
    )


def validate_scenarios(scenarios: Sequence[TwistScenario]) -> None:
    """Reject ambiguous or non-axis-aligned diagnostic commands."""

    if not scenarios:
        raise ValueError("At least one legacy velocity scenario is required")
    names: set[str] = set()
    twists: set[tuple[float, float, float]] = set()
    for scenario in scenarios:
        if not scenario.name or scenario.name.strip() != scenario.name:
            raise ValueError("Scenario names must be non-empty and trimmed")
        if scenario.name in names:
            raise ValueError(f"Duplicate scenario name: {scenario.name}")
        if len(scenario.twist) != 3 or not all(
            math.isfinite(float(value)) for value in scenario.twist
        ):
            raise ValueError(f"Scenario {scenario.name!r} has a non-finite twist")
        nonzero_axes = sum(not math.isclose(value, 0.0) for value in scenario.twist)
        if nonzero_axes > 1:
            raise ValueError(
                f"Scenario {scenario.name!r} is not a signed single-axis command"
            )
        if scenario.twist in twists:
            raise ValueError(f"Duplicate scenario twist: {scenario.twist}")
        names.add(scenario.name)
        twists.add(scenario.twist)


def select_scenarios(
    scenarios: Sequence[TwistScenario], requested: str | None
) -> tuple[TwistScenario, ...]:
    """Select a comma-separated subset while preserving requested order."""

    validate_scenarios(scenarios)
    if requested is None:
        return tuple(scenarios)
    requested_names = tuple(name.strip() for name in requested.split(","))
    if not requested_names or any(not name for name in requested_names):
        raise ValueError("--scenarios must be a non-empty comma-separated list")
    if len(set(requested_names)) != len(requested_names):
        raise ValueError("--scenarios contains duplicate names")
    by_name = {scenario.name: scenario for scenario in scenarios}
    unknown = sorted(set(requested_names) - set(by_name))
    if unknown:
        raise ValueError(f"Unknown legacy velocity scenarios: {unknown}")
    return tuple(by_name[name] for name in requested_names)


def resolve_output_path(
    requested: str | Path | None, *, execute_soft_limit_projection: bool
) -> Path:
    """Select a mode-specific path and protect the raw baseline receipt."""

    output = Path(requested) if requested is not None else (
        DEFAULT_LEGACY_VELOCITY_SOFTCLIP_OUTPUT
        if execute_soft_limit_projection
        else DEFAULT_LEGACY_VELOCITY_RAW_OUTPUT
    )
    if (
        execute_soft_limit_projection
        and output.expanduser().resolve()
        == DEFAULT_LEGACY_VELOCITY_RAW_OUTPUT.expanduser().resolve()
    ):
        raise ValueError(
            "Soft-limit projection diagnostics may not overwrite the raw "
            f"diagnostic path: {DEFAULT_LEGACY_VELOCITY_RAW_OUTPUT}"
        )
    return output


def summarize_samples(values: Sequence[float]) -> dict[str, float | int | None]:
    """Return finite JSON-safe descriptive statistics for scalar samples."""

    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {
            "count": 0,
            "maximum": None,
            "mean": None,
            "minimum": None,
            "p05": None,
            "p95": None,
        }
    ordered = sorted(finite)

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": len(ordered),
        "maximum": ordered[-1],
        "mean": math.fsum(ordered) / len(ordered),
        "minimum": ordered[0],
        "p05": percentile(0.05),
        "p95": percentile(0.95),
    }


def directional_response(result: Mapping[str, Any]) -> dict[str, Any] | None:
    """Describe signed response without imposing an acceptance threshold."""

    command = result["command"]
    twist = tuple(float(command[name]) for name in ("vx_m_s", "vy_m_s", "yaw_rad_s"))
    nonzero = [index for index, value in enumerate(twist) if not math.isclose(value, 0.0)]
    if not nonzero:
        return None
    if len(nonzero) != 1:
        raise ValueError("Directional response requires a single-axis command")
    index = nonzero[0]
    axis = ("vx_m_s", "vy_m_s", "yaw_rad_s")[index]
    measured = result["measured_velocity_body"][axis]["mean"]
    if measured is None:
        return {
            "axis": axis,
            "command": twist[index],
            "measured_mean": None,
            "signed_response": None,
            "sign_matches": False,
        }
    measured = float(measured)
    signed = measured * (1.0 if twist[index] > 0.0 else -1.0)
    return {
        "axis": axis,
        "command": twist[index],
        "measured_mean": measured,
        "signed_response": signed,
        "sign_matches": signed > 0.0,
    }


def build_report(
    *,
    checkpoint: str | Path,
    checkpoint_sha256: str,
    device: str,
    seed: int,
    steps: int,
    settle_steps: int,
    step_dt_s: float,
    results: Sequence[Mapping[str, Any]],
    execute_soft_limit_projection: bool = False,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Aggregate scenario evidence into a diagnostic-only receipt."""

    if steps < 1 or not 0 <= settle_steps < steps:
        raise ValueError("Require steps > settle_steps >= 0")
    if not math.isfinite(step_dt_s) or step_dt_s <= 0.0:
        raise ValueError("step_dt_s must be finite and positive")
    expected_names = [scenario.name for scenario in default_scenarios()]
    names = [str(result["name"]) for result in results]
    if not results or len(names) != len(set(names)):
        raise ValueError("Scenario results must be non-empty and uniquely named")
    unknown = sorted(set(names) - set(expected_names))
    if unknown:
        raise ValueError(f"Unexpected scenario results: {unknown}")

    responses: dict[str, dict[str, Any]] = {}
    for result in results:
        response = directional_response(result)
        if response is not None:
            responses[str(result["name"])] = response

    projected = [
        result
        for result in results
        if int(result["target_soft_limits"]["projected_target_value_count"]) > 0
    ]
    actual_violations = [
        result
        for result in results
        if float(result["target_soft_limits"]["maximum_actual_violation_rad"]) > 0.0
    ]
    bilateral_air_evidence = [
        result
        for result in results
        if all(
            bool(result["foot_evidence"][side]["ever_airborne"])
            for side in ("left", "right")
        )
    ]
    simultaneous_flight = [
        result
        for result in results
        if int(result["foot_evidence"]["both_feet_airborne_step_count"]) > 0
    ]
    timestamp = generated_at_utc or datetime.now(UTC).isoformat()
    execution_mode = (
        "project_actor_absolute_targets_to_current_soft_limits"
        if execute_soft_limit_projection
        else "original_unprojected_actor_actions"
    )
    mismatched_modes = [
        str(result["name"])
        for result in results
        if bool(result["target_soft_limits"]["execution_applied_projection"])
        != execute_soft_limit_projection
    ]
    if mismatched_modes:
        raise ValueError(
            "Scenario execution mode disagrees with report mode: "
            f"{mismatched_modes}"
        )
    report = {
        "checkpoint": {
            "path": str(Path(checkpoint).expanduser().resolve()),
            "sha256": checkpoint_sha256,
        },
        "diagnostic_only": True,
        "environment": {
            "action_execution_mode": execution_mode,
            "nominal_task": "Mjlab-Velocity-Microban",
            "play_configuration": True,
            "policy_output_safety_projection_applied": execute_soft_limit_projection,
            "previous_action_observation": (
                "executed_raw_action_after_soft_limit_projection"
                if execute_soft_limit_projection
                else "original_actor_raw_action"
            ),
        },
        "generated_at_utc": timestamp,
        "results": list(results),
        "revision": LEGACY_VELOCITY_DIAGNOSTIC_REVISION,
        "settings": {
            "device": device,
            "duration_s": steps * step_dt_s,
            "execute_soft_limit_projection": execute_soft_limit_projection,
            "seed": seed,
            "settle_steps": settle_steps,
            "step_dt_s": step_dt_s,
            "steps": steps,
        },
        "summary": {
            "actual_soft_limit_violation_scenario_count": len(actual_violations),
            "bilateral_air_evidence_scenario_count": len(bilateral_air_evidence),
            "completed_scenario_count": sum(bool(item["completed"]) for item in results),
            "directional_response": responses,
            "directionally_correct_scenario_count": sum(
                bool(response["sign_matches"]) for response in responses.values()
            ),
            "fall_scenario_count": sum(bool(item["fell"]) for item in results),
            "nonfinite_scenario_count": sum(
                bool(item["nonfinite_detected"]) for item in results
            ),
            "scenario_count": len(results),
            "simultaneous_flight_scenario_count": len(simultaneous_flight),
            "target_projection_scenario_count": len(projected),
        },
        "warning": (
            "This receipt diagnoses the legacy actor in its original play task. "
            "It is not a teleoperation deployment acceptance gate."
        ),
    }
    # Fail before publication if a future report edit introduces NaN/Infinity.
    json.dumps(report, allow_nan=False)
    return report


def checkpoint_sha256(path: str | Path) -> str:
    """Hash a checkpoint without loading executable pickle content."""

    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish_json_atomic(path: str | Path, report: Mapping[str, Any]) -> Path:
    """Durably publish one complete JSON report in the destination directory."""

    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=resolved.parent,
            prefix=f".{resolved.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(resolved)
        directory_fd = os.open(resolved.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return resolved
