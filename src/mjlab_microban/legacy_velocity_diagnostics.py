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
from pathlib import Path
from typing import Any

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
