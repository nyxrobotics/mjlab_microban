"""Write the robot repository's ``config/home_pose.yaml`` from the training HOME.

The robot (microban) has no MuJoCo at runtime, so it reads a generated YAML
with the HOME joint angles and every derived value it checks: NEUTRAL_POSE
(radians, bit-exact), root position/quaternion, projected gravity at HOME and
the hand-target FK contract (schema 2: no HOME-bound contract strings).
``config/home_pose_tool.py write-robot`` calls ``write_robot_home_pose``.

The output is a strict YAML subset the robot parses without PyYAML: ``#``
comment lines, ``key:`` lines that open a nested mapping (two-space indent),
and ``key: <JSON value>`` lines (numbers, double-quoted strings, booleans and
flow lists).  Floats are ``repr`` (exact round trip) with a ``.0`` before any
exponent so YAML 1.1 readers also see floats.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mjlab_microban.robot.home_pose import HOME, HomePose

ROBOT_HOME_POSE_RELATIVE_PATH = Path("config") / "home_pose.yaml"
ROBOT_HOME_POSE_SCHEMA_VERSION = 2

_HEADER = """\
# Microban HOME pose for the robot runtime -- GENERATED, do not edit by hand.
#
# Source of truth: mjlab_microban config/home_pose.yaml (the training repo).
# Regenerate after changing HOME there (and after retraining every policy):
#   uv run python config/home_pose_tool.py write-robot --microban-repo <this repo>
# src/home_pose.py reads this file; constants.NEUTRAL_POSE, the HOME root pose
# and the HOME projected gravity come from it.  Policies whose home_pose stamp is another HOME are refused.
"""


def robot_home_pose_document(home: HomePose = HOME) -> dict[str, Any]:
    """Return the robot YAML content (ordered) for ``home``."""

    from mjlab_microban.robot.microban_hand_fk import microban_hand_fk_metadata

    if home is not HOME:
        raise ValueError("The robot document is generated for the loaded HOME only")
    analysis = home.analysis
    return {
        "schema_version": ROBOT_HOME_POSE_SCHEMA_VERSION,
        "name": home.name,
        "label": home.label,
        "tag": home.tag,
        "joint_hash": home.joint_hash,
        "trunk_pitch_deg": home.trunk_pitch_deg,
        "trunk_pitch_rad": home.trunk_pitch_rad,
        "root_pos_m": list(home.root_pos),
        "root_quat_wxyz": list(home.root_quat_wxyz),
        "projected_gravity": list(home.projected_gravity),
        "joint_pos_deg": dict(home.joint_pos_deg),
        "joint_pos_rad": dict(home.joint_pos_rad),
        "hand_target_fk": microban_hand_fk_metadata(),
        # Informational FK values (not used by the robot runtime).
        "fk": {
            "head_standing_height_m": home.head_standing_height_m,
            "feet_lateral_m": home.feet_lateral_m,
            "com_m": list(analysis.com),
            "sole_x_range_m": [analysis.sole_x_min, analysis.sole_x_max],
            "heel_margin_m": analysis.heel_margin_m,
            "toe_margin_m": analysis.toe_margin_m,
            "total_mass_kg": analysis.total_mass_kg,
        },
    }


def _format_float(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError(f"Robot HOME YAML values must be finite, got {value!r}")
    text = repr(float(value))
    mantissa, exponent_marker, exponent = text.partition("e")
    if exponent_marker and "." not in mantissa:
        text = f"{mantissa}.0e{exponent}"
    return text


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _format_float(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=True)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format_scalar(item) for item in value) + "]"
    raise TypeError(f"Unsupported robot HOME YAML value {value!r}")


def _render_mapping(mapping: Mapping[str, Any], indent: int, lines: list[str]) -> None:
    for key, value in mapping.items():
        if not isinstance(key, str) or not key.replace("_", "a").isalnum():
            raise ValueError(f"Robot HOME YAML keys must be identifiers, got {key!r}")
        prefix = " " * indent
        if isinstance(value, Mapping):
            lines.append(f"{prefix}{key}:")
            _render_mapping(value, indent + 2, lines)
        else:
            lines.append(f"{prefix}{key}: {_format_scalar(value)}")


def render_robot_home_pose_yaml(document: Mapping[str, Any]) -> str:
    lines: list[str] = []
    _render_mapping(document, 0, lines)
    return _HEADER + "\n" + "\n".join(lines) + "\n"


class NotHomeYamlRobotCheckout(FileNotFoundError):
    """The robot checkout does not read config/home_pose.yaml."""


def require_home_yaml_robot_checkout(microban_repo: Path | str) -> Path:
    """Refuse a robot checkout whose code does not read config/home_pose.yaml.

    Only robot trees with ``src/home_pose.py`` imported by ``src/constants.py``
    take their HOME from the generated YAML; in any other tree writing the
    YAML would change nothing and a retrained policy would be refused only at
    install time.
    """

    repo = Path(microban_repo).expanduser().resolve()
    constants = repo / "src" / "constants.py"
    if not constants.is_file():
        raise NotHomeYamlRobotCheckout(f"{repo} is not a microban robot checkout")
    if not (repo / "src" / "home_pose.py").is_file() or "from home_pose import" not in (
        constants.read_text(encoding="utf-8")
    ):
        raise NotHomeYamlRobotCheckout(
            f"{repo} does not read config/home_pose.yaml (no src/home_pose.py imported by "
            "src/constants.py); use a robot checkout that reads it"
        )
    return repo


def write_robot_home_pose(microban_repo: Path | str, *, check: bool = False) -> tuple[Path, bool]:
    """Write (or with ``check`` only compare) the robot HOME YAML.

    Returns ``(path, up_to_date)``.  ``up_to_date`` is whether the file
    already had exactly this content.
    """

    repo = Path(microban_repo).expanduser().resolve()
    require_home_yaml_robot_checkout(repo)
    path = repo / ROBOT_HOME_POSE_RELATIVE_PATH
    text = render_robot_home_pose_yaml(robot_home_pose_document())
    current = path.read_text(encoding="utf-8") if path.is_file() else None
    up_to_date = current == text
    if not check and not up_to_date:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    return path, up_to_date
