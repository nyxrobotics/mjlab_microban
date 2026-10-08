#!/usr/bin/env python3
"""Inspect config/home_pose.yaml.

Usage (from the training repository root)::

    uv run python config/home_pose_tool.py show

``show`` prints the HOME inputs and everything derived from them by MuJoCo FK
(root pose, projected gravity, COM and sole contact area, heel/toe margins,
head standing height, feet distance, identity hash/tag), plus
``training_line``: whether this checkout's training tasks load at the HOME
(``mjlab_microban.robot.home_pose_training``; exit 1 if they do not).

Every failure (unreadable or unbalanced YAML, ...) is one ``error: ...`` line
and exit status 1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from mjlab_microban.robot.home_pose import (  # noqa: E402
    HOME_POSE_YAML,
    describe_home_yaml_error,
    load_home_pose,
)
from mjlab_microban.robot.home_pose_training import check_training_line  # noqa: E402


class ToolError(Exception):
    """A one-line failure reason."""


def _load(path: Path):
    try:
        return load_home_pose(path)
    except (OSError, ValueError, yaml.YAMLError) as error:
        raise ToolError(f"cannot load {path}: {describe_home_yaml_error(error, path)}") from error


def _training_line(home):
    return check_training_line(
        home.input_joint_pos_deg,
        home.trunk_pitch_deg,
        name=home.name,
        label=home.label,
        path=home.path,
    )


def _show(arguments: argparse.Namespace) -> int:
    home = _load(arguments.yaml)
    summary = home.summary()
    analysis = home.analysis
    summary["margins_mm"] = {
        "heel": analysis.heel_margin_m * 1e3,
        "toe": analysis.toe_margin_m * 1e3,
        "com_minus_sole_centre": analysis.com_offset_x * 1e3,
    }
    training = None if arguments.no_training_check else _training_line(home)
    if training is not None:
        summary["training_line"] = training.summary()
    print(json.dumps(summary, indent=2))
    if training is not None and not training.ok:
        raise ToolError(
            f"this checkout cannot retrain at this HOME: {training.error} "
            "(see config/README.md, 'what this training line accepts')"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    show = commands.add_parser("show", help="print the HOME and its FK-derived values")
    show.add_argument("--yaml", type=Path, default=HOME_POSE_YAML)
    show.add_argument(
        "--no-training-check",
        action="store_true",
        help="skip importing the training tasks at this HOME (a few seconds)",
    )
    show.set_defaults(handler=_show)
    arguments = parser.parse_args(argv)
    try:
        return arguments.handler(arguments)
    except ToolError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
