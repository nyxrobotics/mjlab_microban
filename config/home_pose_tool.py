#!/usr/bin/env python3
"""Inspect config/home_pose.yaml and publish it to the robot repository.

Usage (from the training repository root)::

    uv run python config/home_pose_tool.py show
    uv run python config/home_pose_tool.py write-robot --microban-repo ../microban
    uv run python config/home_pose_tool.py write-robot --microban-repo ../microban --check

``show`` prints the HOME inputs and everything derived from them by MuJoCo FK
(root pose, projected gravity, COM and sole contact area, heel/toe margins,
head standing height, feet distance, identity hash/tag, contract strings),
plus ``training_line``: whether this checkout's training tasks load at the
HOME (``mjlab_microban.robot.home_pose_training``; exit 1 if they do not).
``write-robot`` writes ``<microban-repo>/config/home_pose.yaml``, the robot's
copy (NEUTRAL_POSE, root pose, gravity, contract identifiers, hand FK
contract); it refuses a HOME the training tasks refuse unless ``--force``.
``--check`` exits 1 if that file is missing or stale instead.

Every failure (unreadable or unbalanced YAML, not a robot checkout, ...) is
one ``error: ...`` line and exit status 1.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
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
        home.joint_pos_deg,
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
    if arguments.contracts:
        if arguments.yaml.resolve() != HOME_POSE_YAML.resolve():
            raise ToolError("--contracts describes the checkout's HOME only (drop --yaml)")
        if training is not None and not training.ok:
            raise ToolError(f"--contracts needs the training tasks: {training.error}")
        warnings.filterwarnings("ignore")
        from mjlab_microban.robot.home_pose_robot import robot_contract_strings

        summary["contracts"] = robot_contract_strings()
    print(json.dumps(summary, indent=2))
    if training is not None and not training.ok:
        raise ToolError(
            f"this checkout cannot retrain at this HOME: {training.error} "
            "(see config/README.md, 'what this training line accepts')"
        )
    return 0


def _write_robot(arguments: argparse.Namespace) -> int:
    home = _load(HOME_POSE_YAML)
    if not arguments.check and not arguments.force:
        training = _training_line(home)
        if not training.ok:
            raise ToolError(
                f"this checkout cannot retrain at the HOME in {HOME_POSE_YAML.name}: "
                f"{training.error} (pass --force to publish it anyway)"
            )
    warnings.filterwarnings("ignore")
    from mjlab_microban.robot.home_pose_robot import write_robot_home_pose

    try:
        path, up_to_date = write_robot_home_pose(arguments.microban_repo, check=arguments.check)
    except FileNotFoundError as error:
        raise ToolError(str(error)) from error
    except (OSError, ValueError) as error:
        raise ToolError(f"cannot write the robot HOME: {' '.join(str(error).split())}") from error
    if arguments.check:
        print(f"{path}: {'up to date' if up_to_date else 'STALE or missing'}")
        return 0 if up_to_date else 1
    print(f"{path}: {'unchanged' if up_to_date else 'written'}")
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
    show.add_argument(
        "--contracts",
        action="store_true",
        help="also print the HOME-bound contract strings (imports the training tasks)",
    )
    show.set_defaults(handler=_show)
    write = commands.add_parser("write-robot", help="write the robot repo's config/home_pose.yaml")
    write.add_argument("--microban-repo", type=Path, required=True)
    write.add_argument("--check", action="store_true", help="only report whether it is up to date")
    write.add_argument(
        "--force",
        action="store_true",
        help="publish even if this checkout's training tasks refuse the HOME",
    )
    write.set_defaults(handler=_write_robot)
    arguments = parser.parse_args(argv)
    try:
        return arguments.handler(arguments)
    except ToolError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
