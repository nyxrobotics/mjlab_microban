#!/usr/bin/env python3
"""Inspect config/home_pose.yaml and publish it to the robot repository.

Usage (from the training repository root)::

    uv run python config/home_pose_tool.py show
    uv run python config/home_pose_tool.py write-robot --microban-repo ../microban
    uv run python config/home_pose_tool.py write-robot --microban-repo ../microban --check

``show`` prints the HOME inputs and everything derived from them by MuJoCo FK
(root pose, projected gravity, COM and sole contact area, heel/toe margins,
head standing height, feet distance, identity hash/tag, contract strings).
``write-robot`` writes ``<microban-repo>/config/home_pose.yaml``, the robot's
copy (NEUTRAL_POSE, root pose, gravity, contract identifiers, hand FK
contract); ``--check`` exits 1 if that file is missing or stale instead.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _show(arguments: argparse.Namespace) -> int:
    from mjlab_microban.robot.home_pose import load_home_pose

    home = load_home_pose(arguments.yaml)
    summary = home.summary()
    analysis = home.analysis
    summary["margins_mm"] = {
        "heel": analysis.heel_margin_m * 1e3,
        "toe": analysis.toe_margin_m * 1e3,
        "com_minus_sole_centre": analysis.com_offset_x * 1e3,
    }
    if arguments.contracts:
        warnings.filterwarnings("ignore")
        from mjlab_microban.robot.home_pose_robot import robot_contract_strings

        summary["contracts"] = robot_contract_strings()
    print(json.dumps(summary, indent=2))
    return 0


def _write_robot(arguments: argparse.Namespace) -> int:
    warnings.filterwarnings("ignore")
    from mjlab_microban.robot.home_pose_robot import write_robot_home_pose

    path, up_to_date = write_robot_home_pose(arguments.microban_repo, check=arguments.check)
    if arguments.check:
        print(f"{path}: {'up to date' if up_to_date else 'STALE or missing'}")
        return 0 if up_to_date else 1
    print(f"{path}: {'unchanged' if up_to_date else 'written'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    show = commands.add_parser("show", help="print the HOME and its FK-derived values")
    show.add_argument("--yaml", type=Path, default=REPO_ROOT / "config" / "home_pose.yaml")
    show.add_argument(
        "--contracts",
        action="store_true",
        help="also print the HOME-bound contract strings (imports the training tasks)",
    )
    show.set_defaults(handler=_show)
    write = commands.add_parser("write-robot", help="write the robot repo's config/home_pose.yaml")
    write.add_argument("--microban-repo", type=Path, required=True)
    write.add_argument("--check", action="store_true", help="only report whether it is up to date")
    write.set_defaults(handler=_write_robot)
    arguments = parser.parse_args(argv)
    return arguments.handler(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
