#!/usr/bin/env python3
"""Retrain, judge, export, install and commit every Microban policy at the HOME of config/home_pose.yaml.

    uv run --locked python scripts/retrain_all_for_home.py \\
        --robot-repo ../microban_<label> --robot-branch <label> [--training-branch <label>]

Steps (src/mjlab_microban/pipeline/steps.py; docs/home_pose_workflow.md):

1. home     balance and training-line checks of the YAML, the training test
            suite at this HOME (CPU);
2. walk     one walking run from scratch (Mjlab-Velocity-Microban);
3. pico     the walker's 9x300 source probe (seed 42) and bootstrap gate,
            one PICO run (critic warm-up, hands, feet), then its judgment
            (locomotion, tracking, ONNX; seed 42) and gate file;
4. getup    one get-up run (scheduled IMU latency, calm, effort/push), then
            its judgment (fallen-start stands, push, tremble, posture, clip);
5. export   walk.onnx, getup.onnx, pico_teleop.onnx and manifest.json
            (policy contract, docs/policies.md);
6. install  copy them and the robot HOME YAML into --robot-repo, run its
            tools/validate_policies.py and its tests;
7. commit   commit the robot branch, then config/releases/<tag>/ on the
            training branch, and push both (not with --no-push).

GPU jobs run one at a time; a real run waits while another training uses the
GPU.  A failed check stops the run with exit code 1 and a report; nothing is
rescued.  Fix the cause (a committed recipe or code change) and run the same
command again: steps whose inputs are unchanged are skipped, and a training
that stopped continues from its last checkpoint.  State: <state dir>/
state.json and STATUS.log (default artifacts/home_pipeline/<prefix>_<tag>/);
--status prints it.

--dry-run runs everything at 64 envs with every schedule scaled down
(config/pipeline.yaml `dry:`), forcing past the checks a few-update policy
cannot pass (pipeline/dry.py; the package is marked not deployable), into a
scratch robot clone (its origin push URL must be a local path); the robot
commit stays local and nothing is pushed or committed in this repository.

Exit codes: 0 done, 1 a check failed, 2 a job stalled, 3 bad input,
4 another instance holds the state dir, 130 interrupted.
"""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mjlab_microban.pipeline.core import EXIT_INTERRUPTED, PipelineError  # noqa: E402
from mjlab_microban.pipeline.steps import Pipeline, print_status  # noqa: E402


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--robot-repo", help="robot (microban) worktree to install into")
    parser.add_argument("--robot-branch", help="robot branch to commit on (created from HEAD if missing)")
    parser.add_argument("--training-branch", help="training branch to commit on (default: the current one)")
    parser.add_argument("--state-dir", type=Path, help="default: artifacts/home_pipeline/<prefix>_<tag>/")
    parser.add_argument("--no-push", action="store_true", help="commit but do not push")
    parser.add_argument("--status", action="store_true", help="print the state of --state-dir and exit")
    parser.add_argument("--dry-run", action="store_true", help="the whole run in minutes, not deployable")
    args = parser.parse_args(argv)
    if args.status:
        if not args.state_dir:
            parser.error("--status needs --state-dir")
    elif not args.robot_repo or not args.robot_branch:
        parser.error("--robot-repo and --robot-branch are required")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.status:
        return print_status(args.state_dir)
    pipeline = None

    def on_signal(signum, _frame) -> None:
        raise KeyboardInterrupt(signal.Signals(signum).name)

    signal.signal(signal.SIGTERM, on_signal)
    try:
        pipeline = Pipeline(robot=Path(args.robot_repo), robot_branch=args.robot_branch,
                            training_branch=args.training_branch, state_dir=args.state_dir,
                            dry=args.dry_run, push=not args.no_push)
        return pipeline.execute()
    except PipelineError as error:
        message = f"STOPPED (exit {error.code}): {error}"
        if pipeline is not None:
            pipeline.state.log(message)
        else:
            print(message, file=sys.stderr)
        return error.code
    except KeyboardInterrupt as interrupt:
        if pipeline is not None:
            pipeline.jobs.kill_all()
            pipeline.state.log(f"interrupted ({interrupt}); rerun to continue")
        return EXIT_INTERRUPTED


if __name__ == "__main__":
    sys.exit(main())
