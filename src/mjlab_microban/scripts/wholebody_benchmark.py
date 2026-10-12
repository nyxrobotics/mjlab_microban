"""Evaluate whole-body checkpoints on the evaluation set, per motion kind.

  uv run --locked python src/mjlab_microban/scripts/wholebody_benchmark.py \\
      --checkpoint logs/rsl_rl/microban_wholebody/<run>/model_4999.pt [--checkpoint ...]

Writes ``artifacts/wholebody/<run>/<model>.json`` (verdicts, per kind, per
piece) and ``-kinds.csv`` / ``-pieces.csv`` next to it.  Pass two
checkpoints of one run to see whether success and errors still improve: with
the same seed they meet the same conditions piece by piece.  The kinds of
the training set (``--train-motion-file``) must all be evaluated.

The robot starts every piece at rest at HOME and its first second is not
judged; ``--start reference`` places it on the reference's first frame
instead, as earlier reports did (see benchmark.py).

An evaluation-only set is judged on its own kinds with ``--eval-only-set``
(no training kinds or walk, run and sprint required; the same lines):

  uv run --locked python src/mjlab_microban/scripts/wholebody_benchmark.py \\
      --checkpoint <model.pt> --motion-file data/wholebody/pico_eval/eval --eval-only-set \\
      --output-root artifacts/wholebody_pico
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import mjlab_microban.tasks  # noqa: F401  (registers the task)
from mjlab_microban.policy_contract import ACTION_JOINT_NAMES, sha256_file
from mjlab_microban.wholebody import benchmark
from mjlab_microban.wholebody.env_cfg import EVAL_MOTION_DIR, TASK_ID, TRAIN_MOTION_DIR
from mjlab_microban.wholebody.runner import WholebodyOnPolicyRunner


def benchmark_env_cfg(motion_file: str, clip_seconds: float, num_envs: int, seed: int):
    cfg = load_env_cfg(TASK_ID, play=True)
    motion = cfg.commands["motion"]
    motion.motion_file = motion_file
    motion.sampling_mode = "start"
    cfg.episode_length_s = clip_seconds
    cfg.auto_reset = False
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    return cfg


def evaluate(checkpoint: Path, args: argparse.Namespace) -> dict:
    no_load = benchmark.servo_no_load_speed_rad_s()
    probe = benchmark_env_cfg(args.motion_file, args.clip_seconds, 1, 0)
    step_dt = probe.decimation * probe.sim.mujoco.timestep
    pieces = benchmark.load_pieces(
        args.motion_file, ACTION_JOINT_NAMES, args.clip_seconds, step_dt, no_load, start=args.start
    )
    expected_kinds = set() if args.eval_only_set else benchmark.dataset_kinds(args.train_motion_file)
    results = []
    batches = [pieces[i : i + args.num_envs] for i in range(0, len(pieces), args.num_envs)]
    for index, batch in enumerate(batches):
        cfg = benchmark_env_cfg(args.motion_file, args.clip_seconds, len(batch), args.seed + index)
        env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
        try:
            runner = WholebodyOnPolicyRunner(RslRlVecEnvWrapper(env), asdict(load_rl_cfg(TASK_ID)), device=args.device)
            runner.load(str(checkpoint), load_cfg={"actor": True}, map_location=args.device)
            policy = runner.get_inference_policy(device=args.device)
            batch_results, _ = benchmark.rollout(env, policy, batch, start=args.start)
            results.extend(batch_results)
        finally:
            env.close()
        print(f"[{index + 1}/{len(batches)}] {len(results)}/{len(pieces)} pieces")
    run = checkpoint.parent.name
    metadata = {
        "task": TASK_ID,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "motion_file": str(Path(args.motion_file).resolve()),
        "train_motion_file": None if args.eval_only_set else str(Path(args.train_motion_file).resolve()),
        "eval_only_set": args.eval_only_set,
        "clip_seconds": args.clip_seconds,
        "start": args.start,
        "servo_no_load_speed_rad_s": no_load,
        "pieces": len(pieces),
        "seconds": sum(p.seconds for p in pieces),
        "judged_seconds": sum(p.judged_seconds for p in pieces),
        "num_envs": args.num_envs,
        "seed": args.seed,
    }
    report = benchmark.write_outputs(
        Path(args.output_root) / run, checkpoint.stem, metadata, results, expected_kinds,
        require_locomotion=not args.eval_only_set,
    )
    print(f"{checkpoint}:\n{benchmark.format_summary(report)}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--motion-file", default=EVAL_MOTION_DIR)
    parser.add_argument("--train-motion-file", default=TRAIN_MOTION_DIR, help="Its kinds must all be evaluated")
    parser.add_argument(
        "--eval-only-set",
        action="store_true",
        help="Judge only the kinds of --motion-file (an evaluation-only set such as data/wholebody/pico_eval/eval): "
        "the training kinds and walk, run and sprint are not required",
    )
    parser.add_argument("--clip-seconds", type=float, default=10.0)
    parser.add_argument(
        "--start",
        choices=("home", "reference"),
        default="home",
        help="home: at rest at HOME, the first second not judged (as on the robot); "
        "reference: on the reference's first frame, judged from the first step (earlier reports)",
    )
    parser.add_argument("--num-envs", type=int, default=1024, help="Pieces run at once")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", default="artifacts/wholebody")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    for checkpoint in args.checkpoint:
        evaluate(checkpoint.resolve(strict=True), args)


if __name__ == "__main__":
    main()
