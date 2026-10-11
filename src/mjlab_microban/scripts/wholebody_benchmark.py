"""Evaluate whole-body checkpoints on the evaluation set, per motion kind.

  uv run --locked python src/mjlab_microban/scripts/wholebody_benchmark.py \\
      --checkpoint logs/rsl_rl/microban_wholebody/<run>/model_4999.pt [--checkpoint ...]

Writes ``artifacts/wholebody/<run>/<model>.json`` (verdicts, per kind, per
window) and ``-kinds.csv`` / ``-windows.csv`` next to it.  Pass two
checkpoints of one run to see whether success and errors still improve.
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
from mjlab_microban.wholebody.env_cfg import EVAL_MOTION_DIR, TASK_ID
from mjlab_microban.wholebody.runner import WholebodyOnPolicyRunner


def benchmark_env_cfg(motion_file: str, clip_seconds: float, num_envs: int, seed: int):
    cfg = load_env_cfg(TASK_ID, play=True)
    motion = cfg.commands["motion"]
    motion.motion_file = motion_file
    motion.sampling_mode = "start"
    motion.resample_on_clip_end = False
    cfg.episode_length_s = clip_seconds
    cfg.auto_reset = False
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    return cfg


def evaluate(checkpoint: Path, args: argparse.Namespace) -> dict:
    no_load = benchmark.servo_no_load_speed_rad_s()
    specs = benchmark.load_windows(args.motion_file, ACTION_JOINT_NAMES, args.clip_seconds, no_load)
    windows = benchmark.eligible(specs, args.clip_seconds)
    if not windows:
        raise ValueError(f"no window of {args.clip_seconds} s in {args.motion_file}")
    results = []
    batches = [windows[i : i + args.num_envs] for i in range(0, len(windows), args.num_envs)]
    for index, batch in enumerate(batches):
        cfg = benchmark_env_cfg(args.motion_file, args.clip_seconds, len(batch), args.seed + index)
        control_steps = int(round(args.clip_seconds / (cfg.decimation * cfg.sim.mujoco.timestep)))
        env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
        try:
            runner = WholebodyOnPolicyRunner(RslRlVecEnvWrapper(env), asdict(load_rl_cfg(TASK_ID)), device=args.device)
            runner.load(str(checkpoint), load_cfg={"actor": True}, map_location=args.device)
            policy = runner.get_inference_policy(device=args.device)
            batch_results, _ = benchmark.rollout(env, policy, batch, control_steps)
            results.extend(batch_results)
        finally:
            env.close()
        print(f"[{index + 1}/{len(batches)}] {len(results)}/{len(windows)} windows")
    run = checkpoint.parent.name
    metadata = {
        "task": TASK_ID,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "motion_file": str(Path(args.motion_file).resolve()),
        "clip_seconds": args.clip_seconds,
        "control_steps": control_steps,
        "servo_no_load_speed_rad_s": no_load,
        "windows_total": len(specs),
        "windows_shorter_than_clip": len(specs) - len(windows),
        "seed": args.seed,
    }
    report = benchmark.write_outputs(Path(args.output_root) / run, checkpoint.stem, metadata, results)
    print(f"{checkpoint}:\n{benchmark.format_summary(report)}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--motion-file", default=EVAL_MOTION_DIR)
    parser.add_argument("--clip-seconds", type=float, default=10.0)
    parser.add_argument("--num-envs", type=int, default=256, help="Windows run at once")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", default="artifacts/wholebody")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    for checkpoint in args.checkpoint:
        evaluate(checkpoint.resolve(strict=True), args)


if __name__ == "__main__":
    main()
