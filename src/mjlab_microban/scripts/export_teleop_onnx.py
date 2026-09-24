# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Export a trained Microban PICO teleoperation checkpoint to deployment ONNX.

The resulting model has one 18-wide ``actions`` output.  Its metadata vectors
are in that exact output order and do not contain the three HMD-controlled
head/neck joints.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_checkpoint_path

from mjlab_microban.tasks.microban_policy_export import (
    get_microban_teleop_metadata,
    validate_action_only_onnx,
)

TASK = "Mjlab-Teleop-Microban"
LOG_ROOT = Path("logs/rsl_rl/mjlab_microban_teleop")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint .pt (default: latest teleop run)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("microban_teleop.onnx"),
        help="Output ONNX path",
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint or get_checkpoint_path(LOG_ROOT)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    env_cfg = load_env_cfg(TASK, play=True)
    env_cfg.scene.num_envs = 1
    agent_cfg = load_rl_cfg(TASK)
    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    env = RslRlVecEnvWrapper(raw_env)

    try:
        runner_cls = load_runner_cls(TASK)
        runner = runner_cls(env, asdict(agent_cfg), device=args.device)
        runner.load(
            str(checkpoint),
            load_cfg={"actor": True},
            strict=True,
            map_location=args.device,
        )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = args.output.with_name(f".{args.output.name}.tmp")
        temporary_output.unlink(missing_ok=True)
        try:
            runner.export_policy_to_onnx(str(args.output.parent), temporary_output.name)
            validate_action_only_onnx(temporary_output)
            metadata = get_microban_teleop_metadata(
                raw_env, run_path=checkpoint.parent.name
            )
            attach_metadata_to_onnx(str(temporary_output), metadata)
            # Validate the metadata-bearing artifact itself before atomically
            # replacing the last known-good deployment policy.
            validate_action_only_onnx(temporary_output)
            temporary_output.replace(args.output)
        except Exception:
            temporary_output.unlink(missing_ok=True)
            raise
        print(f"[INFO] Exported validated 18-action policy: {args.output}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
