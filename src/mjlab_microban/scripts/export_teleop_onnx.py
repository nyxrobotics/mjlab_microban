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
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_checkpoint_path

from mjlab_microban.tasks.microban_policy_export import (
    collect_teleop_export_provenance,
    get_microban_teleop_metadata,
    publish_gated_teleop_onnx,
    unique_teleop_onnx_temporary_path,
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
    checkpoint = Path(args.checkpoint or get_checkpoint_path(LOG_ROOT))
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    # Capture the exact bytes before loading.  Publication re-hashes the source
    # checkpoint so a concurrently written/replaced checkpoint fails closed.
    provenance = collect_teleop_export_provenance(checkpoint)
    # Load the exact canonical path that was hashed.  In particular, a retargeted
    # command-line symlink must not select different bytes after provenance capture.
    checkpoint = provenance.checkpoint_path

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
        temporary_output = unique_teleop_onnx_temporary_path(args.output)
        try:
            runner.export_policy_to_onnx(
                str(temporary_output.parent), temporary_output.name
            )
            metadata = get_microban_teleop_metadata(
                raw_env, run_path=checkpoint.parent.name
            )
            parity = publish_gated_teleop_onnx(
                temporary_output,
                args.output,
                pytorch_policy=runner.alg.get_policy().as_onnx(verbose=False),
                policy_metadata=metadata,
                provenance=provenance,
            )
        except Exception:
            temporary_output.unlink(missing_ok=True)
            raise
        print(
            "[INFO] Exported validated 18-action policy: "
            f"{args.output} (checkpoint_sha256={provenance.checkpoint_sha256}, "
            f"max_abs={parity.max_absolute_error:.3g})"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
