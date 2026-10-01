"""Copy a get-up checkpoint with its action std reset, for a calm fine-tune.

Trained get-up policies end with an action std around 10. Under that much
exploration noise, bang-bang targets on the clip are the robust way to hold
a stance, so a calming fine-tune (Mjlab-Getup-Microban-CalmRoll-ImuDelay,
stage 3 in docs/getup_training_export.md) starts from a copy with the std
reset to 0.5 and fresh optimizer moments. The copy goes to a new run
directory, so it can be passed to --agent.load-run.

usage:
  uv run --locked python -m mjlab_microban.scripts.reset_getup_action_std \
    --checkpoint logs/rsl_rl/mjlab_microban_getup/<run>/model_<N>.pt \
    --out-run <new-run-name> [--std 0.5]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

STD_KEY = "distribution.std_param"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out-run", required=True, help="New run directory name under the same experiment")
    parser.add_argument("--std", type=float, default=0.5)
    args = parser.parse_args()

    source = args.checkpoint.resolve(strict=True)
    if args.std <= 0:
        raise ValueError("--std must be positive")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    actor = checkpoint.get("actor_state_dict")
    if not isinstance(actor, dict) or STD_KEY not in actor:
        raise ValueError(f"{source} has no {STD_KEY} (scalar-std actor expected)")
    old = float(actor[STD_KEY].mean())
    actor[STD_KEY] = torch.full_like(actor[STD_KEY], args.std)
    # Adam moments were accumulated for the old std; start them fresh.
    checkpoint["optimizer_state_dict"]["state"] = {}

    out_dir = source.parent.parent / args.out_run
    out_dir.mkdir(parents=False, exist_ok=False)
    out_path = out_dir / source.name
    torch.save(checkpoint, out_path)
    print(f"std {old:.3g} -> {args.std:g}; wrote {out_path}")


if __name__ == "__main__":
    main()
