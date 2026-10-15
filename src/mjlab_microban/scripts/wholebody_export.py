"""Export a passed whole-body checkpoint as ``wholebody.onnx`` (contract microban-wholebody-1).

  uv run --locked python src/mjlab_microban/scripts/wholebody_export.py \\
      --checkpoint logs/rsl_rl/microban_wholebody/<run>/model_29999.pt \\
      --benchmark-report artifacts/wholebody/<run>/model_29999.json \\
      --eval-only-report artifacts/wholebody_pico/<run>/model_29999.json --output wholebody.onnx

Both benchmark reports must be this checkpoint's, start the pieces as in
teleoperation (at HOME, the reference from a standing frame) and pass the
full run (``verdicts["final"]["pass"]``; decided 2026-10-15: the robot does
not fall, the tracking errors are accepted): ``--benchmark-report`` on the
training kinds' evaluation set (the action-rate line), ``--eval-only-report``
on an evaluation-only set (the PICO recordings, ``--eval-only-set``; the
action-rate and fall lines).
``--dry-run`` accepts any report of the checkpoint and marks the file not
deployable.  The
self-test rows (8-64 observation histories) come from an evaluation rollout
of the checkpoint on the CPU, with the torch actor's output for each, and
are re-run in ONNX Runtime before the file is published.
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx
from mjlab.tasks.registry import load_rl_cfg
from tensordict import TensorDict

from mjlab_microban.policy_contract import ACTION_JOINT_NAMES, PolicyContractError, sha256_file
from mjlab_microban.scripts.wholebody_benchmark import benchmark_env_cfg
from mjlab_microban.wholebody import benchmark, contract
from mjlab_microban.wholebody.env_cfg import ACTOR_GROUP, EVAL_MOTION_DIR, TASK_ID
from mjlab_microban.wholebody.runner import WholebodyOnPolicyRunner

SELF_TEST_RECORD_EVERY = 10
SELF_TEST_PIECES = 4


def benchmark_sha256(path: Path, checkpoint_sha256: str, eval_only_set: bool, dry_run: bool) -> str:
    """The SHA-256 of this checkpoint's benchmark report; unless ``dry_run`` it
    must start as in teleoperation, be on the set asked for and pass the full run (``verdicts["final"]["pass"]``)."""
    report = json.loads(path.read_text(encoding="utf-8"))
    metadata = report["metadata"]
    if metadata["checkpoint_sha256"] != checkpoint_sha256:
        raise PolicyContractError(f"{path} is not the benchmark of this checkpoint")
    if not dry_run:
        if metadata.get("start") != benchmark.START:
            raise PolicyContractError(f"{path} does not start the pieces from a standing reference at HOME")
        if bool(metadata.get("eval_only_set")) != eval_only_set:
            kind = "an evaluation-only set" if eval_only_set else "the training kinds' set"
            raise PolicyContractError(f"{path} is not the benchmark on {kind}")
        if not report["verdicts"]["final"]["pass"]:
            raise PolicyContractError(f"{path} does not pass the full-run lines")
    return sha256_file(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--benchmark-report", type=Path, required=True, help="On the training kinds' set")
    parser.add_argument("--eval-only-report", type=Path, required=True, help="On an evaluation-only set (PICO)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--motion-file", default=EVAL_MOTION_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve(strict=True)
    output = args.output.resolve()
    if output.exists() and not args.replace:
        raise FileExistsError(f"{output} exists (use --replace)")
    checkpoint_sha256 = sha256_file(checkpoint)
    report_sha256 = benchmark_sha256(args.benchmark_report, checkpoint_sha256, False, args.dry_run)
    eval_only_report_sha256 = benchmark_sha256(args.eval_only_report, checkpoint_sha256, True, args.dry_run)

    cfg = benchmark_env_cfg(args.motion_file, 10.0, SELF_TEST_PIECES, seed=0)
    pieces = benchmark.load_pieces(
        args.motion_file, ACTION_JOINT_NAMES, 10.0, cfg.decimation * cfg.sim.mujoco.timestep,
        benchmark.servo_no_load_speed_rad_s(),
    )
    pieces = sorted(pieces, key=lambda piece: -piece.steps)[:SELF_TEST_PIECES]
    cfg.scene.num_envs = len(pieces)
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp.onnx")
    try:
        runner = WholebodyOnPolicyRunner(RslRlVecEnvWrapper(env), asdict(load_rl_cfg(TASK_ID)), device="cpu")
        runner.load(str(checkpoint), load_cfg={"actor": True}, map_location="cpu")
        policy = runner.get_inference_policy(device="cpu")
        _, recorded = benchmark.rollout(env, policy, pieces, record_every=SELF_TEST_RECORD_EVERY)
        rows = contract.select_self_test_rows([history.tolist() for history in recorded])
        batch = TensorDict(
            {ACTOR_GROUP: torch.tensor(rows, dtype=torch.float32)}, batch_size=[len(rows)]
        )
        with torch.inference_mode():
            actions = policy(batch).double().tolist()
        metadata = contract.contract_metadata(
            joint_names=env.scene["robot"].joint_names,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            benchmark_report_sha256=report_sha256,
            eval_only_benchmark_report_sha256=eval_only_report_sha256,
            self_test_rows=rows,
            self_test_actions=actions,
            dry_run=args.dry_run,
        )
        contract.export_policy_to_onnx(runner.alg.get_policy(), temporary)
        contract.check_onnx_io(temporary)
        attach_metadata_to_onnx(str(temporary), metadata)
        worst = contract.check_self_test(temporary, metadata)
        if sha256_file(checkpoint) != checkpoint_sha256:
            raise PolicyContractError("checkpoint changed during export")
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, output)
        print(f"self-test {len(rows)} rows at {worst:.3f} x bound; exported {output}")
    finally:
        temporary.unlink(missing_ok=True)
        env.close()


if __name__ == "__main__":
    main()
