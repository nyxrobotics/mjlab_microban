# Microban walk004 tracking training

This is the reproducible initial-stage recipe for the fixed walk004 99-input
tracking policy. It is a simulator training/evaluation path, not yet the live
PICO teleoperation policy and not authorization to send targets to the robot.

## First canary

Run 51 PPO updates and save updates 0, 25 and 50:

```bash
scripts/train_microban_tracking.sh start \
  --target-iterations 51 \
  --save-interval 25 \
  --run-name walk004_startclean_51canary
```

The wrapper pins the known walk004 SHA-256 and the current safe preflight
contract: 2,048 environments, seed 42, synchronized clip start, 267 steps
(5.34 s at 50 Hz), finite-horizon returns, raw actor observations, bounded
actions, learning rate `3e-5`, three learning epochs, four mini-batches, actor
corruption disabled, and all initial-stage domain/motion randomization set to
zero. Foot friction is fixed at 1.0. The training-only anchor and end-effector
termination thresholds are 0.12 m and 1.0 m; the evaluator keeps its independent,
stricter acceptance criteria.

The 2,048-environment limit is mandatory with `nconmax=512` and `njmax=2048`.
The same contact capacity at 4,096 environments caused a measured 12.7 GB
single EPA allocation and exhausted the 24 GiB training GPU.

The existing 51-update run
`2026-09-25_07-43-48_v8l_rawactor_startclean_4096_51canary` failed the full-clip
checkpoint gate at all saved checkpoints. It must not be extended. A new canary
may be extended only after its later checkpoint materially improves the gate
receipt and the selected continuation checkpoint is recorded.

Evaluate each canary checkpoint explicitly:

```bash
RUN=2026-09-25_HH-MM-SS_walk004_startclean_51canary
scripts/train_microban_tracking.sh evaluate "$RUN" --checkpoint-iteration 0
scripts/train_microban_tracking.sh evaluate "$RUN" --checkpoint-iteration 25
scripts/train_microban_tracking.sh evaluate "$RUN" --checkpoint-iteration 50
```

The evaluator exits 0 only on a full pass and otherwise writes a diagnostic JSON
receipt and exits 2. A longer training job should not be justified from TensorBoard
episode length or reward alone; use the full nominal/robust gate measurements.

## Exact continuation

If a checkpoint is selected after evaluation, set the desired **total** update
count, not an additional count:

```bash
scripts/train_microban_tracking.sh resume "$RUN" \
  --target-iterations 251 \
  --save-interval 25 \
  --run-name walk004_startclean_resume_to_251
```

MjLab writes a resumed job to a new timestamp directory. Use that new directory
for the next evaluation or continuation. The wrapper reads
`infos.env_state.common_step_counter` from the newest numeric checkpoint and
divides it by the fixed 24-step rollout. This is the lineage-wide completed
update count. It intentionally does not use `model_N.pt` as `N+1`: RSL-RL starts
a resumed loop at the saved numeric iteration, so suffix arithmetic becomes
off-by-one after the first resume.

Before resuming, the wrapper rejects a source run unless its resolved `env.yaml`
and `agent.yaml` match the pinned initial-stage contract. It also anchors the run
and checkpoint regular expressions, verifies the checkpoint's internal iteration,
and records the selected checkpoint SHA-256. Unknown training options, `--`
passthrough, and `MICROBAN_TRACKING_MOTION_FILE` are rejected.

Use `--evaluate-after` only when the cost of the two 256-environment gate passes
is desired immediately after training. `--dry-run` performs validation and writes
the exact planned command without launching MuJoCo.

## Evidence and later teacher/BC integration

Each launch writes an ignored JSON manifest below
`artifacts/microban_tracking_training/invocations/`. It records the argv array,
shell-rendered command, motion/checkpoint hashes, git commit/status/diff hash,
critical source-file hashes, completed/remaining update counts, and exit status.
MjLab also stores the fully resolved YAML and repository diff in the output run.
Keep the manifest, run directory, and evaluator receipt together.

There is intentionally no guessed teacher/BC flag in this wrapper. When the
teacher or behavior-cloning CLI contract lands, add its exact options to the
explicit allowlist, include them in the manifest contract, and add resume
validation before using them. Do not use arbitrary Tyro passthrough in the
meantime.

## Developer verification

```bash
bash -n scripts/train_microban_tracking.sh
uvx --from shellcheck-py shellcheck scripts/train_microban_tracking.sh
uv run --locked --with pytest python -m pytest -q \
  tests/test_microban_tracking_training_wrapper.py
```
