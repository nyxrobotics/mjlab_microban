# Microban PICO teleoperation contract v10

This is the active reproducible procedure for the dedicated Microban PICO
teleoperation policy. Contract v10 migrates the complete accepted v9
`model_1499.pt` training state, fixes PPO learning rate at `1e-5`, and trains
through boundaries `3000`, `7000`, `10000`, and `15000`. Do not start from
`model_2500.pt`, use the retired v8/v9 wrappers, or override canonical settings.

## Pinned migration source

The only permitted migration pair is:

- checkpoint:
  `logs/rsl_rl/mjlab_microban_teleop/2026-09-25_10-58-50_teleop_v9_safe_v3_v92r_stage0_1500/model_1499.pt`
- checkpoint SHA-256:
  `de8b6139872179679a16d72f3007f6d96cf65c97fa88565841eaa5f89511a65f`
- gate:
  `artifacts/teleop_v9_gates/2026-09-25_10-58-50_teleop_v9_safe_v3_v92r_stage0_1500_boundary_1500_gate.json`
- gate SHA-256:
  `acb2e39411155d70aed2b18561a243bd8ad20eef56e0942d2dafbb4f96f39b7c`

The validator also pins v9 provenance, its source tree, the accepted
safe-velocity checkpoint and receipt, iteration `1499`, common step `36000`,
and source optimizer learning rate `7.593750000000002e-05`. Migration restores
actor, critic, Adam moments, iteration, and common step before changing only the
optimizer/scalar learning rate to `1e-5`.

## Preflight and first canary

From the repository root:

```bash
uv run --locked --with pytest python -m pytest -q
bash -n scripts/train_microban_teleop_v10.sh \
  scripts/evaluate_microban_teleop_v10_stage.sh

uv run --locked python -m mjlab_microban.scripts.teleop_v10_stage \
  validate-migration \
  logs/rsl_rl/mjlab_microban_teleop/2026-09-25_10-58-50_teleop_v9_safe_v3_v92r_stage0_1500/model_1499.pt \
  artifacts/teleop_v9_gates/2026-09-25_10-58-50_teleop_v9_safe_v3_v92r_stage0_1500_boundary_1500_gate.json

scripts/train_microban_teleop_v10.sh migrate \
  logs/rsl_rl/mjlab_microban_teleop/2026-09-25_10-58-50_teleop_v9_safe_v3_v92r_stage0_1500/model_1499.pt \
  artifacts/teleop_v9_gates/2026-09-25_10-58-50_teleop_v9_safe_v3_v92r_stage0_1500_boundary_1500_gate.json \
  --canary --agent.run-name <v10-run-name>

scripts/evaluate_microban_teleop_v10_stage.sh --canary \
  <timestamped-canary-run-directory>
scripts/train_microban_teleop_v10.sh resume \
  <timestamped-canary-run-directory> \
  --agent.run-name <next-run-suffix>
```

MJLab prefixes `--agent.run-name` with a timestamp and every invocation writes a
new run directory. Always copy the complete directory name printed by training
into the evaluator or next resume command.

The canary is exactly 100 PPO updates and evaluates seed 42 over neutral,
low-forward, and left/right low/mid-yaw scenarios. Its receipt is diagnostic and
cannot authorize deployment. Resume validation requires that exact receipt; a
checkpoint cannot skip the canary or a stage gate.

## Canonical stages and gates

Every training process uses 2,048 environments, seed 42, 24 rollout steps,
save interval 100, fixed learning rate `1e-5`, and no caller-provided training
override other than run name. A normal resume runs exactly to its current stage
boundary. Evaluate each boundary before resuming:

```bash
scripts/evaluate_microban_teleop_v10_stage.sh <v10-run-name>
scripts/train_microban_teleop_v10.sh resume <v10-run-name> \
  --agent.run-name <next-stage-run-suffix>
```

Boundary gates use seeds 42, 43, and 44. Boundary `3000` covers signed low/mid
axes; `7000` adds the full locomotion envelope; `10000` adds hand targets and
moving-HMD reports; `15000` adds the full scenario suite, moving-HMD reports,
and deployment performance criteria. Curriculum events remain at updates 500,
1500, 3000, 4500, 6000, 7000, 8500, 10000, 12000, and 15000.

If power is lost, first run the evaluator with `--canary` on the interrupted run
directory, then resume it. RSL-RL periodic saves occur after zero-based
iterations divisible by 100, so a recovery file such as `model_1600.pt`
represents 1,601 completed updates; v10 explicitly accepts both deliberate
100-update segment ends and this one-update-offset recovery cadence. Resume
selects the greatest numeric checkpoint, verifies its SHA/provenance and exact
canary receipt, and preserves the original stage lineage. Do not rename
checkpoints or copy one into a different run directory.

Checkpoint publication itself uses a same-directory temporary file, file and
directory `fsync`, and atomic rename. A power loss during serialization can
therefore leave an ignored `.tmp` file but cannot replace the last complete
`model_<iteration>.pt` with a partial checkpoint.

## Final export

Only `model_14999.pt` plus the schema-3 boundary-15000 gate can produce a
deployment artifact:

```bash
uv run --locked python -m mjlab_microban.scripts.export_teleop_onnx \
  --checkpoint logs/rsl_rl/mjlab_microban_teleop/<v10-run-name>/model_14999.pt \
  --acceptance-receipt \
    artifacts/teleop_v10_gates/<v10-run-name>_boundary_15000_gate.json \
  --require-final-acceptance \
  --device cpu \
  --output artifacts/microban_teleop.onnx
```

The exporter performs deterministic PyTorch/ONNX parity and atomically
publishes only a metadata-complete model. The robot must validate contract `10`,
provenance schema `2`, the `10000->15000` final stage, evaluator and acceptance
revision `v10_1`, all migration/safe-source hashes, and the final receipt before
enabling policy motor commands. See
[`teleop_onnx_export_gate.md`](teleop_onnx_export_gate.md) for the wire metadata.
