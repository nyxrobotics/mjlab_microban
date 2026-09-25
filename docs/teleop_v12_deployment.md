# Contract-v12 hardware deployment package

Only the final `model_14999.pt` checkpoint (15,000 completed PPO updates) may be
packaged for Microban. The 3,000, 7,000 and 10,000 boundaries, activation
canaries, interrupted checkpoints, and every preview checkpoint remain
simulation-only even when one of their diagnostic reports passes.

First create the current final stage gate. This reruns the locomotion, tracking,
PyTorch/ONNX and ONNX Runtime CPU checks and writes a schema-v2 gate whose hashes
bind all three reports and the unmodified checkpoint:

```bash
scripts/evaluate_microban_teleop_v12_stage.sh <run-name> 14999
```

Then build the deployment ONNX:

```bash
scripts/export_microban_teleop_v12_deployment.sh \
  <run-name> artifacts/microban_teleop_v12.onnx
```

To publish directly into the adjacent physical-robot checkout after review:

```bash
scripts/export_microban_teleop_v12_deployment.sh \
  <run-name> ../microban/src/agents/pico_teleop.onnx --force
```

`--force` does not make validation optional. It only permits the final atomic
rename to replace an existing artifact after every check has passed. On any
failure the previous output is left byte-for-byte unchanged.

The packager performs the following fail-closed sequence on CPU:

1. Rebuilds and compares the supplied stage gate using its current evaluator
   code, rehashing the checkpoint, all three reports, and the gate ONNX.
2. Requires exactly `model_14999.pt`, iteration `14999`, 15,000 completed
   updates, canonical-boundary kind, final perturbation tracking profile, and
   `status=pass`.
3. Captures immutable checkpoint bytes before loading the actor, then revalidates
   the pinned legacy checkpoint/probe and frozen legacy tensors.
4. Exports a fresh fixed-shape `obs[1,83] -> actions[1,18]` float32 graph.
5. Copies the hash-bound locomotion, tracking and ONNX evidence into the exact
   metadata keys required by Microban. The per-joint finite-amplitude guard is
   derived from the final tracking envelope; it is never invented or widened by
   the packager. The existing hand wire envelope remains exactly ±0.08 m per
   axis; the smaller reachable joint-box/FK training subset is recorded
   separately in `hand_target_fk` metadata. The metadata also records the shared
   measured dynamic soft-limit overshoot allowance of 5 degrees
   (`0.08726646259971647 rad`) and the distinct commanded-target excess
   tolerance of `1e-7 rad`; the latter is not widened by the dynamic allowance.
6. Runs the same deterministic 64-sample, all-83-column parity corpus through
   PyTorch, ONNX `ReferenceEvaluator`, and ONNX Runtime
   `CPUExecutionProvider`, both before and after metadata attachment.
7. Runs the physical repository's real `tools/validate_pico_policy.py`, including
   its fixed 16-input runtime smoke, against the final temporary file.
8. Embeds and rechecks the SHA-256s of that validator, its
   `src/moves/pico_hybrid.py` contract parser, and the physical repository's
   `uv.lock`, so the runtime checked is the runtime recorded.
9. Rehashes and revalidates the complete gate lineage once more immediately
   before an `os.replace` plus directory `fsync` publishes the file.

The command intentionally has no diagnostic/nonaccepted mode. If the final gate
or its evidence is absent, stale, changed, non-final, or rejected by the current
robot runtime, no deployable ONNX is produced.

Focused CPU-only regression tests:

```bash
uv run --locked --with pytest python -m pytest -q \
  tests/test_teleop_v12_deployment.py
uv run --locked --with ruff ruff check \
  src/mjlab_microban/scripts/export_teleop_v12_deployment.py \
  tests/test_teleop_v12_deployment.py
bash -n scripts/export_microban_teleop_v12_deployment.sh
```
