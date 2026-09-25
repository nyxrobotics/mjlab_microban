# Contract-v12 hardware deployment package

Only the final `model_14999.pt` checkpoint (15,000 completed PPO updates) may be
packaged for Microban. The 3,000, 7,000 and 10,000 boundaries, activation
canaries, interrupted checkpoints, and every preview checkpoint remain
simulation-only even when one of their diagnostic reports passes.

## One-command final release

After training has written `model_14999.pt`, the shortest fail-closed path is:

```bash
scripts/finalize_microban_teleop_v12.sh <run-name>
```

This one command runs the canonical 9x300 locomotion gate, the checkpoint's
required full tracking/safety profile, PyTorch/ONNX/ONNX Runtime parity, creates
and independently validates the hash-bound stage manifest, packages the final
ONNX, and runs the adjacent physical Microban repository's real CPU runtime and
walk-fallback smoke. It publishes these run-specific outputs only after all
checks pass:

```text
artifacts/teleop_v12_gates/<run-name>_model_14999_gate.json
artifacts/teleop_v12_releases/<run-name>_model_14999.onnx
artifacts/teleop_v12_releases/<run-name>_model_14999_deployment_receipt.json
```

The gate JSON is the evaluation manifest; the deployment receipt binds the
checkpoint, manifest, final ONNX, final parity results, and the runtime
validator report by SHA-256. An existing ONNX or receipt is not replaced unless
`--force` is explicit. `--force` only permits atomic replacement and does not
relax a gate.

The active 10,100-to-15,000 run created on 2026-09-26 will write the exact
final checkpoint path:

```text
/home/kanade/Git-projects/mjlab_microban_v12_deferred/logs/rsl_rl/mjlab_microban_teleop_v12/2026-09-26_00-40-44_v12_deadline_post_canary_10100_to15000_20260926/model_14999.pt
```

Its final release command is:

```bash
scripts/finalize_microban_teleop_v12.sh \
  2026-09-26_00-40-44_v12_deadline_post_canary_10100_to15000_20260926
```

To publish the same accepted build directly at the physical runtime's policy
path, use the optional output argument (and `--force` only when replacing an
existing accepted file):

```bash
scripts/finalize_microban_teleop_v12.sh \
  2026-09-26_00-40-44_v12_deadline_post_canary_10100_to15000_20260926 \
  ../microban/src/agents/pico_teleop.onnx --force
```

Do not launch this while training is still active. Before completion it exits
without starting evaluation because the exact final checkpoint is absent.

## Two-step low-level path

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
   updates, canonical-boundary kind, `status=pass`, and either the canonical
   final perturbation profile or the explicitly authorized deadline-final
   profile. The latter is accepted only when the checkpoint contains the exact
   pinned fallback and post-canary lineage; changing only the gate/profile name
   is rejected.
3. Captures immutable checkpoint bytes before loading the actor, then revalidates
   the pinned legacy checkpoint/probe and frozen legacy tensors. It also
   requires the corrected bilateral-site revision and the authenticated
   `swap` migration from the pinned raw `model_9200.pt`; an unmarked pre-fix
   checkpoint, a fresh-but-unrelated revision marker, or the diagnostic
   `zero_hand` migration cannot be exported.
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
   its fixed 16-input runtime smoke, against the final temporary file. A pass
   must contain the separate CPU-only `walk_fallback` load/inference report;
   merely accepting the learned graph is insufficient.
8. Embeds and rechecks the SHA-256s of that validator, its
   `src/moves/pico_hybrid.py` contract parser, the
   `src/moves/policy_selector.py` fallback selector, `src/moves/walk.py`,
   the direct-arm runtime and contract, network-input parser and data contract,
   production entrypoint, scheduler, `src/constants.py`, the exact
   `src/agents/walk.onnx`, and the physical repository's `uv.lock`. The
   validator independently compares those embedded identities with the files
   that actually performed admission, reports the complete 13-entry identity,
   and rehashes all 13 after its CPU smokes to reject a mid-validation change.
   The production learned-policy loader performs the same check before and
   after its fixed CPU smoke, so a post-validation rsync mismatch rejects the
   learned actor and preserves the walk fallback.
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
