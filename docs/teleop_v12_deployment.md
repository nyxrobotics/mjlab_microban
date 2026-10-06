# Contract-v12 hardware deployment package

Only the final `model_14999.pt` checkpoint (15,000 completed PPO updates) may be
packaged for Microban. The 3,000, 7,000 and 10,000 boundaries, activation
canaries and interrupted checkpoints remain simulation-only even when one of
their diagnostic reports passes.

## Packaging

`scripts/retrain_all_for_home.py` packages the final checkpoint after its
stage gate passed (`scripts/evaluate_microban_teleop_v12_stage.sh <run-name>
14999` runs the same three evaluators and writes the schema-v2 gate whose
hashes bind all three reports and the unmodified checkpoint):

```bash
uv run --locked --with onnxruntime --with "protobuf<7" python -m \
  mjlab_microban.scripts.export_teleop_v12_deployment \
  --checkpoint logs/rsl_rl/mjlab_microban_teleop_v12/<run-name>/model_14999.pt \
  --stage-gate artifacts/teleop_v12_gates/<run-name>_model_14999_gate.json \
  --microban-repo ../microban --output ../microban/src/agents/pico_teleop.onnx --force
```

`--force` does not make validation optional. It only permits the final atomic
rename to replace an existing artifact after every check has passed. On any
failure the previous output is left byte-for-byte unchanged.

The packager performs the following fail-closed sequence on CPU:

1. Rebuilds and compares the supplied stage gate using its current evaluator
   code, rehashing the checkpoint, all three reports, and the gate ONNX.
2. Requires exactly `model_14999.pt`, iteration `14999`, 15,000 completed
   updates, canonical-boundary kind, `status=pass`, and the final profile
   (see "Stage-gate profiles" below).
3. Captures immutable checkpoint bytes before loading the actor, then revalidates
   the pinned legacy checkpoint/probe and frozen legacy tensors. It also
   requires the corrected bilateral-site revision (every chain is bootstrapped
   with it; there is no checkpoint migration).
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

## Stage-gate profiles (one table at every HOME)

Each clock has exactly one tracking profile
(`evaluate_teleop_v12_tracking.required_tracking_profile`); there are no
per-HOME, per-recipe, strict, deadline or allowance variants. Every profile
keeps its scenarios, perturbation, target-column ablation targets and
non-accuracy checks (no falls, finite, actual soft limits, raw-action
recurrence, forced HMD motion, observation coverage, ablation response, twist
direction). The profiles that judge accuracy use one table:

| limit | 10000 boundary, 10100 canary | whole body, final |
| --- | --- | --- |
| hand RMS | 0.040 m | 0.040 m |
| hand P95 | 0.05 m | 0.07 m |
| foot RMS | 0.05 m | 0.05 m |
| foot P95 | 0.08 m | 0.08 m |

Hand RMS 0.040 m is the user's decision for every HOME ("手は 0.04mまで許容で
いいんじゃない？"). The accuracy-judging profiles keep their published
`_deployed_accuracy_v1` names; the final one,
`full_body_reachable_performance_perturbation_v2_deployed_accuracy_v1`, is the
`v12_tracking_profile` the robot validator accepts. Every HOME trains a fresh
pose-release chain (`scripts/train_microban_teleop_v12.sh start --source
WALK.pt`). A HOME whose trunk leans forward labels its foot/hand target columns
`robot_home_levelled_trunk_xyz_forward_left_up` and is packaged by packager v7
(forward-lean) or a `<tag>` revision.

## Boundary gates in the package

Checkpoint infos carry no parent hash, so the packager follows the resume
chain each run directory records in `params/agent.yaml` (`load_run: ^RUN$`,
`load_checkpoint: ^model_N[.]pt$`) from the final checkpoint back to the first
run that did not resume (or a corner-rescue seed copy without params). The
pose-release final rescue's seed (`pr_final_rescue_seed_<sha16>/`) carries a
byte copy of its parent run's `params/agent.yaml`, so the walk continues from
the staged `model_14900` to the parent run's gated 10100 canary and 10000
boundary (docs/teleop_v12_hand_pose_release_final_rescue.md).
`--boundary-gate GATE.json` (repeatable) names earlier canonical-boundary or
activation-canary gates explicitly. Each gate is fully revalidated. Its
checkpoint must lie on that resume chain, so a sibling with the same markers
is refused. It must also share the final checkpoint's contract, recipe,
bootstrap, HOME and site-order markers, and the final must carry its
corner-rescue markers unchanged. The packager
records `v12_boundary_stage_gates_json` (clock, checkpoint kind, checkpoint
and gate SHA-256, tracking profile) with
`v12_boundary_stage_gates_semantics`
(`..._resume_ancestor_boundary_gates_..._v2`) in the package.

A pose-release final must record both its 10000 boundary and its 10100
canary (where hand and then foot accuracy are first judged), at every HOME.
Any of the two that is not passed explicitly is discovered as
`artifacts/teleop_v12_gates/{run}_model_{9999|10099}_gate.json` of the
ancestor on the resume chain. If either is still missing, packaging is
refused. So the pipeline records them without extra arguments. The robot runtime only checks the final
`v12_tracking_profile`; the boundary record is informational.

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
```
