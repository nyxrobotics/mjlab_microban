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

## Archived full training chain

The accepted release also versions the exact 3,000, 7,000, 10,000 and 15,000
update checkpoints, every required 100-update activation canary, and each
stage's locomotion, tracking/safety and ONNX evidence. It also archives the
exact deployable `pico_teleop.onnx` bytes admitted by the physical runtime,
rather than retaining only a receipt that names their digest. Verify all
tracked paths, nested report/ONNX SHA-256 bindings and the final
physical-runtime receipt with:

```bash
uv run --locked python scripts/verify_microban_teleop_v12_release_chain.py \
  --paths-only
```

The archive includes the raw update-9,201 `model_9200.pt`, its independently
reconstructed bilateral-order migration, the migration and recovery receipts,
and the authenticated update-9,901 corner-rescue parent. Full verification
rebuilds the migration from the raw checkpoint, compares the complete migrated
payload (actor, critic, optimizer and clocks), and reruns the corner-parent
tracking admission before validating the 10,000-update endpoint. Thus the
7,100-to-10,000 handoff is not accepted merely because the final checkpoint
contains a migration marker. The raw and migrated replay runs' exact agent,
environment and Git-diff files are hash-bound as well; in particular they
record the `model_7099.pt -> model_9200.pt` 2,900-update run and the migrated
`model_9200.pt -> model_9999.pt` 799-update replay. The verifier parses their
resume target, seed, iteration count and save policy, then checks the archived
7,100→9,201→9,901→10,000 checkpoints pairwise: simulation and optimizer clocks,
the exact active actor-column set, frozen actor tensors, zero foot columns and
Adam moments, optimizer inventory, and unchanged bilateral migration marker.

For the complete audit, omit `--paths-only`. The verifier creates disposable
detached worktrees and reruns each gate with the evaluator commit recorded in
`artifacts/teleop_v12_releases/microban_teleop_v12_full_chain_manifest.json`.
It then loads the archived deployable policy with ONNX Runtime's CPU provider
and runs 16 deterministic `[1,83] -> [1,18]` finite-output samples. The audit
requires every evidence file and all verifier scripts to be regular files
whose index and working-tree bytes exactly match committed `HEAD`; it refuses
to mix dirty working files with evidence checked out from another revision.
Only hash-bound files below the teleop artifact and run-data directories may be
overlaid into a pinned evaluator worktree, so release evidence cannot replace
the evaluator's source code.

Some historical reports contain the workstation's then-current absolute
checkout path. During the disposable replay, a fail-closed relocation wrapper
maps only those recorded roots' `artifacts/teleop_v12_*` and
`logs/rsl_rl/mjlab_microban_teleop_v12` files to the same repository-relative
files in the detached worktree. The two exact legacy bootstrap inputs already
stored by the pinned evaluator are readable but are never overlaid from the
release commit. The wrapper does not rewrite the archived JSON or its hash.
Relative paths, path traversal, symlinks, unrecorded absolute roots, and
non-data paths are rejected.

This pin is intentional: the 3,000/7,000 stages predate the authenticated
bilateral-site-order migration and are checked with their historical evaluator;
the migrated 10,000/15,000 stages use the later evaluator. A current evaluator
rejecting a pre-migration checkpoint is therefore not misreported as a failed
historical gate, and no pre-migration checkpoint is deployable.

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
   is rejected. A checkpoint of the active-hand arm pose-release recipe is
   judged at 15000 under the final completion allowance (see below); the
   deployed-accuracy and strict final profiles are also accepted for it.
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

## Final-gate completion allowance (pose-release lineage)

Profile `full_body_reachable_performance_perturbation_v2_completion_allowance_v1`
is the 15000-update profile of the active-hand arm pose-release recipe only.
On this forward-lean branch that recipe is
`forward_lean_home_velocity_source_staged_mask_reachable_fk_elbow_minus10_raw_prev_action_servo_range_pi_home_levelled_targets_level_hmd_receiver_box_hands_active_hand_arm_pose_release_v18`
(the centered line's string is `..._active_hand_arm_pose_release_v12`). It keeps the final profile's
scenarios, perturbation, target-column ablation targets and every
non-accuracy check (no falls, finite, actual soft limits, raw-action
recurrence, forced HMD motion, observation coverage, ablation response, twist
direction). Only the accuracy limits change:

| limit | deployed-accuracy final | completion allowance |
| --- | --- | --- |
| hand RMS | 0.035 m | 0.045 m |
| hand P95 | 0.07 m | 0.08 m |
| foot RMS | 0.05 m | 0.055 m |
| foot P95 | 0.08 m | 0.11 m |

Reason (2026-10-05): the pose-release `model_14999` of run
`2026-10-05_03-31-01_c20k_v12_pr_10100_to15000` passed every non-accuracy final
check and its 9999/10099 gates, and failed the deployed-accuracy final profile
only on accuracy: `mixed_forward_left` hand 0.0403/0.0727 m, foot
0.0517/0.1001 m (RMS/P95); `mixed_backward_right` hand 0.0415/0.0693 m, foot
0.0469/0.0806 m; `max_keypoints_left` foot RMS 0.0521 m. The two perturbed
mixed scenarios are near-fall states for the previous canonical model as well.
The user approved completing the centered-HOME PICO this way ("全部許可するから
一番良いと思う方法で作業完了まで進めて", "本来の基準ってのも別にそんなに意味ない").
The forward-lean pose-release recipe inherits the same allowance (same
reward change, same final scenarios); its release route is a fresh
pose-release chain (`scripts/train_microban_teleop_v12.sh start --source
LEAN_WALK.pt --hand-pose-release`), because no lean model_7099 is pinned as a
recipe-switch parent. Every other boundary and every other lineage keeps its
profile (except the 10000 hand-RMS allowance below). A gate judged
under the allowance records the limits and this reason in
`tracking_profile_completion_allowance`.

## 10000-boundary hand-RMS allowance (pose-release lineage)

Profile
`hmd_hand_reachable_performance_foot_exposure_v2_deployed_accuracy_v1_hand_rms_40mm_v1`
is the 10000-update profile (model_9999 of a fresh pose-release segment or of a
pose-release model_9900 corner rescue, both of which keep the pose-release
recipe revision) of the active-hand arm pose-release recipe only. It is the
HMD/hand deployed-accuracy profile with exactly one change:

| limit | deployed-accuracy HMD/hand | hand-RMS allowance |
| --- | --- | --- |
| hand RMS | 0.035 m | 0.040 m |
| hand P95 | 0.05 m | 0.05 m |
| foot RMS | 0.05 m | 0.05 m |
| foot P95 | 0.08 m | 0.08 m |

Scenarios, checks (falls, finiteness, soft limits, raw-action recurrence,
forced HMD motion, coverage, hand ablation, twist), locomotion and ONNX gates
are unchanged; the deployed-accuracy and strict HMD/hand profiles are also
accepted at that boundary. Interrupted 7101..9999 clocks, the 10100 canary,
the 15000 final and every other lineage keep their profiles.

Reason (2026-10-05): the forward-lean pose-release model_9999 checkpoints (two
fresh 7100->10000 segments and eleven model_9900 corner rescues) passed every
non-accuracy check and missed only the 0.035 m hand RMS by a few millimetres in
one bilateral corner (best max(L, R) about 0.0365 m). User decision: "手は
0.04mまで許容でいいんじゃない？". The gate records the limits and reason in
`tracking_profile_completion_allowance` (revision `hand_rms_40mm_v1`,
`boundary_completed_updates` 10000).

The packager does not see earlier gates through the final checkpoint, so it
takes them explicitly: `--boundary-gate GATE.json` (repeatable) fully
revalidates each earlier canonical-boundary gate, requires its checkpoint to
share the final checkpoint's contract, recipe, bootstrap, HOME and site-order
markers (and the final to carry its corner-rescue / recipe-switch /
LR-migration markers unchanged), and records `v12_boundary_stage_gates_json`
(clock, checkpoint and gate SHA-256, tracking profile, allowance record) with
`v12_boundary_stage_gates_semantics` in the package. The robot runtime only
checks the final `v12_tracking_profile`; the boundary record is informational.

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
