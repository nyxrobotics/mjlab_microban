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
   updates, canonical-boundary kind, `status=pass`, and the canonical final
   perturbation profile. A checkpoint of the active-hand arm pose-release recipe is
   judged at 15000 under the final completion allowance (see below); the
   deployed-accuracy and strict final profiles are also accepted for it.
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

## Final-gate completion allowance (pose-release lineage)

Profile `full_body_reachable_performance_perturbation_v2_completion_allowance_v1`
is the 15000-update profile of the active-hand arm pose-release recipe
only: `..._active_hand_arm_pose_release_v12` at the centered HOME,
`..._home_levelled_targets_level_hmd_receiver_box_hands_active_hand_arm_pose_release_v18`
at the forward-lean HOME (`src/mjlab_microban/robot/home_contracts.py`). It keeps the final profile's
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
At every HOME but the centered one the release route is a fresh pose-release
chain (`scripts/train_microban_teleop_v12.sh start --source WALK.pt
--hand-pose-release`): only the centered HOME pins a model_7099 as a
recipe-switch parent. A HOME whose trunk leans forward labels its foot/hand
target columns `robot_home_levelled_trunk_xyz_forward_left_up` and is packaged
by packager v7 (forward-lean) or a `<tag>` revision.
Every other boundary and every other lineage keeps its profile (except the
10000 hand-RMS allowance below). A gate judged
under the allowance records the limits and this reason in
`tracking_profile_completion_allowance`.

## 10000-boundary hand-RMS allowance (pose-release lineage)

This allowance, its 10100 counterpart and the packager's requirement that a
pose-release final records both gates exist at every HOME except the centered
one (`home_contracts.V12_HAND_RMS_40MM_BOUNDARY_PROFILES`, false in the centered
HOME's `LEGACY_HOME_OVERRIDES` entry). The centered line judged its 10000 / 10100
clocks at 0.035 m (track-centered-home-clip 5b5a9d0), so there the profiles are
unknown to every table and boundary gates are recorded only when passed
explicitly.

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
accepted at that boundary. Interrupted 7101..9999 clocks, the 15000 final and
every other lineage keep their profiles; the 10100 canary has its own
counterpart (below).

Reason (2026-10-05): the forward-lean pose-release model_9999 checkpoints (two
fresh 7100->10000 segments and eleven model_9900 corner rescues) passed every
non-accuracy check and missed only the 0.035 m hand RMS by a few millimetres in
one bilateral corner (best max(L, R) about 0.0365 m). User decision: "手は
0.04mまで許容でいいんじゃない？". The gate records the limits and reason in
`tracking_profile_completion_allowance` (revision `hand_rms_40mm_v1`,
`boundary_completed_updates` 10000).

### 10100-canary counterpart

Profile
`whole_body_foot_activation_canary_reachable_safety_v1_deployed_accuracy_v1_hand_rms_40mm_v1`
is the 10100-update (model_10099 activation canary) profile of the same
pose-release recipe only: the foot-activation canary deployed-accuracy profile
with hand RMS 0.040 m instead of 0.035 m and nothing else changed (hand P95
0.05 m, foot 0.05 / 0.08 m, scenarios, checks, ablation, locomotion, ONNX);
the deployed-accuracy and strict canary profiles stay accepted there and the
interrupted 10001..10099 clocks keep the deployed-accuracy canary profile.
Reason (2026-10-05, same user decision): two forward-lean 10099 canaries
resumed from the 0.040 m-gated model_9999 (itself 0.0365 m) passed every
non-accuracy check, locomotion and hand P95 and missed only the 0.035 m hand
RMS (max 0.0367 / 0.0372 m); a 100-update canary cannot be held tighter than
the boundary it continues. Its gate records `boundary_completed_updates`
10100.

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
corner-rescue / recipe-switch markers unchanged. The packager
records `v12_boundary_stage_gates_json` (clock, checkpoint kind, checkpoint
and gate SHA-256, tracking profile, allowance record) with
`v12_boundary_stage_gates_semantics`
(`..._resume_ancestor_boundary_gates_..._v2`) in the package.

A pose-release final must record both its 10000 boundary and its 10100
canary, because both clocks have a hand-RMS allowance profile on that recipe.
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
