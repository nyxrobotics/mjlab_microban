# Contract-v12 bilateral-site recovery

This is the one-time recovery procedure for the contract-v12 hand stage that
was trained while bilateral hand and foot sites were resolved in XML order
(`right, left`) instead of command order (`left, right`). It preserves the
canonical training clock: the pinned raw `model_9200.pt` is migrated offline,
then exactly 799 corrected updates produce `model_9999.pt` at completed update
10000. Feet remain inactive and their actor/Adam columns remain exact zero for
the complete replay.

This path does not weaken or replace the normal 10000 gate. The recovered
`model_9999.pt` must pass the ordinary strict locomotion, hand tracking, safety,
and ONNX checks before the normal 10000->10100 foot canary can begin.

## Fixed identities

- Raw source iteration: `9200` (completed updates `9201`)
- Raw source common step counter: `220824` (`9201 * 24`)
- Raw source SHA-256:
  `16c9b9d19df6513851b3da26228ae612512fdb2d894542741f0474e4762691c7`
- Replay length: `799` updates
- Recovered endpoint: iteration `9999` (completed updates `10000`)
- Site-order revision: `preserve_requested_left_right_sites_v1`
- Migration strategy: `swap`; `zero_hand` is diagnostic-only and is rejected
  by this recovery route

The migration swaps the hand target and active-flag input columns in the actor,
the matching hand columns and normalizer statistics in the critic, and the
corresponding Adam first/second moments. All other tensors and all untouched
columns must remain bit-identical. The validator independently repeats that
transformation instead of trusting the JSON receipt.

## 1. Commit and use the corrected source tree

Run the procedure only from the committed recovery branch. A dirty source tree
would make the training provenance non-reproducible. Do not run this migration
or replay from the old checkout that generated the raw checkpoint.

The raw checkpoint may remain in its original log directory. Set paths
explicitly so the receipt binds the exact file used:

```bash
cd /home/kanade/Git-projects/mjlab_microban_v12_hand_rescue

RAW=/home/kanade/Git-projects/mjlab_microban_v8j/logs/rsl_rl/mjlab_microban_teleop_v12/2026-09-25_22-18-45_v12_canonical_7100_to10000/model_9200.pt
SEED_RUN=v12_lr_recovery_seed_model9200
SEED_DIR="$PWD/logs/rsl_rl/mjlab_microban_teleop_v12/$SEED_RUN"
MIGRATION_RECEIPT="$PWD/artifacts/teleop_v12_lr_recovery/model9200_migration.json"
RECOVERY_RECEIPT="$PWD/artifacts/teleop_v12_lr_recovery/model9200_recovery_seed.json"

mkdir -p "$SEED_DIR" "$(dirname "$MIGRATION_RECEIPT")"
sha256sum "$RAW"
```

The printed hash must equal the fixed raw source hash above.

## 2. Create the migrated seed and both receipts

Migration and receipt validation are CPU-only and do not open a simulator:

```bash
uv run --locked python -m \
  mjlab_microban.scripts.migrate_teleop_v12_lr_order \
  "$RAW" \
  --expected-sha256 16c9b9d19df6513851b3da26228ae612512fdb2d894542741f0474e4762691c7 \
  --output "$SEED_DIR/model_9200.pt" \
  --receipt "$MIGRATION_RECEIPT" \
  --strategy swap

uv run --locked python -m \
  mjlab_microban.scripts.teleop_v12_lr_recovery create \
  --source "$RAW" \
  --checkpoint "$SEED_DIR/model_9200.pt" \
  --migration-receipt "$MIGRATION_RECEIPT" \
  --output "$RECOVERY_RECEIPT"
```

Neither command overwrites existing output by default. Remove an invalid seed
deliberately, or pass `--force` only after checking the exact destination.

The second receipt proves all of the following before any GPU work starts:

- raw source hash and exact 9201-update clock;
- migrated checkpoint hash and unchanged clock;
- corrected site-order and migration revisions;
- complete actor, critic, normalizer, and optimizer permutation;
- bit identity of every out-of-scope tensor and column;
- exact-zero inactive foot actor columns and Adam moments;
- the fixed 799-update canonical route.

It can be revalidated independently at any time:

```bash
uv run --locked python -m \
  mjlab_microban.scripts.teleop_v12_lr_recovery validate \
  "$RECOVERY_RECEIPT" \
  --source "$RAW" \
  --checkpoint "$SEED_DIR/model_9200.pt" \
  --migration-receipt "$MIGRATION_RECEIPT"
```

## 3. Replay exactly 799 corrected updates

The dedicated launcher accepts no iteration, environment-count, seed, or
curriculum overrides:

```bash
scripts/train_microban_teleop_v12_lr_recovery.sh \
  "$SEED_RUN" "$RAW" "$MIGRATION_RECEIPT" "$RECOVERY_RECEIPT" \
  --agent.run-name v12_lr_corrected_9201_to10000
```

It validates both receipts and the checkpoint again, then launches the normal
`Mjlab-Teleop-V12-Microban` task with the canonical fixed arguments:

- 2048 environments;
- environment and agent seed 42;
- 24 policy steps per update;
- 799 updates;
- checkpoint interval 100;
- TensorBoard logger, upload disabled, NaN guard enabled.

The normal runner independently validates and retains the migration lineage
when loading and saving the recovered checkpoint. The ordinary curriculum is
unchanged: foot commands, reward, and adapter columns do not activate before
the completed-update-10000 boundary.

## 4. Run the unchanged strict gate

Find the single new run directory ending in the requested run name, then run
the standard stage evaluator at iteration 9999:

```bash
scripts/evaluate_microban_teleop_v12_stage.sh \
  RUN_DIRECTORY_NAME 9999
```

Do not start the foot canary unless this gate reports `pass`. If it passes, the
existing canonical driver resumes without special recovery options:

```bash
scripts/train_microban_teleop_v12.sh resume RUN_DIRECTORY_NAME \
  --agent.run-name v12_canonical_10000_to10100
```

If the strict hand gate fails, retain the checkpoint and reports for diagnosis.
Do not extend this launcher, lower the gate thresholds, activate feet early, or
export the failed checkpoint.

## Reproduction checks

Before launching the replay:

```bash
uv run --locked pytest -q tests/test_teleop_v12_lr_order.py \
  tests/test_teleop_v12_lr_recovery.py
bash -n scripts/train_microban_teleop_v12_lr_recovery.sh
git status --short
```

The final command should be empty apart from ignored runtime artifacts.
