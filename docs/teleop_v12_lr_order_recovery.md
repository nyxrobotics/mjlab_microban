# Contract-v12 bilateral site-order recovery

## Why migration is required

Microban's XML lists the right hand/foot sites before the left sites.  The old
`HandTargetCommand` and `FootTargetCommand` resolved a requested `(left,
right)` pair without `preserve_order=True`, so measured positions arrived as
`(right, left)` while the command tensors remained `(left, right)`.  In the
hand-acquisition stage this trained the actor against crossed hand targets.

The corrected command terms preserve the requested order and fail during
environment construction unless the resolved names are exactly the configured
names.  The v12 runner also authenticates the fully resolved actor and critic
term slices.  A raw pre-fix checkpoint cannot be resumed or consumed directly.

## Exact checkpoint transform

For a pre-foot-stage checkpoint, the migration composes the actor and critic
with the self-inverse hand-input permutation:

- actor hand XYZ: `75:78 <-> 78:81`
- actor hand-active flags: `81 <-> 82`
- critic hand XYZ: `90:93 <-> 93:96`
- critic hand-active flags: `96 <-> 97`

It applies the same permutation to first-layer weights, observation-normalizer
mean/variance/std, and the matching Adam `exp_avg`/`exp_avg_sq` tensors.  It
does not reset `iter`, `env_state.common_step_counter`, or any optimizer step.
All untouched tensor bytes are hash-inventoried in the receipt.

Foot adapter columns are intentionally not permuted by this recovery.  The tool
accepts only a checkpoint whose foot columns are absent from the active schedule
and whose actor W0 and Adam foot columns are exactly zero.  They therefore have
no learned L/R meaning yet and start correctly under the fixed environment.

## Reproduce the migration

Use CPU only, pin the source checkpoint bytes, and choose distinct output and
receipt paths:

```bash
cd /home/kanade/Git-projects/mjlab_microban_v12_deferred
SOURCE=/absolute/path/to/model_9200.pt
SOURCE_SHA=$(sha256sum -- "$SOURCE" | awk '{print $1}')
CUDA_VISIBLE_DEVICES='' uv run --locked python -m \
  mjlab_microban.scripts.migrate_teleop_v12_lr_order \
  "$SOURCE" \
  --expected-sha256 "$SOURCE_SHA" \
  --output /tmp/model_9200_lr_swap.pt \
  --receipt /tmp/model_9200_lr_swap_receipt.json \
  --strategy swap
```

`--strategy zero_hand` is an explicit comparison/control: it performs the
critic semantic swap but zeros actor hand W0 and its Adam moments.  It is not
the preferred continuation checkpoint unless measured corrected-order tracking
shows the exact swap is worse.

The JSON receipt is bound to both source and destination SHA-256 values and
contains the exact 83/137-column permutations, source clock, optimizer IDs,
pre-foot zero proof, and hashes for every untouched tensor or tensor slice.
Future runner saves preserve the migration marker so final stage gates and the
exporter can authenticate the original repair lineage.

## CPU comparison

Run the same deterministic corrected-order tracking gate for each candidate:

```bash
CUDA_VISIBLE_DEVICES='' uv run --locked python -m \
  mjlab_microban.scripts.evaluate_teleop_v12_tracking \
  /tmp/model_9200_lr_swap.pt --device cpu \
  --output /tmp/model_9200_lr_swap_tracking.json --force

CUDA_VISIBLE_DEVICES='' uv run --locked python -m \
  mjlab_microban.scripts.evaluate_teleop_v12_tracking \
  /tmp/model_9200_lr_zero_hand.pt --device cpu \
  --output /tmp/model_9200_lr_zero_hand_tracking.json --force
```

These are simulation measurements, not physical deployment authorization.
