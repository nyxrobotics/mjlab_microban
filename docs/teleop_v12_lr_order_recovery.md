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

## Model 9200 decision record

The deterministic CPU gate (`seed=42`, 300 steps, 50 settling steps) produced:

| candidate/environment | `max_hands_left` RMS / P95 | `max_hands_right` RMS / P95 | hand causal response | safety |
| --- | ---: | ---: | --- | --- |
| raw model 9200, old crossed order | 4.12 / 4.88 cm | 5.11 / 6.13 cm | pass | pass |
| exact swap, corrected order | 4.50 / 5.49 cm | 3.38 / 4.13 cm | pass | pass |
| zero hand columns, corrected order | 7.10 / 7.80 cm | 7.42 / 8.46 cm | fail | pass |

Here “safety pass” means all scenarios completed with no fall, no non-finite
state, and no actual soft-limit violation.  The exact swap is the selected
recovery: it preserves strong command causality and is substantially better
than discarding the learned hand adapter.  It is not an acceptance pass by
itself because the strict per-scenario limits remain 3 cm RMS / 5 cm P95.  The
checkpoint must therefore continue from update 9201 through canonical update
10000 under the corrected order before the formal gate.

The measured source checkpoint was SHA-256
`16c9b9d19df6513851b3da26228ae612512fdb2d894542741f0474e4762691c7`.
The specific swap artifact used for this comparison was SHA-256
`bcdb1cddf8d7daff012f884e41e3a89e81d54ee3d564526990346533799a26e2`;
its receipt recorded `iter=9200` and `common_step_counter=220824` unchanged.
