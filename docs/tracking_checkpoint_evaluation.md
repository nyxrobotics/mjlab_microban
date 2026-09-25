# Microban tracking checkpoint evaluation

`evaluate_tracking_checkpoint` is the headless, simulation-only acceptance gate
for a learned walk004 motion-tracking checkpoint. It evaluates the policy itself;
it is separate from the open-loop locomotion-prior replay gate.

Passing this gate is not permission to run the physical robot. The offline
evaluator and exported tracking ONNX still embed the fixed walk004 reference.
For simulation, `live_pico_tracking_sim.py` and `live_pico_tracking_bridge.py`
already validate the real PICO body stream and patch its 42 live target values
into the 99-value observation while preserving the safety latch documented in
`live_pico_tracking99_sim.md`. What remains unavailable is an accepted learned
tracking checkpoint: the evaluated walk004-derived candidates failed their
dynamic gates. Until a learned policy passes those gates, the live adapter is a
simulation integration path only and must not command the physical robot.

## Run the full receipt

From the `mjlab_microban` repository root:

```bash
uv run --locked python -m mjlab_microban.scripts.evaluate_tracking_checkpoint \
  --checkpoint logs/rsl_rl/mjlab_microban_tracking/2026-09-25_07-17-25_v8j_walk004_tracking_4096_251diag/model_250.pt \
  --output artifacts/microban_tracking_checkpoint_walk004_model_250.json
```

The canonical command requires CUDA and always runs both modes with 256
environments and seed 42. Source frame 0 initializes state, and the policy then
tracks target frames 1 through 267: 267 control transitions cover the complete
268-frame clip without replaying frame 0 at the end.

- `nominal`: fixed initial state, actor observation corruption disabled, and no
  MjLab startup event randomization. The configured BAM voltage, voltage-drop,
  current and 3--6 physics-step command-delay envelope remains active.
- `robust`: adds actor observation corruption plus the training task's
  `base_com`, `encoder_bias`, and `foot_friction` startup randomization. Interval
  pushes are intentionally excluded from this bounded full-clip capability test.

The default motion and robot XML are SHA-256 pinned. To evaluate regenerated
inputs, pass both the path and the independently established expected digest:

```bash
uv run --locked python -m mjlab_microban.scripts.evaluate_tracking_checkpoint \
  --checkpoint /path/to/model.pt \
  --motion /path/to/walk004.npz \
  --expected-motion-sha256 <64-hex-digest> \
  --robot-xml /path/to/robot.xml \
  --expected-robot-xml-sha256 <64-hex-digest> \
  --output artifacts/tracking-checkpoint-receipt.json
```

Exit status is `0` only when every check in both modes passes. A complete receipt
with one or more failed checks is still written and exits `2`. Invalid inputs,
contract drift, or simulator errors raise an error and do not publish a partial
receipt. Output publication is atomic, and `receipt_payload_sha256` binds the
complete JSON payload.

The evaluator identifies both the actor distribution and actor observation
normalizer from checkpoint state. This is required for honest historical
evaluation: `model_250.pt` contains the earlier normalized, unbounded
scalar-Gaussian actor, intermediate checkpoints contain a normalized bounded
actor, and current checkpoints use a raw-input bounded actor. Old weights are
strict-loaded with their recorded topology instead of being silently
reinterpreted by the current training config. The selected distribution,
normalizer mode, and algorithm are recorded under each pass's `configuration`.

The receipt also hashes the gate arithmetic, environment/MDP/export sources,
robot configuration, XC330 identification parameters, and `uv.lock`. Together
with the checkpoint, motion, robot XML, and evaluator hashes this makes a local
code or dynamics change visible in every new receipt. The tracking simulator
reserves 512 contacts and 2,048 constraints; a fallen-robot rollout must not be
judged using the earlier overflowing 128/512 capacity.

## Lifecycle and measurements

The evaluator sets `auto_reset=False`. A terminal state is measured before any
reset. To let the remaining vector rows continue, an exited row is then manually
reset but permanently excluded from subsequent metrics; it can never re-enter or
be counted as a full-clip completion. The receipt records a per-step active count,
named termination counts, unattributed terminations, and first-exit statistics.

The pass/fail bounds reuse existing Microban gates rather than relaxing them:

| Check | Required value |
| --- | ---: |
| Completed source sequence | exactly 267 transitions (frame 0 state, targets 1--267), all 256 environments |
| Falls, unexpected terminations, non-finite rows | 0 |
| Self-collision contacts | 0 |
| Minimum root height | at least 0.10 m |
| Actual soft-limit violation | at most 0.000001 rad |
| Policy-target projection to the soft limits | at most 0.001 rad |
| Projected target fraction | at most 0.001 |
| Per-environment p05 forward speed | at least 0.05 m/s |
| Per-environment p95 XY velocity MAE | at most 0.075 m/s |
| Root travel direction | p05 forward displacement strictly positive |
| Foot gait evidence | each foot airborne in all environments; no simultaneous flight |

The JSON also reports diagnostics that currently have no invented acceptance
threshold: anchor pose error, end-effector error, MPKPE, root-relative MPKPE,
joint position/velocity error, per-joint reference and actuator-target errors,
raw policy action magnitude, contact/air-time counts, in-contact foot slip, root
displacement, and the full source-frame trace. These should be used to decide how
the next training run changes; they must not be interpreted as a pass when a
conservative check fails.

## Current `model_250` diagnostic

The command above produced
`artifacts/microban_tracking_checkpoint_walk004_model_250.json` and correctly
returned exit status `2` (`fail`). Nominal completed 256/256 environments and
robust completed 255/256; the robust loss was one `ee_body_pos` termination at
source frame 46. Key failures were:

| Measurement | Nominal | Robust | Required |
| --- | ---: | ---: | ---: |
| Forward velocity p05 | 0.00863 m/s | 0.00840 m/s | at least 0.05 m/s |
| XY velocity MAE p95 | 0.08197 m/s | 0.08235 m/s | at most 0.075 m/s |
| Maximum target projection | 3.030 rad | 3.207 rad | at most 0.001 rad |
| Target clip fraction | 0.2405 | 0.2436 | at most 0.001 |
| Actual soft-limit violation | 0 rad | 0.00636 rad | at most 0.000001 rad |

This confirms that model 250 is useful as a diagnostic baseline but is not a safe
deployment candidate. In particular, its unbounded output relies heavily on the
simulator's target clamp and it does not reproduce the reference's forward speed.

## Developer checks

The receipt/check arithmetic is in the pure, simulator-free
`mjlab_microban.tracking_checkpoint_gate` module. Run its regression tests and
the evaluator's static checks with:

```bash
uv run --locked --with pytest python -m pytest -q tests/test_tracking_checkpoint_gate.py
uv run --locked --with ruff ruff check \
  src/mjlab_microban/tracking_checkpoint_gate.py \
  src/mjlab_microban/scripts/evaluate_tracking_checkpoint.py \
  tests/test_tracking_checkpoint_gate.py
```

The seed and exact local input hashes make runs auditable, but GPU driver and
host-library changes can still affect bit-level results. Keep the receipt with
the checkpoint when comparing runs.
