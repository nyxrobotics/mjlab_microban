# Contract-v12 model9900 corner-pair rescue

This is a one-time, fail-closed replay for the corrected-order contract-v12
checkpoint that reached update 9901 but narrowly missed the update-10000 hand
RMS gate. It does not change learning rate, PPO, action semantics, rewards,
normalization, the frozen legacy actor, or any gate threshold. Foot commands,
foot adapter columns, and their Adam moments remain inactive and exact zero.

## Fixed route

- parent: `model_9900.pt`, completed update 9901, SHA-256
  `063a8f65ebf9007d63395e9a5b98420eb025bd39416dab5727f9f4c06fc6e877`;
- parent strict tracking report SHA-256
  `399db0cee55c137d3d0226ebcb54b0fa3bb84c95496c5c206af55f2fb25837f4`;
- parent optimizer step: 198020 for every initialized Adam state;
- replay: exactly 99 PPO updates, seed 42, 2048 environments, 24 steps;
- endpoint: `model_9999.pt`, completed update 10000, optimizer step 200000.

The sampler uses the existing uniform/independent hand sampler for 40% of
resamples, the currently failing left-forward/right-backward corner for 40%,
and the already-passing left-backward/right-forward corner for 20%. Both hands
are active in the two corner branches. The joint tuples and the 40/40/20 mix
are embedded in every rescue checkpoint marker.

## Launch

Run from a clean commit. The launcher copies the immutable parent into a
dedicated seed run under the canonical experiment directory so the standard
stage evaluator and resume launcher can find the result:

```bash
scripts/train_microban_teleop_v12_corner_rescue.sh \
  /absolute/path/to/model_9900.pt \
  /absolute/path/to/2026-09-25_23-16-50_v12_lrfix_model_9900_tracking.json \
  --agent.run-name v12_corner_rescue_9901_to10000
```

Before simulator startup it deep-validates both fixed inputs, including the
report's exact identity and its sole `hand_tracking_rms` failure. The runner
then reasserts the source and target clocks, every Adam step, frozen foot
normalizer values, exact-zero foot W0/Adam columns, and live zero foot command
tensor/active masks.

## Acceptance

`model_9999.pt` is not promoted merely because training completed, nor because
one tracking run passed. It must pass the ordinary schema-v2 update-10000 gate:
all nine 300-step locomotion scenarios, strict HMD/hand tracking, ONNX parity,
and their hash-bound report validation. Only that full gate may authorize the
normal 10000-to-10100 foot activation canary. Thresholds must not be relaxed;
retain a failing result for diagnosis rather than extending this recipe.

The dedicated wrapper runs that complete gate and creates a second receipt
which binds the fixed rescue lineage, exact optimizer/foot state, tracking
report, schema-v2 gate, all three report hashes, and ONNX artifact:

```bash
scripts/evaluate_microban_teleop_v12_corner_rescue.sh \
  v12_corner_rescue_9901_to10000
```

It exits before ONNX/gate/receipt promotion if strict locomotion or tracking
fails. On success, the gate is written to the canonical
`artifacts/teleop_v12_gates` location expected by the ordinary resume driver:

```bash
scripts/train_microban_teleop_v12.sh resume \
  v12_corner_rescue_9901_to10000 \
  --agent.run-name v12_canonical_10000_to10100
```

CPU-only reproduction checks:

```bash
uv run --locked --with pytest python -m pytest -q \
  tests/test_teleop_v12_corner_rescue.py tests/test_teleop_v12_stage.py
bash -n scripts/train_microban_teleop_v12_corner_rescue.sh
```
