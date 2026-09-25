# PICO full-body check with exact v12 model10099

This route exists only to inspect the accepted deadline foot-activation canary
with a real PICO 4 Ultra and a simulated Microban. It deliberately uses the
canonical `Mjlab-Teleop-V12-Microban` task rather than the preview task or the
older generic `--checkpoint` route.

## What is accepted

- checkpoint SHA-256:
  `86a81f45f34d91036ab138963c835a3e78db98224f523d3484e1d3ed335082ff`;
- checkpoint iteration: `model_10099.pt` (10,100 completed updates);
- post-canary PASS receipt SHA-256:
  `f7fcf511ba75f326cd94a2fc21a05360fa9f7925849fc2bce91709fd8678f27b`;
- actor-only inference from immutable checkpoint bytes.

The receipt keeps the standard locomotion, fall, finite-value, dynamic
soft-limit, directional-response, target-ablation, foot-activation, and ONNX
checks. The deadline exception remains limited to hand RMS 35 mm; hand P95 is
still 50 mm and no safety threshold is relaxed.

The consumer has no training, checkpoint-save, ONNX-export, robot-address,
UDP-motor, or physical-robot output. A failed receipt, changed file, wrong task,
wrong checkpoint, malformed lineage, or strict actor load stops this route
before live inference. Runtime body/input faults can use only the audited
legacy actor inside the local simulation.

## Run later (do not run during headless training)

From this repository:

```bash
receipt=artifacts/teleop_v12_deadline_fallback/2026-09-26_00-18-44_v12_deadline_foot_canary_10000_to10100_20260926_model_10099_post_canary_receipt.json
receipt_sha256="$(sha256sum -- "${receipt}" | cut -d ' ' -f 1)"

scripts/run_pico_v12_deadline_canary_sim.sh \
  logs/rsl_rl/mjlab_microban_teleop_v12/2026-09-26_00-18-44_v12_deadline_foot_canary_10000_to10100_20260926/model_10099.pt \
  "${receipt}" \
  "${receipt_sha256}" \
  --input pico-app
```

The wrapper reuses the isolated simulation pairing on TCP 63903 and camera port
8081. It never selects the generic `--checkpoint` option.

If an earlier preview has been stopped, relaunch it with the same command above;
the control change requires no new launcher flag. No trigger state survives a
stopped process or reconnect.

Before the trial, pair and calibrate all three PICO Motion Trackers (waist, left
ankle, right ankle) in Body Tracking mode. Confirm the PICO client reports
`BODY READY`, `trackers 3/3`, and `joints 24/24`. Stand still with both triggers
released for at least 0.5 seconds so the body and controller origins are
coherent, then check the controls in this order:

1. Hold the right trigger with the left trigger released. The six arm joints
   track the controllers while the locomotion command remains exact zero.
2. Keep the right trigger held and hold the left trigger. The locomotion/full-
   body policy and arm tracking run together.
3. Release only the right trigger. Both arms return to the exact PICO HOME joint
   pose immediately; left-trigger locomotion may continue.
4. Release the left trigger to neutralize locomotion/full-body policy control.
   Both triggers are momentary hold controls, never toggles. Hold the right grip
   separately when neck yaw should face the simulated body's front.

During a momentary invalid controller sample lasting at most 500 ms, locomotion
is neutralized and the arms hold their last validated joint target. A valid
sample resumes held right-trigger tracking without a release/re-press cycle.
Transport disconnect, authority loss, or expiry beyond 500 ms clears the held
target and sends both arms to exact PICO HOME. After reconnect, establish a
fresh released neutral frame before arming again. The fuller preview procedure
and failure semantics are in [the v12 live preview guide](pico_v12_live_preview.md).

## Re-run the CPU-only guard tests

```bash
uv run --locked --with pytest python -m pytest -q \
  tests/test_live_pico_teleop_sim.py \
  tests/test_teleop_v12_deadline_canary_live.py
bash -n scripts/run_pico_v12_deadline_canary_sim.sh
```
