# PICO 4 Ultra → contract-v12 full-body simulation preview

This is the shortest path for trying the real PICO headset, controllers, waist
tracker, and two ankle trackers against a simulated Microban before the complete
15,000-update policy is accepted. It never starts a robot gateway or motor
interface. The isolated PICO control port is `63903`; the simulated stereo-camera
port is `8081`.

The early checkpoint is intentionally a different artifact class from the final
policy. It must retain both `preview_non_deployable=true` and the exact
`teleop_v12_preview` lineage marker. Only the dedicated preview runner accepts
it. The normal v12 runner, stage gate, ONNX exporter, and physical Microban
runtime reject the marker. This preview does not shorten or replace the full
15,000-update training and acceptance path.

## Controller-only arm preview (recommended before Motion Tracker setup)

This dedicated path uses the audited legacy walking actor for balance and all
leg joints, then replaces only the six arm action columns with the controller
mapper's bounded, joint-slew-limited IK solution. It is deliberately more direct
than asking the early learned adapter to approximate the hand target, so visible
controller motion is not attenuated by the phase-1 policy's tracking error.

The phase-1 checkpoint and its strict or visual PASS receipt are still
hash-bound and strict-loaded. The runner accepts only the HMD/hand phase marker,
cannot train/save/export, has no robot output, and remains simulation-only.
Defense in depth enforces all of the following:

- the live target source is the two PICO controllers, never body tracking;
- both foot targets are exact numeric zero on every policy step;
- each commanded arm target is inside the pinned Microban reachable joint box
  and its simulator soft target limit to `1e-7 rad`;
- the Cartesian hand command must exactly match independent Microban FK of the
  supplied joint target;
- right-trigger release, transport loss, a tracking hold older than 500 ms, or
  a contract fault commands exact PICO arm HOME;
- a shorter invalid-controller interval holds only the last validated arm joint
  target while locomotion is neutral, then resumes without a trigger cycle.

The software HOME origin is intentionally unchanged: left shoulder roll is
`+10 deg` and right shoulder roll is `-10 deg`. There is no hidden shoulder
inset. The 45-case dynamic simulation gate separately permits at most `5 deg`
of measured joint overshoot beyond a soft limit; the accepted run measured
`2.013383 deg`. The numeric allowance is shared with the canonical v12 measured
dynamic-state gate. It does not relax the `1e-7 rad` commanded-target limit,
and this simulation-only receipt by itself remains no evidence for physical
deployment.

Start it with the real PICO app and simulated Microban:

```bash
cd /home/kanade/Git-projects/mjlab_microban_v8j
checkpoint=/absolute/path/to/phase1/model_<iteration>.pt
receipt=/absolute/path/to/phase1_pass_receipt.json
overlay_receipt=/absolute/path/to/direct_ik_overlay_acceptance.json
scripts/run_pico_v12_controller_preview.sh \
  "${checkpoint}" \
  "${receipt}" \
  "$(sha256sum -- "${receipt}" | cut -d ' ' -f 1)" \
  "${overlay_receipt}" \
  "$(sha256sum -- "${overlay_receipt}" | cut -d ' ' -f 1)"
```

This mode fixes controller displacement scale at `0.18`; command-line body,
hand, and foot scale overrides are rejected. Motion Tracker packets may still
arrive, but their body and foot targets are ignored by construction.

Reproduce the independent 45-case gate (nine walking scenarios times five arm
profiles), bind its raw bytes into a simulation-only adjudication receipt, then
verify both the receipt and its parent report with:

```bash
cd /home/kanade/Git-projects/mjlab_microban_v8j
raw=artifacts/direct_ik_arm_overlay_model14999_recheck.json
overlay_receipt=artifacts/direct_ik_arm_overlay_model14999_5deg_acceptance_recheck.json

uv run --locked python -m \
  mjlab_microban.scripts.evaluate_direct_ik_arm_overlay \
  --device cuda:0 \
  --output "${raw}"
raw_sha256="$(sha256sum -- "${raw}" | cut -d ' ' -f 1)"

uv run --locked python scripts/adjudicate_direct_ik_overlay.py \
  "${raw}" "${raw_sha256}" "${overlay_receipt}"
overlay_sha256="$(sha256sum -- "${overlay_receipt}" | cut -d ' ' -f 1)"

uv run --locked python scripts/adjudicate_direct_ik_overlay.py \
  --verify-receipt "${overlay_receipt}" "${overlay_sha256}"
```

Use new output filenames for each reproduction; the adjudicator never
overwrites an existing receipt. The launcher repeats the final verification
and refuses to start if the receipt, its raw parent report, any of the 45 cases,
or either hash differs.

## Start

Use a staged-v2 full-body checkpoint and its final hash-bound PASS receipt. The
receipt may be the strict full-body acceptance class or the explicitly
simulation-only visual acceptance class; a phase-1 promotion receipt is not a
live authority.

```bash
cd /home/kanade/Git-projects/mjlab_microban_v8j
receipt=/absolute/path/to/artifacts/final_preview_acceptance.json
scripts/run_pico_v12_preview.sh \
  /absolute/path/to/preview/model_<iteration>.pt \
  "${receipt}" \
  "$(sha256sum -- "${receipt}" | cut -d ' ' -f 1)"
```

If the viewer or preview process is currently stopped, rerun this exact command.
The controls need no new launcher option, and a stopped process retains no
trigger or held-target state.

The launcher checks the pinned legacy walking checkpoint, provisions the PICO
app with the owner-only simulation pairing, installs only the `63903` and `8081`
ADB reverse routes, and opens the MuJoCo viewer. It does not use the physical
control port `63902`.

Before it creates the policy consumer, the launcher requires a regular,
non-symlink checkpoint and receipt, verifies the caller-supplied receipt hash,
and binds the receipt to the checkpoint SHA-256, iteration, staged-v2 full-body
marker, exact training clock, profile, scenarios, and safety checks. A missing,
FAIL, phase-1, legacy-v1, altered, or incomplete receipt is rejected; neither a
targeted precheck nor a phase-1 promotion receipt is sufficient. Once accepted,
an inference or body-tracking fault still latches audited legacy walking until
the left trigger is released; the next activation retries the preview actor.

## Tracker preflight

All three Motion Trackers must be powered, assigned as waist/left ankle/right
ankle, and calibrated in the PICO system UI. A device can be paired but not
connected. Check the current headset state without starting simulation:

```bash
adb devices -l
adb logcat -d -v brief | \
  rg 'mTrackerConnectCount|mBindTrackerCount|mAlgRuningBodyTracking' | tail
```

For a full-body preview, the recent log must no longer show
`mTrackerConnectCount 0`, and body tracking must be running. On 2026-09-25 the
observed headset state was `mBindTrackerCount 3` but `mTrackerConnectCount 0` and
`mAlgRuningBodyTracking 0`; that state can exercise only the legacy fallback,
not hand/foot tracking.

## Hold controls

1. Start with both triggers released and both controllers in a comfortable
   neutral pose. Stand still for at least 0.5 seconds. Fresh released frames
   establish the body and controller origins; reset, recenter, source change, or
   reconnect deliberately starts a new session origin.
2. Hold the left trigger for locomotion/full-body policy control. Release it to
   neutralize locomotion. The left stick commands forward/back/left/right, and
   the right-stick horizontal axis commands yaw.
3. Hold the right trigger for direct controller-arm tracking. It works with the
   left trigger either released (stationary, exact-zero locomotion) or held
   (walking). Release the right trigger and all six arm joints return immediately
   to the exact PICO HOME pose: shoulder pitch `0 deg`, shoulder roll
   left/right `+10/-10 deg`, and elbow `-20 deg`.
4. Hold the right grip separately to request neck-yaw-front; release it for the
   normal HMD-yaw behavior. Arm tracking remains governed only by the right
   trigger.
5. The normal view is PICO passthrough. Hold the left grip (middle-finger button)
   to show the simulator's stereo camera; release it to return to passthrough.

Every operator control is momentary; there are no toggles. If controller
tracking is invalid for at most 500 ms while the right trigger was last known
held, the arms hold the last validated target and locomotion becomes neutral.
Fresh tracking resumes automatically without releasing and pressing the right
trigger. A disconnect, authority loss, or interval beyond 500 ms clears that
hold and returns the arms to exact PICO HOME; after reconnect, establish a fresh
released neutral frame before arming again. Body-tracker/calibration or learned-
policy faults still follow the left-trigger locomotion fallback/rearm rules.

In controller-only mode the controller origin and fixed `0.18` scale drive only
the arms; body and foot tracking are ignored and feet remain exact zero.

The simulator camera uses the existing exact synthetic pinhole contract: two
640×480 eyes in left-first SBS, per-eye tangent bounds, and a 59.016 mm baseline.
Unity reprojects those rays through the live HMD projection instead of stretching
the image to the headset viewport. See the sibling `microban_teleop` document
`docs/pico_unity_simulation.md` for the complete optics and fail-closed display
contract.

## Reproduce the software checks

```bash
cd /home/kanade/Git-projects/mjlab_microban_v8j
uv run --locked --with pytest python -m pytest -q \
  tests/test_live_pico_teleop_sim.py \
  tests/test_simulation_camera.py \
  tests/test_teleop_v12_actor.py \
  tests/test_teleop_v12_bootstrap.py \
  tests/test_teleop_v12_preview.py \
  tests/test_teleop_v12_preview_precheck.py \
  tests/test_teleop_v12_preview_receipts.py
bash -n scripts/run_pico_v12_preview.sh
bash -n scripts/run_pico_v12_controller_preview.sh
```

These checks do not prove that physical trackers are connected or that a preview
policy tracks well; the on-device calibration and visible MuJoCo motion are the
first end-to-end acceptance step.
