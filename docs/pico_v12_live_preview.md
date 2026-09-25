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

## Start

Use a checkpoint produced by the v12 preview training script, then run:

```bash
cd /home/kanade/Git-projects/mjlab_microban_v8j
scripts/run_pico_v12_preview.sh /absolute/path/to/preview/model_<iteration>.pt
```

The launcher checks the pinned legacy walking checkpoint, provisions the PICO
app with the owner-only simulation pairing, installs only the `63903` and `8081`
ADB reverse routes, and opens the MuJoCo viewer. It does not use the physical
control port `63902`.

If the preview checkpoint is absent, altered, has the wrong recipe/provenance,
or fails to load, it is not partially accepted. The process reports the error
and retains the audited legacy joystick actor. An inference or body-tracking
fault likewise latches legacy walking until the left trigger is released; the
next activation retries the preview actor.

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

1. Hold left `X` to select the preview policy.
2. With `X` still held, leave the left trigger released and stand still until
   body-target calibration completes.
3. Hold the left trigger to enable the selected policy. Releasing it returns the
   body to the configured initial pose; it is not a toggle.
4. The left stick commands forward/back/left/right velocity, and the right-stick
   horizontal axis commands yaw.
5. HMD orientation controls the simulated camera neck. While the right trigger
   is held, neck yaw faces the simulated body's front.
6. The normal view is PICO passthrough. Hold the left grip (middle-finger button)
   to show the simulator's stereo camera; release it to return to passthrough.
7. Release left `X` to select the audited legacy walking actor. Preview faults
   also enter this path automatically for the current trigger hold.

Every mode selector uses hold semantics. After stale input, tracker loss,
calibration failure, or a learned-policy fault, release the left trigger before
trying to arm again.

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
  tests/test_teleop_v12_bootstrap.py
bash -n scripts/run_pico_v12_preview.sh
```

These checks do not prove that physical trackers are connected or that a preview
policy tracks well; the on-device calibration and visible MuJoCo motion are the
first end-to-end acceptance step.
