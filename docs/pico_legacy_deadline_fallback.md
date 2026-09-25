# PICO legacy-walk deadline fallback

This is the short path to a usable PICO-to-simulator demo without waiting for
the new hybrid policy to finish training. It runs the already trained
`checkpoints/xc330_velocity/model_14999.pt` velocity actor as the locomotion
backbone. Startup rejects any other bytes; the accepted SHA-256 is
`b0bcdadac39716be784207dd6b2b93157162a3e80650e23c05f490c400b9e141`.

The fallback does **not** project this actor through the rejected hybrid-policy
action closure. With no `--checkpoint`, startup selects the original
`Mjlab-Velocity-Microban` task and runner: the actor receives its native 63-value
observation and its 18 raw outputs go directly to the original action term.
Both wrapper action clipping and action-target clipping are `None`. Raw zero
uses that task's `use_default_offset` and therefore maps exactly to HOME. The
live wrapper changes only session concerns—one environment, deterministic reset,
no random push/domain randomization, and no 20-second timeout—not the policy's
observation or action semantics.

It is simulation-only. No Microban motor/UDP destination exists in this
executable.

## Two commands

From this repository, verify the complete hardware-free path:

```bash
scripts/run_pico_legacy_fallback.sh check
```

Then launch the Unity PICO client, simulator, and stereo stream:

```bash
scripts/run_pico_legacy_fallback.sh
```

The launcher reprovisions the already installed development app with the
isolated simulation pair, starts it, and passes
`~/.config/microban-teleop/native_transport_sim.json` to the simulator. Control
uses loopback TCP 63903 and the camera uses 8081. This intentionally leaves the
physical-camera gateway on 63902/8082 untouched. The simulator camera is served
at `http://127.0.0.1:8081/stream`; Unity reads the matching exact synthetic
pinhole geometry from the calibration endpoint.

Controls are momentary, not toggles:

- release the left trigger once after connecting, then hold it to run;
- left stick is forward/back and left/right translation;
- right-stick X is yaw;
- HMD orientation drives the three neck joints;
- hold the right trigger to force neck yaw to the body's front;
- hold the left grip to request the stereo simulator view; release it for
  passthrough;
- releasing the left trigger immediately returns the body action to HOME.

In this fallback the audited legacy actor always owns locomotion. The left `X`
hybrid-policy selector is deliberately overridden, so an unfinished v11 policy
cannot accidentally be selected. The trigger mapper's disconnect, stale-frame,
rearm and policy-change fail-closed behavior is retained.

## What `check` proves

The check needs neither the PICO nor Microban. It verifies the checkpoint hash,
strict-loads the real 63-input/18-output actor, and feeds synthetic controller
frames through the production control mapping. It covers trigger hold/release,
forward, lateral, yaw, HMD orientation, right-trigger yaw centering, reset
disarming, release-zero action history, and release-to-HOME semantics. It also
starts the real loopback stereo publisher, renders a frame, reads its
health/calibration endpoints and downloads a complete JPEG.

Most importantly, the same command performs nine closed-loop 300-step rollouts
in the nominal legacy live runtime: neutral, two forward, two backward, both
lateral signs, and both yaw signs. The current seed-42 result is 9/9 completed,
0 simulated falls, 0 non-finite states, 0 actual simulated joint soft-limit
violations, and 8/8 signed commands moving in the requested direction.
Representative measured means are `+0.0965 m/s` for a `+0.1 m/s` request,
`+0.1574 m/s` for `+0.2`, `+0.4548 rad/s` for `+0.5` yaw, and
`-0.5288 rad/s` for `-0.5` yaw.

The checked camera geometry is 640x480 per eye, left eye first, 122 degrees
horizontal FOV and 107.065362573 degrees vertical FOV. These are exact synthetic
render parameters, not an estimate of the USB camera lens.

## Why this path is selected

The original-environment diagnostic for the pinned legacy actor completed all
9 command scenarios without a fall, non-finite value, or actual joint-limit
violation. All eight signed movement commands produced the requested direction.
Regenerate that evidence with:

```bash
uv run --locked python -m \
  mjlab_microban.scripts.evaluate_legacy_velocity_checkpoint
```

Do not add a naive soft-limit clamp to that actor: the separately evaluated
clamped variant completed only 8/9 scenarios, fell during the -0.2 m/s backward
case, nearly stopped translating/turning, and reversed both lateral responses.

The fresh v11 actor was stopped at the 100-update canary instead of consuming
the deadline. Its deterministic six-scenario check passed hard safety (no fall,
non-finite action, collision, or actual soft-limit violation) but did not learn
locomotion: the -1.0 rad/s yaw case averaged +0.0062 rad/s. It is therefore not
selected by this launcher. Re-run that bounded diagnostic with:

```bash
scripts/evaluate_microban_teleop_v11_stage.sh --canary \
  2026-09-25_15-07-33_v11_fresh_100canary
```

## Isolated Unity input and optional WebXR

The main command above is the authenticated Unity path. Its two matched secrets
are owner-only local files and are never printed or committed:

- PC: `~/.config/microban-teleop/native_transport_sim.json`, listener 63903
- PICO transfer bundle: `~/.config/microban-teleop/pico-usb-sim.json`, target
  `127.0.0.1:63903`

As of 2026-09-25 the separately launched `microban-physical-camera-gateway`
owns `0.0.0.0:63902`. The fallback never kills it, changes it, or binds that
port. It also refuses to start if its own 63903 or 8081 is occupied.

Controller-only WebXR remains available as a diagnostic alternative:

```bash
scripts/run_pico_legacy_fallback.sh launch-webxr
```

That mode installs ADB reverse routes for 8443/8081; open
`http://localhost:8443/` in PICO Browser.

## Remaining limits

- This fallback proves the controller/actor/camera wiring and nine simulated
  trajectories, not hardware safety. The actor's hypothetical absolute target
  projection is nonzero in every scenario even though actual simulated joints
  remained within their soft limits. Do not infer that raw legacy output is safe
  for unsupported physical operation.
- It does not use the three Motion Trackers for body retargeting. That remains
  the new hybrid/tracking policy path; locomotion here is sticks plus the proven
  velocity actor.
- On 2026-09-25 the installed Unity app reached `Ready / authenticated`, read
  fresh camera frames, and passed projection/FOV coverage. The remaining live
  stereo gate was `eyeBaselineValid=false`: set the PICO IPD to 59.0 mm to
  match the simulator's 59.016 mm camera baseline. IPD/FOV validation remains a
  display-geometry check, not a locomotion authority gate.
- Camera delay, stale/missing frames, or renderer failure must degrade the HMD
  view to passthrough or its explicitly marked last frame while controller
  transport continues. The PC simulation policy catches camera-publisher
  failure and disables only video capture. The advertised `max_frame_age_ms` is
  `30000`: it is a 30-second view sanity bound, never a control-transport gate.
  Final eye order, perceived scale, display latency and comfort still require
  an on-head check.
