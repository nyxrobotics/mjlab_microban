# PICO Motion Tracker -> Microban tracking99 simulation

`live_pico_tracking_sim.py` connects the real PICO 4 Ultra controllers and
three Motion Trackers to a simulated Microban. It drives the TWIST2-style
99-observation tracking actor only. This path is simulation-only: it has no
robot hostname, motor socket, UDP sender, or hardware deployment option.

## What is connected

The actor contract is checked at startup and must be exactly:

| Actor slice | Width | Live value |
|---|---:|---|
| `command` | 36 | 18 absolute Microban body-joint references, then 18 finite-difference velocities |
| `motion_anchor_ori_b` | 6 | current-trunk to desired-PICO-pelvis rotation, first two matrix columns |
| `base_ang_vel` | 3 | simulator measurement, unchanged |
| `joint_pos` | 18 | simulator measurement, unchanged |
| `joint_vel` | 18 | simulator measurement, unchanged |
| `actions` | 18 | simulator action history, unchanged |

The online retargeter uses both elbows, hands, knees and feet. Human segment
directions are applied to the corresponding Microban link lengths, so the
operator/robot size difference is handled without copying human distances.
The released-trigger calibration pose is subtracted from subsequent IK output;
therefore calibration maps exactly to the configured Microban initial pose.
The offline walk004 `MotionCommand` is frozen in this executable: it cannot
advance, wrap, resample, or accept a Viser motion-scrubber reset. Startup and
every simulator reset explicitly write `HOME_FRAME` root state and all 21 joint
positions/zero velocities, including the three neck joints.

Here, “configured initial pose” means the software `HOME_FRAME`, not a measured
physical zero offset.  Its shoulder pitch is 0 degrees, shoulder roll is
right -10 / left +10 degrees, and elbow is -20 degrees.  Any real-servo offset
must be measured and applied separately; this bridge does not infer one.

`head`, `neck_roll`, and `neck_pitch` never enter the 18-joint actor or IK.
They use the existing independent, gravity-compensated HMD neck solver.

## Run

Use a tracking checkpoint only after its offline checkpoint gate passes. A
training checkpoint being loadable is not evidence that it tracks safely.

```bash
cd /path/to/mjlab_microban
uv run python -m mjlab_microban.scripts.live_pico_tracking_sim \
  --checkpoint /absolute/path/to/validated/model.pt \
  --input pico-app \
  --viewer native
```

`--input pico-app` uses the paired Microban Unity application and defaults to
`~/.config/microban-teleop/native_transport.json`. The legacy pinned
XRoboToolkit service can instead be selected with `--input native`. WebXR is
not offered because a controller-only browser frame cannot supply the required
24-joint body skeleton.

### PICO simulator camera route

The optional side-by-side simulator camera defaults to loopback port 8081. The
Unity client uses `/calibration.json` and `/frame.jpg`; `/stream` is only a PC
browser preview. Because the server deliberately listens only on PC loopback,
USB-connected PICO hardware requires a second ADB reverse in addition to the
control port:

```bash
adb reverse tcp:63902 tcp:63902
adb reverse tcp:8081 tcp:8081
adb reverse --list
```

Port 8081 is the Unity default and needs no camera config file. If
`--camera-port PORT` is changed, the ADB ports and the Unity
`Application.persistentDataPath/camera_transport.json` must use that same port:

```json
{
  "schema": "microban_sim_camera_client_v1",
  "base_url": "http://127.0.0.1:PORT"
}
```

The synthetic source publishes separate per-eye tangent bounds and a
59.016 mm baseline; set the PICO IPD to 59.0 mm for its live geometry gate.
Source FOV metadata and Unity reprojection are implemented, but eye ordering,
live PICO projection coverage, orientation and end-to-end age still require the
on-device check documented in the PICO client repository. Until that succeeds,
hardware stereo display is not considered verified. Use `--no-camera` to turn
the server off.

## Controls and arming sequence

1. Wear the HMD plus the waist and two ankle Motion Trackers.
2. Hold the left `X` button to select the PICO tracking policy.
3. Keep the left trigger released and stand still until the console reports
   `calibrated=true`. The existing mapper requires at least 20 coherent samples
   spanning 0.5 seconds. The simulated robot must simultaneously be upright,
   nearly stationary, within 25 mm of HOME trunk height, and within 0.12 rad of
   HOME on every body joint before the online IK baseline is accepted.
4. Keep `X` held and hold the left trigger to enable tracking. Releasing the
   trigger immediately returns the body action to raw zero, which is the exact
   configured Microban initial pose.
5. Hold the right trigger when the neck yaw should face the simulated body's
   front. Otherwise HMD yaw/roll/pitch control the three neck joints.

All controls are hold semantics; there is no hidden toggle state. After a
source gap, stale body data, a discontinuity, an IK error, excessive reference
speed, or an authority change, release the left trigger and recalibrate before
trying again.

## Important limitation: sticks

The tracking99 actor has no translational or yaw-velocity observation. It cannot
honestly implement the requested left-stick translation/right-stick turn by
itself. In this executable, any non-zero stick command while armed fails closed
and disarms instead of being silently ignored. Motion Tracker body poses are the
only body-motion command on this path.

This also means `live_pico_tracking_sim.py` is **not** the deadline fallback:
it does not switch to the legacy velocity actor when a tracker disappears, and
a body-tracking fault currently returns the neck to HOME as well. The simulator
camera publisher remains independent. Use `scripts/run_pico_legacy_fallback.sh`
for controller walking, HMD neck control and the simulator camera when body
tracking is unavailable; that launcher deliberately ignores Motion Tracker
body targets. An automatic same-process tracking-to-walk handoff still requires
a reviewed hybrid/supervisory runtime and must not be inferred from these two
separate executables.

Stick locomotion requires a separately trained hybrid actor (tracking reference
plus a 3-value velocity command) or a reviewed supervisory blend with the
velocity actor. That work must retain the same deadman and soft-limit boundary;
it must not feed a velocity-policy action into the tracking actor as though the
two observation contracts were interchangeable.

## Fail-closed boundaries

The bridge rejects and returns the neutral raw action on any of these:

- body health is invalid/stale, contains a pose jump, or is not the exact named
  24-joint schema;
- host or body timestamps regress, freeze with changed payload, or have a gap
  over 100 ms;
- arming occurs before a released-trigger calibration;
- the returned IK pose has any elbow, hand, knee or foot endpoint more than
  20 mm from its target (configurable only within 5--50 mm); the check is the
  maximum final 3D endpoint error, not an average;
- a calibration-relative reference leaves any 18-joint policy soft limit;
- finite-difference reference speed exceeds 5 rad/s (configurable only within
  0.25--8 rad/s);
- actor output is non-finite or has a shape other than `[1, 18]`;
- the authenticated native input owner changes between sampling and action;
- either stick is non-zero while tracking99 is armed.

Actor output is converted to absolute targets, projected against the simulator's
resolved per-joint soft limits, then converted back to raw action space. The
round-trip reconstruction is checked so floating-point rounding cannot place a
target one ULP outside a limit.

This is not an automatic get-up path. A root below 0.10 m, a trunk up-vector z
below 0.50, non-finite state, or implausible root velocity latches the body
action at neutral and slews the neck target back to HOME. Reset the simulation
upright, then release the physical left trigger. Reset alone, releasing `X`, or
recovering upright while the trigger remains held cannot re-arm the policy; a
fresh released-trigger calibration is required after the latch clears.

## Reproduce the contract checks

```bash
cd /path/to/mjlab_microban
uv run python -m unittest tests.test_live_pico_tracking_bridge -v
uvx ruff check \
  src/mjlab_microban/live_pico_tracking_bridge.py \
  src/mjlab_microban/scripts/live_pico_tracking_sim.py \
  tests/test_live_pico_tracking_bridge.py
```

The unit suite covers exact schema width/order, body validation, robot-ready
calibration and hold semantics, the fall/release latch, frozen HOME reset,
duplicate/regressed/gapped timestamps, final endpoint IK/speed/soft-limit
rejection, the exact 42-value tensor patch, and output target projection.

## Current PICO and checkpoint acceptance (2026-09-25)

The USB-connected PICO 4 Ultra is visible to ADB as model `A9210`, and
`com.microban.teleop` is installed and foreground. The headset OS nevertheless
reports `MotionTracker Connect num: 0` repeatedly. Therefore no physical
three-tracker/24-joint acceptance has occurred yet; power, assign and calibrate
the waist and two ankle trackers in the HMD before expecting body frames.

Neither available tracking checkpoint is adopted:

- v8k
  `2026-09-25_07-37-18_v8k_bounded_progress_4096_oneupdate_preflight/model_0.pt`
  no longer strict-loads into the current actor because the checkpoint contains
  unexpected actor `obs_normalizer` state keys. The older compatibility claim
  is stale.
- v8l
  `2026-09-25_07-43-48_v8l_rawactor_startclean_4096_51canary/model_50.pt`
  strict-loads and starts on the isolated `127.0.0.1:63903` simulation route,
  but its recorded checkpoint gate is `status: fail`. With the current physical
  tracker state it receives an invalid body frame, emits the neutral body
  action, then the simulated root falls to about 0.036 m and the fall latch
  engages. This is startup/failure evidence, not an accepted policy.

The isolation check intentionally did not bind the physical control port
`63902` or contact the robot. Reproduce the non-destructive state checks with:

```bash
adb devices -l
adb shell pidof com.microban.teleop
adb logcat -d -v threadtime | \
  rg 'MotionTracker Connect num|tracker num'
```

Run the tracking-only process again only after a checkpoint receipt passes and
the OS reports all three trackers. A successful physical connection must change
the console from `calibrated=false` to `calibrated=true` after a fresh coherent
24-joint stream and the released-trigger calibration window. Until then, use
the legacy fallback launcher for the runnable PICO-to-simulator path.
