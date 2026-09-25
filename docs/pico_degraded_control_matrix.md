# PICO degraded-control matrix

This is the motor-free recoverability contract for the deadline simulation
path. Run it from this repository with:

```bash
scripts/check_pico_degraded_control.sh
```

The script exercises the sibling `microban_pico_client` and
`microban_teleop` repositories, but opens no robot UDP socket or motor
interface. Use `--skip-unity` only on a machine which has not installed Unity.
On another machine, pass its editor explicitly with `--unity /path/to/Unity`.

## Expected behavior

| Fault | Display/control result | Automated evidence |
|---|---|---|
| Camera HTTP server absent | Robot overlay falls back to system passthrough; joystick/HMD control stays independent | Unity camera self-test plus the simulation broken-publisher test |
| Camera config missing/invalid | Overlay remains unavailable; the native control decision neither reads nor subscribes to camera state | Unity camera, native-transport and degraded-control self-tests |
| Camera frame stale, calibration/FOV/session mismatch | The frame is rejected and the overlay requires grip release before rearming; control is not disconnected | Unity camera age, projection, digest and session tests |
| Zero Motion Trackers or invalid body frame | Unity still sends fresh head/controllers with `body:null`. The no-checkpoint path ignores the X selector; the explicit-checkpoint path uses an independent walk mapper and downgrades in the same frame | Unity raw-sampler test, `test_zero_trackers_do_not_disable_legacy_joystick`, and `test_explicit_checkpoint_body_fault_degrades_and_latches` |
| Learned PICO policy absent, throws, or returns non-finite output | The current policy step is recomputed with the pinned legacy actor instead of terminating the viewer | `test_deadline_fallback_never_calls_optional_learned_actor` and `test_learned_actor_faults_fallback_in_the_same_policy_call` |
| Transient authenticated-control disconnect | Motion becomes neutral during the gap. A new session cannot inherit a held trigger; release once and press again to resume joystick walking | Native server test and `test_transient_reconnect_recovers_after_release_then_press` |

The supported deadline command is:

```bash
scripts/run_pico_legacy_fallback.sh
```

That command deliberately omits `--checkpoint`, pins the audited
`model_14999.pt`, ignores the optional X-button policy selector, and accepts
controller/HMD data even when the body stream is unavailable. Camera failures
never become control failures.

## Explicit-checkpoint downgrade

An explicit `live_pico_teleop_sim.py --checkpoint ...` still attempts the
83-input hybrid policy while left X is held. In parallel, a second mapper sees
the same fresh controller frame with X forced released. It preserves the stick,
left-trigger deadman, HMD orientation and right-trigger yaw-front command
without depending on Motion Tracker body validity.

If the body command is unavailable, checkpoint load fails, inference raises, or
the learned actor returns a wrong-shape/non-finite action, the same policy step
uses the pinned legacy actor. Once degraded, selection stays on legacy walk
while the left trigger remains held. Releasing the trigger clears the latch;
the next fresh press retries the learned policy. This prevents frame-by-frame
flapping between actors while retaining deliberate recovery. See
`test_learned_fault_latch_clears_only_on_release_then_retries`.

The legacy actor is still the final locomotion dependency. The learned-fault
handler returns HOME/neutral if that same-cycle fallback call also fails; it
cannot synthesize walking without a functioning locomotion actor.

No design should continue the last motion command while the authenticated
control socket is actually absent. "Recoverable" means neutral during the
communication gap and automatic connection retry, followed by an explicit
left-trigger release and fresh press. This prevents stale held input from
starting motion after a reconnect while keeping the robot immediately
controllable once the operator rearms it.
