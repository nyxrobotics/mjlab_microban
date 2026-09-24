# Microban PICO hybrid teleoperation policy

This document is the reproducible runbook for `Mjlab-Teleop-Microban`. The task
is separate from both the deployed walking task and get-up task, so training or
changing it cannot silently alter either existing policy.

## Why there are two motion pipelines

The hybrid RL policy in this task is the live controller. It accepts PICO stick
commands and compact end-effector targets every control tick, then produces the
18 arm/leg joint targets that must keep the physical robot balanced.

BeyondMimic/TWIST2-style full-motion tracking is an offline reference-validation
pipeline for now. Retargeted clips are useful for checking reachability,
self-collision and target ranges, and later for curriculum/data generation. A
full reference trajectory is not streamed directly to the servos: it lacks the
online balance correction and hardware observation contract enforced here.

## Deployment contract

The actor observation is 83 floats in this exact term order:

| Term | Width | Physical source |
|---|---:|---|
| `base_ang_vel` | 3 | trunk IMU gyroscope |
| `projected_gravity` | 3 | trunk IMU orientation/gravity |
| `joint_pos` | 21 | all servo encoders, including HMD-controlled head/neck |
| `joint_vel` | 21 | all servo velocity estimates |
| `actions` | 18 | previous policy output |
| `command` | 3 | left-stick `vx/vy` and right-stick yaw rate |
| `foot_target` | 6 | left/right episode-reset-reference XYZ offsets, trunk frame, metres |
| `hand_target` | 8 | left/right episode-reset-reference XYZ offsets, then active flags |

`base_lin_vel`, global/root position and terrain height scans are forbidden from
the actor because the real robot does not provide them. The critic may use
simulation-only signals during asymmetric actor-critic training.

> **Real-robot deployment blocker:** the training `base_ang_vel` is expressed in
> the trunk/body frame, but the current `microban` `Observer` forwards the BMI088
> gyroscope in its raw sensor frame. Only the quaternion/projected-gravity path
> currently applies `IMU_MOUNT_QUAT`. Do not deploy this policy until a live-policy
> adapter rotates angular velocity into the body frame and its axis/sign mapping is
> verified with small, supported-fixture roll/pitch/yaw motions. Training and ONNX
> export may proceed because the simulation side already uses the body frame.

The output is exactly 18 actions in model-natural order: right arm (3), right leg
(6), left arm (3), left leg (6). `head`, `neck_roll` and `neck_pitch` are observed
but never output by this policy; the HMD controller owns them. Processed joint
position targets are clipped to the robot configuration's 90% soft limits.

Foot offsets are trained as stance/keypoint targets and fade out continuously as
the walking command grows. Hand targets remain active while walking; each hand's
flag masks its reward when that controller target is inactive.

The end-effector zero reference is captured once, after forward kinematics has
been refreshed for each episode reset. The 3--8 second training command resample
changes the sampled offset/active state but never moves that reference. A live
adapter must do the equivalent: capture its calibrated robot/tracker reference
when a policy session is reset or enabled, then keep it fixed until the next
explicit reset. It must not reinterpret each incoming tracker packet as a new
zero pose.

The task-specific home pose also sets both shoulder-pitch joints to `+10 deg`,
matching the committed control-runtime contract in
`microban/src/constants.py:NEUTRAL_POSE`. This is a software-contract match, not
a physical measurement or calibration of the assembled robot. Its history does
not establish that either task predates the other. The override is intentionally
local to the PICO task and does not modify the velocity or get-up environments.

## HMD-owned neck motion during training

The body policy cannot safely learn to compensate for the camera neck if all
three observed neck joints remain fixed at zero in simulation. Training therefore
adds the stateful `hmd_neck_target_motion` step event. It owns only `head`,
`neck_roll` and `neck_pitch`; the policy action remains exactly 18-wide.

The requested ranges come from the robot-side `HmdHeadTrackingMove` limits. The
event intersects them with MjLab's 90% articulation soft limits, giving these
effective training ranges:

| Joint | Robot HMD request range | Effective training range | Target slew |
|---|---:|---:|---:|
| `head` yaw | -85 to +85 deg | -81 to +81 deg | 2.5 rad/s |
| `neck_roll` | -23 to +23 deg | -22.5 to +22.5 deg | 2.5 rad/s |
| `neck_pitch` | -85 to +23 deg | -84.25 to +19.25 deg | 2.5 rad/s |

Every environment independently samples a new waypoint every 0.35-1.50 seconds.
The target advances toward it on every 50 Hz policy step, with the same slew
limit used by the physical HMD controller; 20% of waypoints are neutral dwells.
Sampling uses device-side `torch.rand`, so `--env.seed` reproduces the command
sequence. MuJoCo Warp itself is not guaranteed bit-exact, so reproducibility here
means the generated neck commands, not an identical physics trajectory.

The event is omitted entirely from the `play=True` configuration. A viewer or
live player must explicitly own the neck, just as the HMD controller does on the
robot.

## Clean setup and smoke gate

From the repository root:

```bash
uv sync
scripts/train_microban_teleop.sh smoke
```

The command must finish with a JSON report containing `"status": "pass"`, an
actor shape ending in 83, action dimension 18, the three HMD joint names and their
effective ranges. It also checks the resolved joint order, body-action clips,
absence of non-hardware actor terms, HMD seed reproducibility, target slew and
soft limits, play-mode disablement, finite steps, the `+10 deg` shoulder home,
action-aligned export metadata, episode-fixed keypoint references across periodic
resampling, one-pass resume curriculum materialization, and the formal rotation-
command type and ranges. The latter is important because the registered/Tyro CLI
rebuilds dataclasses: a dynamically attached command `build` callback or rotation
fields would otherwise disappear even if direct Python environment construction
appeared to work.

## Training

Start a new run only after the smoke gate passes:

```bash
scripts/train_microban_teleop.sh train
```

The checked wrapper is the canonical entry point. Its defaults are 4,096 CUDA
environments, seed 42 for both environment and agent, 15,000 iterations,
TensorBoard logging, a checkpoint every 500 iterations, model upload disabled,
and the MjLab NaN guard enabled. Reproduce a different bounded run through the
documented environment overrides, for example:

```bash
MICROBAN_TELEOP_NUM_ENVS=1024 \
MICROBAN_TELEOP_TARGET_ITERS=2000 \
MICROBAN_TELEOP_SEED=7 \
scripts/train_microban_teleop.sh train
```

Resume a timestamped run without changing its total-iteration target silently:

```bash
MICROBAN_TELEOP_TARGET_ITERS=15000 \
scripts/train_microban_teleop.sh resume 2026-09-24_12-34-56
```

`TARGET_ITERS` means completed PPO updates across the original run and this
continuation, not additional updates. The wrapper resolves the highest numeric
`model_<N>.pt` inside exactly the named source directory, treats its zero-based
suffix as `N + 1` completed updates, and passes only the remaining count to
RSL-RL. It refuses a missing run/checkpoint, a target already reached, and CLI
arguments that would override the calculated resume fields. MjLab writes the
continuation into a new timestamp directory; use that new directory name for a
later continuation.

Training uses a staged curriculum copied from the last hybrid development state
(`e17b062`): locomotion first, hand targets at 1,000 iterations, foot/single-
support targets at 2,000, and the wider command range at 3,000. Checkpoints and
automatic ONNX exports are written under:

```text
logs/rsl_rl/mjlab_microban_teleop/<timestamp>/
```

On resume, the teleop runner starts at the next (not repeated) PPO iteration,
restores the saved environment step counter, and materializes every curriculum
stage due at that step before collecting another rollout. For a legacy
checkpoint without environment state, it reconstructs the counter as completed
iterations times 24 rollout steps.

Do not copy the resulting model to the robot merely because training completed.
First run a viewer/evaluation pass with neutral, maximum and mixed stick/keypoint
commands and inspect falls, joint-limit saturation, foot slip and self-collision.

## Deterministic export

Export the latest run:

```bash
uv run python -m mjlab_microban.scripts.export_teleop_onnx \
  --device cpu --output artifacts/microban_teleop.onnx
```

Or specify a checkpoint explicitly:

```bash
uv run python -m mjlab_microban.scripts.export_teleop_onnx \
  --checkpoint logs/rsl_rl/mjlab_microban_teleop/<timestamp>/model_<iteration>.pt \
  --output artifacts/microban_teleop.onnx
```

The exporter rejects anything other than one fixed-width 18-action output. Its
metadata contains `action_joint_names`, action-aligned defaults/gains/soft limits,
`observation_joint_names`, all 21 observation-position defaults, 50 Hz control
rate, the versioned observation schema, raw-previous-action/target semantics and
a legacy `joint_names` alias. This avoids the generic MJLab exporter bug for
subset-action policies, where 21 joint names could be paired with only 18
actions.

## Live PICO mapping

Before actor normalization, construct the observation exactly as documented
above. Convert retargeted foot/hand positions to trunk-relative metre offsets
from the fixed policy-session reset reference and preserve left/right order. Set
an inactive hand's XYZ values to zero and its flag to zero. Keep the previous
action in policy output order. The independent HMD controller writes the three
excluded head/neck joints after the policy output is mapped to the 18 arm/leg
servos.

The current `microban_teleop` native mapper exposes only pelvis-relative absolute
positions as a preview, and `_wire_snapshot()` deliberately strips both target
fields. Those values are **not** valid policy offsets. Live enablement still needs
a calibrated per-session reference subtraction, training-range clamps, freshness
and jump rejection, plus an explicit policy-contract opt-in on both ends.

Never substitute world-frame target positions or inferred base linear velocity
without retraining; either change would violate the learned input distribution.
