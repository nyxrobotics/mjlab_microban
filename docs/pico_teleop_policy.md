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
| `actions` | 18 | previous effective action after target soft clipping, mapped back to raw/delta coordinates |
| `command` | 3 | left-stick `vx/vy` and right-stick yaw rate |
| `foot_target` | 6 | left/right episode-reset-reference XYZ offsets, trunk frame, metres |
| `hand_target` | 8 | left/right episode-reset-reference XYZ offsets, then active flags |

`base_lin_vel`, global/root position and terrain height scans are forbidden from
the actor because the real robot does not provide them. The critic may use
simulation-only signals during asymmetric actor-critic training.

> **Supported-robot acceptance blocker:** the `microban` `PicoHybridMove` now
> rotates the raw BMI088 gyro with `IMU_MOUNT_QUAT` before constructing
> `base_ang_vel`, and the ONNX contract requires the body-frame metadata. Unit
> tests lock that software transform, but the physical mount axes/signs have not
> yet been certified on the assembled robot. Do not deploy this policy until
> small, torque-off roll/pitch/yaw motions in a support fixture confirm the
> mapping. Training, simulation and ONNX export may proceed meanwhile.

The output is exactly 18 actions in model-natural order: right arm (3), right leg
(6), left arm (3), left leg (6). `head`, `neck_roll` and `neck_pitch` are observed
but never output by this policy; the HMD controller owns them. The actor applies
a per-joint asymmetric bounded transform before producing its 18 raw deltas.
Processed joint-position targets therefore remain strictly inside the robot
configuration's 90% soft limits; the environment and physical runtime retain
their target clip as an independent defense.

This is training/deployment contract **v7**. In v1, the observation fed back the
unbounded network output even when the actuator target had already saturated.
That created a hidden recurrence and a flat action nullspace: the policy could
keep producing larger shoulder/hip values while the robot received the same
clipped target. V2 introduced, and v7 retains, this computation:

```text
absolute_target = default_joint_pos + raw_action * action_scale
clipped_target = clip(absolute_target, soft_lower, soft_upper)
effective_action = (clipped_target - default_joint_pos) / action_scale
```

`effective_action` is the next observation in simulation and on the robot. ONNX
metadata must contain training-contract version `7`, observation-schema version
`2`, and the exact semantic string
`effective_action_after_absolute_target_soft_clip_in_raw_delta_coordinates`.
The observation-schema version remains `2` because its 83 fields did not change.
The runtime rejects missing, v1/v2/v3/v4/v5/v6, or otherwise different training
metadata.

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

### V7 requires a clean run

Do **not** resume a v1, v2, v3, v4, v5 or v6 checkpoint. Their tensors have
compatible widths, but v7 changes PPO storage and likelihood evaluation from a
float32 physical action reconstructed through an ill-conditioned inverse to the
exact sampled Gaussian latent. It retains v6's neutral-preserving predicted-state
joint guard, v5's bounded environment transform, v4's corrected target-limit
objective, and the velocity-bootstrap normalizer contract. Every v7 checkpoint
stores the training-contract version and exact action
semantic in `infos`; training resume, normal evaluation, automatic export and
explicit export validate those markers before loading weights. They also require
the checkpoint's internal iteration to equal its canonical `model_N.pt` suffix.
An unversioned/v1 or versioned-v2/v3/v4/v5/v6 checkpoint is intentionally
unusable for v7 resume or export. The v1 diagnostics escape hatch below does not
accept v2 through v6.

For historical comparison only, the evaluator has an explicit escape hatch:

```bash
uv run --locked python -m mjlab_microban.scripts.evaluate_teleop_checkpoint \
  --checkpoint logs/rsl_rl/mjlab_microban_teleop/<v1-run>/model_14999.pt \
  --allow-legacy-teleop-contract \
  --output artifacts/model_14999_v1_diagnostic.json
```

That mode reconstructs the v1 raw-action observation, scalar-Gaussian actor and
ordinary RSL-RL `PPO` algorithm rather than the v7 latent-action adapter.
Its report records `legacy_unversioned_v1`, can never produce `status: pass` or
a deployable artifact, exits nonzero, and the loaded runner refuses save and
every ONNX export entry point.

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

V7 teaches capabilities in strict order. It begins with only locomotion over
`vx [-0.2, 0.3] m/s`, `vy [-0.1, 0.1] m/s`, moving yaw
`[-0.4, 0.4] rad/s` and pure yaw `[-0.8, 0.8] rad/s`. At 1,000 iterations this
widens to `[-0.4, 0.5]`, `[-0.2, 0.2]`, `[-0.8, 0.8]` and `[-1.5, 1.5]`
respectively. At 2,500 it reaches the complete asymmetric runtime envelope
(`vx [-0.5, 0.7] m/s`, `vy [-0.3, 0.3] m/s`, moving yaw
`[-1.5, 1.5] rad/s`, pure yaw `[-3, 3] rad/s`) and 25% pure-yaw sampling.
Hands are enabled at iteration 4,500 with weight `1.0`, standard deviation
`0.08 m` and 70% activation, then tightened at 6,500 to weight `2.0` and
`0.05 m`. Feet remain disabled until 7,500, where weight `2.0`, standard
deviation `0.05 m`, 30% single-foot and 5% both-foot sampling are enabled. At
9,500 feet tighten to weight `3.0`, standard deviation `0.03 m` and 10%
both-foot sampling. Both-foot velocity is forced to zero. Active single-foot Z
is sampled from `2.5--50 mm`; simultaneous-both-foot Z starts at `2.5--12 mm`
and expands at 9,500 to `2.5--20 mm`. Exact XYZ zero is the separate inactive-
foot command.
The live bridge projects every target at or below the inclusive `2.5 mm` floor
band to that exact zero, so the canonical evaluator exercises `2.6 mm` as the
first practical active value.

The MLP output is smoothly limited to the mean envelope of a latent diagonal
Gaussian. Each sampled latent coordinate is mapped to its exact asymmetric,
guarded raw-delta interval with a zero-anchored arctangent bijection:

```text
S = raw_upper                  when z >= 0
S = -raw_lower                when z < 0
action = 2*S/pi * atan(pi*z/(2*S))
```

Latent zero therefore remains raw action zero/default pose, and the derivative
at zero is one from both sides. During rollout, v7 stores and scores the exact
sampled latent in PPO's transition; only a separate tensor returned to the
environment is transformed into the bounded physical raw action. The first
minibatch replay therefore has probability ratio exactly one without performing
a float32 inverse. `distribution.mean` is also in that same latent sample space.
The deterministic policy and ONNX export apply the mean limiter and physical
transform exactly once.

The operational latent interval is derived per side as
`min(1024 * physical_side_width, 32)`. The Gaussian mean is limited to three
eighths of it. Standard deviation is constrained between
`min(0.01, nearest_side/64)` and `min(0.15, nearest_side/16)`, leaving ten
standard deviations between the furthest allowed mean and the nearest outer
envelope. An escaped/non-finite latent, a non-finite MLP output, a non-finite
pre-clamp standard-deviation parameter, action clipping in the RSL-RL wrapper,
RND or symmetry augmentation all fail closed. Resume and export also scan every
floating actor-state tensor for non-finite values. Every finite environment
action remains strictly inside its target limits; an exact numerical endpoint
is rejected rather than silently used.

Contract v6 attempted to store the bounded float32 environment action and
recover its latent with the analytic inverse. The very narrow `1e-4 rad`
shoulder side made that round trip poorly conditioned; its production-width
canary reached non-finite PPO likelihoods at iteration 84. That run and every
v6 checkpoint are rejected. V7 removes the inverse from training entirely.

The actor interval normally moves each absolute soft limit inward by 5% of its
span. If that would put the configured default on or outside the interval (the
one-degree shoulder-roll side is the important case), that side is expanded
only by `epsilon = min(1e-4 rad, half of each physical headroom)` around the
default. Raw zero is therefore strictly interior without moving the default pose
or giving up the physical guard. Export metadata records the exact 18 actor
bounds, the wider hard-clip bounds, guard ratio and epsilon. The wider hard clip
is deliberately retained as a second line of defense.

Exploration uses an exact joint-ordered latent log-standard-deviation vector. Each
initial standard deviation is one third of the nearest soft-limit headroom,
capped at `0.15 rad`; shoulder roll is only `0.00581776 rad` (`0.333 deg`)
because its home pose has one degree of headroom. Entropy bonus is zero. A
normalized per-joint L1 target-clip-excess sum (weight `-2`) remains a defensive
contract check, while the actual joint-limit cost (weight `-10`) penalizes a
crossing after it occurs. A second weight-`-1`
asymmetric per-joint L1 sum keeps a 5% target margin wherever possible, but
expands that margin to include the configured default so raw action zero is
always free. Summing rather than averaging prevents one unsafe joint from being
diluted by the other 17; L1 keeps the reward/return signal linear immediately
outside the hinge instead of quadratically small. The smaller numeric weights
compensate for removing the joint mean while still making the per-joint
large-error reward slope 3.6 times stronger than v3. The
raw-action anchor is `-0.01` and action-rate weight is `-0.02`.

V7 retains v6's weight-`-5` measured-state guard on the 18
policy-controlled joints. It checks both current position and the more dangerous
side of `q + 0.12*qdot`; `0.12 s` is the maximum configured six-sample actuator
delay at 50 Hz. The ordinary boundary is 5% inside each soft range, but it is
clamped at the configured default pose. A stationary neutral pose is therefore
exactly penalty-free and is not moved. Gravity sag or predicted motion from that
neutral toward the close shoulder-roll limit is penalized immediately, allowing
the policy to learn an inward holding target while the measured joint stays at
neutral. This term is a training signal, not a formal deployment-time safety
filter; every deterministic checkpoint evaluation still requires zero measured
soft-limit violation.

This corrects the rejected contract-v5 canary
`2026-09-24_22-08-21_v5_bounded_canary`. Its targets remained legal and target
clip fraction stayed zero, but the left shoulder-roll target saturated near
`+9.994 deg` while the XC330 model sagged under gravity past the `+9 deg` soft
limit. Neutral deterministic measured violation was `0.0254458 rad` at
`model_250` and `0.0278908 rad` at `model_500`; `model_0` and `model_100` were
zero. This was sustained tracking error, not a transient target overshoot, and
the evaluator tolerance was not relaxed. The diagnostic v5 reward experiment
`2026-09-24_23-00-04_v5_neutral_preserving_dynamic_margin_canary` produced zero
target clip, fall, self-collision and measured violation at
`model_0/50/100/250/500`, but its v5-marked checkpoints remain non-deployable.

Planar per-axis absolute-error-sum (L1) and yaw-rate absolute-error (L1) costs
have weights `-2.0` and `-0.5`. Teleop alone uses foot-distance minimum `0.07 m` at weight `-100`;
active foot targets are exempt from the otherwise conflicting stationary no-
stepping cost.

PPO uses a fixed `1e-4` learning rate and three learning epochs. There is no
adaptive-KL reduction: a run whose recorded learning rate differs from `1e-4`
is not this v7 recipe.

Checkpoints and automatic ONNX exports are written under:

```text
logs/rsl_rl/mjlab_microban_teleop/<timestamp>/
```

On a **v7-only** resume, the teleop runner starts at the next (not repeated) PPO
iteration, restores the saved environment step counter, and materializes every
curriculum stage due at that step before collecting another rollout. There is no
legacy counter reconstruction or older-contract fine-tuning path; start a clean
run.

### Optional pinned XC330 velocity actor bootstrap

The preferred first v7 canary may initialize the shared actor inputs from the
known XC330 velocity checkpoint. This is explicit opt-in, not resume. The loader
requires the checkpoint path and exact SHA-256, verifies the 63-value source and
83-value target layouts, then maps base angular velocity, gravity, 18 joint
positions, 18 joint velocities, 18 previous actions and three twist commands.
The new HMD, foot and hand first-layer columns are zero. Their normalizer is
initialized to mean zero and variance/std one. The source normalizer's complete
sample count is preserved exactly so the copied 63-column velocity feature
statistics cannot be overwritten in the first few hundred teleop updates. The
new columns remain identity-normalized and learn through their initially-zero
first-layer weights. Downstream actor MLP layers are copied except for the two
shoulder-roll rows of the final head (action indices 1 and 10). Those rows start
with zero weights and inward latent biases `-0.25` (right) and `+0.25` (left),
then remain ordinarily trainable. This maps deterministically to approximately
24.2 degrees of inward shoulder-roll target at bootstrap without changing the
common `-10/+10 deg` shoulder-roll home pose or either physical soft limit. The
smallest diagnosed passing magnitude was `0.15`; `0.25` is pinned as a
`0.10 rad` latent reserve, while remaining below the separately verified
`0.35` case. The production-width preflight below verifies the resulting pose
and dynamics rather than treating that diagnostic sweep as sufficient.

The guarded-head mapping is
`xc330_velocity_63_to_teleop_83_v4_guarded_shoulder_roll_head`. It fixes a
production-width seed-dependent failure observed with the superseded v3
mapping: pristine actors were bit-identical and safe, but seed 43's first PPO
update held both shoulder targets near the neutral edge for over 100 evaluation
steps. Gravity then pulled the left shoulder `0.00799 rad` beyond its measured
soft limit even though its target was finite and unclipped. Cross-seed
evaluation isolated the actor update rather than reset state or simulation seed
as the cause. Checkpoint provenance records the exact guarded rows and biases;
current v7 loading/export rejects a checkpoint that carries the old mapping or
different guarded-head values.

The preserved-normalizer change is another safety correction retained by v7. In the rejected v3 canary
`2026-09-24_21-13-38`, the source count `1,474,560,000` was capped to
`1,000,000`; it had already become `50,250,304` by `model_500`, while the mean
shared-input standard deviation changed from `0.8574` to `0.3829`. Over the same
neutral evaluation, deterministic target clipping rose from `0.00370` at
`model_0` to `0.20778` at `model_500`. An ablation also produced `0.5116` clip
fraction (and a fall) with the model-0 actor plus model-500 normalizer, and
`0.145` with the model-500 actor plus model-0 normalizer, confirming that both
normalizer drift and policy drift mattered. The v3 joint-mean smooth-L1 costs
also diluted sparse violations and made the reward/return signal quadratically
small at their hinges. That run must not be resumed.

The action distribution, critic, optimizer and PPO iteration are deliberately
not copied. Bootstrap plus `resume` is rejected, and an older teleop checkpoint
is not a bootstrap source. Every resulting v7 checkpoint records the source path,
SHA-256, mapping version, normalizer counts and non-copied components in
`infos.velocity_actor_bootstrap`; that provenance is retained across a later v7
resume. The currently audited source is checked into this repository so a fresh
clone can run the exact canary without relying on an ignored local training
directory:

```text
checkpoints/xc330_velocity/model_14999.pt
sha256 b0bcdadac39716be784207dd6b2b93157162a3e80650e23c05f490c400b9e141
```

First run the checked production-width preflight. For each seed 42, 43 and 44,
it trains exactly one update with 4,096 environments, saves the mapped velocity
actor as `model_pristine.pt` before the first rollout, saves `model_0.pt`, then
evaluates both deterministically for 300 neutral steps using the matching seed.
It fails unless every checkpoint has zero falls/non-finite values/self-contact,
exactly zero target clipping and exactly zero measured soft-limit violation:

```bash
scripts/run_microban_teleop_v7_preflight.sh
```

A 64-environment smoke is only an API check and is not a substitute for this
gate: the first 4,096-environment PPO batch previously moved the unbounded actor
in a different and unsafe direction. `model_pristine.pt` is diagnostics-only;
resume, save-as-current and ONNX export reject it.

After that passes, run three independent 251-update production-width locomotion
safety canaries. They verify the local bootstrap bytes before invoking the
normal training wrapper, save their own pristine baselines, materialize
`model_0`, `model_50`, `model_100` and `model_250` (checkpoint suffixes are
zero-based), and differ only in seed:

```bash
for seed in 42 43 44; do
  MICROBAN_TELEOP_NUM_ENVS=4096 \
  MICROBAN_TELEOP_TARGET_ITERS=251 \
  MICROBAN_TELEOP_SAVE_INTERVAL=50 \
  MICROBAN_TELEOP_SEED="${seed}" \
  scripts/train_microban_teleop_v7_canary.sh \
    --agent.run-name "v7_latent_canary_seed${seed}"
done
```

Evaluate all four checkpoints from every seed deterministically under the neutral
scenario before doing any longer run. Any non-finite value, target clip, measured
soft-limit violation, fall or self-collision rejects the entire three-seed gate;
fix the cause and restart all seeds from clean initialization. The checkpoints
make this gate independent of TensorBoard interpolation. Only after all twelve
evaluations pass may one selected v7 source run be extended toward the complete
locomotion envelope with the normal wrapper:

```bash
scripts/evaluate_microban_teleop_v7_canary.sh <seed42-run> 42
scripts/evaluate_microban_teleop_v7_canary.sh <seed43-run> 43
scripts/evaluate_microban_teleop_v7_canary.sh <seed44-run> 44
```

```bash
MICROBAN_TELEOP_TARGET_ITERS=3000 \
MICROBAN_TELEOP_SAVE_INTERVAL=50 \
scripts/train_microban_teleop.sh resume <v7-canary-run>
```

Resume writes a new timestamp directory. Record that continuation name from the
console output and use it, not `<v7-canary-run>`, for evaluation and any later
resume.

If the same audited bytes live elsewhere, set
`MICROBAN_TELEOP_BOOTSTRAP_CHECKPOINT=/absolute/path/model_14999.pt`; changing
the file does not bypass the fixed digest check.

This reaches the full walking envelope but intentionally precedes hand/foot
stages. Evaluate only locomotion at checkpoint `model_2999.pt`; reduced coverage
is expected to exit `3` with `status: diagnostic`, not canonical `pass`:

```bash
uv run --locked python -m mjlab_microban.scripts.evaluate_teleop_checkpoint \
  --checkpoint logs/rsl_rl/mjlab_microban_teleop/<locomotion-continuation-run>/model_2999.pt \
  --steps 300 --settle-steps 50 \
  --scenarios neutral,max_forward,max_backward,max_lateral_left,max_lateral_right,max_stationary_yaw_left,max_stationary_yaw_right \
  --output artifacts/<locomotion-continuation-run>_locomotion_3000.json
```

Proceed only if the report has no hard-safety failure, responds with the correct
sign on every axis, and clip/actual-limit metrics are already trending downward.
Resume from the latest continuation directory to at least 7,000 for hands and
10,000 for feet, each time using the newly written timestamp for the next stage
and inspecting the corresponding saved checkpoints after each capability
boundary. Only the complete 15,000-iteration run is eligible for the full
canonical acceptance suite below.

Do not copy the resulting model to the robot merely because training completed.
First run a viewer/evaluation pass with neutral, maximum and mixed stick/keypoint
commands and inspect falls, joint-limit saturation, foot slip and self-collision.

The canonical first pass is the headless deterministic evaluator. It opens no
viewer or robot network socket and defaults to CPU, so it does not reserve the
training GPU:

```bash
uv run --locked python -m mjlab_microban.scripts.evaluate_teleop_checkpoint \
  --checkpoint logs/rsl_rl/mjlab_microban_teleop/<run>/model_14999.pt \
  --output artifacts/model_14999_evaluation.json
```

Omit `--checkpoint` to select the highest numeric checkpoint in the newest run
that is at least 10 seconds old. MjLab writes checkpoints directly rather than
through an atomic rename, so both explicit and automatic selection reject a
too-recent file and verify that inode, size and modification time remain stable
while loading. This keeps evaluation from accepting a partially written file
from an active training process. The report also records checkpoint size and
SHA-256. Reports are published atomically, and the default create-if-absent
path cannot replace a file that appears during evaluation; replacement requires
an explicit `--force`.

The default suite runs 15 seeded 20-second/1,000-step scenarios from the same
nominal reset: neutral with exact-zero inactive feet, stationary single-foot and
both-feet targets at the `2.6 mm` active lower edge, bounded combined control,
positive and negative runtime velocity extrema, both stationary-yaw extrema,
both maximum live keypoint corners, two mixed locomotion/keypoint corners, and
a maximum bounded both-feet target. The two both-feet cases exercise the
conservative simultaneous-both-foot distribution introduced in v2: live
fixed-reference retargeting can produce two non-zero foot offsets (for example
a shallow squat), so training samples that case from the floor-band boundary
through its maximum Z offset.
Use `--list-scenarios` to see their stable names, or a comma-separated
`--scenarios` subset for a quick diagnostic.

The JSON contains per-scenario falls/timeouts, raw and clipped actions, policy-
target saturation for the 18 action joints, actual soft-limit checks for all 21
joints (including the HMD-controlled neck), finite physics state, foot slip
while in contact, instantaneous self-collision counts, foot/active-hand RMS and
P95 target error, and linear/yaw velocity MAE. `status: pass` requires:

- at least 99% of scenarios to reach their requested time limit;
- no NaN/Inf, fall, self-collision, actual soft-limit exceedance beyond
  `1e-6 rad`, or policy-target clipping above 0.1% of action values;
- active-hand RMS <= 3 cm and P95 <= 5 cm;
- stationary foot RMS <= 1.5 cm and P95 <= 2.5 cm (the foot objective is
  intentionally faded during locomotion, so moving scenarios are not foot-error
  gates);
- linear velocity MAE <= 0.10 m/s and yaw-rate MAE <= 0.20 rad/s.

A failed acceptance report exits with status 2. A passing report still declares
`deployment_certified: false`: the current contact sensor can miss collisions
within the four physics substeps, play mode keeps the HMD-owned neck static, and
simulation cannot certify the assembled IMU axes, camera path, current limits,
fall restraint, or emergency stop.

Only the complete scenario list with at least 1,000 steps and the default
50-step settling window can produce `status: pass`. A shorter/subset run that
meets its applicable checks is labelled `status: diagnostic`; it is useful for
debugging but is not the canonical checkpoint gate. CLI exit codes are `0` only
for `pass`, `2` for a failed/unknown report and `3` for `diagnostic`, so CI cannot
mistake a legacy or reduced-coverage run for deployment approval.

For a quick simulation-only checkpoint preview (no robot network socket), run:

```bash
uv run --locked play Mjlab-Teleop-Microban \
  --checkpoint-file logs/rsl_rl/mjlab_microban_teleop/<run>/model_<iteration>.pt \
  --num-envs 1 --viewer native
```

Close the MuJoCo window to finish. This shares the GPU, so running it alongside
a 4,096-environment training job reduces training throughput even though the
single preview environment uses comparatively little memory.

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
metadata contains `action_joint_names`, action-aligned defaults/gains/soft
limits, the derived raw-action lower/upper bounds as full-precision JSON strings
(the generic three-decimal list serializer is unsafe for the `1e-4 rad`
shoulder allowance), bounded-distribution
semantics, `observation_joint_names`, all 21 observation-position defaults,
50 Hz control rate, the versioned training/observation schemas,
effective-previous-action and target semantics, and a legacy `joint_names`
alias. This avoids the generic MJLab exporter bug for subset-action policies,
where 21 joint names could be paired with only 18 actions.

## Live PICO mapping

Before actor normalization, construct the observation exactly as documented
above. Convert retargeted foot/hand positions to trunk-relative metre offsets
from the fixed policy-session reset reference and preserve left/right order. Set
an inactive hand's XYZ values to zero and its flag to zero. After converting a
raw output into an absolute joint target, soft-clip it and map the clipped target
back to raw/delta coordinates before storing the previous-action observation;
never feed back the unclipped output. Keep it in policy output order. The
independent HMD controller writes the three
excluded head/neck joints after the policy output is mapped to the 18 arm/leg
servos.

The sibling `microban_teleop` native mapper now implements the live offset
contract. While `pico_teleop` is selected and the left trigger is released, it
collects at least 20 unique fresh body frames over at least 0.5 seconds, freezes
the median trunk-frame hand/foot zero, and derives separate arm and leg scales:

```text
scale = 0.9 * Microban reference limb length / operator limb length
```

The Microban reference lengths come from the same MJCF chains as the offline
retargeter and are protected by a cross-repository contract test. Live targets
stay inside 80% of the trained hand/foot bounds and are slew-limited at every
50 Hz packet. Required joint clocks, source/host gaps, body jumps, implausible
limb lengths and sample-to-send age all fail closed. A rejected frame clears
`walk`, zeros velocity and target slew state, and requires another trigger
release before rearming. `_wire_snapshot()` independently repeats the 80% range
and paired-target checks immediately before UDP serialization.

Policy selection is momentary controller state: holding left X selects
`pico_teleop`; releasing X selects `walk`. Changing X while the left trigger is
held immediately disarms motion and requires a trigger release before rearming.
It is not selected through an environment variable. Every native snapshot declares
`body_target_contract: "microban_pico_offsets_v2_both_feet_stationary"` and
`body_target_safety_margin: 0.8`. Before accepting a hybrid walking snapshot,
the robot runtime independently requires those exact values, complete paired
foot and hand targets, and the same 80%-of-training bounds. A support foot at Z
`<= 0.0025 m` must already be projected to exact XYZ zero. Single-foot live
metadata retains a Z lower bound of zero because that exact zero means inactive;
an active single foot is instead Z `(0.0025, 0.040] m` with XY `+/-0.024 m`.
If both foot offsets are active, each uses the narrower XY `+/-0.008 m`, active
Z `(0.0025, 0.016] m` range and the twist must be exactly stationary. A mismatch
stops and disarms walking, clears both target pairs, and requires a trigger
release before rearming. See
`microban_teleop/docs/twist2_microban.md` and
`microban/docs/pico_teleop_runtime.md` for the reproducible operator and robot
runbooks.

Never substitute world-frame target positions or inferred base linear velocity
without retraining; either change would violate the learned input distribution.
