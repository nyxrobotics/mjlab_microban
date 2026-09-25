# Microban safe bounded velocity policy

This task is a clean replacement candidate for the legacy unbounded velocity
teacher.  It is registered as `Mjlab-SafeVelocity-Microban`; the existing
`Mjlab-Velocity-Microban` task and its checkpoints are unchanged.  Nothing in
this document is approved for the physical robot until the deterministic gate
passes.

## Fixed policy contract

The actor is `63 -> 512 -> 256 -> 128 -> 18` with ELU activations.  Its input is
concatenated in this exact order:

| Term | Width |
| --- | ---: |
| `base_ang_vel` | 3 |
| `projected_gravity` | 3 |
| `joint_pos` | 18 |
| `joint_vel` | 18 |
| previous effective `actions` | 18 |
| body-frame velocity `command` | 3 |

The output joint order is the same exact 18-joint order used by tracking99 and
teleop83.  Actions are default-relative radian deltas.  The environment first
adds the configured default pose and then applies an absolute clip equal to the
entity's 0.9-softened joint limits.

The actor has no observation normalizer.  The critic remains normalized.  This
avoids the first-update PPO mismatch caused when RSL-RL records each step's old
log probability under a normalizer that continues to change during the same
rollout.

`MicrobanSafeVelocityBoundedGaussianDistribution` is an asymmetric bounded
Gaussian.  PPO stores and scores the exact Gaussian latent, while
`MicrobanSafeVelocityBoundedPPO.act()` sends only its bounded monotonic transform
to the environment.  The learned standard deviation is projected back into its
numeric envelope after every optimizer step.

All clean output rows start at zero except shoulder-roll indices 1 and 10.  They
retain the validated inward latent biases `-0.25` and `+0.25`, respectively.
The shoulder-roll home positions are only about one degree inside their soft
limits; exact-zero shoulder targets allow gravity sag to cross a limit before
PPO has learned a correction.

## Safety and reward terms

The following terms are active from the first rollout:

- absolute target clip equal to the 18 entity soft-limit intervals;
- a 5% preferred target margin that is expanded to include the configured
  default pose, so the margin never declares the home pose invalid;
- the same default-preserving margin applied to both measured `q` and
  `q + 0.12*qdot`, covering the maximum six-sample actuator delay at 50 Hz;
- a small raw-action L2 anchor and light action-rate penalty;
- an XY-only exponential tracking reward plus non-saturating body-frame
  planar/yaw velocity L1 errors; and
- the existing upright, pose, foot, collision and termination terms.

Contact capacity is deliberately `nconmax=512`, `njmax=2048`.  Previous failed
rollouts reached 248 contacts and 513 constraints; lowering these values can
turn a real fall into silent physics truncation.

The current tracked recipe is v9. It retains the v4 command curriculum, v5
planar tracking, and v6 sagittal exploration, and replaces only v8's ineffective
swing-foot term with a weight-`2.0` bounded body-progress reward.
For command `c_xy` and measured root velocity `v_xy`, the unweighted reward is
`clamp(dot(v_xy, c_xy) / ||c_xy||^2, 0, 1)`; command norms at or below
`0.01 m/s` receive zero. Static, reverse, and orthogonal motion therefore
receive zero, matching the commanded speed receives one, and overspeed is
capped at one.

The curriculum begins with forward-only `0.03..0.07 m/s` and the original
velocity rewards for a 100-update stability warmup. At update 100 it keeps the
same range but changes linear tracking from weight `3.0`, std `0.20` to
weight `5.0`, std `0.10`, and changes the linear L1 error weight from `-8.0` to
`-16.0`. It expands forward speed at updates 300 and 600, adds lateral/yaw at
1200, and adds reverse commands at 2500. `rel_forward_envs` must remain zero:
upstream assigns that special subset a minimum `+0.3 m/s`, bypassing these
deliberately small ranges.  The
curriculum implementation applies every overdue stage in a loop, so restoring
`common_step_counter` on resume reconstructs the correct command and reward
state before the first continued update.

## Reproduce training

Run the CPU contract first:

```bash
uv run --python .venv/bin/python --with pytest \
  python -m pytest -q tests/test_microban_safe_velocity_contract.py
```

Then run the one-update GPU smoke and the canary:

```bash
scripts/train_microban_safe_velocity.sh smoke
scripts/train_microban_safe_velocity.sh canary
scripts/train_microban_safe_velocity.sh reward-canary
```

The canary uses 2048 environments.  A measured 4096-environment attempt on the
24 GiB RTX 4090 failed during MuJoCo-Warp graph construction before training:
the EPA workspace alone requested 12,733,906,944 bytes with the required
`512/2048` contact capacity.  Do not reduce the safety capacity to make 4096 fit.
The v9 body-progress canary runs 501 updates. A later continuation uses the
exact v9 source run name:

```bash
scripts/train_microban_safe_velocity.sh resume \
  TIMESTAMP_safe_velocity_v9_2048_501canary 200
```

The dedicated runner requires `common_step_counter == (iteration + 1) * 24`,
starts a resume at `iteration + 1`, records the parent path/SHA/iteration, and
resets the fresh simulation only after restoring the counter.  That reset
applies overdue curricula and resamples commands before the first continued
policy observation.  A source with missing or mismatched provenance is rejected.

## Validate a checkpoint

The fail-closed loader lives in
`mjlab_microban.tasks.microban_safe_velocity_checkpoint`.  It requires:

- a `model_<iteration>.pt` filename matching the checkpoint's internal `iter`;
- the exact actor state key set and every immutable derived bounded-distribution
  buffer, not only its public lower/upper bounds;
- no actor observation-normalizer tensors;
- exact current lower/upper delta bounds;
- finite parameters and an exploration standard deviation within its envelope;
- the current recipe revision plus an exact iteration/common-step relation;
- syntactically valid recorded immediate-parent path/SHA/iteration when the run
  was resumed (the parent file is not recursively reopened by this inspector);
- an optional caller-supplied SHA-256 match.

This is the loader that a later teleop83/tracking99 bootstrap should use.  It
returns the deterministic bounded actor plus a provenance identity containing
the checkpoint SHA, schema version, recipe revision, topology, observation
schema and joint order.

Run the dynamic gate with:

```bash
scripts/train_microban_safe_velocity.sh evaluate \
  logs/rsl_rl/mjlab_microban_safe_velocity/RUN/model_50.pt \
  artifacts/microban_safe_velocity_RUN_model_50.json
```

The gate runs 64 deterministic first episodes for 200 policy steps at a fixed
`+0.08 m/s`.  It reads the actual command tensor after reset and every step and
requires the exact requested twist.  A pre-reset recorder captures terminal
physics before auto-reset overwrites done rows; no partial reset is used, so
active rows' observation-delay buffers advance exactly once per policy step.
Rows are permanently masked after their first termination.  The gate checks
forward velocity and displacement percentiles, falls/completion, target
clipping, actual `q` soft-limit violation, and actual `q + 0.12*qdot`
soft-limit violation.  The worst preferred-margin lookahead excess remains a
diagnostic and training penalty: crossing the preferred margin is not the same
as crossing the mechanical soft limit.  A failed gate is a diagnostic
checkpoint only and must not be copied into teleop or sent to hardware.

Gate revision `microban_safe_velocity_fixed_forward_v3` changes only the
performance velocity-p05 minimum from `0.020` to `0.010 m/s`; displacement
remains `0.020 m`. Completion, falls, finiteness, target clipping, actual soft
limits, and actual `q + 0.12*qdot` soft-limit thresholds are unchanged from v2.

## 2026-09-25 rejected historical runs

The v1 runs at `08-06-16` and `08-08-36` are invalid.  Upstream
`rel_forward_envs=1.0` silently replaced every intended low-speed command with
`+0.3 m/s`; their early evaluator also measured fallen-row displacement after
auto-reset.  They lack recipe metadata and the current loader rejects them.

The corrected-command v2 run
`2026-09-25_08-20-57_safe_velocity_2048_51canary` was trained from scratch.
Its evaluator verified exact `+0.08 m/s` commands on every step, zero actual
soft-limit violation, and zero target clipping.  It still failed stability and
direction-tail gates:

| Checkpoint | forward velocity p05 | completion | falls | displacement p05 |
| --- | ---: | ---: | ---: | ---: |
| model 0 | -0.0969 m/s | 0/64 | 64/64 | -0.1433 m |
| model 25 | -0.0131 m/s | 25/64 | 39/64 | -0.0496 m |
| model 50 | -0.0498 m/s | 22/64 | 42/64 | -0.1433 m |

Model 25's medians were positive (`+0.0874 m/s`, `+0.1845 m`), so more than half
the environments had learned the requested direction.  Tail behavior regressed
after the curriculum expanded at updates 16 and 32.  The v3 trial therefore
changes only curriculum timing (100/300/800/2000), starts from scratch, and does
not reuse v2 weights.

The v3 scratch run `2026-09-25_08-24-46_safe_velocity_v3_2048_301canary` plus
its 200-update continuation learned a fall-free, soft-limit-safe deterministic
policy, but converged to a slow local optimum.  Forward-velocity p05 increased
from `0.00406 m/s` at model 100 to `0.01065 m/s` at model 450, then declined to
`0.01043 m/s` at model 500; the gate requires `0.02 m/s`.  V4 therefore starts
from scratch, preserves the first 100 stability updates, and then strengthens
only the forward-velocity objective.  No v1, v2, or v3 checkpoint is an
approved teacher or deployment policy; the current v9 loader rejects them by
recipe revision.

## 2026-09-25 v4 canary result

The one-update contract smoke completed at
`2026-09-25_08-36-54_safe_velocity_64x1_smoke`.  The 2048-environment scratch
run `2026-09-25_08-37-15_safe_velocity_v4_2048_501canary` then completed all
501 updates.  Its deterministic fixed-command gates were:

| Checkpoint | velocity p05 | displacement p05 | completion | falls | actual soft / lookahead / target violations |
| --- | ---: | ---: | ---: | ---: | ---: |
| model 100 | 0.00655 m/s | 0.02616 m | 64/64 | 0/64 | 0 / 0 / 0 rad |
| model 200 | 0.00507 m/s | 0.02033 m | 64/64 | 0/64 | 0 / 0 / 0 rad |
| model 300 | 0.00785 m/s | 0.03155 m | 64/64 | 0/64 | 0 / 0 / 0 rad |
| model 400 | 0.01089 m/s | 0.04361 m | 64/64 | 0/64 | 0 / 0 / 0 rad |
| model 500 | 0.01174 m/s | 0.04690 m | 64/64 | 0/64 | 0 / 0 / 0 rad |
| model 600 | 0.01265 m/s | 0.05045 m | 64/64 | 0/64 | 0 / 0 / 0 rad |
| model 700 | 0.01273 m/s | 0.05072 m | 64/64 | 0/64 | 0 / 0 / 0 rad |
| model 800 | 0.01338 m/s | 0.05295 m | 64/64 | 0/64 | 0 / 0 / 0 rad |

V4 preserved deterministic stability and all hard safety contracts, and
improved the final velocity tail over v3 model 500, but it still missed the
`0.02 m/s` velocity gate.  Consequently there is no passing v4 receipt and no
checkpoint from these runs is approved as a teleop teacher or deployment
policy.  The worst preferred-margin lookahead excess was `0.01155 rad`; this is
reported separately from the zero mechanical soft-limit lookahead violation.
The current v9 loader therefore rejects v4 checkpoints too.

## V5 planar-tracking ablation

Inspection of the installed Mjlab reward found that its generic
`track_linear_velocity` combines XY tracking error and squared base vertical
velocity inside one exponential.  With the sharpened `0.10 m/s` scale, this can
reward keeping both feet loaded instead of producing the vertical motion needed
to unload a foot.  The existing L1 tracking term is already XY-only.

V5 was a single-variable ablation from scratch: it replaced only that generic
exponential with an XY-only exponential of the same weight and scale.  It keeps
the bounded actor, inward shoulder initialization, absolute clipping, measured
joint guards, every other reward term, and the full staged curriculum unchanged.
It does not reuse a v4 actor or the `walk004` prior.  A checkpoint remains
unapproved unless the same fixed-command gate emits a passing receipt.

The scratch run `2026-09-25_08-50-46_safe_velocity_v5_2048_501canary` and its
300-update continuation remained fall-free and hard-limit safe, but did not
outperform v4.  The deterministic velocity p05 values were:

| Checkpoint | velocity p05 | displacement p05 | completion | falls |
| --- | ---: | ---: | ---: | ---: |
| model 100 | 0.00372 m/s | 0.01491 m | 64/64 | 0/64 |
| model 200 | 0.00599 m/s | 0.02399 m | 64/64 | 0/64 |
| model 300 | 0.00840 m/s | 0.03384 m | 64/64 | 0/64 |
| model 400 | 0.01066 m/s | 0.04280 m | 64/64 | 0/64 |
| model 500 | 0.01132 m/s | 0.04539 m | 64/64 | 0/64 |
| model 600 | 0.01242 m/s | 0.04961 m | 64/64 | 0/64 |
| model 700 | 0.01132 m/s | 0.04514 m | 64/64 | 0/64 |
| model 800 | 0.01149 m/s | 0.04573 m | 64/64 | 0/64 |

Every row had zero actual joint, actual lookahead, and target-clip violation.
The model 600--800 plateau and the training logs' near-zero air-time and peak
foot-height metrics reject XY-only tracking as a sufficient intervention.

## V6 sagittal exploration ablation

V6 keeps every v5 reward, curriculum, safety bound, PPO parameter, and the
entropy coefficient of zero.  Its only change is the initial bounded-Gaussian
latent standard deviation for the six sagittal leg joints
(`hip_pitch`, `knee`, and `ankle_pitch` on each side), from `0.08` to `0.15`.
Hip yaw/roll and ankle roll remain `0.08`; arms retain their prior values and
the narrow shoulder-roll exploration remains mechanically capped.  This tests
whether the existing `air_time` and swing-height rewards were simply never
discovered.  V6 starts from scratch and neither reuses a v5 actor nor uses the
`walk004` prior.

V6 was stopped just after update 200 because every one of the first 209 logged
`Episode_Reward/air_time` samples was exactly zero.  Raw mean foot air time fell
from about `0.006 s` during the unstable initial exploration to at most
`0.00152 s` over the final 20 samples, far below the inherited `0.125 s` reward
threshold.  Model 200 was deterministic, fall-free, and hard-limit safe, with
velocity p05 `0.00684 m/s` and displacement p05 `0.02793 m`, but failed the
velocity gate.

## V7 discoverable air-time signal

V7 keeps v6's sagittal latent standard deviation at `0.15` and changes only
`air_time.threshold_min` from `0.125 s` to `0.04 s`.  The term's weight remains
`2.0`, its upper threshold remains `0.3 s`, PPO entropy remains zero, and all
other reward, curriculum, action-bound, and measured-state safety settings are
unchanged.  It starts from scratch and does not use any previous actor or the
`walk004` prior.

V7 was stopped after the model 300 evaluation.  Across all 316 logged updates,
the air-time reward remained exactly zero.  Over the last 20 updates raw mean
air time was at most `0.00130 s`, so even `0.04 s` remained unreachable.  Model
300 was deterministic, fall-free, and hard-limit safe, with velocity p05
`0.00890 m/s` and displacement p05 `0.03612 m`, but failed the velocity gate.

## V8 bounded command-aligned swing progress

V8 replaces only the ineffective sparse `air_time` reward function, retaining
its weight `2.0` and every v7 exploration, PPO, curriculum, and safety setting.
For exactly one airborne foot, it multiplies three individually bounded terms:

- positive foot height divided by the `0.02 m` target;
- positive swing-foot horizontal velocity along the normalized body-frame XY
  command, capped at `0.10 m/s`; and
- a time envelope that decreases from one to zero over `0.30 s` in the air.

The command is body-frame while foot velocity is reported in world frame, so
the latter is rotated into the current root body frame before their dot product.
The reward is zero for no XY command, both feet airborne, a static or
reverse-moving swing foot, zero-height contact flicker, and an indefinitely
held foot.  It is fail-closed on non-finite or mismatched sensor tensors and is
mathematically bounded to `[0, 1]` before its fixed weight.  V8 starts from
scratch and does not use a prior actor or `walk004`; the loader rejects v1--v7.

The scratch run `2026-09-25_09-18-49_safe_velocity_v8_2048_501canary` was
stopped after model 200. The new reward was nonzero in 119 of 223 logged
updates, but its tail collapsed: over the last samples its maximum episode
reward was `8.49e-6` and the corresponding support metric was only `1.49e-4`.
The deterministic gates remained hard-limit safe and fall-free but regressed
relative to v6/v7:

| Checkpoint | velocity p05 | displacement p05 | completion | falls | actual soft / lookahead / target violations |
| --- | ---: | ---: | ---: | ---: | ---: |
| model 100 | 0.00440 m/s | 0.01749 m | 64/64 | 0/64 | 0 / 0 / 0 rad |
| model 200 | 0.00609 m/s | 0.02463 m | 64/64 | 0/64 | 0 / 0 / 0 rad |

V8 is rejected: requiring a detectable foot lift made the shaping signal vanish
before it could solve the underlying planar velocity objective.

## V9 bounded body-progress reward

V9 changes only the function occupying v8's weight-`2.0` `air_time` reward
slot. It directly scores command-aligned body-frame XY velocity with the
bounded formula documented above. All action distributions, initial standard
deviations, PPO settings, curriculum stages, tracking terms, and safety guards
remain configured as in v8. Inputs, intermediate arithmetic, and shapes are
checked fail-closed, while the live mean is logged as
`Metrics/commanded_planar_velocity_progress`.

V9 started from scratch; no v8 actor was restored. Its exact recipe was
`scratch_bounded_inward_shoulder_sagittal_bodyprogress_v9`. The run
`2026-09-25_09-27-27_safe_velocity_v9_2048_501canary` completed 501 updates.
The shaping signal was live at every update: its raw metric ranged from
`0.0280` to `0.8061`, and its last-20 mean was `0.1422`. Deterministic results
improved monotonically while remaining fall-free and hard-limit safe:

| Checkpoint | velocity p05 | displacement p05 | completion | falls | actual soft / lookahead / target violations |
| --- | ---: | ---: | ---: | ---: | ---: |
| model 100 | 0.00419 m/s | 0.01671 m | 64/64 | 0/64 | 0 / 0 / 0 rad |
| model 200 | 0.00789 m/s | 0.03234 m | 64/64 | 0/64 | 0 / 0 / 0 rad |
| model 300 | 0.01028 m/s | 0.04243 m | 64/64 | 0/64 | 0 / 0 / 0 rad |
| model 400 | 0.01173 m/s | 0.04759 m | 64/64 | 0/64 | 0 / 0 / 0 rad |
| model 500 | 0.01361 m/s | 0.05431 m | 64/64 | 0/64 | 0 / 0 / 0 rad |

Model 500 failed the former v2 `0.020 m/s` performance threshold. Under the
user-approved v3 threshold it passes all eight checks. The canonical accepted
source and receipt are:

- checkpoint:
  `logs/rsl_rl/mjlab_microban_safe_velocity/2026-09-25_09-27-27_safe_velocity_v9_2048_501canary/model_500.pt`
- checkpoint SHA-256:
  `416a8b16f7f7980822e4e1df81ffaf9515bc18a246e6fc257405a2c46ceece93`
- receipt: `artifacts/microban_safe_velocity_v9_model500_v3_pass.json`
- receipt SHA-256:
  `e68701b11774dd30c8e45a2fd89614a2e4423a9486d01a0d936f0fa6fb760492`

The v3 rerun measured velocity p05 `0.0136098 m/s`, displacement p05
`0.0543144 m`, completion `64/64`, falls `0/64`, and zero target, actual-soft,
actual-lookahead-soft, or non-finite violations. The preferred-margin lookahead
excess `0.0115489 rad` remains diagnostic rather than a mechanical-limit
failure.

## V10 staged body-progress weight

V10 changes only the v9 body-progress weight. Updates 0--99 keep weight `2.0`;
the existing update-100 curriculum stage changes it to `6.0` together with the
already-established tracking changes. Every other reward, command range,
initial distribution parameter, PPO setting, and safety constraint is
unchanged. The reward-override validator accepts only reviewed term/field
combinations and validates the entire mapping before mutation. Resume restores
all overdue stages, including weight `6.0`, before the first continued rollout.

V10 starts from scratch and never restores a v9 actor. Its exact recipe is
`scratch_bounded_inward_shoulder_sagittal_bodyprogress_weight6_v10`. The run
`2026-09-25_09-37-22_safe_velocity_v10_2048_501canary` completed 501 updates.
The raw progress metric remained nonzero at every update, but tripling its
weight reduced deterministic performance relative to v9:

| Checkpoint | velocity p05 | displacement p05 | completion | falls | actual soft / lookahead / target violations |
| --- | ---: | ---: | ---: | ---: | ---: |
| model 100 | 0.00306 m/s | 0.01186 m | 64/64 | 0/64 | 0 / 0 / 0 rad |
| model 200 | 0.00779 m/s | 0.03109 m | 64/64 | 0/64 | 0 / 0 / 0 rad |
| model 300 | 0.00882 m/s | 0.03448 m | 64/64 | 0/64 | 0 / 0 / 0 rad |
| model 400 | 0.01034 m/s | 0.04053 m | 64/64 | 0/64 | 0 / 0 / 0 rad |
| model 500 | 0.01075 m/s | 0.04220 m | 64/64 | 0/64 | 0 / 0 / 0 rad |

V10 is rejected: model 500 is 21% slower than v9 model 500 and missed the
then-current v2 `0.02 m/s` performance gate. There is no passing receipt.

## V11 exact-command curriculum

V11 restores v9's weight-`2.0` body-progress reward and changes only command
sampling. The initial command and all five stage command dictionaries are
exactly `lin_vel_x=(0.08, 0.08)`, `lin_vel_y=(0, 0)`, and
`ang_vel_z=(0, 0)`. The update-100 tracking/L1 sharpening remains, but later
stages no longer broaden the command distribution. This removes the train/eval
command mismatch while preserving every bounded-action and measured-state
safety contract.

V11 started from scratch and never restored a v9 or v10 actor. Its exact recipe
was `scratch_bounded_inward_shoulder_sagittal_fixed008_bodyprogress_v11`.
It was explicitly stopped during update 195 when the decision returned to the
better v9 actor. Its model 100 gate was fall-free and hard-limit safe but slow:
velocity p05 `0.00183 m/s` and displacement p05 `0.00695 m`. V11 is incomplete,
has no passing receipt, and the current strict loader rejects it by recipe.
