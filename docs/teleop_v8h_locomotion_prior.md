# V8h privileged locomotion prior

This note is the reproducible specification for the short-lived TWIST2-derived
locomotion prior used by `Mjlab-Teleop-Microban`. It does **not** change the
deployed actor: the actor remains 83 observations and 18 actions, and no motion
reference is needed by Unity, the PICO bridge, ONNX, or the physical robot.

## Audited input

The repository vendors the exact retargeted artifact at:

```text
data/motions/microban_twist2_walk002_locomotion_prior.npz
```

Its required SHA-256 is:

```text
e789594b7711eb7e001edbde12064c9068e9649dfa6945626478171b48ad9fa0
```

It contains 268 frames at 50 Hz from source session
`twist2-b06178f19a22-0807_yanjie_walk_002`. Training uses only inclusive frames
109 through 267. The command never loops. The source joint order, 18 policy
joint order, HMD-only joint list, body shapes, source schema and retarget profile
are checked at environment construction. A missing, modified or malformed NPZ
fails before rollout. The binary itself is included in the checkpoint training
source manifest, so its SHA is transitively pinned by every canonical checkpoint
and stage receipt.

## Actor/critic boundary

The actor contract is unchanged:

```text
83 observations -> bounded 18-joint action
```

Only the critic receives an extra 39-value command:

```text
[blend, sin(phase), cos(phase), q_ref[12], dq_ref[12], q_lead_5[12]]
```

The 12 values are ordered right hip yaw/roll/pitch, knee, ankle pitch/roll,
then the same six left-leg joints. `dq_ref` is scaled by the phase rate. The
reference is absent from the actor observation config, actor metadata, ONNX and
runtime protocol. Train and play configs retain the same critic/reward/
termination topology; play sets the prior command to disabled so all 39 values,
both imitation rewards and the clip-finished termination are exactly zero.

## Reset and phase behavior

At episode reset the ordinary velocity command is sampled first. Prior
eligibility is then latched for that episode only when all of these hold:

- the prior is enabled and its global blend is greater than zero;
- `vx` is within `+0.06..+0.11 m/s`;
- lateral and yaw commands are exactly zero within `1e-6`.

Eligible environments start at frame 109. Reset overwrites only the twelve leg
joint positions and phase-scaled velocities. It also sets root Z, quaternion,
linear velocity and angular velocity from frame 109, while root X/Y are the
environment origin. It deliberately does not write the six arm joints or the
three HMD-owned joints. The leg position targets are updated to the same reset
pose to avoid a stale-target impulse on the first physics substep.

The nominal clip speed is `0.0780311897 m/s`. Phase advances each 50 Hz policy
step by:

```text
phase_rate = commanded_vx / 0.0780311897
```

Thus the supported command interval gives rates about `0.769..1.410`; the worst
phase-scaled reference joint speed remains below `4.55 rad/s`. Linear
interpolation is used for fractional phases. The action reference is exactly
five source frames ahead and clamps at frame 267. Reaching frame 267 terminates
the eligible episode. It never wraps to frame 109.

## Rewards and fade

Two positive rewards are active only as `eligible * blend`:

| Reward | Weight | Raw value |
|---|---:|---|
| absolute leg action target vs. `q_lead_5` | `+2.0` | `exp(-mean(error^2) / 0.15^2)` |
| measured leg position vs. `q_ref` | `+1.0` | `exp(-mean(error^2) / 0.15^2)` |

The action target is reconstructed from raw action, scale and default offset,
then clipped by the same absolute action limits used by the simulator. The mean
is over the 12 legs (not a sum), so `0.15 rad` retains its per-joint meaning.
There is no root imitation and no joint-velocity imitation reward.

The global blend is `1.0` through `500 * 24` environment steps, fades linearly
to `0.0` at `1000 * 24`, and stays exact zero thereafter. One PPO update is
fixed at 24 steps, so these are exactly updates 500 and 1000 even after resume.

## Command curriculum

The first external gate remains update 1,500. Two internal stages precede it:

| Update | Sampler/prior state |
|---:|---|
| 0 | 20% standing, 80% forward `+0.06..+0.11`; prior blend 1 |
| 500 | 10% standing, 30% forward, 12% each backward/left/right/yaw-left/yaw-right; linear prior fade begins |
| 1,000 | restore v8g 10% standing + 15% each signed direction and its original ranges; permanently disable prior |
| 1,500+ | existing v8 signed-axis, mixed-command, HMD and keypoint stages unchanged |

At the 500 stage, backward is `-0.35..-0.20 m/s`, lateral is
`+0.15..+0.25` / `-0.25..-0.15 m/s`, and yaw is
`+0.8..+1.2` / `-1.2..-0.8 rad/s`. Forward stays in the clip-supported
`+0.06..+0.11 m/s` interval until update 1,000. The prior is already zero and
disabled for the complete 1,000-to-1,500 acquisition interval before the first
external capability gate.

On an iteration-resuming load, the runner first restores the checkpoint global
step, reapplies every due curriculum stage, and then resets all environments
once under that restored configuration. Checkpoints do not preserve simulator
or RNG state, so this discards only the constructor's provisional step-zero
reset. It prevents a frame-109 prior pose or low-forward sample from leaking
into the first rollout of an update-1,000-or-later resume. Actor-only evaluator
loads do not take this training-resume reset path.

## Reproduction and verification

From the repository root:

```bash
uv lock --check
uv sync --locked
sha256sum data/motions/microban_twist2_walk002_locomotion_prior.npz
uv run --locked --with pytest python -m pytest -q tests/test_locomotion_prior.py
scripts/run_microban_teleop_v8_preflight.sh
scripts/train_microban_teleop_v8_stage.sh start \
  --agent.run-name v8h_locomotion_prior_seed42
```

The tracked `uv.lock` is also hashed into checkpoint source provenance. The
preflight includes artifact/schema validation, a mean-vs-sum reward test,
critic-only observation checks, exact-zero play behavior, reset write-scope,
phase scaling/non-looping behavior, curriculum transitions, source provenance,
the full policy contract and the environment smoke. As with every v8 recipe
change, v8h must start from a clean actor; older v8g and diagnostic checkpoints
are rejected by the recipe marker and source provenance.

The first production checkpoint is still `model_1499.pt`. Evaluate and resume
with the ordinary fail-closed stage scripts documented in
[`pico_teleop_policy.md`](pico_teleop_policy.md). Do not export merely because
the prior preflight passes; all canonical capability gates remain required.
