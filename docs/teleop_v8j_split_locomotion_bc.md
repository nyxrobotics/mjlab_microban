# V8j split locomotion teacher and collision-sole gate

V8j replaces the V8i canary recipe after the latter converged to a safe but
stationary policy. At checkpoints 1200 and 1300, both neutral and low-forward
ran for 500 steps without falling, but measured forward velocity remained
within `0.0005 m/s` of zero for a `0.10 m/s` command. Continuing that lineage is
therefore not a locomotion solution.

The deployment interface is unchanged: the actor is still exactly 83 physical
observations to 18 bounded arm/leg joint deltas. The locomotion reference is
critic-only training data and is absent from ONNX and the robot runtime.

## Corrected physical inputs

The Microban MJCF names each sole box
`left_foot_collision_1..6`/`right_foot_collision_1..6`. V8i's entity regex
matched only the obsolete unnumbered name, leaving all twelve boxes at
`condim=1, priority=0`. V8j requires all twelve to resolve to
`condim=3, priority=1` and locks that fact with a CPU contract test.

The retargeted motion is grounded against the lowest world-space corner of the
twelve collision boxes, not the convenience foot sites. The static suitability
gate evaluates frames 109 through 267 and requires:

- maximum grounded-sole error no greater than `0.5 mm`;
- maximum relative swing-sole clearance at least `15 mm`;
- p95 relative swing-sole clearance at least `10 mm`.

The prior artifact, robot XML, actuator-identification JSON, lockfile, resolved
configuration, and relevant source bytes are included in training provenance.
A failed suitability receipt blocks a canonical run.

## Exploration and split actor-only loss

Shoulder-roll keeps its one-third-of-one-degree guarded initialization. The
other sixteen action axes start at latent standard deviation `0.15`; PPO entropy
coefficient is `0.0`. The standard deviation remains trainable through PPO's
likelihood objective, but there is no reward for widening it.

After each PPO update, one accumulated Adam step applies three deployment-space
losses over the full rollout batch. `a` is the deterministic bounded actor
output in raw delta coordinates and every group uses scale `0.15 rad`:

```text
forward: 0.25 * blend * mse(a[sagittal legs], q_lead_2 raw), 6 axes
neutral: 0.50 * blend_max * mse(a[all legs], 0),              12 axes
arms:    0.10 * (forward + neutral weight) * mse(a[arms], 0), 6 axes
```

The sagittal axes are right/left hip pitch, knee, and ankle pitch. Hip yaw,
hip roll, and ankle roll are deliberately not copied from the kinematic forward
clip. A neutral anchor is applied only to exact-zero twist rows. Lateral,
reverse, and yaw rows receive no invented gait teacher. Raw action zero means
the configured home pose because the joint-position action's default offset is
validated at algorithm construction.

Four chunks accumulate one full-batch gradient, followed by one gradient clip
and one optimizer step. The loss is normalized by the complete rollout row
count, not the number of active rows. The critic and learned log-standard
deviation receive no auxiliary gradient. If all weights are zero, even
`zero_grad` and `optimizer.step` are skipped so Adam momentum cannot move the
actor.

## Schedule and identity

The teacher blend is exactly one through update 250, fades linearly, and is
exactly zero at update 500. Reset teleport remains certain through update 100
and fades to zero at update 500. Non-teleported episodes retain the 20-step
smooth launch from their actual reset pose. Lead is two source frames. At update
500 the prior is disabled and the ordinary isolated signed-axis sampler is
restored; the next curriculum boundary remains update 1500.

V8j is a clean-start-only recipe. V8i checkpoints cannot resume or export under
these markers:

```text
microban_teleop_actor_initialization =
  clean_random_except_inward_shoulder_roll_v1_other_action_std_0p15_v1
microban_teleop_recipe_revision =
  v8j_clean_shoulder_std0p15_entropy0_twist2_split_bc_lead2_launch_v1
```

## Reproduction

From the repository root:

```bash
uv lock --check
uv sync --locked
scripts/run_microban_locomotion_prior_gate.sh
scripts/run_microban_teleop_v8_preflight.sh
scripts/train_microban_teleop_v8_stage.sh start \
  --agent.run-name v8j_split_bc_seed42
```

The first 1500-update boundary must pass the ordinary multi-seed neutral and
low signed-axis gate before training continues. No checkpoint from this work is
approved for the physical robot until the complete final acceptance suite,
moving-HMD suite, ONNX parity gate, and runtime receipt validation pass.
