# V8i direct locomotion teacher

> Historical failed canary. V8i produced safe stationary checkpoints but did
> not acquire commanded forward motion. Do not resume or deploy it. The active
> clean-start recipe is documented in
> [`teleop_v8j_split_locomotion_bc.md`](teleop_v8j_split_locomotion_bc.md).

V8i keeps the V8h audited TWIST2 locomotion prior and adds a short-lived,
deployment-invisible supervised update. Its purpose is to avoid the static local
optimum observed in V8h, where the exponential imitation reward was effectively
zero before PPO ever reached the reference gait.

The deployed contract is unchanged: the actor is still exactly 83 observations
to 18 bounded joint deltas. Unity, PICO, ONNX, and the physical robot do not need
the motion file, phase, reference joints, or any additional input.

## Teacher data and loss

The critic-only 39-value locomotion payload remains:

```text
[blend, sin(phase), cos(phase), q_ref[12], dq_ref[12], q_lead_5[12]]
```

At construction, the algorithm locates `locomotion_prior` by its observation
term name and resolved width. It does not assume that the payload is the last 39
critic values. The action scale, default offset, exact 18-joint order, and the 12
leg action indices are also resolved and validated from the live environment.

After each ordinary PPO update, V8i performs one actor-only optimizer step. For
each stored on-policy actor observation, it reconstructs the raw leg action that
would reach `q_lead_5`:

```text
teacher_raw = (q_lead_5 - joint_position_offset) / action_scale
loss = 0.5 * mean(blend * mean(((actor_leg - teacher_raw) / 0.15 rad)^2))
```

`actor_leg` is the deterministic output after the same asymmetric bounded
transform used by ONNX and the robot runtime. The target is projected only into
the closure of that deterministic transform, with a fail-closed maximum allowed
projection of `0.001 rad`. Four memory chunks accumulate one exact full-rollout
gradient; there is only one additional Adam step per PPO update. Exact-zero
twist rows in the same rollout receive a zero leg-delta anchor at the global
teacher blend, so learning the forward clip cannot silently turn neutral into
uncommanded walking. Other inactive rows (reverse/lateral/yaw) are not given an
invented teacher. The auxiliary loss directly selects only the twelve leg
outputs; the shared actor trunk can still change arm predictions indirectly.
The critic and learned exploration standard deviation receive no auxiliary
gradient, and the actor normalizer contains running-statistic buffers rather
than trainable parameters.

The mean is over every rollout row, not only active rows. Consequently the
forward-sample fraction and the existing V8h blend both reduce the teacher
strength naturally. When every blend value is zero, the auxiliary path skips
`zero_grad`, backward, and `optimizer.step` entirely; even Adam momentum cannot
move the actor after the prior is disabled at update 1,000.

Payload values are validated before PPO mutates the model. Non-finite data,
weights outside `[0,1]`, non-unit active phase vectors, nonzero inactive payload,
wrong widths/order, or an unreachable teacher target stop training.

## Schedule and provenance

The direct teacher is full through update 500, fades linearly through update
1,000, and is absent thereafter. Reset teleport remains certain through update
100, then fades linearly to zero at update 500. A non-teleported forward episode
keeps its real reset state and receives a 20-policy-step (0.4 s) smoothstep joint
reference from that state into frame 109 before the source clip advances. This
teaches initiation from the same pose used by play mode and the robot while
avoiding a discontinuous frame-109 target. Reverse, lateral, and yaw actions are
not invented from a forward-only clip; PPO learns those from the staged sampler.
The underlying artifact and clip behavior remain specified in
[`teleop_v8h_locomotion_prior.md`](teleop_v8h_locomotion_prior.md).

The exact recipe marker is:

```text
v8i_clean_shoulder_std1_twist2_direct_bc_launch_v2
```

V8i starts from the same clean actor initialization as V8h. Older checkpoints
are rejected by recipe and source provenance, so V8i must start a fresh run.

## Reproduction

From the repository root:

```bash
uv lock --check
uv sync --locked
sha256sum data/motions/microban_twist2_walk002_locomotion_prior.npz
uv run --locked --with pytest python -m pytest -q \
  tests/test_locomotion_prior.py tests/test_locomotion_prior_bc.py
scripts/run_microban_teleop_v8_preflight.sh
scripts/train_microban_teleop_v8_stage.sh start \
  --agent.run-name v8i_direct_locomotion_bc_seed42
```

The unit tests verify physical-action learning, blend scaling, critic/std
isolation, and the exact-zero disabled path. The CUDA preflight verifies the
83/18 actor contract and 137-value asymmetric critic. Canonical promotion still
requires the normal staged hard-safety, signed-axis tracking, ONNX parity, and
receipt gates; a falling or static canary is never exported to the robot.
