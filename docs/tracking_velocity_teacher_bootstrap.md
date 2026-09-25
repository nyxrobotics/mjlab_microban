# Tracking actor velocity-teacher bootstrap audit

## Decision

Do **not** behavior-clone either legacy velocity checkpoint into the 99-input
Microban tracking actor.  Both candidates fail the projected-teacher dynamic
gate and move opposite the `walk004` reference direction.  No BC optimizer step
or derived tracking checkpoint was produced from either failed teacher.

The reusable loader, safety projection, per-joint BC mask, and dynamic evaluator
remain in the tree so a newly trained bounded velocity controller can be tested
without weakening this decision.

## Why the legacy actor is only a proposal

The legacy velocity actor consumes 63 values in this exact order:

```text
base_ang_vel(3), projected_gravity(3), joint_pos_rel(18),
joint_vel_rel(18), previous_action(18), velocity_command(3)
```

The tracking actor consumes 99 values and has a bounded deterministic output.
The old Gaussian is unbounded, so its weights or raw outputs are never copied
directly.  The frozen teacher loader verifies the complete checkpoint SHA-256,
loads only `actor_state_dict`, disables gradients, and checks every floating
tensor for finiteness.

For the fixed `walk004` dataset, the teacher velocity command is the reference
trunk linear/angular velocity transformed into the reference body frame.  It is
not an unobservable free command: each command value belongs to the same fixed
clip phase already represented by the student's `q_ref/qdot_ref` input.  This
construction must not be reused on a dataset where identical student inputs can
have different hidden velocity commands.

Each proposal `a_old` is converted and projected in this order:

```text
t_old = offset_old + scale_old * a_old
t_soft = clamp(t_old, soft_lower, soft_upper)
a_student = (t_soft - offset_student) / scale_student
a_label = clamp(a_student, student_deterministic_lower,
                student_deterministic_upper)
t_label = clamp(offset_student + scale_student * a_label,
                soft_lower, soft_upper)
```

The student interval is the closure of the actual double arctangent bounded
output, moved one floating-point ULP toward zero so every label has a finite
actor logit.  Projection beyond 1 mrad is recorded as no longer faithful to the
legacy controller.  The projected controller is nevertheless evaluated as a
new controller; only its own complete dynamic pass can make it a BC candidate.

Before a sample is admitted, measured state must also pass:

```text
q_pred = q + 0.12 * qdot
dangerous_lower = min(q, q_pred)
dangerous_upper = max(q, q_pred)
soft_lower <= dangerous_lower and dangerous_upper <= soft_upper
```

`0.12 s` is the configured worst-case six-policy-sample actuator delay.  A hard
lookahead failure rejects the whole sample.  The neutral-preserving 5% margin
is a per-joint BC weight, so one narrow shoulder-roll margin does not erase all
leg supervision.

## Pinned candidates

| Candidate | SHA-256 | Source |
| --- | --- | --- |
| `model_999.pt` | `90dc3a3e8dd7a79b73b09ed60e27dc707ee7656213c8084b1e047e65d1da3305` | `mjlab_microban_velocity/2026-09-25_03-21-43_xc330_velocity_4096_canary/model_999.pt` |
| `model_14999.pt` | `b0bcdadac39716be784207dd6b2b93157162a3e80650e23c05f490c400b9e141` | versioned at `checkpoints/xc330_velocity/model_14999.pt` |

`model_999.pt` is intentionally not copied into the repository because the gate
rejects it.  Supply its exact local path and digest when reproducing the audit.

## Canonical rejection commands

Run from the repository root.  Exit status `1` is expected for these rejected
candidates; the JSON receipt is still written atomically.

```bash
uv run python -m mjlab_microban.scripts.evaluate_tracking_velocity_teacher \
  --teacher-checkpoint /path/to/model_999.pt \
  --teacher-sha256 90dc3a3e8dd7a79b73b09ed60e27dc707ee7656213c8084b1e047e65d1da3305 \
  --teacher-iteration 999 \
  --num-envs 256 \
  --output artifacts/microban_tracking_velocity_teacher_model999_gate256.json

uv run python -m mjlab_microban.scripts.evaluate_tracking_velocity_teacher \
  --teacher-checkpoint checkpoints/xc330_velocity/model_14999.pt \
  --teacher-sha256 b0bcdadac39716be784207dd6b2b93157162a3e80650e23c05f490c400b9e141 \
  --teacher-iteration 14999 \
  --num-envs 256 \
  --output artifacts/microban_tracking_velocity_teacher_model14999_gate256.json
```

Both commands use seed 42, 256 environments per pass, source frame 0 as the
initial state, frames 1 through 267 as the 267 policy targets, exact 50 Hz
control, a deterministic nominal pass, and a robust startup-randomized pass.
They enforce the same fall, soft-limit, contact, direction, forward-velocity,
and velocity-error checks as the tracking-checkpoint gate, plus complete safe
teacher-label admission.

## Recorded results (2026-09-25)

| Teacher/pass | Complete | First loss | Forward velocity p05 | Forward displacement p05 | Safe-label fraction | Legacy-faithful fraction | Max legacy projection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `999` nominal | 0/256 | frame 25 | -0.07885 m/s | -0.06150 m | 99.9763% | 0.0000% | 5.1715 rad |
| `999` robust | 59/256 | frame 24 | -0.07886 m/s | -0.06330 m | 99.9877% | 0.0123% | 5.1522 rad |
| `14999` nominal | 0/256 | frame 35 | -0.06716 m/s | -0.05261 m | 99.9795% | 0.0000% | 5.4280 rad |
| `14999` robust | 22/256 | frame 32 | -0.07189 m/s | -0.05075 m | 99.9875% | 0.9109% | 6.2494 rad |

The negative velocity/displacement is decisive: safe projection changes the
legacy bang-bang joint targets into a controller that travels backward relative
to the required clip.  It also misses the 267-step completion requirement and
the 0.075 m/s XY velocity-error bound.  High safe-label fractions do not rescue
an incapable projected controller.

Two small diagnostic command changes were also rejected for `model_14999`:
reference-command scale `-1` completed 0/16 in both passes; scale `4` completed
5/16 nominal and 4/16 robust, but still had negative forward-velocity p05.
Those are diagnostics only and are not candidate recipes.

## Next teacher candidate

Train a new velocity policy under the **same bounded deterministic action
closure and hard action clip used by the 99-input student**.  Require its own
nominal and robust full-clip dynamic gate to pass before enabling any BC loss.
This removes the multi-radian legacy projection that destroyed both audited
controllers.  If bounded velocity RL cannot pass, the next option is a physics
closed-loop MPC/trajectory controller; raw kinematic `walk004` playback has
already failed its dynamic gate and must not be relabeled as a teacher.

## Unit checks

```bash
uvx ruff check \
  src/mjlab_microban/tasks/microban_tracking_teacher.py \
  src/mjlab_microban/scripts/evaluate_tracking_velocity_teacher.py \
  tests/test_microban_tracking_teacher.py

uv run --locked --with pytest python -m pytest -q \
  tests/test_microban_tracking_teacher.py
```
