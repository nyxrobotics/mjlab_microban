# Microban PICO teleop contract v9

Contract v9 starts the full 83-input PICO policy only from an acceptance-gated
checkpoint of the dedicated bounded/raw `Mjlab-SafeVelocity-Microban` actor.
Early safe-velocity canaries affected by the forced-forward reset-command bug
are not valid sources. Historical normalized,
unbounded `Mjlab-Velocity-Microban` checkpoints (including model999 and
model14999) are not compatible and are rejected before any weights are loaded.

## What is copied

The source actor topology is exactly `63 -> 512 -> 256 -> 128 -> 18`. The
target topology is `83 -> 512 -> 256 -> 128 -> 18`.

- All 63 shared raw observation columns are copied with the explicit
  `VELOCITY_TO_TELEOP_OBSERVATION_INDEX` mapping.
- The 20 new HMD, foot, and hand columns in the first layer are exact zero.
- Every later MLP tensor, including all 18 output rows, is copied exactly.
- The learned log standard deviation and every registered bounded-transform
  buffer are copied only after the complete source and target distribution
  contracts compare equal.
- No observation normalizer exists on either actor. The teleop critic remains
  normalized independently.
- Critic, optimizer, and PPO iteration are never copied.

The source must pass `inspect_safe_velocity_checkpoint`, including canonical
filename/internal iteration agreement, SHA-256, raw 63-input topology, action
order, finite tensors, exact guarded action bounds, and bounded exploration
width. The required recipe is always the current
`MICROBAN_SAFE_VELOCITY_RECIPE_REVISION` (currently
`scratch_bounded_inward_shoulder_sagittal_bodyprogress_v9`); its checkpoint
`infos` marker is mandatory, so rejected v1-v3 and earlier canaries cannot be
relabeled by filename alone.

Structural validation is insufficient by itself. The fixed-forward evaluator
must also have produced a passing JSON receipt with schema `2`, gate
`microban_safe_velocity_fixed_forward_v3`, the exact checkpoint path/SHA/iteration,
the canonical 64-environment/200-step/seed-42/`+0.08 m/s` configuration,
canonical thresholds, and per-step exact-command verification. The validator
recomputes all eight checks from finite metrics and those thresholds, then
requires the emitted checks, failed-check list, summary and status to agree and
pass. Contract v9 hashes and pins the raw receipt bytes; changing only a failed
receipt's `status` cannot make it acceptable.

Gate v3 changes only the performance velocity-p05 minimum from `0.020` to
`0.010 m/s`. The `0.020 m` displacement minimum and every hard completion,
fall, finite, target-clip, actual-soft-limit, and actual-lookahead-soft-limit
threshold remain unchanged. The canonical accepted source is v9 model 500
(SHA-256 `416a8b16f7f7980822e4e1df81ffaf9515bc18a246e6fc257405a2c46ceece93`),
with receipt `artifacts/microban_safe_velocity_v9_model500_v3_pass.json`
(SHA-256 `e68701b11774dd30c8e45a2fd89614a2e4423a9486d01a0d936f0fa6fb760492`).

## HOME and control contract

The commanded software HOME pose uses shoulder pitch `0 deg`, right/left
shoulder roll `-10/+10 deg`, and both elbows `-20 deg`. These values define the
software pose and are not a claim about a measured hardware zero.

The full actor input remains:

1. base angular velocity (3)
2. projected gravity (3)
3. all 21 joint positions, including the three HMD-owned joints
4. all 21 joint velocities
5. previous 18 body actions
6. body-frame velocity command (3)
7. left/right foot targets (6)
8. left/right hand targets and active flags (8)

This retains the PICO hold semantics: the left trigger enables the learned
body policy only while held, releasing it returns to HOME; the right trigger
holds neck yaw toward body-forward; the left grip/middle button shows the robot
camera only while held. Sticks remain full translation and yaw commands. These
are runtime/controller semantics, not environment-variable switches.

The existing step-based training curriculum remains active. It progressively
widens translation/yaw commands, introduces moving HMD inertia and foot/hand
targets, and retains resume-safe application at the restored global step.

Fall diagnostics observed 248 simultaneous contacts and 513 constraints, so
v9 fixes MuJoCo-Warp capacity at `nconmax=512` and `njmax=2048`. With that
required capacity, 4,096 environments requested a measured 12.7 GB single EPA
allocation and OOMed on this RTX 4090 host. The canonical v9 wrapper therefore
fixes the run at exactly 2,048 environments; do not use 4,096 for this contract.

The retargeted walk004 locomotion prior is deliberately not part of v9: its
dynamic gate fell in every evaluated rollout and it is less safe than the new
source actor. The 39-value critic slot remains exact zero only to preserve
critic topology. Its command is disabled from reset, both prior reward weights
and all three BC coefficients are exact zero, no BC configuration reaches the
optimizer, and the prior clip termination is absent. Therefore there is no
walk004 reset teleport, launch transition, clip playback, or post-PPO teacher
step capable of damaging the bootstrapped safe actor.

## Start and resume

Compute the immutable source digest, run the fixed-forward evaluator, then pass
the checkpoint, digest, and passing receipt explicitly:

```bash
SAFE_MODEL=/absolute/path/to/model_<accepted-iteration>.pt
SAFE_SHA256="$(sha256sum -- "${SAFE_MODEL}" | awk '{print $1}')"
PASS_RECEIPT=/absolute/path/to/safe_velocity_acceptance.json
uv run --locked python -m \
  mjlab_microban.scripts.evaluate_safe_velocity_checkpoint \
  --checkpoint "${SAFE_MODEL}" \
  --expected-sha256 "${SAFE_SHA256}" \
  --output "${PASS_RECEIPT}"
scripts/train_microban_teleop_v9.sh start \
  "${SAFE_MODEL}" "${SAFE_SHA256}" "${PASS_RECEIPT}"
```

The start command runs only canonical stage `0->1500`. Gate its new run before
continuing:

```bash
scripts/evaluate_microban_teleop_v9_stage.sh <run-directory-name>
```

Resume from that timestamp/run-name directory:

```bash
scripts/train_microban_teleop_v9.sh resume 2026-09-25_12-34-56
```

Each pass receipt unlocks exactly one adjacent boundary:
`1500, 3000, 4500, 6000, 8000, 12000, 14000, 16000, 18000, 20000`.
The wrapper records and verifies parent checkpoint/gate SHA-256 values, keeps an
interrupted stage on its original interval, and makes the final reproducible
stage exactly `18000->20000`. Generic resume is rejected by the runner.

Because the accepted safe-velocity seed is intentionally slow, boundaries before
20,000 use the explicit `canonical_intermediate_hard_safety_v1` profile. They
still require three seeded, complete deterministic rollouts with no fall,
non-finite value, unexpected termination, self-collision, target clipping, or
actual soft-limit violation. Moving-HMD boundaries also still require measured
target and physical neck excursion. Velocity, hand, and foot tracking metrics
are recorded but do not block continuation at these intermediate boundaries.
The final 20,000-update gate switches back to
`deployment_performance_v1`; every original command-direction, velocity, and
body-target performance threshold is mandatory there. A hard-safety-only report
can never produce a deployable ONNX.

Every resume also records a separate `resume_source_checkpoint_path`,
`resume_source_checkpoint_sha256`, and `resume_source_checkpoint_iteration` in
the new stage manifest. These identify the exact `model_N.pt` loaded by this
process; they do not overload the accepted boundary-parent fields. The runner
re-hashes that path and compares all three values before loading optimizer or
actor state, so replacing an interrupted checkpoint under the same filename is
rejected. On an interrupted `0->1500` resume, the wrapper does not ask the user
to resupply mutable safe-source arguments. Instead, the runner copies the three
safe-source fields from the authenticated teleop checkpoint, verifies them
against its independently validated `safe_velocity_actor_bootstrap` record,
and recomputes the new manifest digest before any subsequent save.

Keep the source checkpoint at the recorded absolute path. Every fresh start,
resume, checkpoint save, and ONNX export revalidates the source path, digest,
recipe, topology, observation schema, action order, raw/bounded contract, and
the pinned acceptance-receipt path, digest, gate, checks, and checkpoint binding.
A v8 checkpoint, missing source, changed source, wrong digest, or retired
bootstrap field fails closed.

Each teleop checkpoint records:

- source path, SHA-256, iteration, schema, recipe, topology, observation schema,
  action order, and raw-observation flag;
- acceptance receipt absolute path, SHA-256, schema, and gate;
- the complete 63-to-83 index mapping and the 20 zero-initialized columns;
- the copied bounded-distribution state keys; and
- explicit `false` markers for actor normalizer, critic, optimizer, and
  iteration copying.

ONNX provenance additionally embeds the safe source checkpoint SHA-256 and
iteration, safe recipe, acceptance-receipt SHA-256/schema/gate, and 63-to-83
mapping revision, so an artifact can be audited without guessing its bootstrap
from a run directory.

The separate Microban hardware repository's `src/moves/pico_hybrid.py` has been
updated for contract v9. It independently requires the safe source checkpoint
SHA-256/iteration/recipe, acceptance receipt SHA-256/schema/gate, and mapping
revision in addition to the final teleop acceptance metadata. Missing or v8
metadata therefore fails before the runtime can command motors.

## CPU-only contract checks

These checks do not launch training:

```bash
uv run --python .venv/bin/python --with pytest python -m pytest -q \
  tests/test_teleop_velocity_bootstrap.py
uv run --python .venv/bin/python --with ruff ruff check \
  src/mjlab_microban/tasks/microban_teleop_bootstrap.py \
  src/mjlab_microban/tasks/microban_policy_export.py \
  src/mjlab_microban/tasks/microban_teleop_env_cfg.py \
  src/mjlab_microban/tasks/microban_teleop_provenance.py \
  tests/test_teleop_velocity_bootstrap.py
bash -n scripts/train_microban_teleop_v9.sh
```
