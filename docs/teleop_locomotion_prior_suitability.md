# Locomotion-prior static and dynamic suitability gates

This gate rejects a retargeted walking clip before policy training when its
Microban foot geometry is not plausibly grounded or does not provide enough
swing-foot clearance.  It is a CPU-only geometric audit.  It does not start an
MjLab environment, step physics, reserve a GPU, or connect to the robot.

## Reproduce the tracked-input audit

From the repository root:

```bash
uv run --locked python -m \
  mjlab_microban.scripts.audit_locomotion_prior_suitability \
  --output artifacts/microban_locomotion_prior_suitability.json
```

The command exits `0` only when all static checks pass and exits `2` after
writing a valid failing receipt when any check fails.  It also exits `2` on an
input-digest mismatch or if the output already exists.  Use `--force` only when
intentionally replacing that receipt.

The defaults bind these exact tracked inputs:

- `data/motions/microban_twist2_walk004_locomotion_prior.npz`
  (`100656a04438e5b7c09d80e63f79f3e758650a20d87326f3e3970616e6fb69e2`)
- `src/mjlab_microban/robot/microban/robot.xml`
  (`27a8a5731bc389ade2fa0cab6d3530515175988667cb0ea386cec577cd68ad8f`)
- inclusive motion frames 109 through 267 (159 frames at 50 Hz)

For a regenerated candidate, calculate and pass its digest explicitly.  This
prevents a path change from silently weakening provenance:

```bash
sha256sum data/motions/<candidate>.npz
uv run --locked python -m \
  mjlab_microban.scripts.audit_locomotion_prior_suitability \
  --prior data/motions/<candidate>.npz \
  --expected-prior-sha256 <64-lowercase-hex-digest> \
  --output artifacts/<candidate>_static_suitability.json
```

If `robot.xml` changes, likewise pass the new file and digest with
`--robot-xml` and `--expected-robot-xml-sha256`.  A candidate should not replace
the tracked training input until its receipt passes review.

## Exact calculation

For every frame the auditor writes the recorded trunk position and WXYZ
quaternion plus every recorded joint position into the MuJoCo model and calls
`mujoco.mj_forward`.  For each of the six named collision boxes on each foot it
constructs all eight local corners from the MuJoCo half-sizes and transforms
them with that geom's exact world rotation and translation.  The sole height is
the minimum world Z across all 48 corners for that foot.  Geom centers, visual
meshes and the foot sites are not used as contact-plane approximations.

The per-frame receipt records:

- left and right sole minimum world Z;
- the lower sole minimum and the translation that would put it on Z=0;
- grounding error, defined as the absolute lower-sole world Z before that
  translation;
- both sole heights after this grounding translation;
- relative swing clearance, the absolute difference between the two sole
  minima.

Relative clearance is invariant to a common root-height offset.  The explicit
grounding error separately detects an incorrectly placed floating root.  The
aggregate uses NumPy's deterministic linear percentile method and requires all
of the following:

- maximum grounding error no greater than 0.5 mm;
- maximum relative swing clearance at least 15 mm;
- P95 relative swing clearance at least 10 mm.

The JSON contains every frame metric, the ordered twelve geom names, thresholds,
input sizes and SHA-256 digests, retargeting source metadata, tool versions and
a canonical `receipt_payload_sha256`.  Host-specific absolute path prefixes and
timestamps are deliberately excluded, so repeated runs over identical bytes in
the locked tool environment produce identical receipt bytes.

The producer metadata is mandatory rather than advisory.  The NPZ must contain
lowercase SHA-256 values for the source capture and retarget model, the exact
`microban-bilateral-sagittal-sole-v2` recipe, a positive root-translation scale,
and the full signed source foot-height signal.  The auditor recomputes the
midpoint-of-P05/P95 baseline, 1 mm deadband side labels, bilateral P95 swing
scales and transition count.  It also requires the exact five-tap phase filter,
the three sagittal correction joint suffixes and correction values, and an
explicit zero artificial swing boost.  The recorded retarget-model digest must
equal the exact `robot.xml` digest used by this audit.  These values and the
source-signal byte digest are copied into
`inputs.locomotion_prior.provenance` in the receipt.

The aggregate gate retains the three thresholds above.  The receipt additionally
reports max/P95 clearance separately for each foot when that foot is higher.
This exposes one-sided clips without changing the specified aggregate static
contract; the dynamic gate below requires both feet to become airborne.

## Historical rejected walk002 result

The existing TWIST2-derived prior is expected to fail this new static gate:

| Metric | Observed | Required |
| --- | ---: | ---: |
| Maximum grounding error | 10.410 mm | <= 0.500 mm |
| Maximum relative swing clearance | 8.822 mm | >= 15.000 mm |
| P95 relative swing clearance | 7.429 mm | >= 10.000 mm |

This failure is useful evidence: the clip teaches a near-double-contact,
foot-dragging trajectory and its stored root height places the lower collision
sole below the ground plane.  Regenerate or repair the retargeted motion rather
than relaxing these thresholds merely to obtain a passing receipt.

## GPU dynamics gate

A passing static receipt is necessary, not sufficient.  It proves only exact
kinematic geometry for the recorded poses.  The checked wrapper runs that audit
first and starts CUDA only after validating its passing, tamper-evident receipt:

```bash
scripts/run_microban_locomotion_prior_gate.sh \
  --prior data/motions/<candidate>.npz \
  --expected-prior-sha256 <64-lowercase-hex-digest> \
  --static-output artifacts/<candidate>_static_suitability.json \
  --dynamic-output artifacts/<candidate>_dynamic_gate.json
```

Use `--force` only to replace both named receipts deliberately.  Without
arguments the wrapper uses the canonical paths and digests embedded in the two
Python entry points.  Keep candidate path and digest explicit until its files
and constants have been reviewed together.

The dynamic gate creates 256 training-mode environments on `cuda:0` with seed
42.  It keeps all five startup domain-randomization events and the randomized
BAM actuator (including 3--6 physics-step command delay and voltage/drop
variation), keeps deterministic reset events, removes interval pushes and the
moving-HMD event, and disables auto reset.  Every environment receives the
exact nominal forward command and is teleported to source frame 109.  The
driver sends frame `n+1` as an absolute joint-position target through the real
18-action transform while the six arm targets remain at their configured home
offsets.  Frame 267 is held for the final termination check.  Each environment's
first terminal observation and terms are captured before it is manually reset;
that row is then permanently excluded from metrics while the remaining rows
continue.  Thus one early failure cannot hide the other 255 outcomes.

All of these checks must pass:

- exactly 256/256 environments terminate via the completed frame-109--267 clip;
- zero falls, unexpected terminations, non-finite environments and self contacts;
- minimum root height at least 0.10 m;
- maximum actual soft-limit violation at most `1e-6` rad;
- maximum absolute-target projection at most 0.001 rad;
- all 256 environments lift the left foot and all 256 lift the right foot at
  least once, with zero simultaneous-flight environment-steps;
- P05 of per-environment mean forward velocity at least 0.05 m/s;
- P95 of per-environment mean XY velocity error at most 0.075 m/s.

### Current walk004 result: rejected by dynamics

The canonical walk004 bytes pass the static gate, but the seed-42, 256-environment
dynamic replay fails and must not be treated as a deployable teacher.  The
schema-v2 clean receipt is
`artifacts/microban_locomotion_prior_walk004_dynamic_gate_per_env_v2_clean.json`.
It records zero completed environments: all 256 first terminate as `fell_over`
between zero-based policy steps 35 and 48 (source target frames 145--158).
Minimum root height is 0.08064 m, forward-velocity P05 is -0.13732 m/s, XY-error
P95 is 0.30329 m/s, and there are 311 simultaneous-flight environment-steps.
Target projection, non-finite state and self-collision checks remain zero; this
does not offset the failed stability and motion checks.

The diagnostic trace makes a phase or one-step reset impulse less likely: the
first target is frame 110 as specified, initial tracking-error P95 is only 0.008
rad, and no environment terminates for 35 steps.  Error then accumulates to about
0.8 rad while roll grows past 50 degrees before the first falls; the largest
whole-run per-joint error is 0.840 rad at the left hip pitch.  This is evidence
of an open-loop actuator/reference tracking and lateral-stability failure under
the required BAM/domain-randomized plant.  It is not evidence for relaxing the
acceptance thresholds.

The dynamic receipt also records contact/airborne counts and contact slip for
each foot, per-joint projection, soft-limit and tracking-error maxima, all
first-termination counts and step histograms, BAM runtime ranges, and exact
input/static-receipt digests.  Its per-step trace includes surviving environment
count, left/right contact count, root height, root roll/pitch and per-joint
tracking error so an initial transient, phase problem or actuator shortfall can
be distinguished.  It is an actuator/reference diagnostic, not proof that a
learned policy will reproduce the gait and not authorization to deploy on
hardware.

## CPU regression tests

The geometry, provenance, static prerequisite, acceptance boundaries and both
receipt digests are tested without creating a simulator:

```bash
uv run --locked python -m unittest -v \
  tests.test_locomotion_prior_suitability \
  tests.test_locomotion_prior_dynamic_gate
```
