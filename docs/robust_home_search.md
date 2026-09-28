# Static HOME diagnostic

`optimize_static_home.py` measures how a MuJoCo robot holds symmetric hip and
ankle pitch HOME candidates under repeatable joint and body perturbations. It
uses a fixed 21-joint position target, not the walking policy. The script does
not change the physical robot.

## Current model validation result (2026-09-28)

The initial simulation accidentally left the twelve sole collision boxes at
`condim=1` because `FULL_COLLISION` matched `*_foot_collision` but the actual
names end in `_1` through `_6`. The collision configuration now uses
`^(left|right)_foot_collision_[1-6]$`. The compiled MuJoCo model was checked:
all twelve sole boxes have `condim=3`, `priority=1`, and friction
`(1.0, 0.005, 0.0001)`.

The original trial setup also varied startup battery voltage and voltage-drop
gain independently between environments, then seeded only at reset. The
diagnostic now seeds before environment construction and fixes every candidate
to the same nominal 10.8 V supply, 0.1 V/Nm voltage-drop gain, and four
physics-step command delay. These are comparison settings, not measured
hardware values. The zero-perturbation controls below each used trial seed 0,
five seconds, no initial joint error, no target waveform, and no push. A fall
means root height below 0.10 m or projected gravity z above −0.5.
P=125 is the walking stand gain in this model; P=900 approximates the A-button
HOME hold register.

| HOME (hip, ankle, shoulder pitch) | P=125 | P=900 |
| --- | --- | --- |
| Old physical A HOME (−10°, 0°, +10°) | Fell at 1.56 s | Fell at 2.36 s |
| Centered training HOME (+1.198384°, −1.198384°, 0°) | Fell at 3.66 s | Survived 5 s |

A second run reproduced all four fall/survival outcomes and their reported
times. Other continuous metrics varied slightly between GPU runs; the table
does not imply exact bitwise deterministic simulation.

The old HOME was observed to work better for walking on the robot. These
static results cannot establish a better walking HOME: the controller is
different, and whether the physical robot can stand in the old HOME with only
the A-button hold has not been measured. The P=900 result is especially
uncertain because the BAM actuator model was identified around P=125. **No
candidate has been selected from this diagnostic, and the deployed HOME has
not been changed.** First check whether the physical robot can hold the old
HOME by itself, then compare contact and actuator behavior with the model.

The contact geometry is another possible contributor. At the old HOME's
computed root height (0.174243 m), only `left_foot_collision_1` and
`right_foot_collision_1` initially touch the plane; the other five boxes per
foot are approximately 0.198, 0.397, 0.597, 0.797, and 0.997 mm higher. The
sole is effectively tilted about 10°. At the centered HOME's computed root
height (0.170555 m), all twelve sole box bottoms are at the same height.
This geometry difference is evidence about the simulated initial contact,
not proof of the physical foot's load distribution. Shoulder pitch also
differs between these two historical HOME definitions, so the control pair
does not isolate hip and ankle effects.

The local diagnostic records are in
`artifacts/home_search/static_zero_fixed_actuator_old.json` and
`artifacts/home_search/static_zero_fixed_actuator_centered.json` (ignored by
Git). Files with `static_zero_friction_fix` in their names used randomized
startup actuator parameters. Older `static_zero_perturb.json` and
`static_centered_zero.json` also predate the foot friction correction. Neither
older set is comparable to the corrected, fixed-actuator controls.

## Reproduce the controls

Use the project environment (`uv run --locked`), or put `src` on `PYTHONPATH`
if reusing another worktree's virtual environment. For the corrected old HOME:

```bash
uv run --locked python -m mjlab_microban.scripts.optimize_static_home \
  --hip-values=-10 --ankle-values=0 --shoulder-deg=10 \
  --kp-values=125,900 --push-levels=0 --seeds=0 \
  --duration-s=5 --target-wave-deg=0 --initial-joint-error-deg=0 \
  --output-prefix artifacts/home_search/static_zero_fixed_actuator_old
```

For the centered HOME, set `--hip-values=1.198384259489`,
`--ankle-values=-1.198384259489`, `--shoulder-deg=0`, and change the output
prefix. Both runs should be repeated after any material contact or actuator
model change before interpreting a perturbation sweep.

## How the diagnostic sweep works

The candidate hips and ankles are left/right symmetric. Knee angles and the
trunk's initial pitch remain zero; shoulder pitch is a separate CLI setting.
MuJoCo geometry sets each candidate's root height so the lowest corner of its
twelve sole collision boxes reaches the floor. The same seed-specific error
is added to all 21 initial joint angles for every candidate. Each joint then
receives a seed-specific, low-frequency target waveform that ramps in over
0.5 seconds. A seed-specific body velocity impulse is applied at the
requested push time and level. All candidates run in parallel with the same
disturbance realization. Automatic environment reset and MjLab terminations
are disabled so early falls cannot be hidden by a fresh reset.

The JSON report includes trial and actuator settings, the robot XML SHA-256,
and the robot collision/configuration source SHA-256. Trial and summary CSVs report falls,
survival, censored recovery time, tilt, joint target error, and actuator
effort. The summary sorts candidates by those simulation metrics only. The
rank is **not** an estimate of the best HOME for walking or for the physical
robot while the static controls remain unvalidated.
