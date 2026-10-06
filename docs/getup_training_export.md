# Microban get-up v5: train and export

> **HOME from config/home_pose.yaml (branch `home-config`).** The HOME, and
> with it the contract string, is the YAML's: `v5` at the centered HOME (trunk
> vertical), `v6` at the forward-lean HOME (trunk +10°, as on
> `forward-lean-centered-home`), `v5_<tag>` / `v6_<tag>` at any other
> (`src/mjlab_microban/robot/home_contracts.py`). With a pitched trunk
> `upright_standing` peaks at the HOME projected gravity (sin p, 0, −cos p),
> `HEAD_STANDING_HEIGHT` / `HOME_FEET_LATERAL_M` are the FK values (0.2953 m /
> 0.0941 m at +10°), the near-HOME reset turns its yaw about world z, and
> `Mjlab-Getup-Microban` defaults to the 10 %, ±5° near-HOME reset (the
> HOME). Only the current contract's stamp (plus, at the centered HOME, a
> `v4` stamp whose recorded env proves v5) is resumed or exported.

> **2026-10-03 unification (contract v5).** Every Microban policy (walking,
> PICO full-body tracking, get-up) now uses the centered HOME
> (`HOME_FRAME` in `microban_constants.py`) and target = HOME + raw action
> × 1.0 with no software clip. The only bound is the XC330's one-turn goal
> range, modeled in training as an absolute ±π clip
> (`SERVO_TARGET_RANGE_RAD`); the old ±1.57 rad clip capped XC330 torque at
> ~70 %. Everything else is unchanged from v4: the policy observes its own
> **raw** previous output and the neck is held at its measured angles. The
> reference numbers, run names and
> the robot policy cited below are **v4 / old-HOME history** (hip-pitch HOME,
> ±1.57 clip); no v5 policy has been measured against them yet.
>
> Runs started on 2026-10-03 just before the version bump (e.g.
> `2026-10-03_13-40-39_chome_servo_s1`, `..._chome_servo_s1_nh5`) trained
> under v5 but their checkpoints are stamped `microban_getup_contract=v4`.
> The exporter and resume accept a `v4` stamp only
> when the run's recorded `params/env.yaml` shows the ±π clip at the centered
> HOME with raw previous-action feedback, and treat it as v5. Centered-HOME
> runs with the ±1.57 clip (e.g. `..._chome_s1b`) and every old-HOME
> checkpoint are refused.

v5 keeps the action rule every get-up policy that ever stood up was trained
under (v4; see `microban_getup_env_cfg.py`'s module docstring for the
evidence), with only the clip widened to the servo range:

* absolute target = centered HOME + raw action, clipped at a flat ±π rad on
  all 18 body joints (the servo goal range; v4 used ±1.57; never each joint's
  soft limit);
* the policy observes its own previous **raw** output (not the clipped target);
* no penalty on raw output beyond the clip until the policy stands (the IMU
  delay and the calm/effort terms are switched on later in the same run).

At the robot's RL servo gain (P=125) standing is active balance that needs
targets far past the joint angle to produce useful torque, so a trained
policy's raw output routinely reaches hundreds of radians. That is expected:
the clip (the servo range) bounds the physical target regardless.

v3 (per-joint soft-limit clip, applied-target feedback, clip-excess penalty)
never produced a standing policy; v2 and earlier also rate-limited the
policy's own target. Their checkpoints and ONNX files are rejected.

The actor reads angular velocity in the IMU sensor's own axes, matching the
unrotated BMI088 gyroscope values sent to the robot's get-up actor.

## Training: one run with scheduled switches (stage C, 2026-10-07)

All servos, head and neck included, run at the policy gain P125
(`SERVO_KP_POLICY`), as on the robot.  `scripts/retrain_all_for_home.py`
trains get-up as one process:

```bash
uv run --locked train Mjlab-Getup-Microban --env.scene.num-envs 4096 --env.seed 42 \
  --agent.seed 42 --agent.logger tensorboard --agent.max-iterations 16500
```

The switches are a table (`GETUP_SCHEDULE` / `GETUP_STAGES` in
`microban_getup_env_cfg.py`, applied by `tasks/curriculum.py`; each prints a
`Curriculum stage ... at step S (update U)` line):

| Updates | What is active | Why (2026-10 chain, v4 / old-HOME measurements) |
|---|---|---|
| 0-2499 | HOME-stance rewards, pose curriculum (reward-driven), no IMU latency, entropy 0.01 | stands by ~1750 at the latest; the delay from scratch kept standing_bonus at ~0.2 for 1400 iterations |
| 2500-3999 | + 0-3 tick IMU latency (the actor's delay buffers are allocated for 3 ticks and held at 0 before) | stage 2: 47/62 -> 61/62 fallen starts standing under delay |
| 4000-9999 | refine: action std 0.5, fresh Adam moments, learning rate back to 1e-3, entropy 0.001 (runner); calm terms: standing joint velocity -4, roll pose (hip/ankle roll, 8.6 deg std) 60, feet width x3, clip barrier -0.2 | stage 3: tremble 0.77 -> 0.08 rad/s; bang-bang targets under std ~10 otherwise |
| 10000-16499 | effort_push: roll pose with shoulder roll, target-vs-measured effort -2, clip barrier -2.0, +-0.3 m/s kicks every 3-6 s | stages 4-5: shoulder 0.44 -> 0.10 Nm, 0.3 m/s kick falls 1/62; clip excess and target error level off ~4000-5000 after the switch |

The evaluation / play config keeps the 0-3 tick latency and no schedule.
Each switch was a separate resumed run before (std reset by a checkpoint
copy, the pose curriculum re-fired at every restart); a run that crashes is
resumed from its last checkpoint with the same table (the curriculum and the
runner's refine flag are restored from the update counter and the
checkpoint).

The training episode lasts 20 seconds. The robot's automatic get-up attempt
also allows up to 20 seconds. Once the upright gravity condition holds for 20
control ticks, the robot runtime keeps the get-up actor running as the
standing balancer until a walk move that can itself balance takes over (see
the robot repository's `docs/pico_teleop_resilience.md`). There is no earlier
progress deadline.

Select a completed checkpoint from the new run directory printed by
training, then export it with both paths explicit (replace `<new-run>` and
`<N>`):

```bash
uv run --locked python -m mjlab_microban.scripts.export_getup_onnx \
  --checkpoint logs/rsl_rl/mjlab_microban_getup/<new-run>/model_<N>.pt \
  --output artifacts/getup.onnx --gate-report <release>/getup_gate.json
```

The exporter refuses an existing output unless `--replace` is passed. It loads
the `Mjlab-Getup-Microban` play environment and actor, checks the checkpoint's
`microban_getup_angular_velocity_frame=imu_sensor_xyz` and complete
`microban_getup_home_pose` (21 joints plus root position and quaternion)
training markers, decides v5 validity from the run's recorded
`params/env.yaml` (centered HOME, flat ±π clip, unit-scale HOME-relative
action, undelayed raw previous-action feedback; checkpoint stamp `v5`, or
`v4` for the runs described at the top), checks the play env's ±π clip, and
checks the
exported 60-input/18-output ONNX: finite initializers, a finite positive
observation normalizer, and finite raw actions at upright, inverted and
sideways initial orientations (non-finite output is the robot runtime's only
actor fault).

The resulting file carries the robot's contract microban-policy-1
(docs/policies.md): the HOME stamp, the layout, the checkpoint and its passed
gate (`--gate-report`, the pipeline's judgment) and the startup self-test rows
of a seeded rollout in the play env.  Its metadata is for the
robot runtime to verify before enabling automatic recovery. Training and
export alone do not prove that the learned maneuver stands the physical robot
up; inspect the new policy in simulation before copying it to the robot.

After inspection, install the ONNX as `src/agents/getup.onnx` in the robot's
`microban` checkout and restart `microban-pico-runtime.service`. The robot
checkout must contain a runtime that accepts get-up contract v5 (centered
HOME, ±π servo-range clip, raw previous-action feedback, no raw-magnitude
fault). A v4 runtime rejects a v5 file (its clip is wider than ±1.57), and a
v5 runtime rejects the installed v4 ONNX until it is replaced.
