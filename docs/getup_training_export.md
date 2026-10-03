# Microban get-up v5: train and export

> **2026-10-03 unification (contract v5).** Every Microban policy (walking,
> PICO full-body tracking, get-up) now uses the centered HOME
> (`HOME_FRAME` in `microban_constants.py`) and target = HOME + raw action
> × 1.0 with no software clip. The only bound is the XC330's one-turn goal
> range, modeled in training as an absolute ±π clip
> (`SERVO_TARGET_RANGE_RAD`); the old ±1.57 rad clip capped XC330 torque at
> ~70 %. Everything else is unchanged from v4: the policy observes its own
> **raw** previous output and the neck is held at its measured angles. The
> stage commands below are unchanged. The reference numbers, run names and
> the robot policy cited below are **v4 / old-HOME history** (hip-pitch HOME,
> ±1.57 clip); no v5 policy has been measured against them yet.
>
> Runs started on 2026-10-03 just before the version bump (e.g.
> `2026-10-03_13-40-39_chome_servo_s1`, `..._chome_servo_s1_nh5`) trained
> under v5 but their checkpoints are stamped `microban_getup_contract=v4`.
> The exporter, resume and `reset_getup_action_std` accept a `v4` stamp only
> when the run's recorded `params/env.yaml` shows the ±π clip at the centered
> HOME with raw previous-action feedback, and treat it as v5. Centered-HOME
> runs with the ±1.57 clip (e.g. `..._chome_s1b`) and every old-HOME
> checkpoint are refused.

Run these commands on the training PC from this repository. Start stage 1
as a **new** training run; stage 2 resumes only from that stage-1 run, never
from a checkpoint of an earlier contract.

v5 keeps the action rule every get-up policy that ever stood up was trained
under (v4; see `microban_getup_env_cfg.py`'s module docstring for the
evidence), with only the clip widened to the servo range:

* absolute target = centered HOME + raw action, clipped at a flat ±π rad on
  all 18 body joints (the servo goal range; v4 used ±1.57; never each joint's
  soft limit);
* the policy observes its own previous **raw** output (not the clipped target);
* no penalty on raw output beyond the clip (the IMU delay comes in only in
  the stage-2 fine-tune below).

At the robot's RL servo gain (P=125) standing is active balance that needs
targets far past the joint angle to produce useful torque, so a trained
policy's raw output routinely reaches hundreds of radians. That is expected:
the clip (the servo range) bounds the physical target regardless.

v3 (per-joint soft-limit clip, applied-target feedback, clip-excess penalty)
never produced a standing policy; v2 and earlier also rate-limited the
policy's own target. Their checkpoints and ONNX files are rejected.

The actor reads angular velocity in the IMU sensor's own axes, matching the
unrotated BMI088 gyroscope values sent to the robot's get-up actor.

Train in two stages (a fresh stage-1 run, then resume it for stage 2):

```bash
uv sync --locked
# Stage 1: HOME-stance rewards, from scratch. Stands with feet together and
# straight legs by ~2000 iterations.
uv run --locked train Mjlab-Getup-Microban --env.scene.num-envs 4096 --agent.logger tensorboard \
  --agent.max-iterations 2000
# Stage 2: same rewards under the walking task's 0-3 tick simulated IMU
# latency, resumed from stage 1 (~500 iterations suffice).
uv run --locked train Mjlab-Getup-Microban-ImuDelay --env.scene.num-envs 4096 --agent.logger tensorboard \
  --agent.max-iterations 1000 --agent.resume True \
  --agent.load-run <stage-1-run> --agent.load-checkpoint model_2000.pt
```

Reference numbers (v4 / old-HOME history, 2026-09-30, 64 envs, fallen starts, 0-3 tick IMU delay
plus sensor noise): stage 1 at 2000 stood 26/62 under delay (52/53 without);
stage 2 at 2500 stood 61/62 (median 2.7 s to stand), feet 10.0 cm apart
(HOME 9.4 cm) with 0.3 cm stagger, every leg joint within 3.3 deg of HOME,
and no falls after 0.2 m/s fore-aft kicks while standing.
Stage 1 reproduced with `--agent.seed 7`: 53/53 fallen starts standing at
1500 (no delay), feet 9.8 cm apart with 0.2 cm stagger. The robot candidate
is stage 2 at 3500 (61/62 and 54/54 under 0-3 tick delay plus noise, 61/62
even under 0-5 ticks).
Training the delay from scratch was much slower (standing_bonus ~0.2 at
iteration 1400), hence the two stages.

### Stage 3: stop the standing tremble (calm fine-tune)

The stage-2 policy stands, but holds the stance with bang-bang targets
(~80 % on the clip) and trembles at ~0.77 rad/s. Its action std has grown to
~10, and under that much noise bang-bang is the robust way to stand. Reset
the std to 0.5 and fine-tune with the calm reward set (standing-gated joint
velocity penalty, tight roll-joint pose, feet width, a clip-excess barrier):

```bash
uv run --locked python -m mjlab_microban.scripts.reset_getup_action_std \
  --checkpoint logs/rsl_rl/mjlab_microban_getup/<stage-2-run>/model_<N>.pt \
  --out-run <stage-2-run>_std05
uv run --locked train Mjlab-Getup-Microban-CalmRoll-ImuDelay --env.scene.num-envs 4096 \
  --agent.logger tensorboard --agent.max-iterations 8000 --agent.algorithm.entropy-coef 0.001 \
  --agent.resume True --agent.load-run <stage-2-run>_std05 --agent.load-checkpoint model_<N>.pt
```

Reproduced (v4 / old-HOME history) on 2026-10-01/02 from stage 2 at 3500 (seed 42): tremble 0.77 ->
0.55 -> 0.29 -> 0.08 rad/s after 2500 / 4500 / 5500 iterations; at 11499,
0.06-0.07 rad/s, fallen starts stood 60/62, 53/54, 57/60, 54/55 (four seeds)
and 59/62 under 0-5 tick delay, feet 9.4 cm apart, tilt ~4 deg, falls after a
0.3 m/s fore-aft kick 2/62. Standing effort drops ~7x (0.9 Nm over 18 joints).


### Stages 4 and 5: release the shoulder, then train push recovery

The stage-3 policy presses its right arm into the 0-deg shoulder_roll stop
at ~0.44 Nm while standing. Its raw output keeps the target on the clip, so
neither a pose term nor an effort term can move it. Stage 4
(`Mjlab-Getup-Microban-CalmEffortStrong-ImuDelay`) adds shoulder_roll to the
roll pose term, penalizes |target - measured| while standing, and raises the
clip-excess barrier 10x. That effort penalty also cost push tolerance, so
stage 5 (`Mjlab-Getup-Microban-CalmPush-ImuDelay`) adds +-0.3 m/s kicks every
3-6 s. Resume each stage from the previous one (keep `--agent.algorithm.entropy-coef
0.001`; no further std reset):

```bash
uv run --locked train Mjlab-Getup-Microban-CalmEffortStrong-ImuDelay ... \
  --agent.max-iterations 6500 --agent.resume True --agent.load-run <stage-3-run> ...
uv run --locked train Mjlab-Getup-Microban-CalmPush-ImuDelay ... \
  --agent.max-iterations 3000 --agent.resume True --agent.load-run <stage-4-run> ...
```

v4 / old-HOME history: the robot's v4 policy (`src/agents/getup.onnx`,
2026-10-02) came from exactly this chain: stage 1 `2026-09-30_11-03-25_v4_S1_posture_scratch`
(2000) -> stage 2 `2026-09-30_12-20-47_v4_S1D_posture_delay_ft` (3500, std
reset) -> stage 3 `2026-10-01_18-46-39_v4_stage3_calmroll_repro_s42` (11499)
-> stage 4 `..._v4_stage4_effort_from_repro` / `_cont` / `_cont2` (18000)
-> stage 5 `2026-10-02_05-56-18_v4_calm_push` (model_20999,
`artifacts/getup_v4_calm_push_20999.onnx`). Measured (64 envs, 0-3 tick IMU
delay plus noise): fallen starts stood 60/62, 52/54, 57/60, 52/55 (four
seeds) and 59/62 under 0-5 ticks; standing tremble 0.05-0.06 rad/s, effort
0.55 Nm over 18 joints (right shoulder 0.10 Nm); falls after a 0.3 m/s
fore-aft kick 1/62, after 0.4 m/s fore-aft / lateral kicks 19/59 / 19/57.
Stages 4-5 were reproduced with `--agent.seed 7` from the same stage-3
checkpoint, in one uninterrupted run each (6500 + 3000 iterations): fallen
starts 61/62, 53/54, 56/60, 54/55 and 60/62 under 0-5 ticks; tremble 0.06
rad/s; effort 0.46 Nm (no joint above 0.08 Nm, the shoulder stop no longer
pressed); falls after a 0.3 m/s fore-aft kick 1/62, after 0.4 m/s fore-aft /
lateral kicks 23/62 / 17/59.

Other registered variants of the same contract: `Mjlab-Getup-Microban-V42`
(the 2026-09-25 recipe), `Mjlab-Getup-Microban-Redesign` (stands, but in a
wide braced stance), and `Mjlab-Getup-Microban-Sym` (stage 1 with left/right
mirror data augmentation).

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
  --output artifacts/getup_v5.onnx
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

The resulting file contains the v5 markers (`microban_getup_contract=v5`,
`microban_getup_previous_action_semantics=raw_policy_output`), the stamp the
checkpoint carried (`microban_getup_checkpoint_contract_stamp`), the HOME
pose as JSON, a checkpoint SHA-256, joint order, default pose, and the
absolute action clip (`action_clip_lower`/`action_clip_upper`, exactly
-π/+π written at full precision). Its metadata is for the
robot runtime to verify before enabling automatic recovery. Training and
export alone do not prove that the learned maneuver stands the physical robot
up; inspect the new policy in simulation before copying it to the robot.

After inspection, install the ONNX as `src/agents/getup.onnx` in the robot's
`microban` checkout and restart `microban-pico-runtime.service`. The robot
checkout must contain a runtime that accepts get-up contract v5 (centered
HOME, ±π servo-range clip, raw previous-action feedback, no raw-magnitude
fault). A v4 runtime rejects a v5 file (its clip is wider than ±1.57), and a
v5 runtime rejects the installed v4 ONNX until it is replaced.
