# Microban get-up: train and export

The HOME, and with it the contract string, is the one of
`config/home_pose.yaml`: `v6` at the forward-lean HOME (trunk +10°),
`v5_<tag>` / `v6_<tag>` at any other HOME with a vertical / pitched trunk
(`src/mjlab_microban/robot/home_contracts.py`). With a pitched trunk
`upright_standing` peaks at the HOME projected gravity (sin p, 0, −cos p),
`HEAD_STANDING_HEIGHT` / `HOME_FEET_LATERAL_M` are the FK values (0.2953 m /
0.0941 m at +10°), and the near-HOME reset turns its yaw about world z.
A checkpoint stamped with another contract or HOME is refused.

The action rule is the one every Microban policy shares:

* absolute target = HOME + raw action, clipped at a flat ±π rad on all 18
  body joints (the XC330's one-turn goal range, `SERVO_TARGET_RANGE_RAD`;
  never each joint's soft limit);
* the policy observes its own previous **raw** output (not the clipped target);
* the head and both neck joints are held at their measured angles;
* no penalty on raw output beyond the clip until the policy stands (the IMU
  delay and the calm/effort terms are switched on later in the same run).

At the robot's RL servo gain (P=125) standing is active balance that needs
targets far past the joint angle to produce useful torque, so a trained
policy's raw output routinely reaches hundreds of radians. That is expected:
the clip (the servo range) bounds the physical target regardless.

The actor reads angular velocity in the IMU sensor's own axes, matching the
unrotated BMI088 gyroscope values sent to the robot's get-up actor.

## Training: one run with scheduled switches

All servos, head and neck included, run at the policy gain P125
(`SERVO_KP_POLICY`), as on the robot.  `scripts/retrain_all_for_home.py`
trains get-up as one process:

```bash
uv run --locked train Mjlab-Getup-Microban --env.scene.num-envs 4096 --env.seed 42 \
  --agent.seed 42 --agent.logger tensorboard --agent.max-iterations 18000
```

The switches are a table (`GETUP_SCHEDULE` / `GETUP_STAGES` in
`microban_getup_env_cfg.py`, applied by `tasks/curriculum.py`; each prints a
`Curriculum stage ... at step S (update U)` line):

| Updates | What is active | Why |
|---|---|---|
| 0-2499 | HOME-stance rewards, pose curriculum (reward-driven), no IMU latency, entropy 0.01 | learns to stand first; with the delay from the first update it learns to stand far later |
| 2500-3999 | + 0-3 tick IMU latency (the actor's delay buffers are allocated for 3 ticks and held at 0 before) | stands under the IMU latency of the evaluation and the robot |
| 4000-9999 | refine: action std 0.5, fresh Adam moments, learning rate back to 1e-3, entropy 0.001 (runner); calm terms: standing joint velocity -4, roll pose (hip/ankle roll, 8.6 deg std) 60, feet width x3, clip barrier -0.2 | calms the standing tremble; otherwise the stance is held by bang-bang targets under a large action std |
| 10000-14999 | effort: roll pose with shoulder roll, target-vs-measured effort -2, clip barrier -2.0 (no pushes) | the arms let go of the shoulder_roll stops and the clip excess and target error level off within the stage |
| 15000-17999 | push: +-0.3 m/s kicks every 3-6 s | the effort terms cost push tolerance; with pushes from the effort switch on, one arm stays pressed into its stop |

The evaluation / play config keeps the 0-3 tick latency and no schedule.
A run that stops before its last update is trained again from scratch
(`scripts/retrain_all_for_home.py`).

The training episode lasts 20 seconds. The robot's automatic get-up attempt
also allows up to 20 seconds. Once the upright gravity condition holds for 20
control ticks, the robot runtime keeps the get-up actor running as the
standing balancer until a walk move that can itself balance takes over.

## Export

`scripts/retrain_all_for_home.py` exports the judged checkpoint; by hand,
with both paths explicit (replace `<run>` and `<N>`):

```bash
uv run --locked python -m mjlab_microban.scripts.export_getup_onnx \
  --checkpoint logs/rsl_rl/mjlab_microban_getup/<run>/model_<N>.pt \
  --output artifacts/getup.onnx --gate-report <release>/getup_gate.json
```

The exporter refuses an existing output unless `--replace` is passed. It loads
the `Mjlab-Getup-Microban` play environment and actor, checks the checkpoint's
`microban_getup_angular_velocity_frame=imu_sensor_xyz` and complete
`microban_getup_home_pose` (21 joints plus root position and quaternion)
training markers, its contract stamp and the run's recorded
`params/env.yaml` (current HOME, flat ±π clip, unit-scale HOME-relative
action, undelayed raw previous-action feedback), checks the play env's ±π
clip, and checks the exported 60-input/18-output ONNX: finite initializers, a
finite positive observation normalizer, and finite raw actions at upright,
inverted and sideways initial orientations (non-finite output is the robot
runtime's only actor fault).

The resulting file carries the robot's contract microban-policy-1
(docs/policies.md): the HOME stamp, the layout, the checkpoint and its passed
gate (`--gate-report`, the pipeline's judgment) and the startup self-test rows
of a seeded rollout in the play env.  The robot verifies them before enabling
automatic recovery.
