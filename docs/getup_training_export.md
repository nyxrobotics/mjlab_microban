# Microban get-up v3: train and export

Run these commands on the training PC from this repository. Start a **new**
training run; do not resume or initialize from an earlier get-up checkpoint.
Earlier (v2 and before) checkpoints used a different previous-action
observation and, in v2, mistakenly rate-limited the active policy's own
commanded target to 0.5 rad/s (a conflation with the unrelated torque-off to
torque-on recovery slew) -- v3 has no rate limit on the policy's own output at
all; its clipped target is written directly, every tick.
The actor reads angular velocity in the IMU sensor's own axes, matching the
unrotated BMI088 gyroscope values sent to the robot's get-up actor.
The training task penalizes raw commands beyond each joint's own absolute
target clip (its simulation soft limit, not a uniform range), in addition to
clipping the actuator target itself -- this reward is required, not optional:
removing it lets a scalar-Gaussian std/entropy runaway develop, since get-up's
own recovery motion needs several joints at or near their limit for much of
an episode, unlike walking's typically comfortable operating range.

```bash
uv sync --locked
uv run --locked train Mjlab-Getup-Microban --env.scene.num-envs 4096 --agent.logger tensorboard
```

The training episode lasts 20 seconds. The robot's automatic get-up attempt
also allows up to 20 seconds; it hands control back once the upright gravity
condition remains stable for 20 control ticks. There is no earlier progress
deadline.

The task is configured for 15,000 training iterations. Select a completed
checkpoint from the new run directory printed by training, then export it with
both paths explicit (replace `<new-run>` with that directory name):

```bash
uv run --locked python -m mjlab_microban.scripts.export_getup_onnx \
  --checkpoint logs/rsl_rl/mjlab_microban_getup/<new-run>/model_14999.pt \
  --output artifacts/getup_v3.onnx
```

The exporter refuses an existing output unless `--replace` is passed. It loads
the `Mjlab-Getup-Microban` play environment and actor, checks the checkpoint's
`microban_getup_contract=v3` and `microban_getup_angular_velocity_frame=imu_sensor_xyz`,
and the complete `microban_getup_home_pose` (21 joints plus root position and
quaternion) training markers, and checks the exported 60-input/18-output ONNX.
In particular, the last 18 observation normalizer means and standard
deviations must fit the reachable previous-action range. That range includes
each joint's simulation soft limits because the applied target begins at the
measured fallen pose, which can be outside each joint's own policy target
clip. The exporter also screens finite raw actions at upright, inverted, and
sideways initial orientations against the robot runtime's 120-action fault
threshold -- a rough policy-sanity ceiling, not a range-of-motion safety
mechanism (that's separately and unconditionally guaranteed by the per-joint
clip regardless of raw magnitude).
Old checkpoints, including pre-v3 checkpoints trained with the mistaken
target-rate slew limit or an earlier -10-degree hip HOME, lack the current
contract or HOME marker and are rejected before resume or export.

The resulting file contains the v3/per-joint-clip markers, the HOME pose as
JSON, a checkpoint SHA-256, joint order, default pose, and each joint's own
absolute action clip (`action_clip_lower`/`action_clip_upper`, not a uniform
range). Its metadata is for the robot runtime to verify before enabling
automatic recovery. Training and export alone do not prove that the learned
maneuver stands the physical robot up; inspect the new policy in simulation
before copying it to the robot.

After inspection, install the ONNX as `src/agents/getup.onnx` in the robot's
`microban` checkout and restart `microban-pico-runtime.service`. The robot
checkout must contain the v3 runtime changes from this update (including the
per-joint action clip fix -- an earlier robot-side revision still checked for
and applied a stale blanket ±1.57 rad range, which rejects a genuinely
v3-compliant export). The currently installed older ONNX is intentionally
rejected until it is replaced.
