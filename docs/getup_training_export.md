# Microban get-up v2: train and export

Run these commands on the training PC from this repository. Start a **new**
training run; do not resume or initialize from an earlier get-up checkpoint.
Earlier checkpoints used a different previous-action observation and did not
model the robot's 0.5 rad/s target slew.
The actor reads angular velocity in the IMU sensor's own axes, matching the
unrotated BMI088 gyroscope values sent to the robot's get-up actor.
The new training task penalizes raw commands beyond the ±1.57 rad absolute
target clip, in addition to clipping the actuator target itself.

```bash
uv sync --locked
uv run --locked train Mjlab-Getup-Microban --env.scene.num-envs 4096
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
  --output artifacts/getup_v2.onnx
```

The exporter refuses an existing output unless `--replace` is passed. It loads
the `Mjlab-Getup-Microban` play environment and actor, checks the checkpoint's
`microban_getup_contract=v2`, `microban_getup_target_slew_rad_s=0.5`, and
`microban_getup_angular_velocity_frame=imu_sensor_xyz`
training markers, and checks the exported 60-input/18-output ONNX. In
particular, the last 18 observation normalizer means and standard deviations
must fit the reachable previous-action range. That range includes each joint's
simulation soft limits because the post-slew target begins at the measured
fallen pose, which can be outside the narrower ±1.57 rad policy target clip.
The exporter also screens finite raw actions at upright, inverted, and sideways
initial orientations against the robot runtime's 120-action fault threshold.
Old checkpoints lack the v2 training markers and are rejected before export.

The resulting file contains the same v2/0.5 markers, a checkpoint SHA-256,
joint order, default pose, and action clips. Its metadata is for the robot
runtime to verify before enabling automatic recovery. Training and export alone
do not prove that the learned maneuver stands the physical robot up; inspect
the new policy in simulation before copying it to the robot.

After inspection, install the ONNX as `src/agents/getup.onnx` in the robot's
`microban` checkout and restart `microban-pico-runtime.service`. The robot
checkout must contain the v2 runtime changes from this update. The currently
installed older ONNX is intentionally rejected until it is replaced.
