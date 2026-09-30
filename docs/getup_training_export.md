# Microban get-up v4: train and export

Run these commands on the training PC from this repository. Start stage 1
as a **new** training run; stage 2 resumes only from that stage-1 run, never
from a checkpoint of an earlier contract.

v4 is the action contract every get-up policy that ever stood up was trained
under (see `microban_getup_env_cfg.py`'s module docstring for the evidence):

* absolute target = default pose + raw action, clipped at a flat ±1.57 rad on
  all 18 body joints (not each joint's soft limit);
* the policy observes its own previous **raw** output (not the clipped target);
* no penalty on raw output beyond the clip (the IMU delay comes in only in
  the stage-2 fine-tune below).

At the robot's RL servo gain (P=125) standing is active balance that needs
targets far past the joint angle to produce useful torque, so a trained
policy's raw output routinely reaches hundreds of radians. That is expected:
the clip bounds the physical target regardless.

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

Reference numbers (2026-09-30, 64 envs, fallen starts, 0-3 tick IMU delay
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

Other registered variants of the same contract: `Mjlab-Getup-Microban-V42`
(the 2026-09-25 recipe), `Mjlab-Getup-Microban-Redesign` (stands, but in a
wide braced stance), and `Mjlab-Getup-Microban-Sym` (stage 1 with left/right
mirror data augmentation).

The training episode lasts 20 seconds. The robot's automatic get-up attempt
also allows up to 20 seconds; it hands control back once the upright gravity
condition remains stable for 20 control ticks. There is no earlier progress
deadline.

Select a completed checkpoint from the new run directory printed by
training, then export it with both paths explicit (replace `<new-run>` and
`<N>`):

```bash
uv run --locked python -m mjlab_microban.scripts.export_getup_onnx \
  --checkpoint logs/rsl_rl/mjlab_microban_getup/<new-run>/model_<N>.pt \
  --output artifacts/getup_v4.onnx
```

The exporter refuses an existing output unless `--replace` is passed. It loads
the `Mjlab-Getup-Microban` play environment and actor, checks the checkpoint's
`microban_getup_contract=v4` and `microban_getup_angular_velocity_frame=imu_sensor_xyz`,
and the complete `microban_getup_home_pose` (21 joints plus root position and
quaternion) training markers, checks the flat ±1.57 rad clip, and checks the
exported 60-input/18-output ONNX: finite initializers, a finite positive
observation normalizer, and finite raw actions at upright, inverted and
sideways initial orientations (non-finite output is the robot runtime's only
actor fault).

The resulting file contains the v4 markers
(`microban_getup_previous_action_semantics=raw_policy_output`), the HOME pose
as JSON, a checkpoint SHA-256, joint order, default pose, and the absolute
action clip (`action_clip_lower`/`action_clip_upper`). Its metadata is for the
robot runtime to verify before enabling automatic recovery. Training and
export alone do not prove that the learned maneuver stands the physical robot
up; inspect the new policy in simulation before copying it to the robot.

After inspection, install the ONNX as `src/agents/getup.onnx` in the robot's
`microban` checkout and restart `microban-pico-runtime.service`. The robot
checkout must contain the v4 runtime (raw previous-action feedback, no
raw-magnitude fault). The currently installed older ONNX is intentionally
rejected until it is replaced.
