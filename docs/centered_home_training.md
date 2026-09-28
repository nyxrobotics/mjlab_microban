# Centered HOME training

Use the `centered-home-training` branch of `mjlab_microban` for a fresh set of
policies. Its shared HOME matches the physical A-button `NEUTRAL_POSE` in all
21 joint angles. Both shoulder pitches and knees are 0 degrees; the trunk is
vertical (root pitch 0 degrees). Both hips pitch +1.198384259489 degrees and
both ankles pitch -1.198384259489 degrees. Root height is 0.170554885633559 m.
In the current MuJoCo `robot.xml`, this keeps both soles flat on the ground and
puts the mass-weighted whole-body center of mass at the midpoint of their
contact patches (residual below 1e-9 mm). The physical neutral is adjusted
separately in the robot runtime to match these angles.
Run `.venv/bin/python scripts/check_home_neutral.py` to reproduce the geometric
measurement. The current PICO walking model still uses its old training HOME.
Keep the existing production branch and its checkpoints/artifacts for the
currently deployed robot.
Do not resume any old-HOME checkpoint or relabel an old ONNX file.

### Foot-contact correction (2026-09-28)

Earlier runs on this branch used a collision-name expression that missed the
twelve `left/right_foot_collision_[1-6]` boxes. Those boxes therefore had
`condim=1` instead of the intended frictional `condim=3`. The expression is
corrected in `microban_constants.py`; all twelve boxes now compile with
`condim=3` and priority 1. Checkpoints saved before this correction, including
the `2026-09-27_18-18-25_centered_home_v4_legacy_walk_15000` run and its
`2026-09-27_23-51-48_centered_home_v4_legacy_walk_resume_30000` continuation,
were trained under the earlier contact model. Do not compare or resume them as
if they had used the corrected contact model; start a fresh run for a controlled
HOME comparison.

### How the HOME pose was chosen

The calculation uses `src/mjlab_microban/robot/microban/robot.xml`, including
every body's MuJoCo mass and center of mass and the twelve
`left/right_foot_collision_[1-6]` sole boxes. Root pitch, both shoulder pitches,
and both knee angles are fixed at exactly zero. The two hips share one pitch
angle; each ankle uses its negative so the soles stay horizontal. For each hip
angle, MuJoCo forward kinematics gives a mass-weighted whole-body COM x and
the midpoint of the left/right `foot_collision_1` box centers (the outermost
contact boxes, whose combined footprint has the same x midpoint). Bisection
solves COM x minus support midpoint x = 0. Root z is then set so the lowest
corner of all twelve sole collision boxes reaches ground z = 0.

At hip/ankle pitch zero, the flat-foot root height is 0.170644236955 m; the
solved height is 0.089351322 mm lower. Run
`.venv/bin/python scripts/check_home_neutral.py --solve` to reproduce the
angles and height from the model, then run the script without `--solve` to
measure the configured HOME. The resulting virtual head-height target for
get-up training is 0.296534095899190 m. This is a simulated geometry match;
new policy checkpoints must still be trained on this HOME.

| Policy | Branch and task | Fresh training command (from `mjlab_microban/`) |
| --- | --- | --- |
| Velocity | `centered-home-training`, `Mjlab-Velocity-Microban` | `uv run --locked train Mjlab-Velocity-Microban --env.scene.num-envs 4096 --agent.logger tensorboard` |
| Get-up | `centered-home-training`, `Mjlab-Getup-Microban` | `uv run --locked train Mjlab-Getup-Microban --env.scene.num-envs 4096 --agent.logger tensorboard` |
| Offline motion tracking | `centered-home-training`, `Mjlab-Tracking-Microban` | `uv run --locked train Mjlab-Tracking-Microban --agent.logger tensorboard` |
| Next PICO full-body teleoperation | `pico-v12-centered-home`, `Mjlab-Teleop-Upright-Fullbody-Microban` | Follow that branch's `docs/teleop_upright_fullbody_training.md`; train a fresh checkpoint. |
| Earlier PICO teleoperation v8 | Historical `Mjlab-Teleop-Microban` | Do not use for the new full-body policy. |

The historical v8 PICO task retains its task-local +10-degree shoulder-pitch
override for its old checkpoint contract. The fresh full-body PICO task on
`pico-v12-centered-home` uses the new zero-degree HOME instead.

On each training PC, select this branch on the first checkout and install its
locked dependencies before starting a job:

```bash
# From mjlab_microban/; first checkout on this PC.
git fetch origin && git switch --track origin/centered-home-training
uv sync --locked
```

The committed `uv.lock` makes `uv sync --locked` and each `uv run --locked`
reproduce the same dependency versions on the training PCs. GitHub SSH access
is needed to fetch the pinned `better-actuator-models` dependency. The commands
above start new runs (the default is `--agent.resume False`) and write local
TensorBoard logs under `logs/rsl_rl/`.

The checked-in tracking default is
`data/motions/microban_twist2_centered_home.npz`. It is a retargeted, pinned
TWIST2 example walk for starting an **offline motion-tracking** training run;
it is not a recording of the user's PICO movements. The source is
`assets/example_motions/0807_yanjie_walk_002.pkl` from
`amazon-far/TWIST2` commit `b06178f19a22f2138cbd31f60c6d494bc263f67d`,
recorded by Yanjie Ze. The source and this derived motion are distributed
under the [TWIST2 MIT notice](../data/motions/TWIST2_LICENSE). The old
`microban_twist2.npz` was retargeted around the former HOME and must not be
used. The tracking task checks the converter's `home_pose_revision`, root
position/quaternion, all 21 HOME joint angles, and joint order against this
branch's HOME. Renaming an old NPZ will not make it usable.

The checked-in NPZ contains 667 frames at 50 Hz (13.32 s), with mean/max
retargeting position error of 6.46/8.09 mm and maximum policy-joint speed of
3.559 rad/s. Its SHA-256 is
`da2b0321e2db72684c9372e7d881e2f09d2f9f028a07921f1cedb9c1043d85e4`;
the source PKL SHA-256 is
`bb564e04da601846e7bb89e9c01c1d0e2cfda87ba86c1c9d8f6141537e832f87`.
To regenerate it from the pinned example, use these commands on a PC with both
repositories checked out as siblings:

```bash
# From microban_teleop/ on centered-home-retargeting.
./tools/retargeting/setup_twist2_examples.sh
mkdir -p recordings
../mjlab_microban/.venv/bin/python \
  tools/retargeting/twist2_pkl_to_xr_jsonl.py \
  tools/retargeting/_upstream/TWIST2-b06178f19a22f2138cbd31f60c6d494bc263f67d/assets/example_motions/0807_yanjie_walk_002.pkl \
  recordings/twist2-g1-walk-002.jsonl
./tools/retargeting/run_converter.sh convert \
  recordings/twist2-g1-walk-002.jsonl \
  /tmp/microban_twist2_centered_home.npz --output-fps 50
./tools/retargeting/run_converter.sh inspect \
  /tmp/microban_twist2_centered_home.npz
```

To train tracking against a new PICO recording, generate a separate NPZ with the
updated converter on the
`centered-home-retargeting` branch of the sibling `microban_teleop` repository:

```bash
# From microban_teleop/; replace the input with the capture to train against.
git fetch origin && git switch --track origin/centered-home-retargeting
uv sync --locked
mkdir -p ../mjlab_microban/data/motions
./tools/retargeting/run_converter.sh convert \
  /path/to/capture.jsonl \
  ../mjlab_microban/data/motions/pico_capture_centered_home.npz \
  --output-fps 50
./tools/retargeting/run_converter.sh inspect \
  ../mjlab_microban/data/motions/pico_capture_centered_home.npz
```

Then launch the same tracking task with that file:

```bash
# From mjlab_microban/.
MICROBAN_TRACKING_MOTION_FILE="$PWD/data/motions/pico_capture_centered_home.npz" \
  uv run --locked train Mjlab-Tracking-Microban --agent.logger tensorboard
```

The get-up runner stores all 21 HOME joint angles and root position/quaternion
in each checkpoint. Resume and ONNX export reject a checkpoint whose HOME
differs or is unknown. The deployed get-up action contract stays `v2`; the new
HOME pose is also recorded in ONNX metadata. Complete and inspect the new
training jobs before changing the robot's deployed policy files.
