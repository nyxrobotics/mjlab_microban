# MjLab Microban

<img width="35%" align="right" alt="image" src="https://github.com/user-attachments/assets/38848ed5-ef34-44f3-ad44-f2137ac0347b" />

[![License: Apache-2.0](https://img.shields.io/badge/Software-Apache--2.0-yellow.svg)](LICENSE)

This repository contains Reinforcement Learning (RL) environments for Microban, a compact, low-cost, fully open-source small humanoid robot. 

If you are interested in learning more about Microban, or even building your own, check out the [Microban repository](https://github.com/MarcDcls/microban).

The environments are built using the [MjLab](https://github.com/mujocolab/mjlab) framework.
A velocity control task is currently implemented, allowing the robot to follow target linear and angular velocities while resisting external disturbances.

<br>
<br>

## Install

To install the repository, you need the uv package manager.
If you don't have it yet, you can install it by following the instructions [here](https://docs.astral.sh/uv/getting-started/installation/#installation-methods).

Then, clone this repository and run the following command in your terminal:
```
uv sync --locked
```

The tracked `uv.lock` is part of the training provenance. Do not regenerate it
inside a canonical run; review and commit dependency updates separately.

## Transferring to the real robot

The transfer on the real robot is always a challenge due to the sim-to-real gap. However, the policies trained in this repository have been successfully transferred to the real Microban robot. It is possible due to a combination of domain randomization and a well-tuned modelisation of the actuators (delays, friction, voltage drop, current clipping, etc.). This modelisation is done using the [BAM](https://github.com/Rhoban/bam) library.

Here is a video of the trained agent being transferred to the real robot: [https://youtu.be/1pnFrT_jfXQ](https://youtu.be/1pnFrT_jfXQ)

<p align="center">
  <img width="70%" alt="image" src="https://github.com/user-attachments/assets/dd91b082-faf0-4c73-a216-fe9b633f51b3" />
</p>

## HOME pose

Every policy (walking, get-up, PICO teleop) is trained and deployed at one
HOME pose, defined only in [`config/home_pose.yaml`](config/README.md).
Root height, gravity at HOME, get-up targets, hand FK bounds and the HOME
identity strings are derived from it by MuJoCo FK
(`src/mjlab_microban/robot/home_pose.py`). Changing it means retraining every
policy at it: `scripts/retrain_all_for_home.py` (below) retrains, judges and
exports them and installs them, with the robot's copy of the YAML
(`config/home_pose_tool.py write-robot`), into a robot worktree whose code
reads `config/home_pose.yaml` (see `config/README.md`).

## Training your own agent

You can modify the environment configuration at `src/mjlab_microban/tasks/microban_velocity_env_cfg.py`.

To test the environment before training, play with a zero or random agent:

```
uv run play Mjlab-Velocity-Microban --agent zero
uv run play Mjlab-Velocity-Microban --agent random
```

Start the training with:

```
uv run train Mjlab-Velocity-Microban --env.scene.num-envs 4096
```

The walking, get-up and PICO teleop policies of a HOME are retrained from
scratch, gated, exported and installed into the robot repository by one
command, `scripts/retrain_all_for_home.py` (see
[`docs/home_pose_workflow.md`](docs/home_pose_workflow.md); the PICO package
is described in [`docs/teleop_v12_deployment.md`](docs/teleop_v12_deployment.md)).

Once training is complete, play back a checkpoint with:

```
uv run play Mjlab-Velocity-Microban --checkpoint-file [path to your checkpoint]
```

Where `[path to your checkpoint]` is typically located at `logs/rsl_rl/mjlab_microban_velocity/[date]/model_[number].pt`.

<p align="center">
  <img width="480" alt="MicrobanSimu" src="https://github.com/user-attachments/assets/fa79d712-e2ff-4452-b3ef-7ac41b87ff13" />
</p>

Random velocity commands are given to the robot at regular intervals.
Linear velocity commands are represented by a blue arrow, while angular velocity commands are represented by a green vertical one.

To push the robot while playing, double-click on the trunk in the simulation window, then hold the left-ctrl key and right-click and drag to apply a force.

You can also play back the last checkpoint in wandb with:

```
uv run play Mjlab-Velocity-Microban --wandb-run-path [path to your wandb run]
```

Where `[path to your wandb run]` is available in the Overview tab of your wandb run.

## Exporting a policy to ONNX

A ONNX is generated during training with the latest checkpoint, but if you want to export a specific checkpoint, you can do so with the following command:

```
uv run python -m mjlab_microban.scripts.export_walk_onnx --checkpoint [path to your checkpoint] --gate-report [release]/walk_gate.json --output walk.onnx
```

The robot's `walk.onnx` carries the passed judgment of its checkpoint: `[release]/walk_gate.json` is the gate report `scripts/retrain_all_for_home.py` writes into its release directory (`<state dir>/release/`), where it also exports the get-up and PICO policies.

## License

Copyright (c) 2026 Marc Duclusaud

This software is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.
