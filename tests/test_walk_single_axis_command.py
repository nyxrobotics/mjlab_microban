"""The walker's training twist: a fifth of the moving commands single-axis and small (WALK_SINGLE_AXIS_*), the
rest mjlab's sampling as before (no rotation-only environments), kept by the train CLI; play, the judgments and
PICO keep their commands."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import mjlab
import pytest
import torch
import tyro
from mjlab.scripts.train import TrainConfig

from mjlab_microban.tasks.mdp import UniformVelocityCommandWithRotationCfg
from mjlab_microban.tasks.microban_teleop_v13_arm_overlay import make_microban_teleop_v13_arm_overlay_env_cfg
from mjlab_microban.tasks.microban_velocity_env_cfg import (
    WALK_COMMAND_RANGES_FINAL,
    WALK_SINGLE_AXIS_FRACTION,
    WALK_SINGLE_AXIS_SHARES,
    WALK_SINGLE_AXIS_SPEEDS,
    SingleAxisVelocityCommand,
    SingleAxisVelocityCommandCfg,
    make_microban_velocity_env_cfg,
)


def test_only_the_walkers_training_draws_single_axis_commands():
    twist = make_microban_velocity_env_cfg().commands["twist"]
    assert type(twist) is SingleAxisVelocityCommandCfg
    assert twist.rel_single_axis_envs == WALK_SINGLE_AXIS_FRACTION == 0.2
    assert twist.single_axis_shares == WALK_SINGLE_AXIS_SHARES == (0.25, 0.5, 0.25)
    assert twist.single_axis_speeds == WALK_SINGLE_AXIS_SPEEDS
    assert (twist.rel_standing_envs, twist.rel_forward_envs, twist.rel_rotation_envs) == (0.1, 0.2, 0.0)
    assert not isinstance(make_microban_velocity_env_cfg(play=True).commands["twist"], SingleAxisVelocityCommandCfg)
    for play in (False, True):
        assert type(make_microban_teleop_v13_arm_overlay_env_cfg(play=play).commands["twist"]) \
            is UniformVelocityCommandWithRotationCfg


def test_the_train_cli_keeps_the_single_axis_command():
    default = TrainConfig.from_task("Mjlab-Velocity-Microban")
    twist = tyro.cli(TrainConfig, args=[], default=default, config=mjlab.TYRO_FLAGS).env.commands["twist"]
    assert type(twist) is SingleAxisVelocityCommandCfg
    assert twist.rel_single_axis_envs == WALK_SINGLE_AXIS_FRACTION


def _command(num_envs: int) -> SingleAxisVelocityCommand:
    cfg = deepcopy(make_microban_velocity_env_cfg().commands["twist"])
    for name, value in WALK_COMMAND_RANGES_FINAL.items():  # after the update-3000 stage
        setattr(cfg.ranges, name, value)
    command = object.__new__(SingleAxisVelocityCommand)
    command.cfg = cfg
    command._env = SimpleNamespace(num_envs=num_envs, device="cpu")
    command.vel_command_b = torch.zeros(num_envs, 3)
    command.vel_command_w = torch.zeros(num_envs, 3)
    command.heading_target = torch.zeros(num_envs)
    for flag in ("is_standing_env", "is_forward_env", "is_heading_env", "is_world_env", "is_rotation_env"):
        setattr(command, flag, torch.zeros(num_envs, dtype=torch.bool))
    return command


def test_a_fifth_of_the_moving_commands_are_single_axis_and_small():
    torch.manual_seed(0)
    n = 40000
    command = _command(n)
    command._resample_command(torch.arange(n))
    twist, standing, forward = command.vel_command_b, command.is_standing_env, command.is_forward_env
    moving = ~standing
    # The box never draws an exact zero and the forward-only commands are flagged.
    single = moving & ~forward & ((twist != 0).sum(-1) == 1)
    assert standing.float().mean().item() == pytest.approx(0.1, abs=0.01)
    assert (single.sum() / moving.sum()).item() == pytest.approx(WALK_SINGLE_AXIS_FRACTION, abs=0.01)
    assert not command.is_rotation_env.any()
    axis = (twist[single] != 0).float().argmax(-1)
    speed = twist[single].abs().sum(-1)
    for i, (share, (low, high)) in enumerate(zip(WALK_SINGLE_AXIS_SHARES, WALK_SINGLE_AXIS_SPEEDS)):
        on = axis == i
        assert (on.sum() / single.sum()).item() == pytest.approx(share, abs=0.02)
        assert low <= speed[on].min().item() and speed[on].max().item() <= high
        assert (twist[single][on, i] > 0).float().mean().item() == pytest.approx(0.5, abs=0.03)
    # The judgment's 0.1 m/s lateral is inside the lateral speeds.
    assert WALK_SINGLE_AXIS_SPEEDS[1][0] < 0.1 < WALK_SINGLE_AXIS_SPEEDS[1][1]
    # The rest keep mjlab's sampling: forward-only on a fifth, the box on the others.
    rest = moving & ~single
    assert (forward[rest].float().mean()).item() == pytest.approx(0.2, abs=0.02)
    box = twist[rest & ~forward]
    assert (box.abs().amax(0) <= torch.tensor([0.7, 0.3, 1.5])).all() and (box != 0).all()
