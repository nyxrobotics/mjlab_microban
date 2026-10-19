"""The left-right mirror of the walker's observations and actions, and its mirror loss.

The mirror is checked on the walking env itself: a state and its reflection
across the world x-z plane, written into two environments, must observe M o
and M' o (and the reflected previous action and command must be the mirrored
ones)."""

from __future__ import annotations

import pytest
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from mjlab_microban.policy_contract import ACTION_JOINT_NAMES, OBSERVATION_WIDTHS
from mjlab_microban.tasks.microban_velocity_env_cfg import (
    MicrobanVelocityRlCfg,
    make_microban_velocity_env_cfg,
)
from mjlab_microban.tasks.microban_velocity_mirror import (
    WALK_MIRROR_ACTION,
    WALK_MIRROR_OBSERVATION,
    WALK_OBSERVATION_JOINTS,
    WALK_OBSERVATION_LAYOUTS,
    apply_mirror,
    check_walk_mirror_layout,
    mirror_actions,
    mirror_augmentation,
    mirror_joint_sign,
    mirror_observations,
    mirror_partner,
)

WIDTHS = {group: sum(width for _, width in layout) for group, layout in WALK_OBSERVATION_LAYOUTS.items()}


def _obs(batch: int, generator: torch.Generator) -> TensorDict:
    return TensorDict(
        {group: torch.randn(batch, width, generator=generator) for group, width in WIDTHS.items()},
        batch_size=[batch],
    )


def test_the_mirror_is_a_signed_permutation_that_undoes_itself():
    assert WIDTHS["actor"] == OBSERVATION_WIDTHS["walk"] == 63
    for name, (index, sign) in {**WALK_MIRROR_OBSERVATION, "action": WALK_MIRROR_ACTION}.items():
        assert sorted(index.tolist()) == list(range(index.numel())), name
        assert torch.equal(sign.abs(), torch.ones_like(sign)), name
    obs = _obs(7, torch.Generator().manual_seed(1))
    twice = mirror_observations(mirror_observations(obs))
    for group in WIDTHS:
        assert torch.equal(twice[group], obs[group]), group
    actions = torch.randn(7, len(ACTION_JOINT_NAMES))
    assert torch.equal(mirror_actions(mirror_actions(actions)), actions)
    both, both_actions = mirror_augmentation(obs=obs, actions=actions)
    assert both.batch_size[0] == 14 and torch.equal(both["actor"][7:], mirror_observations(obs)["actor"])
    assert torch.equal(both_actions[7:], mirror_actions(actions))


def _reflect_root(state: torch.Tensor) -> torch.Tensor:
    """(pos, quat wxyz, lin vel, ang vel) in the world, reflected across the world x-z plane."""

    polar = torch.tensor([1.0, -1.0, 1.0])
    quat = torch.tensor([1.0, -1.0, 1.0, -1.0])
    return torch.cat([state[:3] * polar, state[3:7] * quat, state[7:10] * polar, state[10:13] * -polar])


def _reflect_joints(values: torch.Tensor, names: tuple[str, ...]) -> torch.Tensor:
    return torch.stack([mirror_joint_sign(name) * values[names.index(mirror_partner(name))] for name in names])


@pytest.fixture(scope="module")
def env():
    from mjlab.envs import ManagerBasedRlEnv

    cfg = make_microban_velocity_env_cfg(play=True)
    cfg.scene.num_envs = 2
    cfg.seed = 0
    cfg.events = {}
    for group in cfg.observations.values():
        for term in group.terms.values():
            term.delay_min_lag = term.delay_max_lag = 0
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    env.reset()
    yield env
    env.close()


def test_the_walking_env_matches_the_mirror_layout(env):
    check_walk_mirror_layout(env)


def test_a_reflected_state_observes_the_mirrored_observations(env):
    """Env 0 in a random state in the air (feet at different heights), env 1 in its reflection."""

    generator = torch.Generator().manual_seed(3)
    robot = env.scene["robot"]
    names = tuple(robot.joint_names)
    assert names == WALK_OBSERVATION_JOINTS["critic"]
    home = robot.data.default_joint_pos[0].cpu()
    yaw = 0.7
    root = torch.cat([
        torch.tensor([0.3, 0.2, 0.45]),
        torch.tensor([torch.cos(torch.tensor(yaw / 2)), 0.0, 0.0, torch.sin(torch.tensor(yaw / 2))]),
        torch.zeros(6),
    ])
    tilt = torch.tensor([0.98, 0.1, -0.15, 0.05])  # a tilted trunk, composed with the yaw
    w1, v1 = root[3], root[4:7]
    w2, v2 = tilt[0], tilt[1:]
    root[3:7] = torch.cat([(w1 * w2 - v1 @ v2).reshape(1), w1 * v2 + w2 * v1 + torch.linalg.cross(v1, v2)])
    root[3:7] /= root[3:7].norm()
    root[7:] = torch.randn(6, generator=generator) * 0.5
    joint_pos = home + torch.rand(len(names), generator=generator) * 0.6 - 0.3
    joint_vel = torch.randn(len(names), generator=generator)
    roots = torch.stack([root, _reflect_root(root)])
    positions = torch.stack([joint_pos, home + _reflect_joints(joint_pos - home, names)])
    velocities = torch.stack([joint_vel, _reflect_joints(joint_vel, names)])
    robot.write_root_state_to_sim(roots.to(env.device))
    robot.write_joint_state_to_sim(positions.to(env.device), velocities.to(env.device))
    action = torch.randn(len(ACTION_JOINT_NAMES), generator=generator) * 0.3
    env.action_manager.process_action(torch.stack([action, mirror_actions(action)]).to(env.device))
    command = torch.tensor([0.3, 0.2, -0.6])
    env.command_manager.get_term("twist").vel_command_b[:] = torch.stack(
        [command, command * torch.tensor([1.0, -1.0, -1.0])]
    ).to(env.device)
    env.sim.forward()
    env.scene.update(env.step_dt)
    env.sim.sense()
    obs = env.observation_manager.compute(update_history=True)

    actor = obs["actor"].cpu()
    torch.testing.assert_close(actor[1], apply_mirror(WALK_MIRROR_OBSERVATION["actor"], actor[0]), rtol=0, atol=2e-4)
    critic = obs["critic"].cpu()
    mirrored = apply_mirror(WALK_MIRROR_OBSERVATION["critic"], critic[0])
    # The IMU site is 2 mm off the mid-plane: its linear velocity mirrors up to omega x 4 mm.
    tolerance = torch.full_like(mirrored, 1e-3)
    tolerance[:3] = 0.004 * float(root[10:].norm()) + 1e-3
    assert torch.all((critic[1] - mirrored).abs() <= tolerance), (critic[1] - mirrored).abs().max()
    # The state is not its own mirror: the check compares different numbers.
    assert (actor[1] - actor[0]).abs().max() > 0.1
    heights = critic[0, 72:74]
    assert (heights[0] - heights[1]).abs() > 0.005, heights


def test_the_walker_trains_with_the_mirror_loss():
    symmetry = MicrobanVelocityRlCfg.algorithm.symmetry_cfg
    assert symmetry.use_mirror_loss and not symmetry.use_data_augmentation
    assert symmetry.mirror_loss_coeff > 0.0
    assert symmetry.data_augmentation_func == "mjlab_microban.tasks.microban_velocity_mirror:mirror_augmentation"


def test_ppo_adds_the_mirror_loss_of_all_eighteen_outputs():
    envs, steps = 8, 4
    obs = _obs(envs, torch.Generator().manual_seed(4))
    torch.manual_seed(7)
    gaussian = {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}
    actor = MLPModel(obs, {"actor": ["actor"]}, "actor", 18, hidden_dims=(32,), distribution_cfg=gaussian)
    critic = MLPModel(obs, {"critic": ["critic"]}, "critic", 1, hidden_dims=(32,))
    symmetry = {
        "use_data_augmentation": False,
        "use_mirror_loss": True,
        "mirror_loss_coeff": 0.5,
        "data_augmentation_func": MicrobanVelocityRlCfg.algorithm.symmetry_cfg.data_augmentation_func,
        "_env": None,
    }
    algorithm = PPO(actor, critic, RolloutStorage("rl", envs, steps, obs, [18], "cpu"), num_mini_batches=1,
                    num_learning_epochs=1, learning_rate=0.0, schedule="fixed", symmetry_cfg=symmetry)
    for _ in range(steps):
        algorithm.act(obs)
        algorithm.process_env_step(obs, torch.randn(envs), torch.zeros(envs), {})
    algorithm.compute_returns(obs)
    loss = algorithm.update()["symmetry"]
    with torch.no_grad():
        error = actor(mirror_observations(obs)) - mirror_actions(actor(obs))
    assert loss == pytest.approx(float(error.square().mean()), rel=1e-5)
    assert loss > 0.0
