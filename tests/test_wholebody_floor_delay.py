"""Training's floors and servo-goal delay: one floor (friction, contact time constant) and one goal delay per
environment, the delay redrawn at each reset; evaluation keeps the rigid floor and the robot's delay."""

from __future__ import annotations

import re
from dataclasses import replace

import mujoco
import numpy as np
import pytest
import torch

from mjlab_microban.policy_contract import ACTION_JOINT_NAMES, joint_ranges
from mjlab_microban.robot.home_pose import HOME
from mjlab_microban.wholebody import contract
from mjlab_microban.wholebody.data.hdf5 import FRAME_KEYS, write_motion_shard
from mjlab_microban.wholebody.env_cfg import (
    ACTOR_GROUP,
    COMMAND_DELAY_STEPS,
    FLOOR_FRICTION,
    FLOOR_TIME_CONSTANT_S,
    TRACKED_BODY_NAMES,
    WHOLEBODY_ROBOT_CFG,
    WHOLEBODY_TRAIN_ROBOT_CFG,
    make_wholebody_env_cfg,
)
from mjlab_microban.wholebody.events import contact_time_constant

NUM_ENVS = 8
COLLISION = re.compile(r".*_collision_\d+")


def test_training_draws_floors_and_goal_delays_and_play_keeps_the_robots():
    train, play = make_wholebody_env_cfg(), make_wholebody_env_cfg(play=True)
    friction, softness = train.events["geom_friction"], train.events["floor_softness"]
    assert friction.params["ranges"] == FLOOR_FRICTION == (0.15, 1.5) and friction.params["shared_random"]
    assert softness.func is contact_time_constant and softness.mode == "startup"
    assert softness.params["time_constant_range"] == FLOOR_TIME_CONSTANT_S == (0.02, 0.1)
    assert softness.params["asset_cfg"].geom_names == friction.params["asset_cfg"].geom_names
    assert "geom_friction" not in play.events and "floor_softness" not in play.events
    # Training's goal delay: 0-6 physics steps, drawn once per episode; only the delay differs.
    (trained,) = train.scene.entities["robot"].articulation.actuators
    (robot,) = WHOLEBODY_ROBOT_CFG.articulation.actuators
    assert (trained.delay_min_lag, trained.delay_max_lag) == COMMAND_DELAY_STEPS == (0, 6)
    assert trained.delay_hold_prob == 0.0 and not trained.delay_per_env_phase
    assert trained.delay_update_period > train.episode_length_s / train.sim.mujoco.timestep
    assert replace(trained, delay_min_lag=robot.delay_min_lag, delay_max_lag=robot.delay_max_lag,
                   delay_update_period=robot.delay_update_period,
                   delay_per_env_phase=robot.delay_per_env_phase) == robot
    assert replace(WHOLEBODY_TRAIN_ROBOT_CFG, articulation=WHOLEBODY_ROBOT_CFG.articulation) == WHOLEBODY_ROBOT_CFG
    # Evaluation and play: the robot's own 3-6 steps redrawn every step.
    (played,) = play.scene.entities["robot"].articulation.actuators
    assert played == robot
    assert (played.delay_min_lag, played.delay_max_lag, played.delay_update_period) == (3, 6, 0)


def _standing_shard(path) -> None:
    """The reference standing still at HOME."""
    joints, frames = tuple(joint_ranges()), 600
    clip = {key: np.zeros((frames, len(joints)) if key.startswith("joint")
                          else (frames, len(TRACKED_BODY_NAMES), 4 if "quat" in key else 3)) for key in FRAME_KEYS}
    clip["joint_pos"][:] = [HOME.joint_pos_rad[name] for name in joints]
    clip["body_quat_w"][..., 0] = 1.0
    clip["body_pos_w"][:, 0] = HOME.root_pos
    clip["body_quat_w"][:, 0] = HOME.root_quat_wxyz
    write_motion_shard([{"name": "stand", "fps": 50, **clip}], path, body_names=TRACKED_BODY_NAMES,
                       joint_names=joints, max_window_frames=None)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    from mjlab.envs import ManagerBasedRlEnv

    path = tmp_path_factory.mktemp("floor") / "stand.h5"
    _standing_shard(path)
    cfg = make_wholebody_env_cfg()
    cfg.commands["motion"].motion_file = str(path)
    cfg.scene.num_envs = NUM_ENVS
    cfg.seed = 0
    # No episode ends while the delays are timed.
    cfg.terminations = {"time_out": cfg.terminations["time_out"]}
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    yield env
    env.close()


def test_each_environment_has_its_own_floor_and_its_contacts_use_it(env):
    robot = env.scene["robot"]
    geom_ids = robot.indexing.geom_ids.to(torch.long)
    collision = geom_ids[[i for i, name in enumerate(robot.geom_names) if COLLISION.fullmatch(name)]]
    others = geom_ids[[i for i, name in enumerate(robot.geom_names) if not COLLISION.fullmatch(name)]]
    floor = mujoco.mj_name2id(env.sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
    solref, friction = env.sim.model.geom_solref, env.sim.model.geom_friction
    # Per-world arrays: one time constant and one friction per environment on every collision geom.
    assert solref.shape[0] == friction.shape[0] == NUM_ENVS
    tau, mu = solref[:, collision, 0], friction[:, collision, 0]
    assert torch.equal(tau, tau[:, :1].expand_as(tau)) and torch.equal(mu, mu[:, :1].expand_as(mu))
    assert FLOOR_TIME_CONSTANT_S[0] <= tau.min() and tau.max() <= FLOOR_TIME_CONSTANT_S[1]
    assert FLOOR_FRICTION[0] <= mu.min() and mu.max() <= FLOOR_FRICTION[1]
    assert len(set(tau[:, 0].tolist())) == len(set(mu[:, 0].tolist())) == NUM_ENVS
    assert torch.all(solref[:, collision, 1] == 1.0)  # the damping ratio is MuJoCo's
    default = env.sim.get_default_field("geom_solref")
    for geom in (*others.tolist(), floor):
        assert torch.equal(solref[:, geom], default[geom].expand(NUM_ENVS, 2))
    # The solver's floor contacts carry their world's values (the robot's geoms have priority).
    env.reset()
    actions = torch.zeros(NUM_ENVS, env.action_manager.total_action_dim)
    for _ in range(5):
        env.step(actions)
    data = env.sim.wp_data
    count = int(data.nacon.numpy()[0])
    pairs = data.contact.geom.numpy()[:count]
    world = data.contact.worldid.numpy()[:count]
    on_floor = (pairs == floor).any(axis=1) & np.isin(pairs, collision.numpy()).any(axis=1)
    assert set(world[on_floor].tolist()) == set(range(NUM_ENVS))
    np.testing.assert_allclose(data.contact.solref.numpy()[:count][on_floor, 0], tau[world[on_floor], 0].numpy())
    np.testing.assert_allclose(data.contact.friction.numpy()[:count][on_floor, 0], mu[world[on_floor], 0].numpy())


def _goal_lags(env) -> np.ndarray:
    """Physics steps from a new action to its goal reaching the servo's control law, per environment, for a
    switch to +0.3 rad and back, each after three steps at the previous action."""
    robot = env.scene["robot"]
    (actuator,) = robot.actuators
    column = actuator.target_names.index("right_elbow")
    names = [name for name, _ in contract.OBSERVATION_SCHEMA]
    start = sum(width for _, width in contract.OBSERVATION_SCHEMA[: names.index("actions")])
    goals = []
    compute = actuator.compute

    def record(cmd):
        goals.append(cmd.position_target[:, column].clone())
        return compute(cmd)

    actuator.compute = record
    try:
        lags = []
        for value in (0.3, 0.0):
            for _ in range(3):
                env.step(torch.full((env.num_envs, len(ACTION_JOINT_NAMES)), 0.3 - value))
            before = goals[-1]
            goals.clear()
            action = torch.full((env.num_envs, len(ACTION_JOINT_NAMES)), value)
            obs, *_ = env.step(action)
            # The policy observes its latest output at once, whatever the goal's delay.
            assert torch.equal(obs[ACTOR_GROUP][:, -1, start:start + len(ACTION_JOINT_NAMES)], action)
            for _ in range(2):
                env.step(action)
            moved = (torch.stack(goals) - before).abs() > 0.15  # (physics steps from the switch, envs)
            assert moved.any(dim=0).all()
            lags.append(moved.int().argmax(dim=0).numpy())
    finally:
        del actuator.compute
    return np.stack(lags)


def test_the_goal_delay_holds_through_an_episode_and_is_redrawn_at_reset(env):
    env.reset()
    for _ in range(3):  # past the reset's backfilled goals
        env.step(torch.zeros(NUM_ENVS, len(ACTION_JOINT_NAMES)))
    lags = _goal_lags(env)
    assert lags.min() >= COMMAND_DELAY_STEPS[0] and lags.max() <= COMMAND_DELAY_STEPS[1]
    assert np.array_equal(lags[0], lags[1])  # one delay per episode
    assert len(set(lags[0].tolist())) > 1  # one per environment
    env.reset()
    for _ in range(3):
        env.step(torch.zeros(NUM_ENVS, len(ACTION_JOINT_NAMES)))
    redrawn = _goal_lags(env)
    assert np.array_equal(redrawn[0], redrawn[1])
    assert not np.array_equal(redrawn[0], lags[0])
