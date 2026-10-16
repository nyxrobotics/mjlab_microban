"""Walking's, PICO's and get-up's training floors and servo-goal delay: one floor (friction, contact time constant)
on every collision geom and one goal delay per environment, the delay redrawn at each reset; play and the judgments
keep robot.xml's robot, its foot friction and its goal delay."""

from __future__ import annotations

import re
from dataclasses import replace

import mujoco
import numpy as np
import pytest
import torch

from mjlab_microban.robot.microban_constants import (
    COMMAND_DELAY_STEPS,
    FLOOR_FRICTION,
    FLOOR_TIME_CONSTANT_S,
    MICROBAN_ROBOT_CFG,
    MICROBAN_TRAIN_ROBOT_CFG,
)
from mjlab_microban.tasks.mdp import contact_time_constant
from mjlab_microban.tasks.microban_getup_env_cfg import make_microban_getup_env_cfg
from mjlab_microban.tasks.microban_teleop_v13_arm_overlay import make_microban_teleop_v13_arm_overlay_env_cfg
from mjlab_microban.tasks.microban_velocity_env_cfg import make_microban_velocity_env_cfg

TASKS = {
    "walk": make_microban_velocity_env_cfg,
    "pico": make_microban_teleop_v13_arm_overlay_env_cfg,
    "getup": make_microban_getup_env_cfg,
}
NUM_ENVS = 8
COLLISION = re.compile(r".*_collision_\d+")
SOLE = re.compile(r"(left|right)_foot_collision_[1-6]")
SOLE_NAMES = (r".*left_foot_collision.*", r".*right_foot_collision.*")


@pytest.mark.parametrize("task", TASKS)
def test_training_draws_floors_and_goal_delays_and_play_keeps_the_robots(task):
    train, play = TASKS[task](), TASKS[task](play=True)
    friction, softness = train.events["geom_friction"], train.events["floor_softness"]
    assert "foot_friction" not in train.events
    assert friction.mode == softness.mode == "startup"
    assert friction.params["ranges"] == FLOOR_FRICTION == (0.15, 1.5)
    assert friction.params["operation"] == "abs" and friction.params["shared_random"]
    assert softness.func is contact_time_constant
    assert softness.params["time_constant_range"] == FLOOR_TIME_CONSTANT_S == (0.02, 0.1)
    assert friction.params["asset_cfg"].geom_names == softness.params["asset_cfg"].geom_names == (COLLISION.pattern + "$",)
    # Training's robot: robot.xml's with every collision geom named and the goal delay drawn once per episode.
    assert train.scene.entities["robot"] == MICROBAN_TRAIN_ROBOT_CFG
    (trained,) = train.scene.entities["robot"].articulation.actuators
    (robot,) = MICROBAN_ROBOT_CFG.articulation.actuators
    assert (trained.delay_min_lag, trained.delay_max_lag) == COMMAND_DELAY_STEPS == (3, 8)
    assert trained.delay_hold_prob == 0.0 and not trained.delay_per_env_phase
    assert trained.delay_update_period > train.episode_length_s / train.sim.mujoco.timestep
    assert replace(trained, delay_min_lag=robot.delay_min_lag, delay_max_lag=robot.delay_max_lag,
                   delay_hold_prob=robot.delay_hold_prob, delay_update_period=robot.delay_update_period,
                   delay_per_env_phase=robot.delay_per_env_phase) == robot
    # Play and the judgments: robot.xml's robot, the soles' friction 0.3-1.2, the robot's 3-6 steps every step.
    assert play.scene.entities["robot"] == MICROBAN_ROBOT_CFG
    assert "geom_friction" not in play.events and "floor_softness" not in play.events
    foot = play.events["foot_friction"]
    assert foot.params["ranges"] == (0.3, 1.2) and foot.params["asset_cfg"].geom_names == SOLE_NAMES
    (played,) = play.scene.entities["robot"].articulation.actuators
    assert (played.delay_min_lag, played.delay_max_lag, played.delay_update_period) == (3, 6, 0)
    # Nothing else differs between the two robots' configs.
    assert replace(train.scene.entities["robot"], spec_fn=MICROBAN_ROBOT_CFG.spec_fn,
                   collisions=MICROBAN_ROBOT_CFG.collisions,
                   articulation=MICROBAN_ROBOT_CFG.articulation) == MICROBAN_ROBOT_CFG


def test_the_training_robot_collides_as_the_robot_with_priority_over_the_floor():
    """Same geoms and contact parameters; every collision geom has priority over the floor (MuJoCo's defaults),
    so a floor contact takes the geom's friction and solref, which training draws per environment."""
    play = MICROBAN_ROBOT_CFG.build().spec.compile()
    train = MICROBAN_TRAIN_ROBOT_CFG.build().spec.compile()
    assert play.ngeom == train.ngeom
    for field in ("type", "size", "pos", "quat", "bodyid", "group", "contype", "conaffinity", "condim", "friction",
                  "solref", "solimp", "solmix", "margin", "gap"):
        np.testing.assert_array_equal(getattr(train, f"geom_{field}"), getattr(play, f"geom_{field}"), err_msg=field)
    collides = (train.geom_contype != 0) | (train.geom_conaffinity != 0)
    names = [mujoco.mj_id2name(train, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(train.ngeom)]
    assert [bool(COLLISION.fullmatch(name)) for name in names] == collides.tolist()
    assert sum(bool(SOLE.fullmatch(name)) for name in names) == 12
    assert np.all(train.geom_priority[collides] == 1)
    assert np.all(play.geom_condim[collides] == 3)
    soles = [bool(SOLE.fullmatch(mujoco.mj_id2name(play, mujoco.mjtObj.mjOBJ_GEOM, g) or "")) for g in range(play.ngeom)]
    np.testing.assert_array_equal(play.geom_priority[collides], np.asarray(soles)[collides].astype(int))


def _env(task: str):
    from mjlab.envs import ManagerBasedRlEnv

    cfg = TASKS[task]()
    cfg.scene.num_envs = NUM_ENVS
    cfg.seed = 0
    # No episode ends while the floors and the delays are read, and no curriculum (it needs a runner's clock).
    cfg.terminations = {"time_out": cfg.terminations["time_out"]}
    cfg.curriculum = {}
    return ManagerBasedRlEnv(cfg=cfg, device="cpu")


@pytest.fixture(scope="module", params=list(TASKS))
def env(request):
    env = _env(request.param)
    yield request.param, env
    env.close()


def _zero(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, env.action_manager.total_action_dim)


def _goal_lags(env) -> np.ndarray:
    """Physics steps from a new action to its goal reaching the servo's control law, per environment, for a
    switch of the right knee's action to +0.3 rad and back, each after three steps at the previous action."""
    (actuator,) = env.scene["robot"].actuators
    column = actuator.target_names.index("right_knee")
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
                env.step(torch.full_like(_zero(env), 0.3 - value))
            before = goals[-1]
            goals.clear()
            for _ in range(3):
                env.step(torch.full_like(_zero(env), value))
            moved = (torch.stack(goals) - before).abs() > 0.15  # (physics steps from the switch, envs)
            assert moved.any(dim=0).all()
            lags.append(moved.int().argmax(dim=0).numpy())
    finally:
        del actuator.compute
    return np.stack(lags)


def test_each_environment_has_its_own_floor_on_every_collision_geom_and_its_contacts_use_it(env):
    task, env = env
    robot = env.scene["robot"]
    geom_ids = robot.indexing.geom_ids.to(torch.long)
    collision = geom_ids[[i for i, name in enumerate(robot.geom_names) if COLLISION.fullmatch(name)]]
    soles = geom_ids[[i for i, name in enumerate(robot.geom_names) if SOLE.fullmatch(name)]]
    others = geom_ids[[i for i, name in enumerate(robot.geom_names) if not COLLISION.fullmatch(name)]]
    assert len(collision) == 36 and len(soles) == 12
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
    default_solref, default_friction = env.sim.get_default_field("geom_solref"), env.sim.get_default_field("geom_friction")
    for geom in (*others.tolist(), floor):
        assert torch.equal(solref[:, geom], default_solref[geom].expand(NUM_ENVS, 2))
        assert torch.equal(friction[:, geom], default_friction[geom].expand(NUM_ENVS, 3))
    # The solver's floor contacts carry their world's values (the robot's geoms have priority), the get-up's on
    # its hands, knees and back too.
    env.reset()
    for _ in range(25):  # 0.5 s: the get-up's fallen start has landed
        env.step(_zero(env))
    data = env.sim.wp_data
    count = int(data.nacon.numpy()[0])
    pairs = data.contact.geom.numpy()[:count]
    world = data.contact.worldid.numpy()[:count]
    robot_geom = np.where(pairs[:, 0] == floor, pairs[:, 1], pairs[:, 0])
    on_floor = (pairs == floor).any(axis=1) & np.isin(robot_geom, collision.numpy())
    assert set(world[on_floor].tolist()) == set(range(NUM_ENVS))
    np.testing.assert_allclose(data.contact.solref.numpy()[:count][on_floor, 0], tau[world[on_floor], 0].numpy())
    np.testing.assert_allclose(data.contact.friction.numpy()[:count][on_floor, 0], mu[world[on_floor], 0].numpy())
    if task == "getup":
        assert not np.isin(robot_geom[on_floor], soles.numpy()).all()


def test_the_goal_delay_holds_through_an_episode_and_is_redrawn_at_reset(env):
    _task, env = env
    env.reset()
    for _ in range(3):  # past the reset's backfilled goals
        env.step(_zero(env))
    lags = _goal_lags(env)
    assert lags.min() >= COMMAND_DELAY_STEPS[0] and lags.max() <= COMMAND_DELAY_STEPS[1]
    assert np.array_equal(lags[0], lags[1])  # one delay per episode
    assert len(set(lags[0].tolist())) > 1  # one per environment
    env.reset()
    for _ in range(3):
        env.step(_zero(env))
    redrawn = _goal_lags(env)
    assert np.array_equal(redrawn[0], redrawn[1])
    assert not np.array_equal(redrawn[0], lags[0])
