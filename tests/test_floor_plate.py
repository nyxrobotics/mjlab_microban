"""The training floor that gives sideways (tasks/floor_plate.py): one plate per environment on two slide joints with
a spring and a damper, a third of the environments on the static ground; play, walking and PICO keep the ground."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
import torch

from mjlab_microban.tasks.floor_plate import (
    FLOOR_PLATE,
    FLOOR_PLATE_JOINTS,
    GROUND,
    add_floor_plate,
    floor_plate,
    floor_plate_contact_weight,
)
from mjlab_microban.tasks.microban_getup_env_cfg import make_microban_getup_env_cfg
from mjlab_microban.tasks.microban_teleop_v13_arm_overlay import make_microban_teleop_v13_arm_overlay_env_cfg
from mjlab_microban.tasks.microban_velocity_env_cfg import make_microban_velocity_env_cfg
from mjlab_microban.wholebody.env_cfg import make_wholebody_env_cfg

PLATE_TASKS = {"getup": make_microban_getup_env_cfg, "wholebody": make_wholebody_env_cfg}
GROUND_TASKS = {"walk": make_microban_velocity_env_cfg, "pico": make_microban_teleop_v13_arm_overlay_env_cfg}
NUM_ENVS = 12
ROBOT_MASS_KG = 1.136


@pytest.mark.parametrize("task", PLATE_TASKS)
def test_training_switches_the_plate_on_and_play_keeps_the_ground(task):
    train, play = PLATE_TASKS[task](), PLATE_TASKS[task](play=True)
    assert train.scene.spec_fn is add_floor_plate
    plate, weight = train.events["floor_plate"], train.events["floor_plate_contact_weight"]
    assert plate.func is floor_plate and plate.mode == "startup"
    assert plate.params == {"stiffness_range": (500.0, 5000.0), "damping_ratio_range": (0.3, 1.0),
                            "rigid_fraction": 1.0 / 3.0}
    assert weight.func is floor_plate_contact_weight and weight.mode == "reset"
    # The ground's contact sensors (the get-up's feet and hands) match its subtree: the plane and the plate.
    ground = [s.secondary.mode for s in train.scene.sensors if getattr(s, "secondary", None)
              and s.secondary.pattern == GROUND]
    assert ground == (["subtree", "subtree"] if task == "getup" else [])
    assert play.scene.spec_fn is None
    assert "floor_plate" not in play.events and "floor_plate_contact_weight" not in play.events


@pytest.mark.parametrize("task", GROUND_TASKS)
def test_walking_and_pico_keep_the_ground(task):
    for cfg in (GROUND_TASKS[task](), GROUND_TASKS[task](play=True)):
        assert cfg.scene.spec_fn is None and "floor_plate" not in cfg.events


@pytest.fixture(scope="module")
def env():
    from mjlab.envs import ManagerBasedRlEnv

    cfg = make_microban_getup_env_cfg()
    cfg.scene.num_envs = NUM_ENVS
    cfg.seed = 0
    # No episode ends while the floors are read, and no curriculum (it needs a runner's clock).
    cfg.terminations = {"time_out": cfg.terminations["time_out"]}
    cfg.curriculum = {}
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    yield env
    env.close()


def _floors(env):
    """Per environment: whether its floor is rigid, the plate's springs and dampers (x, y), the plate's position
    and the ground plane's height."""
    mj_model, model = env.sim.mj_model, env.sim.model
    joints = [mj_model.joint(name) for name in FLOOR_PLATE_JOINTS]
    stiffness = model.jnt_stiffness[:, [joint.id for joint in joints]]
    damping = model.dof_damping[:, [int(joint.dofadr[0]) for joint in joints]]
    plate_pos = model.body_pos[:, mj_model.body(FLOOR_PLATE).id]
    plane_z = env.sim.data.geom_xpos[:, mj_model.geom(GROUND).id, 2]
    rigid = plate_pos[:, 2] < env.scene.env_origins[:, 2] - 0.5
    return rigid, stiffness, damping, plate_pos, plane_z


def test_each_environment_reads_back_its_floor(env):
    rigid, stiffness, damping, plate_pos, plane_z = _floors(env)
    origins = env.scene.env_origins
    mj_model = env.sim.mj_model
    assert int(rigid.sum()) == round(NUM_ENVS / 3) == 4
    # Rigid: the static ground at the floor, the plate 1 m below it with no spring or damper.
    torch.testing.assert_close(plane_z[rigid], origins[rigid, 2])
    torch.testing.assert_close(plate_pos[rigid], origins[rigid] - torch.tensor([0.0, 0.0, 1.0]))
    assert torch.all(stiffness[rigid] == 0.0) and torch.all(damping[rigid] == 0.0)
    # Compliant: the plate under the origin and the plane 1 m below it; one spring and one damper per environment,
    # the same in x and y, the damper of a damping ratio 0.3-1.0 for the robot's mass.
    soft = ~rigid
    torch.testing.assert_close(plate_pos[soft], origins[soft])
    torch.testing.assert_close(plane_z[soft], origins[soft, 2] - 1.0)
    assert torch.equal(stiffness[:, 0], stiffness[:, 1]) and torch.equal(damping[:, 0], damping[:, 1])
    k, c = stiffness[soft, 0], damping[soft, 0]
    assert torch.all((k >= 500.0) & (k <= 5000.0)) and len(set(k.tolist())) == int(soft.sum())
    robot_mass = float(mj_model.body_mass[env.scene["robot"].indexing.body_ids.numpy()].sum())
    assert robot_mass == pytest.approx(ROBOT_MASS_KG, abs=0.001)
    ratio = c / (2.0 * torch.sqrt(k * robot_mass))
    assert torch.all((ratio >= 0.3 - 1e-6) & (ratio <= 1.0 + 1e-6))
    # The plate: 0.1 kg on two slide joints, x and y, of +-10 mm.
    assert mj_model.body(FLOOR_PLATE).mass[0] == pytest.approx(0.1)
    for name, axis in zip(FLOOR_PLATE_JOINTS, ((1, 0, 0), (0, 1, 0)), strict=True):
        joint = mj_model.joint(name)
        assert joint.type[0] == mujoco.mjtJoint.mjJNT_SLIDE and tuple(joint.axis) == axis
        assert joint.limited[0] and tuple(joint.range) == pytest.approx((-0.01, 0.01))


def test_rigid_floors_stand_on_the_ground_and_the_others_on_the_plate(env):
    env.reset()
    mj_model = env.sim.mj_model
    # The plate's contacts are regularized as the ground's.
    assert torch.all(env.sim.model.body_invweight0[:, mj_model.body(FLOOR_PLATE).id] == 0.0)
    rigid, *_ = _floors(env)
    robot = env.scene["robot"]
    root = robot.data.default_root_state.clone()
    root[:, :3] += env.scene.env_origins
    robot.write_root_state_to_sim(root)  # standing at HOME
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
    env.sim.forward()
    for _ in range(5):
        env.step(torch.zeros(NUM_ENVS, env.action_manager.total_action_dim))
    data = env.sim.wp_data
    count = int(data.nacon.numpy()[0])
    pairs, world = data.contact.geom.numpy()[:count], data.contact.worldid.numpy()[:count]
    plane, plate = mj_model.geom(GROUND).id, mj_model.geom(FLOOR_PLATE).id
    on_plane = np.bincount(world[(pairs == plane).any(axis=1)], minlength=NUM_ENVS)
    on_plate = np.bincount(world[(pairs == plate).any(axis=1)], minlength=NUM_ENVS)
    np.testing.assert_array_equal(on_plate[rigid.numpy()], 0)
    np.testing.assert_array_equal(on_plane[~rigid.numpy()], 0)
    assert np.all(on_plane[rigid.numpy()] > 0) and np.all(on_plate[~rigid.numpy()] > 0)
    # The feet's ground sensors see both floors.
    found = env.scene["feet_ground_contact"].data.found.reshape(NUM_ENVS, 2)
    assert torch.all(found > 0)


def test_a_sideways_force_bends_the_plate_by_f_over_k_within_its_travel():
    """A plate alone under a constant sideways force, each world with its own spring and damper in MuJoCo Warp,
    settles at F / k; past k times the travel it stops at the +-10 mm limit; a reset centres it again."""
    from mjlab.scene import Scene, SceneCfg
    from mjlab.sim import MujocoCfg, Simulation, SimulationCfg
    from mjlab.terrains import TerrainEntityCfg

    # (k N/m, damping ratio, force N): 2.7 N is the largest sideways foot force of standing and walking.
    cases = [(500.0, 0.3, (2.7, 0.0)), (500.0, 1.0, (0.0, -2.7)), (5000.0, 0.3, (2.7, 2.7)),
             (1500.0, 0.6, (-4.0, 1.0)), (500.0, 1.0, (7.5, 0.0))]
    n = len(cases)
    scene = Scene(SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=n, spec_fn=add_floor_plate),
                  device="cpu")
    sim = Simulation(num_envs=n, cfg=SimulationCfg(mujoco=MujocoCfg(timestep=0.005)), model=scene.compile(),
                     device="cpu")
    sim.expand_model_fields(("jnt_stiffness", "dof_damping"))
    mj_model = sim.mj_model
    joints = [mj_model.joint(name) for name in FLOOR_PLATE_JOINTS]
    assert [int(joint.qposadr[0]) for joint in joints] == [0, 1]
    force = torch.tensor([f for _, _, f in cases])
    stiffness = torch.tensor([k for k, _, _ in cases])
    damping = 2.0 * torch.tensor([z for _, z, _ in cases]) * torch.sqrt(stiffness * ROBOT_MASS_KG)
    for joint in joints:
        sim.model.jnt_stiffness[:, joint.id] = stiffness
        sim.model.dof_damping[:, int(joint.dofadr[0])] = damping
    sim.data.xfrc_applied[:, mj_model.body(FLOOR_PLATE).id, :2] = force
    for _ in range(400):  # 2 s
        sim.step()
    bend = sim.data.qpos[:, :2]
    torch.testing.assert_close(bend[:4], force[:4] / stiffness[:4, None], atol=2e-6, rtol=1e-3)
    assert torch.all(bend[:4].abs() < 0.01)
    # 7.5 N on 500 N/m would bend it 15 mm: the joint limit (MuJoCo's default softness) stops it 0.6 mm past 10 mm.
    assert 0.01 < float(bend[4, 0]) < 0.0107 and abs(float(bend[4, 1])) < 1e-6
    sim.reset()
    assert torch.all(sim.data.qpos[:, :2] == 0.0) and torch.all(sim.data.qvel[:, :2] == 0.0)
