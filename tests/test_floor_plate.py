"""The training floor that gives sideways (tasks/floor_plate.py): one plate per environment on two slide joints with
a spring and a damper, a third of the environments on the static ground, the plate following a robot that walks
away; walking, PICO, the get-up and the whole-body train on it, play keeps the ground."""

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
    FLOOR_PLATE_RECENTRE_M,
    UNUSED_FLOOR_DEPTH_M,
    floor_plate,
    floor_plate_contact_weight,
    floor_plate_follow,
    floor_plate_recentre,
)
from mjlab_microban.tasks.microban_getup_env_cfg import make_microban_getup_env_cfg
from mjlab_microban.tasks.microban_teleop_v13_arm_overlay import make_microban_teleop_v13_arm_overlay_env_cfg
from mjlab_microban.tasks.microban_velocity_env_cfg import make_microban_velocity_env_cfg
from mjlab_microban.wholebody.env_cfg import make_wholebody_env_cfg

PLATE_TASKS = {"getup": make_microban_getup_env_cfg, "wholebody": make_wholebody_env_cfg,
               "walk": make_microban_velocity_env_cfg, "pico": make_microban_teleop_v13_arm_overlay_env_cfg}
# The modes of the ground's contact sensors (the feet's; the get-up's hands too) in training: the plane and the plate.
GROUND_SENSOR_MODES = {"getup": ["subtree", "subtree"], "wholebody": [], "walk": ["subtree"], "pico": ["subtree"]}
NUM_ENVS = 12
ROBOT_MASS_KG = 1.136
PLATE_MASS_KG = 1.0
# The damper's damping ratio is for the mass that sways on the spring: the robot standing on the plate.
SWAYING_MASS_KG = ROBOT_MASS_KG + PLATE_MASS_KG


@pytest.mark.parametrize("task", PLATE_TASKS)
def test_training_switches_the_plate_on_and_play_keeps_the_ground(task):
    train, play = PLATE_TASKS[task](), PLATE_TASKS[task](play=True)
    assert train.scene.spec_fn is add_floor_plate
    plate, weight = train.events["floor_plate"], train.events["floor_plate_contact_weight"]
    assert plate.func is floor_plate and plate.mode == "startup"
    assert plate.params == {"stiffness_range": (500.0, 5000.0), "damping_ratio_range": (0.3, 1.0),
                            "rigid_fraction": 1.0 / 3.0}
    assert weight.func is floor_plate_contact_weight and weight.mode == "reset"
    recentre, follow = train.events["floor_plate_recentre"], train.events["floor_plate_follow"]
    assert recentre.func is floor_plate_recentre and recentre.mode == "reset"
    assert follow.func is floor_plate_follow and follow.mode == "step"
    # The ground's contact sensors match its subtree: the plane and the plate.
    ground = [s.secondary.mode for s in train.scene.sensors if getattr(s, "secondary", None)
              and s.secondary.pattern == GROUND]
    assert ground == GROUND_SENSOR_MODES[task]
    assert play.scene.spec_fn is None
    assert not [name for name in play.events if name.startswith("floor_plate")]


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
    # Rigid: the static ground at the floor, the plate 2 m below it with no spring or damper.
    torch.testing.assert_close(plane_z[rigid], origins[rigid, 2])
    torch.testing.assert_close(plate_pos[rigid], origins[rigid] - torch.tensor([0.0, 0.0, UNUSED_FLOOR_DEPTH_M]))
    assert torch.all(stiffness[rigid] == 0.0) and torch.all(damping[rigid] == 0.0)
    # Compliant: the plate under the origin and the plane 2 m below it; one spring and one damper per environment,
    # the same in x and y, the damper of a damping ratio 0.3-1.0 for the robot's and the plate's mass.
    soft = ~rigid
    torch.testing.assert_close(plate_pos[soft], origins[soft])
    torch.testing.assert_close(plane_z[soft], origins[soft, 2] - UNUSED_FLOOR_DEPTH_M)
    assert torch.equal(stiffness[:, 0], stiffness[:, 1]) and torch.equal(damping[:, 0], damping[:, 1])
    k, c = stiffness[soft, 0], damping[soft, 0]
    assert torch.all((k >= 500.0) & (k <= 5000.0)) and len(set(k.tolist())) == int(soft.sum())
    robot_mass = float(mj_model.body_mass[env.scene["robot"].indexing.body_ids.numpy()].sum())
    assert robot_mass == pytest.approx(ROBOT_MASS_KG, abs=0.001)
    plate_mass = float(mj_model.body(FLOOR_PLATE).mass[0])
    ratio = c / (2.0 * torch.sqrt(k * (robot_mass + plate_mass)))
    assert torch.all((ratio >= 0.3 - 1e-6) & (ratio <= 1.0 + 1e-6))
    # The plate: 1 kg on two slide joints, x and y, of +-10 mm.
    assert plate_mass == pytest.approx(PLATE_MASS_KG)
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
    damping = 2.0 * torch.tensor([z for _, z, _ in cases]) * torch.sqrt(stiffness * SWAYING_MASS_KG)
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


def test_a_load_resting_on_the_plate_sways_with_the_drawn_damping_at_5_ms():
    """A 1.136 kg box (the robot's mass) resting on the plate, the plate pushed by 2.7 N, at the training's 5 ms
    step: the two sway together with the drawn damping ratio, and the box does not slip on the plate.  MuJoCo's
    contact solver does not see the implicit damper, so a light plate fails this: at 0.1 kg the box rang at a
    damping ratio of 0.002 for 0.3 drawn (5000 N/m) and slid 11.6 mm."""
    from mjlab.scene import Scene, SceneCfg
    from mjlab.sim import MujocoCfg, Simulation, SimulationCfg
    from mjlab.terrains import TerrainEntityCfg

    def spec_fn(spec):
        add_floor_plate(spec)
        box = spec.worldbody.add_body(name="load", pos=(0.0, 0.0, 0.0251))
        box.add_freejoint()
        box.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.05, 0.03, 0.025), mass=ROBOT_MASS_KG)
        for key in list(spec.keys):
            spec.delete(key)

    cases = [(500.0, 0.3), (500.0, 1.0), (1500.0, 0.3), (5000.0, 0.3), (5000.0, 1.0)]
    n, force, dt = len(cases), 2.7, 0.005
    scene = Scene(SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=n, spec_fn=spec_fn), device="cpu")
    sim = Simulation(num_envs=n, cfg=SimulationCfg(mujoco=MujocoCfg(timestep=dt, integrator="implicitfast")),
                     model=scene.compile(), device="cpu")
    sim.expand_model_fields(("jnt_stiffness", "dof_damping", "body_invweight0"))
    mj_model = sim.mj_model
    plate = mj_model.body(FLOOR_PLATE).id
    stiffness = torch.tensor([k for k, _ in cases])
    ratio = torch.tensor([z for _, z in cases])
    for name in FLOOR_PLATE_JOINTS:
        joint = mj_model.joint(name)
        sim.model.jnt_stiffness[:, joint.id] = stiffness
        sim.model.dof_damping[:, int(joint.dofadr[0])] = 2.0 * ratio * torch.sqrt(stiffness * SWAYING_MASS_KG)
    # As in a compliant training environment: the plate's contacts as soft as the ground's, the plane 2 m below.
    sim.model.body_invweight0[:, plate] = 0.0
    sim.data.geom_xpos[:, mj_model.geom(GROUND).id, 2] = -UNUSED_FLOOR_DEPTH_M
    sim.forward()
    load = int(mj_model.jnt_qposadr[mj_model.body("load").jntadr[0]])
    for _ in range(100):  # the box settles on the plate
        sim.step()
    plate_x0, load_x0 = sim.data.qpos[:, 0].clone(), sim.data.qpos[:, load].clone()
    sim.data.xfrc_applied[:, plate, 0] = force
    plate_x, slip = [], []
    for _ in range(300):  # 1.5 s
        sim.step()
        plate_x.append(sim.data.qpos[:, 0] - plate_x0)
        slip.append(sim.data.qpos[:, load] - load_x0 - plate_x[-1])
    plate_x, slip = torch.stack(plate_x), torch.stack(slip)
    static = force / stiffness
    torch.testing.assert_close(plate_x[-1], static, rtol=0.01, atol=0.0)
    assert float(slip.abs().max()) < 1.5e-4
    # The damping ratio from the overshoot of the step response: 0.26-0.29 for 0.3 drawn, no overshoot for 1.0.
    overshoot = plate_x.max(dim=0).values / static - 1.0
    log_overshoot = torch.log(overshoot.clamp(min=1e-9))
    measured = -log_overshoot / torch.sqrt(torch.pi**2 + log_overshoot**2)
    low = ratio == 0.3
    assert torch.all((measured[low] > 0.24) & (measured[low] < 0.33)), measured
    assert torch.all(overshoot[~low] < 0.01), overshoot


def test_the_plate_follows_a_robot_that_walks_away_and_comes_back_at_reset(env):
    robot = env.scene["robot"]
    plate = env.sim.mj_model.body(FLOOR_PLATE).id
    origins = env.scene.env_origins
    pose = robot.data.root_link_pose_w.clone()
    # Every other environment's robot 3 m away in x, the others 1 m away (within FLOOR_PLATE_RECENTRE_M).
    far = torch.arange(env.num_envs) % 2 == 0
    pose[:, 0] = origins[:, 0] + torch.where(far, FLOOR_PLATE_RECENTRE_M + 1.0, FLOOR_PLATE_RECENTRE_M - 1.0)
    pose[:, 1] = origins[:, 1]
    robot.write_root_link_pose_to_sim(pose)
    env.sim.forward()
    depth_before = env.sim.model.body_pos[:, plate, 2].clone()
    floor_plate_follow(env, None)
    centre = env.sim.model.body_pos[:, plate]
    assert torch.allclose(centre[far, :2], pose[far, :2])
    assert torch.allclose(centre[~far, :2], origins[~far, :2])
    assert torch.equal(centre[:, 2], depth_before)  # a rigid floor's plate stays out of reach below the ground
    floor_plate_recentre(env, torch.arange(env.num_envs))
    assert torch.allclose(env.sim.model.body_pos[:, plate, :2], origins[:, :2])
