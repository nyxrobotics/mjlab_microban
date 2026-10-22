"""The judgments' soft and slippery floors, one per environment.

A floor is one contact time constant (solref[0], s) and one friction.  The
walker's and PICO's are the soles': the play robot's soles have priority over
the floor (robot/microban_constants.py FULL_COLLISION), so a sole contact
takes the soles' values, as training's floors do on every collision geom
(tasks/mdp.py training_floor_and_delay).  The get-up's are every collision
geom's, as in training (it lies on its hands, knees and back): the robot's
collision geoms are named and given priority over the floor as in training's
robot (TRAINING_COLLISION; with the floor's values the contacts are the play
robot's).  A friction of None keeps the play env's drawn sole friction
(foot_friction, 0.3-1.2) and the other geoms' 1.  The rigid floor is MuJoCo's
0.02 s, the play env's; config/pipeline.yaml ``walk.judgment.floors``,
``pico.judgment.floors`` and ``getup.floors`` name the judged floors, inside
training's range (FLOOR_TIME_CONSTANT_S, FLOOR_FRICTION).

A floor may also give sideways like training's carpet (tasks/floor_plate.py):
``plate: {stiffness_n_m: k, damping_ratio: zeta}`` puts every environment of
it on the plate with that spring and damping ratio (``add_judgment_plates``),
inside training's range (FLOOR_PLATE_STIFFNESS_N_M, FLOOR_PLATE_DAMPING_RATIO).
"""

from __future__ import annotations

import math
from typing import Any

SOLES = (r".*left_foot_collision.*", r".*right_foot_collision.*")
# Every collision geom of training's robot spec (robot/microban_constants.py collision_named_spec), the soles too.
BODY = (r".*_collision_\d+$",)
RIGID_TIME_CONSTANT_S = 0.02


class Floor(tuple):
    """(friction or None, time constant s), carrying the floor's plate (spring N/m, damping ratio) or None as
    ``plate``: the evaluators pass it on as the pair, and ``add_sole_floors`` puts the plate in."""

    plate: tuple[float, float] | None

    def __new__(cls, friction: float | None, time_constant: float, plate: tuple[float, float] | None = None):
        floor = super().__new__(cls, (friction, time_constant))
        floor.plate = plate
        return floor


def parse(floors: dict[str, dict[str, Any]]) -> dict[str, tuple[float | None, float]]:
    """{name: (friction or None, time constant s)} of config/pipeline.yaml's floors, checked against training's range."""

    from mjlab_microban.robot.microban_constants import FLOOR_FRICTION, FLOOR_TIME_CONSTANT_S

    parsed = {}
    for name, floor in floors.items():
        if set(floor) - {"friction", "time_constant_s", "plate"}:
            raise ValueError(f"floor {name}: unknown keys {sorted(set(floor) - {'friction', 'time_constant_s', 'plate'})}")
        friction = floor.get("friction")
        time_constant = float(floor.get("time_constant_s", RIGID_TIME_CONSTANT_S))
        if not FLOOR_TIME_CONSTANT_S[0] <= time_constant <= FLOOR_TIME_CONSTANT_S[1]:
            raise ValueError(f"floor {name}: time constant {time_constant} s outside training's {FLOOR_TIME_CONSTANT_S}")
        if friction is not None and not FLOOR_FRICTION[0] <= float(friction) <= FLOOR_FRICTION[1]:
            raise ValueError(f"floor {name}: friction {friction} outside training's {FLOOR_FRICTION}")
        parsed[name] = Floor(None if friction is None else float(friction), time_constant, parse_plate(name, floor))
    return parsed


def parse_plate(name: str, floor: dict[str, Any]) -> tuple[float, float] | None:
    """(spring N/m, damping ratio) of a floor's ``plate``, checked against training's range; None without one."""

    from mjlab_microban.tasks.floor_plate import FLOOR_PLATE_DAMPING_RATIO, FLOOR_PLATE_STIFFNESS_N_M

    plate = floor.get("plate")
    if plate is None:
        return None
    if set(plate) != {"stiffness_n_m", "damping_ratio"}:
        raise ValueError(f"floor {name}: a plate takes stiffness_n_m and damping_ratio, got {sorted(plate)}")
    stiffness, ratio = float(plate["stiffness_n_m"]), float(plate["damping_ratio"])
    if not FLOOR_PLATE_STIFFNESS_N_M[0] <= stiffness <= FLOOR_PLATE_STIFFNESS_N_M[1]:
        raise ValueError(f"floor {name}: plate spring {stiffness} N/m outside training's {FLOOR_PLATE_STIFFNESS_N_M}")
    if not FLOOR_PLATE_DAMPING_RATIO[0] <= ratio <= FLOOR_PLATE_DAMPING_RATIO[1]:
        raise ValueError(f"floor {name}: plate damping ratio {ratio} outside training's {FLOOR_PLATE_DAMPING_RATIO}")
    return stiffness, ratio


def _judgment_plate(env, env_ids, stiffnesses, damping_ratios) -> None:
    """Each environment's plate (tasks/floor_plate.floor_plate with drawn values replaced by per-environment
    ones): NaN stiffness keeps the static ground."""

    import torch

    from mjlab_microban.tasks.floor_plate import FLOOR_PLATE, FLOOR_PLATE_JOINTS, GROUND, UNUSED_FLOOR_DEPTH_M

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.to(env.device, torch.long)
    mj_model = env.sim.mj_model
    plate_id = mj_model.body(FLOOR_PLATE).id
    mass = float(mj_model.body_mass[env.scene["robot"].indexing.body_ids.cpu().numpy()].sum())
    mass += float(mj_model.body_mass[plate_id])
    stiffness = torch.tensor(stiffnesses, device=env.device, dtype=torch.float32)[env_ids]
    ratio = torch.tensor(damping_ratios, device=env.device, dtype=torch.float32)[env_ids]
    rigid = torch.isnan(stiffness)
    stiffness = torch.nan_to_num(stiffness, nan=0.0)
    damping = torch.where(rigid, 0.0, 2.0 * torch.nan_to_num(ratio, nan=0.0) * torch.sqrt(stiffness * mass))
    model = env.sim.model
    for joint_name in FLOOR_PLATE_JOINTS:
        joint = mj_model.joint(joint_name)
        model.jnt_stiffness[env_ids, joint.id] = stiffness.to(model.jnt_stiffness.dtype)
        model.dof_damping[env_ids, int(joint.dofadr[0])] = damping.to(model.dof_damping.dtype)
    unused = torch.tensor([0.0, 0.0, UNUSED_FLOOR_DEPTH_M], device=env.device)
    model.body_pos[env_ids, plate_id] = (env.scene.env_origins[env_ids] - rigid[:, None] * unused).to(model.body_pos.dtype)
    plane = mj_model.geom(GROUND).id
    env.sim.data.geom_xpos[env_ids, plane, 2] = torch.where(rigid, 0.0, -UNUSED_FLOOR_DEPTH_M).to(
        env.sim.data.geom_xpos.dtype)


def add_judgment_plates(cfg, per_env: list[tuple[float, float] | None]) -> None:
    """Put environment i of ``cfg`` on a plate with the spring and damping ratio ``per_env[i]`` (None: the static
    ground), training's plate (tasks/floor_plate.train_on_floor_plate: following the robot, its contacts as soft
    as the ground's) with nothing drawn.  Nothing to do when no environment has one."""

    from mjlab.managers.event_manager import requires_model_fields

    from mjlab_microban.tasks.floor_plate import train_on_floor_plate

    if len(per_env) != cfg.scene.num_envs:
        raise ValueError(f"{len(per_env)} plates for {cfg.scene.num_envs} environments")
    if all(plate is None for plate in per_env):
        return
    train_on_floor_plate(cfg)
    # The plate's box collider gives a sole up to eight contacts with it: the feet's contact sensors match more
    # than the default 64 (the get-up's 500).
    cfg.sim.contact_sensor_maxmatch = max(cfg.sim.contact_sensor_maxmatch, 500)
    term = cfg.events["floor_plate"]
    term.func = requires_model_fields("body_pos", "jnt_stiffness", "dof_damping")(_judgment_plate)
    term.params = {
        "stiffnesses": [math.nan if plate is None else plate[0] for plate in per_env],
        "damping_ratios": [math.nan if plate is None else plate[1] for plate in per_env],
    }


def _sole_floor(env, env_ids, frictions, time_constants, asset_cfg) -> None:
    """Each environment's sole friction (NaN: kept) and contact time constant, from per-environment lists."""

    import torch

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.to(env.device, torch.long)
    geoms = env.scene[asset_cfg.name].indexing.geom_ids[asset_cfg.geom_ids].to(torch.long)
    friction = torch.tensor(frictions, device=env.device, dtype=torch.float32)[env_ids]
    time_constant = torch.tensor(time_constants, device=env.device, dtype=torch.float32)[env_ids]
    rows, cols = torch.meshgrid(env_ids, geoms, indexing="ij")
    model = env.sim.model
    model.geom_solref[rows, cols, 0] = time_constant[:, None].expand_as(rows).to(model.geom_solref.dtype)
    keep = torch.isnan(friction)
    if (~keep).any():
        r, c = rows[~keep], cols[~keep]
        model.geom_friction[r, c, 0] = friction[~keep][:, None].expand(-1, len(geoms)).to(model.geom_friction.dtype)


def add_sole_floors(cfg, per_env: list[tuple[float | None, float]], geoms: tuple[str, ...] = SOLES) -> None:
    """Give environment i of ``cfg`` the floor ``per_env[i]`` (friction or None, time constant s) at startup, and
    the plate of a ``Floor`` that has one (``add_judgment_plates``).

    The events run after the play env's startup events (foot_friction draws first) and draw nothing, so the
    other randomization of an environment is the one it has without them.
    """

    from mjlab.managers.event_manager import EventTermCfg, requires_model_fields
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    if len(per_env) != cfg.scene.num_envs:
        raise ValueError(f"{len(per_env)} floors for {cfg.scene.num_envs} environments")
    func = requires_model_fields("geom_friction", "geom_solref")(_sole_floor)
    cfg.events["judgment_floor"] = EventTermCfg(
        func=func,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=geoms),
            "frictions": [math.nan if f is None else float(f) for f, _ in per_env],
            "time_constants": [float(t) for _, t in per_env],
        },
    )
    add_judgment_plates(cfg, [getattr(floor, "plate", None) for floor in per_env])


def add_body_floors(cfg, per_env: list[tuple[float | None, float]]) -> None:
    """``add_sole_floors`` on every collision geom (BODY), on training's collision geoms: named, with priority over
    the floor (TRAINING_COLLISION), the play robot's actuators and everything else kept."""

    from dataclasses import replace

    from mjlab_microban.robot.microban_constants import TRAINING_COLLISION, collision_named_spec

    robot = cfg.scene.entities["robot"]
    cfg.scene.entities["robot"] = replace(robot, spec_fn=collision_named_spec, collisions=(TRAINING_COLLISION,))
    add_sole_floors(cfg, per_env, BODY)


def check_body_floors(env, per_env: list[tuple[float | None, float]]) -> None:
    """Raise unless every collision geom of environment i has the time constant of ``per_env[i]`` and, when it has
    one, its friction, with priority over the floor."""

    import torch

    robot = env.scene["robot"]
    local, _names = robot.find_geoms(BODY)
    geoms = robot.indexing.geom_ids[torch.tensor(local, device=env.device)].long()
    mj = env.sim.mj_model
    colliding = [int(g) for g in robot.indexing.geom_ids.tolist() if mj.geom_contype[g] or mj.geom_conaffinity[g]]
    model = env.sim.model
    want_t = torch.tensor([t for _, t in per_env], device=env.device)[:, None]
    want_f = torch.tensor([math.nan if f is None else f for f, _ in per_env], device=env.device)[:, None]
    time_constant = model.geom_solref[:, geoms, 0]
    friction = model.geom_friction[:, geoms, 0]
    set_f = ~torch.isnan(want_f).expand_as(friction)
    if sorted(geoms.tolist()) != sorted(colliding) or not bool((mj.geom_priority[colliding] == 1).all()) \
            or not torch.allclose(time_constant, want_t.expand_as(time_constant).to(time_constant.dtype), rtol=1e-5) \
            or not torch.allclose(friction[set_f], want_f.expand_as(friction)[set_f].to(friction.dtype), rtol=1e-5):
        raise RuntimeError(f"the floors on the {len(colliding)} collision geoms drifted from the ones set")


def sole_floors(env) -> list[tuple[float, float]]:
    """(friction, time constant) of every environment's first sole geom, as simulated."""

    import torch

    robot = env.scene["robot"]
    local, _names = robot.find_geoms(SOLES)
    geom = robot.indexing.geom_ids[torch.tensor(local[:1], device=env.device)].long()[0]
    model = env.sim.model
    return list(zip(model.geom_friction[:, geom, 0].tolist(), model.geom_solref[:, geom, 0].tolist(), strict=True))
