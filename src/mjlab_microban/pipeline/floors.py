"""The walking judgment's soft and slippery floors, one per environment.

A floor is one contact time constant (solref[0], s) and one friction of the
soles.  The play robot's soles have priority over the floor (robot/
microban_constants.py FULL_COLLISION), so a sole contact takes the soles'
values, as training's floors do on every collision geom (tasks/mdp.py
training_floor_and_delay).  A friction of None keeps the play env's drawn sole
friction (foot_friction, 0.3-1.2).  The rigid floor is MuJoCo's 0.02 s, the
play env's; config/pipeline.yaml ``walk.judgment.floors`` names the judged
floors, inside training's range (FLOOR_TIME_CONSTANT_S, FLOOR_FRICTION).
"""

from __future__ import annotations

import math
from typing import Any

SOLES = (r".*left_foot_collision.*", r".*right_foot_collision.*")
RIGID_TIME_CONSTANT_S = 0.02


def parse(floors: dict[str, dict[str, Any]]) -> dict[str, tuple[float | None, float]]:
    """{name: (friction or None, time constant s)} of config/pipeline.yaml's floors, checked against training's range."""

    from mjlab_microban.robot.microban_constants import FLOOR_FRICTION, FLOOR_TIME_CONSTANT_S

    parsed = {}
    for name, floor in floors.items():
        if set(floor) - {"friction", "time_constant_s"}:
            raise ValueError(f"floor {name}: unknown keys {sorted(set(floor) - {'friction', 'time_constant_s'})}")
        friction = floor.get("friction")
        time_constant = float(floor.get("time_constant_s", RIGID_TIME_CONSTANT_S))
        if not FLOOR_TIME_CONSTANT_S[0] <= time_constant <= FLOOR_TIME_CONSTANT_S[1]:
            raise ValueError(f"floor {name}: time constant {time_constant} s outside training's {FLOOR_TIME_CONSTANT_S}")
        if friction is not None and not FLOOR_FRICTION[0] <= float(friction) <= FLOOR_FRICTION[1]:
            raise ValueError(f"floor {name}: friction {friction} outside training's {FLOOR_FRICTION}")
        parsed[name] = (None if friction is None else float(friction), time_constant)
    return parsed


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


def add_sole_floors(cfg, per_env: list[tuple[float | None, float]]) -> None:
    """Give environment i of ``cfg`` the floor ``per_env[i]`` (friction or None, time constant s) at startup.

    The event runs after the play env's startup events (foot_friction draws first) and draws nothing, so the
    other randomization of an environment is the one it has without it.
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
            "asset_cfg": SceneEntityCfg("robot", geom_names=SOLES),
            "frictions": [math.nan if f is None else float(f) for f, _ in per_env],
            "time_constants": [float(t) for _, t in per_env],
        },
    )


def sole_floors(env) -> list[tuple[float, float]]:
    """(friction, time constant) of every environment's first sole geom, as simulated."""

    import torch

    robot = env.scene["robot"]
    local, _names = robot.find_geoms(SOLES)
    geom = robot.indexing.geom_ids[torch.tensor(local[:1], device=env.device)].long()[0]
    model = env.sim.model
    return list(zip(model.geom_friction[:, geom, 0].tolist(), model.geom_solref[:, geom, 0].tolist(), strict=True))
