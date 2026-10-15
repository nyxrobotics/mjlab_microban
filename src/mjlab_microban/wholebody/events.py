"""Training randomization of the floor's softness."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_log_uniform

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@requires_model_fields("geom_solref")
def contact_time_constant(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    time_constant_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
) -> None:
    """One contact time constant (solref[0], s) per environment, drawn log-uniformly, on all of ``asset_cfg``'s geoms.

    MuJoCo's contact is a spring-damper of time constant solref[0] and damping ratio solref[1] (kept): its
    stiffness goes as 1 / solref[0]^2, so a geom's sink under a load about as solref[0]^2.  A floor contact
    takes the solref of the geom with the higher priority, here the robot's.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    geoms = env.scene[asset_cfg.name].indexing.geom_ids[asset_cfg.geom_ids]
    rows, cols = torch.meshgrid(env_ids.to(env.device, torch.long), geoms.to(torch.long), indexing="ij")
    time_constant = sample_log_uniform(*time_constant_range, (len(env_ids), 1), device=env.device)
    env.sim.model.geom_solref[rows, cols, 0] = time_constant.expand_as(rows)
