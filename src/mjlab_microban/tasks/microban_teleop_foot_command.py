"""PICO foot targets, some single-foot ones with a standing twist.

The foot-tracking reward fades out with the commanded speed
(mdp.foot_target_tracking_error_exp, 0 at 0.15 and above): a foot target
under a moving command trains only that walking ignores it.  Two-foot
targets always come with a standing twist (ResetFixedFootTargetCommand);
here a single-foot target does with ``single_support_stationary_probability``
too, drawn at each foot-target resample: the twist is held at zero while the
target lasts and restored when it ends.

Why: with single-foot targets independent of the twist, only about 3.5 % of
the samples train a single-foot target at full weight (30 % of the targets
times the standing and slow commands), too few to learn to lift a foot.
Holding the twist at zero gives the foot targets the standing time to learn
on (the walking reward makes the walker stand still on a standing command).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.envs import ManagerBasedRlEnv

from mjlab_microban.tasks.microban_teleop_mdp import (
    ResetFixedFootTargetCommand,
    ResetFixedFootTargetCommandCfg,
)


class StationaryFootTargetCommand(ResetFixedFootTargetCommand):
    """ResetFixedFootTargetCommand whose single-foot targets may stand too."""

    cfg: StationaryFootTargetCommandCfg

    def __init__(self, cfg: StationaryFootTargetCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        if not 0.0 <= cfg.single_support_stationary_probability <= 1.0:
            raise ValueError("single_support_stationary_probability must be in [0, 1]")
        self.is_stationary_single_support_env = torch.zeros_like(self.is_both_feet_env)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        draw = torch.rand(len(env_ids), device=self.device)
        self.is_stationary_single_support_env[env_ids] = self.is_single_support_env[env_ids] & (
            draw < self.cfg.single_support_stationary_probability
        )

    def _update_command(self) -> None:
        # The parent holds the twist at zero (and restores it) for its
        # two-foot rows; run it on the two-foot rows and the stationary
        # single-foot rows together.
        both_feet = self.is_both_feet_env
        self.is_both_feet_env = both_feet | self.is_stationary_single_support_env
        try:
            super()._update_command()
        finally:
            self.is_both_feet_env = both_feet


@dataclass(kw_only=True)
class StationaryFootTargetCommandCfg(ResetFixedFootTargetCommandCfg):
    single_support_stationary_probability: float = 0.0
    """Share of the single-foot targets that come with a standing twist."""

    def build(self, env: ManagerBasedRlEnv) -> StationaryFootTargetCommand:
        return StationaryFootTargetCommand(self, env)
