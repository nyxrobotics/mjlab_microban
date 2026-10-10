"""PICO foot targets, some single-foot ones with a standing twist.

The foot-tracking reward fades out with the commanded speed
(mdp.foot_target_tracking_error_exp, 0 at 0.01 and above): a foot target
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

The published target moves as the teleop moves it (L3): each foot's target
approaches the drawn one at ``slew_m_s`` (3-D distance per step), and a
target at or below the 2.5 mm floor band reads exactly (0, 0, 0), in that
order (the teleop limits the speed, then the wire drops the floor band).
While both published feet are non-zero the twist is held at zero, so the
policy never sees two lifted feet with a moving command (the robot stops on
that input).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.envs import ManagerBasedRlEnv

from mjlab_microban.tasks.microban_teleop_mdp import (
    MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M,
    ResetFixedFootTargetCommand,
    ResetFixedFootTargetCommandCfg,
)

# The teleop's foot-target speed limit (microban_teleop mapping.py).
MICROBAN_TELEOP_FOOT_TARGET_SLEW_M_S = 0.12


class StationaryFootTargetCommand(ResetFixedFootTargetCommand):
    """ResetFixedFootTargetCommand whose single-foot targets may stand too."""

    cfg: StationaryFootTargetCommandCfg

    def __init__(self, cfg: StationaryFootTargetCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        if not 0.0 <= cfg.single_support_stationary_probability <= 1.0:
            raise ValueError("single_support_stationary_probability must be in [0, 1]")
        if not cfg.slew_m_s > 0.0:
            raise ValueError("slew_m_s must be positive")
        self.is_stationary_single_support_env = torch.zeros_like(self.is_both_feet_env)
        # The drawn target and the slewed one (before the floor band);
        # foot_target_offset_b is what the policy and the reward see.
        self.foot_target_goal_b = torch.zeros_like(self.foot_target_offset_b)
        self.foot_target_slewed_b = torch.zeros_like(self.foot_target_offset_b)
        self._max_step_m = cfg.slew_m_s * env.step_dt

    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        extras = super().reset(env_ids)
        self.foot_target_slewed_b[env_ids] = 0.0
        self.foot_target_offset_b[env_ids] = 0.0
        return extras

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        published = self.foot_target_offset_b[env_ids].clone()
        super()._resample_command(env_ids)
        self.foot_target_goal_b[env_ids] = self.foot_target_offset_b[env_ids]
        self.foot_target_offset_b[env_ids] = published
        draw = torch.rand(len(env_ids), device=self.device)
        self.is_stationary_single_support_env[env_ids] = self.is_single_support_env[env_ids] & (
            draw < self.cfg.single_support_stationary_probability
        )

    def hold_targets(self, env_ids: torch.Tensor, offsets: torch.Tensor) -> None:
        """Set the drawn, slewed and published targets of ``env_ids`` at once."""

        self.foot_target_goal_b[env_ids] = offsets
        self.foot_target_slewed_b[env_ids] = offsets
        self.foot_target_offset_b[env_ids] = self._floor_band(offsets)

    @staticmethod
    def _floor_band(offsets: torch.Tensor) -> torch.Tensor:
        lifted = offsets[..., 2:3] > MICROBAN_TELEOP_FOOT_INACTIVE_Z_MAX_M
        return torch.where(lifted, offsets, torch.zeros_like(offsets))

    def _update_command(self) -> None:
        delta = self.foot_target_goal_b - self.foot_target_slewed_b
        distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        scale = torch.clamp(self._max_step_m / distance.clamp(min=1.0e-12), max=1.0)
        self.foot_target_slewed_b += delta * scale
        self.foot_target_offset_b[:] = self._floor_band(self.foot_target_slewed_b)
        # The parent holds the twist at zero (and restores it) for its
        # two-foot rows; run it on the two-foot rows, the stationary
        # single-foot rows and every row whose two published feet are both
        # non-zero (a target still on its way down) together.
        both_published = (self.foot_target_offset_b != 0.0).any(dim=-1).all(dim=-1)
        both_feet = self.is_both_feet_env
        self.is_both_feet_env = both_feet | self.is_stationary_single_support_env | both_published
        try:
            super()._update_command()
        finally:
            self.is_both_feet_env = both_feet


@dataclass(kw_only=True)
class StationaryFootTargetCommandCfg(ResetFixedFootTargetCommandCfg):
    single_support_stationary_probability: float = 0.0
    """Share of the single-foot targets that come with a standing twist."""
    slew_m_s: float = MICROBAN_TELEOP_FOOT_TARGET_SLEW_M_S
    """Speed (3-D, per foot) at which the published target follows the drawn one."""

    def build(self, env: ManagerBasedRlEnv) -> StationaryFootTargetCommand:
        return StationaryFootTargetCommand(self, env)
