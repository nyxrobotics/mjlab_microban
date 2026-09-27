"""Get-up position action with the same target slew as the robot controller."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

from mjlab_microban.tasks.microban_getup_actuator import GETUP_NECK_JOINT_NAMES


GETUP_TARGET_SLEW_RAD_S = 0.5


@dataclass(kw_only=True)
class SlewLimitedGetupJointPositionActionCfg(JointPositionActionCfg):
    """Clip the absolute policy target, then slew the actuator target toward it."""

    max_target_speed_rad_s: float = GETUP_TARGET_SLEW_RAD_S

    def __post_init__(self) -> None:
        super().__post_init__()
        if not math.isfinite(self.max_target_speed_rad_s) or self.max_target_speed_rad_s <= 0.0:
            raise ValueError("max_target_speed_rad_s must be finite and positive")

    def build(self, env) -> SlewLimitedGetupJointPositionAction:
        return SlewLimitedGetupJointPositionAction(self, env)


class SlewLimitedGetupJointPositionAction(JointPositionAction):
    """Advance the bounded target once per policy step, not per physics substep."""

    cfg: SlewLimitedGetupJointPositionActionCfg

    def __init__(self, cfg: SlewLimitedGetupJointPositionActionCfg, env) -> None:
        super().__init__(cfg, env)
        neck_ids, neck_names = self._entity.find_joints(GETUP_NECK_JOINT_NAMES)
        if set(neck_names) != set(GETUP_NECK_JOINT_NAMES):
            raise ValueError("Get-up action requires all three head/neck joints")
        self._neck_target_ids = torch.tensor(neck_ids, dtype=torch.long, device=self.device)
        # The first policy observation after an episode reset has no previous
        # policy action, even though the target state must begin at measured pose.
        self._has_processed_action = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        # Reset events have already written the randomized fallen joint state.
        # Seeding the target there avoids an instant jump to the default pose.
        self._processed_actions[env_ids] = self._entity.data.joint_pos[:, self._target_ids][env_ids]
        self._has_processed_action[env_ids] = False
        self._hold_measured_neck(env_ids)

    def process_actions(self, actions: torch.Tensor) -> None:
        previous = self._processed_actions
        # Base implementation applies scale, default-pose offset, then absolute
        # target clipping. This happens once per 20 ms policy step in mjlab.
        super().process_actions(actions)
        max_step = self.cfg.max_target_speed_rad_s * self._env.step_dt
        self._processed_actions = previous + torch.clamp(
            self._processed_actions - previous, min=-max_step, max=max_step
        )
        self._has_processed_action[:] = True
        self._hold_measured_neck(slice(None))

    def _hold_measured_neck(self, env_ids: torch.Tensor | slice) -> None:
        # Called once per 20 ms policy step, before mjlab's four physics
        # substeps. Holding measured pose here mirrors the runtime controller's
        # 50 Hz head/neck command without adding action dimensions to the actor.
        measured = self._entity.data.joint_pos[env_ids][:, self._neck_target_ids].clone()
        setter_ids = env_ids[:, None] if isinstance(env_ids, torch.Tensor) else env_ids
        self._entity.set_joint_position_target(
            measured, joint_ids=self._neck_target_ids, env_ids=setter_ids
        )

    @property
    def effective_previous_action(self) -> torch.Tensor:
        """Post-clip, post-slew target in default-relative policy coordinates."""
        effective = (self._processed_actions - self.offset) / self.scale
        return torch.where(self._has_processed_action, effective, torch.zeros_like(effective))


def effective_getup_action_after_target_slew(env, action_name: str = "joint_pos") -> torch.Tensor:
    """Return the action that actually reached the actuator target pipeline."""
    action = env.action_manager.get_term(action_name)
    if not isinstance(action, SlewLimitedGetupJointPositionAction):
        raise TypeError(f"{action_name!r} must be a slew-limited get-up joint-position action")
    return action.effective_previous_action
