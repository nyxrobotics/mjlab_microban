"""Get-up position action: absolute-target clipping plus held neck joints.

No target-rate slew here. An earlier version of this file added one
(GETUP_TARGET_SLEW_RAD_S = 0.5, "the same target slew as the robot
controller"), but that 0.5 rad/s figure belongs to a completely different
part of the robot runtime: getup.py's _RECOVERY_SLEW_RATE_RAD_S, the
deliberately slow/gentle return-to-neutral used only when torque is
switching on from limp. The robot's own ACTIVE get-up policy (once
armed and actually executing) writes its clipped target directly, every
tick, with no rate limit at all -- confirmed against getup.py's own
step()/_step_recover_to_neutral() split. Modeling that same 0.5 rad/s cap
on the policy's own live output here made a real, physically-fast get-up
motion impossible to learn or even express: every joint was capped at
0.01 rad per 20 ms policy step regardless of what the network commanded,
which is why the trained checkpoint's motion looked passive/unwilling to
stand rather than merely imperfect.
"""

from __future__ import annotations

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

# Head and neck: not policy actions; held at their measured angles.
GETUP_NECK_JOINT_NAMES = ("head", "neck_roll", "neck_pitch")


class GetupJointPositionActionCfg(JointPositionActionCfg):
    """Absolute-target-clipped joint-position action, holding the neck at measured pose."""

    def build(self, env) -> GetupJointPositionAction:
        return GetupJointPositionAction(self, env)


class GetupJointPositionAction(JointPositionAction):
    """Write the clipped target directly every policy step; hold the neck joints."""

    cfg: GetupJointPositionActionCfg

    def __init__(self, cfg: GetupJointPositionActionCfg, env) -> None:
        super().__init__(cfg, env)
        neck_ids, neck_names = self._entity.find_joints(GETUP_NECK_JOINT_NAMES)
        if set(neck_names) != set(GETUP_NECK_JOINT_NAMES):
            raise ValueError("Get-up action requires all three head/neck joints")
        self._neck_target_ids = torch.tensor(neck_ids, dtype=torch.long, device=self.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        # Reset events have already written the randomized fallen joint state.
        # Seeding the target there avoids an instant jump to the default pose.
        self._processed_actions[env_ids] = self._entity.data.joint_pos[:, self._target_ids][env_ids]
        self._hold_measured_neck(env_ids)

    def process_actions(self, actions: torch.Tensor) -> None:
        # Base implementation applies scale, default-pose offset, then absolute
        # target clipping -- and nothing else. No rate limit: the real get-up
        # policy's own commanded target is written directly, every tick.
        super().process_actions(actions)
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


def raw_getup_action(env, action_name: str = "joint_pos") -> torch.Tensor:
    """Return the policy's own previous raw output (zero right after a reset).

    The get-up contract (v4-v6) observes this rather than the post-clip target.
    Every policy that stood up (including the one run on the robot) was
    trained observing its raw output; the post-clip variant never stood.
    The robot reproduces it exactly: it is the ONNX model's own last output.
    """
    action = env.action_manager.get_term(action_name)
    if not isinstance(action, GetupJointPositionAction):
        raise TypeError(f"{action_name!r} must be a get-up joint-position action")
    return action.raw_action
