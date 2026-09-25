"""Microban-specific observation helpers for motion tracking.

MjLab's generic tracking command carries the complete reference articulation.  The
three camera-neck joints are intentionally controlled by the PICO HMD outside the
body policy, so the actor and critic must only receive the 18 body-joint portion of
that command.  Keeping this projection here also makes the deployment observation
contract explicit and deterministic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.tracking.mdp import MotionCommand
from rsl_rl.algorithms import PPO

from mjlab_microban.tasks.microban_teleop_mdp import (
    AsymmetricBoundedGaussianDistribution,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


MICROBAN_TRACKING_ACTION_WIDTH = 18
MICROBAN_TRACKING_ARM_ACTION_IDS: tuple[int, ...] = (0, 1, 2, 9, 10, 11)


class MicrobanTrackingBoundedGaussianDistribution(
    AsymmetricBoundedGaussianDistribution
):
    """Bounded tracking distribution with a neutral arm initialization.

    The fixed walking reference keeps all six arm joints at home.  Zeroing the
    six arm output rows prevents clean training from spending its earliest
    rollouts on arbitrary arm motion.  This deliberately replaces the parent
    class' teleop-specific shoulder-roll bias: that bias represents a large
    inward arm move in the tracking coordinate system, whereas this fixed gait
    keeps both shoulder-roll joints exactly at home.
    """

    def init_mlp_weights(self, mlp: torch.nn.Module) -> None:
        super().init_mlp_weights(mlp)
        linear_layers = [
            module for module in mlp.modules() if isinstance(module, torch.nn.Linear)
        ]
        if not linear_layers:
            raise ValueError("Tracking actor MLP must contain a linear output layer")
        final_layer = linear_layers[-1]
        if (
            final_layer.out_features != MICROBAN_TRACKING_ACTION_WIDTH
            or final_layer.bias is None
        ):
            raise ValueError("Tracking actor final layer must have 18 biased rows")
        ids = list(MICROBAN_TRACKING_ARM_ACTION_IDS)
        with torch.no_grad():
            final_layer.weight[ids] = 0.0
            final_layer.bias[ids] = 0.0


class MicrobanTrackingBoundedPPO(PPO):
    """Store PPO latents while stepping bounded physical joint actions."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        distribution = self.actor.distribution
        if not isinstance(distribution, MicrobanTrackingBoundedGaussianDistribution):
            raise TypeError(
                "MicrobanTrackingBoundedPPO requires its bounded distribution"
            )
        if distribution.output_dim != MICROBAN_TRACKING_ACTION_WIDTH:
            raise ValueError("Microban tracking requires exactly 18 actions")
        if self.symmetry is not None:
            raise ValueError("MicrobanTrackingBoundedPPO does not support symmetry")
        if self.rnd is not None:
            raise ValueError("MicrobanTrackingBoundedPPO does not support RND")
        distribution.project_std_parameters_()
        self._std_projection_hook_handle = self.optimizer.register_step_post_hook(
            self._project_std_after_optimizer_step
        )

    def _project_std_after_optimizer_step(
        self,
        _optimizer: torch.optim.Optimizer,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
    ) -> None:
        distribution = self.actor.distribution
        if not isinstance(distribution, MicrobanTrackingBoundedGaussianDistribution):
            raise TypeError("Tracking actor lost its bounded distribution")
        distribution.project_std_parameters_()

    def act(self, obs: Any) -> torch.Tensor:
        latent = super().act(obs)
        distribution = self.actor.distribution
        assert isinstance(distribution, MicrobanTrackingBoundedGaussianDistribution)
        return distribution.to_environment_action(latent)


def controlled_motion_command(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return reference position/velocity for policy-controlled joints only.

    ``MotionCommand`` still loads and resets the complete 21-joint Microban state;
    the NPZ therefore remains compatible with MjLab's standard motion format.  This
    observation removes ``head``, ``neck_roll`` and ``neck_pitch`` from the policy
    input in exactly the same resolved joint order used by the action term.
    """

    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    joint_ids = asset_cfg.joint_ids
    return torch.cat(
        (command.joint_pos[:, joint_ids], command.joint_vel[:, joint_ids]), dim=-1
    )


def motion_anchor_planar_position_error_l1(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    """Return non-saturating horizontal root-position error.

    The generic exponential tracking reward becomes effectively zero once this
    small robot falls a few body lengths behind the reference.  A stationary
    policy can then optimize only the relative-pose rewards.  Keeping the
    horizontal error linear preserves a learning signal all the way through a
    walking clip and makes that stationary solution strictly costly.
    """

    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    error_xy = command.anchor_pos_w[:, :2] - command.robot_anchor_pos_w[:, :2]
    return torch.abs(error_xy).sum(dim=-1)


def motion_anchor_planar_velocity_error_l1(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    """Return non-saturating horizontal root-velocity tracking error."""

    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    error_xy = command.anchor_lin_vel_w[:, :2] - command.robot_anchor_lin_vel_w[:, :2]
    return torch.abs(error_xy).sum(dim=-1)
