"""Fail-closed runner for the pinned 9900->9999 corner-pair rescue."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from mjlab_microban.tasks.microban_teleop_env_cfg import (
    MICROBAN_TELEOP_HAND_TRACKING_FINAL_STD_M,
)
from mjlab_microban.tasks.microban_teleop_v12_actor import (
    TELEOP_V12_FOOT_OBSERVATION_COLUMNS,
    TELEOP_V12_TARGET_POSITION_NORMALIZER_STORED_STD,
)
from mjlab_microban.tasks.microban_teleop_v12_corner_rescue import (
    MICROBAN_TELEOP_V12_CORNER_RESCUE_ACTIVE_COLUMNS,
    MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY,
    MICROBAN_TELEOP_V12_CORNER_RESCUE_PARENT_COMMON_STEP,
    MICROBAN_TELEOP_V12_CORNER_RESCUE_PARENT_ITERATION,
    MICROBAN_TELEOP_V12_CORNER_RESCUE_PARENT_OPTIMIZER_STEP,
    MICROBAN_TELEOP_V12_CORNER_RESCUE_PARENT_SHA256,
    MICROBAN_TELEOP_V12_CORNER_RESCUE_PROCESS_UPDATES,
    MICROBAN_TELEOP_V12_CORNER_RESCUE_RECIPE_REVISION,
    MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_COMMON_STEP,
    MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_OPTIMIZER_STEP,
    CornerPairHandTargetCommand,
    assert_corner_rescue_foot_adapter_zero,
    assert_corner_rescue_optimizer_step,
    corner_rescue_marker,
)
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MICROBAN_TELEOP_V12_RECIPE_REVISION,
)
from mjlab_microban.tasks.microban_teleop_v12_runner import (
    MicrobanTeleopV12OnPolicyRunner,
)


def validate_corner_rescue_parent_payload(
    payload: dict[str, Any], *, checkpoint_sha256: str
) -> None:
    """Validate the only checkpoint from which this runner may resume."""

    if checkpoint_sha256 != MICROBAN_TELEOP_V12_CORNER_RESCUE_PARENT_SHA256:
        raise ValueError("Pinned corner rescue model9900 SHA-256 mismatch")
    iteration = payload.get("iter")
    infos = payload.get("infos")
    if iteration != MICROBAN_TELEOP_V12_CORNER_RESCUE_PARENT_ITERATION:
        raise ValueError("Corner rescue parent must be model_9900.pt")
    if not isinstance(infos, dict):
        raise TypeError("Corner rescue parent infos are missing")
    env_state = infos.get("env_state")
    if (
        not isinstance(env_state, dict)
        or env_state.get("common_step_counter")
        != MICROBAN_TELEOP_V12_CORNER_RESCUE_PARENT_COMMON_STEP
    ):
        raise ValueError("Corner rescue parent clock drifted")
    if infos.get("microban_teleop_recipe_revision") != (
        MICROBAN_TELEOP_V12_RECIPE_REVISION
    ):
        raise ValueError("Corner rescue parent is not the corrected canonical recipe")
    if infos.get("active_actor_columns_at_save") != list(
        MICROBAN_TELEOP_V12_CORNER_RESCUE_ACTIVE_COLUMNS
    ):
        raise ValueError("Corner rescue parent active columns drifted")
    assert_corner_rescue_optimizer_step(
        payload,
        expected_step=MICROBAN_TELEOP_V12_CORNER_RESCUE_PARENT_OPTIMIZER_STEP,
    )
    assert_corner_rescue_foot_adapter_zero(payload)


class MicrobanTeleopV12CornerRescueOnPolicyRunner(
    MicrobanTeleopV12OnPolicyRunner
):
    """Resume one exact parent for 99 updates and mark every resulting save."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._corner_rescue_marker: dict[str, Any] | None = None
        super().__init__(*args, **kwargs)

    def load(
        self,
        path: str | bytes,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        if isinstance(path, bytes):
            raise TypeError("Corner rescue training requires a filesystem checkpoint")
        resolved = Path(path).expanduser().resolve(strict=True)
        from mjlab_microban.tasks.microban_teleop_v12_bootstrap import sha256_file

        before = sha256_file(resolved)
        payload = torch.load(resolved, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError("Corner rescue parent payload is malformed")
        validate_corner_rescue_parent_payload(payload, checkpoint_sha256=before)
        loaded = super().load(
            str(resolved), load_cfg=load_cfg, strict=strict, map_location=map_location
        )
        if sha256_file(resolved) != before:
            raise ValueError("Corner rescue parent changed while loading")
        self._corner_rescue_marker = corner_rescue_marker()
        self._assert_corner_rescue_environment()
        self._assert_live_foot_adapter_zero()
        self._assert_live_optimizer_step(
            MICROBAN_TELEOP_V12_CORNER_RESCUE_PARENT_OPTIMIZER_STEP
        )
        return loaded

    def _contract_infos(self, infos: dict | None = None) -> dict:
        self._assert_corner_rescue_environment()
        result = super()._contract_infos(infos)
        marker = self._corner_rescue_marker
        if marker is None:
            raise RuntimeError("Corner rescue lineage marker is not bound")
        result["microban_teleop_recipe_revision"] = (
            MICROBAN_TELEOP_V12_CORNER_RESCUE_RECIPE_REVISION
        )
        result[MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY] = dict(marker)
        return result

    def _assert_live_foot_adapter_zero(self) -> None:
        first = self._actor.mlp[0].weight
        foot = first[:, TELEOP_V12_FOOT_OBSERVATION_COLUMNS]
        if not torch.equal(foot, torch.zeros_like(foot)):
            raise RuntimeError("Corner rescue activated a foot actor column")
        state = self.alg.optimizer.state.get(self._actor.mlp[0].weight, {})
        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            value = state.get(name)
            if value is None:
                continue
            locked = value[:, TELEOP_V12_FOOT_OBSERVATION_COLUMNS]
            if not torch.equal(locked, torch.zeros_like(locked)):
                raise RuntimeError(f"Corner rescue activated foot Adam {name}")
        normalizer = self._actor.obs_normalizer
        expected_std_values = TELEOP_V12_TARGET_POSITION_NORMALIZER_STORED_STD[:6]
        for name, expected_values in (
            ("_mean", (0.0,) * 6),
            ("_var", tuple(value * value for value in expected_std_values)),
            ("_std", expected_std_values),
        ):
            tensor = getattr(normalizer, name, None)
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Corner rescue normalizer {name} is missing")
            foot_tensor = tensor[:, TELEOP_V12_FOOT_OBSERVATION_COLUMNS]
            expected = foot_tensor.new_tensor(expected_values).unsqueeze(0)
            if not torch.equal(foot_tensor, expected):
                raise RuntimeError(f"Corner rescue foot normalizer {name} drifted")

    def _assert_live_optimizer_step(self, expected_step: int) -> None:
        assert_corner_rescue_optimizer_step(
            {"optimizer_state_dict": self.alg.optimizer.state_dict()},
            expected_step=expected_step,
        )

    def _assert_corner_rescue_environment(self) -> None:
        env = self.env.unwrapped
        common_step = int(env.common_step_counter)
        if not (
            MICROBAN_TELEOP_V12_CORNER_RESCUE_PARENT_COMMON_STEP
            <= common_step
            <= MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_COMMON_STEP
        ):
            raise RuntimeError("Corner rescue environment clock is outside its route")
        hand = env.command_manager.get_term("hand_target")
        foot = env.command_manager.get_term_cfg("foot_target")
        foot_command = env.command_manager.get_term("foot_target")
        rewards = env.reward_manager
        hand_reward = rewards.get_term_cfg("hand_target_tracking")
        if not isinstance(hand, CornerPairHandTargetCommand):
            raise TypeError("Corner rescue hand sampler was not installed")
        if (
            hand.cfg.rel_active != 0.7
            or foot.rel_single_support_envs != 0.0
            or foot.rel_both_feet_envs != 0.0
            or rewards.get_term_cfg("foot_target_tracking").weight != 1.0
            or hand_reward.weight != 2.0
            or hand_reward.params.get("std")
            != MICROBAN_TELEOP_HAND_TRACKING_FINAL_STD_M
        ):
            raise RuntimeError("Corner rescue reward/command contract drifted")
        command = getattr(foot_command, "foot_target_offset_b", None)
        single = getattr(foot_command, "is_single_support_env", None)
        both = getattr(foot_command, "is_both_feet_env", None)
        if (
            not isinstance(command, torch.Tensor)
            or tuple(command.shape) != (2_048, 2, 3)
            or not torch.equal(command, torch.zeros_like(command))
            or not isinstance(single, torch.Tensor)
            or tuple(single.shape) != (2_048,)
            or bool(single.any().item())
            or not isinstance(both, torch.Tensor)
            or tuple(both.shape) != (2_048,)
            or bool(both.any().item())
        ):
            raise RuntimeError(
                "Corner rescue foot command tensor/active masks are not exact zero"
            )
        if tuple(self._actor.active_adapter_columns()) != (
            MICROBAN_TELEOP_V12_CORNER_RESCUE_ACTIVE_COLUMNS
        ):
            raise RuntimeError("Corner rescue adapter columns drifted")

    def learn(
        self,
        num_learning_iterations: int,
        init_at_random_ep_len: bool = False,
    ) -> None:
        if num_learning_iterations != (
            MICROBAN_TELEOP_V12_CORNER_RESCUE_PROCESS_UPDATES
        ):
            raise ValueError("Corner rescue must run exactly 99 PPO updates")
        if self.current_learning_iteration != (
            MICROBAN_TELEOP_V12_CORNER_RESCUE_PARENT_ITERATION + 1
        ):
            raise ValueError("Corner rescue runner did not load the pinned model9900")
        self._assert_corner_rescue_environment()
        self._assert_live_foot_adapter_zero()
        self._assert_live_optimizer_step(
            MICROBAN_TELEOP_V12_CORNER_RESCUE_PARENT_OPTIMIZER_STEP
        )
        result = super().learn(num_learning_iterations, init_at_random_ep_len)
        if int(self.env.unwrapped.common_step_counter) != (
            MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_COMMON_STEP
        ):
            raise RuntimeError("Corner rescue did not stop at completed update 10000")
        self._assert_live_foot_adapter_zero()
        self._assert_live_optimizer_step(
            MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_OPTIMIZER_STEP
        )
        return result

    def save(self, path: str, infos=None) -> None:
        common_step = int(self.env.unwrapped.common_step_counter)
        if common_step > MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_COMMON_STEP:
            raise RuntimeError("Corner rescue may not save beyond update 10000")
        completed_updates = common_step // 24
        if common_step % 24:
            raise RuntimeError("Corner rescue may save only at an update boundary")
        expected_optimizer_step = (
            MICROBAN_TELEOP_V12_CORNER_RESCUE_PARENT_OPTIMIZER_STEP
            + (completed_updates - 9_901) * 20
        )
        if expected_optimizer_step < (
            MICROBAN_TELEOP_V12_CORNER_RESCUE_PARENT_OPTIMIZER_STEP
        ):
            raise RuntimeError("Corner rescue save clock precedes its fixed parent")
        if (
            common_step == MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_COMMON_STEP
            and expected_optimizer_step
            != MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_OPTIMIZER_STEP
        ):
            raise AssertionError("Corner rescue target optimizer formula drifted")
        self._assert_corner_rescue_environment()
        self._assert_live_foot_adapter_zero()
        self._assert_live_optimizer_step(expected_optimizer_step)
        super().save(path, infos)
