"""Fail-closed runner for the recorded-parent 9900->9999 corner-pair rescue."""

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
    MICROBAN_TELEOP_V12_CORNER_RESCUE_PARENT_REPORT_FILENAME,
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

    if (
        not isinstance(checkpoint_sha256, str)
        or len(checkpoint_sha256) != 64
        or any(c not in "0123456789abcdef" for c in checkpoint_sha256)
    ):
        raise ValueError("Corner rescue model9900 SHA-256 is malformed")
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


def validate_corner_rescue_parent_report(
    report_path: Path, *, checkpoint_sha256: str, iteration: int
) -> tuple[str, list[str]]:
    """Validate the parent's strict tracking report and return its SHA-256.

    The report must be the canonical HMD/hand-profile report of exactly this
    parent and its only failing check must be ``hand_tracking_rms``.
    """

    from mjlab_microban.scripts.evaluate_teleop_v12_tracking import HMD_HAND_PROFILE
    from mjlab_microban.scripts.teleop_v12_stage import (
        _load_json,
        _validate_tracking_report,
    )
    from mjlab_microban.tasks.microban_teleop_v12_bootstrap import sha256_file

    resolved = Path(report_path).expanduser().resolve(strict=True)
    if not resolved.is_file() or Path(report_path).is_symlink():
        raise ValueError("Corner rescue parent report must be a regular file")
    before = sha256_file(resolved)
    report = _load_json(resolved)
    _validate_tracking_report(
        report,
        {
            "sha256": checkpoint_sha256,
            "iteration": iteration,
            "completed_updates": iteration + 1,
        },
        profile_override=HMD_HAND_PROFILE,
        allowed_failed_checks=frozenset(("hand_tracking_rms",)),
    )
    failed = sorted(name for name, passed in report["checks"].items() if not passed)
    if report.get("status") != "fail" or failed != ["hand_tracking_rms"]:
        raise ValueError(
            "Corner rescue parent report must fail only the strict hand RMS check"
        )
    if sha256_file(resolved) != before:
        raise ValueError("Corner rescue parent report changed while validating")
    return before, failed


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
        report_path = resolved.parent / (
            MICROBAN_TELEOP_V12_CORNER_RESCUE_PARENT_REPORT_FILENAME
        )
        report_sha256, _ = validate_corner_rescue_parent_report(
            report_path,
            checkpoint_sha256=before,
            iteration=MICROBAN_TELEOP_V12_CORNER_RESCUE_PARENT_ITERATION,
        )
        loaded = super().load(
            str(resolved), load_cfg=load_cfg, strict=strict, map_location=map_location
        )
        if sha256_file(resolved) != before or sha256_file(report_path) != (
            report_sha256
        ):
            raise ValueError("Corner rescue parent or its report changed while loading")
        self._corner_rescue_marker = corner_rescue_marker(
            parent_checkpoint_sha256=before,
            parent_strict_tracking_report_sha256=report_sha256,
        )
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


# ---------------------------------------------------------------------------
# Active-hand arm pose-release variant (fresh pose-release chain, model_9900).
# ---------------------------------------------------------------------------


def validate_hand_pose_release_corner_rescue_parent_payload(
    payload: dict[str, Any], *, checkpoint_sha256: str
) -> None:
    """The pose-release twin of ``validate_corner_rescue_parent_payload``.

    The parent is an unmarked fresh pose-release chain's model_9900 (no recipe
    switch and no rescue marker); every clock/column/Adam/foot check is the
    canonical rescue's.
    """

    from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
        MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION,
    )
    from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release_lineage import (
        HAND_POSE_RELEASE_LINEAGE_FRESH,
        hand_pose_release_lineage,
    )

    infos = payload.get("infos")
    if not isinstance(infos, dict):
        raise TypeError("Corner rescue parent infos are missing")
    if infos.get("microban_teleop_recipe_revision") != (
        MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION
    ):
        raise ValueError("Pose-release corner rescue parent is not the pose-release recipe")
    if infos.get(MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY) is not None:
        raise ValueError("Pose-release corner rescue parent already carries a rescue")
    if (
        hand_pose_release_lineage(
            infos,
            iteration=MICROBAN_TELEOP_V12_CORNER_RESCUE_PARENT_ITERATION,
            verify_parent=False,
        )
        != HAND_POSE_RELEASE_LINEAGE_FRESH
    ):
        raise ValueError("Pose-release corner rescue parent must be a fresh chain")
    canonical_view = dict(payload)
    canonical_view["infos"] = {
        **infos,
        "microban_teleop_recipe_revision": MICROBAN_TELEOP_V12_RECIPE_REVISION,
    }
    validate_corner_rescue_parent_payload(
        canonical_view, checkpoint_sha256=checkpoint_sha256
    )


def validate_hand_pose_release_corner_rescue_parent_report(
    report_path: Path, *, checkpoint_sha256: str, iteration: int
) -> tuple[str, list[str]]:
    """Strict HMD/hand report of the parent failing only hand accuracy checks."""

    from mjlab_microban.scripts.evaluate_teleop_v12_tracking import HMD_HAND_PROFILE
    from mjlab_microban.scripts.teleop_v12_stage import (
        _load_json,
        _validate_tracking_report,
    )
    from mjlab_microban.tasks.microban_teleop_v12_bootstrap import sha256_file
    from mjlab_microban.tasks.microban_teleop_v12_corner_rescue import (
        MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_CORNER_RESCUE_PARENT_FAILED_CHECKS,
    )

    resolved = Path(report_path).expanduser().resolve(strict=True)
    if not resolved.is_file() or Path(report_path).is_symlink():
        raise ValueError("Corner rescue parent report must be a regular file")
    before = sha256_file(resolved)
    report = _load_json(resolved)
    _validate_tracking_report(
        report,
        {
            "sha256": checkpoint_sha256,
            "iteration": iteration,
            "completed_updates": iteration + 1,
        },
        profile_override=HMD_HAND_PROFILE,
        allowed_failed_checks=(
            MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_CORNER_RESCUE_PARENT_FAILED_CHECKS
        ),
    )
    failed = sorted(name for name, passed in report["checks"].items() if not passed)
    if (
        report.get("status") != "fail"
        or not failed
        or not set(failed)
        <= MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_CORNER_RESCUE_PARENT_FAILED_CHECKS
    ):
        raise ValueError(
            "Pose-release corner rescue parent report must fail only strict hand "
            "accuracy checks"
        )
    if sha256_file(resolved) != before:
        raise ValueError("Corner rescue parent report changed while validating")
    return before, failed


from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release_runner import (  # noqa: E402
    MicrobanTeleopV12HandPoseReleaseOnPolicyRunner,
)


class MicrobanTeleopV12HandPoseReleaseCornerRescueOnPolicyRunner(
    MicrobanTeleopV12HandPoseReleaseOnPolicyRunner
):
    """99-update 5/60/35 corner replay from a fresh pose-release model_9900.

    Saves keep the pose-release recipe revision; the base runner writes the
    pose-release corner marker (bound here after the parent is validated) into
    every save, and ordinary pose-release resumes carry it forward.
    """

    def load(
        self,
        path: str | bytes,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        if isinstance(path, bytes):
            raise TypeError("Corner rescue training requires a filesystem checkpoint")
        from mjlab_microban.tasks.microban_teleop_v12_bootstrap import sha256_file

        resolved = Path(path).expanduser().resolve(strict=True)
        before = sha256_file(resolved)
        payload = torch.load(resolved, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError("Corner rescue parent payload is malformed")
        validate_hand_pose_release_corner_rescue_parent_payload(
            payload, checkpoint_sha256=before
        )
        report_path = resolved.parent / (
            MICROBAN_TELEOP_V12_CORNER_RESCUE_PARENT_REPORT_FILENAME
        )
        report_sha256, failed = validate_hand_pose_release_corner_rescue_parent_report(
            report_path,
            checkpoint_sha256=before,
            iteration=MICROBAN_TELEOP_V12_CORNER_RESCUE_PARENT_ITERATION,
        )
        loaded = super().load(
            str(resolved), load_cfg=load_cfg, strict=strict, map_location=map_location
        )
        if sha256_file(resolved) != before or sha256_file(report_path) != (
            report_sha256
        ):
            raise ValueError("Corner rescue parent or its report changed while loading")
        if self.teleop_v12_corner_rescue is not None:
            raise RuntimeError("Pose-release corner rescue parent carried a rescue")
        self.teleop_v12_corner_rescue = corner_rescue_marker(
            parent_checkpoint_sha256=before,
            parent_strict_tracking_report_sha256=report_sha256,
            hand_pose_release=True,
            parent_strict_failed_checks=failed,
        )
        self._assert_corner_rescue_environment()
        self._assert_live_foot_adapter_zero()
        self._assert_live_optimizer_step(
            MICROBAN_TELEOP_V12_CORNER_RESCUE_PARENT_OPTIMIZER_STEP
        )
        return loaded

    def _contract_infos(self, infos: dict | None = None) -> dict:
        self._assert_corner_rescue_environment()
        if self.teleop_v12_corner_rescue is None:
            raise RuntimeError("Corner rescue lineage marker is not bound")
        result = super()._contract_infos(infos)
        if (
            result.get(MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY)
            != self.teleop_v12_corner_rescue
        ):
            raise RuntimeError("Pose-release corner rescue marker was not saved")
        return result

    # Shared invariant checks (no super() calls inside).
    _assert_live_foot_adapter_zero = (
        MicrobanTeleopV12CornerRescueOnPolicyRunner._assert_live_foot_adapter_zero
    )
    _assert_live_optimizer_step = (
        MicrobanTeleopV12CornerRescueOnPolicyRunner._assert_live_optimizer_step
    )

    def _assert_corner_rescue_environment(self) -> None:
        from mjlab_microban.tasks.microban_teleop_v12_corner_rescue import (
            MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_LF_RB_PROBABILITY,
        )

        MicrobanTeleopV12CornerRescueOnPolicyRunner._assert_corner_rescue_environment(
            self
        )
        hand = self.env.unwrapped.command_manager.get_term("hand_target")
        if hand.cfg.lf_rb_probability != (
            MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_LF_RB_PROBABILITY
        ):
            raise RuntimeError("Pose-release corner rescue sampler mix drifted")
        self._assert_hand_pose_release_environment()

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
            raise ValueError("Corner rescue runner did not load model9900")
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
        if common_step % 24:
            raise RuntimeError("Corner rescue may save only at an update boundary")
        completed_updates = common_step // 24
        expected_optimizer_step = (
            MICROBAN_TELEOP_V12_CORNER_RESCUE_PARENT_OPTIMIZER_STEP
            + (completed_updates - 9_901) * 20
        )
        if expected_optimizer_step < (
            MICROBAN_TELEOP_V12_CORNER_RESCUE_PARENT_OPTIMIZER_STEP
        ):
            raise RuntimeError("Corner rescue save clock precedes its fixed parent")
        self._assert_corner_rescue_environment()
        self._assert_live_foot_adapter_zero()
        self._assert_live_optimizer_step(expected_optimizer_step)
        super().save(path, infos)
