"""Fail-closed runner for the recorded-parent 14900->14999 final-scenario rescue."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from mjlab_microban.tasks.microban_teleop_env_cfg import (
    MICROBAN_TELEOP_FOOT_TRACKING_FINAL_STD_M,
    MICROBAN_TELEOP_HAND_TRACKING_FINAL_STD_M,
)
from mjlab_microban.tasks.microban_teleop_v12_corner_rescue import (
    MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY,
    canonical_json_sha256,
    validate_corner_rescue_canonical_lineage,
)
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MICROBAN_TELEOP_V12_RECIPE_REVISION,
)
from mjlab_microban.tasks.microban_teleop_v12_final_rescue import (
    MICROBAN_TELEOP_V12_FINAL_RESCUE_ACTIVE_COLUMNS,
    MICROBAN_TELEOP_V12_FINAL_RESCUE_FAILED_GATE_REPORT_FILENAME,
    MICROBAN_TELEOP_V12_FINAL_RESCUE_INFO_KEY,
    MICROBAN_TELEOP_V12_FINAL_RESCUE_NUM_STEPS_PER_ENV,
    MICROBAN_TELEOP_V12_FINAL_RESCUE_OPTIMIZER_STEPS_PER_UPDATE,
    MICROBAN_TELEOP_V12_FINAL_RESCUE_PARENT_COMMON_STEP,
    MICROBAN_TELEOP_V12_FINAL_RESCUE_PARENT_COMPLETED_UPDATES,
    MICROBAN_TELEOP_V12_FINAL_RESCUE_PARENT_ITERATION,
    MICROBAN_TELEOP_V12_FINAL_RESCUE_PARENT_OPTIMIZER_STEP,
    MICROBAN_TELEOP_V12_FINAL_RESCUE_PARENT_REPORT_FILENAME,
    MICROBAN_TELEOP_V12_FINAL_RESCUE_PROCESS_UPDATES,
    MICROBAN_TELEOP_V12_FINAL_RESCUE_RECIPE_REVISION,
    MICROBAN_TELEOP_V12_FINAL_RESCUE_RESCUABLE_CHECKS,
    MICROBAN_TELEOP_V12_FINAL_RESCUE_TARGET_COMMON_STEP,
    MICROBAN_TELEOP_V12_FINAL_RESCUE_TARGET_COMPLETED_UPDATES,
    MICROBAN_TELEOP_V12_FINAL_RESCUE_TARGET_ITERATION,
    MICROBAN_TELEOP_V12_FINAL_RESCUE_TARGET_OPTIMIZER_STEP,
    FinalRescueFootTargetCommand,
    FinalRescueHandTargetCommand,
    FinalRescueTwistCommand,
    assert_final_rescue_optimizer_step,
    final_rescue_marker,
)
from mjlab_microban.tasks.microban_teleop_v12_runner import (
    MicrobanTeleopV12OnPolicyRunner,
)


def validate_final_rescue_parent_payload(
    payload: dict[str, Any], *, checkpoint_sha256: str
) -> dict[str, Any] | None:
    """Validate the only checkpoint from which this runner may resume.

    Returns the inherited corner-rescue marker (or ``None``).
    """

    if (
        not isinstance(checkpoint_sha256, str)
        or len(checkpoint_sha256) != 64
        or any(c not in "0123456789abcdef" for c in checkpoint_sha256)
    ):
        raise ValueError("Final rescue model14900 SHA-256 is malformed")
    iteration = payload.get("iter")
    infos = payload.get("infos")
    if iteration != MICROBAN_TELEOP_V12_FINAL_RESCUE_PARENT_ITERATION:
        raise ValueError("Final rescue parent must be model_14900.pt")
    if not isinstance(infos, dict):
        raise TypeError("Final rescue parent infos are missing")
    env_state = infos.get("env_state")
    if (
        not isinstance(env_state, dict)
        or env_state.get("common_step_counter")
        != MICROBAN_TELEOP_V12_FINAL_RESCUE_PARENT_COMMON_STEP
    ):
        raise ValueError("Final rescue parent clock drifted")
    if infos.get("microban_teleop_recipe_revision") != (
        MICROBAN_TELEOP_V12_RECIPE_REVISION
    ):
        raise ValueError("Final rescue parent is not the canonical recipe")
    if infos.get(MICROBAN_TELEOP_V12_FINAL_RESCUE_INFO_KEY) is not None:
        raise ValueError("Final rescue parent already carries a final rescue")
    if infos.get("active_actor_columns_at_save") != list(
        MICROBAN_TELEOP_V12_FINAL_RESCUE_ACTIVE_COLUMNS
    ):
        raise ValueError("Final rescue parent active columns drifted")
    from mjlab_microban.tasks.microban_teleop_v12_deadline_fallback import (
        MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_INFO_KEY,
    )

    if infos.get(MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_INFO_KEY) is not None:
        raise ValueError("Final rescue parent cannot carry deadline-fallback lineage")
    assert_final_rescue_optimizer_step(
        payload,
        expected_step=MICROBAN_TELEOP_V12_FINAL_RESCUE_PARENT_OPTIMIZER_STEP,
    )
    return validate_corner_rescue_canonical_lineage(infos, iteration=iteration)


def _validate_final_profile_report(
    report_path: Path, *, identity: dict[str, int | str], require_failure: bool
) -> tuple[str, list[str]]:
    from mjlab_microban.scripts.evaluate_teleop_v12_tracking import (
        required_tracking_profile,
    )
    from mjlab_microban.scripts.teleop_v12_stage import (
        _load_json,
        _validate_tracking_report,
    )
    from mjlab_microban.tasks.microban_teleop_v12_bootstrap import sha256_file

    resolved = Path(report_path).expanduser().resolve(strict=True)
    if not resolved.is_file() or Path(report_path).is_symlink():
        raise ValueError("Final rescue report must be a regular file")
    before = sha256_file(resolved)
    report = _load_json(resolved)
    _validate_tracking_report(
        report,
        identity,
        # Canonical profile of the report's own clock (whole body for the
        # parent at 14901, final for the failed 15000 gate).
        profile_override=required_tracking_profile(
            int(identity["completed_updates"])
        ),
        allowed_failed_checks=MICROBAN_TELEOP_V12_FINAL_RESCUE_RESCUABLE_CHECKS,
    )
    failed = sorted(name for name, passed in report["checks"].items() if not passed)
    if require_failure and (report.get("status") != "fail" or not failed):
        raise ValueError("The failed final gate report must fail an accuracy check")
    if sha256_file(resolved) != before:
        raise ValueError("Final rescue report changed while validating")
    return before, failed


def validate_final_rescue_parent_report(
    report_path: Path, *, checkpoint_sha256: str
) -> tuple[str, list[str]]:
    """The parent's canonical (whole-body) report: safety passes, accuracy may fail."""

    return _validate_final_profile_report(
        report_path,
        identity={
            "sha256": checkpoint_sha256,
            "iteration": MICROBAN_TELEOP_V12_FINAL_RESCUE_PARENT_ITERATION,
            "completed_updates": (
                MICROBAN_TELEOP_V12_FINAL_RESCUE_PARENT_COMPLETED_UPDATES
            ),
        },
        require_failure=False,
    )


def validate_failed_final_gate_report(report_path: Path) -> tuple[str, str, list[str]]:
    """The failed 15000 gate report that triggered the rescue.

    Returns (report SHA-256, its checkpoint SHA-256, failed checks).
    """

    from mjlab_microban.scripts.teleop_v12_stage import _load_json

    report = _load_json(Path(report_path).expanduser().resolve(strict=True))
    identity = report.get("checkpoint")
    checkpoint_sha256 = identity.get("sha256") if isinstance(identity, dict) else None
    if not isinstance(checkpoint_sha256, str):
        raise ValueError("Failed final gate report has no checkpoint identity")
    report_sha256, failed = _validate_final_profile_report(
        report_path,
        identity={
            "sha256": checkpoint_sha256,
            "iteration": MICROBAN_TELEOP_V12_FINAL_RESCUE_TARGET_ITERATION,
            "completed_updates": (
                MICROBAN_TELEOP_V12_FINAL_RESCUE_TARGET_COMPLETED_UPDATES
            ),
        },
        require_failure=True,
    )
    return report_sha256, checkpoint_sha256, failed


class MicrobanTeleopV12FinalRescueOnPolicyRunner(MicrobanTeleopV12OnPolicyRunner):
    """Resume one exact model_14900 for 99 updates and mark every save."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._final_rescue_marker: dict[str, Any] | None = None
        super().__init__(*args, **kwargs)

    def _live_mix(self) -> str:
        env = self.env.unwrapped
        mixes = {
            env.command_manager.get_term(name).cfg.final_rescue_mix
            for name in ("twist", "foot_target", "hand_target")
        }
        if len(mixes) != 1:
            raise RuntimeError("Final rescue command terms disagree on the mix")
        return mixes.pop()

    def load(
        self,
        path: str | bytes,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        if isinstance(path, bytes):
            raise TypeError("Final rescue training requires a filesystem checkpoint")
        resolved = Path(path).expanduser().resolve(strict=True)
        from mjlab_microban.tasks.microban_teleop_v12_bootstrap import sha256_file

        before = sha256_file(resolved)
        payload = torch.load(resolved, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError("Final rescue parent payload is malformed")
        corner = validate_final_rescue_parent_payload(
            payload, checkpoint_sha256=before
        )
        parent_report = resolved.parent / (
            MICROBAN_TELEOP_V12_FINAL_RESCUE_PARENT_REPORT_FILENAME
        )
        failed_report = resolved.parent / (
            MICROBAN_TELEOP_V12_FINAL_RESCUE_FAILED_GATE_REPORT_FILENAME
        )
        parent_report_sha, parent_failed = validate_final_rescue_parent_report(
            parent_report, checkpoint_sha256=before
        )
        failed_report_sha, failed_checkpoint_sha, failed_checks = (
            validate_failed_final_gate_report(failed_report)
        )
        if failed_checkpoint_sha == before:
            raise ValueError("The failed final gate cannot name the parent itself")
        loaded = super().load(
            str(resolved), load_cfg=load_cfg, strict=strict, map_location=map_location
        )
        if (
            sha256_file(resolved) != before
            or sha256_file(parent_report) != parent_report_sha
            or sha256_file(failed_report) != failed_report_sha
        ):
            raise ValueError("Final rescue parent or its reports changed while loading")
        if self.teleop_v12_corner_rescue != corner:
            raise RuntimeError("Final rescue inherited corner lineage drifted on load")
        self._final_rescue_marker = final_rescue_marker(
            parent_checkpoint_sha256=before,
            parent_tracking_report_sha256=parent_report_sha,
            parent_failed_checks=parent_failed,
            failed_gate_checkpoint_sha256=failed_checkpoint_sha,
            failed_gate_tracking_report_sha256=failed_report_sha,
            failed_gate_failed_checks=failed_checks,
            inherited_corner_rescue_marker_sha256=(
                None if corner is None else canonical_json_sha256(corner)
            ),
            sampler_mix=self._live_mix(),
        )
        self._assert_final_rescue_environment()
        self._assert_live_optimizer_step(
            MICROBAN_TELEOP_V12_FINAL_RESCUE_PARENT_OPTIMIZER_STEP
        )
        return loaded

    def _contract_infos(self, infos: dict | None = None) -> dict:
        self._assert_final_rescue_environment()
        result = super()._contract_infos(infos)
        marker = self._final_rescue_marker
        if marker is None:
            raise RuntimeError("Final rescue lineage marker is not bound")
        if marker["inherited_corner_rescue_marker_sha256"] != (
            None
            if result.get(MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY) is None
            else canonical_json_sha256(
                result[MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY]
            )
        ):
            raise RuntimeError("Final rescue inherited corner lineage drifted")
        result["microban_teleop_recipe_revision"] = (
            MICROBAN_TELEOP_V12_FINAL_RESCUE_RECIPE_REVISION
        )
        result[MICROBAN_TELEOP_V12_FINAL_RESCUE_INFO_KEY] = dict(marker)
        return result

    def _assert_live_optimizer_step(self, expected_step: int) -> None:
        assert_final_rescue_optimizer_step(
            {"optimizer_state_dict": self.alg.optimizer.state_dict()},
            expected_step=expected_step,
        )

    def _assert_final_rescue_environment(self) -> None:
        env = self.env.unwrapped
        common_step = int(env.common_step_counter)
        if not (
            MICROBAN_TELEOP_V12_FINAL_RESCUE_PARENT_COMMON_STEP
            <= common_step
            <= MICROBAN_TELEOP_V12_FINAL_RESCUE_TARGET_COMMON_STEP
        ):
            raise RuntimeError("Final rescue environment clock is outside its route")
        commands = env.command_manager
        for name, cls in (
            ("twist", FinalRescueTwistCommand),
            ("foot_target", FinalRescueFootTargetCommand),
            ("hand_target", FinalRescueHandTargetCommand),
        ):
            if not isinstance(commands.get_term(name), cls):
                raise TypeError(f"Final rescue {name} sampler was not installed")
        mix = self._live_mix()
        if (
            self._final_rescue_marker is not None
            and self._final_rescue_marker["sampler_mix"] != mix
        ):
            raise RuntimeError("Final rescue sampler mix drifted")
        hand = commands.get_term_cfg("hand_target")
        foot = commands.get_term_cfg("foot_target")
        rewards = env.reward_manager
        hand_reward = rewards.get_term_cfg("hand_target_tracking")
        foot_reward = rewards.get_term_cfg("foot_target_tracking")
        if (
            hand.rel_active != 0.7
            or foot.rel_single_support_envs != 0.3
            or foot.rel_both_feet_envs != 0.1
            or hand_reward.weight != 2.0
            or hand_reward.params.get("std")
            != MICROBAN_TELEOP_HAND_TRACKING_FINAL_STD_M
            or foot_reward.weight != 3.0
            or foot_reward.params.get("std")
            != MICROBAN_TELEOP_FOOT_TRACKING_FINAL_STD_M
        ):
            raise RuntimeError("Final rescue reward/command contract drifted")
        if tuple(self._actor.active_adapter_columns()) != (
            MICROBAN_TELEOP_V12_FINAL_RESCUE_ACTIVE_COLUMNS
        ):
            raise RuntimeError("Final rescue adapter columns drifted")

    def learn(
        self,
        num_learning_iterations: int,
        init_at_random_ep_len: bool = False,
    ) -> None:
        if num_learning_iterations != MICROBAN_TELEOP_V12_FINAL_RESCUE_PROCESS_UPDATES:
            raise ValueError("Final rescue must run exactly 99 PPO updates")
        if self.current_learning_iteration != (
            MICROBAN_TELEOP_V12_FINAL_RESCUE_PARENT_ITERATION + 1
        ):
            raise ValueError("Final rescue runner did not load model14900")
        self._assert_final_rescue_environment()
        self._assert_live_optimizer_step(
            MICROBAN_TELEOP_V12_FINAL_RESCUE_PARENT_OPTIMIZER_STEP
        )
        result = super().learn(num_learning_iterations, init_at_random_ep_len)
        if int(self.env.unwrapped.common_step_counter) != (
            MICROBAN_TELEOP_V12_FINAL_RESCUE_TARGET_COMMON_STEP
        ):
            raise RuntimeError("Final rescue did not stop at completed update 15000")
        self._assert_live_optimizer_step(
            MICROBAN_TELEOP_V12_FINAL_RESCUE_TARGET_OPTIMIZER_STEP
        )
        return result

    def save(self, path: str, infos=None) -> None:
        common_step = int(self.env.unwrapped.common_step_counter)
        if common_step > MICROBAN_TELEOP_V12_FINAL_RESCUE_TARGET_COMMON_STEP:
            raise RuntimeError("Final rescue may not save beyond update 15000")
        if common_step % MICROBAN_TELEOP_V12_FINAL_RESCUE_NUM_STEPS_PER_ENV:
            raise RuntimeError("Final rescue may save only at an update boundary")
        completed_updates = (
            common_step // MICROBAN_TELEOP_V12_FINAL_RESCUE_NUM_STEPS_PER_ENV
        )
        if completed_updates < MICROBAN_TELEOP_V12_FINAL_RESCUE_PARENT_COMPLETED_UPDATES:
            raise RuntimeError("Final rescue save clock precedes its fixed parent")
        expected_optimizer_step = (
            completed_updates * MICROBAN_TELEOP_V12_FINAL_RESCUE_OPTIMIZER_STEPS_PER_UPDATE
        )
        self._assert_final_rescue_environment()
        self._assert_live_optimizer_step(expected_optimizer_step)
        super().save(path, infos)
