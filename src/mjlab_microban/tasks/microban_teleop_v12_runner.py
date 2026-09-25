"""Fail-closed runner for the legacy-preserving contract-v12 adapter."""

from __future__ import annotations

import hashlib
import math
import os
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import torch
from mjlab.envs.mdp.actions import JointPositionAction
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.velocity import mdp as velocity_mdp

from mjlab_microban.tasks.mdp import MICROBAN_BILATERAL_SITE_ORDER_REVISION
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_HMD_JOINT_NAMES,
    MICROBAN_TELEOP_ACTION_JOINT_NAMES,
    MICROBAN_TELEOP_ACTION_WIDTH,
    MICROBAN_TELEOP_NUM_STEPS_PER_ENV,
    validate_microban_teleop_observation_contract,
)
from mjlab_microban.tasks.microban_teleop_v12_actor import (
    TELEOP_V12_ADAPTER_GRADIENT_SCHEDULE_REVISION,
    TELEOP_V12_ADAPTER_SANITIZATION_REVISION,
    TELEOP_V12_ADAPTER_SANITIZATION_SCHEMA_VERSION,
    TELEOP_V12_EXTRA_OBSERVATION_COLUMNS,
    LegacyAdapterPPO,
    LegacyAdapterTeleopActor,
    teleop_v12_target_normalizer_metadata,
)
from mjlab_microban.tasks.microban_teleop_v12_bootstrap import (
    TeleopV12BootstrapProvenance,
    assert_actor_frozen_against_source,
    bootstrap_legacy_actor,
    serialize_bootstrap_provenance,
    sha256_file,
    validate_bootstrap_provenance,
)
from mjlab_microban.tasks.microban_teleop_v12_corner_rescue import (
    MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY,
    MICROBAN_TELEOP_V12_CORNER_RESCUE_RECIPE_REVISION,
    MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_OPTIMIZER_STEP,
    assert_corner_rescue_foot_adapter_zero,
    assert_corner_rescue_optimizer_step,
    validate_corner_rescue_canonical_lineage,
)
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MICROBAN_TELEOP_V12_FIXED_LEARNING_RATE,
    MICROBAN_TELEOP_V12_RECIPE_REVISION,
    MICROBAN_TELEOP_V12_TRAINING_CONTRACT_VERSION,
    preview_hand_tracking_settings,
)
from mjlab_microban.tasks.microban_teleop_v12_lr_order import (
    BILATERAL_SITE_ORDER_INFO_KEY,
    MIGRATION_INFO_KEY,
    validate_bilateral_site_order_checkpoint,
    validate_lr_order_migration_marker,
)
from mjlab_microban.tasks.microban_teleop_v12_preview import (
    TELEOP_V12_PREVIEW_INFO_KEY,
    TELEOP_V12_PREVIEW_PHASE1_ACCEPTANCE_INFO_KEY,
    TELEOP_V12_PREVIEW_PHASE_FULL_BODY,
    TELEOP_V12_PREVIEW_PHASE_HMD_HAND,
    reject_preview_checkpoint,
    validate_preview_marker,
)

TELEOP_V12_BOOTSTRAP_INFO_KEY = "legacy_velocity_actor_bootstrap_v12"
TELEOP_V12_SANITIZATION_INFO_KEY = "adapter_sanitization"

TELEOP_V12_OBSERVATION_TERM_LAYOUTS = {
    "actor": (
        ("base_ang_vel", 3),
        ("projected_gravity", 3),
        ("joint_pos", 21),
        ("joint_vel", 21),
        ("actions", 18),
        ("command", 3),
        ("foot_target", 6),
        ("hand_target", 8),
    ),
    "critic": (
        ("base_lin_vel", 3),
        ("base_ang_vel", 3),
        ("projected_gravity", 3),
        ("joint_pos", 21),
        ("joint_vel", 21),
        ("actions", 18),
        ("command", 3),
        ("foot_height", 2),
        ("foot_air_time", 2),
        ("foot_contact", 2),
        ("foot_contact_forces", 6),
        ("foot_target", 6),
        ("hand_target", 8),
        ("locomotion_prior", 39),
    ),
}


def teleop_v12_observation_term_slices(manager) -> dict[str, dict[str, slice]]:
    """Resolve and authenticate every actor/critic term slice."""

    result: dict[str, dict[str, slice]] = {}
    for group, expected_layout in TELEOP_V12_OBSERVATION_TERM_LAYOUTS.items():
        names = tuple(manager.active_terms.get(group, ()))
        dimensions = tuple(manager.group_obs_term_dim.get(group, ()))
        widths = tuple(math.prod(value) for value in dimensions)
        actual_layout = tuple(zip(names, widths, strict=True))
        if actual_layout != expected_layout:
            raise ValueError(
                f"Contract-v12 {group} observation layout drifted: {actual_layout}"
            )
        offset = 0
        slices: dict[str, slice] = {}
        for name, width in actual_layout:
            slices[name] = slice(offset, offset + width)
            offset += width
        result[group] = slices
    if result["actor"]["foot_target"] != slice(69, 75) or result["actor"][
        "hand_target"
    ] != slice(75, 83):
        raise RuntimeError("Contract-v12 actor bilateral target slices drifted")
    if result["critic"]["foot_target"] != slice(84, 90) or result["critic"][
        "hand_target"
    ] != slice(90, 98):
        raise RuntimeError("Contract-v12 critic bilateral target slices drifted")
    return result


def validate_teleop_v12_environment_contract(env) -> None:
    """Reject same-width observation/action reorder and clipped recurrence."""

    raw_env = env.unwrapped
    validate_microban_teleop_observation_contract(raw_env)
    teleop_v12_observation_term_slices(raw_env.observation_manager)
    expected_observation_joints = (
        *MICROBAN_HMD_JOINT_NAMES,
        *MICROBAN_TELEOP_ACTION_JOINT_NAMES,
    )
    robot = raw_env.scene["robot"]
    if tuple(robot.joint_names) != expected_observation_joints:
        raise ValueError(
            "Contract-v12 robot natural joint order drifted: "
            f"{tuple(robot.joint_names)}"
        )
    for group_name in ("actor", "critic"):
        for term_name in ("joint_pos", "joint_vel"):
            term = raw_env.observation_manager.get_term_cfg(group_name, term_name)
            asset_cfg = term.params.get("asset_cfg")
            resolved_names = getattr(asset_cfg, "joint_names", None)
            resolved = (
                tuple(robot.joint_names)
                if resolved_names is None
                else tuple(resolved_names)
            )
            if resolved != expected_observation_joints:
                raise ValueError(
                    f"Contract-v12 {group_name}/{term_name} joint order drifted: "
                    f"{resolved}"
                )
        previous = raw_env.observation_manager.get_term_cfg(group_name, "actions")
        if previous.func is not velocity_mdp.last_action or previous.params != {
            "action_name": "joint_pos"
        }:
            raise ValueError(
                f"Contract-v12 {group_name} previous action is not raw joint_pos"
            )

    action = raw_env.action_manager.get_term("joint_pos")
    if not isinstance(action, JointPositionAction):
        raise TypeError("Contract-v12 requires JointPositionAction named joint_pos")
    if tuple(action.target_names) != MICROBAN_TELEOP_ACTION_JOINT_NAMES:
        raise ValueError(
            f"Contract-v12 action joint order drifted: {tuple(action.target_names)}"
        )
    expected_target_ids = tuple(
        robot.joint_names.index(name) for name in MICROBAN_TELEOP_ACTION_JOINT_NAMES
    )
    actual_target_ids = tuple(int(value) for value in action.target_ids.cpu().tolist())
    if actual_target_ids != expected_target_ids:
        raise ValueError(
            "Contract-v12 action target IDs do not match natural joint order"
        )
    scale = torch.as_tensor(action.scale, device=action.raw_action.device)
    if scale.ndim > 1:
        scale = scale[0]
    if not bool(torch.all(scale == 1.0).item()):
        raise ValueError("Contract-v12 action scale must be exactly 1.0")
    offset = torch.as_tensor(action.offset, device=action.raw_action.device)
    if offset.ndim > 1:
        offset = offset[0]
    expected_offset = robot.data.default_joint_pos[0, action.target_ids]
    if tuple(offset.shape) != (MICROBAN_TELEOP_ACTION_WIDTH,) or not torch.equal(
        offset, expected_offset
    ):
        raise ValueError("Contract-v12 action offset drifted from the default pose")
    if getattr(action, "_clip", None) is not None:
        raise ValueError("Contract-v12 action term must not clip raw actor output")


def _atomic_torch_save(payload: object, destination: Path) -> None:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(
            destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


class MicrobanTeleopV12OnPolicyRunner(MjlabOnPolicyRunner):
    """Fresh legacy bootstrap, strict resume, and invariant-checked saves."""

    simulation_preview_capable = False
    allow_missing_preview_phase1_acceptance = False
    require_immutable_checkpoint_bytes = False
    consumer_required_preview_phase = TELEOP_V12_PREVIEW_PHASE_FULL_BODY
    consumer_requires_live_candidate = True

    def __init__(
        self,
        env,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        if not hasattr(env, "clip_actions") or env.clip_actions is not None:
            raise ValueError("Contract-v12 requires wrapper clip_actions=None")
        if getattr(env, "num_actions", None) != MICROBAN_TELEOP_ACTION_WIDTH:
            raise ValueError("Contract-v12 requires exactly 18 actions")
        validate_teleop_v12_environment_contract(env)

        cfg = deepcopy(train_cfg)
        source_path = cfg.pop("legacy_velocity_checkpoint", None)
        source_sha256 = cfg.pop("legacy_velocity_checkpoint_sha256", None)
        probe_path = cfg.pop("legacy_teleop_probe_receipt", None)
        probe_sha256 = cfg.pop("legacy_teleop_probe_receipt_sha256", None)
        save_pristine = cfg.pop("save_pristine_checkpoint", False)
        consumer_mode = cfg.pop("checkpoint_consumer_mode", False)
        preview_mode = cfg.pop("simulation_preview_mode", False)
        if (
            type(save_pristine) is not bool
            or type(consumer_mode) is not bool
            or type(preview_mode) is not bool
        ):
            raise TypeError("Contract-v12 runner flags must be booleans")
        if preview_mode is not self.simulation_preview_capable:
            raise ValueError("V12 preview mode requires the dedicated preview runner")
        resume = bool(cfg.get("resume", False))
        source_values = (source_path, source_sha256, probe_path, probe_sha256)
        if any(value is None for value in source_values) and any(
            value is not None for value in source_values
        ):
            raise ValueError(
                "Fresh v12 bootstrap requires checkpoint/receipt paths and hashes"
            )
        if consumer_mode and resume:
            raise ValueError("Consumer mode cannot resume training")
        if consumer_mode and (log_dir is not None or save_pristine):
            raise ValueError("Consumer mode cannot log or save pristine state")
        if resume and any(value is not None for value in source_values):
            raise ValueError("Resume reads its pinned source from the checkpoint")
        if (
            not resume
            and not consumer_mode
            and any(value is None for value in source_values)
        ):
            raise ValueError("Fresh v12 training requires the authenticated source")
        if resume and save_pristine:
            raise ValueError("A resume cannot create a pristine checkpoint")
        if save_pristine and log_dir is None:
            raise ValueError("Pristine checkpoint requires a log directory")

        algorithm = cfg.get("algorithm")
        if not isinstance(algorithm, dict):
            raise TypeError("Contract-v12 algorithm config must be a dictionary")
        expected_algorithm = {
            "learning_rate": MICROBAN_TELEOP_V12_FIXED_LEARNING_RATE,
            "schedule": "fixed",
            "entropy_coef": 0.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
        }
        if any(
            algorithm.get(name) != value for name, value in expected_algorithm.items()
        ):
            raise ValueError("Contract-v12 PPO optimizer recipe drifted")

        self.checkpoint_consumer_mode = consumer_mode
        self.teleop_v12_training_resume = resume
        self.simulation_preview_mode = preview_mode
        self.teleop_v12_bootstrap: TeleopV12BootstrapProvenance | None = None
        self.teleop_v12_sanitization: dict | None = None
        self.teleop_v12_lr_order_migration: dict | None = None
        self.teleop_v12_corner_rescue: dict | None = None
        self.teleop_v12_preview: dict | None = None
        self.teleop_v12_preview_phase1_acceptance: dict | None = None
        super().__init__(env, cfg, log_dir=log_dir, device=device)
        if not isinstance(self.alg, LegacyAdapterPPO):
            raise TypeError("Contract-v12 runner requires LegacyAdapterPPO")
        actor = self.alg.get_policy()
        if not isinstance(actor, LegacyAdapterTeleopActor):
            raise TypeError("Contract-v12 runner requires LegacyAdapterTeleopActor")
        actor.bind_common_step_provider(
            lambda: int(self.env.unwrapped.common_step_counter)
        )
        if resume or consumer_mode:
            return

        assert source_path is not None
        assert source_sha256 is not None
        assert probe_path is not None
        assert probe_sha256 is not None
        self.teleop_v12_bootstrap = bootstrap_legacy_actor(
            actor,
            source_path,
            source_sha256,
            probe_path,
            probe_sha256,
        )
        actor.assert_optimizer_invariant(self.alg.optimizer)
        actor.assert_schedule_locked_weights_zero()
        if save_pristine and self.gpu_global_rank == 0:
            assert log_dir is not None
            self._save_pristine(Path(log_dir) / "model_pristine.pt")

    @property
    def _actor(self) -> LegacyAdapterTeleopActor:
        actor = self.alg.get_policy()
        if not isinstance(actor, LegacyAdapterTeleopActor):
            raise TypeError("Contract-v12 actor type drifted")
        return actor

    def _validate_live_invariants(self, *, verify_source_files: bool = True) -> None:
        if self.teleop_v12_bootstrap is None:
            raise RuntimeError("Contract-v12 bootstrap provenance is not bound")
        if verify_source_files:
            validate_bootstrap_provenance(
                serialize_bootstrap_provenance(self.teleop_v12_bootstrap),
                verify_files=True,
            )
            assert_actor_frozen_against_source(self._actor, self.teleop_v12_bootstrap)
        self._actor.assert_frozen_legacy_state()
        self._actor.assert_schedule_locked_weights_zero()
        self._actor.assert_optimizer_invariant(self.alg.optimizer)
        groups = self.alg.optimizer.param_groups
        if len(groups) != 1 or groups[0].get("lr") != (
            MICROBAN_TELEOP_V12_FIXED_LEARNING_RATE
        ):
            raise RuntimeError("Contract-v12 optimizer learning rate drifted")
        if self.alg.learning_rate != MICROBAN_TELEOP_V12_FIXED_LEARNING_RATE:
            raise RuntimeError("Contract-v12 PPO learning rate drifted")

    def _contract_infos(self, infos: dict | None = None) -> dict:
        if self.teleop_v12_bootstrap is None:
            raise RuntimeError("Contract-v12 bootstrap provenance is missing")
        result = {
            **(infos or {}),
            "microban_teleop_training_contract_version": (
                MICROBAN_TELEOP_V12_TRAINING_CONTRACT_VERSION
            ),
            "microban_teleop_recipe_revision": MICROBAN_TELEOP_V12_RECIPE_REVISION,
            "previous_action_semantics": "raw_actor_output",
            "action_clip": None,
            "trainable_actor_parameters": ["mlp.0.weight"],
            "trainable_actor_columns": list(TELEOP_V12_EXTRA_OBSERVATION_COLUMNS),
            "adapter_gradient_schedule_revision": (
                TELEOP_V12_ADAPTER_GRADIENT_SCHEDULE_REVISION
            ),
            "active_actor_columns_at_save": list(self._actor.active_adapter_columns()),
            BILATERAL_SITE_ORDER_INFO_KEY: MICROBAN_BILATERAL_SITE_ORDER_REVISION,
            TELEOP_V12_BOOTSTRAP_INFO_KEY: serialize_bootstrap_provenance(
                self.teleop_v12_bootstrap
            ),
        }
        if self.teleop_v12_sanitization is not None:
            result[TELEOP_V12_SANITIZATION_INFO_KEY] = dict(
                self.teleop_v12_sanitization
            )
        if self.teleop_v12_lr_order_migration is not None:
            result[MIGRATION_INFO_KEY] = deepcopy(
                validate_lr_order_migration_marker(self.teleop_v12_lr_order_migration)
            )
        if self.teleop_v12_corner_rescue is not None:
            result[MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY] = deepcopy(
                self.teleop_v12_corner_rescue
            )
        if self.teleop_v12_preview is not None:
            result["preview_non_deployable"] = True
            result[TELEOP_V12_PREVIEW_INFO_KEY] = dict(self.teleop_v12_preview)
            if (
                self.teleop_v12_preview.get("phase")
                == TELEOP_V12_PREVIEW_PHASE_FULL_BODY
            ):
                from mjlab_microban.scripts.promote_teleop_v12_preview_visual import (
                    validate_embedded_phase1_acceptance,
                )

                result[TELEOP_V12_PREVIEW_PHASE1_ACCEPTANCE_INFO_KEY] = deepcopy(
                    self.teleop_v12_preview_phase1_acceptance
                )
                validate_embedded_phase1_acceptance(
                    result, fullbody_marker=self.teleop_v12_preview
                )
        return result

    def _save_pristine(self, path: Path) -> None:
        if path.exists():
            raise FileExistsError(f"Refusing to replace pristine checkpoint: {path}")
        if self.env.unwrapped.common_step_counter != 0:
            raise ValueError("Pristine checkpoint requires common_step_counter == 0")
        self._validate_live_invariants()
        payload = self.alg.save()
        payload["iter"] = -1
        payload["infos"] = self._contract_infos(
            {
                "env_state": {"common_step_counter": 0},
                "pristine_pre_update": True,
            }
        )
        _atomic_torch_save(payload, path)
        print(f"[INFO] Saved contract-v12 pristine checkpoint: {path.resolve()}")

    def learn(
        self,
        num_learning_iterations: int,
        init_at_random_ep_len: bool = False,
    ) -> None:
        if self.checkpoint_consumer_mode:
            raise RuntimeError("Contract-v12 consumer mode cannot train")
        self._validate_live_invariants()
        return super().learn(num_learning_iterations, init_at_random_ep_len)

    def _validate_preview_phase1_acceptance(
        self, infos: dict, preview: dict
    ) -> dict | None:
        """Validate phase-1 lineage, except for the explicit legacy trial class."""

        if preview.get("phase") == TELEOP_V12_PREVIEW_PHASE_FULL_BODY:
            if (
                self.allow_missing_preview_phase1_acceptance
                and TELEOP_V12_PREVIEW_PHASE1_ACCEPTANCE_INFO_KEY not in infos
            ):
                return None
            from mjlab_microban.scripts.promote_teleop_v12_preview_visual import (
                validate_embedded_phase1_acceptance,
            )

            return validate_embedded_phase1_acceptance(infos, fullbody_marker=preview)
        if infos.get(TELEOP_V12_PREVIEW_PHASE1_ACCEPTANCE_INFO_KEY) is not None:
            raise ValueError("Only a full-body preview may embed phase-1 acceptance")
        return None

    def load(
        self,
        path: str | bytes,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        consumer = self.checkpoint_consumer_mode
        resume = self.teleop_v12_training_resume
        if consumer:
            if load_cfg != {
                "actor": True,
                "critic": False,
                "optimizer": False,
                "iteration": False,
                "rnd": False,
            }:
                raise ValueError("Consumer mode requires explicit actor-only load")
        elif resume:
            if load_cfg is not None or strict is not True:
                raise ValueError("Training resume requires strict full-state load")
        else:
            raise ValueError("Fresh bootstrapped runner cannot load another checkpoint")

        if self.require_immutable_checkpoint_bytes and not isinstance(path, bytes):
            raise ValueError(
                "Simulation preview consumer requires immutable checkpoint bytes"
            )

        verified_bytes = path if isinstance(path, bytes) else None
        if verified_bytes is not None:
            if not consumer:
                raise ValueError("Verified checkpoint bytes are consumer-only")
            before_sha256 = hashlib.sha256(verified_bytes).hexdigest()
            payload = torch.load(
                BytesIO(verified_bytes),
                map_location=map_location or "cpu",
                weights_only=False,
            )
            upstream_source: str | BytesIO = BytesIO(verified_bytes)
            resolved: Path | None = None
        else:
            resolved = Path(path).expanduser().resolve()
            before_sha256 = sha256_file(resolved)
            payload = torch.load(
                resolved, map_location=map_location or "cpu", weights_only=False
            )
            upstream_source = str(resolved)
        if not isinstance(payload, dict) or not isinstance(payload.get("infos"), dict):
            raise TypeError("Contract-v12 checkpoint payload is malformed")
        infos = payload["infos"]
        iteration = payload.get("iter")
        if not isinstance(iteration, int) or isinstance(iteration, bool):
            raise TypeError("Checkpoint iteration is malformed")
        if self.simulation_preview_mode:
            preview = validate_preview_marker(
                infos,
                iteration=iteration,
                required_phase=(
                    self.consumer_required_preview_phase if consumer else None
                ),
                require_live_candidate=(
                    consumer and self.consumer_requires_live_candidate
                ),
            )
            phase1_acceptance = self._validate_preview_phase1_acceptance(infos, preview)
        else:
            reject_preview_checkpoint(infos)
            preview = None
            phase1_acceptance = None
        if infos.get("microban_teleop_training_contract_version") != (
            MICROBAN_TELEOP_V12_TRAINING_CONTRACT_VERSION
        ):
            raise ValueError("Checkpoint is not contract-v12")
        corner_rescue = validate_corner_rescue_canonical_lineage(
            infos, iteration=iteration
        )
        if infos.get("microban_teleop_recipe_revision") == (
            MICROBAN_TELEOP_V12_CORNER_RESCUE_RECIPE_REVISION
        ):
            assert corner_rescue is not None
            assert_corner_rescue_foot_adapter_zero(payload)
            assert_corner_rescue_optimizer_step(
                payload,
                expected_step=(
                    MICROBAN_TELEOP_V12_CORNER_RESCUE_TARGET_OPTIMIZER_STEP
                ),
            )
        if infos.get("previous_action_semantics") != "raw_actor_output" or (
            infos.get("action_clip", object()) is not None
        ):
            raise ValueError("Checkpoint raw-action semantics drifted")
        if infos.get("adapter_gradient_schedule_revision") != (
            TELEOP_V12_ADAPTER_GRADIENT_SCHEDULE_REVISION
        ):
            raise ValueError("Checkpoint adapter gradient schedule is not v12-safe")
        provenance = validate_bootstrap_provenance(
            infos.get(TELEOP_V12_BOOTSTRAP_INFO_KEY), verify_files=True
        )
        sanitization = infos.get(TELEOP_V12_SANITIZATION_INFO_KEY)
        if sanitization is not None:
            if not isinstance(sanitization, dict):
                raise TypeError("Checkpoint adapter sanitization lineage is malformed")
            exact_sanitization = {
                "schema_version": TELEOP_V12_ADAPTER_SANITIZATION_SCHEMA_VERSION,
                "revision": TELEOP_V12_ADAPTER_SANITIZATION_REVISION,
                "parent_iteration": sanitization.get("completed_updates", 0) - 1,
                "zeroed_actor_columns": list(TELEOP_V12_EXTRA_OBSERVATION_COLUMNS),
                **teleop_v12_target_normalizer_metadata(),
            }
            if any(
                sanitization.get(name) != value
                for name, value in exact_sanitization.items()
            ):
                raise ValueError("Checkpoint adapter sanitization lineage drifted")
            parent_sha = sanitization.get("parent_checkpoint_sha256")
            if not isinstance(parent_sha, str) or len(parent_sha) != 64:
                raise ValueError("Checkpoint sanitizer parent SHA-256 is malformed")
        if resume and (iteration < 0 or infos.get("pristine_pre_update") is True):
            raise ValueError("Pristine checkpoint cannot resume training")
        lr_order_migration = validate_bilateral_site_order_checkpoint(infos)

        consumer_common_step = (
            int(self.env.unwrapped.common_step_counter) if consumer else None
        )
        loaded_infos = super().load(
            upstream_source,  # type: ignore[arg-type]
            load_cfg=load_cfg,
            strict=strict,
            map_location=map_location,
        )
        after_sha256 = (
            hashlib.sha256(verified_bytes).hexdigest()
            if verified_bytes is not None
            else sha256_file(resolved)  # type: ignore[arg-type]
        )
        if after_sha256 != before_sha256:
            raise ValueError("Checkpoint changed while loading")
        self.teleop_v12_bootstrap = provenance
        self.teleop_v12_sanitization = sanitization
        self.teleop_v12_lr_order_migration = deepcopy(lr_order_migration)
        self.teleop_v12_corner_rescue = deepcopy(corner_rescue)
        self.teleop_v12_preview = preview
        self.teleop_v12_preview_phase1_acceptance = deepcopy(phase1_acceptance)
        assert_actor_frozen_against_source(self._actor, provenance)
        self._actor.bind_frozen_legacy_reference()
        expected_active = list(self._actor.active_adapter_columns())
        if infos.get("active_actor_columns_at_save") != expected_active:
            raise ValueError("Checkpoint active adapter columns drifted from its clock")
        self._actor.assert_schedule_locked_weights_zero()
        self._actor.assert_optimizer_invariant(self.alg.optimizer)
        if consumer:
            # Validate against the saved training clock first, then undo the
            # upstream actor-only load's otherwise surprising env-state write.
            assert consumer_common_step is not None
            self.env.unwrapped.common_step_counter = consumer_common_step
            return loaded_infos

        if self.current_learning_iteration != iteration:
            raise RuntimeError("Full-state load did not restore iteration")
        self.current_learning_iteration += 1
        self._validate_live_invariants()
        env = self.env.unwrapped
        expected_steps = (iteration + 1) * int(self.cfg["num_steps_per_env"])
        if env.common_step_counter != expected_steps:
            raise ValueError(
                "Checkpoint common_step_counter does not match completed updates: "
                f"{env.common_step_counter} != {expected_steps}"
            )
        if getattr(env, "curriculum_manager", None) is not None:
            env.curriculum_manager.compute()
            restored = env.common_step_counter
            env.reset()
            if env.common_step_counter != restored:
                raise RuntimeError("Reset changed restored common_step_counter")
        if self.simulation_preview_mode:
            self._assert_preview_curriculum_active()
        return loaded_infos

    def _assert_preview_curriculum_active(self) -> None:
        """Prove each lifted phase samples/rewards only its declared targets."""

        env = self.env.unwrapped
        marker = self.teleop_v12_preview
        if not isinstance(marker, dict):
            raise TypeError("V12 preview marker is not bound")
        hand = env.command_manager.get_term_cfg("hand_target")
        foot = env.command_manager.get_term_cfg("foot_target")
        rewards = env.reward_manager
        hmd = env.event_manager.get_term_cfg("hmd_neck_target_motion")
        hmd_func = hmd.func
        common_step_counter = int(env.common_step_counter)
        if common_step_counter % MICROBAN_TELEOP_NUM_STEPS_PER_ENV != 0:
            raise RuntimeError("V12 preview training clock is not update-aligned")
        completed_updates = common_step_counter // MICROBAN_TELEOP_NUM_STEPS_PER_ENV
        hand_settings = preview_hand_tracking_settings(completed_updates)
        hand_reward = rewards.get_term_cfg("hand_target_tracking")
        soft_limit_guard = rewards.get_term_cfg("joint_soft_limit_guard")
        phase = marker.get("phase")
        if phase == TELEOP_V12_PREVIEW_PHASE_HMD_HAND:
            expected = (0.7, 0.0, 0.0, 0.0)
        elif phase == TELEOP_V12_PREVIEW_PHASE_FULL_BODY:
            expected = (0.7, 0.3, 0.05, 2.0)
        else:
            raise RuntimeError("V12 preview phase is invalid")
        actual = (
            hand.rel_active,
            foot.rel_single_support_envs,
            foot.rel_both_feet_envs,
            rewards.get_term_cfg("foot_target_tracking").weight,
        )
        hand_actual = (
            hand_reward.weight,
            hand_reward.params.get("std"),
            soft_limit_guard.weight,
        )
        hand_expected = (
            hand_settings.reward_weight,
            hand_settings.reward_std_m,
            hand_settings.joint_soft_limit_guard_weight,
        )
        if (
            actual != expected
            or hand_actual != hand_expected
            or getattr(hmd_func, "neutral_probability", None) != 0.2
        ):
            raise RuntimeError(
                f"V12 preview {phase} curriculum drifted: "
                f"targets={actual} != {expected}; "
                f"hand={hand_actual} != {hand_expected}"
            )

    def save(self, path: str, infos=None) -> None:
        if self.checkpoint_consumer_mode:
            raise RuntimeError("Contract-v12 consumer mode cannot save")
        self._validate_live_invariants()
        if self.simulation_preview_mode:
            self._assert_preview_curriculum_active()
        payload = self.alg.save()
        payload["iter"] = self.current_learning_iteration
        payload["infos"] = self._contract_infos(
            {
                **(infos or {}),
                "env_state": {
                    "common_step_counter": self.env.unwrapped.common_step_counter
                },
            }
        )
        destination = Path(path).expanduser().resolve()
        _atomic_torch_save(payload, destination)

    def export_policy_to_onnx(
        self,
        path: str,
        filename: str = "policy.onnx",
        verbose: bool = False,
    ) -> None:
        if self.simulation_preview_mode:
            raise RuntimeError(
                "Simulation-only preview cannot use canonical ONNX export"
            )
        self._validate_live_invariants()
        super().export_policy_to_onnx(path, filename, verbose)


class MicrobanTeleopV12PreviewOnPolicyRunner(MicrobanTeleopV12OnPolicyRunner):
    """Explicit simulation-only runner for clock-lifted full-body previews."""

    simulation_preview_capable = True


class MicrobanTeleopV12UnacceptedSimulationPreviewOnPolicyRunner(
    MicrobanTeleopV12PreviewOnPolicyRunner
):
    """Read-only loader for one explicit, unaccepted simulation observation."""

    allow_missing_preview_phase1_acceptance = True
    require_immutable_checkpoint_bytes = True

    def __init__(
        self,
        env,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        if (
            train_cfg.get("checkpoint_consumer_mode") is not True
            or train_cfg.get("simulation_preview_mode") is not True
            or train_cfg.get("resume") is not False
            or log_dir is not None
        ):
            raise ValueError(
                "Unaccepted simulation preview is immutable, read-only, and "
                "consumer-only"
            )
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)

    def learn(self, *args, **kwargs) -> None:
        del args, kwargs
        raise RuntimeError("Unaccepted simulation preview cannot train")

    def save(self, *args, **kwargs) -> None:
        del args, kwargs
        raise RuntimeError("Unaccepted simulation preview cannot save")

    def export_policy_to_onnx(self, *args, **kwargs) -> None:
        del args, kwargs
        raise RuntimeError("Unaccepted simulation preview cannot export")


class MicrobanTeleopV12ControllerOnlyPreviewOnPolicyRunner(
    MicrobanTeleopV12PreviewOnPolicyRunner
):
    """Read-only phase-1 consumer for controller hands with inactive feet."""

    consumer_required_preview_phase = TELEOP_V12_PREVIEW_PHASE_HMD_HAND
    consumer_requires_live_candidate = False
    require_immutable_checkpoint_bytes = True

    def __init__(
        self,
        env,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        if (
            train_cfg.get("checkpoint_consumer_mode") is not True
            or train_cfg.get("simulation_preview_mode") is not True
            or train_cfg.get("resume") is not False
            or log_dir is not None
        ):
            raise ValueError(
                "Controller-only preview is immutable, read-only, and consumer-only"
            )
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)

    def learn(self, *args, **kwargs) -> None:
        del args, kwargs
        raise RuntimeError("Controller-only preview cannot train")

    def save(self, *args, **kwargs) -> None:
        del args, kwargs
        raise RuntimeError("Controller-only preview cannot save")

    def export_policy_to_onnx(self, *args, **kwargs) -> None:
        del args, kwargs
        raise RuntimeError("Controller-only preview cannot export")


def preview_actor_load_cfg() -> dict[str, bool]:
    """Return the only actor-only load mask accepted by a preview consumer."""

    return {
        "actor": True,
        "critic": False,
        "optimizer": False,
        "iteration": False,
        "rnd": False,
    }


def make_teleop_v12_preview_consumer(
    env, agent_cfg, device: str
) -> MicrobanTeleopV12PreviewOnPolicyRunner:
    """Construct the explicit simulation-only consumer used by live PICO sim."""

    if is_dataclass(agent_cfg) and not isinstance(agent_cfg, type):
        cfg = asdict(agent_cfg)
    elif isinstance(agent_cfg, dict):
        cfg = deepcopy(agent_cfg)
    else:
        raise TypeError("Preview agent_cfg must be a dataclass instance or dict")
    cfg["checkpoint_consumer_mode"] = True
    cfg["simulation_preview_mode"] = True
    cfg["resume"] = False
    return MicrobanTeleopV12PreviewOnPolicyRunner(env, cfg, device=device)


def make_teleop_v12_unaccepted_simulation_preview_consumer(
    env, agent_cfg, device: str
) -> MicrobanTeleopV12UnacceptedSimulationPreviewOnPolicyRunner:
    """Construct the only consumer allowed to omit old phase-1 evidence."""

    if is_dataclass(agent_cfg) and not isinstance(agent_cfg, type):
        cfg = asdict(agent_cfg)
    elif isinstance(agent_cfg, dict):
        cfg = deepcopy(agent_cfg)
    else:
        raise TypeError("Preview agent_cfg must be a dataclass instance or dict")
    cfg["checkpoint_consumer_mode"] = True
    cfg["simulation_preview_mode"] = True
    cfg["resume"] = False
    return MicrobanTeleopV12UnacceptedSimulationPreviewOnPolicyRunner(
        env, cfg, device=device
    )


def make_teleop_v12_controller_only_preview_consumer(
    env, agent_cfg, device: str
) -> MicrobanTeleopV12ControllerOnlyPreviewOnPolicyRunner:
    """Construct the phase-1 controller-hand, exact-zero-foot consumer."""

    if is_dataclass(agent_cfg) and not isinstance(agent_cfg, type):
        cfg = asdict(agent_cfg)
    elif isinstance(agent_cfg, dict):
        cfg = deepcopy(agent_cfg)
    else:
        raise TypeError("Preview agent_cfg must be a dataclass instance or dict")
    cfg["checkpoint_consumer_mode"] = True
    cfg["simulation_preview_mode"] = True
    cfg["resume"] = False
    return MicrobanTeleopV12ControllerOnlyPreviewOnPolicyRunner(env, cfg, device=device)
