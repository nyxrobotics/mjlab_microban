"""Fail-closed runner for the legacy-preserving contract-v12 adapter."""

from __future__ import annotations

import math
import os
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import torch
from mjlab.envs.mdp.actions import JointPositionAction
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.velocity import mdp as velocity_mdp

from mjlab_microban.tasks.curriculum import bind_update_clock, refuse_resume
from mjlab_microban.tasks.mdp import MICROBAN_BILATERAL_SITE_ORDER_REVISION
from mjlab_microban.schedules import PICO_STEPS_PER_UPDATE
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_HMD_JOINT_NAMES,
    MICROBAN_TELEOP_ACTION_JOINT_NAMES,
    MICROBAN_TELEOP_ACTION_WIDTH,
    validate_microban_teleop_observation_contract,
)
from mjlab_microban.tasks.microban_teleop_v12_actor import (
    TELEOP_RESIDUAL_HIDDEN_DIMS,
    TELEOP_TRAINABLE_ACTOR_PARAMETERS,
    TELEOP_V12_ADAPTER_GRADIENT_SCHEDULE_REVISION,
    TELEOP_V12_EXTRA_OBSERVATION_COLUMNS,
    LegacyAdapterPPO,
    LegacyAdapterTeleopActor,
)
from mjlab_microban.tasks.microban_teleop_v12_bootstrap import (
    TeleopV12BootstrapProvenance,
    assert_actor_frozen_against_source,
    bootstrap_legacy_actor,
    serialize_bootstrap_provenance,
    validate_bootstrap_provenance,
)
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MICROBAN_TELEOP_V12_ACTION_CLIP,
    MICROBAN_TELEOP_V12_FIXED_LEARNING_RATE,
    MICROBAN_TELEOP_V12_RECIPE_REVISION,
    MICROBAN_TELEOP_V12_TRAINING_CONTRACT_VERSION,
)
from mjlab_microban.tasks.microban_teleop_v12_home_pose import (
    TELEOP_V12_HOME_POSE_INFO_KEY,
    teleop_v12_home_pose_marker,
)

TELEOP_V12_BOOTSTRAP_INFO_KEY = "legacy_velocity_actor_bootstrap_v12"
# Every checkpoint records the corrected bilateral (left/right) site order of
# the foot target columns; the packager requires it.
BILATERAL_SITE_ORDER_INFO_KEY = "bilateral_site_order_revision"


def require_bilateral_site_order(infos: dict) -> None:
    """Reject a checkpoint that predates the corrected bilateral site order."""

    if infos.get(BILATERAL_SITE_ORDER_INFO_KEY) != MICROBAN_BILATERAL_SITE_ORDER_REVISION:
        raise ValueError("Checkpoint predates the corrected bilateral site order")


TELEOP_V12_OBSERVATION_TERM_LAYOUTS = {
    "actor": (
        ("base_ang_vel", 3),
        ("projected_gravity", 3),
        ("joint_pos", 21),
        ("joint_vel", 21),
        ("actions", 18),
        ("command", 3),
        ("foot_target", 6),
        ("arm_target", 6),
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
        ("arm_target", 6),
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
        "arm_target"
    ] != slice(75, 81):
        raise RuntimeError("Contract-v12 actor bilateral target slices drifted")
    if result["critic"]["foot_target"] != slice(84, 90) or result["critic"][
        "arm_target"
    ] != slice(90, 96):
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
    home_pose = teleop_v12_home_pose_marker()
    expected_home = torch.tensor(
        home_pose["joint_pos_rad"],
        device=robot.data.default_joint_pos.device,
        dtype=robot.data.default_joint_pos.dtype,
    )
    if not torch.allclose(
        robot.data.default_joint_pos[0], expected_home, rtol=0.0, atol=1.0e-7
    ):
        raise ValueError("Contract-v12 environment HOME pose drifted")
    expected_root = torch.tensor(
        [*home_pose["root_pos_xyz_m"], *home_pose["root_quat_wxyz"]],
        device=robot.data.default_root_state.device,
        dtype=robot.data.default_root_state.dtype,
    )
    if not torch.allclose(
        robot.data.default_root_state[0, :7], expected_root, rtol=0.0, atol=1.0e-7
    ):
        raise ValueError("Contract-v12 environment HOME root pose drifted")
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
    clip = getattr(action, "_clip", None)
    if clip is None or action.cfg.clip is None:
        raise ValueError("Contract-v12 action term must saturate at the servo goal range")
    expected_clip = torch.tensor(
        MICROBAN_TELEOP_V12_ACTION_CLIP, device=clip.device, dtype=clip.dtype
    ).expand(MICROBAN_TELEOP_ACTION_WIDTH, 2)
    if tuple(clip.shape[1:]) != (MICROBAN_TELEOP_ACTION_WIDTH, 2) or not bool(
        torch.all(clip == expected_clip.unsqueeze(0)).item()
    ):
        raise ValueError("Contract-v12 action target bound drifted from the servo goal range (+-pi)")


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
    """Fresh legacy bootstrap and invariant-checked saves; never resumed or reloaded."""

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
        if type(save_pristine) is not bool:
            raise TypeError("Contract-v12 runner flags must be booleans")
        refuse_resume(cfg)
        source_values = (source_path, source_sha256, probe_path, probe_sha256)
        if any(value is None for value in source_values):
            raise ValueError(
                "Fresh v12 training requires the authenticated source "
                "(checkpoint/receipt paths and hashes)"
            )
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

        if int(cfg.get("num_steps_per_env", -1)) != PICO_STEPS_PER_UPDATE:
            raise ValueError(
                f"Contract-v12 runs {PICO_STEPS_PER_UPDATE} env steps per update "
                "(the adapter column schedule is in updates of that length)"
            )
        bind_update_clock(env.unwrapped, PICO_STEPS_PER_UPDATE)
        self.teleop_v12_bootstrap: TeleopV12BootstrapProvenance | None = None
        super().__init__(env, cfg, log_dir=log_dir, device=device)
        if not isinstance(self.alg, LegacyAdapterPPO):
            raise TypeError("Contract-v12 runner requires LegacyAdapterPPO")
        actor = self.alg.get_policy()
        if not isinstance(actor, LegacyAdapterTeleopActor):
            raise TypeError("Contract-v12 runner requires LegacyAdapterTeleopActor")
        actor.bind_common_step_provider(
            lambda: int(self.env.unwrapped.common_step_counter)
        )
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
            TELEOP_V12_HOME_POSE_INFO_KEY: teleop_v12_home_pose_marker(),
            "previous_action_semantics": "raw_actor_output",
            "action_clip": list(MICROBAN_TELEOP_V12_ACTION_CLIP),
            "trainable_actor_parameters": list(TELEOP_TRAINABLE_ACTOR_PARAMETERS),
            "trainable_actor_columns": list(TELEOP_V12_EXTRA_OBSERVATION_COLUMNS),
            "residual_hidden_dims": list(TELEOP_RESIDUAL_HIDDEN_DIMS),
            "residual_trainable_at_save": self._actor.residual_trainable(),
            "adapter_gradient_schedule_revision": (
                TELEOP_V12_ADAPTER_GRADIENT_SCHEDULE_REVISION
            ),
            "active_actor_columns_at_save": list(self._actor.active_adapter_columns()),
            BILATERAL_SITE_ORDER_INFO_KEY: MICROBAN_BILATERAL_SITE_ORDER_REVISION,
            TELEOP_V12_BOOTSTRAP_INFO_KEY: serialize_bootstrap_provenance(
                self.teleop_v12_bootstrap
            ),
        }
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
        self._validate_live_invariants()
        return super().learn(num_learning_iterations, init_at_random_ep_len)

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        raise ValueError(
            "A contract-v12 run starts from its bootstrapped walker and loads no other checkpoint"
        )

    def save(self, path: str, infos=None) -> None:
        self._validate_live_invariants()
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
        self._validate_live_invariants()
        super().export_policy_to_onnx(path, filename, verbose)
