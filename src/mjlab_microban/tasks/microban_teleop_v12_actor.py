"""Legacy-preserving actor components for the contract-v12 teleop policy.

The proven velocity actor is a normalized, unbounded Gaussian policy with a
63-value observation.  Contract v12 expands that input to the teleop 83-value
schema without changing the legacy computation at bootstrap: shared first-layer
columns are copied by semantic name and the 20 new HMD/keypoint columns start at
exact zero.

Only those 20 first-layer columns are trainable.  The empirical normalizer,
legacy columns, downstream trunk, output head, bias, and Gaussian standard
deviation are immutable.  This keeps the deadline path auditable and prevents a
teleop update from silently erasing the already-proven locomotion policy.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.modules.distribution import GaussianDistribution

from mjlab_microban.robot.microban_hand_fk import (
    MICROBAN_HAND_TARGET_NORMALIZER_ABS_BOUND_M,
    microban_hand_fk_metadata,
)
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_HMD_JOINT_NAMES,
    MICROBAN_TELEOP_ACTION_JOINT_NAMES,
    MICROBAN_TELEOP_ACTION_WIDTH,
    MICROBAN_TELEOP_OBSERVATION_WIDTH,
)

LEGACY_VELOCITY_OBSERVATION_WIDTH = 63
LEGACY_VELOCITY_NORMALIZER_EPS = 1.0e-2
LEGACY_VELOCITY_CHECKPOINT_SHA256 = (
    "b0bcdadac39716be784207dd6b2b93157162a3e80650e23c05f490c400b9e141"
)
LEGACY_VELOCITY_CHECKPOINT_ITERATION = 14_999
LEGACY_VELOCITY_ACTOR_TOPOLOGY = (63, 512, 256, 128, 18)
TELEOP_V12_ACTOR_TOPOLOGY = (83, 512, 256, 128, 18)
TELEOP_V12_IDENTITY_NORMALIZER_BOOTSTRAP_MAPPING_VERSION = (
    "normalized_legacy_velocity_63_to_teleop83_masked_extra_columns_v1"
)
TELEOP_V12_BOOTSTRAP_MAPPING_VERSION = (
    "normalized_legacy_velocity_63_to_teleop83_reachable_fk_elbow_minus10_v4"
)

LEGACY_VELOCITY_ACTOR_STATE_KEYS: frozenset[str] = frozenset(
    {
        "obs_normalizer._mean",
        "obs_normalizer._var",
        "obs_normalizer._std",
        "obs_normalizer.count",
        "distribution.std_param",
        "mlp.0.weight",
        "mlp.0.bias",
        "mlp.2.weight",
        "mlp.2.bias",
        "mlp.4.weight",
        "mlp.4.bias",
        "mlp.6.weight",
        "mlp.6.bias",
    }
)


def _semantic_observation_names(*, include_teleop_features: bool) -> tuple[str, ...]:
    body = MICROBAN_TELEOP_ACTION_JOINT_NAMES
    joints = (*MICROBAN_HMD_JOINT_NAMES, *body) if include_teleop_features else body
    names: list[str] = []
    for term, members in (
        ("base_ang_vel", ("x", "y", "z")),
        ("projected_gravity", ("x", "y", "z")),
        ("joint_pos", joints),
        ("joint_vel", joints),
        ("actions", body),
        ("command", ("vx", "vy", "yaw")),
    ):
        names.extend(f"{term}/{member}" for member in members)
    if include_teleop_features:
        names.extend(
            f"foot_target/{member}"
            for member in (
                "left_x",
                "left_y",
                "left_z",
                "right_x",
                "right_y",
                "right_z",
            )
        )
        names.extend(
            f"hand_target/{member}"
            for member in (
                "left_x",
                "left_y",
                "left_z",
                "right_x",
                "right_y",
                "right_z",
                "left_active",
                "right_active",
            )
        )
    return tuple(names)


LEGACY_VELOCITY_OBSERVATION_NAMES = _semantic_observation_names(
    include_teleop_features=False
)
TELEOP_V12_OBSERVATION_NAMES = _semantic_observation_names(include_teleop_features=True)
if len(LEGACY_VELOCITY_OBSERVATION_NAMES) != LEGACY_VELOCITY_OBSERVATION_WIDTH:
    raise RuntimeError("Legacy semantic observation width drifted")
if len(TELEOP_V12_OBSERVATION_NAMES) != MICROBAN_TELEOP_OBSERVATION_WIDTH:
    raise RuntimeError("Teleop semantic observation width drifted")


def legacy_to_teleop_observation_mapping() -> tuple[tuple[int, int], ...]:
    """Map every legacy scalar to its unique teleop scalar by semantic name."""

    if len(set(LEGACY_VELOCITY_OBSERVATION_NAMES)) != len(
        LEGACY_VELOCITY_OBSERVATION_NAMES
    ):
        raise RuntimeError("Legacy semantic observation names are not unique")
    if len(set(TELEOP_V12_OBSERVATION_NAMES)) != len(TELEOP_V12_OBSERVATION_NAMES):
        raise RuntimeError("Teleop semantic observation names are not unique")
    target_by_name = {
        name: index for index, name in enumerate(TELEOP_V12_OBSERVATION_NAMES)
    }
    missing = [
        name for name in LEGACY_VELOCITY_OBSERVATION_NAMES if name not in target_by_name
    ]
    if missing:
        raise RuntimeError(f"Teleop schema is missing legacy observations: {missing}")
    return tuple(
        (source, target_by_name[name])
        for source, name in enumerate(LEGACY_VELOCITY_OBSERVATION_NAMES)
    )


LEGACY_TO_TELEOP_OBSERVATION_INDEX = legacy_to_teleop_observation_mapping()
TELEOP_V12_SHARED_OBSERVATION_COLUMNS = tuple(
    target for _source, target in LEGACY_TO_TELEOP_OBSERVATION_INDEX
)
TELEOP_V12_EXTRA_OBSERVATION_COLUMNS = tuple(
    index
    for index in range(MICROBAN_TELEOP_OBSERVATION_WIDTH)
    if index not in set(TELEOP_V12_SHARED_OBSERVATION_COLUMNS)
)
if len(TELEOP_V12_EXTRA_OBSERVATION_COLUMNS) != 20:
    raise RuntimeError("Contract v12 must introduce exactly 20 actor columns")

TELEOP_V12_HMD_OBSERVATION_COLUMNS = (6, 7, 8, 27, 28, 29)
TELEOP_V12_FOOT_OBSERVATION_COLUMNS = tuple(range(69, 75))
TELEOP_V12_HAND_POSITION_OBSERVATION_COLUMNS = tuple(range(75, 81))
TELEOP_V12_HAND_ACTIVE_OBSERVATION_COLUMNS = tuple(range(81, 83))
TELEOP_V12_HAND_OBSERVATION_COLUMNS = (
    *TELEOP_V12_HAND_POSITION_OBSERVATION_COLUMNS,
    *TELEOP_V12_HAND_ACTIVE_OBSERVATION_COLUMNS,
)
TELEOP_V12_TARGET_POSITION_OBSERVATION_COLUMNS = (
    *TELEOP_V12_FOOT_OBSERVATION_COLUMNS,
    *TELEOP_V12_HAND_POSITION_OBSERVATION_COLUMNS,
)
# EmpiricalNormalization divides by ``stored_std + eps``.  Position commands
# are already bounded in metres, so scale them by their physical per-axis
# maximum instead of leaving centimetre-sized signals near zero.  The active
# flags and the six HMD columns deliberately retain their existing identity
# normalizer state.
TELEOP_V12_TARGET_POSITION_NORMALIZER_DENOMINATORS = (
    0.03,
    0.03,
    0.05,
    0.03,
    0.03,
    0.05,
    *MICROBAN_HAND_TARGET_NORMALIZER_ABS_BOUND_M,
    *MICROBAN_HAND_TARGET_NORMALIZER_ABS_BOUND_M,
)
TELEOP_V12_TARGET_POSITION_NORMALIZER_STORED_STD = tuple(
    denominator - LEGACY_VELOCITY_NORMALIZER_EPS
    for denominator in TELEOP_V12_TARGET_POSITION_NORMALIZER_DENOMINATORS
)
if any(value <= 0.0 for value in TELEOP_V12_TARGET_POSITION_NORMALIZER_STORED_STD):
    raise RuntimeError("Target-position normalizer denominator must exceed epsilon")
TELEOP_V12_UNCHANGED_EXTRA_NORMALIZER_COLUMNS = tuple(
    column
    for column in TELEOP_V12_EXTRA_OBSERVATION_COLUMNS
    if column not in set(TELEOP_V12_TARGET_POSITION_OBSERVATION_COLUMNS)
)
TELEOP_V12_ADAPTER_GRADIENT_SCHEDULE_REVISION = (
    "freeze_extra_to7000_then_hmd_hand_to10000_then_all_v1"
)
TELEOP_V12_ADAPTER_SANITIZATION_SCHEMA_VERSION = 4
TELEOP_V12_ADAPTER_SANITIZATION_REVISION = (
    "zero_pre7000_extra_w0_adam_and_reachable_fk_elbow_minus10_v4"
)


def teleop_v12_target_normalizer_metadata() -> dict[str, object]:
    """Return the exact JSON-safe target-normalizer migration contract."""

    return {
        "source_bootstrap_mapping_version": (
            TELEOP_V12_IDENTITY_NORMALIZER_BOOTSTRAP_MAPPING_VERSION
        ),
        "target_bootstrap_mapping_version": TELEOP_V12_BOOTSTRAP_MAPPING_VERSION,
        "authenticated_source_extra_normalizer": "identity_mean0_var1_std1",
        "normalized_target_position_columns": list(
            TELEOP_V12_TARGET_POSITION_OBSERVATION_COLUMNS
        ),
        "target_position_denominators": list(
            TELEOP_V12_TARGET_POSITION_NORMALIZER_DENOMINATORS
        ),
        "target_position_stored_std": list(
            TELEOP_V12_TARGET_POSITION_NORMALIZER_STORED_STD
        ),
        "target_position_stored_var": [
            value * value for value in TELEOP_V12_TARGET_POSITION_NORMALIZER_STORED_STD
        ],
        "normalizer_eps": LEGACY_VELOCITY_NORMALIZER_EPS,
        "unchanged_identity_normalizer_columns": list(
            TELEOP_V12_UNCHANGED_EXTRA_NORMALIZER_COLUMNS
        ),
        "hand_target_fk": microban_hand_fk_metadata(),
    }


def teleop_v12_active_adapter_columns(common_step_counter: int) -> tuple[int, ...]:
    """Return the only W0 columns allowed to update at this curriculum step."""

    if isinstance(common_step_counter, bool) or common_step_counter < 0:
        raise ValueError("common_step_counter must be a non-negative integer")
    # The provider is read by the gradient hook after rollout collection.  Keep
    # the exact boundary locked so the batch gathered before the curriculum
    # transition cannot update newly enabled columns.  The following rollout
    # ends above the boundary and is the first eligible batch.
    if common_step_counter <= 7_000 * 24:
        return ()
    if common_step_counter <= 10_000 * 24:
        return (
            *TELEOP_V12_HMD_OBSERVATION_COLUMNS,
            *TELEOP_V12_HAND_OBSERVATION_COLUMNS,
        )
    return TELEOP_V12_EXTRA_OBSERVATION_COLUMNS


def transplant_legacy_actor_state_to_teleop83(
    source_state: Mapping[str, torch.Tensor],
    target_template: Mapping[str, torch.Tensor],
    source_to_target_columns: Sequence[tuple[int, int]] = (
        LEGACY_TO_TELEOP_OBSERVATION_INDEX
    ),
) -> dict[str, torch.Tensor]:
    """Map the pinned normalized 63-input actor into the 83-input actor.

    The 63 shared normalization statistics and first-layer columns are copied by
    semantic scalar name.  The 12 bounded foot/hand position columns use their
    physical maximum as the effective normalization denominator; the remaining
    eight teleoperation-only columns retain identity normalization.  All 20 new
    first-layer weights are exact zero.  All downstream tensors, including the
    scalar Gaussian standard deviation, are copied verbatim.
    """

    if set(source_state) != LEGACY_VELOCITY_ACTOR_STATE_KEYS:
        raise ValueError("Legacy source actor state keys drifted")
    if set(target_template) != LEGACY_VELOCITY_ACTOR_STATE_KEYS:
        raise ValueError("83-input target actor state keys drifted")
    pairs = tuple(
        (int(source), int(target)) for source, target in source_to_target_columns
    )
    if tuple(source for source, _target in pairs) != tuple(
        range(LEGACY_VELOCITY_OBSERVATION_WIDTH)
    ):
        raise ValueError("Actor transplant must cover source columns 0..62 in order")
    target_columns = tuple(target for _source, target in pairs)
    if len(set(target_columns)) != LEGACY_VELOCITY_OBSERVATION_WIDTH or any(
        target < 0 or target >= MICROBAN_TELEOP_OBSERVATION_WIDTH
        for target in target_columns
    ):
        raise ValueError("Actor transplant target columns must be 63 unique indices")

    source_shapes = {
        "obs_normalizer._mean": (1, 63),
        "obs_normalizer._var": (1, 63),
        "obs_normalizer._std": (1, 63),
        "obs_normalizer.count": (),
        "distribution.std_param": (18,),
        "mlp.0.weight": (512, 63),
        "mlp.0.bias": (512,),
        "mlp.2.weight": (256, 512),
        "mlp.2.bias": (256,),
        "mlp.4.weight": (128, 256),
        "mlp.4.bias": (128,),
        "mlp.6.weight": (18, 128),
        "mlp.6.bias": (18,),
    }
    target_shapes = dict(source_shapes)
    target_shapes.update(
        {
            "obs_normalizer._mean": (1, 83),
            "obs_normalizer._var": (1, 83),
            "obs_normalizer._std": (1, 83),
            "mlp.0.weight": (512, 83),
        }
    )
    for name, shape in source_shapes.items():
        value = source_state[name]
        if tuple(value.shape) != shape or not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"Legacy source tensor {name!r} is invalid")
    for name, shape in target_shapes.items():
        if tuple(target_template[name].shape) != shape:
            raise ValueError(f"Teleop target tensor {name!r} shape drifted")

    result = {name: value.detach().clone() for name, value in target_template.items()}
    source_columns = torch.tensor(
        [source for source, _target in pairs], dtype=torch.long
    )
    target_columns_tensor = torch.tensor(target_columns, dtype=torch.long)
    for name, fill in (
        ("obs_normalizer._mean", 0.0),
        ("obs_normalizer._var", 1.0),
        ("obs_normalizer._std", 1.0),
    ):
        target = result[name]
        target.fill_(fill)
        target_indices = target_columns_tensor.to(target.device)
        target[:, target_indices] = source_state[name].to(
            device=target.device, dtype=target.dtype
        )[:, source_columns.to(target.device)]

    target_position_columns = torch.tensor(
        TELEOP_V12_TARGET_POSITION_OBSERVATION_COLUMNS, dtype=torch.long
    )
    stored_std = (
        result["obs_normalizer._std"]
        .new_tensor(TELEOP_V12_TARGET_POSITION_NORMALIZER_STORED_STD)
        .unsqueeze(0)
    )
    position_indices = target_position_columns.to(stored_std.device)
    result["obs_normalizer._mean"][:, position_indices] = 0.0
    result["obs_normalizer._std"][:, position_indices] = stored_std
    result["obs_normalizer._var"][:, position_indices] = stored_std.square()

    first = result["mlp.0.weight"]
    first.zero_()
    first[:, target_columns_tensor.to(first.device)] = source_state["mlp.0.weight"].to(
        device=first.device, dtype=first.dtype
    )[:, source_columns.to(first.device)]
    for name in LEGACY_VELOCITY_ACTOR_STATE_KEYS - {
        "obs_normalizer._mean",
        "obs_normalizer._var",
        "obs_normalizer._std",
        "mlp.0.weight",
    }:
        result[name].copy_(
            source_state[name].to(device=result[name].device, dtype=result[name].dtype)
        )
    return result


class FrozenEmpiricalNormalization(EmpiricalNormalization):
    """State-compatible empirical normalizer whose moments can never mutate."""

    def __init__(self, shape: int | tuple[int, ...] | list[int]) -> None:
        super().__init__(shape, eps=LEGACY_VELOCITY_NORMALIZER_EPS)

    @torch.jit.unused
    def update(self, x: torch.Tensor) -> None:
        """Deliberately ignore PPO updates, including while the parent is training."""

        del x


class LegacyAdapterTeleopActor(MLPModel):
    """Standard 83-wide MLP with only the 20 new input columns trainable."""

    def __init__(
        self,
        obs: Any,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = True,
        distribution_cfg: dict | None = None,
    ) -> None:
        if tuple(hidden_dims) != TELEOP_V12_ACTOR_TOPOLOGY[1:-1]:
            raise ValueError("Contract-v12 actor hidden topology drifted")
        if activation != "elu":
            raise ValueError("Contract-v12 actor requires ELU")
        if output_dim != MICROBAN_TELEOP_ACTION_WIDTH:
            raise ValueError("Contract-v12 actor requires 18 outputs")
        if not obs_normalization:
            raise ValueError("Contract-v12 actor requires legacy normalization")
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
        )
        if self.obs_dim != MICROBAN_TELEOP_OBSERVATION_WIDTH:
            raise ValueError(
                "Contract-v12 actor requires an 83-value observation, got "
                f"{self.obs_dim}"
            )
        if not isinstance(self.obs_normalizer, EmpiricalNormalization):
            raise TypeError("Contract-v12 actor normalizer was not empirical")
        frozen = FrozenEmpiricalNormalization(self.obs_dim)
        frozen.load_state_dict(self.obs_normalizer.state_dict(), strict=True)
        self.obs_normalizer = frozen
        if type(self.distribution) is not GaussianDistribution:
            raise TypeError("Contract-v12 actor requires upstream GaussianDistribution")
        if self.distribution.std_type != "scalar":
            raise ValueError("Contract-v12 actor requires scalar std parameterization")

        first = self.mlp[0]
        if not isinstance(first, torch.nn.Linear) or tuple(first.weight.shape) != (
            512,
            MICROBAN_TELEOP_OBSERVATION_WIDTH,
        ):
            raise TypeError("Contract-v12 first actor layer drifted")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        first.weight.requires_grad_(True)
        self._gradient_hook = first.weight.register_hook(self._mask_first_gradient)
        self._frozen_reference: dict[str, torch.Tensor] | None = None
        self._common_step_provider: Callable[[], int] | None = None

    def bind_common_step_provider(self, provider: Callable[[], int]) -> None:
        """Bind the runner clock without adding a checkpoint tensor or parameter."""

        if not callable(provider):
            raise TypeError("Contract-v12 common-step provider must be callable")
        self._common_step_provider = provider

    def active_adapter_columns(self) -> tuple[int, ...]:
        provider = self._common_step_provider
        if provider is None:
            raise RuntimeError("Contract-v12 common-step provider is not bound")
        value = provider()
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError("Contract-v12 common-step provider returned a non-integer")
        return teleop_v12_active_adapter_columns(value)

    def _mask_first_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
        if tuple(gradient.shape) != (512, MICROBAN_TELEOP_OBSERVATION_WIDTH):
            raise RuntimeError("Contract-v12 first-layer gradient shape drifted")
        masked = torch.zeros_like(gradient)
        active = self.active_adapter_columns()
        if active:
            masked[:, active] = gradient[:, active]
        return masked

    def bind_frozen_legacy_reference(self) -> None:
        """Snapshot the loaded bootstrap tensors used by every update invariant."""

        state = self.state_dict()
        self._frozen_reference = {
            name: value.detach().cpu().clone() for name, value in state.items()
        }
        self.assert_frozen_legacy_state()

    def assert_frozen_legacy_state(self) -> None:
        """Require every legacy tensor/column to remain bit-identical."""

        reference = self._frozen_reference
        if reference is None:
            raise RuntimeError("Contract-v12 legacy reference is not bound")
        state = self.state_dict()
        if set(state) != set(reference):
            raise RuntimeError("Contract-v12 actor state keys changed after bootstrap")
        for name, current in state.items():
            if not bool(torch.isfinite(current).all().item()):
                raise RuntimeError(f"Contract-v12 actor state is non-finite: {name}")
            expected = reference[name]
            candidate = current.detach().cpu()
            if name == "mlp.0.weight":
                candidate = candidate[:, TELEOP_V12_SHARED_OBSERVATION_COLUMNS]
                expected = expected[:, TELEOP_V12_SHARED_OBSERVATION_COLUMNS]
            if not torch.equal(candidate, expected):
                raise RuntimeError(f"Frozen contract-v12 actor state changed: {name}")

    def assert_optimizer_invariant(self, optimizer: torch.optim.Optimizer) -> None:
        """Reject Adam momentum that could move a masked legacy W0 column."""

        first = self.mlp[0]
        assert isinstance(first, torch.nn.Linear)
        optimizer_state: Mapping[str, object] = optimizer.state.get(first.weight, {})
        for name, value in optimizer_state.items():
            if isinstance(value, torch.Tensor) and not bool(
                torch.isfinite(value).all().item()
            ):
                raise RuntimeError(
                    f"Contract-v12 optimizer state is non-finite: {name}"
                )
        active = set(self.active_adapter_columns())
        locked_columns = tuple(
            index
            for index in range(MICROBAN_TELEOP_OBSERVATION_WIDTH)
            if index not in active
        )
        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            value = optimizer_state.get(name)
            if value is None:
                continue
            if not isinstance(value, torch.Tensor) or value.shape != first.weight.shape:
                raise RuntimeError(f"Contract-v12 optimizer {name} shape drifted")
            locked = value.detach()[:, locked_columns]
            if not torch.equal(locked, torch.zeros_like(locked)):
                raise RuntimeError(
                    "Contract-v12 optimizer can mutate a schedule-locked W0 "
                    f"column: {name}"
                )

    def assert_schedule_locked_weights_zero(self) -> None:
        """Reject contamination in extra columns not yet exposed by curriculum."""

        active = set(self.active_adapter_columns())
        locked_extra = tuple(
            index
            for index in TELEOP_V12_EXTRA_OBSERVATION_COLUMNS
            if index not in active
        )
        if not locked_extra:
            return
        first = self.mlp[0]
        assert isinstance(first, torch.nn.Linear)
        locked = first.weight.detach()[:, locked_extra]
        if not torch.equal(locked, torch.zeros_like(locked)):
            raise RuntimeError(
                "Contract-v12 schedule-locked adapter W0 column is non-zero"
            )

    def trainable_actor_parameter_names(self) -> tuple[str, ...]:
        return tuple(
            name for name, value in self.named_parameters() if value.requires_grad
        )


class LegacyAdapterPPO(PPO):
    """PPO that verifies frozen legacy weights and Adam state every update."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if not isinstance(self.actor, LegacyAdapterTeleopActor):
            raise TypeError("LegacyAdapterPPO requires LegacyAdapterTeleopActor")
        self._invariant_hook = self.optimizer.register_step_post_hook(
            self._validate_after_optimizer_step
        )

    def _validate_after_optimizer_step(
        self,
        _optimizer: torch.optim.Optimizer,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
    ) -> None:
        self._validate_legacy_adapter()

    def _validate_legacy_adapter(self) -> None:
        if not isinstance(self.actor, LegacyAdapterTeleopActor):
            raise TypeError("LegacyAdapterPPO requires LegacyAdapterTeleopActor")
        self.actor.assert_frozen_legacy_state()
        self.actor.assert_schedule_locked_weights_zero()
        self.actor.assert_optimizer_invariant(self.optimizer)

    def update(self) -> dict[str, float]:
        self._validate_legacy_adapter()
        result = super().update()
        self._validate_legacy_adapter()
        return result
