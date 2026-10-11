# Adapted from Teleopit f926386 train_mimic/tasks/tracking/rl/conv1d_encoder.py
# and temporal_cnn_model.py (Apache-2.0, https://github.com/BotRunner64/Teleopit).
# Changes: one module; every input is a history whose last row is the current
# frame (Teleopit took the current frame as a group of its own); ONNX export
# only.

"""Temporal CNN model: a 1-D CNN over the observation history plus an MLP head.

Follows the extension pattern of ``rsl_rl.models.cnn_model.CNNModel``.  Every
observation group is a history ``(B, T, D)``, oldest first; the MLP sees the
normalized current frames (the last rows) and the CNN encoding of each
normalized history.  The exported ONNX takes one ``obs_<name>`` ``(1, T, D)``
per group.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import EmpiricalNormalization, HiddenState
from tensordict import TensorDict

_ACTIVATIONS = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
}

_POOLS = {
    "avg": nn.AdaptiveAvgPool1d,
    "max": nn.AdaptiveMaxPool1d,
}


class Conv1dEncoder(nn.Module):
    """Encode a ``(B, input_channels, seq_len)`` sequence into ``(B, C)``."""

    def __init__(
        self,
        input_channels: int,
        output_channels: tuple[int, ...] = (64, 32),
        kernel_size: int = 3,
        activation: str = "elu",
        global_pool: str = "avg",
    ) -> None:
        super().__init__()

        act_cls = _ACTIVATIONS[activation]
        layers: list[nn.Module] = []
        in_ch = input_channels
        for out_ch in output_channels:
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2))
            layers.append(act_cls())
            in_ch = out_ch
        pool_cls = _POOLS[global_pool]
        layers.append(pool_cls(1))
        layers.append(nn.Flatten(start_dim=1))
        self.net = nn.Sequential(*layers)
        self._output_dim = output_channels[-1]

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``(B, input_channels, seq_len)``

        Returns:
            ``(B, output_dim)``
        """
        return self.net(x)


def _export_input_name(group_name: str) -> str:
    """Map observation-group names to stable export input names."""
    for prefix in ("actor_", "critic_"):
        if group_name.startswith(prefix):
            return "obs_" + group_name[len(prefix):]
    return group_name


def _current_frames(histories: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
    """The last row of each ``(B, T, D)`` history, concatenated: ``(B, sum D)``."""
    return torch.cat([history[:, -1] for history in histories], dim=-1)


class TemporalCNNModel(MLPModel):
    """MLP over the current frames and the 1-D CNN encodings of the histories.

    Each history group gets its own ``Conv1dEncoder`` and its own
    ``EmpiricalNormalization``; the current frames share the parent's
    normalizer.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        cnn_cfg: dict[str, Any] | None = None,
    ) -> None:
        groups, _ = self._get_obs_dim(obs, obs_groups, obs_set)
        cnn_cfg = cnn_cfg or {}
        # A plain dict until nn.Module is initialized; registered below.
        encoders = {group: Conv1dEncoder(input_channels=dim, **cnn_cfg)
                    for group, dim in zip(groups, self.obs_dims_3d)}
        self.cnn_latent_dim = sum(encoder.output_dim for encoder in encoders.values())
        super().__init__(
            obs, obs_groups, obs_set, output_dim, hidden_dims, activation, obs_normalization, distribution_cfg
        )
        self.cnn_encoders = nn.ModuleDict(encoders)
        self.obs_normalizers_3d = nn.ModuleDict({
            group: EmpiricalNormalization(dim) if obs_normalization else nn.Identity()
            for group, dim in zip(self.obs_groups, self.obs_dims_3d)
        })

    def _get_obs_dim(
        self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str
    ) -> tuple[list[str], int]:
        """The groups and the width of their current frames (the MLP's normalized input)."""
        groups = list(obs_groups[obs_set])
        for group in groups:
            if obs[group].ndim != 3:
                raise ValueError(f"TemporalCNNModel expects (B, T, D) histories, got {tuple(obs[group].shape)} "
                                 f"for '{group}'")
        self.obs_dims_3d = [int(obs[group].shape[-1]) for group in groups]
        self.history_lengths = [int(obs[group].shape[1]) for group in groups]
        return groups, sum(self.obs_dims_3d)

    def _get_latent_dim(self) -> int:
        return self.obs_dim + self.cnn_latent_dim

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        histories = [obs[group] for group in self.obs_groups]
        parts = [self.obs_normalizer(_current_frames(histories))]
        for group, history in zip(self.obs_groups, histories):
            h = self.obs_normalizers_3d[group](history).permute(0, 2, 1)  # (B, D, T) for Conv1d
            parts.append(self.cnn_encoders[group](h))
        return torch.cat(parts, dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if not self.obs_normalization:
            return
        histories = [obs[group] for group in self.obs_groups]
        self.obs_normalizer.update(_current_frames(histories))  # type: ignore[operator]
        for group, history in zip(self.obs_groups, histories):
            batch, length, width = history.shape
            self.obs_normalizers_3d[group].update(history.reshape(batch * length, width))  # type: ignore[operator]

    def as_jit(self) -> nn.Module:
        raise NotImplementedError("the whole-body policy is exported with as_onnx")

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        return _OnnxTemporalCNNModel(self, verbose)


class _OnnxTemporalCNNModel(nn.Module):
    def __init__(self, model: TemporalCNNModel, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.cnn_normalizers = nn.ModuleList([copy.deepcopy(model.obs_normalizers_3d[g]) for g in model.obs_groups])
        self.cnn_encoders = nn.ModuleList([copy.deepcopy(model.cnn_encoders[g]) for g in model.obs_groups])
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self._obs_dims_3d = model.obs_dims_3d
        self._history_lengths = model.history_lengths
        self._input_names = [_export_input_name(g) for g in model.obs_groups]

    def forward(self, *histories: torch.Tensor) -> torch.Tensor:
        """Deterministic action from ``(1, T_i, D_i)`` histories, oldest first."""
        if len(histories) != len(self.cnn_encoders):
            raise ValueError(f"expected inputs {self._input_names}, got {len(histories)}")
        parts = [self.obs_normalizer(_current_frames(histories))]
        for history, normalizer, encoder in zip(histories, self.cnn_normalizers, self.cnn_encoders, strict=True):
            parts.append(encoder(normalizer(history).permute(0, 2, 1)))
        return self.deterministic_output(self.mlp(torch.cat(parts, dim=-1)))

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        return tuple(
            torch.zeros(1, length, width) for length, width in zip(self._history_lengths, self._obs_dims_3d, strict=True)
        )

    @property
    def input_names(self) -> list[str]:
        return list(self._input_names)

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
