"""Regression test for the in-place v12 adapter sanitizer."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from mjlab_microban.scripts.sanitize_teleop_v12_adapter_checkpoint import (
    _authenticate_identity_normalizer_v1_actor,
    _zero_extra_columns_in_place,
)
from mjlab_microban.tasks.microban_teleop_v12_actor import (
    TELEOP_V12_EXTRA_OBSERVATION_COLUMNS,
    TELEOP_V12_SHARED_OBSERVATION_COLUMNS,
    TELEOP_V12_TARGET_POSITION_NORMALIZER_STORED_STD,
    TELEOP_V12_TARGET_POSITION_OBSERVATION_COLUMNS,
    transplant_legacy_actor_state_to_teleop83,
)


def _model(width: int) -> MLPModel:
    observations = TensorDict({"actor": torch.zeros(1, width)}, batch_size=[1])
    return MLPModel(
        obs=observations,
        obs_groups={"actor": ["actor"]},
        obs_set="actor",
        output_dim=18,
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    )


class TeleopV12SanitizerTest(unittest.TestCase):
    def test_reset_mutates_original_tensor_and_only_extra_columns(self) -> None:
        tensor = torch.ones(512, 83)
        storage = tensor.data_ptr()
        _zero_extra_columns_in_place(tensor)
        self.assertEqual(tensor.data_ptr(), storage)
        self.assertTrue(
            torch.equal(
                tensor[:, TELEOP_V12_EXTRA_OBSERVATION_COLUMNS],
                torch.zeros(512, 20),
            )
        )
        self.assertTrue(
            torch.equal(
                tensor[:, TELEOP_V12_SHARED_OBSERVATION_COLUMNS],
                torch.ones(512, 63),
            )
        )

    def test_reset_rejects_wrong_shape(self) -> None:
        with self.assertRaises(ValueError):
            _zero_extra_columns_in_place(torch.ones(512, 82))

    def test_v1_identity_normalizer_is_authenticated_before_scaling(self) -> None:
        torch.manual_seed(17)
        source_state = _model(63).state_dict()
        target_template = _model(83).state_dict()
        scaled = transplant_legacy_actor_state_to_teleop83(
            source_state, target_template
        )
        old_v1 = {name: value.clone() for name, value in scaled.items()}
        for name, fill in (
            ("obs_normalizer._mean", 0.0),
            ("obs_normalizer._var", 1.0),
            ("obs_normalizer._std", 1.0),
        ):
            old_v1[name][:, TELEOP_V12_EXTRA_OBSERVATION_COLUMNS] = fill
        old_v1["mlp.0.weight"][:, TELEOP_V12_EXTRA_OBSERVATION_COLUMNS] = 0.25
        identity = SimpleNamespace(path="source.pt", sha256="a" * 64)
        provenance = SimpleNamespace(source=identity)
        with patch(
            "mjlab_microban.scripts.sanitize_teleop_v12_adapter_checkpoint."
            "inspect_legacy_velocity_checkpoint",
            return_value=(identity, source_state),
        ):
            authenticated = _authenticate_identity_normalizer_v1_actor(
                old_v1, provenance
            )
            self.assertTrue(
                torch.equal(
                    authenticated["obs_normalizer._std"][
                        :, TELEOP_V12_TARGET_POSITION_OBSERVATION_COLUMNS
                    ],
                    torch.tensor([TELEOP_V12_TARGET_POSITION_NORMALIZER_STORED_STD]),
                )
            )
            old_v1["obs_normalizer._std"][
                :, TELEOP_V12_TARGET_POSITION_OBSERVATION_COLUMNS[0]
            ] = 0.5
            with self.assertRaisesRegex(ValueError, "identity normalizer"):
                _authenticate_identity_normalizer_v1_actor(old_v1, provenance)


if __name__ == "__main__":
    unittest.main()
