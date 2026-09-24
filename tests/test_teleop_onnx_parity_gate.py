# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for deterministic teleop ONNX parity and provenance publication."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_TELEOP_ACTION_WIDTH,
    MICROBAN_TELEOP_OBSERVATION_WIDTH,
    TELEOP_ONNX_PARITY_SAMPLE_COUNT,
    TELEOP_ONNX_PARITY_SEED,
    collect_teleop_export_provenance,
    deterministic_teleop_parity_inputs,
    publish_gated_teleop_onnx,
    unique_teleop_onnx_temporary_path,
    validate_pytorch_onnx_parity,
)
from mjlab_microban.tasks.microban_teleop_env_cfg import (
    microban_teleop_action_delta_bounds,
    microban_teleop_initial_action_std,
)
from mjlab_microban.tasks.microban_teleop_mdp import (
    AsymmetricBoundedGaussianDistribution,
)


def _write_zero_policy(path: Path) -> None:
    obs = helper.make_tensor_value_info(
        "obs", TensorProto.FLOAT, [1, MICROBAN_TELEOP_OBSERVATION_WIDTH]
    )
    actions = helper.make_tensor_value_info(
        "actions", TensorProto.FLOAT, [1, MICROBAN_TELEOP_ACTION_WIDTH]
    )
    weight = helper.make_tensor(
        "weight",
        TensorProto.FLOAT,
        [MICROBAN_TELEOP_OBSERVATION_WIDTH, MICROBAN_TELEOP_ACTION_WIDTH],
        [0.0] * (MICROBAN_TELEOP_OBSERVATION_WIDTH * MICROBAN_TELEOP_ACTION_WIDTH),
    )
    graph = helper.make_graph(
        [helper.make_node("MatMul", ("obs", "weight"), ("actions",))],
        "microban_parity_test",
        [obs],
        [actions],
        [weight],
    )
    onnx.save(helper.make_model(graph), path)


class _ZeroPolicy(torch.nn.Module):
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return observation[:, :MICROBAN_TELEOP_ACTION_WIDTH] * 0.0


class _OnePolicy(torch.nn.Module):
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return observation[:, :MICROBAN_TELEOP_ACTION_WIDTH] * 0.0 + 1.0


class _DeterministicMlpPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = torch.nn.Linear(MICROBAN_TELEOP_OBSERVATION_WIDTH, 32)
        self.output = torch.nn.Linear(32, MICROBAN_TELEOP_ACTION_WIDTH)
        with torch.no_grad():
            for index, parameter in enumerate(self.parameters()):
                values = torch.linspace(-0.02, 0.02, parameter.numel())
                parameter.copy_(values.reshape_as(parameter) + index * 0.001)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.tanh(
            self.output(torch.nn.functional.elu(self.hidden(observation)))
        )


class DeterministicCorpusTest(unittest.TestCase):
    def test_corpus_is_repeatable_finite_and_seed_sensitive(self) -> None:
        first = deterministic_teleop_parity_inputs()
        second = deterministic_teleop_parity_inputs()
        different_seed = deterministic_teleop_parity_inputs(
            seed=TELEOP_ONNX_PARITY_SEED + 1
        )

        self.assertEqual(
            first.shape,
            (
                TELEOP_ONNX_PARITY_SAMPLE_COUNT,
                1,
                MICROBAN_TELEOP_OBSERVATION_WIDTH,
            ),
        )
        self.assertEqual(first.dtype, np.float32)
        self.assertTrue(np.isfinite(first).all())
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first[4:], different_seed[4:]))
        self.assertTrue(np.isin(first[4:, 0, -2:], (0.0, 1.0)).all())

    def test_rejects_too_few_samples(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least four"):
            deterministic_teleop_parity_inputs(sample_count=3)


class ParityGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.checkpoint = self.root / "model_42.pt"
        self.checkpoint.write_bytes(b"stable checkpoint bytes")
        self.temporary_onnx = self.root / ".policy.onnx.tmp"
        self.output_onnx = self.root / "policy.onnx"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_collects_explicit_iteration_and_checkpoint_digest(self) -> None:
        provenance = collect_teleop_export_provenance(self.checkpoint)

        self.assertEqual(provenance.checkpoint_iteration, 42)
        self.assertEqual(
            provenance.checkpoint_sha256,
            hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(),
        )
        self.assertEqual(len(provenance.exporter_source_sha256), 64)

    def test_rejects_checkpoint_without_iteration_filename(self) -> None:
        checkpoint = self.root / "latest.pt"
        checkpoint.write_bytes(b"checkpoint")
        with self.assertRaisesRegex(ValueError, r"model_<iteration>\.pt"):
            collect_teleop_export_provenance(checkpoint)

    def test_provenance_keeps_canonical_target_after_symlink_retarget(self) -> None:
        first = self.root / "first" / "model_42.pt"
        second = self.root / "second" / "model_42.pt"
        first.parent.mkdir()
        second.parent.mkdir()
        first.write_bytes(b"first checkpoint")
        second.write_bytes(b"second checkpoint")
        selected = self.root / "selected" / "model_42.pt"
        selected.parent.mkdir()
        selected.symlink_to(first)

        provenance = collect_teleop_export_provenance(selected)
        selected.unlink()
        selected.symlink_to(second)

        self.assertEqual(provenance.checkpoint_path, first.resolve())
        self.assertEqual(
            provenance.checkpoint_sha256,
            hashlib.sha256(first.read_bytes()).hexdigest(),
        )

    def test_unique_temporary_paths_share_only_the_output_directory(self) -> None:
        first = unique_teleop_onnx_temporary_path(self.output_onnx)
        second = unique_teleop_onnx_temporary_path(self.output_onnx)

        self.assertEqual(first.parent, self.output_onnx.parent)
        self.assertEqual(second.parent, self.output_onnx.parent)
        self.assertNotEqual(first, second)
        self.assertNotEqual(first, self.output_onnx)

    def test_parity_rejects_different_pytorch_policy(self) -> None:
        _write_zero_policy(self.temporary_onnx)
        with self.assertRaisesRegex(ValueError, "parity failed"):
            validate_pytorch_onnx_parity(_OnePolicy(), self.temporary_onnx)

    def test_legacy_torch_export_matches_reference_evaluator(self) -> None:
        policy = _DeterministicMlpPolicy().eval()
        torch.onnx.export(
            policy,
            (torch.zeros(1, MICROBAN_TELEOP_OBSERVATION_WIDTH),),
            self.temporary_onnx,
            export_params=True,
            opset_version=18,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={},
            dynamo=False,
        )

        result = validate_pytorch_onnx_parity(policy, self.temporary_onnx)

        self.assertLessEqual(result.max_absolute_error, 1e-5)

    def test_v7_real_actor_onnx_matches_bounded_deterministic_policy(self) -> None:
        torch.manual_seed(37)
        obs = TensorDict(
            {"policy": torch.zeros(1, MICROBAN_TELEOP_OBSERVATION_WIDTH)},
            batch_size=[1],
        )
        lower, upper = microban_teleop_action_delta_bounds()
        actor = MLPModel(
            obs=obs,
            obs_groups={"actor": ["policy"]},
            obs_set="actor",
            output_dim=MICROBAN_TELEOP_ACTION_WIDTH,
            hidden_dims=(16,),
            distribution_cfg={
                "class_name": AsymmetricBoundedGaussianDistribution,
                "init_std": microban_teleop_initial_action_std(),
                "lower_bound": lower,
                "upper_bound": upper,
                "std_type": "log",
            },
        )
        policy = actor.as_onnx(verbose=False).eval()
        torch.onnx.export(
            policy,
            (torch.zeros(1, MICROBAN_TELEOP_OBSERVATION_WIDTH),),
            self.temporary_onnx,
            export_params=True,
            opset_version=18,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={},
            dynamo=False,
        )

        result = validate_pytorch_onnx_parity(policy, self.temporary_onnx)
        self.assertLessEqual(result.max_absolute_error, 1e-5)
        with torch.inference_mode():
            output = policy(torch.as_tensor(deterministic_teleop_parity_inputs()[0]))
        self.assertTrue(torch.isfinite(output).all().item())
        self.assertTrue(torch.all(output > torch.tensor(lower)).item())
        self.assertTrue(torch.all(output < torch.tensor(upper)).item())

    def test_success_publishes_metadata_bearing_artifact(self) -> None:
        _write_zero_policy(self.temporary_onnx)
        self.output_onnx.write_bytes(b"old artifact")
        provenance = collect_teleop_export_provenance(self.checkpoint)

        result = publish_gated_teleop_onnx(
            self.temporary_onnx,
            self.output_onnx,
            pytorch_policy=_ZeroPolicy(),
            policy_metadata={"policy_type": "microban_pico_hybrid_teleop"},
            provenance=provenance,
        )

        self.assertFalse(self.temporary_onnx.exists())
        self.assertEqual(result.sample_count, TELEOP_ONNX_PARITY_SAMPLE_COUNT)
        self.assertEqual(result.max_absolute_error, 0.0)
        metadata = {
            entry.key: entry.value
            for entry in onnx.load(self.output_onnx).metadata_props
        }
        self.assertEqual(metadata["checkpoint_iteration"], "42")
        self.assertEqual(metadata["checkpoint_completed_updates"], "43")
        self.assertEqual(metadata["checkpoint_sha256"], provenance.checkpoint_sha256)
        self.assertEqual(metadata["onnx_parity_verified"], "true")
        self.assertEqual(
            metadata["onnx_parity_runtime"],
            "onnx.reference.ReferenceEvaluator",
        )
        self.assertEqual(metadata["policy_type"], "microban_pico_hybrid_teleop")

    def test_parity_failure_preserves_last_known_good_output(self) -> None:
        _write_zero_policy(self.temporary_onnx)
        known_good = b"last known good"
        self.output_onnx.write_bytes(known_good)
        provenance = collect_teleop_export_provenance(self.checkpoint)

        with self.assertRaisesRegex(ValueError, "parity failed"):
            publish_gated_teleop_onnx(
                self.temporary_onnx,
                self.output_onnx,
                pytorch_policy=_OnePolicy(),
                policy_metadata={},
                provenance=provenance,
            )

        self.assertFalse(self.temporary_onnx.exists())
        self.assertEqual(self.output_onnx.read_bytes(), known_good)

    def test_checkpoint_mutation_preserves_last_known_good_output(self) -> None:
        _write_zero_policy(self.temporary_onnx)
        known_good = b"last known good"
        self.output_onnx.write_bytes(known_good)
        provenance = collect_teleop_export_provenance(self.checkpoint)
        self.checkpoint.write_bytes(b"checkpoint changed during export")

        with self.assertRaisesRegex(RuntimeError, "Checkpoint changed"):
            publish_gated_teleop_onnx(
                self.temporary_onnx,
                self.output_onnx,
                pytorch_policy=_ZeroPolicy(),
                policy_metadata={},
                provenance=provenance,
            )

        self.assertFalse(self.temporary_onnx.exists())
        self.assertEqual(self.output_onnx.read_bytes(), known_good)


if __name__ == "__main__":
    unittest.main()
