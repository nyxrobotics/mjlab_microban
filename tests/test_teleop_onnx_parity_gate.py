# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for deterministic teleop ONNX parity and provenance publication."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from mjlab_microban.scripts import evaluate_teleop_checkpoint as teleop_evaluator
from mjlab_microban.scripts import export_teleop_onnx as teleop_exporter
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_TELEOP_ACTION_WIDTH,
    MICROBAN_TELEOP_ACTOR_INITIALIZATION,
    MICROBAN_TELEOP_OBSERVATION_WIDTH,
    MICROBAN_TELEOP_PREVIOUS_ACTION_SEMANTICS,
    MICROBAN_TELEOP_RECIPE_REVISION,
    MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION,
    TELEOP_ONNX_PARITY_SAMPLE_COUNT,
    TELEOP_ONNX_PARITY_SEED,
    collect_teleop_export_provenance,
    deterministic_teleop_parity_inputs,
    publish_gated_teleop_onnx,
    unique_teleop_onnx_temporary_path,
    validate_pytorch_onnx_parity,
)
from mjlab_microban.tasks.microban_safe_velocity_checkpoint import (
    inspect_safe_velocity_checkpoint,
)
from mjlab_microban.tasks.microban_safe_velocity_env_cfg import (
    microban_safe_velocity_action_delta_bounds,
    microban_safe_velocity_initial_action_std,
)
from mjlab_microban.tasks.microban_safe_velocity_mdp import (
    MICROBAN_SAFE_VELOCITY_RECIPE_INFO_KEY,
    MICROBAN_SAFE_VELOCITY_RECIPE_REVISION,
    MicrobanSafeVelocityBoundedGaussianDistribution,
)
from mjlab_microban.tasks.microban_teleop_bootstrap import (
    SAFE_VELOCITY_ACTOR_BOOTSTRAP_MAPPING_VERSION,
    SAFE_VELOCITY_COPIED_DISTRIBUTION_KEYS,
    SAFE_VELOCITY_NEW_TELEOP_OBSERVATION_COLUMNS,
    SAFE_VELOCITY_TARGET_ACTOR_TOPOLOGY,
    SafeVelocityActorBootstrapProvenance,
    serialize_safe_velocity_actor_bootstrap_provenance,
    validate_safe_velocity_acceptance_receipt,
)
from mjlab_microban.tasks.microban_teleop_env_cfg import (
    microban_teleop_action_delta_bounds,
    microban_teleop_initial_action_std,
)
from mjlab_microban.tasks.microban_teleop_mdp import (
    AsymmetricBoundedGaussianDistribution,
)
from mjlab_microban.tasks.microban_teleop_provenance import (
    MICROBAN_TELEOP_CANONICAL_STAGE_MODE,
    MICROBAN_TELEOP_TRAINING_PROVENANCE_KEY,
    MICROBAN_TELEOP_TRAINING_PROVENANCE_SCHEMA_VERSION,
    MICROBAN_TELEOP_TRAINING_PROVENANCE_SHA256_KEY,
    canonical_json_sha256,
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


class ExportMetadataWiringTest(unittest.TestCase):
    def test_only_acceptance_backed_export_materializes_final_stage_bounds(
        self,
    ) -> None:
        raw_env = object()
        checkpoint = Path("/logs/canonical_run/model_19999.pt")
        for acceptance, expected in ((None, False), (object(), True)):
            with (
                self.subTest(acceptance=acceptance is not None),
                patch.object(
                    teleop_exporter,
                    "get_microban_teleop_metadata",
                    return_value={"marker": expected},
                ) as collect,
            ):
                result = teleop_exporter._collect_export_metadata(
                    raw_env,
                    checkpoint,
                    SimpleNamespace(acceptance=acceptance),
                )
                self.assertEqual(result, {"marker": expected})
                collect.assert_called_once_with(
                    raw_env,
                    run_path="canonical_run",
                    canonical_final_stage=expected,
                )


def _write_accepted_safe_source(root: Path) -> dict[str, object]:
    safe_root = root / "accepted_safe_source"
    safe_root.mkdir(exist_ok=True)
    source_path = safe_root / "model_7.pt"
    lower, upper = microban_safe_velocity_action_delta_bounds()
    actor = MLPModel(
        obs=TensorDict(
            {"actor": torch.zeros((1, 63), dtype=torch.float32)},
            batch_size=[1],
        ),
        obs_groups={"actor": ["actor"]},
        obs_set="actor",
        output_dim=18,
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=False,
        distribution_cfg={
            "class_name": MicrobanSafeVelocityBoundedGaussianDistribution,
            "init_std": microban_safe_velocity_initial_action_std(),
            "lower_bound": lower,
            "upper_bound": upper,
            "std_type": "log",
        },
    )
    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "iter": 7,
            "infos": {
                MICROBAN_SAFE_VELOCITY_RECIPE_INFO_KEY: (
                    MICROBAN_SAFE_VELOCITY_RECIPE_REVISION
                ),
                "env_state": {"common_step_counter": 8 * 24},
            },
        },
        source_path,
    )
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    identity = inspect_safe_velocity_checkpoint(
        source_path, expected_sha256=source_sha256
    )
    receipt_path = safe_root / "acceptance.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "gate": "microban_safe_velocity_fixed_forward_v3",
                "checkpoint": {
                    "path": str(identity.path),
                    "sha256": identity.sha256,
                    "iteration": identity.iteration,
                    "checkpoint_schema_version": identity.schema_version,
                    "recipe_revision": identity.recipe_revision,
                    "actor_topology": list(identity.actor_topology),
                    "actor_obs_normalization": identity.actor_obs_normalization,
                    "observation_schema": [
                        list(item) for item in identity.observation_schema
                    ],
                    "action_joint_names": list(identity.action_joint_names),
                },
                "configuration": {
                    "device": "cuda:0",
                    "num_envs": 64,
                    "steps_requested": 200,
                    "steps_executed": 200,
                    "seed": 42,
                    "command_vx_m_s": 0.08,
                    "actual_command_exactly_verified_each_step": True,
                    "step_dt_s": 0.02,
                    "guard_lookahead_s": 0.12,
                    "guard_margin_ratio": 0.05,
                    "absolute_clip_max_tensor_error_rad": 0.0,
                },
                "thresholds": {
                    "completion_fraction_min": 0.95,
                    "fall_fraction_max": 0.0,
                    "nonfinite_fraction_max": 0.0,
                    "forward_velocity_p05_m_s_min": 0.01,
                    "forward_displacement_p05_m_min": 0.02,
                    "actual_soft_limit_violation_rad_max": 1.0e-6,
                    "actual_lookahead_soft_limit_violation_rad_max": 1.0e-6,
                    "target_clip_rad_max": 1.0e-7,
                },
                "metrics": {
                    "completion_fraction": 1.0,
                    "fall_fraction": 0.0,
                    "nonfinite_fraction": 0.0,
                    "forward_velocity_p05_m_s": 0.03,
                    "forward_velocity_median_m_s": 0.03,
                    "forward_displacement_p05_m": 0.03,
                    "forward_displacement_median_m": 0.03,
                    "maximum_actual_soft_limit_violation_rad": 0.0,
                    "maximum_actual_lookahead_soft_limit_violation_rad": 0.0,
                    "maximum_preferred_margin_lookahead_excess_rad": 0.0,
                    "maximum_target_clip_rad": 0.0,
                    "minimum_root_height_m": 0.2,
                },
                "checks": {
                    name: True
                    for name in (
                        "completion",
                        "no_falls",
                        "finite",
                        "forward_velocity",
                        "forward_displacement",
                        "actual_soft_limits",
                        "actual_lookahead_soft_limits",
                        "absolute_target_clip",
                    )
                },
                "status": "pass",
                "summary": {"passed": True, "failed_checks": []},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    acceptance = validate_safe_velocity_acceptance_receipt(receipt_path, identity)
    return serialize_safe_velocity_actor_bootstrap_provenance(
        SafeVelocityActorBootstrapProvenance(
            source=identity,
            acceptance_receipt=acceptance,
            mapping_version=SAFE_VELOCITY_ACTOR_BOOTSTRAP_MAPPING_VERSION,
            target_actor_topology=SAFE_VELOCITY_TARGET_ACTOR_TOPOLOGY,
            new_teleop_observation_columns=(
                SAFE_VELOCITY_NEW_TELEOP_OBSERVATION_COLUMNS
            ),
            copied_distribution_keys=SAFE_VELOCITY_COPIED_DISTRIBUTION_KEYS,
        )
    )


def _write_generic_checkpoint(
    path: Path,
    bootstrap_info: dict[str, object],
    *,
    iteration: int = 42,
) -> None:
    files = {"src/mjlab_microban/test_recipe.py": "a" * 64}
    manifest = {
        "schema_version": MICROBAN_TELEOP_TRAINING_PROVENANCE_SCHEMA_VERSION,
        "canonical_stage": False,
        "training_contract_version": MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION,
        "recipe_revision": MICROBAN_TELEOP_RECIPE_REVISION,
        "actor_initialization": MICROBAN_TELEOP_ACTOR_INITIALIZATION,
        "resolved_config": {"critical": {}, "environment": {}, "runner": {}},
        "source": {
            "algorithm": "sha256(canonical_json_path_to_sha256_v1)",
            "tree_sha256": canonical_json_sha256(files),
            "files": files,
        },
        "invocation": {
            "mode": "generic",
            "stage_start_boundary": None,
            "stage_target_boundary": None,
            "parent_checkpoint_sha256": None,
            "parent_gate_sha256": None,
        },
    }
    torch.save(
        {
            "iter": iteration,
            "infos": {
                "env_state": {"common_step_counter": (iteration + 1) * 24},
                "microban_teleop_training_contract_version": (
                    MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION
                ),
                "previous_action_semantics": (
                    MICROBAN_TELEOP_PREVIOUS_ACTION_SEMANTICS
                ),
                "microban_teleop_actor_initialization": (
                    MICROBAN_TELEOP_ACTOR_INITIALIZATION
                ),
                "microban_teleop_recipe_revision": MICROBAN_TELEOP_RECIPE_REVISION,
                MICROBAN_TELEOP_TRAINING_PROVENANCE_KEY: manifest,
                MICROBAN_TELEOP_TRAINING_PROVENANCE_SHA256_KEY: (
                    canonical_json_sha256(manifest)
                ),
                "safe_velocity_actor_bootstrap": bootstrap_info,
            },
        },
        path,
    )


def _write_final_canonical_checkpoint(
    path: Path, bootstrap_info: dict[str, object]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    files = {"src/mjlab_microban/final_recipe.py": "b" * 64}
    manifest = {
        "schema_version": MICROBAN_TELEOP_TRAINING_PROVENANCE_SCHEMA_VERSION,
        "canonical_stage": True,
        "training_contract_version": MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION,
        "recipe_revision": MICROBAN_TELEOP_RECIPE_REVISION,
        "actor_initialization": MICROBAN_TELEOP_ACTOR_INITIALIZATION,
        "resolved_config": {
            "critical": {
                "num_envs": 2048,
                "environment_seed": 42,
                "runner_seed": 42,
                "num_steps_per_env": 24,
                "save_interval": 500,
                "max_iterations_for_process": 2000,
                "resume": True,
                "logger": "tensorboard",
                "upload_model": False,
                "wrapper_clip_actions": None,
                "checkpoint_consumer_mode": False,
                "bootstrap_velocity_checkpoint": None,
                "bootstrap_velocity_checkpoint_sha256": None,
                "safe_velocity_checkpoint": None,
                "safe_velocity_checkpoint_sha256": None,
                "safe_velocity_acceptance_receipt": None,
                "save_pristine_checkpoint": False,
            },
            "environment": {},
            "runner": {},
        },
        "source": {
            "algorithm": "sha256(canonical_json_path_to_sha256_v1)",
            "tree_sha256": canonical_json_sha256(files),
            "files": files,
        },
        "invocation": {
            "mode": MICROBAN_TELEOP_CANONICAL_STAGE_MODE,
            "stage_start_boundary": 18_000,
            "stage_target_boundary": 20_000,
            "parent_checkpoint_sha256": "c" * 64,
            "parent_gate_sha256": "d" * 64,
            "resume_source_checkpoint_path": "/pinned/model_17999.pt",
            "resume_source_checkpoint_sha256": "c" * 64,
            "resume_source_checkpoint_iteration": 17_999,
        },
    }
    torch.save(
        {
            "iter": 19_999,
            "infos": {
                "env_state": {"common_step_counter": 20_000 * 24},
                "microban_teleop_training_contract_version": (
                    MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION
                ),
                "previous_action_semantics": (
                    MICROBAN_TELEOP_PREVIOUS_ACTION_SEMANTICS
                ),
                "microban_teleop_actor_initialization": (
                    MICROBAN_TELEOP_ACTOR_INITIALIZATION
                ),
                "microban_teleop_recipe_revision": MICROBAN_TELEOP_RECIPE_REVISION,
                MICROBAN_TELEOP_TRAINING_PROVENANCE_KEY: manifest,
                MICROBAN_TELEOP_TRAINING_PROVENANCE_SHA256_KEY: (
                    canonical_json_sha256(manifest)
                ),
                "safe_velocity_actor_bootstrap": bootstrap_info,
            },
        },
        path,
    )


def _write_final_acceptance_receipt(
    root: Path,
    checkpoint: Path,
    *,
    training_provenance_sha256: str,
) -> tuple[Path, list[Path]]:
    checkpoint = checkpoint.resolve()
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    scenario_reports = [
        {"name": scenario.name} for scenario in teleop_evaluator.default_scenarios()
    ]
    report_paths: list[Path] = []
    moving_paths: list[Path] = []
    for moving_hmd, destination in ((False, report_paths), (True, moving_paths)):
        for seed in (42, 43, 44):
            report = {
                "schema_version": 8,
                "seed": seed,
                "checkpoint": str(checkpoint),
                "checkpoint_iteration": 19_999,
                "checkpoint_sha256": checkpoint_sha256,
                "evaluator_revision": teleop_evaluator.TELEOP_EVALUATOR_REVISION,
                "acceptance_revision": teleop_evaluator.TELEOP_ACCEPTANCE_REVISION,
                "steps_per_scenario": 1000,
                "settle_steps": 50,
                "status": "diagnostic" if moving_hmd else "pass",
                "training_contract": {
                    "version": MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION,
                    "training_provenance_sha256": training_provenance_sha256,
                    "canonical_training_stage": True,
                    "deployment_compatible": True,
                },
                "summary": {
                    "hard_safety_checks_passed": True,
                    "acceptance_checks_passed": True,
                    "canonical_coverage": not moving_hmd,
                },
                "scenarios": scenario_reports,
                "nominal_environment": {"hmd_neck_motion": moving_hmd},
                "hmd_neck_motion": {
                    "enabled": moving_hmd,
                    "params": ({"neutral_probability": 0.0} if moving_hmd else None),
                    "evidence": (
                        {
                            "passed": True,
                            "active_event_membership": {
                                "all_scenarios": True,
                                "inactive_scenarios": [],
                                "malformed_scenarios": [],
                            },
                            "minimum_required_target_peak_to_peak_rad": 0.10,
                            "minimum_required_actual_peak_to_peak_rad": 0.05,
                            "minimum_observed_target_peak_to_peak_rad_by_axis": {
                                "head": 0.11,
                                "neck_roll": 0.11,
                                "neck_pitch": 0.11,
                            },
                            "minimum_observed_actual_peak_to_peak_rad_by_axis": {
                                "head": 0.06,
                                "neck_roll": 0.06,
                                "neck_pitch": 0.06,
                            },
                        }
                        if moving_hmd
                        else None
                    ),
                },
            }
            suffix = "moving" if moving_hmd else "nominal"
            report_path = root / f"seed{seed}_{suffix}.json"
            report_path.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            destination.append(report_path.resolve())

    evaluator_source_sha256 = hashlib.sha256(
        Path(teleop_evaluator.__file__).read_bytes()
    ).hexdigest()
    receipt = {
        "schema_version": 3,
        "status": "pass",
        "training_contract_version": MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION,
        "recipe_revision": MICROBAN_TELEOP_RECIPE_REVISION,
        "training_provenance_sha256": training_provenance_sha256,
        "evaluator_revision": teleop_evaluator.TELEOP_EVALUATOR_REVISION,
        "acceptance_revision": teleop_evaluator.TELEOP_ACCEPTANCE_REVISION,
        "evaluator_source_sha256": evaluator_source_sha256,
        "run_name": checkpoint.parent.name,
        "completed_iterations": 20_000,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "evaluation_seeds": [42, 43, 44],
        "scenarios": "canonical",
        "reports": [str(path) for path in report_paths],
        "report_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in report_paths
        },
        "moving_hmd_reports": [str(path) for path in moving_paths],
        "moving_hmd_report_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in moving_paths
        },
    }
    receipt_path = root / "final_gate.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt_path, [*report_paths, *moving_paths]


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
        self.bootstrap_info = _write_accepted_safe_source(self.root)
        self.checkpoint = self.root / "model_42.pt"
        _write_generic_checkpoint(self.checkpoint, self.bootstrap_info)
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
        self.assertFalse(provenance.training.canonical_stage)
        self.assertEqual(provenance.training.mode, "generic")
        self.assertIsNone(provenance.acceptance)

    def test_generic_checkpoint_cannot_satisfy_canonical_deployment(self) -> None:
        with self.assertRaisesRegex(ValueError, "canonical training stage"):
            collect_teleop_export_provenance(
                self.checkpoint, require_canonical_stage=True
            )
        with self.assertRaisesRegex(ValueError, "canonical training stage"):
            collect_teleop_export_provenance(
                self.checkpoint, require_final_acceptance=True
            )

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
        _write_generic_checkpoint(first, self.bootstrap_info)
        _write_generic_checkpoint(second, self.bootstrap_info)
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
        self.assertEqual(metadata["canonical_training_stage"], "false")
        self.assertEqual(metadata["training_provenance_mode"], "generic")
        self.assertEqual(
            metadata["training_provenance_sha256"], provenance.training.sha256
        )
        self.assertEqual(
            metadata["training_source_tree_sha256"],
            provenance.training.source_tree_sha256,
        )
        self.assertEqual(metadata["training_stage_start_boundary"], "none")
        self.assertEqual(
            metadata["safe_velocity_source_checkpoint_sha256"],
            self.bootstrap_info["source_checkpoint_sha256"],
        )
        self.assertEqual(
            metadata["safe_velocity_acceptance_receipt_sha256"],
            self.bootstrap_info["source_acceptance_receipt_sha256"],
        )
        self.assertEqual(
            metadata["safe_velocity_bootstrap_mapping_version"],
            self.bootstrap_info["mapping_version"],
        )
        self.assertEqual(metadata["deployment_accepted"], "false")
        self.assertEqual(metadata["acceptance_receipt_sha256"], "none")

    def test_final_receipt_is_verified_and_bound_into_onnx(self) -> None:
        run = self.root / "canonical_run"
        checkpoint = run / "model_19999.pt"
        _write_final_canonical_checkpoint(checkpoint, self.bootstrap_info)
        unaccepted = collect_teleop_export_provenance(
            checkpoint, require_final_canonical_stage=True
        )
        receipt, _reports = _write_final_acceptance_receipt(
            self.root,
            checkpoint,
            training_provenance_sha256=unaccepted.training.sha256,
        )

        provenance = collect_teleop_export_provenance(
            checkpoint,
            acceptance_receipt=receipt,
            require_final_acceptance=True,
        )
        self.assertIsNotNone(provenance.acceptance)
        _write_zero_policy(self.temporary_onnx)
        publish_gated_teleop_onnx(
            self.temporary_onnx,
            self.output_onnx,
            pytorch_policy=_ZeroPolicy(),
            policy_metadata={},
            provenance=provenance,
        )
        metadata = {
            item.key: item.value for item in onnx.load(self.output_onnx).metadata_props
        }
        self.assertEqual(metadata["canonical_training_stage"], "true")
        self.assertEqual(
            metadata["training_provenance_mode"],
            MICROBAN_TELEOP_CANONICAL_STAGE_MODE,
        )
        self.assertEqual(metadata["training_stage_start_boundary"], "18000")
        self.assertEqual(metadata["training_stage_target_boundary"], "20000")
        self.assertEqual(
            metadata["training_resume_source_checkpoint_sha256"],
            "c" * 64,
        )
        self.assertEqual(
            metadata["training_resume_source_checkpoint_iteration"],
            "17999",
        )
        self.assertEqual(metadata["deployment_accepted"], "true")
        self.assertEqual(metadata["acceptance_receipt_schema_version"], "3")
        self.assertEqual(metadata["acceptance_status"], "pass")
        self.assertEqual(metadata["acceptance_boundary"], "20000")
        self.assertEqual(
            metadata["acceptance_checkpoint_sha256"], metadata["checkpoint_sha256"]
        )
        self.assertEqual(
            metadata["acceptance_training_provenance_sha256"],
            metadata["training_provenance_sha256"],
        )
        self.assertEqual(
            metadata["acceptance_recipe_revision"],
            metadata["training_recipe_revision"],
        )
        assert provenance.acceptance is not None
        self.assertEqual(
            metadata["acceptance_receipt_sha256"], provenance.acceptance.sha256
        )

    def test_final_receipt_fails_closed_on_hashed_report_mutation(self) -> None:
        run = self.root / "canonical_run"
        checkpoint = run / "model_19999.pt"
        _write_final_canonical_checkpoint(checkpoint, self.bootstrap_info)
        unaccepted = collect_teleop_export_provenance(checkpoint)
        receipt, reports = _write_final_acceptance_receipt(
            self.root,
            checkpoint,
            training_provenance_sha256=unaccepted.training.sha256,
        )
        reports[0].write_text("{}\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "report SHA-256 mismatch"):
            collect_teleop_export_provenance(
                checkpoint,
                acceptance_receipt=receipt,
                require_final_acceptance=True,
            )

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
