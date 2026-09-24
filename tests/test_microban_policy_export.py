# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for the Microban teleoperation deployment contract."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import onnx
from onnx import TensorProto, helper

from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_TELEOP_OBSERVATION_SCHEMA,
    MICROBAN_TELEOP_OBSERVATION_WIDTH,
    validate_action_only_onnx,
    validate_microban_teleop_observation_contract,
)


def _write_policy(
    path: Path,
    *,
    input_shape: list[int | str | None] | None = None,
    output_shape: list[int | str | None] | None = None,
) -> None:
    """Write a minimal, checker-valid linear policy with requested I/O shapes."""

    input_shape = input_shape or [1, 83]
    output_shape = output_shape or [1, 18]
    obs = helper.make_tensor_value_info("obs", TensorProto.FLOAT, input_shape)
    actions = helper.make_tensor_value_info("actions", TensorProto.FLOAT, output_shape)
    weight = helper.make_tensor(
        "weight",
        TensorProto.FLOAT,
        [83, 18],
        [0.0] * (83 * 18),
    )
    graph = helper.make_graph(
        [helper.make_node("MatMul", ("obs", "weight"), ("actions",))],
        "microban_policy_contract_test",
        [obs],
        [actions],
        [weight],
    )
    model = helper.make_model(graph)
    onnx.save(model, path)


def _fake_env(
    *,
    names: tuple[str, ...] | None = None,
    widths: tuple[int, ...] | None = None,
) -> SimpleNamespace:
    expected_names = tuple(name for name, _ in MICROBAN_TELEOP_OBSERVATION_SCHEMA)
    expected_widths = tuple(width for _, width in MICROBAN_TELEOP_OBSERVATION_SCHEMA)
    resolved_widths = widths or expected_widths
    manager = SimpleNamespace(
        active_terms={"actor": list(names or expected_names)},
        group_obs_concatenate={"actor": True},
        group_obs_term_dim={"actor": [(width,) for width in resolved_widths]},
        group_obs_dim={"actor": (sum(resolved_widths),)},
    )
    return SimpleNamespace(observation_manager=manager)


class ObservationContractTest(unittest.TestCase):
    def test_exact_schema_is_83_values(self) -> None:
        self.assertEqual(MICROBAN_TELEOP_OBSERVATION_WIDTH, 83)
        validate_microban_teleop_observation_contract(_fake_env())

    def test_rejects_reordered_actor_terms(self) -> None:
        names = tuple(name for name, _ in MICROBAN_TELEOP_OBSERVATION_SCHEMA)
        with self.assertRaisesRegex(ValueError, "observation order"):
            validate_microban_teleop_observation_contract(
                _fake_env(names=(names[1], names[0], *names[2:]))
            )

    def test_rejects_term_width_drift(self) -> None:
        widths = tuple(width for _, width in MICROBAN_TELEOP_OBSERVATION_SCHEMA)
        with self.assertRaisesRegex(ValueError, "observation widths"):
            validate_microban_teleop_observation_contract(
                _fake_env(widths=(*widths[:-1], widths[-1] + 1))
            )


class OnnxContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary_directory.name) / "policy.onnx"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_accepts_exact_fixed_shapes(self) -> None:
        _write_policy(self.path)
        validate_action_only_onnx(self.path)

    def test_rejects_dynamic_batch(self) -> None:
        _write_policy(self.path, input_shape=["batch", 83])
        with self.assertRaisesRegex(ValueError, "input shape"):
            validate_action_only_onnx(self.path)

    def test_rejects_wrong_input_width(self) -> None:
        _write_policy(self.path, input_shape=[1, 82])
        with self.assertRaisesRegex(ValueError, "input shape"):
            validate_action_only_onnx(self.path)

    def test_rejects_wrong_output_width(self) -> None:
        _write_policy(self.path, output_shape=[1, 17])
        with self.assertRaisesRegex(ValueError, "output shape"):
            validate_action_only_onnx(self.path)


if __name__ == "__main__":
    unittest.main()
