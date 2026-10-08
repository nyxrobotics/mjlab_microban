# Copyright 2026 nyxrobotics

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for the Microban teleoperation deployment contract."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_TELEOP_OBSERVATION_SCHEMA,
    MICROBAN_TELEOP_OBSERVATION_WIDTH,
    validate_microban_teleop_observation_contract,
)


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


if __name__ == "__main__":
    unittest.main()
