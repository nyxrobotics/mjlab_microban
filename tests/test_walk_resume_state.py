"""A resumed walking run continues its adaptive learning rate (CPU, fake algorithm)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from mjlab_microban.tasks import microban_velocity_runner as walk_runner
from mjlab_microban.tasks.microban_velocity_runner import (
    WALK_LEARNING_RATE_INFO_KEY,
    MicrobanVelocityOnPolicyRunner,
)

PARENT = MicrobanVelocityOnPolicyRunner.__mro__[1]


def fake_runner(learning_rate: float) -> MicrobanVelocityOnPolicyRunner:
    runner = object.__new__(MicrobanVelocityOnPolicyRunner)
    parameter = torch.nn.Parameter(torch.zeros(3))
    runner.alg = SimpleNamespace(
        learning_rate=learning_rate, optimizer=torch.optim.Adam([parameter], lr=learning_rate)
    )
    runner.current_learning_iteration = 0
    return runner


class WalkResumeStateTest(unittest.TestCase):
    def test_the_checkpoint_records_the_rate_and_a_resume_continues_it(self) -> None:
        saved = {}
        with mock.patch.object(PARENT, "save", lambda self, path, infos=None: saved.update(infos)):
            fake_runner(3.4e-5).save("model_1000.pt")
        self.assertEqual(saved[WALK_LEARNING_RATE_INFO_KEY], 3.4e-5)

        resumed = fake_runner(1.0e-3)
        with mock.patch.object(walk_runner, "require_current_home_walk_checkpoint"), mock.patch.object(
            PARENT, "load", lambda self, *args, **kwargs: saved
        ):
            resumed.load("model_1000.pt")
        self.assertEqual(resumed.alg.learning_rate, 3.4e-5)
        self.assertEqual(resumed.alg.optimizer.param_groups[0]["lr"], 3.4e-5)
        self.assertEqual(resumed.current_learning_iteration, 1)

    def test_a_weights_only_load_keeps_the_configured_rate(self) -> None:
        runner = fake_runner(1.0e-3)
        with mock.patch.object(walk_runner, "require_current_home_walk_checkpoint"), mock.patch.object(
            PARENT, "load", lambda self, *args, **kwargs: {WALK_LEARNING_RATE_INFO_KEY: 3.4e-5}
        ):
            runner.load("model_1000.pt", load_cfg={"actor": True, "iteration": False})
        self.assertEqual(runner.alg.learning_rate, 1.0e-3)


if __name__ == "__main__":
    unittest.main()
