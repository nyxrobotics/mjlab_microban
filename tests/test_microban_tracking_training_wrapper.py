# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Contract tests for the fail-closed walk004 training wrapper."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/train_microban_tracking.sh"


class TrackingTrainingWrapperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.script = SCRIPT.read_text(encoding="utf-8")

    def test_shell_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(SCRIPT)], cwd=ROOT, check=True)

    def test_help_is_side_effect_free(self) -> None:
        result = subprocess.run(
            ["bash", str(SCRIPT), "--help"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("Exact total PPO updates", result.stdout)
        self.assertIn("evaluate RUN_NAME", result.stdout)

    def test_unknown_override_is_rejected(self) -> None:
        result = subprocess.run(
            [
                "bash",
                str(SCRIPT),
                "start",
                "--agent.algorithm.learning-rate",
                "0.01",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Unsupported training option", result.stderr)

    def test_contract_pins_all_initial_stage_axes(self) -> None:
        required = (
            "readonly NUM_ENVS=2048",
            "readonly CLIP_STEPS=267",
            "readonly CLIP_DURATION_S=5.34",
            "readonly LEARNING_RATE=3e-5",
            "readonly LEARNING_EPOCHS=3",
            "--env.is-finite-horizon True",
            "--env.observations.actor.enable-corruption False",
            "--env.commands.motion.sampling-mode start",
            "--env.commands.motion.joint-position-range '(0.0,0.0)'",
            "--env.events.encoder-bias.params.bias-range",
            "--env.events.foot-friction.params.ranges",
            "--agent.actor.obs-normalization False",
            '--agent.algorithm.learning-rate "${LEARNING_RATE}"',
        )
        for item in required:
            with self.subTest(item=item):
                self.assertIn(item, self.script)

    def test_resume_uses_persisted_steps_not_suffix_plus_one(self) -> None:
        self.assertIn('env_state.get("common_step_counter")', self.script)
        self.assertIn("common_steps // rollout_steps", self.script)
        self.assertNotIn(
            "completed_iterations=$((checkpoint_iteration + 1))", self.script
        )
        self.assertIn("target_iterations - completed_iterations", self.script)

    def test_motion_and_resume_inputs_are_pinned(self) -> None:
        self.assertIn(
            "100656a04438e5b7c09d80e63f79f3e758650a20d87326f3e3970616e6fb69e2",
            self.script,
        )
        self.assertIn("validate_resume_contract", self.script)
        self.assertIn('"^${source_run}$"', self.script)
        self.assertIn('"^model_${checkpoint_iteration}[.]pt$"', self.script)
        self.assertIn("Arbitrary passthrough is disabled", self.script)

    def test_command_and_source_provenance_are_recorded(self) -> None:
        for field in (
            '"command": command',
            '"command_shell": shlex.join(command)',
            '"git_commit"',
            '"git_diff_sha256"',
            '"source_checkpoint"',
            '"sha256": source_checkpoint_sha256',
        ):
            with self.subTest(field=field):
                self.assertIn(field, self.script)


if __name__ == "__main__":
    unittest.main()
