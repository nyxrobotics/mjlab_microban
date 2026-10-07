"""The release pipeline (src/mjlab_microban/pipeline) on the CPU, without training."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from mjlab_microban.pipeline import core, steps
from mjlab_microban.pipeline.core import PipelineError, State, hash_inputs, is_training_command, load_config

GATE = load_config(dry=False)["getup"]["gate"]


def fake_pipeline(state_dir: Path) -> steps.Pipeline:
    pipeline = steps.Pipeline.__new__(steps.Pipeline)
    pipeline.state = State(state_dir)
    pipeline.cfg = load_config(dry=False)
    pipeline.dry = False
    pipeline.sched = {"walk_max": 20000}
    return pipeline


class StateTest(unittest.TestCase):
    def test_done_with_same_inputs_is_skipped_and_changed_inputs_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pipeline = fake_pipeline(root / "state")
            output = root / "model_9.pt"
            output.write_bytes(b"walker")
            self.assertTrue(pipeline.begin("walk", "a" * 64))
            pipeline.finish("walk", {"checkpoint": str(output)}, [output])
            reloaded = fake_pipeline(root / "state")
            self.assertFalse(reloaded.begin("walk", "a" * 64))
            self.assertTrue(reloaded.begin("walk", "b" * 64))  # inputs changed
            self.assertEqual(reloaded.state.step("walk")["status"], "running")

    def test_a_changed_output_file_reruns_the_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pipeline = fake_pipeline(root / "state")
            output = root / "model_9.pt"
            output.write_bytes(b"walker")
            pipeline.begin("walk", "a" * 64)
            pipeline.finish("walk", {}, [output])
            output.write_bytes(b"other")
            self.assertTrue(fake_pipeline(root / "state").begin("walk", "a" * 64))

    def test_a_failure_stops_until_the_inputs_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline = fake_pipeline(Path(directory) / "state")
            pipeline.begin("getup", "a" * 64)
            pipeline.fail("getup", "push falls 0.20 > 0.1")
            again = fake_pipeline(Path(directory) / "state")
            with self.assertRaisesRegex(PipelineError, "push falls"):
                again.begin("getup", "a" * 64)
            self.assertTrue(again.begin("getup", "c" * 64))

    def test_inputs_cover_the_training_keys_not_the_checks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline = fake_pipeline(Path(directory) / "state")
            pipeline.cfg = {"seed": 42, "walk": {"task": "T", "envs": 4096, "save_interval": 500,
                                                 "check": {"seeds": "101"}}}
            before = pipeline.inputs("walk")
            pipeline.cfg["walk"]["check"] = {"seeds": "101,102", "line": 0.5}
            pipeline.cfg["walk"]["gate"] = {"x": 1}
            self.assertEqual(pipeline.inputs("walk"), before)
            pipeline.cfg["walk"]["envs"] = 2048
            self.assertNotEqual(pipeline.inputs("walk"), before)

    def test_running_steps_continue(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline = fake_pipeline(Path(directory) / "state")
            pipeline.begin("pico", "a" * 64)  # then the process died
            again = fake_pipeline(Path(directory) / "state")
            self.assertTrue(again.begin("pico", "a" * 64))
            self.assertEqual(again.state.step("pico")["inputs"], "a" * 64)

    def test_a_crashed_or_stalled_training_continues_and_a_refusal_fails(self) -> None:
        def run(state_dir: Path, cmd: list[str], kind: str) -> steps.Pipeline:
            pipeline = fake_pipeline(state_dir)
            pipeline.home = {"tag": "t", "joint_hash": "h"}
            pipeline.prefix = "home_h"
            pipeline.jobs = core.Jobs(pipeline.state, stall_minutes={"train": 20, "eval": 30},
                                      wait_for_gpu=False, external_min_envs=1024)
            pipeline.preflight = lambda: None
            for name in steps.STEPS:
                setattr(pipeline, f"step_{name}", lambda: None)

            def step_walk() -> None:
                if pipeline.begin("walk", "a" * 64):
                    pipeline.jobs.run("train_walk" if kind == "train" else "export_walk", cmd, kind, gpu=False)

            pipeline.step_walk = step_walk
            return pipeline

        def execute(pipeline: steps.Pipeline) -> None:
            try:
                pipeline.execute()
            finally:
                pipeline.state._lock_file.close()  # the process ended

        with tempfile.TemporaryDirectory() as directory:
            state_dir = Path(directory) / "state"
            crash = ["bash", "-c", "exit 1"]  # e.g. CUDA out of memory on the shared GPU
            with self.assertRaises(core.JobStopped) as raised:
                execute(run(state_dir, crash, "train"))
            self.assertEqual(raised.exception.code, core.EXIT_STALL)
            self.assertEqual(State(state_dir).step("walk")["status"], "running")
            with self.assertRaises(core.JobStopped):  # the rerun enters the step again
                execute(run(state_dir, crash, "train"))
            self.assertEqual(State(state_dir).step("walk")["status"], "running")

        with tempfile.TemporaryDirectory() as directory:
            state_dir = Path(directory) / "state"
            with self.assertRaises(PipelineError) as raised:  # an exporter refusing is a verdict
                execute(run(state_dir, ["bash", "-c", "exit 1"], "eval"))
            self.assertNotIsInstance(raised.exception, core.JobStopped)
            self.assertEqual(State(state_dir).step("walk")["status"], "failed")

        with tempfile.TemporaryDirectory() as directory:
            state_dir = Path(directory) / "state"
            pipeline = run(state_dir, ["bash", "-c", "sleep 600"], "train")
            pipeline.jobs.stall_s["train"] = 0.0
            with self.assertRaisesRegex(core.JobStopped, "STALL"):
                execute(pipeline)
            self.assertEqual(State(state_dir).step("walk")["status"], "running")

    def test_one_instance_per_state_dir(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = State(Path(directory))
            first.lock()
            with self.assertRaises(PipelineError) as raised:
                State(Path(directory)).lock()
            self.assertEqual(raised.exception.code, core.EXIT_BUSY)


class InputsTest(unittest.TestCase):
    def test_step_inputs_cover_only_their_training_files(self) -> None:
        walk = set(steps.STEP_INPUTS["walk"])
        self.assertIn("src/mjlab_microban/tasks/microban_velocity_*.py", walk)
        self.assertFalse(any("teleop" in pattern or "getup_env" in pattern for pattern in walk))
        self.assertFalse(any("velocity" in pattern for pattern in steps.STEP_INPUTS["getup"]))
        # PICO trains on the walking task's config (its base) and rewards.
        self.assertIn("src/mjlab_microban/tasks/microban_velocity_env_cfg.py", steps.STEP_INPUTS["pico"])

    def test_hash_changes_with_a_listed_file_only(self) -> None:
        with tempfile.TemporaryDirectory(dir=core.REPO / "logs" if (core.REPO / "logs").is_dir() else None) \
                as directory:
            root = Path(directory)
            listed = root / "listed.py"
            other = root / "other.py"
            listed.write_text("a = 1\n")
            other.write_text("b = 1\n")
            pattern = str(listed.relative_to(core.REPO)) if listed.is_relative_to(core.REPO) else None
            if pattern is None:
                self.skipTest("needs a directory inside the repository")
            before = hash_inputs([pattern], {"x": 1})
            other.write_text("b = 2\n")
            self.assertEqual(hash_inputs([pattern], {"x": 1}), before)
            listed.write_text("a = 2\n")
            self.assertNotEqual(hash_inputs([pattern], {"x": 1}), before)
            self.assertNotEqual(hash_inputs([pattern], {"x": 2}), hash_inputs([pattern], {"x": 1}))


class ConfigTest(unittest.TestCase):
    def test_dry_section_overrides_only_a_dry_run(self) -> None:
        real, dry = load_config(dry=False), load_config(dry=True)
        self.assertEqual(real["walk"]["envs"], 4096)
        self.assertEqual(dry["walk"]["envs"], 64)
        self.assertEqual(dry["walk"]["probe_thresholds"], real["walk"]["probe_thresholds"])
        self.assertNotIn("dry", real)
        self.assertLess(dry["schedule_scale"], 1.0)

    def test_judgment_thresholds(self) -> None:
        self.assertEqual(GATE["max_standing_targets_beyond_1p57"], 0.05)
        self.assertEqual(GATE["min_fallen_standing_fraction"], 0.85)


class JudgmentTest(unittest.TestCase):
    def test_getup_gate(self) -> None:
        good = {"min_fallen_standing_fraction": 0.9, "push_fall_fraction": 0.05,
                "standing_joint_abs_vel_rad_s": 0.1, "standing_targets_beyond_1p57": 0.03,
                "posture_standing_fraction": 0.9, "final_tilt_deg": 4.0}
        self.assertEqual(steps.getup_gate_failures(good, GATE), [])
        for key, value in (("min_fallen_standing_fraction", 0.8), ("push_fall_fraction", 0.2),
                           ("standing_joint_abs_vel_rad_s", 0.4), ("standing_targets_beyond_1p57", 0.06),
                           ("posture_standing_fraction", 0.7)):
            with self.subTest(key=key):
                self.assertEqual(len(steps.getup_gate_failures({**good, key: value}, GATE)), 1)

    def test_walker_probe_verdict(self) -> None:
        walk = load_config(dry=False)["walk"]
        thresholds = walk["probe_thresholds"]
        receipt = {
            "summary": {"completed_scenario_count": 9, "fall_scenario_count": 0, "nonfinite_scenario_count": 0,
                        "directionally_correct_scenario_count": 8, "raw_action_recurrence_all_steps": True,
                        "maximum_actual_soft_limit_violation_rad": 0.01},
            "results": [{"scenario": name, "directional_response": {"signed_response": value + 0.01}}
                        for name, value in thresholds.items()] + [{"scenario": "neutral"}],
        }
        verdict = steps.probe_verdict(receipt, walk)
        self.assertTrue(verdict["ok"])
        self.assertAlmostEqual(verdict["worst_margin"], 0.01)
        receipt["results"][0]["directional_response"]["signed_response"] = 0.0
        self.assertFalse(steps.probe_verdict(receipt, walk)["ok"])

    def test_getup_summary(self) -> None:
        stand = {"fallen_standing_fraction": 0.9, "standing_joint_abs_vel_rad_s": 0.1,
                 "standing_targets_beyond_1p57": 0.02, "final_tilt_deg": 3.0}
        summary = steps.getup_summary({
            "stand_s11": stand, "stand_s5": {**stand, "fallen_standing_fraction": 0.88},
            "push_s11": {**stand, "push_fell_within_3s": 3, "push_standing_before": 60},
            "posture_s11": {"standing_fraction": 0.95},
        })
        self.assertEqual(summary["min_fallen_standing_fraction"], 0.88)
        self.assertAlmostEqual(summary["push_fall_fraction"], 0.05)


class GpuTest(unittest.TestCase):
    def test_what_counts_as_another_training(self) -> None:
        self.assertTrue(is_training_command("/x/.venv/bin/python /x/.venv/bin/train Mjlab-Velocity-Microban", 1024))
        self.assertTrue(is_training_command("python -m x --env.scene.num-envs 4096", 1024))
        self.assertFalse(is_training_command("python -m x --env.scene.num-envs 64", 1024))
        self.assertFalse(is_training_command("python -m mjlab_microban.pipeline.getup_eval stand", 1024))

    def test_a_dry_run_does_not_wait(self) -> None:
        jobs = core.Jobs(SimpleNamespace(log=print), stall_minutes={"train": 1, "eval": 1}, wait_for_gpu=False,
                         external_min_envs=1024)
        with mock.patch.object(core.Jobs, "external_training", side_effect=AssertionError("queried")):
            jobs.wait_for_free_gpu("x")


class LatestCheckpointTest(unittest.TestCase):
    def test_newest_over_resumed_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for run, models in (("2026-10-07_01-00-00_label", (0, 500, 1000)),
                                ("2026-10-07_02-00-00_label", (1001, 1500)),
                                ("2026-10-07_03-00-00_other", (9000,))):
                (root / "exp" / run).mkdir(parents=True)
                for index in models:
                    (root / "exp" / run / f"model_{index}.pt").write_bytes(b"x")
            with mock.patch.object(core, "LOG_ROOT", root):
                self.assertEqual(core.latest_checkpoint("exp", "label").name, "model_1500.pt")
                self.assertEqual(core.find_checkpoint("exp", "label", 500).parent.name,
                                 "2026-10-07_01-00-00_label")
                self.assertIsNone(core.find_checkpoint("exp", "label", 9000))


class StatusTest(unittest.TestCase):
    def test_status_lists_every_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = State(Path(directory))
            state.data["steps"]["walk"] = {"status": "done"}
            state.save()
            with mock.patch("builtins.print") as printed:
                self.assertEqual(steps.print_status(Path(directory)), 0)
            text = "\n".join(str(call.args[0]) for call in printed.call_args_list)
            for name in steps.STEPS:
                self.assertIn(name, text)
            self.assertIn(json.dumps(None), text)


if __name__ == "__main__":
    unittest.main()
