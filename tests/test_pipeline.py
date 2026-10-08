"""The release pipeline (src/mjlab_microban/pipeline) on the CPU, without training."""

from __future__ import annotations

import json
import os
import subprocess
import sys
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

    def test_the_best_walking_check(self) -> None:
        def verdict(failed=(), margin=0.03):
            checks = {name: name not in failed for name in ("W4", "W5", "W6")}
            return {"passed": not failed, "probe": {"checks": checks},
                    "nine_by_300": {"ok": "9x300" not in failed, "worst_margin": margin}}

        checks = {"4000": verdict(), "5000": verdict(("W5", "9x300")), "6000": verdict(("W6",), margin=0.01),
                  "7000": verdict(("W6",), margin=0.02), "7999": verdict(("W4",), margin=0.015)}
        # 4000 is too early; 6000, 7000, 7999 fail one item; 7000 has the largest 9x300 margin.
        self.assertEqual(steps.best_walk_check(checks, 5000), 7000)
        self.assertEqual(steps.walk_check_failures(checks["7000"]), ["W6"])
        checks["7999"] = verdict(("W4",), margin=0.02)
        self.assertEqual(steps.best_walk_check(checks, 5000), 7999)  # a tie: the later one
        checks["6000"] = {"passed": False, "error": "RuntimeError: boom"}
        self.assertEqual(steps.walk_check_score(checks["6000"]), (0, float("-inf")))
        self.assertIsNone(steps.best_walk_check({"4000": verdict()}, 5000))

    def test_a_step_hashes_only_its_own_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline = fake_pipeline(Path(directory) / "state")
            pipeline.cfg = {"seed": 42, "walk": {"task": "W"}, "getup": {"task": "G"}}
            pipeline.sched = {"walk_max": 8000, "walk_min_final": 5000, "getup": {"effort": 10000},
                              "getup_total": 18000, "stages": {"walk": {"widen": 3000}, "getup": {"effort": 10000}}}
            walk, getup = pipeline.inputs("walk"), pipeline.inputs("getup")
            pipeline.sched["getup"] = {"effort": 10000, "push": 15000}
            pipeline.sched["stages"]["getup"] = {"effort": 10000, "push": 15000}
            pipeline.sched["walk_min_final"] = 4000  # an early-stop rule, not a training input
            self.assertEqual(pipeline.inputs("walk"), walk)
            self.assertNotEqual(pipeline.inputs("getup"), getup)
            pipeline.sched["walk_max"] = 9000
            self.assertNotEqual(pipeline.inputs("walk"), walk)
            self.assertNotIn("src/mjlab_microban/schedules.py", steps.STEP_INPUTS["walk"])

    def test_the_run_label_is_kept_by_the_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline = fake_pipeline(Path(directory) / "state")
            pipeline.prefix = "home_h"
            pipeline.begin("pico", "a" * 64)
            self.assertEqual(pipeline.state.step("pico")["label"], "home_h_pico_aaaaaaaa")
            pipeline.state.step("pico")["label"] = "home_h_pico_kept"
            pipeline.state.save()
            again = fake_pipeline(Path(directory) / "state")
            again.prefix = "home_h"
            again.begin("pico", "a" * 64)
            self.assertEqual(again.state.step("pico")["label"], "home_h_pico_kept")

    def test_a_failed_step_is_judged_again_only_when_its_judgment_changed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline = fake_pipeline(Path(directory) / "state")
            pipeline.judge_inputs = "j1"
            self.assertTrue(pipeline.begin("walk", "a" * 64))
            pipeline.state.step("walk").update(checks={"5000": {"passed": False}}, adopted="m.pt",
                                               adopted_by="best_checked")
            pipeline.fail("walk", "W1 fails")
            again = fake_pipeline(Path(directory) / "state")
            again.judge_inputs = "j1"  # the same test: a valid failing test stays failed
            with self.assertRaisesRegex(PipelineError, "Question the test first"):
                again.begin("walk", "a" * 64)
            again.judge_inputs = "j2"  # the test was fixed: re-judge, keep the training
            self.assertTrue(again.begin("walk", "a" * 64))
            record = again.state.step("walk")
            self.assertEqual((record["status"], record["inputs"], record["judge_inputs"]),
                             ("running", "a" * 64, "j2"))
            for key in ("checks", "adopted", "adopted_by", "error"):
                self.assertNotIn(key, record)
            # The trained run's entry gate stays.
            record["walker_probe"] = {"ok": True}
            again.fail("walk", "fails again")
            again.judge_inputs = "j3"
            again.begin("walk", "a" * 64)
            self.assertEqual(again.state.step("walk")["walker_probe"], {"ok": True})

    def test_a_training_started_by_hand_is_found_and_watched(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = State(Path(directory) / "state")
            log = state.dir / "logs" / "1007-000000_train_walk.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text("run lbl_x\n")
            proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3)", "train", "--agent.run-name",
                                     "lbl_x", "end"], start_new_session=True)
            try:
                self.assertEqual(core.running_training("lbl_x"), os.getpgid(proc.pid))
                self.assertIsNone(core.running_training("lbl"))
                jobs = core.Jobs(state, stall_minutes={"train": 20, "eval": 30}, wait_for_gpu=False,
                                 external_min_envs=1024, own_prefix="lbl")
                self.assertIn("--agent.run-name lbl_", jobs.own_marker)
                polls = []
                with mock.patch.object(core.time, "sleep", lambda _s: proc.wait()):
                    jobs.watch("train_walk", os.getpgid(proc.pid), log, poll=lambda: polls.append(1))
                self.assertTrue(polls)
                self.assertFalse(core.group_alive(proc.pid))
            finally:
                proc.kill()
                proc.wait()

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
        self.assertEqual(dry["walk"]["check"]["w4_falls"], real["walk"]["check"]["w4_falls"])
        self.assertNotIn("dry", real)
        self.assertLess(dry["schedule_scale"], 1.0)

    def test_judgment_thresholds(self) -> None:
        self.assertEqual(GATE["max_standing_targets_on_clip"], 0.05)
        self.assertEqual(GATE["min_fallen_standing_fraction"], 0.85)


class JudgmentTest(unittest.TestCase):
    def test_getup_gate(self) -> None:
        good = {"min_fallen_standing_fraction": 0.9, "push_fall_fraction": 0.05,
                "standing_joint_abs_vel_rad_s": 0.1, "standing_targets_on_clip": 0.03,
                "posture_standing_fraction": 0.9, "final_tilt_deg": 4.0}
        self.assertEqual(steps.getup_gate_failures(good, GATE), [])
        # Targets between 1.57 and the +-pi clip are allowed (torque authority).
        self.assertEqual(steps.getup_gate_failures({**good, "standing_targets_beyond_1p57": 0.5}, GATE), [])
        for key, value in (("min_fallen_standing_fraction", 0.8), ("push_fall_fraction", 0.2),
                           ("standing_joint_abs_vel_rad_s", 0.4), ("standing_targets_on_clip", 0.06),
                           ("posture_standing_fraction", 0.7)):
            with self.subTest(key=key):
                self.assertEqual(len(steps.getup_gate_failures({**good, key: value}, GATE)), 1)

    def test_walker_probe_verdict(self) -> None:
        # Each moving scenario passes at the fixed signed-response minimum
        # (0.1 / 0.2 forward 0.04 / 0.08, backward 0.02 / 0.04, lateral 0.02,
        # yaw 0.2); the joints may overshoot 0.25 rad.
        commands = {
            "forward_0p1": (0.1, 0.0, 0.0), "forward_0p2": (0.2, 0.0, 0.0),
            "backward_0p1": (-0.1, 0.0, 0.0), "backward_0p2": (-0.2, 0.0, 0.0),
            "lateral_left_0p1": (0.0, 0.1, 0.0), "lateral_right_0p1": (0.0, -0.1, 0.0),
            "yaw_left_0p5": (0.0, 0.0, 0.5), "yaw_right_0p5": (0.0, 0.0, -0.5),
        }
        axes = ("vx_m_s", "vy_m_s", "yaw_rad_s")

        def result(name, twist, measured):
            index = next((i for i, v in enumerate(twist) if v != 0.0), None)
            return {"scenario": name, "command": dict(zip(axes, twist, strict=True)),
                    "measured_velocity_body": {a: {"mean": m} for a, m in zip(axes, measured, strict=True)},
                    "directional_response": None if index is None else {
                        "signed_response": measured[index] * (1 if twist[index] > 0 else -1)}}

        def receipt(measured, overshoot=0.2):
            return {
                "summary": {"completed_scenario_count": 9, "fall_scenario_count": 0, "nonfinite_scenario_count": 0,
                            "directionally_correct_scenario_count": 8, "raw_action_recurrence_all_steps": True,
                            "maximum_actual_soft_limit_violation_rad": overshoot},
                "results": [result("neutral", (0.0, 0.0, 0.0), (0.03, 0.0, 0.0))]
                + [result(name, twist, measured(twist)) for name, twist in commands.items()],
            }

        # Half of every command: 0.01 above the backward / lateral minimums.
        verdict = steps.probe_verdict(receipt(lambda c: [0.5 * x for x in c]))
        self.assertTrue(verdict["ok"])
        self.assertAlmostEqual(verdict["worst_margin"], 0.01, places=6)
        # A tenth of every command is below them.
        self.assertFalse(steps.probe_verdict(receipt(lambda c: [0.1 * x for x in c]))["ok"])
        self.assertTrue(steps.probe_verdict(receipt(lambda c: [2.0 * x for x in c]))["ok"])
        # The first release walker's answer to 0.1 m/s forward (-0.0396 m/s).
        bad = receipt(lambda c: [0.5 * x for x in c])
        bad["results"][1]["measured_velocity_body"]["vx_m_s"]["mean"] = -0.0396
        verdict = steps.probe_verdict(bad)
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["below"], ["forward_0p1"])
        # Drift on the other axes does not count (the former tables).
        drift = receipt(lambda c: [0.5 * x for x in c])
        drift["results"][5]["measured_velocity_body"]["yaw_rad_s"]["mean"] = 1.2
        self.assertTrue(steps.probe_verdict(drift)["ok"])
        # The joint overshoot allowance is 0.25 rad.
        self.assertTrue(steps.probe_verdict(receipt(lambda c: [0.5 * x for x in c], 0.25))["ok"])
        self.assertFalse(steps.probe_verdict(receipt(lambda c: [0.5 * x for x in c], 0.2501))["ok"])

    def test_getup_summary(self) -> None:
        stand = {"fallen_standing_fraction": 0.9, "standing_joint_abs_vel_rad_s": 0.1,
                 "standing_targets_on_clip": 0.02, "standing_targets_beyond_1p57": 0.3,
                 "final_tilt_deg": 3.0}
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
