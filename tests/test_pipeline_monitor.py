"""The pipeline's training monitor and the walking check rules (CPU, recorded logs)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from mjlab_microban.pipeline import core, monitor as m
from mjlab_microban.pipeline.core import PipelineError, State, load_config
from mjlab_microban.pipeline.walk_probe import DIAGONAL, SINGLE, evaluate

CFG = load_config(dry=False)


class FakeProcess:
    def __init__(self, rc=None):
        self.rc = rc
        self.pid = 12345

    def poll(self):
        return self.rc

    def wait(self, timeout=None):
        return self.rc


def make_monitor(root: Path, *, checks, min_final, verdicts=None, **kwargs) -> m.Monitor:
    state = State(root / "state")
    jobs = SimpleNamespace(children={}, kill=lambda proc: None)
    verdicts = verdicts or {}

    def start_check(checkpoint: Path) -> m.Check:
        update = int(checkpoint.stem.split("_")[1])
        return m.Check(update=-1, process=FakeProcess(rc=0), log=root / "check.log",
                       verdict=lambda _log: {"passed": verdicts.get(update, False)})

    return m.Monitor(state=state, jobs=jobs, record=state.step("walk"), experiment="exp", label="run",
                     expected_stages=kwargs.pop("stages", {}), stage_tolerance=1, check_updates=checks,
                     min_final_update=min_final, start_check=start_check, **kwargs)


def save(root: Path, *updates: int) -> None:
    run = root / "logs" / "exp" / "2026-10-07_01-00-00_run"
    run.mkdir(parents=True, exist_ok=True)
    for update in updates:
        (run / f"model_{update}.pt").write_bytes(b"x")


class MonitorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)
        self.patch = mock.patch.object(core, "LOG_ROOT", self.root / "logs")
        self.patch.start()

    def tearDown(self) -> None:
        self.patch.stop()
        self.dir.cleanup()

    def run_checks(self, monitor: m.Monitor) -> None:
        for _ in range(10):
            monitor.collect()
            monitor.start_due()

    def test_two_consecutive_passes_end_the_run(self) -> None:
        save(self.root, 1000, 2000)
        monitor = make_monitor(self.root, checks=[1000, 2000, 3000, 4000], min_final=0,
                               verdicts={2000: True, 3000: True, 4000: True})
        monitor.start_due()  # the oldest due checkpoint first; every one is checked
        self.run_checks(monitor)
        self.assertEqual(set(monitor.record["checks"]), {"1000", "2000"})
        self.assertFalse(monitor.record["checks"]["1000"]["passed"])
        save(self.root, 3000)
        with self.assertRaises(m.EarlyStop) as stop:
            self.run_checks(monitor)
        self.assertEqual(stop.exception.checkpoint.name, "model_3000.pt")
        self.assertTrue(monitor.record["adopted"].endswith("model_3000.pt"))

    def test_finish_checks_every_checkpoint_left(self) -> None:
        save(self.root, 1000, 2000, 3000, 3999)
        monitor = make_monitor(self.root, checks=[1000, 2000, 3000, 3999], min_final=0,
                               verdicts={2000: True, 3999: True})
        monitor.finish()  # the training ended before any check ran (or it was not watched)
        self.assertEqual(set(monitor.record["checks"]), {"1000", "2000", "3000", "3999"})
        self.assertNotIn("adopted", monitor.record)

    def test_a_failed_check_restarts_the_pair(self) -> None:
        monitor = make_monitor(self.root, checks=[1000, 2000, 3000, 4000], min_final=0,
                               verdicts={1000: True, 2000: False, 3000: True})
        for update in (1000, 2000, 3000):
            save(self.root, update)
            self.run_checks(monitor)
        self.assertNotIn("adopted", monitor.record)

    def test_no_early_end_before_the_minimum(self) -> None:
        save(self.root, 11000)
        monitor = make_monitor(self.root, checks=[11000, 12000, 13000], min_final=13249,
                               verdicts={11000: True, 12000: True, 13000: True})
        self.run_checks(monitor)
        save(self.root, 12000)
        self.run_checks(monitor)  # 11000 + 12000 pass, but 12000 < 13249
        save(self.root, 13000)
        self.run_checks(monitor)
        self.assertNotIn("adopted", monitor.record)
        monitor.record["checks"]["13000"] = {"passed": True}
        save(self.root, 14000)
        monitor.check_updates.append(14000)
        monitor.start_check = lambda ckpt: m.Check(-1, FakeProcess(0), self.root / "c.log", lambda _l: {"passed": True})
        with self.assertRaises(m.EarlyStop) as stop:
            self.run_checks(monitor)
        self.assertEqual(stop.exception.checkpoint.name, "model_14000.pt")

    def test_dry_monitor_never_stops(self) -> None:
        save(self.root, 1, 2)
        monitor = make_monitor(self.root, checks=[1, 2], min_final=0, verdicts={1: True, 2: True},
                               early_stop=False)
        self.run_checks(monitor)
        self.assertEqual(set(monitor.record["checks"]), {"1", "2"})
        self.assertNotIn("adopted", monitor.record)

    def test_stage_lines_are_checked_against_the_table(self) -> None:
        log = self.root / "train.log"
        log.write_text("\x1b[1m Learning iteration 2999/20000 \x1b[0m\n"
                       "Curriculum stage 1 penalize stepping + increase velocity at step 72024 (update 3001)\n")
        monitor = make_monitor(self.root, checks=[], min_final=0,
                               stages={"penalize stepping + increase velocity": 3000})
        monitor.log_path = log
        monitor.read_log()
        self.assertEqual(monitor.record["stages_seen"], {"penalize stepping + increase velocity": 3001})
        late = make_monitor(self.root / "late", checks=[], min_final=0,
                            stages={"tighten foot tracking": 6000})
        log.write_text("Curriculum stage 4 tighten foot tracking at step 150000 (update 6250)\n")
        late.log_path = log
        with self.assertRaisesRegex(PipelineError, "the schedule did not run as written"):
            late.read_log()
        resumed = make_monitor(self.root / "resumed", checks=[], min_final=0,
                               stages={"tighten foot tracking": 6000}, resumed_from=6250)
        resumed.log_path = log
        resumed.read_log()  # re-applied where the continued run started

    def test_every_due_stage_must_be_seen_and_named_in_the_table(self) -> None:
        stages = {"start": 0, "refine": 4000, "effort_push": 10000}
        log = self.root / "train.log"
        log.write_text("Curriculum stage 1 start at step 0 (update 0)\n"
                       "Curriculum stage 2 refine at step 96000 (update 4000)\n")
        monitor = make_monitor(self.root, checks=[], min_final=0, stages=stages)
        monitor.log_path = log
        monitor.read_log()
        monitor.require_stages(10000)  # effort_push may still come within the tolerance
        with self.assertRaisesRegex(PipelineError, r"without applying the curriculum stages \['effort_push'\]"):
            monitor.require_stages(10001)
        renamed = make_monitor(self.root / "renamed", checks=[], min_final=0, stages=stages)
        log.write_text("Curriculum stage 3 effort and push at step 240000 (update 10000)\n")
        renamed.log_path = log
        with self.assertRaisesRegex(PipelineError, "not in the schedule table"):
            renamed.read_log()

    def test_the_pipeline_reads_the_task_tables(self) -> None:
        from mjlab_microban.pipeline.steps import stage_tables
        from mjlab_microban.schedules import GETUP_SCHEDULE, PICO_SCHEDULE, WALK_WIDEN_UPDATE

        tables = stage_tables()
        self.assertEqual(tables["walk"], {"penalize stepping + increase velocity": WALK_WIDEN_UPDATE})
        self.assertEqual(sorted(tables["pico"].values()), sorted(PICO_SCHEDULE.values()))
        self.assertEqual(tables["getup"]["refine exploration (std, Adam, learning rate, entropy)"],
                         GETUP_SCHEDULE["refine"])
        self.assertEqual(set(tables["getup"]), {"start without IMU latency", *GETUP_SCHEDULE,
                                                "refine exploration (std, Adam, learning rate, entropy)"})

    def test_walking_ends_only_after_both_checks_saw_its_stage(self) -> None:
        from mjlab_microban.schedules import WALK_MIN_FINAL_UPDATES, WALK_WIDEN_UPDATE

        # The earlier of the two passing checks (N - 1000) trained 1000
        # updates with the widened commands and the no-stepping penalty.
        self.assertGreaterEqual(WALK_MIN_FINAL_UPDATES - 1000, WALK_WIDEN_UPDATE + 1000)
        save(self.root, *range(1000, WALK_MIN_FINAL_UPDATES + 1, 1000))
        monitor = make_monitor(self.root, checks=list(range(1000, 20000, 1000)), min_final=WALK_MIN_FINAL_UPDATES)
        monitor.record["checks"] = {str(u): {"passed": True} for u in range(1000, WALK_MIN_FINAL_UPDATES, 1000)}
        monitor.decide_early_stop()  # every earlier pair passes, but ends before the stage was trained
        self.assertNotIn("adopted", monitor.record)
        monitor.record["checks"][str(WALK_MIN_FINAL_UPDATES)] = {"passed": True}
        with self.assertRaises(m.EarlyStop) as stop:
            monitor.decide_early_stop()
        self.assertEqual(stop.exception.checkpoint.name, f"model_{WALK_MIN_FINAL_UPDATES}.pt")

    def test_no_early_end_before_the_minimum(self) -> None:
        save(self.root, 11000)
        monitor = make_monitor(self.root, checks=[11000, 12000, 13000], min_final=13249,
                               verdicts={11000: True, 12000: True, 13000: True})
        self.run_checks(monitor)
        save(self.root, 12000)
        self.run_checks(monitor)  # 11000 + 12000 pass, but 12000 < 13249
        save(self.root, 13000)
        self.run_checks(monitor)
        self.assertNotIn("adopted", monitor.record)
        monitor.record["checks"]["13000"] = {"passed": True}
        save(self.root, 14000)
        monitor.check_updates.append(14000)
        monitor.start_check = lambda ckpt: m.Check(-1, FakeProcess(0), self.root / "c.log", lambda _l: {"passed": True})
        with self.assertRaises(m.EarlyStop) as stop:
            self.run_checks(monitor)
        self.assertEqual(stop.exception.checkpoint.name, "model_14000.pt")

    def test_dry_monitor_never_stops(self) -> None:
        save(self.root, 1, 2)
        monitor = make_monitor(self.root, checks=[1, 2], min_final=0, verdicts={1: True, 2: True},
                               early_stop=False)
        self.run_checks(monitor)
        self.assertEqual(set(monitor.record["checks"]), {"1", "2"})
        self.assertNotIn("adopted", monitor.record)

    def test_stage_lines_are_checked_against_the_table(self) -> None:
        log = self.root / "train.log"
        log.write_text("\x1b[1m Learning iteration 2999/20000 \x1b[0m\n"
                       "Curriculum stage 1 penalize stepping + increase velocity at step 72024 (update 3001)\n")
        monitor = make_monitor(self.root, checks=[], min_final=0,
                               stages={"penalize stepping + increase velocity": 3000})
        monitor.log_path = log
        monitor.read_log()
        self.assertEqual(monitor.record["stages_seen"], {"penalize stepping + increase velocity": 3001})
        late = make_monitor(self.root / "late", checks=[], min_final=0,
                            stages={"tighten foot tracking": 6000})
        log.write_text("Curriculum stage 4 tighten foot tracking at step 150000 (update 6250)\n")
        late.log_path = log
        with self.assertRaisesRegex(PipelineError, "the schedule did not run as written"):
            late.read_log()
        resumed = make_monitor(self.root / "resumed", checks=[], min_final=0,
                               stages={"tighten foot tracking": 6000}, resumed_from=6250)
        resumed.log_path = log
        resumed.read_log()  # re-applied where the continued run started

    def test_every_due_stage_must_be_seen_and_named_in_the_table(self) -> None:
        stages = {"start": 0, "refine": 4000, "effort_push": 10000}
        log = self.root / "train.log"
        log.write_text("Curriculum stage 1 start at step 0 (update 0)\n"
                       "Curriculum stage 2 refine at step 96000 (update 4000)\n")
        monitor = make_monitor(self.root, checks=[], min_final=0, stages=stages)
        monitor.log_path = log
        monitor.read_log()
        monitor.require_stages(10000)  # effort_push may still come within the tolerance
        with self.assertRaisesRegex(PipelineError, r"without applying the curriculum stages \['effort_push'\]"):
            monitor.require_stages(10001)
        renamed = make_monitor(self.root / "renamed", checks=[], min_final=0, stages=stages)
        log.write_text("Curriculum stage 3 effort and push at step 240000 (update 10000)\n")
        renamed.log_path = log
        with self.assertRaisesRegex(PipelineError, "not in the schedule table"):
            renamed.read_log()

    def test_the_pipeline_reads_the_task_tables(self) -> None:
        from mjlab_microban.pipeline.steps import stage_tables
        from mjlab_microban.schedules import GETUP_SCHEDULE, PICO_SCHEDULE, WALK_WIDEN_UPDATE

        tables = stage_tables()
        self.assertEqual(tables["walk"], {"penalize stepping + increase velocity": WALK_WIDEN_UPDATE})
        self.assertEqual(sorted(tables["pico"].values()), sorted(PICO_SCHEDULE.values()))
        self.assertEqual(tables["getup"]["refine exploration (std, Adam, learning rate, entropy)"],
                         GETUP_SCHEDULE["refine"])
        self.assertEqual(set(tables["getup"]), {"start without IMU latency", *GETUP_SCHEDULE,
                                                "refine exploration (std, Adam, learning rate, entropy)"})

    def test_walking_ends_only_after_both_checks_saw_its_stage(self) -> None:
        from mjlab_microban.schedules import WALK_MIN_FINAL_UPDATES, WALK_WIDEN_UPDATE

        # The earlier of the two passing checks (N - 1000) trained 1000
        # updates with the widened commands and the no-stepping penalty.
        self.assertGreaterEqual(WALK_MIN_FINAL_UPDATES - 1000, WALK_WIDEN_UPDATE + 1000)
        all_pass = {u: True for u in range(1000, 7000, 1000)}
        monitor = make_monitor(self.root, checks=sorted(all_pass), min_final=WALK_MIN_FINAL_UPDATES,
                               verdicts=all_pass)
        monitor.record["checks"] = {str(u): {"passed": True} for u in range(1000, WALK_MIN_FINAL_UPDATES, 1000)}
        monitor.decide_early_stop()  # 1000+2000 ... 3000+4000 pass but end too early
        self.assertNotIn("adopted", monitor.record)


class WalkRulesTest(unittest.TestCase):
    def rows(self, *, scale=(0.5, 0.5, 0.5), falls=0, single=0.1, still=(0.0, 0.0, 0.0), touchdowns=0.0):
        rows = []
        for push in ("none", "p30_15"):
            for name, twist in DIAGONAL.items():
                for rep in range(3):
                    motion = [twist[i] * scale[i] for i in range(3)]
                    rows.append({"push": push, "cmd": name, "twist": twist, "fell": rep < falls and push == "none",
                                 "mean": motion})
        for name, twist in SINGLE.items():
            for _ in range(3):
                rows.append({"push": "none", "cmd": name, "twist": twist, "fell": False,
                             "touchdowns_per_s": touchdowns if name == "S" else 8.0,
                             "mean": list(still) if name == "S" else [
                                 (single * 3 if i == 2 else single) * (1 if v > 0 else -1 if v < 0 else 0)
                                 for i, v in enumerate(twist)]})
        return rows

    def test_rules(self) -> None:
        rules = CFG["walk"]["check"]
        good = evaluate(self.rows(), rules)
        self.assertTrue(good["passed"], good)
        self.assertEqual(set(good["checks"]), {"W4", "W5", "W6"})
        self.assertLess(good["angle_deg"][0], 1.0)
        # The diagonals' ratio is recorded, not judged (no W1-W3).
        self.assertTrue(evaluate(self.rows(scale=(0.5, 0.0, 0.5)), rules)["passed"])
        fallen = evaluate(self.rows(falls=2), rules)
        self.assertFalse(fallen["checks"]["W4"])
        # W5: every single-axis command at its fixed minimum (0.2 forward
        # 0.08, backward 0.04, lateral 0.02, yaw 0.2; yaw rows get 3x).
        self.assertTrue(evaluate(self.rows(single=0.08), rules)["checks"]["W5"])
        self.assertFalse(evaluate(self.rows(single=0.079), rules)["checks"]["W5"])
        self.assertFalse(evaluate(self.rows(single=-0.1), rules)["checks"]["W5"])
        # W5: standing drifts at most 0.05, 0.05 m/s and 0.2 rad/s.
        self.assertTrue(evaluate(self.rows(still=(0.049, -0.049, 0.19)), rules)["checks"]["W5"])
        self.assertFalse(evaluate(self.rows(still=(0.06, 0.0, 0.0)), rules)["checks"]["W5"])
        self.assertFalse(evaluate(self.rows(still=(0.0, 0.0, 0.25)), rules)["checks"]["W5"])
        # W6: standing with the feet still; stepping in place (8 per second) fails.
        self.assertEqual(rules["still_touchdowns_per_s"], 0.5)
        self.assertTrue(evaluate(self.rows(touchdowns=0.5), rules)["checks"]["W6"])
        stepping = evaluate(self.rows(touchdowns=7.9), rules)
        self.assertFalse(stepping["checks"]["W6"] or stepping["passed"])
        self.assertAlmostEqual(stepping["still_touchdowns_per_s"], 7.9)

if __name__ == "__main__":
    unittest.main()
