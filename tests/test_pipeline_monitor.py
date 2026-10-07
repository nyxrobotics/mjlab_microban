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
        save(self.root, 1000, 2000, 3000)
        monitor = make_monitor(self.root, checks=[1000, 2000, 3000, 4000], min_final=0,
                               verdicts={2000: True, 3000: True, 4000: True})
        monitor.start_due()  # only the newest due checkpoint runs; older ones are skipped
        for update in ("1000", "2000"):
            self.assertEqual(monitor.record["checks"][update]["skipped"], "the previous check still ran")
        self.run_checks(monitor)  # 3000 passes, but 2000 was not checked: no pair yet
        self.assertTrue(monitor.record["checks"]["3000"]["passed"])
        save(self.root, 4000)
        with self.assertRaises(m.EarlyStop) as stop:
            self.run_checks(monitor)
        self.assertEqual(stop.exception.checkpoint.name, "model_4000.pt")
        self.assertTrue(monitor.record["adopted"].endswith("model_4000.pt"))

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

    def test_metrics_are_read_per_update(self) -> None:
        log = self.root / "train.log"
        log.write_text("Learning iteration 499/16500\n    Mean episode length: 88.50\n"
                       "Episode_Reward/standing_bonus: 2.9000\nLearning iteration 500/16500\n"
                       "Mean episode length: 120.00\n")
        monitor = make_monitor(self.root, checks=[], min_final=0)
        monitor.log_path = log
        monitor.read_log()
        self.assertEqual(monitor._metrics["Mean episode length"], [(499, 88.5), (500, 120.0)])
        self.assertEqual(monitor._metrics["Episode_Reward/standing_bonus"], [(499, 2.9)])


class RuleTest(unittest.TestCase):
    def test_walk_episode_length_floors(self) -> None:
        rules = m.walk_abort_rules(CFG["walk"]["check"])
        self.assertIsNone(rules(400, {"Mean episode length": [(400, 50.0)]}))
        self.assertIn("< 100", rules(500, {"Mean episode length": [(500, 90.0)]}))
        self.assertIsNone(rules(2000, {"Mean episode length": [(500, 150.0), (2000, 350.0)]}))
        self.assertIn("< 300", rules(2100, {"Mean episode length": [(500, 150.0), (2000, 250.0)]}))

    def test_walk_hopeless_at_12000(self) -> None:
        check = m.walk_check_abort(CFG["walk"]["check"])
        probe = {"checks": {"W1": False, "W4": True}, "angle_deg": [50.0, 50.0], "speed": [0.1, 0.1]}
        self.assertIsNone(check(12000, {"probe": probe}))
        self.assertIsNone(check(11000, {"probe": {**probe, "checks": {"W4": False}}}))
        self.assertIn("W4", check(12000, {"probe": {**probe, "checks": {"W4": False}}}))

    def test_getup_standing_bonus(self) -> None:
        c = CFG["getup"]["check"]
        rules = m.getup_abort_rules(c, [2500, 4000, 10000])
        name = "Episode_Reward/standing_bonus"
        self.assertIsNone(rules(2499, {name: [(2499, 4.4)]}))
        self.assertIn("< 3.0", rules(2499, {name: [(2499, 2.0)]}))
        self.assertIn("< 3.5", rules(4500, {name: [(2499, 4.4), (4500, 3.0)]}))
        dip = [(u, 2.5) for u in range(4000, 4400)]  # inside the 500-update grace after refine
        self.assertIsNone(rules(4400, {name: [(2499, 4.4), *dip]}))
        long_dip = [(u, 2.5) for u in range(5000, 5200)]
        self.assertIn("for 200 updates", rules(5200, {name: [(2499, 4.4), (4500, 4.4), *long_dip]}))


class WalkRulesTest(unittest.TestCase):
    def rows(self, *, scale=(0.5, 0.5, 0.5), falls=0, single=0.1, still=(0.0, 0.0, 0.0)):
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
                             "mean": list(still) if name == "S" else [
                                 (single * 3 if i == 2 else single) * (1 if v > 0 else -1 if v < 0 else 0)
                                 for i, v in enumerate(twist)]})
        return rows

    def test_rules(self) -> None:
        rules = CFG["walk"]["check"]
        good = evaluate(self.rows(), rules)
        self.assertTrue(good["passed"], good)
        self.assertLess(good["angle_deg"][0], 1.0)
        self.assertAlmostEqual(good["worst_value"][0], 0.75, places=6)
        # Off the ray as the reward judges it: 40 % of the commanded lateral
        # part (about 23 deg off) still beats standing still; none of it
        # (40-45 deg off) is worse than standing.
        off_ray = evaluate(self.rows(scale=(0.5, 0.2, 0.5)), rules)
        self.assertGreater(off_ray["angle_deg"][0], 20.0)
        self.assertTrue(off_ray["checks"]["W1"] and off_ray["checks"]["W2"], off_ray)
        given_up = evaluate(self.rows(scale=(0.5, 0.0, 0.5)), rules)
        self.assertFalse(given_up["checks"]["W1"] or given_up["checks"]["W2"])
        # Against the command on the diagonals: worse than standing.
        backward = evaluate(self.rows(scale=(-0.5, -0.5, -0.5)), rules)
        self.assertFalse(backward["checks"]["W1"] or backward["checks"]["W2"])
        fallen = evaluate(self.rows(falls=2), rules)
        self.assertFalse(fallen["checks"]["W4"])
        # A tenth of a single-axis command still beats standing; the wrong way does not.
        self.assertTrue(evaluate(self.rows(single=0.01), rules)["checks"]["W3"])
        self.assertFalse(evaluate(self.rows(single=-0.01), rules)["checks"]["W3"])
        # Standing drift below the smallest checked command (0.1 m/s forward).
        self.assertTrue(evaluate(self.rows(still=(0.09, 0.0, 0.0)), rules)["checks"]["W5"])
        self.assertFalse(evaluate(self.rows(still=(0.11, 0.0, 0.0)), rules)["checks"]["W5"])
        self.assertFalse(evaluate(self.rows(still=(0.0, 0.05, 0.0)), rules)["checks"]["W5"])


if __name__ == "__main__":
    unittest.main()
