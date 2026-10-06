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
        probe = {"checks": {"W4": True}, "angle_deg": [30.0, 32.0], "speed": [0.4, 0.3]}
        self.assertIsNone(check(12000, {"probe": probe}))
        self.assertIsNone(check(11000, {"probe": {**probe, "angle_deg": [50.0, 50.0]}}))
        self.assertIn("ratio angle", check(12000, {"probe": {**probe, "angle_deg": [36.0, 30.0]}}))
        self.assertIn("W4", check(12000, {"probe": {**probe, "checks": {"W4": False}}}))
        self.assertIn("speed", check(12000, {"probe": {**probe, "speed": [0.1, 0.1]}}))

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
    def rows(self, *, angle_scale=1.0, falls=0, single=0.1):
        rows = []
        for push in ("none", "p30_15"):
            for name, twist in DIAGONAL.items():
                for rep in range(3):
                    motion = [twist[0] * 0.5, twist[1] * 0.5 * angle_scale, twist[2] * 0.5]
                    rows.append({"push": push, "cmd": name, "twist": twist, "fell": rep < falls and push == "none",
                                 "mean": motion})
        for name, twist in SINGLE.items():
            for _ in range(3):
                rows.append({"push": "none", "cmd": name, "twist": twist, "fell": False,
                             "mean": [0.0, 0.0, 0.0] if name == "S" else [
                                 (single * 3 if i == 2 else single) * (1 if v > 0 else -1 if v < 0 else 0)
                                 for i, v in enumerate(twist)]})
        return rows

    def test_rules(self) -> None:
        rules = CFG["walk"]["check"]
        good = evaluate(self.rows(), rules)
        self.assertTrue(good["passed"], good)
        self.assertLess(good["angle_deg"][0], 1.0)
        skewed = evaluate(self.rows(angle_scale=0.1), rules)
        self.assertFalse(skewed["checks"]["W1"])
        fallen = evaluate(self.rows(falls=2), rules)
        self.assertFalse(fallen["checks"]["W4"])
        slow = evaluate(self.rows(single=0.01), rules)
        self.assertFalse(slow["checks"]["W5"])


if __name__ == "__main__":
    unittest.main()
