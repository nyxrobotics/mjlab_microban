"""The pipeline's stage watch and the walking judgment rules (CPU, recorded logs)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mjlab_microban.pipeline import monitor as m
from mjlab_microban.pipeline.core import PipelineError, State, load_config
from mjlab_microban.pipeline.walk_probe import DIAGONAL, SINGLE, evaluate

CFG = load_config(dry=False)


def make_monitor(root: Path, stages: dict[str, int]) -> m.Monitor:
    state = State(root / "state")
    return m.Monitor(state=state, record=state.step("walk"), expected_stages=stages, stage_tolerance=1)


class MonitorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)

    def tearDown(self) -> None:
        self.dir.cleanup()

    def test_stage_lines_are_checked_against_the_table(self) -> None:
        log = self.root / "train.log"
        log.write_text("\x1b[1m Learning iteration 2999/30000 \x1b[0m\n"
                       "Curriculum stage 1 penalize stepping + increase velocity at step 72024 (update 3001)\n")
        monitor = make_monitor(self.root, {"penalize stepping + increase velocity": 3000})
        monitor.log_path = log
        monitor.read_log()
        self.assertEqual(monitor.record["stages_seen"], {"penalize stepping + increase velocity": 3001})
        late = make_monitor(self.root / "late", {"tighten foot tracking": 6000})
        log.write_text("Curriculum stage 4 tighten foot tracking at step 150000 (update 6250)\n")
        late.log_path = log
        with self.assertRaisesRegex(PipelineError, "the schedule did not run as written"):
            late.read_log()

    def test_every_due_stage_must_be_seen_and_named_in_the_table(self) -> None:
        stages = {"start": 0, "refine": 4000, "effort_push": 10000}
        log = self.root / "train.log"
        log.write_text("Curriculum stage 1 start at step 0 (update 0)\n"
                       "Curriculum stage 2 refine at step 96000 (update 4000)\n")
        monitor = make_monitor(self.root, stages)
        monitor.log_path = log
        monitor.read_log()
        monitor.require_stages(10000)  # effort_push may still come within the tolerance
        with self.assertRaisesRegex(PipelineError, r"without applying the curriculum stages \['effort_push'\]"):
            monitor.require_stages(10001)
        renamed = make_monitor(self.root / "renamed", stages)
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
        rules = CFG["walk"]["judgment"]
        good = evaluate(self.rows(), rules)
        self.assertTrue(good["passed"], good)
        self.assertEqual(set(good["checks"]), {"falls", "direction", "standing_still"})
        # The diagonals are judged by their falls only.
        self.assertTrue(evaluate(self.rows(scale=(0.5, 0.0, 0.5)), rules)["passed"])
        fallen = evaluate(self.rows(falls=2), rules)
        self.assertFalse(fallen["checks"]["falls"])
        # direction: every single-axis command at its fixed minimum (0.2 forward
        # 0.08, backward 0.04, lateral 0.02, yaw 0.2; yaw rows get 3x).
        self.assertTrue(evaluate(self.rows(single=0.08), rules)["checks"]["direction"])
        self.assertFalse(evaluate(self.rows(single=0.079), rules)["checks"]["direction"])
        self.assertFalse(evaluate(self.rows(single=-0.1), rules)["checks"]["direction"])
        # direction: standing drifts at most 0.05, 0.05 m/s and 0.2 rad/s.
        self.assertTrue(evaluate(self.rows(still=(0.049, -0.049, 0.19)), rules)["checks"]["direction"])
        self.assertFalse(evaluate(self.rows(still=(0.06, 0.0, 0.0)), rules)["checks"]["direction"])
        self.assertFalse(evaluate(self.rows(still=(0.0, 0.0, 0.25)), rules)["checks"]["direction"])
        # standing_still: standing with the feet still; stepping in place (8 per second) fails.
        self.assertEqual(rules["still_touchdowns_per_s"], 0.5)
        self.assertTrue(evaluate(self.rows(touchdowns=0.5), rules)["checks"]["standing_still"])
        stepping = evaluate(self.rows(touchdowns=7.9), rules)
        self.assertFalse(stepping["checks"]["standing_still"] or stepping["passed"])
        self.assertAlmostEqual(stepping["still_touchdowns_per_s"], 7.9)


if __name__ == "__main__":
    unittest.main()
