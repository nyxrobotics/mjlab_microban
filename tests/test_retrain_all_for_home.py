"""Pure parts of scripts/retrain_all_for_home.py and scripts/home_pipeline/robot_pins.py."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "home_pipeline"))
import robot_pins  # noqa: E402

_spec = importlib.util.spec_from_file_location("retrain_all_for_home", REPO / "scripts" / "retrain_all_for_home.py")
pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pipeline)


def probe_receipt(responses: dict[str, float], *, falls: int = 0, overshoot: float = 0.0) -> dict:
    results = [{"name": "neutral", "directional_response": None}]
    results += [{"name": k, "directional_response": {"signed_response": v}} for k, v in responses.items()]
    return {
        "summary": {
            "completed_scenario_count": 9, "fall_scenario_count": falls, "nonfinite_scenario_count": 0,
            "directionally_correct_scenario_count": 8, "raw_action_recurrence_all_steps": True,
            "maximum_actual_soft_limit_violation_rad": overshoot,
        },
        "results": results,
    }


GOOD = {k: v + 0.01 for k, v in pipeline.PROBE_THRESHOLDS.items()}


class ProbeSelectionTest(unittest.TestCase):
    def verdict(self, receipt: dict) -> dict:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "probe.json"
            path.write_text(json.dumps(receipt))
            return pipeline.probe_verdict(path)

    def test_verdict_margins_and_rules(self):
        v = self.verdict(probe_receipt(GOOD))
        self.assertTrue(v["ok"])
        self.assertAlmostEqual(v["worst_margin"], 0.01)
        low = dict(GOOD, lateral_left_0p1=0.015)
        v = self.verdict(probe_receipt(low))
        self.assertFalse(v["ok"])
        self.assertEqual(v["worst"], "lateral_left_0p1")
        self.assertAlmostEqual(v["worst_margin"], -0.005)
        self.assertFalse(self.verdict(probe_receipt(GOOD, falls=1))["ok"])
        self.assertFalse(self.verdict(probe_receipt(GOOD, overshoot=0.1))["ok"])

    def test_selection_uses_worst_case_over_repeats(self):
        def v(margin, ok=True):
            return {"ok": ok, "worst_margin": margin}

        results = {"a": [v(0.02), v(0.001)], "b": [v(0.009), v(0.008)], "c": [v(0.05), v(-0.01, False)]}
        best, row, fallback = pipeline.select_walker(results)
        self.assertEqual(best, "b")
        self.assertFalse(fallback)
        self.assertAlmostEqual(row["worst_case_margin"], 0.008)
        best, row, fallback = pipeline.select_walker({"c": results["c"], "d": [v(-0.002, False)]})
        self.assertEqual(best, "d")
        self.assertTrue(fallback)


class RobotPinsTest(unittest.TestCase):
    def test_training_home_table_and_packager_json(self):
        text = 'X = 1\nTRAINING_HOME_DEG = {\n    "left_knee": 0.0,\n    "left_hip_pitch": 1.0,\n}\n'
        new, changed = robot_pins.set_training_home_deg(text, {"left_knee": 15.0, "left_hip_pitch": -6.5})
        self.assertTrue(changed)
        self.assertIn('    "left_knee": 15.0,\n    "left_hip_pitch": -6.5,\n}', new)
        value = json.dumps({"revision": "a_b", "joint_pos_rad": [0.1] * 30}, separators=(",", ":"))
        text = "PACKAGER_V12_HOME_POSE_JSON = (\n    '{}'\n)\n"
        new, changed = robot_pins.set_packager_home_json(text, value)
        self.assertTrue(changed)
        namespace: dict = {}
        exec(new, namespace)
        self.assertEqual(namespace["PACKAGER_V12_HOME_POSE_JSON"], value)
        self.assertEqual(robot_pins.set_packager_home_json(new, value), (new, False))

    def test_run_pins(self):
        text = 'A_SHA256 = (\n    "' + "0" * 64 + '"\n)\nA_ITERATION = 20_000\n'
        new, changed = robot_pins.set_hex_pin(text, "A_SHA256", "f" * 64)
        self.assertTrue(changed)
        self.assertIn('"' + "f" * 64 + '"', new)
        new, changed = robot_pins.set_int_pin(new, "A_ITERATION", 29998)
        self.assertIn("A_ITERATION = 29_998\n", new)
        with self.assertRaises(ValueError):
            robot_pins.set_hex_pin(text, "MISSING_SHA256", "f" * 64)
        with self.assertRaises(ValueError):
            robot_pins.set_hex_pin(text, "A_SHA256", "XYZ")

    def test_packager_metadata_and_source_comment(self):
        text = ('    "v12_legacy_source_checkpoint_sha256": (\n        "' + "0" * 64 + '"\n    ),\n'
                '    "v12_legacy_source_checkpoint_iteration": "20000",\n')
        new, changed = robot_pins.set_quoted_values(text, {
            "v12_legacy_source_checkpoint_sha256": "a" * 64,
            "v12_legacy_source_checkpoint_iteration": "29998"})
        self.assertTrue(changed)
        self.assertIn('        "' + "a" * 64 + '"\n', new)
        self.assertIn('"v12_legacy_source_checkpoint_iteration": "29998"', new)
        text = ("# pinned on the robot.\n#\n# Centered-HOME chain (old): the frozen\n# source is x.\n"
                'EXPECTED_V12_LEGACY_SOURCE_CHECKPOINT_SHA256 = (\n    "' + "0" * 64 + '"\n)\n')
        new, changed = robot_pins.set_source_comment(text, tag="t", source_path="repo://c/model_1.pt",
                                                     iteration=1, probe_path="repo://p.json")
        self.assertTrue(changed)
        self.assertTrue(new.startswith("# pinned on the robot.\n#\n# HOME chain (mjlab_microban"))
        self.assertIn("#   repo://c/model_1.pt\n", new)
        self.assertNotIn("Centered-HOME", new)
        again, changed = robot_pins.set_source_comment(new, tag="t", source_path="repo://c/model_1.pt",
                                                       iteration=1, probe_path="repo://p.json")
        self.assertEqual((again, changed), (new, False))

    def test_robot_yaml_subset_parser(self):
        text = '# c\nschema_version: 1\ntag: "t"\njoint_pos_deg:\n  head: 0.0\nroot_pos_m: [0.0, 0.0, 0.17]\n'
        self.assertEqual(robot_pins.parse_robot_home_yaml(text),
                         {"schema_version": 1, "tag": "t", "joint_pos_deg": {"head": 0.0},
                          "root_pos_m": [0.0, 0.0, 0.17]})
        self.assertEqual(robot_pins.signed_degree_token(1.198384259489), "plus1p198384259489")
        self.assertEqual(robot_pins.signed_degree_token(-10.0), "minus10")


if __name__ == "__main__":
    unittest.main()


def make_pipeline(state_dir: Path | None = None, *extra: str):
    args = pipeline.parse_args(["--robot-repo", "/nonexistent", "--robot-branch", "x", *extra])
    p = pipeline.Pipeline(args)
    if state_dir is not None:
        p.state_dir = state_dir
        p.owns_state = True
    return p


class GpuSchedulingTest(unittest.TestCase):
    """Review round 4: no head-of-line blocking, no stall-check deadlock."""

    def test_reservations_do_not_block_a_fitting_job(self):
        p = make_pipeline()
        p.gpu_free_mib = lambda: 5000
        self.assertIsNone(p.gpu_reserve(11000))  # the big job does not fit, holds nothing
        first = p.gpu_reserve(3000)
        self.assertIsNotNone(first)
        # The first job's memory is not visible yet: its reservation counts.
        self.assertIsNone(p.gpu_reserve(3000))
        p.gpu_release(first)
        self.assertIsNotNone(p.gpu_reserve(3000))

    def test_reservations_expire(self):
        p = make_pipeline()
        p.gpu_free_mib = lambda: 5000
        reservation = p.gpu_reserve(4000)
        p.gpu_reservations[reservation] = (4000, 0.0)  # expired
        self.assertIsNotNone(p.gpu_reserve(4000))

    def test_non_waiting_job_returns_none_when_full(self):
        p = make_pipeline()
        p.gpu_free_mib = lambda: 1000
        self.assertIsNone(p.gpu_job(3000, "probe", ["true"], "probe", wait=False))

    def test_default_pico_memory_fits_a_2048_env_v12_job(self):
        self.assertGreaterEqual(make_pipeline().args.pico_gpu_mib, 17000)


class StateOwnershipTest(unittest.TestCase):
    def test_refused_second_instance_leaves_the_live_state_alone(self):
        with tempfile.TemporaryDirectory() as d:
            state = Path(d)
            (state / "state.json").write_text('{"children": {"1": {"name": "walk"}}}')
            (state / "STATUS.log").write_text("live\n")
            p = make_pipeline()
            p.state_dir = state  # as open_state sets it before the lock is refused
            p.put("stopped", {"code": 4})
            p.log("STOPPED: another instance holds it")
            self.assertEqual(json.loads((state / "state.json").read_text()),
                             {"children": {"1": {"name": "walk"}}})
            self.assertEqual((state / "STATUS.log").read_text(), "live\n")

    def test_unexpected_errors_are_reported_and_stop_own_jobs(self):
        with tempfile.TemporaryDirectory() as d:
            state = Path(d)
            killed = []

            def execute(self):
                self.state_dir, self.owns_state = state, True
                raise KeyError("summary")

            original = pipeline.Pipeline.execute
            original_kill = pipeline.Pipeline.kill_all_children
            pipeline.Pipeline.execute = execute
            pipeline.Pipeline.kill_all_children = lambda self: killed.append(True)
            try:
                code = pipeline.main(["--robot-repo", "/x", "--robot-branch", "x"])
            finally:
                pipeline.Pipeline.execute = original
                pipeline.Pipeline.kill_all_children = original_kill
            self.assertEqual(code, pipeline.EXIT_FAILED)
            self.assertEqual(killed, [True])
            stopped = json.loads((state / "state.json").read_text())["stopped"]
            self.assertIn("unexpected KeyError", stopped["reason"])
            self.assertIn("STOPPED: unexpected KeyError", (state / "STATUS.log").read_text())


class HomeYamlGuardTest(unittest.TestCase):
    def test_an_edit_during_the_run_stops_it(self):
        with tempfile.TemporaryDirectory() as d:
            yaml_path = Path(d) / "home_pose.yaml"
            yaml_path.write_text("a: 1\n")
            original = pipeline.HOME_YAML
            pipeline.HOME_YAML = yaml_path
            try:
                p = make_pipeline(Path(d))
                p.yaml_sha256 = pipeline.sha256(yaml_path)
                p.verify_home_yaml()
                p.capture(["true"])
                yaml_path.write_text("a: 2\n")
                with self.assertRaisesRegex(pipeline.PipelineError, "changed during the run") as raised:
                    p.capture(["true"])
                self.assertEqual(raised.exception.code, pipeline.EXIT_INPUT)
                with self.assertRaisesRegex(pipeline.PipelineError, "changed during the run"):
                    p.run("job", ["true"], "cpu")
            finally:
                pipeline.HOME_YAML = original

    def test_an_edit_while_a_job_runs_stops_at_its_end(self):
        """Verifier round 3: a job loads the YAML seconds after it starts."""

        with tempfile.TemporaryDirectory() as d:
            yaml_path = Path(d) / "home_pose.yaml"
            yaml_path.write_text("a: 1\n")
            original = pipeline.HOME_YAML
            pipeline.HOME_YAML = yaml_path
            try:
                p = make_pipeline(Path(d))
                p.yaml_sha256 = pipeline.sha256(yaml_path)
                with self.assertRaises(pipeline.HomeYamlChanged) as raised:
                    p.run("job", ["sh", "-c", f"echo 'a: 2' > {yaml_path}"], "cpu")
                self.assertGreater(raised.exception.edit_time, 0)
                self.assertIsNotNone(p.job_log[-1]["ended"])
            finally:
                pipeline.HOME_YAML = original

    def test_output_of_jobs_that_may_have_loaded_the_edit_is_quarantined(self):
        import os
        import time

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            saved = (pipeline.LOG_ROOT, pipeline.GATE_ROOT, pipeline.PROBE_ROOT)
            pipeline.LOG_ROOT = root / "logs" / "rsl_rl"
            pipeline.GATE_ROOT = root / "artifacts" / "teleop_v12_gates"
            pipeline.PROBE_ROOT = root / "artifacts" / "legacy_teleop_probe"
            try:
                state = root / "state"
                (state / "pico").mkdir(parents=True)
                p = make_pipeline(state)
                now = time.time()
                exp = pipeline.LOG_ROOT / "mjlab_microban_teleop_v12"
                for name, t in (("old_run", now - 7200), ("long_run", now - 3600), ("new_run", now - 20)):
                    (exp / name).mkdir(parents=True)
                    (exp / name / "params.yaml").write_text("x")
                    os.utime(exp / name / "params.yaml", (t, t))
                (exp / "long_run" / "model_5.pt").write_text("x")  # still written after the edit
                pipeline.GATE_ROOT.mkdir(parents=True)
                (pipeline.GATE_ROOT / "old_gate.json").write_text("{}")
                os.utime(pipeline.GATE_ROOT / "old_gate.json", (now - 7200, now - 7200))
                (pipeline.GATE_ROOT / "new_gate.json").write_text("{}")
                (state / "pico" / "new_report.json").write_text("{}")
                p.job_log = [
                    {"name": "done_before", "started": now - 7300, "ended": now - 7000},
                    {"name": "long", "started": now - 3600, "ended": None},  # loaded the HOME long ago
                    {"name": "racer", "started": now - 25, "ended": now - 5},  # loaded the edit
                ]
                p.quarantine_after_yaml_change(now - 22)
                left = sorted(x.name for x in exp.iterdir())
                self.assertEqual(left, ["long_run", "old_run"])
                self.assertEqual(sorted(x.name for x in pipeline.GATE_ROOT.iterdir()), ["old_gate.json"])
                self.assertFalse((state / "pico" / "new_report.json").exists())
                record = next(iter(p.get("quarantined").values()))
                self.assertEqual(record["jobs"], ["racer"])
                self.assertEqual(len(record["moved"]), 3)
                quarantine = next((state / "quarantine").iterdir())
                self.assertTrue((quarantine / "rsl_rl" / "mjlab_microban_teleop_v12" / "new_run").is_dir())
                # No job in the load window: nothing is moved.
                p.job_log = [{"name": "long", "started": now - 3600, "ended": None}]
                p.quarantine_after_yaml_change(now)
                self.assertEqual(sorted(x.name for x in exp.iterdir()), ["long_run", "old_run"])
            finally:
                pipeline.LOG_ROOT, pipeline.GATE_ROOT, pipeline.PROBE_ROOT = saved


class StaleStopTest(unittest.TestCase):
    def test_a_resumed_run_moves_the_old_stop_to_history(self):
        with tempfile.TemporaryDirectory() as d:
            state = Path(d) / "state"
            state.mkdir()
            (state / "state.json").write_text(json.dumps({
                "stopped": {"at": "x", "code": 1, "reason": "old"}, "dry_run": False, "prefix": "home_x",
                "home_identity": {"joint_hash": "h", "label": "l", "tag": "t"}}))
            yaml_path = Path(d) / "home_pose.yaml"
            yaml_path.write_text("a: 1\n")
            original = pipeline.HOME_YAML
            pipeline.HOME_YAML = yaml_path
            try:
                p = make_pipeline(None, "--state-dir", str(state))
                p.capture = lambda cmd, **kw: unittest.mock.Mock(
                    returncode=0, stdout=json.dumps({"tag": "t", "joint_hash": "h", "label": "l"}), stderr="")
                p.open_state()
            finally:
                pipeline.HOME_YAML = original
            saved = json.loads((state / "state.json").read_text())
            self.assertNotIn("stopped", saved)
            self.assertEqual(saved["previous_stops"], [{"at": "x", "code": 1, "reason": "old"}])


class PreflightTest(unittest.TestCase):
    def robot(self, d: Path, *, reads_yaml: bool) -> Path:
        (d / ".git").mkdir()
        (d / "tools").mkdir()
        (d / "tools" / "validate_pico_policy.py").write_text("")
        (d / "src").mkdir()
        if reads_yaml:
            (d / "src" / "home_pose.py").write_text("")
            (d / "src" / "constants.py").write_text("from home_pose import (\n)\n")
        else:
            (d / "src" / "constants.py").write_text("NEUTRAL_POSE = {}\n")
        return d

    def test_robot_checkout_must_read_the_home_yaml(self):
        with tempfile.TemporaryDirectory() as d:
            p = make_pipeline()
            p.robot = self.robot(Path(d), reads_yaml=False)
            with self.assertRaisesRegex(pipeline.PipelineError, "does not read config/home_pose.yaml") as raised:
                p.preflight()
            self.assertEqual(raised.exception.code, pipeline.EXIT_INPUT)
        with tempfile.TemporaryDirectory() as d:
            p = make_pipeline()
            p.robot = self.robot(Path(d), reads_yaml=True)
            p.preflight()


class WalkerFallbackTest(unittest.TestCase):
    def test_next_candidate_after_a_failed_start_probe(self):
        with tempfile.TemporaryDirectory() as d:
            p = make_pipeline(Path(d))
            installed = []
            p.install_walker = lambda best, row, fallback: (
                installed.append(best),
                p.put("walk", "selected", {"from": best, "sha256": best[-1] * 64}),
            )
            v = lambda margin: {"ok": True, "worst_margin": margin}  # noqa: E731
            p.put("walk", "candidates", {"/r/model_1.pt": [v(0.03)], "/r/model_2.pt": [v(0.02)],
                                         "/r/model_3.pt": [v(0.01)]})
            p.put("walk", "selected", {"from": "/r/model_1.pt", "sha256": "1" * 64})
            self.assertTrue(p.reselect_walker("probe failed"))
            self.assertEqual(installed, ["/r/model_2.pt"])
            self.assertTrue(p.reselect_walker("probe failed"))
            self.assertEqual(installed, ["/r/model_2.pt", "/r/model_3.pt"])
            self.assertFalse(p.reselect_walker("probe failed"))
            self.assertEqual(len(p.get("walk", "rejected")), 3)

    def test_dry_run_reprobes_the_candidates_before_stopping(self):
        with tempfile.TemporaryDirectory() as d:
            p = make_pipeline(Path(d), "--dry-run")
            installed = []
            p.install_walker = lambda best, row, fallback: (
                installed.append(best),
                p.put("walk", "selected", {"from": best, "sha256": best[-4] * 64}),
            )
            v = lambda margin: {"ok": True, "worst_margin": margin}  # noqa: E731
            p.put("walk", "candidates", {"/r/model_1.pt": [v(0.03)], "/r/model_2.pt": [v(0.02)]})
            p.put("walk", "selected", {"from": "/r/model_1.pt", "sha256": "1" * 64})
            for _ in range(pipeline.DRY_START_ATTEMPTS - 1):
                self.assertTrue(p.reselect_walker("probe failed"))
            self.assertEqual(installed, ["/r/model_2.pt", "/r/model_1.pt", "/r/model_2.pt"])
            self.assertFalse(p.reselect_walker("probe failed"))
            self.assertEqual(len(p.get("walk", "rejected")), pipeline.DRY_START_ATTEMPTS)


class DryRunSmokeCorpusTest(unittest.TestCase):
    def test_short_corpus_is_cycled_to_sixteen_rows(self):
        sys.path.insert(0, str(REPO / "scripts" / "home_pipeline"))
        import dry_run_tools

        self.assertEqual(dry_run_tools.SMOKE_ROWS, 16)
        source = (REPO / "scripts" / "home_pipeline" / "dry_run_tools.py").read_text()
        self.assertIn("rows[i % len(rows)] for i in range(SMOKE_ROWS)", source)


class ArgumentsTest(unittest.TestCase):
    def test_a_dry_run_needs_a_walker_of_this_home_or_plumbing_mode(self):
        with tempfile.TemporaryDirectory() as d:
            robot = PreflightTest().robot(Path(d), reads_yaml=True)
            p = make_pipeline(None, "--dry-run")
            p.robot = robot
            with self.assertRaisesRegex(pipeline.PipelineError, "--dry-run-plumbing") as raised:
                p.preflight()
            self.assertEqual(raised.exception.code, pipeline.EXIT_INPUT)
            p = make_pipeline(None, "--dry-run", "--dry-run-walk-init", str(Path(d) / "model_7.pt"))
            p.robot = robot
            with self.assertRaisesRegex(pipeline.PipelineError, "model_<N>.pt"):
                p.preflight()
            (Path(d) / "model_7.pt").write_bytes(b"x")
            p.preflight()
            p = make_pipeline(None, "--dry-run", "--dry-run-plumbing")
            p.robot = robot
            p.preflight()


class DryRunToolsTest(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(REPO / "scripts" / "home_pipeline"))
        import dry_run_tools

        self.tools = dry_run_tools

    def test_forced_probe_keeps_the_measured_values(self):
        receipt = {"summary": {"scenario_count": 9, "completed_scenario_count": 4, "fall_scenario_count": 5,
                               "directionally_correct_scenario_count": 1},
                   "results": [{"completed": False, "fell": True, "executed_steps": 120,
                                "maximum_actual_soft_limit_violation_rad": 0.5}] * 9}
        forced = self.tools.forced_probe_report(receipt, 0.0873)
        for key, value in self.tools.PROBE_PASS_SUMMARY.items():
            self.assertIs(type(forced["summary"][key]), type(value))
            self.assertEqual(forced["summary"][key], value)
        self.assertTrue(all(r["completed"] and not r["fell"] and r["executed_steps"] == 300
                            and r["maximum_actual_soft_limit_violation_rad"] == 0.0 for r in forced["results"]))
        self.assertEqual(forced["dry_run_original_summary"]["fall_scenario_count"], 5)
        self.assertEqual(forced["dry_run_original_results"][0]["executed_steps"], 120)
        self.assertTrue(forced["dry_run_forced_pass_not_deployable"])
        self.assertEqual(receipt["summary"]["fall_scenario_count"], 5)  # input untouched
        with self.assertRaises(ValueError):
            self.tools.forced_probe_report({"summary": {}, "results": []}, 0.0873)

    def test_package_forces_the_locomotion_pass_fields_the_robot_rechecks(self):
        # A from-scratch plumbing policy falls in every scenario; the robot
        # validator re-checks these fields in the package metadata.
        with tempfile.TemporaryDirectory() as d:
            prefix = str(Path(d) / "run_model_14999")
            Path(prefix + "_9x300.json").write_text(json.dumps({
                "status": "fail",
                "summary": {"scenario_count": 9, "completed_scenario_count": 0, "fall_scenario_count": 9,
                            "nonfinite_scenario_count": 0, "directionally_correct_scenario_count": 3,
                            "directional_scenario_count": 8},
                "results": [{"completed": False, "fell": True, "executed_steps": 40,
                             "maximum_actual_soft_limit_violation_rad": 0.105}] * 9}))
            Path(prefix + "_tracking.json").write_text(json.dumps({
                "status": "fail", "runtime_smoke_observations": [[0.0], [1.0], [2.0]]}))
            loco, tracking = self.tools.forced_stage_reports(prefix, Path(d), 0.0873)
            loco = json.loads(loco.read_text())
            self.assertEqual(loco["status"], "pass")
            self.assertEqual(loco["dry_run_original_status"], "fail")
            self.assertEqual((loco["summary"]["fall_scenario_count"],
                              loco["summary"]["directionally_correct_scenario_count"]), (0, 8))
            self.assertEqual(loco["dry_run_original_summary"]["fall_scenario_count"], 9)
            self.assertTrue(all(r["maximum_actual_soft_limit_violation_rad"] == 0.0 for r in loco["results"]))
            tracking = json.loads(tracking.read_text())
            self.assertEqual(len(tracking["runtime_smoke_observations"]), 16)
            self.assertEqual(tracking["dry_run_smoke_rows_original_count"], 3)
        with unittest.mock.patch("sys.stderr"):  # boundary arguments come in pairs
            self.assertEqual(self.tools.main(["package", "c", "p", "g", "o", "r", "b"]), 2)

class SerialGpuTest(unittest.TestCase):
    def test_one_gpu_job_at_a_time(self):
        with tempfile.TemporaryDirectory() as d:
            p = make_pipeline(Path(d), "--serial-gpu")
            self.assertTrue(p.args.sequential)
            p.gpu_free_mib = lambda: 30000
            seen = []

            def run(name, cmd, kind, **kwargs):
                # A nested non-waiting job (a walking probe from the training poll) is deferred.
                seen.append(p.gpu_job(100, "nested", ["true"], "probe", wait=False))
                return 0

            p.run = run
            self.assertEqual(p.gpu_job(100, "train", ["true"], "train"), 0)
            self.assertEqual(seen, [None])
            self.assertTrue(p.gpu_serial.acquire(blocking=False))  # released after the job
            p.gpu_serial.release()
        self.assertTrue(make_pipeline(None, "--dry-run", "--dry-run-plumbing").args.serial_gpu)


class UnmeasuredFootScenarioTest(unittest.TestCase):
    """A from-scratch plumbing policy can fall before a foot target is sampled."""

    def test_unmeasured_error_fails_instead_of_crashing(self):
        from mjlab_microban.scripts.evaluate_teleop_v12_tracking import _measured_within

        self.assertTrue(_measured_within({"rms": 0.01}, "rms", 0.02))
        self.assertFalse(_measured_within({"rms": 0.03}, "rms", 0.02))
        self.assertFalse(_measured_within({"rms": None, "sample_count": 0}, "rms", 0.02))


class BoundaryGateArgsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.original = pipeline.GATE_ROOT
        pipeline.GATE_ROOT = self.root
        self.p = make_pipeline(self.root / "state")
        (self.root / "state").mkdir()
        self.p.prefix = "home_x"
        self.p.put("pico", "b9999", "passed", {"run": "rescue", "sha256": "0" * 64})
        self.p.put("pico", "segments", "10000_to10100", "canary")
        self.p.latest_v12 = lambda seg: "newer_canary_retry_not_continued_from"
        (self.root / "rescue_model_9999_gate.json").write_text("{}")
        (self.root / "canary_model_10099_gate.json").write_text("{}")

    def tearDown(self):
        pipeline.GATE_ROOT = self.original
        self.tmp.cleanup()

    def test_required_gates_of_the_continued_lineage_are_passed(self):
        self.p.home = {"v12_required_boundary_gate_clocks": [10000, 10100]}
        self.p.gate_ok = lambda run, end: True
        self.assertEqual(self.p.boundary_gate_args(),
                         ["--boundary-gate", str(self.root / "rescue_model_9999_gate.json"),
                          "--boundary-gate", str(self.root / "canary_model_10099_gate.json")])

    def test_a_missing_required_gate_stops_instead_of_being_dropped(self):
        self.p.home = {"v12_required_boundary_gate_clocks": [10000, 10100]}
        self.p.gate_ok = lambda run, end: run == "rescue"
        with self.assertRaisesRegex(pipeline.PipelineError, "10100 .canary/model_10099"):
            self.p.boundary_gate_args()
        # An old state without the field requires both too.
        self.p.home = {}
        with self.assertRaisesRegex(pipeline.PipelineError, "10100"):
            self.p.boundary_gate_args()

    def test_centered_home_records_only_validating_gates(self):
        self.p.home = {"v12_required_boundary_gate_clocks": []}
        self.p.gate_ok = lambda run, end: run == "rescue"
        self.assertEqual(self.p.boundary_gate_args(),
                         ["--boundary-gate", str(self.root / "rescue_model_9999_gate.json")])

    def test_dry_package_gets_the_10000_and_10100_gates(self):
        self.assertEqual(self.p.dry_boundary_args(), [])
        for run, end in (("rescue", 9999), ("canary", 10099)):
            for suffix in ("_9x300.json", "_tracking.json", "_onnx.json"):
                (self.root / f"{run}_model_{end}{suffix}").write_text("{}")
        self.assertEqual(self.p.dry_boundary_args(), [
            str(pipeline.V12_EXP / "rescue" / "model_9999.pt"), str(self.root / "rescue_model_9999"),
            str(pipeline.V12_EXP / "canary" / "model_10099.pt"), str(self.root / "canary_model_10099")])


class CanaryRetryResumeTest(unittest.TestCase):
    """A canary retry interrupted before it saved model_<end> is retried on resume."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.original = pipeline.V12_EXP
        pipeline.V12_EXP = root / "v12"
        pipeline.V12_EXP.mkdir()
        self.state = root / "state"
        self.state.mkdir()
        self.mk("2026-01-01_00-00-00_home_x_v12_0_to3000", 2999)
        self.mk("2026-01-01_00-00-01_home_x_v12_3000_to3100", 3099)  # fails on accuracy only

    def tearDown(self):
        pipeline.V12_EXP = self.original
        self.tmp.cleanup()

    def mk(self, name: str, end: int) -> None:
        (pipeline.V12_EXP / name).mkdir(exist_ok=True)
        (pipeline.V12_EXP / name / f"model_{end}.pt").write_bytes(name.encode())

    def run_pico(self, *, interrupt: bool, retry_fails: bool = False):
        p = make_pipeline(self.state)
        path = self.state / "state.json"
        p.state = json.loads(path.read_text()) if path.exists() else {}
        p.prefix = "home_x"
        p.check_recipe = lambda run, end: None
        calls = []

        def judged_gate(run, end, key):
            calls.append(("gate", run))
            bad = run.endswith("00-00-01_home_x_v12_3000_to3100") or (retry_fails and "00-00-02" in run)
            return (not bad), (["hand_tracking_rms"] if bad else []), []

        def v12_train(seg, prev, lift_to=None, seed=None):
            calls.append(("train", seg))
            if seg.endswith("3000_to3100"):
                self.assertEqual(seed, 43)  # a canary retry trains with a new seed
            if interrupt:
                raise SystemExit(130)  # Ctrl-C while waiting for GPU memory
            if seg.endswith("3000_to3100"):
                self.mk("2026-01-01_00-00-02_home_x_v12_3000_to3100", 3099)
                return
            raise RuntimeError("later segments are not modelled")

        p.judged_gate, p.v12_train = judged_gate, v12_train
        return p, calls

    def test_resume_runs_the_interrupted_retry(self):
        p, calls = self.run_pico(interrupt=True)
        with self.assertRaises(SystemExit):
            p.step_pico()
        self.assertEqual(calls[-1], ("train", "home_x_v12_3000_to3100"))
        p, calls = self.run_pico(interrupt=False)
        with self.assertRaisesRegex(RuntimeError, "not modelled"):
            p.step_pico()  # the retry ran and passed; the chain went on to 3100->7000
        self.assertIn(("gate", "2026-01-01_00-00-02_home_x_v12_3000_to3100"), calls)

    def test_resume_after_the_retry_created_its_run_dir_keeps_the_retry_seed(self):
        """Verifier round 3: the retry is usually interrupted after its run dir exists."""

        p, calls = self.run_pico(interrupt=True)
        with self.assertRaises(SystemExit):
            p.step_pico()
        partial = pipeline.V12_EXP / "2026-01-01_00-00-02_home_x_v12_3000_to3100"
        (partial / "params").mkdir(parents=True)  # created by the launcher, no model_3099 yet
        p = make_pipeline(self.state)
        p.state = json.loads((self.state / "state.json").read_text())
        p.prefix = "home_x"
        p.check_recipe = lambda run, end: None
        seeds, gates = [], []

        def v12_train(seg, prev, lift_to=None, seed=None):
            seeds.append((seg, seed))
            if seg.endswith("3000_to3100"):
                self.mk("2026-01-01_00-00-03_home_x_v12_3000_to3100", 3099)
                return
            raise RuntimeError("later segments are not modelled")

        def judged_gate(run, end, key):
            gates.append(run)
            bad = run.endswith("00-00-01_home_x_v12_3000_to3100")
            return (not bad), (["hand_tracking_rms"] if bad else []), []

        p.judged_gate, p.v12_train = judged_gate, v12_train
        with self.assertRaisesRegex(RuntimeError, "not modelled"):
            p.step_pico()
        self.assertEqual(seeds[0], ("home_x_v12_3000_to3100", pipeline.V12_TRAIN_SEED + 1))
        self.assertIn("2026-01-01_00-00-03_home_x_v12_3000_to3100", gates)

    def test_a_failed_retry_still_stops(self):
        p, calls = self.run_pico(interrupt=False, retry_fails=True)
        with self.assertRaisesRegex(pipeline.PipelineError, "v12 gate 3099 failed for 2026-01-01_00-00-02"):
            p.step_pico()
        self.assertEqual([c for c in calls if c[0] == "train"], [("train", "home_x_v12_3000_to3100")])


class GateCrashTest(unittest.TestCase):
    """An evaluator that died without reports is no verdict: not cached, not a failed gate."""

    def test_crash_is_reevaluated_on_rerun(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            originals = pipeline.V12_EXP, pipeline.GATE_ROOT
            pipeline.V12_EXP, pipeline.GATE_ROOT = root / "v12", root / "gates"
            pipeline.GATE_ROOT.mkdir()
            run = "2026-01-01_00-00-01_home_x_v12_3000_to3100"
            (pipeline.V12_EXP / run).mkdir(parents=True)
            (pipeline.V12_EXP / run / "model_3099.pt").write_bytes(b"ckpt")
            (root / "state").mkdir()
            try:
                evaluations = []

                def make(writes_reports: bool):
                    p = make_pipeline(root / "state")
                    path = root / "state" / "state.json"
                    p.state = json.loads(path.read_text()) if path.exists() else {}
                    p.gate_ok = lambda r, e: False

                    def gpu_job(need, name, cmd, kind, **kw):
                        evaluations.append(name)
                        if writes_reports:
                            prefix = pipeline.GATE_ROOT / f"{run}_model_3099"
                            for suffix in ("_9x300.json", "_tracking.json", "_onnx.json"):
                                Path(f"{prefix}{suffix}").write_text(json.dumps({"checks": {"a": True}}))
                            return 0
                        return 137  # OOM-killed on the shared GPU

                    p.gpu_job = gpu_job
                    p.run = lambda name, cmd, kind, **kw: evaluations.append(name) or 0  # gate create
                    return p

                with self.assertRaisesRegex(pipeline.PipelineError, "wrote no report"):
                    make(False).judged_gate(run, 3099, "canary:3000_to3100")
                self.assertIsNone(make(False).get("pico", "gates", f"{run}_model_3099", "checkpoint_sha256"))
                self.assertEqual(make(True).judged_gate(run, 3099, "canary:3000_to3100"), (True, [], []))
                # Three evaluators each time (one by one, as the stage script), then the gate create.
                self.assertEqual(len(evaluations), 7)
                self.assertEqual(evaluations[-1], "gate_3099_create")
                # A real verdict is cached: no further evaluation.
                self.assertEqual(make(True).judged_gate(run, 3099, "canary:3000_to3100"), (True, [], []))
                self.assertEqual(len(evaluations), 7)
            finally:
                pipeline.V12_EXP, pipeline.GATE_ROOT = originals


class UntrackedPreflightTest(unittest.TestCase):
    """The training-repo check counts untracked files, as the 9999 corner rescue does."""

    def test_untracked_files_are_refused_up_front(self):
        import subprocess

        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            git = lambda *a: subprocess.run(["git", "-C", d, *a], check=True, capture_output=True)  # noqa: E731
            git("init", "-q", "-b", "main")
            git("config", "user.email", "t@example.com")
            git("config", "user.name", "t")
            (repo / "config").mkdir()
            (repo / "config" / "home_pose.yaml").write_text("a: 1\n")
            git("add", ".")
            git("commit", "-qm", "base")
            original = pipeline.REPO
            pipeline.REPO = repo
            try:
                for extra, dry in (((), False), (("--allow-dirty",), False), (("--allow-dirty", "--dry-run",
                                                                                "--dry-run-plumbing"), True)):
                    p = make_pipeline(None, *extra)
                    p.state_dir = repo / ".git" / "state"
                    p.state_dir.mkdir(exist_ok=True)
                    p.owns_state = True
                    p.robot = repo  # the robot half is not under test
                    p.args.robot_branch = "main"
                    (repo / "config" / "home_pose.yaml").write_text("a: 2\n")
                    (repo / "notes.txt").write_text("x\n")
                    if dry:
                        p.prepare_branches()
                        continue
                    with self.assertRaisesRegex(pipeline.PipelineError, "notes.txt"):
                        p.prepare_branches()
                (repo / "notes.txt").unlink()
                p = make_pipeline(None)
                p.state_dir = repo / ".git" / "state"
                p.owns_state = True
                p.robot = repo
                p.args.robot_branch = "main"
                p.prepare_branches()  # only the HOME yaml is changed
            finally:
                pipeline.REPO = original


class LiftClockParentTest(unittest.TestCase):
    """A dry lift records its parent like a resumed run, for the packager's ancestry walk."""

    def test_lift_records_the_resume_parent(self):
        sys.path.insert(0, str(REPO / "scripts" / "home_pipeline"))
        import dry_run_tools
        import torch

        from mjlab_microban.tasks.microban_teleop_v12_actor import teleop_v12_active_adapter_columns

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "run_a").mkdir()
            source = root / "run_a" / "model_9999.pt"
            torch.save({"iter": 9999, "infos": {
                "env_state": {"common_step_counter": 10000 * 24},
                "active_actor_columns_at_save": list(teleop_v12_active_adapter_columns(10000 * 24))}}, source)
            dry_run_tools.lift_clock(str(source), str(root / "lift"), "10096")
            text = (root / "lift" / "params" / "agent.yaml").read_text()
            self.assertIn("load_run: ^run_a$\n", text)
            self.assertIn("load_checkpoint: ^model_9999[.]pt$\n", text)
            from mjlab_microban.scripts.export_teleop_v12_deployment import _resume_ancestry

            (root / "final").mkdir()
            (root / "final" / "params").mkdir()
            (root / "final" / "params" / "agent.yaml").write_text(
                "resume: true\nload_run: ^lift$\nload_checkpoint: ^model_10096[.]pt$\n")
            self.assertEqual(_resume_ancestry(root / "final" / "model_14999.pt"),
                             [(root / "lift" / "model_10096.pt").resolve(), source.resolve()])


class RealGateVerdictTest(unittest.TestCase):
    """A real gate failing on tracking is a verdict (every evaluator runs), not a crash.

    scripts/evaluate_microban_teleop_v12_stage.sh (set -e) stops after the
    tracking evaluator returns 1 for "fail", so its ONNX report is never
    written; the pipeline runs the evaluators one by one instead.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.originals = pipeline.V12_EXP, pipeline.GATE_ROOT
        pipeline.V12_EXP, pipeline.GATE_ROOT = root / "v12", root / "gates"
        self.run_name = "2026-01-01_00-00-01_home_x_v12_3000_to3100"
        (pipeline.V12_EXP / self.run_name).mkdir(parents=True)
        (pipeline.V12_EXP / self.run_name / "model_3099.pt").write_bytes(b"ckpt")
        self.state = root / "state"
        self.state.mkdir()
        self.jobs: list = []

    def tearDown(self):
        pipeline.V12_EXP, pipeline.GATE_ROOT = self.originals
        self.tmp.cleanup()

    def make(self, verdicts: dict):
        """verdicts: evaluator -> (rc, checks or None for "no report written")."""

        p = make_pipeline(self.state)
        path = self.state / "state.json"
        p.state = json.loads(path.read_text()) if path.exists() else {}
        p.gate_ok = lambda r, e: False
        names = {"loco": "_9x300.json", "tracking": "_tracking.json", "onnx": "_onnx.json"}

        def gpu_job(need, name, cmd, kind, **kw):
            evaluator = name.rsplit("_", 1)[1]
            self.jobs.append(evaluator)
            rc, checks = verdicts[evaluator]
            if checks is not None:
                out = Path(cmd[cmd.index("--output") + 1])
                self.assertEqual(out.name, f"{self.run_name}_model_3099{names[evaluator]}")
                out.write_text(json.dumps({"status": "pass" if rc == 0 else "fail", "checks": checks}))
            return rc

        p.gpu_job = gpu_job
        p.run = lambda name, cmd, kind, **kw: self.jobs.append(name) or 0
        return p

    def test_tracking_failure_is_a_cached_verdict(self):
        verdicts = {"loco": (0, {"a": True}), "tracking": (1, {"hand_tracking_rms": False, "b": True}),
                    "onnx": (0, {"c": True})}
        got = self.make(verdicts).judged_gate(self.run_name, 3099, "canary:3000_to3100")
        self.assertEqual(got, (False, ["hand_tracking_rms"], []))
        self.assertEqual(self.jobs, ["loco", "tracking", "onnx"])  # no gate create after a failure
        self.assertEqual(self.make(verdicts).judged_gate(self.run_name, 3099, "canary:3000_to3100"), got)
        self.assertEqual(len(self.jobs), 3)  # cached

    def test_locomotion_failure_still_runs_tracking_and_onnx(self):
        verdicts = {"loco": (1, {"falls": False}), "tracking": (0, {"a": True}), "onnx": (0, {"c": True})}
        got = self.make(verdicts).judged_gate(self.run_name, 3099, "canary:3000_to3100")
        self.assertEqual(got, (False, [], ["falls"]))

    def test_a_missing_report_is_still_a_crash(self):
        verdicts = {"loco": (0, {"a": True}), "tracking": (137, None), "onnx": (0, {"c": True})}
        with self.assertRaisesRegex(pipeline.PipelineError, "wrote no report"):
            self.make(verdicts).judged_gate(self.run_name, 3099, "canary:3000_to3100")

    def test_stale_reports_are_removed_before_evaluating(self):
        pipeline.GATE_ROOT.mkdir(parents=True)
        stale = pipeline.GATE_ROOT / f"{self.run_name}_model_3099_tracking.json"
        stale.write_text(json.dumps({"checks": {"a": True}}))
        verdicts = {"loco": (0, {"a": True}), "tracking": (137, None), "onnx": (0, {"c": True})}
        with self.assertRaisesRegex(pipeline.PipelineError, "wrote no report"):
            self.make(verdicts).judged_gate(self.run_name, 3099, "canary:3000_to3100")
        self.assertFalse(stale.exists())

    def test_a_refused_gate_create_is_a_failed_verdict(self):
        verdicts = {"loco": (0, {"a": True}), "tracking": (0, {"b": True}), "onnx": (0, {"c": True})}
        p = self.make(verdicts)
        p.run = lambda name, cmd, kind, **kw: 2
        self.assertEqual(p.judged_gate(self.run_name, 3099, "canary:3000_to3100"),
                         (False, [], ["stage_gate_create"]))


class TrainingSeedTest(unittest.TestCase):
    """Retrains from the same gated parent use a new seed (forward-lean-v2 eb02a05)."""

    def commands(self, **kwargs) -> list[str]:
        p = make_pipeline()
        cmds = []
        p.gpu_job = lambda need, name, cmd, kind, **kw: cmds.append(cmd)
        p.v12_train("seg", "parent_run", **kwargs)
        return cmds[0]

    def test_seed_flag(self):
        self.assertNotIn("--seed", self.commands())
        self.assertNotIn("--seed", self.commands(seed=42))
        cmd = self.commands(seed=43)
        self.assertEqual(cmd[cmd.index("--seed") + 1], "43")

    def test_trainer_accepts_seed(self):
        text = (REPO / "scripts" / "train_microban_teleop_v12.sh").read_text()
        self.assertIn('--env.seed "${train_seed}" --agent.seed "${train_seed}"', text)
        self.assertNotIn("--env.seed 42", text)

class TrainingSuiteStepTest(unittest.TestCase):
    """Verifier round 3: a HOME branch's own training suite must pass at its HOME."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.p = make_pipeline(Path(self.tmp.name))
        self.p.yaml_sha256 = "y" * 64
        self.p.git = lambda repo, *args, check=True: "c" * 40
        self.runs = []
        self.known = pipeline.known_test_failures()

    def tearDown(self):
        self.tmp.cleanup()

    def fake_run(self, text: str, rc: int = 1):
        def run(name, cmd, kind, *, env=None, allow_fail=False, stdout_path=None, **kwargs):
            self.runs.append((cmd, env))
            stdout_path.write_text(text)
            return rc

        self.p.run = run

    def test_known_failures_pass_and_the_verdict_is_cached(self):
        self.assertIn("tests/test_teleop_v12_deployment.py::test_runtime_validator_requires_cpu_only_pass",
                      self.known)
        lines = [f"FAILED {node} - AssertionError" for node in self.known[:3]]
        self.fake_run("..F\n" + "\n".join(lines) + "\n=== 3 failed, 700 passed in 120.0s ===\n")
        self.p.step_training_suite()
        record = self.p.get("training_suite")
        self.assertTrue(record["passed"])
        self.assertEqual(record["new_failures"], [])
        cmd, env = self.runs[0]
        self.assertEqual(cmd[-1], "tests")
        self.assertEqual(env, {"CUDA_VISIBLE_DEVICES": ""})
        self.p.step_training_suite()  # same commit and YAML: not run again
        self.assertEqual(len(self.runs), 1)

    def test_a_new_failure_stops_the_run(self):
        self.fake_run("F\nFAILED tests/test_walk_export_contract.py::T::test_x - boom\n"
                      "=== 1 failed, 700 passed in 120.0s ===\n")
        with self.assertRaisesRegex(pipeline.PipelineError, "test_walk_export_contract.py::T::test_x") as raised:
            self.p.step_training_suite()
        self.assertEqual(raised.exception.code, pipeline.EXIT_INPUT)
        self.assertFalse(self.p.get("training_suite", "passed"))
        self.fake_run("=== 700 passed in 120.0s ===\n", rc=0)
        self.p.step_training_suite()  # fixed: run again and passes
        self.assertEqual(len(self.runs), 2)

    def test_an_incomplete_run_stops(self):
        self.fake_run("Traceback: collection crashed\n", rc=2)
        with self.assertRaisesRegex(pipeline.PipelineError, "did not complete"):
            self.p.step_training_suite()

    def test_skip_is_a_dry_run_option(self):
        with self.assertRaises(SystemExit):
            pipeline.parse_args(["--robot-repo", "/x", "--robot-branch", "x", "--skip-training-suite"])
        p = make_pipeline(Path(self.tmp.name), "--dry-run", "--dry-run-plumbing", "--skip-training-suite")
        p.run = lambda *a, **k: self.fail("skipped")
        p.step_training_suite()
