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
    def test_rescue_mixes_and_dry_run_options(self):
        args = make_pipeline().args
        self.assertEqual(args.pr_corner_rescue_mixes, ["lf60", "lf90", "lf72", "lf65"])
        self.assertEqual(args.v12_9999_attempts, 2)
        args = make_pipeline(None, "--pr-corner-rescue-mixes", "lf72, lf90").args
        self.assertEqual(args.pr_corner_rescue_mixes, ["lf72", "lf90"])
        for bad in (["--pr-corner-rescue-mixes", "lf61"], ["--v12-9999-attempts", "0"], ["--dry-run-plumbing"],
                    ["--dry-run-simulate-failures"], ["--serial-gpu", "--dry-run-plumbing"]):
            with self.assertRaises(SystemExit):
                with unittest.mock.patch("sys.stderr"):
                    make_pipeline(None, *bad)
        p = make_pipeline(None, "--dry-run", "--dry-run-plumbing")
        self.assertTrue(p.plumbing)
        self.assertFalse(make_pipeline(None, "--dry-run").plumbing)

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


class SimulatedFailureTest(unittest.TestCase):
    def schedule(self, mode: str) -> dict[str, bool]:
        with tempfile.TemporaryDirectory() as d:
            p = make_pipeline(Path(d), "--dry-run", "--dry-run-plumbing", "--dry-run-simulate-failures",
                              "--dry-run-simulate-9999", mode)
            keys = ["canary:3000_to3100", "canary:3000_to3100", "9999:a1", "rescue:a1r1_lf60",
                    "rescue:a1r2_lf90", "9999:a2", "rescue:a2r1_lf60", "stage:3100_to7000"]
            return [p.simulated_failure(k) for k in keys]

    def test_routes(self):
        self.assertEqual(self.schedule("retrain"), [True, False, True, True, True, False, True, False])
        self.assertEqual(self.schedule("rescue"), [True, False, True, True, False, False, True, False])
        self.assertEqual(self.schedule("stop"), [True, False, True, True, True, True, True, False])

    def test_real_runs_never_simulate(self):
        p = make_pipeline()
        self.assertFalse(p.simulated_failure("9999:a1"))


class Boundary9999Test(unittest.TestCase):
    """The automatic 10000-boundary escalation: gate -> corner rescues -> retrain -> stop."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.v12 = root / "v12"
        self.v12.mkdir()
        self.original_v12 = pipeline.V12_EXP
        pipeline.V12_EXP = self.v12
        self.p = make_pipeline(root / "state")
        (root / "state").mkdir()
        self.p.prefix = "home_x"
        self.p.check_recipe = lambda run, end: None
        self.calls: list = []

    def tearDown(self):
        pipeline.V12_EXP = self.original_v12
        self.tmp.cleanup()

    def make_run(self, name: str, end: int = 9999) -> str:
        (self.v12 / name).mkdir(exist_ok=True)
        (self.v12 / name / f"model_{end}.pt").write_bytes(name.encode())
        return name

    def wire(self, gates: dict[str, bool], rescues: dict[int, str | None]):
        def judged_gate(run, end, key):
            self.calls.append(("gate", run, key))
            return gates[key], ([] if gates[key] else ["hand_tracking_rms"]), []

        def corner_rescues(run, attempt):
            self.calls.append(("rescues", run, attempt))
            return rescues.get(attempt)

        def attempt_9999(parent, attempt):
            self.calls.append(("retrain", parent, attempt))
            return self.make_run(f"r_a{attempt}")

        self.p.judged_gate, self.p.corner_rescues, self.p.attempt_9999 = judged_gate, corner_rescues, attempt_9999

    def test_passing_gate(self):
        self.wire({"9999:a1": True}, {})
        first = self.make_run("first")
        self.assertEqual(self.p.boundary_9999("p7099", first), "first")
        self.assertEqual(self.p.get("pico", "b9999", "passed", "kind"), "segment")

    def test_rescue_then_resume_skips_everything(self):
        self.wire({"9999:a1": False}, {1: self.make_run("rescue_lf90")})
        self.assertEqual(self.p.boundary_9999("p7099", self.make_run("first")), "rescue_lf90")
        self.assertEqual(self.p.get("pico", "b9999", "passed", "kind"), "corner_rescue")
        self.calls.clear()
        self.assertEqual(self.p.boundary_9999("p7099", "first"), "rescue_lf90")
        self.assertEqual(self.calls, [])

    def test_retrain_from_the_gated_7099_after_every_rescue_failed(self):
        self.wire({"9999:a1": False, "9999:a2": True}, {})
        self.assertEqual(self.p.boundary_9999("p7099", self.make_run("first")), "r_a2")
        self.assertEqual(self.calls, [("gate", "first", "9999:a1"), ("rescues", "first", 1),
                                      ("retrain", "p7099", 2), ("gate", "r_a2", "9999:a2")])
        self.assertEqual(self.p.get("pico", "b9999", "passed", "attempt"), 2)

    def test_stops_after_the_last_attempt(self):
        self.wire({"9999:a1": False, "9999:a2": False}, {})
        with self.assertRaisesRegex(pipeline.PipelineError, "9999 boundary failed after 2 attempt"):
            self.p.boundary_9999("p7099", self.make_run("first"))
        self.assertEqual([c[0] for c in self.calls], ["gate", "rescues", "retrain", "gate", "rescues"])
        self.assertIsNone(self.p.get("pico", "b9999", "passed"))


class CornerRescueMixesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.v12 = root / "v12"
        self.v12.mkdir()
        self.original_v12 = pipeline.V12_EXP
        pipeline.V12_EXP = self.v12
        (root / "state").mkdir()
        self.p = make_pipeline(root / "state")
        self.p.prefix = "home_x"
        (self.v12 / "first").mkdir()
        (self.v12 / "first" / "model_9900.pt").write_bytes(b"parent")
        (self.v12 / "first" / "model_9999.pt").write_bytes(b"first")
        (root / "state" / "pico").mkdir()
        (root / "state" / "pico" / "first_model_9900_strict_tracking.json").write_text("{}")
        self.p.check_recipe = lambda run, end: None
        self.p.ensure_committed_tree = lambda: self.trained.append("commit")
        self.trained: list = []
        self.validated: list = []

        def gpu_job(need, name, cmd, kind, **kwargs):
            seg = cmd[cmd.index("--agent.run-name") + 1]
            self.trained.append(cmd[cmd.index("--mix") + 1])
            (self.v12 / f"2026-01-01_00-00-0{len(self.trained)}_{seg}").mkdir()
            (self.v12 / f"2026-01-01_00-00-0{len(self.trained)}_{seg}" / "model_9999.pt").write_bytes(seg.encode())

        self.p.gpu_job = gpu_job

    def tearDown(self):
        pipeline.V12_EXP = self.original_v12
        self.tmp.cleanup()

    def validator(self, rc: int):
        def capture(cmd, **kwargs):
            self.validated.append(kwargs["env"]["MICROBAN_V12_PR_CORNER_RESCUE_MIX"])
            return unittest.mock.Mock(returncode=rc, stdout="", stderr="ValueError: refused")
        self.p.capture = capture

    def test_mixes_in_order_until_one_passes_and_resume(self):
        self.validator(0)
        self.p.judged_gate = lambda run, end, key: (key == "rescue:a1r2_lf90", [], [])
        rescue = self.p.corner_rescues("first", 1)
        self.assertTrue(rescue.endswith("_home_x_v12_pr_rescue_a1r2_lf90_9901_to10000"))
        self.assertEqual(self.trained, ["commit", "lf60", "commit", "lf90"])
        self.assertEqual(self.validated, ["lf60", "lf90"])
        self.assertFalse(self.p.get("pico", "b9999", "rescues", "a1r1_lf60", "passed"))
        # A rerun skips the failed lf60 and reuses the trained lf90 run.
        self.trained.clear()
        self.validated.clear()
        self.assertEqual(self.p.corner_rescues("first", 1), rescue)
        self.assertEqual(self.trained, [])
        self.assertEqual(self.validated, ["lf90"])

    def test_every_mix_is_tried_and_repeats_are_new_runs(self):
        self.validator(0)
        self.p.judged_gate = lambda run, end, key: (False, ["hand_tracking_rms"], [])
        self.assertIsNone(self.p.corner_rescues("first", 1))
        self.assertEqual([m for m in self.trained if m != "commit"], ["lf60", "lf90", "lf72", "lf65"])
        runs = {v["run"] for v in self.p.get("pico", "b9999", "rescues").values()}
        self.assertEqual(len(runs), 4)

    def test_a_refused_parent_skips_the_rescues(self):
        self.validator(1)
        self.p.judged_gate = lambda run, end, key: self.fail("no gate without a rescue")
        self.assertIsNone(self.p.corner_rescues("first", 1))
        self.assertEqual(self.trained, [])
        self.assertIn("refused", self.p.get("pico", "b9999", "attempts", "1", "rescue_parent", "refused"))

    def test_no_model_9900(self):
        (self.v12 / "first" / "model_9900.pt").unlink()
        self.assertIsNone(self.p.corner_rescues("first", 1))


class CommittedTreeTest(unittest.TestCase):
    def test_only_the_home_yaml_is_committed_for_the_rescue(self):
        import subprocess

        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            git = lambda *a: subprocess.run(["git", "-C", d, *a], check=True, capture_output=True)  # noqa: E731
            git("init", "-q")
            git("config", "user.email", "t@example.com")
            git("config", "user.name", "t")
            (repo / "config").mkdir()
            yaml_path = repo / "config" / "home_pose.yaml"
            yaml_path.write_text("a: 1\n")
            git("add", ".")
            git("commit", "-qm", "base")
            originals = pipeline.REPO, pipeline.HOME_YAML
            pipeline.REPO, pipeline.HOME_YAML = repo, yaml_path
            try:
                p = make_pipeline(repo / "state")
                p.home = {"tag": "knee15", "joint_hash": "abc", "trunk_pitch_deg": 0.0}
                yaml_path.write_text("a: 2\n")
                p.yaml_sha256 = pipeline.sha256(yaml_path)
                (repo / "state").mkdir()
                (repo / ".gitignore").write_text("state/\n")
                git("add", ".gitignore")
                git("commit", "-qm", "ignore")
                p.ensure_committed_tree()
                log = subprocess.run(["git", "-C", d, "log", "-1", "--format=%s"], capture_output=True,
                                     text=True).stdout
                self.assertIn("Set HOME knee15", log)
                p.ensure_committed_tree()  # clean: nothing to do
                (repo / "other.py").write_text("x\n")
                with self.assertRaisesRegex(pipeline.PipelineError, "other.py"):
                    p.ensure_committed_tree()
            finally:
                pipeline.REPO, pipeline.HOME_YAML = originals


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

    def test_stamped_rescue_carries_the_pose_release_corner_lineage(self):
        from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
            MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION as PR,
        )
        from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release_lineage import (
            HAND_POSE_RELEASE_LINEAGE_FRESH_CORNER_RESCUE,
            hand_pose_release_lineage,
        )

        for mix in ("lf60", "lf72", "lf90"):
            infos = self.tools.corner_rescue_infos({"microban_teleop_recipe_revision": PR},
                                                   parent_sha256="a" * 64, report_sha256="b" * 64, mix=mix,
                                                   provenance={})
            self.assertEqual(hand_pose_release_lineage(infos, iteration=10099),
                             HAND_POSE_RELEASE_LINEAGE_FRESH_CORNER_RESCUE)
            self.assertTrue(infos["dry_run_synthetic_corner_rescue"]["not_deployable"])
            with self.assertRaises(ValueError):  # a rescue of a rescue
                self.tools.corner_rescue_infos(infos, parent_sha256="a" * 64, report_sha256="b" * 64, mix=mix,
                                               provenance={})


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

    def test_canary_retry_and_9999_attempt_use_new_seeds(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            original = pipeline.V12_EXP
            pipeline.V12_EXP = root / "v12"
            pipeline.V12_EXP.mkdir()
            try:
                p = make_pipeline(root / "state")
                (root / "state").mkdir()
                p.prefix = "home_x"
                seeds = []

                def v12_train(seg, prev, lift_to=None, seed=None):
                    seeds.append((seg, seed))
                    (pipeline.V12_EXP / f"2026-01-01_00-00-09_{seg}").mkdir()
                    (pipeline.V12_EXP / f"2026-01-01_00-00-09_{seg}" / "model_9999.pt").write_bytes(b"x")

                p.v12_train = v12_train
                p.attempt_9999("p7099", 2)
                self.assertEqual(seeds, [("home_x_v12_7100_to10000_a2", 43)])
            finally:
                pipeline.V12_EXP = original
        source = (REPO / "scripts" / "retrain_all_for_home.py").read_text()
        self.assertIn("lift_to=end - 3 if self.dry else None, seed=V12_TRAIN_SEED + 1)", source)


class Boundary15000Test(unittest.TestCase):
    """The automatic 15000-boundary escalation: gate -> final rescues -> retrain (new seed) -> stop."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.v12 = root / "v12"
        self.v12.mkdir()
        self.original_v12 = pipeline.V12_EXP
        pipeline.V12_EXP = self.v12
        (root / "state").mkdir()
        self.p = make_pipeline(root / "state")
        self.p.prefix = "home_x"
        self.p.check_recipe = lambda run, end: None
        self.calls: list = []

    def tearDown(self):
        pipeline.V12_EXP = self.original_v12
        self.tmp.cleanup()

    def make_run(self, name: str) -> str:
        (self.v12 / name).mkdir(exist_ok=True)
        (self.v12 / name / "model_14999.pt").write_bytes(name.encode())
        return name

    def wire(self, gates: dict[str, bool], rescues: dict[int, str | None]):
        def judged_gate(run, end, key):
            self.calls.append(("gate", run, key))
            return gates[key], ([] if gates[key] else ["twist_directional_response"]), []

        def final_rescues(run, attempt, seed, trk, other):
            self.calls.append(("rescues", run, attempt, seed))
            return rescues.get(attempt)

        def attempt_15000(parent, attempt):
            self.calls.append(("retrain", parent, attempt))
            return self.make_run(f"r_a{attempt}")

        self.p.judged_gate, self.p.final_rescues = judged_gate, final_rescues
        self.p.attempt_15000 = attempt_15000

    def test_passing_gate(self):
        self.wire({"15000:a1": True}, {})
        self.assertEqual(self.p.boundary_15000("p10099", self.make_run("first")), "first")
        self.assertEqual(self.p.get("pico", "b15000", "passed", "kind"), "segment")

    def test_rescue_then_resume_skips_everything(self):
        self.wire({"15000:a1": False}, {1: self.make_run("frescue_pr_v2")})
        self.assertEqual(self.p.boundary_15000("p10099", self.make_run("first")), "frescue_pr_v2")
        self.assertEqual(self.p.get("pico", "b15000", "passed", "kind"), "final_rescue")
        self.calls.clear()
        self.assertEqual(self.p.boundary_15000("p10099", "first"), "frescue_pr_v2")
        self.assertEqual(self.calls, [])

    def test_retrain_with_the_next_seed_after_every_rescue_failed(self):
        self.wire({"15000:a1": False, "15000:a2": True}, {})
        self.assertEqual(self.p.boundary_15000("p10099", self.make_run("first")), "r_a2")
        self.assertEqual(self.calls, [("gate", "first", "15000:a1"), ("rescues", "first", 1, 42),
                                      ("retrain", "p10099", 2), ("gate", "r_a2", "15000:a2")])
        self.assertEqual(self.p.get("pico", "b15000", "attempts", "2", "seed"), 43)

    def test_stops_after_the_last_attempt(self):
        self.wire({"15000:a1": False, "15000:a2": False}, {})
        with self.assertRaisesRegex(pipeline.PipelineError, "15000 boundary failed after 2 attempt"):
            self.p.boundary_15000("p10099", self.make_run("first"))
        self.assertEqual([c[0] for c in self.calls], ["gate", "rescues", "retrain", "gate", "rescues"])
        self.assertIsNone(self.p.get("pico", "b15000", "passed"))

    def test_attempt_trains_from_the_gated_10099_with_its_seed(self):
        p = make_pipeline(self.p.state_dir)
        p.prefix = "home_x"
        seeds = []

        def v12_train(seg, prev, lift_to=None, seed=None):
            seeds.append((seg, prev, seed))
            self.make_run(f"2026-01-01_00-00-09_{seg}")

        p.v12_train = v12_train
        p.attempt_15000("p10099", 3)
        self.assertEqual(seeds, [("home_x_v12_10100_to15000_a3", "p10099", 44)])


class FinalRescueMixesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.v12 = root / "v12"
        self.v12.mkdir()
        self.originals = pipeline.V12_EXP, pipeline.GATE_ROOT
        pipeline.V12_EXP, pipeline.GATE_ROOT = self.v12, root / "gates"
        (root / "state").mkdir()
        self.p = make_pipeline(root / "state")
        self.p.prefix = "home_x"
        (self.v12 / "first").mkdir()
        (self.v12 / "first" / "model_14900.pt").write_bytes(b"parent")
        (self.v12 / "first" / "model_14999.pt").write_bytes(b"first")
        self.p.check_recipe = lambda run, end: None
        self.p.ensure_committed_tree = lambda boundary=9999: self.trained.append(f"commit{boundary}")
        self.trained: list = []
        self.validated: list = []

        def gpu_job(need, name, cmd, kind, **kwargs):
            self.assertEqual(cmd[0], "scripts/train_microban_teleop_v12_hand_pose_release_final_rescue.sh")
            self.assertEqual(Path(cmd[2]).name, "first_model_14999_tracking.json")
            seg = cmd[cmd.index("--agent.run-name") + 1]
            self.trained.append((cmd[cmd.index("--mix") + 1], cmd[cmd.index("--seed") + 1]))
            run = self.v12 / f"2026-01-01_00-00-0{len(self.trained)}_{seg}"
            run.mkdir()
            (run / "model_14999.pt").write_bytes(seg.encode())

        self.p.gpu_job = gpu_job

    def tearDown(self):
        pipeline.V12_EXP, pipeline.GATE_ROOT = self.originals
        self.tmp.cleanup()

    def validator(self, accepted: set[str]):
        def capture(cmd, **kwargs):
            mix = cmd[cmd.index("--mix") + 1]
            self.validated.append((mix, cmd[cmd.index("--seed") + 1]))
            ok = mix in accepted
            return unittest.mock.Mock(returncode=0 if ok else 1, stdout="",
                                      stderr="" if ok else "ValueError: mix does not replay every failed scenario")
        self.p.capture = capture

    def test_refused_mixes_are_skipped_and_the_first_passing_rescue_wins(self):
        self.validator({"pr_v2", "pr_v4"})
        self.p.judged_gate = lambda run, end, key: (key == "frescue:a1r4_pr_v4", [], [])
        rescue = self.p.final_rescues("first", 1, 42, ["twist_directional_response"], [])
        self.assertTrue(rescue.endswith("_home_x_v12_pr_final_rescue_a1r4_pr_v4_14901_to15000"))
        self.assertEqual(self.validated, [("pr_v1", "42"), ("pr_v2", "42"), ("pr_v3", "42"), ("pr_v4", "42")])
        self.assertEqual(self.trained, ["commit15000", ("pr_v2", "42"), "commit15000", ("pr_v4", "42")])
        self.assertFalse(self.p.get("pico", "b15000", "rescues", "a1r1_pr_v1", "passed"))
        # A rerun skips refused and failed mixes and reuses the trained rescue.
        self.trained.clear()
        self.validated.clear()
        self.assertEqual(self.p.final_rescues("first", 1, 42, ["twist_directional_response"], []), rescue)
        self.assertEqual(self.trained, [])
        self.assertEqual(self.validated, [("pr_v4", "42")])

    def test_unrescuable_gates_are_not_rescued(self):
        self.validator({"pr_v1"})
        self.p.judged_gate = lambda run, end, key: self.fail("no rescue gate")
        for trk, other in ((["twist_directional_response"], ["falls"]), (["hmd_motion"], []), ([], ["onnx"])):
            self.assertIsNone(self.p.final_rescues("first", 1, 42, trk, other))
        self.assertEqual(self.validated, [])

    def test_no_model_14900(self):
        (self.v12 / "first" / "model_14900.pt").unlink()
        self.validator({"pr_v1"})
        self.assertIsNone(self.p.final_rescues("first", 1, 42, ["actual_soft_limits"], []))
        self.assertEqual(self.validated, [])


class FinalRescueRegistrationTest(unittest.TestCase):
    def test_pipeline_mixes_match_the_registered_mixes(self):
        from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release_final_rescue import (
            MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_MIXES as MIXES,
            MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_RESCUABLE_CHECKS as RESCUABLE,
        )

        self.assertEqual(set(pipeline.V12_FINAL_RESCUE_MIXES), set(MIXES))
        self.assertEqual(pipeline.V12_FINAL_RESCUABLE_CHECKS, set(RESCUABLE))

    def test_arguments(self):
        args = make_pipeline().args
        self.assertEqual(args.pr_final_rescue_mixes, ["pr_v1", "pr_v2", "pr_v3", "pr_v4"])
        self.assertEqual((args.v12_15000_attempts, args.dry_run_simulate_15000), (2, "pass"))
        for bad in (["--pr-final-rescue-mixes", "pr_v9"], ["--v12-15000-attempts", "0"],
                    ["--dry-run-simulate-15000", "rescue"]):
            with self.assertRaises(SystemExit):
                with unittest.mock.patch("sys.stderr"):
                    make_pipeline(None, *bad)

    def test_simulated_15000_routes(self):
        keys = ["15000:a1", "frescue:a1r1_pr_v1", "frescue:a1r2_pr_v2", "15000:a2", "frescue:a2r1_pr_v1"]
        expected = {"pass": [False] * 5, "rescue": [True, True, False, False, True],
                    "retrain": [True, True, True, False, True], "stop": [True] * 5}
        for mode, want in expected.items():
            with tempfile.TemporaryDirectory() as d:
                p = make_pipeline(Path(d), "--dry-run", "--dry-run-plumbing", "--dry-run-simulate-failures",
                                  "--dry-run-simulate-15000", mode)
                self.assertEqual([p.simulated_failure(k) for k in keys], want, mode)

    def test_stamped_final_rescue_carries_the_final_lineage(self):
        sys.path.insert(0, str(REPO / "scripts" / "home_pipeline"))
        import dry_run_tools

        from mjlab_microban.tasks.microban_teleop_v12_actor import teleop_v12_active_adapter_columns
        from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
            MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION as PR,
        )
        from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release_lineage import (
            HAND_POSE_RELEASE_LINEAGE_FRESH_CORNER_FINAL_RESCUE,
            HAND_POSE_RELEASE_LINEAGE_FRESH_FINAL_RESCUE,
            hand_pose_release_lineage,
        )

        base = {"microban_teleop_recipe_revision": PR, "env_state": {"common_step_counter": 15000 * 24},
                "active_actor_columns_at_save": list(teleop_v12_active_adapter_columns(15000 * 24))}
        corner = dry_run_tools.corner_rescue_infos(base, parent_sha256="a" * 64, report_sha256="b" * 64,
                                                   mix="lf72", provenance={})
        for infos, lineage in ((base, HAND_POSE_RELEASE_LINEAGE_FRESH_FINAL_RESCUE),
                               (corner, HAND_POSE_RELEASE_LINEAGE_FRESH_CORNER_FINAL_RESCUE)):
            for mix in ("pr_v1", "pr_v4"):
                out = dry_run_tools.final_rescue_infos(infos, parent_sha256="c" * 64, failed_sha256="d" * 64,
                                                       report_sha256="e" * 64, mix=mix, seed=43, provenance={})
                self.assertEqual(hand_pose_release_lineage(out, iteration=14999), lineage)
                with self.assertRaises(ValueError):  # a rescue of a rescue
                    dry_run_tools.final_rescue_infos(out, parent_sha256="c" * 64, failed_sha256="d" * 64,
                                                     report_sha256="e" * 64, mix=mix, seed=43, provenance={})
