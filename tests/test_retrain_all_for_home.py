"""Pure parts of scripts/retrain_all_for_home.py and scripts/home_pipeline/robot_pins.py."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
import unittest
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


CENTERED = {
    "tag": "centered_home",
    "root_pos_m": [0.0, 0.0, 0.170554885633559],
    "joint_pos_deg": {"left_hip_pitch": 1.198384259489, "left_ankle_pitch": -1.198384259489,
                      "left_knee": 0.0, "left_elbow": -20.0},
    "joint_pos_rad": {"left_hip_pitch": math.radians(1.198384259489),
                      "left_ankle_pitch": -math.radians(1.198384259489), "left_knee": 0.0},
}
KNEE15 = {
    "tag": "centered_home_e7afd03eb9",
    "root_pos_m": [0.0, 0.0, 0.167836754864191],
    "joint_pos_deg": {"left_hip_pitch": -6.513901179137, "left_ankle_pitch": -8.486098820863,
                      "left_knee": 15.0, "left_elbow": -20.0},
    "joint_pos_rad": {"left_hip_pitch": math.radians(-6.513901179137),
                      "left_ankle_pitch": math.radians(-8.486098820863), "left_knee": math.radians(15.0)},
}


class RobotPinsTest(unittest.TestCase):
    def test_home_tokens_follow_the_new_home(self):
        mapping, ambiguous = robot_pins.build_home_token_map(CENTERED, KNEE15)
        self.assertEqual(ambiguous, [])
        text = (
            '    "left_hip_pitch": 1.198384259489,\n'
            "        self.assertEqual(HOME_ROOT_POS_Z_M, 0.170554885633559)\n"
            '    "centered_home_hip_plus1p198384259489_ankle_minus1p198384259489_shoulder_zero_v5"\n'
            '        "v3_centered_home_servo_range"\n'
            "    def test_centered_home_values(self):\n"
            "        x = -math.radians(1.198384259489)\n"
            "        y = math.radians(1.198384259489)\n"
            "        elbow = -20.0\n"
        )
        new, count = robot_pins.substitute_home_tokens(text, mapping)
        self.assertIn('"left_hip_pitch": -6.513901179137,', new)
        self.assertIn("0.167836754864191", new)
        self.assertIn("centered_home_e7afd03eb9_hip_minus6p513901179137_ankle_minus8p486098820863", new)
        self.assertIn('"v3_centered_home_e7afd03eb9_servo_range"', new)
        self.assertIn("def test_centered_home_values", new)  # test names keep their wording
        self.assertIn("x = math.radians(-8.486098820863)", new)
        self.assertIn("y = math.radians(-6.513901179137)", new)
        self.assertIn("elbow = -20.0", new)
        self.assertGreater(count, 5)
        same, none = robot_pins.build_home_token_map(CENTERED, CENTERED)
        self.assertEqual((same, none), ({}, []))
        self.assertEqual(robot_pins.substitute_home_tokens(text, same), (text, 0))

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
