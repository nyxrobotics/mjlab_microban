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
