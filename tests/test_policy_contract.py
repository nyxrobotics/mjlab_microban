"""microban-policy-1 as the robot (microban runtime-cleanup src/policy_contract.py) reads it."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mjlab_microban import policy_contract as p  # noqa: E402
from mjlab_microban.robot.home_pose import HOME  # noqa: E402
from mjlab_microban.schedules import pico_schedule_record  # noqa: E402

# Every key the robot's parse_policy reads (common, then PICO only).
ROBOT_COMMON_KEYS = {
    "microban_policy_contract", "microban_policy_kind", "microban_recipe", "home_pose", "joint_names",
    "default_joint_pos", "action_joint_names", "action_scale", "action_clip_lower", "action_clip_upper",
    "previous_action_semantics", "base_ang_vel_frame", "control_hz", "observation_schema_json",
    "observation_joint_names", "checkpoint_filename", "checkpoint_iteration", "checkpoint_sha256",
    "gate_status", "gate_report_sha256", "self_test_observations_json", "self_test_actions_json",
}
ROBOT_PICO_KEYS = {
    "pico_walk_checkpoint_sha256", "pico_target_frame", "pico_hand_target_fk_json",
    "pico_foot_target_lower_json", "pico_foot_target_upper_json", "pico_both_feet_target_lower_json",
    "pico_both_feet_target_upper_json", "pico_hand_target_lower_json", "pico_hand_target_upper_json",
    "pico_raw_action_guard_json", "pico_curriculum_json", "pico_active_adapter_columns_json",
}
SHA = "ab" * 32


def _row(kind: str, scale: float = 0.0) -> list[float]:
    """A possible robot state at HOME (gravity of the HOME trunk lean)."""

    row = [0.0] * p.OBSERVATION_WIDTHS[kind]
    row[3:6] = list(HOME.projected_gravity)
    row[6] = scale
    return row


def _metadata(kind: str, rows: int = 8, **overrides) -> dict[str, str]:
    kwargs = dict(
        joint_names=sorted(HOME.joint_pos_rad),
        checkpoint=Path("/runs/x/model_8999.pt"),
        checkpoint_sha256=SHA,
        gate_report_sha256="cd" * 32,
        self_test_observations=[_row(kind, 0.01 * i) for i in range(rows)],
        self_test_actions=[[0.1] * 18 for _ in range(rows)],
        dry_run=False,
    )
    kwargs.update(overrides)
    return p.contract_metadata(kind, **kwargs)


class PolicyContractTest(unittest.TestCase):
    def test_layout_is_the_training_layout(self) -> None:
        from mjlab_microban.tasks.microban_policy_export import (
            MICROBAN_HMD_JOINT_NAMES,
            MICROBAN_TELEOP_ACTION_JOINT_NAMES,
            MICROBAN_TELEOP_OBSERVATION_SCHEMA,
        )

        self.assertEqual(p.ACTION_JOINT_NAMES, MICROBAN_TELEOP_ACTION_JOINT_NAMES)
        self.assertEqual(p.HEAD_JOINTS, MICROBAN_HMD_JOINT_NAMES)
        self.assertEqual(p.OBSERVATION_SCHEMAS["pico"], MICROBAN_TELEOP_OBSERVATION_SCHEMA)
        self.assertEqual(p.OBSERVATION_WIDTHS, {"walk": 63, "getup": 60, "pico": 83})
        self.assertEqual(p.SERVO_TARGET_RANGE_RAD, math.pi)
        self.assertEqual(set(p.joint_ranges()), set(HOME.joint_pos_rad))

    def test_every_kind_carries_the_robot_keys(self) -> None:
        for kind in p.KINDS:
            metadata = _metadata(kind)
            self.assertEqual(set(metadata), ROBOT_COMMON_KEYS, kind)
            self.assertTrue(all(isinstance(value, str) for value in metadata.values()))
            self.assertEqual(metadata["microban_policy_contract"], "microban-policy-1")
            self.assertEqual(metadata["microban_recipe"], p.RECIPES[kind])
            self.assertEqual(metadata["checkpoint_iteration"], "8999")
            self.assertEqual(metadata["previous_action_semantics"], "raw_policy_output")
            self.assertEqual(float(metadata["control_hz"]), 50.0)
            # Full precision, the robot compares with 1e-6.
            defaults = dict(zip(metadata["joint_names"].split(","),
                                (float(v) for v in metadata["default_joint_pos"].split(","))))
            self.assertEqual(defaults, dict(HOME.joint_pos_rad))
            self.assertEqual([float(v) for v in metadata["action_clip_upper"].split(",")], [math.pi] * 18)
            home = json.loads(metadata["home_pose"])
            self.assertEqual(home["joint_pos_rad"], dict(HOME.joint_pos_rad))
            self.assertEqual(home["root_pos_m"], list(HOME.root_pos))
            self.assertEqual(home["root_quat_wxyz"], list(HOME.root_quat_wxyz))
        self.assertEqual(_metadata("walk", dry_run=True)[p.DRY_RUN_METADATA_KEY], "true")

    def test_self_test_rows_must_be_possible_states(self) -> None:
        upside_down = _row("walk")
        upside_down[3:6] = [0.0, 0.0, 2.0]
        fast = _row("walk")
        fast[6 + 18 + 3] = 13.0
        beyond = _row("walk")
        beyond[6 + p.ACTION_JOINT_NAMES.index("right_knee")] = 3.0
        for bad in (upside_down, fast, beyond, _row("walk")[:-1], [math.nan] * 63):
            self.assertIsNotNone(p.physical_row_problem("walk", bad))
            with self.assertRaises(p.PolicyContractError):
                _metadata("walk", self_test_observations=[bad] + [_row("walk")] * 7)
        # A head joint is not checked (the robot checks the 18 body joints).
        head = _row("pico")
        head[6] = 3.0
        self.assertIsNone(p.physical_row_problem("pico", head))
        for count in (7, 65):
            with self.assertRaises(p.PolicyContractError):
                _metadata("walk", rows=count)
        rows = [_row("walk")] * 5 + [fast] * 3
        with self.assertRaises(p.PolicyContractError):
            p.select_self_test_rows("walk", rows)
        picked = p.select_self_test_rows("walk", [_row("walk", 0.001 * i) for i in range(40)] + [fast])
        self.assertEqual(len(picked), 32)
        self.assertEqual(picked[0][6], 0.0)
        self.assertAlmostEqual(picked[-1][6], 0.039)

    def test_pico_keys(self) -> None:
        record = pico_schedule_record()
        self.assertEqual(tuple(record), p.PICO_CURRICULUM_KEYS)
        keys = p.pico_metadata(walk_checkpoint_sha256=SHA, target_frame="robot_trunk_xyz_forward_left_up",
                               hand_target_fk={"revision": "x"}, raw_action_guard=[1.0] * 18,
                               curriculum=record, active_adapter_columns=p.PICO_ADAPTER_COLUMNS,
                               checkpoint_iteration=record["total"] - 1)
        self.assertEqual(set(keys), ROBOT_PICO_KEYS)
        self.assertEqual(json.loads(keys["pico_curriculum_json"]), record)
        self.assertEqual(json.loads(keys["pico_foot_target_upper_json"]), [0.03, 0.03, 0.05] * 2)
        for bad in (dict(raw_action_guard=[0.0] * 18), dict(raw_action_guard=[1e39] * 18),
                    dict(checkpoint_iteration=record["foot_tighten"] - 2),
                    dict(active_adapter_columns=p.PICO_ADAPTER_COLUMNS[:-1])):
            arguments = dict(walk_checkpoint_sha256=SHA, target_frame="f", hand_target_fk={},
                             raw_action_guard=[1.0] * 18, curriculum=record,
                             active_adapter_columns=p.PICO_ADAPTER_COLUMNS,
                             checkpoint_iteration=record["total"] - 1)
            arguments.update(bad)
            with self.assertRaises(p.PolicyContractError):
                p.pico_metadata(**arguments)

    def test_gate_report_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gate = root / "walk_gate.json"
            p.write_gate_report(gate, "walk", SHA, passed=True, dry_run=False, evidence={})
            self.assertEqual(p.gate_report("walk", gate, SHA, dry_run=False), p.sha256_file(gate))
            for kind, sha in (("getup", SHA), ("walk", "ef" * 32)):
                with self.assertRaises(p.PolicyContractError):
                    p.gate_report(kind, gate, sha, dry_run=False)
            with self.assertRaises(p.PolicyContractError):
                p.write_gate_report(gate, "walk", SHA, passed=False, dry_run=False, evidence={})
            p.write_gate_report(gate, "walk", SHA, passed=False, dry_run=True, evidence={})
            with self.assertRaises(p.PolicyContractError):
                p.gate_report("walk", gate, SHA, dry_run=False)
            p.gate_report("walk", gate, SHA, dry_run=True)

            files = {}
            for kind, name in p.POLICY_FILES.items():
                files[kind] = root / name
                files[kind].write_bytes(kind.encode())
            manifest = p.manifest(files=files, checkpoint_sha256={k: SHA for k in p.KINDS},
                                  training_commit="0" * 40, dry_run=True, extra={"created": "now"})
            self.assertEqual(manifest["contract"], "microban-policy-1")
            self.assertEqual(manifest["home_tag"], HOME.tag)
            self.assertIs(manifest["dry_run"], True)
            self.assertEqual(set(manifest["policies"]), {"walk", "getup", "pico"})
            self.assertEqual(manifest["policies"]["pico"],
                             {"file": "pico_teleop.onnx", "sha256": p.sha256_file(files["pico"]),
                              "checkpoint_sha256": SHA})

    def test_pipeline_reads_the_robot_constants(self) -> None:
        from mjlab_microban.pipeline.steps import robot_contract_constants

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "policy_contract.py"
            source.write_text(
                'POLICY_CONTRACT = "microban-policy-1"\n'
                f"RECIPES: Mapping[str, str] = {dict(p.RECIPES)!r}\n"
                "KINDS = tuple(RECIPES)\n"
            )
            self.assertEqual(robot_contract_constants(source),
                             {"POLICY_CONTRACT": p.POLICY_CONTRACT, "RECIPES": dict(p.RECIPES)})


if __name__ == "__main__":
    unittest.main()
