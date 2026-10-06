"""Trunk pitch is just a value of config/home_pose.yaml.

The two HOMEs with trained artifacts reproduce the branches they were trained
on exactly:

* the centered HOME (config/home_pose.yaml, trunk vertical) reproduces
  mjlab_microban track-centered-home-clip (5b5a9d0);
* the forward-lean HOME (tests/fixtures/home_pose_forward_lean.yaml: what
  ``config/balance_home_pose.py --trunk-pitch-deg 10 --write`` writes into a copy
  of the centered YAML, with name/label edited) reproduces forward-lean-v2
  with its pose-release final rescue and its push-replay mixes (forward-lean-v2
  cd0ea78, which contains lean-final-rescue fc1c313..3fc519b and 005f55c; its
  walking and get-up tasks are those of forward-lean-centered-home).

"Reproduces" means every module constant, every HOME-derived function result
and the repr of every registered task's env / play / RL config and runner of
the reference branch (tests/fixtures/home_equivalence/*.json, sha256 per key,
recorded with tests/home_equivalence.py) is equal here.  The only allowed
differences are new names this tree adds, the names of the modules, constants
and tasks it deleted (never trained by the HOME pipeline), the values listed in
INTENDED_CHANGES (each with its reason), and at the centered HOME the new
command-config fields left at their no-op defaults (``trunk_pitch=0.0``,
``lf_rb_probability=0.9``) and the walking runner that stamps checkpoints with
their HOME (a subclass of mjlab's).  The 0.040 m hand-RMS profiles of the
pose-release 10000 boundary / 10100 canary (forward-lean-v2 e3271de, ec67f1e)
and the packager's required 10000 / 10100 gates (7ceb280) exist at every HOME
but the centered one, so the centered profile tables equal the reference too;
ForwardLeanOnlyTestsTest runs their tests at the forward-lean HOME.

``MJLAB_MICROBAN_EXPORT_EQUIVALENCE=1`` also re-exports the walking and get-up
checkpoints of both HOMEs (CPU, a few minutes) from the git objects of
forward-lean-v2 and requires byte-identical ONNX files.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures"
LEAN_YAML = FIXTURES / "home_pose_forward_lean.yaml"
CENTERED_YAML = FIXTURES / "home_pose_centered.yaml"
REFERENCES = {
    "centered": FIXTURES / "home_equivalence" / "centered_home_track-centered-home-clip_5b5a9d0.json",
    "forward_lean": FIXTURES / "home_equivalence" / "forward_lean_home_forward-lean-v2_cd0ea78.json",
}
sys.path.insert(0, str(REPO_ROOT / "tests"))
import home_equivalence  # noqa: E402


def _environment(yaml_path: Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONHASHSEED": "0",
            "CUDA_VISIBLE_DEVICES": "",
            "MJLAB_MICROBAN_HOME_POSE_YAML": str(yaml_path),
            "PYTHONPATH": os.pathsep.join(
                [str(REPO_ROOT / "src"), *filter(None, [os.environ.get("PYTHONPATH")])]
            ),
        }
    )
    return environment


def _run_python(yaml_path: Path, code: str) -> str:
    completed = subprocess.run(
        [sys.executable, "-c", code],
        env=_environment(yaml_path),
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stderr[-3000:])
    return completed.stdout.strip().splitlines()[-1]


# Task ids of the reference branches that the HOME pipeline never trained and
# this tree no longer registers (their env / play / RL / runner keys are absent).
UNREGISTERED_TASKS = frozenset({
    "Mjlab-SafeVelocity-Microban",
    "Mjlab-Getup-Microban-V42",
    "Mjlab-Getup-Microban-Sym",
    "Mjlab-Getup-Microban-NearHome5deg",
    "Mjlab-Getup-Microban-Tipping",
    "Mjlab-Tracking-Microban",
    "Mjlab-Teleop-Microban",
    "Mjlab-Teleop-Upright-Fullbody-Microban",
    "Mjlab-Teleop-V12-Preview-Microban",
})


# Values this tree changed on purpose, key -> reason.  Every entry was checked
# by diffing the full dumps (tests/home_equivalence.py dump) before and after
# the change: an RL config entry differs only by the removed runner options.
_V12_TASKS = (
    "Mjlab-Teleop-V12-Microban",
    "Mjlab-Teleop-V12-HandPoseRelease-Microban",
    "Mjlab-Teleop-V12-Corner-Rescue-Microban",
    "Mjlab-Teleop-V12-HandPoseRelease-Corner-Rescue-Microban",
    "Mjlab-Teleop-V12-Final-Rescue-Microban",
    "Mjlab-Teleop-V12-HandPoseRelease-Final-Rescue-Microban",
)
INTENDED_CHANGES = {
    **{
        f"task:{task}:rl": "runner options of the deleted consumer, preview and "
        "deadline-fallback modes removed (checkpoint_consumer_mode, "
        "simulation_preview_mode, deadline_fallback_resume*)"
        for task in _V12_TASKS
    },
    "const:mjlab_microban.scripts.evaluate_teleop_v12_tracking.TRACKING_PROFILES": (
        "deadline-fallback profiles removed"
    ),
    "const:mjlab_microban.scripts.teleop_v12_stage.TRACKING_PROFILES": (
        "deadline-fallback profiles removed"
    ),
    "const:mjlab_microban.scripts.export_teleop_v12_deployment.SUPPORTED_FINAL_TRACKING_PROFILES": (
        "deadline-fallback final profile removed"
    ),
    "const:mjlab_microban.scripts.export_teleop_v12_deployment._BOUNDARY_GATE_INHERITED_INFO_KEYS": (
        "left/right-order migration marker removed (no checkpoint migration)"
    ),
}


def _deleted_here(key: str) -> bool:
    """A reference key whose module, constant or task this tree deleted (not a regression).

    Keys of modules and constants that still exist, and of tasks that are still
    registered, must be present and equal (or listed in INTENDED_CHANGES).
    """

    kind, _, rest = key.partition(":")
    if kind == "task":
        return rest.rsplit(":", 1)[0] in UNREGISTERED_TASKS
    if kind == "import":
        module = rest
    elif kind == "const":
        module = rest.rsplit(".", 1)[0]
    elif kind == "call" and rest.startswith("mjlab_microban."):
        module = rest.rsplit(".", 1)[0]
    else:
        return False
    relative = Path(*module.split("."))
    source = REPO_ROOT / "src"
    if not (source / relative.with_suffix(".py")).is_file() and not (source / relative).is_dir():
        return True
    # A constant the module (which imports cleanly, see _compare) no longer
    # defines or imports: the dump records every one it has.
    return kind == "const"


class HomeEquivalenceTest(unittest.TestCase):
    """Side-by-side with the reference branches (subprocess per HOME, ~40 s each)."""

    def _compare(self, yaml_path: Path, reference: Path, *, centered: bool) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dump.json"
            completed = subprocess.run(
                [sys.executable, str(REPO_ROOT / "tests" / "home_equivalence.py"), "dump",
                 str(output), str(REPO_ROOT)],
                env=_environment(yaml_path),
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=1800,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr[-3000:])
            values = json.loads(output.read_text())
        if centered:
            values = {
                key: home_equivalence.without_centered_default_fields(value)
                for key, value in values.items()
            }
        allowed = {}
        if centered:
            # The walking runner stamps checkpoints with the training HOME (a
            # subclass of mjlab's VelocityOnPolicyRunner, as on the
            # forward-lean branches); the centered branch used mjlab's own.
            allowed["task:Mjlab-Velocity-Microban:runner"] = (
                "mjlab_microban.tasks.microban_velocity_runner.MicrobanVelocityOnPolicyRunner"
            )
        for key, value in allowed.items():
            self.assertEqual(values.get(key), value)
        import_errors = sorted(key for key, value in values.items() if key.startswith("import:"))
        self.assertEqual(import_errors, [], "modules that failed to import")
        current = home_equivalence.digest(values)
        expected = json.loads(reference.read_text())
        missing = sorted(key for key in set(expected) - set(current) if not _deleted_here(key))
        different = sorted(
            key
            for key in expected
            if key in current
            and current[key] != expected[key]
            and key not in allowed
            and key not in INTENDED_CHANGES
        )
        self.assertEqual(missing, [], "names of the reference branch missing here")
        self.assertEqual(different, [], "values that differ from the reference branch")
        # Every registered task of the reference exists and matches (checked
        # above), except the ones this tree no longer registers.
        self.assertGreaterEqual(
            sum(key.startswith("task:") for key in current),
            sum(key.startswith("task:") and not _deleted_here(key) for key in expected),
        )

    def test_centered_home_reproduces_track_centered_home_clip(self):
        self._compare(CENTERED_YAML, REFERENCES["centered"], centered=True)

    def test_forward_lean_home_reproduces_forward_lean_v2(self):
        self._compare(LEAN_YAML, REFERENCES["forward_lean"], centered=False)


class ForwardLeanHomeValuesTest(unittest.TestCase):
    """The forward-lean YAML gives the forward-lean branches' literal values."""

    def test_home_values_and_contract_strings(self):
        result = json.loads(
            _run_python(
                LEAN_YAML,
                "import json\n"
                "from mjlab_microban.robot.home_pose import HOME\n"
                "from mjlab_microban.robot import home_contracts as c\n"
                "print(json.dumps({'tag': HOME.tag, 'hash': HOME.joint_hash,"
                " 'rad': dict(HOME.joint_pos_rad), 'deg_in': dict(HOME.input_joint_pos_deg),"
                " 'root': HOME.root_pos, 'quat': HOME.root_quat_wxyz, 'g': HOME.projected_gravity,"
                " 'head': HOME.head_standing_height_m, 'feet': HOME.feet_lateral_m,"
                " 'robot': c.robot_contract_strings(), 'getup_legacy': c.GETUP_LEGACY_STAMP,"
                " 'switch': c.V12_POSE_RELEASE_SWITCH_PARENT_SHA256,"
                " 'unstamped': c.ACCEPTS_UNSTAMPED_WALK_CHECKPOINTS}))",
            )
        )
        self.assertEqual(result["tag"], "forward_lean_home")
        self.assertEqual(result["hash"], "481503d292")
        # The YAML holds the canonical 12-decimal values; HOME publishes the
        # forward-lean branch's unrounded ones bit for bit.
        self.assertEqual(result["deg_in"]["left_hip_pitch"], -14.166561199931)
        self.assertEqual(result["rad"]["left_hip_pitch"], float(np.deg2rad(-14.166561199931119)))
        self.assertEqual(result["rad"]["right_ankle_pitch"], float(np.deg2rad(4.127976841869204)))
        self.assertEqual(result["root"], [0.0, 0.0, 0.170430569776402])
        pitch = float(np.deg2rad(10.0))
        self.assertEqual(
            result["quat"],
            [float(np.cos(pitch / 2.0)), 0.0, float(np.sin(pitch / 2.0)), 0.0],
        )
        self.assertEqual(result["g"], [math.sin(pitch), 0.0, -math.cos(pitch)])
        self.assertEqual((result["head"], result["feet"]), (0.2953, 0.0941))
        self.assertEqual(
            result["robot"],
            {
                "walk_contract_version": "v4_forward_lean_home_servo_range",
                "getup_contract_version": "v6",
                "getup_checkpoint_stamp": "v6",
                "v12_home_pose_revision": (
                    "forward_lean10_hip_minus14p166561199931_ankle_plus4p127976841869_"
                    "shoulder_zero_v6"
                ),
                "v12_recipe_revision": (
                    "forward_lean_home_velocity_source_staged_mask_reachable_fk_elbow_minus10_"
                    "raw_prev_action_servo_range_pi_home_levelled_targets_level_hmd_"
                    "receiver_box_hands_v17"
                ),
                "v12_hand_pose_release_recipe_revision": (
                    "forward_lean_home_velocity_source_staged_mask_reachable_fk_elbow_minus10_"
                    "raw_prev_action_servo_range_pi_home_levelled_targets_level_hmd_"
                    "receiver_box_hands_active_hand_arm_pose_release_v18"
                ),
                "v12_packager_revision": (
                    "microban_teleop_v12_final_deployment_packager_v7_forward_lean_home_servo_range"
                ),
                "v12_target_frame": "robot_home_levelled_trunk_xyz_forward_left_up",
            },
        )
        self.assertIsNone(result["getup_legacy"])
        self.assertIsNone(result["switch"])
        self.assertFalse(result["unstamped"])

    def test_fixture_is_the_balance_tool_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "home_pose.yaml"
            shutil.copyfile(CENTERED_YAML, path)
            completed = subprocess.run(
                [sys.executable, str(REPO_ROOT / "config" / "balance_home_pose.py"), "--yaml",
                 str(path), "--trunk-pitch-deg", "10", "--write", "--no-training-check"],
                cwd=REPO_ROOT, capture_output=True, text=True, timeout=600, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            written = path.read_text().splitlines()
        fixture = LEAN_YAML.read_text().splitlines()
        # Same file except the fixture's header comment and name/label.
        values = lambda lines: [  # noqa: E731
            line for line in lines
            if line and not line.startswith("#") and not line.startswith(("name:", "label:"))
        ]
        self.assertEqual(values(written), values(fixture))


class RequiredBoundaryClocksTest(unittest.TestCase):
    """The pipeline's HOME check reports the packager's required boundary clocks."""

    def test_home_check_agrees_with_the_packager(self):
        code = (
            "import json, sys\n"
            "sys.path.insert(0, 'scripts/home_pipeline')\n"
            "import home_check\n"
            "from mjlab_microban.robot.home_pose import HOME\n"
            "from mjlab_microban.scripts import export_teleop_v12_deployment as d\n"
            "print(json.dumps([home_check.check(HOME.path)['v12_required_boundary_gate_clocks'],"
            " list(d.POSE_RELEASE_REQUIRED_BOUNDARY_COMPLETED_UPDATES)]))"
        )
        for yaml_path, expected in ((CENTERED_YAML, []), (LEAN_YAML, [10000, 10100])):
            with self.subTest(yaml=yaml_path.name):
                self.assertEqual(json.loads(_run_python(yaml_path, code)), [expected, expected])


class DerivedHomeStringsTest(unittest.TestCase):
    """Any other HOME gets "<label>_<hash>" strings of its trunk's mechanism."""

    def test_pitched_and_vertical_derived_strings(self):
        with tempfile.TemporaryDirectory() as directory:
            for pitch, knee in ((5.0, 0.0), (0.0, 15.0)):
                path = Path(directory) / f"home_{pitch}_{knee}.yaml"
                shutil.copyfile(CENTERED_YAML, path)
                from mjlab_microban.robot.home_pose import rewrite_home_pose_yaml

                rewrite_home_pose_yaml(path, joint_pos_deg={"left_knee": knee, "right_knee": knee})
                completed = subprocess.run(
                    [sys.executable, str(REPO_ROOT / "config" / "balance_home_pose.py"),
                     "--yaml", str(path), "--trunk-pitch-deg", str(pitch), "--write",
                     "--no-training-check"],
                    cwd=REPO_ROOT, capture_output=True, text=True, timeout=600, check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                result = json.loads(
                    _run_python(
                        path,
                        "import json\n"
                        "from mjlab_microban.robot.home_pose import HOME\n"
                        "from mjlab_microban.robot import home_contracts as c\n"
                        "print(json.dumps({'tag': HOME.tag, 'walk': c.WALK_CONTRACT_VERSION,"
                        " 'getup': c.GETUP_CONTRACT_VERSION, 'legacy': c.GETUP_LEGACY_STAMP,"
                        " 'recipe': c.V12_RECIPE_REVISION, 'packager': c.V12_PACKAGER_REVISION,"
                        " 'corner': c.V12_CORNER_RESCUE_RECIPE_REVISION,"
                        " 'pr_final': [c.V12_PR_FINAL_RESCUE_MARKER_REVISION,"
                        " c.V12_PR_FINAL_RESCUE_SAMPLER_REVISION],"
                        " 'switch': c.V12_POSE_RELEASE_SWITCH_PARENT_SHA256}))",
                    )
                )
                tag = result["tag"]
                self.assertRegex(tag, r"^centered_home_[0-9a-f]{10}$")
                version = "v3" if pitch == 0.0 else "v4"
                self.assertEqual(result["walk"], f"{version}_{tag}_servo_range")
                self.assertEqual(result["getup"], f"{'v5' if pitch == 0.0 else 'v6'}_{tag}")
                self.assertIsNone(result["legacy"])
                self.assertIsNone(result["switch"])
                self.assertTrue(result["recipe"].startswith(f"{tag}_velocity_source_"))
                self.assertEqual(
                    result["recipe"].endswith("_v11"), pitch == 0.0, result["recipe"]
                )
                self.assertIn(tag, result["packager"])
                self.assertIn(tag, result["corner"])
                # The pose-release final rescue's strings carry the HOME too.
                self.assertTrue(all(tag in value for value in result["pr_final"]), result["pr_final"])
                self.assertEqual("home_levelled" in result["pr_final"][1], pitch != 0.0)


class ForwardLeanOnlyTestsTest(unittest.TestCase):
    """Tests skipped at the centered HOME pass at the forward-lean HOME.

    The 0.040 m hand-RMS boundary allowance and the required pose-release
    10000 / 10100 boundary gates exist at every HOME but the centered one.
    """

    NODES = (
        "tests/test_teleop_v12_stage.py::TeleopV12StageTest::"
        "test_hand_rms_40mm_is_the_pose_release_10000_profile_only",
        "tests/test_teleop_v12_stage.py::TeleopV12StageTest::"
        "test_hand_rms_40mm_is_the_pose_release_10100_canary_profile_only",
        "tests/test_teleop_v12_stage.py::TeleopV12StageTest::"
        "test_pose_release_10000_gate_uses_and_records_the_hand_rms_allowance",
        "tests/test_teleop_v12_deployment.py::"
        "test_boundary_gates_record_the_10000_hand_rms_allowance",
        "tests/test_teleop_v12_deployment.py::"
        "test_boundary_gates_record_the_10100_canary_hand_rms_allowance",
        "tests/test_teleop_v12_deployment.py::"
        "test_boundary_gates_reject_foreign_or_nonboundary_gates[missing_canary]",
        "tests/test_teleop_v12_deployment.py::"
        "test_boundary_gate_controls_pass_on_the_exact_ancestry",
    )

    def test_forward_lean_only_tests_pass_at_the_forward_lean_home(self):
        try:
            import pytest  # noqa: F401
        except ImportError:
            self.skipTest("pytest is not installed")
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rs", *self.NODES],
            env=_environment(LEAN_YAML), cwd=REPO_ROOT, capture_output=True, text=True,
            timeout=1800, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout[-3000:] + completed.stderr[-2000:])
        self.assertIn(f"{len(self.NODES)} passed", completed.stdout)
        self.assertNotIn("skipped", completed.stdout.splitlines()[-1])

    def test_pose_release_final_rescue_tests_pass_at_the_forward_lean_home(self):
        """forward-lean-v2's final rescue (fc1c313..cd0ea78) with its published strings."""

        try:
            import pytest  # noqa: F401
        except ImportError:
            self.skipTest("pytest is not installed")
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rs",
             "tests/test_teleop_v12_hand_pose_release_final_rescue.py"],
            env=_environment(LEAN_YAML), cwd=REPO_ROOT, capture_output=True, text=True,
            timeout=1800, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout[-3000:] + completed.stderr[-2000:])
        self.assertNotIn("skipped", completed.stdout.splitlines()[-1])
        marker = _run_python(LEAN_YAML, (
            "from mjlab_microban.tasks import microban_teleop_v12_hand_pose_release_final_rescue as m;"
            "print(m.MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_MARKER_REVISION + ' ' + "
            "m.MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_SAMPLER_REVISION)"))
        self.assertEqual(marker.split(), [
            "recorded_pose_release_model14900_failed_final_scenarios_replay_99_updates_forward_lean_v1",
            "episode_shared_twist_foot_hand_failed_scenario_replay_home_levelled_pose_release_v1",
        ])


class HomePinnedTestsTest(unittest.TestCase):
    """The tests pinned to one published HOME pass at that HOME, from any checkout.

    A HOME branch's own config/home_pose.yaml runs its pinned tests directly
    (tests/home_cases.py); the other published HOME's pinned tests are run here
    in a subprocess at its fixture YAML, so the centered branch still checks
    the forward-lean mechanisms and the reverse.  Any other HOME runs both.
    """

    def _run_marked(self, yaml_path: Path, marker: str) -> None:
        try:
            import pytest  # noqa: F401
        except ImportError:
            self.skipTest("pytest is not installed")
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rs", "-m", marker,
             "tests"],
            env=_environment(yaml_path), cwd=REPO_ROOT, capture_output=True, text=True,
            timeout=3000, check=False,
        )
        summary = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
        self.assertEqual(completed.returncode, 0, completed.stdout[-4000:] + completed.stderr[-2000:])
        self.assertIn(" passed", summary)
        self.assertNotIn("skipped", summary, completed.stdout[-4000:])

    def test_forward_lean_pinned_tests_pass_at_the_forward_lean_home(self):
        import home_cases

        if home_cases.AT_FORWARD_LEAN_HOME:
            self.skipTest("this checkout is the forward-lean HOME: its pinned tests ran directly")
        self._run_marked(LEAN_YAML, "forward_lean_home_pinned")

    def test_centered_pinned_tests_pass_at_the_centered_home(self):
        import home_cases

        if home_cases.AT_CENTERED_HOME:
            self.skipTest("this checkout is the centered HOME: its pinned tests ran directly")
        self._run_marked(CENTERED_YAML, "centered_home_pinned")


def _git_show(commit: str, path: str, destination: Path) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", f"{commit}:{path}"],
        capture_output=True, check=False,
    )
    if completed.returncode != 0:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(completed.stdout)
    return True


@unittest.skipUnless(
    os.environ.get("MJLAB_MICROBAN_EXPORT_EQUIVALENCE") == "1",
    "set MJLAB_MICROBAN_EXPORT_EQUIVALENCE=1 (CPU re-export of four checkpoints, minutes)",
)
class ExportEquivalenceTest(unittest.TestCase):
    """Re-exporting each HOME's walking / get-up checkpoint is byte-identical."""

    SOURCE_COMMIT = "5442e8838535c48e3e36bbb57ed016614e29ff3b"  # forward-lean-v2
    CASES = (
        # (HOME yaml, exporter, run directory, checkpoint, published ONNX, its sha256)
        (LEAN_YAML, "export_walk_onnx", "logs/rsl_rl/mjlab_microban_velocity/2026-10-05_06-14-02_lean_walk_cont2",
         "model_29000.pt", "artifacts/walk_v4_forward_lean_home_servo_cont2_29000.onnx",
         "b33cd9ea7dbebbfe4543c0bb616a54d9ba713ded1ffdda0891b79dad09e2c1d2"),
        (LEAN_YAML, "export_getup_onnx", "logs/rsl_rl/mjlab_microban_getup/2026-10-05_04-37-04_lean_s5_push",
         "model_21495.pt", "artifacts/getup_v6_lean_home_push_21495.onnx",
         "ce6cdc0489b451b32a35c3a791830ca123cadb308f3b2f4eaf1019c3721678bf"),
        (CENTERED_YAML, "export_walk_onnx", "logs/rsl_rl/mjlab_microban_velocity/2026-10-04_03-05-49_chome_servo_walk_cont",
         "model_20000.pt", "artifacts/walk_v3_centered_home_servo_cont_20000.onnx",
         "c9cdd8527704046d5c8058fc63fc3ef148716d659509666d0dc844b64c18fa40"),
        (CENTERED_YAML, "export_getup_onnx", "logs/rsl_rl/mjlab_microban_getup/2026-10-04_01-35-01_servo_s5_push",
         "model_21495.pt", "artifacts/getup_v5_chome_servo_push_21495.onnx",
         "80cd7ddb13066f6563b07927e85e6fb7916149d4275218b9ad80ee58f5b08cfa"),
    )

    def test_reexports_are_byte_identical(self):
        for yaml_path, exporter, run, checkpoint, published, published_sha in self.CASES:
            with self.subTest(published=published), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run_dir = root / Path(run).name
                ok = _git_show(self.SOURCE_COMMIT, f"{run}/{checkpoint}", run_dir / checkpoint)
                ok = ok and _git_show(self.SOURCE_COMMIT, f"{run}/params/env.yaml", run_dir / "params" / "env.yaml")
                ok = ok and _git_show(self.SOURCE_COMMIT, f"{run}/params/agent.yaml", run_dir / "params" / "agent.yaml")
                ok = ok and _git_show(self.SOURCE_COMMIT, published, root / "published.onnx")
                if not ok:
                    self.skipTest(f"git objects of {self.SOURCE_COMMIT[:7]} are not available")
                self.assertEqual(
                    hashlib.sha256((root / "published.onnx").read_bytes()).hexdigest(), published_sha
                )
                output = root / "export.onnx"
                completed = subprocess.run(
                    [sys.executable, "-m", f"mjlab_microban.scripts.{exporter}", "--checkpoint",
                     str(run_dir / checkpoint), "--output", str(output), "--device", "cpu"],
                    env=_environment(yaml_path), cwd=REPO_ROOT, capture_output=True, text=True,
                    timeout=1800, check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr[-3000:])
                self.assertEqual(hashlib.sha256(output.read_bytes()).hexdigest(), published_sha)


if __name__ == "__main__":
    unittest.main()
