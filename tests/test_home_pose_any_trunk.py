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

"Reproduces" means every HOME-derived value of the reference branch -- the
constants of mjlab_microban.robot (HOME, its FK-derived quantities and
contract strings) and the HOME-derived function results (hand FK metadata and
evaluation offsets, the HOME stamps and markers, the ONNX parity corpus;
tests/fixtures/home_equivalence/*.json, sha256 per key, recorded with
tests/home_equivalence.py) -- is equal here.  The training recipes (task
configs, rewards, schedules, gates) were rebuilt in stage C (2026-10-07) and
are not compared.

``MJLAB_MICROBAN_EXPORT_EQUIVALENCE=1`` also re-exports the walking and get-up
checkpoints of both HOMEs (CPU, a few minutes) from the git objects of
forward-lean-v2 and requires ONNX files byte-identical to the published ones
once the policy-contract keys (docs/policies.md) are removed.
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


def _deleted_here(key: str) -> bool:
    """A reference key whose module or constant this tree deleted (not a regression).

    Keys of modules and constants that still exist must be present and equal.
    """

    kind, _, rest = key.partition(":")
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
        expected = {
            key: digest
            for key, digest in json.loads(reference.read_text()).items()
            if key.startswith(("const:mjlab_microban.robot.", "call:", "import:mjlab_microban.robot"))
        }
        missing = sorted(key for key in set(expected) - set(current) if not _deleted_here(key))
        different = sorted(
            key
            for key in expected
            if key in current
            and current[key] != expected[key]
            and key not in allowed
        )
        self.assertEqual(missing, [], "names of the reference branch missing here")
        self.assertEqual(different, [], "values that differ from the reference branch")

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
                    "receiver_box_hands_active_hand_arm_pose_release_twist_ratio_one_run_"
                    "warmup1000_total9000_v1"
                ),
                "v12_packager_revision": (
                    "microban_pico_packager_one_run_v1_forward_lean_home_servo_range"
                ),
                "v12_target_frame": "robot_home_levelled_trunk_xyz_forward_left_up",
            },
        )
        self.assertIsNone(result["getup_legacy"])
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
                        " 'recipe': c.V12_RECIPE_REVISION, 'packager': c.V12_PACKAGER_REVISION}))",
                    )
                )
                tag = result["tag"]
                self.assertRegex(tag, r"^centered_home_[0-9a-f]{10}$")
                version = "v3" if pitch == 0.0 else "v4"
                self.assertEqual(result["walk"], f"{version}_{tag}_servo_range")
                self.assertEqual(result["getup"], f"{'v5' if pitch == 0.0 else 'v6'}_{tag}")
                self.assertIsNone(result["legacy"])
                self.assertTrue(result["recipe"].startswith(f"{tag}_velocity_source_"))
                self.assertEqual(
                    result["recipe"].endswith("_v11"), pitch == 0.0, result["recipe"]
                )
                self.assertIn(tag, result["packager"])


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
                # The policy-contract keys (2026-10-07) are the only addition:
                # without them the file is byte-identical to the published one.
                self.assertEqual(hashlib.sha256(_without_contract_keys(output)).hexdigest(), published_sha)


def _without_contract_keys(path: Path) -> bytes:
    import onnx

    from mjlab_microban.policy_contract import contract_metadata

    model = onnx.load(str(path))
    keys = set(contract_metadata())
    kept = [prop for prop in model.metadata_props if prop.key not in keys]
    if len(kept) != len(model.metadata_props) - len(keys):
        raise AssertionError("the export lacks some policy-contract keys")
    del model.metadata_props[:]
    model.metadata_props.extend(kept)
    with tempfile.TemporaryDirectory() as directory:
        stripped = Path(directory) / "stripped.onnx"
        onnx.save(model, str(stripped))
        return stripped.read_bytes()


if __name__ == "__main__":
    unittest.main()
