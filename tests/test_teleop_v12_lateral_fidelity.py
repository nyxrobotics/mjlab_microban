"""Lateral-fidelity variant of the forward-lean pose-release recipe."""

from __future__ import annotations

import math
import os
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from mjlab_microban.robot.microban_constants import HOME_TRUNK_PITCH_RAD
from mjlab_microban.scripts.export_teleop_v12_deployment import (
    _BOUNDARY_GATE_INHERITED_INFO_KEYS,
    _lateral_fidelity_metadata,
)
from mjlab_microban.scripts.teleop_v12_stage import lateral_fidelity_gate_marker
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION,
)
from mjlab_microban.tasks.microban_teleop_v12_final_rescue import (
    MICROBAN_TELEOP_V12_FINAL_RESCUE_INFO_KEY,
)
from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release import (
    make_microban_teleop_v12_hand_pose_release_env_cfg,
)
from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release_lineage import (
    HAND_POSE_RELEASE_LINEAGE_FRESH,
    MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_SWITCH_INFO_KEY,
    hand_pose_release_lineage,
)
from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release_runner import (
    MicrobanTeleopV12HandPoseReleaseOnPolicyRunner,
)
from mjlab_microban.tasks.microban_teleop_v12_home_pose import (
    TELEOP_V12_HOME_POSE_INFO_KEY,
    teleop_v12_home_pose_marker,
)
from mjlab_microban.tasks.microban_teleop_v12_lateral_fidelity import (
    MICROBAN_TELEOP_V12_LATERAL_FIDELITY_INFO_KEY,
    MICROBAN_TELEOP_V12_LATERAL_FIDELITY_REWARD_NAME,
    MICROBAN_TELEOP_V12_LATERAL_FIDELITY_TASK_ID,
    MICROBAN_TELEOP_V12_LATERAL_FIDELITY_WEIGHT_ENV,
    apply_lateral_fidelity,
    installed_lateral_fidelity_weight,
    lateral_fidelity_marker,
    make_microban_teleop_v12_lateral_fidelity_env_cfg,
    mixed_command_lateral_deficit,
    mixed_command_lateral_deficit_l1,
    selected_lateral_fidelity_weight_label,
    validate_lateral_fidelity_infos,
    validate_lateral_fidelity_marker,
    validate_lateral_fidelity_parent_payload,
)

_PARENT_CHECKPOINT = (
    "repo://logs/rsl_rl/mjlab_microban_teleop_v12/"
    "2026-10-05_14-02-37_lean_v12_pr_7000_to7100/model_7099.pt"
)
_PARENT_GATE = (
    "repo://artifacts/teleop_v12_gates/"
    "2026-10-05_14-02-37_lean_v12_pr_7000_to7100_model_7099_gate.json"
)
_ROOT = Path(__file__).resolve().parents[1]


def _deficit(command, velocity, min_abs=0.05):
    return mixed_command_lateral_deficit(
        torch.tensor(command, dtype=torch.float32),
        torch.tensor(velocity, dtype=torch.float32),
        min_abs,
    )


def _marker(weight: float = -8.0, **overrides) -> dict:
    values = {
        "parent_checkpoint_path": _PARENT_CHECKPOINT,
        "parent_checkpoint_sha256": "a" * 64,
        "parent_stage_gate_path": _PARENT_GATE,
        "parent_stage_gate_sha256": "b" * 64,
        "weight": weight,
        **overrides,
    }
    return lateral_fidelity_marker(**values)


def _infos(**extra) -> dict:
    return {
        "microban_teleop_recipe_revision": (
            MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION
        ),
        TELEOP_V12_HOME_POSE_INFO_KEY: teleop_v12_home_pose_marker(),
        **extra,
    }


class LateralDeficitTermTest(unittest.TestCase):
    def test_lean_and_centered_behaviours_on_the_diagnosed_command(self) -> None:
        # (0.7, 0.3) command; lean 14999 (0.27, -0.03), centered (0.20, +0.066).
        values = _deficit([[0.7, 0.3], [0.7, 0.3]], [[0.27, -0.03], [0.20, 0.066]])
        ratio = 0.3 / 0.7
        self.assertAlmostEqual(float(values[0]), ratio * 0.27 + 0.03, places=6)
        self.assertAlmostEqual(float(values[1]), ratio * 0.20 - 0.066, places=6)
        self.assertGreater(float(values[0]) * 8.0, 1.1)
        self.assertLess(float(values[1]) * 8.0, 0.2)

    def test_mirrored_and_backward_commands_use_the_commanded_signs(self) -> None:
        base = float(_deficit([[0.7, 0.3]], [[0.27, -0.03]])[0])
        mirrored = float(_deficit([[0.7, -0.3]], [[0.27, 0.03]])[0])
        self.assertAlmostEqual(base, mirrored, places=6)
        # Backward-left: progress is -v_x; 0.2 m/s backward expects 0.1 left.
        backward = _deficit(
            [[-0.6, 0.3], [-0.6, -0.3], [-0.6, 0.3]],
            [[-0.2, 0.0], [-0.2, 0.0], [-0.2, 0.1]],
        )
        self.assertAlmostEqual(float(backward[0]), 0.1, places=6)
        self.assertAlmostEqual(float(backward[1]), 0.1, places=6)
        self.assertAlmostEqual(float(backward[2]), 0.0, places=6)
        # Moving against the forward command expects no lateral progress.
        self.assertAlmostEqual(float(_deficit([[0.6, 0.3]], [[-0.2, 0.0]])[0]), 0.0)

    def test_extra_lateral_speed_is_never_penalised_and_target_is_capped(self) -> None:
        values = _deficit(
            [[0.7, 0.3], [0.7, 0.3], [0.1, 0.3]],
            [[0.2, 0.3], [0.0, 0.5], [0.5, 0.1]],
        )
        self.assertEqual(float(values[0]), 0.0)
        self.assertEqual(float(values[1]), 0.0)
        # r * p_x = 1.5 is capped at |c_y| = 0.3.
        self.assertAlmostEqual(float(values[2]), 0.2, places=6)

    def test_pure_axis_and_small_components_are_inactive(self) -> None:
        values = _deficit(
            [[0.7, 0.0], [0.0, 0.3], [0.04, 0.3], [0.7, -0.049], [0.0, 0.0]],
            [[0.7, -0.5], [0.5, -0.3], [0.5, -0.3], [0.7, 0.5], [0.3, -0.3]],
        )
        self.assertTrue(torch.equal(values, torch.zeros(5)))
        boundary = _deficit([[0.05, 0.05]], [[0.05, -0.05]])
        self.assertAlmostEqual(float(boundary[0]), 0.1, places=6)

    def test_env_term_reads_the_home_levelled_frame(self) -> None:
        # Robot at HOME (trunk pitched forward by HOME_TRUNK_PITCH_RAD) moving
        # (0.27, -0.03) in the world: the levelled frame reads it unchanged.
        half = 0.5 * HOME_TRUNK_PITCH_RAD
        quat = torch.tensor([[math.cos(half), 0.0, math.sin(half), 0.0]])
        data = SimpleNamespace(
            root_link_quat_w=quat,
            root_link_lin_vel_w=torch.tensor([[0.27, -0.03, 0.0]]),
            root_link_lin_vel_b=None,
        )
        command = torch.tensor([[0.7, 0.3, 1.5]])
        env = SimpleNamespace(
            scene={"robot": SimpleNamespace(data=data)},
            command_manager=SimpleNamespace(get_command=lambda name: command),
        )
        value = mixed_command_lateral_deficit_l1(env)
        self.assertAlmostEqual(float(value[0]), 0.3 / 0.7 * 0.27 + 0.03, places=5)


class LateralFidelityEnvTest(unittest.TestCase):
    def test_only_the_new_term_is_added(self) -> None:
        base = make_microban_teleop_v12_hand_pose_release_env_cfg()
        cfg = make_microban_teleop_v12_lateral_fidelity_env_cfg(weight_label="8")
        self.assertEqual(
            set(cfg.rewards) - set(base.rewards),
            {MICROBAN_TELEOP_V12_LATERAL_FIDELITY_REWARD_NAME},
        )
        for name, term in base.rewards.items():
            self.assertEqual(cfg.rewards[name].weight, term.weight, name)
            self.assertIs(cfg.rewards[name].func, term.func, name)
        term = cfg.rewards[MICROBAN_TELEOP_V12_LATERAL_FIDELITY_REWARD_NAME]
        self.assertEqual(term.weight, -8.0)
        self.assertIs(term.func, mixed_command_lateral_deficit_l1)
        self.assertEqual(term.params["trunk_pitch"], HOME_TRUNK_PITCH_RAD)
        self.assertEqual(
            base.curriculum["staged_curriculum"].params["stages"].__len__(),
            cfg.curriculum["staged_curriculum"].params["stages"].__len__(),
        )
        v2 = make_microban_teleop_v12_lateral_fidelity_env_cfg(weight_label="16")
        self.assertEqual(
            v2.rewards[MICROBAN_TELEOP_V12_LATERAL_FIDELITY_REWARD_NAME].weight, -16.0
        )
        with self.assertRaises(ValueError):
            apply_lateral_fidelity(base, "12")
        with self.assertRaises(ValueError):
            apply_lateral_fidelity(cfg, "8")

    def test_weight_label_comes_from_the_launcher_environment(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(MICROBAN_TELEOP_V12_LATERAL_FIDELITY_WEIGHT_ENV, None)
            self.assertEqual(selected_lateral_fidelity_weight_label(), "8")
        with mock.patch.dict(
            os.environ, {MICROBAN_TELEOP_V12_LATERAL_FIDELITY_WEIGHT_ENV: "16"}
        ):
            self.assertEqual(selected_lateral_fidelity_weight_label(), "16")
        with mock.patch.dict(
            os.environ, {MICROBAN_TELEOP_V12_LATERAL_FIDELITY_WEIGHT_ENV: "4"}
        ):
            with self.assertRaises(ValueError):
                selected_lateral_fidelity_weight_label()

    def test_task_is_registered(self) -> None:
        import mjlab_microban.tasks  # noqa: F401
        from mjlab.tasks.registry import list_tasks

        self.assertIn(MICROBAN_TELEOP_V12_LATERAL_FIDELITY_TASK_ID, list_tasks())


class LateralFidelityMarkerTest(unittest.TestCase):
    def test_marker_is_exact(self) -> None:
        marker = _marker()
        self.assertEqual(validate_lateral_fidelity_marker(marker), marker)
        self.assertEqual(marker["reward_weight"], -8.0)
        self.assertEqual(marker["parent_iteration"], 7099)
        self.assertEqual(
            marker["recipe_revision"],
            MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION,
        )
        for drift in (
            {"reward_weight": -10.0},
            {"min_abs_command_m_s": 0.1},
            {"parent_iteration": 9999},
            {"extra": 1},
            {"reason": "x"},
        ):
            with self.assertRaises(ValueError, msg=str(drift)):
                validate_lateral_fidelity_marker({**marker, **drift})
        with self.assertRaises(ValueError):
            _marker(parent_checkpoint_path=_PARENT_CHECKPOINT.replace("7099", "9999"))
        with self.assertRaises(ValueError):
            _marker(parent_stage_gate_path="/abs/gate.json")
        with self.assertRaises(ValueError):
            _marker(weight=-4.0)

    def test_lineage_keeps_fresh_chain_and_refuses_other_extensions(self) -> None:
        infos = _infos(**{MICROBAN_TELEOP_V12_LATERAL_FIDELITY_INFO_KEY: _marker()})
        self.assertEqual(
            hand_pose_release_lineage(infos, iteration=9999, verify_parent=False),
            HAND_POSE_RELEASE_LINEAGE_FRESH,
        )
        with self.assertRaisesRegex(ValueError, "clock"):
            hand_pose_release_lineage(infos, iteration=7099, verify_parent=False)
        bad = dict(infos)
        bad[MICROBAN_TELEOP_V12_LATERAL_FIDELITY_INFO_KEY] = {
            **_marker(),
            "reward_weight": -1.0,
        }
        with self.assertRaises(ValueError):
            hand_pose_release_lineage(bad, verify_parent=False)
        for key in (
            MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_SWITCH_INFO_KEY,
            MICROBAN_TELEOP_V12_FINAL_RESCUE_INFO_KEY,
        ):
            with self.assertRaises(ValueError, msg=key):
                hand_pose_release_lineage(
                    {**infos, key: {"x": 1}}, verify_parent=False
                )
        self.assertIsNone(validate_lateral_fidelity_infos(_infos(), iteration=9999))

    def test_parent_payload_is_an_unmarked_fresh_model_7099(self) -> None:
        good = {
            "iter": 7099,
            "infos": _infos(env_state={"common_step_counter": 7100 * 24}),
        }
        validate_lateral_fidelity_parent_payload(good)
        for payload, message in (
            ({**good, "iter": 9900}, "7099"),
            (
                {**good, "infos": {**good["infos"], "env_state": {"common_step_counter": 1}}},
                "clock",
            ),
            (
                {
                    **good,
                    "infos": {
                        **good["infos"],
                        MICROBAN_TELEOP_V12_LATERAL_FIDELITY_INFO_KEY: _marker(),
                    },
                },
                "already",
            ),
            (
                {
                    **good,
                    "infos": {
                        **good["infos"],
                        "microban_teleop_recipe_revision": "other",
                    },
                },
                "pose-release",
            ),
        ):
            with self.assertRaisesRegex(ValueError, message):
                validate_lateral_fidelity_parent_payload(payload)

    def test_gate_and_package_record_the_marker(self) -> None:
        marker = _marker()
        infos = _infos(**{MICROBAN_TELEOP_V12_LATERAL_FIDELITY_INFO_KEY: marker})
        self.assertEqual(
            lateral_fidelity_gate_marker(infos),
            (MICROBAN_TELEOP_V12_LATERAL_FIDELITY_INFO_KEY, marker),
        )
        self.assertIsNone(lateral_fidelity_gate_marker(_infos()))
        gate = {MICROBAN_TELEOP_V12_LATERAL_FIDELITY_INFO_KEY: marker}
        metadata = _lateral_fidelity_metadata(gate, infos)
        self.assertEqual(
            metadata["v12_lateral_fidelity_revision"],
            "hand_pose_release_lateral_fidelity_v1",
        )
        self.assertEqual(metadata["v12_lateral_fidelity_reward_weight"], "-8.0")
        self.assertEqual(_lateral_fidelity_metadata({}, _infos()), {})
        with self.assertRaises(ValueError):
            _lateral_fidelity_metadata({}, infos)
        with self.assertRaises(ValueError):
            _lateral_fidelity_metadata(gate, _infos())
        self.assertIn(
            MICROBAN_TELEOP_V12_LATERAL_FIDELITY_INFO_KEY,
            _BOUNDARY_GATE_INHERITED_INFO_KEYS,
        )


class _RewardManager:
    def __init__(self, weight: float | None) -> None:
        self.weight = weight
        cfg = make_microban_teleop_v12_lateral_fidelity_env_cfg(weight_label="8")
        self.term = cfg.rewards[MICROBAN_TELEOP_V12_LATERAL_FIDELITY_REWARD_NAME]
        if weight is not None:
            self.term.weight = weight

    @property
    def active_terms(self):
        return (
            [] if self.weight is None else [MICROBAN_TELEOP_V12_LATERAL_FIDELITY_REWARD_NAME]
        )

    def get_term_cfg(self, name):
        assert name == MICROBAN_TELEOP_V12_LATERAL_FIDELITY_REWARD_NAME
        return self.term


def _runner(weight: float | None, marker: dict | None):
    env = SimpleNamespace(unwrapped=SimpleNamespace(reward_manager=_RewardManager(weight)))
    return SimpleNamespace(env=env, teleop_v12_lateral_fidelity=marker)


class LateralFidelityRunnerEnvTest(unittest.TestCase):
    check = staticmethod(
        MicrobanTeleopV12HandPoseReleaseOnPolicyRunner._assert_lateral_fidelity_environment
    )

    def test_term_and_marker_must_agree(self) -> None:
        self.check(_runner(None, None))
        self.check(_runner(-8.0, _marker()))
        self.check(_runner(-16.0, _marker(weight=-16.0)))
        with self.assertRaisesRegex(RuntimeError, "carrying its marker"):
            self.check(_runner(-8.0, None))
        with self.assertRaisesRegex(RuntimeError, "own task"):
            self.check(_runner(None, _marker()))
        with self.assertRaisesRegex(RuntimeError, "differs"):
            self.check(_runner(-16.0, _marker()))
        self.assertEqual(installed_lateral_fidelity_weight(_runner(-8.0, None).env), -8.0)


class LateralFidelityLauncherTest(unittest.TestCase):
    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(_ROOT / "scripts" / "train_microban_teleop_v12.sh"), *args],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_option_requires_pose_release_and_resume(self) -> None:
        result = self._run("resume", "some_run", "--lateral-fidelity")
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires --hand-pose-release", result.stderr)
        result = self._run("resume", "some_run", "--lateral-fidelity-weight", "16")
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires --lateral-fidelity", result.stderr)
        result = self._run(
            "resume", "some_run", "--hand-pose-release", "--lateral-fidelity",
            "--lateral-fidelity-weight", "12",
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("must be 8 or 16", result.stderr)
        help_text = self._run("--help").stdout
        self.assertIn("--lateral-fidelity", help_text)


if __name__ == "__main__":
    unittest.main()
