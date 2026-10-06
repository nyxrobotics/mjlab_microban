"""Active-hand arm pose-release recipe: reward math and recipe gating."""

from __future__ import annotations

import unittest
from unittest import mock
from copy import deepcopy
from types import SimpleNamespace

import torch
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp as velocity_mdp

from mjlab_microban.tasks.microban_teleop_v12_corner_rescue import (
    validate_corner_rescue_canonical_lineage,
)
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION,
    MICROBAN_TELEOP_V12_RECIPE_REVISION,
)
from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release import (
    active_hand_arm_released_posture,
)
from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release import (
    MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_SWITCH_INFO_KEY,
)
from mjlab_microban.tasks import (
    microban_teleop_v12_hand_pose_release_lineage as lineage_module,
)
from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release_lineage import (
    HAND_POSE_RELEASE_LINEAGE_EXPERIMENTAL_SWITCH,
    HAND_POSE_RELEASE_LINEAGE_FRESH,
    HAND_POSE_RELEASE_LINEAGE_RELEASE_SWITCH,
    HAND_POSE_RELEASE_RECIPE_SWITCH_PARENT_CHECKPOINT_SHA256,
    MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_EXPERIMENTAL_SWITCH_INFO_KEY,
    MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_SWITCH_INFO_KEY,
    hand_pose_release_lineage,
    hand_pose_release_recipe_switch_marker,
    validate_hand_pose_release_recipe_switch_marker,
    validate_hand_pose_release_switch_parent_payload,
    verify_hand_pose_release_switch_parent,
)
from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release_runner import (
    validate_hand_pose_release_switch_marker,
)
from mjlab_microban.tasks.microban_teleop_v12_bootstrap import (
    resolve_bootstrap_artifact_path,
    sha256_file,
)
from mjlab_microban.scripts.evaluate_teleop_v12_checkpoint import (
    hand_pose_release_report_settings,
)
from mjlab_microban.scripts.export_teleop_v12_deployment import (
    _deployment_recipe_revision,
)
from mjlab_microban.scripts.teleop_v12_stage import checkpoint_recipe_kind
from mjlab_microban.tasks.microban_teleop_v12_home_pose import (
    TELEOP_V12_HOME_POSE_INFO_KEY,
    teleop_v12_home_pose_marker,
    validate_teleop_v12_home_pose,
)

JOINTS = [
    "head",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_elbow",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_elbow",
    "left_knee",
]
STD = {r".*": 0.1}


class _Asset:
    def __init__(self, joint_pos: torch.Tensor) -> None:
        self.data = SimpleNamespace(
            default_joint_pos=torch.zeros_like(joint_pos), joint_pos=joint_pos
        )

    def find_joints(self, names):
        del names
        return list(range(len(JOINTS))), list(JOINTS)


def _env(joint_pos: torch.Tensor, active: torch.Tensor):
    asset = _Asset(joint_pos)
    hand = SimpleNamespace(is_active=active)
    twist = torch.zeros(joint_pos.shape[0], 3)
    commands = SimpleNamespace(
        get_command=lambda name: twist, get_term=lambda name: hand
    )
    return SimpleNamespace(
        scene={"robot": asset}, device="cpu", command_manager=commands
    )


def _cfg() -> RewardTermCfg:
    asset_cfg = SceneEntityCfg("robot", joint_names=(".*",))
    asset_cfg.joint_ids = slice(None)
    return RewardTermCfg(
        func=active_hand_arm_released_posture,
        weight=1.0,
        params={
            "asset_cfg": asset_cfg,
            "command_name": "twist",
            "hand_command_name": "hand_target",
            "std_standing": STD,
            "std_walking": STD,
            "std_running": STD,
            "walking_threshold": 0.01,
        },
    )


def _call(term, env, cfg):
    return term(env, **cfg.params)


class HandPoseReleaseRewardTest(unittest.TestCase):
    def test_inactive_rows_equal_v11_and_active_arms_are_dropped(self) -> None:
        joint_pos = torch.zeros(4, len(JOINTS))
        joint_pos[:, 1:4] = 0.3  # left arm far from HOME
        joint_pos[:, 4:7] = 0.2  # right arm
        joint_pos[:, 7] = 0.05
        active = torch.tensor(
            [[False, False], [True, False], [False, True], [True, True]]
        )
        env = _env(joint_pos, active)
        cfg = _cfg()
        term = active_hand_arm_released_posture(cfg, env)
        base = velocity_mdp.variable_posture(cfg, env)
        released = _call(term, env, cfg)
        params = dict(cfg.params)
        params.pop("hand_command_name")
        v11 = base(env, **params)
        self.assertTrue(torch.equal(released[0], v11[0]))
        err = joint_pos.square() / 0.01
        expected = torch.stack(
            [
                torch.exp(-err[0].mean()),
                torch.exp(-err[1, [0, 4, 5, 6, 7]].mean()),
                torch.exp(-err[2, [0, 1, 2, 3, 7]].mean()),
                torch.exp(-err[3, [0, 7]].mean()),
            ]
        )
        torch.testing.assert_close(released, expected)
        self.assertGreater(float(released[3]), float(v11[3]))

    def test_no_active_hand_is_bit_identical_to_v11(self) -> None:
        joint_pos = torch.randn(8, len(JOINTS)) * 0.2
        env = _env(joint_pos, torch.zeros(8, 2, dtype=torch.bool))
        cfg = _cfg()
        params = dict(cfg.params)
        params.pop("hand_command_name")
        v11 = velocity_mdp.variable_posture(cfg, env)(env, **params)
        released = _call(active_hand_arm_released_posture(cfg, env), env, cfg)
        self.assertTrue(torch.equal(released, v11))


_EXPERIMENTAL_MARKER = {
    "schema_version": 1,
    "release_eligible": False,
    "parent_recipe_revision": MICROBAN_TELEOP_V12_RECIPE_REVISION,
    "recipe_revision": MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION,
    "parent_checkpoint_sha256": "0" * 64,
    "parent_iteration": 7099,
}
_PARENT_CHECKPOINT = (
    "repo://logs/rsl_rl/mjlab_microban_teleop_v12/"
    "2026-10-04_10-18-35_c20k_v12_7000_to7100/model_7099.pt"
)
_PARENT_GATE = (
    "repo://artifacts/teleop_v12_gates/"
    "2026-10-04_10-18-35_c20k_v12_7000_to7100_model_7099_gate.json"
)


# Only the centered HOME pins a switch parent (its gated model_7099); every
# other HOME (the forward-lean one included) trains a fresh chain only.  The
# switch mechanism itself is exercised at every HOME under this test-only pin.
_TEST_PARENT_SHA256 = "d" * 64


def _pin_test_parent():
    return mock.patch.object(
        lineage_module,
        "HAND_POSE_RELEASE_RECIPE_SWITCH_PARENT_CHECKPOINT_SHA256",
        _TEST_PARENT_SHA256,
    )


class NoPinnedSwitchParentTest(unittest.TestCase):
    @unittest.skipUnless(
        HAND_POSE_RELEASE_RECIPE_SWITCH_PARENT_CHECKPOINT_SHA256 is None,
        "this HOME pins a switch parent (the centered HOME's gated model_7099)",
    )
    def test_no_parent_is_pinned_and_every_switch_is_refused(self) -> None:
        self.assertIsNone(HAND_POSE_RELEASE_RECIPE_SWITCH_PARENT_CHECKPOINT_SHA256)
        with self.assertRaisesRegex(ValueError, "fresh pose-release chain"):
            hand_pose_release_recipe_switch_marker(
                parent_checkpoint_path=_PARENT_CHECKPOINT,
                parent_checkpoint_sha256=_TEST_PARENT_SHA256,
                parent_stage_gate_path=_PARENT_GATE,
                parent_stage_gate_sha256="b" * 64,
            )
        with _pin_test_parent():
            marker = _release_marker()
        with self.assertRaisesRegex(ValueError, "drifted"):
            validate_hand_pose_release_recipe_switch_marker(marker)
        infos = {
            "microban_teleop_recipe_revision": (
                MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION
            ),
            TELEOP_V12_HOME_POSE_INFO_KEY: teleop_v12_home_pose_marker(),
            MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_SWITCH_INFO_KEY: marker,
        }
        with self.assertRaises(ValueError):
            hand_pose_release_lineage(infos, verify_parent=False)
        with self.assertRaisesRegex(ValueError, "fresh pose-release chain"):
            validate_hand_pose_release_switch_parent_payload(
                {"iter": 7099, "infos": {}}, checkpoint_sha256=_TEST_PARENT_SHA256
            )

    def test_fresh_chain_uses_the_home_pose_release_string(self) -> None:
        from mjlab_microban.robot.home_pose import HOME

        self.assertTrue(
            MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION.startswith(f"{HOME.tag}_")
        )
        self.assertIn(
            "active_hand_arm_pose_release",
            MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION,
        )
        self.assertNotEqual(
            MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION,
            MICROBAN_TELEOP_V12_RECIPE_REVISION,
        )
        infos = {
            "microban_teleop_recipe_revision": (
                MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION
            ),
            TELEOP_V12_HOME_POSE_INFO_KEY: teleop_v12_home_pose_marker(),
        }
        self.assertEqual(
            hand_pose_release_lineage(infos, iteration=14_999),
            HAND_POSE_RELEASE_LINEAGE_FRESH,
        )
        self.assertEqual(
            _deployment_recipe_revision(infos),
            MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION,
        )


def _release_marker(**overrides) -> dict:
    values = {
        "parent_checkpoint_path": _PARENT_CHECKPOINT,
        "parent_checkpoint_sha256": _TEST_PARENT_SHA256,
        "parent_stage_gate_path": _PARENT_GATE,
        "parent_stage_gate_sha256": "b" * 64,
        **overrides,
    }
    return hand_pose_release_recipe_switch_marker(**values)


class HandPoseReleaseRecipeGateTest(unittest.TestCase):
    def setUp(self) -> None:
        patcher = _pin_test_parent()
        patcher.start()
        self.addCleanup(patcher.stop)

    def _infos(self, recipe: str, **extra) -> dict:
        return {
            "microban_teleop_recipe_revision": recipe,
            TELEOP_V12_HOME_POSE_INFO_KEY: teleop_v12_home_pose_marker(),
            **extra,
        }

    def test_fresh_chain_is_release_eligible_without_a_flag(self) -> None:
        infos = self._infos(MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION)
        validate_teleop_v12_home_pose(infos)
        self.assertIsNone(
            validate_corner_rescue_canonical_lineage(infos, iteration=9999)
        )
        self.assertEqual(
            hand_pose_release_lineage(infos, iteration=9999),
            HAND_POSE_RELEASE_LINEAGE_FRESH,
        )
        validate_teleop_v12_home_pose(
            self._infos(MICROBAN_TELEOP_V12_RECIPE_REVISION),
            allow_hand_pose_release_recipe=True,
        )

    def test_experimental_switch_stays_evidence_only(self) -> None:
        self.assertEqual(
            MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_SWITCH_INFO_KEY,
            MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_EXPERIMENTAL_SWITCH_INFO_KEY,
        )
        infos = self._infos(
            MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION,
            **{
                MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_SWITCH_INFO_KEY: dict(
                    _EXPERIMENTAL_MARKER
                )
            },
        )
        with self.assertRaisesRegex(ValueError, "not release-eligible"):
            validate_teleop_v12_home_pose(infos)
        with self.assertRaisesRegex(ValueError, "not release-eligible"):
            validate_corner_rescue_canonical_lineage(infos, iteration=9999)
        validate_teleop_v12_home_pose(infos, allow_hand_pose_release_recipe=True)
        self.assertIsNone(
            validate_corner_rescue_canonical_lineage(
                infos, iteration=9999, allow_hand_pose_release_recipe=True
            )
        )
        self.assertEqual(
            hand_pose_release_lineage(infos, allow_experimental=True),
            HAND_POSE_RELEASE_LINEAGE_EXPERIMENTAL_SWITCH,
        )
        settings = hand_pose_release_report_settings(infos, allow_experimental=True)
        self.assertFalse(settings["canonical_stage_gate_accepts_recipe"])
        with self.assertRaisesRegex(ValueError, "not release-eligible"):
            _deployment_recipe_revision(infos)

    def test_experimental_marker_is_never_release_eligible(self) -> None:
        validate_hand_pose_release_switch_marker(dict(_EXPERIMENTAL_MARKER))
        with self.assertRaises(ValueError):
            validate_hand_pose_release_switch_marker(
                {**_EXPERIMENTAL_MARKER, "release_eligible": True}
            )

    def test_release_marker_is_exact_and_pinned_to_the_gated_model_7099(
        self,
    ) -> None:
        marker = _release_marker()
        self.assertTrue(marker["release_eligible"])
        self.assertEqual(marker["parent_iteration"], 7099)
        self.assertEqual(marker["parent_completed_updates"], 7100)
        self.assertEqual(
            marker["recipe_revision"],
            MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION,
        )
        self.assertEqual(
            marker["parent_recipe_revision"], MICROBAN_TELEOP_V12_RECIPE_REVISION
        )
        self.assertTrue(marker["reason"])
        self.assertEqual(validate_hand_pose_release_recipe_switch_marker(marker), marker)
        for bad in (
            {**marker, "release_eligible": False},
            {**marker, "parent_iteration": 9999},
            {**marker, "parent_checkpoint_sha256": "c" * 64},
            {**marker, "parent_checkpoint_path": "/abs/model_7099.pt"},
            {**marker, "parent_checkpoint_path": _PARENT_CHECKPOINT.replace("7099", "6999")},
            {**marker, "parent_stage_gate_sha256": "nothex"},
            {**marker, "reason": "other"},
            {**marker, "extra": 1},
            {k: v for k, v in marker.items() if k != "reason"},
        ):
            with self.subTest(bad=sorted(set(bad.items()) ^ set(marker.items()), key=str)):
                with self.assertRaises(ValueError):
                    validate_hand_pose_release_recipe_switch_marker(bad)
        with self.assertRaises(ValueError):
            _release_marker(parent_checkpoint_sha256="c" * 64)

    def test_release_switch_lineage_rules(self) -> None:
        marker = _release_marker()
        infos = self._infos(
            MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION,
            **{MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_SWITCH_INFO_KEY: marker},
        )
        # Structural checks (no parent I/O) accept it without any flag.
        validate_teleop_v12_home_pose(infos)
        self.assertEqual(
            hand_pose_release_lineage(infos, iteration=7199, verify_parent=False),
            HAND_POSE_RELEASE_LINEAGE_RELEASE_SWITCH,
        )
        settings = hand_pose_release_report_settings(infos)
        self.assertTrue(settings["canonical_stage_gate_accepts_recipe"])
        self.assertEqual(settings["release_recipe_switch"], marker)
        self.assertEqual(
            _deployment_recipe_revision(infos),
            MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION,
        )
        self.assertEqual(
            _deployment_recipe_revision(
                self._infos(MICROBAN_TELEOP_V12_RECIPE_REVISION)
            ),
            MICROBAN_TELEOP_V12_RECIPE_REVISION,
        )
        self.assertEqual(
            hand_pose_release_report_settings(
                self._infos(MICROBAN_TELEOP_V12_RECIPE_REVISION)
            ),
            {},
        )
        for iteration in (7099, 7000, True):
            with self.subTest(iteration=iteration), self.assertRaises(ValueError):
                hand_pose_release_lineage(
                    infos, iteration=iteration, verify_parent=False
                )
        both = {
            **infos,
            MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_SWITCH_INFO_KEY: dict(
                _EXPERIMENTAL_MARKER
            ),
        }
        with self.assertRaisesRegex(ValueError, "both"):
            hand_pose_release_lineage(both, allow_experimental=True)
        rescued = {**infos, "microban_teleop_v12_corner_pair_rescue": {}}
        with self.assertRaisesRegex(ValueError, "rescue"):
            hand_pose_release_lineage(rescued, verify_parent=False)
        # Every real load re-validates the parent files: a missing parent fails.
        with self.assertRaises(ValueError):
            hand_pose_release_lineage(
                {
                    **infos,
                    MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_SWITCH_INFO_KEY: (
                        _release_marker(
                            parent_checkpoint_path="repo://missing/model_7099.pt"
                        )
                    ),
                },
                iteration=7199,
            )

    def test_parent_payload_must_be_the_unmarked_canonical_model_7099(
        self,
    ) -> None:
        sha = _TEST_PARENT_SHA256
        payload = {
            "iter": 7099,
            "infos": {
                "microban_teleop_recipe_revision": MICROBAN_TELEOP_V12_RECIPE_REVISION,
                "env_state": {"common_step_counter": 7100 * 24},
            },
        }
        validate_hand_pose_release_switch_parent_payload(
            payload, checkpoint_sha256=sha
        )
        with self.assertRaisesRegex(ValueError, "only the gated"):
            validate_hand_pose_release_switch_parent_payload(
                payload, checkpoint_sha256="a" * 64
            )
        for mutate in (
            lambda p: p.update(iter=9999),
            lambda p: p["infos"].update(env_state={"common_step_counter": 1}),
            lambda p: p["infos"].update(
                microban_teleop_recipe_revision=(
                    MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION
                )
            ),
            lambda p: p["infos"].update(
                {"microban_teleop_v12_corner_pair_rescue": {"x": 1}}
            ),
            lambda p: p["infos"].update(
                {MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_SWITCH_INFO_KEY: {}}
            ),
        ):
            candidate = deepcopy(payload)
            mutate(candidate)
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                validate_hand_pose_release_switch_parent_payload(
                    candidate, checkpoint_sha256=sha
                )


_REAL_PARENT = resolve_bootstrap_artifact_path(_PARENT_CHECKPOINT)
_REAL_GATE = resolve_bootstrap_artifact_path(_PARENT_GATE)


@unittest.skipUnless(
    HAND_POSE_RELEASE_RECIPE_SWITCH_PARENT_CHECKPOINT_SHA256 is not None
    and _REAL_PARENT.is_file()
    and _REAL_GATE.is_file(),
    "no switch parent is pinned at this HOME, or its gated model_7099 is not present",
)
class HandPoseReleaseRealParentTest(unittest.TestCase):
    """Validate the pinned parent and its stage gate exactly as consumers do."""

    def setUp(self) -> None:
        self.marker = _release_marker(
            parent_stage_gate_sha256=sha256_file(_REAL_GATE)
        )
        payload = torch.load(_REAL_PARENT, map_location="cpu", weights_only=False)
        self.parent_infos = payload["infos"]

    def test_real_parent_gate_validates_and_descendant_is_accepted(self) -> None:
        self.assertEqual(
            sha256_file(_REAL_PARENT),
            HAND_POSE_RELEASE_RECIPE_SWITCH_PARENT_CHECKPOINT_SHA256,
        )
        parent_infos = verify_hand_pose_release_switch_parent(self.marker)
        self.assertEqual(
            parent_infos["microban_teleop_recipe_revision"],
            MICROBAN_TELEOP_V12_RECIPE_REVISION,
        )
        descendant = {
            **self.parent_infos,
            "microban_teleop_recipe_revision": (
                MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION
            ),
            MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_SWITCH_INFO_KEY: self.marker,
        }
        self.assertEqual(
            hand_pose_release_lineage(descendant, iteration=9999),
            HAND_POSE_RELEASE_LINEAGE_RELEASE_SWITCH,
        )
        self.assertIsNone(
            validate_corner_rescue_canonical_lineage(descendant, iteration=9999)
        )
        drifted = deepcopy(descendant)
        drifted["action_clip"] = [-1.0, 1.0]
        with self.assertRaisesRegex(ValueError, "differs from its parent"):
            hand_pose_release_lineage(drifted, iteration=9999)

    def test_wrong_gate_hash_is_refused(self) -> None:
        marker = _release_marker(parent_stage_gate_sha256="e" * 64)
        with self.assertRaisesRegex(ValueError, "stage gate changed"):
            verify_hand_pose_release_switch_parent(marker)

    def test_canonical_parent_reports_canonical_recipe_kind(self) -> None:
        self.assertEqual(checkpoint_recipe_kind(_REAL_PARENT), "canonical")


if __name__ == "__main__":
    unittest.main()
