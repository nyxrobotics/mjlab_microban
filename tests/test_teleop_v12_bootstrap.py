"""CPU gates for the authenticated contract-v12 bootstrap chain."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from mjlab_microban.tasks.microban_teleop_v12_actor import (
    LEGACY_TO_TELEOP_OBSERVATION_INDEX,
    LEGACY_VELOCITY_NORMALIZER_EPS,
    TELEOP_V12_EXTRA_OBSERVATION_COLUMNS,
    TELEOP_V12_HMD_OBSERVATION_COLUMNS,
    TELEOP_V12_TARGET_POSITION_NORMALIZER_DENOMINATORS,
    TELEOP_V12_TARGET_POSITION_NORMALIZER_STORED_STD,
    TELEOP_V12_TARGET_POSITION_OBSERVATION_COLUMNS,
    TELEOP_TRAINABLE_ACTOR_PARAMETERS,
    LegacyAdapterTeleopActor,
)
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_HMD_JOINT_NAMES,
    MICROBAN_TELEOP_ACTION_JOINT_NAMES,
)
from mjlab_microban.tasks.microban_teleop_v12_bootstrap import (
    LEGACY_TELEOP_PROBE_REVISION,
    assert_actor_frozen_against_source,
    bootstrap_legacy_actor,
    inspect_legacy_velocity_checkpoint,
    serialize_bootstrap_provenance,
    sha256_file,
    validate_bootstrap_provenance,
)
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MICROBAN_TELEOP_V12_ACTION_CLIP,
    MICROBAN_TELEOP_V12_RECIPE_REVISION,
    MicrobanTeleopV12RlCfg,
    make_microban_teleop_v12_env_cfg,
)

# The frozen source is now any centered-HOME velocity checkpoint chosen on the
# command line, so these CPU tests build a synthetic source and a receipt with
# the exact passing 9x300 schema instead of pinning one historical file.
_TEMPORARY = tempfile.TemporaryDirectory(prefix="teleop_v12_bootstrap_test_")
CHECKPOINT = Path(_TEMPORARY.name) / "model_499.pt"
RECEIPT = Path(_TEMPORARY.name) / "probe_9x300.json"


def _observations(width: int) -> TensorDict:
    return TensorDict({"actor": torch.zeros(1, width)}, batch_size=[1])


def _source_model() -> MLPModel:
    return MLPModel(
        obs=_observations(63),
        obs_groups={"actor": ["actor"]},
        obs_set="actor",
        output_dim=18,
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    )


def _target_model() -> LegacyAdapterTeleopActor:
    return LegacyAdapterTeleopActor(
        obs=_observations(81),
        obs_groups={"actor": ["actor"]},
        obs_set="actor",
        output_dim=18,
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    )


def _write_synthetic_source() -> tuple[str, str]:
    torch.manual_seed(7)
    source = _source_model()
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.normal_(0.0, 0.1)
    state = source.state_dict()
    state["obs_normalizer._mean"].normal_(0.0, 0.3)
    state["obs_normalizer._var"].uniform_(0.2, 2.0)
    state["obs_normalizer._std"].copy_(state["obs_normalizer._var"].sqrt())
    state["obs_normalizer.count"].fill_(500 * 4096 * 24)
    torch.save({"actor_state_dict": state, "iter": 499, "infos": {}}, CHECKPOINT)
    source_sha256 = sha256_file(CHECKPOINT)
    result = {
        "completed": True,
        "fell": False,
        "nonfinite": None,
        "executed_steps": 300,
        "raw_action_recurrence_verified_steps": 300,
        "neutral_target_verified_steps": 300,
        "maximum_actual_soft_limit_violation_rad": 0.0,
    }
    receipt = {
        "probe": LEGACY_TELEOP_PROBE_REVISION,
        "checkpoint": {"path": str(CHECKPOINT), "sha256": source_sha256},
        "settings": {
            "seed": 42,
            "steps": 300,
            "settle_steps": 50,
            "step_dt_s": 0.02,
            "action_clip": MICROBAN_TELEOP_V12_ACTION_CLIP,
            "previous_action": "raw_actor_output",
            "foot_target": "exact_zero_inactive",
            "arm_target": "home_overlay",
        },
        "mapping": {
            "legacy_observation_width": 63,
            "teleop_observation_width": 81,
            "legacy_joint_names": list(MICROBAN_TELEOP_ACTION_JOINT_NAMES),
            "teleop_joint_names": [
                *MICROBAN_HMD_JOINT_NAMES,
                *MICROBAN_TELEOP_ACTION_JOINT_NAMES,
            ],
            "teleop_joint_indices_for_legacy": list(range(3, 21)),
            "legacy_action_names": list(MICROBAN_TELEOP_ACTION_JOINT_NAMES),
            "teleop_action_indices_for_legacy": list(range(18)),
            "legacy_source_columns_to_teleop_target_columns": [
                list(pair) for pair in LEGACY_TO_TELEOP_OBSERVATION_INDEX
            ],
            "new_teleop_columns": list(TELEOP_V12_EXTRA_OBSERVATION_COLUMNS),
        },
        "summary": {
            "scenario_count": 9,
            "completed_scenario_count": 9,
            "fall_scenario_count": 0,
            "nonfinite_scenario_count": 0,
            "actual_soft_limit_violation_scenario_count": 0,
            "directionally_correct_scenario_count": 8,
            "directional_scenario_count": 8,
            "neutral_target_contract_all_steps": True,
            "raw_action_recurrence_all_steps": True,
        },
        "results": [dict(result) for _ in range(9)],
    }
    RECEIPT.write_text(json.dumps(receipt), encoding="utf-8")
    return source_sha256, sha256_file(RECEIPT)


LEGACY_VELOCITY_CHECKPOINT_SHA256, PINNED_LEGACY_TELEOP_PROBE_SHA256 = (
    _write_synthetic_source()
)


class TeleopV12BootstrapTest(unittest.TestCase):
    def test_authenticated_pristine_actor_parity(self) -> None:
        source_identity, source_state = inspect_legacy_velocity_checkpoint(
            CHECKPOINT, LEGACY_VELOCITY_CHECKPOINT_SHA256
        )
        source = _source_model()
        source.load_state_dict(source_state, strict=True)
        target = _target_model()
        provenance = bootstrap_legacy_actor(
            target,
            CHECKPOINT,
            LEGACY_VELOCITY_CHECKPOINT_SHA256,
            RECEIPT,
            PINNED_LEGACY_TELEOP_PROBE_SHA256,
        )
        self.assertEqual(provenance.source, source_identity)
        self.assertEqual(
            int(target.obs_normalizer.count.item()), source_identity.normalizer_count
        )
        trainable = [name for name, value in target.named_parameters() if value.requires_grad]
        self.assertEqual(sorted(trainable), sorted(TELEOP_TRAINABLE_ACTOR_PARAMETERS))
        state = target.state_dict()
        target_columns = TELEOP_V12_TARGET_POSITION_OBSERVATION_COLUMNS
        expected_std = torch.tensor([TELEOP_V12_TARGET_POSITION_NORMALIZER_STORED_STD])
        torch.testing.assert_close(
            state["obs_normalizer._std"][:, target_columns],
            expected_std,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            state["obs_normalizer._var"][:, target_columns],
            expected_std.square(),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            state["obs_normalizer._std"][:, target_columns]
            + LEGACY_VELOCITY_NORMALIZER_EPS,
            torch.tensor([TELEOP_V12_TARGET_POSITION_NORMALIZER_DENOMINATORS]),
            rtol=0.0,
            atol=4.0e-9,
        )
        identity_columns = TELEOP_V12_HMD_OBSERVATION_COLUMNS
        self.assertTrue(
            torch.equal(
                state["obs_normalizer._std"][:, identity_columns],
                torch.ones((1, len(identity_columns))),
            )
        )

        generator = torch.Generator().manual_seed(20260925)
        teleop = torch.randn(10_000, 81, generator=generator)
        legacy = teleop[
            :, [target for _source, target in LEGACY_TO_TELEOP_OBSERVATION_INDEX]
        ]
        with torch.inference_mode():
            expected = source(TensorDict({"actor": legacy}, batch_size=[10_000]))
            actual = target(TensorDict({"actor": teleop}, batch_size=[10_000]))
        max_absolute = float(torch.max(torch.abs(actual - expected)).item())
        self.assertLessEqual(max_absolute, 2.0e-5)
        torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-5)

        validated = validate_bootstrap_provenance(
            serialize_bootstrap_provenance(provenance), verify_files=True
        )
        self.assertEqual(validated, provenance)
        assert_actor_frozen_against_source(target, provenance)

    def test_source_guard_allows_only_new_first_layer_columns(self) -> None:
        target = _target_model()
        provenance = bootstrap_legacy_actor(
            target,
            CHECKPOINT,
            LEGACY_VELOCITY_CHECKPOINT_SHA256,
            RECEIPT,
            PINNED_LEGACY_TELEOP_PROBE_SHA256,
        )
        first = target.mlp[0]
        assert isinstance(first, torch.nn.Linear)
        with torch.no_grad():
            first.weight[0, TELEOP_V12_EXTRA_OBSERVATION_COLUMNS[0]] += 0.5
        assert_actor_frozen_against_source(target, provenance)
        target.assert_frozen_legacy_state()
        with torch.no_grad():
            first.weight[0, 0] += 0.5
        with self.assertRaisesRegex(RuntimeError, "pinned legacy"):
            assert_actor_frozen_against_source(target, provenance)

    def test_receipt_with_unclipped_settings_is_rejected(self) -> None:
        report = json.loads(RECEIPT.read_text(encoding="utf-8"))
        report["settings"]["action_clip"] = None
        stale = Path(_TEMPORARY.name) / "stale_probe.json"
        stale.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "settings drifted"):
            bootstrap_legacy_actor(
                _target_model(),
                CHECKPOINT,
                LEGACY_VELOCITY_CHECKPOINT_SHA256,
                stale,
                sha256_file(stale),
            )

    def test_source_iteration_and_count_are_recorded_not_pinned(self) -> None:
        identity, _state = inspect_legacy_velocity_checkpoint(
            CHECKPOINT, LEGACY_VELOCITY_CHECKPOINT_SHA256
        )
        self.assertEqual(identity.iteration, 499)
        self.assertEqual(identity.normalizer_count, 500 * 4096 * 24)
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            inspect_legacy_velocity_checkpoint(CHECKPOINT, "0" * 64)

    def test_v12_environment_and_runner_config_use_raw_legacy_semantics(self) -> None:
        cfg = make_microban_teleop_v12_env_cfg(play=True)
        # No software clip: only the servo's one-turn goal range bounds the
        # absolute target.
        self.assertEqual(
            cfg.actions["joint_pos"].clip, {r".*": (-math.pi, math.pi)}
        )
        self.assertEqual(MICROBAN_TELEOP_V12_ACTION_CLIP, [-math.pi, math.pi])
        self.assertIn("servo_range_pi", MICROBAN_TELEOP_V12_RECIPE_REVISION)
        self.assertIn("joint_soft_limit_guard", cfg.rewards)
        self.assertEqual(
            cfg.observations["actor"].terms["actions"].func.__name__, "last_action"
        )
        self.assertEqual(
            cfg.observations["critic"].terms["actions"].func.__name__, "last_action"
        )
        self.assertTrue(MicrobanTeleopV12RlCfg.actor.obs_normalization)
        self.assertEqual(
            MicrobanTeleopV12RlCfg.actor.distribution_cfg["std_type"], "scalar"
        )
        self.assertEqual(MicrobanTeleopV12RlCfg.clip_actions, None)


if __name__ == "__main__":
    unittest.main()
