"""CPU tests for the isolated legacy-actor/teleop-environment probe."""

from __future__ import annotations

import unittest

import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from mjlab_microban.scripts.probe_legacy_actor_in_teleop_env import (
    ActorLayout,
    _assemble_legacy_observation,
    _indices_by_name,
)
from mjlab_microban.tasks.microban_teleop_v12_actor import (
    LEGACY_VELOCITY_NORMALIZER_EPS,
    TELEOP_V12_HAND_ACTIVE_OBSERVATION_COLUMNS,
    TELEOP_V12_HMD_OBSERVATION_COLUMNS,
    TELEOP_V12_TARGET_POSITION_NORMALIZER_DENOMINATORS,
    TELEOP_V12_TARGET_POSITION_NORMALIZER_STORED_STD,
    TELEOP_V12_TARGET_POSITION_OBSERVATION_COLUMNS,
    transplant_legacy_actor_state_to_teleop83,
)

BODY_JOINTS = tuple(f"body_{index}" for index in range(18))
HMD_JOINTS = ("head", "neck_roll", "neck_pitch")


def _legacy_layout() -> ActorLayout:
    return ActorLayout(
        terms={
            "base_ang_vel": slice(0, 3),
            "projected_gravity": slice(3, 6),
            "joint_pos": slice(6, 24),
            "joint_vel": slice(24, 42),
            "actions": slice(42, 60),
            "command": slice(60, 63),
        },
        width=63,
        joint_pos_names=BODY_JOINTS,
        joint_vel_names=BODY_JOINTS,
        action_names=BODY_JOINTS,
    )


def _teleop_layout() -> ActorLayout:
    return ActorLayout(
        terms={
            "base_ang_vel": slice(0, 3),
            "projected_gravity": slice(3, 6),
            "joint_pos": slice(6, 27),
            "joint_vel": slice(27, 48),
            "actions": slice(48, 66),
            "command": slice(66, 69),
            "foot_target": slice(69, 75),
            "hand_target": slice(75, 83),
        },
        width=83,
        joint_pos_names=HMD_JOINTS + BODY_JOINTS,
        joint_vel_names=HMD_JOINTS + BODY_JOINTS,
        action_names=BODY_JOINTS,
    )


def _model(width: int) -> MLPModel:
    observation = TensorDict({"actor": torch.zeros(1, width)}, batch_size=[1])
    return MLPModel(
        obs=observation,
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


class LegacyTeleopObservationMappingTest(unittest.TestCase):
    def test_named_projection_is_exact_and_twist_is_injected(self) -> None:
        legacy = _legacy_layout()
        teleop = _teleop_layout()
        joint_indices = _indices_by_name(
            BODY_JOINTS, teleop.joint_pos_names, label="joint"
        )
        action_indices = _indices_by_name(
            BODY_JOINTS, teleop.action_names, label="action"
        )
        source = torch.arange(83, dtype=torch.float32).unsqueeze(0)
        twist = torch.tensor([[101.0, 102.0, 103.0]])
        projected = _assemble_legacy_observation(
            source,
            teleop=teleop,
            legacy=legacy,
            joint_pos_indices=joint_indices,
            joint_vel_indices=joint_indices,
            action_indices=action_indices,
            twist=twist,
        )
        expected = torch.cat(
            (
                source[:, 0:6],
                source[:, 9:27],
                source[:, 30:48],
                source[:, 48:66],
                twist,
            ),
            dim=1,
        )
        self.assertTrue(torch.equal(projected, expected))
        self.assertEqual(joint_indices, tuple(range(3, 21)))

    def test_named_projection_rejects_missing_or_duplicate_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing legacy joint"):
            _indices_by_name(("a", "b"), ("a", "c"), label="joint")
        with self.assertRaisesRegex(ValueError, "Duplicate target joint"):
            _indices_by_name(("a",), ("a", "a"), label="joint")


class LegacyActorTransplantTest(unittest.TestCase):
    def test_83_input_transplant_preserves_legacy_deterministic_actor(self) -> None:
        torch.manual_seed(7)
        legacy_model = _model(63)
        target_model = _model(83)
        source_state = legacy_model.state_dict()
        source_state["obs_normalizer._mean"].normal_()
        source_state["obs_normalizer._var"].uniform_(0.1, 2.0)
        source_state["obs_normalizer._std"].copy_(
            torch.sqrt(source_state["obs_normalizer._var"])
        )
        source_state["obs_normalizer.count"].fill_(1_474_560_000)
        legacy_model.load_state_dict(source_state, strict=True)

        teleop = _teleop_layout()
        legacy = _legacy_layout()
        body_indices = _indices_by_name(
            BODY_JOINTS, teleop.joint_pos_names, label="joint"
        )
        action_indices = _indices_by_name(
            BODY_JOINTS, teleop.action_names, label="action"
        )
        target_columns = (
            *range(6),
            *(teleop.terms["joint_pos"].start + index for index in body_indices),
            *(teleop.terms["joint_vel"].start + index for index in body_indices),
            *(teleop.terms["actions"].start + index for index in action_indices),
            *range(teleop.terms["command"].start, teleop.terms["command"].stop),
        )
        mapping = tuple(enumerate(target_columns))
        transplanted = transplant_legacy_actor_state_to_teleop83(
            source_state, target_model.state_dict(), mapping
        )
        target_model.load_state_dict(transplanted, strict=True)

        teleop_obs = torch.randn(16, 83)
        legacy_obs = teleop_obs[:, list(target_columns)]
        expected = legacy_model(TensorDict({"actor": legacy_obs}, batch_size=[16]))
        actual = target_model(TensorDict({"actor": teleop_obs}, batch_size=[16]))
        torch.testing.assert_close(actual, expected, rtol=2.0e-6, atol=2.0e-6)

        new_columns = sorted(set(range(83)) - set(target_columns))
        self.assertTrue(
            torch.equal(
                transplanted["mlp.0.weight"][:, new_columns],
                torch.zeros((512, 20)),
            )
        )
        self.assertTrue(
            torch.equal(
                transplanted["obs_normalizer._mean"][:, new_columns],
                torch.zeros((1, 20)),
            )
        )
        identity_columns = (
            *TELEOP_V12_HMD_OBSERVATION_COLUMNS,
            *TELEOP_V12_HAND_ACTIVE_OBSERVATION_COLUMNS,
        )
        self.assertEqual(
            new_columns,
            sorted(
                (*identity_columns, *TELEOP_V12_TARGET_POSITION_OBSERVATION_COLUMNS)
            ),
        )
        torch.testing.assert_close(
            transplanted["obs_normalizer._std"][
                :, TELEOP_V12_TARGET_POSITION_OBSERVATION_COLUMNS
            ],
            torch.tensor([TELEOP_V12_TARGET_POSITION_NORMALIZER_STORED_STD]),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            transplanted["obs_normalizer._std"][
                :, TELEOP_V12_TARGET_POSITION_OBSERVATION_COLUMNS
            ]
            + LEGACY_VELOCITY_NORMALIZER_EPS,
            torch.tensor([TELEOP_V12_TARGET_POSITION_NORMALIZER_DENOMINATORS]),
            rtol=0.0,
            atol=4.0e-9,
        )
        self.assertTrue(
            torch.equal(
                transplanted["obs_normalizer._std"][:, identity_columns],
                torch.ones((1, len(identity_columns))),
            )
        )
        self.assertEqual(legacy.width, 63)

    def test_transplant_rejects_incomplete_mapping(self) -> None:
        with self.assertRaisesRegex(ValueError, "cover source columns"):
            transplant_legacy_actor_state_to_teleop83(
                _model(63).state_dict(),
                _model(83).state_dict(),
                tuple((index, index) for index in range(62)),
            )


if __name__ == "__main__":
    unittest.main()
