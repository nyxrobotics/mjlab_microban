"""Unit tests for the fail-closed frozen tracking teacher."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from mjlab_microban.tasks.microban_tracking_mdp import (
    MicrobanTrackingBoundedGaussianDistribution,
)
from mjlab_microban.tasks.microban_tracking_teacher import (
    MICROBAN_TRACKING_TEACHER_CHECKPOINT_ITERATION,
    MICROBAN_TRACKING_TEACHER_OBSERVATION_WIDTH,
    assemble_velocity_teacher_observation,
    bounded_student_deterministic_action_closure,
    load_frozen_velocity_teacher,
    project_velocity_teacher_labels,
    reference_velocity_command_b,
    sha256_file,
)


def _legacy_actor() -> MLPModel:
    observation = TensorDict({"teacher": torch.zeros(1, 63)}, batch_size=[1])
    return MLPModel(
        obs=observation,
        obs_groups={"actor": ["teacher"]},
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


class TrackingTeacherObservationTest(unittest.TestCase):
    def test_schema_is_exact_and_reference_velocity_is_body_relative(self) -> None:
        batch = 2
        observation = assemble_velocity_teacher_observation(
            base_ang_vel=torch.full((batch, 3), 1.0),
            projected_gravity=torch.full((batch, 3), 2.0),
            joint_pos=torch.full((batch, 18), 3.0),
            joint_vel=torch.full((batch, 18), 4.0),
            previous_action=torch.full((batch, 18), 5.0),
            command=torch.full((batch, 3), 6.0),
        )
        self.assertEqual(observation.shape, (batch, 63))
        self.assertEqual(MICROBAN_TRACKING_TEACHER_OBSERVATION_WIDTH, 63)
        self.assertTrue(torch.equal(observation[:, :3], torch.ones(batch, 3)))
        self.assertTrue(torch.equal(observation[:, -3:], torch.full((batch, 3), 6.0)))

        half = math.sqrt(0.5)
        command = reference_velocity_command_b(
            reference_quat_w=torch.tensor([[half, 0.0, 0.0, half]]),
            reference_lin_vel_w=torch.tensor([[1.0, 0.0, 0.0]]),
            reference_ang_vel_w=torch.tensor([[0.0, 0.0, 0.25]]),
        )
        torch.testing.assert_close(command, torch.tensor([[0.0, -1.0, 0.25]]), atol=1e-6, rtol=0.0)

    def test_loader_verifies_hash_and_freezes_actor(self) -> None:
        actor = _legacy_actor()
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "model_999.pt"
            torch.save(
                {
                    "actor_state_dict": actor.state_dict(),
                    "iter": MICROBAN_TRACKING_TEACHER_CHECKPOINT_ITERATION,
                },
                checkpoint,
            )
            digest = sha256_file(checkpoint)
            teacher = load_frozen_velocity_teacher(
                checkpoint, checkpoint_sha256=digest
            )
            output = teacher(torch.zeros(3, 63))
            self.assertEqual(output.shape, (3, 18))
            self.assertFalse(teacher.training)
            self.assertTrue(all(not value.requires_grad for value in teacher.parameters()))
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_frozen_velocity_teacher(checkpoint, checkpoint_sha256="0" * 64)


class TrackingTeacherProjectionTest(unittest.TestCase):
    def _project(self, teacher_action: torch.Tensor, **overrides: object):
        batch = teacher_action.shape[0]
        kwargs: dict[str, object] = {
            "teacher_scale": 1.0,
            "teacher_offset": torch.zeros(18),
            "student_scale": 1.0,
            "student_offset": torch.zeros(18),
            "soft_lower": torch.full((18,), -1.0),
            "soft_upper": torch.full((18,), 1.0),
            "student_action_lower": torch.full((18,), -0.8),
            "student_action_upper": torch.full((18,), 0.8),
            "joint_pos": torch.zeros(batch, 18),
            "joint_vel": torch.zeros(batch, 18),
            "default_joint_pos": torch.zeros(18),
        }
        kwargs.update(overrides)
        return project_velocity_teacher_labels(teacher_action, **kwargs)

    def test_projection_is_safe_label_but_only_small_projection_is_faithful(self) -> None:
        action = torch.zeros(2, 18)
        action[0, 0] = 0.8005
        action[1, 0] = 0.802
        result = self._project(action)
        self.assertTrue(torch.equal(result.label_action[:, 0], torch.full((2,), 0.8)))
        self.assertTrue(torch.equal(result.accepted, torch.tensor([True, True])))
        self.assertTrue(
            torch.equal(result.faithful_to_legacy, torch.tensor([True, False]))
        )
        torch.testing.assert_close(
            result.maximum_target_projection_rad,
            torch.tensor([0.0005, 0.002]),
            atol=1e-7,
            rtol=0.0,
        )

    def test_lookahead_is_hard_gate_and_margin_is_bc_weight(self) -> None:
        action = torch.zeros(3, 18)
        joint_pos = torch.zeros(3, 18)
        joint_vel = torch.zeros(3, 18)
        # q + 0.12*qdot = 1.08: beyond the hard upper limit.
        joint_vel[1, 0] = 9.0
        # Inside the hard range, but beyond the 5% preferred range (+0.9).
        joint_pos[2, 0] = 0.91
        result = self._project(action, joint_pos=joint_pos, joint_vel=joint_vel)
        self.assertTrue(torch.equal(result.accepted, torch.tensor([True, False, True])))
        self.assertEqual(result.bc_weight.shape, (3, 18))
        self.assertTrue(bool((result.bc_weight[0] == 1.0).all().item()))
        self.assertTrue(bool((result.bc_weight[1] == 0.0).all().item()))
        self.assertEqual(float(result.bc_weight[2, 0].item()), 0.0)
        self.assertTrue(bool((result.bc_weight[2, 1:] == 1.0).all().item()))

    def test_student_deterministic_closure_is_strictly_interior(self) -> None:
        distribution = MicrobanTrackingBoundedGaussianDistribution(
            18,
            (0.03,) * 18,
            (-1.0,) * 18,
            (1.0,) * 18,
            std_type="log",
        )
        lower, upper = bounded_student_deterministic_action_closure(distribution)
        self.assertTrue(bool((lower > -1.0).all().item()))
        self.assertTrue(bool((upper < 1.0).all().item()))
        self.assertTrue(bool((lower < 0.0).all().item()))
        self.assertTrue(bool((upper > 0.0).all().item()))


if __name__ == "__main__":
    unittest.main()
