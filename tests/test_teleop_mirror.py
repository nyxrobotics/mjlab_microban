"""The left-right mirror of the PICO observations/actions and the leg-only mirror loss."""

from __future__ import annotations

import unittest

import mujoco
import numpy as np
import torch
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from mjlab_microban.policy_contract import PICO_ARM_JOINT_NAMES
from mjlab_microban.robot.home_pose import HOME
from mjlab_microban.robot.microban_constants import MICROBAN_XML
from mjlab_microban.schedules import PICO_SCHEDULE
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_HMD_JOINT_NAMES,
    MICROBAN_TELEOP_ACTION_JOINT_NAMES,
)
from mjlab_microban.tasks.microban_teleop_mirror import (
    MIRROR_ACTION,
    MIRROR_OBSERVATION,
    mirror_actions,
    mirror_augmentation,
    mirror_joint_sign,
    mirror_observations,
    mirror_partner,
)
from mjlab_microban.tasks.microban_teleop_v12_actor import (
    LEGACY_TO_TELEOP_OBSERVATION_INDEX,
    TELEOP_MIRROR_LOSS_ACTION_COLUMNS,
    LegacyAdapterPPO,
    LegacyAdapterTeleopActor,
    transplant_legacy_actor_state_to_teleop,
)
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import MicrobanTeleopV12RlCfg

JOINTS = (*MICROBAN_HMD_JOINT_NAMES, *MICROBAN_TELEOP_ACTION_JOINT_NAMES)
REFLECT = np.diag([1.0, -1.0, 1.0])
GAUSSIAN = {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}


def _model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(MICROBAN_XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _obs(batch: int, generator: torch.Generator) -> TensorDict:
    return TensorDict(
        {
            "actor": torch.randn(batch, 81, generator=generator),
            "critic": torch.randn(batch, 96, generator=generator),
        },
        batch_size=[batch],
    )


class MirrorTest(unittest.TestCase):
    def test_mirror_is_a_signed_permutation_that_undoes_itself(self) -> None:
        for name, (index, sign) in {**MIRROR_OBSERVATION, "action": MIRROR_ACTION}.items():
            self.assertEqual(sorted(index.tolist()), list(range(index.numel())), name)
            self.assertTrue(torch.equal(sign.abs(), torch.ones_like(sign)), name)
        obs = _obs(7, torch.Generator().manual_seed(1))
        twice = mirror_observations(mirror_observations(obs))
        for group in ("actor", "critic"):
            self.assertTrue(torch.equal(twice[group], obs[group]), group)
        actions = torch.randn(7, 18)
        self.assertTrue(torch.equal(mirror_actions(mirror_actions(actions)), actions))

    def test_augmentation_appends_the_mirror(self) -> None:
        obs = _obs(5, torch.Generator().manual_seed(2))
        actions = torch.randn(5, 18)
        both, both_actions = mirror_augmentation(obs=obs, actions=actions)
        self.assertEqual(both.batch_size[0], 10)
        self.assertTrue(torch.equal(both["actor"][:5], obs["actor"]))
        self.assertTrue(torch.equal(both["critic"][5:], mirror_observations(obs)["critic"]))
        self.assertTrue(torch.equal(both_actions[5:], mirror_actions(actions)))
        self.assertEqual(mirror_augmentation(obs=None, actions=actions)[0], None)

    def test_joint_signs_follow_the_mjcf_axes(self) -> None:
        # The mirror of a joint's axis a is -R a (an axial vector); the angle
        # keeps its sign when that is the partner's axis.
        model, data = _model()
        for name in JOINTS:
            axis = data.xaxis[model.joint(name).id]
            partner = data.xaxis[model.joint(mirror_partner(name)).id]
            np.testing.assert_allclose(
                -REFLECT @ axis, mirror_joint_sign(name) * partner, atol=1e-6, err_msg=name
            )

    def test_mirrored_joint_angles_mirror_the_body(self) -> None:
        model, data = _model()
        rng = np.random.default_rng(3)
        angles = {name: float(rng.uniform(-0.6, 0.6)) for name in JOINTS}

        def sites(joint_angles: dict[str, float]) -> dict[str, np.ndarray]:
            for name, value in joint_angles.items():
                data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
            mujoco.mj_forward(model, data)
            return {model.site(i).name: data.site_xpos[i].copy() for i in range(model.nsite)}

        original = sites(angles)
        mirrored = sites(
            {mirror_partner(name): mirror_joint_sign(name) * value for name, value in angles.items()}
        )
        for name in ("left_foot", "right_foot", "left_hand", "right_hand"):
            np.testing.assert_allclose(
                mirrored[mirror_partner(name)], REFLECT @ original[name], atol=2e-4, err_msg=name
            )

    def test_imu_frame_signs_follow_the_imu_site(self) -> None:
        model, data = _model()
        site = model.site("imu")
        trunk_from_imu = (
            data.xmat[site.bodyid[0]].reshape(3, 3).T @ data.site_xmat[site.id].reshape(3, 3)
        )
        layout = dict(
            zip(("base_lin_vel", "base_ang_vel"), (slice(0, 3), slice(3, 6)), strict=True)
        )
        _index, sign = MIRROR_OBSERVATION["critic"]
        for term, polar in (("base_lin_vel", REFLECT), ("base_ang_vel", -REFLECT)):
            np.testing.assert_allclose(
                trunk_from_imu.T @ polar @ trunk_from_imu,
                np.diag(sign[layout[term]].numpy()),
                atol=1e-9,
                err_msg=term,
            )

    def test_a_symmetric_state_is_its_own_mirror(self) -> None:
        # HOME, a pitch rate (imu x = trunk y), gravity in the sagittal
        # plane, a straight twist, mirror-image foot and arm targets.
        home = torch.tensor([HOME.joint_pos_rad[name] for name in JOINTS])
        arms = {"left_shoulder_pitch": 0.4, "left_shoulder_roll": 0.7, "left_elbow": -0.5}
        arms |= {mirror_partner(n): mirror_joint_sign(n) * v for n, v in list(arms.items())}
        actions = home[3:] * 0.2
        actor = torch.cat(
            [
                torch.tensor([0.3, 0.0, 0.0]),
                torch.tensor([0.17, 0.0, -0.98]),
                home * 0.1,
                home * -0.3,
                actions,
                torch.tensor([0.2, 0.0, 0.0]),
                torch.tensor([0.01, 0.02, 0.03, 0.01, -0.02, 0.03]),
                torch.tensor([arms[name] for name in PICO_ARM_JOINT_NAMES]),
            ]
        ).unsqueeze(0)
        obs = TensorDict({"actor": actor}, batch_size=[1])
        torch.testing.assert_close(mirror_observations(obs)["actor"], actor, rtol=0, atol=0)
        torch.testing.assert_close(mirror_actions(actions), actions, rtol=0, atol=0)


def _actor() -> LegacyAdapterTeleopActor:
    torch.manual_seed(7)
    legacy = MLPModel(
        obs=TensorDict({"actor": torch.zeros(1, 63)}, batch_size=[1]),
        obs_groups={"actor": ["actor"]},
        obs_set="actor",
        output_dim=18,
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg=dict(GAUSSIAN),
    )
    actor = LegacyAdapterTeleopActor(
        obs=TensorDict({"actor": torch.zeros(1, 81)}, batch_size=[1]),
        obs_groups={"actor": ["actor"]},
        obs_set="actor",
        output_dim=18,
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg=dict(GAUSSIAN),
    )
    actor.load_state_dict(
        transplant_legacy_actor_state_to_teleop(
            legacy.state_dict(), actor.state_dict(), LEGACY_TO_TELEOP_OBSERVATION_INDEX
        ),
        strict=True,
    )
    actor.bind_frozen_legacy_reference()
    actor.bind_common_step_provider(lambda: PICO_SCHEDULE["arm"] * 24 + 24)
    return actor


class MirrorLossTest(unittest.TestCase):
    def test_the_mirror_loss_covers_the_leg_outputs_only(self) -> None:
        self.assertEqual(
            [MICROBAN_TELEOP_ACTION_JOINT_NAMES[i] for i in TELEOP_MIRROR_LOSS_ACTION_COLUMNS],
            [n for n in MICROBAN_TELEOP_ACTION_JOINT_NAMES if "hip" in n or "knee" in n or "ankle" in n],
        )
        envs, steps = 8, 4
        obs = _obs(envs, torch.Generator().manual_seed(4))
        actor = _actor()
        critic = MLPModel(obs, {"critic": ["critic"]}, "critic", 1, hidden_dims=(32,))
        symmetry = {
            "use_data_augmentation": False,
            "use_mirror_loss": True,
            "mirror_loss_coeff": 0.1,
            "data_augmentation_func": mirror_augmentation,
            "_env": None,
        }
        algorithm = LegacyAdapterPPO(
            actor,
            critic,
            RolloutStorage("rl", envs, steps, obs, [18], "cpu"),
            num_mini_batches=1,
            num_learning_epochs=1,
            learning_rate=0.0,
            schedule="fixed",
            symmetry_cfg=symmetry,
        )
        for _ in range(steps):
            algorithm.act(obs)
            algorithm.process_env_step(obs, torch.randn(envs), torch.zeros(envs), {})
        algorithm.compute_returns(obs)
        loss = algorithm.update()["symmetry"]

        self.assertIsNone(actor.mirror_loss_columns)
        with torch.no_grad():
            error = actor(mirror_observations(obs)) - mirror_actions(actor(obs))
        legs = list(TELEOP_MIRROR_LOSS_ACTION_COLUMNS)
        self.assertAlmostEqual(loss, float(error[:, legs].square().sum(-1).mean()) / 18, places=5)
        # The random walker's arm outputs are asymmetric too: they would count.
        self.assertGreater(float(error.square().mean()), 1.2 * loss)

    def test_the_task_trains_with_the_mirror_loss(self) -> None:
        symmetry = MicrobanTeleopV12RlCfg.algorithm.symmetry_cfg
        self.assertTrue(symmetry.use_mirror_loss)
        self.assertFalse(symmetry.use_data_augmentation)
        self.assertGreater(symmetry.mirror_loss_coeff, 0.0)


if __name__ == "__main__":
    unittest.main()
