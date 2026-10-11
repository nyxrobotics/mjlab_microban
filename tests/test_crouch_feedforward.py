"""The PICO crouch offsets (robot/crouch_feedforward.py) and their use in the task."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import mujoco
import numpy as np
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp as velocity_mdp

from mjlab_microban import policy_contract
from mjlab_microban.robot import crouch_feedforward as cf
from mjlab_microban.robot.home_pose import HOME, MICROBAN_XML
from mjlab_microban.tasks import microban_teleop_mdp as mdp
from mjlab_microban.tasks.microban_teleop_mirror import _TERM_MIRRORS, _joints
from mjlab_microban.tasks.microban_teleop_v13_arm_overlay import (
    MICROBAN_TELEOP_LEG_JOINT_NAMES,
    make_microban_teleop_v13_arm_overlay_env_cfg,
)

TABLE = cf.build_table(HOME.joint_pos_rad, HOME.trunk_pitch_rad)
HOME_LEGS = np.array([HOME.joint_pos_rad[name] for name in cf.LEG_JOINT_NAMES])


def _both(d) -> np.ndarray:
    d = np.asarray(d, dtype=np.float64).reshape(-1, 3)
    return np.hstack((d, d))


class KinematicsTest(unittest.TestCase):
    def test_the_leg_chains_are_robot_xml(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(MICROBAN_XML))
        data = mujoco.MjData(model)
        rng = np.random.default_rng(0)
        for side in ("right", "left"):
            names = [name for name in cf.LEG_JOINT_NAMES if name.startswith(side)]
            for q in rng.uniform(-1.0, 1.0, (8, 6)):
                data.qpos[:] = 0.0
                data.qpos[3] = 1.0
                for name, value in zip(names, q):
                    data.qpos[model.joint(name).qposadr[0]] = value
                mujoco.mj_kinematics(model, data)
                site = model.site(f"{side}_foot").id
                pos, rot, _, _ = cf.leg_fk(side, q[None])
                np.testing.assert_allclose(pos[0], data.site_xpos[site] - data.qpos[:3], atol=1e-12)
                np.testing.assert_allclose(rot[0], data.site_xmat[site].reshape(3, 3), atol=1e-12)

    def test_the_crouch_keeps_both_soles_and_lowers_the_trunk(self) -> None:
        """MuJoCo FK of HOME + offsets with the root moved by -(dx, dy, dz)
        (the HOME-levelled frame is the world's at HOME): both foot sites stay
        at their HOME pose, at grid nodes and between them."""

        model = mujoco.MjModel.from_xml_path(str(MICROBAN_XML))
        data = mujoco.MjData(model)

        def feet(root_shift, legs):
            data.qpos[:] = 0.0
            data.qpos[0:3] = np.asarray(HOME.root_pos) + root_shift
            data.qpos[3:7] = HOME.root_quat_wxyz
            for name, value in HOME.joint_pos_rad.items():
                data.qpos[model.joint(name).qposadr[0]] = value
            for name, value in zip(cf.LEG_JOINT_NAMES, legs):
                data.qpos[model.joint(name).qposadr[0]] = value
            mujoco.mj_kinematics(model, data)
            sites = [model.site(f"{side}_foot").id for side in ("left", "right")]
            return data.site_xpos[sites].copy(), data.site_xmat[sites].copy(), data.qpos[2]

        home_pos, home_rot, home_z = feet(np.zeros(3), HOME_LEGS)
        rng = np.random.default_rng(1)
        targets = np.vstack((
            [[0.0, 0.0, 0.008], [0.0, 0.0, 0.016], [0.008, 0.008, 0.016], [-0.008, -0.008, 0.016]],
            np.column_stack((rng.uniform(-0.01, 0.01, (40, 2)), rng.uniform(0.0026, 0.02, 40))),
        ))
        offsets = cf.leg_offsets(TABLE, _both(targets))
        for d, offset in zip(targets, offsets):
            pos, rot, z = feet(-d, HOME_LEGS + offset)
            np.testing.assert_allclose(pos, home_pos, atol=5e-5)  # 0.05 mm
            np.testing.assert_allclose(rot, home_rot, atol=2e-3)
            self.assertAlmostEqual(home_z - z, d[2], delta=1e-12)
        # The knee bends (16 mm: about 0.84 rad), hip and ankle pitch take the rest.
        knee = cf.LEG_JOINT_NAMES.index("right_knee")
        self.assertGreater(offsets[1, knee], 0.8)

    def test_the_knees_bend_forward_and_more_with_dz(self) -> None:
        """Every node's knee angle (HOME + offset) is >= 0 and grows with dz."""

        for side in ("left", "right"):
            column = cf.LEG_JOINT_NAMES.index(f"{side}_knee")
            knee = HOME.joint_pos_rad[f"{side}_knee"] + TABLE["values"][..., column]
            self.assertTrue(np.all(knee >= 0.0), side)
            self.assertTrue(np.all(np.diff(knee, axis=2) > 0.0), side)

    def test_a_backward_knee_is_refused(self) -> None:
        backward = dict(HOME.joint_pos_rad)
        for side in ("left", "right"):
            backward[f"{side}_knee"] = -1.2  # the solver starts 0.6 rad bent: still backward
        with self.assertRaisesRegex(RuntimeError, "backward knee"):
            cf.solve_offsets(backward, HOME.trunk_pitch_rad, [[0.0, 0.0, 0.004]])

    def test_the_table_matches_the_exact_solution(self) -> None:
        rng = np.random.default_rng(2)
        targets = np.column_stack((rng.uniform(-0.01, 0.01, (500, 2)), rng.uniform(0.0026, 0.02, 500)))
        exact = cf.solve_offsets(HOME.joint_pos_rad, HOME.trunk_pitch_rad, targets)
        np.testing.assert_allclose(cf.leg_offsets(TABLE, _both(targets)), exact, atol=3e-3)


class RowsTest(unittest.TestCase):
    def test_zero_without_a_two_foot_target(self) -> None:
        feet = np.array([
            [0.0] * 6,  # no target (standing or walking)
            [0.01, 0.0, 0.03, 0.0, 0.0, 0.0],  # left foot up
            [0.0, 0.0, 0.0, 0.0, -0.01, 0.05],  # right foot up
            [0.0, 0.0, 0.0025, 0.0, 0.0, 0.0025],  # both at the floor band
            [0.0, 0.0, 0.016, 0.0, 0.0, 0.0],
        ])
        np.testing.assert_array_equal(cf.leg_offsets(TABLE, feet), np.zeros((5, 12)))

    def test_two_feet_take_the_mean_xy_and_the_lower_height(self) -> None:
        feet = np.array([[0.004, -0.002, 0.012, 0.002, 0.0, 0.02]])
        np.testing.assert_allclose(
            cf.leg_offsets(TABLE, feet), cf.leg_offsets(TABLE, _both([0.003, -0.001, 0.012])), atol=1e-15
        )
        # Continuous while one foot leaves a crouch: the lower foot sets it.
        going_up = np.array([[0.0, 0.0, 0.012, 0.0, 0.0, 0.012], [0.0, 0.0, 0.03, 0.0, 0.0, 0.012]])
        offsets = cf.leg_offsets(TABLE, going_up)
        np.testing.assert_allclose(offsets[0], offsets[1], atol=1e-15)

    def test_torch_and_numpy_give_the_same_offsets(self) -> None:
        rng = np.random.default_rng(3)
        feet = np.hstack((rng.uniform(-0.012, 0.012, (64, 2)), rng.uniform(0.0, 0.025, (64, 1)),
                          rng.uniform(-0.012, 0.012, (64, 2)), rng.uniform(0.0, 0.025, (64, 1))))
        table = {key: torch.as_tensor(value) for key, value in TABLE.items()}
        torch_offsets = cf.leg_offsets(table, torch.as_tensor(feet)).numpy()
        np.testing.assert_allclose(torch_offsets, cf.leg_offsets(TABLE, feet), atol=1e-12)

    def test_mirror_symmetric(self) -> None:
        """The offsets of the mirrored targets are the mirrored offsets."""

        rng = np.random.default_rng(4)
        feet = np.hstack((rng.uniform(-0.01, 0.01, (64, 2)), rng.uniform(0.003, 0.02, (64, 1))))
        feet = np.hstack((feet, feet + rng.uniform(-0.002, 0.002, (64, 3))))
        foot_index, foot_sign = _TERM_MIRRORS["foot_target"]
        leg_index, leg_sign = _joints(cf.LEG_JOINT_NAMES)
        mirrored = cf.leg_offsets(TABLE, feet[:, foot_index] * foot_sign)
        np.testing.assert_allclose(mirrored, cf.leg_offsets(TABLE, feet)[:, leg_index] * leg_sign, atol=1e-9)

    def test_the_contract_record_reproduces_its_probes(self) -> None:
        record = policy_contract.pico_crouch_record()
        self.assertEqual(record["contract"], "microban_pico_crouch_feedforward_v1")
        self.assertEqual(record["joint_names"], list(MICROBAN_TELEOP_LEG_JOINT_NAMES))
        offsets = np.asarray(record["leg_offsets_rad"])
        np.testing.assert_allclose(offsets, cf.leg_offsets(TABLE, np.asarray(record["foot_targets"])))
        self.assertTrue(np.any(offsets[:5] != 0.0))
        np.testing.assert_array_equal(offsets[5:7], np.zeros((2, 12)))


class TaskTest(unittest.TestCase):
    def test_the_task_offsets_actions_and_actor_joints(self) -> None:
        cfg = make_microban_teleop_v13_arm_overlay_env_cfg()
        self.assertEqual(cfg.actions["joint_pos"].crouch_command_name, "foot_target")
        self.assertIs(cfg.observations["actor"].terms["joint_pos"].func, mdp.crouch_joint_pos_rel)
        self.assertEqual(cfg.observations["actor"].terms["joint_pos"].params["command_name"], "foot_target")
        # The critic keeps the measured joints; the previous action stays the raw output.
        self.assertIs(cfg.observations["critic"].terms["joint_pos"].func, velocity_mdp.joint_pos_rel)
        self.assertIs(cfg.observations["actor"].terms["actions"].func, velocity_mdp.last_action)

    def test_the_action_and_the_observation_use_the_same_offsets(self) -> None:
        joint_names = ["head", "neck_roll", "neck_pitch", *policy_contract.ACTION_JOINT_NAMES]
        n = 4
        foot = torch.tensor([
            [0.0, 0.0, 0.016, 0.0, 0.0, 0.016],
            [0.008, 0.008, 0.016, 0.008, 0.008, 0.016],
            [0.0, 0.0, 0.03, 0.0, 0.0, 0.0],
            [0.0] * 6,
        ])
        offsets = mdp.crouch_leg_offsets(
            SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: SimpleNamespace(command=foot))),
            "foot_target",
        )
        self.assertTrue(bool((offsets[:2] != 0.0).any()) and bool((offsets[2:] == 0.0).all()))
        default = torch.tensor([[HOME.joint_pos_rad[name] for name in joint_names]]).repeat(n, 1)
        leg_ids = [joint_names.index(name) for name in cf.LEG_JOINT_NAMES]
        joint_pos = default.clone()
        joint_pos[:, leg_ids] += offsets + 0.01
        robot = SimpleNamespace(
            joint_names=joint_names,
            data=SimpleNamespace(joint_pos=joint_pos, default_joint_pos=default),
        )
        env = SimpleNamespace(
            scene={"robot": robot},
            device="cpu",
            command_manager=SimpleNamespace(get_term=lambda name: SimpleNamespace(command=foot)),
        )
        asset_cfg = SceneEntityCfg("robot", joint_names=(r".*",))
        asset_cfg.joint_ids = slice(None)
        term = mdp.crouch_joint_pos_rel(SimpleNamespace(params={"asset_cfg": asset_cfg}), env)
        observed = term(env, asset_cfg=asset_cfg, command_name="foot_target")
        torch.testing.assert_close(observed[:, leg_ids], torch.full((n, 12), 0.01))
        torch.testing.assert_close(observed[:, :3], torch.zeros(n, 3))

        # The action: legs HOME + offsets + raw, arms the overlay, raw kept.
        action_names = list(policy_contract.ACTION_JOINT_NAMES)
        action = object.__new__(mdp.PicoArmOverlayJointPositionAction)
        action.cfg = SimpleNamespace(crouch_command_name="foot_target", clip=True)
        action._env = env
        action._raw_actions = torch.zeros(n, 18)
        action._scale = 1.0
        action._offset = torch.tensor([[HOME.joint_pos_rad[name] for name in action_names]]).repeat(n, 1)
        action._clip = torch.tensor([-torch.pi, torch.pi]).repeat(n, 18, 1)
        action.arm_columns = torch.tensor([action_names.index(name) for name in policy_contract.PICO_ARM_JOINT_NAMES])
        action.arm_target_rad = action._offset[:, action.arm_columns].clone()
        action.arm_goal_rad = action.arm_target_rad.clone()
        action.arm_max_step_rad = 0.08
        action.arm_lower_rad = torch.full((1, 6), -3.0)
        action.arm_upper_rad = torch.full((1, 6), 3.0)
        action.crouch_columns = torch.tensor([action_names.index(name) for name in cf.LEG_JOINT_NAMES])
        raw = torch.linspace(-0.1, 0.1, 18).repeat(n, 1)
        action.process_actions(raw)
        legs = action.crouch_columns
        torch.testing.assert_close(
            action._processed_actions[:, legs], action._offset[:, legs] + offsets + raw[:, legs]
        )
        torch.testing.assert_close(action._raw_actions, raw)
        # Both are relative to the same pose: target - raw = measured - observed = HOME + offsets.
        torch.testing.assert_close(
            action._processed_actions[:, legs] - raw[:, legs], joint_pos[:, leg_ids] - observed[:, leg_ids]
        )


if __name__ == "__main__":
    unittest.main()
