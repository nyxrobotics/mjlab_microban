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

    def test_the_trunk_shift_moves_the_centre_of_mass_by_the_support_reward_share(self) -> None:
        """support_zmp moves its level point by -share (dx, dy) of a two-foot
        target: the crouch pose's centre of mass from the soles' midpoint is
        the straight-down crouch's less share (dx, dy), and HOME's is on it."""

        from mjlab_microban.robot.home_pose import _pose_model
        from mjlab_microban.tasks.microban_teleop_env_cfg import MICROBAN_TELEOP_TRUNK_SHIFT_COM_SHARE

        pose = _pose_model(str(MICROBAN_XML.resolve()))

        def com_from_midpoint(goal):
            joints = dict(HOME.joint_pos_rad)
            if any(goal):
                offsets = cf.solve_offsets(HOME.joint_pos_rad, HOME.trunk_pitch_rad, np.asarray([goal]))[0]
                for name, value in zip(cf.LEG_JOINT_NAMES, offsets):
                    joints[name] += value
            pose.set_pose(joints, HOME.trunk_pitch_rad)
            soles = [pose.data.geom_xpos[pose.sole_geom_ids[side][0]][:2] for side in ("left", "right")]
            return pose.data.subtree_com[pose.trunk_id][:2] - 0.5 * (soles[0] + soles[1])

        np.testing.assert_allclose(com_from_midpoint((0.0, 0.0, 0.0)), 0.0, atol=1.0e-4)
        for dz in (0.004, 0.012, 0.02):
            straight = com_from_midpoint((0.0, 0.0, dz))
            self.assertLess(np.linalg.norm(straight), 0.002)  # knees forward: under 2 mm
            for dx, dy in ((0.01, 0.0), (0.0, -0.01), (-0.008, 0.008), (0.01, 0.01)):
                with self.subTest(dx=dx, dy=dy, dz=dz):
                    shift = com_from_midpoint((dx, dy, dz)) - straight
                    np.testing.assert_allclose(shift, -MICROBAN_TELEOP_TRUNK_SHIFT_COM_SHARE * np.array([dx, dy]),
                                               atol=2.0e-4)

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
            # Both up, one beyond the two-foot support (passing between two
            # single-foot targets): no crouch, as the robot drops such targets.
            [0.0, 0.0, 0.0284, 0.0, 0.0, 0.0216],
            [0.0, 0.0, 0.025, 0.0, 0.0, 0.015],
            [0.012, 0.0, 0.01, 0.0, 0.0, 0.01],
            [0.0, 0.0, 0.01, 0.0, -0.0105, 0.01],
        ])
        np.testing.assert_array_equal(cf.leg_offsets(TABLE, feet), np.zeros((9, 12)))
        # The support's edge is still inside.
        edge = np.array([[0.01, -0.01, 0.02] * 2])
        self.assertTrue(np.any(cf.leg_offsets(TABLE, edge) != 0.0))

    def test_two_feet_take_the_mean_xy_and_the_lower_height(self) -> None:
        feet = np.array([[0.004, -0.002, 0.012, 0.002, 0.0, 0.02]])
        np.testing.assert_allclose(
            cf.leg_offsets(TABLE, feet), cf.leg_offsets(TABLE, _both([0.003, -0.001, 0.012])), atol=1e-15
        )
        # Continuous while one foot leaves a crouch: the lower foot sets it.
        going_up = np.array([[0.0, 0.0, 0.012, 0.0, 0.0, 0.012], [0.0, 0.0, 0.018, 0.0, 0.0, 0.012]])
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
        self.assertEqual(record["contract"], "microban_pico_crouch_feedforward_v2")
        self.assertEqual((record["policy_dt_s"], record["max_step_rad"]), (0.02, 0.2))
        self.assertEqual(record["joint_names"], list(MICROBAN_TELEOP_LEG_JOINT_NAMES))
        offsets = np.asarray(record["leg_offsets_rad"])
        np.testing.assert_allclose(offsets, cf.leg_offsets(TABLE, np.asarray(record["foot_targets"])))
        self.assertTrue(np.any(offsets[:5] != 0.0))
        np.testing.assert_array_equal(offsets[5:], np.zeros((len(offsets) - 5, 12)))


def _ramp(start, goal, steps=60):
    """Published foot targets (steps, 6) of each foot moved at the teleop's
    0.12 m/s (0.0024 m per step), then the floor band (as the gate's ramp)."""

    s = np.array(start, dtype=np.float64).reshape(2, 3)
    g = np.array(goal, dtype=np.float64).reshape(2, 3)
    rows = []
    for _ in range(steps):
        rows.append(np.where(s[:, 2:3] <= cf.FLOOR_BAND_M, 0.0, s).reshape(6))
        d = g - s
        n = np.linalg.norm(d, axis=1, keepdims=True)
        s = s + d * np.minimum(1.0, 0.0024 / np.maximum(n, 1e-12))
    return np.array(rows)


def _follow_all(feet, start=None):
    offsets = np.zeros((1, 12)) if start is None else start
    out = []
    for row in feet:
        offsets = cf.follow(offsets, cf.leg_offsets(TABLE, row[None]))
        out.append(offsets[0])
    return np.array(out)


class FollowTest(unittest.TestCase):
    def test_every_switch_is_a_bounded_ramp(self) -> None:
        """Into and out of a crouch, between one and two feet, across a swap of
        the lifted foot, and when the robot drops both targets at once, the
        offsets move by at most MAX_STEP_RAD per step."""

        z = [0.0] * 3
        crouch = [0.0, 0.0, 0.016] * 2
        cases = [
            _ramp(z + z, crouch),
            _ramp(crouch, z + z),
            _ramp([0.02, 0.0, 0.03] + z, crouch),
            _ramp(crouch, [0.02, 0.01, 0.03] + z),
            _ramp([0.0, 0.0, 0.05] + z, z + [0.0, 0.0, 0.05]),
            _ramp([0.03, 0.03, 0.05] + z, z + [-0.03, -0.03, 0.05]),
            np.vstack((_ramp(z + z, crouch)[:20], np.zeros((20, 6)))),  # dropped at once
        ]
        for feet in cases:
            offsets = _follow_all(feet)
            steps = np.linalg.norm(np.diff(np.vstack((np.zeros(12), offsets)), axis=0), axis=1)
            self.assertLessEqual(steps.max(), cf.MAX_STEP_RAD + 1e-12)
            # And it ends at the target of the last row.
            np.testing.assert_allclose(offsets[-1], cf.leg_offsets(TABLE, feet[-1:])[0], atol=1e-12)
        # A swap of a 50 mm lifted foot never crouches.
        np.testing.assert_array_equal(_follow_all(cases[4]), np.zeros((60, 12)))

    def test_a_held_crouch_is_reached_and_zero_stays_zero(self) -> None:
        feet = _ramp([0.0] * 6, [0.0, 0.0, 0.016] * 2)
        offsets = _follow_all(feet)
        target = cf.leg_offsets(TABLE, feet[-1:])[0]
        # The teleop's ramp arrives at 16 mm on step 7, the offsets two steps later.
        np.testing.assert_array_equal(feet[7], [0.0, 0.0, 0.016] * 2)
        self.assertGreater(np.linalg.norm(offsets[8] - target), 0.0)
        np.testing.assert_allclose(offsets[9:], np.broadcast_to(target, offsets[9:].shape), atol=1e-12)
        np.testing.assert_array_equal(_follow_all(np.zeros((5, 6))), np.zeros((5, 12)))
        decayed = _follow_all(np.zeros((10, 6)), start=target[None])
        np.testing.assert_array_equal(decayed[-1], np.zeros(12))

    def test_torch_and_numpy_follow_alike(self) -> None:
        rng = np.random.default_rng(5)
        offsets, target = rng.uniform(-1.0, 1.0, (2, 64, 12)) * rng.uniform(0.0, 0.3, (2, 64, 1))
        np.testing.assert_allclose(
            cf.follow(torch.as_tensor(offsets), torch.as_tensor(target)).numpy(),
            cf.follow(offsets, target),
            atol=1e-12,
        )


class TaskTest(unittest.TestCase):
    def test_the_task_offsets_actions_and_actor_joints(self) -> None:
        cfg = make_microban_teleop_v13_arm_overlay_env_cfg()
        self.assertEqual(cfg.actions["joint_pos"].crouch_command_name, "foot_target")
        self.assertIs(cfg.observations["actor"].terms["joint_pos"].func, mdp.crouch_joint_pos_rel)
        self.assertEqual(cfg.observations["actor"].terms["joint_pos"].params["action_name"], "joint_pos")
        # The critic keeps the measured joints; the previous action stays the raw output.
        self.assertIs(cfg.observations["critic"].terms["joint_pos"].func, velocity_mdp.joint_pos_rel)
        self.assertIs(cfg.observations["actor"].terms["actions"].func, velocity_mdp.last_action)

    def test_the_action_and_the_observation_use_the_same_offsets(self) -> None:
        """The observation after a step subtracts exactly the offsets the next
        action keeps and adds; they follow the target at the bounded rate."""

        joint_names = ["head", "neck_roll", "neck_pitch", *policy_contract.ACTION_JOINT_NAMES]
        n = 4
        foot = torch.tensor([
            [0.0, 0.0, 0.016, 0.0, 0.0, 0.016],
            [0.008, 0.008, 0.016, 0.008, 0.008, 0.016],
            [0.0, 0.0, 0.03, 0.0, 0.0, 0.0],
            [0.0] * 6,
        ])
        target = mdp.crouch_offsets(foot)
        self.assertTrue(bool((target[:2] != 0.0).any()) and bool((target[2:] == 0.0).all()))

        # The action: legs HOME + offsets + raw, arms the overlay, raw kept.
        action_names = list(policy_contract.ACTION_JOINT_NAMES)
        action = object.__new__(mdp.PicoArmOverlayJointPositionAction)
        action.cfg = SimpleNamespace(crouch_command_name="foot_target", clip=True)
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
        action.crouch_offsets_rad = torch.zeros(n, 12)

        default = torch.tensor([[HOME.joint_pos_rad[name] for name in joint_names]]).repeat(n, 1)
        leg_ids = [joint_names.index(name) for name in cf.LEG_JOINT_NAMES]
        robot = SimpleNamespace(
            joint_names=joint_names,
            data=SimpleNamespace(joint_pos=default.clone(), default_joint_pos=default),
        )
        env = SimpleNamespace(
            scene={"robot": robot},
            device="cpu",
            command_manager=SimpleNamespace(get_term=lambda name: SimpleNamespace(command=foot)),
            action_manager=SimpleNamespace(get_term=lambda name: action),
        )
        action._env = env
        asset_cfg = SceneEntityCfg("robot", joint_names=(r".*",))
        asset_cfg.joint_ids = slice(None)
        term = mdp.crouch_joint_pos_rel(SimpleNamespace(params={"asset_cfg": asset_cfg}), env)

        raw = torch.linspace(-0.1, 0.1, 18).repeat(n, 1)
        legs = action.crouch_columns
        previous = torch.zeros(n, 12)
        for _ in range(12):
            observed = term(env, asset_cfg=asset_cfg, action_name="joint_pos")
            action.process_actions(raw)
            used = action.crouch_offsets_rad
            # Observed relative to HOME + the offsets this action added.
            torch.testing.assert_close(robot.data.joint_pos[:, leg_ids] - observed[:, leg_ids], default[:, leg_ids] + used)
            torch.testing.assert_close(observed[:, :3], torch.zeros(n, 3))
            torch.testing.assert_close(action._processed_actions[:, legs], action._offset[:, legs] + used + raw[:, legs])
            torch.testing.assert_close(action._raw_actions, raw)
            self.assertLessEqual(float(torch.linalg.vector_norm(used - previous, dim=-1).max()), cf.MAX_STEP_RAD + 1e-6)
            previous = used.clone()
            robot.data.joint_pos[:, leg_ids] = default[:, leg_ids] + used + 0.01
        torch.testing.assert_close(action.crouch_offsets_rad, target)
        torch.testing.assert_close(term(env, asset_cfg=asset_cfg, action_name="joint_pos")[:, leg_ids], torch.full((n, 12), 0.01))


if __name__ == "__main__":
    unittest.main()
