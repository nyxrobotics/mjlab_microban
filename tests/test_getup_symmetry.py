"""Tests for the get-up left/right mirror augmentation (microban_getup_symmetry)."""

from __future__ import annotations

import unittest

import mujoco
import numpy as np
import torch
from tensordict import TensorDict

from mjlab_microban.tasks.microban_getup_symmetry import (
    build_action_mirror,
    build_group_mirrors,
    getup_symmetry_augmentation,
    mirror_joint_partner,
)

_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
_K = 4  # random states; envs K..2K-1 hold their mirrors


def _make_env(imu_delay_max_lag: int = 0):
    from mjlab.envs import ManagerBasedRlEnv

    from mjlab_microban.tasks.microban_getup_env_cfg import make_microban_getup_env_cfg

    cfg = make_microban_getup_env_cfg(reward_set="posture", imu_delay_max_lag=imu_delay_max_lag)
    cfg.scene.num_envs = 2 * _K
    cfg.curriculum = {}
    for name in list(cfg.events):
        if cfg.events[name].mode in ("startup", "interval"):
            del cfg.events[name]
    for group in cfg.observations.values():
        group.enable_corruption = False
    return ManagerBasedRlEnv(cfg=cfg, device=_DEVICE)


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


class MirrorMapTest(unittest.TestCase):
    def test_partner_is_an_involution(self) -> None:
        for name in ("right_hip_roll", "left_knee", "robot/right_ankle_roll", "head", "neck_pitch"):
            partner, sign = mirror_joint_partner(name)
            back, sign_back = mirror_joint_partner(partner)
            self.assertEqual(back, name)
            self.assertEqual(sign, sign_back)


class GetupSymmetrySimTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.env = _make_env()
        cls.env.reset()
        cls.robot = cls.env.scene["robot"]
        cls.group_mirrors = build_group_mirrors(cls.env)
        cls.action_mirror = build_action_mirror(cls.env)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    # (a) mirror(mirror(x)) == x, batch layout of the augmentation function.
    def test_involution_and_augmentation_layout(self) -> None:
        gen = torch.Generator(device=_DEVICE).manual_seed(0)
        om = self.env.observation_manager
        obs = TensorDict(
            {g: torch.randn(16, om.group_obs_dim[g][0], generator=gen, device=_DEVICE) for g in om.active_terms},
            batch_size=[16],
        )
        act = torch.randn(16, 18, generator=gen, device=_DEVICE)
        for g, mirror in self.group_mirrors.items():
            torch.testing.assert_close(mirror(mirror(obs[g])), obs[g], rtol=0, atol=0)
            self.assertFalse(torch.equal(mirror(obs[g]), obs[g]))
        torch.testing.assert_close(self.action_mirror(self.action_mirror(act)), act, rtol=0, atol=0)
        obs_aug, act_aug = getup_symmetry_augmentation(env=self.env, obs=obs, actions=act)
        self.assertEqual(obs_aug.batch_size[0], 32)
        self.assertEqual(tuple(act_aug.shape), (32, 18))
        for g in om.active_terms:
            torch.testing.assert_close(obs_aug[g][:16], obs[g])
            torch.testing.assert_close(obs_aug[g][16:], self.group_mirrors[g](obs[g]))
        torch.testing.assert_close(act_aug[16:], self.action_mirror(act))
        none_obs, only_act = getup_symmetry_augmentation(env=self.env, obs=None, actions=act)
        self.assertIsNone(none_obs)
        self.assertEqual(tuple(only_act.shape), (32, 18))

    # (b1) Kinematics: mirrored joint configurations reflect every body COM and
    # collision-geom centre across the trunk x-z plane.
    def test_kinematic_mirror(self) -> None:
        m = self.env.sim.mj_model
        d = mujoco.MjData(m)
        names = list(self.robot.joint_names)
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "robot/" + n) for n in names]
        qadr = [m.jnt_qposadr[j] for j in jid]
        body_name = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)]
        robot_bodies = [b for b in range(m.nbody) if body_name[b].startswith("robot/")]
        trunk = body_name.index("robot/trunk")
        bpair = {b: b for b in robot_bodies}
        for n, j in zip(names, jid, strict=True):
            p, _ = mirror_joint_partner(n)
            bpair[m.jnt_bodyid[j]] = m.jnt_bodyid[jid[names.index(p)]]
        coll = [g for g in range(m.ngeom) if m.geom_bodyid[g] in bpair and (m.geom_contype[g] or m.geom_conaffinity[g])]
        gpair = {}
        for b in robot_bodies:
            ga = [g for g in coll if m.geom_bodyid[g] == b]
            gb = [g for g in coll if m.geom_bodyid[g] == bpair[b]]
            self.assertEqual(len(ga), len(gb))
            gpair.update(zip(ga, gb, strict=True))
        refl = np.diag([1.0, -1.0, 1.0])

        def points(q):
            d.qpos[:] = 0.0
            d.qpos[3] = 1.0
            for a, v in zip(qadr, q, strict=True):
                d.qpos[a] = v
            mujoco.mj_kinematics(m, d)
            rot, p0 = d.xmat[trunk].reshape(3, 3), d.xpos[trunk]
            com = {b: rot.T @ (d.xipos[b] - p0) for b in robot_bodies}
            geo = {g: rot.T @ (d.geom_xpos[g] - p0) for g in coll}
            return com, geo

        perm = [names.index(mirror_joint_partner(n)[0]) for n in names]
        sign = np.array([mirror_joint_partner(n)[1] for n in names], dtype=float)
        rng = np.random.default_rng(0)
        worst_com = worst_geom = 0.0
        for _ in range(100):
            q = np.array([rng.uniform(*m.jnt_range[j]) for j in jid])
            qm = sign * q[perm]
            c1, g1 = points(q)
            c2, g2 = points(qm)
            worst_com = max(worst_com, max(np.linalg.norm(refl @ c1[b] - c2[bpair[b]]) for b in robot_bodies))
            worst_geom = max(worst_geom, max(np.linalg.norm(refl @ g1[g] - g2[gpair[g]]) for g in coll))
        # 0.96 mm is a static 0.5 mm lateral COM offset of roll_pitch_link.
        self.assertLess(worst_com, 1.0e-3)
        self.assertLess(worst_geom, 1.0e-4)

    # (b2) Observations of a physically mirrored state equal the mirrored observations.
    def test_physical_observation_mirror(self) -> None:
        env, robot = self.env, self.robot
        gen = torch.Generator(device=_DEVICE).manual_seed(1)

        def rnd(*shape, scale=1.0):
            return (torch.rand(*shape, generator=gen, device=_DEVICE) * 2 - 1) * scale

        names = list(robot.joint_names)
        jperm = torch.tensor([names.index(mirror_joint_partner(n)[0]) for n in names], device=_DEVICE)
        jsign = torch.tensor([float(mirror_joint_partner(n)[1]) for n in names], device=_DEVICE)
        lo = robot.data.joint_pos_limits[0, :, 0] if hasattr(robot.data, "joint_pos_limits") else None
        default = robot.data.default_joint_pos[:_K]
        q = default + rnd(_K, len(names), scale=0.6)
        if lo is not None:
            q = torch.clamp(q, robot.data.joint_pos_limits[:_K, :, 0], robot.data.joint_pos_limits[:_K, :, 1])
        qd = rnd(_K, len(names), scale=3.0)
        quat = torch.nn.functional.normalize(rnd(_K, 4), dim=-1)
        lin = rnd(_K, 3, scale=1.0)
        ang = rnd(_K, 3, scale=3.0)
        origins = env.scene.env_origins
        pos = origins[:_K] + torch.tensor([0.0, 0.0, 0.4], device=_DEVICE) + rnd(_K, 3, scale=0.1)

        flip = torch.tensor([1.0, -1.0, 1.0], device=_DEVICE)
        pos_m = origins[_K:] + (pos - origins[:_K]) * flip
        quat_m = quat * torch.tensor([1.0, -1.0, 1.0, -1.0], device=_DEVICE)  # M R M, M = diag(1,-1,1)
        lin_m = lin * flip
        ang_m = -ang * flip  # pseudo-vector
        root = torch.cat([torch.cat([pos, quat, lin, ang], -1), torch.cat([pos_m, quat_m, lin_m, ang_m], -1)])
        joint_pos = torch.cat([q, q[:, jperm] * jsign])
        joint_vel = torch.cat([qd, qd[:, jperm] * jsign])
        robot.write_root_state_to_sim(root)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        term = env.action_manager.get_term("joint_pos")
        raw = rnd(_K, 18, scale=1.0)
        term._raw_actions[:] = torch.cat([raw, self.action_mirror(raw)])
        env.sim.forward()
        env.scene.update(dt=0.0) if hasattr(env.scene, "update") else None
        obs = env.observation_manager.compute(update_history=True)

        om = env.observation_manager
        for group in om.active_terms:
            mirrored = self.group_mirrors[group](obs[group][:_K])
            actual = obs[group][_K:]
            offset = 0
            for name, dim in zip(om.active_terms[group], om.group_obs_term_dim[group], strict=True):
                sl = slice(offset, offset + dim[0])
                offset += dim[0]
                # The IMU site sits 2 mm off the mirror plane: imu_lin_vel carries
                # a |w x 4 mm| mismatch (<= 0.021 m/s at |w| <= 5.2 rad/s).
                atol = 0.025 if name == "base_lin_vel" else 1.0e-4
                err = (mirrored[:, sl] - actual[:, sl]).abs().max().item()
                self.assertLess(err, atol, f"{group}/{name}: max mirror mismatch {err:.3g}")


if __name__ == "__main__":
    unittest.main()
