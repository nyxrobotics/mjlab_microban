"""Ratio-keeping twist velocity reward (microban_twist_ratio_mdp)."""

from __future__ import annotations

import itertools
import math
import unittest
from types import SimpleNamespace

import torch

from mjlab_microban.tasks.microban_twist_ratio_mdp import (
    TWIST_RATIO_AXIS_SCALE,
    TWIST_RATIO_BASE,
    TWIST_RATIO_DIRECTION_PENALTY,
    TWIST_RATIO_MIN_COMMAND_NORM,
    twist_ratio,
    twist_ratio_reward,
    twist_ratio_velocity_reward,
)

SCALE = torch.tensor(TWIST_RATIO_AXIS_SCALE)
HOME_TRUNK_PITCH_RAD = math.radians(10.0)  # forward-lean HOME (any value works)
G = (0.7, 0.3, 1.5)  # infeasible diagonal: forward, left, counter-clockwise


def parts(commands, twists):
    return twist_ratio(
        torch.tensor(commands, dtype=torch.float32),
        torch.tensor(twists, dtype=torch.float32),
    )


def reward(commands, twists):
    return twist_ratio_reward(
        torch.tensor(commands, dtype=torch.float32),
        torch.tensor(twists, dtype=torch.float32),
        base=0.0,
    )


def scaled(command, k):
    return [k * value for value in command]


def unit_and_normal(command):
    """Unit command direction u and a unit normal w, in normalized units."""

    c_hat = torch.tensor(command, dtype=torch.float32) / SCALE
    u = c_hat / c_hat.norm()
    w = torch.linalg.cross(u, torch.tensor([0.0, 0.0, 1.0]))
    if float(w.norm()) < 1e-3:
        w = torch.linalg.cross(u, torch.tensor([1.0, 0.0, 0.0]))
    return c_hat, u, w / w.norm()


class DecompositionTest(unittest.TestCase):
    def test_default_axis_scale_is_the_moving_command_envelope(self) -> None:
        # Robot scale_velocity while translating: 0.7 m/s, 0.3 m/s, 1.5 rad/s.
        self.assertEqual(TWIST_RATIO_AXIS_SCALE, (0.7, 0.3, 1.5))

    def test_kept_ratio_costs_nothing_and_earns_its_scale(self) -> None:
        commands = [list(G), [-0.4, 0.2, -0.9], [0.3, -0.1, 0.0], [0.0, 0.25, -1.2]]
        for k in (0.0, 0.25, 0.5, 1.0):
            result = parts(commands, [scaled(c, k) for c in commands])
            self.assertTrue(torch.allclose(result.speed, torch.full((4,), k), atol=1e-5))
            self.assertTrue(torch.allclose(result.error, torch.zeros(4), atol=1e-5))

    def test_forward_only_on_a_diagonal_command_is_penalised(self) -> None:
        c_hat, u, _ = unit_and_normal(G)
        lean = [0.289, -0.017, 0.94]  # the lean final's answer to G under push
        result = parts([list(G)], [lean])
        v_hat = torch.tensor(lean) / SCALE
        along = float(v_hat @ u)
        self.assertAlmostEqual(float(result.speed[0]), along / float(c_hat.norm()), places=5)
        perpendicular = math.sqrt(float(v_hat @ v_hat) - along**2)
        self.assertAlmostEqual(float(result.error[0]), perpendicular, places=5)
        self.assertGreater(float(result.error[0]), 0.45)
        # The same along-command progress with the commanded ratio wins.
        kept = scaled(G, float(result.speed[0]))
        values = reward([list(G)] * 2, [lean, kept])
        self.assertGreater(float(values[1]), float(values[0]) + 0.2)

    def test_fast_motion_in_another_direction_earns_only_its_projection(self) -> None:
        c_hat, u, w = unit_and_normal(G)
        v_hat = 0.1 * u + math.sqrt(4.0 - 0.01) * w  # |v^| = 2, mostly perpendicular
        result = parts([list(G)], [(v_hat * SCALE).tolist()])
        self.assertAlmostEqual(float(result.speed[0]), 0.1 / float(c_hat.norm()), places=5)
        self.assertAlmostEqual(float(result.error[0]), math.sqrt(4.0 - 0.01), places=4)
        slow = parts([list(G)], [(0.2 * u * SCALE).tolist()])
        self.assertGreater(float(slow.speed[0]), float(result.speed[0]))
        self.assertAlmostEqual(float(slow.error[0]), 0.0, places=5)

    def test_overshoot_is_not_rewarded(self) -> None:
        commands = [list(G), [0.3, 0.0, 0.0], [0.0, -0.2, 0.0]]
        exact = reward(commands, commands)
        for k in (1.2, 2.0):
            over = parts(commands, [scaled(c, k) for c in commands])
            self.assertTrue(torch.allclose(over.speed, torch.ones(3)))
            self.assertTrue(
                torch.allclose(over.error, (k - 1.0) * over.command_norm, atol=1e-5)
            )
            self.assertTrue(bool((reward(commands, [scaled(c, k) for c in commands]) < exact).all()))

    def test_opposite_motion_earns_nothing_and_costs_at_least_perpendicular(self) -> None:
        for command in (G, (0.7, -0.3, -1.5), (-0.4, 0.25, 0.9), (0.3, 0.0, 0.0), (0.0, 0.0, -2.0)):
            _, u, w = unit_and_normal(command)
            k = 0.5
            rows = [(-k * u * SCALE).tolist(), (k * w * SCALE).tolist()]
            result = parts([list(command)] * 2, rows)
            self.assertEqual(float(result.speed[0]), 0.0)
            self.assertAlmostEqual(float(result.speed[1]), 0.0, places=5)
            self.assertAlmostEqual(float(result.error[0]), 2.0 * k, places=5)
            self.assertAlmostEqual(float(result.error[1]), k, places=5)
            self.assertGreaterEqual(float(result.error[0]), float(result.error[1]))
            values = reward([list(command)] * 2, rows)
            self.assertLess(float(values[0]), float(values[1]))

    def test_error_grows_and_speed_falls_with_the_angle_to_the_command(self) -> None:
        for command in (G, (0.3, 0.0, 0.0), (-0.3, 0.2, 0.0)):
            _, u, w = unit_and_normal(command)
            rows = [
                ((math.cos(a) * u + math.sin(a) * w) * 0.4 * SCALE).tolist()
                for a in (i * math.pi / 24 for i in range(25))
            ]
            result = parts([list(command)] * len(rows), rows)
            for previous, current in itertools.pairwise(result.error.tolist()):
                self.assertGreaterEqual(current + 1e-6, previous)
            for previous, current in itertools.pairwise(result.speed.tolist()):
                self.assertLessEqual(current, previous + 1e-6)

    def test_single_axis_commands_track_their_axis(self) -> None:
        for command in ([0.5, 0.0, 0.0], [-0.3, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, -1.0]):
            candidates = [
                command,
                scaled(command, 0.7),
                scaled(command, 1.3),
                [command[0] + 0.05, command[1] + 0.05, command[2]],
                [command[0], command[1], command[2] + 0.3],
            ]
            values = reward([command] * len(candidates), candidates)
            self.assertEqual(int(torch.argmax(values)), 0, command)
            self.assertAlmostEqual(float(values[0]), 1.0, places=5)

    def test_standing_and_small_commands(self) -> None:
        standing = parts([[0.0, 0.0, 0.0]] * 2, [[0.0, 0.0, 0.0], [0.07, -0.03, 0.3]])
        self.assertTrue(torch.equal(standing.speed, torch.zeros(2)))
        self.assertAlmostEqual(float(standing.error[0]), 0.0)
        self.assertAlmostEqual(float(standing.error[1]), math.sqrt(0.01 + 0.01 + 0.04), places=5)
        # A command below min_command_norm earns at most n / min_command_norm.
        small = [0.07, 0.0, 0.0]  # n = 0.1
        result = parts([small], [small])
        self.assertAlmostEqual(float(result.speed[0]), 0.1 / TWIST_RATIO_MIN_COMMAND_NORM, places=5)
        self.assertAlmostEqual(float(result.error[0]), 0.0, places=6)

    def test_yaw_is_part_of_the_ratio(self) -> None:
        command = [0.5, 0.0, 1.0]
        result = parts([command] * 3, [[0.5, 0.0, 0.0], [0.25, 0.0, 0.5], [0.0, 0.0, 1.0]])
        self.assertGreater(float(result.error[0]), 0.4)
        self.assertAlmostEqual(float(result.error[1]), 0.0, places=5)
        self.assertAlmostEqual(float(result.speed[1]), 0.5, places=5)
        self.assertGreater(float(result.error[2]), 0.4)

    def test_mirror_symmetry(self) -> None:
        def mirror(t):
            return [t[0], -t[1], -t[2]]

        commands = [list(G), [0.6, 0.3, 1.2], [-0.4, 0.25, 0.9], [0.0, 0.0, 0.0], [0.0, 0.2, 0.0]]
        twists = [[0.289, -0.017, 0.94], [0.2, 0.05, 0.3], [-0.1, -0.2, 0.4], [0.05, 0.02, 0.1], [0.0, 0.1, -0.2]]
        base = reward(commands, twists)
        mirrored = reward([mirror(c) for c in commands], [mirror(t) for t in twists])
        self.assertTrue(torch.allclose(base, mirrored, atol=1e-6))

    def test_best_reachable_twist_keeps_the_ratio_at_the_largest_scale(self) -> None:
        # A robot that reaches |v_x| <= 0.3, |v_y| <= 0.08, |w_z| <= 1.0 can do
        # at most 8/30 of G along the ratio (v_y binds).
        grid = torch.cartesian_prod(
            torch.linspace(-0.3, 0.3, 61),
            torch.linspace(-0.08, 0.08, 33),
            torch.linspace(-1.0, 1.0, 81),
        )
        values = twist_ratio_reward(torch.tensor([G]).expand(len(grid), 3), grid)
        best = grid[int(torch.argmax(values))]
        expected = torch.tensor(G) * (0.08 / 0.3)
        self.assertTrue(torch.allclose(best, expected, atol=0.011), best)

    def test_feasible_command_is_best_tracked_exactly(self) -> None:
        command = [0.3, 0.1, 0.6]
        grid = torch.cartesian_prod(
            torch.linspace(0.0, 0.5, 51),
            torch.linspace(-0.1, 0.3, 41),
            torch.linspace(0.0, 1.2, 61),
        )
        values = twist_ratio_reward(torch.tensor([command]).expand(len(grid), 3), grid)
        best = grid[int(torch.argmax(values))]
        self.assertTrue(torch.allclose(best, torch.tensor(command), atol=0.011), best)

    def test_toppling_along_the_command_earns_less_than_standing(self) -> None:
        command = torch.tensor([[0.5, 0.0, 0.0]] * 3)
        twist = torch.tensor([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
        # Falling forward (v_z -0.8 m/s, pitching 3 rad/s), walking with a
        # small bob and sway, standing still.
        uncommanded = torch.tensor([[-0.8, 0.0, 3.0], [0.05, 0.2, 0.2], [0.0, 0.0, 0.0]])
        values = twist_ratio_reward(command, twist, uncommanded=uncommanded, base=0.0)
        self.assertLess(float(values[0]), float(values[2]))
        self.assertAlmostEqual(float(values[2]), 0.0, places=6)
        self.assertGreater(float(values[1]), 0.75)
        bob = math.sqrt((0.05 / 0.7) ** 2 + 2 * (0.2 / 1.5) ** 2)
        self.assertAlmostEqual(float(values[1]), 1.0 - bob, places=5)
        # Without the uncommanded motion the fall would earn the full reward.
        planar = twist_ratio_reward(command, twist, base=0.0)
        self.assertAlmostEqual(float(planar[0]), 1.0, places=5)

    def test_uncommanded_motion_is_off_direction_motion(self) -> None:
        command = torch.tensor([list(G)])
        twist = torch.tensor([scaled(G, 0.4)])
        result = twist_ratio(command, twist, uncommanded=torch.tensor([[0.0, 0.0, 0.0]]))
        self.assertAlmostEqual(float(result.error[0]), 0.0, places=5)
        result = twist_ratio(command, twist, uncommanded=torch.tensor([[0.07, 0.0, -0.15]]))
        self.assertAlmostEqual(float(result.error[0]), math.sqrt(0.01 + 0.01), places=5)
        self.assertAlmostEqual(float(result.speed[0]), 0.4, places=5)
        with self.assertRaises(ValueError):
            twist_ratio(command, twist, uncommanded=torch.zeros(1, 3), uncommanded_scale=None)
        with self.assertRaises(ValueError):
            twist_ratio(command, twist, uncommanded=torch.zeros(1, 3), uncommanded_scale=(0.7, 1.5))

    def test_base_keeps_a_positive_reward_for_standing_upright(self) -> None:
        command = torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.1, 0.6], [0.3, 0.1, 0.6], list(G)])
        twist = torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.1, 0.6], [0.0, 0.0, 0.0], scaled(G, 0.3)])
        values = twist_ratio_reward(command, twist)
        self.assertEqual(TWIST_RATIO_BASE, 1.0)
        expected = [1.0, 2.0, 1.0, 1.3]
        for value, want in zip(values.tolist(), expected, strict=True):
            self.assertAlmostEqual(value, want, places=5)
        shifted = twist_ratio_reward(command, twist, base=0.0)
        self.assertTrue(torch.allclose(values - shifted, torch.ones(4)))

    def test_bad_inputs(self) -> None:
        with self.assertRaises(ValueError):
            twist_ratio(torch.zeros(2, 2), torch.zeros(2, 2))
        with self.assertRaises(ValueError):
            twist_ratio(torch.zeros(2, 3), torch.zeros(2, 3), axis_scale=(0.7, 0.0, 1.5))
        with self.assertRaises(ValueError):
            twist_ratio_reward(torch.zeros(1, 3), torch.zeros(1, 3), direction_penalty=0.0)


class RewardTermTest(unittest.TestCase):
    def test_env_term_reads_the_home_levelled_twist(self) -> None:
        # Robot at HOME (trunk pitched forward by HOME_TRUNK_PITCH_RAD) heading
        # +x: its world twist reads unchanged in the HOME-levelled frame.
        half = 0.5 * HOME_TRUNK_PITCH_RAD
        twist = torch.tensor([[0.289, -0.017, 0.94], [0.2, 0.0857, 0.4286]])
        data = SimpleNamespace(
            root_link_quat_w=torch.tensor([[math.cos(half), 0.0, math.sin(half), 0.0]] * 2),
            root_link_lin_vel_w=torch.stack((twist[:, 0], twist[:, 1], torch.zeros(2)), dim=-1),
            root_link_ang_vel_w=torch.stack((torch.zeros(2), torch.zeros(2), twist[:, 2]), dim=-1),
            root_link_lin_vel_b=None,
            root_link_ang_vel_b=None,
        )
        command = torch.tensor([list(G)] * 2)
        env = SimpleNamespace(
            scene={"robot": SimpleNamespace(data=data)},
            command_manager=SimpleNamespace(get_command=lambda name: command),
            extras={"log": {}},
        )
        value = twist_ratio_velocity_reward(env, trunk_pitch=HOME_TRUNK_PITCH_RAD)
        expected = twist_ratio_reward(command, twist)
        self.assertTrue(torch.allclose(value, expected, atol=1e-5))
        self.assertGreater(float(value[1]), float(value[0]))
        self.assertIn("Metrics/twist_ratio_speed", env.extras["log"])
        self.assertEqual(TWIST_RATIO_DIRECTION_PENALTY, 1.0)

    def test_untilted_frame_reads_the_body_twist(self) -> None:
        data = SimpleNamespace(
            root_link_lin_vel_b=torch.tensor([[0.2, 0.1, 0.05]]),
            root_link_ang_vel_b=torch.tensor([[0.3, -0.2, 0.6]]),
        )
        command = torch.tensor([[0.4, 0.2, 1.2]])
        env = SimpleNamespace(
            scene={"robot": SimpleNamespace(data=data)},
            command_manager=SimpleNamespace(get_command=lambda name: command),
        )
        # Half the command along its direction; the vertical velocity and the
        # roll/pitch rates (0.05 m/s, 0.3 and -0.2 rad/s) are uncommanded.
        value = twist_ratio_velocity_reward(env)
        uncommanded = math.sqrt((0.05 / 0.7) ** 2 + (0.3 / 1.5) ** 2 + (0.2 / 1.5) ** 2)
        self.assertAlmostEqual(float(value[0]), 1.5 - uncommanded, places=5)
        planar = twist_ratio_velocity_reward(env, uncommanded_scale=None)
        self.assertAlmostEqual(float(planar[0]), 1.5, places=5)

    def test_matches_the_task_home_levelled_velocity(self) -> None:
        # Same frame as mjlab_microban.tasks.mdp.home_levelled_root_*_vel_b
        # (skipped where that helper does not exist).
        try:
            from mjlab_microban.tasks.mdp import (
                home_levelled_root_ang_vel_b,
                home_levelled_root_lin_vel_b,
            )
        except ImportError:
            self.skipTest("no task home_levelled_root_*_vel_b helpers")
        from mjlab_microban.tasks.microban_twist_ratio_mdp import home_levelled_twist

        torch.manual_seed(0)
        quat = torch.nn.functional.normalize(torch.randn(8, 4), dim=-1)
        data = SimpleNamespace(
            root_link_quat_w=quat,
            root_link_lin_vel_w=torch.randn(8, 3),
            root_link_ang_vel_w=torch.randn(8, 3),
        )
        env = SimpleNamespace(scene={"robot": SimpleNamespace(data=data)})
        twist = home_levelled_twist(env, HOME_TRUNK_PITCH_RAD)
        linear = home_levelled_root_lin_vel_b(env, HOME_TRUNK_PITCH_RAD)
        angular = home_levelled_root_ang_vel_b(env, HOME_TRUNK_PITCH_RAD)
        self.assertTrue(torch.equal(twist[:, :2], linear[:, :2]))
        self.assertTrue(torch.equal(twist[:, 2], angular[:, 2]))


if __name__ == "__main__":
    unittest.main()
