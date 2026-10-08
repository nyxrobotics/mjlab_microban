"""Ratio-keeping twist velocity reward (microban_twist_ratio_mdp)."""

from __future__ import annotations

import itertools
import math
import unittest
from types import SimpleNamespace

import torch

from mjlab_microban.tasks.microban_twist_ratio_mdp import (
    TWIST_RATIO_AXIS_SCALE,
    TWIST_RATIO_DIRECTION_PENALTY,
    TWIST_RATIO_EPS,
    twist_ratio,
    twist_ratio_reward,
    twist_ratio_velocity,
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
            self.assertAlmostEqual(float(values[0]), 1.0, places=5)  # the best value

    def test_standing_command_is_best_answered_by_standing_still(self) -> None:
        # n = 0: u = 0, p = a = 0, speed 1 (nothing left undone); all motion
        # is off-command motion.
        standing = parts(
            [[0.0, 0.0, 0.0]] * 3,
            [[0.0, 0.0, 0.0], [0.007, -0.003, 0.03], [0.07, -0.03, 0.3]],
        )
        self.assertTrue(torch.equal(standing.speed, torch.ones(3)))
        self.assertTrue(torch.equal(standing.along, torch.zeros(3)))
        self.assertAlmostEqual(float(standing.error[0]), 0.0)
        moved = math.sqrt(0.01**2 + 0.01**2 + 0.02**2)
        self.assertAlmostEqual(float(standing.error[1]), moved, places=5)
        self.assertAlmostEqual(float(standing.error[2]), math.sqrt(0.01 + 0.01 + 0.04), places=5)
        torch.manual_seed(1)
        moves = torch.randn(256, 3) * torch.tensor([0.2, 0.1, 0.5])
        values = twist_ratio_reward(torch.zeros(257, 3), torch.cat((torch.zeros(1, 3), moves)))
        self.assertAlmostEqual(float(values[0]), 1.0, places=6)  # full marks, as exact tracking
        self.assertTrue(bool((values[1:] < values[0]).all()))

    def test_drifting_forward_on_the_standing_command_costs(self) -> None:
        # The 2026-10-07 walker drifted about 0.06 m/s forward on the standing
        # command; standing still must clearly beat that (it scored 0.5
        # against 0.46 with the old speed a / max(n, eps)).
        values = twist_ratio_reward(torch.zeros(3, 3), torch.tensor([[0.0, 0.0, 0.0], [0.06, 0.0, 0.0],
                                                                    [0.06, 0.0, -0.1]]))
        self.assertAlmostEqual(float(values[0]), 1.0, places=6)
        self.assertAlmostEqual(float(values[1]), math.exp(-0.06 / 0.7), places=6)
        self.assertLess(float(values[1]), 0.92)
        self.assertLess(float(values[2]), float(values[1]))

    def test_one_speed_formula_for_every_command(self) -> None:
        # speed = 1 - (n - clamp(p, 0, n)) / max(n, eps) with u = c^ / max(n, eps),
        # also for the 0.1 m/s commands and below eps; for n >= eps it is the
        # along-command fraction clamp(p, 0, n) / n.
        torch.manual_seed(2)
        for norm in (0.002, 0.009, TWIST_RATIO_EPS, 0.03, 0.1 / 0.7, 0.5, 1.2):
            direction = torch.randn(64, 3)
            c_hat = direction / direction.norm(dim=-1, keepdim=True) * norm
            v_hat = c_hat * torch.rand(64, 1) * 2.0 + 0.05 * torch.randn(64, 3)
            result = twist_ratio(c_hat * SCALE, v_hat * SCALE)
            floor = max(norm, TWIST_RATIO_EPS)
            along = (v_hat * c_hat).sum(dim=-1) / floor
            progress = torch.clamp(along, min=0.0).clamp(max=norm)
            speed = 1.0 - (norm - progress) / floor
            self.assertTrue(torch.allclose(result.along, along, atol=1e-5), norm)
            self.assertTrue(torch.allclose(result.speed, speed, atol=1e-5), norm)
            if norm >= TWIST_RATIO_EPS:  # the same value as the former a / max(n, eps)
                self.assertTrue(torch.allclose(result.speed, progress / floor, atol=1e-5), norm)

    def test_no_nan_or_infinity_for_any_command_size(self) -> None:
        torch.manual_seed(3)
        sizes = (0.0, 1e-30, 1e-12, 1e-6, 1e-3, TWIST_RATIO_EPS, 0.1, 1.0, 1e3)
        direction = torch.randn(len(sizes) * 32, 3)
        direction = direction / direction.norm(dim=-1, keepdim=True)
        norms = torch.tensor(sizes).repeat_interleave(32).unsqueeze(-1)
        commands = direction * norms * SCALE
        for twists in (torch.zeros_like(commands), commands, torch.randn_like(commands) * 3.0):
            result = twist_ratio(commands, twists, uncommanded=torch.randn_like(commands))
            for field in result:
                self.assertTrue(bool(torch.isfinite(field).all()))
            values = twist_ratio_reward(commands, twists)
            self.assertTrue(bool(torch.isfinite(values).all()))
            self.assertGreaterEqual(float(values.min()), 0.0)
            self.assertLessEqual(float(values.max()), 1.0)

    def test_reward_is_continuous_where_the_command_crosses_eps_and_zero(self) -> None:
        torch.manual_seed(4)
        direction = torch.randn(128, 3)
        direction = direction / direction.norm(dim=-1, keepdim=True)
        twists = torch.randn(128, 3) * 0.02 * SCALE
        for norm in (TWIST_RATIO_EPS, 0.0):
            below = twist_ratio_reward(direction * max(norm * (1 - 1e-5), 1e-9) * SCALE, twists)
            at = twist_ratio_reward(direction * norm * SCALE, twists)
            above = twist_ratio_reward(direction * (norm * (1 + 1e-5) + 1e-9) * SCALE, twists)
            self.assertLess(float((below - at).abs().max()), 1e-5, norm)
            self.assertLess(float((above - at).abs().max()), 1e-5, norm)
        # The command 0.1 m/s forward scored by the one formula.
        command = [[0.1, 0.0, 0.0]] * 4
        values = reward(command, [[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
        expected = [0.5, 0.75, 1.0, math.exp(-0.1 / 0.7)]
        for value, want in zip(values.tolist(), expected, strict=True):
            self.assertAlmostEqual(value, want, places=5)

    def test_any_progress_along_a_moving_command_beats_standing_still(self) -> None:
        # Every command from 0.1 eps up, at half, exactly and twice the command
        # and along it at the robot's slowest gait (0.02 normalized).  Below
        # eps standing still keeps 1 - n / eps of the speed (1 at n = 0).
        torch.manual_seed(5)
        for norm in (0.1 * TWIST_RATIO_EPS, 0.5 * TWIST_RATIO_EPS, 0.999 * TWIST_RATIO_EPS,
                     TWIST_RATIO_EPS, 0.05, 0.1 / 0.7, 0.2 / 0.7, 0.6, 1.5):
            direction = torch.randn(32, 3)
            c_hat = direction / direction.norm(dim=-1, keepdim=True) * norm
            commands = c_hat * SCALE
            still = twist_ratio_reward(commands, torch.zeros_like(commands))
            standing_speed = max(0.0, 1.0 - norm / TWIST_RATIO_EPS)
            self.assertTrue(torch.allclose(still, torch.full((32,), 0.5 * (1.0 + standing_speed)), atol=1e-6))
            for k in (0.5, 1.0, 2.0):
                moving = twist_ratio_reward(commands, commands * k)
                # The overshoot costs (k - 1) n: twice the command beats
                # standing for n < ln 2 (0.48 m/s forward, beyond the reach).
                wins = (k - 1.0) * norm < math.log(2.0) - math.log(1.0 + standing_speed)
                self.assertEqual(bool((moving > still).all()), wins, (norm, k))
            slow = twist_ratio_reward(commands, c_hat / c_hat.norm(dim=-1, keepdim=True) * 0.02 * SCALE)
            self.assertTrue(bool((slow > still).all()), norm)

    def test_moving_against_a_command_is_worse_than_standing_still(self) -> None:
        torch.manual_seed(6)
        for norm in (0.5 * TWIST_RATIO_EPS, TWIST_RATIO_EPS, 0.1 / 0.7, 0.8):
            direction = torch.randn(32, 3)
            commands = direction / direction.norm(dim=-1, keepdim=True) * norm * SCALE
            still = twist_ratio_reward(commands, torch.zeros_like(commands))
            for k in (0.5, 1.0, 2.0):
                values = twist_ratio_reward(commands, -k * commands)
                self.assertTrue(bool((values < still).all()), (norm, k))

    def test_moving_commands_are_best_tracked_exactly(self) -> None:
        # 0.1 and 0.2 m/s forward and backward, 0.1 m/s lateral, 0.5 rad/s yaw:
        # the best over a fine grid of single-axis answers is the command.
        for axis, value in ((0, 0.1), (0, 0.2), (0, -0.1), (0, -0.2), (1, 0.1), (1, -0.1), (2, 0.5), (2, -0.5)):
            command = [0.0, 0.0, 0.0]
            command[axis] = value
            grid = torch.zeros(401, 3)
            grid[:, axis] = torch.linspace(-2.0 * abs(value), 3.0 * abs(value), 401)
            grid = torch.cat((grid, grid + torch.tensor([0.0, 0.01, 0.05])), dim=0)
            values = twist_ratio_reward(torch.tensor([command]).expand(len(grid), 3), grid)
            best = grid[int(torch.argmax(values))]
            self.assertTrue(torch.allclose(best, torch.tensor(command), atol=abs(value) * 0.01), (command, best))
            self.assertAlmostEqual(float(values.max()), 1.0, places=5)

    def test_every_moving_command_has_the_same_best_value(self) -> None:
        commands = [[0.07, 0.0, 0.0], [0.0, 0.03, 0.1], [0.3, 0.1, 0.6], list(G), [0.0, 0.0, -3.0], [0.007, 0.0, 0.0]]
        values = twist_ratio_reward(torch.tensor(commands), torch.tensor(commands))
        self.assertTrue(torch.allclose(values, torch.ones(len(commands)), atol=1e-5))

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

        commands = [list(G), [0.6, 0.3, 1.2], [-0.4, 0.25, 0.9], [0.0, 0.0, 0.0], [0.0, 0.2, 0.0], [0.003, 0.001, 0.004]]
        twists = [[0.289, -0.017, 0.94], [0.2, 0.05, 0.3], [-0.1, -0.2, 0.4], [0.05, 0.02, 0.1], [0.0, 0.1, -0.2], [0.004, 0.0, 0.01]]
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
        values = twist_ratio_reward(command, twist, uncommanded=uncommanded)
        self.assertLess(float(values[0]), float(values[2]))
        self.assertAlmostEqual(float(values[2]), 0.5, places=6)  # still on a moving command
        bob = math.sqrt((0.05 / 0.7) ** 2 + 2 * (0.2 / 1.5) ** 2)
        self.assertAlmostEqual(float(values[1]), math.exp(-bob), places=5)
        self.assertGreater(float(values[1]), 0.75)
        # Without the uncommanded motion the fall would earn the full reward.
        planar = twist_ratio_reward(command, twist)
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

    def test_reward_is_bounded_and_never_negative(self) -> None:
        # Falling can never pay: every value is in [0, 1].  Standing still
        # is worth 1 on the standing command and 1/2 on every moving one.
        torch.manual_seed(0)
        commands = torch.randn(4096, 3) * torch.tensor([0.5, 0.25, 1.2])
        commands[:256] = 0.0
        twists = torch.randn(4096, 3) * torch.tensor([1.0, 1.0, 3.0])
        uncommanded = torch.randn(4096, 3) * torch.tensor([1.0, 3.0, 3.0])
        values = twist_ratio_reward(commands, twists, uncommanded=uncommanded)
        self.assertGreaterEqual(float(values.min()), 0.0)
        self.assertLessEqual(float(values.max()), 1.0)
        still = twist_ratio_reward(
            torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.1, 0.6], list(G)]), torch.zeros(3, 3)
        )
        for value, want in zip(still.tolist(), [1.0, 0.5, 0.5], strict=True):
            self.assertAlmostEqual(value, want, places=6)
        half = twist_ratio_reward(torch.tensor([list(G)]), torch.tensor([scaled(G, 0.3)]))
        self.assertAlmostEqual(float(half[0]), 0.65, places=5)

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
            num_envs=2,
            device="cpu",
            step_dt=0.02,
        )
        term = twist_ratio_velocity(SimpleNamespace(params={}), env)
        value = term(env, trunk_pitch=HOME_TRUNK_PITCH_RAD, filter_time_constant=0.0)
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
            num_envs=1,
            device="cpu",
            step_dt=0.02,
        )
        # Half the command along its direction; the vertical velocity and the
        # roll/pitch rates (0.05 m/s, 0.3 and -0.2 rad/s) are uncommanded.
        term = twist_ratio_velocity(SimpleNamespace(params={}), env)
        value = term(env, filter_time_constant=0.0)
        uncommanded = math.sqrt((0.05 / 0.7) ** 2 + (0.3 / 1.5) ** 2 + (0.2 / 1.5) ** 2)
        self.assertAlmostEqual(float(value[0]), 0.75 * math.exp(-uncommanded), places=5)
        planar = twist_ratio_velocity(SimpleNamespace(params={}), env)(
            env, uncommanded_scale=None, filter_time_constant=0.0
        )
        self.assertAlmostEqual(float(planar[0]), 0.75, places=5)


class _FilterEnv:
    """Minimal env for the filtered term: untilted root, settable motion."""

    def __init__(self, commands) -> None:
        self.num_envs = len(commands)
        self.device = "cpu"
        self.step_dt = 0.02
        self.extras = {"log": {}}
        self.command = torch.tensor(commands, dtype=torch.float32)
        self.data = SimpleNamespace(
            root_link_lin_vel_b=torch.zeros(self.num_envs, 3),
            root_link_ang_vel_b=torch.zeros(self.num_envs, 3),
        )
        self.scene = {"robot": SimpleNamespace(data=self.data)}
        self.command_manager = SimpleNamespace(get_command=lambda name: self.command)

    def move(self, twist, uncommanded=None) -> None:
        twist = torch.tensor(twist, dtype=torch.float32)
        unc = torch.zeros_like(twist) if uncommanded is None else torch.tensor(uncommanded, dtype=torch.float32)
        self.data.root_link_lin_vel_b = torch.stack((twist[:, 0], twist[:, 1], unc[:, 0]), dim=-1)
        self.data.root_link_ang_vel_b = torch.stack((unc[:, 1], unc[:, 2], twist[:, 2]), dim=-1)


class FilteredTermTest(unittest.TestCase):
    @staticmethod
    def make(env, tau=0.5):
        from mjlab_microban.tasks.microban_twist_ratio_mdp import twist_ratio_velocity

        term = twist_ratio_velocity(SimpleNamespace(params={}), env)
        return lambda: term(env, filter_time_constant=tau), term

    def test_first_step_uses_the_current_values(self) -> None:
        env = _FilterEnv([list(G), [0.3, 0.0, 0.0]])
        env.move([scaled(G, 0.4), [0.3, 0.0, 0.0]])
        call, _term = self.make(env)
        expected = twist_ratio_reward(env.command, torch.tensor([scaled(G, 0.4), [0.3, 0.0, 0.0]]))
        self.assertTrue(torch.allclose(call(), expected, atol=1e-6))

    def test_zero_time_constant_is_the_instantaneous_reward(self) -> None:
        env = _FilterEnv([list(G)] * 3)
        call, _term = self.make(env, tau=0.0)
        torch.manual_seed(0)
        for _ in range(5):
            twist = (torch.randn(3, 3) * torch.tensor([0.3, 0.2, 1.0])).tolist()
            env.move(twist)
            self.assertTrue(
                torch.allclose(call(), twist_ratio_reward(env.command, torch.tensor(twist)), atol=1e-6)
            )

    def test_stride_sway_averages_out(self) -> None:
        # Walking forward at 0.2 m/s with +-0.15 m/s lateral sway at 1.6 Hz
        # (the command's own direction) and +-0.1 m/s vertical bob at 3.2 Hz.
        env = _FilterEnv([[0.2, 0.0, 0.0]] * 2)
        call, _term = self.make(env)
        instant, _ = self.make(env, tau=0.0)
        filtered_values, instant_values = [], []
        for step in range(300):
            t = step * env.step_dt
            sway = 0.15 * math.sin(2 * math.pi * 1.6 * t)
            bob = 0.1 * math.sin(2 * math.pi * 3.2 * t)
            env.move([[0.2, sway, 0.0], [0.0, 0.0, 0.0]], [[bob, 0.0, 0.0], [0.0, 0.0, 0.0]])
            filtered_values.append(call())
            instant_values.append(instant())
        filtered = torch.stack(filtered_values[100:]).mean(dim=0)
        instant = torch.stack(instant_values[100:]).mean(dim=0)
        # Walking with sway earns clearly more than standing still (1/2).
        self.assertGreater(float(filtered[0]), 0.85)
        self.assertGreater(float(filtered[0]), float(instant[0]) + 0.1)
        self.assertAlmostEqual(float(filtered[1]), 0.5, places=5)

    def test_sustained_drift_is_not_averaged_out(self) -> None:
        env = _FilterEnv([[0.2, 0.0, 0.0]])
        call, _term = self.make(env)
        for _ in range(200):
            env.move([[0.2, 0.1, 0.0]])
            value = call()
        expected = twist_ratio_reward(env.command, torch.tensor([[0.2, 0.1, 0.0]]))
        self.assertAlmostEqual(float(value[0]), float(expected[0]), places=4)
        self.assertLess(float(value[0]), 0.75)

    def test_a_robot_that_follows_a_new_command_at_once_stays_on_target(self) -> None:
        env = _FilterEnv([[0.3, 0.0, 0.0]])
        call, _term = self.make(env)
        env.move([[0.3, 0.0, 0.0]])
        for _ in range(100):
            call()
        env.command = torch.tensor([[0.0, 0.2, 0.0]])
        env.move([[0.0, 0.2, 0.0]])
        values = [float(call()[0]) for _ in range(100)]
        self.assertGreater(min(values), 0.999)

    def test_reset_restarts_the_filters_of_those_envs_only(self) -> None:
        env = _FilterEnv([[0.3, 0.0, 0.0]] * 2)
        call, term = self.make(env)
        env.move([[0.3, 0.0, 0.0]] * 2)
        for _ in range(50):
            call()
        env.move([[0.0, 0.0, 0.0]] * 2)
        term.reset(env_ids=torch.tensor([1]))
        value = call()
        self.assertAlmostEqual(float(value[1]), 0.5, places=5)  # restarted: standing now
        self.assertGreater(float(value[0]), 0.9)  # filter still near the walk

    def test_moving_commands_keep_the_filtered_reward(self) -> None:
        # n >= 0.05 (normalized): the value is the filtered reward exactly.
        from mjlab_microban.tasks.microban_twist_ratio_mdp import TWIST_RATIO_BLEND_NORM

        self.assertEqual(TWIST_RATIO_BLEND_NORM, 0.05)
        commands = [[0.035, 0.0, 0.0], [0.0, 0.015, 0.0], [0.0, 0.0, 0.075], list(G), [0.1, 0.0, 0.0]]
        env = _FilterEnv(commands)
        call, term = self.make(env)
        torch.manual_seed(3)
        for _ in range(80):
            twist = torch.tensor(commands) + 0.1 * torch.randn(5, 3)
            unc = 0.2 * torch.randn(5, 3)
            env.move(twist.tolist(), unc.tolist())
            value = call()
            filtered = twist_ratio(term.command, term.twist, uncommanded=term.uncommanded)
            expected = 0.5 * (1.0 + filtered.speed) * torch.exp(-filtered.error)
            self.assertTrue(torch.allclose(value, expected, atol=1e-6))

    def test_standing_still_on_the_standing_command_is_full_marks(self) -> None:
        env = _FilterEnv([[0.0, 0.0, 0.0]])
        call, _term = self.make(env)
        env.move([[0.0, 0.0, 0.0]])
        for _ in range(50):
            value = call()
        self.assertAlmostEqual(float(value[0]), 1.0, places=6)

    def test_stepping_in_place_on_the_standing_command_costs(self) -> None:
        # Stepping in place: lateral sway +-0.1 m/s at 2 Hz, bob and roll rate,
        # zero mean.  Filtered it nearly vanished (about 0.95); the blend
        # evaluates the instantaneous motion on a standing command.
        env = _FilterEnv([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        call, _term = self.make(env)
        values = []
        for step in range(300):
            t = step * env.step_dt
            sway = 0.1 * math.sin(2 * math.pi * 2.0 * t)
            bob = 0.05 * math.sin(2 * math.pi * 4.0 * t)
            roll = 0.6 * math.sin(2 * math.pi * 2.0 * t)
            env.move([[0.0, sway, 0.0], [0.0, 0.0, 0.0]], [[bob, roll, 0.0], [0.0, 0.0, 0.0]])
            values.append(call())
        mean = torch.stack(values[100:]).mean(dim=0)
        self.assertAlmostEqual(float(mean[1]), 1.0, places=6)  # still
        self.assertLess(float(mean[0]), 0.8)  # stepping in place

    def test_the_blend_is_continuous_where_the_command_crosses_its_norm(self) -> None:
        torch.manual_seed(4)
        twist = (0.05 * torch.randn(1, 3)).tolist()
        unc = (0.2 * torch.randn(1, 3)).tolist()
        values = []
        for norm in (0.0, 1e-6, 0.05 * (1 - 1e-6), 0.05, 0.05 * (1 + 1e-6)):
            env = _FilterEnv([[norm * 0.7, 0.0, 0.0]])
            call, _term = self.make(env)
            env.move([[0.0, 0.0, 0.0]])
            call()
            env.move(twist, unc)
            values.append(float(call()[0]))
        self.assertLess(abs(values[0] - values[1]), 1e-4)
        self.assertLess(abs(values[2] - values[3]), 1e-4)
        self.assertLess(abs(values[3] - values[4]), 1e-4)

    def test_the_blend_has_no_nan(self) -> None:
        sizes = (0.0, 1e-30, 1e-12, 1e-6, 0.05, 1.0)
        env = _FilterEnv([[s * 0.7, 0.0, 0.0] for s in sizes])
        call, _term = self.make(env)
        torch.manual_seed(5)
        for _ in range(20):
            env.move((torch.randn(len(sizes), 3)).tolist(), (torch.randn(len(sizes), 3)).tolist())
            value = call()
            self.assertTrue(bool(torch.isfinite(value).all()))
            self.assertTrue(bool(((value >= 0.0) & (value <= 1.0)).all()))

# Outputs of the filtered term and the pure reward as implemented when the
# walker of the comparison was trained (form B3: exp/twist-ratio-validation
# cf91eec, run trw6B3_s42) on the motion of _golden_sequence().
GOLDEN_B3 = {"filtered": [[0.409094, 0.409808, 0.310659, 0.209858, 0.189795, 0.400346], [0.414879, 0.412083, 0.3203, 0.226528, 0.205094, 0.407476], [0.426553, 0.411424, 0.328397, 0.255392, 0.210388, 0.421391], [0.437867, 0.413884, 0.332636, 0.271767, 0.214798, 0.428114], [0.450201, 0.43436, 0.346801, 0.293713, 0.22407, 0.431423], [0.464829, 0.445849, 0.347669, 0.307413, 0.236955, 0.440628], [0.480302, 0.463376, 0.349086, 0.314652, 0.235362, 0.45546], [0.481048, 0.488019, 0.355664, 0.328646, 0.237605, 0.459444], [0.487112, 0.493575, 0.360248, 0.336748, 0.253155, 0.463668], [0.504501, 0.498174, 0.349412, 0.351894, 0.263591, 0.481011], [0.513295, 0.492951, 0.352668, 0.359312, 0.271202, 0.484424], [0.525134, 0.508077, 0.349495, 0.374088, 0.283431, 0.496396], [0.526055, 0.516091, 0.353248, 0.37106, 0.296744, 0.499247], [0.521316, 0.520437, 0.367538, 0.377166, 0.29885, 0.503702], [0.528091, 0.512986, 0.367977, 0.395646, 0.302201, 0.508423], [0.530555, 0.497766, 0.372659, 0.420832, 0.315032, 0.513992], [0.530141, 0.496352, 0.379028, 0.416282, 0.312875, 0.508678], [0.539496, 0.499422, 0.384963, 0.416878, 0.301732, 0.505896], [0.544256, 0.502471, 0.389662, 0.434988, 0.307331, 0.499558], [0.547209, 0.504597, 0.421379, 0.426965, 0.317845, 0.496567], [0.541887, 0.510119, 0.482781, 0.435466, 0.33557, 0.500246], [0.539302, 0.52427, 0.52092, 0.436758, 0.349043, 0.503263], [0.548332, 0.530792, 0.517786, 0.451848, 0.353522, 0.502574], [0.557892, 0.538203, 0.520398, 0.456602, 0.350496, 0.519453], [0.566944, 0.544126, 0.621399, 0.471635, 0.361342, 0.536725]], "instantaneous_last": [0.552736, 0.566233, 0.170201, 0.689134, 0.285068, 0.478691]}


def _golden_sequence():
    gen = torch.Generator().manual_seed(7)
    commands = torch.tensor(
        [[0.7, 0.3, 1.5], [0.0, 0.0, -0.5], [0.0, 0.0, 0.0], [-0.4, 0.25, 0.9], [0.07, 0.0, 0.0], [0.3, -0.1, 0.0]]
    )
    steps = []
    for k in range(25):
        twist = commands * (0.4 + 0.4 * math.sin(0.3 * k)) + 0.15 * torch.randn(6, 3, generator=gen)
        unc = 0.2 * torch.randn(6, 3, generator=gen)
        steps.append((twist, unc))
    return commands, steps



class GoldenTest(unittest.TestCase):
    def test_reproduces_the_implementation_the_comparison_trained_with(self) -> None:
        # The comparison's small-command branch (below n = 0.2) is gone; the
        # commands at or above 0.2 (columns 0, 1, 3, 5) score as they did.
        kept = [0, 1, 3, 5]
        commands, steps = _golden_sequence()
        env = _FilterEnv(commands.tolist())
        term = twist_ratio_velocity(SimpleNamespace(params={}), env)
        for (twist, unc), expected in zip(steps, GOLDEN_B3["filtered"], strict=True):
            env.data.root_link_lin_vel_b = torch.stack((twist[:, 0], twist[:, 1], unc[:, 0]), -1)
            env.data.root_link_ang_vel_b = torch.stack((unc[:, 1], unc[:, 2], twist[:, 2]), -1)
            value = term(env)[kept]
            self.assertTrue(torch.allclose(value, torch.tensor(expected)[kept], atol=2e-6))
        last = twist_ratio_reward(commands, steps[-1][0], uncommanded=steps[-1][1])[kept]
        self.assertTrue(torch.allclose(last, torch.tensor(GOLDEN_B3["instantaneous_last"])[kept], atol=2e-6))


if __name__ == "__main__":
    unittest.main()
