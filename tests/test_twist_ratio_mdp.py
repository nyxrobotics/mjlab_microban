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
    TWIST_RATIO_MIN_COMMAND_NORM,
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

    def test_standing_and_small_commands(self) -> None:
        # Standing still earns the same speed 1 as exact tracking of a moving
        # command; any motion lowers it on the scale of min_command_norm.
        standing = parts(
            [[0.0, 0.0, 0.0]] * 3,
            [[0.0, 0.0, 0.0], [0.007, -0.003, 0.03], [0.07, -0.03, 0.3]],
        )
        self.assertAlmostEqual(float(standing.speed[0]), 1.0)
        moved = math.sqrt(0.01**2 + 0.01**2 + 0.02**2)
        self.assertAlmostEqual(
            float(standing.speed[1]), 1.0 - moved / TWIST_RATIO_MIN_COMMAND_NORM, places=5
        )
        self.assertEqual(float(standing.speed[2]), 0.0)  # clamped at 0
        self.assertAlmostEqual(float(standing.error[0]), 0.0)
        self.assertAlmostEqual(float(standing.error[1]), moved, places=5)
        self.assertAlmostEqual(float(standing.error[2]), math.sqrt(0.01 + 0.01 + 0.04), places=5)
        # A small command (n = 0.02) tracked exactly earns speed 1, standing
        # still on it 1 - n / min_command_norm = 0.5.
        small = [0.014, 0.0, 0.0]
        result = parts([small] * 2, [small, [0.0, 0.0, 0.0]])
        self.assertAlmostEqual(float(result.speed[0]), 1.0, places=5)
        self.assertAlmostEqual(float(result.speed[1]), 0.5, places=5)
        self.assertAlmostEqual(float(result.error[0]), 0.0, places=6)

    def test_every_command_has_the_same_best_value(self) -> None:
        commands = [[0.0, 0.0, 0.0], [0.07, 0.0, 0.0], [0.0, 0.03, 0.1], [0.3, 0.1, 0.6], list(G), [0.0, 0.0, -3.0]]
        values = twist_ratio_reward(torch.tensor(commands), torch.tensor(commands))
        self.assertTrue(torch.allclose(values, torch.ones(len(commands)), atol=1e-5))

    def test_the_speed_branches_meet_on_the_ray(self) -> None:
        # n exactly at min_command_norm: both branches give s on the ray.
        unit = torch.tensor([1.0, 1.0, 0.0]) / math.sqrt(2.0)
        command = (unit * TWIST_RATIO_MIN_COMMAND_NORM * SCALE).tolist()
        for k in (0.0, 0.3, 0.7, 1.0):
            below = parts([[x * 0.999999 for x in command]], [scaled(command, k)])
            above = parts([command], [scaled(command, k)])
            self.assertAlmostEqual(float(below.speed[0]), k, places=4)
            self.assertAlmostEqual(float(above.speed[0]), k, places=4)

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
        # is worth 1 on a standing command and 1/2 on a moving one.
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
        # The comparison ran with min_command_norm 0.2 (the release uses 0.04).
        commands, steps = _golden_sequence()
        env = _FilterEnv(commands.tolist())
        term = twist_ratio_velocity(SimpleNamespace(params={}), env)
        for (twist, unc), expected in zip(steps, GOLDEN_B3["filtered"], strict=True):
            env.data.root_link_lin_vel_b = torch.stack((twist[:, 0], twist[:, 1], unc[:, 0]), -1)
            env.data.root_link_ang_vel_b = torch.stack((unc[:, 1], unc[:, 2], twist[:, 2]), -1)
            self.assertTrue(torch.allclose(term(env, min_command_norm=0.2), torch.tensor(expected), atol=2e-6))
        last = twist_ratio_reward(commands, steps[-1][0], min_command_norm=0.2, uncommanded=steps[-1][1])
        self.assertTrue(torch.allclose(last, torch.tensor(GOLDEN_B3["instantaneous_last"]), atol=2e-6))


if __name__ == "__main__":
    unittest.main()
