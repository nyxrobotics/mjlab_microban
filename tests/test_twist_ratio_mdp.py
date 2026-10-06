"""Ratio-keeping twist velocity reward (microban_twist_ratio_mdp, form B4)."""

from __future__ import annotations

import itertools
import math
import unittest
from types import SimpleNamespace

import torch

from mjlab_microban.tasks.microban_twist_ratio_mdp import (
    TWIST_RATIO_AXIS_SCALE,
    TWIST_RATIO_DEVIATION_SCALE,
    TWIST_RATIO_PROGRESS_FLOOR,
    home_levelled_twist,
    twist_ratio,
    twist_ratio_reward,
    twist_ratio_velocity,
)

SCALE = torch.tensor(TWIST_RATIO_AXIS_SCALE)
G = (0.7, 0.3, 1.5)  # infeasible diagonal: forward, left, counter-clockwise
HOME_TRUNK_PITCH_RAD = math.radians(10.0)  # forward-lean HOME (any value works)


def parts(commands, twists, uncommanded=None):
    return twist_ratio(
        torch.tensor(commands, dtype=torch.float32),
        torch.tensor(twists, dtype=torch.float32),
        uncommanded=None if uncommanded is None else torch.tensor(uncommanded, dtype=torch.float32),
    )


def reward(commands, twists, uncommanded=None):
    return twist_ratio_reward(
        torch.tensor(commands, dtype=torch.float32),
        torch.tensor(twists, dtype=torch.float32),
        uncommanded=None if uncommanded is None else torch.tensor(uncommanded, dtype=torch.float32),
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

    def test_kept_ratio_has_no_deviation_and_earns_its_scale(self) -> None:
        commands = [list(G), [-0.4, 0.2, -0.9], [0.3, -0.1, 0.0], [0.0, 0.25, -1.2]]
        for k in (0.0, 0.25, 0.5, 1.0):
            result = parts(commands, [scaled(c, k) for c in commands])
            self.assertTrue(torch.allclose(result.speed, torch.full((4,), k), atol=1e-5))
            self.assertTrue(torch.allclose(result.deviation, torch.zeros(4), atol=1e-5))
            values = reward(commands, [scaled(c, k) for c in commands])
            self.assertTrue(torch.allclose(values, torch.full((4,), 1.0 + k), atol=1e-5))

    def test_fast_motion_in_another_direction_earns_only_its_projection(self) -> None:
        c_hat, u, w = unit_and_normal(G)
        v_hat = 0.1 * u + math.sqrt(4.0 - 0.01) * w  # |v^| = 2, mostly perpendicular
        result = parts([list(G)], [(v_hat * SCALE).tolist()])
        self.assertAlmostEqual(float(result.speed[0]), 0.1 / float(c_hat.norm()), places=5)
        self.assertAlmostEqual(float(result.deviation[0]), math.sqrt(4.0 - 0.01), places=4)

    def test_bad_inputs(self) -> None:
        with self.assertRaises(ValueError):
            twist_ratio(torch.zeros(2, 2), torch.zeros(2, 2))
        with self.assertRaises(ValueError):
            twist_ratio(torch.zeros(2, 3), torch.zeros(2, 3), axis_scale=(0.7, 0.0, 1.5))
        with self.assertRaises(ValueError):
            twist_ratio(
                torch.zeros(1, 3), torch.zeros(1, 3), uncommanded=torch.zeros(1, 3), uncommanded_scale=None
            )
        with self.assertRaises(ValueError):
            twist_ratio_reward(torch.zeros(1, 3), torch.zeros(1, 3), progress_floor=1.0)


class FormB4Test(unittest.TestCase):
    def test_still_below_deviating_progress_below_exact_progress(self) -> None:
        c_hat, u, w = unit_and_normal(G)
        k = 0.3 * float(c_hat.norm())
        still = [0.0, 0.0, 0.0]
        deviating = ((k * u + 0.6 * w) * SCALE).tolist()  # same progress, far off the ray
        exact = ((k * u) * SCALE).tolist()
        values = reward([list(G)] * 3, [still, deviating, exact]).tolist()
        self.assertAlmostEqual(values[0], 1.0, places=6)
        self.assertGreater(values[1], 1.0 + TWIST_RATIO_PROGRESS_FLOOR * 0.3 - 1e-6)
        self.assertLess(values[1], values[2])
        self.assertAlmostEqual(values[2], 1.3, places=5)

    def test_any_progress_however_deviated_beats_standing_still(self) -> None:
        _c, u, w = unit_and_normal(G)
        for deviation in (0.1, 0.5, 2.0, 10.0):
            value = reward([list(G)], [((0.05 * u + deviation * w) * SCALE).tolist()])
            self.assertGreater(float(value[0]), 1.0)

    def test_sideways_equals_standing_and_opposite_is_worse(self) -> None:
        for command in (G, (0.7, -0.3, -1.5), (-0.4, 0.25, 0.9), (0.3, 0.0, 0.0), (0.0, 0.0, -2.0)):
            c_hat, u, w = unit_and_normal(command)
            k = 0.5
            values = reward(
                [list(command)] * 3,
                [[0.0, 0.0, 0.0], (k * w * SCALE).tolist(), (-k * u * SCALE).tolist()],
            ).tolist()
            self.assertAlmostEqual(values[0], 1.0, places=6)
            self.assertAlmostEqual(values[1], 1.0, places=5)
            self.assertAlmostEqual(values[2], 1.0 - k / float(c_hat.norm()), places=5)
            self.assertLess(values[2], values[0])

    def test_no_bonus_beyond_the_command(self) -> None:
        commands = [list(G), [0.3, 0.0, 0.0], [0.0, -0.2, 0.0]]
        exact = reward(commands, commands)
        self.assertTrue(torch.allclose(exact, torch.full((3,), 2.0), atol=1e-5))
        for k in (1.2, 2.0):
            over = reward(commands, [scaled(c, k) for c in commands])
            self.assertTrue(bool((over < exact).all()))
            self.assertTrue(bool((over > 1.0 + TWIST_RATIO_PROGRESS_FLOOR - 1e-6).all()))

    def test_turning_with_drift_and_bob_beats_standing_still(self) -> None:
        # Turning right in place at the command with lateral drift, forward
        # creep, vertical bob and roll/pitch rates.
        command = [0.0, 0.0, -0.5]
        values = reward(
            [command] * 3,
            [[0.0, 0.0, 0.0], [0.05, 0.08, -0.5], [0.05, 0.08, -0.25]],
            [[0.0, 0.0, 0.0], [0.08, 0.4, 0.3], [0.08, 0.4, 0.3]],
        ).tolist()
        self.assertAlmostEqual(values[0], 1.0, places=6)
        self.assertGreater(values[1], 1.0 + TWIST_RATIO_PROGRESS_FLOOR - 1e-6)
        self.assertGreater(values[2], 1.0)
        self.assertGreater(values[1], values[2])

    def test_left_right_mirror_symmetry(self) -> None:
        def mirror(t):
            return [t[0], -t[1], -t[2]]

        commands = [list(G), [0.6, 0.3, 1.2], [-0.4, 0.25, 0.9], [0.0, 0.0, 0.5], [0.0, 0.2, 0.0], [0.0, 0.0, 0.0]]
        twists = [[0.289, -0.017, 0.94], [0.2, 0.05, 0.3], [-0.1, -0.2, 0.4], [0.05, 0.08, 0.5], [0.0, 0.1, -0.2], [0.02, 0.01, 0.1]]
        unc = [[0.05, 0.3, -0.2], [0.0, 0.1, 0.1], [0.02, -0.4, 0.2], [0.08, 0.4, 0.3], [0.0, 0.0, 0.0], [0.01, 0.1, 0.0]]
        base = reward(commands, twists, unc)
        mirrored = reward(
            [mirror(c) for c in commands],
            [mirror(t) for t in twists],
            [[u[0], -u[1], u[2]] for u in unc],  # roll rate flips, pitch rate does not
        )
        self.assertTrue(torch.allclose(base, mirrored, atol=1e-6))

    def test_every_command_has_the_same_best_value(self) -> None:
        commands = [[0.0, 0.0, 0.0], [0.07, 0.0, 0.0], [0.0, 0.03, 0.1], [0.3, 0.1, 0.6], list(G), [0.0, 0.0, -3.0]]
        values = reward(commands, commands)
        self.assertTrue(torch.allclose(values, torch.full((len(commands),), 2.0), atol=1e-5))

    def test_standing_and_small_commands(self) -> None:
        # A standing command: still = 2, moving lowers it toward 1, never below.
        values = reward(
            [[0.0, 0.0, 0.0]] * 3,
            [[0.0, 0.0, 0.0], [0.014, -0.006, 0.06], [0.7, -0.3, 1.5]],
        ).tolist()
        self.assertAlmostEqual(values[0], 2.0, places=6)
        self.assertTrue(1.0 < values[1] < 2.0)
        self.assertAlmostEqual(values[2], 1.0, places=6)
        # A small command (n = 0.1) tracked exactly earns 2, still 1.5.
        small = [0.07, 0.0, 0.0]
        values = reward([small] * 2, [small, [0.0, 0.0, 0.0]]).tolist()
        self.assertAlmostEqual(values[0], 2.0, places=5)
        self.assertAlmostEqual(values[1], 1.5, places=5)

    def test_toppling_along_the_command_earns_less_than_walking(self) -> None:
        command = [[0.5, 0.0, 0.0]] * 2
        twist = [[0.5, 0.0, 0.0]] * 2
        values = reward(command, twist, [[-0.8, 0.0, 3.0], [0.0, 0.0, 0.0]]).tolist()
        self.assertLess(values[0], values[1] - 0.5)

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

    def test_deviation_gain_rises_sharply_near_agreement(self) -> None:
        c_hat, u, w = unit_and_normal(G)
        k = 0.5 * float(c_hat.norm())
        values = [
            float(reward([list(G)], [((k * u + d * w) * SCALE).tolist()])[0])
            for d in (0.0, TWIST_RATIO_DEVIATION_SCALE, 3 * TWIST_RATIO_DEVIATION_SCALE)
        ]
        self.assertAlmostEqual(values[0], 1.5, places=5)
        drop = 0.5 * (1 - TWIST_RATIO_PROGRESS_FLOOR) * (1 - math.exp(-1))
        self.assertAlmostEqual(values[1], 1.5 - drop, places=5)
        for previous, current in itertools.pairwise(values):
            self.assertLess(current, previous)


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
        term = twist_ratio_velocity(SimpleNamespace(params={}), env)
        return lambda: term(env, filter_time_constant=tau), term

    def test_first_step_uses_the_current_values(self) -> None:
        env = _FilterEnv([list(G), [0.3, 0.0, 0.0]])
        env.move([scaled(G, 0.4), [0.3, 0.0, 0.0]])
        call, _term = self.make(env)
        expected = twist_ratio_reward(
            env.command, torch.tensor([scaled(G, 0.4), [0.3, 0.0, 0.0]]), uncommanded=torch.zeros(2, 3)
        )
        self.assertTrue(torch.allclose(call(), expected, atol=1e-6))
        self.assertIn("Metrics/twist_ratio_speed", env.extras["log"])

    def test_zero_time_constant_is_the_instantaneous_reward(self) -> None:
        env = _FilterEnv([list(G)] * 3)
        call, _term = self.make(env, tau=0.0)
        torch.manual_seed(0)
        for _ in range(5):
            twist = (torch.randn(3, 3) * torch.tensor([0.3, 0.2, 1.0])).tolist()
            env.move(twist)
            expected = twist_ratio_reward(env.command, torch.tensor(twist), uncommanded=torch.zeros(3, 3))
            self.assertTrue(torch.allclose(call(), expected, atol=1e-6))

    def test_stride_sway_averages_out(self) -> None:
        # Walking forward at 0.2 m/s with +-0.15 m/s lateral sway at 1.6 Hz and
        # +-0.1 m/s vertical bob at 3.2 Hz.
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
        self.assertGreater(float(filtered[0]), 1.6)
        self.assertGreater(float(filtered[0]), float(instant[0]) + 0.2)
        self.assertAlmostEqual(float(filtered[1]), 1.0, places=5)  # standing on a moving command

    def test_sustained_drift_is_not_averaged_out(self) -> None:
        env = _FilterEnv([[0.2, 0.0, 0.0]])
        call, _term = self.make(env)
        for _ in range(200):
            env.move([[0.2, 0.1, 0.0]])
            value = call()
        expected = twist_ratio_reward(env.command, torch.tensor([[0.2, 0.1, 0.0]]), uncommanded=torch.zeros(1, 3))
        self.assertAlmostEqual(float(value[0]), float(expected[0]), places=4)
        self.assertLess(float(value[0]), 1.6)

    def test_a_robot_that_follows_a_new_command_at_once_stays_on_target(self) -> None:
        env = _FilterEnv([[0.3, 0.0, 0.0]])
        call, _term = self.make(env)
        env.move([[0.3, 0.0, 0.0]])
        for _ in range(100):
            call()
        env.command = torch.tensor([[0.0, 0.2, 0.0]])
        env.move([[0.0, 0.2, 0.0]])
        values = [float(call()[0]) for _ in range(100)]
        self.assertGreater(min(values), 1.999)

    def test_reset_restarts_the_filters_of_those_envs_only(self) -> None:
        env = _FilterEnv([[0.3, 0.0, 0.0]] * 2)
        call, term = self.make(env)
        env.move([[0.3, 0.0, 0.0]] * 2)
        for _ in range(50):
            call()
        env.move([[0.0, 0.0, 0.0]] * 2)
        term.reset(env_ids=torch.tensor([1]))
        value = call()
        self.assertAlmostEqual(float(value[1]), 1.0, places=5)  # restarted: standing now
        self.assertGreater(float(value[0]), 1.9)  # filter still near the walk


class FrameTest(unittest.TestCase):
    def test_env_term_reads_the_home_levelled_twist(self) -> None:
        half = 0.5 * HOME_TRUNK_PITCH_RAD
        twist = torch.tensor([[0.289, -0.017, 0.94], [0.2, 0.0857, 0.4286]])
        data = SimpleNamespace(
            root_link_quat_w=torch.tensor([[math.cos(half), 0.0, math.sin(half), 0.0]] * 2),
            root_link_lin_vel_w=torch.stack((twist[:, 0], twist[:, 1], torch.zeros(2)), dim=-1),
            root_link_ang_vel_w=torch.stack((torch.zeros(2), torch.zeros(2), twist[:, 2]), dim=-1),
        )
        env = SimpleNamespace(scene={"robot": SimpleNamespace(data=data)})
        self.assertTrue(torch.allclose(home_levelled_twist(env, HOME_TRUNK_PITCH_RAD), twist, atol=1e-6))

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
        torch.manual_seed(0)
        quat = torch.nn.functional.normalize(torch.randn(8, 4), dim=-1)
        data = SimpleNamespace(
            root_link_quat_w=quat,
            root_link_lin_vel_w=torch.randn(8, 3),
            root_link_ang_vel_w=torch.randn(8, 3),
        )
        env = SimpleNamespace(scene={"robot": SimpleNamespace(data=data)})
        twist = home_levelled_twist(env, HOME_TRUNK_PITCH_RAD)
        self.assertTrue(
            torch.equal(twist[:, :2], home_levelled_root_lin_vel_b(env, HOME_TRUNK_PITCH_RAD)[:, :2])
        )
        self.assertTrue(
            torch.equal(twist[:, 2], home_levelled_root_ang_vel_b(env, HOME_TRUNK_PITCH_RAD)[:, 2])
        )


if __name__ == "__main__":
    unittest.main()
