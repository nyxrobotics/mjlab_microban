"""The one get-up training run: its scheduled switches and the refine reset.

CPU only: the real env config's terms behind fake managers, and the runner's
refine switch on a fake algorithm.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from mjlab.utils.buffers.delay_buffer import DelayBuffer

from mjlab_microban.tasks.curriculum import StagedCurriculum, bind_update_clock
from mjlab_microban.tasks.microban_getup_env_cfg import (
    GETUP_MAX_ITERATIONS,
    GETUP_REFINE_ACTION_STD,
    GETUP_REFINE_ENTROPY_COEF,
    GETUP_SCHEDULE,
    GETUP_STAGES,
    make_microban_getup_env_cfg,
)
from mjlab_microban.tasks.microban_getup_runner import (
    MicrobanGetupOnPolicyRunner,
)

STEPS = 24
CALM = ("standing_joint_vel", "raw_target_clip_excess", "roll_pose", "roll_pose_shoulder",
        "standing_target_error")


def expected(update: int) -> dict:
    """The table of docs/getup_training_export.md at ``update``."""

    delay = 3 if update >= GETUP_SCHEDULE["imu_delay"] else 0
    weights = dict.fromkeys(CALM, 0.0)
    feet_lateral, push = 10.0, (0.0, 0.0)
    if update >= GETUP_SCHEDULE["refine"]:
        weights.update(standing_joint_vel=-4.0, raw_target_clip_excess=-0.2, roll_pose=60.0)
        feet_lateral = 30.0
    if update >= GETUP_SCHEDULE["effort"]:
        weights.update(roll_pose=0.0, roll_pose_shoulder=60.0, standing_target_error=-2.0,
                       raw_target_clip_excess=-2.0)
    if update >= GETUP_SCHEDULE["push"]:
        push = (-0.3, 0.3)
    return {"delay": delay, "weights": weights, "feet_lateral": feet_lateral, "push": push}


class FakeEnv:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.buffers = {
            name: DelayBuffer(min_lag=0, max_lag=cfg.observations["actor"].terms[name].delay_max_lag,
                              batch_size=2)
            for name in ("base_ang_vel", "projected_gravity")
        }
        self.reward_manager = SimpleNamespace(get_term_cfg=cfg.rewards.__getitem__)
        self.event_manager = SimpleNamespace(get_term_cfg=cfg.events.__getitem__)
        self.command_manager = SimpleNamespace(get_term_cfg=cfg.commands.__getitem__)
        self.observation_manager = SimpleNamespace(
            get_term_cfg=lambda group, name: cfg.observations[group].terms[name],
            _group_obs_term_delay_buffer={"actor": self.buffers},
        )
        self.common_step_counter = 0

    def state(self) -> dict:
        rewards = self.cfg.rewards
        return {
            "delay": {buffer.max_lag for buffer in self.buffers.values()}.pop(),
            "weights": {name: rewards[name].weight for name in CALM},
            "feet_lateral": rewards["feet_lateral"].weight,
            "push": tuple(self.cfg.events["push_robot"].params["velocity_range"]["x"]),
        }


class GetupScheduleTest(unittest.TestCase):
    def test_every_switch_happens_at_its_update(self) -> None:
        env = FakeEnv(make_microban_getup_env_cfg())
        term = StagedCurriculum(SimpleNamespace(params={"stages": GETUP_STAGES}), env)
        term(env, None, GETUP_STAGES)
        self.assertEqual(env.state(), expected(0))
        bind_update_clock(env, STEPS)
        for update in (1, 2499, 2500, 3999, 4000, 9999, 10000, GETUP_MAX_ITERATIONS):
            env.common_step_counter = update * STEPS
            term(env, None, GETUP_STAGES)
            self.assertEqual(env.state(), expected(update), update)

    def test_play_config_has_the_final_latency_and_no_schedule(self) -> None:
        cfg = make_microban_getup_env_cfg(play=True)
        self.assertEqual(cfg.curriculum, {})
        for name in ("base_ang_vel", "projected_gravity"):
            self.assertEqual(cfg.observations["actor"].terms[name].delay_max_lag, 3)
        self.assertNotIn("joint_torques_l2", cfg.rewards)

    def test_run_length(self) -> None:
        from mjlab_microban.tasks.microban_getup_env_cfg import MicrobanGetupRlCfg

        self.assertEqual(MicrobanGetupRlCfg.max_iterations, GETUP_MAX_ITERATIONS)
        # Effort without pushes for 5000 updates, then 3000 with pushes.
        self.assertEqual(GETUP_SCHEDULE["push"] - GETUP_SCHEDULE["effort"], 5000)
        self.assertEqual(GETUP_MAX_ITERATIONS - GETUP_SCHEDULE["push"], 3000)


class FakeAlgorithm:
    def __init__(self) -> None:
        self.std = torch.nn.Parameter(torch.full((18,), 9.0))
        self.policy = SimpleNamespace(distribution=SimpleNamespace(std_param=self.std))
        self.optimizer = torch.optim.Adam([self.std], lr=3e-4)
        self.std.grad = torch.ones(18)
        self.optimizer.step()
        self.learning_rate = 3e-4
        self.entropy_coef = 0.01

    def get_policy(self):
        return self.policy


class RefineSwitchTest(unittest.TestCase):
    def _runner(self, counter: int):
        env = SimpleNamespace(common_step_counter=counter)
        bind_update_clock(env, STEPS)
        runner = SimpleNamespace(
            env=SimpleNamespace(unwrapped=env), alg=FakeAlgorithm(), exploration_refined=False,
            initial_learning_rate=1e-3,
        )
        return runner

    def test_switch_once_at_the_refine_update(self) -> None:
        runner = self._runner(GETUP_SCHEDULE["refine"] * STEPS - 1)
        refine = MicrobanGetupOnPolicyRunner.maybe_refine_exploration
        self.assertFalse(refine(runner))
        runner.env.unwrapped.common_step_counter += 1
        self.assertTrue(refine(runner))
        alg = runner.alg
        self.assertTrue(torch.all(alg.std == GETUP_REFINE_ACTION_STD))
        self.assertEqual(alg.optimizer.state, {})
        self.assertEqual(alg.learning_rate, 1e-3)
        self.assertEqual(alg.optimizer.param_groups[0]["lr"], 1e-3)
        self.assertEqual(alg.entropy_coef, GETUP_REFINE_ENTROPY_COEF)
        # Never again (the std keeps learning afterwards).
        with torch.no_grad():
            alg.std.fill_(2.0)
        runner.env.unwrapped.common_step_counter += 10 * STEPS
        self.assertFalse(refine(runner))
        self.assertTrue(torch.all(alg.std == 2.0))

if __name__ == "__main__":
    unittest.main()
