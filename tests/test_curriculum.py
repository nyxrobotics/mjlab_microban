"""The one step-scheduled curriculum (tasks/curriculum.py) on fake managers."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from mjlab.utils.buffers.delay_buffer import DelayBuffer

from mjlab_microban import schedules
from mjlab_microban.tasks import curriculum
from mjlab_microban.tasks.curriculum import (
    Setting,
    Stage,
    StagedCurriculum,
    apply_to_cfg,
    bind_update_clock,
    final_settings,
    refuse_resume,
)

STAGES = (
    Stage("start", 0, (Setting("reward", "a", "weight", 0.0),)),
    Stage("widen", 10, (Setting("command", "twist", "ranges.lin_vel_x", (-0.7, 0.7)),)),
    Stage("push", 20, (Setting("event", "push", "params.velocity_range", {"x": (-0.3, 0.3)}),)),
)


def term(**kwargs):
    return SimpleNamespace(params={}, **kwargs)


class FakeEnv:
    def __init__(self) -> None:
        self.rewards = {"a": term(weight=1.0)}
        self.commands = {"twist": term(ranges=SimpleNamespace(lin_vel_x=(-0.5, 0.5)))}
        self.events = {"push": SimpleNamespace(params={"velocity_range": {"x": (0.0, 0.0)}})}
        self.obs_cfg = term(delay_max_lag=3)
        self.buffer = DelayBuffer(min_lag=0, max_lag=3, batch_size=4)
        self.reward_manager = SimpleNamespace(get_term_cfg=self.rewards.__getitem__)
        self.command_manager = SimpleNamespace(get_term_cfg=self.commands.__getitem__)
        self.event_manager = SimpleNamespace(get_term_cfg=self.events.__getitem__)
        self.observation_manager = SimpleNamespace(
            get_term_cfg=lambda group, name: self.obs_cfg,
            _group_obs_term_delay_buffer={"actor": {"gyro": self.buffer}},
        )
        self.common_step_counter = 0


def make_term(stages=STAGES) -> StagedCurriculum:
    return StagedCurriculum(SimpleNamespace(params={"stages": stages}), None)


class StagedCurriculumTest(unittest.TestCase):
    def test_applies_each_stage_at_its_update(self) -> None:
        env = FakeEnv()
        term_ = make_term()
        term_(env, None, STAGES)  # before the runner binds: only update-0 stages
        self.assertEqual(env.rewards["a"].weight, 0.0)
        bind_update_clock(env, 24)
        env.common_step_counter = 10 * 24 - 1
        term_(env, None, STAGES)
        self.assertEqual(env.commands["twist"].ranges.lin_vel_x, (-0.5, 0.5))
        env.common_step_counter = 10 * 24
        self.assertEqual(term_(env, None, STAGES), {"stage": 2})
        self.assertEqual(env.commands["twist"].ranges.lin_vel_x, (-0.7, 0.7))
        self.assertEqual(env.events["push"].params["velocity_range"], {"x": (0.0, 0.0)})

    def test_every_due_stage_applies_in_one_call(self) -> None:
        env = FakeEnv()
        bind_update_clock(env, 24)
        env.common_step_counter = 25 * 24
        term_ = make_term()
        self.assertEqual(term_(env, None, STAGES), {"stage": 3})
        self.assertEqual(env.events["push"].params["velocity_range"], {"x": (-0.3, 0.3)})
        # Idempotent afterwards.
        self.assertEqual(term_(env, None, STAGES), {"stage": 3})

    def test_a_resume_is_refused(self) -> None:
        refuse_resume({"resume": False})
        with self.assertRaisesRegex(ValueError, "train it again from update 0"):
            refuse_resume({"resume": True})

    def test_unbound_clock_after_start_is_an_error(self) -> None:
        env = FakeEnv()
        env.common_step_counter = 5
        with self.assertRaises(RuntimeError):
            make_term()(env, None, STAGES)

    def test_update_clock_follows_the_rollout_length(self) -> None:
        env = FakeEnv()
        bind_update_clock(env, 8)
        env.common_step_counter = 10 * 8
        self.assertEqual(make_term()(env, None, STAGES), {"stage": 2})

    def test_stage_log_line(self) -> None:
        self.assertEqual(
            curriculum.stage_log_line(2, "widen", 240, 24),
            "Curriculum stage 2 widen at step 240 (update 10)",
        )

    def test_delay_bound_changes_the_live_buffer_only(self) -> None:
        env = FakeEnv()
        curriculum.apply_to_env(env, (Setting("observation", "actor/gyro", "delay_max_lag", 0),))
        self.assertEqual(env.buffer.max_lag, 0)
        self.assertEqual(env.obs_cfg.delay_max_lag, 3)  # the buffer keeps its history
        obs = torch.arange(4.0).unsqueeze(-1)
        for step in range(5):
            env.buffer.append(obs + step)
            self.assertTrue(torch.equal(env.buffer.compute(), obs + step))
        curriculum.apply_to_env(env, (Setting("observation", "actor/gyro", "delay_max_lag", 3),))
        self.assertEqual(env.buffer.max_lag, 3)
        with self.assertRaises(ValueError):
            curriculum.apply_to_env(env, (Setting("observation", "actor/gyro", "delay_max_lag", 4),))

    def test_rejects_unordered_tables_and_unknown_targets(self) -> None:
        with self.assertRaises(ValueError):
            make_term((STAGES[1], STAGES[0]))
        env = FakeEnv()
        with self.assertRaises(AttributeError):
            curriculum.apply_to_env(env, (Setting("command", "twist", "ranges.nope", 1),))
        with self.assertRaises(KeyError):
            curriculum.apply_to_env(env, (Setting("event", "push", "params.nope", 1),))

    def test_final_settings_on_a_config(self) -> None:
        env = FakeEnv()
        cfg = SimpleNamespace(rewards=env.rewards, commands=env.commands, events=env.events)
        apply_to_cfg(cfg, final_settings(STAGES))
        self.assertEqual(cfg.rewards["a"].weight, 0.0)
        self.assertEqual(cfg.commands["twist"].ranges.lin_vel_x, (-0.7, 0.7))
        self.assertEqual(cfg.events["push"].params["velocity_range"], {"x": (-0.3, 0.3)})

    def test_schedule_scale(self) -> None:
        import os

        old = os.environ.get(schedules.SCHEDULE_SCALE_ENV)
        try:
            os.environ[schedules.SCHEDULE_SCALE_ENV] = "0.002"
            self.assertEqual(
                [schedules.scaled(i) for i in (0, 2500, 4000, 10000, 16500)], [0, 5, 8, 20, 33]
            )
            os.environ[schedules.SCHEDULE_SCALE_ENV] = "2"
            with self.assertRaises(ValueError):
                schedules.scaled(10)
        finally:
            if old is None:
                os.environ.pop(schedules.SCHEDULE_SCALE_ENV, None)
            else:
                os.environ[schedules.SCHEDULE_SCALE_ENV] = old
        self.assertEqual(schedules.scaled(3000), 3000)


if __name__ == "__main__":
    unittest.main()
