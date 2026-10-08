"""One step-scheduled curriculum for every Microban task.

A task's schedule is a table of stages.  Each stage has a name, the PPO update
(iteration) at which it starts, and the settings it writes: plain values of
reward, command, event or observation terms.  ``StagedCurriculum`` applies, at
every curriculum call (episode resets), every stage whose start the
environment's ``common_step_counter`` has reached.  It advances in a loop, so
stages that start at the same update (as in a scaled dry run) apply in one
call.

A run is never resumed from a checkpoint (``refuse_resume``): a resumed run
would not replay its curriculum, its adaptive learning rate and its
reward-gated stages as one uninterrupted run does.  A run that stopped is
trained again from update 0.

The update clock is ``iteration * steps_per_update``.  The runner binds
``steps_per_update`` (its ``num_steps_per_env``) to the environment
(``bind_update_clock``), so a run with another rollout length switches at the
same updates.  Every applied stage prints one line in a fixed format::

    Curriculum stage <k> <name> at step <S> (update <U>)

which the pipeline monitor checks against the table.

The tables' updates come from mjlab_microban/schedules.py, which scales them
for dry runs (``MICROBAN_SCHEDULE_SCALE``).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, NamedTuple

STEPS_PER_UPDATE_ATTR = "microban_steps_per_update"


class Setting(NamedTuple):
    """One value a stage writes.

    ``manager`` is "reward", "command", "event" or "observation" ("<group>/<term>"
    as ``term``).  ``path`` is a dotted attribute path below the term config;
    a "params." prefix addresses a key of its params dict.  The observation
    path "delay_max_lag" is special in a running environment: it sets the live
    delay buffer's bound only (0 serves the newest frame), within the lag the
    config allocated, and keeps the term config's lag so the buffer keeps its
    history.
    """

    manager: str
    term: str
    path: str
    value: Any


class Stage(NamedTuple):
    name: str
    iteration: int
    settings: tuple[Setting, ...]


def _write(target: Any, path: str, value: Any) -> None:
    keys = path.split(".")
    if keys[0] == "params":
        if len(keys) != 2:
            raise ValueError(f"params path must be params.<key>: {path}")
        if keys[1] not in target.params:
            raise KeyError(f"no parameter {keys[1]!r} to set")
        target.params[keys[1]] = value
        return
    for key in keys[:-1]:
        target = getattr(target, key)
    if not hasattr(target, keys[-1]):
        raise AttributeError(f"no attribute {path!r} to set")
    setattr(target, keys[-1], value)


def _cfg_term(cfg: Any, setting: Setting) -> Any:
    table = {"reward": cfg.rewards, "command": cfg.commands, "event": cfg.events}
    if setting.manager == "observation":
        group, term = setting.term.split("/")
        return cfg.observations[group].terms[term]
    return table[setting.manager][setting.term]


def _env_term(env: Any, setting: Setting) -> Any:
    if setting.manager == "reward":
        return env.reward_manager.get_term_cfg(setting.term)
    if setting.manager == "command":
        return env.command_manager.get_term_cfg(setting.term)
    if setting.manager == "event":
        return env.event_manager.get_term_cfg(setting.term)
    if setting.manager == "observation":
        group, term = setting.term.split("/")
        return env.observation_manager.get_term_cfg(group, term)
    raise ValueError(f"unknown manager {setting.manager!r}")


def apply_to_cfg(cfg: Any, settings: Iterable[Setting]) -> None:
    """Write ``settings`` into an environment config (before the env exists)."""

    for setting in settings:
        if setting.manager == "observation" and setting.path == "delay_max_lag":
            term = _cfg_term(cfg, setting)
            if setting.value > term.delay_max_lag:
                raise ValueError(f"{setting.term}: delay above the allocated lag")
        _write(_cfg_term(cfg, setting), setting.path, setting.value)


def apply_to_env(env: Any, settings: Iterable[Setting]) -> None:
    """Write ``settings`` into a running environment's managers."""

    for setting in settings:
        term_cfg = _env_term(env, setting)
        if setting.manager == "observation" and setting.path == "delay_max_lag":
            if term_cfg.delay_max_lag <= 0:
                raise ValueError(f"{setting.term}: no delay buffer was allocated")
            group, term = setting.term.split("/")
            buffer = env.observation_manager._group_obs_term_delay_buffer[group][term]
            allocated = term_cfg.delay_max_lag
            if not 0 <= setting.value <= allocated:
                raise ValueError(f"{setting.term}: delay {setting.value} outside 0..{allocated}")
            buffer.max_lag = setting.value
            # Lags already drawn above the new bound are clamped at once.
            buffer._current_lags.clamp_(max=setting.value)
            continue
        _write(term_cfg, setting.path, setting.value)


def validate_stages(stages: Sequence[Stage]) -> None:
    previous = -1
    for index, stage in enumerate(stages):
        if not isinstance(stage, Stage):
            raise TypeError(f"curriculum stage {index} must be a Stage")
        if not isinstance(stage.iteration, int) or isinstance(stage.iteration, bool):
            raise TypeError(f"curriculum stage {index} iteration must be an int")
        if stage.iteration <= previous:
            raise ValueError("curriculum stage iterations must be non-negative and strictly increasing")
        if not all(isinstance(s, Setting) for s in stage.settings):
            raise TypeError(f"curriculum stage {index} settings must be Setting tuples")
        previous = stage.iteration


def final_settings(stages: Sequence[Stage]) -> tuple[Setting, ...]:
    """Every setting of the table in order (the state after its last stage)."""

    return tuple(setting for stage in stages for setting in stage.settings)


def bind_update_clock(env: Any, steps_per_update: int) -> None:
    """Record the runner's rollout length (env steps per PPO update) on ``env``."""

    if not isinstance(steps_per_update, int) or steps_per_update <= 0:
        raise ValueError("steps_per_update must be a positive int")
    setattr(env, STEPS_PER_UPDATE_ATTR, steps_per_update)


def refuse_resume(train_cfg: Mapping[str, Any]) -> None:
    """Refuse ``--agent.resume`` (see module doc)."""

    if train_cfg.get("resume"):
        raise ValueError("A Microban run is not resumed from a checkpoint: train it again from update 0")


def stage_log_line(index: int, name: str, step: int, steps_per_update: int) -> str:
    return f"Curriculum stage {index} {name} at step {step} (update {step // steps_per_update})"


class StagedCurriculum:
    """Curriculum term: apply every stage of ``stages`` that is due (see module doc)."""

    def __init__(self, cfg: Any, env: Any) -> None:
        del env
        validate_stages(cfg.params["stages"])
        self.current_stage = 0

    def __call__(self, env: Any, env_ids: Any, stages: Sequence[Stage]) -> dict[str, int]:
        del env_ids
        steps = getattr(env, STEPS_PER_UPDATE_ATTR, None)
        counter = int(env.common_step_counter)
        while self.current_stage < len(stages):
            stage = stages[self.current_stage]
            if stage.iteration > 0:
                if steps is None:
                    if counter == 0:
                        break
                    raise RuntimeError(
                        "StagedCurriculum: the runner did not bind its update clock "
                        "(curriculum.bind_update_clock)"
                    )
                if counter < stage.iteration * steps:
                    break
            apply_to_env(env, stage.settings)
            self.current_stage += 1
            print(stage_log_line(self.current_stage, stage.name, counter, steps or 1), flush=True)
        return {"stage": self.current_stage}
