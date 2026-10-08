"""Walking runner: the curriculum's update clock; a run is never resumed."""

from __future__ import annotations

from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from mjlab_microban.tasks.curriculum import bind_update_clock, refuse_resume


class MicrobanVelocityOnPolicyRunner(VelocityOnPolicyRunner):
    """mjlab's velocity runner plus the update clock."""

    def __init__(self, env, train_cfg: dict, *args, **kwargs) -> None:
        refuse_resume(train_cfg)
        bind_update_clock(env.unwrapped, int(train_cfg["num_steps_per_env"]))
        super().__init__(env, train_cfg, *args, **kwargs)
