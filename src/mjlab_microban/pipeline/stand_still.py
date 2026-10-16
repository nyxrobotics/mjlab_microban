"""Standing still: the leg joints' speed and the falls of a policy that should stand.

The walker and PICO stand at HOME on a zero command (PICO with its foot target
and arms neutral) for 10 s, judged from 2 s on; the get-up starts fallen and
is judged over the last 5 s of its 20 s (the robot's get-up timeout), on the
environments standing through them.  Each environment holds one servo goal
delay of DELAYS_MS (1-10 physics steps of 5 ms; play draws 3-6, training 3-8)
and one foot friction of FRICTIONS (the play env's range 0.3-1.2) on the play
env's rigid floor, with the training's observation noise (without it a policy
that trembled on the robot stands still here; config/pipeline.yaml
``stand_still`` has the measurements).

Measured, while up (the trunk-fixed head point above 0.9 x the standing head
height, the get-up's standing line):

* ``leg_speed_rms_rad_s``: per delay, the RMS over its environments, judged
  steps and the twelve leg joints of the simulated joint speed; the judged
  value is the worst delay's;
* ``falls``: environments that went down (the walker and PICO: at any step;
  the get-up: after standing 1 s);
* touchdowns per second (both feet, recorded only).

usage: python -m mjlab_microban.pipeline.stand_still {walk,pico,getup} TASK CKPT OUT.json
           --max-leg-speed-rms 0.05 --max-falls 0 [--device cuda:0]
Prints one ``RESULT {json}`` line; OUT.json has ``checks`` and ``status``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

DELAYS_MS = (5, 10, 15, 20, 25, 30, 35, 40, 45, 50)
FRICTIONS = (0.3, 0.6, 0.9, 1.2)
PHYSICS_STEP_MS = 5
# (total s, first judged s) of a run: the walker and PICO from HOME at rest,
# the get-up from a fallen start over the last 5 s of its 20 s.
TIMING_S = {"walk": (10.0, 2.0), "pico": (10.0, 2.0), "getup": (20.0, 15.0)}
GETUP_STOOD_STEPS = 50  # 1 s up before a get-up's drop counts as a fall
LEG_JOINT_KEYS = ("hip", "knee", "ankle")


def _env_cfg(kind: str, task: str, envs: int, seed: int, delays: list[int], seconds: float, noise: bool):
    from mjlab.tasks.registry import load_env_cfg

    from mjlab_microban.scripts.probe_legacy_actor_in_teleop_env import _configure_fixed_twist

    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = envs
    cfg.seed = seed
    cfg.observations["actor"].enable_corruption = noise
    if kind == "getup":  # its own 20 s episode (the get-up runner requires it)
        cfg.events["reset_near_home"].params["rel_near_home_envs"] = 0.0
    else:
        cfg.episode_length_s = seconds + 1.0
        _configure_fixed_twist(cfg.commands["twist"])
    # One goal delay per environment, set after the reset and never redrawn.
    robot = cfg.scene.entities["robot"]
    actuators = tuple(replace(a, delay_min_lag=min(delays), delay_max_lag=max(delays), delay_hold_prob=1.0,
                              delay_update_period=0) for a in robot.articulation.actuators)
    cfg.scene.entities["robot"] = replace(robot, articulation=replace(robot.articulation, actuators=actuators))
    return cfg


def _policy(kind: str, task: str, checkpoint: str, wrapped, device: str):
    import torch
    from tensordict import TensorDict

    n = wrapped.num_envs
    if kind == "getup":
        from mjlab.tasks.registry import load_rl_cfg, load_runner_cls

        runner = load_runner_cls(task)(wrapped, asdict(load_rl_cfg(task)), device=device)
        runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
        return runner.get_inference_policy(device=device)
    if kind == "walk":
        from mjlab_microban.scripts.teleop_v12_bootstrap_gate import _legacy_model
        from mjlab_microban.tasks.microban_teleop_v12_bootstrap import inspect_legacy_velocity_checkpoint

        digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        _identity, state = inspect_legacy_velocity_checkpoint(checkpoint, digest)
        actor = _legacy_model().to(device)
        actor.load_state_dict(state, strict=True)
    else:
        from mjlab_microban.scripts.evaluate_teleop_v12_checkpoint import _load_actor

        actor, _iteration, _infos = _load_actor(Path(checkpoint), device=device)
    actor.eval()

    def policy(obs):
        with torch.inference_mode():
            return actor(TensorDict({"actor": obs["actor"]}, batch_size=[n]))

    return policy


def run(kind: str, task: str, checkpoint: str, *, device: str, seed: int, reps: int, noise: bool = True,
        delays_ms=DELAYS_MS, frictions=FRICTIONS) -> dict:
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab.rl import RslRlVecEnvWrapper

    from mjlab_microban.tasks.mdp import _head_height
    from mjlab_microban.tasks.microban_getup_env_cfg import HEAD_STANDING_HEIGHT

    delays = [int(d) // PHYSICS_STEP_MS for d in delays_ms]
    cells = [(d, f) for d in delays for f in frictions for _ in range(reps)]
    n = len(cells)
    seconds, judged_from = TIMING_S[kind]
    cfg = _env_cfg(kind, task, n, seed, delays, seconds, noise)
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
    try:
        policy = _policy(kind, task, checkpoint, wrapped, device)
        robot = env.scene["robot"]
        lag = torch.tensor([d for d, _ in cells], device=device)
        friction = torch.tensor([f for _, f in cells], device=device)
        foot = env.event_manager.get_term_cfg("foot_friction")
        for value in frictions:
            ids = torch.nonzero(friction == value).flatten()
            foot.func(env, ids, **{**foot.params, "ranges": (value, value)})
        legs = [i for i, name in enumerate(robot.joint_names) if any(k in name for k in LEG_JOINT_KEYS)]
        head_cfg = SceneEntityCfg("robot", body_names=("head",))
        feet = env.scene.sensors["feet_ground_contact"]
        threshold = 0.9 * HEAD_STANDING_HEIGHT
        # The get-up's episode ends at its 1000th step (a reset before it is read): 999 steps.
        steps = round(seconds / env.step_dt) - (kind == "getup")
        first = round(judged_from / env.step_dt)

        obs, _ = wrapped.reset()
        for actuator in robot.actuators:
            actuator.set_lags(lag)
        if kind == "pico":
            from mjlab_microban.scripts.probe_legacy_actor_in_teleop_env import _zero_neutral_targets

            _zero_neutral_targets(env)
            obs = wrapped.get_observations()

        up = torch.zeros(steps, n, dtype=torch.bool, device=device)
        speed_sq = torch.zeros(steps, n, device=device)
        touchdowns = torch.zeros(steps, n, device=device)
        down = feet.data.found.reshape(n, -1)[:, :2] > 0
        ended = torch.zeros(n, dtype=torch.bool, device=device)
        for t in range(steps):
            if kind != "getup":
                env.command_manager.get_term("twist").vel_command_b.zero_()
            obs, _, dones, extras = wrapped.step(policy(obs))
            # A terminated environment was reset in the step: down from then on.
            ended = ended | (dones.bool() & ~extras["time_outs"].bool())
            up[t] = (_head_height(env, head_cfg) > threshold) & ~ended
            speed_sq[t] = robot.data.joint_vel[:, legs].square().mean(dim=-1)
            now_down = feet.data.found.reshape(n, -1)[:, :2] > 0
            touchdowns[t] = (now_down & ~down).sum(dim=-1).float()
            down = now_down
        applied = torch.stack([a._delay_buffer.current_lags for a in robot.actuators if a._delay_buffer is not None])
        if not torch.equal(applied[:, ~ended], lag[~ended].expand_as(applied[:, ~ended])):
            raise RuntimeError("the goal delays drifted from the ones set")
    finally:
        wrapped.close()

    if kind == "getup":
        # Down after 1 s standing (or terminated); judged: standing through the window.
        run_up = torch.zeros(n, device=device)
        stood = torch.zeros(n, dtype=torch.bool, device=device)
        fell = ended.clone()
        for t in range(steps):
            fell |= stood & ~up[t]
            run_up = (run_up + 1) * up[t]
            stood |= run_up >= GETUP_STOOD_STEPS
        judged = up[first:].all(dim=0) & ~fell
        weight = judged.float().expand(steps - first, n)
    else:
        fell = ~up.all(dim=0)
        alive = torch.cumprod(up.float(), dim=0)  # up at every step so far
        weight = alive[first:]
        judged = weight.sum(dim=0) > 0
    window_sq, window_td = speed_sq[first:], touchdowns[first:]
    per_delay = {}
    for d in delays:
        mask = lag == d
        w = weight[:, mask]
        count = float(w.sum())
        per_delay[d * PHYSICS_STEP_MS] = {
            "leg_speed_rms_rad_s": (float((window_sq[:, mask] * w).sum()) / count) ** 0.5 if count else None,
            "falls": int(fell[mask].sum()),
            "judged_envs": int(judged[mask].sum()),
            "envs": int(mask.sum()),
            "touchdowns_per_s": float((window_td[:, mask] * w).sum()) / (count * env.step_dt) if count else None,
        }
    env_rms = ((window_sq * weight).sum(0) / weight.sum(0).clamp(min=1)).sqrt()
    by_friction = {}
    for value in frictions:
        mask = (friction == value) & judged
        by_friction[value] = float(env_rms[mask].max()) if mask.any() else None
    worst = max(per_delay, key=lambda k: float("inf") if per_delay[k]["leg_speed_rms_rad_s"] is None
                else per_delay[k]["leg_speed_rms_rad_s"])
    return {
        "kind": kind, "task": task, "checkpoint": checkpoint, "seed": seed, "envs": n, "reps": reps,
        "seconds": seconds, "judged_from_s": judged_from, "delays_ms": list(delays_ms), "frictions": list(frictions),
        "noise": noise,
        "leg_speed_rms_rad_s": per_delay[worst]["leg_speed_rms_rad_s"], "worst_delay_ms": worst,
        "falls": int(fell.sum()), "judged_envs": int(judged.sum()),
        "per_delay": per_delay, "max_env_leg_speed_rms_rad_s_by_friction": by_friction,
    }


def judge(result: dict, max_leg_speed_rms: float, max_falls: int) -> dict:
    rms = result["leg_speed_rms_rad_s"]
    checks = {
        "leg_speed": rms is not None and all(v["judged_envs"] > 0 for v in result["per_delay"].values())
        and rms <= max_leg_speed_rms,
        "falls": result["falls"] <= max_falls,
    }
    return {**result, "lines": {"max_leg_speed_rms_rad_s": max_leg_speed_rms, "max_falls": max_falls},
            "checks": checks, "status": "pass" if all(checks.values()) else "fail"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("kind", choices=tuple(TIMING_S))
    parser.add_argument("task")
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--max-leg-speed-rms", type=float, required=True)
    parser.add_argument("--max-falls", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=107)
    parser.add_argument("--reps", type=int, default=2)
    args = parser.parse_args()
    started = time.time()
    result = judge(run(args.kind, args.task, args.checkpoint, device=args.device, seed=args.seed, reps=args.reps),
                   args.max_leg_speed_rms, args.max_falls)
    result["wall_s"] = round(time.time() - started, 1)
    with open(args.output, "w") as stream:
        json.dump(result, stream, indent=1)
    print("RESULT " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
